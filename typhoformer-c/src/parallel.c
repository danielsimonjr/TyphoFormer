/*
 * parallel.c — synchronous data-parallel SGD over pthreads (see parallel.h).
 */
#include "parallel.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* One Worker == one full model REPLICA plus everything it needs to process its
 * shard of the minibatch on its own thread. Every field here is thread-private:
 * two workers never touch the same memory during the parallel region, which is
 * what makes the design race-free without any locks. `model` and `pl` are the
 * replica's weights+gradients; the Mats are its private forward/backward
 * scratch; the ds/idx/start/count/lambda block is the job handed to it each
 * batch; `loss` is its output; `tid` is its pthread handle. */
typedef struct {
    Model     model;
    ParamList pl;
    /* per-worker scratch (no sharing between threads) */
    Mat xn, xt, yp, Y, dpred, dgate;
    Mat nbr, vel;   /* per-sample aux inputs: co-active neighbours, seed velocity */
    /* per-batch job description */
    const Dataset *ds;
    const int     *idx;
    int   start, count;
    float lambda;
    /* Dropout RNG state, private to this worker and carried FORWARD across
     * batches (worker_run seeds from it, then writes the advanced state back).
     * Distinct seeds per worker (see partrainer_new) mean each replica draws its
     * own dropout masks, so the parallel run is not a biased repeat of one mask
     * pattern. This is deterministic given the worker count, which is why the
     * serial path (threads=1) is used for the byte-exact golden test. */
    unsigned long dseed;   /* per-worker dropout RNG state, threaded across batches */
    double loss;       /* out: summed loss over this worker's shard */
    pthread_t tid;
} Worker;

/* The trainer owns the fixed pool of `n` replicas and the Config they were
 * built from. It is created once and reused for every minibatch. */
struct ParTrainer {
    int     n;
    Config  cfg;
    Worker *w;
};

/* Build the replica pool. Allocates `n_workers` full Models from the SAME
 * Config, so every replica has an identical parameter LAYOUT (same tensors, same
 * sizes, registered in the same order) even though their initial values differ —
 * the values are made identical later by partrainer_broadcast. Each replica also
 * gets its own scratch tensors sized from the Config and its own dropout seed. */
ParTrainer *partrainer_new(const Config *cfg, int n_workers) {
    if (n_workers < 1) n_workers = 1;      /* at least one replica */
    ParTrainer *pt = (ParTrainer *)malloc(sizeof *pt);
    assert(pt);
    pt->n   = n_workers;
    pt->cfg = *cfg;
    pt->w   = (Worker *)calloc((size_t)n_workers, sizeof(Worker));
    assert(pt->w);
    const Config *c = cfg;
    for (int i = 0; i < n_workers; ++i) {
        Worker *w = &pt->w[i];
        plist_init(&w->pl);
        w->model = model_new(c, &w->pl);   /* full independent replica + its ParamList */
        w->xn    = mat_new(c->in_len, c->d_num);
        w->xt    = mat_new(c->in_len, c->d_text);
        w->yp    = mat_new(1, 2);
        w->Y     = mat_new(c->pred_len, 2);
        w->dpred = mat_new(c->pred_len, c->out_dim);
        w->dgate = mat_new(c->in_len, c->d_model);
        w->nbr   = mat_new(TF_NBR_K, TF_NBR_NF);
        w->vel   = mat_new(1, 2);
        /* Seed = FNV-prime * (i+1) + 1: a cheap way to give each worker a
         * well-separated, nonzero starting dropout state. */
        w->dseed = 0x100000001b3UL * (unsigned long)(i + 1) + 1;   /* distinct per worker */
    }
    return pt;
}

void partrainer_free(ParTrainer *pt) {
    if (!pt) return;
    for (int i = 0; i < pt->n; ++i) {
        Worker *w = &pt->w[i];
        mat_free(&w->xn); mat_free(&w->xt); mat_free(&w->yp); mat_free(&w->Y);
        mat_free(&w->dpred); mat_free(&w->dgate);
        mat_free(&w->nbr); mat_free(&w->vel);
        model_free(&w->model);
        plist_free(&w->pl);
    }
    free(pt->w);
    free(pt);
}

int partrainer_workers(const ParTrainer *pt) { return pt->n; }

Model *partrainer_model(ParTrainer *pt, int i) { return &pt->w[i].model; }

/* Broadcast: overwrite every replica's parameter VALUES with the master's. This
 * must be called before each step so all replicas compute gradients at the exact
 * same weights the optimizer is about to update — the defining property of
 * synchronous data parallelism. The asserts pin the layout invariant (same
 * tensor count, same per-tensor sizes) that partrainer_new guarantees by
 * construction; the inner loop is a plain element-wise copy. Only `v` (values)
 * is copied, not `g` (gradients) — grads are freshly computed each step. */
void partrainer_broadcast(ParTrainer *pt, const ParamList *master) {
    for (int i = 0; i < pt->n; ++i) {
        ParamList *r = &pt->w[i].pl;
        assert(r->count == master->count);
        for (int p = 0; p < master->count; ++p) {
            assert(r->item[p].n == master->item[p].n);
            /* Each tensor's values are one contiguous float block, so a memcpy
             * replaces the element loop — this runs once per replica per batch
             * over ALL parameters (e.g. 5.1M floats x N workers on the full
             * config), so bulk copy speed matters. */
            memcpy(r->item[p].v, master->item[p].v,
                   (size_t)master->item[p].n * sizeof(float));
        }
    }
}

/* The thread body: process one worker's shard of the minibatch. Runs entirely
 * on thread-private state (this Worker's model, scratch, and grad buffers), so
 * it needs no locking. Steps:
 *   1. seed the dropout RNG from this worker's carried state — nn_dropout_seed
 *      sets a THREAD-LOCAL generator, so concurrent workers never race on it;
 *   2. zero this replica's gradient accumulators;
 *   3. for each sample in [start, start+count): fetch it, run forward, add its
 *      loss, run backward — model_backward ACCUMULATES grads into w->pl->g, so
 *      after the loop w->pl holds the SUM of per-sample grads over the shard
 *      (raw, un-averaged: the 1/bs division happens once, at reduction);
 *   4. write the advanced RNG state back so the next batch uses fresh masks. */
static void *worker_run(void *arg) {
    Worker *w = (Worker *)arg;
    nn_dropout_seed(w->dseed);      /* per-thread dropout RNG (race-free) */
    plist_zero_grad(&w->pl);
    w->loss = 0.0;
    for (int k = 0; k < w->count; ++k) {
        int s = w->idx[w->start + k];                       /* shuffled sample index */
        dataset_get(w->ds, s, w->xn, w->xt, w->yp, w->Y);
        /* Per-sample auxiliary inputs, mirroring the serial loop and evaluate():
         * co-active neighbours (used only when the model was built --co_spatial)
         * and the seed velocity (used only by the cv/gru/xattn decoders). Both
         * are cheap copies and no-ops for model heads that ignore them, and both
         * read only immutable per-record dataset tables — race-free. This is
         * what makes --cv/--gru/--xattn/--co_spatial work with --threads>1. */
        int nc; dataset_neighbors(w->ds, s, w->nbr, &nc);
        model_set_neighbors(&w->model, w->nbr, nc);
        dataset_seed_velocity(w->ds, s, w->vel);
        model_set_seed_velocity(&w->model, w->vel);
        model_forward(&w->model, w->xn, w->xt, w->yp);
        /* raw (un-averaged) gradients; the 1/bs scale is applied at reduction */
        w->loss += model_loss(w->model.pred, w->Y, w->model.pgf.gate, w->lambda,
                              w->dpred, w->dgate);
        model_backward(&w->model, w->dpred, w->dgate);      /* accumulates into w->pl->g */
    }
    w->dseed = nn_dropout_state();   /* advance so masks differ across batches */
    return NULL;
}

/* Run one minibatch in parallel and reduce to the averaged master gradient.
 * Three phases: PARTITION the batch, DISPATCH the threads, then REDUCE. The
 * caller is expected to have already broadcast the master weights into the
 * replicas (partrainer_broadcast) so all shards use identical parameters. */
double partrainer_step_grads(ParTrainer *pt, ParamList *master,
                             const Dataset *ds, const int *idx,
                             int b, int bs, float lambda) {
    /* --- PARTITION: split the bs samples into contiguous shards. Each worker
     * gets `base = bs/n` samples; the `rem = bs%n` leftovers are handed one each
     * to the first `rem` workers, so shard sizes differ by at most 1 and cover
     * exactly idx[b .. b+bs) with no gaps or overlap. `off` walks the start
     * offset forward so shards are disjoint contiguous ranges. */
    int base = bs / pt->n, rem = bs % pt->n, off = 0;
    int active = 0;
    for (int i = 0; i < pt->n; ++i) {
        Worker *w = &pt->w[i];
        w->count  = base + (i < rem ? 1 : 0);
        w->start  = b + off;
        w->ds     = ds; w->idx = idx; w->lambda = lambda;
        off += w->count;
    }
    /* --- DISPATCH: spawn a thread for workers 1..n-1, then run worker 0 inline
     * on the CALLING thread (saves one thread create/join, and worker 0 always
     * has the largest shard). Workers with an empty shard (possible when bs < n)
     * are skipped rather than spawned. */
    for (int i = 1; i < pt->n; ++i) {
        if (pt->w[i].count == 0) continue;
        int rc = pthread_create(&pt->w[i].tid, NULL, worker_run, &pt->w[i]);
        assert(rc == 0); (void)rc; ++active;
    }
    worker_run(&pt->w[0]);                 /* worker 0 on this thread */
    /* Join every spawned thread — a full barrier. After this line all replicas'
     * gradient buffers are final and safe to read from this thread. */
    for (int i = 1; i < pt->n; ++i)
        if (pt->w[i].count > 0) pthread_join(pt->w[i].tid, NULL);
    (void)active;

    /* --- REDUCE: master->g = (1/bs) * sum_workers replica->g.
     * Each replica already holds the SUM of its shard's per-sample gradients, so
     * summing across replicas yields the sum over the whole batch, and scaling
     * by 1/bs gives the mean-batch gradient — bit-identical in MATH to the serial
     * loop, and equal up to floating-point summation ORDER (the batch is added in
     * a different grouping across threads). That reordering is the only reason
     * results match the serial path to ~1e-5 rather than exactly; test_parallel.c
     * pins this, and the byte-exact golden test uses threads=1 to avoid it. */
    double total = 0.0;
    float inv = 1.0f / (float)bs;
    plist_zero_grad(master);
    for (int i = 0; i < pt->n; ++i) {
        total += pt->w[i].loss;            /* accumulate loss across shards too */
        ParamList *r = &pt->w[i].pl;
        for (int p = 0; p < master->count; ++p)
            for (int e = 0; e < master->item[p].n; ++e)
                master->item[p].g[e] += r->item[p].g[e];
    }
    /* Apply the single 1/bs averaging pass (deferred out of the workers). */
    for (int p = 0; p < master->count; ++p)
        for (int e = 0; e < master->item[p].n; ++e)
            master->item[p].g[e] *= inv;
    return total / (double)bs;             /* mean batch loss */
}
