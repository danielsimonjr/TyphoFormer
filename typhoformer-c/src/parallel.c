/*
 * parallel.c — synchronous data-parallel SGD over pthreads (see parallel.h).
 */
#include "parallel.h"

#include <assert.h>
#include <pthread.h>
#include <stdlib.h>

typedef struct {
    Model     model;
    ParamList pl;
    /* per-worker scratch (no sharing between threads) */
    Mat xn, xt, yp, Y, dpred, dgate;
    /* per-batch job description */
    const Dataset *ds;
    const int     *idx;
    int   start, count;
    float lambda;
    unsigned long dseed;   /* per-worker dropout RNG state, threaded across batches */
    double loss;       /* out: summed loss over this worker's shard */
    pthread_t tid;
} Worker;

struct ParTrainer {
    int     n;
    Config  cfg;
    Worker *w;
};

ParTrainer *partrainer_new(const Config *cfg, int n_workers) {
    if (n_workers < 1) n_workers = 1;
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
        w->model = model_new(c, &w->pl);
        w->xn    = mat_new(c->in_len, c->d_num);
        w->xt    = mat_new(c->in_len, c->d_text);
        w->yp    = mat_new(1, 2);
        w->Y     = mat_new(c->pred_len, 2);
        w->dpred = mat_new(c->pred_len, c->out_dim);
        w->dgate = mat_new(c->in_len, c->d_model);
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
        model_free(&w->model);
        plist_free(&w->pl);
    }
    free(pt->w);
    free(pt);
}

int partrainer_workers(const ParTrainer *pt) { return pt->n; }

void partrainer_broadcast(ParTrainer *pt, const ParamList *master) {
    for (int i = 0; i < pt->n; ++i) {
        ParamList *r = &pt->w[i].pl;
        assert(r->count == master->count);
        for (int p = 0; p < master->count; ++p) {
            assert(r->item[p].n == master->item[p].n);
            for (int e = 0; e < master->item[p].n; ++e)
                r->item[p].v[e] = master->item[p].v[e];
        }
    }
}

static void *worker_run(void *arg) {
    Worker *w = (Worker *)arg;
    nn_dropout_seed(w->dseed);      /* per-thread dropout RNG (race-free) */
    plist_zero_grad(&w->pl);
    w->loss = 0.0;
    for (int k = 0; k < w->count; ++k) {
        int s = w->idx[w->start + k];
        dataset_get(w->ds, s, w->xn, w->xt, w->yp, w->Y);
        model_forward(&w->model, w->xn, w->xt, w->yp);
        /* raw (un-averaged) gradients; the 1/bs scale is applied at reduction */
        w->loss += model_loss(w->model.pred, w->Y, w->model.pgf.gate, w->lambda,
                              w->dpred, w->dgate);
        model_backward(&w->model, w->dpred, w->dgate);
    }
    w->dseed = nn_dropout_state();   /* advance so masks differ across batches */
    return NULL;
}

double partrainer_step_grads(ParTrainer *pt, ParamList *master,
                             const Dataset *ds, const int *idx,
                             int b, int bs, float lambda) {
    /* split the batch into contiguous shards (remainder spread over the first
     * few workers) */
    int base = bs / pt->n, rem = bs % pt->n, off = 0;
    int active = 0;
    for (int i = 0; i < pt->n; ++i) {
        Worker *w = &pt->w[i];
        w->count  = base + (i < rem ? 1 : 0);
        w->start  = b + off;
        w->ds     = ds; w->idx = idx; w->lambda = lambda;
        off += w->count;
    }
    /* dispatch: the first worker runs on the calling thread to save one spawn */
    for (int i = 1; i < pt->n; ++i) {
        if (pt->w[i].count == 0) continue;
        int rc = pthread_create(&pt->w[i].tid, NULL, worker_run, &pt->w[i]);
        assert(rc == 0); (void)rc; ++active;
    }
    worker_run(&pt->w[0]);
    for (int i = 1; i < pt->n; ++i)
        if (pt->w[i].count > 0) pthread_join(pt->w[i].tid, NULL);
    (void)active;

    /* reduce: master->g = (1/bs) * sum_workers replica->g */
    double total = 0.0;
    float inv = 1.0f / (float)bs;
    plist_zero_grad(master);
    for (int i = 0; i < pt->n; ++i) {
        total += pt->w[i].loss;
        ParamList *r = &pt->w[i].pl;
        for (int p = 0; p < master->count; ++p)
            for (int e = 0; e < master->item[p].n; ++e)
                master->item[p].g[e] += r->item[p].g[e];
    }
    for (int p = 0; p < master->count; ++p)
        for (int e = 0; e < master->item[p].n; ++e)
            master->item[p].g[e] *= inv;
    return total / (double)bs;
}
