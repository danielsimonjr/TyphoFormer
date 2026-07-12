/*
 * test_parallel.c — verify that data-parallel multicore training produces the
 * same minibatch gradient as the serial path (up to floating-point summation
 * order), and that the loss agrees. This is the correctness proof for
 * src/parallel.c: if the reduced gradient matches, multi-threaded training is
 * doing exactly what single-threaded training does, only faster.
 */
#include "parallel.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build a small self-contained prewindowed dataset (no data files needed). */
static Dataset synth_dataset(int n, int in_len, int pred_len, int d_num, int d_text) {
    Dataset d; memset(&d, 0, sizeof d);
    d.n_samples = n; d.in_len = in_len; d.pred_len = pred_len;
    d.d_num = d_num; d.d_text = d_text; d.prewindowed = 1;
    /* identity coordinate stats: memset left cstd at 0, and cnorm() divides by
     * cstd — without this every coordinate became inf/NaN and the whole test
     * compared NaN losses (which pass any '>' threshold check vacuously). */
    d.cstd[0] = d.cstd[1] = 1.0f;
    int F = d_num + d_text;
    d.win_in = (float *)malloc((size_t)n * in_len * F * sizeof(float));
    d.win_tg = (float *)malloc((size_t)n * pred_len * 2 * sizeof(float));
    for (long i = 0; i < (long)n * in_len * F; ++i)  d.win_in[i] = nn_uniform(-1, 1);
    for (long i = 0; i < (long)n * pred_len * 2; ++i) d.win_tg[i] = nn_uniform(-1, 1);
    return d;
}

/* Serial reference: raw grads summed over the batch into pl, then scaled 1/bs
 * (exactly what partrainer_step_grads does, but on one thread in order). Feeds
 * the same per-sample aux inputs the workers feed (neighbours, seed velocity)
 * so the decoder-variant models see identical inputs on both paths. */
static double serial_grads(Model *m, ParamList *pl, const Dataset *ds,
                           const int *idx, int b, int bs, float lambda) {
    Mat xn = mat_new(ds->in_len, ds->d_num), xt = mat_new(ds->in_len, ds->d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(ds->pred_len, 2);
    Mat dpred = mat_new(ds->pred_len, 2), dgate = mat_new(ds->in_len, m->cfg.d_model);
    Mat nbr = mat_new(TF_NBR_K, TF_NBR_NF), vel = mat_new(1, 2);
    plist_zero_grad(pl);
    double loss = 0.0;
    for (int k = 0; k < bs; ++k) {
        dataset_get(ds, idx[b + k], xn, xt, yp, Y);
        int nc; dataset_neighbors(ds, idx[b + k], nbr, &nc);
        model_set_neighbors(m, nbr, nc);
        dataset_seed_velocity(ds, idx[b + k], vel);
        model_set_seed_velocity(m, vel);
        model_forward(m, xn, xt, yp);
        loss += model_loss(m->pred, Y, m->pgf.gate, lambda, dpred, dgate);
        model_backward(m, dpred, dgate);
    }
    float inv = 1.0f / (float)bs;
    for (int p = 0; p < pl->count; ++p)
        for (int e = 0; e < pl->item[p].n; ++e) pl->item[p].g[e] *= inv;
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    mat_free(&dpred); mat_free(&dgate); mat_free(&nbr); mat_free(&vel);
    return loss / bs;
}

/* Compare the serial reference gradient against the parallel reduction for the
 * CURRENT model architecture (whatever the model_set_* globals say). Returns 1
 * on mismatch. `label` names the variant in the output. */
static int check_equivalence(const Config *c, const Dataset *ds, const int *idx,
                             int bs, const char *label) {
    const float lambda = 0.1f;
    ParamList mpl; plist_init(&mpl);
    Model master = model_new(c, &mpl);
    double sloss = serial_grads(&master, &mpl, ds, idx, 0, bs, lambda);
    long np = plist_num_params(&mpl);
    float *gserial = (float *)malloc((size_t)np * sizeof(float));
    long o = 0;
    for (int p = 0; p < mpl.count; ++p)
        for (int e = 0; e < mpl.item[p].n; ++e) gserial[o++] = mpl.item[p].g[e];

    int worst_fail = 0;
    for (int nw = 2; nw <= 3; ++nw) {
        ParTrainer *pt = partrainer_new(c, nw);
        partrainer_broadcast(pt, &mpl);
        double ploss = partrainer_step_grads(pt, &mpl, ds, idx, 0, bs, lambda);
        float max_abs = 0.0f;
        o = 0;
        for (int p = 0; p < mpl.count; ++p)
            for (int e = 0; e < mpl.item[p].n; ++e) {
                float d = fabsf(mpl.item[p].g[e] - gserial[o++]);
                if (d > max_abs) max_abs = d;
            }
        double lerr = fabs(ploss - sloss);
        int fail = !(max_abs <= 1e-4f) || !(lerr <= 1e-4) || isnan(sloss) || isnan(ploss);
        printf("%-11s threads=%d | loss |d|=%.2e | grad max abs=%.2e  %s\n",
               label, nw, lerr, max_abs, fail ? "FAIL" : "ok");
        if (fail) worst_fail = 1;
        partrainer_free(pt);
    }
    free(gserial);
    model_free(&master); plist_free(&mpl);
    return worst_fail;
}

/* Reset every architecture toggle to the defaults. */
static void reset_variants(void) {
    model_set_delta(0); model_set_cv(0); model_set_gru(0); model_set_xattn(0);
    model_set_co_spatial(0);
}

int main(void) {
    nn_seed(7);
    Config c = config_default();
    c.d_num = 4; c.d_text = 8; c.in_len = 4; c.pred_len = 2;
    c.d_model = 16; c.d_ff = 32; c.n_layers = 1; c.n_heads = 2;

    Dataset ds = synth_dataset(20, c.in_len, c.pred_len, c.d_num, c.d_text);
    int idx[20]; for (int i = 0; i < 20; ++i) idx[i] = i;
    const int bs = 12; const float lambda = 0.1f;

    /* master model holds the shared weights */
    ParamList mpl; plist_init(&mpl);
    Model master = model_new(&c, &mpl);

    /* serial reference gradient */
    double sloss = serial_grads(&master, &mpl, &ds, idx, 0, bs, lambda);
    long np = plist_num_params(&mpl);
    float *gserial = (float *)malloc((size_t)np * sizeof(float));
    long o = 0;
    for (int p = 0; p < mpl.count; ++p)
        for (int e = 0; e < mpl.item[p].n; ++e) gserial[o++] = mpl.item[p].g[e];

    int worst_fail = 0;
    for (int nw = 2; nw <= 4; ++nw) {
        ParTrainer *pt = partrainer_new(&c, nw);
        partrainer_broadcast(pt, &mpl);   /* replicas get the master weights */
        double ploss = partrainer_step_grads(pt, &mpl, &ds, idx, 0, bs, lambda);

        float max_rel = 0.0f, max_abs = 0.0f;
        o = 0;
        for (int p = 0; p < mpl.count; ++p)
            for (int e = 0; e < mpl.item[p].n; ++e) {
                float a = mpl.item[p].g[e], b = gserial[o++];
                float abserr = fabsf(a - b);
                float rel = abserr / (fabsf(a) + fabsf(b) + 1e-6f);
                if (abserr > max_abs) max_abs = abserr;
                if (rel > max_rel)    max_rel = rel;
            }
        double lerr = fabs(ploss - sloss);
        /* NaN-proof comparison: !(x <= tol) is true for x > tol AND for NaN,
         * so a NaN loss or gradient can never slip through as a pass. */
        int fail = !(max_abs <= 1e-4f) || !(lerr <= 1e-4) || isnan(sloss) || isnan(ploss);
        printf("threads=%d | loss serial=%.6f parallel=%.6f (|d|=%.2e) | grad max abs=%.2e rel=%.2e  %s\n",
               nw, sloss, ploss, lerr, max_abs, max_rel, fail ? "FAIL" : "ok");
        if (fail) worst_fail = 1;
        partrainer_free(pt);
    }

    free(gserial);
    model_free(&master); plist_free(&mpl); dataset_free(&ds);

    /* ---- decoder-variant / co-spatial equivalence on the real dataset ----
     * The cv/gru/xattn decoders consume a per-sample seed velocity and
     * --co_spatial consumes per-sample neighbour tables; both are fed by the
     * workers (parallel.c) and must match the serial loop. The synthetic
     * prewindowed dataset above carries neither, so this phase needs the real
     * CSV; skip (still passing) when it is not available. */
    FILE *probe = fopen("../HURDAT_2new_3000.csv", "r");
    if (!probe) {
        printf(worst_fail ? "FAIL: parallel gradient mismatch\n"
                          : "parallel data-parallel gradient check passed "
                            "(variant phase SKIPPED: no ../ dataset)\n");
        return worst_fail;
    }
    fclose(probe);

    Config vc = config_default();
    vc.in_len = 6; vc.pred_len = 2;                 /* small but multi-step rollout */
    vc.d_model = 16; vc.d_ff = 32; vc.n_layers = 1; vc.n_heads = 2;
    Dataset rds = dataset_load("../HURDAT_2new_3000.csv", "../embedding_chunks",
                               vc.in_len, vc.pred_len);
    dataset_build_neighbors(&rds);                  /* co_spatial's aux tables */
    dataset_standardize(&rds);
    vc.d_num = rds.d_num; vc.d_text = rds.d_text;
    int ridx[12]; for (int i = 0; i < 12; ++i) ridx[i] = i * 97;   /* spread across storms */

    struct { const char *label; void (*set)(int); } variants[] = {
        { "cv",         model_set_cv },
        { "gru",        model_set_gru },
        { "xattn",      model_set_xattn },
        { "co_spatial", model_set_co_spatial },
    };
    for (unsigned v = 0; v < sizeof variants / sizeof variants[0]; ++v) {
        reset_variants();
        nn_seed(11 + v);                             /* fresh, deterministic weights */
        variants[v].set(1);
        if (check_equivalence(&vc, &rds, ridx, 12, variants[v].label)) worst_fail = 1;
    }
    reset_variants();
    dataset_free(&rds);

    printf(worst_fail ? "FAIL: parallel gradient mismatch\n"
                      : "parallel data-parallel gradient check passed\n");
    return worst_fail;
}
