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
 * (exactly what partrainer_step_grads does, but on one thread in order). */
static double serial_grads(Model *m, ParamList *pl, const Dataset *ds,
                           const int *idx, int b, int bs, float lambda) {
    Mat xn = mat_new(ds->in_len, ds->d_num), xt = mat_new(ds->in_len, ds->d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(ds->pred_len, 2);
    Mat dpred = mat_new(ds->pred_len, 2), dgate = mat_new(ds->in_len, m->cfg.d_model);
    plist_zero_grad(pl);
    double loss = 0.0;
    for (int k = 0; k < bs; ++k) {
        dataset_get(ds, idx[b + k], xn, xt, yp, Y);
        model_forward(m, xn, xt, yp);
        loss += model_loss(m->pred, Y, m->pgf.gate, lambda, dpred, dgate);
        model_backward(m, dpred, dgate);
    }
    float inv = 1.0f / (float)bs;
    for (int p = 0; p < pl->count; ++p)
        for (int e = 0; e < pl->item[p].n; ++e) pl->item[p].g[e] *= inv;
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    mat_free(&dpred); mat_free(&dgate);
    return loss / bs;
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
    printf(worst_fail ? "FAIL: parallel gradient mismatch\n"
                      : "parallel data-parallel gradient check passed\n");
    return worst_fail;
}
