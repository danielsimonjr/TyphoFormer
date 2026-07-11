/*
 * test_module.c — gradient-check a Sequential built from pluggable Modules,
 * proving the extension interface composes and backprops correctly.
 */
#include "module.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define S 3
#define D 8
#define HEADS 2
#define FF 16

static Sequential seq;
static ParamList  pl;
static Mat        gx, gt;

static double loss_only(void) {
    Mat y = mat_new(S, D);
    seq_forward(&seq, gx, y);
    double L = 0.0;
    for (int i = 0; i < S * D; ++i) { double d = y.data[i] - gt.data[i]; L += d * d; }
    L /= (double)(S * D);
    mat_free(&y);
    return L;
}

int main(void) {
    nn_seed(42);
    plist_init(&pl);
    seq_init(&seq, S, D);
    seq_add(&seq, module_block(D, HEADS, FF, 0, &pl, "temporal"));  /* full attention */
    seq_add(&seq, module_block(D, HEADS, FF, 1, &pl, "spatial"));   /* single-node   */

    gx = mat_new(S, D); gt = mat_new(S, D);
    for (int i = 0; i < S * D; ++i) { gx.data[i] = nn_uniform(-1, 1); gt.data[i] = nn_uniform(-1, 1); }

    Mat y = mat_new(S, D), dy = mat_new(S, D), dx = mat_new(S, D);
    seq_forward(&seq, gx, y);
    for (int i = 0; i < S * D; ++i) dy.data[i] = 2.0f / (S * D) * (y.data[i] - gt.data[i]);
    plist_zero_grad(&pl);
    seq_backward(&seq, dy, dx);

    const float eps = 1e-3f, floor = 1e-2f;
    float max_err = 0.0f, max_abs = 0.0f;
    for (int p = 0; p < pl.count; ++p)
        for (int e = 0; e < pl.item[p].n; ++e) {
            float *w = &pl.item[p].v[e], save = *w;
            *w = save + eps; double lp = loss_only();
            *w = save - eps; double lm = loss_only();
            *w = save;
            float g = (float)((lp - lm) / (2.0 * eps)), a = pl.item[p].g[e];
            float abserr = fabsf(g - a), err = abserr / (fabsf(g) + fabsf(a) + floor);
            if (err > max_err) max_err = err;
            if (abserr > max_abs) max_abs = abserr;
        }

    printf("Sequential(%d modules) params: %ld | max rel err = %.2e, max abs err = %.2e\n",
           seq.n, plist_num_params(&pl), max_err, max_abs);
    int fail = (max_err >= 2e-2f && max_abs >= 1e-3f);
    printf(fail ? "FAIL: module interface gradients\n" : "module interface gradient check passed\n");

    seq_free(&seq); plist_free(&pl);
    mat_free(&gx); mat_free(&gt); mat_free(&y); mat_free(&dy); mat_free(&dx);
    return fail;
}
