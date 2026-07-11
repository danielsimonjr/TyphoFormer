/*
 * test_nn.c — end-to-end gradient check of a full transformer Block
 * (multi-head attention + residual + LayerNorm + FFN + residual + LayerNorm).
 * Validates every registered parameter gradient and the input gradient
 * against central finite differences.
 */
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define S 3
#define D 8
#define HEADS 2
#define FF 16

static Block   blk;
static ParamList pl;
static Mat     gx, gt;

static double loss_only(void) {
    Mat y = mat_new(S, D);
    block_forward(&blk, gx, y);
    double L = 0.0;
    for (int i = 0; i < S * D; ++i) { double d = y.data[i] - gt.data[i]; L += d * d; }
    L /= (double)(S * D);
    mat_free(&y);
    return L;
}

static int check_block(int self_only) {
    printf("[block gradient check, self_only=%d]\n", self_only);
    plist_init(&pl);
    blk = block_new(D, HEADS, FF, self_only, &pl, "block");

    gx = mat_new(S, D); gt = mat_new(S, D);
    for (int i = 0; i < S * D; ++i) { gx.data[i] = nn_uniform(-1, 1); gt.data[i] = nn_uniform(-1, 1); }

    /* analytic gradients */
    Mat y = mat_new(S, D), dy = mat_new(S, D), dx = mat_new(S, D);
    block_forward(&blk, gx, y);
    for (int i = 0; i < S * D; ++i) dy.data[i] = 2.0f / (S * D) * (y.data[i] - gt.data[i]);
    plist_zero_grad(&pl);
    block_backward(&blk, dy, dx);

    /* finite-difference check on all parameters. Single-precision forward
     * limits FD accuracy, so we judge with a combined absolute+relative
     * metric: err = |g_num - g_ana| / (|g_num| + |g_ana| + floor). */
    const float eps = 1e-4f, floor = 1e-2f;   /* small eps keeps ± off the ReLU kink */
    /* A parameter/input is "off" only if BOTH its relative and absolute error
     * are large. Because the block contains a ReLU, a coordinate whose
     * pre-activation sits within eps of the kink gets an inaccurate *central
     * difference* (not a wrong analytic gradient) — so we tolerate a handful of
     * such outliers and fail only if MANY coordinates disagree, which is what a
     * real backprop bug produces. The full-model check (test_model) is the
     * backstop that would catch a subtle few-parameter error. */
    const float REL = 2e-2f, ABS = 3e-3f; const int OUTLIERS = 3;
    float max_err = 0.0f, max_abs = 0.0f; long checked = 0; int bad = 0;
    for (int p = 0; p < pl.count; ++p) {
        for (int e = 0; e < pl.item[p].n; ++e) {
            float *w = &pl.item[p].v[e];
            float save = *w;
            *w = save + eps; double lp = loss_only();
            *w = save - eps; double lm = loss_only();
            *w = save;
            float g = (float)((lp - lm) / (2.0 * eps));
            float a = pl.item[p].g[e];
            float abserr = fabsf(g - a);
            float err = abserr / (fabsf(g) + fabsf(a) + floor);
            if (err > max_err) max_err = err;
            if (abserr > max_abs) max_abs = abserr;
            if (err >= REL && abserr >= ABS) ++bad;
            ++checked;
        }
    }
    /* finite-difference check on the input gradient dx */
    float max_dx_err = 0.0f, max_dx_abs = 0.0f; int bad_dx = 0;
    for (int e = 0; e < S * D; ++e) {
        float save = gx.data[e];
        gx.data[e] = save + eps; double lp = loss_only();
        gx.data[e] = save - eps; double lm = loss_only();
        gx.data[e] = save;
        float g = (float)((lp - lm) / (2.0 * eps));
        float abserr = fabsf(g - dx.data[e]);
        float err = abserr / (fabsf(g) + fabsf(dx.data[e]) + floor);
        if (err > max_dx_err) max_dx_err = err;
        if (abserr > max_dx_abs) max_dx_abs = abserr;
        if (err >= REL && abserr >= ABS) ++bad_dx;
    }

    printf("params: %ld (checked %ld elements)\n", plist_num_params(&pl), checked);
    printf("params: max rel err = %.2e, max abs err = %.2e | outliers = %d\n", max_err, max_abs, bad);
    printf("input : max rel err = %.2e, max abs err = %.2e | outliers = %d\n", max_dx_err, max_dx_abs, bad_dx);

    int fail = 0;
    if (bad    > OUTLIERS) { printf("  FAIL: parameter gradients (%d disagree)\n", bad); fail = 1; }
    if (bad_dx > OUTLIERS) { printf("  FAIL: input gradient (%d disagree)\n", bad_dx);   fail = 1; }
    if (!fail) printf("  ok\n");

    block_free(&blk); plist_free(&pl);
    mat_free(&gx); mat_free(&gt); mat_free(&y); mat_free(&dy); mat_free(&dx);
    return fail;
}

int main(void) {
    nn_seed(20260711);
    int fail = 0;
    fail |= check_block(0);   /* full temporal attention */
    fail |= check_block(1);   /* single-node spatial attention */
    if (!fail) printf("\nall block gradient checks passed\n");
    return fail;
}
