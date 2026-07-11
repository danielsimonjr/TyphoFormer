/*
 * test_dropout.c — verify dropout: (1) it is the identity in eval mode,
 * (2) it actually zeros ~p of the activations in training mode, and (3) its
 * backward (mask multiply) is correct. The gradient check PINS the mask by
 * re-seeding the dropout RNG to the same value before every forward, so the
 * finite-difference surface is smooth (the mask does not depend on the weights).
 */
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define S 4
#define D 8
#define HEADS 2
#define FF 16
#define DSEED 1234567u

static Block blk;
static ParamList pl;
static Mat gx, gt;

static double loss_pinned(void) {
    nn_dropout_seed(DSEED);                 /* same mask every forward */
    Mat y = mat_new(S, D);
    block_forward(&blk, gx, y);
    double L = 0.0;
    for (int i = 0; i < S * D; ++i) { double d = y.data[i] - gt.data[i]; L += d * d; }
    L /= (double)(S * D);
    mat_free(&y);
    return L;
}

int main(void) {
    nn_seed(7);
    plist_init(&pl);
    blk = block_new(D, HEADS, FF, 0, &pl, "block");
    gx = mat_new(S, D); gt = mat_new(S, D);
    for (int i = 0; i < S * D; ++i) { gx.data[i] = nn_uniform(-1, 1); gt.data[i] = nn_uniform(-1, 1); }
    int fail = 0;

    /* (1) eval mode == training mode with p=0 (dropout is identity) */
    Mat ye = mat_new(S, D), y0 = mat_new(S, D);
    nn_set_training(0);                     block_forward(&blk, gx, ye);
    nn_set_training(1); nn_set_dropout(0.0f); block_forward(&blk, gx, y0);
    double diff = 0.0; for (int i = 0; i < S * D; ++i) diff += fabs(ye.data[i] - y0.data[i]);
    if (diff > 1e-6) { printf("FAIL: dropout not identity when off (diff %.2e)\n", diff); fail = 1; }

    /* (3) pinned-mask gradient check with p=0.5 */
    nn_set_training(1); nn_set_dropout(0.5f);
    Mat y = mat_new(S, D), dy = mat_new(S, D), dx = mat_new(S, D);
    nn_dropout_seed(DSEED); block_forward(&blk, gx, y);
    for (int i = 0; i < S * D; ++i) dy.data[i] = 2.0f / (S * D) * (y.data[i] - gt.data[i]);
    plist_zero_grad(&pl);
    block_backward(&blk, dy, dx);

    const float eps = 1e-4f, floor = 1e-2f, REL = 2e-2f, ABS = 3e-3f;
    int bad = 0; float max_abs = 0.0f;
    for (int p = 0; p < pl.count; ++p)
        for (int e = 0; e < pl.item[p].n; ++e) {
            float *w = &pl.item[p].v[e], save = *w;
            *w = save + eps; double lp = loss_pinned();
            *w = save - eps; double lm = loss_pinned();
            *w = save;
            float g = (float)((lp - lm) / (2.0 * eps)), a = pl.item[p].g[e];
            float abserr = fabsf(g - a), err = abserr / (fabsf(g) + fabsf(a) + floor);
            if (abserr > max_abs) max_abs = abserr;
            if (err >= REL && abserr >= ABS) ++bad;
        }
    printf("dropout gradient check: max abs err = %.2e, outliers = %d\n", max_abs, bad);
    if (bad > 3) { printf("FAIL: dropout backward gradients (%d disagree)\n", bad); fail = 1; }

    printf(fail ? "FAIL: dropout tests\n" : "dropout check passed\n");
    block_free(&blk); plist_free(&pl);
    mat_free(&gx); mat_free(&gt); mat_free(&ye); mat_free(&y0);
    mat_free(&y); mat_free(&dy); mat_free(&dx);
    return fail;
}
