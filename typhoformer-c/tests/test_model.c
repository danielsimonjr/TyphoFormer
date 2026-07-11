/*
 * test_model.c — finite-difference gradient check of the whole model
 * (PGF + spatio-temporal encoder + autoregressive decoder + loss) on a
 * small configuration. This validates the full forward/backward composition.
 */
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static Model     M;
static ParamList pl;
static Mat       xnum, xtext, yprev, Y;
static float     LAMBDA = 0.1f;

static double loss_only(void) {
    model_forward(&M, xnum, xtext, yprev);
    Mat none = {0, 0, NULL};
    return model_loss(M.pred, Y, M.pgf.gate, LAMBDA, none, none);
}

int main(void) {
    nn_seed(7);
    Config c; c.d_num = 3; c.d_text = 5; c.d_model = 8; c.out_dim = 2;
    c.in_len = 4; c.pred_len = 1; c.d_ff = 16; c.n_heads = 2; c.n_layers = 1;

    plist_init(&pl);
    M = model_new(&c, &pl);

    xnum  = mat_new(c.in_len, c.d_num);
    xtext = mat_new(c.in_len, c.d_text);
    yprev = mat_new(1, c.out_dim);
    Y     = mat_new(c.pred_len, c.out_dim);
    for (int i = 0; i < c.in_len * c.d_num;  ++i) xnum.data[i]  = nn_uniform(-1, 1);
    for (int i = 0; i < c.in_len * c.d_text; ++i) xtext.data[i] = nn_uniform(-1, 1);
    for (int i = 0; i < c.out_dim; ++i)           yprev.data[i] = nn_uniform(-1, 1);
    for (int i = 0; i < c.pred_len * c.out_dim; ++i) Y.data[i]  = nn_uniform(-1, 1);

    /* analytic gradients */
    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);
    model_forward(&M, xnum, xtext, yprev);
    model_loss(M.pred, Y, M.pgf.gate, LAMBDA, dpred, dgate);
    plist_zero_grad(&pl);
    model_backward(&M, dpred, dgate);

    /* finite differences over every parameter */
    const float eps = 1e-3f, floor = 1e-2f;
    float max_err = 0.0f, max_abs = 0.0f; long checked = 0;
    for (int p = 0; p < pl.count; ++p)
        for (int e = 0; e < pl.item[p].n; ++e) {
            float *w = &pl.item[p].v[e], save = *w;
            *w = save + eps; double lp = loss_only();
            *w = save - eps; double lm = loss_only();
            *w = save;
            float g = (float)((lp - lm) / (2.0 * eps)), a = pl.item[p].g[e];
            float abserr = fabsf(g - a);
            float err = abserr / (fabsf(g) + fabsf(a) + floor);
            if (err > max_err) max_err = err;
            if (abserr > max_abs) max_abs = abserr;
            ++checked;
        }

    printf("model params: %ld (checked %ld)\n", plist_num_params(&pl), checked);
    printf("max rel err = %.2e, max abs err = %.2e\n", max_err, max_abs);
    int fail = (max_err >= 2e-2f && max_abs >= 1e-3f);
    printf(fail ? "FAIL: model gradients\n" : "\nmodel gradient check passed\n");

    model_free(&M); plist_free(&pl);
    mat_free(&xnum); mat_free(&xtext); mat_free(&yprev); mat_free(&Y);
    mat_free(&dpred); mat_free(&dgate);
    return fail;
}
