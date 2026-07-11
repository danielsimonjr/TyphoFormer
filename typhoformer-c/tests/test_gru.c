/*
 * test_gru.c — finite-difference gradient check of the GRU cell over a TWO-STEP
 * rollout. Checks every gate parameter, both step inputs, and the initial hidden
 * state — so it exercises the cross-step hidden-gradient threading (dh from step
 * 1 flowing back into step 0), which is where a recurrent backward usually goes
 * wrong.
 */
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define IN  3
#define HID 5

static GRU        g;
static ParamList  pl;
static Mat        x0, x1, h0, tgt;

/* loss = MSE(h2, tgt) over a 2-step rollout h0 -> h1 -> h2 */
static double loss_only(void) {
    Mat h1 = mat_new(1, HID), h2 = mat_new(1, HID);
    gru_forward(&g, 0, x0, h0, h1);
    gru_forward(&g, 1, x1, h1, h2);
    double L = 0.0;
    for (int i = 0; i < HID; ++i) { double d = h2.data[i] - tgt.data[i]; L += d * d; }
    L /= HID;
    mat_free(&h1); mat_free(&h2);
    return L;
}

static int fd_check(const char *what, float *base, const float *ana, int n) {
    const float eps = 1e-4f, floor = 1e-2f, REL = 2e-2f, ABS = 3e-3f;
    float max_err = 0.0f, max_abs = 0.0f; int bad = 0;
    for (int e = 0; e < n; ++e) {
        float save = base[e];
        base[e] = save + eps; double lp = loss_only();
        base[e] = save - eps; double lm = loss_only();
        base[e] = save;
        float gnum = (float)((lp - lm) / (2.0 * eps));
        float abserr = fabsf(gnum - ana[e]);
        float err = abserr / (fabsf(gnum) + fabsf(ana[e]) + floor);
        if (err > max_err) max_err = err;
        if (abserr > max_abs) max_abs = abserr;
        if (err >= REL && abserr >= ABS) ++bad;
    }
    printf("  %-8s max rel err = %.2e, max abs err = %.2e | outliers = %d\n", what, max_err, max_abs, bad);
    return bad > 2;
}

int main(void) {
    nn_seed(20260711);
    printf("[GRU gradient check, 2-step rollout, in=%d hid=%d]\n", IN, HID);
    plist_init(&pl);
    g = gru_new(IN, HID, 2, &pl, "gru");

    x0 = mat_new(1, IN); x1 = mat_new(1, IN); h0 = mat_new(1, HID); tgt = mat_new(1, HID);
    for (int i = 0; i < IN;  ++i) { x0.data[i] = nn_uniform(-1, 1); x1.data[i] = nn_uniform(-1, 1); }
    for (int i = 0; i < HID; ++i) { h0.data[i] = nn_uniform(-1, 1); tgt.data[i] = nn_uniform(-1, 1); }

    /* analytic grads */
    Mat h1 = mat_new(1, HID), h2 = mat_new(1, HID);
    gru_forward(&g, 0, x0, h0, h1);
    gru_forward(&g, 1, x1, h1, h2);
    Mat dh2 = mat_new(1, HID), dh1 = mat_new(1, HID), dh0 = mat_new(1, HID);
    Mat dx1 = mat_new(1, IN), dx0 = mat_new(1, IN);
    for (int i = 0; i < HID; ++i) dh2.data[i] = 2.0f / HID * (h2.data[i] - tgt.data[i]);
    plist_zero_grad(&pl);
    gru_backward(&g, 1, dh2, dx1, dh1);          /* step 1 -> dx1, dh1 */
    gru_backward(&g, 0, dh1, dx0, dh0);          /* step 0 -> dx0, dh0 (initial state) */

    int fail = 0;
    for (int p = 0; p < pl.count; ++p)
        if (fd_check("param", pl.item[p].v, pl.item[p].g, pl.item[p].n)) fail = 1;
    fail |= fd_check("dx0", x0.data, dx0.data, IN);
    fail |= fd_check("dx1", x1.data, dx1.data, IN);
    fail |= fd_check("dh0", h0.data, dh0.data, HID);

    if (!fail) printf("  ok\n\nGRU gradient check passed\n");
    else       printf("  FAIL: GRU gradients disagree\n");

    gru_free(&g); plist_free(&pl);
    mat_free(&x0); mat_free(&x1); mat_free(&h0); mat_free(&tgt);
    mat_free(&h1); mat_free(&h2); mat_free(&dh2); mat_free(&dh1); mat_free(&dh0);
    mat_free(&dx1); mat_free(&dx0);
    return fail;
}
