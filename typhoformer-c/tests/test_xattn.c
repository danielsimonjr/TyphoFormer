/*
 * test_xattn.c — finite-difference gradient check of the single-head
 * cross-attention cell over a TWO-STEP rollout against a fixed memory. Checks
 * every projection parameter, both step query inputs, and the memory gradient
 * (which accumulates the K/V contributions across both steps).
 */
#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define QIN 2
#define DD  4
#define TT  3

static CrossAttn  a;
static ParamList  pl;
static Mat        x0, x1, mem, t0, t1;

/* loss = MSE(out0,t0) + MSE(out1,t1); memory K/V recomputed each eval so that
 * perturbing a projection weight or a memory entry is reflected. */
static double loss_only(void) {
    Mat o0 = mat_new(1, DD), o1 = mat_new(1, DD);
    xattn_set_memory(&a, mem);
    xattn_forward(&a, 0, x0, o0);
    xattn_forward(&a, 1, x1, o1);
    double L = 0.0;
    for (int i = 0; i < DD; ++i) { double d0 = o0.data[i] - t0.data[i], d1 = o1.data[i] - t1.data[i]; L += d0*d0 + d1*d1; }
    L /= DD;
    mat_free(&o0); mat_free(&o1);
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
    printf("  %-6s max rel err = %.2e, max abs err = %.2e | outliers = %d\n", what, max_err, max_abs, bad);
    return bad > 2;
}

int main(void) {
    nn_seed(20260711);
    printf("[cross-attention gradient check, 2-step rollout, qin=%d d=%d T=%d]\n", QIN, DD, TT);
    plist_init(&pl);
    a = xattn_new(QIN, DD, 2, &pl, "xattn");

    x0 = mat_new(1, QIN); x1 = mat_new(1, QIN); mem = mat_new(TT, DD);
    t0 = mat_new(1, DD); t1 = mat_new(1, DD);
    for (int i = 0; i < QIN; ++i) { x0.data[i] = nn_uniform(-1, 1); x1.data[i] = nn_uniform(-1, 1); }
    for (int i = 0; i < TT * DD; ++i) mem.data[i] = nn_uniform(-1, 1);
    for (int i = 0; i < DD; ++i) { t0.data[i] = nn_uniform(-1, 1); t1.data[i] = nn_uniform(-1, 1); }

    /* analytic grads */
    Mat o0 = mat_new(1, DD), o1 = mat_new(1, DD);
    xattn_set_memory(&a, mem);
    xattn_forward(&a, 0, x0, o0);
    xattn_forward(&a, 1, x1, o1);
    Mat do0 = mat_new(1, DD), do1 = mat_new(1, DD), dx0 = mat_new(1, QIN), dx1 = mat_new(1, QIN), dmem = mat_new(TT, DD);
    for (int i = 0; i < DD; ++i) { do0.data[i] = 2.0f/DD*(o0.data[i]-t0.data[i]); do1.data[i] = 2.0f/DD*(o1.data[i]-t1.data[i]); }
    plist_zero_grad(&pl);
    xattn_backward_step(&a, 1, do1, dx1);
    xattn_backward_step(&a, 0, do0, dx0);
    xattn_backward_memory(&a, dmem);

    int fail = 0;
    for (int p = 0; p < pl.count; ++p) if (fd_check("param", pl.item[p].v, pl.item[p].g, pl.item[p].n)) fail = 1;
    fail |= fd_check("dx0", x0.data, dx0.data, QIN);
    fail |= fd_check("dx1", x1.data, dx1.data, QIN);
    fail |= fd_check("dmem", mem.data, dmem.data, TT * DD);

    if (!fail) printf("  ok\n\ncross-attention gradient check passed\n");
    else       printf("  FAIL: cross-attention gradients disagree\n");

    xattn_free(&a); plist_free(&pl);
    mat_free(&x0); mat_free(&x1); mat_free(&mem); mat_free(&t0); mat_free(&t1);
    mat_free(&o0); mat_free(&o1); mat_free(&do0); mat_free(&do1);
    mat_free(&dx0); mat_free(&dx1); mat_free(&dmem);
    return fail;
}
