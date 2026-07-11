/*
 * test_tensor.c — unit tests for the tensor core:
 *   1. exact matmul / matmul_bt / matmul_atb against hand-computed values
 *   2. a finite-difference gradient check through  linear -> relu -> MSE,
 *      validating the backward primitives used by every layer.
 */
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_fail = 0;
#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (!(cond)) { printf("  FAIL: %s\n", (msg)); ++g_fail; } \
        else         { printf("  ok:   %s\n", (msg)); }         \
    } while (0)

static int approx(float a, float b, float tol) { return fabsf(a - b) <= tol; }

/* ---- 1. exact matmul checks ------------------------------------------ */
static void test_matmul_exact(void) {
    printf("[matmul exact]\n");
    Mat A = mat_new(2, 2), B = mat_new(2, 2), C = mat_new(2, 2);
    float a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8};
    for (int i = 0; i < 4; ++i) { A.data[i] = a[i]; B.data[i] = b[i]; }

    mat_matmul(A, B, C);           /* [[19,22],[43,50]] */
    CHECK(approx(C.data[0], 19, 1e-5f) && approx(C.data[1], 22, 1e-5f) &&
          approx(C.data[2], 43, 1e-5f) && approx(C.data[3], 50, 1e-5f),
          "A @ B");

    mat_matmul_bt(A, B, C);        /* A @ B^T = [[17,23],[39,53]] */
    CHECK(approx(C.data[0], 17, 1e-5f) && approx(C.data[1], 23, 1e-5f) &&
          approx(C.data[2], 39, 1e-5f) && approx(C.data[3], 53, 1e-5f),
          "A @ B^T");

    mat_matmul_atb(A, B, C);       /* A^T @ B = [[26,30],[38,44]] */
    CHECK(approx(C.data[0], 26, 1e-5f) && approx(C.data[1], 30, 1e-5f) &&
          approx(C.data[2], 38, 1e-5f) && approx(C.data[3], 44, 1e-5f),
          "A^T @ B");

    mat_free(&A); mat_free(&B); mat_free(&C);
}

/* ---- 2. gradient check:  a = relu(x W^T + b);  L = mean((a - t)^2) ---- */
#define T 4
#define IN 5
#define OUT 3

static float frand(void) { return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f; }

/* Forward pass; returns scalar loss. */
static double forward(const Mat x, const Mat W, const float *b, const Mat t) {
    Mat h = mat_new(T, OUT);
    mat_matmul_bt(x, W, h);        /* x[T,IN] @ W^T[IN,OUT] -> h[T,OUT] */
    mat_add_bias(h, b);
    mat_relu(h);
    double loss = 0.0;
    for (int i = 0; i < T * OUT; ++i) {
        double d = (double)h.data[i] - (double)t.data[i];
        loss += d * d;
    }
    loss /= (double)(T * OUT);
    mat_free(&h);
    return loss;
}

static void test_gradcheck(void) {
    printf("[gradient check: linear -> relu -> mse]\n");
    srand(1234);
    Mat x = mat_new(T, IN), W = mat_new(OUT, IN), t = mat_new(T, OUT);
    float b[OUT];
    for (int i = 0; i < T * IN; ++i)  x.data[i] = frand();
    for (int i = 0; i < OUT * IN; ++i) W.data[i] = frand();
    for (int i = 0; i < OUT; ++i)      b[i]      = frand();
    for (int i = 0; i < T * OUT; ++i)  t.data[i] = frand();

    /* analytic gradients */
    Mat h = mat_new(T, OUT), dh = mat_new(T, OUT);
    mat_matmul_bt(x, W, h);
    mat_add_bias(h, b);
    Mat a = mat_new(T, OUT);
    mat_copy(a, h);
    mat_relu(a);
    for (int i = 0; i < T * OUT; ++i) {
        float da = 2.0f / (T * OUT) * (a.data[i] - t.data[i]);
        dh.data[i] = (h.data[i] > 0.0f) ? da : 0.0f;   /* relu' */
    }
    Mat dW = mat_new(OUT, IN), dx = mat_new(T, IN);
    float db[OUT];
    mat_matmul_atb(dh, x, dW);     /* dW = dh^T @ x */
    mat_colsum(dh, db);            /* db = sum_rows dh */
    mat_matmul(dh, W, dx);         /* dx = dh @ W */

    /* numeric gradients via central differences */
    const float eps = 1e-3f;
    float max_err = 0.0f;

    #define NUMERIC(store, ana)                                          \
        do {                                                             \
            float save = *(store);                                      \
            *(store) = save + eps; double lp = forward(x, W, b, t);     \
            *(store) = save - eps; double lm = forward(x, W, b, t);     \
            *(store) = save;                                            \
            float g = (float)((lp - lm) / (2.0 * eps));                 \
            float err = fabsf(g - (ana)) / (fabsf(g) + 1e-3f);          \
            if (err > max_err) max_err = err;                          \
        } while (0)

    for (int i = 0; i < OUT * IN; ++i) NUMERIC(&W.data[i], dW.data[i]);
    for (int i = 0; i < OUT; ++i)      NUMERIC(&b[i],      db[i]);
    for (int i = 0; i < T * IN; ++i)   NUMERIC(&x.data[i], dx.data[i]);
    #undef NUMERIC

    printf("  max relative grad error = %.2e\n", max_err);
    CHECK(max_err < 2e-2f, "analytic grads match finite differences");

    mat_free(&x); mat_free(&W); mat_free(&t);
    mat_free(&h); mat_free(&dh); mat_free(&a);
    mat_free(&dW); mat_free(&dx);
}

int main(void) {
    test_matmul_exact();
    test_gradcheck();
    if (g_fail) { printf("\n%d check(s) FAILED\n", g_fail); return 1; }
    printf("\nall tensor tests passed\n");
    return 0;
}
