/*
 * test_backend.c — verify the ACTIVE backend's kernels produce the same numbers as
 * a straightforward CPU reference. Built with OPENCL=1, so mat_* here are the
 * OpenCL implementations; the reference is computed inline in plain C. Every
 * kernel is checked against random inputs to a tight tolerance.
 *
 *   make OPENCL=1 backends/opencl/test_opencl && ./backends/opencl/test_opencl
 */
#include "tensor.h"

#include <stdint.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Deterministic pseudo-random source: a xorshift64 generator seeded with a fixed
 * constant so every run produces the identical inputs (reproducible pass/fail). frand
 * folds the top 53 bits into a double in [0,1), then maps to [-1,1) so test matrices
 * hold a mix of positive and negative values (important for exercising relu/sigmoid). */
static uint64_t rng = 88172645463325252ULL;
static float frand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 11) / 9007199254740992.0) * 2.0f - 1.0f; }

/* Populate a matrix with frand() values. */
static void fill(Mat m) { for (int i = 0; i < m.rows * m.cols; ++i) m.data[i] = frand(); }
/* Worst-case (L-infinity) elementwise error between two flat arrays — the metric each
 * kernel is judged by. Using max |a-b| rather than an average catches a single wrong
 * element that a mean would dilute. */
static float maxabs_diff(const float *a, const float *b, int n) {
    float d = 0; for (int i = 0; i < n; ++i) { float e = fabsf(a[i] - b[i]); if (e > d) d = e; } return d; }

/* Pass threshold. 1e-4 (not exact) because the GPU kernels accumulate in float and use
 * the device's exp(); tiny rounding differences from the CPU reference are expected and
 * acceptable, but a real indexing/logic bug would blow well past this. */
#define TOL 1e-4f

int main(void) {
    /* Non-square, mutually distinct dims (M!=K!=N) on purpose: a kernel that swaps a
     * row/column index or confuses two dimensions would still pass on square matrices,
     * so distinct sizes make such bugs surface as shape mismatches or wrong numbers. */
    const int M = 7, K = 5, N = 6;
    int fail = 0;
    float *ref = NULL;

    /* ---- matmul: C = A@B ----
     * Run the kernel, then recompute the same product with the naive CPU triple loop
     * into `ref` using identical row-major indexing (A[i*K+p], B[p*N+j]), and record
     * the max abs difference d1. Each subsequent block follows this same run-then-verify
     * shape for a different op. */
    Mat A = mat_new(M, K), B = mat_new(K, N), C = mat_new(M, N);
    fill(A); fill(B);
    mat_matmul(A, B, C);
    ref = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float s = 0; for (int p = 0; p < K; ++p) s += A.data[i*K+p] * B.data[p*N+j]; ref[i*N+j] = s; }
    float d1 = maxabs_diff(C.data, ref, M*N); free(ref);

    /* ---- matmul_bt: C = A@Bt, B is [N,K] ----
     * Reference reads Bt row j contiguously (Bt.data[j*K+p]) — the transpose expressed
     * through indexing, matching k_matmul_bt. */
    Mat Bt = mat_new(N, K), C2 = mat_new(M, N);
    fill(A); fill(Bt);
    mat_matmul_bt(A, Bt, C2);
    ref = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float s = 0; for (int p = 0; p < K; ++p) s += A.data[i*K+p] * Bt.data[j*K+p]; ref[i*N+j] = s; }
    float d2 = maxabs_diff(C2.data, ref, M*N); free(ref);

    /* ---- matmul_atb: C = At@B, A is [K,M], B is [K,N] ----
     * Contraction runs over the K rows; reference reads Aa.data[p*M+i] (column i of Aa =
     * row i of Aa^T) and Bb.data[p*N+j], matching k_matmul_atb. */
    Mat Aa = mat_new(K, M), Bb = mat_new(K, N), C3 = mat_new(M, N);
    fill(Aa); fill(Bb);
    mat_matmul_atb(Aa, Bb, C3);
    ref = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
        float s = 0; for (int p = 0; p < K; ++p) s += Aa.data[p*M+i] * Bb.data[p*N+j]; ref[i*N+j] = s; }
    float d3 = maxabs_diff(C3.data, ref, M*N); free(ref);

    /* ---- pointwise: relu, sigmoid, scale, axpy, add_bias ----
     * For each op the reference is computed BEFORE calling the (in-place) kernel, since
     * the kernel overwrites the matrix; then diff the kernel result against the saved
     * reference. n = total element count for these flat elementwise ops. */
    int n = M * N;
    Mat R = mat_new(M, N); fill(R);
    ref = malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) ref[i] = R.data[i] < 0 ? 0 : R.data[i];
    mat_relu(R); float d4 = maxabs_diff(R.data, ref, n); free(ref);

    Mat S = mat_new(M, N); fill(S);
    ref = malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) ref[i] = 1.0f / (1.0f + expf(-S.data[i]));
    mat_sigmoid(S); float d5 = maxabs_diff(S.data, ref, n); free(ref);

    Mat Sc = mat_new(M, N); fill(Sc);
    ref = malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) ref[i] = Sc.data[i] * 2.5f;
    mat_scale(Sc, 2.5f); float d6 = maxabs_diff(Sc.data, ref, n); free(ref);

    Mat Y = mat_new(M, N), X = mat_new(M, N); fill(Y); fill(X);
    ref = malloc(sizeof(float) * n);
    for (int i = 0; i < n; ++i) ref[i] = Y.data[i] + (-0.75f) * X.data[i];
    mat_axpy(Y, -0.75f, X); float d7 = maxabs_diff(Y.data, ref, n); free(ref);

    /* add_bias: per-column bias (bias[j] added to every row of column j). The bias[6]
     * array matches N==6; the reference adds bias[j] to element (i,j), which is exactly
     * the kernel's b[i%cols] mapping for a row-major flat index. */
    Mat Bi = mat_new(M, N); fill(Bi);
    float bias[6]; for (int j = 0; j < N; ++j) bias[j] = frand();
    ref = malloc(sizeof(float) * n);
    for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) ref[i*N+j] = Bi.data[i*N+j] + bias[j];
    mat_add_bias(Bi, bias); float d8 = maxabs_diff(Bi.data, ref, n); free(ref);

    printf("backend kernels vs CPU reference (max abs diff):\n");
    printf("  matmul       %.2e\n  matmul_bt    %.2e\n  matmul_atb   %.2e\n", d1, d2, d3);
    printf("  relu         %.2e\n  sigmoid      %.2e\n  scale        %.2e\n", d4, d5, d6);
    printf("  axpy         %.2e\n  add_bias     %.2e\n", d7, d8);
    /* Reduce the eight per-op errors to a single worst case; the suite fails if ANY op
     * exceeds TOL. `ds` holds d2..d8 (worst is seeded with d1), so the loop scans 7. */
    float worst = d1; float ds[] = {d2,d3,d4,d5,d6,d7,d8};
    for (int i = 0; i < 7; ++i) if (ds[i] > worst) worst = ds[i];
    fail = worst > TOL;
    printf("%s (worst %.2e, tol %.0e)\n", fail ? "FAIL: OpenCL kernels diverge" :
           "backend matches CPU reference", worst, (double)TOL);

    mat_free(&A); mat_free(&B); mat_free(&C); mat_free(&Bt); mat_free(&C2);
    mat_free(&Aa); mat_free(&Bb); mat_free(&C3); mat_free(&R); mat_free(&S);
    mat_free(&Sc); mat_free(&Y); mat_free(&X); mat_free(&Bi);
    return fail;
}
