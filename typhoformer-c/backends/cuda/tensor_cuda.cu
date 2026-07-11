/*
 * tensor_cuda.cu — a CUDA reference backend for TyphoFormer-C.
 *
 * This translation unit implements the SAME kernel set declared in
 * include/tensor.h (the backend contract documented in include/backend.h),
 * but keeps Mat.data resident in device (GPU) memory. Link this file in place
 * of src/tensor.c and the entire model runs on the GPU — no layer code changes,
 * because every layer is written only in terms of these primitives.
 *
 * STATUS: compile-ready reference. It requires the CUDA toolkit (nvcc) which is
 * not part of the standard-library-only default build, so it is NOT compiled or
 * run by the top-level Makefile and has not been executed in this repository's
 * CI. It is here to (a) prove the seam is real — the CPU model is expressed
 * purely through these ~13 functions — and (b) give an engineer a correct,
 * readable starting point. Build it with backends/cuda/Makefile (see there).
 *
 * Design choices kept deliberately simple for readability over peak FLOPS:
 *   - Mat.data is a device pointer (cudaMalloc). rows/cols live on the host.
 *   - One thread per output element for GEMM (no shared-memory tiling). This is
 *     correct and easy to read; the natural next exercise is a tiled GEMM.
 *   - Reductions (mat_colsum) use atomicAdd for brevity.
 * Every function matches the host-visible signature in tensor.h exactly, so the
 * rest of the program is oblivious to where the data lives.
 */

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

extern "C" {
#include "tensor.h"
}

/* ---- error plumbing -------------------------------------------------- */

extern "C" _Noreturn void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess)                                             \
            die("CUDA error %s at %s:%d", cudaGetErrorString(err__),          \
                __FILE__, __LINE__);                                           \
    } while (0)

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* ---- memory ---------------------------------------------------------- */

extern "C" Mat mat_new(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    CUDA_CHECK(cudaMalloc((void **)&m.data, (size_t)rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(m.data, 0, (size_t)rows * cols * sizeof(float)));
    return m;
}

extern "C" void mat_free(Mat *m) {
    if (m && m->data) {
        cudaFree(m->data);
        m->data = NULL;
        m->rows = m->cols = 0;
    }
}

extern "C" void mat_zero(Mat m) {
    CUDA_CHECK(cudaMemset(m.data, 0, (size_t)m.rows * m.cols * sizeof(float)));
}

extern "C" void mat_copy(Mat dst, const Mat src) {
    CUDA_CHECK(cudaMemcpy(dst.data, src.data,
                          (size_t)src.rows * src.cols * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

/* ---- GEMM: one thread per output element ----------------------------- */

__global__ void k_matmul(const float *A, const float *B, float *C,
                         int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[i * k + p] * B[p * n + j];
    C[i * n + j] = acc;
}

__global__ void k_matmul_bt(const float *A, const float *B, float *C,
                            int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[i * k + p] * B[j * k + p];
    C[i * n + j] = acc;
}

__global__ void k_matmul_atb(const float *A, const float *B, float *C,
                             int k, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[p * m + i] * B[p * n + j];
    C[i * n + j] = acc;
}

static void launch_gemm(void (*kern)(const float *, const float *, float *, int, int, int),
                        const Mat A, const Mat B, Mat out, int a1, int a2, int a3) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(out.cols, block.x), ceil_div(out.rows, block.y));
    kern<<<grid, block>>>(A.data, B.data, out.data, a1, a2, a3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void mat_matmul(const Mat A, const Mat B, Mat out) {
    launch_gemm(k_matmul, A, B, out, A.rows, A.cols, B.cols);
}
extern "C" void mat_matmul_bt(const Mat A, const Mat B, Mat out) {
    launch_gemm(k_matmul_bt, A, B, out, A.rows, A.cols, B.rows);
}
extern "C" void mat_matmul_atb(const Mat A, const Mat B, Mat out) {
    launch_gemm(k_matmul_atb, A, B, out, A.rows, A.cols, B.cols);
}

/* ---- bias / reductions ---------------------------------------------- */

__global__ void k_add_bias(float *m, const float *bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    m[idx] += bias[idx % cols];
}

__global__ void k_colsum(const float *m, float *out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    atomicAdd(&out[idx % cols], m[idx]);
}

extern "C" void mat_add_bias(Mat m, const float *bias) {
    /* bias is expected device-resident (same address space as m.data). */
    int n = m.rows * m.cols, t = 256;
    k_add_bias<<<ceil_div(n, t), t>>>(m.data, bias, m.rows, m.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void mat_colsum(const Mat m, float *out) {
    CUDA_CHECK(cudaMemset(out, 0, (size_t)m.cols * sizeof(float)));
    int n = m.rows * m.cols, t = 256;
    k_colsum<<<ceil_div(n, t), t>>>(m.data, out, m.rows, m.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ---- pointwise / scalar --------------------------------------------- */

__global__ void k_relu(float *m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && m[i] < 0.0f) m[i] = 0.0f;
}
__global__ void k_sigmoid(float *m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] = 1.0f / (1.0f + expf(-m[i]));
}
__global__ void k_scale(float *m, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) m[i] *= s;
}
__global__ void k_axpy(float *y, float a, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

extern "C" void mat_relu(Mat m) {
    int n = m.rows * m.cols, t = 256;
    k_relu<<<ceil_div(n, t), t>>>(m.data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void mat_sigmoid(Mat m) {
    int n = m.rows * m.cols, t = 256;
    k_sigmoid<<<ceil_div(n, t), t>>>(m.data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void mat_scale(Mat m, float s) {
    int n = m.rows * m.cols, t = 256;
    k_scale<<<ceil_div(n, t), t>>>(m.data, s, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
extern "C" void mat_axpy(Mat y, float a, const Mat x) {
    int n = y.rows * y.cols, t = 256;
    k_axpy<<<ceil_div(n, t), t>>>(y.data, a, x.data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
