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

extern "C" TF_NORETURN void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * Wrap every CUDA runtime call so a non-success status aborts with a decoded message
 * and the exact file:line. CUDA APIs return a cudaError_t; the do/while(0) idiom lets
 * this expand as a single statement at any call site. Kernel *launches* don't return a
 * status directly, so after each launch we CUDA_CHECK(cudaGetLastError()) to catch
 * launch-config errors and cudaDeviceSynchronize() to surface async execution errors. */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess)                                             \
            die("CUDA error %s at %s:%d", cudaGetErrorString(err__),          \
                __FILE__, __LINE__);                                           \
    } while (0)

/* Round-up integer division: number of blocks needed to cover `a` items in groups of
 * `b`. Used everywhere to size the launch grid so it fully covers the data even when
 * the count isn't a multiple of the block size (the kernels then guard the tail). */
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* ---- memory ----------------------------------------------------------
 * Unlike the OpenCL backend (host-resident data), here Mat.data is a DEVICE pointer.
 * These four functions therefore differ from src/tensor.c only in swapping the C stdlib
 * calls for their CUDA equivalents: calloc->cudaMalloc+cudaMemset, free->cudaFree,
 * memset->cudaMemset, memcpy->cudaMemcpy(...DeviceToDevice). The Mat struct (rows/cols
 * on the host, data pointer) is unchanged, so callers can't tell where the data lives —
 * that is the whole point of the backend seam. Note m.data must never be dereferenced
 * on the host; only kernels and cuda* calls may touch it. */

extern "C" Mat mat_new(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    /* Allocate rows*cols floats on the device and zero them (calloc-equivalent). */
    CUDA_CHECK(cudaMalloc((void **)&m.data, (size_t)rows * cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(m.data, 0, (size_t)rows * cols * sizeof(float)));
    return m;
}

extern "C" void mat_free(Mat *m) {
    if (m && m->data) {
        cudaFree(m->data);   /* release device allocation */
        m->data = NULL;
        m->rows = m->cols = 0;
    }
}

extern "C" void mat_zero(Mat m) {
    CUDA_CHECK(cudaMemset(m.data, 0, (size_t)m.rows * m.cols * sizeof(float)));
}

extern "C" void mat_copy(Mat dst, const Mat src) {
    /* Device-to-device copy: both pointers are GPU addresses, no host round-trip. */
    CUDA_CHECK(cudaMemcpy(dst.data, src.data,
                          (size_t)src.rows * src.cols * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

/* ---- GEMM: one thread per output element -----------------------------
 *
 * These three __global__ kernels are the CUDA twins of the OpenCL kernels in KSRC and
 * of the CPU loops in src/tensor.c. The parallelization is identical to OpenCL: each
 * thread computes exactly one output element C[i][j]. The only surface difference is
 * how a thread learns its (i,j): OpenCL uses get_global_id(1)/get_global_id(0), CUDA
 * composes it from block and thread coordinates:
 *     i (row)    = blockIdx.y * blockDim.y + threadIdx.y   (grid Y axis)
 *     j (column) = blockIdx.x * blockDim.x + threadIdx.x   (grid X axis)
 * The `if (i>=m || j>=n) return` guard drops the surplus threads in the last blocks,
 * since the grid is rounded up with ceil_div and rarely divides the matrix evenly.
 * The inner loop and row-major indexing are byte-identical to the CPU/OpenCL versions,
 * which is what guarantees matching numerics. */

/* C = A@B: A[m,k] B[k,n] C[m,n]; contraction over p, standard indexing. */
__global__ void k_matmul(const float *A, const float *B, float *C,
                         int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[i * k + p] * B[p * n + j];
    C[i * n + j] = acc;
}

/* C = A@B^T: B is stored [n,k]; reading B[j*k+p] walks row j of B, i.e. column j of B^T. */
__global__ void k_matmul_bt(const float *A, const float *B, float *C,
                            int m, int k, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[i * k + p] * B[j * k + p];
    C[i * n + j] = acc;
}

/* C = A^T@B: A is stored [k,m]; A[p*m+i] is column i of A = row i of A^T. Here `k` is
 * the contraction dim (row count of A and B), `m` the output height, `n` its width. */
__global__ void k_matmul_atb(const float *A, const float *B, float *C,
                             int k, int m, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) acc += A[p * m + i] * B[p * n + j];
    C[i * n + j] = acc;
}

/*
 * Shared launcher for the GEMM kernels. Launch configuration:
 *   block = 16x16 = 256 threads (a 2D tile of the output),
 *   grid  = ceil_div(out.cols, 16) x ceil_div(out.rows, 16) blocks — enough tiles to
 *           cover the whole output matrix, rounded up (the kernels guard the overhang).
 * Grid X maps to columns and grid Y to rows, matching the index math in each kernel.
 * The <<<grid, block>>> triple-angle syntax is the CUDA kernel-launch operator. After
 * launching we check cudaGetLastError() (launch/config errors) and then synchronize —
 * cudaDeviceSynchronize() blocks until the kernel finishes, which the callers rely on
 * so that `out` is fully computed before they use it (and gives a synchronous, easy-to-
 * read reference; a production path would overlap instead). a1/a2/a3 are the kernel's
 * three int dimension args, whose meaning depends on the specific kernel (see wrappers). */
static void launch_gemm(void (*kern)(const float *, const float *, float *, int, int, int),
                        const Mat A, const Mat B, Mat out, int a1, int a2, int a3) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(out.cols, block.x), ceil_div(out.rows, block.y));
    kern<<<grid, block>>>(A.data, B.data, out.data, a1, a2, a3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* Public GEMM entry points — same signatures and dimension-passing as the CPU/OpenCL
 * backends. Pointers passed to the kernels are device addresses (Mat.data). */
extern "C" void mat_matmul(const Mat A, const Mat B, Mat out) {
    launch_gemm(k_matmul, A, B, out, A.rows, A.cols, B.cols);       /* (m,k,n) */
}
extern "C" void mat_matmul_bt(const Mat A, const Mat B, Mat out) {
    launch_gemm(k_matmul_bt, A, B, out, A.rows, A.cols, B.rows);    /* n = B.rows (B is [n,k]) */
}
extern "C" void mat_matmul_atb(const Mat A, const Mat B, Mat out) {
    launch_gemm(k_matmul_atb, A, B, out, A.rows, A.cols, B.cols);   /* first arg = contraction dim = A.rows */
}

/* ---- bias / reductions ----------------------------------------------
 * These two use a 1D grid: one thread per flattened element, idx = block*blockDim +
 * thread, guarded by idx < rows*cols. Column of element idx is idx % cols (row-major). */

/* Add per-column bias in place: element idx gets bias[column]. Same mapping as the
 * OpenCL k_add_bias and the CPU mat_add_bias (which adds bias[j] to entry (i,j)). */
__global__ void k_add_bias(float *m, const float *bias, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    m[idx] += bias[idx % cols];
}

/* Column-sum reduction. Every element atomically adds itself into its column's slot of
 * `out`. atomicAdd serializes the concurrent updates to each out[j] so no contribution
 * is lost when many threads (all rows of a column) hit the same slot simultaneously.
 * This is the offloaded counterpart of the host-side loop the OpenCL backend keeps on
 * the CPU — possible here because `out` is device memory in this design. Simple, not
 * the fastest (atomic contention on `cols` slots); a shared-memory reduction is the
 * natural optimization. */
__global__ void k_colsum(const float *m, float *out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    atomicAdd(&out[idx % cols], m[idx]);
}

extern "C" void mat_add_bias(Mat m, const float *bias) {
    /* bias is expected device-resident (same address space as m.data). */
    /* t = 256 threads/block; grid = ceil_div(n, t) blocks to cover all n elements. */
    int n = m.rows * m.cols, t = 256;
    k_add_bias<<<ceil_div(n, t), t>>>(m.data, bias, m.rows, m.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void mat_colsum(const Mat m, float *out) {
    /* Clear the `cols` accumulators first — k_colsum only ever adds into them. */
    CUDA_CHECK(cudaMemset(out, 0, (size_t)m.cols * sizeof(float)));
    int n = m.rows * m.cols, t = 256;
    k_colsum<<<ceil_div(n, t), t>>>(m.data, out, m.rows, m.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ---- pointwise / scalar ---------------------------------------------
 * Flat 1D elementwise kernels, one thread per element (i = block*blockDim+thread,
 * guarded by i<n). Each is the exact arithmetic of its CPU/OpenCL twin: relu clamps
 * negatives to 0, sigmoid = 1/(1+exp(-x)) (expf = float exp, the source of the tiny
 * rounding the cross-check tolerates), scale multiplies by s, axpy does y += a*x. */

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

/* Public wrappers. All share the 1D launch pattern: 256 threads/block, ceil_div(n,256)
 * blocks, launch, check-last-error, synchronize. Operate in place on device memory. */
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
    /* Both y and x are device pointers; the kernel reads x and updates y in place. */
    int n = y.rows * y.cols, t = 256;
    k_axpy<<<ceil_div(n, t), t>>>(y.data, a, x.data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
