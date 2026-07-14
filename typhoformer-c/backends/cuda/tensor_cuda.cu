/*
 * tensor_cuda.cu — a CUDA reference backend for TyphoFormer-C.
 *
 * This translation unit implements the SAME kernel set declared in
 * include/tensor.h (the backend contract documented in include/backend.h).
 * Link it in place of src/tensor.c and every layer's math runs on the GPU — no
 * layer code changes, because every layer is written only in terms of these
 * primitives.
 *
 * STATUS: RUN-VERIFIED on real hardware (Quadro P1000, sm_61, CUDA 12.8) via
 *   make CUDA=1 test-cuda
 * which runs the backend kernel cross-check plus the tensor and full-model
 * gradient checks *through* these kernels. CI still only compile-checks this
 * file (GitHub runners have no NVIDIA GPU), so the executing gate is that
 * local target.
 *
 * Until 2026-07-14 this file had only ever been COMPILED, and it did not work:
 * it made Mat.data a cudaMalloc'd DEVICE pointer while asserting that "the rest
 * of the program is oblivious to where the data lives." It is not — nn.c and
 * model.c dereference Mat.data[i] directly in ~189 places — so the first host
 * read segfaulted. See the memory section below.
 *
 * Design choices kept deliberately simple for readability over peak FLOPS:
 *   - Mat.data is HOST-resident; each wrapper stages its operands to the device
 *     and copies results back (same choice the OpenCL backend documents). Slow,
 *     but correct against the code that actually calls it.
 *   - One thread per output element for GEMM (no shared-memory tiling). This is
 *     correct and easy to read; the natural next exercise is a tiled GEMM.
 *   - Reductions (mat_colsum) use atomicAdd for brevity.
 */

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
 * Mat.data is HOST-resident (plain malloc'd float*), exactly as in src/tensor.c and the
 * OpenCL backend. Each kernel wrapper below stages its operands to the device, launches,
 * and copies the result back.
 *
 * This is NOT a performance choice — a per-call host<->device round trip is the slowest
 * thing you could do — it is a CORRECTNESS requirement, and it is why this backend now
 * runs at all. The seam in tensor.h is only ~13 kernels wide, but the rest of the program
 * does NOT confine itself to that seam: nn.c and model.c dereference `Mat.data[i]`
 * directly in ~189 places (weight init, loss, softmax, and all the glue), as does the
 * backend cross-check. An earlier version of this file made Mat.data a cudaMalloc'd
 * DEVICE pointer and asserted "callers can't tell where the data lives" — they can, and
 * they do: the first host read of a device pointer segfaults. That version compiled, CI
 * only ever compile-checked it, and it had never once been executed. It could not have
 * worked with this codebase.
 *
 * So: host-resident data, device staging per call. Slow, correct, and it actually runs.
 * (Making the data genuinely device-resident is a real option, but it requires routing
 * those ~189 direct accesses through the seam first — a much larger change.) */

/* ---- device scratch pool ----
 * cudaMalloc and cudaFree cost on the order of a MILLISECOND each. Doing a
 * malloc/free pair per matrix op is correct but unusable: a full-model
 * finite-difference gradient check issues millions of ops, and the first version
 * of this file (allocating per call) was still grinding after 30 minutes on a
 * P1000 — the allocator, not the kernels, was the entire cost.
 *
 * Instead, keep a handful of device buffers alive and grow them on demand. Reusing
 * fixed slots across calls is safe because every launch below is immediately
 * followed by cudaDeviceSynchronize(), so no kernel is ever still in flight when
 * the next call reuses a slot. Three slots is enough: the widest op (GEMM) needs
 * A, B and out simultaneously. */
#define TF_CUDA_SLOTS 3
static float *g_buf[TF_CUDA_SLOTS];
static size_t g_cap[TF_CUDA_SLOTS];

/* Device buffer for slot i, at least n floats. Grows (never shrinks). */
static float *dslot(int i, size_t n) {
    if (n > g_cap[i]) {
        if (g_buf[i]) CUDA_CHECK(cudaFree(g_buf[i]));
        CUDA_CHECK(cudaMalloc((void **)&g_buf[i], n * sizeof(float)));
        g_cap[i] = n;
    }
    return g_buf[i];
}
/* Stage n floats host->device into slot i. */
static float *dev_up(int i, const float *host, size_t n) {
    float *d = dslot(i, n);
    CUDA_CHECK(cudaMemcpy(d, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}
/* Zeroed device scratch in slot i (for outputs that are written, not read). */
static float *dev_zero(int i, size_t n) {
    float *d = dslot(i, n);
    CUDA_CHECK(cudaMemset(d, 0, n * sizeof(float)));
    return d;
}
/* Copy n floats device->host. */
static void dev_down(float *host, const float *dev, size_t n) {
    CUDA_CHECK(cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C" Mat mat_new(int rows, int cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)calloc((size_t)rows * cols, sizeof(float));
    if (!m.data) die("mat_new: out of memory (%dx%d)", rows, cols);
    return m;
}

extern "C" void mat_free(Mat *m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
        m->rows = m->cols = 0;
    }
}

extern "C" void mat_zero(Mat m) {
    memset(m.data, 0, (size_t)m.rows * m.cols * sizeof(float));
}

extern "C" void mat_copy(Mat dst, const Mat src) {
    memcpy(dst.data, src.data, (size_t)src.rows * src.cols * sizeof(float));
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
    size_t nA = (size_t)A.rows * A.cols, nB = (size_t)B.rows * B.cols,
           nO = (size_t)out.rows * out.cols;
    float *dA = dev_up(0, A.data, nA), *dB = dev_up(1, B.data, nB), *dO = dev_zero(2, nO);
    dim3 block(16, 16);
    dim3 grid(ceil_div(out.cols, block.x), ceil_div(out.rows, block.y));
    kern<<<grid, block>>>(dA, dB, dO, a1, a2, a3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_down(out.data, dO, nO);
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
    /* Both m.data and bias are HOST pointers (tensor.h's contract) — stage both. */
    int n = m.rows * m.cols, t = 256;
    float *dM = dev_up(0, m.data, (size_t)n), *dB = dev_up(1, bias, (size_t)m.cols);
    k_add_bias<<<ceil_div(n, t), t>>>(dM, dB, m.rows, m.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_down(m.data, dM, (size_t)n);          /* in-place: write the result back */
}

extern "C" void mat_colsum(const Mat m, float *out) {
    /* `out` is a HOST float[cols]. dev_new zeroes the accumulators — k_colsum only adds. */
    int n = m.rows * m.cols, t = 256;
    float *dM = dev_up(0, m.data, (size_t)n), *dO = dev_zero(1, (size_t)m.cols);
    k_colsum<<<ceil_div(n, t), t>>>(dM, dO, m.rows, m.cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_down(out, dO, (size_t)m.cols);
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

/* Public wrappers. All share the 1D launch pattern: stage the host data to the device,
 * 256 threads/block over ceil_div(n,256) blocks, launch, check-last-error, synchronize,
 * copy the (in-place) result back to the host buffer. */
extern "C" void mat_relu(Mat m) {
    int n = m.rows * m.cols, t = 256;
    float *dM = dev_up(0, m.data, (size_t)n);
    k_relu<<<ceil_div(n, t), t>>>(dM, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_down(m.data, dM, (size_t)n);
}
extern "C" void mat_sigmoid(Mat m) {
    int n = m.rows * m.cols, t = 256;
    float *dM = dev_up(0, m.data, (size_t)n);
    k_sigmoid<<<ceil_div(n, t), t>>>(dM, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_down(m.data, dM, (size_t)n);
}
extern "C" void mat_scale(Mat m, float s) {
    int n = m.rows * m.cols, t = 256;
    float *dM = dev_up(0, m.data, (size_t)n);
    k_scale<<<ceil_div(n, t), t>>>(dM, s, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_down(m.data, dM, (size_t)n);
}
extern "C" void mat_axpy(Mat y, float a, const Mat x) {
    /* y is read-modify-write, x is read-only; both are host buffers. */
    int n = y.rows * y.cols, t = 256;
    float *dY = dev_up(0, y.data, (size_t)n), *dX = dev_up(1, x.data, (size_t)n);
    k_axpy<<<ceil_div(n, t), t>>>(dY, a, dX, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    dev_down(y.data, dY, (size_t)n);
}
