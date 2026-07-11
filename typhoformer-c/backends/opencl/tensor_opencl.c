/*
 * tensor_opencl.c — an OpenCL backend for TyphoFormer-C.
 *
 * This is a drop-in replacement for src/tensor.c: it implements the exact same
 * kernel set declared in include/tensor.h (the backend contract in
 * include/backend.h), but runs the compute-heavy operations — the three GEMMs
 * and the elementwise/pointwise ops — as OpenCL kernels on whatever device the
 * ICD loader exposes (a GPU, or a CPU via POCL).
 *
 * Design choice: Mat.data stays HOST-resident (plain malloc'd float*), exactly
 * as in the CPU backend. Each kernel call copies its operands to device buffers,
 * runs the kernel, and copies the result back. This keeps the whole rest of the
 * program — including the glue code that reads Mat.data directly (loss, softmax,
 * decoder feedback) — working unchanged, so the *entire model and every gradient
 * check* runs end-to-end through OpenCL. The per-call host<->device copy makes it
 * slower than a fully device-resident design (see backends/cuda/ for that shape),
 * but the point here is verification: the OpenCL kernels must produce the same
 * numbers as the CPU backend, and this design lets us prove that against the real
 * model and the finite-difference gradient checks.
 *
 * Reductions (mat_colsum) stay on the host — a tree reduction adds complexity
 * without changing what we're verifying. Everything else offloads.
 *
 * Build:  make OPENCL=1        (links -lOpenCL, uses this file instead of tensor.c)
 * Verify: make OPENCL=1 test-opencl
 */
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensor.h"

/* ---- error plumbing (this TU replaces tensor.c, so it must define die) ---- */
_Noreturn void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/*
 * Convenience wrapper around the OpenCL error convention. Almost every OpenCL
 * host API returns a cl_int status (CL_SUCCESS == 0) either as its return value
 * or through an out-parameter. Rather than check each call site by hand, we funnel
 * them through here: on any non-success code we abort with the numeric error and a
 * label identifying which call failed. `what` is a short static string the caller
 * passes so the fatal message pinpoints the failing step.
 */
static void cl_check(cl_int e, const char *what) {
    if (e != CL_SUCCESS) die("OpenCL error %d at %s", (int)e, what);
}

/* ---- kernel source (mirrors the CPU/CUDA math exactly) -------------------
 *
 * OpenCL compiles device code from source *strings at runtime* (see cl_boot's
 * clCreateProgramWithSource / clBuildProgram). KSRC below is that source: a single
 * translation unit of OpenCL C holding all eight kernels. Because it is a C string
 * literal, the device code is written as one "..\n" line per source line — the \n
 * are cosmetic (they make the build log's line numbers meaningful). Do NOT insert
 * host-style /* *\/ or // comments *inside* these quotes: that text would become part
 * of the kernel program. Any explanation of the kernels lives out here in host land.
 *
 * How the kernels mirror src/tensor.c (the CPU backend), kernel by kernel:
 *
 *   k_matmul(A,B,C,m,k,n)  ==  mat_matmul: C = A@B, A[m,k] B[k,n] C[m,n].
 *     Each work-item owns one output element (i,j). get_global_id(0) is the fast
 *     axis -> column j; get_global_id(1) -> row i (this matches the 2D global size
 *     {out.cols, out.rows} set in gemm()). The bounds guard `if(i>=m||j>=n) return`
 *     discards work-items in the padding when the NDRange is rounded up. The inner
 *     loop is the dot product sum_p A[i*k+p]*B[p*n+j] — identical arithmetic to the
 *     CPU's ikj triple loop, just with every (i,j) computed in parallel instead of
 *     accumulated in place. Row-major indexing A[i*k+p], B[p*n+j], C[i*n+j] matches
 *     the host layout exactly, which is what makes the numbers agree.
 *
 *   k_matmul_bt(A,B,C,m,k,n)  ==  mat_matmul_bt: C = A @ B^T, A[m,k] B[n,k] C[m,n].
 *     Same tiling, but B is read as B[j*k+p]: row j of B walked contiguously, i.e.
 *     the transpose is expressed purely through indexing (no data is moved).
 *
 *   k_matmul_atb(A,B,C,kk,m,n)  ==  mat_matmul_atb: C = A^T @ B, A[kk,m] B[kk,n] C[m,n].
 *     Here the contraction dimension is the number of *rows* of A and B, passed as
 *     kk (renamed from k to avoid shadowing the m below). A is read as A[p*m+i]:
 *     column i of A, i.e. row i of A^T. B[p*n+j] as usual. Result C[i*n+j].
 *
 *   k_add_bias(M,b,cols,N)  ==  mat_add_bias: adds a per-column bias vector.
 *     Flattened 1D over all N=rows*cols elements; element i lives in column i%cols,
 *     so it gets b[i%cols]. (The CPU version loops i,j and adds bias[j]; i%cols is
 *     exactly that j for a row-major flat index.)
 *
 *   k_relu / k_sigmoid / k_scale / k_axpy  ==  the matching pointwise CPU ops,
 *     one work-item per flattened element i, guarded by if(i<N): relu clamps
 *     negatives to 0, sigmoid is 1/(1+exp(-x)), scale multiplies by s, and axpy
 *     does Y[i]+=a*X[i]. exp() here is the OpenCL builtin (float precision), which
 *     is why the cross-check tolerance in test_opencl.c is 1e-4 rather than exact.
 */
static const char *KSRC =
"__kernel void k_matmul(__global const float*A,__global const float*B,__global float*C,int m,int k,int n){\n"
"  int j=get_global_id(0), i=get_global_id(1); if(i>=m||j>=n) return;\n"
"  float acc=0.0f; for(int p=0;p<k;++p) acc+=A[i*k+p]*B[p*n+j]; C[i*n+j]=acc; }\n"
"__kernel void k_matmul_bt(__global const float*A,__global const float*B,__global float*C,int m,int k,int n){\n"
"  int j=get_global_id(0), i=get_global_id(1); if(i>=m||j>=n) return;\n"
"  float acc=0.0f; for(int p=0;p<k;++p) acc+=A[i*k+p]*B[j*k+p]; C[i*n+j]=acc; }\n"
"__kernel void k_matmul_atb(__global const float*A,__global const float*B,__global float*C,int kk,int m,int n){\n"
"  int j=get_global_id(0), i=get_global_id(1); if(i>=m||j>=n) return;\n"
"  float acc=0.0f; for(int p=0;p<kk;++p) acc+=A[p*m+i]*B[p*n+j]; C[i*n+j]=acc; }\n"
"__kernel void k_add_bias(__global float*M,__global const float*b,int cols,int N){\n"
"  int i=get_global_id(0); if(i>=N) return; M[i]+=b[i%cols]; }\n"
"__kernel void k_relu(__global float*M,int N){ int i=get_global_id(0); if(i<N&&M[i]<0.0f) M[i]=0.0f; }\n"
"__kernel void k_sigmoid(__global float*M,int N){ int i=get_global_id(0); if(i<N) M[i]=1.0f/(1.0f+exp(-M[i])); }\n"
"__kernel void k_scale(__global float*M,float s,int N){ int i=get_global_id(0); if(i<N) M[i]*=s; }\n"
"__kernel void k_axpy(__global float*Y,float a,__global const float*X,int N){ int i=get_global_id(0); if(i<N) Y[i]+=a*X[i]; }\n";

/* ---- lazy global context -------------------------------------------------
 *
 * The OpenCL runtime state is created once and cached in file-scope globals. A
 * context binds a device; a command queue serializes the buffer copies and kernel
 * launches we submit to it; a program is the compiled KSRC; and each cl_kernel is a
 * callable handle to one __kernel function inside that program. g_ready is the
 * "have we booted yet?" flag so cl_boot() is idempotent. Keeping these global (rather
 * than threading a handle through every mat_* call) is what lets these functions keep
 * the exact same signatures as the CPU backend in src/tensor.c. Single-queue,
 * single-device: no multi-GPU or async concerns to reason about here.
 */
static cl_context       g_ctx;
static cl_command_queue g_q;
static cl_program       g_prog;
static cl_kernel g_matmul, g_matmul_bt, g_matmul_atb, g_add_bias, g_relu, g_sigmoid, g_scale, g_axpy;
static int g_ready = 0;

/* Look up one compiled kernel by name inside the built program (g_prog) and return
 * its handle. clCreateKernel fails if the name doesn't match a __kernel in KSRC, so
 * this doubles as a spelling check between the strings above and the mk() calls below. */
static cl_kernel mk(const char *name) {
    cl_int e; cl_kernel k = clCreateKernel(g_prog, name, &e);
    cl_check(e, name); return k;
}

/*
 * One-time device/context/queue/program setup — the full OpenCL host lifecycle.
 * Called (cheaply, guarded by g_ready) at the top of every op so users never have to
 * initialize anything explicitly. Steps:
 *   1. clGetPlatformIDs: ask the ICD loader for the first installed OpenCL platform
 *      (an implementation such as a GPU vendor's driver, or POCL for CPU execution).
 *   2. clGetDeviceIDs: prefer the platform's DEFAULT device; if that yields nothing,
 *      fall back to CL_DEVICE_TYPE_ALL so a CPU-only POCL install still works.
 *   3. clCreateContext: a container that owns the device's memory objects and programs.
 *   4. clCreateCommandQueue: the ordered channel we push writes/launches/reads onto.
 *   5. clCreateProgramWithSource + clBuildProgram: JIT-compile KSRC for this device.
 *      On build failure we pull CL_PROGRAM_BUILD_LOG (the compiler's diagnostics —
 *      queried twice: once for its size, once to fill the buffer) and abort with it,
 *      because a kernel-source typo is otherwise silent.
 *   6. mk(...) for each kernel: resolve the eight callable handles.
 * Setting g_ready = 1 last means a failure anywhere above leaves us un-booted and
 * will retry (though die() has already aborted the process in practice).
 */
static void cl_boot(void) {
    if (g_ready) return;
    cl_platform_id plat; cl_uint np = 0;
    cl_check(clGetPlatformIDs(1, &plat, &np), "clGetPlatformIDs");
    if (np == 0) die("OpenCL: no platform found");
    cl_device_id dev; cl_uint nd = 0;
    /* Try the platform's default device first; some ICDs report none, so retry ALL. */
    cl_int e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, &nd);
    if (e != CL_SUCCESS || nd == 0) cl_check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, &nd), "clGetDeviceIDs");
    g_ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e); cl_check(e, "clCreateContext");
    g_q = clCreateCommandQueue(g_ctx, dev, 0, &e); cl_check(e, "clCreateCommandQueue");
    /* Hand the loader a pointer to our single source string; NULL length => NUL-terminated. */
    g_prog = clCreateProgramWithSource(g_ctx, 1, &KSRC, NULL, &e); cl_check(e, "clCreateProgramWithSource");
    e = clBuildProgram(g_prog, 1, &dev, "", NULL, NULL);
    if (e != CL_SUCCESS) {
        /* Build failed: fetch the compiler log (size first, then contents) and abort. */
        size_t n = 0; clGetProgramBuildInfo(g_prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
        char *log = (char *)malloc(n + 1);
        clGetProgramBuildInfo(g_prog, dev, CL_PROGRAM_BUILD_LOG, n, log, NULL); log[n] = 0;
        die("OpenCL build failed:\n%s", log);
    }
    /* Resolve one callable handle per __kernel; names must match KSRC exactly. */
    g_matmul = mk("k_matmul"); g_matmul_bt = mk("k_matmul_bt"); g_matmul_atb = mk("k_matmul_atb");
    g_add_bias = mk("k_add_bias"); g_relu = mk("k_relu"); g_sigmoid = mk("k_sigmoid");
    g_scale = mk("k_scale"); g_axpy = mk("k_axpy");
    g_ready = 1;
}

/*
 * Allocate a device buffer of `bytes` and, if `host` is non-NULL, upload those bytes
 * into it before returning. `rw` is the access hint (CL_MEM_READ_ONLY / WRITE_ONLY /
 * READ_WRITE) describing how the *kernel* will use it. The upload uses a BLOCKING
 * write (CL_TRUE): the call does not return until the copy is done, so the caller can
 * treat the buffer as ready immediately. Every op below follows the same shape —
 * dbuf() to stage operands on the device, run the kernel, read results back, free. */
static cl_mem dbuf(size_t bytes, const void *host, cl_mem_flags rw) {
    cl_int e; cl_mem m = clCreateBuffer(g_ctx, rw, bytes, NULL, &e); cl_check(e, "clCreateBuffer");
    if (host) cl_check(clEnqueueWriteBuffer(g_q, m, CL_TRUE, 0, bytes, host, 0, NULL, NULL), "write");
    return m;
}

/* ---- host-side memory ops (identical to the CPU backend) -----------------
 * Because this backend keeps Mat.data host-resident (see the file header), these
 * four functions are byte-for-byte the same as src/tensor.c: allocation, free,
 * zero-fill, and copy all happen in ordinary host memory. Only the compute ops
 * touch the device. */
Mat mat_new(int rows, int cols) {
    assert(rows > 0 && cols > 0);
    Mat m; m.rows = rows; m.cols = cols;
    m.data = (float *)calloc((size_t)rows * cols, sizeof(float));
    assert(m.data != NULL);
    return m;
}
void mat_free(Mat *m) { if (m && m->data) { free(m->data); m->data = NULL; m->rows = m->cols = 0; } }
void mat_zero(Mat m) { memset(m.data, 0, (size_t)m.rows * m.cols * sizeof(float)); }
void mat_copy(Mat dst, const Mat src) {
    assert(dst.rows == src.rows && dst.cols == src.cols);
    memcpy(dst.data, src.data, (size_t)src.rows * src.cols * sizeof(float));
}

/* ---- GEMMs (offloaded) ---------------------------------------------------
 *
 * Shared driver for all three matrix multiplies. `kern` selects which compiled
 * kernel to run; a1/a2/a3 are the three int dimension arguments that kernel expects
 * (their exact meaning differs per kernel — see each mat_matmul* wrapper below and
 * the KSRC documentation). The flow is the canonical offload pattern:
 *   1. size the three operands from their Mat rows*cols,
 *   2. stage A and B on the device READ_ONLY (uploaded from host), allocate C
 *      WRITE_ONLY (no upload — the kernel fills it),
 *   3. bind the six kernel arguments in order: the three buffers then the three ints
 *      (clSetKernelArg copies the *value pointed to*, hence &dA / &a1),
 *   4. launch a 2D NDRange. global = {out.cols, out.rows} means one work-item per
 *      output element, with dimension 0 = columns (matching get_global_id(0)->j) and
 *      dimension 1 = rows (get_global_id(1)->i). local size NULL lets the runtime pick
 *      the work-group shape,
 *   5. blocking read C back into out.data (CL_TRUE waits for the kernel to finish —
 *      this is also the synchronization point for the whole op),
 *   6. release the three device buffers (host Mat memory is untouched).
 */
static void gemm(cl_kernel kern, const Mat A, const Mat B, Mat out, int a1, int a2, int a3) {
    cl_boot();
    size_t bytesA = (size_t)A.rows * A.cols * sizeof(float);
    size_t bytesB = (size_t)B.rows * B.cols * sizeof(float);
    size_t bytesC = (size_t)out.rows * out.cols * sizeof(float);
    cl_mem dA = dbuf(bytesA, A.data, CL_MEM_READ_ONLY);   /* upload A */
    cl_mem dB = dbuf(bytesB, B.data, CL_MEM_READ_ONLY);   /* upload B */
    cl_mem dC = dbuf(bytesC, NULL, CL_MEM_WRITE_ONLY);    /* output only, no upload */
    clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
    clSetKernelArg(kern, 3, sizeof(int), &a1);
    clSetKernelArg(kern, 4, sizeof(int), &a2);
    clSetKernelArg(kern, 5, sizeof(int), &a3);
    /* 2D grid over the output: axis0=cols (j), axis1=rows (i); one work-item per cell. */
    size_t global[2] = { (size_t)out.cols, (size_t)out.rows };
    cl_check(clEnqueueNDRangeKernel(g_q, kern, 2, NULL, global, NULL, 0, NULL, NULL), "gemm enqueue");
    cl_check(clEnqueueReadBuffer(g_q, dC, CL_TRUE, 0, bytesC, out.data, 0, NULL, NULL), "gemm read");
    clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC);
}
/* out = A@B. Mirrors mat_matmul in src/tensor.c. Passes (m,k,n)=(A.rows,A.cols,B.cols):
 * a1=m rows of A, a2=k shared/contraction dim, a3=n cols of B. Asserts the shape
 * contract (A.cols==B.rows) exactly as the CPU version does before touching the GPU. */
void mat_matmul(const Mat A, const Mat B, Mat out) {
    assert(A.cols == B.rows && out.rows == A.rows && out.cols == B.cols);
    cl_boot();   /* boot before reading the kernel global (arg eval precedes callee) */
    gemm(g_matmul, A, B, out, A.rows, A.cols, B.cols);
}
/* out = A@B^T with B stored [n,k]. Mirrors mat_matmul_bt. Passes (m,k,n) as
 * (A.rows, A.cols, B.rows): the contraction dim k is A.cols==B.cols, and the output
 * width n is B.rows (each row of B becomes a column of B^T). */
void mat_matmul_bt(const Mat A, const Mat B, Mat out) {
    assert(A.cols == B.cols && out.rows == A.rows && out.cols == B.rows);
    cl_boot();
    gemm(g_matmul_bt, A, B, out, A.rows, A.cols, B.rows);
}
/* out = A^T@B with A stored [k,m], B stored [k,n]. Mirrors mat_matmul_atb. Here the
 * kernel's first int (kk) is the contraction dim = A.rows = B.rows; a2=m=A.cols is the
 * output height and a3=n=B.cols the output width. (kk is named apart from m in KSRC.) */
void mat_matmul_atb(const Mat A, const Mat B, Mat out) {
    assert(A.rows == B.rows && out.rows == A.cols && out.cols == B.cols);
    cl_boot();
    gemm(g_matmul_atb, A, B, out, A.rows, A.cols, B.cols);
}

/* ---- pointwise / scalar / bias (offloaded) -------------------------------
 *
 * Shared driver for the in-place elementwise kernels (relu, sigmoid, scale, axpy).
 * These operate over the matrix as a flat 1D array of N = rows*cols floats, so the
 * NDRange is 1D. The matrix buffer dM is READ_WRITE because the kernel both reads and
 * overwrites it in place; it is uploaded from m.data and read back into m.data.
 *
 * `extra_is_scale` selects the argument shape, because the kernels differ in signature:
 *   0 -> relu/sigmoid: args are (M, N)                       — no scalar.
 *   1 -> scale:        args are (M, s, N)                    — one float scalar sval.
 *   2 -> axpy:         args are (Y, a, X, N)                 — a float aval and the
 *        already-staged X buffer (xbuf). The `arg` counter walks the argument index so
 *        N is always bound last, matching each kernel's final int parameter.
 * The 1D global size is N (one work-item per element); the trailing blocking read is
 * the op's synchronization point, after which m.data holds the result. */
static void run1d(cl_kernel kern, Mat m, int extra_is_scale, float sval,
                  cl_mem xbuf /* for axpy */, float aval) {
    cl_boot();
    int N = m.rows * m.cols;
    size_t bytes = (size_t)N * sizeof(float);
    cl_mem dM = dbuf(bytes, m.data, CL_MEM_READ_WRITE);
    int arg = 0;
    clSetKernelArg(kern, arg++, sizeof(cl_mem), &dM);
    if (extra_is_scale == 1) clSetKernelArg(kern, arg++, sizeof(float), &sval);        /* scale */
    if (extra_is_scale == 2) { clSetKernelArg(kern, arg++, sizeof(float), &aval);      /* axpy: a, X */
        clSetKernelArg(kern, arg++, sizeof(cl_mem), &xbuf); }
    clSetKernelArg(kern, arg++, sizeof(int), &N);
    size_t global = (size_t)N;
    cl_check(clEnqueueNDRangeKernel(g_q, kern, 1, NULL, &global, NULL, 0, NULL, NULL), "1d enqueue");
    cl_check(clEnqueueReadBuffer(g_q, dM, CL_TRUE, 0, bytes, m.data, 0, NULL, NULL), "1d read");
    clReleaseMemObject(dM);
}
/* Thin wrappers mirroring the CPU pointwise ops in src/tensor.c; the unused trailing
 * zeros are the scalar/axpy slots that these kernels don't use (extra_is_scale==0/1). */
void mat_relu(Mat m)    { cl_boot(); run1d(g_relu, m, 0, 0, 0, 0); }
void mat_sigmoid(Mat m) { cl_boot(); run1d(g_sigmoid, m, 0, 0, 0, 0); }
void mat_scale(Mat m, float s) { cl_boot(); run1d(g_scale, m, 1, s, 0, 0); }

/* y += a*x. Mirrors mat_axpy. Unlike the single-buffer ops, axpy needs a *second*
 * operand on the device, so we stage x READ_ONLY into dX here and hand it to run1d
 * (extra_is_scale==2); run1d uploads/downloads y as usual, and we free dX afterward. */
void mat_axpy(Mat y, float a, const Mat x) {
    assert(y.rows == x.rows && y.cols == x.cols);
    cl_boot();
    size_t bytes = (size_t)x.rows * x.cols * sizeof(float);
    cl_mem dX = dbuf(bytes, x.data, CL_MEM_READ_ONLY);
    run1d(g_axpy, y, 2, 0, dX, a);
    clReleaseMemObject(dX);
}

/* Add a per-column bias vector to every row. Mirrors mat_add_bias. Bias has `cols`
 * entries; the kernel maps flat index i to column i%cols. Handled separately from
 * run1d because it needs the bias buffer plus the `cols` argument. Flow is the usual
 * upload M and bias -> bind (M, bias, cols, N) -> 1D launch over N -> read M back. */
void mat_add_bias(Mat m, const float *bias) {
    cl_boot();
    int N = m.rows * m.cols, cols = m.cols;
    size_t bytes = (size_t)N * sizeof(float);
    cl_mem dM = dbuf(bytes, m.data, CL_MEM_READ_WRITE);
    cl_mem dB = dbuf((size_t)cols * sizeof(float), bias, CL_MEM_READ_ONLY);
    clSetKernelArg(g_add_bias, 0, sizeof(cl_mem), &dM);
    clSetKernelArg(g_add_bias, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(g_add_bias, 2, sizeof(int), &cols);
    clSetKernelArg(g_add_bias, 3, sizeof(int), &N);
    size_t global = (size_t)N;
    cl_check(clEnqueueNDRangeKernel(g_q, g_add_bias, 1, NULL, &global, NULL, 0, NULL, NULL), "add_bias enqueue");
    cl_check(clEnqueueReadBuffer(g_q, dM, CL_TRUE, 0, bytes, m.data, 0, NULL, NULL), "add_bias read");
    clReleaseMemObject(dM); clReleaseMemObject(dB);
}

/* ---- reduction kept on host (documented) ---------------------------------
 * mat_colsum sums each column into out[j]. It is deliberately NOT offloaded: a
 * correct parallel reduction needs either atomics or a tree, which adds machinery
 * without exercising anything the cross-check cares about. Since Mat.data is already
 * host-resident, computing it here in plain C (byte-for-byte the same loop as
 * src/tensor.c) costs nothing extra and keeps this the one op that never touches the
 * device. The CUDA backend, where data lives on the GPU, DOES offload this via
 * atomicAdd — see backends/cuda/tensor_cuda.cu k_colsum. */
void mat_colsum(const Mat m, float *out) {
    for (int j = 0; j < m.cols; ++j) out[j] = 0.0f;
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out[j] += m.data[i * m.cols + j];
}
