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

static void cl_check(cl_int e, const char *what) {
    if (e != CL_SUCCESS) die("OpenCL error %d at %s", (int)e, what);
}

/* ---- kernel source (mirrors the CPU/CUDA math exactly) ------------------- */
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

/* ---- lazy global context ------------------------------------------------- */
static cl_context       g_ctx;
static cl_command_queue g_q;
static cl_program       g_prog;
static cl_kernel g_matmul, g_matmul_bt, g_matmul_atb, g_add_bias, g_relu, g_sigmoid, g_scale, g_axpy;
static int g_ready = 0;

static cl_kernel mk(const char *name) {
    cl_int e; cl_kernel k = clCreateKernel(g_prog, name, &e);
    cl_check(e, name); return k;
}

static void cl_boot(void) {
    if (g_ready) return;
    cl_platform_id plat; cl_uint np = 0;
    cl_check(clGetPlatformIDs(1, &plat, &np), "clGetPlatformIDs");
    if (np == 0) die("OpenCL: no platform found");
    cl_device_id dev; cl_uint nd = 0;
    cl_int e = clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, &nd);
    if (e != CL_SUCCESS || nd == 0) cl_check(clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, &nd), "clGetDeviceIDs");
    g_ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &e); cl_check(e, "clCreateContext");
    g_q = clCreateCommandQueue(g_ctx, dev, 0, &e); cl_check(e, "clCreateCommandQueue");
    g_prog = clCreateProgramWithSource(g_ctx, 1, &KSRC, NULL, &e); cl_check(e, "clCreateProgramWithSource");
    e = clBuildProgram(g_prog, 1, &dev, "", NULL, NULL);
    if (e != CL_SUCCESS) {
        size_t n = 0; clGetProgramBuildInfo(g_prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &n);
        char *log = (char *)malloc(n + 1);
        clGetProgramBuildInfo(g_prog, dev, CL_PROGRAM_BUILD_LOG, n, log, NULL); log[n] = 0;
        die("OpenCL build failed:\n%s", log);
    }
    g_matmul = mk("k_matmul"); g_matmul_bt = mk("k_matmul_bt"); g_matmul_atb = mk("k_matmul_atb");
    g_add_bias = mk("k_add_bias"); g_relu = mk("k_relu"); g_sigmoid = mk("k_sigmoid");
    g_scale = mk("k_scale"); g_axpy = mk("k_axpy");
    g_ready = 1;
}

/* Allocate a device buffer and optionally upload host bytes. */
static cl_mem dbuf(size_t bytes, const void *host, cl_mem_flags rw) {
    cl_int e; cl_mem m = clCreateBuffer(g_ctx, rw, bytes, NULL, &e); cl_check(e, "clCreateBuffer");
    if (host) cl_check(clEnqueueWriteBuffer(g_q, m, CL_TRUE, 0, bytes, host, 0, NULL, NULL), "write");
    return m;
}

/* ---- host-side memory ops (identical to the CPU backend) ----------------- */
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

/* ---- GEMMs (offloaded) --------------------------------------------------- */
static void gemm(cl_kernel kern, const Mat A, const Mat B, Mat out, int a1, int a2, int a3) {
    cl_boot();
    size_t bytesA = (size_t)A.rows * A.cols * sizeof(float);
    size_t bytesB = (size_t)B.rows * B.cols * sizeof(float);
    size_t bytesC = (size_t)out.rows * out.cols * sizeof(float);
    cl_mem dA = dbuf(bytesA, A.data, CL_MEM_READ_ONLY);
    cl_mem dB = dbuf(bytesB, B.data, CL_MEM_READ_ONLY);
    cl_mem dC = dbuf(bytesC, NULL, CL_MEM_WRITE_ONLY);
    clSetKernelArg(kern, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(kern, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(kern, 2, sizeof(cl_mem), &dC);
    clSetKernelArg(kern, 3, sizeof(int), &a1);
    clSetKernelArg(kern, 4, sizeof(int), &a2);
    clSetKernelArg(kern, 5, sizeof(int), &a3);
    size_t global[2] = { (size_t)out.cols, (size_t)out.rows };
    cl_check(clEnqueueNDRangeKernel(g_q, kern, 2, NULL, global, NULL, 0, NULL, NULL), "gemm enqueue");
    cl_check(clEnqueueReadBuffer(g_q, dC, CL_TRUE, 0, bytesC, out.data, 0, NULL, NULL), "gemm read");
    clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC);
}
void mat_matmul(const Mat A, const Mat B, Mat out) {
    assert(A.cols == B.rows && out.rows == A.rows && out.cols == B.cols);
    cl_boot();   /* boot before reading the kernel global (arg eval precedes callee) */
    gemm(g_matmul, A, B, out, A.rows, A.cols, B.cols);
}
void mat_matmul_bt(const Mat A, const Mat B, Mat out) {
    assert(A.cols == B.cols && out.rows == A.rows && out.cols == B.rows);
    cl_boot();
    gemm(g_matmul_bt, A, B, out, A.rows, A.cols, B.rows);
}
void mat_matmul_atb(const Mat A, const Mat B, Mat out) {
    assert(A.rows == B.rows && out.rows == A.cols && out.cols == B.cols);
    cl_boot();
    gemm(g_matmul_atb, A, B, out, A.rows, A.cols, B.cols);
}

/* ---- pointwise / scalar / bias (offloaded) ------------------------------- */
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
void mat_relu(Mat m)    { cl_boot(); run1d(g_relu, m, 0, 0, 0, 0); }
void mat_sigmoid(Mat m) { cl_boot(); run1d(g_sigmoid, m, 0, 0, 0, 0); }
void mat_scale(Mat m, float s) { cl_boot(); run1d(g_scale, m, 1, s, 0, 0); }

void mat_axpy(Mat y, float a, const Mat x) {
    assert(y.rows == x.rows && y.cols == x.cols);
    cl_boot();
    size_t bytes = (size_t)x.rows * x.cols * sizeof(float);
    cl_mem dX = dbuf(bytes, x.data, CL_MEM_READ_ONLY);
    run1d(g_axpy, y, 2, 0, dX, a);
    clReleaseMemObject(dX);
}

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

/* ---- reduction kept on host (documented) --------------------------------- */
void mat_colsum(const Mat m, float *out) {
    for (int j = 0; j < m.cols; ++j) out[j] = 0.0f;
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out[j] += m.data[i * m.cols + j];
}
