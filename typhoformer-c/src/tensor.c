/*
 * tensor.c — implementation of the core matrix primitives.
 *
 * Everything here operates on the row-major Mat type from tensor.h: a matrix
 * with `rows` rows, `cols` columns, and a flat buffer `data` where element
 * (i, j) is stored at data[i * cols + j]. The three matmul variants and the
 * elementwise helpers below are the numerical foundation the whole transformer
 * (linear layers, attention, layernorm, GRU, FFN) is built out of.
 */
#include "tensor.h"

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Fatal-error helper: format a message to stderr and terminate. Marked
 * _Noreturn so the compiler knows control never comes back (callers need no
 * fallthrough handling). Uses the C stdarg mechanism to forward the variadic
 * argument list to vfprintf, mirroring printf's format semantics. */
_Noreturn void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/* Allocate a rows x cols matrix. calloc both reserves rows*cols floats AND
 * zero-fills them, so a freshly created Mat starts at all-zeros (relied upon by
 * accumulate-in-place callers). The size is computed as (size_t)rows * cols to
 * do the multiply in size_t and avoid 32-bit int overflow on large matrices.
 * The asserts enforce the invariants that dimensions are positive and that the
 * allocation succeeded before the buffer is ever dereferenced. */
Mat mat_new(int rows, int cols) {
    assert(rows > 0 && cols > 0);
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)calloc((size_t)rows * cols, sizeof(float));
    assert(m.data != NULL);
    return m;
}

/* Release the backing buffer and reset the handle to an empty, safe-to-reuse
 * state. Guards against NULL and double-free: after this returns m->data is
 * NULL, so a second mat_free is a no-op. Takes Mat* (not Mat by value) because
 * it must mutate the caller's handle, not just a local copy. */
void mat_free(Mat *m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
        m->rows = m->cols = 0;
    }
}

/* Set every element to 0.0f. Valid because IEEE-754 float 0.0 has an all-bits-
 * zero representation, so a single memset of the whole buffer is correct and
 * fast. Takes Mat by value: only the elements pointed at change, the handle
 * does not. */
void mat_zero(Mat m) {
    memset(m.data, 0, (size_t)m.rows * m.cols * sizeof(float));
}

/* Deep-copy src's contents into dst. This copies the underlying floats (unlike
 * assigning one Mat to another, which would only alias the shared pointer). The
 * assert enforces that both matrices have identical shape so the contiguous
 * memcpy of rows*cols floats lands element-for-element. */
void mat_copy(Mat dst, const Mat src) {
    assert(dst.rows == src.rows && dst.cols == src.cols);
    memcpy(dst.data, src.data, (size_t)src.rows * src.cols * sizeof(float));
}

/* out = A @ B  (standard matrix product), A[m,k] B[k,n] -> out[m,n].
 * Mathematically out[i,j] = sum over p of A[i,p] * B[p,j].
 *
 * Loop order is i-p-j ("ikj"), NOT the textbook i-j-p. The reason is memory
 * access: in the innermost j-loop we read B's row p (B.data[p*n + j], stride 1)
 * and write out's row i (out.data[i*n + j], stride 1). Both march contiguously
 * through memory, which is cache-friendly and lets the compiler auto-vectorize
 * the inner loop into a scalar-times-vector fused multiply-add. The classic
 * i-j-p order would instead walk down a column of B with stride n (cache-hostile).
 *
 * Because out is accumulated with += , it must start at zero: mat_zero(out)
 * clears it first. Then for each (i,p) we hold a = A[i,p] fixed and add a*B[p,:]
 * into out[i,:], so after the p-loop finishes out[i,:] holds the full dot
 * products. The assert pins the shape contract: A.cols == B.rows (shared
 * summed dimension k) and out is exactly A.rows x B.cols.
 *
 * `restrict` on Arow/Brow/Crow promises the compiler these pointers do not
 * alias each other, so it may keep values in registers and vectorize the inner
 * loop without inserting defensive reload/aliasing checks. */
void mat_matmul(const Mat A, const Mat B, Mat out) {
    /* ikj order: B and out are streamed row-contiguously (cache friendly,
     * and auto-vectorizable). */
    assert(A.cols == B.rows && out.rows == A.rows && out.cols == B.cols);
    const int m = A.rows, k = A.cols, n = B.cols;
    mat_zero(out);
    for (int i = 0; i < m; ++i) {
        const float *restrict Arow = &A.data[i * k];   /* A[i, :]  */
        float *restrict Crow = &out.data[i * n];        /* out[i, :] */
        for (int p = 0; p < k; ++p) {
            const float a = Arow[p];                    /* scalar A[i,p]     */
            const float *restrict Brow = &B.data[p * n];/* B[p, :], stride 1 */
            for (int j = 0; j < n; ++j) Crow[j] += a * Brow[j]; /* out[i,:] += A[i,p]*B[p,:] */
        }
    }
}

/* out = A @ B^T  (B transposed), A[m,k], B[n,k] -> out[m,n].
 * Mathematically out[i,j] = sum over p of A[i,p] * B[j,p]. Note B is indexed by
 * ROW j and column p — i.e. we dot A's row i against B's row j, which is exactly
 * the (i,j) entry of A @ B^T. This is the natural layout for a linear layer's
 * forward pass y = x @ W^T where W is stored [out_features, in_features]: each
 * output neuron j owns a contiguous weight row B[j,:].
 *
 * Here the shared/summed dimension is k = A.cols = B.cols (asserted). Both
 * Arow = A[i,:] and Brow = B[j,:] are contiguous, so the innermost p-loop is a
 * clean stride-1 dot product accumulated into a local `acc`. Because each
 * out[i,j] is written exactly once from acc, no mat_zero is needed (unlike the
 * accumulate-in-place variants). `restrict` again asserts non-aliasing to help
 * the compiler vectorize the reduction. */
void mat_matmul_bt(const Mat A, const Mat B, Mat out) {
    /* out = A @ B^T ; A[m,k], B[n,k], out[m,n] (contiguous dot products) */
    assert(A.cols == B.cols && out.rows == A.rows && out.cols == B.rows);
    const int m = A.rows, k = A.cols, n = B.rows;
    for (int i = 0; i < m; ++i) {
        const float *restrict Arow = &A.data[i * k];        /* A[i, :] */
        for (int j = 0; j < n; ++j) {
            const float *restrict Brow = &B.data[j * k];    /* B[j, :] */
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) acc += Arow[p] * Brow[p]; /* dot(A[i,:], B[j,:]) */
            out.data[i * n + j] = acc;                      /* single write, no += */
        }
    }
}

/* out = A^T @ B  (A transposed), A[k,m], B[k,n] -> out[m,n].
 * Mathematically out[i,j] = sum over p of A[p,i] * B[p,j]. Now the summed index
 * p runs over ROWS of both A and B (their shared leading dimension k, asserted
 * A.rows == B.rows). This is the weight-gradient shape in backprop: with dy[k,n]
 * and x[k,m] (k = batch/sequence rows), dW = dy^T @ x accumulates the outer
 * products across all rows.
 *
 * Loop order is p-i-j. For each row p we hold A[p,:] and B[p,:] (both stride 1);
 * then for each i we take the scalar a = A[p,i] and add a*B[p,:] into out[i,:].
 * Across all p this sums the rank-1 outer products A[p,:]^T (x) B[p,:], which is
 * exactly A^T @ B. Because out is built by accumulation, it is cleared first
 * with mat_zero. The innermost j-loop streams out[i,:] and B[p,:] contiguously
 * (cache-friendly, vectorizable), and `restrict` marks the pointers non-aliasing. */
void mat_matmul_atb(const Mat A, const Mat B, Mat out) {
    /* out = A^T @ B ; A[k,m], B[k,n], out[m,n].
     * pij order: A and B are read row-contiguously, out streamed by row. */
    assert(A.rows == B.rows && out.rows == A.cols && out.cols == B.cols);
    const int k = A.rows, m = A.cols, n = B.cols;
    mat_zero(out);
    for (int p = 0; p < k; ++p) {
        const float *restrict Arow = &A.data[p * m];    /* A[p, :] */
        const float *restrict Brow = &B.data[p * n];    /* B[p, :] */
        for (int i = 0; i < m; ++i) {
            const float a = Arow[i];                    /* scalar A[p,i] */
            float *restrict Crow = &out.data[i * n];    /* out[i, :]     */
            for (int j = 0; j < n; ++j) Crow[j] += a * Brow[j]; /* out[i,:] += A[p,i]*B[p,:] */
        }
    }
}

/* Add a per-column bias vector to every row, in place: m[i,j] += bias[j].
 * `bias` is a length-m.cols array broadcast down the rows — this is the "+ b"
 * step of an affine layer y = xW^T + b, where one bias per output feature is
 * added to that feature's column across all rows (tokens/timesteps). */
void mat_add_bias(Mat m, const float *bias) {
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            m.data[i * m.cols + j] += bias[j];
}

/* Column-wise reduction: out[j] = sum over rows i of m[i,j], for each column j.
 * `out` must have m.cols entries. This is the gradient of a bias term: since
 * the bias is broadcast-added to every row in the forward pass, its gradient is
 * the sum of the incoming gradients over all rows. Accumulators are cleared to
 * 0.0f first, then the double loop adds each element into its column's slot. */
void mat_colsum(const Mat m, float *out) {
    for (int j = 0; j < m.cols; ++j) out[j] = 0.0f;
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out[j] += m.data[i * m.cols + j];
}

/* ReLU activation, in place: m[i] = max(m[i], 0). Iterates the flat buffer as
 * one length rows*cols vector (shape is irrelevant for an elementwise op).
 * Negative entries are clamped to zero, non-negative entries pass through. */
void mat_relu(Mat m) {
    const int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i)
        if (m.data[i] < 0.0f) m.data[i] = 0.0f;
}

/* Logistic sigmoid, in place: m[i] = 1 / (1 + exp(-m[i])), mapping each element
 * into (0, 1). Uses expf (single-precision) to match the float buffer. Flat
 * traversal since the operation is elementwise. */
void mat_sigmoid(Mat m) {
    const int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i)
        m.data[i] = 1.0f / (1.0f + expf(-m.data[i]));
}

/* Scale every element by scalar s, in place: m[i] *= s. Flat elementwise pass;
 * used e.g. to apply the 1/sqrt(d_k) attention-score scaling or to average. */
void mat_scale(Mat m, float s) {
    const int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i) m.data[i] *= s;
}

/* AXPY ("a x plus y"), the classic BLAS saxpy: y[i] += a * x[i], elementwise,
 * with y and x required to have identical shape (asserted). Both matrices are
 * treated as flat vectors of length rows*cols. This is the workhorse for
 * gradient accumulation and scaled residual adds throughout the network. */
void mat_axpy(Mat y, float a, const Mat x) {
    assert(y.rows == x.rows && y.cols == x.cols);
    const int n = y.rows * y.cols;
    for (int i = 0; i < n; ++i) y.data[i] += a * x.data[i];
}
