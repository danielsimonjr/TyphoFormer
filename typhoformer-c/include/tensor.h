/*
 * tensor.h — minimal row-major float matrix and the core linear-algebra
 * primitives used throughout TyphoFormer-C. Standard library only.
 */
#ifndef TYPHOFORMER_TENSOR_H
#define TYPHOFORMER_TENSOR_H

/* Portable "does not return" marker: _Noreturn in C11, [[noreturn]] in C++
 * (so device backends compiled as C++ — e.g. CUDA .cu — accept this header). */
#if defined(__cplusplus)
#  define TF_NORETURN [[noreturn]]
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define TF_NORETURN _Noreturn
#else
#  define TF_NORETURN
#endif

/* Print a message to stderr and exit(1). Used for unrecoverable I/O and
 * format errors throughout the CLI. */
TF_NORETURN void die(const char *fmt, ...);

/* Row-major dense matrix of 32-bit floats (data has rows*cols elements).
 *
 * Memory model:
 *   - `data` points to a single contiguous block of rows*cols floats.
 *   - Storage is ROW-MAJOR: the element at logical position (i, j) — row i,
 *     column j — lives at the flat index i*cols + j. So consecutive columns of
 *     a fixed row are adjacent in memory, but stepping down a column jumps by
 *     `cols` floats. Every loop in tensor.c is written to walk memory in this
 *     contiguous (row-inner) order for cache efficiency.
 *   - The Mat is passed BY VALUE (a small {rows, cols, ptr} handle), but the
 *     `data` pointer is shared: copying a Mat aliases the same buffer, it does
 *     NOT deep-copy. Only mat_new allocates and only mat_free releases; the
 *     other ops assume `data` is already sized rows*cols. Ownership therefore
 *     lives with whoever called mat_new, not with the by-value copies. */
typedef struct {
    int    rows;   /* number of rows (m)                                    */
    int    cols;   /* number of columns (n); row stride, in floats, is cols */
    float *data;   /* owning pointer to rows*cols contiguous floats         */
} Mat;

/* Allocation / lifetime helpers.
 *   mat_new  : allocate a rows x cols matrix, zero-initialised (via calloc).
 *   mat_free : release the buffer and null it out (safe to call twice).
 *   mat_zero : overwrite all elements with 0.0f (buffer kept).
 *   mat_copy : deep-copy src's elements into dst (shapes must match). */
Mat  mat_new(int rows, int cols);
void mat_free(Mat *m);
void mat_zero(Mat m);
void mat_copy(Mat dst, const Mat src);

/* Matrix products (out must be pre-allocated with matching shape):
 *   mat_matmul     : out = A   @ B     A[m,k] B[k,n] -> out[m,n]
 *   mat_matmul_bt  : out = A   @ B^T   A[m,k] B[n,k] -> out[m,n]
 *   mat_matmul_atb : out = A^T @ B     A[k,m] B[k,n] -> out[m,n]
 * These three cover a linear layer's forward pass and both of its gradients.
 * Concretely, for a layer y = x @ W^T + b with x[m,k], W[n,k]:
 *   - forward           y  = x @ W^T             -> mat_matmul_bt(x, W, y)
 *   - grad w.r.t. input dx = dy @ W              -> mat_matmul(dy, W, dx)
 *   - grad w.r.t. weight dW = dy^T @ x           -> mat_matmul_atb(dy, x, dW)
 * The shared inner dimension (named k below) is the one summed over; it must
 * agree between the two operands, and `out` must already have the result
 * shape — these functions never allocate. */
void mat_matmul(const Mat A, const Mat B, Mat out);
void mat_matmul_bt(const Mat A, const Mat B, Mat out);
void mat_matmul_atb(const Mat A, const Mat B, Mat out);

/* Bias handling for y = xW^T + b (bias broadcast over rows). */
void mat_add_bias(Mat m, const float *bias);      /* in place: m[i,:] += bias */
void mat_colsum(const Mat m, float *out);         /* out[j] = sum_i m[i,j]    */

/* Elementwise activations, in place (forward). */
void mat_relu(Mat m);
void mat_sigmoid(Mat m);

/* Scalar / elementwise helpers. */
void mat_scale(Mat m, float s);                   /* m *= s */
void mat_axpy(Mat y, float a, const Mat x);       /* y += a * x (same shape) */

#endif /* TYPHOFORMER_TENSOR_H */
