/*
 * tensor.h — minimal row-major float matrix and the core linear-algebra
 * primitives used throughout TyphoFormer-C. Standard library only.
 */
#ifndef TYPHOFORMER_TENSOR_H
#define TYPHOFORMER_TENSOR_H

/* Row-major dense matrix of 32-bit floats (data has rows*cols elements). */
typedef struct {
    int    rows;
    int    cols;
    float *data;
} Mat;

/* Allocation. mat_new zero-initialises. */
Mat  mat_new(int rows, int cols);
void mat_free(Mat *m);
void mat_zero(Mat m);
void mat_copy(Mat dst, const Mat src);

/* Matrix products (out must be pre-allocated with matching shape):
 *   mat_matmul     : out = A   @ B     A[m,k] B[k,n] -> out[m,n]
 *   mat_matmul_bt  : out = A   @ B^T   A[m,k] B[n,k] -> out[m,n]
 *   mat_matmul_atb : out = A^T @ B     A[k,m] B[k,n] -> out[m,n]
 * These three cover a linear layer's forward pass and both of its gradients.
 */
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
