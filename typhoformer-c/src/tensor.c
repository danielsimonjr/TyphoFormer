/*
 * tensor.c — implementation of the core matrix primitives.
 */
#include "tensor.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

Mat mat_new(int rows, int cols) {
    assert(rows > 0 && cols > 0);
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float *)calloc((size_t)rows * cols, sizeof(float));
    assert(m.data != NULL);
    return m;
}

void mat_free(Mat *m) {
    if (m && m->data) {
        free(m->data);
        m->data = NULL;
        m->rows = m->cols = 0;
    }
}

void mat_zero(Mat m) {
    memset(m.data, 0, (size_t)m.rows * m.cols * sizeof(float));
}

void mat_copy(Mat dst, const Mat src) {
    assert(dst.rows == src.rows && dst.cols == src.cols);
    memcpy(dst.data, src.data, (size_t)src.rows * src.cols * sizeof(float));
}

void mat_matmul(const Mat A, const Mat B, Mat out) {
    assert(A.cols == B.rows && out.rows == A.rows && out.cols == B.cols);
    const int m = A.rows, k = A.cols, n = B.cols;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += A.data[i * k + p] * B.data[p * n + j];
            out.data[i * n + j] = acc;
        }
    }
}

void mat_matmul_bt(const Mat A, const Mat B, Mat out) {
    /* out = A @ B^T ; A[m,k], B[n,k], out[m,n] */
    assert(A.cols == B.cols && out.rows == A.rows && out.cols == B.rows);
    const int m = A.rows, k = A.cols, n = B.rows;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += A.data[i * k + p] * B.data[j * k + p];
            out.data[i * n + j] = acc;
        }
    }
}

void mat_matmul_atb(const Mat A, const Mat B, Mat out) {
    /* out = A^T @ B ; A[k,m], B[k,n], out[m,n] */
    assert(A.rows == B.rows && out.rows == A.cols && out.cols == B.cols);
    const int k = A.rows, m = A.cols, n = B.cols;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < k; ++p)
                acc += A.data[p * m + i] * B.data[p * n + j];
            out.data[i * n + j] = acc;
        }
    }
}

void mat_add_bias(Mat m, const float *bias) {
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            m.data[i * m.cols + j] += bias[j];
}

void mat_colsum(const Mat m, float *out) {
    for (int j = 0; j < m.cols; ++j) out[j] = 0.0f;
    for (int i = 0; i < m.rows; ++i)
        for (int j = 0; j < m.cols; ++j)
            out[j] += m.data[i * m.cols + j];
}

void mat_relu(Mat m) {
    const int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i)
        if (m.data[i] < 0.0f) m.data[i] = 0.0f;
}

void mat_sigmoid(Mat m) {
    const int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i)
        m.data[i] = 1.0f / (1.0f + expf(-m.data[i]));
}

void mat_scale(Mat m, float s) {
    const int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i) m.data[i] *= s;
}

void mat_axpy(Mat y, float a, const Mat x) {
    assert(y.rows == x.rows && y.cols == x.cols);
    const int n = y.rows * y.cols;
    for (int i = 0; i < n; ++i) y.data[i] += a * x.data[i];
}
