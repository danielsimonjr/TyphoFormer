/*
 * backend.h — the compute-backend seam.
 *
 * Every heavy operation in the model goes through a small set of kernels
 * declared in tensor.h. Those kernels ARE the backend interface; `src/tensor.c`
 * is the reference CPU backend. To retarget the model to another device (GPU,
 * accelerator), implement the same functions over device-resident `Mat.data`
 * and link that translation unit instead of `src/tensor.c` — nothing else in
 * the codebase needs to change, because every layer is written only in terms of
 * these calls.
 *
 * The backend kernel set a device must provide (see tensor.h for signatures):
 *
 *   memory:   mat_new  mat_free  mat_zero  mat_copy
 *   gemm:     mat_matmul  mat_matmul_bt  mat_matmul_atb
 *   vector:   mat_add_bias  mat_colsum  mat_scale  mat_axpy
 *   pointwise:mat_relu  mat_sigmoid
 *
 * Two reference backends are provided under backends/ (neither is built by the
 * default Makefile): opencl/ — a runnable OpenCL backend, verified end-to-end on
 * a CPU device via POCL (`make OPENCL=1 test-opencl`); and cuda/ — a
 * device-resident CUDA backend that compiles with nvcc. See backends/README.md.
 *
 * This header intentionally declares nothing new: it documents the contract so
 * the seam is discoverable. Include tensor.h to call the kernels.
 */
#ifndef TYPHOFORMER_BACKEND_H
#define TYPHOFORMER_BACKEND_H
#include "tensor.h"
#endif /* TYPHOFORMER_BACKEND_H */
