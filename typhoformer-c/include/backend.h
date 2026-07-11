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
 * How the seam works mechanically: every layer calls ONLY the functions above,
 * never raw loops over Mat.data, so the set of kernels is a hard, exhaustive
 * boundary between "model logic" and "how arithmetic runs on a device." A
 * backend is selected at LINK TIME, not runtime: there is no vtable or function
 * pointer here — you compile exactly one implementation of these symbols into
 * the binary (src/tensor.c for CPU, backends/opencl/… or backends/cuda/… for a
 * GPU) and the linker binds every call site to it. Swapping backends is a build
 * choice (which .c/.cu to link), which is why retargeting needs zero edits to
 * layer code.
 *
 * Device residency: on the CPU backend Mat.data is an ordinary host float*; on
 * the CUDA backend it is a device pointer and the same kernels run on the GPU,
 * so activations never round-trip to host between layers. The Mat struct is the
 * shared handle; what its `data` points at is the backend's business.
 *
 * This header intentionally declares nothing new: it documents the contract so
 * the seam is discoverable. Include tensor.h to call the kernels.
 */
#ifndef TYPHOFORMER_BACKEND_H
#define TYPHOFORMER_BACKEND_H
#include "tensor.h"
#endif /* TYPHOFORMER_BACKEND_H */
