# Compute backends

Every heavy operation in TyphoFormer-C flows through a small, fixed set of
kernels declared in [`include/tensor.h`](../include/tensor.h) and documented as
a contract in [`include/backend.h`](../include/backend.h). Those ~13 functions
**are** the backend interface:

```
memory:    mat_new  mat_free  mat_zero  mat_copy
gemm:      mat_matmul  mat_matmul_bt  mat_matmul_atb
vector:    mat_add_bias  mat_colsum
pointwise: mat_relu  mat_sigmoid
scalar:    mat_scale  mat_axpy
```

The default build uses `src/tensor.c` — a portable, cache-blocked **CPU
backend** (standard library only). To retarget the model to another device,
implement the same functions over device-resident `Mat.data` and link that
translation unit *instead of* `src/tensor.c`. No layer code changes, because
every layer (`src/nn.c`, `src/model.c`) is written purely in terms of these
calls.

## `cuda/` — GPU reference backend

[`cuda/tensor_cuda.cu`](cuda/tensor_cuda.cu) is a **compile-ready reference**
implementation of the full kernel set, keeping `Mat.data` in GPU memory
(`cudaMalloc`). It is deliberately simple (one thread per output element, no
shared-memory tiling) so it reads as a starting point, not a tuned library.

**It is not built or run by the top-level Makefile** and has not been executed
in this repository's CI, because the CUDA toolkit (`nvcc`) is outside the
"standard library only" default build. Build it separately once you have CUDA:

```sh
cd backends/cuda && make      # produces libtyphoformer_cuda.a
```

### One honest caveat: the "hot path" vs. the "glue"

The seam is genuinely clean for the compute-heavy layers — attention, the
feed-forward blocks, and the linear layers touch matrices only through the
kernels above, so on GPU those run entirely on-device.

The **glue code**, however, still reads and writes `Mat.data` directly with
plain C loops in a few places — the MSE loss, the softmax inside attention, the
autoregressive decoder feedback (`src/model.c`), and normalization. Those loops
assume host memory. To run *fully* on the GPU without host round-trips you would
either:

1. **Copy at the boundary** (simplest): `cudaMemcpy` the small tensors involved
   (a handful of floats per step) to the host for those elementwise steps and
   back. Correct immediately; fine for a teaching/reference build since these
   tensors are tiny (`pred_len × output_dim`).
2. **Add kernels** for softmax / loss / the decoder feedback (the natural next
   exercise), keeping everything device-resident.

This is called out deliberately: the point of the seam is that **the expensive
work is already abstracted**, and what remains is small, bounded, and explicitly
enumerated — not hidden.

## Adding your own backend

1. Create `backends/<name>/tensor_<name>.{c,cpp,cu}` implementing every function
   in `include/tensor.h`.
2. Provide a `die()` implementation (or link the one in `src/tensor.c`).
3. Link it in place of `src/tensor.c`. A gradient check
   (`tests/test_tensor`, `tests/test_model`) is the acceptance test — a correct
   backend passes them unchanged.
