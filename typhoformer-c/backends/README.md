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

The default build uses `src/tensor.c` — a portable, cache-aware **CPU
backend** (standard library only; loop-ordered accumulation plus an unrolled
8-accumulator dot product that auto-vectorizes at `-O3`). To retarget the model to another device,
implement the same functions and link that translation unit *instead of*
`src/tensor.c`. No layer code changes, because every layer (`src/nn.c`,
`src/model.c`) is written purely in terms of these calls.

Two reference GPU/accelerator backends ship here, illustrating the **two ways** a
backend can satisfy the contract:

| Backend | Data lives | Runnable here? | Status |
|:--|:--|:--|:--|
| [`opencl/`](opencl) | host, offloaded per call | **yes** — verified on POCL (CPU OpenCL) | numerically checked vs CPU + full-model gradient check |
| [`cuda/`](cuda) | device-resident (`cudaMalloc`) | needs an NVIDIA GPU | compiles with `nvcc`; not executable without a GPU |

## `opencl/` — a runnable OpenCL backend  ✅ verified

[`opencl/tensor_opencl.c`](opencl/tensor_opencl.c) implements the full kernel set
as OpenCL kernels. It runs on **any** OpenCL device — a real GPU, or a **CPU via
[POCL](http://portablecl.org/)** (`pocl-opencl-icd`), which is how it is verified
in this repo without a GPU.

Design: `Mat.data` stays **host-resident** (plain `malloc`), exactly like the CPU
backend; each kernel call copies operands to device buffers, runs, and copies the
result back. That keeps the whole program — *including* the glue code that reads
`Mat.data` directly — working unchanged, so the entire model (forward **and**
backward) runs end-to-end through OpenCL. Per-call copies make it slower than a
device-resident design, but the point is correctness verification.

```sh
sudo apt-get install pocl-opencl-icd ocl-icd-opencl-dev opencl-headers   # CPU OpenCL
make OPENCL=1 test-opencl        # cross-check kernels, then gradient-check the whole model
```

Verified: every kernel matches the CPU reference at the float-rounding level
(test tolerance 1e-4; observed worst ~1e-7 — the exact figure shifts with the
CPU kernel's summation order), and the full-model finite-difference gradient
check passes *through OpenCL* at `pred_len` 1 and 3.

## `cuda/` — device-resident GPU reference

[`cuda/tensor_cuda.cu`](cuda/tensor_cuda.cu) implements the full kernel set
keeping `Mat.data` in GPU memory (`cudaMalloc`), one thread per output element —
a readable starting point, not a tuned library.

```sh
cd backends/cuda && make      # produces libtyphoformer_cuda.a  (requires nvcc)
```

It **compiles with `nvcc`** but cannot execute without an NVIDIA GPU + driver, so
it is not run by CI. Being device-resident, it also hits the caveat below.

### Device-resident caveat: the "hot path" vs. the "glue"

The seam is genuinely clean for the compute-heavy layers — attention, the
feed-forward blocks, and the linear layers touch matrices only through the
kernels above, so on a device those run entirely on-device.

The **glue code**, however, reads and writes `Mat.data` directly with plain C
loops in a few places — the MSE loss, the softmax inside attention, the
autoregressive decoder feedback (`src/model.c`), and normalization. Those loops
assume host memory. A *device-resident* backend (like `cuda/`) must therefore
either:

1. **Copy at the boundary** (simplest): move the small tensors involved (a handful
   of floats per step) to the host for those elementwise steps and back.
2. **Add kernels** for softmax / loss / the decoder feedback, keeping everything
   device-resident.

The `opencl/` backend sidesteps this entirely by staying host-resident and
offloading per call — which is exactly why it runs the whole model unmodified.
The point of the seam is that **the expensive work is already abstracted**; what
remains is small, bounded, and explicitly enumerated — not hidden.

## Adding your own backend

1. Create `backends/<name>/tensor_<name>.{c,cpp,cu}` implementing every function
   in `include/tensor.h` (pick host-resident offload or device-resident — both
   satisfy the contract; see the table above).
2. Provide a `die()` implementation (or link the one in `src/tensor.c`).
3. Link it in place of `src/tensor.c`. A gradient check
   (`tests/test_tensor`, `tests/test_model`) is the acceptance test — a correct
   backend passes them unchanged. The OpenCL backend adds
   `backends/opencl/test_opencl.c`, a direct kernel-vs-CPU cross-check you can
   copy.
