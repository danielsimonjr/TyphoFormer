# Extending the Model

How to add layers/ops and modify the architecture safely. The golden rule:
**every change to a forward pass needs a matching backward, and you prove the
backward with a finite-difference gradient check.** If it passes, ship it.

---

## 1. The module pattern

Every learnable module in this codebase follows the same five-part shape. Copy it
for anything new.

```c
// 1. STRUCT: parameters (+ their grads), forward caches, and backward scratch.
typedef struct {
    Mat   W, dW;         // parameter + its gradient (same shape)
    float *b, *db;
    Mat   xcache;        // forward cache (what backward needs)
    Mat   s_tmp;         // persistent backward scratch (reused, not malloc'd)
} MyLayer;

// 2. NEW: allocate buffers, init params, REGISTER params with the ParamList.
MyLayer mylayer_new(int in, int out, ParamList *pl, const char *name) {
    MyLayer L = { ... };
    // init W ~ U(-1/sqrt(in), 1/sqrt(in));  b, dW, db = 0;  caches = {0,0,NULL}
    plist_add(pl, L.W.data, L.dW.data, in*out, name);   // value ptr, grad ptr, count
    plist_add(pl, L.b,      L.db,      out,    name);
    return L;
}

// 3. FORWARD: compute y from x; cache what backward needs.
void mylayer_forward(MyLayer *L, const Mat x, Mat y) { /* ... cache x ... */ }

// 4. BACKWARD: given dy, ACCUMULATE param grads (+=) and write dx.
void mylayer_backward(MyLayer *L, const Mat dy, Mat dx) { /* dW += ...; dx = ... */ }

// 5. FREE: release every buffer the struct owns.
void mylayer_free(MyLayer *L) { mat_free(&L->W); /* ... */ }
```

Key conventions (all enforced by the existing code):

- **Register params** with `plist_add(pl, value_ptr, grad_ptr, count, name)`.
  This is the *only* thing the optimizer and checkpoint need — do it and your
  layer is automatically trained and serialized.
- **Grads accumulate** (`+=`), never assign. The trainer zeros them per batch, so
  a mini-batch sums correctly.
- **Scratch is persistent.** Allocate backward temporaries as `s_*` struct fields
  and grow them with the `ensure(&m, r, c)` idiom, so steady state does no heap
  allocation. Do not `mat_new`/`mat_free` inside `*_backward`.
- **Caches couple forward→backward.** Whatever backward reads (inputs,
  pre-activations, softmax weights) must be stored by forward.

---

## 2. Worked example — add a GELU activation

Say you want `FFN` to use GELU instead of ReLU. Forward and backward of
(tanh-approx) GELU:

```
g(x)  = 0.5·x·(1 + tanh( √(2/π)·(x + 0.044715·x³) ))
g'(x) = ... (derivative of the above)
```

1. Add a `mat_gelu(Mat)` to `tensor.c` and, for backward, either cache the
   pre-activation `h` (already cached in `FFN`) and recompute `g'(h)`, or cache
   `g'(h)` directly.
2. In `ffn_forward`, replace `mat_relu(ff->a)` with `mat_gelu(ff->a)`.
3. In `ffn_backward`, replace the ReLU mask `(h>0)` with `g'(h)`.
4. **Gradient-check it.** Rebuild and run `make test` — `test_nn.c` exercises the
   FFN inside the transformer block. If `g'` is wrong, the check fails with a
   large relative error. Tune until abs error ~1e-4.

That is the whole loop: change forward, change backward, gradient-check.

---

## 3. Adding a gradient check for a brand-new layer

Model your test on `tests/test_nn.c`. The recipe:

```c
// build the layer with a ParamList; make random input x and target t
// forward -> y ; loss = mean((y-t)^2) ; dy = 2/N (y-t)
plist_zero_grad(&pl);
mylayer_backward(&L, dy, dx);                 // analytic grads
// for each parameter element w: central difference
//   w+=eps -> loss+ ; w-=eps -> loss- ; g_num = (loss+ - loss-)/(2 eps)
//   compare g_num to the analytic .g  (abs+rel metric, ~1e-4 abs floor)
```

Single-precision limits the finite-difference accuracy, so judge with a combined
metric `|g_num − g_ana| / (|g_num| + |g_ana| + 1e-2)` and an absolute floor of
~1e-3 (see the existing tests). Add the test binary to `TESTS` in the `Makefile`.

---

## 4. Common architecture changes

| Change | What to touch |
|:--|:--|
| **Multi-step decoding** (`pred_len > 1`) | Already implemented: `decoder_forward` caches per-step `z`/`h1`/`a`, and `decoder_backward` runs from the last step to the first, feeding `dz[D:D+2]` back into the previous step's output gradient (the autoregressive feedback). `tests/test_model` gradient-checks it at `pred_len=3`. Read it as the reference for any recurrent backprop. |
| **Pre-norm blocks** | reorder `block_forward` to `y = x + MHA(LN1(x))` etc.; mirror the residual/LN order in `block_backward`. |
| **Dropout** | *Implemented* — see `dropout_apply` in `src/nn.c` and the block's `drop1`/`drop2` masks. A cached 0/(1/(1-p)) mask in forward, multiplied into the gradient in backward, gated by `nn_set_training`. Model your own stochastic layer on it. |
| **Learned positional encoding** | add a `[T,D]` parameter to the encoder, `+=` it after `input_proj`, register it, and add its (identity) gradient in backward. |
| **New numerical features** | change `NUMCOL`/`d_num` in `data.c` and `Config.d_num`; retrain (the checkpoint config guards against loading a mismatched model). |
| **Bigger/smaller model** | just change `Config` fields — the whole graph is config-driven. |

---

## 5. Pitfalls (each is caught by the gradient check)

- **Missing residual gradient.** In `block_backward`, forgetting `+ dr2` / `+
  dr1` silently halves a gradient path.
- **Dropping the attention scale.** `dscores` must be multiplied by `1/√hd`, the
  same factor used in forward.
- **Aliasing scratch.** Two `s_*` buffers used as input and output of the same op
  must be distinct `Mat`s; overwriting an input mid-computation corrupts results.
- **Assigning instead of accumulating** param grads breaks mini-batching.
- **Forgetting to register** a new parameter means it never trains and never
  saves — silent, and *not* caught by the gradient check. Grep for `plist_add`.

---

## 6. Extension seams (build behind an interface, not by forking the core)

Three parts of the codebase are deliberately factored as **seams** so you can add
capability without editing the core model. Each has a reference implementation
and a test that is its acceptance criterion.

### 6a. Pluggable layers — the `Module` interface
[`include/module.h`](../include/module.h) defines a tiny vtable
(`self` + `forward`/`backward`/`free`) and a `Sequential` container. Wrap any
layer as a `Module` and chain it — no core edits, and it is gradient-checkable
exactly like a built-in. `module_block()` adapts a transformer `Block`; model
your own on it. **Acceptance:** extend `tests/test_module.c` and watch the
`Sequential` gradient-check pass.

### 6b. New compute device — the backend seam
[`include/backend.h`](../include/backend.h) documents the ~13-kernel contract
(GEMMs, bias, pointwise, scalar) that *every* layer is built from. `src/tensor.c`
is the CPU backend; implement the same functions and link that file instead to
retarget the whole model. Two references ship:

- [`backends/opencl/tensor_opencl.c`](../backends/opencl/tensor_opencl.c) — a
  **runnable** OpenCL backend. It keeps `Mat.data` host-resident and offloads each
  kernel, so the entire model runs through it. Verify on a CPU device with no GPU:
  `sudo apt-get install pocl-opencl-icd ocl-icd-opencl-dev opencl-headers`, then
  `make OPENCL=1 test-opencl`.
- [`backends/cuda/tensor_cuda.cu`](../backends/cuda/tensor_cuda.cu) — a
  device-resident CUDA reference; `make -C backends/cuda` compiles it with `nvcc`.

See [`backends/README.md`](../backends/README.md) for the host-resident vs.
device-resident trade-off (the latter hits the glue-code elementwise loops).
**Acceptance:** the CPU gradient checks (`test_tensor`, `test_model`) pass
unchanged against your backend — as they do for OpenCL.

### 6c. Multicore training — the data-parallel seam
[`include/parallel.h`](../include/parallel.h) / `src/parallel.c` replicate the
model across threads, broadcast weights, and reduce gradients for one optimizer
step (`--threads=N`). **Acceptance:** `tests/test_parallel` shows the reduced
gradient equals the serial gradient to ≈1e-7, and ThreadSanitizer is clean.

---

## 7. Good student projects

Ordered roughly easy → hard. (Items marked ✓ are already implemented — study the
reference, then extend or vary it.)

1. Swap ReLU→GELU (§2) and compare validation ΔR.
2. Add dropout and measure its effect on overfitting.
3. Implement pre-norm blocks and compare training stability.
4. ✓ *Done:* normalization stats in the checkpoint ([INTEGRATION.md](INTEGRATION.md)
   §3). Extend it to per-storm normalization and gradient-check nothing breaks.
5. ✓ *Done:* multi-step autoregressive backprop (`decoder_backward`). Add
   teacher forcing as a training-time option and compare.
6. Add a **tiled/blocked matmul** in a new backend file (§6b) and benchmark cache
   behavior vs. the current `ikj`/`pij` kernels with `bench`.
7. ✓ *Done:* multicore data-parallel training (§6c). Extend it to gradient
   **accumulation** across microbatches, or a persistent thread pool, and confirm
   `test_parallel` still holds.
8. Write a new **backend** (SIMD-intrinsics or GPU) behind §6b's contract.
9. Add a new **Module** (a GRU decoder, a graph-attention spatial block) behind
   §6a and plug it into a `Sequential`.

Each is a self-contained, gradient-checkable change. Start from the module
pattern in §1 (or the matching seam in §6) and let the tests be your guide.
