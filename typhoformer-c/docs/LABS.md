# Labs — hands-on exercises

A graded set of exercises that turn TyphoFormer-C into a course. Each lab has a
**goal**, the **files** you'll touch, **steps**, and a **check** — an objective
way to know you succeeded (almost always: *a gradient check passes* or *the
golden loss is unchanged*). The single discipline running through all of them:

> **Never trust a backward pass you haven't finite-difference checked.**
> Analytic gradient vs. `(L(w+ε) − L(w−ε))/2ε`. If they disagree, you're wrong.

Set up once:

```sh
cd typhoformer-c
make test          # everything green before you start
```

---

## Track A — Students: understand the machine

### Lab A1 — Read the gradient checker first
**Goal:** understand the tool you'll rely on for everything else.
**Files:** `tests/test_tensor.c`, `tests/test_nn.c`.
**Steps:** read how each test perturbs one weight by `±ε`, measures the loss
change, and compares to the analytic `.g`. Note the *relative* error metric and
the noise floor (`~1e-3` in float32).
**Check:** explain, in one sentence, why the check divides by
`|analytic| + |numeric| + floor`.

### Lab A2 — Break a backward pass on purpose
**Goal:** see the checker catch a real bug.
**Files:** `src/nn.c` (`mha_backward`), then revert.
**Steps:** delete the `1/√hd` scale in the attention gradient (or drop one
residual add in `block_backward`). Rebuild and run `make test`.
**Check:** `tests/test_nn` now reports a large error and **FAILS**. Restore the
line; it passes again. You now trust the checker.

### Lab A3 — Derive LayerNorm's backward
**Goal:** the first non-obvious Jacobian.
**Files:** `docs/ARCHITECTURE.md` (LayerNorm section), `src/nn.c`
(`layernorm_backward`).
**Steps:** with the forward `ŷ = γ(x−μ)/σ + β`, derive `∂L/∂x` by hand
(remember μ and σ both depend on every `xⱼ`). Match your result term-by-term to
the code.
**Check:** `tests/test_nn` passes — and you can now say what each of the three
terms (`dxhat`, the mean subtraction, the `xhat·mean` subtraction) is for.

### Lab A4 — Add a new activation: GELU
**Goal:** implement a forward+backward from scratch.
**Files:** `src/tensor.c` / `include/tensor.h` (add `mat_gelu` and its
derivative), or add it inside a copy of `FFN`.
**Steps:** implement the tanh-approximation GELU forward and its derivative,
swap it in for the ReLU inside `ffn_forward`/`ffn_backward`.
**Check:** write/extend a gradient check (mirror `tests/test_nn.c`) over the FFN;
it must pass at the noise floor.

### Lab A5 — A second decoder step, by hand
**Goal:** understand autoregressive backprop (the feedback path).
**Files:** `src/model.c` (`decoder_forward`/`decoder_backward`),
`tests/test_model.c`.
**Steps:** train with `--pred_len=1` vs `--pred_len=3`; read how
`decoder_backward` accumulates `dynext` from step `s` into step `s−1`.
**Check:** `tests/test_model` gradient-checks at **both** `pred_len=1` and
`pred_len=3` (it already calls `check_model(3)` — make sure you can explain why
the feedback term is needed for the check to pass).

### Lab A6 — Swap TimeMix for mean-pooling
**Goal:** see that pooling is just another differentiable op.
**Files:** `src/model.c` (`timemix_forward`/`timemix_backward`).
**Steps:** replace the learned `[out,in]` mixing matrix with a fixed `1/in_len`
average; derive the (trivial) backward.
**Check:** full-model gradient check still passes; compare validation ΔR to the
learned TimeMix (`./typhoformer 20`).

---

## Track B — Extenders: add models behind the seams

### Lab B1 — Add a layer as a pluggable Module
**Goal:** use the extension interface instead of editing the core model.
**Files:** `include/module.h`, `src/module.c`, `tests/test_module.c`.
**Steps:** wrap a new layer (e.g. your GELU-FFN block) as a `Module`
(`self/forward/backward/free`), add it to a `Sequential`.
**Check:** `tests/test_module` gradient-checks your `Sequential` end to end.

### Lab B2 — A different encoder
**Goal:** change architecture without touching the optimizer or checkpoint I/O.
**Files:** `src/model.c` (`Encoder`).
**Steps:** add a pre-norm variant, or a third block type; register any new
weights with the same `ParamList`.
**Check:** full-model gradient check passes; the checkpoint round-trip test still
passes (parameters are discovered via the `ParamList`, so I/O adapts for free).

### Lab B3 — A new optimizer (SGD+momentum or AdamW→Lion)
**Goal:** the optimizer only needs a `ParamList`.
**Files:** `src/optim.c` (copy `Adam`).
**Steps:** implement the update rule over `pl->item[p].v/.g`.
**Check:** training converges (`./typhoformer 20`) and beats persistence.

### Lab B4 — Read a new dataset
**Goal:** feed your own records through the pipeline.
**Files:** `src/data.c` (`dataset_load`), `tools/`.
**Steps:** point the loader at a different CSV schema, or precompute embeddings
with a different sentence model into `.npy`.
**Check:** `tests/test_npy` still passes; `./typhoformer prepare` writes a `.tfb`;
`./typhoformer 5 --bin=...` trains.

---

## Track C — Engineers: performance & deployment

### Lab C1 — Profile, then parallelize
**Goal:** measure before optimizing.
**Files:** `bench` subcommand, `include/parallel.h`.
**Steps:** `./typhoformer bench --full --iters=50` for a baseline; then train
with `--threads=1`, `2`, `4`, `8` and record epoch time.
**Check:** multicore reduces epoch wall-clock; `tests/test_parallel` proves the
gradient is unchanged (≈`1e-7`) so speed didn't cost correctness. Confirm with
ThreadSanitizer: `cc -fsanitize=thread ... tests/test_parallel.c`.

### Lab C2 — A GPU (or SIMD) backend
**Goal:** retarget the compute seam.
**Files:** `include/tensor.h` (contract), `backends/README.md`,
`backends/cuda/tensor_cuda.cu` (reference).
**Steps:** with CUDA available, `cd backends/cuda && make`; or write a new
backend for your device implementing the same ~13 kernels. Start from the honest
caveat in `backends/README.md` about the glue-code elementwise loops.
**Check:** the CPU gradient checks are the acceptance test — a correct backend
passes `tests/test_tensor` and `tests/test_model` unchanged.

### Lab C3 — Embed as a library
**Goal:** run inference from another program.
**Files:** `docs/INTEGRATION.md`, `make lib`.
**Steps:** `make lib` → `libtyphoformer.a`; write a small `main` that loads a
checkpoint (`checkpoint_load_config/params/stats`), calls `model_forward`, and
prints ŷ.
**Check:** your program reproduces `./typhoformer eval` numbers on the same
checkpoint.

### Lab C4 — Guard against regressions
**Goal:** make correctness non-negotiable in CI.
**Files:** `tests/test_golden.c`, `.github/workflows/c-ci.yml`.
**Steps:** after any change, confirm the golden loss (43.21483) is unchanged for
the serial path; if you *intend* to change numerics, update the constant and
explain why in the commit.
**Check:** `make test` and `make test-san` green under both gcc and clang; the
valgrind job is clean.

---

## The rule, restated

Every lab ends at a **green check**, not "it looks right." The gradient checker,
the golden loss, and the sanitizers are the graders. If your change keeps them
green, it's correct by construction; if it doesn't, they tell you exactly where
to look.
