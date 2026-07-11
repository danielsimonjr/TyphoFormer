# TyphoFormer-C Documentation

In-depth documentation for the pure-C TyphoFormer implementation — written both
for **students** learning how a Transformer (and its backpropagation) works from
first principles, and for **engineers** embedding the model into real
applications.

All math in these docs is written in plain text / code blocks (no LaTeX), so it
renders everywhere including mobile.

## The documents

| Doc | For | What's inside |
|:--|:--|:--|
| [ARCHITECTURE.md](ARCHITECTURE.md) | students | The model end to end, and the **full forward + backward math** for every layer (Linear, LayerNorm, softmax, attention, FFN, block, PGF, decoder, loss, Adam). Derivations match the code line for line. |
| [THEORY_MAP.md](THEORY_MAP.md) | both | A one-page **theory ↔ code map**: every equation → the exact `file:function` that implements it (forward and backward), plus the extension seams. |
| [GLOSSARY.md](GLOSSARY.md) | both | Plain-language **definitions** of every term (architecture, training, data, systems), each linked to where it lives in code. |
| [LABS.md](LABS.md) | both | **Hands-on exercises** graded by an objective check (a gradient check passes / the golden loss is unchanged) — three tracks: understand, extend, deploy. |
| [API.md](API.md) | engineers | Complete public API reference and the **memory / ownership / lifetime model** — who allocates, who frees, what aliases, and what is thread-safe. |
| [INTEGRATION.md](INTEGRATION.md) | engineers | Embedding the code as a **library** in your own C application: inference from a checkpoint, the on-disk file formats (byte-exact), error handling, threading, and performance. |
| [EXTENDING.md](EXTENDING.md) | both | How to **add a new layer or op** with the gradient-check discipline, the pluggable Module interface, the compute-backend seam, and multicore training. |

## Suggested learning path (students)

1. **Read the code in dependency order.** `include/tensor.h` → `include/nn.h` →
   `include/model.h`. Each header is short and commented.
2. **Work through [ARCHITECTURE.md](ARCHITECTURE.md)** with the source open next
   to it. For each layer, read the forward, then re-derive the backward yourself
   before reading the derivation.
3. **Run the gradient checks** (`make test`). Open `tests/test_nn.c` and
   `tests/test_model.c` — the finite-difference check is the single most
   important tool for understanding whether your backward pass is correct.
4. **Break something on purpose.** Introduce a bug in one backward pass (e.g.
   drop the `scale` in attention) and watch the gradient check catch it. This is
   how you build intuition for what each term does.
5. **Extend it** — follow [EXTENDING.md](EXTENDING.md) to add a layer (GELU, a
   bias-free LayerNorm, dropout, a second decoder step) and gradient-check it.
6. **Do the labs** — work through [LABS.md](LABS.md) Track A, then B/C. Keep
   [GLOSSARY.md](GLOSSARY.md) and [THEORY_MAP.md](THEORY_MAP.md) open beside the
   source as a dictionary and an index.

## Suggested path (engineers)

1. Skim [ARCHITECTURE.md](ARCHITECTURE.md) §1–§2 for the data shapes and the
   module graph.
2. Read [API.md](API.md) — especially the **Memory model** and **Ownership**
   sections — before calling anything.
3. Follow [INTEGRATION.md](INTEGRATION.md) for a minimal inference program and
   the checkpoint / dataset formats.

## Conventions used throughout

- `[R, C]` denotes a matrix with `R` rows and `C` columns, stored **row-major**
  in a flat `float` array (`Mat`).
- `T` = input length (time steps), `D` = model dim, `H` = number of heads,
  `hd = D/H` = head dim, `S` = a generic sequence length.
- `⊙` = elementwise (Hadamard) product, `σ` = logistic sigmoid, `Σ` = sum,
  `∂L/∂x` (written `dx` in code) = gradient of the scalar loss w.r.t. `x`.
- A function that computes gradients **accumulates** into `.g` buffers with `+=`
  (so a mini-batch sums naturally); the optimizer zeroes them each step.
