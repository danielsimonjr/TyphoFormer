# Glossary

Plain-language definitions of the terms used in the code, the docs, and the
paper. Grouped by theme. Where a term maps to code, the `file:symbol` is given.

## Model architecture

**TyphoFormer** — the model in this repo: a Transformer for typhoon/hurricane
track forecasting that fuses numerical track features with language-model
embeddings of a natural-language description of each record.

**Prompt-aware Gating Fusion (PGF)** — the input stage. It learns a per-feature
gate `g = σ(W_g[xₙ;xₜ])` and blends the (projected) numerical stream `xₙ′` and
text stream `xₜ′` as `x̃ = g⊙xₙ′ + (1−g)⊙xₜ′`. "Prompt-aware" because the gate
is driven by the language embedding. Code: `PGF`, `src/model.c`.

**Spatio-temporal encoder** — stacks **temporal** Transformer blocks (attention
across time steps) and **spatial** blocks, then pools the time axis to a single
vector. Code: `Encoder`, `src/model.c`.

**Temporal vs. spatial attention** — temporal attention lets each time step
attend to all others (`self_only=0`). Spatial attention here is *single-node*
(`self_only=1`): with one storm track there is no spatial neighborhood, so each
position attends only to itself — the identity attention map. The seam is left
in so multi-node/graph attention can drop in later.

**TimeMix** — a small learned linear map over the *time* axis that pools
`[in_len, D] → [1, D]` (a learned weighted average across steps), rather than a
fixed mean-pool. Code: `TimeMix`, `src/model.c`.

**Autoregressive (AR) decoder** — produces the forecast one step at a time,
feeding each predicted coordinate back in as the input for the next step. Code:
`Decoder`, `src/model.c`. With `pred_len > 1` the backward pass threads the
gradient back through the feedback path (`decoder_backward`).

**Encoder / decoder** — the encoder turns the input window into a fixed context
vector `h_enc`; the decoder turns `h_enc` (plus the last known coordinate) into
the multi-step forecast.

## Transformer building blocks

**Linear layer** — `y = xWᵀ + b`. The workhorse; three matrix products cover its
forward and both gradients. Code: `Linear`, `src/nn.c`.

**Multi-head attention (MHA)** — projects inputs to queries `Q`, keys `K`,
values `V`; computes `softmax(QKᵀ/√hd)·V` in `H` parallel heads; concatenates and
projects out. Code: `MHA`, `src/nn.c`.

**Head / head dim (hd)** — attention is split into `H` independent
lower-dimensional sub-spaces of size `hd = D/H`; `√hd` scales the dot products.

**Softmax** — turns a row of scores into a probability distribution
`e^{zᵢ}/Σe^{zⱼ}`; here it normalizes attention weights over keys. Code:
`softmax_rows`, `src/nn.c`.

**LayerNorm** — normalizes each row to zero mean / unit variance, then applies a
learned scale `γ` and shift `β`. Stabilizes training. Code: `LayerNorm`.

**Feed-forward network (FFN)** — the per-position `Linear→ReLU→Linear` sub-layer
inside a Transformer block. Code: `FFN`, `src/nn.c`.

**Residual connection** — adding a sub-layer's input to its output (`x + f(x)`)
so gradients flow directly; used around MHA and FFN. Code: inside `block_forward`.

**Post-norm** — LayerNorm applied *after* the residual add (this code's
convention), vs. pre-norm which normalizes before the sub-layer.

**Transformer block** — MHA + residual + LN, then FFN + residual + LN. Code:
`Block`, `src/nn.c`.

## Training & optimization

**Forward pass** — compute the output (and cache intermediates needed later).

**Backward pass / backpropagation** — apply the chain rule in reverse to get
`∂loss/∂parameter` for every weight, reusing the cached forward values.

**Gradient** — the vector of partial derivatives of the scalar loss w.r.t. a
parameter; points in the direction of steepest increase, so we step *against* it.

**Finite-difference gradient check** — the correctness test used everywhere here:
compare the analytic backward gradient to `(L(w+ε) − L(w−ε)) / 2ε`. If they
disagree beyond the single-precision noise floor, the backward pass is wrong.
Code: `tests/test_*.c`.

**Loss** — what training minimizes: mean-squared error between predicted and true
coordinates, plus a gate penalty `λ·mean(relu(0.6−g)²)` that discourages the gate
from collapsing. Code: `model_loss`, `src/model.c`.

**Adam** — an adaptive optimizer keeping per-parameter first/second-moment
estimates (`m`, `v`) with bias correction; combined here with decoupled weight
decay. Code: `Adam`, `src/optim.c`.

**Weight decay** — an L2-style pull of weights toward zero; regularization.

**Learning rate (lr) / LR schedule** — the step size; `--lr_decay` multiplies it
each epoch. **Early stopping** halts when validation stops improving
(`--patience`).

**Epoch / minibatch / batch size** — one epoch = one pass over the training set;
a minibatch is a small group of samples whose gradients are averaged before one
optimizer step (`--batch`).

**Data-parallel SGD** — process a minibatch across CPU cores: replicate the model,
run each core on a shard, sum the gradients, take one step (`--threads`). Code:
`ParTrainer`, `src/parallel.c`. Equivalent to serial training up to
floating-point summation order.

## Data & evaluation

**HURDAT2** — NOAA's Atlantic hurricane best-track database; the numerical source
records (position, wind, pressure, wind radii). Records are 6-hourly.

**Sliding window / sample** — `in_len` consecutive records as input and the next
`pred_len` records as the target, slid along each storm track. Code: `Dataset`,
`src/data.c`.

**Embedding** — a fixed-length numeric vector summarizing text. Here, MiniLM
(`all-MiniLM-L6-v2`) turns each record's description into a 384-d vector,
precomputed offline into `.npy` files.

**Standardization / z-score** — rescaling each numerical feature to zero mean,
unit std using training statistics; those `mean`/`std` are persisted in the
checkpoint so inference reproduces them. Code: `checkpoint_save2`.

**MAE (mean absolute error)** — average absolute coordinate error (in the model's
lat/lon units).

**ΔR / spherical-distance error** — great-circle distance in kilometers between
predicted and true position (haversine). The headline metric. Code: `haversine`,
`src/train.c`.

**Persistence baseline** — "assume it doesn't move": predict the last observed
position. The floor any real model must beat. Code: `cmd_baseline`.

**CLIPER / constant-velocity baseline** — extrapolate the last observed velocity
linearly. A stronger classical baseline. Code: `cmd_baseline`.

**Forecast horizon** — how far ahead a prediction is (6h, 12h, …). Per-horizon
metrics are reported because error grows with lead time. Code: `Eval`,
`src/train.c`.

## Systems & implementation

**`Mat`** — the row-major float matrix struct (`rows`, `cols`, `data`); the one
data type all math operates on. Code: `include/tensor.h`.

**ParamList** — a flat registry of every trainable parameter (value + gradient
buffer). One list is shared by the model, optimizer, checkpoint I/O, and the
gradient checker, so they all see parameters uniformly. Code: `nn.h`.

**Kernel / backend** — the small fixed set of matrix primitives
(`mat_matmul`, `mat_relu`, …) that every layer is built from. Swapping the file
that implements them retargets the whole model to a new device. Code:
`include/backend.h`, `src/tensor.c` (CPU), `backends/cuda/` (GPU reference).

**Module (vtable)** — the polymorphic extension interface: a struct of function
pointers (`forward`/`backward`/`free`) wrapping any layer, so new layers compose
in a `Sequential` without editing the core model. Code: `include/module.h`.

**Scratch buffer** — a persistent per-module workspace reused every step (via the
`ensure()` idiom), so training does no per-step heap allocation.

**Checkpoint (`.ckpt`, TFW1/TFW2)** — the on-disk model file: a magic tag, the
config header, optional normalization stats (TFW2), then parameters in
registration order. Code: `src/checkpoint.c`.

**`.tfb`** — a flat cache of pre-windowed samples (no coordinates), produced by
`prepare` / `npy_dict_to_bin.py`. Code: `dataset_load_bin`.

**`.npy`** — NumPy's array file format; the C loader reads v1.0/2.0 float32
arrays for the embeddings. Code: `npy_load_2d`, `src/data.c`.

**Sanitizer (ASan/UBSan/TSan)** — compiler instrumentation that catches memory
errors, undefined behavior, and data races at runtime. `make test-san` runs the
suite under ASan+UBSan; `make test-valgrind` runs it under valgrind.

**Determinism** — same seed → same result. Guaranteed by a seeded xorshift RNG
(`nn_seed`/`nn_uniform`) and pinned by the golden-loss regression test.
