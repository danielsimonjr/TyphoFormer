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

**AdamW** — an adaptive optimizer keeping per-parameter first/second-moment
estimates (`m`, `v`) with bias correction, using **decoupled** weight decay (the
decay acts on the weight directly, not through the gradient). Code: `Adam`,
`src/optim.c`.

**Weight decay (decoupled)** — an L2-style pull of weights toward zero for
regularization; "decoupled" (AdamW) means it is applied separately from the
adaptive gradient term.

**Gradient clipping** — rescaling all gradients so their global L2 norm does not
exceed a cap (`--clip`), preventing a single large step from destabilizing
training. Code: `plist_clip_grad_norm`, `src/nn.c`.

**Dropout** — during training, randomly zeroing a fraction `p` of activations
(and scaling the rest by `1/(1-p)`) so the model can't over-rely on any unit;
disabled at eval. Applied after attention and after the FFN in each block.
Code: `dropout_apply`, `src/nn.c`.

**Learning rate (lr) / schedule** — the step size. **Warmup** (`--warmup`) ramps
it up linearly over the first N steps; **decay** (`--lr_decay`) multiplies it each
epoch. **Early stopping** halts when validation stops improving (`--patience`).

**Data leakage** — when information from validation/test data influences training
(e.g. normalization fit on the whole dataset, or overlapping windows split across
sets). It makes reported metrics look better than real generalization. Avoided
here by storm-level splitting and train-only statistics.

**Storm-level split** — assigning whole storms (not individual sliding windows)
to train/val/test, so overlapping windows from one storm never straddle two
sets. Code: `dataset_split3`, `src/data.c`.

**Held-out test set** — a partition never used for training *or* model
selection; the honest generalization estimate. Distinct from the validation set
used for early stopping / checkpoint selection.

**Coordinate normalization** — z-scoring the target lat/lon (with train-only
stats) so the regression targets are unit-scaled and balanced; predictions are
de-normalized back to degrees for metrics. Code: `dataset_denorm`, `src/data.c`.

**Motion features** — the storm's position and velocity (`lat, lon, Δlat, Δlon`)
added to the model input (`--motion`). By default the inputs are intensity only,
so the model is blind to how the storm is moving; adding motion is the single
biggest improvement to forecast skill. Code: `dataset_add_motion`, `src/data.c`.

**Displacement head (delta mode)** — a decoder that outputs the *change* from the
previous coordinate (`ŷ_t = ŷ_{t-1} + Δ`) instead of the absolute position, with
the output layer zero-initialised so it starts at persistence and learns the
correction (`--delta`). Improves accuracy and collapses cross-split variance.
Code: `model_set_delta`, `decoder_forward`. (Also called a residual head.)

**km-aware loss** — weighting the longitude error by `cos²(latitude)` so the
training objective approximates squared kilometres (a degree of longitude is
`cos(lat)` shorter than a degree of latitude). Implemented as `--km_loss`; tested
and found *not* to help on this data (an honest null result).

**Constant-velocity / CLIPER baseline** — extrapolate the last observed velocity
linearly. The honest bar a learned model must beat (~39 km @ 6h here), much
stronger than persistence. Code: `cmd_baseline`, `src/train.c`.

**Resume** — continuing an interrupted training run from a checkpoint plus its
`.opt` sidecar (Adam moments + step + lr), so the optimizer state is restored
rather than restarted. Code: `checkpoint_save_optim`, `--resume`.

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
