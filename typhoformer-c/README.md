# TyphoFormer-C

A **clean-room, pure-C reimplementation** of the TyphoFormer typhoon-track
forecasting model — Prompt-aware Gating Fusion (PGF) + a spatio-temporal
Transformer encoder + an autoregressive decoder.

## 📚 Documentation

In-depth docs for students and engineers live in [`docs/`](docs/):

| Doc | For | Contents |
|:--|:--|:--|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | students | The model + **full forward/backward math** for every layer, matching the code. |
| [docs/THEORY_MAP.md](docs/THEORY_MAP.md) | both | **Theory ↔ code map**: every equation → the exact `file:function` implementing it. |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | both | Plain-language **definitions** of every term, linked to code. |
| [docs/LABS.md](docs/LABS.md) | both | **Hands-on labs** graded by a gradient check / the golden loss — understand, extend, deploy. |
| [docs/API.md](docs/API.md) | engineers | Complete API reference + the **memory / ownership / concurrency** model. |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | engineers | Embed as a library (`make lib`), inference from a checkpoint, byte-exact file formats, serving. |
| [docs/EXTENDING.md](docs/EXTENDING.md) | both | Add layers/ops (gradient-check discipline), the **Module** interface, the **backend** seam, multicore. |

Start at [docs/README.md](docs/README.md) for the suggested learning path.

## Scope & design

- **Pure C, standard library only.** No third-party dependencies — all math
  (matmul, softmax, layer norm, attention, Adam) is hand-written. Links only
  against `libm` (and `pthread` for optional multicore training).
- **Full pipeline.** Data preparation, training (forward + backprop + Adam),
  multi-step autoregressive inference, evaluation, and baselines are all in C.
- **Built to be extended.** A pluggable [`Module`](include/module.h) interface
  for new layers, a documented [compute-backend seam](include/backend.h) with
  **two reference backends** under [`backends/`](backends/) — a runnable
  **OpenCL** backend (verified on a CPU device via POCL) and a **CUDA** backend
  (compiles with `nvcc`) — and [data-parallel multicore](include/parallel.h)
  training via pthreads. Each is covered by its own gradient/equivalence test.
- **Language branch is precomputed.** The GPT-4o descriptions and MiniLM
  sentence embeddings are produced offline (outside this C code); the C
  program consumes the resulting numerical embedding vectors.

## Clean-room / licensing note

This directory is an **independent implementation written from the published
method** — the paper and the algorithm pseudocode / data-flow specification in
the repository root (`TyphoFormer_algorithm.tex`, README figures) — *not* a
line-by-line translation of the upstream Python. Because it is original code
implementing (non-copyrightable) algorithms and math, it carries its own
license: **MIT** (see [`LICENSE`](LICENSE)). This license applies only to the
contents of this `typhoformer-c/` directory, not to the upstream repository.
The copyright line reads "TyphoFormer-C contributors" — edit it to your name
or organization as appropriate.

## Build & run

```bash
cd typhoformer-c
make            # build the ./typhoformer binary
make test       # build and run all unit tests (gradient checks + data loader)

./typhoformer 30                     # 30 epochs, compact demo config, CSV data
./typhoformer 5 --full               # full paper config (d_model=256, 3 layers)
./typhoformer 30 --csv=PATH --emb=DIR
```

Requires a C11 compiler (`gcc`/`clang`) and `make`. The core is portable C;
the data loader uses POSIX directory APIs (`scandir`) with a Windows
`FindFirstFile` branch provided (compiled on POSIX here, not yet on Windows).

## Command-line tools

The `./typhoformer` binary provides subcommands (the default is `train`, so
`./typhoformer 30` still trains):

| Command | Purpose | Example |
|:--|:--|:--|
| `train`    | Train; writes a checkpoint (config header + normalization stats). | `./typhoformer train 30 --full --save=m.ckpt` |
| `eval`     | Load a checkpoint — or a comma-separated **ensemble**, predictions averaged — and evaluate (MAE + spherical ΔR, per horizon). `--split=test --split_seed=S` re-derives the training partition and scores only the held-out storms; `--threads=N` shards eval across cores. | `./typhoformer eval --weights=a.ckpt,b.ckpt --split=test --split_seed=5` |
| `prepare`  | Build sliding-window samples from CSV + embeddings and write a `.tfb`. | `./typhoformer prepare --out=data.tfb` |
| `predict`  | Write predicted vs. true tracks (with per-step error) to CSV. | `./typhoformer predict --weights=m.ckpt --n=50 --out=pred.csv` |
| `baseline` | Report persistence and constant-velocity (CLIPER-style) baselines. | `./typhoformer baseline --pred_len=4` |
| `bench`    | Forward / forward+backward throughput (ms/sample, samples/s). | `./typhoformer bench --full --iters=50` |

### Training flags

`train` takes hyperparameters on the command line (all optional):

| Flag | Meaning | Default |
|:--|:--|:--|
| `--full` | Full paper config (`d_model=256`, 3 layers) instead of the compact demo. | compact |
| `--d_model= --d_ff= --n_layers= --n_heads= --in_len= --pred_len=` | Architecture overrides. | config |
| `--batch=` | Minibatch size. | 8 |
| `--threads=` | **Data-parallel training across N CPU cores** (1 = serial). | 1 |
| `--lr= --wd= --lambda=` | Learning rate, **AdamW** weight decay, gate-penalty weight. | 1e-3 / 1e-5 / 0.1 |
| `--dropout=` | Dropout rate (post-attention and post-FFN; off at eval). | 0.1 |
| `--clip=` | Global-norm gradient clipping (0 = off). | 1.0 |
| `--warmup=` | Linear LR warmup steps (0 = off). | 0 |
| `--lr_decay=` | Per-epoch LR multiplier (1.0 = off). | 1.0 |
| `--patience=` | Early stop after N epochs without val improvement (0 = off). | 0 |
| `--swa` / `--swa_start=E` | **Stochastic Weight Averaging**: average the weights over the late-epoch tail (default: second half; `E` sets the first epoch) instead of keeping the single best-val checkpoint. Disables early stopping so the window is well-defined; reports **both** the best-checkpoint and SWA held-out numbers, and saves the average as `<save>.swa`. Measured **neutral at 6h but −6 km at 48h (4/5 seeds)** — recommended for long-horizon forecasting (FINDINGS §20). | off |
| `--resume=CKPT` | Resume weights **and optimizer state** from a checkpoint + its `.opt` sidecar. | — |
| `--motion` | **Feed position + velocity** (lat, lon, Δlat, Δlon) as input features — the trajectory signal the model otherwise never sees. | off |
| `--physics` | **Second-order physics features**: acceleration (Δ²lat, Δ²lon), translation speed, heading unit vector, and seasonal day-of-year phase (sin/cos). Composes with `--motion` (+7 features). | off |
| `--delta` | **Displacement head**: the decoder predicts the change from the seed (`ŷ_t = ŷ_{t-1} + Δ`, fc2 zero-init → starts at persistence, learns the correction). | off |
| `--cv` | **Constant-velocity decoder** (2nd-order delta): anchors the rollout at constant-velocity extrapolation (`ŷ_t = y_{t-1} + v + fc2(...)`, v threaded across steps, fc2 zero-init) so an untrained model starts *at the CLIPER baseline* and learns only curvature. Supersedes `--delta`. | off |
| `--rotframe` | cv correction predicted in the **motion-aligned frame** (along/cross-track, rotated by the velocity direction; implies `--cv`; parameter-free but eval must match). Tested; did **not** help (worse at 6h — FINDINGS §13). | off |
| `--gru` | Constant-velocity decoder whose curvature correction is produced by a **GRU** with hidden state carried across the rollout (initialised from the encoder context) — gives the multi-step rollout real memory. | off |
| `--xattn` | Constant-velocity decoder whose per-step context comes from **cross-attention over the encoder's full sequence** (not the pooled vector). | off |
| `--km_loss` | Weight the longitude error by `cos²(lat)` (km-aware objective). Tested; did **not** help — off by default. | off |
| `--huber=` | **Huber loss** with transition point δ on the normalized residual (quadratic core = MSE, linear tails) — tempers fast-moving outlier storms. 0 = plain MSE. | 0 |
| `--hweight=` | **Horizon-weighted loss**: forecast step h weighted `(h+1)^γ` (mean-normalized) — γ>0 upweights the long horizons that dominate the km error. 0 = uniform. | 0 |
| `--tf=` | **Teacher forcing** on the cv rollout: each training step's output state is replaced by ground truth with a probability annealed 1→0 over E epochs (recurrence gradient cut at forced steps; eval always autoregressive). | 0 |
| `--no_lon` | **Ablation**: zero `--motion`'s absolute-longitude column (keeps d_num/checkpoint layout; pass to `eval` too). Tests climatology-signal vs memorization. | off |
| `--defer_dw` | Deferred per-batch dW GEMMs instead of per-sample accumulation. Bit-identical math; measured **neutral** on the reference 4-core box (backward is compute-bound at these sizes) — kept for hardware where gradients exceed the last-level cache. | off |
| `--spatial` | Restore the paper's N=1 spatial encoder blocks. They are **off by default** (their Q/K never train and dropping them is accuracy-neutral — FINDINGS §7 — for ~2× less encoder compute); required to load checkpoints trained before the default changed. `--no_spatial` is accepted as a no-op. | off |
| `--posenc` | Learned positional encoding after `input_proj` — makes temporal attention order-aware. | off |
| `--pool=last` | Pool the encoder by the last time step instead of the learned TimeMix average. | off |
| `--prenorm` | Pre-norm transformer blocks (LN before each sublayer) instead of post-norm. | off |
| `--timebias` | ALiBi-style temporal-distance bias in the temporal attention (`score[i,j] −= slopeₕ·\|i−j\|`) — a parameter-free sense of recency. | off |
| `--co_spatial` | **Real** multi-node spatial attention: the encoded context attends over the relative states of storms active at the same timestep. | off |
| `--no_text` | **Ablation**: zero the language-embedding branch (numbers-only model). | off |
| `--split_seed=` | Seed for the storm-level train/val/test partition (vary for a variance estimate). | 42 |
| `--seed=` | RNG seed (determinism). | 20260711 |
| `--csv= --emb= --bin= --save=` | Data source / checkpoint path. `--emb=none` = **text-free mode**: the language branch is fed zeros (== `--no_text`), so CSVs converted from raw HURDAT2 train without any embeddings. | repo defaults |

**For accuracy, train with `--motion --physics --cv --huber=0.1`** (the
best-known recipe): motion features give the model the trajectory signal, the
physics features add the curvature signal, the constant-velocity anchor lets
it start at CLIPER and learn only the correction, and the Huber loss tames
outlier storms. On the full 826-storm dataset
(`--csv=../HURDAT_full.csv --emb=none`) it **beats split-matched
constant-velocity over the 48-hour rollout on all five splits** — crossover at
24h, **−8.4% at 48h** — while one-step (6h) dead reckoning remains ahead; on
the bundled 98-storm sample it scores 38.5 km @ 6h, under that sample's
39.4 km CLIPER bar (see [docs/FINDINGS.md](docs/FINDINGS.md) §14–§15).
All decoder variants work on both the serial and the data-parallel
(`--threads=N`) paths. `eval`/`predict` auto-detect `--motion`/`--physics` from
the checkpoint's feature count; pass `--delta`/`--cv` to `eval` to match the
checkpoint's decoder.

`--no_text` is the key scientific control: it tests whether the GPT-4o/MiniLM
language branch — the paper's central premise — actually helps versus a
numbers-only model. Vary `--split_seed` to see how much the held-out number moves
with the storm partition (a poor-man's cross-validation).

Training uses a **leakage-safe** pipeline: whole storms are split into
train/val/test, feature and coordinate normalization is fit on the **training
storms only**, and the final number is scored on a **held-out test set** the
model never saw during training or selection.

Set `--pred_len=N` (N > 1) to forecast multiple 6-hourly steps autoregressively;
`eval`/`predict` then report **per-horizon** metrics (6h, 12h, 18h, …).

### Multicore training

`--threads=N` runs synchronous data-parallel SGD: the minibatch is split across
N model replicas (one per core), gradients are summed, and the optimizer takes a
single step. It is numerically equivalent to serial training up to
floating-point summation order (`tests/test_parallel` pins this to ≈1e-7), and
`--threads=1` keeps the original serial path byte-for-byte.

Workers feed the same per-sample auxiliary inputs as the serial loop (seed
velocity, co-active neighbours), so **every decoder variant (`--cv`/`--gru`/
`--xattn`) and `--co_spatial` trains data-parallel too** — `tests/test_parallel`
pins serial/parallel gradient equivalence for each of them on the real dataset.
The one remaining serial-only flag is `--km_loss`.

```bash
./typhoformer train 30 --full --threads=8 --save=m.ckpt
```

Plus the offline preprocessing under `tools/` — these two stages need large
external models (GPT-4o, MiniLM), so they are **Python by necessity** (the C
core consumes their output, precomputed embeddings):

| Tool | Purpose |
|:--|:--|
| `tools/gen_descriptions.py` | GPT-4o natural-language descriptions per record (needs `OPENAI_API_KEY`). |
| `tools/gen_embeddings.py`   | MiniLM (`all-MiniLM-L6-v2`) 384-d embeddings → `emb_chunk_*.npy`. |
| `tools/npy_dict_to_bin.py`  | Convert the pre-split pickled `data/{train,val,test}/*.npy` dicts → `.tfb`. |
| `tools/hurdat2_to_csv.py`   | Convert a **raw NOAA HURDAT2** database file → this repo's CSV schema, for scaling past the bundled 98 storms. Train the result text-free: `--csv=... --emb=none` (the language branch measurably doesn't help — FINDINGS §2/§6 — so no GPT-4o/MiniLM regeneration is needed). |

End-to-end pipeline (from raw records):

```bash
OPENAI_API_KEY=... python tools/gen_descriptions.py ../HURDAT_2new_3000.csv ../desc.csv
python tools/gen_embeddings.py ../desc.csv ../embedding_chunks
./typhoformer prepare --out=../data.tfb        # optional: cache windows
./typhoformer train 30 --save=model.ckpt
./typhoformer eval  --weights=model.ckpt
```

### Consuming the repository's pre-split `.npy` data

The pre-split sample files under `data/{train,val,test}/` are pickled Python
dicts, which a pure-C loader cannot read directly. Convert them once (needs
Python + numpy) to a flat `.tfb` binary, then train from it:

```bash
python tools/npy_dict_to_bin.py ../data/val ../data/val.tfb
./typhoformer 20 --bin=../data/val.tfb
```

> Those files store only `input` (numerical + embedding features) and
> `target` — no coordinates — so this path seeds the decoder with the first
> target coordinate, reproducing the upstream setup. The CSV path uses the
> true last-observed coordinate and is the sound default.

### Performance

Math is hand-written and cache-aware (`ikj`/`pij` accumulation loop orders,
`restrict` pointers), built at `-O3`. The linear-layer forward kernel
(`mat_matmul_bt`) splits each dot product across 8 independent accumulators,
breaking the float-addition dependency chain so the compiler can pipeline and
auto-vectorize it without `-ffast-math` — this alone made the full-config
forward pass 5.6× faster. Backward passes reuse persistent per-module scratch
buffers, so training does no per-step heap allocation.

Measured on a 4-core container, single-threaded (epoch = 1097 train samples +
per-epoch validation; your hardware will vary — rerun `bench` to check). The
default encoder now skips the dead N=1 spatial blocks (full config 2.77 M
params; `--spatial` restores the paper's 5.14 M — bench it with
`bench --full --spatial`):

| Build | Compact config | Full config (2.77 M params) |
|:--|:--:|:--:|
| `make` (portable `-O3`) | ~1 s/epoch | ~23 s/epoch |
| `make NATIVE=1` (`-march=native`) | ~1 s/epoch | **~18 s/epoch** |

(`bench`, full config: portable 4.1 ms forward / 14.5 ms forward+backward per
sample; NATIVE 3.3 / 10.9 ms. For a portable AVX2 binary between the two, use
`make MARCH=x86-64-v3`.) `--threads=N` adds data-parallel scaling across cores
on top (see **Multicore training** above).
For a GPU or other accelerator, implement the ~13-kernel contract in
[`include/backend.h`](include/backend.h): [`backends/opencl/`](backends/opencl/)
is a **runnable** OpenCL backend (`make OPENCL=1 test-opencl` verifies the whole
model through it — works on a real GPU or on a CPU via POCL), and
[`backends/cuda/`](backends/cuda/) is a CUDA backend that compiles with `nvcc`.
See [`backends/README.md`](backends/README.md).

## Development

| Target | What it does |
|:--|:--|
| `make` | Build `./typhoformer` (portable `-O3`). |
| `make lib` | Build `libtyphoformer.a` for embedding in another application (see [docs/INTEGRATION.md](docs/INTEGRATION.md)). |
| `make test` | Build + run all unit tests (gradient checks, golden, module, parallel, checkpoint, npy). |
| `make test-san` | Rebuild with **AddressSanitizer + UBSan** and run the tests. |
| `make test-valgrind` | Run the whole test suite under **valgrind** (memcheck + leak-check). |
| `make OPENCL=1 test-opencl` | Build the model on the **OpenCL** backend and verify it (kernels vs CPU + full-model gradient check). Needs an OpenCL ICD (`pocl-opencl-icd` for CPU). |
| `make NATIVE=1` | Build with `-march=native -funroll-loops` (faster, non-portable binary). |
| `make MARCH=x86-64-v3` | Portable SIMD tier between plain `-O3` and NATIVE (AVX2+FMA; runs on any x86-64 CPU from ~2015 on). |
| `make -C backends/cuda` | Compile the **CUDA** backend with `nvcc` → `libtyphoformer_cuda.a`. |
| `make clean` | Remove build artifacts. |

The code builds warning-clean under `-Wall -Wextra -Wpedantic`, is
AddressSanitizer/UBSan/valgrind-clean and ThreadSanitizer-clean (multicore path)
across all runtime paths, and is exercised in CI (`.github/workflows/c-ci.yml`):
a **gcc + clang** build/test matrix, sanitizer tests, a valgrind pass, an
**OpenCL** job (runs the model through POCL), a **CUDA** `nvcc` compile-check,
and a train/multicore/eval/prepare/predict/baseline/bench smoke test.

The unit tests double as the correctness contract for every extension seam:

| Test | Proves |
|:--|:--|
| `test_tensor`, `test_nn`, `test_model` | forward/backward gradients match finite differences (incl. `pred_len=3`). |
| `test_golden` | training loss is bit-stable (deterministic regression guard). |
| `test_module` | a `Sequential` of pluggable `Module`s backprops correctly. |
| `test_parallel` | multicore gradient == serial gradient (≈1e-7), incl. the cv/gru/xattn/co_spatial variants on the real dataset. |
| `test_checkpoint`, `test_npy` | checkpoint round-trip (TFW1/2/3 + optimizer sidecar), `.npy` dtype/fortran validation, storm-safe split. |
| `test_dropout` | dropout is identity at eval; its backward (pinned-mask) gradient-checks. |
| `backends/opencl/test_opencl` | OpenCL kernels match the CPU reference (float-rounding level); model runs through OpenCL. |

## Status — complete

- [x] Tensor core (matmul variants, activations) + gradient-check harness
- [x] NN layers (Linear, multi-head attention, LayerNorm, FFN) — block gradient check
- [x] Model (PGF fusion, ST-encoder, autoregressive decoder) — full-model gradient check
- [x] `.npy` data loader + sliding-window dataset
- [x] Loss (MSE + gate penalty), Adam optimizer, training loop
- [x] Evaluation (MAE, spherical-distance error) + persistence / CLIPER baselines
- [x] Multi-step autoregressive decoding + per-horizon metrics
- [x] `train`/`eval`/`prepare`/`predict`/`baseline`/`bench` subcommands + config CLI
- [x] **Leakage-safe** data pipeline: storm-level train/val/**test** split, train-only stats
- [x] Dropout, AdamW (decoupled), gradient clipping, LR warmup, resume from checkpoint
- [x] Best-checkpoint saving (with feature + coord stats), LR decay, early stopping
- [x] Pluggable `Module` interface + compute-backend seam (CPU + runnable OpenCL + CUDA)
- [x] Data-parallel multicore training (pthreads) — gradient-equivalence tested
- [x] gcc/clang CI matrix, sanitizers, valgrind, golden regression

Every layer is validated by finite-difference gradient checks (tensor core, a
full transformer block, and the whole model — all at the single-precision noise
floor). But the more important question is whether it *forecasts* — see
[docs/FINDINGS.md](docs/FINDINGS.md). On the **held-out test storms** (never seen
in training or selection), varying only the storm split:

Held-out test ΔR across 5 storm splits (compact config). The honest bar is
**constant-velocity extrapolation (~39 km @ 6h)**, not persistence (~123 km):

| model | held-out test ΔR | vs persistence |
|:--|:--:|:--:|
| default (intensity + text) | 131.1 ± 39.6 km | ~parity |
| **`--motion`** (feed position + velocity) | 76.2 ± 30.6 km | beats it |
| **`--motion --delta`** (predict displacement) | 48.4 ± 2.6 km | 2.5× better |
| `--motion --delta --no_text` (numbers only) | 46.7 ± 4.8 km | 2.6× better |
| **`--motion --cv`** (constant-velocity decoder) | **40.8 km** | **reaches CLIPER (~39 km)** |
| **`--motion --physics --cv`** (+ heading/accel/season; no-spatial default) | **39.1 km** | **on CLIPER; better on 5/5 splits** |
| **`--motion --physics --cv --huber=0.1`** (the recipe) | **38.5 km** | **under CLIPER (~39.4 km); best-known at 48h too** |

`--cv` anchors the decoder at constant-velocity extrapolation and learns only the
curvature — the first architectural change that reaches the constant-velocity
baseline the model had been trailing (see [FINDINGS §9](docs/FINDINGS.md)), and
at **longer horizons** it passes it: over a 48-hour rollout, `--cv` beats
split-matched constant-velocity on 4 of 5 storm splits (−9.3% at 48h). Adding
decoder memory (`--gru`) or cross-attention (`--xattn`) on top did **not**
survive a five-split re-test — an earlier three-split signal turned out to be
split noise (see [FINDINGS §10–§11](docs/FINDINGS.md)).

The full story is in [**docs/FINDINGS.md**](docs/FINDINGS.md). In short: the
default model was **blind to motion** (its inputs are intensity + text; position
and velocity were never fed in), so it lost to a two-line physics baseline.
Feeding motion, anchoring at constant-velocity, adding physics features and a
Huber loss takes held-out ΔR from **131 km → 38.5 km** on this sample — and
trained on the full three-basin dataset (826 storms, `tools/fetch_hurdat2.sh`)
the same recipe **beats split-matched constant-velocity over the 48-hour
rollout on all five splits** (−8.4% at 48h; one-step dead reckoning stays
ahead at 6h). And the
**language branch still doesn't help** even with a working model
(numbers-only is marginally *better*), a robust negative result on this data.

> **Honesty note.** Earlier versions reported ΔR ≈ 79 km "beating persistence."
> That was data leakage — normalization fit on the whole dataset and overlapping
> windows split randomly, so validation storms bled into training. With a
> storm-level split, train-only statistics, and a real held-out test set (all now
> enforced), the favourable number disappears. The gradient checks, golden test,
> and cross-backend agreement confirm the *math* is correct; this much data (~98
> storms) is simply too little for the model to beat persistence robustly. The
> point of the fixes is that the numbers you see are now real.

## Notes

- `make test` runs from `typhoformer-c/`; the data loader reads the CSV and
  embedding chunks from the repository root (`../`).
- The training binary uses a compact instance of the architecture (smaller
  `d_model`/`d_ff`) for a fast self-contained demo; the full paper
  configuration (`--full`) runs through the identical code path.
- The decoder is seeded with the last *observed* coordinate (paper faithful);
  targets are coordinate-normalized for training and de-normalized to km for
  metrics.
