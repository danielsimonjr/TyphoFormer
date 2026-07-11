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
| `eval`     | Load a checkpoint and evaluate (MAE + spherical-distance ΔR, per horizon). | `./typhoformer eval --weights=m.ckpt` |
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
| `--resume=CKPT` | Resume weights **and optimizer state** from a checkpoint + its `.opt` sidecar. | — |
| `--no_text` | **Ablation**: zero the language-embedding branch (numbers-only model). | off |
| `--split_seed=` | Seed for the storm-level train/val/test partition (vary for a variance estimate). | 42 |
| `--seed=` | RNG seed (determinism). | 20260711 |
| `--csv= --emb= --bin= --save=` | Data source / checkpoint path. | repo defaults |

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

Math is hand-written and cache-blocked (`ikj`/`pij` loop orders, `restrict`
pointers), built at `-O3`. Backward passes reuse persistent per-module scratch
buffers, so training does no per-step heap allocation.

| Build | Compact config | Full paper config (5.1 M params) |
|:--|:--:|:--:|
| `make` (portable `-O3`) | ~3 s/epoch | ~190 s/epoch |
| `make NATIVE=1` (`-march=native`) | ~3 s/epoch | **~99 s/epoch** |

Native SIMD roughly halves the full-config epoch time; `--threads=N` adds
data-parallel scaling across cores on top (see **Multicore training** above).
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
| `test_parallel` | multicore gradient == serial gradient (≈1e-7). |
| `test_checkpoint`, `test_npy` | checkpoint round-trip (TFW1/2/3 + optimizer sidecar), `.npy` dtype/fortran validation, storm-safe split. |
| `test_dropout` | dropout is identity at eval; its backward (pinned-mask) gradient-checks. |
| `backends/opencl/test_opencl` | OpenCL kernels match the CPU reference (≤1.2e-7); model runs through OpenCL. |

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

| what | params | held-out test ΔR | persistence |
|:--|:--:|:--:|:--:|
| model, 5 different splits | 198 K | **128.5 ± 41.3 km** (min 95, max 206) | 123.5 km |
| **numbers-only** (`--no_text`, split 42) | 198 K | **156.0 km** | 102.7 km |
| with-text (compact, split 42) | 198 K | 172.9 km | 102.7 km |
| with-text (**full config**, split 42) | 5.1 M | 158.0 km | 102.7 km |

Three honest conclusions: the model is at **rough parity with persistence** (the
±41 km split noise dwarfs the ~5 km gap); the **language branch does not help**
here — numbers-only is ~17 km *better* on the tested split, contradicting the
paper's premise on this data; and **26× more parameters** (the full paper config)
does **not** close the gap, so the bottleneck is the ~98-storm dataset, not model
size.

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
