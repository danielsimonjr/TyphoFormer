# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

TyphoFormer: a language-augmented Transformer for typhoon/hurricane track forecasting (ACM SIGSPATIAL 2025). Two implementations of the same model live side by side:

- **`typhoformer-c/`** — the active codebase. A clean-room, dependency-free pure-C (C11) reimplementation of the full pipeline: training, evaluation, inference, baselines. Links only `libm` + `pthread`. MIT-licensed independently of the rest of the repo.
- **`legacy/`** — the original PyTorch implementation, kept for reference/reproducibility. Superseded by the C port; changes normally go to `typhoformer-c/`.

Shared data at the repo root: `HURDAT_2new_3000.csv` (raw records, 2020–2024), `embedding_chunks/*.npy` (precomputed MiniLM embeddings of GPT-4o descriptions), `data/{train,val,test}/` (pickled `.npy` window samples). The C data loader reads the CSV and embedding chunks from the repo root via `../`, so build and run from inside `typhoformer-c/`.

## Task tracking & changelog

- **`TODO.md`** is the task list for new/pending work. When starting a task that matches an item, work from it; check the item off (`- [x]`) when it's done and delete stale entries. Add newly discovered follow-up work as new items.
- **`CHANGELOG.md`** follows Keep a Changelog: record every notable codebase change under `[Unreleased]` (categories: Added / Changed / Fixed / Docs) as part of the same commit or PR that makes the change.
- Experimental results — including negative ones — go in `typhoformer-c/docs/FINDINGS.md`, not the changelog.

## Commands (typhoformer-c/)

All from `typhoformer-c/`:

```bash
make                    # build ./typhoformer (portable -O3)
make NATIVE=1           # -march=native SIMD build (~2x faster full config)
make lib                # libtyphoformer.a for embedding (no main)
make test               # all unit tests (gradient checks, golden, module, parallel, checkpoint, npy, dropout, gru, xattn)
make test-san           # clean rebuild with ASan+UBSan, run tests
make test-valgrind      # tests under valgrind
make OPENCL=1 test-opencl  # verify the OpenCL backend (needs an ICD, e.g. pocl-opencl-icd)
make -C backends/cuda   # compile the CUDA backend with nvcc
```

**Run a single test:** `make tests/test_nn && ./tests/test_nn` (each `tests/test_*.c` builds to its own binary against the library objects).

**Train / evaluate:**

```bash
./typhoformer 30                          # train 30 epochs, compact demo config
./typhoformer 30 --full --threads=8       # full paper config (d_model=256, 3 layers), data-parallel
./typhoformer train 30 --motion --cv --save=m.ckpt      # the accurate configuration (works serial or --threads=N)
./typhoformer eval --weights=m.ckpt       # MAE + spherical ΔR, per horizon
./typhoformer predict --weights=m.ckpt --n=50 --out=pred.csv
./typhoformer baseline --pred_len=4       # persistence + constant-velocity baselines
./typhoformer bench --full --iters=50
```

The full flag reference is in `typhoformer-c/README.md`. Notable: `--no_text` (language-branch ablation), `--cv`/`--gru`/`--xattn` (decoder variants), `--split_seed` (vary the storm partition).

Legacy PyTorch (run from the repo root so `data/` and the `model` package resolve): `python legacy/train_typhoformer.py`, `python legacy/eval_typhoformer.py`, `python legacy/prepare_typhoformer_data.py`. Hyperparameters are constants at the top of `train_typhoformer.py`.

CI (`.github/workflows/c-ci.yml`) runs on changes under `typhoformer-c/`: gcc+clang build/test matrix, sanitizer tests, valgrind, an OpenCL job through POCL, a CUDA compile-check, and a subcommand smoke test. Code must stay warning-clean under `-Wall -Wextra -Wpedantic`.

## Architecture

The model (both implementations): **PGF fusion → spatio-temporal Transformer encoder → autoregressive decoder.** Per time step, numerical features and a mean-pooled 384-d MiniLM prompt embedding are blended by a learned sigmoid gate (Prompt-aware Gating Fusion); the fused sequence goes through alternating temporal/spatial self-attention blocks; a decoder rolls out future (lat, lon) autoregressively, seeded with the last observed coordinate. Loss = MSE + a gate-collapse penalty `λ_g·max(0, τ−g)²`. The LaTeX pseudocode in `TyphoFormer_algorithm.tex` and the README diagrams are the spec the C port was written from.

The GPT-4o description + MiniLM embedding stages are offline Python by necessity (`typhoformer-c/tools/gen_descriptions.py`, `gen_embeddings.py`); the C core only consumes their `.npy` output. `tools/npy_dict_to_bin.py` converts the pickled pre-split `data/*/*.npy` dicts to the flat `.tfb` binary the C loader can read.

### C code layout

One `.c` per header, layered bottom-up:

- `tensor.c/h` — Mat primitives, cache-aware matmuls (`ikj`/`pij` orders, unrolled-accumulator dot products), activations. `backend.h` defines the ~13-kernel compute seam; `backends/opencl/tensor_opencl.c` (runnable, POCL-verified) and `backends/cuda/tensor_cuda.cu` are drop-in replacements for `src/tensor.c` (the Makefile swaps the object with `OPENCL=1`).
- `nn.c/h` — Linear, multi-head attention, LayerNorm, FFN, transformer block, each with hand-written forward *and* backward.
- `model.c/h` — PGF, ST-encoder, decoder variants, full model.
- `module.c/h` — pluggable Module vtable + Sequential for new layers.
- `data.c/h` — CSV/`.npy`/`.tfb` loaders, sliding windows, **storm-level** train/val/test split with train-only normalization (leakage-safe — preserve this in any data changes).
- `optim.c/h` (AdamW), `checkpoint.c/h` (TFW formats + optimizer sidecar), `parallel.c/h` (synchronous data-parallel SGD across pthreads), `train.c` (main + all six subcommands).

### The correctness discipline

Every forward pass has a matching hand-written backward, proven by a **finite-difference gradient check** — this is the repo's core invariant. When touching any forward/backward pair, follow `docs/EXTENDING.md`:

- Register parameters with `plist_add(pl, value_ptr, grad_ptr, count, name)` — that alone makes a layer trained and checkpointed.
- Gradients **accumulate** (`+=`), never assign; the trainer zeros per batch.
- Backward scratch buffers are persistent struct fields grown with `ensure()` — no malloc/free inside `*_backward`.
- After any math change, run `make test`; `test_golden` pins the training loss bit-exactly, so an intentional numerics change requires updating the golden value.

### Findings context (matters for experiment work)

`typhoformer-c/docs/FINDINGS.md` is the honest experimental record. Key conclusions: the default input featurization is blind to motion (intensity + text only), so train with `--motion --cv` for real accuracy (held-out ΔR 131→40.8 km @ 6h; beats constant-velocity at 48h; works with `--threads=N`); the language branch does **not** help on this data (`--no_text` is marginally better); the honest baseline is constant-velocity (~39 km @ 6h), not persistence; decoder memory (`--gru`/`--xattn`) did not survive a five-split re-test (§11) — single-split results on 98 storms can flip under float-rounding-level perturbations, so always test across multiple `--split_seed`s. An earlier "beats persistence" result was data leakage — the storm-level split and train-only stats exist to prevent that regression. Report held-out test numbers, not validation numbers.

Other docs: `docs/ARCHITECTURE.md` (full layer-by-layer math), `docs/THEORY_MAP.md` (equation → `file:function`), `docs/API.md` (memory/ownership/concurrency model), `docs/INTEGRATION.md` (embedding as a library, file formats), `docs/LABS.md`, `docs/GLOSSARY.md`.
