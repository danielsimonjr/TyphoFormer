# Changelog

Notable changes to this repository, per [Keep a Changelog](https://keepachangelog.com/): new work goes under **[Unreleased]** as it lands, using the categories **Added / Changed / Fixed / Docs**. The project has no versioned releases yet, so the history below is grouped by date, newest first.

## [Unreleased]

### Added
- Nothing yet.

## 2026-07-12

### Added
- `CLAUDE.md` and `AGENTS.md` (guidance for AI coding agents), `CHANGELOG.md`, `TODO.md`.

## 2026-07-11 — the pure-C reimplementation

The bulk of the current codebase: `typhoformer-c/`, a clean-room, dependency-free C11 port of the full pipeline, plus the honest re-evaluation of the model it enabled.

### Added
- **C port core**: tensor primitives with cache-blocked matmuls (`tensor.c`); NN layers with hand-written forward *and* backward (Linear, LayerNorm, FFN, multi-head attention, transformer block — `nn.c`); the full model (PGF fusion, spatio-temporal encoder, autoregressive decoder — `model.c`); CSV/`.npy` data loader with sliding windows (`data.c`); Adam optimizer and end-to-end training loop. Every layer validated by finite-difference gradient checks.
- **CLI**: six subcommands — `train`, `eval`, `prepare`, `predict`, `baseline`, `bench` — with full hyperparameter flags, multi-step autoregressive decoding, and per-horizon metrics (MAE + spherical ΔR).
- **Training hygiene**: dropout, AdamW (decoupled weight decay), gradient clipping, LR warmup/decay, early stopping, best-checkpoint saving, resume with optimizer sidecar.
- **Extension seams**: pluggable `Module` vtable + `Sequential` (`module.h`); a documented ~13-kernel compute-backend seam (`backend.h`) with a runnable **OpenCL** backend (verified end-to-end via POCL) and a **CUDA** backend (nvcc-clean); data-parallel multicore training via pthreads (`--threads=N`), gradient-equivalence-tested against serial.
- **Experiment flags**: `--motion` (position + velocity inputs) and `--delta` (displacement head); `--no_text` language-branch ablation and `--split_seed`; encoder options (`--no_spatial`, `--posenc`, `--pool=last`, `--prenorm`, `--timebias`, `--co_spatial`); decoder variants `--cv` (constant-velocity anchor), `--gru` (recurrent rollout state), `--xattn` (cross-attention over the encoder sequence); `--km_loss` (tested, did not help — off by default).
- **Tests + CI**: gradient checks (tensor/nn/model), golden-loss regression, module, parallel-equivalence, checkpoint round-trip, `.npy` validation, dropout, GRU, cross-attention; gcc+clang CI matrix with sanitizers, valgrind, an OpenCL job (POCL), a CUDA compile-check, and a subcommand smoke test.
- **Offline tools** (`tools/`): GPT-4o description generation, MiniLM embedding generation, `.npy`-dict → `.tfb` converter.

### Changed
- Original PyTorch implementation moved to `legacy/` (kept for reference; superseded by the C port).

### Fixed
- **Data leakage** in evaluation: storm-level train/val/test split, train-only normalization stats, coordinate normalization, and a real held-out test set. The previously reported ΔR ≈ 79 km "beats persistence" result was retracted as a leakage artifact; results were re-measured honestly.
- `--co_spatial` training divergence, fixed via a zero-initialized attention output projection (starts as a no-op).

### Docs
- Full documentation suite under `typhoformer-c/docs/`: ARCHITECTURE (layer-by-layer math), THEORY_MAP (equation → `file:function`), GLOSSARY, LABS, API, INTEGRATION, EXTENDING, and **FINDINGS.md** — the honest experimental record (motion blindness fixed: held-out ΔR 128 → 48 km; language branch does not help on this data; `--cv` reaches the CLIPER baseline; `--gru`/`--xattn` help at long horizons but are split-dependent).
- Exhaustive explanatory comments across the entire C source.

## 2026-07-10 — paper-release packaging

### Added
- `TyphoFormer_algorithm.tex`: paper-style pseudocode (LaTeX/algorithmicx) inferred from the implementation, rendered to SVG/PNG in `assets/`.
- Mermaid end-to-end data-flow diagram in the README.

### Changed
- README overhauled into a polished paper-release page (method, results tables, quick start, repository structure).
- Licensing settled: fork-added licenses removed; the repo defers to upstream LabRAI/TyphoFormer (the later C port under `typhoformer-c/` carries its own MIT license).

## 2025-11-12 → 2025-12-14 — upstream

- Original PyTorch implementation, bundled HURDAT2 sample data (2020–2024), GPT-4o descriptions, and MiniLM embeddings, as published by the paper authors (LabRAI/TyphoFormer; this repo is a fork).
