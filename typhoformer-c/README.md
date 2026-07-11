# TyphoFormer-C

A **clean-room, pure-C reimplementation** of the TyphoFormer typhoon-track
forecasting model — Prompt-aware Gating Fusion (PGF) + a spatio-temporal
Transformer encoder + an autoregressive decoder.

## Scope & design

- **Pure C, standard library only.** No third-party dependencies — all math
  (matmul, softmax, layer norm, attention, Adam) is hand-written. Links only
  against `libm`.
- **Full pipeline.** Data preparation, training (forward + backprop + Adam),
  and inference are all implemented in C.
- **Language branch is precomputed.** The GPT-4o descriptions and MiniLM
  sentence embeddings are produced offline (outside this C code); the C
  program consumes the resulting numerical embedding vectors.

## Clean-room / licensing note

This directory is an **independent implementation written from the published
method** — the paper and the algorithm pseudocode / data-flow specification in
the repository root (`TyphoFormer_algorithm.tex`, README figures) — *not* a
line-by-line translation of the upstream Python. Because it is original code
implementing (non-copyrightable) algorithms and math, it can carry its own
license. A `LICENSE` will be added once the copyright holder is confirmed.

## Build

```bash
cd typhoformer-c
make            # build everything
make test       # build and run unit tests (incl. gradient checks)
```

Requires only a C11 compiler (`gcc`/`clang`) and `make`.

## Status

Under active construction. Progress:

- [x] Tensor core (matmul variants, activations) + gradient-check harness
- [ ] NN layers (Linear, multi-head attention, LayerNorm, FFN)
- [ ] Model (PGF fusion, ST-encoder, autoregressive decoder)
- [ ] `.npy` data loader + sliding-window dataset
- [ ] Loss (MSE + gate penalty), Adam optimizer, training loop
- [ ] Evaluation (MAE, spherical-distance error)
