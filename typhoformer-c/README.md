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

## Build & run

```bash
cd typhoformer-c
make            # build the ./typhoformer training binary
make test       # build and run all unit tests (gradient checks + data loader)

./typhoformer 30                     # train 30 epochs on the bundled data
./typhoformer 30 ../HURDAT_2new_3000.csv ../embedding_chunks
```

Requires only a C11 compiler (`gcc`/`clang`) and `make`.

## Status — complete

- [x] Tensor core (matmul variants, activations) + gradient-check harness
- [x] NN layers (Linear, multi-head attention, LayerNorm, FFN) — block gradient check
- [x] Model (PGF fusion, ST-encoder, autoregressive decoder) — full-model gradient check
- [x] `.npy` data loader + sliding-window dataset
- [x] Loss (MSE + gate penalty), Adam optimizer, training loop
- [x] Evaluation (MAE, spherical-distance error) + persistence baseline

Every layer is validated by finite-difference gradient checks (tensor core,
a full transformer block, and the whole model — all at the single-precision
noise floor). Training on the bundled 5-year sample converges and beats the
persistence baseline:

```
epoch  0 (init)  | val MAE 44.53 | val dR 7382.81 km  (persistence 125.70 km)
epoch 10         | val MAE 0.82  | val dR 129.62 km
epoch 30         | val MAE 0.51  | val dR  79.41 km    (~37% better than persistence)
```

## Notes

- `make test` runs from `typhoformer-c/`; the data loader reads the CSV and
  embedding chunks from the repository root (`../`).
- The training binary uses a compact instance of the architecture (smaller
  `d_model`/`d_ff`) for a fast self-contained demo; the full paper
  configuration (`config_default()`) runs through the identical code path.
- The decoder is seeded with the last *observed* coordinate (paper faithful),
  so the reported ΔR is compared against a persistence baseline.
