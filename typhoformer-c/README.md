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
implementing (non-copyrightable) algorithms and math, it carries its own
license: **MIT** (see [`LICENSE`](LICENSE)). This license applies only to the
contents of this `typhoformer-c/` directory, not to the upstream repository.
The copyright line reads "TyphoFormer-C contributors" — edit it to your name
or organization as appropriate.

## Build & run

```bash
cd typhoformer-c
make            # build the ./typhoformer training binary
make test       # build and run all unit tests (gradient checks + data loader)

./typhoformer 30                     # 30 epochs, compact demo config, CSV data
./typhoformer 5 --full               # full paper config (d_model=256, 3 layers)
./typhoformer 30 --csv=PATH --emb=DIR
```

Requires only a C11 compiler (`gcc`/`clang`) and `make`.

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

Math is hand-written and cache-blocked (`ikj`/`pij` loop orders), built at
`-O3`. The compact demo config runs ~6 s/epoch and the full 5.1 M-parameter
config ~190 s/epoch, single-threaded, on the bundled data.

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
