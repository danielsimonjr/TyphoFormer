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
make            # build the ./typhoformer binary
make test       # build and run all unit tests (gradient checks + data loader)

./typhoformer 30                     # 30 epochs, compact demo config, CSV data
./typhoformer 5 --full               # full paper config (d_model=256, 3 layers)
./typhoformer 30 --csv=PATH --emb=DIR
```

Requires only a C11 compiler (`gcc`/`clang`) and `make`.

## Command-line tools

The `./typhoformer` binary provides subcommands (the default is `train`, so
`./typhoformer 30` still trains):

| Command | Purpose | Example |
|:--|:--|:--|
| `train`   | Train; writes a checkpoint (with a config header). | `./typhoformer train 30 --full --save=m.ckpt` |
| `eval`    | Load a checkpoint and evaluate (MAE + spherical-distance ΔR). | `./typhoformer eval --weights=m.ckpt` |
| `prepare` | Build sliding-window samples from CSV + embeddings and write a `.tfb`. | `./typhoformer prepare --out=data.tfb` |

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
