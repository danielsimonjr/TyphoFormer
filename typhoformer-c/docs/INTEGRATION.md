# Integration Guide — Embedding the Model in Applications

How to use TyphoFormer-C as a **library** inside your own C program: build it,
load a trained model, run inference, and read/write its on-disk formats. Read
[API.md](API.md) (memory model + concurrency) alongside this.

---

## 1. Build as a static library

```bash
cd typhoformer-c
make lib            # produces libtyphoformer.a (everything except the CLI main)
```

Compile your app against the headers and link the archive plus `libm`:

```bash
cc -std=c11 -O3 -Ityphoformer-c/include myapp.c \
   -Ltyphoformer-c -ltyphoformer -lm -o myapp
# faster, non-portable: add  -march=native
```

Everything is C11 + `libm`; there are no other dependencies.

---

## 2. Minimal inference program

Load a checkpoint, rebuild the exact model, and predict the next `(lat, lon)`
from one window of features:

```c
#include "checkpoint.h"
#include "model.h"
#include "nn.h"
#include <stdio.h>

int main(void) {
    // 1. Rebuild the model from the checkpoint's config header, then load weights.
    Config c = checkpoint_load_config("model.ckpt");
    ParamList pl; plist_init(&pl);
    // If the checkpoint used the displacement head (--delta), call
    // model_set_delta(1) here, BEFORE model_new. If trained with --motion,
    // c.d_num is 18 (not 14) and you must feed lat,lon,Δlat,Δlon in the last 4
    // columns — always size xnum from c.d_num.
    Model m = model_new(&c, &pl);
    checkpoint_load_params("model.ckpt", &pl);

    // 2. Fill one input window (see §3 for where these come from).
    Mat xnum  = mat_new(c.in_len, c.d_num);    // [T, d_num] standardized numerics (14, or 18 w/ motion)
    Mat xtext = mat_new(c.in_len, c.d_text);   // [T,384] MiniLM embeddings
    Mat yprev = mat_new(1, 2);                  // last observed (lat, lon)
    // ... populate xnum, xtext, yprev.data[0]=lat, yprev.data[1]=lon ...

    // 3. Forward pass. Prediction lands in m.pred [pred_len,2].
    model_forward(&m, xnum, xtext, yprev);
    printf("predicted next (lat,lon) = %.3f, %.3f\n",
           m.pred.data[0], m.pred.data[1]);

    // 4. Teardown.
    mat_free(&xnum); mat_free(&xtext); mat_free(&yprev);
    model_free(&m); plist_free(&pl);
    return 0;
}
```

`model_forward` also fills `m.pgf.gate` if you want to inspect how much the model
trusted each modality per step.

---

## 3. Producing the input features (important)

The model consumes two things per time step:

- **`xnum` — 14 numerical features**, in the column order of
  `data.c` (`max_wind`, `min_pressure`, then the twelve 34/50/64-kt wind radii),
  **z-score standardized**. Training standardizes with per-column mean/std
  computed over the dataset (`Dataset.mean/std`). For inference you must apply
  the *same* transform: `x' = (x − mean) / std`.

  > ⚠️ **Deployment note.** The checkpoint stores model weights only, not the
  > normalization stats. Persist `Dataset.mean[0..13]` / `Dataset.std[0..13]`
  > next to your checkpoint (e.g. a 28-float side file) and apply them at
  > inference. Skipping this is the most common cause of garbage predictions.

- **`xtext` — a 384-d MiniLM embedding** of a natural-language description of the
  step. These require an external model, so they are produced offline by
  `tools/gen_descriptions.py` (GPT-4o) + `tools/gen_embeddings.py` (MiniLM). At
  serving time, run the same sentence encoder on your description and pass the
  vector in. If you have no description, a zero vector makes the gate lean on the
  numerical branch (the PGF penalty was designed to keep that branch alive).

`yprev` is the last *observed* `(lat, lon)`; the decoder predicts the delta from
it implicitly.

---

## 4. On-disk formats (byte-exact, little-endian)

### 4.1 Checkpoint `.ckpt` — `checkpoint.h`

```
offset  bytes  content
0       4      magic "TFW3"  (also reads legacy "TFW1"/"TFW2")
4       36     9 × int32 : d_num d_text d_model out_dim in_len pred_len d_ff n_heads n_layers
40      4      int32 n_stats            (0 or d_num)
44      ...    float32 mean[n_stats], std[n_stats]        (feature normalization)
..      4      int32 has_coord          (0 or 1)
..      16     float32 cmean[2], cstd[2]                  (lat/lon normalization)
..      ...    float32 parameters, concatenated in ParamList registration order
```

Training also writes a `.opt` sidecar (magic "TFO1": Adam moments + step + lr +
epoch) next to the checkpoint; `--resume=CKPT` restores weights and optimizer
state from it. Inference only needs the `.ckpt`.

The parameter order is defined by the order modules are constructed in
`model_new` (PGF → Encoder(input_proj, output_proj, then per layer temporal &
spatial blocks, then TimeMix) → Decoder), and within each layer by its `*_new`
(`Linear` registers `W` then `b`; `LayerNorm` registers `γ` then `β`). You never
need to compute this by hand: load the config, build the model (which
re-registers in the same order), then stream the floats in. A **wrong config
produces a size mismatch** and `die()`s — a useful safety check.

### 4.2 Dataset cache `.tfb` — produced by `prepare` / `tools/npy_dict_to_bin.py`

```
offset  bytes  content
0       4      magic "TFB2"  (also reads legacy "TFB1")
4       20     5 × int32 : n_samples in_len feat_dim pred_len out_dim   (feat_dim = 14+384 = 398)
24      ...    per sample: float32 input[in_len·feat_dim], target[pred_len·out_dim],
               then (TFB2 only) float32 seed[2]  — the true last-observed coordinate
```

`input` interleaves the 14 numerics and 384 embedding dims per step
(`[num(14) | emb(384)]`). TFB2 stores the real decoder seed, so it is not seeded
with a target label; legacy TFB1 files fall back to the first target. For a
leakage-free train/val/test split, prefer the CSV path (the `.tfb` path has no
storm info and splits by sample).

### 4.3 Embeddings `.npy`

`npy_load_2d` reads NumPy v1.0/2.0 headers for **2-D little-endian float32**
(`<f4`) arrays only. This is the subset the MiniLM chunks use; it is not a
general `.npy` reader.

> **Endianness.** All three formats are little-endian and are read with raw
> `fread` into `float`/`int`. On a big-endian host you would need byte-swapping;
> the code currently assumes a little-endian host (x86/ARM as configured).

---

## 5. Serving / concurrency

- A `Model` instance mutates internal caches on every `model_forward`, so use
  **one instance per worker thread**. For a thread pool, create N instances
  (each `model_new` + `checkpoint_load_params` from the same file). Weights are
  duplicated per instance; at `d_model=256` that is ~20 MB each.
- Instances are otherwise independent — no shared mutable state at inference (the
  global RNG is only touched during construction/shuffling).
- Determinism: same inputs → same outputs, bit-for-bit, on the same build.

---

## 6. Performance knobs

| Knob | Effect |
|:--|:--|
| `make NATIVE=1` | `-march=native` SIMD; ~2× on the full config. Non-portable binary. |
| `Config` size (`d_model`, `d_ff`, `n_layers`) | Dominates compute. The compact demo (`d_model=64`) is ~60× cheaper than the full paper config. |
| Batch | Inference is per-sample; batch by looping. Throughput scales ~linearly with cores if you shard across instances. |
| `--threads=N` (training) | Data-parallel SGD across cores via `ParTrainer` ([parallel.h](../include/parallel.h)); near-linear scaling to the core count, gradient-equivalent to serial. |
| Backend | Every layer is built from the ~13 kernels in [backend.h](../include/backend.h). Swap `src/tensor.c` for a SIMD/GPU backend (CUDA reference in [backends/](../backends/)) to move compute off the CPU. |

Each `model_forward` is hand-written, cache-aware (`ikj`/`pij` loop orders,
unrolled-accumulator dot products that auto-vectorize at `-O3`) and does no
per-step heap allocation, so latency is stable and predictable — a good fit for
embedded / edge deployment where a BLAS/GPU stack is unavailable. Training scales
across cores with `--threads=N`; inference scales by running one instance per
worker thread.

---

## 7. Error handling for library users

Unrecoverable I/O and format errors call `die()` (message to `stderr`,
`exit(1)`). If your process must not exit, validate before calling: check the
file exists and begins with the expected magic, and confirm the checkpoint's
`Config` matches what you expect before `model_new`. All compute functions
(`*_forward`/`*_backward`, `model_loss`, `adam_step`) do not perform I/O and do
not `die()`.
