# Findings — does this model actually forecast?

After fixing the data leakage (storm-level split, train-only statistics,
held-out test set), the codebase can finally answer the questions that matter
more than "is it well-built." These are the honest results on the bundled
5-year HURDAT2 sample (~98 storms). All numbers are **held-out test** ΔR
(great-circle km), scored on storms never used for training or model selection.

## 1. The single-split headline is not trustworthy

The test ΔR swings wildly with *which storms* land in the test set. Compact
config, 25 epochs, early stopping, varying only `--split_seed`:

| split_seed | model ΔR (km) | persistence ΔR (km) | model wins? |
|:--:|:--:|:--:|:--:|
| 1 | 132.6 | 124.8 | no |
| 2 | 111.9 | 122.2 | **yes** |
| 3 | 95.1 | 134.1 | **yes** |
| 4 | 96.3 | 120.4 | **yes** |
| 5 | 206.4 | 115.8 | no |
| **mean ± std** | **128.5 ± 41.3** | **123.5** | **3 / 5** |

The model is, on average, at **rough parity with persistence** (128 vs 123 km) —
but the **±41 km spread dwarfs the ~5 km gap**. Quoting any single split (the old
"79 km", or the 172 km from split 42) is misleading. The honest statement is:
*on this data the model is about as good as persistence, and the split noise is
larger than the effect.*

## 2. The language branch does not earn its keep

The paper's central premise is that GPT-4o/MiniLM text descriptions improve
forecasting. The `--no_text` ablation (identical seed and split, language branch
zeroed) tests it directly. On `--split_seed=42`:

| model | held-out test ΔR (km) |
|:--|:--:|
| with text (full model) | 172.9 |
| **numbers only (`--no_text`)** | **156.0** |

The **numbers-only model is better by ~17 km**. On this data and configuration,
the language branch is not helping — it adds parameters and, if anything, noise.
(One split is not the last word; combined with the huge split variance above, the
honest reading is that the text branch provides no *robust* benefit here — a
proper study would repeat the ablation across many splits and, ideally, more
data.)

## 3. More capacity does not fix it

Scaling from the compact demo (198 K params) to the **full paper config**
(`--full`, 5.1 M params — 26× larger), same split 42:

| model | params | held-out test ΔR (km) | persistence |
|:--|:--:|:--:|:--:|
| compact, numbers-only | 198 K | 156.0 | 102.7 |
| compact, with text | 198 K | 172.9 | 102.7 |
| **full config, with text** | **5.1 M** | **158.0** | **102.7** |

The full model lands between the compact variants and is still **~55 km worse
than persistence** on this split. Adding 26× the parameters did not close the
gap — the bottleneck is the **~98-storm dataset**, not model size. (Consistent
with §1: 158 km sits inside the 95–206 km split-variance band.)

## 4. What this means

- The engineering is sound; the **forecaster is not a clear win over the trivial
  baseline** on this small sample, and the marquee "language helps" claim does
  not reproduce here.
- This is not a failure of the reimplementation — the gradient checks, golden
  test, and cross-backend agreement all say the math is correct. It is an honest
  measurement that the *earlier favourable numbers were leakage*, and that this
  much data (~98 storms) is likely too little for the model to beat persistence
  robustly, let alone to show a language benefit.

## Reproduce

```sh
# split variance
for s in 1 2 3 4 5; do ./typhoformer train 25 --patience=8 --split_seed=$s; done
# text ablation
./typhoformer train 25 --patience=8 --split_seed=42            # with text
./typhoformer train 25 --patience=8 --split_seed=42 --no_text  # numbers only
```

(See [LABS.md](LABS.md) Track D for these as guided exercises.) The full paper
config was tried (§3) and does not change the verdict. A km-aware loss and — most
importantly — **more data** are the remaining levers; on ~98 storms, no amount of
model capacity made this beat persistence.
