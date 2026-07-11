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

## 4. The diagnosis: the model was blind to motion

The honest bar is not persistence (~123 km) — it is **constant-velocity
extrapolation** (add the last observed velocity), which this repo's own
`baseline` subcommand puts at **39 km @ 6h**. The default model (§1–3, ~128 km)
is *3–4× worse than that two-line physics baseline*.

Why? Its inputs (`NUMCOL` in `src/data.c`) are **intensity only** — max wind, min
pressure, 12 wind-radii columns — plus the text embedding. The storm's
**position and velocity are never fed to the model**; they are used only to seed
the decoder and form the targets. So the model is asked to predict *where the
storm goes* while never being shown *how it is moving*. That is why it loses to
constant-velocity, and why the language ablation was flat: **neither branch
carries the motion signal**, so the whole input representation is the bottleneck,
not the text.

## 5. The fix: feed motion, predict displacement

Two changes, each a CLI flag, tested across the same 5 storm splits (compact
config, 30 epochs, early stopping):

| model | held-out test ΔR (km) | vs persistence | vs const-velocity |
|:--|:--:|:--:|:--:|
| default (intensity + text) | 128.5 ± 41.3 | ~parity | 3–4× worse |
| **+ `--motion`** (position + velocity inputs) | 79.0 ± 26.8 | beats it | ~2× worse |
| **+ `--motion --delta`** (predict displacement) | **48.1 ± 2.7** | **2.5× better** | competitive (39) |

- **`--motion`** (add `lat, lon, Δlat, Δlon` to the inputs) is the dominant fix:
  it hands the model the signal constant-velocity already uses. Mean ΔR 128 → 79.
- **`--delta`** (the **displacement head**: the decoder predicts the change from
  the seed, `fc2` zero-initialised so it *starts* at persistence) halves the error again
  and — just as important — **collapses the variance** (std 27 → 3 km): every
  split now lands in 44–53 km. Starting from a sensible prior and learning the
  correction is far more stable than regressing absolute coordinates.

The fixed model (**48 km**) is **2.5× better than persistence** and finally
**competitive with the constant-velocity baseline (~39 km)** — from a starting
point of being several times worse than a two-line heuristic.

## 6. Even with a working model, the language branch still does not help

Repeating the `--no_text` ablation on the *fixed* model, across all 5 splits:

| fixed model | held-out test ΔR (km) |
|:--|:--:|
| `--motion --delta` **with** text | 48.1 ± 2.7 |
| `--motion --delta` **without** text (`--no_text`) | **46.5 ± 3.9** |

Numbers-only is again **marginally better** (~1.6 km). This is now a robust
result — five splits, and a model that actually forecasts well — and it still
says the GPT-4o/MiniLM language branch, the paper's central premise, provides no
benefit on this data. If anything it adds a little noise.

## 7. The encoder is not the bottleneck

With the model fixed (`--motion --delta`), we swept the encoder architecture one
option at a time, each across three storm-split seeds (held-out test ΔR, km):

| encoder variant | seed 1 | seed 3 | seed 5 | mean |
|:--|:--:|:--:|:--:|:--:|
| base (`--motion --delta`) | 47.2 | 51.9 | 50.6 | 49.9 |
| `--no_spatial` (drop dead N=1 blocks) | 54.4 | 49.8 | 48.4 | 50.9 |
| `--posenc` (learned positional encoding) | 47.9 | 51.2 | 49.2 | 49.5 |
| `--pool=last` (recency pooling) | 51.3 | 56.3 | 47.7 | 51.8 |
| `--prenorm` (pre-norm blocks) | 52.6 | 52.9 | 48.5 | 51.3 |
| all four together | 47.1 | 47.0 | 49.2 | **47.7** |

Every single-option mean sits inside the ±3 km split-to-split noise of the base
— none is a real improvement. Stacking all four is marginally the best (47.7 vs
49.9), but still within noise, and it costs a learned positional table. The
honest read: **the encoder architecture is not what limits this model.** Once
motion is in the inputs and the head predicts displacement, the remaining error
is dominated by the data (98 storms) and the intrinsic hardness of the horizon,
not by how the 12-step context is mixed. `--no_spatial` is worth keeping anyway
— it drops parameters whose Q/K never train, at no accuracy cost.

## 8. Giving attention a sense of time and space — also neutral

Two targeted attention upgrades, each on the fixed `--motion --delta` model,
three storm-split seeds (held-out test ΔR, km):

**`--timebias`** — an ALiBi-style relative-distance bias in the *temporal*
attention (`score[i,j] −= slopeₕ·|i−j|`), giving self-attention an intrinsic
sense of recency without any new parameters.

**`--co_spatial`** — *real* multi-node spatial attention. The paper's "spatial"
blocks are degenerate (N=1: every position attends only to itself). Here the
encoded context instead attends over the relative states — (Δlat, Δlon,
Δmax_wind) — of every storm active at the same timestep (38% of timesteps have
≥2 co-active storms). This is genuine cross-storm attention, reusing MHA over a
`[context; neighbour…]` token sequence.

| variant | seed 1 | seed 3 | seed 5 | mean |
|:--|:--:|:--:|:--:|:--:|
| base (`--motion --delta`) | 47.2 | 51.9 | 50.6 | 49.9 |
| `--timebias` | 46.2 | 52.5 | 51.6 | 50.1 |
| `--co_spatial` (random-init residual) | **93.6** | 54.2 | 48.9 | 65.6 ⚠ |
| `--co_spatial` (zero-init residual) | 50.0 | 49.7 | 53.5 | 51.1 |
| `--timebias --co_spatial` (zero-init) | 51.2 | 52.5 | 48.9 | 50.9 |

Two things worth recording honestly:

1. **A real bug we measured, then fixed.** The first `--co_spatial` build put a
   *randomly-initialised* attention residual into the decoder's context. On seed
   1 that kicked training off course and ΔR blew up to 93.6 km — nearly 2× the
   base, and close to persistence. Zero-initialising the attention's output
   projection (so the module starts as an exact no-op and learns the correction
   from zero — the delta-head trick) removed the divergence: the seed-1 result
   dropped from 93.6 → 50.0 and the spread collapsed back into the noise band.
2. **Once stable, it does not help.** Zero-init `--co_spatial` means 51.1, and
   `--timebias` 50.1 — both inside the ±3 km split noise of the 49.9 base, i.e.
   neither is a real improvement. Giving the temporal attention a sense of time,
   and making the spatial attention genuinely multi-node, are both *architecturally*
   the right thing to do — but on this data they change nothing measurable.

This is the same lesson as §7 from a different angle: the ceiling here is the
data and the horizon, not how cleverly 12 steps of context (or a handful of
co-active storms) are attended over. The value of building `--co_spatial`
properly was the measurement — and catching a real training divergence that a
single-split run would have hidden.

## 9. The decoder: anchoring at constant velocity finally reaches CLIPER

Everything above says the *inputs* and the *parametrization* are the levers, not
the network internals. The decoder is where that pays off again. `--delta`
anchored the rollout at **persistence** (`ŷ_t = y_{t-1} + Δ`) and learned the
correction down to ~48 km. But the bar we actually chase is **constant-velocity
(CLIPER, 39.4 km at 6h)** — and persistence is a needlessly bad place to start
from when the seed velocity is right there in the data.

`--cv` is a second-order delta: it threads a velocity state through the rollout
and anchors at constant-velocity extrapolation,

```
ŷ_t = y_{t-1} + v + fc2(relu(fc1([h ; y_{t-1} ; v]))),   v ← v + correction
```

with `fc2` zero-initialised so the *untrained* model emits exact constant-velocity
— it starts at CLIPER and learns only the curvature on top. (Verified directly:
epoch-0 held-out ΔR is **39.4 km** with `--cv` vs **137 km** with `--delta`.)

Held-out test ΔR (km), fixed `--motion` model, three storm-split seeds:

| decoder | seed 1 | seed 3 | seed 5 | mean |
|:--|:--:|:--:|:--:|:--:|
| `--delta` (anchor at persistence) | 47.2 | 51.9 | 50.6 | 49.9 |
| `--cv` (anchor at constant-velocity) | 42.2 | 41.3 | **37.9** | **40.5** |
| *constant-velocity / CLIPER baseline* | — | — | — | *39.4* |

This is the first **architectural** change in the whole exercise that moves the
needle: **49.9 → 40.5 km (~19%), consistently on every split** (each `--cv` split
beats its `--delta` counterpart), landing essentially *on* CLIPER — and on seed 5
a hair *under* it (37.9 vs 39.4). The learned model has finally caught the
constant-velocity forecaster it was losing to.

It has not decisively *passed* CLIPER, and at a single horizon (6h) it arguably
shouldn't be expected to — constant-velocity is very strong for one step. The
place a learned curvature term should actually win is **longer horizons**, where
storms recurve and constant-velocity drifts; that (with more than 98 storms) is
the honest next test. But the lesson is clean and consistent with §5: give the
decoder the right physical anchor — motion in, velocity threaded, correction
learned from zero — and it matches the classical baseline instead of trailing it.

## 10. What this means

- The engineering was always sound (gradient checks, golden, cross-backend
  agreement). The *modeling* had a concrete, fixable flaw — the inputs omitted
  motion — and fixing it moved held-out ΔR from ~128 km to **48 km**, turning a
  sub-persistence model into one that beats persistence 2.5× and rivals
  constant-velocity.
- The **language branch does not earn its keep** here, before or after the fix.
  Whether it would on a larger, harder dataset (rapid intensification, recurvature,
  extratropical transition — where text might carry signal the numbers don't) is
  the honest open question.
- **The gap to constant-velocity is now closed** (§9). `--cv` anchors the decoder
  at constant-velocity instead of persistence and learns only the curvature:
  49.9 → **40.5 km**, matching the 39.4 km CLIPER baseline (and beating it on one
  split). The right *parametrization* — like motion in the inputs — is a real
  lever; a km-aware loss (`--km_loss`) by contrast was **tested and did not help**.
  The remaining frontier is **longer horizons** (where curvature should let the
  learned model finally *pass* constant-velocity) and — the ceiling on everything
  — **more than 98 storms**.
- **The network internals are not the lever** (§7, §8). An encoder sweep (no_spatial /
  posenc / pool=last / prenorm) and two attention upgrades — a temporal
  distance bias (`--timebias`) and *real* multi-node spatial attention over
  co-active storms (`--co_spatial`) — are all neutral within split noise on the
  fixed model. Each is the architecturally right thing to do; none moves ΔR on
  this data. Building `--co_spatial` properly still paid off: it surfaced (and
  we fixed, via a zero-init residual) a real training divergence that a
  single-split run would have hidden. The levers that *did* move ΔR were all about
  the physics the model sees — motion features, a displacement head, a
  constant-velocity anchor — not the depth or attention of the network.

## Reproduce

```sh
# the fix, across splits
for s in 1 2 3 4 5; do ./typhoformer train 30 --patience=8 --motion --delta --split_seed=$s; done
# the constant-velocity decoder — reaches CLIPER (§9)
for s in 1 3 5; do ./typhoformer train 30 --patience=8 --motion --cv --threads=1 --split_seed=$s; done
# does text help the fixed model?
./typhoformer train 30 --patience=8 --motion --delta --split_seed=42            # with text
./typhoformer train 30 --patience=8 --motion --delta --split_seed=42 --no_text  # numbers only
# the honest baseline the model must beat
./typhoformer baseline --pred_len=4
# architecture explorations (all neutral, §7–§8) — vary --split_seed to see the noise
./typhoformer train 30 --patience=8 --motion --delta --no_spatial          # encoder sweep
./typhoformer train 30 --patience=8 --motion --delta --timebias            # temporal distance bias
./typhoformer train 30 --patience=8 --motion --delta --co_spatial --threads=1  # real spatial attn
```

(See [LABS.md](LABS.md) Track D for these as guided exercises.)
