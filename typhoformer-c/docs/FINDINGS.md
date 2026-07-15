# Findings — does this model actually forecast?

After fixing the data leakage (storm-level split, train-only statistics,
held-out test set), the codebase can finally answer the questions that matter
more than "is it well-built." These are the honest results on the bundled
5-year HURDAT2 sample (~98 storms). All numbers are **held-out test** ΔR
(great-circle km), scored on storms never used for training or model selection.

> **Re-measured.** Every §1–§9 number below was re-run under one uniform
> protocol — compact config, 30 epochs, early stopping (`--patience=8`),
> `--threads=1` — after the matmul vectorization changed float summation order
> (see §11's sensitivity control). **Every qualitative conclusion survived**;
> individual numbers moved by a few km, exactly the float-level sensitivity §11
> quantifies. §10 is kept unchanged as the historical pre-vectorization record.
>
> **Encoder default change (post-§12):** the degenerate N=1 spatial blocks are
> now skipped by default (motivated by §7). §1–§12 were measured with those
> blocks present — pass `--spatial` to reproduce them exactly. §13's fresh
> baselines cover the new default (which is ~1 km *better* at 6h).

## 1. The single-split headline is not trustworthy

The test ΔR swings wildly with *which storms* land in the test set. Compact
config, varying only `--split_seed`:

| split_seed | model ΔR (km) | persistence ΔR (km) | model wins? |
|:--:|:--:|:--:|:--:|
| 1 | 130.0 | 124.8 | no |
| 2 | 111.2 | 122.2 | **yes** |
| 3 | 110.5 | 134.1 | **yes** |
| 4 | 104.0 | 120.4 | **yes** |
| 5 | 199.9 | 115.8 | no |
| **mean ± std** | **131.1 ± 39.6** | **123.5** | **3 / 5** |

The model is, on average, at **rough parity with persistence** (131 vs 123 km) —
but the **±40 km spread dwarfs the ~8 km gap**. Quoting any single split (the old
"79 km", or the 186 km from split 42) is misleading. The honest statement is:
*on this data the model is about as good as persistence, and the split noise is
larger than the effect.*

## 2. The language branch does not earn its keep

The paper's central premise is that GPT-4o/MiniLM text descriptions improve
forecasting. The `--no_text` ablation (identical seed and split, language branch
zeroed) tests it directly. On `--split_seed=42`:

| model | held-out test ΔR (km) |
|:--|:--:|
| with text (full model) | 185.7 |
| **numbers only (`--no_text`)** | **171.3** |

The **numbers-only model is better by ~14 km**. On this data and configuration,
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
| compact, numbers-only | 198 K | 171.3 | 102.7 |
| compact, with text | 198 K | 185.7 | 102.7 |
| **full config, with text** | **5.1 M** | **163.1** | **102.7** |

The full model edges out the compact variants on this split but is still
**~60 km worse than persistence**. Adding 26× the parameters did not close the
gap — the bottleneck is the **~98-storm dataset**, not model size. (Consistent
with §1: 163 km sits inside the 104–200 km split-variance band.)

## 4. The diagnosis: the model was blind to motion

The honest bar is not persistence (~123 km) — it is **constant-velocity
extrapolation** (add the last observed velocity), which this repo's own
`baseline` subcommand puts at **39 km @ 6h**. The default model (§1–3, ~131 km)
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
| default (intensity + text) | 131.1 ± 39.6 | ~parity | 3–4× worse |
| **+ `--motion`** (position + velocity inputs) | 76.2 ± 30.6 | beats it | ~2× worse |
| **+ `--motion --delta`** (predict displacement) | **48.4 ± 2.6** | **2.5× better** | competitive (39) |

- **`--motion`** (add `lat, lon, Δlat, Δlon` to the inputs) is the dominant fix:
  it hands the model the signal constant-velocity already uses. Mean ΔR 131 → 76.
- **`--delta`** (the **displacement head**: the decoder predicts the change from
  the seed, `fc2` zero-initialised so it *starts* at persistence) cuts the error
  by a third again and — just as important — **collapses the variance**
  (std 31 → 3 km): every split now lands in 45–51 km. Starting from a sensible
  prior and learning the correction is far more stable than regressing absolute
  coordinates.

The fixed model (**48 km**) is **2.5× better than persistence** and finally
**competitive with the constant-velocity baseline (~39 km)** — from a starting
point of being several times worse than a two-line heuristic.

## 6. Even with a working model, the language branch still does not help

Repeating the `--no_text` ablation on the *fixed* model, across all 5 splits:

| fixed model | held-out test ΔR (km) |
|:--|:--:|
| `--motion --delta` **with** text | 48.4 ± 2.6 |
| `--motion --delta` **without** text (`--no_text`) | **46.7 ± 4.8** |

Numbers-only is again **marginally better** (~1.7 km). This is now a robust
result — five splits, and a model that actually forecasts well — and it still
says the GPT-4o/MiniLM language branch, the paper's central premise, provides no
benefit on this data. If anything it adds a little noise.

## 7. The encoder is not the bottleneck

With the model fixed (`--motion --delta`), we swept the encoder architecture one
option at a time, each across three storm-split seeds (held-out test ΔR, km):

| encoder variant | seed 1 | seed 3 | seed 5 | mean |
|:--|:--:|:--:|:--:|:--:|
| base (`--motion --delta`) | 48.0 | 51.4 | 50.4 | 49.9 |
| `--no_spatial` (drop dead N=1 blocks) | 48.0 | 53.1 | 45.5 | 48.9 |
| `--posenc` (learned positional encoding) | 49.1 | 50.8 | 51.2 | 50.4 |
| `--pool=last` (recency pooling) | 48.3 | 52.5 | 49.0 | 49.9 |
| `--prenorm` (pre-norm blocks) | 54.4 | 50.9 | 47.2 | 50.8 |
| all four together | 47.1 | 47.4 | 45.5 | **46.7** |

Every single-option mean sits inside the ±3 km split-to-split noise of the base
— none is a real improvement. Stacking all four is marginally the best (46.7 vs
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
| base (`--motion --delta`) | 48.0 | 51.4 | 50.4 | 49.9 |
| `--timebias` | 48.7 | 52.4 | 55.3 | 52.1 |
| `--co_spatial` (random-init residual, historical) | **93.6** | 54.2 | 48.9 | 65.6 ⚠ |
| `--co_spatial` (zero-init residual) | 60.5 | 50.8 | 50.2 | 53.8 |
| `--timebias --co_spatial` (zero-init) | 53.3 | 48.1 | 50.5 | 50.6 |

(The random-init row is the historical measurement that motivated the fix — that
build no longer exists to re-measure.)

Two things worth recording honestly:

1. **A real bug we measured, then fixed.** The first `--co_spatial` build put a
   *randomly-initialised* attention residual into the decoder's context. On seed
   1 that kicked training off course and ΔR blew up to 93.6 km — nearly 2× the
   base, and close to persistence. Zero-initialising the attention's output
   projection (so the module starts as an exact no-op and learns the correction
   from zero — the delta-head trick) removed the divergence and collapsed the
   spread back toward the noise band (seed 1: 93.6 → 50.0 at the time of the
   fix; 60.5 under the re-measured kernel — elevated, but no blow-up).
2. **Once stable, it does not help.** Zero-init `--co_spatial` means 53.8, and
   `--timebias` 52.1 — at or a few km *above* the 49.9 base, i.e. neither is an
   improvement (if anything they cost a little here). Giving the temporal
   attention a sense of time, and making the spatial attention genuinely
   multi-node, are both *architecturally* the right thing to do — but on this
   data they measurably do not help.

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
| `--delta` (anchor at persistence) | 48.0 | 51.4 | 50.4 | 49.9 |
| `--cv` (anchor at constant-velocity) | 43.4 | 41.1 | **37.8** | **40.8** |
| *constant-velocity / CLIPER baseline* | — | — | — | *39.4* |

This is the first **architectural** change in the whole exercise that moves the
needle: **49.9 → 40.8 km (~18%), consistently on every split** (each `--cv` split
beats its `--delta` counterpart), landing essentially *on* CLIPER — and on seed 5
a hair *under* it (37.8 vs 39.4). The learned model has finally caught the
constant-velocity forecaster it was losing to.

It has not decisively *passed* CLIPER, and at a single horizon (6h) it arguably
shouldn't be expected to — constant-velocity is very strong for one step. The
place a learned curvature term should actually win is **longer horizons**, where
storms recurve and constant-velocity drifts; that (with more than 98 storms) is
the honest next test. But the lesson is clean and consistent with §5: give the
decoder the right physical anchor — motion in, velocity threaded, correction
learned from zero — and it matches the classical baseline instead of trailing it.

## 10. Recurrent state and cross-attention — the first signal, at long horizons

The cv decoder (§9) still threads only a 4-number state (position + velocity) and
re-reads a single pooled context. Two upgrades give it more:

- **`--gru`** carries a full d_model hidden state across the rollout (initialised
  from the encoder context) to produce the curvature correction — real memory.
- **`--xattn`** replaces the static pooled context with per-step **cross-attention
  over the encoder's full sequence**, so each forecast step reads the history it
  needs rather than one averaged vector.

Both anchor at constant-velocity (zero-init output), both are gradient-checked as
standalone cells *and* through the full model. Held-out test ΔR (km), `--motion`,
three storm splits:

| decoder | 6h — s1 | s3 | s5 | mean | | 6–24h mean — s1 | s3 | s5 | mean |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `--cv`    | 42.2 | 41.3 | 37.9 | 40.5 | | 144.1 | 140.6 | 133.3 | 139.3 |
| `--gru`   | 42.5 | 41.8 | 38.8 | 41.0 | | 145.7 | 135.1 | 125.0 | 135.3 |
| `--xattn` | 42.3 | 41.5 | 37.8 | 40.5 | | 144.5 | 151.5 | 115.3 | 137.1 |
| *CLIPER*  |  —   |  —   |  —   | 39.4 | |   —   |   —   |   —   | 141.3 |

**At a single step (6h) they do nothing** — all three sit at ~40.5 km, identical to
cv. No surprise: there is no multi-step memory to exploit, and a 6h forecast is
already close to constant-velocity.

**At multiple steps a signal appears.** The means edge below cv (gru 135.3,
xattn 137.1 vs 139.3) and below the CLIPER mean (141.3) — but the multi-step
split noise is large (±10 km), so the *means* alone are only suggestive. The
per-horizon breakdown on the favourable split (seed 5) is where it becomes
legible:

| horizon | CLIPER | `--cv` | `--gru` | `--xattn` |
|:--|:--:|:--:|:--:|:--:|
|  6h | 39.4 | 37.8 | 36.0 | **34.5** |
| 12h | 97.9 | 91.3 | 85.5 | **81.6** |
| 18h |171.4 |160.1 |149.4 | **138.9** |
| 24h |256.4 |244.0 |229.1 | **206.3** |

The gain **grows monotonically with horizon**: negligible at 6h (~2–3 km), but at
24h `--xattn` beats cv by 38 km and constant-velocity by **50 km (20%)**, with
`--gru` in between. This is the first place in the whole exercise where a *network*
change — recurrent memory, real cross-attention — clearly helps, and it lands
exactly where the hypothesis (§10 of the earlier discussion) said it should: at
long horizons, where storms recurve and constant-velocity drifts.

Two honest caveats keep this from being a clean win:

1. **It is split-dependent.** Seed 5 shows the effect strongly; on seed 3
   `--xattn` is actually *worse* than cv (151.5 vs 140.6). `--gru` is the more
   consistent of the two (beats cv on 2 of 3 splits, never much worse), `--xattn`
   the higher-ceiling / higher-variance one. With only 98 storms, a three-split
   spread this wide means "promising", not "established".
2. **The absolute numbers are still large.** Even the best 24h forecast (~206 km)
   is a coarse result; on this much data the model is matching-to-slightly-beating
   a two-line physics baseline, not superseding it.

So: unlike the encoder and attention tweaks of §7–§8 (flatly neutral), the
decoder's *memory* is a real lever at long horizons — the honest next step is to
confirm it with more storms and longer training, where the trend here suggests
the learned curvature term should pull further ahead of constant-velocity.

> **Superseded by §11.** The five-split re-test below does not confirm this
> section's signal: the gru/xattn edge over cv disappears (and reverses at 48h),
> and the favourable seed-5 result flips under a float-rounding-level
> perturbation. Treat §10 as the honest record of a promising-looking effect
> that did not survive a stricter test.

## 11. Five-split re-test: the §10 signal does not survive — but cv beats CLIPER at 48h

Section §10 was flagged "promising, not established" on three splits. This
section re-tests it properly: **five** split seeds, two horizon settings
(`--pred_len=4` = 24h, `--pred_len=8` = 48h), and a **split-matched CLIPER
baseline** — a 0-epoch `--cv` checkpoint evaluated on each split's own held-out
test storms (the untrained cv decoder emits exact constant-velocity, so this is
CLIPER on precisely the storms the model is scored on, unlike the all-sample
`baseline` subcommand).

**First, a numerics-sensitivity control.** Between §10 and this re-test the
matmul kernel's summation order changed (an 8-accumulator vectorization —
a float-rounding-level perturbation, ~1e-7 relative). Rebuilding the *old*
kernel reproduces §10's seed-5 headline exactly (mean 115.3 km, 24h 206.3 km);
the *new* kernel on the identical command gives 127.7 / 232.8 km. **A 26 km
swing at 24h from rounding order alone** — the "xattn beats constant-velocity
by 20%" result was split noise amplified by chaotic training sensitivity, not
a real effect. (Every §11 number below uses the new kernel consistently.)

**24h (`--pred_len=4`), held-out test ΔR mean over 6–24h, five splits:**

| decoder | s1 | s2 | s3 | s4 | s5 | mean |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| *CLIPER (split-matched)* | *150.4* | *137.4* | *152.5* | *127.5* | *120.9* | *137.8* |
| `--cv` | 143.0 | 134.7 | 141.1 | 135.7 | 127.8 | **136.5** |
| `--gru` | 143.4 | 136.2 | 140.6 | 143.5 | 129.1 | 138.6 |
| `--xattn` | 153.2 | 132.1 | 146.6 | 121.9 | 127.7 | 136.3 |

`--gru` never meaningfully beats cv and is 8 km worse on seed 4. `--xattn`
swings from 14 km better (s4) to 10 km worse (s1) — mean parity, huge variance.
The §10 ordering (xattn > gru > cv) is gone.

**48h (`--pred_len=8`), held-out test ΔR mean over 6–48h, five splits:**

| decoder | s1 | s2 | s3 | s4 | s5 | mean |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| *CLIPER (split-matched)* | *316.8* | *327.2* | *338.0* | *269.1* | *247.5* | *299.7* |
| `--cv` | 301.8 | 295.7 | 297.5 | 244.3 | 262.3 | **280.3** |
| `--gru` | 324.3 | 299.1 | 294.6 | 259.3 | 286.8 | 292.8 |
| `--xattn` | 319.7 | 298.5 | 277.2 | 269.3 | 287.7 | 290.5 |

At the longer rollout the decoder-memory variants are simply **worse** than the
plain cv MLP (by 10–12 km on average) — more parameters and state to fit on the
same 98 storms.

**The real long-horizon result is the plain cv decoder itself.** At 48h,
split-matched CLIPER vs `--cv`:

| horizon 48h | s1 | s2 | s3 | s4 | s5 | mean |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| *CLIPER* | *638.8* | *704.0* | *699.4* | *564.3* | *529.6* | *627.2* |
| `--cv` | 608.0 | 584.7 | 594.1 | 497.1 | 559.1 | **568.6** |

`--cv` beats constant-velocity on **4 of 5 splits**, by **−58.6 km (−9.3%) at
48h** and −19.4 km (−6.5%) over the full 6–48h curve. This is where §9 predicted
a learned curvature term should finally win — storms recurve over two days and
dead reckoning drifts — and it does. The lesson of the whole §9–§11 arc: the
constant-velocity **anchor plus a simple learned correction** is the winning
recipe at every horizon; adding recurrent memory or cross-attention on top only
adds variance on this much data.

## 12. Data-parallel training is variant-complete — and a cautionary tale

Multicore training (`--threads=N`) originally covered only the plain and delta
decoders: the workers never fed the per-sample auxiliary inputs — the seed
velocity the cv/gru/xattn decoders consume, and the co-active neighbour tables
`--co_spatial` consumes — so those flags were documented serial-only. That gap
is now closed: the workers feed both inputs exactly as the serial loop does,
and `tests/test_parallel` pins serial/parallel **gradient equivalence per
variant on the real dataset** (max abs diff ≤ 3e-8; batch loss identical to
~1e-16).

End-to-end, the parallel path reproduces the serial results. Held-out test ΔR
(km), `--motion`, 30 epochs, `--patience=8`:

| config | serial | `--threads=4` |
|:--|:--:|:--:|
| `--cv`, seed 1 | 43.4 | 42.4 |
| `--cv`, seed 3 | 41.1 | 41.4 |
| `--cv`, seed 5 | 37.8 | 38.4 |
| `--gru`, seed 5 | 39.1 | 40.0 |
| `--xattn`, seed 5 | 37.2 | 37.9 |
| **`--cv` 3-seed mean** | **40.8** | **40.7** |

Parallel runs are not bit-identical to serial — each worker draws its own
dropout mask stream, and gradient reduction reorders float sums — but every
config lands within ±1 km of its serial counterpart, far inside the split
noise, and the cv 3-seed means agree to 0.1 km.

**The cautionary tale.** The pre-fix code did not *refuse*
`--cv --threads=4` — it printed a note and trained anyway, with the workers
silently feeding **zero seed velocity**. Such a run looks perfectly healthy on
the training loss (it falls exactly like a good run), because the model
consistently learns curvature corrections for a zero-velocity anchor. But eval
feeds the *real* seed velocity, the learned pseudo-velocity double-counts it,
and the held-out result collapses: **105.8 km vs 37.8 serial** — worse than
doing nothing. Two things caught it instantly: the **epoch-0 anchor check**
(an untrained cv model must score exactly CLIPER on val) and **held-out
evaluation**. The lesson generalizes: a train/eval *input* mismatch is
invisible in the loss curve — validate plumbing changes on val/test metrics,
never on training loss.

## 13. Second-round levers: physics features, robust loss, seed ensembles

Three follow-ups from the optimization review, each tested with the standard
protocol (30 epochs, `--patience=8`, five split seeds unless noted). All §13
runs use the **new no-spatial default encoder** (§7 measured the paper's N=1
spatial blocks as accuracy-neutral dead weight, so the default now skips them;
`--spatial` restores the old architecture — and is required to reproduce
§1–§12's numbers exactly).

**Fresh cv baseline under the new default (6h):** 41.6 / 38.3 / 41.0 / 39.5 /
37.6 → **mean 39.6 km** — about 1 km under the spatial-default baseline (40.8,
§9) and statistically *on* the 39.4 km CLIPER bar. Dropping the dead blocks
cost nothing and may have helped slightly.

**`--physics` (acceleration, speed, heading, seasonal phase) — small but
sign-consistent.** Held-out test ΔR at 6h, `--motion [--physics] --cv`:

| seed | 1 | 2 | 3 | 4 | 5 | mean |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| baseline | 41.60 | 38.34 | 40.95 | 39.49 | 37.56 | 39.59 |
| `--physics` | 40.28 | 38.15 | 40.49 | 39.45 | 37.13 | **39.10** |

Physics wins on **all five splits** (−1.32 to −0.04 km, mean −0.49 km, ~1.2%).
The magnitude is inside single-split noise, but a 5/5 sign pattern is not —
this is the first *feature* addition since `--motion` that helps at all, and it
helps in exactly the physically-motivated way (heading and acceleration are the
curvature signal). Honest label: suggestive-to-real; more storms decide.

**Loss shaping at 48h (`--pred_len=8`, 6–48h mean ΔR):**

| loss | s1 | s2 | s3 | s4 | s5 | mean ± std |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| MSE (baseline) | 300.1 | 348.2 | 303.8 | 259.9 | 261.4 | 294.7 ± 36 |
| `--huber=0.1` | 303.1 | 272.3 | 269.7 | 282.9 | 274.7 | **280.5 ± 14** |
| `--hweight=1` | 304.7 | 327.7 | 301.7 | 248.0 | 265.4 | 289.5 ± 33 |

`--huber` wins the mean (−14 km, −4.8%) but only 2 of 5 seeds — its real effect
is the **variance collapse (±36 → ±14)**: the catastrophic splits vanish (worst
348 → 303). This is the `--delta` lesson (§5) again, in the objective this
time: on 98 storms, robustness mechanisms pay by taming tails, not by shifting
means. `--hweight` is neutral within noise — upweighting far horizons does not
help when the far-horizon errors are already what dominates the gradient.

**Seed ensembles (3 members per split, `--seed=101/102/103`, scored on the
held-out test storms via `eval --weights=a,b,c --split=test`):**

| split | member 1 | member 2 | member 3 | ensemble |
|:--|:--:|:--:|:--:|:--:|
| 1 | 43.17 | 42.75 | 42.09 | **42.16** |
| 3 | 41.30 | 40.68 | 40.99 | **40.83** |
| 5 | 37.99 | 39.91 | 37.82 | **37.98** |

The ensemble lands at ~the best member on every split without knowing which
member is best — worth ~0.5 km (~1.2%) over the *expected* single model. The
gain is smaller than typical deep-ensemble lore because cv members are highly
correlated: they share the constant-velocity anchor and learn similar small
corrections. Cheap insurance, not a lever.

**`--physics` also helps at 48h.** Repeating the comparison at `--pred_len=8`
(6–48h mean): no-physics cv 294.7 ± 36 → with physics **285.0 ± 26**, better on
4 of 5 splits. The curvature signal pays where curvature matters most.

**`--rotframe` (motion-aligned correction frame) — tested, did not help.** The
cv correction predicted in (along-track, cross-track) coordinates and rotated
into lat/lon by the velocity direction — rotation-invariant curvature, exact
backward through `∂u/∂v`, gradient-checked. On `--motion --physics --cv`:

| horizon | baseline | `--rotframe` | verdict |
|:--|:--:|:--:|:--|
| 6h mean (5 seeds) | 39.10 | 40.43 | **worse on 4/5 splits** (+1.3 km) |
| 6–48h mean (5 seeds) | 285.0 ± 26 | 279.9 ± 15 | neutral (2/5 wins, −5 km inside noise) |

The physically-elegant idea loses at 6h, plausibly because the frame itself is
noisy exactly when it matters least: a quasi-stationary storm has an
ill-defined heading, and the rotation injects that jitter into the correction.
Another entry for the §7/§8 pattern — architectural cleverness beyond the
anchor does not beat giving the network the raw signal (`--physics` already
feeds heading explicitly, and the MLP can learn its own frame). The flag stays
available and off by default.

**Takeaway.** All four behave exactly as the small-data hypothesis predicts:
physically-motivated inputs give a small consistent win (at both horizons),
robustness mechanisms collapse variance, averaging buys insurance, and imposing
structure the network can already learn does not help — nothing moves the
number a lot, because the ceiling is still 98 storms.

## 14. Third round: tuning, composition, and the ablations that close the book

The remaining suggestions from the optimization review, run as one battery
(~80 runs, standard protocol, no-spatial default; baselines are §13's
`--motion --physics --cv` numbers: 39.10 km @ 6h, 285.0 @ 48h).

**Capacity is flat — the compact config was already at the knee.** Held-out 6h
mean over seeds 1/3/5, `d_ff = 2·d_model`: d_model 32 → 40.1, 48 → 40.3,
**64 → 39.3**, 96 → 40.0. Nothing to gain in either direction; §3's lesson
(capacity is not the constraint) holds at every scale tried.

**Hyperparameters were already near-optimal — except dropout.** A 16-config
random search on split 3 (selected on VAL, never test) spans a val range of
only 34.8–36.3 km against the default's 35.3 — the defaults sit mid-pack of a
tight field. The best-val config differs from the defaults only in
**dropout 0.2**, and validates across all five splits: **38.50 vs 39.10 km**
(better on 4/5). A decade of defaults-tuning fear, worth half a kilometre.

**The recipe: `--motion --physics --cv --huber=0.1` is the new best-known
configuration at BOTH horizons.**

| config | 6h mean (5 seeds) | 48h mean (5 seeds) |
|:--|:--:|:--:|
| physics baseline (§13) | 39.10 | 285.0 |
| + `--huber=0.1` (**the recipe**) | **38.45 ± 1.4** | **275.3** |
| + `--dropout=0.2` as well | 38.29 ± 2.4 | 284.9 |
| *CLIPER bar* | *39.4* | *≈300 (§11)* |

The recipe beats the physics baseline on 4/5 splits at both horizons and puts
the 6h mean **under the CLIPER bar**. Two honest wrinkles: (1) **gains do not
stack** — physics, huber, and dropout 0.2 are each worth ~0.6 km alone but
compose to ~0.8, not 1.9: they are overlapping regularizers, not independent
signals; (2) the extra dropout **helps at 6h but hurts at 48h** (284.9 vs
275.3), so the recommended recipe leaves dropout at the 0.1 default.
Recipe ensembles (3 members/split) land at-or-under the best member on every
split (40.20 / 40.28 / 36.69 on splits 1/3/5) — the §13 insurance, preserved.

**Window length: 12 steps was right, and longer is worse.** 6h mean over seeds
1/3/5: in_len 8 → 38.9, **12 → 39.3**, 16 → 42.0, 20 → 42.3. Three extra days
of history costs 3 km — more attention surface to fit, fewer training windows
per storm, no added signal. (8 is marginally better than 12 but within noise.)

**Absolute longitude carries no robust signal.** Zeroing it (`--no_lon`):
38.83 vs 39.10 mean, better on 3/5 splits — neutral-to-marginally-better. The
model was partly *geolocating* rather than generalizing; with more storms this
ablation is worth repeating in reverse (does longitude start to pay when the
basin is densely sampled?).

**Teacher forcing: neutral.** Annealing forced rollouts over 10 epochs at 48h:
283.7 vs 285.0, better on 3/5 — inside noise. The cv anchor already gives the
early rollout a sane trajectory, which is presumably why truth-feeding adds
nothing; the flag stays for larger-data retests.

**And one speed hypothesis refuted en route:** deferring dW accumulation to
one GEMM per layer per batch (`--defer_dw`) is bit-identical and measured
*neutral* — the backward is compute-bound, not gradient-traffic-bound, at
these model sizes. Recorded so nobody re-derives that theory.

**Where this leaves the number.** From the original honest baseline (131 km,
blind to motion) to **38.5 km @ 6h** — under constant-velocity — via, in
order of impact: motion features, the cv anchor, physics features, a robust
loss, and one dropout notch. Every remaining candidate on the board is either
measured-neutral or waiting on more storms.

## 15. The 826-storm retest — the data finally arrived, and the verdicts hold

`HURDAT_full.csv` (24,177 records, **826 storms**, 2004–2025, three basins —
8.4× the bundled sample) makes the questions of §7–§14 answerable. Protocol as
always: five split seeds, 30 epochs, `--patience=8`, text-free (`--emb=none`),
split-matched CLIPER via 0-epoch cv checkpoints. One critical calibration:
**the bars move with the data** — split-matched CLIPER at 6h is now **32.0 km**
(vs 39.4 on the 98-storm sample; the multi-basin mix tracks more steadily), so
only internal comparisons are valid, never cross-dataset ones.

**The headline: the recipe beats constant-velocity over the 48h rollout on all
five splits — and the crossover is exactly where theory says it should be.**
Per-horizon held-out ΔR (km), 5-seed means, recipe (`--motion --physics --cv
--huber=0.1`) vs split-matched CLIPER:

| horizon | 6h | 12h | 18h | 24h | 30h | 36h | 42h | 48h |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| *CLIPER* | *28.0* | *68.3* | *119.0* | *178.3* | *245.2* | *318.9* | *398.3* | *482.9* |
| recipe | 31.3 | 70.7 | 120.1 | **175.5** | **236.1** | **301.4** | **370.3** | **442.3** |

Dead reckoning wins the first three steps; the learned curvature takes over at
24h and pulls away monotonically to **−40.6 km (−8.4%) at 48h**. Over the full
6–48h curve the recipe wins on **5/5 splits** (227.9 vs 235.0 mean). At 6h
alone, CLIPER wins on 5/5 (32.0 vs 35.2): the small-sample "under the bar at
6h" result (§14) did **not** survive scaling — the bar strengthened more than
the model. Honest framing: one-step extrapolation on steady storms is nearly
unbeatable; this model's value begins where forecasting gets hard.

**The levers matter MORE at scale.** Plain `--motion --cv` (no physics, MSE):
49.9 km at 6h — 14.7 km behind the recipe (on 98 storms the same gap was ~1
km). At 48h (seeds 1/3/5): plain cv 230.7 ≈ CLIPER 230.1, recipe 223.6. The
physics features and robust loss are not small-sample artifacts; they are what
lets the model use the extra data.

**Small-sample verdicts, re-tested at scale (seeds 1/3/5, 48h means):**

| variant | 826-storm result | small-sample verdict |
|:--|:--:|:--|
| `--gru` | 256.2 vs cv 230.7 — **worse by 25 km, decisively** | "parity-or-worse" (§11) → now conclusive |
| `--xattn` | 229.5 vs cv 230.7 — parity | "worse at 48h" (§11) → parity; still no reason to pay for it |
| `--rotframe` (on the recipe, 6h) | 37.1 vs 35.3 — worse on 3/3 | negative (§13) → confirmed |

Decoder memory is dead as an idea on this task; the frame rotation too. The
§11 methodology lesson also held up: the noise floor dropped with the data and
none of the "promising" small-sample wiggles reappeared.

**Absolute progress, for the record.** Trained on 826 storms the recipe scores
35.2 km @ 6h and 227.9 over 6–48h — versus 38.5 and 275.3 when trained on 98
storms. More data improved the model everywhere, exactly as every section
since §3 predicted; it just improved the baseline's *test conditions* too.

**The scaled re-tune: the recipe's defaults survive (48 runs).** Fifteen
configs (lr, lr_decay, dropout, batch, weight decay, Huber δ) × splits {1, 3},
selected on VAL, never test. The val landscape is *flat* — 30.95 to 32.06 km
across all fifteen — with the recipe itself tied-best at 30.95. Two lessons
inside that flatness:

1. **Tuning does not transfer across dataset sizes.** Dropout 0.2, the clear
   winner of the 98-storm search (§14), is now among the *worst* configs
   (32.06). More data wants less regularization; re-tune when the data changes.
2. **Val-tied challengers validated across all five splits at both horizons,
   and neither clears the bar.** `lr_decay=0.9` gains ~1.3 km at 6h but blows
   up on two 48h splits (288/269 vs the recipe's 213/231) — rejected.
   `lr=2e-3 --lr_decay=0.9` shows −1.9 @ 6h / −1.3 @ 48h on means but only
   3/5 splits each, with its largest win on the volatile split-3 test set, and
   it did not beat the recipe on val — adopting it would be test-set
   selection. **The defaults stand.**

**The split-3 "outlier," resolved.** Three init seeds of the identical recipe
on split 3 score 38.2 / 40.0 / 41.8 at 6h while its val barely moves
(30.1–30.5): that split's test storms are intrinsically volatile, not
mis-tuned — run-to-run spread on one test set can reach ±2 km even at 826
storms. Multi-seed reporting remains mandatory.

## 16. What this means

- The engineering was sound **on the platform it was tested on** (gradient
  checks, golden, OpenCL cross-backend agreement) — but only there. §17 audits
  the two paths CI never executed, and both were broken. The *modeling* had a
  concrete, fixable flaw — the inputs omitted motion — and fixing it moved
  held-out ΔR from ~131 km to **48 km**, turning a sub-persistence model into
  one that beats persistence 2.5× and rivals constant-velocity.
- The **language branch does not earn its keep** here, before or after the fix.
  Whether it would on a larger, harder dataset (rapid intensification, recurvature,
  extratropical transition — where text might carry signal the numbers don't) is
  the honest open question.
- **The gap to constant-velocity is now closed** (§9) **and, at long horizons,
  passed** (§11). `--cv` anchors the decoder at constant-velocity instead of
  persistence and learns only the curvature: 49.9 → **40.8 km** at 6h, matching
  the 39.4 km CLIPER baseline — and at 48h it **beats split-matched CLIPER on 4
  of 5 splits (−9.3%)**, exactly where a learned curvature term should win. The
  right *parametrization* — like motion in the inputs — is a real lever; a
  km-aware loss (`--km_loss`) by contrast was **tested and did not help**. The
  ceiling on everything remains **more than 98 storms**.
- **The network internals barely move the single-step number** (§7, §8). An encoder
  sweep (no_spatial / posenc / pool=last / prenorm) and two attention upgrades — a
  temporal distance bias (`--timebias`) and *real* multi-node spatial attention over
  co-active storms (`--co_spatial`) — are all neutral within split noise at 6h.
  Building `--co_spatial` properly still paid off: it surfaced (and we fixed, via a
  zero-init residual) a real training divergence that a single-split run would have
  hidden. The levers that moved the 6h number were all about the physics the model
  sees — motion features, a displacement head, a constant-velocity anchor.
- **The second round of levers behaved exactly as small-data theory predicts**
  (§13). Physics features (heading, acceleration, seasonal phase) give a small
  win that is **sign-consistent across all five splits** — the first feature
  addition since `--motion` that helps at all. A Huber loss collapses the 48h
  split variance (±36 → ±14 km) without reliably shifting the mean; horizon
  weighting is neutral; 3-seed ensembles recover ~best-member skill (~1%
  insurance). Inputs and robustness keep paying small dividends; capacity and
  machinery still pay nothing.
- **Decoder *memory* did not survive the re-test** (§10 → §11). On three splits,
  `--gru`/`--xattn` looked like the first network change that helps at long
  horizons. On **five** splits, at both 24h and 48h, the effect is gone: both
  variants sit at parity with the plain cv MLP at 24h and are 10–12 km *worse*
  at 48h. The §10 headline (xattn 20% under CLIPER at 24h) flipped under a
  float-rounding-level change in matmul summation order — a 26 km swing from
  numerics alone, which is the clearest demonstration yet of how little a
  single-split result means on 98 storms. The long-horizon win belongs to the
  **plain cv decoder** (previous bullet); extra decoder machinery just adds
  variance at this data scale.

## 17. The portability audit — both untested paths were broken

Two code paths had never been *executed*: the Windows build (the data loader's
`FindFirstFile` branch carried the comment *"Untested on Windows"*) and the CUDA
backend (CI compile-checks the `.cu`; the runners have no GPU). Running each for
the first time broke it immediately. Neither failure is exotic; both were
invisible to a Linux-only CI, which is the whole point of recording them here.

**Windows: the RNG was 32-bit, and every weight initialized to the same value.**
The xorshift64 state lived in `unsigned long` — 64-bit on LP64 (Linux/macOS),
**32-bit on LLP64 (Windows)**. The seed truncated
(`88172645463325252` → `3418323524`), `(13,7,17)` is not a full-period triple at
32 bits, and — fatally — `nn_uniform` takes `xorshift() >> 11` and divides by
`2^53`. With ~21 significant bits over a `2^53` denominator, **every draw
collapses to ~0**:

| `nn_uniform(-0.5, 0.5)` | as shipped | with `uint64_t` |
|:--|:--:|:--:|
| draw 1 | **-0.499999999986** | -0.025741013236 |
| draw 2 | **-0.499999999935** | -0.335152426809 |
| draw 3 | **-0.499999999931** | -0.312758417299 |

Every parameter starts at the bottom of its range, so the network is born
**perfectly symmetric** and the analytic backward cannot agree with finite
differences: `test_nn` failed at **max rel err 9.99e-01, 72 outliers**. The build
was clean and the losses looked plausible the whole time. The same truncation hit
the storm-split shuffles, the per-worker dropout seeds, and an FNV multiplier
(`0x100000001b3`, itself > 2^32).

Fixed with `uint64_t` throughout. On LP64 that is a **no-op** — `unsigned long`
*is* 64-bit there — and the proof is that `test_golden`, which pins the training
loss bit-exactly and was pinned on Linux, now **passes on Windows**: both
platforms generate an identical stream. The full suite is 12/12 green on
mingw-w64, and the `FindFirstFile` branch now discovers the embedding chunks and
trains end-to-end.

**CUDA: the backend could never have run.** First execution on a Quadro P1000
(sm_61, CUDA 12.8) segfaulted in about four seconds. `tensor_cuda.cu` made
`Mat.data` a `cudaMalloc`'d **device** pointer, asserting that "the rest of the
program is oblivious to where the data lives." It is not: `nn.c` and `model.c`
dereference `Mat.data[i]` **directly in ~189 places** (weight init, loss, softmax,
all the glue), as does the backend cross-check. The first host read of a device
pointer dies. The backend was structurally incompatible with its own callers.
`tensor_opencl.c` — which works — documents the opposite choice for exactly this
reason: host-resident data, staged per call. CUDA now does the same.

**Cross-backend agreement, finally measured for CUDA** (`make CUDA=1 test-cuda`,
new; mirrors `test-opencl`):

| gate | result |
|:--|:--|
| kernel cross-check, 11 kernels vs CPU reference | worst **1.19e-07** (tol 1e-04) |
| tensor gradient check *through* CUDA | pass |
| full-model gradient check *through* CUDA | **22/22 variants, 0 outliers** |

**The strongest check is not a gradient check — it is the pipeline.** Gradient
checks only prove the backward agrees with the *forward*; a self-consistently
wrong forward passes them. So the whole program was run through CUDA
(`train → eval → predict → baseline`, all exit 0) and compared against the CPU
backend on the **identical command** (`./typhoformer 1`, 1 epoch, demo config):

| | init val ΔR | epoch-1 loss | val ΔR | held-out test ΔR |
|:--|:--:|:--:|:--:|:--:|
| CPU | 2120.89 km | 0.27334 | 572.03 km | **758.86 km** |
| CUDA | 2120.89 km | 0.27333 | 572.15 km | **759.08 km** |

Agreement to ~4 significant figures across a full training epoch; the residual
drift is 1e-07-level kernel divergence compounding over thousands of updates,
exactly as expected. (Those absolute numbers are terrible because the demo config
is the motion-blind model of §4 after one epoch — the GPU is faithfully
reproducing a bad model, which is the point.)

**A gap the cross-check was hiding.** Chasing those numbers revealed that the
backend cross-check only ever tested **8 of the 13** seam functions: `mat_colsum`,
`mat_copy` and `mat_zero` were never checked — and `colsum` is the one genuinely
non-trivial reduction (CUDA does it with `atomicAdd` across every row of a column).
A backend could pass every other op and still get it wrong. All three are now in
the shared check (colsum 1.19e-07; copy and zero exact).

**A full training run is NOT bit-reproducible across backends — and that is §11,
live.** Seed 1 of the §15 recipe (30 epochs, `--pred_len=8`, 826 storms) was
trained end-to-end on both backends. They start **bit-identical** — the untrained
model scores val ΔR 221.94 km on each — but by epoch 1 the val curves already part
(232.8 vs 246.3 km) and they keep diverging. The reason is not a bug: SGD is
chaotic, so the per-op 1e-7 kernel differences compound over ~7,400 gradient steps
an epoch, and the two weight trajectories separate even though every individual
operation still agrees to 1e-7. Both runs then find essentially the **same best
val** (CPU 215.20 vs CUDA 215.97 km, 0.4% apart) — the model quality is identical —
but early stopping, on the flat val landscape §15 documented, selects **different
epochs** (CPU epoch 16, CUDA epoch 5), and those two checkpoints give held-out
per-horizon curves ~3% apart (CUDA 31.2/122.5/471.3 vs CPU 32.3/125.6/487.9 km at
6h/18h/48h). This is exactly the sensitivity §11 built the multi-seed methodology
around: swapping CPU→CUDA is a float-rounding-level perturbation, and it moved the
single-seed result by ~3% and flipped the selected epoch — which is precisely why
§15 reports **five-seed means, never single-seed numbers**. The qualitative result
(the per-horizon shape, the CLIPER-beating crossover) reproduces on the GPU; the
exact single-seed km values are not backend-reproducible, by the nature of the
optimisation, not any fault of the port.

**And a negative performance result, which is the honest headline.** This backend
is a **correctness reference, not a speed win**. The model is tiny (`d_model=64`,
T=12), so per-op host↔device transfer and the launch/sync overhead dominate
completely: the GPU idles at **~7% utilisation** and the full-model gradient check
takes **2150 s (36 min)** — against ~1 s on the CPU. An earlier version that did a
`cudaMalloc`/`cudaFree` pair per matrix op was worse still (those cost ~1 ms
*each*; it was still grinding after 30 minutes), fixed with a three-slot
device-buffer pool — but pooling only removed the allocator, not the transfers.

**Why device-resident data does not rescue it — the seam is too narrow.** The
tempting fix is to keep `Mat.data` on the GPU across ops. It does not help, and
tracing *why* is the real lesson. The `tensor.h` seam the backend accelerates is
only ~13 ops (the GEMMs, `relu`, `sigmoid`, …). But most of the model's arithmetic
runs **outside** it, in plain host C: `layernorm_forward` computes mean/variance
straight off `x.data[]`; softmax, the GRU gates, PGF gating and the entire decoder
rollout do the same. Those are the ~189 direct `.data[]` accesses — not glue, the
core layer math. And they are **interleaved** with the seam ops: a matmul (GPU)
feeds a layernorm (host) feeds the next matmul (GPU) feeds a softmax (host). So the
data *structurally* has to live on the host; making it device-resident would force
a sync-down after every single seam op anyway. The per-op transfers are not
redundant — the narrow seam requires them.

The arithmetic makes it concrete. The largest GEMM (`[12,64]@[64,128]`) is ~196K
FLOPs ≈ **100 ns** of compute on the P1000. Each op pays a host↔device round trip +
launch + `cudaDeviceSynchronize` ≈ **20–50 µs** (higher under WSL2's paravirtualised
GPU). That is a **200–500× overhead-to-compute ratio** — the GPU does 100 ns of work
then waits tens of microseconds, which is exactly why it idles at ~7%.

So beating the CPU is not an optimization of this backend; it is a different
backend. It requires porting the layer math itself — layernorm, attention
(softmax + scaling), GRU, PGF, the decoder rollout — into GPU kernels so the whole
forward+backward stays on the device, and *then* batching samples into the encoder
GEMMs so each launch does enough work to amortise its overhead. On a Pascal P1000
at `d_model=64` even that wins only modestly; the payoff scales with model size
(`--full`, d_model=256) and batch. Given training is already ~14 min on the CPU and
the real bottlenecks (data, motion-blind inputs) are solved, that port was judged
not worth its cost. The seam is *proven* — the same model math runs unmodified on
CPU, OpenCL and CUDA and agrees to 1e-07 — but deliberately not *exploited*. This
mirrors §7 from the other side: knowing where the time goes matters more than adding
hardware, and here the time is in the transfers a narrow seam forces, not the maths.

**Methodological note.** "It compiles" is not "it runs", and one platform is not
all platforms. Both bugs sat behind green CI for the entire life of the project.
The gates that would have caught them — a Windows build, and an *executing* CUDA
target — now exist; only the second still cannot run in CI, for want of a GPU
runner.

## Reproduce

The commands below record the exact protocol used for the numbers above
(`--threads=1` throughout). The cv/gru/xattn/co_spatial paths now also train
data-parallel — add `--threads=N` for a faster, statistically equivalent run
(different dropout streams and summation order, same distribution).

```sh
# §17 portability gates
make test                                        # 12/12, also green on Windows (mingw-w64)
make OPENCL=1 test-opencl                        # kernel cross-check + gradient checks via OpenCL
make CUDA=1 CUDA_ARCH=sm_61 test-cuda            # same, on a real NVIDIA GPU (needs a 12.x toolkit)
make CUDA=1 CUDA_ARCH=sm_61 typhoformer && ./typhoformer 1   # the PIPELINE through CUDA; compare to CPU

# the fix, across splits
for s in 1 2 3 4 5; do ./typhoformer train 30 --patience=8 --motion --delta --split_seed=$s; done
# the constant-velocity decoder — reaches CLIPER (§9)
for s in 1 3 5; do ./typhoformer train 30 --patience=8 --motion --cv --threads=1 --split_seed=$s; done
# decoder memory at long horizons (§10) — compare per-horizon dR at pred_len=4
./typhoformer train 30 --patience=8 --motion --cv    --threads=1 --pred_len=4 --split_seed=5
./typhoformer train 30 --patience=8 --motion --gru   --threads=1 --pred_len=4 --split_seed=5
./typhoformer train 30 --patience=8 --motion --xattn --threads=1 --pred_len=4 --split_seed=5
# the five-split re-test at 24h and 48h (§11); a 0-epoch cv run = split-matched CLIPER
for P in 4 8; do for s in 1 2 3 4 5; do for dec in cv gru xattn; do
  ./typhoformer train 30 --patience=8 --motion --$dec --threads=1 --pred_len=$P --split_seed=$s
done; ./typhoformer train 0 --motion --cv --threads=1 --pred_len=$P --split_seed=$s; done; done
# does text help the fixed model?
./typhoformer train 30 --patience=8 --motion --delta --split_seed=42            # with text
./typhoformer train 30 --patience=8 --motion --delta --split_seed=42 --no_text  # numbers only
# the honest baseline the model must beat
./typhoformer baseline --pred_len=4
# architecture explorations (all neutral, §7–§8) — vary --split_seed to see the noise
./typhoformer train 30 --patience=8 --motion --delta --no_spatial          # encoder sweep
./typhoformer train 30 --patience=8 --motion --delta --timebias            # temporal distance bias
./typhoformer train 30 --patience=8 --motion --delta --co_spatial --threads=1  # real spatial attn
# second-round levers (§13) — note: §1-§12 predate the no-spatial default; add
# --spatial to those commands to reproduce their numbers exactly
for s in 1 2 3 4 5; do
  ./typhoformer train 30 --patience=8 --motion --physics --cv --split_seed=$s   # physics features
  ./typhoformer train 30 --patience=8 --motion --cv --pred_len=8 --huber=0.1 --split_seed=$s
  ./typhoformer train 30 --patience=8 --motion --cv --pred_len=8 --hweight=1 --split_seed=$s
done
for sd in 101 102 103; do ./typhoformer train 30 --patience=8 --motion --cv --split_seed=5 --seed=$sd --save=m$sd.ckpt; done
./typhoformer eval --weights=m101.ckpt,m102.ckpt,m103.ckpt --cv --split=test --split_seed=5   # ensemble
```

(See [LABS.md](LABS.md) Track D for these as guided exercises.)
