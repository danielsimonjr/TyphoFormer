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

**Takeaway.** All three behave exactly as the small-data hypothesis predicts:
physically-motivated inputs give a small consistent win, robustness mechanisms
collapse variance, averaging buys insurance — and nothing moves the number a
lot, because the ceiling is still 98 storms.

## 14. What this means

- The engineering was always sound (gradient checks, golden, cross-backend
  agreement). The *modeling* had a concrete, fixable flaw — the inputs omitted
  motion — and fixing it moved held-out ΔR from ~131 km to **48 km**, turning a
  sub-persistence model into one that beats persistence 2.5× and rivals
  constant-velocity.
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

## Reproduce

The commands below record the exact protocol used for the numbers above
(`--threads=1` throughout). The cv/gru/xattn/co_spatial paths now also train
data-parallel — add `--threads=N` for a faster, statistically equivalent run
(different dropout streams and summation order, same distribution).

```sh
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
