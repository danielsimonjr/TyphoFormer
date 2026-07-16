# TODO

Open work, sourced from the docs (chiefly `typhoformer-c/docs/FINDINGS.md`) and the current state of the code. The completed history lives in `CHANGELOG.md` and the "Status — complete" section of `typhoformer-c/README.md`.

## Research (the "honest next steps" from FINDINGS.md)

- [x] **Confirm the long-horizon decoder gains.** Done — they did *not* confirm: across five splits at 24h and 48h, `--gru`/`--xattn` are at parity-or-worse vs plain `--cv`, and the §10 headline flipped under a summation-order numerics perturbation (FINDINGS §11).
- [x] **Push horizons past 24h.** Done — at `--pred_len=8` (48h) the plain `--cv` decoder beats split-matched constant-velocity on 4/5 splits (−9.3% at 48h), the predicted long-horizon win (FINDINGS §11).
- [x] **Scale past 98 storms — the ceiling on everything. THE DATA IS IN THE REPO.** `bash tools/fetch_hurdat2.sh` was run on Daniel's machine (NOAA *is* reachable there; the dev container's 403 was environmental, not a bug) and `HURDAT_full.csv` is committed: **24,177 records / 826 storms / 2004-2025**, three basins — **8.4× the 98-storm sample**, including the full 2025 season. The raw NOAA sources are pinned too, so the dataset is reproducible even though NOAA revises HURDAT2 between releases.
- [x] 🟢 **Re-run the FINDINGS grid on the real data.** Done — FINDINGS §15 (37 runs on 826 storms): **the recipe beats split-matched constant-velocity over the 48h rollout on all five splits** (crossover at 24h, −8.4% at 48h); at 6h the stronger multi-basin CLIPER bar stays ahead (32.0 vs 35.2). The levers matter *more* at scale (motion-only cv trails the recipe by 14.7 km at 6h). Re-tests: `--gru` now decisively worse (+25 km at 48h), `--xattn` parity, `--rotframe` confirmed negative. The language branch could not be re-tested (no embeddings exist for the new records — tracked below).
- [x] **Re-tune on the full dataset.** Done (48 runs, FINDINGS §15) — the recipe's defaults are **at the val optimum** on 826 storms (val landscape flat: 30.95–32.06 across 15 configs, recipe tied-best); the 98-storm winner (dropout 0.2) inverted to worst — tuning doesn't transfer across dataset sizes; both val-tied challengers failed five-split validation (`lr_decay=0.9` blows up at 48h; `lr=2e-3+decay` is parity within run noise and didn't win on val). Split-3 "outlier" resolved as test-set volatility (init seeds 38.2/40.0/41.8 on unmoved val) — multi-seed reporting remains mandatory.
- [ ] **Re-test the language branch on harder data.** The robust negative result (`--no_text` marginally *better*) may not hold where text could carry signal the numbers don't — rapid intensification, recurvature, extratropical transition (the open question of FINDINGS §16). Requires generating GPT-4o descriptions + MiniLM embeddings for the 24,177 new records (`tools/gen_descriptions.py` + `gen_embeddings.py`, needs `OPENAI_API_KEY`).

## Speed (from the 2026-07 optimization review)

- [x] **Unrolled reductions in the backward path and attention kernels.** Done — `linear_backward`'s bias reduction and all six MHA loops restructured to contiguous/multi-accumulator form; full-config forward+backward 39.6 → 16.6 ms/sample combined with the default change below.
- [x] **Make the no-spatial encoder the default.** Done — `--spatial` restores the paper architecture (and loads pre-change checkpoints); `--no_spatial` kept as a no-op; golden re-pinned; both paths still gradient-checked.
- [x] **Parallelize `evaluate()`.** Done — eval shards across the ParTrainer replicas (training validation passes and `eval --threads=N`); metrics identical to serial.
- [x] **Portable SIMD tier.** Done — `make MARCH=x86-64-v3`.
- [x] **Batch samples into the GEMMs — MEASURED, and it does NOT help. Do not build this.** The premise ("weights re-streamed per sample") is false. `mat_matmul` is `ikj`-order, which reads each weight row **once per OUTPUT row**; 8×`[12,k]@[k,n]` and 1×`[96,k]@[k,n]` produce the same 96 output rows, so the weight traffic is identical — stacking the minibatch changes nothing. Measured directly with the repo's own kernel (2026-07-15, i7-8850H, `-O3 -march=native`): recipe shapes give **0.57–1.04×** (the 128×64 case is 1.75× *slower* batched, from blowing the L1 working set); even `--full`-scale weights (256×1024, 1 MB, spilling L2→L3) give **≤1.03×**. GEMMs are 52% (recipe) / 62% (`--full`) of fwd+bwd time, but sample-batching cannot claim any of it. The real (uncertain, localized) GEMM lever is a **register-blocked microkernel** — tile the `m` loop so a block of output rows reuses each `B[p][:]` from registers — NOT sample batching. Recorded so this invasive refactor is never re-proposed on the false premise.
- [ ] ⚪ **GPU backend cannot beat the CPU without a full model port (analysed 2026-07-14, FINDINGS §17).** The `tensor.h` seam is only ~13 ops; the model's real arithmetic (layernorm, softmax, GRU, PGF, decoder rollout — the ~189 host `.data[]` accesses) runs on the host and is *interleaved* with the seam ops, so data structurally can't stay device-resident. Each GPU op is ~100 ns of compute against ~20–50 µs of transfer/launch/sync overhead (200–500×), so the P1000 idles at ~7%. Beating the CPU means porting the layer math into kernels **and** batching — a from-scratch GPU model, judged not worth it for a tiny model whose CPU training is already ~14 min. Kept here as the recorded verdict, not an open task.

## Accuracy (from the 2026-07 optimization review)

- [x] **Physics features v2.** Done — `--physics` (acceleration, speed, heading, seasonal phase) beats the cv baseline on **all five splits** at 6h (39.59 → 39.10 km mean); small but sign-consistent (FINDINGS §13).
- [x] **Seed ensembling at eval.** Done — `eval --weights=a,b,c` with `--split=test` scoring; 3-member cv ensembles recover ~best-member skill on every split (~1% over the expected single model; FINDINGS §13).
- [x] **Loss shaping at long horizons.** Done for Huber + horizon weights — `--huber=0.1` collapses 48h split variance (±36 → ±14 km, mean −4.8%); `--hweight` is neutral (FINDINGS §13). The teacher-forcing-with-annealing test remains open (tracked below).
- [x] **Teacher forcing with annealing.** Done (`--tf=E`) — and neutral: 283.7 vs 285.0 km at 48h across five splits (FINDINGS §14). The cv anchor already stabilizes early rollouts.
- [x] **Motion-aligned frame for the cv correction.** Done (`--rotframe`, exact backward through ∂u/∂v, gradient-checked serial+parallel) — and honestly negative: worse on 4/5 splits at 6h, neutral at 48h (FINDINGS §13). Off by default.
- [x] **Text-free data path to scale storms.** Done — `tools/hurdat2_to_csv.py` converts raw NOAA HURDAT2 to the repo's CSV schema, and `--emb=none` trains without embedding chunks (language branch fed zeros == `--no_text`). The actual 20-year retrain still needs the HURDAT2 download (tracked in the research item above; NOAA is unreachable from this environment's network policy).

## Improvement roadmap (2026-07-15 — worked one at a time, easy → hard)

Grounded in FINDINGS: the model's own measurements say **architecture is neutral**
(§3, §7, §8) and the levers that moved it were **data** (98→826 storms) and
**physics in the inputs** (`--motion`). So this roadmap attacks *information* and
*training statistics*, not the Transformer. Each item: investigate → measure →
implement or refute → gradient-check → FINDINGS/CHANGELOG → PR. Negatives are wins.

- [x] **1. Right-size sweep (speed · EASY · possibly free).** ~~Adopt the smallest
      size within noise of the recipe.~~ **Done — default UNCHANGED (FINDINGS §18).**
      Measured 5 sizes × 5 seeds. At **6h** every size down to `d16L2` (7× fewer
      params, 8.66× faster) is within noise and d16L2 has the best mean — looked
      free. But the confirmation at **pred_len=8 (48h rollout)** inverts it: the
      recipe's capacity is load-bearing at long horizons, and smaller models regress
      toward the CLIPER bar (d16L2 keeps half the margin; d32L2 ties CLIPER). Verdict:
      keep `d64L2` default; `d16L2` documented as a fast/short-horizon option. The 6h
      sweep alone would have silently killed the model's only advantage.
- [x] **2. Register-blocked GEMM microkernel (speed · MEDIUM).** **Done — REFUTED,
      reverted (FINDINGS §19).** MR=4 blocking of `mat_matmul` is bit-exact (golden
      unchanged) and a clean **1.55× in an isolated microbench** — but in-situ it
      **regresses the recipe 16%** (mat_matmul itself 6.5% slower in place: the bigger
      blocked body costs i-cache/cold-data that the tight-loop microbench hides). It
      helps only `--full` (1.15×), the capacity-overkill config. A size guard recovers
      recipe parity but needs a machine-specific `k·n` threshold that doesn't belong in
      a portable kernel. Isolated ≠ in-situ; measured, not shipped.
- [x] **3. Stochastic Weight Averaging (accuracy · MEDIUM-EASY).** **Done — SHIPPED as
      `--swa`, the roadmap's first accuracy win (FINDINGS §20).** Averages the late-epoch
      tail instead of one best-val checkpoint. Measured 5 seeds: **neutral at 6h (+1.2 km,
      2/5) but −6.0 km at 48h (4/5 seeds)** — the chaotic-training noise it smooths
      compounds over the rollout, so it pays off at long range. Extends the recipe's margin
      over CLIPER from 7.1 to 14.0 km at 6–48h. Opt-in (neutral at 6h, changes training
      behavior); bit-neutral when off. Honest SEM caveat ~1.6σ.
- [x] **4. Great-circle (km) loss (accuracy · MEDIUM).** **Done — reworked + measured
      NEUTRAL (FINDINGS §21).** Rebuilt `--km_loss` from a serial-only *inconsistent*
      gradient hack into a proper equirectangular km objective in `model_loss`: consistent
      loss+gradient (gradient-checked), both training paths (no longer serial-only). Chose
      equirectangular over haversine deliberately (haversine's gradient is singular at
      pred→true). Measured 5-seed: neutral at 6h (−0.5, 2/5) AND 48h (+1.6 ± 11, 3/5, 0.3σ)
      — the `--cv` anchor + short-range steps put the km and MSE optima nearly together.
      **Kept** (repairs a broken flag, unlike Item 2's reverted new complexity), off by default.
- [x] **5. Direct multi-horizon head (speed + accuracy · MEDIUM-HARD).** **Done — SHIPPED
      as `--direct`, the roadmap's BIGGEST win (FINDINGS §22).** Predicts all horizons at
      once (no rollout), anchored at seed-CV. Refuted my prediction that the AR recurrence
      would win: direct beats `--cv` at 6h (32.1 vs 35.2, ties CLIPER, σ 0.86 vs 2.71) AND
      48h (−10.2 km 6–48h mean, 5/5 seeds, ~4.8σ), worse on no seed, and faster. Erases the
      "loses at 6h" caveat; widens the CLIPER margin 7.1→17.2 km. First architecture change
      that helps — removes rollout error compounding rather than adding capacity. Gradient-
      checked. **Candidate to replace `--cv` as the default decoder (ADR — surfaced to Daniel).**
      ⏭ open: does it stack with SWA (§20)?
- [x] **6. Probabilistic / NLL uncertainty head (accuracy · HARD).** **Done — REFUTED,
      reverted (FINDINGS §23).** Built + gradient-checked a heteroscedastic Gaussian NLL head
      on the direct decoder (per-horizon log-variance channel, race-free). It **regresses the
      mean at every horizon** — +73% @ 6h, +24% @ 48h, 0/10 seed-horizons, some seeds collapse.
      Mechanism: `∂loss/∂r = r·e^{−lv}` lets the model inflate variance to abandon hard samples.
      The detached-variance fix breaks the FD gradient-check invariant, so the only FD-consistent
      NLL is the coupled one — and it eats the mean. Calibrated uncertainty, if wanted, needs a
      separate variance model over frozen residuals (a 2nd pass), not a coupled loss.
- [x] **7. Global basins via IBTrACS, esp. West Pacific (accuracy · HARD · data).** **Done —
      SHIPPED, the first data win (FINDINGS §24).** Built `ibtracs_to_csv.py` (synoptic-time +
      spur-track handling) + `fetch_ibtracs.sh`; committed `IBTRACS_WP.csv` (**1,446 WPac storms,
      1.75× HURDAT**). The recipe (`--direct`) beats split-matched CLIPER on West Pacific by
      **−11.4% over 6–48h (5/5 seeds)** — a *wider* margin than the Atlantic's −7.3%: the model
      helps more where storms recurve more, on the basin it's named for. Confirms §3 (data is the
      ceiling). ⏭ other basins + a global model are one `fetch_ibtracs.sh <BASIN>` away.
- [~] **8. Environmental steering predictors from ERA5 (accuracy · HARDEST · data + physics).**
      **PREMISE VALIDATED + data path proven; full feature NOT yet built (FINDINGS §25).** The
      physical gap: track is governed by steering flow the model can't see. Proof-of-concept
      (`tools/era5_steering_poc.py`) collocated 500 hPa ERA5 wind with HURDAT points →
      **corr(u,Δlon)=+0.63, corr(v,Δlat)=+0.91** — the steering flow predicts the motion.
      Data reachable with **no credential wall** (public ARCO-ERA5, anon; zarr stack on Windows
      Python — WSL Python has no pip). ⏭ REMAINING (a real multi-hour build): (1) efficient
      collocation batched by timestamp (naive ~27 s/point → days for ~16k points); (2) leakage-
      safe CSV columns for deep-layer (u,v)[+shear]; (3) `data.c` loader extension + `--steering`
      flag; (4) retrain + measure held-out ΔR (expected largest at long horizons). Coverage caveat:
      ARCO ends 2021, HURDAT runs to 2025 → restrict to ≤2021 or add a later ERA5 source.

> **Deliberately NOT on the list:** more work on the language branch. The GPT-4o text is
> generated *from* the track numbers, so it is largely redundant — which is why `--no_text`
> keeps winning (§6, §16). It would only help carrying *independent* signal (forecaster
> discussion, environmental analysis), which the current descriptions don't. One clean
> re-test on harder cases is the research item above; beyond that, not worth the effort.

## Engineering

- [x] **Multicore support for the serial-only paths.** Done — the data-parallel workers now feed the per-sample aux inputs (seed velocity, co-active neighbours), so `--cv`/`--gru`/`--xattn`/`--co_spatial` all train with `--threads=N`; `test_parallel` pins serial/parallel gradient equivalence for each variant on the real dataset. Only `--km_loss` remains serial-only.
- [x] **Run-verify the CUDA backend on a real GPU. Done — and it did not work.** Executed on a **Quadro P1000 (sm_61, CUDA 12.8, WSL2)** via the new `make CUDA=1 test-cuda`. It **segfaulted immediately**: the backend made `Mat.data` a `cudaMalloc`'d **device** pointer while asserting "the rest of the program is oblivious to where the data lives" — it is not, `nn.c`/`model.c` dereference `Mat.data[i]` directly in **~189 places**, so the first host read dies. It compiled for years and could never have run. Fixed by making the data **host-resident with per-call device staging** (the choice the OpenCL backend documents), plus a **device-buffer pool** — `cudaMalloc`/`cudaFree` cost ~1 ms each, and doing them per matrix op left the gradient check still grinding after 30 minutes. Now: kernel cross-check **exact** (worst 1.19e-07 vs 1e-04 tol), tensor + full-model gradient checks pass **through the GPU**.
      ⚠ **Slow by construction:** the model is tiny (`d_model=64`), so per-op host↔device transfer dominates and the GPU idles at ~7%. This backend is a **correctness reference, not a speed win.** A speedup is NOT a contained "make the data device-resident" change — the `tensor.h` seam is only ~13 ops and the model's real math (layernorm/softmax/gru/pgf/decoder, the ~189 host `.data[]` accesses) is interleaved with it, so the data structurally must live on the host. Beating the CPU means porting that math into kernels + batching (a from-scratch GPU model). Full analysis + verdict: FINDINGS §17 and the GPU item under **Speed** above.
- [x] **Build and test on Windows. Done — and it found a real bug.** Built with MSYS2 mingw-w64 gcc 12.1 (`make`, warning-clean under `-Wall -Wextra -Wpedantic`), full suite **12/12 green**, and the `FindFirstFile` branch (whose own comment read *"Untested on Windows"*) is now **executed**: it discovers the 8 `emb_chunk_*.npy` files and trains end-to-end. Also verified on the real 826-storm CSV with `--threads=4` (winpthreads). **The bug:** the xorshift64 RNG lived in `unsigned long`, which is 32-bit on Windows — every weight initialized to the bottom of its range, the gradient check failed, and the model was silently garbage. Now `uint64_t` throughout; `test_golden` passes on Windows, so both platforms generate a bit-identical stream.
- [x] **Record the decoder variant in the checkpoint. Done — after it bit us for real.** The `TFW4` header now carries a mode bitfield and `eval`/`predict` self-configure from it (`decoder/arch (from checkpoint): cv+no_spatial`). This was not cosmetic: the size-mismatch guard **cannot** catch `--cv`/`--delta`/`--rotframe` because they are parameter-neutral, so a cv-trained checkpoint evaluated without `--cv` loaded cleanly and reported **3726 km** where training had said **31.02 km** — a silent ~120× error, caught only because the number was absurd. Older checkpoints still load and now warn loudly. Round-trip pinned in `tests/test_checkpoint`.

## Conventions for this file

Check items off (`- [x]`) as they land, and record the outcome under `[Unreleased]` in `CHANGELOG.md`. Add new pending tasks here as they come up. For research items, FINDINGS.md is the record of results — including negative ones.
