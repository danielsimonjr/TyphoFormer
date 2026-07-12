# TODO

Open work, sourced from the docs (chiefly `typhoformer-c/docs/FINDINGS.md`) and the current state of the code. The completed history lives in `CHANGELOG.md` and the "Status — complete" section of `typhoformer-c/README.md`.

## Research (the "honest next steps" from FINDINGS.md)

- [x] **Confirm the long-horizon decoder gains.** Done — they did *not* confirm: across five splits at 24h and 48h, `--gru`/`--xattn` are at parity-or-worse vs plain `--cv`, and the §10 headline flipped under a summation-order numerics perturbation (FINDINGS §11).
- [x] **Push horizons past 24h.** Done — at `--pred_len=8` (48h) the plain `--cv` decoder beats split-matched constant-velocity on 4/5 splits (−9.3% at 48h), the predicted long-horizon win (FINDINGS §11).
- [ ] **Scale past 98 storms — the ceiling on everything.** Now ONE command wherever `www.nhc.noaa.gov` is reachable: `bash tools/fetch_hurdat2.sh` (from `typhoformer-c/`) downloads the latest Atlantic + NE Pacific HURDAT2, converts (default `--since=2004`, ~700 storms with wind radii), and merges into `../HURDAT_full.csv`; then train the recipe text-free: `./typhoformer train 30 --patience=8 --motion --physics --cv --huber=0.1 --csv=../HURDAT_full.csv --emb=none --threads=N`. **The download attempt from this dev environment was blocked** (the network policy 403s NOAA on every pathway — proxy CONNECT and WebFetch — and no allowlisted host mirrors the data), so either run the script locally and commit `HURDAT_full.csv` to the repo, or allow `www.nhc.noaa.gov` in the environment's network settings (https://code.claude.com/docs/en/claude-code-on-the-web) and re-ask. Then re-run the FINDINGS grid — this resolves every "within noise" question in §7–§14.
- [ ] **Re-test the language branch on harder data.** The robust negative result (`--no_text` marginally *better*) may not hold where text could carry signal the numbers don't — rapid intensification, recurvature, extratropical transition (the open question of FINDINGS §15).

## Speed (from the 2026-07 optimization review)

- [x] **Unrolled reductions in the backward path and attention kernels.** Done — `linear_backward`'s bias reduction and all six MHA loops restructured to contiguous/multi-accumulator form; full-config forward+backward 39.6 → 16.6 ms/sample combined with the default change below.
- [x] **Make the no-spatial encoder the default.** Done — `--spatial` restores the paper architecture (and loads pre-change checkpoints); `--no_spatial` kept as a no-op; golden re-pinned; both paths still gradient-checked.
- [x] **Parallelize `evaluate()`.** Done — eval shards across the ParTrainer replicas (training validation passes and `eval --threads=N`); metrics identical to serial.
- [x] **Portable SIMD tier.** Done — `make MARCH=x86-64-v3`.
- [ ] **Batch samples into the GEMMs.** Everything runs per-sample at T=12 rows (low arithmetic intensity, weights re-streamed per sample). Stacking a minibatch into `[B·T, d]` for the encoder GEMMs is the biggest remaining speed lever — and the most invasive (per-sample decoder state; encoder-only batching is the pragmatic scope).

## Accuracy (from the 2026-07 optimization review)

- [x] **Physics features v2.** Done — `--physics` (acceleration, speed, heading, seasonal phase) beats the cv baseline on **all five splits** at 6h (39.59 → 39.10 km mean); small but sign-consistent (FINDINGS §13).
- [x] **Seed ensembling at eval.** Done — `eval --weights=a,b,c` with `--split=test` scoring; 3-member cv ensembles recover ~best-member skill on every split (~1% over the expected single model; FINDINGS §13).
- [x] **Loss shaping at long horizons.** Done for Huber + horizon weights — `--huber=0.1` collapses 48h split variance (±36 → ±14 km, mean −4.8%); `--hweight` is neutral (FINDINGS §13). The teacher-forcing-with-annealing test remains open (tracked below).
- [x] **Teacher forcing with annealing.** Done (`--tf=E`) — and neutral: 283.7 vs 285.0 km at 48h across five splits (FINDINGS §14). The cv anchor already stabilizes early rollouts.
- [x] **Motion-aligned frame for the cv correction.** Done (`--rotframe`, exact backward through ∂u/∂v, gradient-checked serial+parallel) — and honestly negative: worse on 4/5 splits at 6h, neutral at 48h (FINDINGS §13). Off by default.
- [x] **Text-free data path to scale storms.** Done — `tools/hurdat2_to_csv.py` converts raw NOAA HURDAT2 to the repo's CSV schema, and `--emb=none` trains without embedding chunks (language branch fed zeros == `--no_text`). The actual 20-year retrain still needs the HURDAT2 download (tracked in the research item above; NOAA is unreachable from this environment's network policy).

## Engineering

- [x] **Multicore support for the serial-only paths.** Done — the data-parallel workers now feed the per-sample aux inputs (seed velocity, co-active neighbours), so `--cv`/`--gru`/`--xattn`/`--co_spatial` all train with `--threads=N`; `test_parallel` pins serial/parallel gradient equivalence for each variant on the real dataset. Only `--km_loss` remains serial-only.
- [ ] **Run-verify the CUDA backend on a real GPU.** It compiles with `nvcc` (CI compile-checks it) but, unlike OpenCL (verified end-to-end via POCL), has never been executed against the kernel cross-check and gradient tests.
- [ ] **Build and test on Windows.** The data loader has a `FindFirstFile` branch written but only the POSIX (`scandir`) branch has ever been compiled.
- [ ] **Record the decoder variant in the checkpoint.** `eval`/`predict` auto-detect `--motion` from the checkpoint's feature count, but `--delta` (and the `--cv`/`--gru`/`--xattn` variants) must be re-specified by hand to match the checkpoint — store it in the TFW header instead.

## Conventions for this file

Check items off (`- [x]`) as they land, and record the outcome under `[Unreleased]` in `CHANGELOG.md`. Add new pending tasks here as they come up. For research items, FINDINGS.md is the record of results — including negative ones.
