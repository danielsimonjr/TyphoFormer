# MEMORY.md — durable project memory

Persistent context for anyone (human or AI agent) picking this repo up cold: the
load-bearing decisions, invariants, and hard-won lessons that are **not** obvious
from the code and easy to re-break. It complements the other docs, it does not
repeat them:

- **`AGENTS.md` / `CLAUDE.md`** — how to work here (guidance, commands, workflow).
- **`typhoformer-c/docs/FINDINGS.md`** — experimental results, including negatives.
- **`CHANGELOG.md`** — what changed, when.
- **`TODO.md`** — open and closed work.
- **This file** — *why* things are the way they are, and the traps that cost time.

Keep it terse and curated. A fact belongs here only if getting it wrong would
cost hours, and it isn't already stated somewhere better.

---

## Where things stand (2026-07-15)

- **The model works, and the science is settled.** On the full 826-storm dataset
  the recipe (`--motion --physics --cv --huber=0.1`) beats split-matched
  constant-velocity over the 48h rollout on all five splits (crossover ~24h,
  −8.4% at 48h); dead reckoning still wins the first few hours. The language
  branch does **not** help. Details: `FINDINGS.md` §15–16.
- **The real dataset is committed.** `HURDAT_full.csv` (24,177 records, 826 storms,
  2004–2025, three basins) plus the raw NOAA `.txt` sources, pinned **binary**
  (`.gitattributes`) so a NOAA re-release can't silently change them. `git` must
  not renormalize their line endings.
- **Three numerically-verified backends:** native CPU (gcc), CUDA, OpenCL. All
  agree op-for-op to 1e-07. None is faster than the CPU for this model — see below.
- **Open work is the roadmap (`TODO.md`, 2026-07-15):** 8 speed/accuracy items worked
  easy→hard. **Items 1–3 done:** (1) right-size — default unchanged (capacity invariant
  below); (2) register-blocked GEMM — refuted (below); (3) **SWA (`--swa`) — SHIPPED, the
  first accuracy win**: neutral at 6h, **−6 km at 48h (4/5 seeds)**, doubling the recipe's
  margin over CLIPER; opt-in, bit-neutral when off (FINDINGS §20). (4) km-aligned loss —
  reworked `--km_loss` into a proper both-paths gradient-checked equirectangular objective,
  but **measured NEUTRAL** (§21): the `--cv` anchor + short-range steps put km and MSE
  optima nearly together. Kept (off) as a correct flag. (5) **`--direct` multi-horizon head
  — the BIGGEST win and the first architecture change that helps (§22).** Predicts all
  horizons at once (no autoregressive rollout), so no error compounding: beats `--cv` at
  6h (32.1 vs 35.2 — **ties CLIPER**, erasing the "loses at 6h" caveat) AND 48h (−10.2 km
  6–48h mean, **5/5 seeds**, ~4.8σ), tighter variance, faster. Refuted my prediction that
  the AR evolving-velocity recurrence would win. **Strong candidate to replace `--cv` as
  the default decoder — ADR, needs Daniel's OK.** Next: uncertainty head (Item 6), then
  the data/physics levers (7–8, need external data). ⏭ does `--direct` stack with `--swa`?

---

## Load-bearing invariants (break these and the model silently rots)

- **RNG state is `uint64_t`, never `unsigned long`.** `unsigned long` is 64-bit on
  LP64 (Linux/macOS) and **32-bit on LLP64 (Windows)**. The xorshift64 RNG in
  32 bits truncates its seed and — via `nn_uniform`'s `>>11` then `/2^53` — collapses
  every draw to ~0, so every weight initializes to the low end of its range and the
  network is born symmetric. It compiles clean and trains to garbage. `test_golden`
  passing on Windows *is* the proof both platforms share one stream. (Fixed; do not
  reintroduce `unsigned long` for any RNG/hash/mask/offset.)
- **The storm-level split + train-only normalization is the anti-leakage barrier.**
  An earlier "beats persistence" result was leakage. `data.c` assigns whole *storms*
  to train/val/test and fits stats on train only. Preserve this in any data change.
  Report **held-out test** numbers, not validation.
- **Every forward has a hand-written backward proven by a finite-difference gradient
  check.** This is the repo's core invariant (`docs/EXTENDING.md`). `test_golden`
  pins the training loss bit-exactly — an intentional numerics change means updating
  the golden value; an *un*intentional change means you broke something.
- **Report five-seed means, never single-seed numbers.** The optimization is chaotic:
  a float-rounding-level perturbation moves single-seed results ~3% and flips the
  early-stopping epoch. This is why every `FINDINGS.md` table is a multi-seed mean.
- **Do NOT shrink the default (`d64L2`) on a 6h result.** Capacity is neutral at 6h —
  `d16L2` (7× fewer params, 8.66× faster) matches the recipe there and looks like a
  free win — but **load-bearing at 48h**: over the full rollout the smaller models
  regress toward the constant-velocity bar and give back the model's only advantage
  (`d32L2` *ties* CLIPER). A right-sizing study must measure the horizon the model is
  *for*. `d16L2` is a documented fast/short-horizon option, not a replacement.
  (`FINDINGS.md` §18.) *Sweep harness trap:* concurrent runs need a **unique `--save`**
  — the default checkpoint is shared and the held-out eval reloads it, so a race
  either aborts (different geometry) or silently scores the wrong split (same geometry).

---

## Backends — the one thing everyone gets wrong

The `tensor.h` **seam is only ~13 ops** (the GEMMs, relu, sigmoid, …). Most of the
model's arithmetic — layernorm, attention softmax, GRU gates, PGF gating, the whole
decoder rollout — runs in **plain host C reading `Mat.data[]` directly** (~189 sites),
*interleaved* with the seam ops. Consequences that repeatedly mislead people:

- **`Mat.data` must stay HOST-resident in every backend.** The CUDA backend once made
  it a `cudaMalloc`'d device pointer "so callers can't tell where the data lives" —
  they can, and the first host read segfaults. It compiled for years and had never
  run. Both CUDA and OpenCL now keep data host-resident and stage per op (documented
  in `tensor_opencl.c` / `tensor_cuda.cu`).
- **No GPU/OpenCL speedup is possible without a full model port.** Because the seam is
  narrow and the host math is interleaved, the data can't stay device-resident, so
  every op pays a host↔device round trip. Op compute ≈100 ns vs ≈20–50 µs overhead
  (200–500×); the GPU idles at ~7%. Beating the CPU needs porting the layer math into
  kernels **and** batching — a from-scratch GPU model, judged not worth it. Do **not**
  re-propose "just make `Mat.data` device-resident" as a contained fix; it isn't.
- **Sample-batching the GEMMs does NOT help either — measured.** `mat_matmul` is
  `ikj`-order and reads each weight row once *per output row*, so stacking B samples
  into one `[B·T, d]` GEMM has identical weight traffic (0.57–1.04× at recipe scale,
  ≤1.03× at `--full`). The `TODO.md` "biggest speed lever" was a false premise. (§17.)
- **The register-blocked microkernel is ALSO refuted for the recipe — measured (§19).**
  MR=4 blocking of `mat_matmul` is bit-exact (golden unchanged) and 1.55× in an
  *isolated* microbench, but **in-situ it regresses the recipe 16%** (the bigger blocked
  body costs i-cache/cold-data that the tight loop hides; `mat_matmul` itself 6.5%
  slower in place). It helps only `--full` (capacity-overkill config, 1.15×); a size
  guard needs a machine-specific `k·n` threshold that doesn't belong in a portable
  kernel. **Do not re-derive it from the microbench and ship it — the isolated number
  lies.** Both "obvious" GEMM speed levers are now measured dead ends for the recipe.
- **A checkpoint records its decoder mode (TFW4).** The parameter-size guard catches
  `--gru`/`--xattn`/`--posenc` (they change tensor shapes) but is **blind** to
  `--cv`/`--delta`/`--rotframe` (parameter-neutral, they only change what the output
  *means*). Before TFW4, evaluating a cv checkpoint without `--cv` loaded cleanly and
  reported 5071 km where the truth was 29 km — silent, no error. `eval`/`predict` now
  self-configure from the header; pre-TFW4 checkpoints warn loudly.

---

## Build / CI / environment traps

- **CI only *compiles* the CUDA backend** (GitHub runners have no GPU). The executing
  gate is `make CUDA=1 test-cuda`, which must be run on a real NVIDIA GPU by hand. A
  green CI does not mean the GPU path runs. OpenCL *is* executed in CI via POCL (CPU).
- **CI is path-filtered to `typhoformer-c/**`.** Changes to root `TODO.md` / `MEMORY.md`
  don't trigger it — that's expected, not a broken pipeline.
- **`test-san` leaves ASan/UBSan-instrumented objects behind.** Any later plain `make`
  that doesn't `make clean` first relinks them without `-fsanitize` and dies on
  undefined `__asan_*` symbols. This kept `build-test` red on `main` for days once.
- **OpenCL on the GPU is not available under WSL2 here** — NVIDIA ships CUDA but not
  the OpenCL ICD in WSL. OpenCL runs via **POCL on the CPU**. (Convenient: POCL-on-CPU
  vs native-CPU is the *same hardware*, which is what proved the backend divergence is
  implementation, not hardware.)
- **On Windows, `unsigned long` is 32-bit** (see the RNG invariant). The data loader's
  `FindFirstFile` branch is the Windows path; build with MSYS2 mingw-w64.

---

## The meta-lesson, earned repeatedly this project

**"It compiles" is not "it runs"; passing tests is not the pipeline working; and a
green single number is not a reproduced result.** Every serious bug here — the
never-run CUDA backend, the 32-bit Windows RNG, the silent 170× eval mismatch, the
CI that was red for days — sat behind a green-looking signal. The fix is always the
same: exercise the real artifact end-to-end and read what it actually did. And when
a tempting optimization is "obviously" worth it (GPU speedup, GEMM batching),
**measure the premise first** — twice this project, minutes of measurement saved days
of building the wrong thing.
