# TODO

Open work, sourced from the docs (chiefly `typhoformer-c/docs/FINDINGS.md`) and the current state of the code. The completed history lives in `CHANGELOG.md` and the "Status — complete" section of `typhoformer-c/README.md`.

## Research (the "honest next steps" from FINDINGS.md)

- [x] **Confirm the long-horizon decoder gains.** Done — they did *not* confirm: across five splits at 24h and 48h, `--gru`/`--xattn` are at parity-or-worse vs plain `--cv`, and the §10 headline flipped under a summation-order numerics perturbation (FINDINGS §11).
- [x] **Push horizons past 24h.** Done — at `--pred_len=8` (48h) the plain `--cv` decoder beats split-matched constant-velocity on 4/5 splits (−9.3% at 48h), the predicted long-horizon win (FINDINGS §11).
- [ ] **Scale past 98 storms — the ceiling on everything.** The repo ships 2020–2024 only; the paper uses 20+ years of HURDAT2. Regenerate GPT-4o descriptions (`tools/gen_descriptions.py`, needs `OPENAI_API_KEY`) and MiniLM embeddings (`tools/gen_embeddings.py`) for a longer record span, then re-run the FINDINGS experiment grid.
- [ ] **Re-test the language branch on harder data.** The robust negative result (`--no_text` marginally *better*) may not hold where text could carry signal the numbers don't — rapid intensification, recurvature, extratropical transition (the open question of FINDINGS §13).

## Engineering

- [x] **Multicore support for the serial-only paths.** Done — the data-parallel workers now feed the per-sample aux inputs (seed velocity, co-active neighbours), so `--cv`/`--gru`/`--xattn`/`--co_spatial` all train with `--threads=N`; `test_parallel` pins serial/parallel gradient equivalence for each variant on the real dataset. Only `--km_loss` remains serial-only.
- [ ] **Run-verify the CUDA backend on a real GPU.** It compiles with `nvcc` (CI compile-checks it) but, unlike OpenCL (verified end-to-end via POCL), has never been executed against the kernel cross-check and gradient tests.
- [ ] **Build and test on Windows.** The data loader has a `FindFirstFile` branch written but only the POSIX (`scandir`) branch has ever been compiled.
- [ ] **Extend the unrolled-reduction treatment to the attention kernels.** `mat_matmul_bt` got the 8-accumulator dot product (5.6× forward); the MHA per-head score/context loops and the xattn step loops still use single-accumulator reductions. Small share of runtime at `in_len=12`, but worth it for long input windows.
- [ ] **Record the decoder variant in the checkpoint.** `eval`/`predict` auto-detect `--motion` from the checkpoint's feature count, but `--delta` (and the `--cv`/`--gru`/`--xattn` variants) must be re-specified by hand to match the checkpoint — store it in the TFW header instead.

## Conventions for this file

Check items off (`- [x]`) as they land, and record the outcome under `[Unreleased]` in `CHANGELOG.md`. Add new pending tasks here as they come up. For research items, FINDINGS.md is the record of results — including negative ones.
