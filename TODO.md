# TODO

Open work, sourced from the docs (chiefly `typhoformer-c/docs/FINDINGS.md`) and the current state of the code. The completed history lives in `CHANGELOG.md` and the "Status — complete" section of `typhoformer-c/README.md`.

## Research (the "honest next steps" from FINDINGS.md)

- [ ] **Confirm the long-horizon decoder gains.** `--gru` / `--xattn` beat the constant-velocity decoder at 24h on the favourable split but are split-dependent (FINDINGS §10 calls them "promising, not established"). Repeat across more splits and longer training runs.
- [ ] **Scale past 98 storms — the ceiling on everything.** The repo ships 2020–2024 only; the paper uses 20+ years of HURDAT2. Regenerate GPT-4o descriptions (`tools/gen_descriptions.py`, needs `OPENAI_API_KEY`) and MiniLM embeddings (`tools/gen_embeddings.py`) for a longer record span, then re-run the FINDINGS experiment grid.
- [ ] **Re-test the language branch on harder data.** The robust negative result (`--no_text` marginally *better*) may not hold where text could carry signal the numbers don't — rapid intensification, recurvature, extratropical transition (the open question of FINDINGS §11).
- [ ] **Push horizons past 24h.** Longer horizons are where the learned curvature term should finally pass constant-velocity (FINDINGS §9–§10); `--pred_len` already supports it, the experiments haven't been run.

## Engineering

- [ ] **Multicore support for the serial-only paths.** `--cv`, `--gru`, `--xattn`, and `--co_spatial` require `--threads=1`; extend `parallel.c` replica setup to cover the decoder variants and co-active spatial attention.
- [ ] **Run-verify the CUDA backend on a real GPU.** It compiles with `nvcc` (CI compile-checks it) but, unlike OpenCL (verified end-to-end via POCL), has never been executed against the kernel cross-check and gradient tests.
- [ ] **Build and test on Windows.** The data loader has a `FindFirstFile` branch written but only the POSIX (`scandir`) branch has ever been compiled.
- [ ] **Record the decoder variant in the checkpoint.** `eval`/`predict` auto-detect `--motion` from the checkpoint's feature count, but `--delta` (and the `--cv`/`--gru`/`--xattn` variants) must be re-specified by hand to match the checkpoint — store it in the TFW header instead.

## Conventions for this file

Check items off (`- [x]`) as they land, and record the outcome under `[Unreleased]` in `CHANGELOG.md`. Add new pending tasks here as they come up. For research items, FINDINGS.md is the record of results — including negative ones.
