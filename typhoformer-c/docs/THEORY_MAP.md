# Theory ↔ Code Map

A single table that connects each mathematical object in the model to the exact
place it lives in the source. Use it as an index: pick a concept, jump to the
equation in [ARCHITECTURE.md](ARCHITECTURE.md), then open the `file:function`
and read the implementation. Line numbers drift as the code changes — search by
function name, which is stable.

Notation: `[R,C]` = row-major matrix; `T` = input steps; `D` = model dim;
`H` = heads; `hd = D/H`; `P` = prediction steps.

## Data flow (one sample), end to end

```
xnum[T,d_num], xtext[T,d_text]
      │  Prompt-aware Gating Fusion
      ▼
x̃[T,D]  ──►  temporal blocks ──►  spatial blocks ──►  TimeMix pool ──►  h_enc[1,D]
      │            (attention over time)   (per-node)   (T→1)
      ▼
h_enc, y_prev ──►  autoregressive decoder (P steps) ──►  ŷ[P,2]
      │
      ▼
loss = MSE(ŷ, Y) + λ·gate-penalty
```

## The map

| Concept / equation | Math (see ARCHITECTURE.md) | Forward | Backward |
|:--|:--|:--|:--|
| **Dense matrix, GEMM** | `C = AB`, `ABᵀ`, `AᵀB` | `mat_matmul*` in `src/tensor.c` | (same file) |
| **Linear layer** | `y = xWᵀ + b` | `linear_forward` `src/nn.c` | `linear_backward` |
| **LayerNorm** | `ŷ = γ·(x−μ)/σ + β` | `layernorm_forward` `src/nn.c` | `layernorm_backward` |
| **Softmax (attention rows)** | `softmax(z)_i = e^{z_i}/Σe^{z_j}` | `softmax_rows` (static) `src/nn.c` | folded into `mha_backward` |
| **Scaled dot-product attention** | `A = softmax(QKᵀ/√hd)`, `O = AV` | `mha_forward` `src/nn.c` | `mha_backward` |
| **Multi-head split/merge** | per-head `Q,K,V`; concat then `Wₒ` | `mha_forward` | `mha_backward` |
| **Single-node ("self-only") spatial attention** | `A = I` (each position attends to itself) | `mha_forward` (`self_only=1`) | `mha_backward` |
| **Feed-forward network** | `Linear→ReLU→Linear` | `ffn_forward` `src/nn.c` | `ffn_backward` |
| **Transformer block (post-norm)** | `x→+Dropout(MHA)→LN→+Dropout(FFN)→LN` | `block_forward` `src/nn.c` | `block_backward` |
| **Dropout** | train: `y=mask⊙x`, `mask∈{0,1/(1−p)}`; eval: identity | `dropout_apply` `src/nn.c` | folded into `block_backward` |
| **Prompt-aware Gating Fusion** | `g = σ(W_g[xₙ;xₜ])`, `x̃ = g⊙xₙ′ + (1−g)⊙xₜ′` | `pgf_forward` `src/model.c` | `pgf_backward` |
| **TimeMix pooling** | `y = A·x + c`, `[out,in]·[in,D]` | `timemix_forward` `src/model.c` | `timemix_backward` |
| **Spatio-temporal encoder** | stack temporal+spatial blocks, pool T→1 | `encoder_forward` `src/model.c` | `encoder_backward` |
| **Autoregressive decoder** | `ŷ_s = f(h_enc, ŷ_{s−1})`, feeding own output | `decoder_forward` `src/model.c` | `decoder_backward` |
| **Displacement head** (delta, `--delta`) | `ŷ_s = ŷ_{s−1} + Δ`, `fc2` zero-init (start at persistence) | `decoder_forward` (`model_delta`) | `decoder_backward` (identity term) |
| **Motion features** (`--motion`) | append `lat, lon, Δlat, Δlon` to the input | `dataset_add_motion` `src/data.c` | — |
| **Full model** | compose PGF→encoder→decoder | `model_forward` `src/model.c` | `model_backward` |
| **Loss** | `MSE(ŷ,Y) + λ·mean(relu(0.6−g)²)` | `model_loss` `src/model.c` | (returns `dpred`, `dgate`) |
| **AdamW (decoupled decay)** | `m,v` moments; bias-corrected step; `w−=lr·wd·w` | `adam_step` `src/optim.c` | — |
| **Gradient clipping** | scale grads if `‖g‖₂ > clip` | `plist_clip_grad_norm` `src/nn.c` | — |
| **Parameter registry** | flat list bridging optimizer + I/O | `plist_*` `src/nn.c` | — |
| **Finite-difference gradient check** | `∂L/∂w ≈ (L(w+ε)−L(w−ε))/2ε` | `tests/test_*.c` | (ground truth for all above) |

## Extension seams (where students/engineers plug in)

| Seam | Interface | Reference implementation |
|:--|:--|:--|
| **New layer/block** as a pluggable module | `Module` vtable, `Sequential` | `include/module.h`, `src/module.c`, `tests/test_module.c` |
| **New compute device** (GPU, accelerator) | the ~13 kernels in `include/tensor.h` | CPU: `src/tensor.c`; runnable OpenCL: `backends/opencl/tensor_opencl.c`; CUDA: `backends/cuda/tensor_cuda.cu` (see `backends/README.md`) |
| **Multicore data-parallel training** | replicate + broadcast + reduce | `include/parallel.h`, `src/parallel.c`, `tests/test_parallel.c` |
| **New optimizer** | consume a `ParamList` | `src/optim.c` (Adam) |
| **New embedding/description model** | precompute → `.npy` the C loader reads | `tools/gen_*.py`, `src/data.c` |

## Reading order for a full derivation

1. `mat_matmul*` — the only place real arithmetic happens.
2. `linear_forward`/`linear_backward` — the three GEMMs of a dense layer.
3. `layernorm_backward` and `softmax_rows` — the two non-obvious Jacobians.
4. `mha_backward` — attention assembled from the pieces above.
5. `block_backward` — residual + norm plumbing.
6. `pgf_backward`, `encoder_backward`, `decoder_backward` — the model-specific glue.
7. `model_loss` → `model_backward` — how the scalar loss seeds the whole reverse pass.

Every one of these is checked against finite differences in `tests/` — if you
change a forward pass, the matching gradient check is your proof of correctness.
