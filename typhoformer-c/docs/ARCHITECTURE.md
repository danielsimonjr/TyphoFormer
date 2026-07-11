# Architecture & Backpropagation Math

This is the "how it works" reference. It walks through the whole model and gives
the **forward and backward equations for every layer**, matching the C source
line for line. If you understand this document you understand the entire
codebase.

> Notation (see [docs README](README.md)): `[R,C]` = row-major matrix; `T` =
> time steps; `D` = model dim; `H` = heads; `hd = D/H`; `⊙` = elementwise
> product; `σ(x) = 1/(1+e^-x)`; a code-name `dx` means `∂L/∂x` for the scalar
> loss `L`. Gradients into parameters **accumulate** (`+=`); gradients into
> activations are freshly written.

---

## 1. Problem and data shapes

We forecast the next position of a storm from a window of past observations.
One training sample is:

```
xnum   [T, 14]     numerical features per step (max wind, pressure, 12 wind radii)
xtext  [T, 384]    per-step MiniLM embedding of an LLM description of that step
yprev  [1, 2]      the last observed (lat, lon) — the decoder's seed
Y      [H_p, 2]    the future (lat, lon) target(s);  H_p = pred_len (=1 here)
```

`T = in_len = 12`, `D = d_model`, `H_p = pred_len = 1`. Everything is processed
**one sample at a time**; a mini-batch is a loop that sums gradients.

---

## 2. The module graph

```
 xnum ─┐
       ├─▶ PGF ──▶ x̃ [T,D] ──▶ ST-Encoder ──▶ h [1,D] ──▶ AR-Decoder ──▶ ŷ [H_p,2]
 xtext┘        └─▶ gate g [T,D] ─────────────────────────────────────────┐
                                                                          ▼
                                                              Loss = MSE(ŷ,Y) + λ·penalty(g)
```

Forward = left to right. Backward = right to left, plus a **second gradient path
into the gate** `g` coming straight from the loss penalty (not through the
encoder). That second path is the one subtlety in the whole model; keep it in
mind for §4.1 and §4.4.

Source map: `PGF`, `Encoder`, `Decoder`, `Model` live in `src/model.c`; the
generic layers (`Linear`, `LayerNorm`, `FFN`, `MHA`, `Block`) in `src/nn.c`; the
matrix kernels in `src/tensor.c`.

---

## 3. Building blocks

### 3.1 Linear — `linear_forward` / `linear_backward`

Forward (`y = x Wᵀ + b`), with `x:[T,in]`, `W:[out,in]`, `b:[out]`, `y:[T,out]`:

```
y[t,o] = Σ_i x[t,i]·W[o,i] + b[o]
```

Backward, given `dy:[T,out]` (we cached `x`):

```
dW[o,i] += Σ_t dy[t,o]·x[t,i]          # dW += dyᵀ x     [out,in]
db[o]   += Σ_t dy[t,o]                 # db += colsum(dy)
dx[t,i]  = Σ_o dy[t,o]·W[o,i]          # dx  = dy W       [T,in]
```

These three products are exactly `mat_matmul_atb`, `mat_colsum`, and
`mat_matmul`. Every learnable layer reduces to this.

### 3.2 ReLU — `mat_relu`

```
forward:  y = max(0, x)
backward: dx = dy ⊙ 1[x>0]
```

The mask `1[x>0]` is recovered from the cached pre-activation (see FFN).

### 3.3 LayerNorm — `layernorm_forward` / `layernorm_backward`

Per row (over the `D` features), with learnable `γ, β:[D]` and `ε=1e-5`:

```
μ    = (1/D) Σ_j x_j
v    = (1/D) Σ_j (x_j-μ)²
rstd = 1/√(v+ε)
x̂_j  = (x_j-μ)·rstd
y_j  = γ_j·x̂_j + β_j
```

Backward, given `dy` (cache `x̂` and `rstd`):

```
dγ_j += Σ_t dy[t,j]·x̂[t,j]
dβ_j += Σ_t dy[t,j]

# per row, let dx̂_j = dy_j·γ_j :
dx_j = rstd·( dx̂_j − mean_k(dx̂_k) − x̂_j·mean_k(dx̂_k·x̂_k) )
```

`mean_k` averages over the `D` features of that row. In code these two means are
`m1` and `m2`. The two subtracted terms are what make LayerNorm's gradient
zero-mean and de-correlated from `x̂` — that is the whole point of the layer.

### 3.4 Softmax (row-wise) — `softmax_rows` + attention backward

```
forward:  p_i = e^{s_i} / Σ_k e^{s_k}     (max-subtracted for stability)
backward: ds_i = p_i·( dp_i − Σ_k dp_k·p_k )
```

The bracket is the Jacobian-vector product `(diag(p) − p pᵀ)·dp`. Written this
way it costs `O(S)` per row, no `S×S` Jacobian.

### 3.5 Multi-head self-attention — `mha_forward` / `mha_backward`

Project `x:[S,D]` to `Q,K,V:[S,D]` with linears `q,k,v`. Split into `H` heads of
width `hd`. Per head `h` (slice `[:, h·hd : (h+1)·hd]`), with `scale = 1/√hd`:

```
scores = (Qh Khᵀ)·scale        [S,S]
P      = softmax_rows(scores)  [S,S]
Oh     = P Vh                  [S,hd]
```

Concatenate the `Oh` into `Ocat:[S,D]`, then `out = o(Ocat)` (linear). Backward,
given `dout`:

```
dOcat = o.backward(dout)                          # [S,D]

# per head (dOh = dOcat[:,head]):
dP[i,j]  = Σ_d dOh[i,d]·Vh[j,d]                    # dP  = dOh Vhᵀ
dVh[j,d] = Σ_i P[i,j]·dOh[i,d]                     # dVh = Pᵀ dOh
dscores  = softmax_backward(P, dP) · scale         # §3.4, then ×scale
dQh[i,d] = Σ_j dscores[i,j]·Kh[j,d]               # dQh = dscores Kh
dKh[j,d] = Σ_i dscores[i,j]·Qh[i,d]               # dKh = dscoresᵀ Qh

dx = q.backward(dQ) + k.backward(dK) + v.backward(dV)   # three paths share x
```

**Single-node mode (`self_only=1`).** When the sequence has length 1 (the
encoder's "spatial" attention over `num_nodes = 1`), `softmax` over one element
is always `1`, so `P = I`, `Oh = Vh`, and the output is just `o(v(x))`. The
gradient flows only through `v` and `o`; `Q,K` are dead (`dQ = dK = 0`). The code
takes this shortcut in `mha_forward`/`mha_backward` under `if (m->self_only)`.

### 3.6 Feed-forward — `ffn_forward` / `ffn_backward`

```
forward:  h = fc1(x);  a = relu(h);  y = fc2(a)     # D → F → D
backward: da = fc2.backward(dy)
          dh = da ⊙ 1[h>0]                          # cached h gives the mask
          dx = fc1.backward(dh)
```

### 3.7 Transformer block (post-norm) — `block_forward` / `block_backward`

```
a  = MHA(x)
r1 = x + a;    y1 = LN1(r1)         # attention sublayer + residual + norm
f  = FFN(y1)
r2 = y1 + f;   y  = LN2(r2)         # FFN sublayer + residual + norm
```

Backward — note how each residual **adds** a gradient path:

```
dr2 = LN2.backward(dy)
dy1 = dr2 + FFN.backward(dr2)        # residual path (dr2) + through the FFN
dr1 = LN1.backward(dy1)
dx  = dr1 + MHA.backward(dr1)        # residual path (dr1) + through attention
```

If you forget either `+ dr2` / `+ dr1` (the residual contributions), the
gradient check in `tests/test_nn.c` fails immediately — a good exercise.

---

## 4. The TyphoFormer-specific modules

### 4.1 Prompt-aware Gating Fusion — `pgf_forward` / `pgf_backward`

Learnable linears `W_g` (`fc_gate`), `W_x` (`proj_num`), `W_p` (`proj_text`).
Per time step, with `x = xnum[t]`, `p = xtext[t]`:

```
g   = σ( W_g·[x ; p] + b_g )        [D]     # gate in (0,1)
xn  = W_x·x                          [D]
xt  = W_p·p                          [D]
x̃   = g ⊙ xn + (1−g) ⊙ xt           [D]     # convex blend of the two modalities
```

The gate decides, per feature per step, how much to trust the numbers vs. the
language embedding. Backward is the subtle part, because `g` is used **twice** —
once inside `x̃` (reaches the loss through the encoder) and once directly in the
loss penalty (§4.4). Let `dx̃` be the gradient arriving from the encoder and
`dg_pen` the gradient arriving from the penalty:

```
dg  = dx̃ ⊙ (xn − xt) + dg_pen        # BOTH paths sum into the gate
dxn = dx̃ ⊙ g
dxt = dx̃ ⊙ (1−g)
dz  = dg ⊙ g ⊙ (1−g)                 # through the sigmoid, σ' = σ(1−σ)

fc_gate.backward(dz);  proj_num.backward(dxn);  proj_text.backward(dxt)
```

The inputs (`xnum`, `xtext`) are data, not parameters, so we don't propagate a
gradient into them — the linears just accumulate their `dW`/`db` (the code
passes a NULL `dx`).

### 4.2 Spatio-temporal encoder — `encoder_forward` / `encoder_backward`

```
e  = input_proj(x̃)                         # Linear D→D
for l in 0..N-1:  e = temporal_block[l](e)  # attention over the T time steps
for l in 0..N-1:  e = spatial_block[l](e)   # self_only attention (num_nodes=1)
h1 = TimeMix(e)                             # pool T steps → 1   (see §4.2.1)
h  = output_proj(h1)                        # Linear D→D  → context vector [1,D]
```

Temporal blocks mix information **across time**; with a single track the spatial
blocks reduce to a per-step transform (§3.5, `self_only`). The encoder uses two
ping-pong buffers (`b0/b1`, `db0/db1`) so no per-call allocation happens.
Backward runs the same list in reverse (`output_proj` → `TimeMix` → spatial →
temporal → `input_proj`).

#### 4.2.1 TimeMix — `timemix_forward` / `timemix_backward`

A learnable pooling over the time axis: `A:[1,T]`, `c:[1]`, `x:[T,D]`,
`y:[1,D]`.

```
forward:  y[0,d] = Σ_t A[0,t]·x[t,d] + c[0]

backward: dA += dy·xᵀ           # dA[0,t] += Σ_d dy[0,d]·x[t,d]
          dc += Σ_d dy[0,d]
          dx  = Aᵀ·dy           # dx[t,d] = A[0,t]·dy[0,d]
```

### 4.3 Autoregressive decoder — `decoder_forward` / `decoder_backward`

Two linears `fc1:(D+2)→D`, `fc2:D→2`. Seeded with `yprev`, it rolls out
`pred_len` steps; training uses one step:

```
z  = [h ; y_prev]        # concat context and previous coord → [1, D+2]
h1 = fc1(z)
a  = relu(h1)
ŷ  = fc2(a)              # predicted (lat, lon)
```

Backward (single step):

```
da   = fc2.backward(dŷ)
dh1  = da ⊙ 1[h1>0]
dz   = fc1.backward(dh1)
dh   = dz[0:D]           # gradient w.r.t. the encoder context (y_prev is data)
```

For multi-step rollout you would additionally propagate `dz[D:D+2]` back to the
previous step's output — see [EXTENDING.md](EXTENDING.md).

### 4.4 Loss — `model_loss`

```
MSE      = (1/N_o) Σ (ŷ − Y)²                    N_o = pred_len·2
penalty  = (1/N_g) Σ max(0, τ − g)²              N_g = T·D,  τ = 0.6
L        = MSE + λ·penalty                        λ = 0.1
```

The penalty pushes the gate up toward `τ`, discouraging it from collapsing to 0
(which would ignore the numerical branch). Gradients:

```
dŷ      = (2/N_o)·(ŷ − Y)                        # → decoder
dg_pen  = λ·(−2/N_g)·max(0, τ − g)               # → straight into the gate (§4.1)
```

`dg_pen` is exactly the "second path into `g`" from §2. It is `0` wherever
`g ≥ τ`.

### 4.5 Optimizer — Adam (`adam_step`)

For every scalar parameter `w` with gradient `g` (plus L2 weight decay `wd`):

```
g      ← g + wd·w
m      ← β1·m + (1−β1)·g            (β1 = 0.9)
v      ← β2·v + (1−β2)·g²           (β2 = 0.999)
m̂      = m / (1 − β1ᵗ)              # bias correction, t = step count
v̂      = v / (1 − β2ᵗ)
w      ← w − lr·m̂ / (√v̂ + ε)        (lr = 1e-3, ε = 1e-8)
```

Adam iterates the `ParamList`, so it is completely decoupled from the model
architecture — see [API.md](API.md) §ParamList.

---

## 5. One training step, end to end

```
1.  xnum,xtext,yprev,Y = dataset_get(sample)
2.  x̃, g   = PGF(xnum, xtext)          # forward
3.  h      = Encoder(x̃)
4.  ŷ      = Decoder(h, yprev)
5.  L, dŷ, dg_pen = model_loss(ŷ, Y, g)
6.  scale dŷ, dg_pen by 1/batch          # mini-batch mean
7.  dh      = Decoder.backward(dŷ)       # backward
8.  dx̃      = Encoder.backward(dh)
9.  _       = PGF.backward(dx̃, dg_pen)   # accumulates dW/db everywhere
10. (repeat 1–9 for the batch, summing grads)
11. adam_step()                          # update, then zero grads next batch
```

Steps 2–4 fill each module's caches; steps 7–9 consume them. Because gradients
accumulate with `+=`, the batch loop in step 10 simply sums — no special-casing.

---

## 6. Why the gradient checks are the ground truth

For a scalar loss `L` and any parameter `w`, the definition of the derivative
gives a numerical estimate independent of our backward code:

```
∂L/∂w  ≈  ( L(w+ε) − L(w−ε) ) / (2ε)        (central difference)
```

`tests/test_nn.c` and `tests/test_model.c` compute this for **every** parameter
and compare it to the analytic gradient the backward pass produced. Agreement to
the single-precision floating-point floor (abs error ~1e-4) means the backward
math above is implemented correctly. This is why you can extend the model with
confidence: **add a layer, then gradient-check it** — if it passes, your
derivation is right. See [EXTENDING.md](EXTENDING.md).
