/*
 * nn.h — neural-network building blocks (forward + backward), standard
 * library only. Each parametric module registers its weights with a shared
 * ParamList so the optimizer and the gradient-check harness can treat all
 * parameters uniformly.
 */
#ifndef TYPHOFORMER_NN_H
#define TYPHOFORMER_NN_H

#include "tensor.h"

/* ---- reproducible RNG + init ---------------------------------------- */
void  nn_seed(unsigned long s);
float nn_uniform(float lo, float hi);

/* ---- dropout mode (train vs eval) ----------------------------------- */
/* Dropout is active only when training is on AND the rate is > 0. Eval /
 * inference / gradient-checks leave training off (the default), so dropout is
 * the identity and everything stays deterministic. The dropout RNG is
 * thread-local (seed it per worker) so multicore training is race-free. */
void nn_set_training(int on);
void nn_set_dropout(float p);
void nn_dropout_seed(unsigned long s);
unsigned long nn_dropout_state(void);   /* current thread's dropout RNG (to thread it across workers) */
int  nn_training(void);

/* Pre-norm transformer blocks (LN before each sublayer) instead of the default
 * post-norm. Off by default. Read at block_forward/backward time. */
void nn_set_prenorm(int on);

/* ALiBi-style relative-distance bias on the (temporal) attention scores:
 * score[i,j] -= slope_h·|i−j|, giving attention an intrinsic sense of temporal
 * distance (no new parameters). Applied only to full self-attention (the
 * temporal blocks), not the single-node spatial path. Off by default. */
void nn_set_timebias(int on);

/* ---- parameter registry --------------------------------------------- */
/* A Param is a flat view onto one weight tensor and its gradient buffer:
 *   v    -> the parameter values   (length n floats, owned by the module)
 *   g    -> the matching gradient  (length n floats, same layout as v)
 *   n    -> element count (so W[out,in] registers with n = out*in)
 *   name -> label for debugging / grad-check reporting (not owned)
 * Registering (v,g) as a pair lets the optimizer step v -= lr*g and lets the
 * finite-difference checker perturb v[e] and read back g[e] without knowing
 * which module produced it. */
typedef struct { float *v; float *g; int n; const char *name; } Param;
/* Dynamic array of Params. item is a grow-by-doubling buffer of `count`
 * live entries with `cap` slots allocated. The list does NOT own the float
 * buffers it points at — each module frees its own weights/grads. */
typedef struct { Param *item; int count, cap; } ParamList;

void plist_init(ParamList *pl);
void plist_add(ParamList *pl, float *v, float *g, int n, const char *name);
void plist_zero_grad(ParamList *pl);
long plist_num_params(const ParamList *pl);
void plist_free(ParamList *pl);
/* Global-norm gradient clipping. Returns the pre-clip L2 norm of all gradients;
 * if it exceeds max_norm (and max_norm > 0), scales every gradient by
 * max_norm/norm in place. */
float plist_clip_grad_norm(ParamList *pl, float max_norm);

/* ---- Linear:  y[T,out] = x[T,in] W^T + b ---------------------------- */
/* Affine map applied row-wise to a batch of T tokens. W is stored [out,in]
 * (row i is the weight vector for output unit i), so the forward uses W^T
 * implicitly: y[t,i] = Σ_j x[t,j]·W[i,j] + b[i]. */
typedef struct {
    int   in, out;     /* input / output feature widths                    */
    Mat   W, dW;       /* [out,in] parameter matrix and its gradient       */
    float *b, *db;     /* [out]    bias vector and its gradient            */
    Mat   xcache;      /* [T,in] forward input, cached for the backward   */
} Linear;

Linear linear_new(int in, int out, ParamList *pl, const char *name);
void   linear_forward(Linear *l, const Mat x, Mat y);        /* caches x */
void   linear_backward(Linear *l, const Mat dy, Mat dx);     /* += grads; dx may be NULL */
void   linear_free(Linear *l);

/* ---- LayerNorm over the last dimension ------------------------------ */
/* Normalizes each row to zero mean / unit variance, then applies a learned
 * per-feature affine:  y[t,j] = g[j]·xhat[t,j] + b[j], where
 *   xhat[t,j] = (x[t,j] - mean_t) / sqrt(var_t + eps).
 * The backward needs xhat and the reciprocal std per row, so both are cached. */
typedef struct {
    int   dim;                /* feature width D that is normalized          */
    float *g, *dg, *b, *db;   /* [dim] scale/shift params and their grads   */
    Mat   xhat;               /* [T,dim] cached normalized activations       */
    float *rstd;              /* [T] cached 1/sqrt(var+eps) per row          */
    int   Tcap;               /* allocated length of rstd (grows on demand) */
} LayerNorm;

LayerNorm layernorm_new(int dim, ParamList *pl, const char *name);
void      layernorm_forward(LayerNorm *ln, const Mat x, Mat y);
void      layernorm_backward(LayerNorm *ln, const Mat dy, Mat dx);
void      layernorm_free(LayerNorm *ln);

/* ---- FFN:  Linear(D->F) -> ReLU -> Linear(F->D) --------------------- */
/* Position-wise feed-forward network: y = fc2(ReLU(fc1(x))). The hidden
 * width F is usually a few times D. Both the pre-activation h = fc1(x) and
 * the post-ReLU a are cached because the ReLU backward needs the sign of h. */
typedef struct {
    Linear fc1, fc2;      /* D->F expansion, then F->D projection            */
    int    d, f;          /* model width D and hidden width F                */
    Mat    h, a;          /* h = fc1(x) pre-act, a = ReLU(h); both [T,F]    */
    Mat    s_da, s_dh;    /* backward scratch: dA and dH, both [T,F]        */
} FFN;

FFN  ffn_new(int d, int f, ParamList *pl, const char *name);
void ffn_forward(FFN *ff, const Mat x, Mat y);
void ffn_backward(FFN *ff, const Mat dy, Mat dx);
void ffn_free(FFN *ff);

/* ---- GRU cell:  h_t = GRU(x_t, h_{t-1}) ----------------------------- */
/* Gated recurrent unit, built from three Linear gates (reset/update/candidate):
 *   r = σ(Lr[x;h]),  u = σ(Lu[x;h]),  n = tanh(Ln[x; r⊙h]),  h' = (1-u)⊙n + u⊙h
 * Per-step forward state is cached in arrays sized to max_steps so the
 * autoregressive rollout can back-propagate through the whole sequence. */
typedef struct {
    int    in, hid, max_steps;               /* input dim I, hidden dim H, rollout length */
    Linear lr, lu, ln;                       /* reset, update, candidate gates */
    /* Per-step forward caches, one row [1,·] per timestep (length max_steps):
     *   zruc[t] = [x_t ; h_{t-1}]   input to reset+update gates   [1, I+H]
     *   znc[t]  = [x_t ; r_t⊙h_{t-1}] input to candidate gate     [1, I+H]
     *   rc[t]   = reset gate r_t     [1,H]
     *   uc[t]   = update gate u_t    [1,H]
     *   nc[t]   = candidate n_t      [1,H]
     *   hpc[t]  = h_{t-1} snapshot   [1,H]  (needed by the update-gate backward) */
    Mat   *zruc, *znc, *rc, *uc, *nc, *hpc;  /* per-step caches [1,·]          */
    Mat    s_dzru, s_dzn, s_dg;              /* backward scratch: dzru/dzn [1,I+H], dg [1,H] */
} GRU;
GRU  gru_new(int in, int hid, int max_steps, ParamList *pl, const char *name);
void gru_forward(GRU *g, int step, const Mat x, const Mat hprev, Mat hout);
/* Accumulates gate grads; writes dx and dhprev (either may be a NULL matrix). */
void gru_backward(GRU *g, int step, const Mat dh, Mat dx, Mat dhprev);
void gru_free(GRU *g);

/* ---- Single-head cross-attention (query -> fixed memory) ------------ */
/* Scaled dot-product attention of a per-step query over a FIXED memory sequence
 * (e.g. the encoder's per-timestep states). K/V are projected from the memory
 * once (shared across steps); each step supplies its own query. Per-step caches
 * let an autoregressive rollout back-propagate through every step, and the K/V
 * gradients accumulate across steps then flow back to the memory in one pass. */
typedef struct {
    int    qin, d, max_steps, T;        /* query in-dim, model dim d, rollout len, memory len T */
    Linear lq, lk, lv, lo;              /* query / key / value / output proj */
    Mat    K, V, memc;                  /* K=lk(mem), V=lv(mem), memc=mem copy; all [T,d] */
    /* Per-step forward caches (length max_steps), one row per decode step:
     *   qc[t]  = projected query q_t   [1,d]
     *   ac[t]  = attention weights a_t [1,T] (softmax over the T memory keys)
     *   cxc[t] = context Σ_j a_{t,j}V_j [1,d]
     *   xc[t]  = raw step input x_t    [1,qin] (lq input, cached for backward) */
    Mat   *qc, *ac, *cxc, *xc;          /* per-step: q[1,d] a[1,T] ctx[1,d] x[1,qin] */
    /* Backward scratch. s_dK/s_dV are ACCUMULATORS over all steps (zeroed in
     * xattn_set_memory) that flow to the memory once in xattn_backward_memory;
     * the rest are transient per-step temporaries. */
    Mat    s_dctx, s_da, s_dsc, s_dq, s_dK, s_dV;  /* backward scratch        */
} CrossAttn;
CrossAttn xattn_new(int qin, int d, int max_steps, ParamList *pl, const char *name);
void xattn_set_memory(CrossAttn *a, const Mat mem);          /* project K,V (T = mem.rows) */
void xattn_forward(CrossAttn *a, int step, const Mat x, Mat out);      /* out[1,d] */
void xattn_backward_step(CrossAttn *a, int step, const Mat dout, Mat dx);  /* dx[1,qin]; accum dK,dV */
void xattn_backward_memory(CrossAttn *a, Mat dmem);          /* dK,dV -> dmem[T,d] (after all steps) */
void xattn_free(CrossAttn *a);

/* ---- Multi-head self-attention over a sequence [S,D] ---------------- */
typedef struct {
    int    d_model, n_heads, head_dim;   /* D, H, and hd = D/H (D must be divisible by H) */
    int    self_only;       /* 1 => each position attends only to itself
                             *      (spatial attention with a single node) */
    Linear q, k, v, o;      /* per-token Q/K/V projections and the output proj, all D->D */
    Mat    Q, K, V, Ocat;   /* [S,D] caches: projected queries/keys/values and the
                             * concatenated per-head context (input to o) */
    Mat   *P;               /* n_heads x [S,S] attention weights (rows sum to 1) */
    int    Scap;            /* tracks allocated S for P (unused past ensure()) */
    Mat    s_dOcat, s_dQ, s_dK, s_dV, s_dxq, s_dxk;  /* backward scratch [S,D] */
    Mat    s_dP, s_dsc;                              /* backward scratch [S,S]: dP (post-softmax) and dscores (pre-softmax) */
} MHA;

MHA  mha_new(int d_model, int n_heads, int self_only, ParamList *pl, const char *name);
void mha_forward(MHA *m, const Mat x, Mat y);        /* self-attn over rows of x */
void mha_backward(MHA *m, const Mat dy, Mat dx);
void mha_free(MHA *m);

/* ---- Transformer block: MHA + res + LN -> FFN + res + LN (post-norm) */
typedef struct {
    MHA       attn;         /* self-attention sublayer                         */
    LayerNorm ln1, ln2;     /* the two layer-norms (post-attn and post-FFN)   */
    FFN       ff;           /* feed-forward sublayer                           */
    int       d;            /* model width D                                   */
    /* Forward caches [S,D]. Post-norm meaning:
     *   attn_out = MHA(x) (post-dropout), r1 = x + attn_out, y1 = LN1(r1),
     *   ff_out   = FFN(y1) (post-dropout), r2 = y1 + ff_out, y = LN2(r2). */
    Mat       attn_out, r1, y1, ff_out, r2;   /* caches */
    Mat       drop1, drop2;                    /* dropout keep-masks (attn / ffn), [S,D] */
    Mat       s_dr2, s_dy1, s_dr1, s_dtmp, s_dd; /* backward scratch [S,D] */
} Block;

Block block_new(int d_model, int n_heads, int ff_dim, int self_only, ParamList *pl, const char *name);
void  block_forward(Block *b, const Mat x, Mat y);
void  block_backward(Block *b, const Mat dy, Mat dx);
void  block_free(Block *b);

#endif /* TYPHOFORMER_NN_H */
