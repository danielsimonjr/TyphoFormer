/*
 * nn.c — implementation of the neural-network building blocks.
 */
#include "nn.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- RNG ------------------------------------------------------------- */
/* Marsaglia xorshift64. A single 64-bit state gives a fast, fully
 * deterministic stream — essential for the byte-stable golden regression:
 * the same seed reproduces the exact same weight initialization every run.
 * The initial state is an arbitrary nonzero constant (0 is a fixed point of
 * xorshift and must never be used, hence nn_seed maps 0 -> 1).
 *
 * The state MUST be uint64_t, never `unsigned long`: that type is 64-bit on
 * LP64 (Linux/macOS) but 32-bit on LLP64 (Windows). At 32 bits the seed
 * truncates, (13,7,17) is not a full-period triple, and — fatally —
 * nn_uniform's `>> 11` then divide by 2^53 leaves ~21 significant bits over a
 * 2^53 denominator, so EVERY draw collapses to ~0. Weight init returns the low
 * end of the range for every parameter, the network starts perfectly
 * symmetric, and the finite-difference gradient check fails. It compiles and
 * runs; the model is silently garbage. */
static uint64_t g_rng = 88172645463325252ULL;
void nn_seed(uint64_t s) { g_rng = s ? s : 1ULL; }
static uint64_t xorshift(void) {
    /* Three shift-xor rounds; the (13,7,17) triple is a known full-period set. */
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}
float nn_uniform(float lo, float hi) {
    /* Take the top 53 bits (>>11) and divide by 2^53 to get a uniform double
     * in [0,1), then affine-map into [lo,hi). Using the high bits avoids the
     * weaker low-order bits of xorshift. */
    double u = (double)(xorshift() >> 11) / 9007199254740992.0; /* [0,1) */
    return (float)(lo + (hi - lo) * u);
}

/* ---- dropout mode + thread-local dropout RNG ------------------------ */
/* Dropout uses a SEPARATE rng from weight init so that toggling training on/off
 * does not perturb the init stream, and so each worker thread can own its own
 * mask sequence. g_drng is _Thread_local: seed it per worker (nn_dropout_seed)
 * and hand off state across steps with nn_dropout_state — no locks, no races. */
static int   g_training = 0;      /* master train/eval switch (eval => no dropout) */
static float g_dropout  = 0.1f;   /* drop probability p                            */
static _Thread_local uint64_t g_drng = 88172645463325252ULL;
void nn_set_training(int on) { g_training = on; }
void nn_set_dropout(float p) { g_dropout = p; }
void nn_dropout_seed(uint64_t s) { g_drng = s ? s : 1ULL; }
uint64_t nn_dropout_state(void) { return g_drng; }
int  nn_training(void) { return g_training; }
static float drop_uniform(void) {
    /* Same xorshift64 as the init RNG but on the thread-local state. */
    g_drng ^= g_drng << 13; g_drng ^= g_drng >> 7; g_drng ^= g_drng << 17;
    return (float)((double)(g_drng >> 11) / 9007199254740992.0);
}
/* Inverted dropout: each element is kept with prob (1-p) and rescaled by
 * 1/(1-p), or zeroed with prob p. The rescale keeps the expected activation
 * unchanged, so eval (which skips dropout entirely) needs no compensation.
 * The keep/scale factor is stored in `mask` so the backward can multiply dy by
 * the exact same per-element factor. Applies in place to m. */
static void dropout_apply(Mat m, Mat mask, float p) {
    float scale = 1.0f / (1.0f - p);
    int n = m.rows * m.cols;
    for (int i = 0; i < n; ++i) {
        float keep = (drop_uniform() >= p) ? scale : 0.0f;
        mask.data[i] = keep;
        m.data[i] *= keep;
    }
}
/* Dropout is active only while training AND p>0; otherwise it is the identity. */
static int dropout_on(void) { return g_training && g_dropout > 0.0f; }
/* Public draw from the thread-local training RNG (shared with the dropout mask
 * stream — one decorrelated stochastic stream per worker). Used by the decoder
 * for per-step teacher-forcing decisions. */
float nn_dropout_draw(void) { return drop_uniform(); }

/* ---- pre-norm vs post-norm block ------------------------------------ */
static int g_prenorm = 0;
void nn_set_prenorm(int on) { g_prenorm = on; }

/* ---- deferred weight-gradient accumulation --------------------------- */
/* See nn.h. Plain global read during forward/backward; the trainer sets it
 * once before training starts (before any worker threads run), so there is
 * no race. Default off keeps every gradient-check test and the golden loop
 * on the immediate (original) path. */
static int g_defer = 0;
void nn_set_defer_grads(int on) { g_defer = on; }
int  nn_defer_grads(void) { return g_defer; }

/* ---- ALiBi temporal-distance bias ----------------------------------- */
static int g_timebias = 0;
void nn_set_timebias(int on) { g_timebias = on; }
/* Standard ALiBi geometric slope for head h of H:  slope_h = 2^(-8(h+1)/H).
 * Heads get a geometric ladder of slopes from 2^(-8/H) down to 2^-8, so
 * different heads penalize temporal distance at different rates. The bias
 * -slope_h·|i-j| is added to the raw scores (see mha_forward); it introduces
 * no learnable parameters — the recency prior is fixed. */
static float alibi_slope(int h, int H) {
    return (float)pow(2.0, -8.0 * (double)(h + 1) / (double)H);
}

/* ---- ParamList ------------------------------------------------------- */
void plist_init(ParamList *pl) { pl->item = NULL; pl->count = pl->cap = 0; }
void plist_add(ParamList *pl, float *v, float *g, int n, const char *name) {
    if (pl->count == pl->cap) {
        pl->cap = pl->cap ? pl->cap * 2 : 16;
        pl->item = (Param *)realloc(pl->item, (size_t)pl->cap * sizeof(Param));
        assert(pl->item);
    }
    pl->item[pl->count++] = (Param){ v, g, n, name };
}
/* Reset every registered gradient buffer to 0. Because all module backwards
 * ACCUMULATE (+=) into these buffers, this must be called once before each
 * fresh forward/backward pass or grads from prior passes would leak in. */
void plist_zero_grad(ParamList *pl) {
    for (int i = 0; i < pl->count; ++i)
        memset(pl->item[i].g, 0, (size_t)pl->item[i].n * sizeof(float));
}
long plist_num_params(const ParamList *pl) {
    long t = 0; for (int i = 0; i < pl->count; ++i) t += pl->item[i].n; return t;
}
void plist_free(ParamList *pl) { free(pl->item); plist_init(pl); }
/* Global-norm gradient clipping. Computes the L2 norm over ALL parameters
 * concatenated:  norm = sqrt(Σ_all g^2). If it exceeds max_norm, every grad is
 * scaled by max_norm/(norm+eps) so the whole vector's norm becomes max_norm,
 * preserving its direction. The eps guards against divide-by-zero. Returns the
 * PRE-clip norm (useful for logging exploding-gradient events). */
float plist_clip_grad_norm(ParamList *pl, float max_norm) {
    double sq = 0.0;
    for (int i = 0; i < pl->count; ++i)
        for (int e = 0; e < pl->item[i].n; ++e) { double g = pl->item[i].g[e]; sq += g * g; }
    float norm = (float)sqrt(sq);
    if (max_norm > 0.0f && norm > max_norm) {
        float s = max_norm / (norm + 1e-6f);
        for (int i = 0; i < pl->count; ++i)
            for (int e = 0; e < pl->item[i].n; ++e) pl->item[i].g[e] *= s;
    }
    return norm;
}

/* ---- shape helper ---------------------------------------------------- */
/* Lazily (re)allocate m to exactly [r,c]. If it is already that shape this is
 * a no-op, so caches are allocated once on the first pass and then reused
 * across all subsequent passes of the same size (no per-step malloc churn). */
static void ensure(Mat *m, int r, int c) {
    if (m->data == NULL || m->rows != r || m->cols != c) {
        if (m->data) mat_free(m);
        *m = mat_new(r, c);
    }
}

/* ---- Linear ---------------------------------------------------------- */
Linear linear_new(int in, int out, ParamList *pl, const char *name) {
    Linear l; l.in = in; l.out = out;
    l.W = mat_new(out, in); l.dW = mat_new(out, in);
    l.b = (float *)calloc(out, sizeof(float));
    l.db = (float *)calloc(out, sizeof(float));
    l.xcache = (Mat){0, 0, NULL};
    l.st_x = (Mat){0, 0, NULL}; l.st_dy = (Mat){0, 0, NULL};
    l.st_n = l.st_cap = 0;
    l.s_dWtmp = (Mat){0, 0, NULL};
    /* Kaiming-style uniform init matching PyTorch's nn.Linear default: both W
     * and b are drawn from U(-k, k) with k = 1/sqrt(fan_in). Sampling order
     * (all of W, then b) is part of the golden-regression contract. */
    float k = 1.0f / sqrtf((float)in);           /* PyTorch nn.Linear default */
    for (int i = 0; i < out * in; ++i) l.W.data[i] = nn_uniform(-k, k);
    for (int i = 0; i < out; ++i)      l.b[i]      = nn_uniform(-k, k);  /* bias U(-1/√in,·) */
    /* Register W and b (with their grads) as two separate Params. */
    plist_add(pl, l.W.data, l.dW.data, out * in, name);
    plist_add(pl, l.b, l.db, out, name);
    return l;
}
/* Forward:  y[t,i] = Σ_j x[t,j]·W[i,j] + b[i]   (i.e. y = x·W^T + b).
 * mat_matmul_bt multiplies x by W transposed ("bt" = B-transposed). Caches x
 * for the backward's dW computation. */
void linear_forward(Linear *l, const Mat x, Mat y) {
    assert(x.cols == l->in && y.rows == x.rows && y.cols == l->out);
    ensure(&l->xcache, x.rows, x.cols); mat_copy(l->xcache, x);
    mat_matmul_bt(x, l->W, y);
    mat_add_bias(y, l->b);
}
/* Backward for y = x·W^T + b. Given dy = dL/dy [T,out]:
 *   dW[i,j] += Σ_t dy[t,i]·x[t,j]     (dW = dy^T · x)
 *   db[i]   += Σ_t dy[t,i]            (column sum of dy)
 *   dx[t,j]  = Σ_i dy[t,i]·W[i,j]     (dx = dy · W)   — only if dx is non-NULL
 * IMPORTANT asymmetry: dW and db are ACCUMULATED with += so gradients from
 * every token (and every reuse of a shared weight, e.g. the same Linear called
 * once per GRU step) sum correctly across the whole pass. dx, in contrast, is
 * OVERWRITTEN by mat_matmul because it is the local input gradient for exactly
 * this call and must not carry state from a previous one. Any accumulation of
 * dx across sublayers is done by the CALLER (see mha_backward / block_backward). */
void linear_backward(Linear *l, const Mat dy, Mat dx) {
    const Mat x = l->xcache;
    const int T = dy.rows, out = l->out, in = l->in;
    assert(dy.cols == out && x.rows == T);
    if (g_defer) {
        /* Deferred path: append this call's (x, dy) rows to the stash instead
         * of touching dW/db — the whole batch becomes one GEMM in
         * linear_flush. The stash grows geometrically and persists (no
         * steady-state allocation). dx is still computed immediately below,
         * because the caller's backward chain needs it now. */
        if (l->st_n + T > l->st_cap) {
            int cap = l->st_cap ? l->st_cap : 32;
            while (cap < l->st_n + T) cap *= 2;
            Mat nx = mat_new(cap, in), ndy = mat_new(cap, out);
            if (l->st_n) {
                memcpy(nx.data,  l->st_x.data,  (size_t)l->st_n * in  * sizeof(float));
                memcpy(ndy.data, l->st_dy.data, (size_t)l->st_n * out * sizeof(float));
            }
            mat_free(&l->st_x); mat_free(&l->st_dy);
            l->st_x = nx; l->st_dy = ndy; l->st_cap = cap;
        }
        memcpy(&l->st_x.data[(size_t)l->st_n * in],  x.data,  (size_t)T * in  * sizeof(float));
        memcpy(&l->st_dy.data[(size_t)l->st_n * out], dy.data, (size_t)T * out * sizeof(float));
        l->st_n += T;
    } else {
        for (int p = 0; p < T; ++p) {               /* dW += dy^T @ x (pij) */
            const float *dyr = &dy.data[p * out];
            const float *xr  = &x.data[p * in];
            for (int i = 0; i < out; ++i) {
                const float a = dyr[i];
                float *dWr = &l->dW.data[i * in];
                for (int j = 0; j < in; ++j) dWr[j] += a * xr[j];
            }
        }
        /* db += colsum(dy), accumulated row-by-row: the p-outer/i-inner order
         * walks dy contiguously and updates db[i] with independent adds
         * (vectorizable), instead of a strided reduction per column. */
        for (int p = 0; p < T; ++p) {
            const float *restrict dyr = &dy.data[p * out];
            for (int i = 0; i < out; ++i) l->db[i] += dyr[i];
        }
    }
    if (dx.data) mat_matmul(dy, l->W, dx);          /* dx = dy @ W (overwrites dx) */
}

/* Flush the deferred stash: dW += st_dyᵀ · st_x as ONE [st_n]-row GEMM (via
 * the existing atb kernel into scratch, then an axpy — no new backend kernel
 * needed), db += column sums of st_dy. Idempotent: resets st_n to 0. */
void linear_flush(Linear *l) {
    if (l->st_n == 0) return;
    const int in = l->in, out = l->out;
    Mat sx  = { l->st_n, in,  l->st_x.data };       /* filled prefix of the stash */
    Mat sdy = { l->st_n, out, l->st_dy.data };
    ensure(&l->s_dWtmp, out, in);
    mat_matmul_atb(sdy, sx, l->s_dWtmp);            /* dyᵀ·x over the whole batch */
    mat_axpy(l->dW, 1.0f, l->s_dWtmp);
    for (int p = 0; p < l->st_n; ++p) {
        const float *restrict dyr = &sdy.data[p * out];
        for (int i = 0; i < out; ++i) l->db[i] += dyr[i];
    }
    l->st_n = 0;
}
void linear_free(Linear *l) {
    mat_free(&l->W); mat_free(&l->dW); free(l->b); free(l->db); mat_free(&l->xcache);
    mat_free(&l->st_x); mat_free(&l->st_dy); mat_free(&l->s_dWtmp);
}

/* ---- LayerNorm ------------------------------------------------------- */
LayerNorm layernorm_new(int dim, ParamList *pl, const char *name) {
    LayerNorm ln; ln.dim = dim;
    ln.g = (float *)malloc(dim * sizeof(float));
    ln.b = (float *)calloc(dim, sizeof(float));
    ln.dg = (float *)calloc(dim, sizeof(float));
    ln.db = (float *)calloc(dim, sizeof(float));
    for (int i = 0; i < dim; ++i) ln.g[i] = 1.0f;
    ln.xhat = (Mat){0, 0, NULL}; ln.rstd = NULL; ln.Tcap = 0;
    plist_add(pl, ln.g, ln.dg, dim, name);
    plist_add(pl, ln.b, ln.db, dim, name);
    return ln;
}
/* Forward, per row i:
 *   mean_i = (1/D) Σ_j x[i,j]
 *   var_i  = (1/D) Σ_j (x[i,j]-mean_i)^2          (biased / population variance)
 *   rstd_i = 1/sqrt(var_i + eps)
 *   xhat[i,j] = (x[i,j]-mean_i)·rstd_i            (cached for backward)
 *   y[i,j]    = g[j]·xhat[i,j] + b[j]
 * mean/var are accumulated in double for stability; eps=1e-5 guards constant
 * rows (var=0). rstd is cached per row so the backward avoids recomputing it. */
void layernorm_forward(LayerNorm *ln, const Mat x, Mat y) {
    const int T = x.rows, D = ln->dim;
    assert(x.cols == D && y.rows == T && y.cols == D);
    ensure(&ln->xhat, T, D);
    if (ln->Tcap < T) { free(ln->rstd); ln->rstd = (float *)malloc(T * sizeof(float)); ln->Tcap = T; }
    const float eps = 1e-5f;
    for (int i = 0; i < T; ++i) {
        const float *xr = &x.data[i * D];
        double mean = 0.0; for (int j = 0; j < D; ++j) mean += xr[j]; mean /= D;
        double var = 0.0; for (int j = 0; j < D; ++j) { double d = xr[j] - mean; var += d * d; }
        var /= D;
        float rstd = (float)(1.0 / sqrt(var + eps)); ln->rstd[i] = rstd;
        for (int j = 0; j < D; ++j) {
            float xh = (float)((xr[j] - mean)) * rstd;
            ln->xhat.data[i * D + j] = xh;
            y.data[i * D + j] = xh * ln->g[j] + ln->b[j];
        }
    }
}
/* Backward. Param grads (accumulated over all rows, +=):
 *   dg[j] += Σ_i dy[i,j]·xhat[i,j]
 *   db[j] += Σ_i dy[i,j]
 * Input grad, per row i, using the standard LN Jacobian. Let dxh = dy·g be the
 * grad w.r.t. the normalized value. With m1 = mean_j(dxh) and
 * m2 = mean_j(dxh·xhat) the mean/variance coupling contracts to:
 *   dx[i,j] = rstd_i · ( dxh[i,j] - m1 - xhat[i,j]·m2 )
 * The -m1 term removes the mean-subtraction's contribution, and the
 * -xhat·m2 term removes the variance-normalization's. dx is OVERWRITTEN here
 * (one row's local grad), unlike the accumulated dg/db. */
void layernorm_backward(LayerNorm *ln, const Mat dy, Mat dx) {
    const int T = dy.rows, D = ln->dim;
    for (int j = 0; j < D; ++j) {                    /* param grads */
        float dgj = 0.0f, dbj = 0.0f;
        for (int i = 0; i < T; ++i) {
            dgj += dy.data[i * D + j] * ln->xhat.data[i * D + j];
            dbj += dy.data[i * D + j];
        }
        ln->dg[j] += dgj; ln->db[j] += dbj;
    }
    for (int i = 0; i < T; ++i) {                    /* input grad, per row */
        const float *xh = &ln->xhat.data[i * D];
        double m1 = 0.0, m2 = 0.0;                   /* m1=mean(dxh), m2=mean(dxh·xhat) */
        for (int j = 0; j < D; ++j) {
            float dxh = dy.data[i * D + j] * ln->g[j];
            m1 += dxh; m2 += dxh * xh[j];
        }
        m1 /= D; m2 /= D;
        float rstd = ln->rstd[i];
        for (int j = 0; j < D; ++j) {
            float dxh = dy.data[i * D + j] * ln->g[j];
            dx.data[i * D + j] = rstd * (dxh - (float)m1 - xh[j] * (float)m2);
        }
    }
}
void layernorm_free(LayerNorm *ln) { free(ln->g); free(ln->b); free(ln->dg); free(ln->db); mat_free(&ln->xhat); free(ln->rstd); }

/* ---- FFN ------------------------------------------------------------- */
FFN ffn_new(int d, int f, ParamList *pl, const char *name) {
    FFN ff; ff.d = d; ff.f = f;
    ff.fc1 = linear_new(d, f, pl, name);
    ff.fc2 = linear_new(f, d, pl, name);
    ff.h = (Mat){0,0,NULL}; ff.a = (Mat){0,0,NULL};
    ff.s_da = (Mat){0,0,NULL}; ff.s_dh = (Mat){0,0,NULL};
    return ff;
}
/* Forward:  h = fc1(x) [T,F];  a = ReLU(h);  y = fc2(a) [T,D].
 * Keeps both h (pre-activation, for the ReLU mask) and a (fc2's input). */
void ffn_forward(FFN *ff, const Mat x, Mat y) {
    const int T = x.rows;
    ensure(&ff->h, T, ff->f); ensure(&ff->a, T, ff->f);
    linear_forward(&ff->fc1, x, ff->h);
    mat_copy(ff->a, ff->h); mat_relu(ff->a);
    linear_forward(&ff->fc2, ff->a, y);
}
/* Backward:  da = fc2^T(dy) gives dL/da;  ReLU'(h)=1[h>0] so
 *   dh[i] = (h[i] > 0) ? da[i] : 0    (gradient is cut at the ReLU kink;
 *            exactly-zero pre-activations are treated as off, matching forward);
 * then dx = fc1^T(dh). The two linear_backward calls also accumulate dW/db for
 * fc2 and fc1 respectively. */
void ffn_backward(FFN *ff, const Mat dy, Mat dx) {
    const int T = dy.rows;
    ensure(&ff->s_da, T, ff->f); ensure(&ff->s_dh, T, ff->f);
    Mat da = ff->s_da, dh = ff->s_dh;
    linear_backward(&ff->fc2, dy, da);
    for (int i = 0; i < T * ff->f; ++i) dh.data[i] = (ff->h.data[i] > 0.0f) ? da.data[i] : 0.0f;
    linear_backward(&ff->fc1, dh, dx);
}
void ffn_flush(FFN *ff) { linear_flush(&ff->fc1); linear_flush(&ff->fc2); }
void ffn_free(FFN *ff) {
    linear_free(&ff->fc1); linear_free(&ff->fc2);
    mat_free(&ff->h); mat_free(&ff->a); mat_free(&ff->s_da); mat_free(&ff->s_dh);
}

/* ---- GRU cell ------------------------------------------------------- */
static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
/* Allocate an array of n single-row [1,cols] matrices, used to hold one
 * cached value per rollout step. */
static Mat  *alloc_mats(int n, int cols) {
    Mat *a = (Mat *)calloc(n, sizeof(Mat));
    for (int i = 0; i < n; ++i) a[i] = mat_new(1, cols);
    return a;
}
GRU gru_new(int in, int hid, int max_steps, ParamList *pl, const char *name) {
    GRU g; g.in = in; g.hid = hid; g.max_steps = max_steps;
    g.lr = linear_new(in + hid, hid, pl, name);   /* reset gate     from [x;h]     */
    g.lu = linear_new(in + hid, hid, pl, name);   /* update gate    from [x;h]     */
    g.ln = linear_new(in + hid, hid, pl, name);   /* candidate      from [x; r⊙h]  */
    g.zruc = alloc_mats(max_steps, in + hid);     /* [x;h] per step */
    g.znc  = alloc_mats(max_steps, in + hid);     /* [x; r⊙h]       */
    g.rc = alloc_mats(max_steps, hid); g.uc = alloc_mats(max_steps, hid);
    g.nc = alloc_mats(max_steps, hid); g.hpc = alloc_mats(max_steps, hid);
    g.s_dzru = mat_new(1, in + hid); g.s_dzn = mat_new(1, in + hid); g.s_dg = mat_new(1, hid);
    return g;
}
/* One GRU step. Gate equations (⊙ = elementwise, [;] = concat):
 *   r_t = σ(Lr[x_t ; h_{t-1}])                    reset gate
 *   u_t = σ(Lu[x_t ; h_{t-1}])                    update gate
 *   n_t = tanh(Ln[x_t ; r_t⊙h_{t-1}])            candidate state
 *   h_t = (1 - u_t)⊙n_t + u_t⊙h_{t-1}            blend
 * Note the update-gate convention here: u=1 KEEPS the old state (h_t=h_{t-1})
 * and u=0 fully replaces it with the candidate. All intermediates (zru, zn, r,
 * u, n, and a snapshot hp of h_{t-1}) are cached under `step` for the backward. */
void gru_forward(GRU *g, int step, const Mat x, const Mat hprev, Mat hout) {
    const int I = g->in, H = g->hid;
    Mat zru = g->zruc[step], zn = g->znc[step];
    Mat r = g->rc[step], u = g->uc[step], n = g->nc[step], hp = g->hpc[step];
    mat_copy(hp, hprev);
    memcpy(&zru.data[0], x.data, I * sizeof(float));           /* zru = [x ; h_prev] */
    memcpy(&zru.data[I], hprev.data, H * sizeof(float));
    linear_forward(&g->lr, zru, r); for (int i = 0; i < H; ++i) r.data[i] = sigmoidf(r.data[i]);
    linear_forward(&g->lu, zru, u); for (int i = 0; i < H; ++i) u.data[i] = sigmoidf(u.data[i]);
    memcpy(&zn.data[0], x.data, I * sizeof(float));            /* zn = [x ; r⊙h_prev] */
    for (int i = 0; i < H; ++i) zn.data[I + i] = r.data[i] * hprev.data[i];
    linear_forward(&g->ln, zn, n); for (int i = 0; i < H; ++i) n.data[i] = tanhf(n.data[i]);
    for (int i = 0; i < H; ++i) hout.data[i] = (1.0f - u.data[i]) * n.data[i] + u.data[i] * hp.data[i];
}
/* Backward through one GRU step. Input dh = dL/dh_t. Produces:
 *   dx      = dL/dx_t              [1,I]  (overwritten)
 *   dhprev  = dL/dh_{t-1}          [1,H]  (the cross-step hidden gradient that
 *             the caller feeds as dh into step t-1 — this is what threads BPTT
 *             through the whole rollout), and accumulates dW/db of the 3 gates.
 * h_{t-1} feeds h_t through FOUR paths, all summed into dhp_acc:
 *   (1) the candidate via r⊙h_prev, (2) the reset gate r, (3) the update gate u,
 *   (4) the direct blend term u⊙h_prev. Each Linear's xcache is restored (the
 *   [x;h] / [x;r⊙h] snapshot) before its backward, because forward reused the
 *   same Linear objects across steps and overwrote their xcache. */
void gru_backward(GRU *g, int step, const Mat dh, Mat dx, Mat dhprev) {
    const int I = g->in, H = g->hid;
    Mat zru = g->zruc[step], zn = g->znc[step];
    Mat r = g->rc[step], u = g->uc[step], n = g->nc[step], hp = g->hpc[step];
    Mat dzru = g->s_dzru, dzn = g->s_dzn, dg = g->s_dg;
    float dhp[64] = {0}; /* hid <= 64 in practice; fall back to hp cols */
    float *dhp_acc = dhp;                                      /* accumulates dL/dh_{t-1} over all 4 paths */
    if (H > 64) dhp_acc = (float *)calloc(H, sizeof(float));   /* heap fallback for large hidden dims */
    for (int i = 0; i < H; ++i) dhp_acc[i] = 0.0f;
    /* h = (1-u)⊙n + u⊙hp  =>  dn_pre = dh·(1-u)·tanh'(pre_n), tanh'=1-n^2 */
    for (int i = 0; i < H; ++i) {                              /* candidate pre-activation grad */
        float dn = dh.data[i] * (1.0f - u.data[i]);            /* ∂h/∂n = 1-u */
        dg.data[i] = dn * (1.0f - n.data[i] * n.data[i]);      /* tanh' */
    }
    mat_copy(g->ln.xcache, zn);                                /* restore Ln's input for its backward */
    linear_backward(&g->ln, dg, dzn);                          /* dzn = [dx_n ; d(r⊙hp)] */
    for (int i = 0; i < I; ++i) if (dx.data) dx.data[i] = dzn.data[i];   /* dx path 1 (candidate input) */
    /* r⊙hp path: the product d(r⊙hp) splits by product rule into dr and dhp */
    for (int i = 0; i < H; ++i) {
        float drh = dzn.data[I + i];                           /* dL/d(r_i·hp_i) */
        dhp_acc[i] += drh * r.data[i];                         /* into h_prev via candidate (path 1) */
        float dr = drh * hp.data[i];                           /* dL/dr_i */
        dg.data[i] = dr * r.data[i] * (1.0f - r.data[i]);      /* reuse dg for reset-gate pre-act (σ') */
    }
    mat_copy(g->lr.xcache, zru);                               /* restore Lr's input [x;h_prev] */
    linear_backward(&g->lr, dg, dzru);                         /* dzru = [dx_r ; dhp_r] */
    for (int i = 0; i < I; ++i) if (dx.data) dx.data[i] += dzru.data[i];   /* dx path 2 (reset gate) */
    for (int i = 0; i < H; ++i) dhp_acc[i] += dzru.data[I + i];            /* into h_prev via reset (path 2) */
    /* update gate: ∂h/∂u = hp - n  => du_pre = dh·(hp-n)·σ'(u); plus the direct
     * blend term ∂h/∂hp = u contributes dh·u straight to h_prev. */
    for (int i = 0; i < H; ++i) {
        float du = dh.data[i] * (hp.data[i] - n.data[i]);
        dg.data[i] = du * u.data[i] * (1.0f - u.data[i]);      /* sigmoid' */
        dhp_acc[i] += dh.data[i] * u.data[i];                  /* u⊙hp direct path (path 4) */
    }
    mat_copy(g->lu.xcache, zru);                               /* restore Lu's input [x;h_prev] */
    linear_backward(&g->lu, dg, dzru);                         /* dzru = [dx_u ; dhp_u] */
    for (int i = 0; i < I; ++i) if (dx.data) dx.data[i] += dzru.data[i];   /* dx path 3 (update gate) */
    for (int i = 0; i < H; ++i) dhp_acc[i] += dzru.data[I + i];            /* into h_prev via update (path 3) */
    if (dhprev.data) for (int i = 0; i < H; ++i) dhprev.data[i] = dhp_acc[i];  /* export cross-step grad */
    if (dhp_acc != dhp) free(dhp_acc);
}
void gru_flush(GRU *g) { linear_flush(&g->lr); linear_flush(&g->lu); linear_flush(&g->ln); }
void gru_free(GRU *g) {
    linear_free(&g->lr); linear_free(&g->lu); linear_free(&g->ln);
    for (int s = 0; s < g->max_steps; ++s) {
        mat_free(&g->zruc[s]); mat_free(&g->znc[s]); mat_free(&g->rc[s]);
        mat_free(&g->uc[s]); mat_free(&g->nc[s]); mat_free(&g->hpc[s]);
    }
    free(g->zruc); free(g->znc); free(g->rc); free(g->uc); free(g->nc); free(g->hpc);
    mat_free(&g->s_dzru); mat_free(&g->s_dzn); mat_free(&g->s_dg);
}

/* ---- Single-head cross-attention ------------------------------------ */
CrossAttn xattn_new(int qin, int d, int max_steps, ParamList *pl, const char *name) {
    CrossAttn a; a.qin = qin; a.d = d; a.max_steps = max_steps; a.T = 0;
    a.lq = linear_new(qin, d, pl, name);
    a.lk = linear_new(d,   d, pl, name);
    a.lv = linear_new(d,   d, pl, name);
    a.lo = linear_new(d,   d, pl, name);
    a.K = (Mat){0,0,NULL}; a.V = (Mat){0,0,NULL}; a.memc = (Mat){0,0,NULL};
    a.qc  = (Mat *)calloc(max_steps, sizeof(Mat));
    a.ac  = (Mat *)calloc(max_steps, sizeof(Mat));
    a.cxc = (Mat *)calloc(max_steps, sizeof(Mat));
    a.xc  = (Mat *)calloc(max_steps, sizeof(Mat));
    a.s_dctx = (Mat){0,0,NULL}; a.s_da = (Mat){0,0,NULL}; a.s_dsc = (Mat){0,0,NULL};
    a.s_dq = (Mat){0,0,NULL}; a.s_dK = (Mat){0,0,NULL}; a.s_dV = (Mat){0,0,NULL};
    return a;
}
/* Project the FIXED memory once:  K = lk(mem), V = lv(mem), both [T,D].
 * These are shared across every decode step, so this runs once per rollout
 * (not per step). Also zeroes the s_dK/s_dV accumulators, which every
 * xattn_backward_step will add into before xattn_backward_memory pushes them
 * back through lk/lv to the memory. */
void xattn_set_memory(CrossAttn *a, const Mat mem) {
    const int T = mem.rows, D = a->d;
    a->T = T;
    ensure(&a->memc, T, D); mat_copy(a->memc, mem);
    ensure(&a->K, T, D); ensure(&a->V, T, D);
    linear_forward(&a->lk, mem, a->K);            /* caches mem in lk.xcache */
    linear_forward(&a->lv, mem, a->V);            /* caches mem in lv.xcache */
    ensure(&a->s_dK, T, D); ensure(&a->s_dV, T, D);
    mat_zero(a->s_dK); mat_zero(a->s_dV);         /* accumulate over steps */
}
/* One decode step. Given step input x [1,qin]:
 *   q      = lq(x)                              [1,D]  the per-step query
 *   s_j    = (q·K_j)/sqrt(D)                     scaled dot-product over T keys
 *   a      = softmax_j(s)                        [1,T] attention weights
 *   ctx    = Σ_j a_j·V_j                         [1,D] attended memory value
 *   out    = lo(ctx)                             [1,D]
 * The 1/sqrt(D) scaling keeps logits from growing with dimension (so softmax
 * doesn't saturate). Softmax subtracts the row max (mx) before exp for numeric
 * stability (exp of a large logit would overflow). q, a, ctx, and x are cached
 * per step for the backward. */
void xattn_forward(CrossAttn *a, int step, const Mat x, Mat out) {
    const int T = a->T, D = a->d;
    const float scale = 1.0f / sqrtf((float)D);
    ensure(&a->qc[step], 1, D); ensure(&a->ac[step], 1, T);
    ensure(&a->cxc[step], 1, D); ensure(&a->xc[step], 1, a->qin);
    mat_copy(a->xc[step], x);
    linear_forward(&a->lq, x, a->qc[step]);       /* q [1,D] (caches x, overwritten per step) */
    Mat q = a->qc[step], av = a->ac[step], ctx = a->cxc[step];
    for (int j = 0; j < T; ++j) {                 /* scores = q·K_j * scale */
        const float *kj = &a->K.data[j * D];
        float s = 0.0f; for (int i = 0; i < D; ++i) s += q.data[i] * kj[i];
        av.data[j] = s * scale;
    }
    float mx = av.data[0]; for (int j = 1; j < T; ++j) if (av.data[j] > mx) mx = av.data[j];   /* max for stability */
    double sum = 0.0; for (int j = 0; j < T; ++j) { av.data[j] = expf(av.data[j] - mx); sum += av.data[j]; }
    for (int j = 0; j < T; ++j) av.data[j] /= (float)sum;   /* softmax (av now holds the weights) */
    for (int i = 0; i < D; ++i) {                 /* ctx = Σ a_j V_j */
        float c = 0.0f; for (int j = 0; j < T; ++j) c += av.data[j] * a->V.data[j * D + i];
        ctx.data[i] = c;
    }
    linear_forward(&a->lo, ctx, out);             /* out = o(ctx); caches ctx per step */
}
/* Backward for one step. Input dout = dL/dout [1,D]. Reverses xattn_forward:
 *   dctx      = lo^T(dout)                                     [1,D]
 *   da_j      = Σ_i dctx_i·V_{j,i} ;  dV_j += a_j·dctx         (ctx = Σ a_j V_j)
 *   softmax backward:  with dot = Σ_j a_j·da_j,
 *     dsc_j = a_j·(da_j - dot)·scale                            (Jacobian of softmax,
 *             folding in the 1/sqrt(D) scale from the forward scores)
 *   dq_i    = Σ_j dsc_j·K_{j,i} ;  dK_j += dsc_j·q             (scores = q·K^T)
 *   dx      = lq^T(dq)                                          [1,qin]
 * dV and dK ACCUMULATE into the shared s_dV/s_dK (every step contributes,
 * because K/V are shared); dx and dq are per-step. lo/lq xcaches are restored
 * from this step's ctx/x snapshots before their linear_backward. */
void xattn_backward_step(CrossAttn *a, int step, const Mat dout, Mat dx) {
    const int T = a->T, D = a->d;
    const float scale = 1.0f / sqrtf((float)D);
    ensure(&a->s_dctx, 1, D); ensure(&a->s_da, 1, T); ensure(&a->s_dsc, 1, T); ensure(&a->s_dq, 1, D);
    Mat q = a->qc[step], av = a->ac[step];
    Mat dctx = a->s_dctx, da = a->s_da, dsc = a->s_dsc, dq = a->s_dq;
    mat_copy(a->lo.xcache, a->cxc[step]);         /* restore lo's input (this step's ctx) */
    linear_backward(&a->lo, dout, dctx);          /* dout -> dctx */
    for (int j = 0; j < T; ++j) {                 /* ctx = Σ a_j V_j :  da_j, dV_j */
        const float *vj = &a->V.data[j * D];
        float g = 0.0f;
        for (int i = 0; i < D; ++i) { g += dctx.data[i] * vj[i]; a->s_dV.data[j * D + i] += av.data[j] * dctx.data[i]; }
        da.data[j] = g;
    }
    double dot = 0.0; for (int j = 0; j < T; ++j) dot += av.data[j] * da.data[j];   /* softmax backward: dot=Σ a·da */
    for (int j = 0; j < T; ++j) dsc.data[j] = av.data[j] * (da.data[j] - (float)dot) * scale;
    for (int i = 0; i < D; ++i) {                 /* scores = q·Kᵀ:  dq, dK */
        float g = 0.0f;
        for (int j = 0; j < T; ++j) { g += dsc.data[j] * a->K.data[j * D + i]; a->s_dK.data[j * D + i] += dsc.data[j] * q.data[i]; }
        dq.data[i] = g;
    }
    mat_copy(a->lq.xcache, a->xc[step]);          /* restore lq's input (this step's x) */
    linear_backward(&a->lq, dq, dx);              /* dq -> dx [1,qin] */
}
/* After ALL steps have accumulated s_dK/s_dV, flow them back to the memory in
 * one pass:  dmem = lk^T(dK) + lv^T(dV)   [T,D]. lk/lv still hold mem in their
 * xcache (from xattn_set_memory, never overwritten since), so their backwards
 * also accumulate dW/db for the key/value projections. The lk path writes dmem
 * directly (overwrite via linear_backward), then the lv path is computed into a
 * borrowed [T,D] scratch (s_dctx reused) and ADDED so both contributions sum. */
void xattn_backward_memory(CrossAttn *a, Mat dmem) {
    const int T = a->T, D = a->d;
    linear_backward(&a->lk, a->s_dK, dmem);       /* dK -> dmem (lk.xcache=mem) */
    ensure(&a->s_dctx, 1, T * D);                 /* borrow as [T,D] temp for lv path */
    Mat tmp = { T, D, a->s_dctx.data };
    linear_backward(&a->lv, a->s_dV, tmp);        /* dV -> tmp */
    if (dmem.data) for (int i = 0; i < T * D; ++i) dmem.data[i] += tmp.data[i];   /* dmem += lv path */
}
void xattn_flush(CrossAttn *a) {
    linear_flush(&a->lq); linear_flush(&a->lk); linear_flush(&a->lv); linear_flush(&a->lo);
}
void xattn_free(CrossAttn *a) {
    linear_free(&a->lq); linear_free(&a->lk); linear_free(&a->lv); linear_free(&a->lo);
    mat_free(&a->K); mat_free(&a->V); mat_free(&a->memc);
    for (int s = 0; s < a->max_steps; ++s) { mat_free(&a->qc[s]); mat_free(&a->ac[s]); mat_free(&a->cxc[s]); mat_free(&a->xc[s]); }
    free(a->qc); free(a->ac); free(a->cxc); free(a->xc);
    mat_free(&a->s_dctx); mat_free(&a->s_da); mat_free(&a->s_dsc);
    mat_free(&a->s_dq); mat_free(&a->s_dK); mat_free(&a->s_dV);
}

/* ---- softmax over rows (in place) ----------------------------------- */
/* For each row: r_j <- exp(r_j - max_k r_k) / Σ_k exp(r_k - max_k r_k).
 * Subtracting the row max is the standard numerical-stability trick: it makes
 * the largest exponent 0 (so exp can't overflow) without changing the result,
 * since softmax is invariant to a per-row additive constant. Sum in double. */
static void softmax_rows(Mat m) {
    for (int i = 0; i < m.rows; ++i) {
        float *r = &m.data[i * m.cols];
        float mx = r[0]; for (int j = 1; j < m.cols; ++j) if (r[j] > mx) mx = r[j];
        double s = 0.0;
        for (int j = 0; j < m.cols; ++j) { r[j] = expf(r[j] - mx); s += r[j]; }
        float inv = (float)(1.0 / s);
        for (int j = 0; j < m.cols; ++j) r[j] *= inv;
    }
}

/* ---- Multi-head self-attention -------------------------------------- */
MHA mha_new(int d_model, int n_heads, int self_only, ParamList *pl, const char *name) {
    assert(d_model % n_heads == 0);
    MHA m; m.d_model = d_model; m.n_heads = n_heads; m.head_dim = d_model / n_heads;
    m.self_only = self_only;
    m.q = linear_new(d_model, d_model, pl, name);
    m.k = linear_new(d_model, d_model, pl, name);
    m.v = linear_new(d_model, d_model, pl, name);
    m.o = linear_new(d_model, d_model, pl, name);
    m.Q = (Mat){0,0,NULL}; m.K = (Mat){0,0,NULL}; m.V = (Mat){0,0,NULL}; m.Ocat = (Mat){0,0,NULL};
    m.P = (Mat *)calloc(n_heads, sizeof(Mat)); m.Scap = 0;
    m.s_dOcat = (Mat){0,0,NULL}; m.s_dQ = (Mat){0,0,NULL}; m.s_dK = (Mat){0,0,NULL};
    m.s_dV = (Mat){0,0,NULL}; m.s_dxq = (Mat){0,0,NULL}; m.s_dxk = (Mat){0,0,NULL};
    m.s_dP = (Mat){0,0,NULL}; m.s_dsc = (Mat){0,0,NULL};
    return m;
}
/* Forward. Full self-attention over S rows (D-dim tokens). Project once:
 *   Q = q(x), K = k(x), V = v(x)   all [S,D], then split each row into H heads
 *   of width hd = D/H (head h occupies columns [off, off+hd), off = h·hd).
 * Per head h:
 *   scores[i,j] = (Q_h[i]·K_h[j])/sqrt(hd) - slope_h·|i-j|   (ALiBi bias if on)
 *   P_h = softmax_j(scores)                                   [S,S], rows sum to 1
 *   O_h[i] = Σ_j P_h[i,j]·V_h[j]                              per-head context
 * The per-head O_h are written back into their column slice of Ocat [S,D]
 * (the "concat" of heads), then y = o(Ocat). 1/sqrt(hd) is the standard scale.
 *
 * self_only branch: each position attends ONLY to itself, so attention is the
 * identity map on values and collapses to y = o(v(x)) with no Q/K/scores at all
 * (used for the single-node spatial path where there is nothing else to attend
 * to). ALiBi is deliberately skipped here (spatial, not temporal). */
void mha_forward(MHA *m, const Mat x, Mat y) {
    const int S = x.rows, D = m->d_model, H = m->n_heads, hd = m->head_dim;
    ensure(&m->V, S, D); ensure(&m->Ocat, S, D);
    if (m->self_only) {
        /* single-element attention: output = O(V(x)) per position */
        linear_forward(&m->v, x, m->V);
        mat_copy(m->Ocat, m->V);
        linear_forward(&m->o, m->Ocat, y);
        return;
    }
    ensure(&m->Q, S, D); ensure(&m->K, S, D);
    for (int h = 0; h < H; ++h) ensure(&m->P[h], S, S);
    linear_forward(&m->q, x, m->Q);
    linear_forward(&m->k, x, m->K);
    linear_forward(&m->v, x, m->V);
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < H; ++h) {
        const int off = h * hd;                                      /* this head's column offset in [S,D] */
        const float slope = g_timebias ? alibi_slope(h, H) : 0.0f;   /* ALiBi recency prior */
        Mat P = m->P[h];
        const int hd4 = hd & ~3;                                     /* multiple of 4 <= hd */
        for (int i = 0; i < S; ++i) {
            const float *restrict Qrow = &m->Q.data[i * D + off];
            for (int j = 0; j < S; ++j) {
                /* Q_h[i]·K_h[j]: contiguous dot over hd dims, split across 4
                 * accumulators to break the float-add dependency chain (same
                 * trick as mat_matmul_bt; head widths are multiples of 4). */
                const float *restrict Krow = &m->K.data[j * D + off];
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (int d = 0; d < hd4; d += 4) {
                    a0 += Qrow[d    ] * Krow[d    ];
                    a1 += Qrow[d + 1] * Krow[d + 1];
                    a2 += Qrow[d + 2] * Krow[d + 2];
                    a3 += Qrow[d + 3] * Krow[d + 3];
                }
                float acc = (a0 + a1) + (a2 + a3);
                for (int d = hd4; d < hd; ++d) acc += Qrow[d] * Krow[d];
                P.data[i * S + j] = acc * scale - slope * (float)abs(i - j);   /* bias is a constant */
            }
        }
        softmax_rows(P);
        /* O_h[i] = Σ_j P[i,j]·V_h[j], accumulated j-by-j so the inner d-loop
         * streams V's row contiguously (independent adds, vectorizable) instead
         * of a strided reduction per (i,d). */
        for (int i = 0; i < S; ++i) {
            float *restrict Orow = &m->Ocat.data[i * D + off];
            for (int d = 0; d < hd; ++d) Orow[d] = 0.0f;
            for (int j = 0; j < S; ++j) {
                const float p = P.data[i * S + j];
                const float *restrict Vrow = &m->V.data[j * D + off];
                for (int d = 0; d < hd; ++d) Orow[d] += p * Vrow[d];
            }
        }
    }
    linear_forward(&m->o, m->Ocat, y);
}
/* Backward. dy = dL/dy [S,D]. First  dOcat = o^T(dy). Then per head, reverse
 * O_h = P_h·V_h and the scaled-dot-product softmax:
 *   dP[i,j]  = Σ_d dO_h[i,d]·V_h[j,d]
 *   dV_h[j]  = Σ_i P[i,j]·dO_h[i,d]
 *   softmax bwd (row i, dot = Σ_k dP[i,k]·P[i,k]):
 *     dsc[i,j] = P[i,j]·(dP[i,j] - dot)·scale
 *   dQ_h[i]  = Σ_j dsc[i,j]·K_h[j]
 *   dK_h[j]  = Σ_i dsc[i,j]·Q_h[i]
 * The ALiBi bias is additive and independent of Q/K/V, so it contributes zero
 * gradient and simply does not appear here. Finally dx = q^T(dQ) + k^T(dK) +
 * v^T(dV): v's contribution is written first (overwrite), then the q and k
 * contributions are ADDED, since x fans out to all three projections.
 *
 * self_only branch: forward was y=o(v(x)) with Ocat=V, so dOcat is exactly dV
 * and there is no Q/K path — dx = v^T(dOcat). */
void mha_backward(MHA *m, const Mat dy, Mat dx) {
    const int S = dy.rows, D = m->d_model, H = m->n_heads, hd = m->head_dim;
    const float scale = 1.0f / sqrtf((float)hd);
    ensure(&m->s_dOcat, S, D);
    Mat dOcat = m->s_dOcat;
    linear_backward(&m->o, dy, dOcat);
    if (m->self_only) {                     /* dOcat == dV; q,k carry no grad */
        linear_backward(&m->v, dOcat, dx);
        return;
    }
    ensure(&m->s_dQ, S, D); ensure(&m->s_dK, S, D); ensure(&m->s_dV, S, D);
    ensure(&m->s_dP, S, S); ensure(&m->s_dsc, S, S);
    Mat dQ = m->s_dQ, dK = m->s_dK, dV = m->s_dV, dP = m->s_dP, dsc = m->s_dsc;
    for (int h = 0; h < H; ++h) {
        const int off = h * hd;
        Mat P = m->P[h];
        const int hd4 = hd & ~3;
        /* dP[i,j] = sum_d dOh[i,d] * Vh[j,d] — contiguous dot, 4 accumulators
         * (see mha_forward's score loop for why the chain is split). */
        for (int i = 0; i < S; ++i) {
            const float *restrict dOrow = &dOcat.data[i * D + off];
            for (int j = 0; j < S; ++j) {
                const float *restrict Vrow = &m->V.data[j * D + off];
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                for (int d = 0; d < hd4; d += 4) {
                    a0 += dOrow[d    ] * Vrow[d    ];
                    a1 += dOrow[d + 1] * Vrow[d + 1];
                    a2 += dOrow[d + 2] * Vrow[d + 2];
                    a3 += dOrow[d + 3] * Vrow[d + 3];
                }
                float acc = (a0 + a1) + (a2 + a3);
                for (int d = hd4; d < hd; ++d) acc += dOrow[d] * Vrow[d];
                dP.data[i * S + j] = acc;
            }
        }
        /* dVh[j,d] = sum_i P[i,j]*dOh[i,d], accumulated i-by-i so the inner
         * d-loop streams both rows contiguously (independent adds). */
        for (int j = 0; j < S; ++j) {
            float *restrict dVrow = &dV.data[j * D + off];
            for (int d = 0; d < hd; ++d) dVrow[d] = 0.0f;
        }
        for (int i = 0; i < S; ++i) {
            const float *restrict dOrow = &dOcat.data[i * D + off];
            for (int j = 0; j < S; ++j) {
                const float p = P.data[i * S + j];
                float *restrict dVrow = &dV.data[j * D + off];
                for (int d = 0; d < hd; ++d) dVrow[d] += p * dOrow[d];
            }
        }
        /* softmax backward per row: dsc[i,:] = P[i,:] * (dP[i,:] - sum_k dP[i,k]P[i,k]) */
        for (int i = 0; i < S; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < S; ++j) dot += dP.data[i * S + j] * P.data[i * S + j];
            for (int j = 0; j < S; ++j)
                dsc.data[i * S + j] = P.data[i * S + j] * (dP.data[i * S + j] - dot) * scale;
        }
        /* dQh[i,d] = sum_j dsc[i,j]*Kh[j,d] ; dKh[j,d] = sum_i dsc[i,j]*Qh[i,d]
         * — both accumulated with a contiguous inner d-loop over the source row. */
        for (int i = 0; i < S; ++i) {
            float *restrict dQrow = &dQ.data[i * D + off];
            for (int d = 0; d < hd; ++d) dQrow[d] = 0.0f;
            for (int j = 0; j < S; ++j) {
                const float s = dsc.data[i * S + j];
                const float *restrict Krow = &m->K.data[j * D + off];
                for (int d = 0; d < hd; ++d) dQrow[d] += s * Krow[d];
            }
        }
        for (int j = 0; j < S; ++j) {
            float *restrict dKrow = &dK.data[j * D + off];
            for (int d = 0; d < hd; ++d) dKrow[d] = 0.0f;
            for (int i = 0; i < S; ++i) {
                const float s = dsc.data[i * S + j];
                const float *restrict Qrow = &m->Q.data[i * D + off];
                for (int d = 0; d < hd; ++d) dKrow[d] += s * Qrow[d];
            }
        }
    }
    ensure(&m->s_dxq, S, D); ensure(&m->s_dxk, S, D);
    Mat dxq = m->s_dxq, dxk = m->s_dxk;
    linear_backward(&m->q, dQ, dxq);         /* x's grad via the query proj */
    linear_backward(&m->k, dK, dxk);         /* x's grad via the key proj   */
    linear_backward(&m->v, dV, dx);          /* dx starts as v's contribution (overwrite) */
    for (int i = 0; i < S * D; ++i) dx.data[i] += dxq.data[i] + dxk.data[i];   /* + q and k paths */
}
void mha_flush(MHA *m) {
    linear_flush(&m->q); linear_flush(&m->k); linear_flush(&m->v); linear_flush(&m->o);
}
void mha_free(MHA *m) {
    linear_free(&m->q); linear_free(&m->k); linear_free(&m->v); linear_free(&m->o);
    mat_free(&m->Q); mat_free(&m->K); mat_free(&m->V); mat_free(&m->Ocat);
    for (int h = 0; h < m->n_heads; ++h) mat_free(&m->P[h]);
    free(m->P);
    mat_free(&m->s_dOcat); mat_free(&m->s_dQ); mat_free(&m->s_dK); mat_free(&m->s_dV);
    mat_free(&m->s_dxq); mat_free(&m->s_dxk); mat_free(&m->s_dP); mat_free(&m->s_dsc);
}

/* ---- Transformer block (post-norm) ---------------------------------- */
Block block_new(int d_model, int n_heads, int ff_dim, int self_only, ParamList *pl, const char *name) {
    Block b; b.d = d_model;
    b.attn = mha_new(d_model, n_heads, self_only, pl, name);
    b.ln1 = layernorm_new(d_model, pl, name);
    b.ln2 = layernorm_new(d_model, pl, name);
    b.ff = ffn_new(d_model, ff_dim, pl, name);
    b.attn_out = (Mat){0,0,NULL}; b.r1 = (Mat){0,0,NULL}; b.y1 = (Mat){0,0,NULL};
    b.ff_out = (Mat){0,0,NULL}; b.r2 = (Mat){0,0,NULL};
    b.drop1 = (Mat){0,0,NULL}; b.drop2 = (Mat){0,0,NULL};
    b.s_dr2 = (Mat){0,0,NULL}; b.s_dy1 = (Mat){0,0,NULL};
    b.s_dr1 = (Mat){0,0,NULL}; b.s_dtmp = (Mat){0,0,NULL}; b.s_dd = (Mat){0,0,NULL};
    return b;
}
void block_forward(Block *b, const Mat x, Mat y) {
    const int S = x.rows, D = b->d;
    ensure(&b->attn_out, S, D); ensure(&b->r1, S, D); ensure(&b->y1, S, D);
    ensure(&b->ff_out, S, D); ensure(&b->r2, S, D);
    int drop = dropout_on();
    if (drop) { ensure(&b->drop1, S, D); ensure(&b->drop2, S, D); }
    if (g_prenorm) {
        /* pre-norm: y1 = x + Drop(MHA(LN1(x))); y = y1 + Drop(FFN(LN2(y1))) */
        layernorm_forward(&b->ln1, x, b->r1);                    /* r1 = LN1(x) */
        mha_forward(&b->attn, b->r1, b->attn_out);
        if (drop) dropout_apply(b->attn_out, b->drop1, g_dropout);
        for (int i = 0; i < S * D; ++i) b->y1.data[i] = x.data[i] + b->attn_out.data[i];
        layernorm_forward(&b->ln2, b->y1, b->r2);                /* r2 = LN2(y1) */
        ffn_forward(&b->ff, b->r2, b->ff_out);
        if (drop) dropout_apply(b->ff_out, b->drop2, g_dropout);
        for (int i = 0; i < S * D; ++i) y.data[i] = b->y1.data[i] + b->ff_out.data[i];
        return;
    }
    /* POST-NORM (default): LN is applied AFTER the residual add.
     *   r1 = x  + Drop(MHA(x));   y1 = LN1(r1)
     *   r2 = y1 + Drop(FFN(y1));  y  = LN2(r2)                              */
    mha_forward(&b->attn, x, b->attn_out);
    if (drop) dropout_apply(b->attn_out, b->drop1, g_dropout);   /* post-attn dropout */
    for (int i = 0; i < S * D; ++i) b->r1.data[i] = x.data[i] + b->attn_out.data[i];
    layernorm_forward(&b->ln1, b->r1, b->y1);
    ffn_forward(&b->ff, b->y1, b->ff_out);
    if (drop) dropout_apply(b->ff_out, b->drop2, g_dropout);     /* post-FFN dropout */
    for (int i = 0; i < S * D; ++i) b->r2.data[i] = b->y1.data[i] + b->ff_out.data[i];
    layernorm_forward(&b->ln2, b->r2, y);
}
void block_backward(Block *b, const Mat dy, Mat dx) {
    const int S = dy.rows, D = b->d;
    ensure(&b->s_dr2, S, D); ensure(&b->s_dy1, S, D); ensure(&b->s_dr1, S, D);
    ensure(&b->s_dtmp, S, D); ensure(&b->s_dd, S, D);
    Mat dr2 = b->s_dr2, dy1 = b->s_dy1, dr1 = b->s_dr1, dtmp = b->s_dtmp, dd = b->s_dd;
    int drop = dropout_on();
    if (g_prenorm) {
        /* reverse of the pre-norm forward above */
        if (drop) { for (int i = 0; i < S * D; ++i) dd.data[i] = dy.data[i] * b->drop2.data[i];
                    ffn_backward(&b->ff, dd, dr2); }             /* dr2 = d(n2) */
        else        ffn_backward(&b->ff, dy, dr2);
        layernorm_backward(&b->ln2, dr2, dtmp);                  /* dtmp = d(y1) via LN2 */
        for (int i = 0; i < S * D; ++i) dy1.data[i] = dy.data[i] + dtmp.data[i];   /* + residual */
        if (drop) { for (int i = 0; i < S * D; ++i) dd.data[i] = dy1.data[i] * b->drop1.data[i];
                    mha_backward(&b->attn, dd, dr1); }           /* dr1 = d(n1) */
        else        mha_backward(&b->attn, dy1, dr1);
        layernorm_backward(&b->ln1, dr1, dtmp);                  /* dtmp = d(x) via LN1 */
        for (int i = 0; i < S * D; ++i) dx.data[i] = dy1.data[i] + dtmp.data[i];   /* + residual */
        return;
    }
    /* Reverse of the post-norm forward, top to bottom. Each residual y=a+f(a)
     * routes gradient BOTH straight through the skip connection AND through the
     * sublayer, hence the "+= dr2" / "dr1 + dtmp" adds below. Dropout is undone
     * by multiplying by the saved keep-mask (same per-element factor as forward). */
    layernorm_backward(&b->ln2, dy, dr2);            /* y = LN2(r2)  =>  dr2 = dL/dr2 */
    if (drop) { for (int i = 0; i < S * D; ++i) dd.data[i] = dr2.data[i] * b->drop2.data[i];
                ffn_backward(&b->ff, dd, dy1); }     /* r2 = y1 + dropout(ff(y1)) */
    else        ffn_backward(&b->ff, dr2, dy1);
    for (int i = 0; i < S * D; ++i) dy1.data[i] += dr2.data[i];   /* + FFN residual skip */
    layernorm_backward(&b->ln1, dy1, dr1);           /* y1 = LN1(r1)  =>  dr1 = dL/dr1 */
    if (drop) { for (int i = 0; i < S * D; ++i) dd.data[i] = dr1.data[i] * b->drop1.data[i];
                mha_backward(&b->attn, dd, dtmp); }   /* r1 = x + dropout(attn(x)) */
    else        mha_backward(&b->attn, dr1, dtmp);
    for (int i = 0; i < S * D; ++i) dx.data[i] = dr1.data[i] + dtmp.data[i];   /* attn skip + attn path */
}
void block_flush(Block *b) { mha_flush(&b->attn); ffn_flush(&b->ff); }
void block_free(Block *b) {
    mha_free(&b->attn); layernorm_free(&b->ln1); layernorm_free(&b->ln2); ffn_free(&b->ff);
    mat_free(&b->attn_out); mat_free(&b->r1); mat_free(&b->y1); mat_free(&b->ff_out); mat_free(&b->r2);
    mat_free(&b->drop1); mat_free(&b->drop2);
    mat_free(&b->s_dr2); mat_free(&b->s_dy1); mat_free(&b->s_dr1); mat_free(&b->s_dtmp); mat_free(&b->s_dd);
}
