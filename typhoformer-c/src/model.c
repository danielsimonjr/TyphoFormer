/*
 * model.c — PGF fusion, spatio-temporal encoder, autoregressive decoder,
 * and the full TyphoFormer model with its loss.
 */
#include "model.h"
#include "data.h"   /* TF_NBR_K / TF_NBR_NF for co-active spatial attention */

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const Mat NULLMAT = {0, 0, NULL};

/* Decoder delta mode (predict displacement from the seed). Off by default. */
static int g_delta = 0;
void model_set_delta(int on) { g_delta = on; }
int  model_delta(void) { return g_delta; }

/* Constant-velocity decoder (2nd-order: anchor at CLIPER, learn the curvature
 * correction). Off by default; takes precedence over delta when both are set. */
static int g_cv = 0;
void model_set_cv(int on) { g_cv = on; }

/* Motion-aligned frame for the cv correction (parameter-free; changes NO
 * parameter layout, so checkpoints are interchangeable — but eval must use the
 * same flag or the correction is interpreted in the wrong frame). The MLP's
 * output is read as (along-track, cross-track) components and rotated into
 * lat/lon by the current velocity direction u = v/|v|:
 *     c_glob = R(u)·c_local,   R = [[u0, −u1], [u1, u0]]
 * so the learned curvature is rotation-invariant — "turn left by x" means the
 * same thing whatever the storm's heading, which is how recurvature actually
 * works. When |v| ≈ 0 the frame is undefined; we fall back to the identity
 * rotation (and treat the frame as constant in the backward). Off by default. */
static int g_rotframe = 0;
void model_set_rotframe(int on) { g_rotframe = on; }
#define ROTF_EPS 1e-6f

/* Teacher-forcing probability for the cv rollout (training only; eval always
 * rolls out autoregressively). With probability p, a step's OUTPUT state is
 * replaced by ground truth before the next step consumes it: p ← Y_s and the
 * truth-implied velocity v ← Y_s − p_entry. Forced steps cut the recurrence
 * gradient (truth is a constant), which is exactly the intended effect: early
 * in training the rollout does not backpropagate through its own noisy
 * predictions. The trainer anneals p → 0 (--tf=E). 0 = off (default). */
static float g_tf_prob = 0.0f;
void model_set_tf_prob(float p) { g_tf_prob = p; }

/* Recurrent (GRU) decoder correction. Its rollout branch anchors at
 * constant-velocity itself (zero-init fc_out), so it needs no separate g_cv.
 * Off by default. */
static int g_gru = 0;
void model_set_gru(int on) { g_gru = on; }

/* Decoder cross-attention over the encoder sequence (declared here so the
 * setter below sees it). */
static int g_xattn = 0;

/* Cross-attention decoder correction (anchors at cv in its own branch) and makes
 * the encoder expose its pre-pool sequence. Off by default. */
void model_set_xattn(int on) { g_xattn = on; }

/* Encoder architecture options (read at encoder_new time).
 * g_no_spatial defaults ON: the paper's N=1 "spatial" blocks are degenerate
 * (each position attends only to itself; their Q/K never train) and FINDINGS §7
 * measured dropping them as accuracy-neutral — so the default encoder skips
 * them for ~2× less encoder compute. `--spatial` restores the paper-faithful
 * architecture (required to load checkpoints trained with spatial blocks). */
static int g_no_spatial = 1, g_posenc = 0, g_pool_last = 0, g_co_spatial = 0;
void model_set_no_spatial(int on) { g_no_spatial = on; }
void model_set_posenc(int on)     { g_posenc = on; }
void model_set_pool_last(int on)  { g_pool_last = on; }
void model_set_co_spatial(int on) { g_co_spatial = on; }

Config config_default(void) {
    Config c;
    c.d_num = 14; c.d_text = 384; c.d_model = 256; c.out_dim = 2;
    c.in_len = 12; c.pred_len = 1; c.d_ff = 1024; c.n_heads = 4; c.n_layers = 3;
    return c;
}

/* =====================================================================
 * PGF
 * ===================================================================== */
PGF pgf_new(const Config *c, ParamList *pl) {
    PGF p; p.d_num = c->d_num; p.d_text = c->d_text; p.d_model = c->d_model;
    p.fc_gate   = linear_new(c->d_num + c->d_text, c->d_model, pl, "pgf.gate");
    p.proj_num  = linear_new(c->d_num,  c->d_model, pl, "pgf.proj_num");
    p.proj_text = linear_new(c->d_text, c->d_model, pl, "pgf.proj_text");
    p.cat = NULLMAT; p.gate = NULLMAT; p.xn = NULLMAT; p.xt = NULLMAT;
    p.s_dz = NULLMAT; p.s_dxn = NULLMAT; p.s_dxt = NULLMAT;
    return p;
}
/* Lazily (re)allocate *m to exactly r×cc floats, reusing the buffer when the
 * shape already matches. Lets every module keep persistent scratch matrices that
 * grow to the current batch/sequence length without reallocating each call. */
static void ensure(Mat *m, int r, int cc) {
    if (m->data == NULL || m->rows != r || m->cols != cc) { if (m->data) mat_free(m); *m = mat_new(r, cc); }
}
/* Forward fusion. Builds cat=[xnum|xtext] per step, computes the sigmoid gate
 * from it, projects each stream to D, then blends:
 *   x~[i] = g[i]·xn[i] + (1-g[i])·xt[i]      (element-wise over all T·D entries).
 * cat/gate/xn/xt are all cached because the backward needs them. */
void pgf_forward(PGF *p, const Mat xnum, const Mat xtext, Mat xtilde) {
    const int T = xnum.rows, D = p->d_model;
    ensure(&p->cat, T, p->d_num + p->d_text);
    ensure(&p->gate, T, D); ensure(&p->xn, T, D); ensure(&p->xt, T, D);
    for (int i = 0; i < T; ++i) {
        /* copy the numerical block then the text block into row i of cat */
        memcpy(&p->cat.data[i * (p->d_num + p->d_text)], &xnum.data[i * p->d_num],
               p->d_num * sizeof(float));
        memcpy(&p->cat.data[i * (p->d_num + p->d_text) + p->d_num], &xtext.data[i * p->d_text],
               p->d_text * sizeof(float));
    }
    linear_forward(&p->fc_gate, p->cat, p->gate);   /* gate logits [T,D] */
    mat_sigmoid(p->gate);                           /* -> gate in (0,1), cached */
    linear_forward(&p->proj_num, xnum, p->xn);      /* xn = proj_num(xnum)  [T,D] */
    linear_forward(&p->proj_text, xtext, p->xt);    /* xt = proj_text(xtext) [T,D] */
    for (int i = 0; i < T * D; ++i)
        xtilde.data[i] = p->gate.data[i] * p->xn.data[i] + (1.0f - p->gate.data[i]) * p->xt.data[i];
}
/* Backward of the blend. Given dL/dx~ (and an optional direct gate penalty grad):
 *   x~ = g·xn + (1-g)·xt   =>
 *     d/dg   = dx~·(xn - xt)          (+ dgate_pen from the loss's gate penalty)
 *     d/dxn  = dx~·g
 *     d/dxt  = dx~·(1-g)
 *   the gate passed through sigmoid, so dz (grad on the logit) = dg·g·(1-g).
 * The three streams (xnum, xtext) are raw data inputs, so no input-grad is
 * requested from the Linears (NULLMAT dx) — only their weight grads accumulate. */
void pgf_backward(PGF *p, const Mat dxtilde, const Mat dgate_pen) {
    const int T = dxtilde.rows, D = p->d_model;
    ensure(&p->s_dz, T, D); ensure(&p->s_dxn, T, D); ensure(&p->s_dxt, T, D);
    Mat dz = p->s_dz, dxn = p->s_dxn, dxt = p->s_dxt;
    for (int i = 0; i < T * D; ++i) {
        float g = p->gate.data[i];
        float dgate = dxtilde.data[i] * (p->xn.data[i] - p->xt.data[i]);
        if (dgate_pen.data) dgate += dgate_pen.data[i];   /* gate-collapse penalty grad */
        dz.data[i]  = dgate * g * (1.0f - g);        /* through sigmoid */
        dxn.data[i] = dxtilde.data[i] * g;
        dxt.data[i] = dxtilde.data[i] * (1.0f - g);
    }
    linear_backward(&p->fc_gate, dz, NULLMAT);       /* inputs are data: no dx */
    linear_backward(&p->proj_num, dxn, NULLMAT);
    linear_backward(&p->proj_text, dxt, NULLMAT);
}
void pgf_free(PGF *p) {
    linear_free(&p->fc_gate); linear_free(&p->proj_num); linear_free(&p->proj_text);
    mat_free(&p->cat); mat_free(&p->gate); mat_free(&p->xn); mat_free(&p->xt);
    mat_free(&p->s_dz); mat_free(&p->s_dxn); mat_free(&p->s_dxt);
}

/* =====================================================================
 * TimeMix
 * ===================================================================== */
/* Init the mix matrix A and bias c with the same uniform(-1/in, 1/in) scheme
 * nn.Linear uses (fan-in = in_steps), and register both with the ParamList so
 * they train and gradient-check like any other weight. */
TimeMix timemix_new(int in_steps, int out_steps, ParamList *pl) {
    TimeMix t; t.in_steps = in_steps; t.out_steps = out_steps;
    t.A = mat_new(out_steps, in_steps); t.dA = mat_new(out_steps, in_steps);
    t.c = (float *)calloc(out_steps, sizeof(float));
    t.dc = (float *)calloc(out_steps, sizeof(float));
    float k = 1.0f / (float)in_steps;
    for (int i = 0; i < out_steps * in_steps; ++i) t.A.data[i] = nn_uniform(-k, k);
    for (int i = 0; i < out_steps; ++i) t.c[i] = nn_uniform(-k, k);   /* bias like nn.Linear */
    t.xcache = NULLMAT; t.s_tmp = NULLMAT;
    plist_add(pl, t.A.data, t.dA.data, out_steps * in_steps, "timemix.A");
    plist_add(pl, t.c, t.dc, out_steps, "timemix.c");
    return t;
}
/* y[o,d] = Σ_i A[o,i]·x[i,d] + c[o]. The matmul mixes over the time axis (rows);
 * the bias broadcasts across the D feature columns. x is cached for the backward. */
void timemix_forward(TimeMix *t, const Mat x, Mat y) {
    ensure(&t->xcache, x.rows, x.cols); mat_copy(t->xcache, x);
    mat_matmul(t->A, x, y);                          /* [out,in] @ [in,D] */
    for (int o = 0; o < t->out_steps; ++o)
        for (int d = 0; d < y.cols; ++d) y.data[o * y.cols + d] += t->c[o];
}
/* Backward of y = A·x + c:
 *   dA[o,i] += Σ_d dy[o,d]·x[i,d]   (= dy · x^T)
 *   dc[o]   += Σ_d dy[o,d]          (bias summed over the feature axis)
 *   dx      =  A^T · dy             (grad back into the [in,D] sequence)
 * Grads accumulate (+=) so multiple calls in a batch add up. */
void timemix_backward(TimeMix *t, const Mat dy, Mat dx) {
    const int D = dy.cols;
    ensure(&t->s_tmp, t->out_steps, t->in_steps);
    Mat tmp = t->s_tmp;
    mat_matmul_bt(dy, t->xcache, tmp);               /* dA = dy @ x^T */
    for (int i = 0; i < t->out_steps * t->in_steps; ++i) t->dA.data[i] += tmp.data[i];
    for (int o = 0; o < t->out_steps; ++o) {
        float s = 0.0f; for (int d = 0; d < D; ++d) s += dy.data[o * D + d];
        t->dc[o] += s;
    }
    if (dx.data) mat_matmul_atb(t->A, dy, dx);        /* dx = A^T @ dy */
}
void timemix_free(TimeMix *t) {
    mat_free(&t->A); mat_free(&t->dA); free(t->c); free(t->dc);
    mat_free(&t->xcache); mat_free(&t->s_tmp);
}

/* =====================================================================
 * Encoder
 * ===================================================================== */
/* Build the encoder. NOTE the registration ORDER of parameters (input_proj,
 * posenc, output_proj, then per-layer temporal[/spatial] blocks, then tmix)
 * defines the flat parameter layout a checkpoint must match — the architecture
 * switches are snapshotted here from the g_* globals so a saved model is only
 * loadable with the same options. */
Encoder encoder_new(const Config *c, ParamList *pl) {
    Encoder e; e.cfg = *c;
    e.use_spatial = !g_no_spatial; e.use_posenc = g_posenc; e.pool_last = g_pool_last;
    e.input_proj  = linear_new(c->d_model, c->d_model, pl, "enc.input_proj");
    /* learned positional encoding, registered right after input_proj. Unlike a
     * fixed sinusoid this is a trainable [T,D] table added to the projected
     * input so the attention can tell time steps apart. */
    e.posenc = NULLMAT; e.dposenc = NULLMAT;
    if (e.use_posenc) {
        e.posenc = mat_new(c->in_len, c->d_model); e.dposenc = mat_new(c->in_len, c->d_model);
        float k = 0.02f;                              /* small init, like a PE table */
        for (int i = 0; i < c->in_len * c->d_model; ++i) e.posenc.data[i] = nn_uniform(-k, k);
        plist_add(pl, e.posenc.data, e.dposenc.data, c->in_len * c->d_model, "enc.posenc");
    }
    e.output_proj = linear_new(c->d_model, c->d_model, pl, "enc.output_proj");
    e.temporal = (Block *)malloc(c->n_layers * sizeof(Block));
    e.spatial  = e.use_spatial ? (Block *)malloc(c->n_layers * sizeof(Block)) : NULL;
    for (int l = 0; l < c->n_layers; ++l) {
        /* temporal block: self_only=0 -> full self-attention across the T steps.
         * This is the path the optional ALiBi time bias (nn_set_timebias) acts on:
         * it subtracts slope_h·|i-j| from the scores so nearer time steps attend
         * more strongly, giving attention a built-in sense of temporal distance.
         * spatial block: self_only=1 -> each position attends only to itself
         * (a placeholder for multi-node attention; degenerate with a single storm). */
        e.temporal[l] = block_new(c->d_model, c->n_heads, c->d_ff, 0, pl, "enc.temporal");
        if (e.use_spatial) e.spatial[l] = block_new(c->d_model, c->n_heads, c->d_ff, 1, pl, "enc.spatial");
    }
    e.tmix = timemix_new(c->in_len, 1, pl);
    e.use_encseq = g_xattn;                          /* expose the pre-pool sequence for --xattn */
    e.encseq = NULLMAT; e.dencseq = NULLMAT;
    if (e.use_encseq) { e.encseq = mat_new(c->in_len, c->d_model); e.dencseq = mat_new(c->in_len, c->d_model); }
    e.b0  = mat_new(c->in_len, c->d_model); e.b1  = mat_new(c->in_len, c->d_model);
    e.db0 = mat_new(c->in_len, c->d_model); e.db1 = mat_new(c->in_len, c->d_model);
    e.tmid = mat_new(1, c->d_model); e.dtmid = mat_new(1, c->d_model);
    return e;
}
/* Forward: x~[T,D] -> henc[1,D]. cur/nxt are two [T,D] buffers ping-ponged so
 * each block writes into the spare buffer and we swap, avoiding per-layer allocs. */
void encoder_forward(Encoder *e, const Mat xtilde, Mat henc) {
    const int L = e->cfg.n_layers, T = e->cfg.in_len, D = e->cfg.d_model;
    Mat cur = e->b0, nxt = e->b1, tmp;
    linear_forward(&e->input_proj, xtilde, cur);
    if (e->use_posenc) for (int i = 0; i < T * D; ++i) cur.data[i] += e->posenc.data[i];  /* + learned PE */
    /* alternate temporal then (optionally) spatial WITHIN each layer (not two
     * separate stacks): every layer mixes over time, then optionally over nodes. */
    for (int l = 0; l < L; ++l) {
        block_forward(&e->temporal[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp;
        if (e->use_spatial) { block_forward(&e->spatial[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp; }
    }
    if (e->use_encseq) mat_copy(e->encseq, cur);   /* expose the pre-pool sequence (--xattn memory) */
    /* pool the [T,D] sequence to a single [1,D] summary tmid: either the last
     * step (causal "most recent state") or the learned TimeMix weighted average. */
    if (e->pool_last) memcpy(e->tmid.data, &cur.data[(T - 1) * D], D * sizeof(float)); /* last step */
    else              timemix_forward(&e->tmix, cur, e->tmid);                          /* [1,D] */
    linear_forward(&e->output_proj, e->tmid, henc);   /* [1,D] context */
}
/* Backward: mirror of encoder_forward, unwinding in reverse.
 *   output_proj^T -> dtmid, then "un-pool" dtmid back to a [T,D] grad (scatter to
 *   the last row for pool_last, or timemix_backward for the learned average),
 *   add the cross-attention memory grad (dencseq) if --xattn, run the blocks in
 *   reverse (spatial before temporal — opposite of the forward order), fold the
 *   grad into the positional table, and finally input_proj^T -> dxtilde. */
void encoder_backward(Encoder *e, const Mat dhenc, Mat dxtilde) {
    const int L = e->cfg.n_layers, T = e->cfg.in_len, D = e->cfg.d_model;
    linear_backward(&e->output_proj, dhenc, e->dtmid);
    Mat cur = e->db0, nxt = e->db1, tmp;
    if (e->pool_last) { mat_zero(cur); memcpy(&cur.data[(T - 1) * D], e->dtmid.data, D * sizeof(float)); }
    else              timemix_backward(&e->tmix, e->dtmid, cur);
    if (e->use_encseq) for (int i = 0; i < T * D; ++i) cur.data[i] += e->dencseq.data[i];  /* + xattn grad */
    for (int l = L - 1; l >= 0; --l) {
        if (e->use_spatial) { block_backward(&e->spatial[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp; }
        block_backward(&e->temporal[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp;
    }
    if (e->use_posenc) for (int i = 0; i < T * D; ++i) e->dposenc.data[i] += cur.data[i]; /* PE added, so its grad = cur */
    linear_backward(&e->input_proj, cur, dxtilde);
}
void encoder_free(Encoder *e) {
    linear_free(&e->input_proj); linear_free(&e->output_proj);
    for (int l = 0; l < e->cfg.n_layers; ++l) {
        block_free(&e->temporal[l]);
        if (e->use_spatial) block_free(&e->spatial[l]);
    }
    free(e->temporal); free(e->spatial); timemix_free(&e->tmix);
    mat_free(&e->posenc); mat_free(&e->dposenc);
    mat_free(&e->encseq); mat_free(&e->dencseq);
    mat_free(&e->b0); mat_free(&e->b1); mat_free(&e->db0); mat_free(&e->db1);
    mat_free(&e->tmid); mat_free(&e->dtmid);
}

/* =====================================================================
 * Decoder (autoregressive; training uses steps == 1)
 * ===================================================================== */
/* Allocate exactly one correction head according to the active flag. The heads
 * are mutually exclusive and their final layer is zero-initialised in every
 * "anchored" mode (delta/cv/gru/xattn) so the untrained rollout emits the
 * physical baseline and gradient descent only has to learn the residual. */
Decoder decoder_new(const Config *c, ParamList *pl) {
    Decoder d; d.hidden = c->d_model; d.out = c->out_dim; d.max_steps = c->pred_len;
    d.use_cv = g_cv; d.use_gru = g_gru; d.use_xattn = g_xattn;
    /* rotframe applies to the plain cv head only (out_dim 2 by construction). */
    d.use_rotframe = (g_rotframe && g_cv && !g_gru && !g_xattn && c->out_dim == 2);
    d.frc = d.use_rotframe ? (Mat *)calloc(c->pred_len, sizeof(Mat)) : NULL;
    d.tf_y = NULL; d.forced = (int *)calloc(c->pred_len, sizeof(int));
    if (g_xattn) {
        /* per-step context = cross-attention([p;v] query over encoder sequence);
         * fed to the cv MLP in place of the static pooled context. */
        d.xattn = xattn_new(2 * c->out_dim, c->d_model, c->pred_len, pl, "dec.xattn");
        d.fc1 = linear_new(c->d_model + 2 * c->out_dim, c->d_model, pl, "dec.fc1");
        d.fc2 = linear_new(c->d_model, c->out_dim, pl, "dec.fc2");
        memset(d.fc2.W.data, 0, (size_t)d.fc2.out * d.fc2.in * sizeof(float));   /* start at cv */
        memset(d.fc2.b, 0, (size_t)d.fc2.out * sizeof(float));
    } else if (g_gru) {
        /* GRU correction: input is [p; v] (2*out); hidden = d_model, initialised
         * from the encoder context. fc_out projects hidden -> out, zero-init so
         * the untrained model still starts at constant-velocity. */
        d.gru    = gru_new(2 * c->out_dim, c->d_model, c->pred_len, pl, "dec.gru");
        d.fc_out = linear_new(c->d_model, c->out_dim, pl, "dec.fc_out");
        memset(d.fc_out.W.data, 0, (size_t)d.fc_out.out * d.fc_out.in * sizeof(float));
        memset(d.fc_out.b, 0, (size_t)d.fc_out.out * sizeof(float));
    } else {
        /* cv mode also feeds the current velocity, so fc1's input is [h; y_prev; v]. */
        int fc1_in = c->d_model + c->out_dim + (g_cv ? c->out_dim : 0);
        d.fc1 = linear_new(fc1_in, c->d_model, pl, "dec.fc1");
        d.fc2 = linear_new(c->d_model, c->out_dim, pl, "dec.fc2");
        if (g_delta || g_cv) {          /* start at persistence (delta) / const-velocity (cv) */
            memset(d.fc2.W.data, 0, (size_t)d.fc2.out * d.fc2.in * sizeof(float));
            memset(d.fc2.b, 0, (size_t)d.fc2.out * sizeof(float));
        }
    }
    /* Per-step caches: one Mat slot per rollout step so the whole autoregressive
     * sequence can be replayed in the backward pass. All scratch starts empty
     * (NULLMAT) and is grown lazily by ensure() on first use. s_dmem is the only
     * one sized eagerly, since it must exist to receive the xattn memory grad. */
    d.zc  = (Mat *)calloc(d.max_steps, sizeof(Mat));
    d.h1c = (Mat *)calloc(d.max_steps, sizeof(Mat));
    d.ac  = (Mat *)calloc(d.max_steps, sizeof(Mat));
    d.s_yt = NULLMAT; d.s_a = NULLMAT; d.s_ytn = NULLMAT; d.s_v = NULLMAT;
    d.s_da = NULLMAT; d.s_dh1 = NULLMAT; d.s_dz = NULLMAT;
    d.s_dyt = NULLMAT; d.s_dynext = NULLMAT; d.s_dvnext = NULLMAT; d.s_dhacc = NULLMAT;
    d.s_hid0 = NULLMAT; d.s_dhid = NULLMAT; d.s_dhc = NULLMAT; d.s_dx2 = NULLMAT;
    d.s_xq = NULLMAT; d.s_dmem = g_xattn ? mat_new(c->in_len, c->d_model) : NULLMAT;
    return d;
}
/* Autoregressive rollout. State threads across steps as p (last position) and,
 * in the second-order heads, v (velocity). Every step writes preds[s] = y and
 * then sets p := y so the next step consumes its own output. The four branches
 * differ only in how the correction c is produced and what the anchor is. */
void decoder_forward(Decoder *d, const Mat henc, const Mat yprev, const Mat vseed, int steps, Mat preds) {
    const int H = d->hidden, O = d->out;
    assert(steps <= d->max_steps);
    if (d->use_xattn) {
        /* Constant-velocity rollout whose context is per-step cross-attention over
         * the encoder sequence (memory set by the model via xattn_set_memory).
         * Each step forms query = [p ; v], attends over encseq to get ctx, and
         * feeds fc1 input [ctx ; p ; v]; fc2 zero-init -> ŷ = p+v at init (cv).
         * ctx replaces the STATIC pooled henc — the context is recomputed every
         * step from the current state, unlike the cv/gru heads. */
        ensure(&d->s_yt, 1, O); ensure(&d->s_a, 1, H); ensure(&d->s_ytn, 1, O);
        ensure(&d->s_v, 1, O); ensure(&d->s_hid0, 1, H); ensure(&d->s_xq, 1, 2 * O);
        Mat p = d->s_yt, a = d->s_a, c = d->s_ytn, v = d->s_v, ctx = d->s_hid0, xq = d->s_xq;
        mat_copy(p, yprev);
        if (vseed.data) mat_copy(v, vseed); else mat_zero(v);   /* seed velocity (0 if none) */
        for (int s = 0; s < steps; ++s) {
            ensure(&d->zc[s], 1, H + 2 * O); ensure(&d->h1c[s], 1, H); ensure(&d->ac[s], 1, H);
            memcpy(&xq.data[0], p.data, O * sizeof(float));        /* query = [p ; v] */
            memcpy(&xq.data[O], v.data, O * sizeof(float));
            xattn_forward(&d->xattn, s, xq, ctx);                  /* context over encoder seq */
            memcpy(&d->zc[s].data[0],       ctx.data, H * sizeof(float));  /* z = [ctx ; p ; v] */
            memcpy(&d->zc[s].data[H],       p.data,   O * sizeof(float));
            memcpy(&d->zc[s].data[H + O],   v.data,   O * sizeof(float));
            linear_forward(&d->fc1, d->zc[s], d->h1c[s]);
            mat_copy(a, d->h1c[s]); mat_relu(a); mat_copy(d->ac[s], a);  /* a = relu(fc1(z)), cached */
            linear_forward(&d->fc2, a, c);                              /* curvature correction c */
            for (int i = 0; i < O; ++i) {
                float y = p.data[i] + v.data[i] + c.data[i];   /* cv anchor + correction */
                v.data[i] += c.data[i];                        /* v_{s+1} = v_s + c */
                p.data[i]  = y;                                /* p_{s+1} = y */
                preds.data[s * O + i] = y;
            }
        }
        return;
    }
    if (d->use_gru) {
        /* Recurrent constant-velocity rollout. Same anchor arithmetic as cv
         * (ŷ = p + v + c ; v += c), but the correction is
         *   c = fc_out( GRU([p;v], hid_{s-1}) )
         * with the hidden state carried across steps (hid_0 = encoder context).
         * The GRU gives the multi-step rollout real memory that a memoryless MLP
         * lacks; fc_out zero-init keeps the untrained anchor at cv. ac[s] doubles
         * as the cached hidden state hid_s that feeds the next step. */
        ensure(&d->s_yt, 1, O); ensure(&d->s_ytn, 1, O); ensure(&d->s_v, 1, O); ensure(&d->s_hid0, 1, H);
        Mat p = d->s_yt, c = d->s_ytn, v = d->s_v, hid_prev = d->s_hid0;
        mat_copy(p, yprev);
        if (vseed.data) mat_copy(v, vseed); else mat_zero(v);
        mat_copy(hid_prev, henc);                      /* hid_0 = encoder context */
        for (int s = 0; s < steps; ++s) {
            ensure(&d->zc[s], 1, 2 * O); ensure(&d->ac[s], 1, H);
            memcpy(&d->zc[s].data[0], p.data, O * sizeof(float));   /* x_s = [p ; v] */
            memcpy(&d->zc[s].data[O], v.data, O * sizeof(float));
            gru_forward(&d->gru, s, d->zc[s], hid_prev, d->ac[s]);  /* hid_s -> ac[s] */
            linear_forward(&d->fc_out, d->ac[s], c);               /* correction */
            for (int i = 0; i < O; ++i) {
                float y = p.data[i] + v.data[i] + c.data[i];
                v.data[i] += c.data[i];
                p.data[i]  = y;
                preds.data[s * O + i] = y;
            }
            hid_prev = d->ac[s];                        /* carry hidden state to step s+1 */
        }
        return;
    }
    if (d->use_cv) {
        /* Constant-velocity rollout: state (p = y_prev, v = velocity).
         *   z    = [h ; p ; v]
         *   c    = fc2(relu(fc1(z)))            (learned curvature correction)
         *   ŷ_s  = p + v + c                    (== p at next step)
         *   v   += c,  p = ŷ_s                  (v_{s+1}=v_s+c ; p_{s+1}=ŷ_s)
         * fc2 zero-init -> ŷ starts at p+v = constant-velocity extrapolation. */
        const int Z = H + 2 * O;
        ensure(&d->s_yt, 1, O); ensure(&d->s_a, 1, H); ensure(&d->s_ytn, 1, O); ensure(&d->s_v, 1, O);
        Mat p = d->s_yt, a = d->s_a, c = d->s_ytn, v = d->s_v;
        mat_copy(p, yprev);
        if (vseed.data) mat_copy(v, vseed); else mat_zero(v);
        for (int s = 0; s < steps; ++s) {
            ensure(&d->zc[s], 1, Z); ensure(&d->h1c[s], 1, H); ensure(&d->ac[s], 1, H);
            memcpy(&d->zc[s].data[0],     henc.data, H * sizeof(float));   /* z = [h ; p ; v] */
            memcpy(&d->zc[s].data[H],     p.data,    O * sizeof(float));
            memcpy(&d->zc[s].data[H + O], v.data,    O * sizeof(float));
            linear_forward(&d->fc1, d->zc[s], d->h1c[s]);
            mat_copy(a, d->h1c[s]); mat_relu(a);
            mat_copy(d->ac[s], a);
            linear_forward(&d->fc2, a, c);                                /* correction */
            if (d->use_rotframe) {
                /* Read c as (along-track, cross-track) and rotate into lat/lon
                 * by the CURRENT velocity direction u = v/|v| (pre-update v_s):
                 *   c_glob = R(u)·c_local,  R = [[u0,−u1],[u1,u0]].
                 * |v|≈0 → identity frame (marked by sp=0 in the cache, so the
                 * backward treats the frame as constant there). Cache what the
                 * backward needs: u, |v|, and the local correction. */
                ensure(&d->frc[s], 1, 5);
                float v0 = v.data[0], v1 = v.data[1];
                float sp = sqrtf(v0 * v0 + v1 * v1);
                float u0 = 1.0f, u1 = 0.0f;
                if (sp > ROTF_EPS) { u0 = v0 / sp; u1 = v1 / sp; } else sp = 0.0f;
                float c0 = c.data[0], c1 = c.data[1];
                d->frc[s].data[0] = u0; d->frc[s].data[1] = u1; d->frc[s].data[2] = sp;
                d->frc[s].data[3] = c0; d->frc[s].data[4] = c1;
                c.data[0] = c0 * u0 - c1 * u1;                            /* R·c_local */
                c.data[1] = c0 * u1 + c1 * u0;
            }
            for (int i = 0; i < O; ++i) {
                float y = p.data[i] + v.data[i] + c.data[i];
                v.data[i] += c.data[i];
                p.data[i]  = y;
                preds.data[s * O + i] = y;
            }
            /* Teacher forcing (training only; d->tf_y is NULL otherwise): with
             * prob g_tf_prob, hand the NEXT step ground-truth state instead of
             * the model's own — position ← Y_s, velocity ← the truth-implied
             * step displacement Y_s − p_entry (p at this step's input, still in
             * zc). The prediction just emitted is untouched; only the rollout
             * state is corrected. forced[s] tells the backward that no gradient
             * flows into this step from later steps (truth is constant). */
            d->forced[s] = 0;
            if (d->tf_y && s + 1 < steps && nn_dropout_draw() < g_tf_prob) {
                d->forced[s] = 1;
                for (int i = 0; i < O; ++i) {
                    float p_entry = d->zc[s].data[H + i];
                    p.data[i] = d->tf_y[s * O + i];
                    v.data[i] = d->tf_y[s * O + i] - p_entry;
                }
            }
        }
        return;
    }
    /* First-order head (plain or delta). z = [h ; y_prev]; ytn = fc2(relu(fc1(z))).
     *   plain:  y_t = ytn                 (fc2 predicts the absolute next coord)
     *   delta:  y_t = y_{t-1} + ytn       (fc2 predicts a displacement; zero-init
     *           fc2 -> untrained model outputs persistence y_t = y_{t-1})
     * The static pooled context h is re-fed at every step (there is no velocity). */
    ensure(&d->s_yt, 1, O); ensure(&d->s_a, 1, H); ensure(&d->s_ytn, 1, O);
    Mat yt = d->s_yt, a = d->s_a, ytn = d->s_ytn;
    mat_copy(yt, yprev);
    for (int s = 0; s < steps; ++s) {
        ensure(&d->zc[s], 1, H + O); ensure(&d->h1c[s], 1, H); ensure(&d->ac[s], 1, H);
        memcpy(&d->zc[s].data[0], henc.data, H * sizeof(float));   /* z = [h ; y_prev] */
        memcpy(&d->zc[s].data[H], yt.data, O * sizeof(float));
        linear_forward(&d->fc1, d->zc[s], d->h1c[s]);
        mat_copy(a, d->h1c[s]); mat_relu(a);
        mat_copy(d->ac[s], a);
        linear_forward(&d->fc2, a, ytn);
        if (g_delta) { for (int i = 0; i < O; ++i) yt.data[i] += ytn.data[i]; }  /* y_t = y_{t-1}+Δ */
        else         mat_copy(yt, ytn);                                          /* y_t = Δ (absolute) */
        memcpy(&preds.data[s * O], yt.data, O * sizeof(float));
    }
}
/* Autoregressive backward: y_t is used both as this step's output and as the
 * next step's decoder input, so gradients from step s+1 flow back into it. */
void decoder_backward(Decoder *d, const Mat dpreds, Mat dhenc) {
    const int H = d->hidden, O = d->out, steps = dpreds.rows;
    if (d->use_xattn) {
        /* Backward of the cross-attention cv rollout. dp/dv threading matches cv;
         * fc1's context-grad (dz[0:H]) drives the per-step cross-attention backward,
         * whose query-grad adds into dp/dv and whose K/V-grad accumulates for the
         * memory. The pooled context is unused here, so dhenc is zero and all the
         * encoder gradient flows through the attended memory (s_dmem). */
        ensure(&d->s_da, 1, H); ensure(&d->s_dh1, 1, H); ensure(&d->s_dz, 1, H + 2 * O);
        ensure(&d->s_dyt, 1, O); ensure(&d->s_dynext, 1, O); ensure(&d->s_dvnext, 1, O);
        ensure(&d->s_dhc, 1, H); ensure(&d->s_dx2, 1, 2 * O);
        Mat da = d->s_da, dh1 = d->s_dh1, dz = d->s_dz, dc = d->s_dyt;
        Mat dpnext = d->s_dynext, dvnext = d->s_dvnext, dctx = d->s_dhc, dxq = d->s_dx2;
        mat_zero(dpnext); mat_zero(dvnext);
        for (int s = steps - 1; s >= 0; --s) {
            /* dc_s: c feeds pred[s] and (via v+=c, p=y) both p_{s+1} and v_{s+1},
             * whose accumulated grads arrived as dpnext/dvnext from step s+1. */
            for (int i = 0; i < O; ++i)
                dc.data[i] = dpreds.data[s * O + i] + dpnext.data[i] + dvnext.data[i];
            mat_copy(d->fc2.xcache, d->ac[s]); linear_backward(&d->fc2, dc, da);
            for (int i = 0; i < H; ++i) dh1.data[i] = (d->h1c[s].data[i] > 0.0f) ? da.data[i] : 0.0f;  /* relu' */
            mat_copy(d->fc1.xcache, d->zc[s]); linear_backward(&d->fc1, dh1, dz);
            for (int i = 0; i < H; ++i) dctx.data[i] = dz.data[i];       /* fc1's ctx-slot grad -> attn */
            xattn_backward_step(&d->xattn, s, dctx, dxq);                /* dxq = [dp_x ; dv_x] query grads */
            /* Thread the state grads back to step s-1. Same structure as the cv
             * recurrence but with the extra query-path terms dxq from the per-step
             * cross-attention (the query was [p;v]). dp_net/dv_net are fc1's grads
             * on the p/v slots of z. */
            for (int i = 0; i < O; ++i) {
                float gs = dpreds.data[s * O + i];
                float dp_old = dpnext.data[i], dv_old = dvnext.data[i];
                float dp_net = dz.data[H + i], dv_net = dz.data[H + O + i];
                dpnext.data[i] = gs + dp_old + dp_net + dxq.data[i];               /* dp_s */
                dvnext.data[i] = gs + dp_old + dv_old + dv_net + dxq.data[O + i];  /* dv_s */
            }
        }
        xattn_backward_memory(&d->xattn, d->s_dmem);   /* K/V grads -> encoder-sequence grad */
        mat_zero(dhenc);                               /* pooled context unused in xattn mode */
        return;
    }
    if (d->use_gru) {
        /* Backward of the recurrent cv rollout. dp/dv threading is identical to
         * the cv MLP; the correction's grad flows fc_out -> hid_s, is combined
         * with the hidden grad from step s+1, and back-propagates through the GRU
         * (which threads dhid into step s-1 and returns dx = [dp_net; dv_net]).
         * The initial hidden grad is the encoder-context grad. */
        ensure(&d->s_dyt, 1, O); ensure(&d->s_dynext, 1, O); ensure(&d->s_dvnext, 1, O);
        ensure(&d->s_dhid, 1, H); ensure(&d->s_dhc, 1, H); ensure(&d->s_dh1, 1, H); ensure(&d->s_dx2, 1, 2 * O);
        Mat dc = d->s_dyt, dpnext = d->s_dynext, dvnext = d->s_dvnext;
        Mat dhid_next = d->s_dhid, dhc = d->s_dhc, dhprev = d->s_dh1, dx2 = d->s_dx2;
        mat_zero(dpnext); mat_zero(dvnext); mat_zero(dhid_next);
        for (int s = steps - 1; s >= 0; --s) {
            for (int i = 0; i < O; ++i)
                dc.data[i] = dpreds.data[s * O + i] + dpnext.data[i] + dvnext.data[i];
            mat_copy(d->fc_out.xcache, d->ac[s]);
            linear_backward(&d->fc_out, dc, dhc);                  /* dc -> grad into hid_s */
            for (int i = 0; i < H; ++i) dhc.data[i] += dhid_next.data[i];   /* + grad from step s+1 */
            gru_backward(&d->gru, s, dhc, dx2, dhprev);            /* dx2=[dp_net; dv_net], dhprev */
            for (int i = 0; i < O; ++i) {
                float gs = dpreds.data[s * O + i];
                float dp_old = dpnext.data[i], dv_old = dvnext.data[i];
                dpnext.data[i] = gs + dp_old + dx2.data[i];         /* dp_s */
                dvnext.data[i] = gs + dp_old + dv_old + dx2.data[O + i];  /* dv_s */
            }
            mat_copy(dhid_next, dhprev);                            /* grad into hid_{s-1} */
        }
        memcpy(dhenc.data, dhid_next.data, H * sizeof(float));      /* hid_0 = encoder context */
        return;
    }
    if (d->use_cv) {
        /* Backward of the constant-velocity recurrence. Grads on the state
         * entering step s (dp, dv) and on the correction output (dc):
         *   dc_s = g_s + dp_next + dv_next          (c feeds pred, p_{s+1}, v_{s+1})
         *   dp_s = g_s + dp_next        + dp_net     (p feeds pred, p_{s+1}, network)
         *   dv_s = g_s + dp_next + dv_next + dv_net  (v feeds pred, p_{s+1}, v_{s+1}, net)
         * where [dhenc | dp_net | dv_net] = fc1 backward of z = [h; p; v]. */
        ensure(&d->s_da, 1, H); ensure(&d->s_dh1, 1, H); ensure(&d->s_dz, 1, H + 2 * O);
        ensure(&d->s_dyt, 1, O); ensure(&d->s_dynext, 1, O); ensure(&d->s_dvnext, 1, O);
        ensure(&d->s_dhacc, 1, H);
        Mat da = d->s_da, dh1 = d->s_dh1, dz = d->s_dz;
        Mat dc = d->s_dyt, dpnext = d->s_dynext, dvnext = d->s_dvnext;
        mat_zero(d->s_dhacc); mat_zero(dpnext); mat_zero(dvnext);
        for (int s = steps - 1; s >= 0; --s) {
            /* Teacher-forced step: its produced state was replaced by truth, so
             * no gradient arrives from later steps — cut the recurrence here. */
            if (d->forced[s]) { mat_zero(dpnext); mat_zero(dvnext); }
            /* dc is the grad on the GLOBAL correction c_glob (it feeds pred_s,
             * p_{s+1}, v_{s+1}). */
            for (int i = 0; i < O; ++i)
                dc.data[i] = dpreds.data[s * O + i] + dpnext.data[i] + dvnext.data[i];
            float dvf0 = 0.0f, dvf1 = 0.0f;          /* frame's extra grad into v_s */
            if (d->use_rotframe) {
                /* Reverse c_glob = R(u(v))·c_local. Two paths:
                 *   (1) into the MLP:   dc_local = Rᵀ·dc_glob
                 *   (2) into v via u:   dv += (∂c_glob/∂v)ᵀ·dc_glob
                 * With J = ∂u/∂v = (I − u·uᵀ)/|v| (symmetric) and the 90°
                 * rotation P = [[0,−1],[1,0]] (Pᵀ = −P):
                 *   ∂c_glob/∂v = c0·J + c1·P·J
                 *   dv_frame   = J·w,  w = c0·dc_glob − c1·P·dc_glob.
                 * Identity frame (|v|≈0, sp stored as 0): both reduce to
                 * dc_local = dc_glob, dv_frame = 0. */
                const float *fr = d->frc[s].data;
                float u0 = fr[0], u1 = fr[1], sp = fr[2], c0 = fr[3], c1 = fr[4];
                float g0 = dc.data[0], g1 = dc.data[1];
                if (sp > 0.0f) {
                    float w0 = c0 * g0 + c1 * g1;    /* w = c0·g − c1·P·g ; P·g = (−g1, g0) */
                    float w1 = c0 * g1 - c1 * g0;
                    float udw = u0 * w0 + u1 * w1;   /* J·w = (w − u(u·w))/|v| */
                    dvf0 = (w0 - u0 * udw) / sp;
                    dvf1 = (w1 - u1 * udw) / sp;
                }
                dc.data[0] =  u0 * g0 + u1 * g1;     /* Rᵀ·dc_glob → grad on c_local */
                dc.data[1] = -u1 * g0 + u0 * g1;
            }
            mat_copy(d->fc2.xcache, d->ac[s]);
            linear_backward(&d->fc2, dc, da);
            for (int i = 0; i < H; ++i) dh1.data[i] = (d->h1c[s].data[i] > 0.0f) ? da.data[i] : 0.0f;
            mat_copy(d->fc1.xcache, d->zc[s]);
            linear_backward(&d->fc1, dh1, dz);
            for (int i = 0; i < H; ++i) d->s_dhacc.data[i] += dz.data[i];        /* h feeds every step */
            for (int i = 0; i < O; ++i) {
                float gs = dpreds.data[s * O + i];
                float dp_old = dpnext.data[i], dv_old = dvnext.data[i];
                float dp_net = dz.data[H + i], dv_net = dz.data[H + O + i];
                float dv_frame = (i == 0) ? dvf0 : dvf1;                 /* rotframe's u(v_s) path */
                dpnext.data[i] = gs + dp_old + dp_net;                   /* dp_s */
                dvnext.data[i] = gs + dp_old + dv_old + dv_net + dv_frame;  /* dv_s */
            }
        }
        memcpy(dhenc.data, d->s_dhacc.data, H * sizeof(float));
        return;
    }
    ensure(&d->s_da, 1, H); ensure(&d->s_dh1, 1, H); ensure(&d->s_dz, 1, H + O);
    ensure(&d->s_dyt, 1, O); ensure(&d->s_dynext, 1, O); ensure(&d->s_dhacc, 1, H);
    Mat da = d->s_da, dh1 = d->s_dh1, dz = d->s_dz, dyt = d->s_dyt, dynext = d->s_dynext;
    mat_zero(d->s_dhacc); mat_zero(dynext);
    for (int s = steps - 1; s >= 0; --s) {
        for (int i = 0; i < O; ++i) dyt.data[i] = dpreds.data[s * O + i] + dynext.data[i];
        mat_copy(d->fc2.xcache, d->ac[s]);            /* restore this step's caches */
        linear_backward(&d->fc2, dyt, da);
        for (int i = 0; i < H; ++i) dh1.data[i] = (d->h1c[s].data[i] > 0.0f) ? da.data[i] : 0.0f;
        mat_copy(d->fc1.xcache, d->zc[s]);
        linear_backward(&d->fc1, dh1, dz);
        for (int i = 0; i < H; ++i) d->s_dhacc.data[i] += dz.data[i];      /* h feeds every step */
        /* grad into the previous step's y: through fc1's input, plus (delta mode)
         * the identity path y_t = y_{t-1} + Δ. */
        for (int i = 0; i < O; ++i)
            dynext.data[i] = dz.data[H + i] + (g_delta ? dyt.data[i] : 0.0f);
    }
    memcpy(dhenc.data, d->s_dhacc.data, H * sizeof(float));
}
void decoder_free(Decoder *d) {
    if (d->use_gru)        { gru_free(&d->gru); linear_free(&d->fc_out); }
    else                   { linear_free(&d->fc1); linear_free(&d->fc2); }
    if (d->use_xattn)      { xattn_free(&d->xattn); mat_free(&d->s_xq); mat_free(&d->s_dmem); }
    for (int s = 0; s < d->max_steps; ++s) { mat_free(&d->zc[s]); mat_free(&d->h1c[s]); mat_free(&d->ac[s]); }
    if (d->frc) { for (int s = 0; s < d->max_steps; ++s) mat_free(&d->frc[s]); free(d->frc); }
    free(d->forced);
    free(d->zc); free(d->h1c); free(d->ac);
    mat_free(&d->s_yt); mat_free(&d->s_a); mat_free(&d->s_ytn); mat_free(&d->s_v);
    mat_free(&d->s_da); mat_free(&d->s_dh1); mat_free(&d->s_dz);
    mat_free(&d->s_dyt); mat_free(&d->s_dynext); mat_free(&d->s_dvnext); mat_free(&d->s_dhacc);
    mat_free(&d->s_hid0); mat_free(&d->s_dhid); mat_free(&d->s_dhc); mat_free(&d->s_dx2);
}

/* =====================================================================
 * Co-active spatial cross-attention (reuses MHA over [1+cnt, D])
 * ===================================================================== */
static CoSpatial cospatial_new(int nf, int d, ParamList *pl) {
    CoSpatial cs; cs.nf = nf; cs.d = d; cs.cnt = 0;
    cs.proj = linear_new(nf, d, pl, "co.proj");
    cs.attn = mha_new(d, 1, 0, pl, "co.attn");        /* single-head self-attn */
    /* Zero the attention's OUTPUT projection so the module starts as a no-op:
     * out = henc + o(...) = henc at init. Same idea as the delta head's zero-init
     * fc2 — a randomly-initialised residual into the decoder's context otherwise
     * kicks training off course on some splits (a real divergence we measured).
     * Gradients still flow, so the spatial signal is learned from zero. */
    for (int i = 0; i < cs.attn.o.out * cs.attn.o.in; ++i) cs.attn.o.W.data[i] = 0.0f;
    for (int i = 0; i < cs.attn.o.out; ++i) cs.attn.o.b[i] = 0.0f;
    cs.tok = NULLMAT; cs.seq = NULLMAT; cs.out = NULLMAT; cs.dmha = NULLMAT; cs.dseq = NULLMAT;
    return cs;
}
static void ensure_m(Mat *m, int r, int c) {
    if (m->data == NULL || m->rows != r || m->cols != c) { if (m->data) mat_free(m); *m = mat_new(r, c); }
}
/* out[1,D] = henc + attention(context over [context; neighbour tokens]).
 * Builds a token sequence whose row 0 is the encoded context and rows 1..c are
 * the co-active storms' states projected to D, runs self-attention over it, and
 * keeps row 0 as a residual add onto henc. So the context gets to attend over
 * the storms sharing its timestep. With zero neighbours it is exactly identity. */
static void cospatial_forward(CoSpatial *cs, const Mat henc, const Mat nbr, int cnt, Mat out) {
    const int D = cs->d;
    int c = cnt < 0 ? 0 : (cnt > nbr.rows ? nbr.rows : cnt);   /* clamp count to [0, rows] */
    cs->cnt = c;
    if (c == 0) { mat_copy(out, henc); return; }       /* no neighbours -> identity */
    Mat nbr_c = { c, cs->nf, nbr.data };               /* first c neighbour rows */
    ensure_m(&cs->tok, c, D);
    linear_forward(&cs->proj, nbr_c, cs->tok);         /* project neighbours to D */
    ensure_m(&cs->seq, 1 + c, D); ensure_m(&cs->out, 1 + c, D);
    memcpy(&cs->seq.data[0], henc.data, (size_t)D * sizeof(float));         /* row 0 = context */
    memcpy(&cs->seq.data[D], cs->tok.data, (size_t)c * D * sizeof(float));  /* rows 1..c = neighbours */
    mha_forward(&cs->attn, cs->seq, cs->out);          /* context (row 0) attends over all */
    for (int i = 0; i < D; ++i) out.data[i] = henc.data[i] + cs->out.data[i];   /* residual */
}
static void cospatial_backward(CoSpatial *cs, const Mat dout, Mat dhenc) {
    const int D = cs->d; int c = cs->cnt;
    if (c == 0) { mat_copy(dhenc, dout); return; }
    ensure_m(&cs->dmha, 1 + c, D); ensure_m(&cs->dseq, 1 + c, D);
    mat_zero(cs->dmha);
    memcpy(&cs->dmha.data[0], dout.data, (size_t)D * sizeof(float));   /* only row 0 has grad */
    mha_backward(&cs->attn, cs->dmha, cs->dseq);
    for (int i = 0; i < D; ++i) dhenc.data[i] = dout.data[i] + cs->dseq.data[i]; /* residual + query path */
    Mat dtok = { c, D, &cs->dseq.data[D] };            /* grad into neighbour tokens */
    /* proj.xcache still holds the neighbour input from cospatial_forward */
    linear_backward(&cs->proj, dtok, (Mat){0,0,NULL});    /* accumulate proj grads; drop input grad */
}
static void cospatial_free(CoSpatial *cs) {
    linear_free(&cs->proj); mha_free(&cs->attn);
    mat_free(&cs->tok); mat_free(&cs->seq); mat_free(&cs->out);
    mat_free(&cs->dmha); mat_free(&cs->dseq);
}

/* =====================================================================
 * Full model
 * ===================================================================== */
/* Assemble the full model. Sub-modules register their parameters with pl in
 * construction order (pgf, encoder, [co-spatial], decoder), which fixes the flat
 * parameter vector a checkpoint is saved/loaded against — so all the g_* option
 * flags must be set BEFORE this call and match at load time. */
Model model_new(const Config *c, ParamList *pl) {
    Model m; m.cfg = *c;
    m.pgf = pgf_new(c, pl);
    m.enc = encoder_new(c, pl);
    m.use_co = g_co_spatial;
    if (m.use_co) m.co = cospatial_new(TF_NBR_NF, c->d_model, pl);
    m.dec = decoder_new(c, pl);
    m.nbr = mat_new(TF_NBR_K, TF_NBR_NF); m.nbr_cnt = 0;
    m.vseed = mat_new(1, c->out_dim); mat_zero(m.vseed);
    m.tf_Y = mat_new(c->pred_len, c->out_dim); m.tf_set = 0;
    m.xtilde  = mat_new(c->in_len, c->d_model);
    m.henc    = mat_new(1, c->d_model);
    m.henc2   = mat_new(1, c->d_model);
    m.dhenc   = mat_new(1, c->d_model);
    m.dhenc2  = mat_new(1, c->d_model);
    m.dxtilde = mat_new(c->in_len, c->d_model);
    m.pred    = mat_new(c->pred_len, c->out_dim);
    return m;
}
void model_set_neighbors(Model *m, const Mat nbr, int cnt) {
    mat_copy(m->nbr, nbr); m->nbr_cnt = cnt;
}
void model_set_seed_velocity(Model *m, const Mat v) {
    mat_copy(m->vseed, v);
}
void model_set_teacher(Model *m, const Mat Y) {
    mat_copy(m->tf_Y, Y); m->tf_set = 1;
}
/* Full forward: fuse -> encode -> (co-spatial refine) -> decode.
 * The decoder's context is henc, or henc2 when co-spatial attention is on.
 * In --xattn mode the encoder's per-step sequence is handed to the decoder's
 * cross-attention as the fixed K/V memory before the rollout. */
void model_forward(Model *m, const Mat xnum, const Mat xtext, const Mat yprev) {
    /* Teacher forcing is armed only while training with p>0 and a teacher set;
     * eval and gradient checks (training off) always roll out autoregressively. */
    m->dec.tf_y = (nn_training() && g_tf_prob > 0.0f && m->tf_set) ? m->tf_Y.data : NULL;
    pgf_forward(&m->pgf, xnum, xtext, m->xtilde);
    encoder_forward(&m->enc, m->xtilde, m->henc);
    if (m->dec.use_xattn) xattn_set_memory(&m->dec.xattn, m->enc.encseq);   /* memory = encoder seq */
    Mat ctx = m->henc;
    if (m->use_co) { cospatial_forward(&m->co, m->henc, m->nbr, m->nbr_cnt, m->henc2); ctx = m->henc2; }
    decoder_forward(&m->dec, ctx, yprev, m->vseed, m->cfg.pred_len, m->pred);
}
/* Full backward: exact reverse of model_forward. The decoder writes the context
 * grad into dhenc2 (then co-spatial backward folds it into dhenc) or straight
 * into dhenc. In --xattn mode the decoder's context grad is zero (see
 * decoder_backward) and all the encoder gradient instead arrives through the
 * attended-memory grad s_dmem, which is routed into the encoder's dencseq. */
void model_backward(Model *m, const Mat dpred, const Mat dgate_pen) {
    if (m->use_co) {
        decoder_backward(&m->dec, dpred, m->dhenc2);
        cospatial_backward(&m->co, m->dhenc2, m->dhenc);
    } else {
        decoder_backward(&m->dec, dpred, m->dhenc);
    }
    if (m->dec.use_xattn) mat_copy(m->enc.dencseq, m->dec.s_dmem);  /* route attended-memory grad */
    encoder_backward(&m->enc, m->dhenc, m->dxtilde);
    pgf_backward(&m->pgf, m->dxtilde, dgate_pen);
}
/* Flush every Linear's deferred weight-gradient stash (see nn_set_defer_grads).
 * Must be called after the last model_backward of a batch and before the
 * optimizer (or gradient reduction) reads the gradients. No-op when deferral
 * is off or nothing is stashed. Walks exactly the Linears that exist for the
 * current architecture flags — the same set model_new allocated. */
void model_flush_grads(Model *m) {
    if (!nn_defer_grads()) return;
    linear_flush(&m->pgf.fc_gate); linear_flush(&m->pgf.proj_num); linear_flush(&m->pgf.proj_text);
    linear_flush(&m->enc.input_proj); linear_flush(&m->enc.output_proj);
    for (int l = 0; l < m->cfg.n_layers; ++l) {
        block_flush(&m->enc.temporal[l]);
        if (m->enc.use_spatial) block_flush(&m->enc.spatial[l]);
    }
    if (m->use_co) { linear_flush(&m->co.proj); mha_flush(&m->co.attn); }
    if (m->dec.use_gru)        { gru_flush(&m->dec.gru); linear_flush(&m->dec.fc_out); }
    else                       { linear_flush(&m->dec.fc1); linear_flush(&m->dec.fc2); }
    if (m->dec.use_xattn)      xattn_flush(&m->dec.xattn);
}

void model_free(Model *m) {
    pgf_free(&m->pgf); encoder_free(&m->enc); decoder_free(&m->dec);
    if (m->use_co) cospatial_free(&m->co);
    mat_free(&m->nbr); mat_free(&m->vseed); mat_free(&m->tf_Y);
    mat_free(&m->xtilde); mat_free(&m->henc); mat_free(&m->henc2);
    mat_free(&m->dhenc); mat_free(&m->dhenc2);
    mat_free(&m->dxtilde); mat_free(&m->pred);
}

/* ---- loss-shaping options (both off by default; set before training) ----
 * g_huber:   Huber transition point δ on the NORMALIZED residual. Inside |r|<=δ
 *            the term is the usual r² ; outside it grows linearly as δ(2|r|−δ)
 *            (value- and gradient-continuous at ±δ), so fast-moving outlier
 *            storms stop dominating the objective. 0 = plain MSE.
 * g_hweight: horizon-weight exponent γ. Forecast step h is weighted
 *            w_h = (h+1)^γ, normalized so mean(w) = 1 (keeps the loss scale
 *            comparable across γ). γ>0 upweights the far steps that dominate
 *            the km error budget; 0 = uniform. */
static float g_huber = 0.0f;
static float g_hweight = 0.0f;
/* g_km_loss: equirectangular great-circle (km) objective. The loss is scored on
 * spherical ΔR (kilometres) but trained on normalized-coordinate MSE — a mismatch,
 * because a normalized unit of longitude is a different number of kilometres than a
 * normalized unit of latitude (different train-set std) AND longitude kilometres
 * shrink by cos(lat) toward the poles. This scales the LONGITUDE residual by
 * (cstd_lon/cstd_lat)·cos(lat) so the objective becomes, up to a global constant,
 * the squared kilometre error. Latitude is the reference (weight 1). Both the loss
 * term and its gradient are reweighted consistently (unlike the old serial-only
 * gradient-reweight hack), and it lives in model_loss so BOTH the serial and the
 * data-parallel paths use it. The small-angle equirectangular form is deliberate:
 * the exact haversine gradient carries a 1/sin(d) factor that blows up as pred→true
 * (d→0), i.e. exactly at convergence — a poor training loss. cmean/cstd are the
 * TRAIN-set coordinate stats (leakage-safe), injected via model_set_km_loss. */
static int   g_km_loss = 0;
static float g_km_clat_mean = 0.0f, g_km_clat_std = 1.0f, g_km_clon_std = 1.0f;
void model_set_huber(float delta) { g_huber = delta; }
void model_set_hweight(float g)   { g_hweight = g; }
void model_set_km_loss(int on, float clat_mean, float clat_std, float clon_std) {
    g_km_loss = on; g_km_clat_mean = clat_mean; g_km_clat_std = clat_std; g_km_clon_std = clon_std;
}

/* Loss = mean_i w(i)·huber(pred_i − Y_i) + lambda·mean( relu(0.6 - gate)^2 ).
 * With both options off this reduces exactly to the original
 * MSE(pred,Y) + gate-collapse penalty. The second term activates only where a
 * gate channel drops below 0.6, pushing the fusion to keep listening to the
 * numerical stream instead of leaning entirely on text. Gradients (optional
 * outputs):
 *   dpred[i]     = (w_h/No)·huber'(r_i)         huber'(r)=2r inside, 2δ·sign(r) outside
 *   dgate_pen[i] = lambda·(-2/Ng)·(0.6 - gate)  where 0.6 - gate > 0, else 0
 * (the minus sign is d/dgate of (0.6 - gate)^2). dgate_pen feeds pgf_backward. */
double model_loss(const Mat pred, const Mat Y, const Mat gate, float lambda,
                  Mat dpred, Mat dgate_pen) {
    const int No = pred.rows * pred.cols, Ng = gate.rows * gate.cols;
    const int H = pred.rows, C = pred.cols;
    /* Per-horizon weights w_h = (h+1)^γ · H / Σ(k+1)^γ  (mean 1). Uniform 1
     * when γ==0 so the default path is bit-identical to the original. */
    double wsum = 0.0;
    for (int h = 0; h < H; ++h) wsum += pow((double)(h + 1), (double)g_hweight);
    const double wnorm = (double)H / wsum;
    double err = 0.0;
    for (int h = 0; h < H; ++h) {
        const double w = (g_hweight != 0.0f) ? pow((double)(h + 1), (double)g_hweight) * wnorm : 1.0;
        for (int j = 0; j < C; ++j) {
            const int i = h * C + j;
            double r = pred.data[i] - Y.data[i];
            /* km-loss: reweight the longitude residual to kilometres (lat = reference,
             * weight 1). kw==1 when km-loss is off, so this path is then bit-identical
             * to normalized MSE/Huber. reff is the km-weighted residual; the gradient
             * w.r.t. pred picks up an extra factor kw by the chain rule (reff = kw·r). */
            double kw = 1.0;
            if (g_km_loss && j == 1 && C >= 2) {                 /* longitude channel */
                double lat_deg = (double)Y.data[h * C] * g_km_clat_std + g_km_clat_mean;
                kw = (g_km_clon_std / g_km_clat_std) * cos(lat_deg * 0.017453292519943295);
            }
            double reff = kw * r;
            double term, grad;                                   /* grad = d(term)/d(reff) */
            if (g_huber > 0.0f && fabs(reff) > (double)g_huber) {/* linear tail */
                term = (double)g_huber * (2.0 * fabs(reff) - (double)g_huber);
                grad = 2.0 * (double)g_huber * (reff > 0 ? 1.0 : -1.0);
            } else {                                             /* quadratic core (== MSE) */
                term = reff * reff;
                grad = 2.0 * reff;
            }
            err += w * term;
            if (dpred.data) dpred.data[i] = (float)(w * grad * kw / (double)No);  /* chain: ×kw */
        }
    }
    err /= (double)No;
    double pen = 0.0;
    for (int i = 0; i < Ng; ++i) { float r = 0.6f - gate.data[i]; if (r > 0.0f) pen += (double)r * r; }
    pen /= (double)Ng;
    if (dgate_pen.data)
        for (int i = 0; i < Ng; ++i) {
            float r = 0.6f - gate.data[i];
            dgate_pen.data[i] = (r > 0.0f) ? (lambda * (-2.0f) / Ng * r) : 0.0f;
        }
    return err + (double)lambda * pen;
}
