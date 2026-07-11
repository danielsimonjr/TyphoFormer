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

/* Encoder architecture options (read at encoder_new time). Off by default. */
static int g_no_spatial = 0, g_posenc = 0, g_pool_last = 0, g_co_spatial = 0;
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
static void ensure(Mat *m, int r, int cc) {
    if (m->data == NULL || m->rows != r || m->cols != cc) { if (m->data) mat_free(m); *m = mat_new(r, cc); }
}
void pgf_forward(PGF *p, const Mat xnum, const Mat xtext, Mat xtilde) {
    const int T = xnum.rows, D = p->d_model;
    ensure(&p->cat, T, p->d_num + p->d_text);
    ensure(&p->gate, T, D); ensure(&p->xn, T, D); ensure(&p->xt, T, D);
    for (int i = 0; i < T; ++i) {
        memcpy(&p->cat.data[i * (p->d_num + p->d_text)], &xnum.data[i * p->d_num],
               p->d_num * sizeof(float));
        memcpy(&p->cat.data[i * (p->d_num + p->d_text) + p->d_num], &xtext.data[i * p->d_text],
               p->d_text * sizeof(float));
    }
    linear_forward(&p->fc_gate, p->cat, p->gate);
    mat_sigmoid(p->gate);
    linear_forward(&p->proj_num, xnum, p->xn);
    linear_forward(&p->proj_text, xtext, p->xt);
    for (int i = 0; i < T * D; ++i)
        xtilde.data[i] = p->gate.data[i] * p->xn.data[i] + (1.0f - p->gate.data[i]) * p->xt.data[i];
}
void pgf_backward(PGF *p, const Mat dxtilde, const Mat dgate_pen) {
    const int T = dxtilde.rows, D = p->d_model;
    ensure(&p->s_dz, T, D); ensure(&p->s_dxn, T, D); ensure(&p->s_dxt, T, D);
    Mat dz = p->s_dz, dxn = p->s_dxn, dxt = p->s_dxt;
    for (int i = 0; i < T * D; ++i) {
        float g = p->gate.data[i];
        float dgate = dxtilde.data[i] * (p->xn.data[i] - p->xt.data[i]);
        if (dgate_pen.data) dgate += dgate_pen.data[i];
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
void timemix_forward(TimeMix *t, const Mat x, Mat y) {
    ensure(&t->xcache, x.rows, x.cols); mat_copy(t->xcache, x);
    mat_matmul(t->A, x, y);                          /* [out,in] @ [in,D] */
    for (int o = 0; o < t->out_steps; ++o)
        for (int d = 0; d < y.cols; ++d) y.data[o * y.cols + d] += t->c[o];
}
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
Encoder encoder_new(const Config *c, ParamList *pl) {
    Encoder e; e.cfg = *c;
    e.use_spatial = !g_no_spatial; e.use_posenc = g_posenc; e.pool_last = g_pool_last;
    e.input_proj  = linear_new(c->d_model, c->d_model, pl, "enc.input_proj");
    /* learned positional encoding, registered right after input_proj */
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
        e.temporal[l] = block_new(c->d_model, c->n_heads, c->d_ff, 0, pl, "enc.temporal");
        if (e.use_spatial) e.spatial[l] = block_new(c->d_model, c->n_heads, c->d_ff, 1, pl, "enc.spatial");
    }
    e.tmix = timemix_new(c->in_len, 1, pl);
    e.b0  = mat_new(c->in_len, c->d_model); e.b1  = mat_new(c->in_len, c->d_model);
    e.db0 = mat_new(c->in_len, c->d_model); e.db1 = mat_new(c->in_len, c->d_model);
    e.tmid = mat_new(1, c->d_model); e.dtmid = mat_new(1, c->d_model);
    return e;
}
void encoder_forward(Encoder *e, const Mat xtilde, Mat henc) {
    const int L = e->cfg.n_layers, T = e->cfg.in_len, D = e->cfg.d_model;
    Mat cur = e->b0, nxt = e->b1, tmp;
    linear_forward(&e->input_proj, xtilde, cur);
    if (e->use_posenc) for (int i = 0; i < T * D; ++i) cur.data[i] += e->posenc.data[i];
    /* alternate temporal then (optionally) spatial WITHIN each layer */
    for (int l = 0; l < L; ++l) {
        block_forward(&e->temporal[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp;
        if (e->use_spatial) { block_forward(&e->spatial[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp; }
    }
    if (e->pool_last) memcpy(e->tmid.data, &cur.data[(T - 1) * D], D * sizeof(float)); /* last step */
    else              timemix_forward(&e->tmix, cur, e->tmid);                          /* [1,D] */
    linear_forward(&e->output_proj, e->tmid, henc);   /* [1,D] */
}
void encoder_backward(Encoder *e, const Mat dhenc, Mat dxtilde) {
    const int L = e->cfg.n_layers, T = e->cfg.in_len, D = e->cfg.d_model;
    linear_backward(&e->output_proj, dhenc, e->dtmid);
    Mat cur = e->db0, nxt = e->db1, tmp;
    if (e->pool_last) { mat_zero(cur); memcpy(&cur.data[(T - 1) * D], e->dtmid.data, D * sizeof(float)); }
    else              timemix_backward(&e->tmix, e->dtmid, cur);
    for (int l = L - 1; l >= 0; --l) {
        if (e->use_spatial) { block_backward(&e->spatial[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp; }
        block_backward(&e->temporal[l], cur, nxt); tmp = cur; cur = nxt; nxt = tmp;
    }
    if (e->use_posenc) for (int i = 0; i < T * D; ++i) e->dposenc.data[i] += cur.data[i]; /* += identity path */
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
    mat_free(&e->b0); mat_free(&e->b1); mat_free(&e->db0); mat_free(&e->db1);
    mat_free(&e->tmid); mat_free(&e->dtmid);
}

/* =====================================================================
 * Decoder (autoregressive; training uses steps == 1)
 * ===================================================================== */
Decoder decoder_new(const Config *c, ParamList *pl) {
    Decoder d; d.hidden = c->d_model; d.out = c->out_dim; d.max_steps = c->pred_len;
    d.fc1 = linear_new(c->d_model + c->out_dim, c->d_model, pl, "dec.fc1");
    d.fc2 = linear_new(c->d_model, c->out_dim, pl, "dec.fc2");
    if (g_delta) {                                   /* start at persistence (zero delta) */
        memset(d.fc2.W.data, 0, (size_t)d.fc2.out * d.fc2.in * sizeof(float));
        memset(d.fc2.b, 0, (size_t)d.fc2.out * sizeof(float));
    }
    d.zc  = (Mat *)calloc(d.max_steps, sizeof(Mat));
    d.h1c = (Mat *)calloc(d.max_steps, sizeof(Mat));
    d.ac  = (Mat *)calloc(d.max_steps, sizeof(Mat));
    d.s_yt = NULLMAT; d.s_a = NULLMAT; d.s_ytn = NULLMAT;
    d.s_da = NULLMAT; d.s_dh1 = NULLMAT; d.s_dz = NULLMAT;
    d.s_dyt = NULLMAT; d.s_dynext = NULLMAT; d.s_dhacc = NULLMAT;
    return d;
}
void decoder_forward(Decoder *d, const Mat henc, const Mat yprev, int steps, Mat preds) {
    const int H = d->hidden, O = d->out;
    assert(steps <= d->max_steps);
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
    linear_free(&d->fc1); linear_free(&d->fc2);
    for (int s = 0; s < d->max_steps; ++s) { mat_free(&d->zc[s]); mat_free(&d->h1c[s]); mat_free(&d->ac[s]); }
    free(d->zc); free(d->h1c); free(d->ac);
    mat_free(&d->s_yt); mat_free(&d->s_a); mat_free(&d->s_ytn);
    mat_free(&d->s_da); mat_free(&d->s_dh1); mat_free(&d->s_dz);
    mat_free(&d->s_dyt); mat_free(&d->s_dynext); mat_free(&d->s_dhacc);
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
/* out[1,D] = henc + attention(context over [context; neighbour tokens]) */
static void cospatial_forward(CoSpatial *cs, const Mat henc, const Mat nbr, int cnt, Mat out) {
    const int D = cs->d;
    int c = cnt < 0 ? 0 : (cnt > nbr.rows ? nbr.rows : cnt);
    cs->cnt = c;
    if (c == 0) { mat_copy(out, henc); return; }       /* no neighbours -> identity */
    Mat nbr_c = { c, cs->nf, nbr.data };               /* first c neighbour rows */
    ensure_m(&cs->tok, c, D);
    linear_forward(&cs->proj, nbr_c, cs->tok);         /* project neighbours to D */
    ensure_m(&cs->seq, 1 + c, D); ensure_m(&cs->out, 1 + c, D);
    memcpy(&cs->seq.data[0], henc.data, (size_t)D * sizeof(float));
    memcpy(&cs->seq.data[D], cs->tok.data, (size_t)c * D * sizeof(float));
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
Model model_new(const Config *c, ParamList *pl) {
    Model m; m.cfg = *c;
    m.pgf = pgf_new(c, pl);
    m.enc = encoder_new(c, pl);
    m.use_co = g_co_spatial;
    if (m.use_co) m.co = cospatial_new(TF_NBR_NF, c->d_model, pl);
    m.dec = decoder_new(c, pl);
    m.nbr = mat_new(TF_NBR_K, TF_NBR_NF); m.nbr_cnt = 0;
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
void model_forward(Model *m, const Mat xnum, const Mat xtext, const Mat yprev) {
    pgf_forward(&m->pgf, xnum, xtext, m->xtilde);
    encoder_forward(&m->enc, m->xtilde, m->henc);
    Mat ctx = m->henc;
    if (m->use_co) { cospatial_forward(&m->co, m->henc, m->nbr, m->nbr_cnt, m->henc2); ctx = m->henc2; }
    decoder_forward(&m->dec, ctx, yprev, m->cfg.pred_len, m->pred);
}
void model_backward(Model *m, const Mat dpred, const Mat dgate_pen) {
    if (m->use_co) {
        decoder_backward(&m->dec, dpred, m->dhenc2);
        cospatial_backward(&m->co, m->dhenc2, m->dhenc);
    } else {
        decoder_backward(&m->dec, dpred, m->dhenc);
    }
    encoder_backward(&m->enc, m->dhenc, m->dxtilde);
    pgf_backward(&m->pgf, m->dxtilde, dgate_pen);
}
void model_free(Model *m) {
    pgf_free(&m->pgf); encoder_free(&m->enc); decoder_free(&m->dec);
    if (m->use_co) cospatial_free(&m->co);
    mat_free(&m->nbr);
    mat_free(&m->xtilde); mat_free(&m->henc); mat_free(&m->henc2);
    mat_free(&m->dhenc); mat_free(&m->dhenc2);
    mat_free(&m->dxtilde); mat_free(&m->pred);
}

double model_loss(const Mat pred, const Mat Y, const Mat gate, float lambda,
                  Mat dpred, Mat dgate_pen) {
    const int No = pred.rows * pred.cols, Ng = gate.rows * gate.cols;
    double mse = 0.0;
    for (int i = 0; i < No; ++i) { double d = pred.data[i] - Y.data[i]; mse += d * d; }
    mse /= (double)No;
    double pen = 0.0;
    for (int i = 0; i < Ng; ++i) { float r = 0.6f - gate.data[i]; if (r > 0.0f) pen += (double)r * r; }
    pen /= (double)Ng;
    if (dpred.data)
        for (int i = 0; i < No; ++i) dpred.data[i] = 2.0f / No * (pred.data[i] - Y.data[i]);
    if (dgate_pen.data)
        for (int i = 0; i < Ng; ++i) {
            float r = 0.6f - gate.data[i];
            dgate_pen.data[i] = (r > 0.0f) ? (lambda * (-2.0f) / Ng * r) : 0.0f;
        }
    return mse + (double)lambda * pen;
}
