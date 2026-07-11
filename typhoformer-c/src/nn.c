/*
 * nn.c — implementation of the neural-network building blocks.
 */
#include "nn.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- RNG ------------------------------------------------------------- */
static unsigned long g_rng = 88172645463325252UL;
void nn_seed(unsigned long s) { g_rng = s ? s : 1UL; }
static unsigned long xorshift(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17;
    return g_rng;
}
float nn_uniform(float lo, float hi) {
    double u = (double)(xorshift() >> 11) / 9007199254740992.0; /* [0,1) */
    return (float)(lo + (hi - lo) * u);
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
void plist_zero_grad(ParamList *pl) {
    for (int i = 0; i < pl->count; ++i)
        memset(pl->item[i].g, 0, (size_t)pl->item[i].n * sizeof(float));
}
long plist_num_params(const ParamList *pl) {
    long t = 0; for (int i = 0; i < pl->count; ++i) t += pl->item[i].n; return t;
}
void plist_free(ParamList *pl) { free(pl->item); plist_init(pl); }

/* ---- shape helper ---------------------------------------------------- */
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
    float k = 1.0f / sqrtf((float)in);           /* Xavier-ish uniform */
    for (int i = 0; i < out * in; ++i) l.W.data[i] = nn_uniform(-k, k);
    plist_add(pl, l.W.data, l.dW.data, out * in, name);
    plist_add(pl, l.b, l.db, out, name);
    return l;
}
void linear_forward(Linear *l, const Mat x, Mat y) {
    assert(x.cols == l->in && y.rows == x.rows && y.cols == l->out);
    ensure(&l->xcache, x.rows, x.cols); mat_copy(l->xcache, x);
    mat_matmul_bt(x, l->W, y);
    mat_add_bias(y, l->b);
}
void linear_backward(Linear *l, const Mat dy, Mat dx) {
    const Mat x = l->xcache;
    const int T = dy.rows, out = l->out, in = l->in;
    assert(dy.cols == out && x.rows == T);
    for (int i = 0; i < out; ++i) {                 /* dW += dy^T @ x */
        for (int j = 0; j < in; ++j) {
            float acc = 0.0f;
            for (int p = 0; p < T; ++p) acc += dy.data[p * out + i] * x.data[p * in + j];
            l->dW.data[i * in + j] += acc;
        }
        float bsum = 0.0f;                          /* db += colsum(dy) */
        for (int p = 0; p < T; ++p) bsum += dy.data[p * out + i];
        l->db[i] += bsum;
    }
    if (dx.data) mat_matmul(dy, l->W, dx);          /* dx = dy @ W */
}
void linear_free(Linear *l) { mat_free(&l->W); mat_free(&l->dW); free(l->b); free(l->db); mat_free(&l->xcache); }

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
        double m1 = 0.0, m2 = 0.0;
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
    return ff;
}
void ffn_forward(FFN *ff, const Mat x, Mat y) {
    const int T = x.rows;
    ensure(&ff->h, T, ff->f); ensure(&ff->a, T, ff->f);
    linear_forward(&ff->fc1, x, ff->h);
    mat_copy(ff->a, ff->h); mat_relu(ff->a);
    linear_forward(&ff->fc2, ff->a, y);
}
void ffn_backward(FFN *ff, const Mat dy, Mat dx) {
    const int T = dy.rows;
    Mat da = mat_new(T, ff->f), dh = mat_new(T, ff->f);
    linear_backward(&ff->fc2, dy, da);
    for (int i = 0; i < T * ff->f; ++i) dh.data[i] = (ff->h.data[i] > 0.0f) ? da.data[i] : 0.0f;
    linear_backward(&ff->fc1, dh, dx);
    mat_free(&da); mat_free(&dh);
}
void ffn_free(FFN *ff) { linear_free(&ff->fc1); linear_free(&ff->fc2); mat_free(&ff->h); mat_free(&ff->a); }

/* ---- softmax over rows (in place) ----------------------------------- */
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
MHA mha_new(int d_model, int n_heads, ParamList *pl, const char *name) {
    assert(d_model % n_heads == 0);
    MHA m; m.d_model = d_model; m.n_heads = n_heads; m.head_dim = d_model / n_heads;
    m.q = linear_new(d_model, d_model, pl, name);
    m.k = linear_new(d_model, d_model, pl, name);
    m.v = linear_new(d_model, d_model, pl, name);
    m.o = linear_new(d_model, d_model, pl, name);
    m.Q = (Mat){0,0,NULL}; m.K = (Mat){0,0,NULL}; m.V = (Mat){0,0,NULL}; m.Ocat = (Mat){0,0,NULL};
    m.P = (Mat *)calloc(n_heads, sizeof(Mat)); m.Scap = 0;
    return m;
}
void mha_forward(MHA *m, const Mat x, Mat y) {
    const int S = x.rows, D = m->d_model, H = m->n_heads, hd = m->head_dim;
    ensure(&m->Q, S, D); ensure(&m->K, S, D); ensure(&m->V, S, D); ensure(&m->Ocat, S, D);
    for (int h = 0; h < H; ++h) ensure(&m->P[h], S, S);
    linear_forward(&m->q, x, m->Q);
    linear_forward(&m->k, x, m->K);
    linear_forward(&m->v, x, m->V);
    const float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < H; ++h) {
        const int off = h * hd;
        Mat P = m->P[h];
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) {
                float acc = 0.0f;
                for (int d = 0; d < hd; ++d) acc += m->Q.data[i * D + off + d] * m->K.data[j * D + off + d];
                P.data[i * S + j] = acc * scale;
            }
        softmax_rows(P);
        for (int i = 0; i < S; ++i)
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < S; ++j) acc += P.data[i * S + j] * m->V.data[j * D + off + d];
                m->Ocat.data[i * D + off + d] = acc;
            }
    }
    linear_forward(&m->o, m->Ocat, y);
}
void mha_backward(MHA *m, const Mat dy, Mat dx) {
    const int S = dy.rows, D = m->d_model, H = m->n_heads, hd = m->head_dim;
    const float scale = 1.0f / sqrtf((float)hd);
    Mat dOcat = mat_new(S, D);
    linear_backward(&m->o, dy, dOcat);
    Mat dQ = mat_new(S, D), dK = mat_new(S, D), dV = mat_new(S, D);
    Mat dP = mat_new(S, S), dsc = mat_new(S, S);
    for (int h = 0; h < H; ++h) {
        const int off = h * hd;
        Mat P = m->P[h];
        /* dP[i,j] = sum_d dOh[i,d] * Vh[j,d] ;  dVh[j,d] = sum_i P[i,j]*dOh[i,d] */
        for (int i = 0; i < S; ++i)
            for (int j = 0; j < S; ++j) {
                float acc = 0.0f;
                for (int d = 0; d < hd; ++d) acc += dOcat.data[i * D + off + d] * m->V.data[j * D + off + d];
                dP.data[i * S + j] = acc;
            }
        for (int j = 0; j < S; ++j)
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int i = 0; i < S; ++i) acc += P.data[i * S + j] * dOcat.data[i * D + off + d];
                dV.data[j * D + off + d] = acc;
            }
        /* softmax backward per row: dsc[i,:] = P[i,:] * (dP[i,:] - sum_k dP[i,k]P[i,k]) */
        for (int i = 0; i < S; ++i) {
            float dot = 0.0f;
            for (int j = 0; j < S; ++j) dot += dP.data[i * S + j] * P.data[i * S + j];
            for (int j = 0; j < S; ++j)
                dsc.data[i * S + j] = P.data[i * S + j] * (dP.data[i * S + j] - dot) * scale;
        }
        /* dQh[i,d] = sum_j dsc[i,j]*Kh[j,d] ; dKh[j,d] = sum_i dsc[i,j]*Qh[i,d] */
        for (int i = 0; i < S; ++i)
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < S; ++j) acc += dsc.data[i * S + j] * m->K.data[j * D + off + d];
                dQ.data[i * D + off + d] = acc;
            }
        for (int j = 0; j < S; ++j)
            for (int d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (int i = 0; i < S; ++i) acc += dsc.data[i * S + j] * m->Q.data[i * D + off + d];
                dK.data[j * D + off + d] = acc;
            }
    }
    Mat dxq = mat_new(S, D), dxk = mat_new(S, D);
    linear_backward(&m->q, dQ, dxq);
    linear_backward(&m->k, dK, dxk);
    linear_backward(&m->v, dV, dx);          /* dx starts as v's contribution */
    for (int i = 0; i < S * D; ++i) dx.data[i] += dxq.data[i] + dxk.data[i];
    mat_free(&dOcat); mat_free(&dQ); mat_free(&dK); mat_free(&dV);
    mat_free(&dP); mat_free(&dsc); mat_free(&dxq); mat_free(&dxk);
}
void mha_free(MHA *m) {
    linear_free(&m->q); linear_free(&m->k); linear_free(&m->v); linear_free(&m->o);
    mat_free(&m->Q); mat_free(&m->K); mat_free(&m->V); mat_free(&m->Ocat);
    for (int h = 0; h < m->n_heads; ++h) mat_free(&m->P[h]);
    free(m->P);
}

/* ---- Transformer block (post-norm) ---------------------------------- */
Block block_new(int d_model, int n_heads, int ff_dim, ParamList *pl, const char *name) {
    Block b; b.d = d_model;
    b.attn = mha_new(d_model, n_heads, pl, name);
    b.ln1 = layernorm_new(d_model, pl, name);
    b.ln2 = layernorm_new(d_model, pl, name);
    b.ff = ffn_new(d_model, ff_dim, pl, name);
    b.attn_out = (Mat){0,0,NULL}; b.r1 = (Mat){0,0,NULL}; b.y1 = (Mat){0,0,NULL};
    b.ff_out = (Mat){0,0,NULL}; b.r2 = (Mat){0,0,NULL};
    return b;
}
void block_forward(Block *b, const Mat x, Mat y) {
    const int S = x.rows, D = b->d;
    ensure(&b->attn_out, S, D); ensure(&b->r1, S, D); ensure(&b->y1, S, D);
    ensure(&b->ff_out, S, D); ensure(&b->r2, S, D);
    mha_forward(&b->attn, x, b->attn_out);
    for (int i = 0; i < S * D; ++i) b->r1.data[i] = x.data[i] + b->attn_out.data[i];
    layernorm_forward(&b->ln1, b->r1, b->y1);
    ffn_forward(&b->ff, b->y1, b->ff_out);
    for (int i = 0; i < S * D; ++i) b->r2.data[i] = b->y1.data[i] + b->ff_out.data[i];
    layernorm_forward(&b->ln2, b->r2, y);
}
void block_backward(Block *b, const Mat dy, Mat dx) {
    const int S = dy.rows, D = b->d;
    Mat dr2 = mat_new(S, D), dy1 = mat_new(S, D), dr1 = mat_new(S, D), dtmp = mat_new(S, D);
    layernorm_backward(&b->ln2, dy, dr2);            /* y = LN2(r2) */
    ffn_backward(&b->ff, dr2, dy1);                  /* r2 = y1 + ff(y1) */
    for (int i = 0; i < S * D; ++i) dy1.data[i] += dr2.data[i];
    layernorm_backward(&b->ln1, dy1, dr1);           /* y1 = LN1(r1) */
    mha_backward(&b->attn, dr1, dtmp);               /* r1 = x + attn(x) */
    for (int i = 0; i < S * D; ++i) dx.data[i] = dr1.data[i] + dtmp.data[i];
    mat_free(&dr2); mat_free(&dy1); mat_free(&dr1); mat_free(&dtmp);
}
void block_free(Block *b) {
    mha_free(&b->attn); layernorm_free(&b->ln1); layernorm_free(&b->ln2); ffn_free(&b->ff);
    mat_free(&b->attn_out); mat_free(&b->r1); mat_free(&b->y1); mat_free(&b->ff_out); mat_free(&b->r2);
}
