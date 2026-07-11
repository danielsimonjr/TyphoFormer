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

/* ---- parameter registry --------------------------------------------- */
typedef struct { float *v; float *g; int n; const char *name; } Param;
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
typedef struct {
    int   in, out;
    Mat   W, dW;       /* [out,in] */
    float *b, *db;     /* [out]    */
    Mat   xcache;      /* [T,in] cached input for backward */
} Linear;

Linear linear_new(int in, int out, ParamList *pl, const char *name);
void   linear_forward(Linear *l, const Mat x, Mat y);        /* caches x */
void   linear_backward(Linear *l, const Mat dy, Mat dx);     /* += grads; dx may be NULL */
void   linear_free(Linear *l);

/* ---- LayerNorm over the last dimension ------------------------------ */
typedef struct {
    int   dim;
    float *g, *dg, *b, *db;   /* [dim] */
    Mat   xhat;               /* [T,dim] cache */
    float *rstd;              /* [T] cache */
    int   Tcap;
} LayerNorm;

LayerNorm layernorm_new(int dim, ParamList *pl, const char *name);
void      layernorm_forward(LayerNorm *ln, const Mat x, Mat y);
void      layernorm_backward(LayerNorm *ln, const Mat dy, Mat dx);
void      layernorm_free(LayerNorm *ln);

/* ---- FFN:  Linear(D->F) -> ReLU -> Linear(F->D) --------------------- */
typedef struct {
    Linear fc1, fc2;
    int    d, f;
    Mat    h, a;          /* pre/post activation caches [T,F] */
    Mat    s_da, s_dh;    /* backward scratch [T,F] */
} FFN;

FFN  ffn_new(int d, int f, ParamList *pl, const char *name);
void ffn_forward(FFN *ff, const Mat x, Mat y);
void ffn_backward(FFN *ff, const Mat dy, Mat dx);
void ffn_free(FFN *ff);

/* ---- Multi-head self-attention over a sequence [S,D] ---------------- */
typedef struct {
    int    d_model, n_heads, head_dim;
    int    self_only;       /* 1 => each position attends only to itself
                             *      (spatial attention with a single node) */
    Linear q, k, v, o;
    Mat    Q, K, V, Ocat;   /* [S,D] caches */
    Mat   *P;               /* n_heads x [S,S] attention weights */
    int    Scap;
    Mat    s_dOcat, s_dQ, s_dK, s_dV, s_dxq, s_dxk;  /* backward scratch [S,D] */
    Mat    s_dP, s_dsc;                              /* backward scratch [S,S] */
} MHA;

MHA  mha_new(int d_model, int n_heads, int self_only, ParamList *pl, const char *name);
void mha_forward(MHA *m, const Mat x, Mat y);        /* self-attn over rows of x */
void mha_backward(MHA *m, const Mat dy, Mat dx);
void mha_free(MHA *m);

/* ---- Transformer block: MHA + res + LN -> FFN + res + LN (post-norm) */
typedef struct {
    MHA       attn;
    LayerNorm ln1, ln2;
    FFN       ff;
    int       d;
    Mat       attn_out, r1, y1, ff_out, r2;   /* caches */
    Mat       drop1, drop2;                    /* dropout masks (attn / ffn) */
    Mat       s_dr2, s_dy1, s_dr1, s_dtmp, s_dd; /* backward scratch [S,D] */
} Block;

Block block_new(int d_model, int n_heads, int ff_dim, int self_only, ParamList *pl, const char *name);
void  block_forward(Block *b, const Mat x, Mat y);
void  block_backward(Block *b, const Mat dy, Mat dx);
void  block_free(Block *b);

#endif /* TYPHOFORMER_NN_H */
