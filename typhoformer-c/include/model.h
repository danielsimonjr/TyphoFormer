/*
 * model.h — the TyphoFormer model: Prompt-aware Gating Fusion (PGF), a
 * spatio-temporal Transformer encoder, and an autoregressive decoder.
 * Clean-room from the paper / algorithm spec.
 */
#ifndef TYPHOFORMER_MODEL_H
#define TYPHOFORMER_MODEL_H

#include "nn.h"

typedef struct {
    int d_num;     /* numerical feature dim              */
    int d_text;    /* language-embedding dim             */
    int d_model;   /* fused / hidden dim                 */
    int out_dim;   /* prediction dim (lat, lon) = 2      */
    int in_len;    /* input time steps                   */
    int pred_len;  /* prediction steps                   */
    int d_ff;      /* feed-forward hidden dim            */
    int n_heads;
    int n_layers;
} Config;

Config config_default(void);   /* the paper's configuration */

/* ---- Prompt-aware Gating Fusion ------------------------------------- */
typedef struct {
    int    d_num, d_text, d_model;
    Linear fc_gate, proj_num, proj_text;
    Mat    cat, gate, xn, xt;   /* caches ([T, .]) */
} PGF;
PGF  pgf_new(const Config *c, ParamList *pl);
void pgf_forward(PGF *p, const Mat xnum, const Mat xtext, Mat xtilde);  /* gate in p->gate */
void pgf_backward(PGF *p, const Mat dxtilde, const Mat dgate_pen);
void pgf_free(PGF *p);

/* ---- Time-mixing projection: y[out,D] = A[out,in] @ x[in,D] + c ----- */
typedef struct {
    int   in_steps, out_steps;
    Mat   A, dA; float *c, *dc;
    Mat   xcache;
} TimeMix;
TimeMix timemix_new(int in_steps, int out_steps, ParamList *pl);
void    timemix_forward(TimeMix *t, const Mat x, Mat y);
void    timemix_backward(TimeMix *t, const Mat dy, Mat dx);
void    timemix_free(TimeMix *t);

/* ---- Encoder -------------------------------------------------------- */
typedef struct {
    Config  cfg;
    Linear  input_proj, output_proj;
    Block  *temporal;   /* n_layers, attention over time  */
    Block  *spatial;    /* n_layers, single-node attention */
    TimeMix tmix;       /* pool in_len -> 1                */
    Mat     b0, b1, db0, db1, tmid, dtmid;   /* work buffers */
} Encoder;
Encoder encoder_new(const Config *c, ParamList *pl);
void    encoder_forward(Encoder *e, const Mat xtilde, Mat henc);  /* henc [1,d_model] */
void    encoder_backward(Encoder *e, const Mat dhenc, Mat dxtilde);
void    encoder_free(Encoder *e);

/* ---- Autoregressive decoder ----------------------------------------- */
typedef struct {
    Linear fc1, fc2;
    int    hidden, out;
    Mat    z, h1;       /* caches (single-step) */
} Decoder;
Decoder decoder_new(const Config *c, ParamList *pl);
void    decoder_forward(Decoder *d, const Mat henc, const Mat yprev, int steps, Mat preds);
void    decoder_backward(Decoder *d, const Mat dpreds, Mat dhenc);   /* steps==1 */
void    decoder_free(Decoder *d);

/* ---- Full model ----------------------------------------------------- */
typedef struct {
    Config  cfg;
    PGF     pgf;
    Encoder enc;
    Decoder dec;
    Mat     xtilde, henc, dhenc, dxtilde, pred;
} Model;
Model model_new(const Config *c, ParamList *pl);
void  model_forward(Model *m, const Mat xnum, const Mat xtext, const Mat yprev); /* -> m->pred */
void  model_backward(Model *m, const Mat dpred, const Mat dgate_pen);
void  model_free(Model *m);

/* Loss = MSE(pred,Y) + lambda * mean(relu(0.6 - gate)^2).
 * If dpred / dgate_pen are non-NULL they receive the upstream gradients. */
double model_loss(const Mat pred, const Mat Y, const Mat gate, float lambda,
                  Mat dpred, Mat dgate_pen);

#endif /* TYPHOFORMER_MODEL_H */
