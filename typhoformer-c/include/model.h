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
    Mat    cat, gate, xn, xt;         /* caches ([T, .]) */
    Mat    s_dz, s_dxn, s_dxt;        /* backward scratch [T,d_model] */
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
    Mat   s_tmp;              /* backward scratch [out_steps,in_steps] */
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
    Block  *spatial;    /* n_layers, single-node attention (omitted if !use_spatial) */
    TimeMix tmix;       /* pool in_len -> 1                */
    Mat     posenc, dposenc;                 /* learned positional encoding [in_len,D] */
    int     use_spatial, use_posenc, pool_last;
    Mat     b0, b1, db0, db1, tmid, dtmid;   /* work buffers */
} Encoder;
Encoder encoder_new(const Config *c, ParamList *pl);
void    encoder_forward(Encoder *e, const Mat xtilde, Mat henc);  /* henc [1,d_model] */
void    encoder_backward(Encoder *e, const Mat dhenc, Mat dxtilde);
void    encoder_free(Encoder *e);

/* ---- Encoder architecture options (set BEFORE model_new) ------------ */
/* Each is off by default so the golden path and existing tests are unchanged.
 * They change the parameter set/order, so a checkpoint must be evaluated with
 * the same options (the size-mismatch guard in checkpoint_load_params catches
 * a mismatch). */
void model_set_no_spatial(int on);   /* drop the degenerate N=1 spatial blocks */
void model_set_posenc(int on);       /* add a learned positional encoding      */
void model_set_pool_last(int on);    /* pool by taking the last step, not TimeMix */

/* ---- Autoregressive decoder ----------------------------------------- */
typedef struct {
    Linear fc1, fc2;
    int    hidden, out, max_steps, use_cv;
    Mat   *zc, *h1c, *ac;               /* per-step caches (autoregressive rollout) */
    Mat    s_yt, s_a, s_ytn, s_v;       /* forward scratch (s_v: cv velocity) */
    Mat    s_da, s_dh1, s_dz, s_dyt, s_dynext, s_dvnext, s_dhacc;  /* backward scratch */
} Decoder;
Decoder decoder_new(const Config *c, ParamList *pl);
/* vseed is the seed velocity (normalized coord units); used only in cv mode and
 * may be a NULL matrix otherwise. */
void    decoder_forward(Decoder *d, const Mat henc, const Mat yprev, const Mat vseed, int steps, Mat preds);
void    decoder_backward(Decoder *d, const Mat dpreds, Mat dhenc);   /* steps==1 */
void    decoder_free(Decoder *d);

/* Delta mode: the decoder predicts the DISPLACEMENT from the seed instead of the
 * absolute coordinate (y_t = y_{t-1} + fc2(...)), with fc2 zero-initialised so
 * an untrained model starts at persistence and only learns the correction. Set
 * BEFORE building the model. Off by default. */
void    model_set_delta(int on);
int     model_delta(void);

/* Constant-velocity decoder: a second-order extension of delta. The rollout is
 * anchored at constant-velocity extrapolation (ŷ_t = y_{t-1} + v + fc2(...)),
 * with v threaded across steps (v_{t+1} = v_t + correction) and fc2 zero-init,
 * so an untrained model starts *at the CLIPER baseline* and learns only the
 * curvature correction. The decoder also takes the seed velocity as input. Set
 * BEFORE building the model; feed each sample's seed velocity with
 * model_set_seed_velocity. Off by default; takes precedence over --delta. */
void    model_set_cv(int on);

/* ---- Co-active spatial cross-attention ------------------------------ */
/* The encoded context attends over the relative states of storms active at the
 * same timestep — real multi-node spatial attention (reuses MHA over a
 * [1+cnt, D] token sequence). Off by default; enable with model_set_co_spatial
 * BEFORE model_new, and feed each sample's neighbours with model_set_neighbors. */
typedef struct {
    int    nf, d;
    Linear proj;                       /* neighbour features (NF) -> D  */
    MHA    attn;                       /* self-attention over [1+cnt, D] */
    Mat    tok, seq, out, dmha, dseq;  /* scratch */
    int    cnt;                        /* cached neighbour count         */
} CoSpatial;
void model_set_co_spatial(int on);

/* ---- Full model ----------------------------------------------------- */
typedef struct {
    Config    cfg;
    PGF       pgf;
    Encoder   enc;
    Decoder   dec;
    CoSpatial co;
    int       use_co;
    Mat       nbr;  int nbr_cnt;       /* current sample's neighbours    */
    Mat       vseed;                   /* current sample's seed velocity (cv mode) */
    Mat       xtilde, henc, henc2, dhenc, dhenc2, dxtilde, pred;
} Model;
Model model_new(const Config *c, ParamList *pl);
void  model_set_neighbors(Model *m, const Mat nbr, int cnt);   /* set before model_forward */
void  model_set_seed_velocity(Model *m, const Mat v);          /* set before model_forward (cv) */
void  model_forward(Model *m, const Mat xnum, const Mat xtext, const Mat yprev); /* -> m->pred */
void  model_backward(Model *m, const Mat dpred, const Mat dgate_pen);
void  model_free(Model *m);

/* Loss = MSE(pred,Y) + lambda * mean(relu(0.6 - gate)^2).
 * If dpred / dgate_pen are non-NULL they receive the upstream gradients. */
double model_loss(const Mat pred, const Mat Y, const Mat gate, float lambda,
                  Mat dpred, Mat dgate_pen);

#endif /* TYPHOFORMER_MODEL_H */
