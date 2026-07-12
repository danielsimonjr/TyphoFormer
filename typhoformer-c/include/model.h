/*
 * model.h — the TyphoFormer model: Prompt-aware Gating Fusion (PGF), a
 * spatio-temporal Transformer encoder, and an autoregressive decoder.
 * Clean-room from the paper / algorithm spec.
 */
#ifndef TYPHOFORMER_MODEL_H
#define TYPHOFORMER_MODEL_H

#include "nn.h"

/* Hyper-parameters shared by every sub-module. One Config is built once
 * (config_default) and threaded read-only through every *_new constructor so
 * the shapes below are globally consistent. */
typedef struct {
    int d_num;     /* numerical feature dim  (per-step track/intensity numbers) */
    int d_text;    /* language-embedding dim (per-step prompt/text embedding)   */
    int d_model;   /* fused / hidden dim D — the width everything runs at        */
    int out_dim;   /* prediction dim: (lat, lon) = 2                            */
    int in_len;    /* input time steps T (history length fed to the encoder)    */
    int pred_len;  /* prediction steps  (autoregressive rollout length; 1 here) */
    int d_ff;      /* transformer feed-forward hidden dim                       */
    int n_heads;   /* attention heads per block                                 */
    int n_layers;  /* stacked encoder layers (each = temporal [+ spatial])      */
} Config;

Config config_default(void);   /* the paper's configuration (D=256, T=12, ...) */

/* ---- Prompt-aware Gating Fusion ------------------------------------- */
/* PGF fuses the numerical stream and the language stream into one per-step
 * hidden vector. A learned per-channel gate g in (0,1) decides, element-wise,
 * how much of the numerical projection vs. the text projection to keep:
 *   g       = sigmoid( fc_gate([xnum ; xtext]) )        [T,D]
 *   x~       = g ⊙ proj_num(xnum) + (1-g) ⊙ proj_text(xtext)   [T,D]
 * The gate is also exported (p->gate) so the loss can add a penalty that keeps
 * it from collapsing to ~0 (i.e. ignoring the numbers).
 * Fields: fc_gate [d_num+d_text->D] produces the gate logits; proj_num/proj_text
 * project each stream to D. Caches (all [T,·]): cat=[xnum|xtext] concat input,
 * gate=sigmoid output (also the loss's penalty target), xn=proj_num(xnum),
 * xt=proj_text(xtext). Backward scratch [T,D]: s_dz=gate-logit grad, s_dxn/s_dxt
 * = grads of the two projected streams. */
typedef struct {
    int    d_num, d_text, d_model;
    Linear fc_gate, proj_num, proj_text;
    Mat    cat, gate, xn, xt;         /* caches ([T, .]) */
    Mat    s_dz, s_dxn, s_dxt;        /* backward scratch [T,d_model] */
} PGF;
PGF  pgf_new(const Config *c, ParamList *pl);
void pgf_forward(PGF *p, const Mat xnum, const Mat xtext, Mat xtilde);  /* writes x~; gate left in p->gate */
void pgf_backward(PGF *p, const Mat dxtilde, const Mat dgate_pen);      /* dgate_pen: extra grad on the gate */
void pgf_free(PGF *p);

/* ---- Time-mixing projection: y[out,D] = A[out,in] @ x[in,D] + c ----- */
/* A learned linear pooling ACROSS the time axis (a small mix matrix over steps,
 * applied identically to every feature channel). The encoder uses it with
 * out_steps==1 to collapse the [T,D] sequence into a single [1,D] context —
 * a learned weighted average over time, in contrast to the fixed "take the last
 * step" pooling. Each output step o is  y[o,:] = Σ_i A[o,i]·x[i,:] + c[o].
 * A [out_steps,in_steps] is the mix-weight parameter, dA its grad; c/dc are the
 * [out_steps] per-output bias and its grad; xcache holds the [in_steps,D] input
 * for the backward; s_tmp is scratch for the dA outer product. */
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
/* Spatio-temporal Transformer that maps the fused sequence x~ [T,D] to a single
 * context vector henc [1,D]:
 *   1. input_proj: [T,D] -> [T,D]  (+ optional learned positional encoding)
 *   2. n_layers of  temporal-block  then optional spatial-block, alternating
 *      WITHIN each layer (temporal = self-attention over the T time steps;
 *      spatial = single-node/self-only attention, degenerate here with N=1).
 *   3. pool over time to [1,D]: learned TimeMix average, or just the last step.
 *   4. output_proj: [1,D] -> [1,D] context henc.
 * With --xattn the pre-pool [T,D] sequence is also exported (encseq) so the
 * decoder can cross-attend to every timestep, not just the pooled summary.
 * input_proj/output_proj are [D->D]; posenc/dposenc are the [in_len,D] learned
 * positional table and its grad; the use_* ints snapshot the architecture
 * switches (posenc, spatial, pool_last, expose-sequence) from the g_* globals;
 * encseq/dencseq are the exported pre-pool [in_len,D] sequence and the grad the
 * decoder injects back into it. Work buffers: b0/b1 ([T,D]) ping-pong forward
 * activations between layers, db0/db1 do the same in the backward, and
 * tmid/dtmid ([1,D]) hold the pooled context and its grad. */
typedef struct {
    Config  cfg;
    Linear  input_proj, output_proj;
    Block  *temporal;   /* n_layers, attention over time  */
    Block  *spatial;    /* n_layers, single-node attention (omitted if !use_spatial) */
    TimeMix tmix;       /* pool in_len -> 1                */
    Mat     posenc, dposenc;                 /* learned positional encoding [in_len,D] */
    int     use_spatial, use_posenc, pool_last;
    int     use_encseq;                      /* expose the pre-pool [in_len,D] sequence */
    Mat     encseq, dencseq;                 /* pre-pool sequence + its injected grad (--xattn) */
    Mat     b0, b1, db0, db1, tmid, dtmid;   /* work buffers */
} Encoder;
Encoder encoder_new(const Config *c, ParamList *pl);
void    encoder_forward(Encoder *e, const Mat xtilde, Mat henc);  /* x~[T,D] -> henc [1,D] */
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
/* Rolls out pred_len future coordinates one step at a time, feeding each
 * prediction back in as the next step's input. Four interchangeable "correction"
 * heads, selected by the module-level flags at build time:
 *   plain / delta : MLP fc2(relu(fc1([h ; y_prev])))
 *   cv            : MLP with velocity, anchored at const-velocity extrapolation
 *   gru (--gru)   : the cv correction produced by a GRU with carried hidden state
 *   xattn (--xattn): the cv correction fed by per-step cross-attention over encseq
 * All the "anchored" heads zero-init their final layer so an untrained model
 * emits the physical baseline (persistence for delta, p+v for cv) and only
 * learns the residual correction. Only one head is allocated per Decoder.
 * hidden=H=d_model, out=O=out_dim, max_steps=pred_len; use_cv/use_gru/use_xattn
 * select the (mutually exclusive) active head. Per-step caches [max_steps]: zc =
 * fc1 input z, h1c = pre-activation, ac = post-relu activation (or the GRU hidden
 * hid_s). Forward scratch: s_yt=p (last position), s_a=relu act, s_ytn=correction
 * c, s_v=velocity v. Backward scratch: s_da=dact, s_dh1=d(pre-relu), s_dz=d(fc1
 * input z), s_dyt=dc, s_dynext=dp threaded from step s+1, s_dvnext=dv threaded,
 * s_dhacc [1,H]=accumulated encoder-context grad (h feeds every step). GRU/xattn
 * scratch: s_hid0=initial hidden/context, s_dhid=dhid carried across steps,
 * s_dhc=dhid from fc_out, s_dx2=dx=[dp;dv]. Xattn scratch: s_xq [1,2O]=query
 * [p;v], s_dmem [T,D]=accumulated grad routed back to the encoder memory. */
typedef struct {
    Linear fc1, fc2;                    /* MLP correction (plain/delta/cv modes) */
    GRU    gru; Linear fc_out;          /* recurrent correction (--gru mode)     */
    CrossAttn xattn;                    /* per-step encoder cross-attention (--xattn) */
    int    hidden, out, max_steps, use_cv, use_gru, use_xattn;
    Mat   *zc, *h1c, *ac;               /* per-step caches (autoregressive rollout) */
    Mat    s_yt, s_a, s_ytn, s_v;       /* forward scratch (s_v: cv velocity) */
    Mat    s_da, s_dh1, s_dz, s_dyt, s_dynext, s_dvnext, s_dhacc;  /* backward scratch */
    Mat    s_hid0, s_dhid, s_dhc, s_dx2;  /* gru scratch (initial hidden, hidden grads, dx) */
    Mat    s_xq, s_dmem;                  /* xattn scratch (query input, memory grad) */
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

/* Recurrent decoder: the constant-velocity rollout's curvature correction is
 * produced by a GRU whose hidden state is carried across steps (initialised from
 * the encoder context), instead of a memoryless per-step MLP. Gives the
 * multi-step rollout real memory. Implies --cv anchoring; fc_out zero-init so it
 * still starts at CLIPER. Set BEFORE model_new. Off by default. */
void    model_set_gru(int on);

/* Cross-attention decoder: each rollout step attends over the encoder's
 * per-timestep sequence (not the pooled context) to build its own context,
 * which feeds the curvature correction. Implies --cv anchoring (fc2 zero-init).
 * Set BEFORE model_new. Off by default. */
void    model_set_xattn(int on);

/* ---- Co-active spatial cross-attention ------------------------------ */
/* The encoded context attends over the relative states of storms active at the
 * same timestep — real multi-node spatial attention (reuses MHA over a
 * [1+cnt, D] token sequence). Off by default; enable with model_set_co_spatial
 * BEFORE model_new, and feed each sample's neighbours with model_set_neighbors. */
/* Fields: nf = neighbour feature dim, d = D. proj maps neighbour features (NF)
 * -> D; attn is single-head self-attention over the [1+cnt,D] token sequence.
 * Scratch: tok [cnt,D] projected neighbour tokens; seq [1+cnt,D] = context (row
 * 0) followed by those tokens; out [1+cnt,D] attention output; dmha/dseq
 * [1+cnt,D] backward scratch. cnt caches this sample's neighbour count. */
typedef struct {
    int    nf, d;
    Linear proj;                       /* neighbour features (NF) -> D  */
    MHA    attn;                       /* self-attention over [1+cnt, D] */
    Mat    tok, seq, out, dmha, dseq;  /* scratch */
    int    cnt;                        /* cached neighbour count         */
} CoSpatial;
void model_set_co_spatial(int on);

/* ---- Full model ----------------------------------------------------- */
/* Wires the pipeline: xnum/xtext --PGF--> x~ --Encoder--> henc [--CoSpatial-->
 * henc2] --Decoder--> pred. Owns all the inter-stage buffers and their grads.
 * Inter-stage buffers: xtilde [T,D] PGF output / encoder input; henc [1,D]
 * encoder context; henc2 [1,D] co-spatial-refined context (when use_co); dhenc/
 * dhenc2 [1,D] their grads; dxtilde [T,D] grad of x~ flowing back into the PGF;
 * pred [pred_len,out_dim] the predicted coords. nbr [TF_NBR_K,TF_NBR_NF] holds
 * the current sample's neighbour states (nbr_cnt valid) and vseed [1,out_dim]
 * its seed velocity (cv/gru/xattn). */
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

/* Loss = mean_i w_h(i)·huber(pred_i − Y_i) + lambda * mean(relu(0.6 - gate)^2).
 * With both loss-shaping options off (the default) this is exactly
 * MSE(pred,Y) + the gate penalty. If dpred / dgate_pen are non-NULL they
 * receive the upstream gradients. */
double model_loss(const Mat pred, const Mat Y, const Mat gate, float lambda,
                  Mat dpred, Mat dgate_pen);

/* Loss shaping (both off by default; set before training — they are read on
 * every model_loss call, on the serial and the data-parallel path alike):
 *   model_set_huber(δ)  : Huber loss with transition point δ on the NORMALIZED
 *                         residual (quadratic core == MSE inside |r|<=δ, linear
 *                         tails outside). δ=0 disables (plain MSE).
 *   model_set_hweight(γ): weight forecast step h by (h+1)^γ, normalized to
 *                         mean 1 — γ>0 upweights long horizons. γ=0 disables. */
void model_set_huber(float delta);
void model_set_hweight(float gamma);

#endif /* TYPHOFORMER_MODEL_H */
