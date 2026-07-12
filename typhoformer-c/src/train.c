/*
 * train.c — TyphoFormer-C command-line driver.
 *
 * This translation unit is the *only* entry point of the program: it owns
 * main(), parses the CLI for every subcommand, and glues together the four
 * library layers declared in the headers included below:
 *   - data.h        : dataset loading, storm-level split, standardization,
 *                     windowing, neighbour lookup, denormalization.
 *   - model.h       : the transformer encoder + track decoder and its loss.
 *   - optim.h       : AdamW optimizer and the parameter list (ParamList).
 *   - parallel.h    : the data-parallel trainer (ParTrainer) worker pool.
 *   - checkpoint.h  : save/load of weights, config, normalization stats, and
 *                     the optimizer sidecar used by --resume.
 *
 * Everything below is either CLI plumbing, the metric loop (evaluate()), or
 * the training loop (cmd_train). No model math lives here — this file decides
 * *what* to run, in *what* order, on *which* data split.
 *
 * Subcommands (default is "train", so `./typhoformer 30` still trains):
 *   ./typhoformer train   [epochs] [--full] [--csv=F --emb=D | --bin=F] [--save=CKPT]
 *   ./typhoformer eval     [--weights=CKPT] [--csv=F --emb=D | --bin=F]
 *   ./typhoformer prepare  --csv=F --emb=D --out=DATA.tfb [--in_len=12 --pred_len=1]
 */
#include "checkpoint.h"
#include "data.h"
#include "model.h"
#include "optim.h"
#include "parallel.h"

#include <pthread.h>

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R_EARTH 6371.0                     /* mean Earth radius in km — sets the units of every ΔR metric */
#define PI_ 3.14159265358979323846         /* π to double precision; used for degrees→radians conversion */
#define DEF_CSV "../HURDAT_2new_3000.csv"  /* default HURDAT2-style track CSV (raw numeric + storm ids)  */
#define DEF_EMB "../embedding_chunks"      /* default directory of per-record text embedding chunks       */
#define DEF_CKPT "typhoformer.ckpt"        /* default checkpoint path for save (train) and load (eval)    */

/* ---- helpers -------------------------------------------------------- */
/* Great-circle distance between two (lat,lon) points in DEGREES, returned in
 * kilometres. This is the numerically stable "haversine" form: it works with
 * the half-versed-sine `a` (always in [0,1]) instead of a raw acos, which
 * loses precision for nearby points. All track error is reported in km via
 * this function, so predictions and targets must be DENORMALIZED to degrees
 * before being passed in (normalized coords would give meaningless distances). */
static double haversine(float lat1, float lon1, float lat2, float lon2) {
    double p = PI_ / 180.0;                          /* degrees → radians scale factor */
    /* a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2), written via the
     * identity sin²(x/2) = (1 − cos x)/2 to avoid a separate sin() call. */
    double a = 0.5 - cos((lat2 - lat1) * p) / 2.0 +
               cos(lat1 * p) * cos(lat2 * p) * (1.0 - cos((lon2 - lon1) * p)) / 2.0;
    /* central angle = 2·asin(√a); ×R_EARTH turns radians into kilometres.
     * Clamp a≥0 to guard against tiny negative round-off feeding sqrt(). */
    return 2.0 * R_EARTH * asin(sqrt(a < 0 ? 0 : a));
}

#define MAXH 32                                     /* hard cap on forecast horizons a single Eval can hold */
/* Aggregate metrics returned by evaluate(). Three parallel per-horizon arrays
 * plus their scalar means. Index h is the (h+1)-th forecast step (6·(h+1) hours
 * out for 6-hourly HURDAT2 data). */
typedef struct {
    int    H;                                   /* number of horizons actually populated (min(pred_len,MAXH)) */
    double mae[MAXH], km[MAXH], base_km[MAXH];  /* per forecast horizon: model MAE, model ΔR km, persistence ΔR km */
    double mmae, mkm, mbase;                    /* horizon means of the three arrays above */
} Eval;

/* Accumulate raw metric SUMS for samples idx[i0..i1) (or samples i0..i1-1 when
 * idx==NULL) into mae/km/base (each MAXH-long, zeroed by the caller). Scores
 * the ENSEMBLE MEAN of the nm models' predictions (nm==1 is the ordinary
 * single-model case; averaging is done in normalized coordinate space, which
 * commutes with the linear denorm). Thread-safe as long as each concurrent
 * call has its own Model instance(s). */
static void eval_span(Model *const *ms, int nm, const Dataset *d, const int *idx,
                      int i0, int i1, int H, double *mae, double *km, double *base) {
    /* Scratch tensors reused across this span's samples. Shapes:
     *   xn  [in_len × d_num]  numeric feature window
     *   xt  [in_len × d_text] text-embedding window
     *   yp  [1 × 2]           seed coordinate (last observed lat,lon, normalized)
     *   Y   [pred_len × 2]    ground-truth future coords (normalized) */
    Mat xn = mat_new(d->in_len, d->d_num), xt = mat_new(d->in_len, d->d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(d->pred_len, 2);
    Mat nbr = mat_new(TF_NBR_K, TF_NBR_NF);          /* co-active neighbour features [K × NF] for --co_spatial */
    Mat vel = mat_new(1, 2);                         /* seed velocity (Δlat,Δlon) for the cv/gru/xattn decoder */
    double apred[2 * MAXH];                          /* ensemble-mean prediction, normalized coords */
    for (int i = i0; i < i1; ++i) {
        int s = idx ? idx[i] : i;                    /* map dense loop index to the real sample id */
        dataset_get(d, s, xn, xt, yp, Y);
        for (int h = 0; h < 2 * H; ++h) apred[h] = 0.0;
        for (int k = 0; k < nm; ++k) {
            Model *m = ms[k];
            /* Feed the same auxiliary inputs the model saw in training so eval is
             * apples-to-apples. These are no-ops for the model heads that ignore
             * them, so it is safe to always populate both. */
            int nc; dataset_neighbors(d, s, nbr, &nc); model_set_neighbors(m, nbr, nc);   /* spatial context */
            dataset_seed_velocity(d, s, vel); model_set_seed_velocity(m, vel);            /* constant-velocity anchor */
            model_forward(m, xn, xt, yp);
            for (int h = 0; h < 2 * H; ++h) apred[h] += m->pred.data[h];
        }
        for (int h = 0; h < 2 * H; ++h) apred[h] /= nm;
        float seed[2] = { yp.data[0], yp.data[1] };  /* copy seed before denorm; used for persistence baseline */
        dataset_denorm(d, seed);                        /* coords back to degrees for metrics */
        for (int h = 0; h < H; ++h) {
            float p[2] = { (float)apred[h * 2], (float)apred[h * 2 + 1] };  /* (mean) prediction at horizon h */
            float t[2] = { Y.data[h * 2],       Y.data[h * 2 + 1] };        /* ground truth at horizon h    */
            dataset_denorm(d, p); dataset_denorm(d, t);   /* everything in degrees before distance/MAE */
            /* MAE is on DENORMALIZED degrees, averaged over the 2 coord axes.
             * (Note: this reports degree error, not the normalized-space loss
             * the network minimizes — it is a human-readable companion to ΔR.) */
            mae[h]  += (fabs(p[0] - t[0]) + fabs(p[1] - t[1])) / 2.0;
            km[h]   += haversine(p[0], p[1], t[0], t[1]);              /* model great-circle error, km */
            /* Persistence baseline: assume the storm does not move, i.e. every
             * future position equals the seed. A useful model must beat this. */
            base[h] += haversine(seed[0], seed[1], t[0], t[1]);   /* persistence */
        }
    }
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&nbr); mat_free(&vel);
}

/* One thread's eval job: a replica model and a [i0,i1) span of the sample list. */
typedef struct {
    Model *m; const Dataset *d; const int *idx;
    int i0, i1, H;
    double mae[MAXH], km[MAXH], base[MAXH];
    pthread_t tid;
} EvalJob;

static void *eval_worker(void *arg) {
    EvalJob *j = (EvalJob *)arg;
    eval_span(&j->m, 1, j->d, j->idx, j->i0, j->i1, j->H, j->mae, j->km, j->base);
    return NULL;
}

/* Score model `m` over the `n` samples named by index array `idx` (or samples
 * 0..n-1 when idx==NULL). Returns per-horizon and mean metrics. This is called
 * with the VAL split each epoch (for checkpoint selection) and with the TEST
 * split once at the end; it never touches optimizer state or gradients.
 *
 * When a ParTrainer pool is supplied (pt != NULL, from --threads=N), the eval
 * set is SHARDED across the replica models: `pl` (the master weights) is
 * broadcast to the replicas, each worker thread scores a contiguous span with
 * its own replica, and the per-span metric sums are reduced here. Every replica
 * owns its scratch, so this is race-free; only the double-precision summation
 * order differs from serial (≈1e-12 relative on the printed means). Pass
 * pt=NULL (or n too small to shard) for the original single-thread path. */
static Eval evaluate(Model *m, const ParamList *pl, ParTrainer *pt,
                     const Dataset *d, const int *idx, int n) {
    nn_set_training(0);                              /* dropout off for evaluation (deterministic forward) */
    int H = d->pred_len < MAXH ? d->pred_len : MAXH; /* clamp horizons to the Eval array bound */
    Eval e; memset(&e, 0, sizeof e); e.H = H;
    int W = pt ? partrainer_workers(pt) : 1;
    if (pt && W > 1 && n >= 2 * W) {
        partrainer_broadcast(pt, pl);                /* replicas <- current master weights */
        EvalJob *jobs = (EvalJob *)calloc((size_t)W, sizeof(EvalJob));
        int base_n = n / W, rem = n % W, off = 0;
        for (int w = 0; w < W; ++w) {                /* contiguous spans, sizes differ by <=1 */
            jobs[w].m = partrainer_model(pt, w);
            jobs[w].d = d; jobs[w].idx = idx; jobs[w].H = H;
            jobs[w].i0 = off; off += base_n + (w < rem ? 1 : 0); jobs[w].i1 = off;
        }
        for (int w = 1; w < W; ++w) pthread_create(&jobs[w].tid, NULL, eval_worker, &jobs[w]);
        eval_worker(&jobs[0]);                       /* worker 0 on the calling thread */
        for (int w = 1; w < W; ++w) pthread_join(jobs[w].tid, NULL);
        for (int w = 0; w < W; ++w)
            for (int h = 0; h < H; ++h) {
                e.mae[h] += jobs[w].mae[h]; e.km[h] += jobs[w].km[h]; e.base_km[h] += jobs[w].base[h];
            }
        free(jobs);
    } else {
        eval_span(&m, 1, d, idx, 0, n, H, e.mae, e.km, e.base_km);
    }
    for (int h = 0; h < H; ++h) {
        e.mae[h] /= n; e.km[h] /= n; e.base_km[h] /= n;                    /* sum → mean over samples */
        e.mmae += e.mae[h]; e.mkm += e.km[h]; e.mbase += e.base_km[h];     /* accumulate for horizon mean */
    }
    e.mmae /= H; e.mkm /= H; e.mbase /= H;           /* mean across horizons — mkm is the model-selection metric */
    return e;
}

/* Pretty-print the per-horizon breakdown of an Eval. Skipped when there is
 * only a single horizon (the scalar means already tell the whole story).
 * The "%2dh" column is the lead time in hours: horizon h → 6·(h+1) hours. */
static void print_horizons(const Eval *e) {
    if (e->H <= 1) return;
    printf("  per-horizon:\n");
    for (int h = 0; h < e->H; ++h)
        printf("    %2dh | MAE %.4f | dR %.2f km | persistence %.2f km\n",
               (h + 1) * 6, e->mae[h], e->km[h], e->base_km[h]);
}

/* Load a dataset from one of two mutually exclusive sources:
 *   - bin != NULL : a pre-windowed .tfb binary produced by `prepare`. Fast to
 *     load and self-standardizing, but it stores no raw lat/lon history, so the
 *     decoder must be seeded from the first target and there is no storm-level
 *     split (it splits by sample instead).
 *   - otherwise   : the raw CSV + text-embedding directory, windowed on the fly
 *     into (in_len → pred_len) samples that retain full coordinate history. */
static Dataset load_source(const char *bin, const char *csv, const char *emb, int in_len, int pred_len) {
    if (bin) {
        printf("Loading pre-windowed data: %s\n", bin);
        Dataset d = dataset_load_bin(bin);
        printf("NOTE: .tfb files carry no coordinates; decoder seeded with the first target.\n");
        return d;
    }
    printf("Loading data: %s + %s\n", csv, emb);
    return dataset_load(csv, emb, in_len, pred_len);
}

/* ---- train ---------------------------------------------------------- */
/* The main workhorse: parse flags, load+split+standardize data, build the
 * model and optimizer, run the epoch loop with checkpointing/early-stopping,
 * and finish with a held-out TEST score. Every knob below has a CLI flag; the
 * -1 / 0 sentinels mean "unset — fall back to config_default() or off". */
static int cmd_train(int argc, char **argv) {
    int epochs = 10, full = 0, batch = 8, threads = 1;   /* epochs, full-size model, minibatch size, worker threads */
    /* Architecture overrides. -1 means "leave at the config default"; a
     * positive value forces that dimension. See config_default() / the --full
     * preset below for the actual defaults these override. */
    int pred_len = -1, d_model = -1, d_ff = -1, n_layers = -1, n_heads = -1, in_len = -1;
    /* Optimization hyperparameters:
     *   lambda   — weight of the auxiliary gate-sparsity term in model_loss()
     *   lr       — base AdamW learning rate (before warmup/decay multipliers)
     *   wd       — AdamW weight decay (decoupled L2)
     *   lr_decay — per-epoch multiplicative LR decay (1.0 = constant)
     *   dropout  — dropout probability inside the encoder blocks
     *   clip     — global gradient-norm clip threshold */
    float lambda = 0.1f, lr = 1e-3f, wd = 1e-5f, lr_decay = 1.0f, dropout = 0.1f, clip = 1.0f;
    int patience = 0;                 /* early stopping: stop after this many non-improving epochs (0 = disabled) */
    int warmup = 0;                   /* linear LR warmup steps: lr ramps 0→lr over the first `warmup` steps (0 = off) */
    int no_text = 0;                  /* ablation: zero out the text-embedding stream (numbers-only model) */
    int motion = 0;                   /* feature aug: append (lat,lon,dlat,dlon) to numeric inputs — CHANGES d_num */
    int no_lon = 0;                   /* ablation: zero --motion's ABSOLUTE-longitude column (memorization test) */
    int physics = 0;                  /* feature aug v2: acceleration, speed, heading, day-of-year — CHANGES d_num (+7) */
    float huber = 0.0f;               /* Huber loss transition point on the normalized residual (0 = MSE) */
    float hweight = 0.0f;             /* horizon-weight exponent: step h weighted (h+1)^γ, mean-normalized (0 = uniform) */
    int tf_epochs = 0;                /* teacher forcing: anneal force-prob 1→0 over this many epochs (0 = off; cv head) */
    int defer_dw = 0;                 /* deferred per-batch dW GEMMs (bit-identical; neutral on the reference box) */
    int delta = 0;                    /* decoder predicts displacement from the seed instead of absolute coords */
    /* --- decoder variants ---
     * These change the parameter set, so they must be set before model_new AND
     * matched at eval time. cv/gru/xattn all anchor the forecast at a constant-
     * velocity extrapolation and learn a curvature correction on top. All work
     * on both the serial and the data-parallel (--threads=N) paths: the workers
     * feed the same per-sample aux inputs (seed velocity, neighbours). */
    int cv = 0;                       /* constant-velocity anchor + learned curvature (2nd-order motion model) */
    int rotframe = 0;                 /* cv correction in the motion-aligned (along/cross-track) frame */
    int gru = 0;                      /* cv + a recurrent GRU memory over horizons (implies cv) */
    int xattn = 0;                    /* cv + decoder cross-attention over the encoder sequence (implies cv) */
    int km_loss = 0;                  /* reweight the longitude gradient by cos^2(lat) so loss ≈ km error (serial only) */
    /* Encoder structural options (see cmd_eval — each must be re-passed to
     * evaluate a checkpoint trained with it, since they alter the param set).
     * spatial=0 is the DEFAULT (no N=1 spatial blocks — FINDINGS §7); pass
     * --spatial for the paper-faithful architecture. */
    int spatial = 0, posenc = 0, pool_last = 0, prenorm = 0, timebias = 0, co_spatial = 0;  /* encoder options */
    unsigned long seed = 20260711, split_seed = 42;   /* weight-init/dropout RNG seed; storm-split RNG seed */
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *save = DEF_CKPT, *resume = NULL;
    /* Hand-rolled getopt: each arg is either a "--flag" toggle or a
     * "--key=value" pair matched by prefix (strncmp with the exact length that
     * includes the '='), with the value parsed from argv[i]+offset. A bare
     * leading-digit token is taken as the epoch count for backward compat. */
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--full"))     full = 1;             /* use the larger config_default() model */
        else if (!strncmp(argv[i], "--csv=", 6)) csv = argv[i] + 6;    /* raw track CSV path */
        else if (!strncmp(argv[i], "--emb=", 6)) emb = argv[i] + 6;    /* text-embedding chunk directory */
        else if (!strncmp(argv[i], "--bin=", 6)) bin = argv[i] + 6;    /* pre-windowed .tfb (overrides csv/emb) */
        else if (!strncmp(argv[i], "--save=", 7)) save = argv[i] + 7;  /* checkpoint output path */
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);  /* forecast horizons */
        else if (!strncmp(argv[i], "--d_model=", 10))  d_model = atoi(argv[i] + 10);   /* model width */
        else if (!strncmp(argv[i], "--d_ff=", 7))      d_ff = atoi(argv[i] + 7);       /* FFN hidden width */
        else if (!strncmp(argv[i], "--n_layers=", 11)) n_layers = atoi(argv[i] + 11);  /* encoder depth */
        else if (!strncmp(argv[i], "--n_heads=", 10))  n_heads = atoi(argv[i] + 10);   /* attention heads */
        else if (!strncmp(argv[i], "--in_len=", 9))    in_len = atoi(argv[i] + 9);     /* input window length */
        else if (!strncmp(argv[i], "--batch=", 8))     batch = atoi(argv[i] + 8);      /* minibatch size */
        else if (!strncmp(argv[i], "--threads=", 10))  threads = atoi(argv[i] + 10);   /* >1 selects data-parallel path */
        else if (!strncmp(argv[i], "--lr=", 5))        lr = (float)atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--wd=", 5))        wd = (float)atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--lambda=", 9))    lambda = (float)atof(argv[i] + 9);   /* gate-sparsity weight */
        else if (!strncmp(argv[i], "--lr_decay=", 11)) lr_decay = (float)atof(argv[i] + 11);
        else if (!strncmp(argv[i], "--dropout=", 10))  dropout = (float)atof(argv[i] + 10);
        else if (!strncmp(argv[i], "--clip=", 7))      clip = (float)atof(argv[i] + 7);
        else if (!strncmp(argv[i], "--warmup=", 9))    warmup = atoi(argv[i] + 9);      /* LR warmup steps */
        else if (!strncmp(argv[i], "--resume=", 9))    resume = argv[i] + 9;            /* checkpoint to continue from */
        else if (!strcmp(argv[i], "--no_text"))        no_text = 1;
        else if (!strcmp(argv[i], "--motion"))         motion = 1;
        else if (!strcmp(argv[i], "--no_lon"))         no_lon = 1;
        else if (!strcmp(argv[i], "--physics"))        physics = 1;
        else if (!strncmp(argv[i], "--huber=", 8))     huber = (float)atof(argv[i] + 8);
        else if (!strncmp(argv[i], "--hweight=", 10))  hweight = (float)atof(argv[i] + 10);
        else if (!strncmp(argv[i], "--tf=", 5))        tf_epochs = atoi(argv[i] + 5);
        else if (!strcmp(argv[i], "--defer_dw"))       defer_dw = 1;
        else if (!strcmp(argv[i], "--delta"))          delta = 1;
        /* --gru and --xattn each force cv=1 too: they are curvature corrections
         * layered on top of the constant-velocity anchor, not standalone modes. */
        else if (!strcmp(argv[i], "--cv"))             cv = 1;
        else if (!strcmp(argv[i], "--rotframe"))     { rotframe = 1; cv = 1; }  /* implies the cv anchor */
        else if (!strcmp(argv[i], "--gru"))          { gru = 1; cv = 1; }
        else if (!strcmp(argv[i], "--xattn"))        { xattn = 1; cv = 1; }
        else if (!strcmp(argv[i], "--km_loss"))        km_loss = 1;
        else if (!strcmp(argv[i], "--spatial"))        spatial = 1;      /* restore the paper's N=1 spatial blocks */
        else if (!strcmp(argv[i], "--no_spatial"))     spatial = 0;      /* legacy: now the default */
        else if (!strcmp(argv[i], "--posenc"))         posenc = 1;       /* add sinusoidal positional encoding */
        else if (!strcmp(argv[i], "--pool=last"))      pool_last = 1;    /* pool the last token instead of mean */
        else if (!strcmp(argv[i], "--prenorm"))        prenorm = 1;      /* pre-LN transformer block ordering */
        else if (!strcmp(argv[i], "--timebias"))       timebias = 1;     /* learned relative-time attention bias */
        else if (!strcmp(argv[i], "--co_spatial"))     co_spatial = 1;   /* feed co-active storm neighbours */
        else if (!strncmp(argv[i], "--split_seed=", 13)) split_seed = strtoul(argv[i] + 13, NULL, 10);  /* storm split RNG */
        else if (!strncmp(argv[i], "--patience=", 11)) patience = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--seed=", 7))      seed = strtoul(argv[i] + 7, NULL, 10);           /* weight/dropout RNG */
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') epochs = atoi(argv[i]);       /* bare number => epoch count */
    }
    nn_seed(seed);                                   /* seed the weight-init / shuffle RNG */
    nn_set_dropout(dropout);                         /* global dropout probability */
    nn_dropout_seed(seed ^ 0xD4090CULL);             /* separate RNG stream for dropout masks (decorrelated from init) */
    Config c = config_default();
    /* Default is a small/fast model; --full keeps config_default()'s larger
     * dimensions. Explicit --d_model/--d_ff/... below override either preset. */
    if (!full) { c.d_model = 64; c.d_ff = 128; c.n_layers = 2; }
    if (d_model  > 0) c.d_model  = d_model;
    if (d_ff     > 0) c.d_ff     = d_ff;
    if (n_layers > 0) c.n_layers = n_layers;
    if (n_heads  > 0) c.n_heads  = n_heads;
    if (in_len   > 0) c.in_len   = in_len;
    if (pred_len > 0) c.pred_len = pred_len;

    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    /* A .tfb file dictates its own window/embedding shapes; adopt them so the
     * model is built to match the data rather than the requested config. */
    if (bin) { c.in_len = ds.in_len; c.pred_len = ds.pred_len; c.d_text = ds.d_text; }
    if (motion) dataset_add_motion(&ds);             /* +lat,lon,dlat,dlon (before standardize) — grows d_num by 4 */
    if (physics) dataset_add_physics(&ds);           /* +accel,speed,heading,seasonal phase — grows d_num by 7 */
    /* --no_lon ablation: zero --motion's absolute-longitude column (index
     * old_d_num+1 = d_num-3 with motion alone, or -10 with physics stacked on
     * top). Tests whether absolute longitude is climatology signal or a
     * memorization vector on 98 storms. Zeroing (not removing) keeps d_num —
     * and therefore the checkpoint layout — unchanged; standardize sees a
     * constant column and clamps its std to 1, so it stays zeros. */
    if (no_lon && motion) {
        int lon_col = ds.d_num - (physics ? 10 : 3);
        for (int i = 0; i < ds.n_records; ++i) ds.num[(size_t)i * ds.d_num + lon_col] = 0.0f;
        printf("ablation: absolute-longitude feature zeroed (--no_lon)\n");
    }
    if (co_spatial) dataset_build_neighbors(&ds);    /* precompute co-active storm neighbours for the spatial branch */
    c.d_num = ds.d_num;                              /* lock the model's numeric input width to the (possibly grown) dataset */
    /* Leakage-safe: split whole STORMS into train/val/test, then fit feature +
     * coordinate normalization on the TRAIN storms only. (The .tfb path has no
     * storm info, so it splits by sample and standardizes itself.)
     * Splitting by storm — not by sample — prevents windows from the same
     * cyclone leaking across the train/val/test boundary, which would let the
     * model "cheat" by memorizing a track it also scores on. */
    ds.no_text |= no_text;                           /* propagate the numbers-only ablation (--emb=none already set it) */
    /* 0.15 val + 0.15 test => 0.70 train, keyed by split_seed for reproducibility. */
    Split sp = dataset_split3(&ds, 0.15f, 0.15f, split_seed);
    dataset_standardize(&ds);                        /* z-score features + coords using TRAIN-only statistics */
    int *train = sp.train, *val = sp.val, *test = sp.test;   /* arrays of sample ids per split */
    int ntr = sp.n_train, nva = sp.n_val, nte = sp.n_test;   /* their counts */
    printf("records=%d storms=%d samples=%d  train=%d val=%d test=%d | split_seed=%lu | d_num=%d%s%s%s\n",
           ds.n_records, ds.n_storms, ds.n_samples, ntr, nva, nte, split_seed, ds.d_num,
           motion ? " (+motion)" : "", physics ? " (+physics)" : "", no_text ? " | NO-TEXT" : "");
    /* Announce the active decoder branch. The order mirrors precedence below:
     * gru → xattn → cv → delta, each printed only if the higher ones are off. */
    if (gru)        printf("decoder: constant-velocity + GRU memory (recurrent curvature)\n");
    else if (xattn) printf("decoder: constant-velocity + cross-attention over encoder sequence\n");
    else if (cv)    printf("decoder: constant-velocity mode (anchor at CLIPER, learn curvature)\n");
    else if (delta) printf("decoder: delta mode (predict displacement from seed)\n");
    /* --km_loss reweights gradients inside the serial inner loop, which the
     * data-parallel workers never execute — warn if the user combined them. */
    if (km_loss && threads > 1) printf("note: --km_loss applies on the serial path; use --threads=1\n");

    if (threads < 1) threads = 1;                    /* clamp: threads=0 or negative is meaningless */
    /* architecture options — all set BEFORE model_new (they change the param set)
     * These are process-global toggles read when model_new() allocates params,
     * so they MUST be established before construction and re-established (via the
     * same flags) whenever the resulting checkpoint is later loaded for eval. */
    if (gru)        model_set_gru(1);         /* gru/xattn anchor at cv in their own branch (own curvature params) */
    else if (xattn) model_set_xattn(1);
    else if (cv)    model_set_cv(1);          /* cv is a superset of delta; takes precedence */
    else if (delta) model_set_delta(1);
    model_set_rotframe(rotframe);             /* plain-cv head only (no-op for gru/xattn) */
    if (rotframe && (gru || xattn)) printf("note: --rotframe applies to the plain cv head only\n");
    else if (rotframe) printf("decoder: motion-aligned (along/cross-track) correction frame\n");
    model_set_no_spatial(!spatial); model_set_posenc(posenc); model_set_pool_last(pool_last);
    nn_set_prenorm(prenorm); nn_set_timebias(timebias); model_set_co_spatial(co_spatial);
    /* Loss shaping (globals read by every model_loss call, both training paths). */
    model_set_huber(huber); model_set_hweight(hweight);
    /* Deferred weight-gradient accumulation (--defer_dw): one dW GEMM per layer
     * per BATCH instead of per sample. Bit-identical to the immediate path
     * (pinned by tests/test_parallel's defer check) but measured NEUTRAL on the
     * reference 4-core container — the backward is compute-bound, not
     * dW-traffic-bound, at these model sizes (see FINDINGS). Kept opt-in for
     * hardware where the gradient working set exceeds the last-level cache. */
    nn_set_defer_grads(defer_dw);
    if (tf_epochs > 0) printf("teacher forcing: prob 1 -> 0 over %d epochs (cv rollout)\n", tf_epochs);
    if (huber > 0.0f)    printf("loss: Huber, delta=%.3f (normalized residual)\n", huber);
    if (hweight != 0.0f) printf("loss: horizon-weighted, gamma=%.2f\n", hweight);
    ParamList pl; plist_init(&pl);                   /* flat registry of all trainable tensors + their grads */
    Model m = model_new(&c, &pl);                    /* build the model, registering its params into `pl` */
    Adam opt = adam_new(&pl, lr, wd);                /* AdamW state (m,v moments) sized to `pl` */
    /* Data-parallel worker pool: only allocated when --threads>1. Each worker
     * holds a full model replica; the master `pl` remains the source of truth. */
    ParTrainer *pt = (threads > 1) ? partrainer_new(&c, threads) : NULL;
    int start_ep = 0;                                /* epoch to resume after (0 = fresh run) */
    if (resume) {                                    /* resume weights + optimizer state */
        char opath[1024]; snprintf(opath, sizeof opath, "%s.opt", resume);   /* sidecar = "<ckpt>.opt" */
        checkpoint_load_params(resume, &pl);         /* restore weights into the freshly built model */
        /* The optimizer sidecar carries Adam moments + the last epoch. If it is
         * present the schedule continues seamlessly; if not, weights are kept
         * but Adam's moment estimates restart from zero. */
        if (checkpoint_load_optim(opath, &opt, &start_ep))
            printf("resumed from %s (+ %s) at epoch %d\n", resume, opath, start_ep);
        else
            printf("resumed weights from %s (no optimizer state; Adam restarts)\n", resume);
    }
    printf("model params = %ld | d_model=%d layers=%d heads=%d d_ff=%d pred_len=%d | threads=%d\n\n",
           plist_num_params(&pl), c.d_model, c.n_layers, c.n_heads, c.d_ff, c.pred_len, threads);

    /* Per-sample scratch tensors reused across the whole run (see evaluate()
     * for the xn/xt/yp/Y shapes). The two gradient buffers are model outputs:
     *   dpred [pred_len × out_dim] — ∂loss/∂prediction fed to model_backward
     *   dgate [in_len × d_model]   — ∂loss/∂gate (the auxiliary sparsity path) */
    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);
    Mat nbr = mat_new(TF_NBR_K, TF_NBR_NF);          /* neighbour feature scratch (co_spatial) */
    Mat vel = mat_new(1, 2);                         /* seed velocity scratch (cv/gru/xattn) */

    /* Epoch 0 = the randomly initialized model, scored for a baseline. */
    Eval e0 = evaluate(&m, &pl, pt, &ds, val, nva);
    printf("epoch  0 (init)          | val MAE %.4f | val dR %.2f km  (persistence %.2f km)\n",
           e0.mmae, e0.mkm, e0.mbase);
    /* Best-checkpoint tracking: `best` is the lowest val ΔR (km) seen so far;
     * since_improve counts consecutive non-improving epochs for early stopping. */
    double best = 1e300; int best_ep = 0, since_improve = 0; Eval best_e = e0;
    long gstep = 0; float decay_mult = 1.0f;         /* LR schedule state: global optimizer step count, per-epoch decay factor */
    for (int ep = start_ep + 1; ep <= epochs; ++ep) {
        /* Fisher–Yates shuffle of the train index array: for i from n-1 down to
         * 1, swap element i with a uniformly-random j in [0,i]. Gives an unbiased
         * permutation each epoch so minibatches differ run-to-run within a seed.
         * (j>i clamp guards the vanishingly rare nn_uniform()==1.0 case.) */
        for (int i = ntr - 1; i > 0; --i) { int j = (int)(nn_uniform(0, 1) * (i + 1)); if (j > i) j = i;
            int t = train[i]; train[i] = train[j]; train[j] = t; }
        double run_loss = 0.0; int nb = 0;           /* running sum of per-batch mean loss; batch counter */
        nn_set_training(1);                          /* dropout on for training */
        /* Teacher-forcing schedule: linear anneal 1 → 0 over tf_epochs, then
         * fully autoregressive. Set once per epoch, before any worker runs. */
        model_set_tf_prob(tf_epochs > 0 && ep <= tf_epochs
                          ? 1.0f - (float)(ep - 1) / (float)tf_epochs : 0.0f);
        /* Iterate the shuffled train set in contiguous minibatches. The last
         * batch may be short (bs < batch) when ntr is not a multiple of batch. */
        for (int b = 0; b < ntr; b += batch) {
            int bs = (b + batch <= ntr) ? batch : (ntr - b);   /* actual size of this (possibly ragged) batch */
            if (pt) {                                /* data-parallel across cores */
                /* Push the current master weights to every replica, then have
                 * the workers forward+backward their shard and reduce gradients
                 * back into the master `pl`. Workers feed the same per-sample
                 * aux inputs as the serial loop (neighbours, seed velocity);
                 * only --km_loss remains serial-only. */
                partrainer_broadcast(pt, &pl);       /* replicas <- master weights */
                run_loss += partrainer_step_grads(pt, &pl, &ds, train, b, bs, lambda);
            } else {                                 /* serial */
                plist_zero_grad(&pl);                /* clear grads; we accumulate over the batch below */
                double bl = 0.0;                     /* summed (un-normalized) loss over this batch */
                for (int k = 0; k < bs; ++k) {
                    dataset_get(&ds, train[b + k], xn, xt, yp, Y);
                    /* Serial-path-only auxiliary inputs, matched in evaluate(): */
                    if (co_spatial) { int nc; dataset_neighbors(&ds, train[b + k], nbr, &nc); model_set_neighbors(&m, nbr, nc); }
                    if (cv) { dataset_seed_velocity(&ds, train[b + k], vel); model_set_seed_velocity(&m, vel); }
                    if (tf_epochs > 0) model_set_teacher(&m, Y);   /* rollout state correction targets */
                    model_forward(&m, xn, xt, yp);
                    /* Forward loss and its gradients (dpred, dgate) for one sample. */
                    bl += model_loss(m.pred, Y, m.pgf.gate, lambda, dpred, dgate);
                    if (km_loss) {                       /* equirectangular: down-weight lon by cos^2(lat) */
                        /* Denormalize the seed latitude to degrees, then scale the
                         * longitude gradient by cos²(lat). On an equirectangular
                         * grid one degree of longitude spans cos(lat)·111 km, so
                         * this makes the optimized loss track kilometre error more
                         * faithfully at high latitudes. Latitude grad is untouched. */
                        float latd = yp.data[0] * ds.cstd[0] + ds.cmean[0];
                        float w = cosf(latd * (float)(PI_ / 180.0)); w *= w;
                        for (int h = 0; h < c.pred_len; ++h) dpred.data[h * 2 + 1] *= w;
                    }
                    /* Scale each sample's gradient by 1/bs BEFORE accumulating so
                     * the summed grad over the batch equals the MEAN gradient —
                     * matching the 1/bs averaging the parallel path does. */
                    for (int i = 0; i < c.pred_len * c.out_dim; ++i) dpred.data[i] /= bs;
                    for (int i = 0; i < c.in_len * c.d_model; ++i)  dgate.data[i] /= bs;
                    model_backward(&m, dpred, dgate);    /* accumulates into pl grads (no zero between k's) */
                }
                model_flush_grads(&m);               /* flush deferred dW: one GEMM per layer per batch */
                run_loss += bl / bs;                 /* record the batch's MEAN loss */
            }
            /* One optimizer step per minibatch, shared by both paths. */
            plist_clip_grad_norm(&pl, clip);         /* global-norm gradient clipping: rescale so ||g||₂ ≤ clip */
            ++gstep;
            /* Linear warmup: on step g<warmup the LR is scaled by g/warmup so it
             * ramps 0→1; afterwards warm=1. Multiplied by the per-epoch decay. */
            float warm = (warmup > 0 && gstep < warmup) ? (float)gstep / (float)warmup : 1.0f;
            opt.lr = lr * decay_mult * warm;         /* effective LR = base × epoch-decay × warmup */
            adam_step(&opt, &pl);                    /* AdamW update using the (clipped) master grads */
            ++nb;                                    /* batch count for the epoch-mean loss print */
        }
        /* End-of-epoch validation. Model selection is on val ΔR (mkm), NOT on
         * training loss, so we keep the checkpoint that generalizes best. */
        Eval e = evaluate(&m, &pl, pt, &ds, val, nva);
        printf("epoch %2d | train loss %.5f | val MAE %.4f | val dR %.2f km\n", ep, run_loss / nb, e.mmae, e.mkm);
        if (e.mkm < best) {                          /* keep the best model on disk */
            best = e.mkm; best_ep = ep; best_e = e; since_improve = 0;   /* new best: reset the patience counter */
            /* Persist config + BOTH normalization stat sets (feature mean/std and
             * coordinate cmean/cstd) alongside weights, so eval can reproduce the
             * exact preprocessing. */
            checkpoint_save3(save, c, ds.mean, ds.std, ds.d_num, ds.cmean, ds.cstd, &pl);
            char opath[1024]; snprintf(opath, sizeof opath, "%s.opt", save);
            checkpoint_save_optim(opath, &opt, ep);  /* sidecar for --resume: Adam moments + this epoch number */
        } else if (++since_improve >= patience && patience > 0) {
            /* Early stopping: bail once val ΔR has failed to improve for
             * `patience` consecutive epochs (only when patience>0). */
            printf("early stop: no val improvement for %d epochs\n", patience); break;
        }
        decay_mult *= lr_decay;                       /* per-epoch LR decay (1.0 = off) */
    }
    /* Degenerate case (0 epochs, or nothing ever beat init): still emit a
     * checkpoint so downstream eval/predict have weights to load. */
    if (best_ep == 0) checkpoint_save3(save, c, ds.mean, ds.std, ds.d_num, ds.cmean, ds.cstd, &pl);
    printf("best val dR %.2f km at epoch %d  ->  saved %s (%ld params)\n",
           best_ep ? best : e0.mkm, best_ep, save, plist_num_params(&pl));
    print_horizons(&best_e);

    /* Honest generalization estimate: reload the best checkpoint and score the
     * held-out TEST storms (never seen in training or model selection).
     * Reloading matters — the in-memory weights are from the LAST epoch, which
     * may be worse than the best; loading `save` restores the selected model. */
    if (nte > 0) {
        checkpoint_load_params(save, &pl);           /* overwrite live weights with the best checkpoint */
        Eval te = evaluate(&m, &pl, pt, &ds, test, nte);
        printf("HELD-OUT TEST: dR %.2f km | MAE %.4f  (persistence %.2f km) over %d samples\n",
               te.mkm, te.mmae, te.mbase, nte);
        print_horizons(&te);
    }

    /* Tear everything down in reverse order of allocation (scratch tensors,
     * worker pool, split arrays, optimizer, model, param list, dataset). */
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&dpred); mat_free(&dgate); mat_free(&nbr); mat_free(&vel);
    if (pt) partrainer_free(pt);
    split_free(&sp); adam_free(&opt); model_free(&m); plist_free(&pl); dataset_free(&ds);
    return 0;
}

/* Reproduce a checkpoint's feature augmentation by width arithmetic: --motion
 * adds exactly 4 numeric columns, --physics adds 7, both together add 11 — so
 * the difference between the checkpoint's d_num and the raw dataset's uniquely
 * identifies which augmentations to re-apply before evaluation. */
static void apply_feature_aug(Dataset *ds, int ckpt_dnum) {
    if (ds->prewindowed) return;
    int diff = ckpt_dnum - ds->d_num;
    if (diff == 4 || diff == 11) dataset_add_motion(ds);
    if (diff == 7 || diff == 11) dataset_add_physics(ds);
}

/* Apply a checkpoint's (train-only) normalization to a freshly loaded raw
 * dataset. No-op for the .tfb path, which standardizes itself.
 * At eval/predict time the dataset is loaded raw; to reproduce training-time
 * preprocessing exactly we must reuse the SAME statistics the checkpoint was
 * fit with — recomputing them on the eval data would leak and shift scales.
 * Two stat sets are loaded independently: feature mean/std (ns of them) and the
 * 2-D coordinate cmean/cstd (hc==nonzero if present); either may be absent. */
static void apply_ckpt_stats(Dataset *ds, const char *weights) {
    if (ds->prewindowed) return;
    float mean[64], std[64], cmean[2], cstd[2];
    int ns = checkpoint_load_stats(weights, mean, std);          /* count of feature stats found (0 => none) */
    int hc = checkpoint_load_coord_stats(weights, cmean, cstd);  /* nonzero => coord stats present */
    dataset_apply_stats(ds, ns ? mean : NULL, ns ? std : NULL, ns,
                        hc ? cmean : NULL, hc ? cstd : NULL);
}

/* ---- eval ----------------------------------------------------------- */
/* Load a checkpoint and score it over ALL samples of a dataset (no split — the
 * whole thing is treated as the eval set). Because the architecture flags
 * change the parameter layout, the SAME structural flags used at train time
 * must be repeated here or model_new() will build a mismatched model and the
 * weight load will be nonsense. Note these flags call the model_set_* toggles
 * directly (no local bool needed) except co_spatial, which also needs a local
 * flag to trigger neighbour precomputation on the dataset. */
static int cmd_eval(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *weights = DEF_CKPT;
    int no_text = 0, co_spatial = 0, threads = 1, no_lon = 0;
    /* Optional split-restricted scoring: --split_seed=S --split=test re-derives
     * the storm-level partition a checkpoint was trained with (same S => same
     * deterministic split) and scores ONLY that subset — the honest way to
     * report held-out numbers from `eval` (the default scores ALL samples,
     * which mixes training storms in). -1 = no restriction. */
    unsigned long split_seed = 42; int split_which = -1;
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;   /* checkpoint to evaluate */
        else if (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6))      bin = argv[i] + 6;
        else if (!strncmp(argv[i], "--threads=", 10)) threads = atoi(argv[i] + 10);  /* shard eval across cores */
        else if (!strncmp(argv[i], "--split_seed=", 13)) split_seed = strtoul(argv[i] + 13, NULL, 10);
        else if (!strcmp(argv[i], "--split=test"))   split_which = 2;   /* score the held-out test storms only */
        else if (!strcmp(argv[i], "--split=val"))    split_which = 1;
        else if (!strcmp(argv[i], "--split=train"))  split_which = 0;
        else if (!strcmp(argv[i], "--no_text"))       no_text = 1;
        else if (!strcmp(argv[i], "--no_lon"))        no_lon = 1;   /* must match a --no_lon checkpoint */
        /* Structural flags — MUST match the checkpoint's training flags: */
        else if (!strcmp(argv[i], "--delta"))         model_set_delta(1);
        else if (!strcmp(argv[i], "--cv"))            model_set_cv(1);
        else if (!strcmp(argv[i], "--rotframe"))    { model_set_cv(1); model_set_rotframe(1); }
        else if (!strcmp(argv[i], "--gru"))           model_set_gru(1);
        else if (!strcmp(argv[i], "--xattn"))         model_set_xattn(1);
        else if (!strcmp(argv[i], "--spatial"))       model_set_no_spatial(0);  /* ckpt trained with spatial blocks */
        else if (!strcmp(argv[i], "--no_spatial"))    model_set_no_spatial(1);  /* legacy: now the default */
        else if (!strcmp(argv[i], "--posenc"))        model_set_posenc(1);
        else if (!strcmp(argv[i], "--pool=last"))     model_set_pool_last(1);
        else if (!strcmp(argv[i], "--prenorm"))       nn_set_prenorm(1);
        else if (!strcmp(argv[i], "--timebias"))      nn_set_timebias(1);
        else if (!strcmp(argv[i], "--co_spatial"))  { model_set_co_spatial(1); co_spatial = 1; }
    }
    /* --weights=a.ckpt,b.ckpt,... loads an ENSEMBLE: predictions are averaged
     * in normalized coordinate space before scoring. Every member must share
     * the same Config (checked) and — for honest numbers — the same split and
     * normalization stats (train members with the same --split_seed and
     * different --seed values; stats are taken from the FIRST member). */
    #define MAX_ENS 16
    char wbuf[4096]; strncpy(wbuf, weights, sizeof wbuf - 1); wbuf[sizeof wbuf - 1] = 0;
    const char *wlist[MAX_ENS]; int K = 0;
    for (char *tok = strtok(wbuf, ","); tok && K < MAX_ENS; tok = strtok(NULL, ","))
        wlist[K++] = tok;
    if (K == 0) { fprintf(stderr, "eval: empty --weights\n"); return 1; }

    Config c = checkpoint_load_config(wlist[0]);     /* dimensions come from the checkpoint, not the CLI */
    printf("Loaded config from %s | d_model=%d layers=%d heads=%d d_ff=%d%s\n",
           wlist[0], c.d_model, c.n_layers, c.n_heads, c.d_ff,
           K > 1 ? " (ensemble)" : "");
    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    if (bin && ds.d_text != c.d_text) { die("d_text mismatch (data %d vs model %d)", ds.d_text, c.d_text); }

    ds.no_text |= no_text;                           /* --emb=none already set it */
    apply_feature_aug(&ds, c.d_num);                /* re-apply the ckpt's --motion/--physics (width arithmetic) */
    /* --no_lon checkpoints trained with the absolute-longitude column zeroed;
     * reproduce that here (same column arithmetic as cmd_train). */
    if (no_lon && !ds.prewindowed && c.d_num > 14) {
        int lon_col = ds.d_num - ((c.d_num - 14 == 11) ? 10 : 3);
        for (int i = 0; i < ds.n_records; ++i) ds.num[(size_t)i * ds.d_num + lon_col] = 0.0f;
    }
    if (co_spatial) dataset_build_neighbors(&ds);
    apply_ckpt_stats(&ds, wlist[0]);                 /* reuse the checkpoint's train-only normalization */
    ParamList pls[MAX_ENS]; Model msv[MAX_ENS]; Model *mps[MAX_ENS];
    for (int k = 0; k < K; ++k) {
        if (k > 0) {                                 /* members must be architecturally identical */
            Config ck = checkpoint_load_config(wlist[k]);
            if (memcmp(&ck, &c, sizeof(Config)) != 0)
                die("ensemble member %s: config mismatch vs %s", wlist[k], wlist[0]);
        }
        plist_init(&pls[k]);
        msv[k] = model_new(&c, &pls[k]);
        checkpoint_load_params(wlist[k], &pls[k]);   /* fill each member with its saved weights */
        mps[k] = &msv[k];
    }
    /* Optional eval sharding: a replica pool (same architecture — the model_set_*
     * flags above are already in effect) lets evaluate() split the samples
     * across cores. Single-model path only; the ensemble path runs serial. */
    /* Restrict scoring to one side of the storm-level split, if requested. */
    Split sp; memset(&sp, 0, sizeof sp);
    const int *eidx = NULL; int en = ds.n_samples;
    if (split_which >= 0) {
        sp = dataset_split3(&ds, 0.15f, 0.15f, split_seed);
        if      (split_which == 2) { eidx = sp.test;  en = sp.n_test; }
        else if (split_which == 1) { eidx = sp.val;   en = sp.n_val; }
        else                       { eidx = sp.train; en = sp.n_train; }
        printf("scoring the %s split only (split_seed=%lu): %d samples\n",
               split_which == 2 ? "TEST" : split_which == 1 ? "VAL" : "TRAIN", split_seed, en);
    }
    ParTrainer *ept = (threads > 1 && K == 1) ? partrainer_new(&c, threads) : NULL;
    Eval e;
    if (K == 1) {
        e = evaluate(&msv[0], &pls[0], ept, &ds, eidx, en);  /* idx==NULL => samples 0..n-1 */
    } else {
        nn_set_training(0);
        int H = ds.pred_len < MAXH ? ds.pred_len : MAXH;
        memset(&e, 0, sizeof e); e.H = H;
        eval_span(mps, K, &ds, eidx, 0, en, H, e.mae, e.km, e.base_km);
        for (int h = 0; h < H; ++h) {
            e.mae[h] /= en; e.km[h] /= en; e.base_km[h] /= en;
            e.mmae += e.mae[h]; e.mkm += e.km[h]; e.mbase += e.base_km[h];
        }
        e.mmae /= H; e.mkm /= H; e.mbase /= H;
    }
    printf("eval on %d samples%s | MAE %.4f | dR %.2f km  (persistence %.2f km)\n",
           en, K > 1 ? " (ensemble mean)" : "", e.mmae, e.mkm, e.mbase);
    print_horizons(&e);

    if (ept) partrainer_free(ept);
    if (split_which >= 0) split_free(&sp);
    for (int k = 0; k < K; ++k) { model_free(&msv[k]); plist_free(&pls[k]); }
    dataset_free(&ds);
    return 0;
}

/* ---- prepare (CSV + embeddings -> .tfb) ----------------------------- */
/* Offline windowing: read the raw CSV+embeddings, materialize every
 * (in_len → pred_len) training window, and write them to a compact binary .tfb
 * so later runs can `--bin=` load instantly without re-parsing. Numeric and
 * text features are concatenated per timestep into a single `feat`-wide row. */
static int cmd_prepare(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB, *out = NULL;
    int in_len = 12, pred_len = 1;                   /* default window: 12 steps in (72 h), 1 step out (6 h) */
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--out=", 6))      out = argv[i] + 6;    /* required output .tfb path */
        else if (!strncmp(argv[i], "--in_len=", 9))   in_len = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);
    }
    if (!out) { fprintf(stderr, "prepare: --out=FILE.tfb required\n"); return 1; }
    Dataset ds = dataset_load(csv, emb, in_len, pred_len);
    int feat = ds.d_num + ds.d_text, out_dim = 2;    /* per-timestep width = numeric+text; targets are (lat,lon) */
    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    int hdr[5] = {ds.n_samples, ds.in_len, feat, ds.pred_len, out_dim};   /* fixed 5-int header after the magic */
    /* TFB2: store the true last-observed seed coordinate per sample so the
     * decoder is not seeded with a target label. Features are written raw;
     * dataset_load_bin standardizes them on load.
     * On-disk layout: magic "TFB2", then hdr[5], then per sample:
     *   in_len rows of `feat` floats, then pred_len×2 target floats,
     *   then 2 seed floats. All little-endian float32 in file order. */
    fwrite("TFB2", 1, 4, f); fwrite(hdr, sizeof(int), 5, f);
    Mat xn = mat_new(ds.in_len, ds.d_num), xt = mat_new(ds.in_len, ds.d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(ds.pred_len, 2);
    float *row = malloc((size_t)feat * sizeof(float));   /* scratch to splice numeric+text into one contiguous row */
    for (int s = 0; s < ds.n_samples; ++s) {
        dataset_get(&ds, s, xn, xt, yp, Y);           /* coords raw (identity stats here) */
        for (int t = 0; t < ds.in_len; ++t) {
            /* Interleave: [numeric d_num values | text d_text values] per step. */
            memcpy(row, &xn.data[t * ds.d_num], ds.d_num * sizeof(float));
            memcpy(row + ds.d_num, &xt.data[t * ds.d_text], ds.d_text * sizeof(float));
            fwrite(row, sizeof(float), feat, f);
        }
        fwrite(Y.data, sizeof(float), (size_t)ds.pred_len * out_dim, f);   /* ground-truth future coords */
        fwrite(yp.data, sizeof(float), 2, f);          /* seed coordinate */
    }
    fclose(f);
    printf("wrote %s (TFB2): %d samples, in_len=%d, feat=%d, pred_len=%d\n", out, ds.n_samples, ds.in_len, feat, ds.pred_len);
    free(row); mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); dataset_free(&ds);
    return 0;
}

/* ---- predict (write predicted vs. true tracks to CSV) --------------- */
/* Run a checkpoint over the dataset and dump one CSV row per (sample, horizon)
 * with seed / truth / prediction coords in degrees plus the model and
 * persistence errors in km — the raw material for plotting tracks or error
 * curves. --n=K caps the number of samples written. */
static int cmd_predict(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *weights = DEF_CKPT, *out = "predictions.csv";
    int limit = -1;                                  /* -1 => all samples; else write at most `limit` */
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
        else if (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6))      bin = argv[i] + 6;
        else if (!strncmp(argv[i], "--out=", 6))      out = argv[i] + 6;    /* output CSV path */
        else if (!strncmp(argv[i], "--n=", 4))        limit = atoi(argv[i] + 4);
    }
    Config c = checkpoint_load_config(weights);
    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    apply_feature_aug(&ds, c.d_num);                 /* re-apply the ckpt's --motion/--physics (width arithmetic) */
    apply_ckpt_stats(&ds, weights);                  /* reproduce train-time normalization */
    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    checkpoint_load_params(weights, &pl);
    int n = (limit > 0 && limit < ds.n_samples) ? limit : ds.n_samples;   /* clamp --n to available samples */
    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    fprintf(f, "sample,horizon_h,seed_lat,seed_lon,true_lat,true_lon,pred_lat,pred_lon,error_km,persistence_km\n");
    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text), yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    for (int s = 0; s < n; ++s) {
        dataset_get(&ds, s, xn, xt, yp, Y);
        model_forward(&m, xn, xt, yp);               /* note: this path does not feed neighbours/velocity */
        float seed[2] = { yp.data[0], yp.data[1] };  /* last observed position; persistence anchor */
        dataset_denorm(&ds, seed);                       /* all outputs in degrees */
        for (int h = 0; h < c.pred_len; ++h) {
            float p[2] = { m.pred.data[h * 2], m.pred.data[h * 2 + 1] };   /* prediction at horizon h */
            float t[2] = { Y.data[h * 2],       Y.data[h * 2 + 1] };       /* ground truth at horizon h */
            dataset_denorm(&ds, p); dataset_denorm(&ds, t);
            fprintf(f, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f\n",
                    s, (h + 1) * 6, seed[0], seed[1], t[0], t[1], p[0], p[1],
                    haversine(p[0], p[1], t[0], t[1]),          /* model great-circle error, km */
                    haversine(seed[0], seed[1], t[0], t[1]));   /* persistence error, km */
        }
    }
    fclose(f);
    printf("wrote %s: %d samples x %d horizon(s)\n", out, n, c.pred_len);
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    model_free(&m); plist_free(&pl); dataset_free(&ds);
    return 0;
}

/* ---- baseline (persistence + constant-velocity / CLIPER-style) ------ */
/* Pure no-learning reference numbers computed straight from the coordinate
 * history — no model at all. Two classic track baselines any forecaster must
 * beat:
 *   persistence     — the storm stays put (future = last position).
 *   const-velocity  — extrapolate the last observed step's velocity linearly
 *                     (a CLIPER-style dead-reckoning forecast).
 * Both are reported as mean ΔR km per horizon. */
static int cmd_baseline(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB;
    int in_len = 12, pred_len = 4;                   /* default 4 horizons (24 h) for the baseline table */
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--csv=", 6))       csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))       emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);
    }
    Dataset ds = dataset_load(csv, emb, in_len, pred_len);   /* needs coordinate history */
    int H = pred_len < MAXH ? pred_len : MAXH;
    double pkm[MAXH] = {0}, vkm[MAXH] = {0};         /* accumulators: persistence km, const-velocity km per horizon */
    for (int s = 0; s < ds.n_samples; ++s) {
        int last = ds.start[s] + in_len - 1;         /* index of the last OBSERVED step of this window */
        /* Velocity = last minus previous observed position (one 6-h step). */
        float vlat = ds.lat[last] - ds.lat[last - 1], vlon = ds.lon[last] - ds.lon[last - 1];
        for (int h = 0; h < H; ++h) {
            float t_lat = ds.lat[last + 1 + h], t_lon = ds.lon[last + 1 + h];   /* truth at horizon h */
            pkm[h] += haversine(ds.lat[last], ds.lon[last], t_lat, t_lon);      /* persistence: no movement */
            /* const-velocity: last position + (h+1) linear velocity steps. */
            vkm[h] += haversine(ds.lat[last] + (h + 1) * vlat, ds.lon[last] + (h + 1) * vlon, t_lat, t_lon);
        }
    }
    printf("baselines on %d samples (ΔR km):\n", ds.n_samples);
    printf("  horizon | persistence | const-velocity (CLIPER-style)\n");
    for (int h = 0; h < H; ++h)
        printf("   %2dh    |   %7.2f   |   %7.2f\n", (h + 1) * 6, pkm[h] / ds.n_samples, vkm[h] / ds.n_samples);
    dataset_free(&ds);
    return 0;
}

/* ---- bench (forward / forward+backward throughput) ------------------ */
/* Micro-benchmark: build a model on RANDOM inputs (no dataset loaded) and time
 * pure forward vs. forward+loss+backward, reporting ms/sample and samples/s.
 * Handy for sizing configs and spotting perf regressions in the kernels. */
static int cmd_bench(int argc, char **argv) {
    int full = 0, iters = 200, d_model = -1, d_ff = -1, n_layers = -1, pred_len = -1;   /* iters = timed repetitions */
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--full"))           full = 1;
        else if (!strcmp(argv[i], "--spatial"))        model_set_no_spatial(0);  /* bench the paper-faithful encoder */
        else if (!strncmp(argv[i], "--iters=", 8))     iters = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--d_model=", 10))  d_model = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--d_ff=", 7))      d_ff = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--n_layers=", 11)) n_layers = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);
    }
    nn_seed(1);                                      /* fixed seed => reproducible random weights/inputs */
    Config c = config_default();
    if (!full) { c.d_model = 64; c.d_ff = 128; c.n_layers = 2; }   /* small preset unless --full */
    if (d_model  > 0) c.d_model  = d_model;
    if (d_ff     > 0) c.d_ff     = d_ff;
    if (n_layers > 0) c.n_layers = n_layers;
    if (pred_len > 0) c.pred_len = pred_len;
    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text), yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    for (int i = 0; i < c.in_len * c.d_num;  ++i) xn.data[i] = nn_uniform(-1, 1);   /* random numeric inputs */
    for (int i = 0; i < c.in_len * c.d_text; ++i) xt.data[i] = nn_uniform(-1, 1);   /* random text inputs   */
    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);

    for (int i = 0; i < 5; ++i) model_forward(&m, xn, xt, yp);            /* warmup: prime caches, exclude from timing */
    clock_t t0 = clock();
    for (int i = 0; i < iters; ++i) model_forward(&m, xn, xt, yp);        /* time forward only */
    double fwd = (double)(clock() - t0) / CLOCKS_PER_SEC / iters;         /* seconds per forward pass */
    t0 = clock();
    for (int i = 0; i < iters; ++i) {                                     /* time the full training step */
        model_forward(&m, xn, xt, yp);
        model_loss(m.pred, Y, m.pgf.gate, 0.1f, dpred, dgate);
        model_backward(&m, dpred, dgate);
    }
    double fb = (double)(clock() - t0) / CLOCKS_PER_SEC / iters;          /* seconds per forward+backward */
    /* The training step as cmd_train actually runs it: deferred dW, one flush
     * per batch of 8 (see nn_set_defer_grads) — same math, less dW traffic. */
    nn_set_defer_grads(1);
    t0 = clock();
    for (int i = 0; i < iters; ++i) {
        model_forward(&m, xn, xt, yp);
        model_loss(m.pred, Y, m.pgf.gate, 0.1f, dpred, dgate);
        model_backward(&m, dpred, dgate);
        if ((i + 1) % 8 == 0) model_flush_grads(&m);
    }
    model_flush_grads(&m);
    double fbd = (double)(clock() - t0) / CLOCKS_PER_SEC / iters;
    nn_set_defer_grads(0);

    printf("bench | d_model=%d d_ff=%d layers=%d heads=%d pred_len=%d | %ld params\n",
           c.d_model, c.d_ff, c.n_layers, c.n_heads, c.pred_len, plist_num_params(&pl));
    printf("  forward         : %8.3f ms/sample  (%8.1f samples/s)\n", fwd * 1e3, 1.0 / fwd);
    printf("  forward+backward: %8.3f ms/sample  (%8.1f samples/s)\n", fb * 1e3, 1.0 / fb);
    printf("  fwd+bwd deferred: %8.3f ms/sample  (%8.1f samples/s)  [dW flushed per 8-batch]\n", fbd * 1e3, 1.0 / fbd);
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&dpred); mat_free(&dgate);
    model_free(&m); plist_free(&pl);
    return 0;
}

/* Subcommand dispatch. The first non-program token selects a handler; each is
 * called with (argc-1, argv+1) so the subcommand name is stripped and every
 * cmd_* sees its own args starting at index 1 (matching the parse loops above).
 * Any invocation without a recognized leading keyword defaults to training,
 * preserving the historical `./typhoformer [epochs] ...` form. */
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "eval"))     return cmd_eval(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "prepare"))  return cmd_prepare(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "predict"))  return cmd_predict(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "baseline")) return cmd_baseline(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "bench"))    return cmd_bench(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "train"))    return cmd_train(argc - 1, argv + 1);
    return cmd_train(argc, argv);   /* default / backward compatible */
}
