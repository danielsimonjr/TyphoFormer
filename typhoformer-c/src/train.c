/*
 * train.c — TyphoFormer-C command-line driver.
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

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R_EARTH 6371.0
#define PI_ 3.14159265358979323846
#define DEF_CSV "../HURDAT_2new_3000.csv"
#define DEF_EMB "../embedding_chunks"
#define DEF_CKPT "typhoformer.ckpt"

/* ---- helpers -------------------------------------------------------- */
static double haversine(float lat1, float lon1, float lat2, float lon2) {
    double p = PI_ / 180.0;
    double a = 0.5 - cos((lat2 - lat1) * p) / 2.0 +
               cos(lat1 * p) * cos(lat2 * p) * (1.0 - cos((lon2 - lon1) * p)) / 2.0;
    return 2.0 * R_EARTH * asin(sqrt(a < 0 ? 0 : a));
}

#define MAXH 32
typedef struct {
    int    H;
    double mae[MAXH], km[MAXH], base_km[MAXH];  /* per forecast horizon */
    double mmae, mkm, mbase;                    /* horizon means */
} Eval;

static Eval evaluate(Model *m, const Dataset *d, const int *idx, int n) {
    nn_set_training(0);                              /* dropout off for evaluation */
    int H = d->pred_len < MAXH ? d->pred_len : MAXH;
    Mat xn = mat_new(d->in_len, d->d_num), xt = mat_new(d->in_len, d->d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(d->pred_len, 2);
    Eval e; memset(&e, 0, sizeof e); e.H = H;
    for (int i = 0; i < n; ++i) {
        int s = idx ? idx[i] : i;
        dataset_get(d, s, xn, xt, yp, Y);
        model_forward(m, xn, xt, yp);
        float seed[2] = { yp.data[0], yp.data[1] };
        dataset_denorm(d, seed);                        /* coords back to degrees for metrics */
        for (int h = 0; h < H; ++h) {
            float p[2] = { m->pred.data[h * 2], m->pred.data[h * 2 + 1] };
            float t[2] = { Y.data[h * 2],       Y.data[h * 2 + 1] };
            dataset_denorm(d, p); dataset_denorm(d, t);
            e.mae[h]     += (fabs(p[0] - t[0]) + fabs(p[1] - t[1])) / 2.0;
            e.km[h]      += haversine(p[0], p[1], t[0], t[1]);
            e.base_km[h] += haversine(seed[0], seed[1], t[0], t[1]);   /* persistence */
        }
    }
    for (int h = 0; h < H; ++h) {
        e.mae[h] /= n; e.km[h] /= n; e.base_km[h] /= n;
        e.mmae += e.mae[h]; e.mkm += e.km[h]; e.mbase += e.base_km[h];
    }
    e.mmae /= H; e.mkm /= H; e.mbase /= H;
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    return e;
}

static void print_horizons(const Eval *e) {
    if (e->H <= 1) return;
    printf("  per-horizon:\n");
    for (int h = 0; h < e->H; ++h)
        printf("    %2dh | MAE %.4f | dR %.2f km | persistence %.2f km\n",
               (h + 1) * 6, e->mae[h], e->km[h], e->base_km[h]);
}

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
static int cmd_train(int argc, char **argv) {
    int epochs = 10, full = 0, batch = 8, threads = 1;
    int pred_len = -1, d_model = -1, d_ff = -1, n_layers = -1, n_heads = -1, in_len = -1;
    float lambda = 0.1f, lr = 1e-3f, wd = 1e-5f, lr_decay = 1.0f, dropout = 0.1f, clip = 1.0f;
    int patience = 0;                 /* 0 = disabled (early stopping) */
    int warmup = 0;                   /* linear LR warmup steps (0 = off)  */
    int no_text = 0;                  /* numbers-only ablation             */
    int motion = 0;                   /* add position+velocity input features */
    int delta = 0;                    /* decoder predicts displacement     */
    int km_loss = 0;                  /* weight longitude error by cos^2(lat) */
    unsigned long seed = 20260711, split_seed = 42;
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *save = DEF_CKPT, *resume = NULL;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--full"))     full = 1;
        else if (!strncmp(argv[i], "--csv=", 6)) csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6)) emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6)) bin = argv[i] + 6;
        else if (!strncmp(argv[i], "--save=", 7)) save = argv[i] + 7;
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--d_model=", 10))  d_model = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--d_ff=", 7))      d_ff = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--n_layers=", 11)) n_layers = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--n_heads=", 10))  n_heads = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--in_len=", 9))    in_len = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--batch=", 8))     batch = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--threads=", 10))  threads = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--lr=", 5))        lr = (float)atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--wd=", 5))        wd = (float)atof(argv[i] + 5);
        else if (!strncmp(argv[i], "--lambda=", 9))    lambda = (float)atof(argv[i] + 9);
        else if (!strncmp(argv[i], "--lr_decay=", 11)) lr_decay = (float)atof(argv[i] + 11);
        else if (!strncmp(argv[i], "--dropout=", 10))  dropout = (float)atof(argv[i] + 10);
        else if (!strncmp(argv[i], "--clip=", 7))      clip = (float)atof(argv[i] + 7);
        else if (!strncmp(argv[i], "--warmup=", 9))    warmup = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--resume=", 9))    resume = argv[i] + 9;
        else if (!strcmp(argv[i], "--no_text"))        no_text = 1;
        else if (!strcmp(argv[i], "--motion"))         motion = 1;
        else if (!strcmp(argv[i], "--delta"))          delta = 1;
        else if (!strcmp(argv[i], "--km_loss"))        km_loss = 1;
        else if (!strncmp(argv[i], "--split_seed=", 13)) split_seed = strtoul(argv[i] + 13, NULL, 10);
        else if (!strncmp(argv[i], "--patience=", 11)) patience = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--seed=", 7))      seed = strtoul(argv[i] + 7, NULL, 10);
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') epochs = atoi(argv[i]);
    }
    nn_seed(seed);
    nn_set_dropout(dropout);
    nn_dropout_seed(seed ^ 0xD4090CULL);
    Config c = config_default();
    if (!full) { c.d_model = 64; c.d_ff = 128; c.n_layers = 2; }
    if (d_model  > 0) c.d_model  = d_model;
    if (d_ff     > 0) c.d_ff     = d_ff;
    if (n_layers > 0) c.n_layers = n_layers;
    if (n_heads  > 0) c.n_heads  = n_heads;
    if (in_len   > 0) c.in_len   = in_len;
    if (pred_len > 0) c.pred_len = pred_len;

    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    if (bin) { c.in_len = ds.in_len; c.pred_len = ds.pred_len; c.d_text = ds.d_text; }
    if (motion) dataset_add_motion(&ds);             /* +lat,lon,dlat,dlon (before standardize) */
    c.d_num = ds.d_num;
    /* Leakage-safe: split whole STORMS into train/val/test, then fit feature +
     * coordinate normalization on the TRAIN storms only. (The .tfb path has no
     * storm info, so it splits by sample and standardizes itself.) */
    ds.no_text = no_text;
    Split sp = dataset_split3(&ds, 0.15f, 0.15f, split_seed);
    dataset_standardize(&ds);
    int *train = sp.train, *val = sp.val, *test = sp.test;
    int ntr = sp.n_train, nva = sp.n_val, nte = sp.n_test;
    printf("records=%d storms=%d samples=%d  train=%d val=%d test=%d | split_seed=%lu | d_num=%d%s%s\n",
           ds.n_records, ds.n_storms, ds.n_samples, ntr, nva, nte, split_seed, ds.d_num,
           motion ? " (+motion)" : "", no_text ? " | NO-TEXT" : "");
    if (delta) printf("decoder: delta mode (predict displacement from seed)\n");
    if (km_loss && threads > 1) printf("note: --km_loss applies on the serial path; use --threads=1\n");

    if (threads < 1) threads = 1;
    if (delta) model_set_delta(1);                   /* before model_new (zero-inits fc2) */
    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    Adam opt = adam_new(&pl, lr, wd);
    ParTrainer *pt = (threads > 1) ? partrainer_new(&c, threads) : NULL;
    int start_ep = 0;
    if (resume) {                                    /* resume weights + optimizer state */
        char opath[1024]; snprintf(opath, sizeof opath, "%s.opt", resume);
        checkpoint_load_params(resume, &pl);
        if (checkpoint_load_optim(opath, &opt, &start_ep))
            printf("resumed from %s (+ %s) at epoch %d\n", resume, opath, start_ep);
        else
            printf("resumed weights from %s (no optimizer state; Adam restarts)\n", resume);
    }
    printf("model params = %ld | d_model=%d layers=%d heads=%d d_ff=%d pred_len=%d | threads=%d\n\n",
           plist_num_params(&pl), c.d_model, c.n_layers, c.n_heads, c.d_ff, c.pred_len, threads);

    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);

    Eval e0 = evaluate(&m, &ds, val, nva);
    printf("epoch  0 (init)          | val MAE %.4f | val dR %.2f km  (persistence %.2f km)\n",
           e0.mmae, e0.mkm, e0.mbase);
    double best = 1e300; int best_ep = 0, since_improve = 0; Eval best_e = e0;
    long gstep = 0; float decay_mult = 1.0f;         /* LR schedule state */
    for (int ep = start_ep + 1; ep <= epochs; ++ep) {
        for (int i = ntr - 1; i > 0; --i) { int j = (int)(nn_uniform(0, 1) * (i + 1)); if (j > i) j = i;
            int t = train[i]; train[i] = train[j]; train[j] = t; }
        double run_loss = 0.0; int nb = 0;
        nn_set_training(1);                          /* dropout on for training */
        for (int b = 0; b < ntr; b += batch) {
            int bs = (b + batch <= ntr) ? batch : (ntr - b);
            if (pt) {                                /* data-parallel across cores */
                partrainer_broadcast(pt, &pl);       /* replicas <- master weights */
                run_loss += partrainer_step_grads(pt, &pl, &ds, train, b, bs, lambda);
            } else {                                 /* serial */
                plist_zero_grad(&pl);
                double bl = 0.0;
                for (int k = 0; k < bs; ++k) {
                    dataset_get(&ds, train[b + k], xn, xt, yp, Y);
                    model_forward(&m, xn, xt, yp);
                    bl += model_loss(m.pred, Y, m.pgf.gate, lambda, dpred, dgate);
                    if (km_loss) {                       /* equirectangular: down-weight lon by cos^2(lat) */
                        float latd = yp.data[0] * ds.cstd[0] + ds.cmean[0];
                        float w = cosf(latd * (float)(PI_ / 180.0)); w *= w;
                        for (int h = 0; h < c.pred_len; ++h) dpred.data[h * 2 + 1] *= w;
                    }
                    for (int i = 0; i < c.pred_len * c.out_dim; ++i) dpred.data[i] /= bs;
                    for (int i = 0; i < c.in_len * c.d_model; ++i)  dgate.data[i] /= bs;
                    model_backward(&m, dpred, dgate);
                }
                run_loss += bl / bs;
            }
            plist_clip_grad_norm(&pl, clip);         /* global-norm gradient clipping */
            ++gstep;
            float warm = (warmup > 0 && gstep < warmup) ? (float)gstep / (float)warmup : 1.0f;
            opt.lr = lr * decay_mult * warm;         /* warmup x per-epoch decay */
            adam_step(&opt, &pl);
            ++nb;
        }
        Eval e = evaluate(&m, &ds, val, nva);
        printf("epoch %2d | train loss %.5f | val MAE %.4f | val dR %.2f km\n", ep, run_loss / nb, e.mmae, e.mkm);
        if (e.mkm < best) {                          /* keep the best model on disk */
            best = e.mkm; best_ep = ep; best_e = e; since_improve = 0;
            checkpoint_save3(save, c, ds.mean, ds.std, ds.d_num, ds.cmean, ds.cstd, &pl);
            char opath[1024]; snprintf(opath, sizeof opath, "%s.opt", save);
            checkpoint_save_optim(opath, &opt, ep);  /* sidecar for --resume */
        } else if (++since_improve >= patience && patience > 0) {
            printf("early stop: no val improvement for %d epochs\n", patience); break;
        }
        decay_mult *= lr_decay;                       /* per-epoch LR decay (1.0 = off) */
    }
    if (best_ep == 0) checkpoint_save3(save, c, ds.mean, ds.std, ds.d_num, ds.cmean, ds.cstd, &pl);
    printf("best val dR %.2f km at epoch %d  ->  saved %s (%ld params)\n",
           best_ep ? best : e0.mkm, best_ep, save, plist_num_params(&pl));
    print_horizons(&best_e);

    /* Honest generalization estimate: reload the best checkpoint and score the
     * held-out TEST storms (never seen in training or model selection). */
    if (nte > 0) {
        checkpoint_load_params(save, &pl);
        Eval te = evaluate(&m, &ds, test, nte);
        printf("HELD-OUT TEST: dR %.2f km | MAE %.4f  (persistence %.2f km) over %d samples\n",
               te.mkm, te.mmae, te.mbase, nte);
        print_horizons(&te);
    }

    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&dpred); mat_free(&dgate);
    if (pt) partrainer_free(pt);
    split_free(&sp); adam_free(&opt); model_free(&m); plist_free(&pl); dataset_free(&ds);
    return 0;
}

/* Apply a checkpoint's (train-only) normalization to a freshly loaded raw
 * dataset. No-op for the .tfb path, which standardizes itself. */
static void apply_ckpt_stats(Dataset *ds, const char *weights) {
    if (ds->prewindowed) return;
    float mean[64], std[64], cmean[2], cstd[2];
    int ns = checkpoint_load_stats(weights, mean, std);
    int hc = checkpoint_load_coord_stats(weights, cmean, cstd);
    dataset_apply_stats(ds, ns ? mean : NULL, ns ? std : NULL, ns,
                        hc ? cmean : NULL, hc ? cstd : NULL);
}

/* ---- eval ----------------------------------------------------------- */
static int cmd_eval(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *weights = DEF_CKPT;
    int no_text = 0;
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
        else if (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6))      bin = argv[i] + 6;
        else if (!strcmp(argv[i], "--no_text"))       no_text = 1;
        else if (!strcmp(argv[i], "--delta"))         model_set_delta(1);
    }
    Config c = checkpoint_load_config(weights);
    printf("Loaded config from %s | d_model=%d layers=%d heads=%d d_ff=%d\n",
           weights, c.d_model, c.n_layers, c.n_heads, c.d_ff);
    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    if (bin && ds.d_text != c.d_text) { die("d_text mismatch (data %d vs model %d)", ds.d_text, c.d_text); }

    ds.no_text = no_text;
    if (!ds.prewindowed && c.d_num == ds.d_num + 4) dataset_add_motion(&ds);  /* ckpt trained --motion */
    apply_ckpt_stats(&ds, weights);
    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    checkpoint_load_params(weights, &pl);
    Eval e = evaluate(&m, &ds, NULL, ds.n_samples);
    printf("eval on %d samples | MAE %.4f | dR %.2f km  (persistence %.2f km)\n",
           ds.n_samples, e.mmae, e.mkm, e.mbase);
    print_horizons(&e);

    model_free(&m); plist_free(&pl); dataset_free(&ds);
    return 0;
}

/* ---- prepare (CSV + embeddings -> .tfb) ----------------------------- */
static int cmd_prepare(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB, *out = NULL;
    int in_len = 12, pred_len = 1;
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--out=", 6))      out = argv[i] + 6;
        else if (!strncmp(argv[i], "--in_len=", 9))   in_len = atoi(argv[i] + 9);
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);
    }
    if (!out) { fprintf(stderr, "prepare: --out=FILE.tfb required\n"); return 1; }
    Dataset ds = dataset_load(csv, emb, in_len, pred_len);
    int feat = ds.d_num + ds.d_text, out_dim = 2;
    FILE *f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    int hdr[5] = {ds.n_samples, ds.in_len, feat, ds.pred_len, out_dim};
    /* TFB2: store the true last-observed seed coordinate per sample so the
     * decoder is not seeded with a target label. Features are written raw;
     * dataset_load_bin standardizes them on load. */
    fwrite("TFB2", 1, 4, f); fwrite(hdr, sizeof(int), 5, f);
    Mat xn = mat_new(ds.in_len, ds.d_num), xt = mat_new(ds.in_len, ds.d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(ds.pred_len, 2);
    float *row = malloc((size_t)feat * sizeof(float));
    for (int s = 0; s < ds.n_samples; ++s) {
        dataset_get(&ds, s, xn, xt, yp, Y);           /* coords raw (identity stats here) */
        for (int t = 0; t < ds.in_len; ++t) {
            memcpy(row, &xn.data[t * ds.d_num], ds.d_num * sizeof(float));
            memcpy(row + ds.d_num, &xt.data[t * ds.d_text], ds.d_text * sizeof(float));
            fwrite(row, sizeof(float), feat, f);
        }
        fwrite(Y.data, sizeof(float), (size_t)ds.pred_len * out_dim, f);
        fwrite(yp.data, sizeof(float), 2, f);          /* seed coordinate */
    }
    fclose(f);
    printf("wrote %s (TFB2): %d samples, in_len=%d, feat=%d, pred_len=%d\n", out, ds.n_samples, ds.in_len, feat, ds.pred_len);
    free(row); mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); dataset_free(&ds);
    return 0;
}

/* ---- predict (write predicted vs. true tracks to CSV) --------------- */
static int cmd_predict(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *weights = DEF_CKPT, *out = "predictions.csv";
    int limit = -1;
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
        else if (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6))      bin = argv[i] + 6;
        else if (!strncmp(argv[i], "--out=", 6))      out = argv[i] + 6;
        else if (!strncmp(argv[i], "--n=", 4))        limit = atoi(argv[i] + 4);
    }
    Config c = checkpoint_load_config(weights);
    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    if (!ds.prewindowed && c.d_num == ds.d_num + 4) dataset_add_motion(&ds);   /* ckpt trained --motion */
    apply_ckpt_stats(&ds, weights);
    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    checkpoint_load_params(weights, &pl);
    int n = (limit > 0 && limit < ds.n_samples) ? limit : ds.n_samples;
    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); return 1; }
    fprintf(f, "sample,horizon_h,seed_lat,seed_lon,true_lat,true_lon,pred_lat,pred_lon,error_km,persistence_km\n");
    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text), yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    for (int s = 0; s < n; ++s) {
        dataset_get(&ds, s, xn, xt, yp, Y);
        model_forward(&m, xn, xt, yp);
        float seed[2] = { yp.data[0], yp.data[1] };
        dataset_denorm(&ds, seed);                       /* all outputs in degrees */
        for (int h = 0; h < c.pred_len; ++h) {
            float p[2] = { m.pred.data[h * 2], m.pred.data[h * 2 + 1] };
            float t[2] = { Y.data[h * 2],       Y.data[h * 2 + 1] };
            dataset_denorm(&ds, p); dataset_denorm(&ds, t);
            fprintf(f, "%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f\n",
                    s, (h + 1) * 6, seed[0], seed[1], t[0], t[1], p[0], p[1],
                    haversine(p[0], p[1], t[0], t[1]),
                    haversine(seed[0], seed[1], t[0], t[1]));
        }
    }
    fclose(f);
    printf("wrote %s: %d samples x %d horizon(s)\n", out, n, c.pred_len);
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    model_free(&m); plist_free(&pl); dataset_free(&ds);
    return 0;
}

/* ---- baseline (persistence + constant-velocity / CLIPER-style) ------ */
static int cmd_baseline(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB;
    int in_len = 12, pred_len = 4;
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--csv=", 6))       csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))       emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);
    }
    Dataset ds = dataset_load(csv, emb, in_len, pred_len);   /* needs coordinate history */
    int H = pred_len < MAXH ? pred_len : MAXH;
    double pkm[MAXH] = {0}, vkm[MAXH] = {0};
    for (int s = 0; s < ds.n_samples; ++s) {
        int last = ds.start[s] + in_len - 1;
        float vlat = ds.lat[last] - ds.lat[last - 1], vlon = ds.lon[last] - ds.lon[last - 1];
        for (int h = 0; h < H; ++h) {
            float t_lat = ds.lat[last + 1 + h], t_lon = ds.lon[last + 1 + h];
            pkm[h] += haversine(ds.lat[last], ds.lon[last], t_lat, t_lon);
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
static int cmd_bench(int argc, char **argv) {
    int full = 0, iters = 200, d_model = -1, d_ff = -1, n_layers = -1, pred_len = -1;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--full"))           full = 1;
        else if (!strncmp(argv[i], "--iters=", 8))     iters = atoi(argv[i] + 8);
        else if (!strncmp(argv[i], "--d_model=", 10))  d_model = atoi(argv[i] + 10);
        else if (!strncmp(argv[i], "--d_ff=", 7))      d_ff = atoi(argv[i] + 7);
        else if (!strncmp(argv[i], "--n_layers=", 11)) n_layers = atoi(argv[i] + 11);
        else if (!strncmp(argv[i], "--pred_len=", 11)) pred_len = atoi(argv[i] + 11);
    }
    nn_seed(1);
    Config c = config_default();
    if (!full) { c.d_model = 64; c.d_ff = 128; c.n_layers = 2; }
    if (d_model  > 0) c.d_model  = d_model;
    if (d_ff     > 0) c.d_ff     = d_ff;
    if (n_layers > 0) c.n_layers = n_layers;
    if (pred_len > 0) c.pred_len = pred_len;
    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text), yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    for (int i = 0; i < c.in_len * c.d_num;  ++i) xn.data[i] = nn_uniform(-1, 1);
    for (int i = 0; i < c.in_len * c.d_text; ++i) xt.data[i] = nn_uniform(-1, 1);
    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);

    for (int i = 0; i < 5; ++i) model_forward(&m, xn, xt, yp);            /* warmup */
    clock_t t0 = clock();
    for (int i = 0; i < iters; ++i) model_forward(&m, xn, xt, yp);
    double fwd = (double)(clock() - t0) / CLOCKS_PER_SEC / iters;
    t0 = clock();
    for (int i = 0; i < iters; ++i) {
        model_forward(&m, xn, xt, yp);
        model_loss(m.pred, Y, m.pgf.gate, 0.1f, dpred, dgate);
        model_backward(&m, dpred, dgate);
    }
    double fb = (double)(clock() - t0) / CLOCKS_PER_SEC / iters;

    printf("bench | d_model=%d d_ff=%d layers=%d heads=%d pred_len=%d | %ld params\n",
           c.d_model, c.d_ff, c.n_layers, c.n_heads, c.pred_len, plist_num_params(&pl));
    printf("  forward         : %8.3f ms/sample  (%8.1f samples/s)\n", fwd * 1e3, 1.0 / fwd);
    printf("  forward+backward: %8.3f ms/sample  (%8.1f samples/s)\n", fb * 1e3, 1.0 / fb);
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&dpred); mat_free(&dgate);
    model_free(&m); plist_free(&pl);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "eval"))     return cmd_eval(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "prepare"))  return cmd_prepare(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "predict"))  return cmd_predict(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "baseline")) return cmd_baseline(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "bench"))    return cmd_bench(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "train"))    return cmd_train(argc - 1, argv + 1);
    return cmd_train(argc, argv);   /* default / backward compatible */
}
