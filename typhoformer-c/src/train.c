/*
 * train.c — TyphoFormer-C command-line driver.
 *
 * Subcommands (default is "train", so `./typhoformer 30` still trains):
 *   ./typhoformer train   [epochs] [--full] [--csv=F --emb=D | --bin=F] [--save=CKPT]
 *   ./typhoformer eval     [--weights=CKPT] [--csv=F --emb=D | --bin=F]
 *   ./typhoformer prepare  --csv=F --emb=D --out=DATA.tfb [--in_len=12 --pred_len=1]
 */
#include "data.h"
#include "model.h"
#include "optim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define R_EARTH 6371.0
#define PI_ 3.14159265358979323846
#define CKPT_MAGIC "TFW1"
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

typedef struct { double mae, km, base_km; } Eval;

static Eval evaluate(Model *m, const Dataset *d, const int *idx, int n) {
    Mat xn = mat_new(d->in_len, d->d_num), xt = mat_new(d->in_len, d->d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(d->pred_len, 2);
    Eval e = {0, 0, 0};
    for (int i = 0; i < n; ++i) {
        int s = idx ? idx[i] : i;
        dataset_get(d, s, xn, xt, yp, Y);
        model_forward(m, xn, xt, yp);
        float plat = m->pred.data[0], plon = m->pred.data[1];
        e.mae += (fabs(plat - Y.data[0]) + fabs(plon - Y.data[1])) / 2.0;
        e.km  += haversine(plat, plon, Y.data[0], Y.data[1]);
        e.base_km += haversine(yp.data[0], yp.data[1], Y.data[0], Y.data[1]);
    }
    e.mae /= n; e.km /= n; e.base_km /= n;
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    return e;
}

/* checkpoint = magic + Config (9 ints) + flat float32 parameters (in order) */
static void ckpt_save(const char *path, Config c, const ParamList *pl) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    int h[9] = {c.d_num, c.d_text, c.d_model, c.out_dim, c.in_len, c.pred_len, c.d_ff, c.n_heads, c.n_layers};
    fwrite(CKPT_MAGIC, 1, 4, f); fwrite(h, sizeof(int), 9, f);
    for (int p = 0; p < pl->count; ++p) fwrite(pl->item[p].v, sizeof(float), pl->item[p].n, f);
    fclose(f);
    printf("saved checkpoint -> %s (%ld params)\n", path, plist_num_params(pl));
}
static Config ckpt_read_config(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { die("cannot open %s", path); }
    char magic[4]; int h[9];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, CKPT_MAGIC, 4)) { die("%s: bad checkpoint", path); }
    if (fread(h, sizeof(int), 9, f) != 9) die("%s: unexpected end of file", path);
    fclose(f);
    Config c = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};
    return c;
}
static void ckpt_read_params(const char *path, ParamList *pl) {
    FILE *f = fopen(path, "rb");
    fseek(f, 4 + 9 * (long)sizeof(int), SEEK_SET);
    for (int p = 0; p < pl->count; ++p)
        if (fread(pl->item[p].v, sizeof(float), pl->item[p].n, f) != (size_t)pl->item[p].n) {
            die("%s: parameter size mismatch", path);
        }
    fclose(f);
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
    int epochs = 10, full = 0;
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *save = DEF_CKPT;
    const float lambda = 0.1f; const int batch = 8;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--full"))     full = 1;
        else if (!strncmp(argv[i], "--csv=", 6)) csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6)) emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6)) bin = argv[i] + 6;
        else if (!strncmp(argv[i], "--save=", 7)) save = argv[i] + 7;
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') epochs = atoi(argv[i]);
    }
    nn_seed(20260711);
    Config c = config_default();
    if (!full) { c.d_model = 64; c.d_ff = 128; c.n_layers = 2; }

    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    if (bin) { c.in_len = ds.in_len; c.pred_len = ds.pred_len; c.d_text = ds.d_text; }
    int *train, *val, ntr, nva;
    dataset_split(&ds, 0.15f, 42, &train, &ntr, &val, &nva);
    printf("records=%d samples=%d  train=%d val=%d\n", ds.n_records, ds.n_samples, ntr, nva);

    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    Adam opt = adam_new(&pl, 1e-3f, 1e-5f);
    printf("model params = %ld | d_model=%d layers=%d heads=%d d_ff=%d\n\n",
           plist_num_params(&pl), c.d_model, c.n_layers, c.n_heads, c.d_ff);

    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);

    Eval e0 = evaluate(&m, &ds, val, nva);
    printf("epoch  0 (init)          | val MAE %.4f | val dR %.2f km  (persistence %.2f km)\n",
           e0.mae, e0.km, e0.base_km);
    for (int ep = 1; ep <= epochs; ++ep) {
        for (int i = ntr - 1; i > 0; --i) { int j = (int)(nn_uniform(0, 1) * (i + 1)); if (j > i) j = i;
            int t = train[i]; train[i] = train[j]; train[j] = t; }
        double run_loss = 0.0; int nb = 0;
        for (int b = 0; b < ntr; b += batch) {
            int bs = (b + batch <= ntr) ? batch : (ntr - b);
            plist_zero_grad(&pl);
            double bl = 0.0;
            for (int k = 0; k < bs; ++k) {
                dataset_get(&ds, train[b + k], xn, xt, yp, Y);
                model_forward(&m, xn, xt, yp);
                bl += model_loss(m.pred, Y, m.pgf.gate, lambda, dpred, dgate);
                for (int i = 0; i < c.pred_len * c.out_dim; ++i) dpred.data[i] /= bs;
                for (int i = 0; i < c.in_len * c.d_model; ++i)  dgate.data[i] /= bs;
                model_backward(&m, dpred, dgate);
            }
            adam_step(&opt, &pl);
            run_loss += bl / bs; ++nb;
        }
        Eval e = evaluate(&m, &ds, val, nva);
        printf("epoch %2d | train loss %.5f | val MAE %.4f | val dR %.2f km\n", ep, run_loss / nb, e.mae, e.km);
    }
    ckpt_save(save, c, &pl);

    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&dpred); mat_free(&dgate);
    free(train); free(val); adam_free(&opt); model_free(&m); plist_free(&pl); dataset_free(&ds);
    return 0;
}

/* ---- eval ----------------------------------------------------------- */
static int cmd_eval(int argc, char **argv) {
    const char *csv = DEF_CSV, *emb = DEF_EMB, *bin = NULL, *weights = DEF_CKPT;
    for (int i = 1; i < argc; ++i) {
        if      (!strncmp(argv[i], "--weights=", 10)) weights = argv[i] + 10;
        else if (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6))      bin = argv[i] + 6;
    }
    Config c = ckpt_read_config(weights);
    printf("Loaded config from %s | d_model=%d layers=%d heads=%d d_ff=%d\n",
           weights, c.d_model, c.n_layers, c.n_heads, c.d_ff);
    Dataset ds = load_source(bin, csv, emb, c.in_len, c.pred_len);
    if (bin && ds.d_text != c.d_text) { die("d_text mismatch (data %d vs model %d)", ds.d_text, c.d_text); }

    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    ckpt_read_params(weights, &pl);
    Eval e = evaluate(&m, &ds, NULL, ds.n_samples);
    printf("eval on %d samples | MAE %.4f | dR %.2f km  (persistence %.2f km)\n",
           ds.n_samples, e.mae, e.km, e.base_km);

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
    fwrite("TFB1", 1, 4, f); fwrite(hdr, sizeof(int), 5, f);
    Mat xn = mat_new(ds.in_len, ds.d_num), xt = mat_new(ds.in_len, ds.d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(ds.pred_len, 2);
    float *row = malloc((size_t)feat * sizeof(float));
    for (int s = 0; s < ds.n_samples; ++s) {
        dataset_get(&ds, s, xn, xt, yp, Y);
        for (int t = 0; t < ds.in_len; ++t) {
            memcpy(row, &xn.data[t * ds.d_num], ds.d_num * sizeof(float));
            memcpy(row + ds.d_num, &xt.data[t * ds.d_text], ds.d_text * sizeof(float));
            fwrite(row, sizeof(float), feat, f);
        }
        fwrite(Y.data, sizeof(float), (size_t)ds.pred_len * out_dim, f);
    }
    fclose(f);
    printf("wrote %s: %d samples, in_len=%d, feat=%d, pred_len=%d\n", out, ds.n_samples, ds.in_len, feat, ds.pred_len);
    free(row); mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); dataset_free(&ds);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "eval"))    return cmd_eval(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "prepare")) return cmd_prepare(argc - 1, argv + 1);
    if (argc > 1 && !strcmp(argv[1], "train"))   return cmd_train(argc - 1, argv + 1);
    return cmd_train(argc, argv);   /* default / backward compatible */
}
