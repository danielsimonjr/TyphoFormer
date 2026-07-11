/*
 * train.c — TyphoFormer-C training entry point.
 *
 * Usage: ./typhoformer [epochs] [csv] [embdir]
 * Trains the model on the bundled HURDAT2 sample data and reports, per epoch,
 * the training loss and validation MAE / spherical-distance error (km).
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
        dataset_get(d, idx[i], xn, xt, yp, Y);
        model_forward(m, xn, xt, yp);
        float plat = m->pred.data[0], plon = m->pred.data[1];
        e.mae += (fabs(plat - Y.data[0]) + fabs(plon - Y.data[1])) / 2.0;
        e.km  += haversine(plat, plon, Y.data[0], Y.data[1]);
        e.base_km += haversine(yp.data[0], yp.data[1], Y.data[0], Y.data[1]); /* persistence */
    }
    e.mae /= n; e.km /= n; e.base_km /= n;
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    return e;
}

int main(int argc, char **argv) {
    int epochs = 10, full = 0;
    const char *csv = "../HURDAT_2new_3000.csv";
    const char *emb = "../embedding_chunks";
    const char *bin = NULL;
    const float lambda = 0.1f;
    const int   batch = 8;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--full"))          full = 1;
        else if (!strncmp(argv[i], "--csv=", 6))      csv = argv[i] + 6;
        else if (!strncmp(argv[i], "--emb=", 6))      emb = argv[i] + 6;
        else if (!strncmp(argv[i], "--bin=", 6))      bin = argv[i] + 6;
        else if (argv[i][0] >= '0' && argv[i][0] <= '9') epochs = atoi(argv[i]);
    }

    nn_seed(20260711);

    /* Full paper configuration, or a compact instance for a fast demo
     * (identical code path, smaller dims). */
    Config c = config_default();
    if (!full) { c.d_model = 64; c.d_ff = 128; c.n_layers = 2; }

    Dataset ds;
    if (bin) {
        printf("Loading pre-windowed data: %s\n", bin);
        ds = dataset_load_bin(bin);
        c.in_len = ds.in_len; c.pred_len = ds.pred_len; c.d_text = ds.d_text;
        printf("NOTE: .tfb files carry no coordinates, so the decoder is seeded with the\n"
               "      first target coordinate (reproducing the upstream setup). For a\n"
               "      sound last-observed-coordinate seed, use the CSV path instead.\n");
    } else {
        printf("Loading data: %s + %s\n", csv, emb);
        ds = dataset_load(csv, emb, c.in_len, c.pred_len);
    }
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
        /* shuffle training indices */
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
                for (int i = 0; i < c.pred_len * c.out_dim; ++i) dpred.data[i] /= bs;  /* batch mean */
                for (int i = 0; i < c.in_len * c.d_model; ++i)  dgate.data[i] /= bs;
                model_backward(&m, dpred, dgate);
            }
            adam_step(&opt, &pl);
            run_loss += bl / bs; ++nb;
        }
        Eval e = evaluate(&m, &ds, val, nva);
        printf("epoch %2d | train loss %.5f | val MAE %.4f | val dR %.2f km\n",
               ep, run_loss / nb, e.mae, e.km);
    }

    /* save learned parameters (flat float32 dump) */
    FILE *fp = fopen("typhoformer_weights.bin", "wb");
    if (fp) { for (int p = 0; p < pl.count; ++p) fwrite(pl.item[p].v, sizeof(float), pl.item[p].n, fp); fclose(fp);
              printf("\nsaved weights -> typhoformer_weights.bin (%ld params)\n", plist_num_params(&pl)); }

    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&dpred); mat_free(&dgate);
    free(train); free(val); adam_free(&opt); model_free(&m); plist_free(&pl); dataset_free(&ds);
    return 0;
}
