/*
 * test_golden.c — determinism / behavior-regression guard.
 *
 * Trains the compact model for 2 epochs on the bundled data with the default
 * hyperparameters and asserts the training loss matches a pinned "golden"
 * value. Because the RNG is seeded and there is no wall-clock/rand() use, this
 * is exactly reproducible; any change that alters the numerics (init, order of
 * ops, optimizer, loss) will trip this test.
 *
 * If you intentionally change training behavior, update GOLDEN_LOSS below.
 * Run from the typhoformer-c/ directory (reads ../ data). Skips (passes) if the
 * data is unavailable, so it never blocks a build without the dataset.
 */
#include "data.h"
#include "model.h"
#include "optim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define GOLDEN_LOSS   0.02700   /* + 8-accumulator matmul_bt summation order (tensor.c) */
#define GOLDEN_TOL    1e-4

int main(void) {
    FILE *probe = fopen("../HURDAT_2new_3000.csv", "r");
    if (!probe) { printf("SKIP: dataset not found (../HURDAT_2new_3000.csv)\n"); return 0; }
    fclose(probe);

    nn_seed(20260711);
    Config c = config_default();
    c.d_model = 64; c.d_ff = 128; c.n_layers = 2;   /* the compact default */

    Dataset ds = dataset_load("../HURDAT_2new_3000.csv", "../embedding_chunks", c.in_len, c.pred_len);
    Split sp = dataset_split3(&ds, 0.15f, 0.15f, 42);   /* leakage-safe pipeline */
    dataset_standardize(&ds);
    int *train = sp.train, ntr = sp.n_train;

    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    Adam opt = adam_new(&pl, 1e-3f, 1e-5f);
    const int batch = 8; const float lambda = 0.1f;

    Mat xn = mat_new(c.in_len, c.d_num), xt = mat_new(c.in_len, c.d_text);
    Mat yp = mat_new(1, 2), Y = mat_new(c.pred_len, 2);
    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);

    double last = 0.0;
    for (int ep = 1; ep <= 2; ++ep) {
        for (int i = ntr - 1; i > 0; --i) { int j = (int)(nn_uniform(0, 1) * (i + 1)); if (j > i) j = i;
            int t = train[i]; train[i] = train[j]; train[j] = t; }
        double run = 0.0; int nb = 0;
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
            run += bl / bs; ++nb;
        }
        last = run / nb;
    }

    printf("golden: epoch-2 loss = %.5f (expected %.5f)\n", last, GOLDEN_LOSS);
    int fail = fabs(last - GOLDEN_LOSS) > GOLDEN_TOL;
    printf(fail ? "FAIL: training behavior changed (update GOLDEN_LOSS if intentional)\n"
                : "golden regression check passed\n");

    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y); mat_free(&dpred); mat_free(&dgate);
    split_free(&sp); adam_free(&opt); model_free(&m); plist_free(&pl); dataset_free(&ds);
    return fail;
}
