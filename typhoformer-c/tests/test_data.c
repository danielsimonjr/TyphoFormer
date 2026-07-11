/*
 * test_data.c — smoke test for the CSV + .npy dataset loader.
 * Run from the typhoformer-c/ directory (data lives one level up).
 */
#include "data.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int main(void) {
    Dataset d = dataset_load("../HURDAT_2new_3000.csv", "../embedding_chunks", 12, 1);
    printf("records = %d, samples = %d, d_num = %d, d_text = %d\n",
           d.n_records, d.n_samples, d.d_num, d.d_text);

    double m = 0.0, s2 = 0.0;
    for (int i = 0; i < d.n_records; ++i) { float x = d.num[i * d.d_num]; m += x; s2 += x * x; }
    m /= d.n_records; s2 = sqrt(s2 / d.n_records - m * m);
    printf("num col0 after z-score: mean = %.2e, std = %.3f (expect ~0, ~1)\n", m, s2);

    Mat xn = mat_new(12, 14), xt = mat_new(12, 384), yp = mat_new(1, 2), Y = mat_new(1, 2);
    dataset_get(&d, 0, xn, xt, yp, Y);
    printf("sample 0: yprev = (%.2f, %.2f), Y = (%.2f, %.2f)\n",
           yp.data[0], yp.data[1], Y.data[0], Y.data[1]);

    int *tr, *va, ntr, nva;
    dataset_split(&d, 0.1f, 42, &tr, &ntr, &va, &nva);
    printf("split: train = %d, val = %d\n", ntr, nva);

    int fail = !(d.n_records == 3004 && d.n_samples > 0 && fabs(m) < 1e-3 && fabs(s2 - 1.0) < 1e-2);
    printf(fail ? "FAIL: data loader\n" : "data loader ok\n");

    free(tr); free(va);
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    dataset_free(&d);
    return fail;
}
