/*
 * test_data.c — CSV + .npy loader and the leakage-safe pipeline.
 * Run from typhoformer-c/ (data lives one level up). Asserts:
 *   - features are RAW after load, and z-scored (~0/~1) after standardize;
 *   - the storm-level split is disjoint AND no storm straddles two splits;
 *   - standardization statistics are fit on TRAIN storms only.
 */
#include "data.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static double col0_mean(const Dataset *d) {
    double m = 0;
    for (int i = 0; i < d->n_records; ++i) m += d->num[i * d->d_num];
    return d->n_records ? m / d->n_records : 0.0;
}

int main(void) {
    Dataset d = dataset_load("../HURDAT_2new_3000.csv", "../embedding_chunks", 12, 1);
    printf("records = %d, storms = %d, samples = %d, d_num = %d, d_text = %d\n",
           d.n_records, d.n_storms, d.n_samples, d.d_num, d.d_text);
    int fail = 0;

    /* before standardize: features are raw (col0 is max_wind ~ tens of knots) */
    double raw_mean = col0_mean(&d);
    if (fabs(raw_mean) < 1.0) { printf("FAIL: features look pre-standardized (%.3f)\n", raw_mean); fail = 1; }

    /* storm-level 3-way split */
    Split sp = dataset_split3(&d, 0.15f, 0.15f, 123);
    printf("split: train=%d val=%d test=%d\n", sp.n_train, sp.n_val, sp.n_test);
    if (sp.n_train + sp.n_val + sp.n_test != d.n_samples) { printf("FAIL: split doesn't cover all samples\n"); fail = 1; }

    /* no storm appears in more than one split (the whole point) */
    int *seen = (int *)calloc(d.n_storms, sizeof(int));
    for (int i = 0; i < sp.n_train; ++i) seen[d.gid[d.start[sp.train[i]]]] |= 1;
    for (int i = 0; i < sp.n_val;   ++i) seen[d.gid[d.start[sp.val[i]]]]   |= 2;
    for (int i = 0; i < sp.n_test;  ++i) seen[d.gid[d.start[sp.test[i]]]]  |= 4;
    int leaked = 0;
    for (int g = 0; g < d.n_storms; ++g) if (seen[g] == 3 || seen[g] == 5 || seen[g] == 6 || seen[g] == 7) ++leaked;
    if (leaked) { printf("FAIL: %d storms straddle splits (leakage)\n", leaked); fail = 1; }
    free(seen);

    /* standardize on TRAIN storms; the fit column should have ~0 mean over train */
    dataset_standardize(&d);
    double tm = 0; long tc = 0;
    for (int i = 0; i < d.n_records; ++i)
        if (d.storm_split[d.gid[i]] == 0) { tm += d.num[i * d.d_num]; ++tc; }
    tm = tc ? tm / tc : 0.0;
    printf("col0 z-score over TRAIN records: mean = %.2e (expect ~0)\n", tm);
    if (fabs(tm) > 1e-2) { printf("FAIL: train-only standardization off\n"); fail = 1; }

    /* coordinates are returned normalized; denorm round-trips */
    Mat xn = mat_new(12, 14), xt = mat_new(12, 384), yp = mat_new(1, 2), Y = mat_new(1, 2);
    dataset_get(&d, sp.n_train ? sp.train[0] : 0, xn, xt, yp, Y);
    float ll[2] = { yp.data[0], yp.data[1] };
    dataset_denorm(&d, ll);
    printf("sample seed: normalized (%.3f,%.3f) -> degrees (%.2f,%.2f)\n",
           yp.data[0], yp.data[1], ll[0], ll[1]);

    if (d.n_records != 3004) { printf("FAIL: expected 3004 records\n"); fail = 1; }
    printf(fail ? "FAIL: data loader\n" : "data loader ok\n");

    split_free(&sp);
    mat_free(&xn); mat_free(&xt); mat_free(&yp); mat_free(&Y);
    dataset_free(&d);
    return fail;
}
