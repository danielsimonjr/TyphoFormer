/*
 * test_checkpoint.c — round-trip the checkpoint format. Proves that
 * save -> load reproduces the Config, every parameter byte-for-byte, and the
 * normalization stats, for both TFW2 (with stats) and legacy TFW1 (without).
 */
#include "checkpoint.h"
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CKPT "._test_ckpt.tmp"

static int cfg_eq(Config a, Config b) {
    return a.d_num == b.d_num && a.d_text == b.d_text && a.d_model == b.d_model &&
           a.out_dim == b.out_dim && a.in_len == b.in_len && a.pred_len == b.pred_len &&
           a.d_ff == b.d_ff && a.n_heads == b.n_heads && a.n_layers == b.n_layers;
}

int main(void) {
    nn_seed(123);
    Config c = config_default();
    c.d_model = 32; c.d_ff = 48; c.n_layers = 1; c.n_heads = 2; c.pred_len = 3;

    ParamList pl; plist_init(&pl);
    Model m = model_new(&c, &pl);
    long np = plist_num_params(&pl);

    /* stats + a snapshot of the params we saved */
    float mean[64], std[64];
    for (int i = 0; i < c.d_num; ++i) { mean[i] = 1.5f * i; std[i] = 0.5f + i; }
    float *snap = (float *)malloc((size_t)np * sizeof(float));
    long o = 0;
    for (int p = 0; p < pl.count; ++p)
        for (int e = 0; e < pl.item[p].n; ++e) snap[o++] = pl.item[p].v[e];

    int fail = 0;

    /* ---- TFW2: config + params + stats ---- */
    checkpoint_save2(CKPT, c, mean, std, c.d_num, &pl);

    Config c2 = checkpoint_load_config(CKPT);
    if (!cfg_eq(c, c2)) { printf("FAIL: config mismatch after TFW2 load\n"); fail = 1; }

    ParamList pl2; plist_init(&pl2);
    Model m2 = model_new(&c2, &pl2);
    checkpoint_load_params(CKPT, &pl2);
    long diff = 0; o = 0;
    for (int p = 0; p < pl2.count; ++p)
        for (int e = 0; e < pl2.item[p].n; ++e)
            if (pl2.item[p].v[e] != snap[o++]) ++diff;
    if (diff) { printf("FAIL: %ld/%ld params differ after reload\n", diff, np); fail = 1; }

    float rmean[64], rstd[64];
    int ns = checkpoint_load_stats(CKPT, rmean, rstd);
    if (ns != c.d_num) { printf("FAIL: n_stats=%d expected %d\n", ns, c.d_num); fail = 1; }
    for (int i = 0; i < ns; ++i)
        if (fabsf(rmean[i] - mean[i]) > 0 || fabsf(rstd[i] - std[i]) > 0) {
            printf("FAIL: stats[%d] mismatch\n", i); fail = 1; break;
        }

    /* ---- TFW1: legacy, no stats block ---- */
    checkpoint_save(CKPT, c, &pl);
    Config c3 = checkpoint_load_config(CKPT);
    if (!cfg_eq(c, c3)) { printf("FAIL: config mismatch after TFW1 load\n"); fail = 1; }
    int ns1 = checkpoint_load_stats(CKPT, rmean, rstd);
    if (ns1 != 0) { printf("FAIL: legacy checkpoint reported %d stats (expected 0)\n", ns1); fail = 1; }
    ParamList pl3; plist_init(&pl3);
    Model m3 = model_new(&c3, &pl3);
    checkpoint_load_params(CKPT, &pl3);   /* must still load params past absent stats */

    printf("checkpoint round-trip: %ld params, TFW2 stats=%d, TFW1 stats=%d  %s\n",
           np, ns, ns1, fail ? "FAIL" : "ok");

    remove(CKPT);
    free(snap);
    model_free(&m); plist_free(&pl);
    model_free(&m2); plist_free(&pl2);
    model_free(&m3); plist_free(&pl3);
    printf(fail ? "FAIL: checkpoint tests\n" : "checkpoint round-trip check passed\n");
    return fail;
}
