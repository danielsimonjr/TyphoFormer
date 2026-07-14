/*
 * test_checkpoint.c — round-trip the checkpoint format. Proves that
 * save -> load reproduces the Config, every parameter byte-for-byte, and the
 * normalization stats, for both TFW2 (with stats) and legacy TFW1 (without).
 */
#include "checkpoint.h"
#include "model.h"
#include "optim.h"

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

    /* ---- TFW3: coordinate stats round-trip ---- */
    float cmean[2] = { 25.5f, -70.25f }, cstd[2] = { 8.0f, 12.5f };
    checkpoint_save3(CKPT, c, mean, std, c.d_num, cmean, cstd, &pl);
    float rcm[2], rcs[2];
    int hc = checkpoint_load_coord_stats(CKPT, rcm, rcs);
    if (!hc || rcm[0] != cmean[0] || rcm[1] != cmean[1] || rcs[0] != cstd[0] || rcs[1] != cstd[1]) {
        printf("FAIL: coord stats round-trip\n"); fail = 1;
    }
    if (checkpoint_load_stats(CKPT, rmean, rstd) != c.d_num) { printf("FAIL: TFW3 feat stats\n"); fail = 1; }
    { ParamList q; plist_init(&q); Model mq = model_new(&c, &q);
      checkpoint_load_params(CKPT, &q);         /* params load past both stat blocks */
      model_free(&mq); plist_free(&q); }

    /* ---- optimizer-state sidecar round-trip ---- */
    Adam opt = adam_new(&pl, 1e-3f, 1e-5f);
    for (long i = 0; i < opt.n; ++i) { opt.fm[i] = 0.5f * i; opt.sm[i] = 0.25f * i; }
    opt.t = 42; opt.lr = 7e-4f;
    checkpoint_save_optim("._test_ckpt.opt", &opt, 9);
    Adam opt2 = adam_new(&pl, 1e-3f, 1e-5f); int rep = 0;
    if (!checkpoint_load_optim("._test_ckpt.opt", &opt2, &rep)) { printf("FAIL: optim load\n"); fail = 1; }
    if (opt2.t != 42 || rep != 9 || fabsf(opt2.lr - 7e-4f) > 0) { printf("FAIL: optim scalars\n"); fail = 1; }
    for (long i = 0; i < opt.n; ++i)
        if (opt2.fm[i] != opt.fm[i] || opt2.sm[i] != opt.sm[i]) { printf("FAIL: optim moments\n"); fail = 1; break; }
    adam_free(&opt); adam_free(&opt2); remove("._test_ckpt.opt");

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

    /* ---- TFW4: the mode bitfield -------------------------------------
     * Before v4 the decoder mode was NOT recorded. The size-mismatch guard cannot
     * catch --cv/--delta/--rotframe because they are parameter-NEUTRAL, so a
     * cv-trained checkpoint evaluated without --cv loaded cleanly and reported
     * 5071 km where the truth was 29 km. Pin the round-trip so that cannot regress. */
    unsigned modes_w = TF_MODE_CV | TF_MODE_ROTFRAME | TF_MODE_NO_SPATIAL;
    checkpoint_save4(CKPT, c, mean, std, c.d_num, cmean, cstd, modes_w, &pl);

    unsigned modes_r = 0;
    if (!checkpoint_load_modes(CKPT, &modes_r)) {
        printf("FAIL: TFW4 checkpoint did not report modes\n"); fail = 1;
    } else if (modes_r != modes_w) {
        printf("FAIL: modes round-trip %u != %u\n", modes_r, modes_w); fail = 1;
    }

    /* A TFW4 file must still load config, stats and params like a TFW3 one — the
     * readers have to SKIP the new word, not trip over it. */
    Config c4 = checkpoint_load_config(CKPT);
    if (!cfg_eq(c, c4)) { printf("FAIL: config mismatch after TFW4 load\n"); fail = 1; }
    if (checkpoint_load_stats(CKPT, rmean, rstd) != c.d_num) {
        printf("FAIL: TFW4 feature stats lost\n"); fail = 1;
    }
    if (!checkpoint_load_coord_stats(CKPT, rcm, rcs)) {
        printf("FAIL: TFW4 coord stats lost\n"); fail = 1;
    }
    ParamList pl4; plist_init(&pl4);
    Model m4 = model_new(&c4, &pl4);
    checkpoint_load_params(CKPT, &pl4);   /* must seek PAST the modes word */

    /* An older checkpoint must report "no modes recorded" rather than garbage. */
    checkpoint_save3(CKPT, c, mean, std, c.d_num, cmean, cstd, &pl);
    unsigned modes_old = 0xDEADBEEF;
    if (checkpoint_load_modes(CKPT, &modes_old)) {
        printf("FAIL: TFW3 checkpoint claimed to carry modes\n"); fail = 1;
    }
    if (modes_old != 0xDEADBEEF) {
        printf("FAIL: load_modes clobbered the caller's value on a TFW3 file\n"); fail = 1;
    }

    printf("TFW4 modes round-trip: %s (%s)  %s\n",
           checkpoint_modes_str(modes_r), "cv+rotframe+no_spatial expected",
           fail ? "FAIL" : "ok");
    model_free(&m4); plist_free(&pl4);

    remove(CKPT);
    free(snap);
    model_free(&m); plist_free(&pl);
    model_free(&m2); plist_free(&pl2);
    model_free(&m3); plist_free(&pl3);
    printf(fail ? "FAIL: checkpoint tests\n" : "checkpoint round-trip check passed\n");
    return fail;
}
