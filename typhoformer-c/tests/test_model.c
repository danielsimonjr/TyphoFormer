/*
 * test_model.c — finite-difference gradient check of the whole model
 * (PGF + spatio-temporal encoder + autoregressive decoder + loss) on a small
 * configuration, for both single-step and multi-step (autoregressive) decoding.
 */
#include "model.h"
#include "data.h"   /* TF_NBR_K / TF_NBR_NF for the co_spatial variant */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static Model     M;
static ParamList pl;
static Mat       xnum, xtext, yprev, Y;
static float     LAMBDA = 0.1f;

static double loss_only(void) {
    model_forward(&M, xnum, xtext, yprev);
    Mat none = {0, 0, NULL};
    return model_loss(M.pred, Y, M.pgf.gate, LAMBDA, none, none);
}

/* check_model reads whatever architecture flags are currently set globally
 * (delta / no_spatial / posenc / pool_last / prenorm); main sets and resets
 * them around each call. */
static int check_model(const char *label, int pred_len) {
    printf("[model gradient check, %s, pred_len=%d]\n", label, pred_len);
    Config c; c.d_num = 3; c.d_text = 5; c.d_model = 8; c.out_dim = 2;
    c.in_len = 4; c.pred_len = pred_len; c.d_ff = 16; c.n_heads = 2; c.n_layers = 1;

    plist_init(&pl);
    M = model_new(&c, &pl);

    /* co-active spatial attention needs a (fixed) neighbour block fed in before
     * every forward — synthesise a couple of neighbours so the module is on the
     * active path during the gradient check. */
    Mat nbr = mat_new(TF_NBR_K, TF_NBR_NF);
    if (M.use_co) {
        for (int i = 0; i < TF_NBR_K * TF_NBR_NF; ++i) nbr.data[i] = nn_uniform(-1, 1);
        model_set_neighbors(&M, nbr, 2);
    }
    /* cv decoder needs a (fixed) seed velocity fed before every forward. */
    Mat vel = mat_new(1, c.out_dim);
    if (M.dec.use_cv) {
        for (int i = 0; i < c.out_dim; ++i) vel.data[i] = nn_uniform(-1, 1);
        model_set_seed_velocity(&M, vel);
    }

    xnum  = mat_new(c.in_len, c.d_num);
    xtext = mat_new(c.in_len, c.d_text);
    yprev = mat_new(1, c.out_dim);
    Y     = mat_new(c.pred_len, c.out_dim);
    for (int i = 0; i < c.in_len * c.d_num;  ++i) xnum.data[i]  = nn_uniform(-1, 1);
    for (int i = 0; i < c.in_len * c.d_text; ++i) xtext.data[i] = nn_uniform(-1, 1);
    for (int i = 0; i < c.out_dim; ++i)           yprev.data[i] = nn_uniform(-1, 1);
    for (int i = 0; i < c.pred_len * c.out_dim; ++i) Y.data[i]  = nn_uniform(-1, 1);
    model_set_teacher(&M, Y);   /* teacher targets (consumed only by the tf-forced variant) */

    Mat dpred = mat_new(c.pred_len, c.out_dim), dgate = mat_new(c.in_len, c.d_model);
    model_forward(&M, xnum, xtext, yprev);
    model_loss(M.pred, Y, M.pgf.gate, LAMBDA, dpred, dgate);
    plist_zero_grad(&pl);
    model_backward(&M, dpred, dgate);

    /* eps=1e-4 keeps the ± perturbation off the ReLU kink; tolerate a few
     * kink-affected outliers (see tests/test_nn.c for the rationale). */
    const float eps = 1e-4f, floor = 1e-2f, REL = 2e-2f, ABS = 3e-3f;
    float max_err = 0.0f, max_abs = 0.0f; long checked = 0; int bad = 0;
    for (int p = 0; p < pl.count; ++p)
        for (int e = 0; e < pl.item[p].n; ++e) {
            float *w = &pl.item[p].v[e], save = *w;
            *w = save + eps; double lp = loss_only();
            *w = save - eps; double lm = loss_only();
            *w = save;
            float g = (float)((lp - lm) / (2.0 * eps)), a = pl.item[p].g[e];
            float abserr = fabsf(g - a);
            float err = abserr / (fabsf(g) + fabsf(a) + floor);
            if (err > max_err) max_err = err;
            if (abserr > max_abs) max_abs = abserr;
            if (err >= REL && abserr >= ABS) ++bad;
            ++checked;
        }

    printf("  params: %ld (checked %ld) | max rel err = %.2e, max abs err = %.2e | outliers = %d\n",
           plist_num_params(&pl), checked, max_err, max_abs, bad);
    int fail = (bad > 3);
    if (fail) printf("  FAIL: model gradients (%d disagree)\n", bad); else printf("  ok\n");

    model_free(&M); plist_free(&pl);
    mat_free(&xnum); mat_free(&xtext); mat_free(&yprev); mat_free(&Y);
    mat_free(&dpred); mat_free(&dgate); mat_free(&nbr); mat_free(&vel);
    return fail;
}

int main(void) {
    nn_seed(7);
    int fail = 0;
    fail |= check_model("baseline", 1);
    fail |= check_model("multistep", 3);
    model_set_delta(1);
    fail |= check_model("delta", 1);
    fail |= check_model("delta+multistep", 3);
    model_set_delta(0);
    model_set_cv(1);
    fail |= check_model("cv", 1);
    fail |= check_model("cv+multistep", 3);
    /* the motion-aligned frame's rotation backward (Rᵀ path + the ∂u/∂v
     * Jacobian into the velocity recurrence); vel is random nonzero so the
     * frame is active, and multistep exercises the threaded dv path. */
    model_set_rotframe(1);
    fail |= check_model("cv+rotframe", 1);
    fail |= check_model("cv+rotframe+multistep", 3);
    model_set_rotframe(0);
    /* teacher forcing with prob 1: every rollout step's state is replaced by
     * the (constant) teacher target, so the stochastic path is deterministic
     * and the recurrence-cutting backward FD-checks exactly. training must be
     * on for the path to arm; dropout off keeps the check deterministic. */
    nn_set_dropout(0.0f); nn_set_training(1); model_set_tf_prob(1.0f);
    fail |= check_model("cv+teacher-forced", 3);
    model_set_tf_prob(0.0f); nn_set_training(0); nn_set_dropout(0.1f);
    model_set_cv(0);
    model_set_gru(1);
    fail |= check_model("gru", 1);
    fail |= check_model("gru+multistep", 3);
    model_set_gru(0);
    model_set_xattn(1);
    fail |= check_model("xattn", 1);
    fail |= check_model("xattn+multistep", 3);
    model_set_xattn(0);
    /* loss-shaping options: the FD check differentiates THROUGH model_loss, so
     * this validates the Huber tails (small delta pushes most residuals into
     * the linear branch) and the horizon weights on a multi-step rollout. */
    model_set_huber(0.05f); model_set_hweight(1.0f);
    fail |= check_model("huber+hweight", 3);
    model_set_huber(0.0f); model_set_hweight(0.0f);
    /* --km_loss: the equirectangular km objective scales the longitude residual by
     * (clon_std/clat_std)·cos(lat); the FD check validates that reweight AND the
     * extra chain-rule factor kw in its gradient (representative HURDAT-ish stats).
     * Multistep so per-horizon cos(lat) varies; also composed with Huber. */
    model_set_km_loss(1, 25.0f, 10.0f, 15.0f);
    fail |= check_model("km_loss", 3);
    model_set_km_loss(1, 25.0f, 10.0f, 15.0f); model_set_huber(0.05f);
    fail |= check_model("km_loss+huber", 3);
    model_set_km_loss(0, 0.0f, 1.0f, 1.0f); model_set_huber(0.0f);
    /* encoder architecture variants (each set, checked, reset).
     * no-spatial is now the DEFAULT (covered by every check above); this
     * exercises the paper-faithful spatial (self_only) path explicitly. */
    model_set_no_spatial(0); fail |= check_model("spatial (paper)", 1); model_set_no_spatial(1);
    model_set_posenc(1);     fail |= check_model("posenc", 1);     model_set_posenc(0);
    model_set_pool_last(1);  fail |= check_model("pool_last", 1);  model_set_pool_last(0);
    nn_set_prenorm(1);       fail |= check_model("prenorm", 1);    nn_set_prenorm(0);
    nn_set_timebias(1);      fail |= check_model("timebias", 1);   nn_set_timebias(0);
    model_set_co_spatial(1); fail |= check_model("co_spatial", 1); model_set_co_spatial(0);
    model_set_co_spatial(1); fail |= check_model("co_spatial+multistep", 3); model_set_co_spatial(0);
    /* everything on at once (multi-step; spatial ON to combine both paths) */
    model_set_delta(1); model_set_no_spatial(0); model_set_posenc(1);
    model_set_pool_last(1); nn_set_prenorm(1);
    fail |= check_model("all-options", 3);
    model_set_delta(0); model_set_no_spatial(1); model_set_posenc(0);
    model_set_pool_last(0); nn_set_prenorm(0);
    if (!fail) printf("\nmodel gradient check passed\n");
    return fail;
}
