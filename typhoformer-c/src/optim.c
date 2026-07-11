/*
 * optim.c — Adam / AdamW implementation.
 */
#include "optim.h"

#include <math.h>
#include <stdlib.h>

/* Construct an optimizer for the given parameter list.
 * Hyperparameters use the canonical Adam defaults: b1 = 0.9, b2 = 0.999,
 * eps = 1e-8. lr and the weight-decay wd are supplied by the caller.
 * n = total number of scalar parameters across the whole ParamList, so the two
 * moment arrays fm (first moment m) and sm (second moment v) each get one slot
 * per trainable scalar. calloc zero-initialises them, which encodes the Adam
 * convention m_0 = 0, v_0 = 0. t starts at 0 and is pre-incremented on the
 * first step so bias correction sees t = 1. */
Adam adam_new(const ParamList *pl, float lr, float wd) {
    Adam a;
    a.lr = lr; a.b1 = 0.9f; a.b2 = 0.999f; a.eps = 1e-8f; a.wd = wd;
    a.n = plist_num_params(pl); a.t = 0;
    a.fm = (float *)calloc(a.n, sizeof(float));
    a.sm = (float *)calloc(a.n, sizeof(float));
    return a;
}

/* One optimizer step: apply the gradients currently sitting in each Param's g[]
 * buffer to its v[] values, updating the moment estimates along the way.
 *
 * Bias correction. Because m and v were initialised to 0, the EMAs are biased
 * toward 0 during early steps. Adam corrects this by dividing by (1 - beta^t).
 * bc1 = 1 - b1^t and bc2 = 1 - b2^t are computed once per step in double
 * precision (pow) for accuracy; as t grows they approach 1 and the correction
 * fades out.
 *
 * Iteration. `idx` is a flat cursor into the optimizer's fm/sm arrays. The
 * outer loop walks the ParamList's `count` tensors; the inner loop walks the n
 * scalars of each tensor, advancing idx in lockstep so every scalar parameter
 * maps to its own persistent (fm[idx], sm[idx]) moment pair. This flat indexing
 * is why adam_new sized fm/sm to plist_num_params — the total scalar count.
 *
 * Per-parameter math (for each scalar, with g = its gradient):
 *   m = b1*m + (1-b1)*g              first moment  (EMA of gradient)
 *   v = b2*v + (1-b2)*g*g            second moment (EMA of squared gradient)
 *   mhat = m / (1 - b1^t)            bias-corrected first moment
 *   vhat = v / (1 - b2^t)            bias-corrected second moment
 *   param -= lr * ( mhat/(sqrt(vhat)+eps) + wd*param )
 * The eps in the denominator prevents division by zero (and blow-up) when vhat
 * is tiny. sqrtf(vhat)+eps adaptively shrinks the effective step for parameters
 * with large/noisy gradients and grows it for consistently small ones. */
void adam_step(Adam *a, ParamList *pl) {
    ++a->t;
    double bc1 = 1.0 - pow((double)a->b1, (double)a->t);   /* 1 - b1^t */
    double bc2 = 1.0 - pow((double)a->b2, (double)a->t);   /* 1 - b2^t */
    long idx = 0;
    for (int p = 0; p < pl->count; ++p) {
        float *v = pl->item[p].v, *g = pl->item[p].g;      /* this tensor's values & grads */
        for (int e = 0; e < pl->item[p].n; ++e, ++idx) {
            float grad = g[e];                             /* raw gradient */
            a->fm[idx] = a->b1 * a->fm[idx] + (1.0f - a->b1) * grad;         /* update m */
            a->sm[idx] = a->b2 * a->sm[idx] + (1.0f - a->b2) * grad * grad;  /* update v */
            float mhat = (float)(a->fm[idx] / bc1);        /* bias-corrected m */
            float vhat = (float)(a->sm[idx] / bc2);        /* bias-corrected v */
            /* AdamW: decoupled weight decay (decay the parameter directly, not
             * the gradient — so it does not interact with the adaptive term).
             * Contrast with classic L2, which adds wd*param INTO grad above and
             * would therefore be rescaled by sqrt(vhat); here wd*v[e] is added
             * outside the adaptive ratio, giving the cleaner AdamW update. */
            v[e] -= a->lr * (mhat / (sqrtf(vhat) + a->eps) + a->wd * v[e]);
        }
    }
}

/* Release the moment buffers and null the pointers (idempotent, mirrors the
 * calloc in adam_new). The Adam struct itself is caller-owned (by value). */
void adam_free(Adam *a) { free(a->fm); free(a->sm); a->fm = a->sm = NULL; }
