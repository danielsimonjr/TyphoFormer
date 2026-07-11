/*
 * optim.c — Adam.
 */
#include "optim.h"

#include <math.h>
#include <stdlib.h>

Adam adam_new(const ParamList *pl, float lr, float wd) {
    Adam a;
    a.lr = lr; a.b1 = 0.9f; a.b2 = 0.999f; a.eps = 1e-8f; a.wd = wd;
    a.n = plist_num_params(pl); a.t = 0;
    a.fm = (float *)calloc(a.n, sizeof(float));
    a.sm = (float *)calloc(a.n, sizeof(float));
    return a;
}

void adam_step(Adam *a, ParamList *pl) {
    ++a->t;
    double bc1 = 1.0 - pow((double)a->b1, (double)a->t);
    double bc2 = 1.0 - pow((double)a->b2, (double)a->t);
    long idx = 0;
    for (int p = 0; p < pl->count; ++p) {
        float *v = pl->item[p].v, *g = pl->item[p].g;
        for (int e = 0; e < pl->item[p].n; ++e, ++idx) {
            float grad = g[e] + a->wd * v[e];              /* L2 weight decay */
            a->fm[idx] = a->b1 * a->fm[idx] + (1.0f - a->b1) * grad;
            a->sm[idx] = a->b2 * a->sm[idx] + (1.0f - a->b2) * grad * grad;
            float mhat = (float)(a->fm[idx] / bc1);
            float vhat = (float)(a->sm[idx] / bc2);
            v[e] -= a->lr * mhat / (sqrtf(vhat) + a->eps);
        }
    }
}

void adam_free(Adam *a) { free(a->fm); free(a->sm); a->fm = a->sm = NULL; }
