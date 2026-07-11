/*
 * optim.h — AdamW optimizer over a ParamList (decoupled weight decay).
 *
 * Adam (Adaptive Moment Estimation) keeps two running averages per scalar
 * parameter: the first moment (an EMA of the gradient, ~ its mean/momentum) and
 * the second moment (an EMA of the squared gradient, ~ its uncentered variance).
 * Each parameter is then stepped by mean/sqrt(variance), which adaptively
 * rescales the learning rate per coordinate. This is the "W" (AdamW) variant:
 * weight decay is applied DIRECTLY to the parameter (decoupled) rather than
 * folded into the gradient as an L2 penalty.
 *
 * The optimizer treats the model as one flat list of scalars via ParamList
 * (nn.h): a ParamList holds `count` Param entries, and each Param{v, g, n, name}
 * exposes a value buffer v[n] and its gradient buffer g[n]. The optimizer's own
 * per-scalar state (fm/sm) is a pair of flat arrays indexed in lockstep as it
 * walks every Param's n elements in order.
 */
#ifndef TYPHOFORMER_OPTIM_H
#define TYPHOFORMER_OPTIM_H

#include "nn.h"

typedef struct {
    float  lr;        /* learning rate (base step size)                       */
    float  b1;        /* beta1: decay rate for the first-moment EMA (~0.9)    */
    float  b2;        /* beta2: decay rate for the second-moment EMA (~0.999) */
    float  eps;       /* epsilon: denominator floor for numerical stability   */
    float  wd;        /* decoupled weight-decay coefficient (AdamW)           */
    float *fm, *sm;   /* first/second moment, one slot per scalar parameter   */
    long   n;         /* total scalar-parameter count (length of fm and sm)   */
    long   t;         /* timestep, incremented each step (drives bias correction) */
} Adam;

/* adam_new  : allocate zeroed moment buffers sized to the ParamList, set hypers.
 * adam_step : consume the current gradients (g) in pl and update the values (v).
 * adam_free : release the moment buffers. */
Adam adam_new(const ParamList *pl, float lr, float wd);
void adam_step(Adam *a, ParamList *pl);
void adam_free(Adam *a);

#endif /* TYPHOFORMER_OPTIM_H */
