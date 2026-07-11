/*
 * optim.h — AdamW optimizer over a ParamList (decoupled weight decay).
 */
#ifndef TYPHOFORMER_OPTIM_H
#define TYPHOFORMER_OPTIM_H

#include "nn.h"

typedef struct {
    float  lr, b1, b2, eps, wd;
    float *fm, *sm;   /* first/second moment, one slot per scalar parameter */
    long   n;
    long   t;
} Adam;

Adam adam_new(const ParamList *pl, float lr, float wd);
void adam_step(Adam *a, ParamList *pl);
void adam_free(Adam *a);

#endif /* TYPHOFORMER_OPTIM_H */
