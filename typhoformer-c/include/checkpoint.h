/*
 * checkpoint.h — save/load model weights with a self-describing header.
 *
 * Format v3 ("TFW3", little-endian):
 *   magic  "TFW3"                                   (4 bytes)
 *   int32  d_num d_text d_model out_dim in_len pred_len d_ff n_heads n_layers
 *   int32  n_stats                                  (0, or d_num)
 *   float32 mean[n_stats], std[n_stats]             (feature normalization)
 *   int32  has_coord                                (0 or 1)
 *   float32 cmean[2], cstd[2]                        (lat/lon normalization)
 *   float32 parameters, in ParamList registration order (== model_new order)
 *
 * "TFW2" (feature stats, no coord block) and "TFW1" (no stats) still readable.
 *
 * Optimizer state for resuming training is stored SEPARATELY (a ".opt" sidecar,
 * see checkpoint_save_optim) so inference checkpoints stay small.
 */
#ifndef TYPHOFORMER_CHECKPOINT_H
#define TYPHOFORMER_CHECKPOINT_H

#include "model.h"   /* Config    */
#include "nn.h"      /* ParamList */
#include "optim.h"   /* Adam      */

void   checkpoint_save (const char *path, Config c, const ParamList *pl);           /* no stats  */
void   checkpoint_save2(const char *path, Config c, const float *mean, const float *std,
                        int n_stats, const ParamList *pl);                          /* + feat stats */
void   checkpoint_save3(const char *path, Config c, const float *mean, const float *std,
                        int n_stats, const float *cmean, const float *cstd,
                        const ParamList *pl);                                       /* + coord stats */
Config checkpoint_load_config(const char *path);
void   checkpoint_load_params(const char *path, ParamList *pl);
/* Load feature normalization stats into mean/std (caller buffers). Returns the
 * number of stats (0 for a legacy checkpoint without them). */
int    checkpoint_load_stats(const char *path, float *mean, float *std);
/* Load coordinate (lat/lon) stats. Returns 1 if present (TFW3), else 0. */
int    checkpoint_load_coord_stats(const char *path, float *cmean, float *cstd);

/* Optimizer-state sidecar (Adam moments + step + lr) for resuming training.
 * checkpoint_load_optim returns 1 on success, 0 if the file is absent/mismatched. */
void   checkpoint_save_optim(const char *path, const Adam *a, int epoch);
int    checkpoint_load_optim(const char *path, Adam *a, int *epoch);

#endif /* TYPHOFORMER_CHECKPOINT_H */
