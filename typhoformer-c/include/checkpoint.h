/*
 * checkpoint.h — save/load model weights with a self-describing header.
 *
 * Format v2 ("TFW2", little-endian):
 *   magic  "TFW2"                                   (4 bytes)
 *   int32  d_num d_text d_model out_dim in_len pred_len d_ff n_heads n_layers
 *   int32  n_stats                                  (0, or d_num)
 *   float32 mean[n_stats], std[n_stats]             (feature normalization)
 *   float32 parameters, in ParamList registration order (== model_new order)
 *
 * Legacy "TFW1" checkpoints (no stats block) are still readable.
 */
#ifndef TYPHOFORMER_CHECKPOINT_H
#define TYPHOFORMER_CHECKPOINT_H

#include "model.h"   /* Config    */
#include "nn.h"      /* ParamList */

void   checkpoint_save (const char *path, Config c, const ParamList *pl);           /* no stats  */
void   checkpoint_save2(const char *path, Config c, const float *mean, const float *std,
                        int n_stats, const ParamList *pl);                          /* with stats */
Config checkpoint_load_config(const char *path);
void   checkpoint_load_params(const char *path, ParamList *pl);
/* Load feature normalization stats into mean/std (caller buffers). Returns the
 * number of stats (0 for a legacy checkpoint without them). */
int    checkpoint_load_stats(const char *path, float *mean, float *std);

#endif /* TYPHOFORMER_CHECKPOINT_H */
