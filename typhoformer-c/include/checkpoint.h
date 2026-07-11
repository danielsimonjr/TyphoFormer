/*
 * checkpoint.h — save/load model weights with a self-describing header.
 *
 * Format (little-endian):
 *   magic  "TFW1"                                   (4 bytes)
 *   int32  d_num d_text d_model out_dim in_len pred_len d_ff n_heads n_layers
 *   float32 parameters, in ParamList registration order (== model_new order)
 *
 * Loading is two-phase: read the config, build an identically-configured model
 * (which re-registers parameters in the same order), then read the weights.
 */
#ifndef TYPHOFORMER_CHECKPOINT_H
#define TYPHOFORMER_CHECKPOINT_H

#include "model.h"   /* Config    */
#include "nn.h"      /* ParamList */

void   checkpoint_save(const char *path, Config c, const ParamList *pl);
Config checkpoint_load_config(const char *path);
void   checkpoint_load_params(const char *path, ParamList *pl);

#endif /* TYPHOFORMER_CHECKPOINT_H */
