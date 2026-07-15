/*
 * checkpoint.h — save/load model weights with a self-describing header.
 *
 * Format v4 ("TFW4", little-endian):
 *   magic  "TFW4"                                   (4 bytes)
 *   int32  d_num d_text d_model out_dim in_len pred_len d_ff n_heads n_layers
 *   int32  n_stats                                  (0, or d_num)
 *   float32 mean[n_stats], std[n_stats]             (feature normalization)
 *   int32  has_coord                                (0 or 1)
 *   float32 cmean[2], cstd[2]                        (lat/lon normalization)
 *   uint32 modes                                    (TF_MODE_* bitfield)   <-- v4
 *   float32 parameters, in ParamList registration order (== model_new order)
 *
 * "TFW3" (no modes word), "TFW2" (no coord block) and "TFW1" (no stats) still
 * readable. The four versions are nested supersets, so a v4 reader loads all of
 * them by branching on the magic tag to decide which optional blocks are present.
 *
 * WHY the modes word exists (v4). The header carries the model GEOMETRY (the 9
 * ints), and the untagged parameter blob makes a geometry mismatch fatal: the
 * reader recomputes each tensor's element count from the Config and demands
 * exactly that many floats, so a wrong d_model or a toggled --posenc/--gru/
 * --xattn trips a "parameter size mismatch".
 *
 * That guard does NOT cover the decoder SEMANTICS. `--cv`, `--delta` and
 * `--rotframe` are parameter-NEUTRAL: they reuse the same fc1/fc2 tensors and
 * only change what the decoder's output MEANS (a correction to a constant-
 * velocity anchor, versus an absolute coordinate). The byte accounting sails
 * through, the model loads cleanly, and it then interprets its own output
 * wrongly. Measured 2026-07-14 on a real run: a cv-trained checkpoint evaluated
 * without `--cv` reported **5071 km** where the truth was **29 km** — a silent
 * 170x error with no warning.
 *
 * So v4 records the mode flags and eval/predict self-configure from them. Older
 * checkpoints have no modes word; loading one leaves the flags to the caller,
 * exactly as before.
 *
 * Optimizer state for resuming training is stored SEPARATELY (a ".opt" sidecar,
 * magic "TFO1", see checkpoint_save_optim) so inference checkpoints stay small —
 * the Adam moments double the on-disk size and are useless for inference.
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

/* ---- Model mode flags (TFW4) ----------------------------------------
 * The architecture/decoder switches that model_new() reads from its module-level
 * globals. They are NOT in Config, so before v4 nothing recorded them and
 * eval/predict had to be told again by hand — see the header comment above for
 * why that failed silently. Keep these bit values STABLE: they are on disk. */
#define TF_MODE_DELTA      (1u << 0)   /* model_set_delta      */
#define TF_MODE_CV         (1u << 1)   /* model_set_cv         */
#define TF_MODE_GRU        (1u << 2)   /* model_set_gru        */
#define TF_MODE_XATTN      (1u << 3)   /* model_set_xattn      */
#define TF_MODE_ROTFRAME   (1u << 4)   /* model_set_rotframe   */
#define TF_MODE_NO_SPATIAL (1u << 5)   /* model_set_no_spatial */
#define TF_MODE_POSENC     (1u << 6)   /* model_set_posenc     */
#define TF_MODE_POOL_LAST  (1u << 7)   /* model_set_pool_last  */
#define TF_MODE_CO_SPATIAL (1u << 8)   /* model_set_co_spatial */
#define TF_MODE_DIRECT     (1u << 9)   /* model_set_direct     */

/* Write a TFW4 checkpoint: TFW3 plus the mode bitfield. This is what training
 * writes; the older savers remain for the tests and for back-compat. */
void   checkpoint_save4(const char *path, Config c, const float *mean, const float *std,
                        int n_stats, const float *cmean, const float *cstd,
                        unsigned modes, const ParamList *pl);
/* Read the mode bitfield. Returns 1 and sets *modes for a TFW4 checkpoint; returns
 * 0 (and leaves *modes untouched) for TFW1/2/3, which do not record them. */
int    checkpoint_load_modes(const char *path, unsigned *modes);
/* Apply a mode bitfield to the module-level model_set_* globals. Call BEFORE
 * model_new(). Sets every flag explicitly (including to 0), so it fully overrides
 * whatever the CLI parse left behind. */
void   checkpoint_apply_modes(unsigned modes);
/* Human-readable summary, e.g. "cv+rotframe" or "plain". Returns a static buffer. */
const char *checkpoint_modes_str(unsigned modes);

Config checkpoint_load_config(const char *path);
void   checkpoint_load_params(const char *path, ParamList *pl);
/* Load feature normalization stats into mean/std (caller buffers). Returns the
 * number of stats (0 for a legacy checkpoint without them). */
int    checkpoint_load_stats(const char *path, float *mean, float *std);
/* Load coordinate (lat/lon) stats. Returns 1 if present (TFW3+), else 0. */
int    checkpoint_load_coord_stats(const char *path, float *cmean, float *cstd);

/* Optimizer-state sidecar (Adam moments + step + lr) for resuming training.
 * checkpoint_load_optim returns 1 on success, 0 if the file is absent/mismatched. */
void   checkpoint_save_optim(const char *path, const Adam *a, int epoch);
int    checkpoint_load_optim(const char *path, Adam *a, int *epoch);

#endif /* TYPHOFORMER_CHECKPOINT_H */
