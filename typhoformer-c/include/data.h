/*
 * data.h — dataset loading: parse the HURDAT2 CSV numerical features and the
 * precomputed MiniLM embedding chunks (.npy), group by storm, and build
 * sliding-window samples.
 *
 * Leakage-safe methodology (CSV path):
 *   1. dataset_load()      — parse RAW features/coords, group by storm, build
 *                            windows. No standardization yet.
 *   2. dataset_split3()    — assign whole STORMS to train/val/test (so no storm,
 *                            and no overlapping window, straddles two splits).
 *   3. dataset_standardize() — fit feature + coordinate stats on the TRAIN
 *                            storms only, then apply. Nothing from val/test
 *                            leaks into the normalization statistics.
 */
#ifndef TYPHOFORMER_DATA_H
#define TYPHOFORMER_DATA_H

#include <stdint.h>

#include "tensor.h"

/*
 * Dataset — the single in-memory container for every representation of the
 * data used during training/eval. There are two mutually-exclusive population
 * paths, selected by the `prewindowed` flag:
 *
 *   CSV path (prewindowed==0):  per-record arrays (num/emb/lat/lon/gid/tkey)
 *       are filled by dataset_load, then sliding windows are described lazily
 *       by `start` (each sample is a length-in_len+pred_len slice of records).
 *       dataset_get() reconstructs a sample on demand from these arrays.
 *   TFB path (prewindowed==1):  dataset_load_bin fills the already-windowed
 *       tensors win_in/win_tg/win_seed directly; the per-record arrays and
 *       `start` stay NULL. dataset_get() just copies out of win_*.
 *
 * Units convention, important for reasoning about leakage and denorm:
 *   - `num`  starts RAW (physical units from the CSV) and becomes z-scored
 *            (standardized, mean 0 / std 1 per column) IN PLACE after
 *            dataset_standardize / dataset_apply_stats / the .tfb loader.
 *   - `lat`/`lon` stay RAW degrees for the whole lifetime; coordinates are
 *            normalized on the fly (see cnorm) only when handed out by
 *            dataset_get, and never mutated in place.
 */
typedef struct {
    int    n_records;    /* total CSV records parsed (rows across all storms) */
    int    d_num, d_text, in_len, pred_len;
                         /* d_num: numeric feature width (14 raw; +4 after
                          *   dataset_add_motion). d_text: MiniLM embedding
                          *   width (384). in_len: encoder/history length.
                          *   pred_len: decoder/forecast horizon.            */
    float *num;          /* [n_records * d_num] numeric features, row-major.
                          *   RAW at load; z-scored IN PLACE after standardize */
    float *emb;          /* [n_records * d_text] MiniLM text embeddings, raw  */
    float *lat, *lon;    /* [n_records] each; RAW degrees, never mutated      */
    int   *gid;          /* [n_records] storm group id per record (0-based)   */
    int    n_storms;     /* number of distinct storms (max gid + 1)           */
    int   *storm_split;  /* [n_storms] 0=train 1=val 2=test; NULL until
                          *   dataset_split3 runs. The per-STORM label is what
                          *   makes standardization leakage-safe.             */
    float  mean[64], std[64];   /* per-column feature z-score stats, fit on
                          *   TRAIN records only. Indexed [0,d_num). 64 is a
                          *   fixed upper bound (>= max d_num of 18).          */
    float  cmean[2], cstd[2];   /* coordinate (lat,lon) normalization stats.
                          *   Identity {mean 0, std 1} means "raw / unset"
                          *   (the .tfb path and any pre-standardize state);
                          *   real values are fit on TRAIN records only.      */
    int   *start;        /* [n_samples] first record index of each window
                          *   (CSV path only; NULL on the .tfb path).         */
    int    n_samples;    /* number of sliding-window samples                  */
    int    prewindowed;  /* 1 => samples come from a .tfb file (win_* below,
                          *   per-record arrays unused).                      */
    int    no_text;      /* 1 => dataset_get zeros the text branch (ablation) */
    float *win_in;       /* [n_samples * in_len * (d_num+d_text)] .tfb inputs;
                          *   numeric cols standardized in place by the loader */
    float *win_tg;       /* [n_samples * pred_len * 2] .tfb target coords
                          *   (RAW degrees; cnorm applied when handed out)     */
    float *win_seed;     /* [n_samples * 2] true decoder seed coord (TFB2);
                          *   NULL for legacy TFB1 (which leaks via tg[0])     */
    long  *tkey;         /* [n_records] timestamp key = date*10000+time; used
                          *   to find co-active storms sharing a moment.       */
    float *nbr;          /* [n_records * TF_NBR_K * TF_NBR_NF] relative
                          *   neighbour vectors (Δlat,Δlon,Δmax_wind) per rec  */
    int   *nbr_cnt;      /* [n_records] how many neighbour slots are filled    */
} Dataset;

/* Co-active spatial neighbours: up to K storms sharing a timestamp, each a
 * relative (Δlat, Δlon, Δmax_wind) vector. Real multi-node spatial signal. */
#define TF_NBR_K  3
#define TF_NBR_NF 3

/* Load records from `csv` and embeddings from `embdir` (emb_chunk_*.npy) and
 * build sliding windows. Features are left RAW; call dataset_split3 then
 * dataset_standardize to normalize without leakage. */
Dataset dataset_load(const char *csv, const char *embdir, int in_len, int pred_len);

/* Load pre-windowed samples from a .tfb file (tools/npy_dict_to_bin.py or the
 * `prepare` subcommand). TFB2 files carry the true decoder seed coordinate;
 * legacy TFB1 files do not, so those fall back to seeding with the first target
 * (a mild label leak — prefer the CSV path or regenerate as TFB2). */
Dataset dataset_load_bin(const char *path);

/* A train/val/test partition of sample indices (into d->start). */
typedef struct {
    int *train, *val, *test;
    int  n_train, n_val, n_test;
} Split;

/* Deterministic STORM-level split: whole storms go to train/val/test, so
 * overlapping windows never straddle splits. Records d->storm_split. */
Split dataset_split3(Dataset *d, float val_frac, float test_frac, uint64_t seed);
void  split_free(Split *s);

/* Append motion features to the input: per record, the (raw) position lat, lon
 * and the step-to-step velocity dlat, dlon (0 at a storm's first record). This
 * gives the model the trajectory signal it otherwise never sees. Increases
 * d->d_num by 4. Call BEFORE dataset_standardize. No-op for the .tfb path (it
 * carries no coordinate history). */
void dataset_add_motion(Dataset *d);

/* Append 7 second-order physics features: acceleration (Δ²lat, Δ²lon),
 * translation speed |v|, the heading unit vector (v/|v|), and the seasonal
 * day-of-year phase (sin, cos). Increases d->d_num by 7; composes with
 * dataset_add_motion (apply motion first, then physics: +11 total). Call
 * BEFORE dataset_standardize. No-op for the .tfb path. */
void dataset_add_physics(Dataset *d);

/* Precompute co-active neighbours (call after load, before use). Accessor fills
 * nbrmat[TF_NBR_K, TF_NBR_NF] and *cnt for sample s's seed timestep. */
void dataset_build_neighbors(Dataset *d);
void dataset_neighbors(const Dataset *d, int s, Mat nbrmat, int *cnt);

/* Seed velocity (last observed displacement) for sample s, in normalized coord
 * units. Fills vout[1,2]; zero for the pre-windowed path (no coord history). */
void dataset_seed_velocity(const Dataset *d, int s, Mat vout);

/* Fit feature + coordinate normalization on the TRAIN storms only (per
 * d->storm_split) and apply the feature z-score to d->num in place. Coordinate
 * stats are stored and applied on the fly in dataset_get. */
void dataset_standardize(Dataset *d);

/* Apply externally-provided stats (e.g. loaded from a checkpoint) to a freshly
 * loaded raw dataset: sets mean/std + cmean/cstd and z-scores d->num in place.
 * Pass cmean=cstd=NULL to leave coordinates raw ({0,1}). */
void dataset_apply_stats(Dataset *d, const float *mean, const float *std, int n,
                         const float *cmean, const float *cstd);

/* Materialise sample `s` into caller-provided matrices:
 *   xnum [in_len,d_num], xtext [in_len,d_text], yprev [1,2], Y [pred_len,2].
 * Coordinates (yprev, Y) are returned NORMALIZED by cmean/cstd. Use
 * dataset_denorm() to convert model outputs / targets back to degrees. */
void dataset_get(const Dataset *d, int s, Mat xnum, Mat xtext, Mat yprev, Mat Y);

/* Convert a normalized (lat,lon) pair back to raw degrees (in place, len-2). */
void dataset_denorm(const Dataset *d, float *latlon);

/* Legacy 2-way sample-level split (deterministic shuffle). Kept as a utility;
 * NOT leakage-safe for overlapping windows — use dataset_split3 for training. */
void dataset_split(const Dataset *d, float val_frac, uint64_t seed,
                   int **train, int *n_train, int **val, int *n_val);

void dataset_free(Dataset *d);

/* Load a 2-D float32 .npy array. Returns malloc'd data; sets *rows,*cols. */
float *npy_load_2d(const char *path, int *rows, int *cols);

#endif /* TYPHOFORMER_DATA_H */
