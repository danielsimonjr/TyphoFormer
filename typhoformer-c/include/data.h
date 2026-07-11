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

#include "tensor.h"

typedef struct {
    int    n_records;
    int    d_num, d_text, in_len, pred_len;
    float *num;          /* n_records * d_num  (standardized in place)      */
    float *emb;          /* n_records * d_text                              */
    float *lat, *lon;    /* n_records (raw degrees)                         */
    int   *gid;          /* storm group id per record                       */
    int    n_storms;     /* number of distinct storms (max gid + 1)         */
    int   *storm_split;  /* n_storms: 0=train 1=val 2=test (NULL until split)*/
    float  mean[64], std[64];   /* feature normalization (train-only fit)   */
    float  cmean[2], cstd[2];   /* coord (lat,lon) normalization; identity  */
                                /* {0,1} means "raw" (bin path / unset)     */
    int   *start;        /* n_samples window start record indices           */
    int    n_samples;
    int    prewindowed;  /* 1 => samples come from a .tfb file (below)       */
    int    no_text;      /* 1 => dataset_get zeros the text branch (ablation)*/
    float *win_in;       /* n_samples * in_len * (d_num+d_text)             */
    float *win_tg;       /* n_samples * pred_len * 2                        */
    float *win_seed;     /* n_samples * 2: true decoder seed (TFB2) or NULL */
    long  *tkey;         /* n_records: date*10000+time (co-activity key)    */
    float *nbr;          /* n_records * TF_NBR_K * TF_NBR_NF (co-active nbrs)*/
    int   *nbr_cnt;      /* n_records: number of co-active neighbours        */
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
Split dataset_split3(Dataset *d, float val_frac, float test_frac, unsigned long seed);
void  split_free(Split *s);

/* Append motion features to the input: per record, the (raw) position lat, lon
 * and the step-to-step velocity dlat, dlon (0 at a storm's first record). This
 * gives the model the trajectory signal it otherwise never sees. Increases
 * d->d_num by 4. Call BEFORE dataset_standardize. No-op for the .tfb path (it
 * carries no coordinate history). */
void dataset_add_motion(Dataset *d);

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
void dataset_split(const Dataset *d, float val_frac, unsigned long seed,
                   int **train, int *n_train, int **val, int *n_val);

void dataset_free(Dataset *d);

/* Load a 2-D float32 .npy array. Returns malloc'd data; sets *rows,*cols. */
float *npy_load_2d(const char *path, int *rows, int *cols);

#endif /* TYPHOFORMER_DATA_H */
