/*
 * data.h — dataset loading: parse the HURDAT2 CSV numerical features and the
 * precomputed MiniLM embedding chunks (.npy), group by storm, and build
 * sliding-window samples.
 */
#ifndef TYPHOFORMER_DATA_H
#define TYPHOFORMER_DATA_H

#include "tensor.h"

typedef struct {
    int    n_records;
    int    d_num, d_text, in_len, pred_len;
    float *num;          /* n_records * d_num  (standardized in place) */
    float *emb;          /* n_records * d_text */
    float *lat, *lon;    /* n_records */
    int   *gid;          /* storm group id per record */
    float  mean[64], std[64];
    int   *start;        /* n_samples window start record indices */
    int    n_samples;
} Dataset;

/* Load records from `csv` and embeddings from `embdir` (emb_chunk_*.npy),
 * build sliding windows, and standardize the numerical features. */
Dataset dataset_load(const char *csv, const char *embdir, int in_len, int pred_len);

/* Materialise sample `s` into caller-provided matrices:
 *   xnum [in_len,d_num], xtext [in_len,d_text], yprev [1,2], Y [pred_len,2].
 * yprev is the last *observed* coordinate (decoder seed). */
void dataset_get(const Dataset *d, int s, Mat xnum, Mat xtext, Mat yprev, Mat Y);

/* Deterministic shuffle + split of sample indices into train/val. */
void dataset_split(const Dataset *d, float val_frac, unsigned long seed,
                   int **train, int *n_train, int **val, int *n_val);

void dataset_free(Dataset *d);

/* Load a 2-D float32 .npy array. Returns malloc'd data; sets *rows,*cols. */
float *npy_load_2d(const char *path, int *rows, int *cols);

#endif /* TYPHOFORMER_DATA_H */
