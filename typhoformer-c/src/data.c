/*
 * data.c — CSV + .npy loading and sliding-window dataset construction.
 */
#define _DEFAULT_SOURCE      /* scandir, alphasort, strsep, dirent */
#define _POSIX_C_SOURCE 200809L
#include "data.h"

#include <assert.h>
#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- .npy (v1.0/2.0) 2-D float32 loader ----------------------------- */
float *npy_load_2d(const char *path, int *rows, int *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) { die("cannot open %s", path); }
    unsigned char h[8];
    if (fread(h, 1, 8, f) != 8 || h[0] != 0x93 || memcmp(h + 1, "NUMPY", 5)) {
        die("%s: not a .npy file", path);
    }
    unsigned int hlen;
    if (h[6] == 1) { unsigned char b[2]; if(fread(b,1,2,f)!=2)die("%s: unexpected end of file", path); hlen = b[0] | (b[1] << 8); }
    else           { unsigned char b[4]; if(fread(b,1,4,f)!=4)die("%s: unexpected end of file", path); hlen = b[0] | (b[1]<<8) | (b[2]<<16) | ((unsigned)b[3]<<24); }
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) die("%s: unexpected end of file", path);
    hdr[hlen] = 0;
    if (!strstr(hdr, "f4")) { die("%s: expected float32", path); }
    char *s = strstr(hdr, "'shape':"); assert(s); s = strchr(s, '(');
    int r = 0, c = 0;
    if (sscanf(s, "(%d, %d", &r, &c) != 2) { die("%s: bad shape", path); }
    free(hdr);
    float *data = (float *)malloc((size_t)r * c * sizeof(float));
    if (fread(data, sizeof(float), (size_t)r * c, f) != (size_t)r * c) die("%s: unexpected end of file", path);
    fclose(f);
    *rows = r; *cols = c;
    return data;
}

/* ---- small parsing helpers ------------------------------------------ */
static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) ++s;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    return s;
}
static float parse_latlon(const char *v) {
    char buf[32]; strncpy(buf, v, 31); buf[31] = 0; char *t = trim(buf);
    size_t n = strlen(t);
    if (n == 0) return 0.0f;
    char hemi = t[n - 1];
    if (hemi == 'N' || hemi == 'S' || hemi == 'E' || hemi == 'W') t[n - 1] = 0;
    float x = (float)atof(t);
    if (hemi == 'S' || hemi == 'W') x = -x;
    return x;
}

/* numerical feature columns (0-based) in the HURDAT2 CSV: max_wind,
 * min_pressure, then the twelve 34/50/64-kt wind radii. */
static const int NUMCOL[14] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
#define LATCOL 5
#define LONCOL 6
#define IDCOL  0
#define NFIELD 22

static int emb_name_filter(const struct dirent *e) {
    return strncmp(e->d_name, "emb_chunk_", 10) == 0 &&
           strstr(e->d_name, ".npy") != NULL;
}

Dataset dataset_load(const char *csv, const char *embdir, int in_len, int pred_len) {
    Dataset d; memset(&d, 0, sizeof(d));
    d.d_num = 14; d.d_text = 384; d.in_len = in_len; d.pred_len = pred_len;

    /* ---- CSV ---- */
    FILE *f = fopen(csv, "r");
    if (!f) { die("cannot open %s", csv); }
    int cap = 4096; d.num = malloc((size_t)cap * d.d_num * sizeof(float));
    d.lat = malloc(cap * sizeof(float)); d.lon = malloc(cap * sizeof(float));
    d.gid = malloc(cap * sizeof(int));
    char line[4096]; int n = 0, gid = -1; char prev_id[64] = "";
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        if (lineno++ == 0) continue;                 /* header */
        if (!strchr(line, ',')) continue;
        char *fields[NFIELD]; int nf = 0;
        char *p = line, *tok;
        while (nf < NFIELD && (tok = strsep(&p, ",")) != NULL) fields[nf++] = tok;
        if (nf < 21) continue;
        if (n == cap) {
            cap *= 2;
            d.num = realloc(d.num, (size_t)cap * d.d_num * sizeof(float));
            d.lat = realloc(d.lat, cap * sizeof(float));
            d.lon = realloc(d.lon, cap * sizeof(float));
            d.gid = realloc(d.gid, cap * sizeof(int));
        }
        char *id = trim(fields[IDCOL]);
        if (strcmp(id, prev_id) != 0) { ++gid; strncpy(prev_id, id, 63); prev_id[63] = 0; }
        d.gid[n] = gid;
        d.lat[n] = parse_latlon(fields[LATCOL]);
        d.lon[n] = parse_latlon(fields[LONCOL]);
        for (int j = 0; j < d.d_num; ++j) d.num[n * d.d_num + j] = (float)atof(trim(fields[NUMCOL[j]]));
        ++n;
    }
    fclose(f);
    d.n_records = n;

    /* ---- embeddings ---- */
    struct dirent **names; int nc = scandir(embdir, &names, emb_name_filter, alphasort);
    if (nc <= 0) { die("no emb_chunk_*.npy in %s", embdir); }
    d.emb = malloc((size_t)n * d.d_text * sizeof(float));
    int got = 0;
    for (int i = 0; i < nc; ++i) {
        char path[1024]; snprintf(path, sizeof(path), "%s/%s", embdir, names[i]->d_name);
        int r, c; float *chunk = npy_load_2d(path, &r, &c);
        assert(c == d.d_text);
        if (got + r > n) r = n - got;                 /* guard */
        memcpy(&d.emb[(size_t)got * d.d_text], chunk, (size_t)r * d.d_text * sizeof(float));
        got += r; free(chunk); free(names[i]);
    }
    free(names);
    if (got != n) { die("embeddings (%d) != records (%d)", got, n); }

    /* ---- standardize numerical features (z-score per column) ---- */
    for (int j = 0; j < d.d_num; ++j) {
        double m = 0.0; for (int i = 0; i < n; ++i) m += d.num[i * d.d_num + j]; m /= n;
        double v = 0.0; for (int i = 0; i < n; ++i) { double t = d.num[i * d.d_num + j] - m; v += t * t; }
        v = sqrt(v / n); if (v < 1e-6) v = 1.0;
        d.mean[j] = (float)m; d.std[j] = (float)v;
        for (int i = 0; i < n; ++i) d.num[i * d.d_num + j] = (float)((d.num[i * d.d_num + j] - m) / v);
    }

    /* ---- sliding windows within a storm ---- */
    int need = in_len + pred_len;
    d.start = malloc((size_t)n * sizeof(int)); d.n_samples = 0;
    for (int i = 0; i + need <= n; ++i)
        if (d.gid[i] == d.gid[i + need - 1]) d.start[d.n_samples++] = i;

    return d;
}

Dataset dataset_load_bin(const char *path) {
    Dataset d; memset(&d, 0, sizeof(d));
    FILE *f = fopen(path, "rb");
    if (!f) { die("cannot open %s", path); }
    char magic[4]; int hdr[5];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TFB1", 4)) { die("%s: bad magic", path); }
    if (fread(hdr, sizeof(int), 5, f) != 5) die("%s: unexpected end of file", path);
    d.n_samples = hdr[0]; d.in_len = hdr[1];
    int feat = hdr[2]; d.pred_len = hdr[3]; int out = hdr[4];
    d.d_num = 14; d.d_text = feat - d.d_num; d.prewindowed = 1;
    assert(out == 2 && d.d_text > 0);
    size_t in_sz = (size_t)d.in_len * feat, tg_sz = (size_t)d.pred_len * out;
    d.win_in = malloc((size_t)d.n_samples * in_sz * sizeof(float));
    d.win_tg = malloc((size_t)d.n_samples * tg_sz * sizeof(float));
    for (int s = 0; s < d.n_samples; ++s) {
        if (fread(&d.win_in[s * in_sz], sizeof(float), in_sz, f) != in_sz) die("%s: unexpected end of file", path);
        if (fread(&d.win_tg[s * tg_sz], sizeof(float), tg_sz, f) != tg_sz) die("%s: unexpected end of file", path);
    }
    fclose(f);
    /* standardize the numerical feature columns (0..d_num-1) across all steps */
    for (int j = 0; j < d.d_num; ++j) {
        double m = 0.0; long cnt = (long)d.n_samples * d.in_len;
        for (int s = 0; s < d.n_samples; ++s)
            for (int t = 0; t < d.in_len; ++t) m += d.win_in[s * in_sz + (size_t)t * feat + j];
        m /= cnt;
        double v = 0.0;
        for (int s = 0; s < d.n_samples; ++s)
            for (int t = 0; t < d.in_len; ++t) { double x = d.win_in[s * in_sz + (size_t)t * feat + j] - m; v += x * x; }
        v = sqrt(v / cnt); if (v < 1e-6) v = 1.0;
        d.mean[j] = (float)m; d.std[j] = (float)v;
        for (int s = 0; s < d.n_samples; ++s)
            for (int t = 0; t < d.in_len; ++t) {
                size_t off = s * in_sz + (size_t)t * feat + j;
                d.win_in[off] = (float)((d.win_in[off] - m) / v);
            }
    }
    return d;
}

void dataset_get(const Dataset *d, int s, Mat xnum, Mat xtext, Mat yprev, Mat Y) {
    if (d->prewindowed) {
        const int F = d->d_num + d->d_text;
        const float *in = &d->win_in[(size_t)s * d->in_len * F];
        for (int t = 0; t < d->in_len; ++t) {
            memcpy(&xnum.data[t * d->d_num],  &in[(size_t)t * F], d->d_num * sizeof(float));
            memcpy(&xtext.data[t * d->d_text], &in[(size_t)t * F + d->d_num], d->d_text * sizeof(float));
        }
        const float *tg = &d->win_tg[(size_t)s * d->pred_len * 2];
        yprev.data[0] = tg[0]; yprev.data[1] = tg[1];   /* seed = first target (upstream) */
        for (int k = 0; k < d->pred_len; ++k) { Y.data[k * 2] = tg[k * 2]; Y.data[k * 2 + 1] = tg[k * 2 + 1]; }
        return;
    }
    int r0 = d->start[s];
    for (int t = 0; t < d->in_len; ++t) {
        memcpy(&xnum.data[t * d->d_num],  &d->num[(size_t)(r0 + t) * d->d_num], d->d_num * sizeof(float));
        memcpy(&xtext.data[t * d->d_text], &d->emb[(size_t)(r0 + t) * d->d_text], d->d_text * sizeof(float));
    }
    int last = r0 + d->in_len - 1;                    /* last observed coord = decoder seed */
    yprev.data[0] = d->lat[last]; yprev.data[1] = d->lon[last];
    for (int k = 0; k < d->pred_len; ++k) {
        Y.data[k * 2 + 0] = d->lat[r0 + d->in_len + k];
        Y.data[k * 2 + 1] = d->lon[r0 + d->in_len + k];
    }
}

void dataset_split(const Dataset *d, float val_frac, unsigned long seed,
                   int **train, int *n_train, int **val, int *n_val) {
    int N = d->n_samples;
    int *idx = malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) idx[i] = i;
    unsigned long rng = seed ? seed : 1;              /* xorshift shuffle */
    for (int i = N - 1; i > 0; --i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        int j = (int)(rng % (unsigned long)(i + 1));
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
    int nv = (int)(val_frac * N);
    *n_val = nv; *n_train = N - nv;
    *val = malloc((nv ? nv : 1) * sizeof(int));
    *train = malloc((N - nv ? N - nv : 1) * sizeof(int));
    for (int i = 0; i < nv; ++i) (*val)[i] = idx[i];
    for (int i = nv; i < N; ++i) (*train)[i - nv] = idx[i];
    free(idx);
}

void dataset_free(Dataset *d) {
    free(d->num); free(d->emb); free(d->lat); free(d->lon); free(d->gid); free(d->start);
    free(d->win_in); free(d->win_tg);
    memset(d, 0, sizeof(*d));
}
