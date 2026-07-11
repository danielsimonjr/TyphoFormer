/*
 * data.c — CSV + .npy loading and sliding-window dataset construction.
 */
#define _DEFAULT_SOURCE      /* scandir, alphasort, dirent (POSIX) */
#define _POSIX_C_SOURCE 200809L
#include "data.h"

#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Portable helpers: no strdup/strsep/scandir dependency in the public path so
 * the loader builds on POSIX and (via the Win32 branch below) on Windows. */
static char *dupstr(const char *s) {
    size_t n = strlen(s) + 1; char *d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

#ifdef _WIN32
#include <windows.h>
static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}
/* List emb_chunk_*.npy in `dir`, sorted by name. Returns count; *out is a
 * malloc'd array of malloc'd names (caller frees). Untested on Windows. */
static int list_emb_chunks(const char *dir, char ***out) {
    char pat[1024]; snprintf(pat, sizeof pat, "%s\\emb_chunk_*.npy", dir);
    WIN32_FIND_DATAA fd; HANDLE h = FindFirstFileA(pat, &fd);
    *out = NULL;
    if (h == INVALID_HANDLE_VALUE) return 0;
    char **names = NULL; int n = 0, cap = 0;
    do {
        if (n == cap) { cap = cap ? cap * 2 : 16; names = (char **)realloc(names, (size_t)cap * sizeof(char *)); }
        names[n++] = dupstr(fd.cFileName);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    qsort(names, (size_t)n, sizeof(char *), cmp_str);
    *out = names; return n;
}
#else
#include <dirent.h>
static int emb_name_filter(const struct dirent *e) {
    return strncmp(e->d_name, "emb_chunk_", 10) == 0 &&
           strstr(e->d_name, ".npy") != NULL;
}
static int list_emb_chunks(const char *dir, char ***out) {
    struct dirent **ents; int nc = scandir(dir, &ents, emb_name_filter, alphasort);
    *out = NULL;
    if (nc <= 0) return 0;
    char **names = (char **)malloc((size_t)nc * sizeof(char *));
    for (int i = 0; i < nc; ++i) { names[i] = dupstr(ents[i]->d_name); free(ents[i]); }
    free(ents);
    *out = names; return nc;
}
#endif

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
    /* dtype: require little-endian (or not-applicable) float32. The reader is
     * row-major and host-endian, so reject big-endian and non-f4 explicitly
     * instead of silently misreading. */
    char *descr = strstr(hdr, "'descr'");
    if (!descr) die("%s: .npy header missing 'descr'", path);
    char *q = strchr(descr, ':'); if (q) q = strchr(q, '\'');
    if (!q) die("%s: malformed .npy descr", path);
    char dt[8] = {0}; int di = 0; ++q;
    while (*q && *q != '\'' && di < 7) dt[di++] = *q++;
    if (!(strcmp(dt, "<f4") == 0 || strcmp(dt, "|f4") == 0 || strcmp(dt, "=f4") == 0))
        die("%s: expected little-endian float32 ('<f4'), got '%s'", path, dt);
    /* fortran_order must be False (we read C-order row-major). */
    char *fo = strstr(hdr, "'fortran_order'");
    if (fo && strstr(fo, "True")) die("%s: fortran_order=True is not supported (save as C-order)", path);
    char *s = strstr(hdr, "'shape'");
    if (!s) die("%s: .npy header missing 'shape'", path);
    s = strchr(s, '(');
    int r = 0, c = 0;
    if (!s || sscanf(s, "(%d, %d", &r, &c) != 2) die("%s: bad or non-2D shape", path);
    if (r <= 0 || c <= 0) die("%s: non-positive shape (%d,%d)", path, r, c);
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
        char *p = line;                              /* portable comma split (no strsep) */
        while (nf < NFIELD && p) {
            fields[nf++] = p;
            char *comma = strchr(p, ',');
            if (comma) { *comma = 0; p = comma + 1; } else p = NULL;
        }
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
    d.n_storms = gid + 1;

    /* ---- embeddings ---- */
    char **names; int nc = list_emb_chunks(embdir, &names);
    if (nc <= 0) { die("no emb_chunk_*.npy in %s", embdir); }
    d.emb = malloc((size_t)n * d.d_text * sizeof(float));
    int got = 0;
    for (int i = 0; i < nc; ++i) {
        char path[1024]; snprintf(path, sizeof(path), "%s/%s", embdir, names[i]);
        int r, c; float *chunk = npy_load_2d(path, &r, &c);
        assert(c == d.d_text);
        if (got + r > n) r = n - got;                 /* guard */
        memcpy(&d.emb[(size_t)got * d.d_text], chunk, (size_t)r * d.d_text * sizeof(float));
        got += r; free(chunk); free(names[i]);
    }
    free(names);
    if (got != n) { die("embeddings (%d) != records (%d)", got, n); }

    /* Features/coords are left RAW here; dataset_split3 + dataset_standardize
     * fit and apply normalization on the TRAIN storms only (no leakage).
     * Until then, coordinate stats are identity so dataset_get returns degrees. */
    d.cmean[0] = d.cmean[1] = 0.0f; d.cstd[0] = d.cstd[1] = 1.0f;

    /* ---- sliding windows within a storm ---- */
    int need = in_len + pred_len;
    d.start = malloc((size_t)n * sizeof(int)); d.n_samples = 0;
    for (int i = 0; i + need <= n; ++i)
        if (d.gid[i] == d.gid[i + need - 1]) d.start[d.n_samples++] = i;

    return d;
}

/* Fit a z-score over the records flagged in `use` (or all records if use==NULL);
 * writes mean/std and normalizes d->num in place. */
static void fit_apply_features(Dataset *d, const char *use) {
    for (int j = 0; j < d->d_num; ++j) {
        double m = 0.0; long c = 0;
        for (int i = 0; i < d->n_records; ++i)
            if (!use || use[i]) { m += d->num[i * d->d_num + j]; ++c; }
        if (c == 0) c = 1;
        m /= c;
        double v = 0.0;
        for (int i = 0; i < d->n_records; ++i)
            if (!use || use[i]) { double t = d->num[i * d->d_num + j] - m; v += t * t; }
        v = sqrt(v / c); if (v < 1e-6) v = 1.0;
        d->mean[j] = (float)m; d->std[j] = (float)v;
        for (int i = 0; i < d->n_records; ++i)
            d->num[i * d->d_num + j] = (float)((d->num[i * d->d_num + j] - m) / v);
    }
}

void dataset_standardize(Dataset *d) {
    if (d->prewindowed) return;                    /* bin path standardizes itself */
    char *use = (char *)malloc((size_t)d->n_records);
    int have_split = (d->storm_split != NULL);
    for (int i = 0; i < d->n_records; ++i)
        use[i] = have_split ? (d->storm_split[d->gid[i]] == 0) : 1;   /* train storms */
    fit_apply_features(d, use);
    /* coordinate stats on the same train records */
    double lm = 0, om = 0; long c = 0;
    for (int i = 0; i < d->n_records; ++i) if (use[i]) { lm += d->lat[i]; om += d->lon[i]; ++c; }
    if (c == 0) c = 1;
    lm /= c; om /= c;
    double lv = 0, ov = 0;
    for (int i = 0; i < d->n_records; ++i) if (use[i]) {
        double a = d->lat[i] - lm, b = d->lon[i] - om; lv += a * a; ov += b * b; }
    lv = sqrt(lv / c);
    ov = sqrt(ov / c);
    if (lv < 1e-6) lv = 1.0;
    if (ov < 1e-6) ov = 1.0;
    d->cmean[0] = (float)lm; d->cmean[1] = (float)om;
    d->cstd[0]  = (float)lv; d->cstd[1]  = (float)ov;
    free(use);
}

void dataset_apply_stats(Dataset *d, const float *mean, const float *std, int n,
                         const float *cmean, const float *cstd) {
    for (int j = 0; j < d->d_num && j < n; ++j) {
        float m = mean[j], s = (std[j] > 1e-12f) ? std[j] : 1.0f;
        d->mean[j] = m; d->std[j] = s;
        for (int i = 0; i < d->n_records; ++i)
            d->num[i * d->d_num + j] = (d->num[i * d->d_num + j] - m) / s;
    }
    if (cmean && cstd) {
        d->cmean[0] = cmean[0]; d->cmean[1] = cmean[1];
        d->cstd[0]  = (cstd[0] > 1e-12f) ? cstd[0] : 1.0f;
        d->cstd[1]  = (cstd[1] > 1e-12f) ? cstd[1] : 1.0f;
    }
}

Dataset dataset_load_bin(const char *path) {
    Dataset d; memset(&d, 0, sizeof(d));
    FILE *f = fopen(path, "rb");
    if (!f) { die("cannot open %s", path); }
    char magic[4]; int hdr[5];
    if (fread(magic, 1, 4, f) != 4 ||
        (memcmp(magic, "TFB1", 4) && memcmp(magic, "TFB2", 4))) { die("%s: bad magic", path); }
    int has_seed = (memcmp(magic, "TFB2", 4) == 0);   /* TFB2 stores a true seed coord */
    if (fread(hdr, sizeof(int), 5, f) != 5) die("%s: unexpected end of file", path);
    d.n_samples = hdr[0]; d.in_len = hdr[1];
    int feat = hdr[2]; d.pred_len = hdr[3]; int out = hdr[4];
    d.d_num = 14; d.d_text = feat - d.d_num; d.prewindowed = 1;
    assert(out == 2 && d.d_text > 0);
    d.cmean[0] = d.cmean[1] = 0.0f; d.cstd[0] = d.cstd[1] = 1.0f;   /* coords raw */
    size_t in_sz = (size_t)d.in_len * feat, tg_sz = (size_t)d.pred_len * out;
    d.win_in = malloc((size_t)d.n_samples * in_sz * sizeof(float));
    d.win_tg = malloc((size_t)d.n_samples * tg_sz * sizeof(float));
    if (has_seed) d.win_seed = malloc((size_t)d.n_samples * 2 * sizeof(float));
    for (int s = 0; s < d.n_samples; ++s) {
        if (fread(&d.win_in[s * in_sz], sizeof(float), in_sz, f) != in_sz) die("%s: unexpected end of file", path);
        if (fread(&d.win_tg[s * tg_sz], sizeof(float), tg_sz, f) != tg_sz) die("%s: unexpected end of file", path);
        if (has_seed && fread(&d.win_seed[s * 2], sizeof(float), 2, f) != 2) die("%s: unexpected end of file", path);
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

/* Normalize a (lat,lon) pair with the dataset's coordinate stats. */
static void cnorm(const Dataset *d, float *lat, float *lon) {
    *lat = (*lat - d->cmean[0]) / d->cstd[0];
    *lon = (*lon - d->cmean[1]) / d->cstd[1];
}

void dataset_denorm(const Dataset *d, float *latlon) {
    latlon[0] = latlon[0] * d->cstd[0] + d->cmean[0];
    latlon[1] = latlon[1] * d->cstd[1] + d->cmean[1];
}

/* Zero the text branch in place (numbers-only ablation). */
static void maybe_drop_text(const Dataset *d, Mat xtext) {
    if (d->no_text) memset(xtext.data, 0, (size_t)xtext.rows * xtext.cols * sizeof(float));
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
        /* TFB2 stores the true last-observed seed; TFB1 falls back to tg[0]. */
        if (d->win_seed) { yprev.data[0] = d->win_seed[s * 2]; yprev.data[1] = d->win_seed[s * 2 + 1]; }
        else             { yprev.data[0] = tg[0];              yprev.data[1] = tg[1]; }
        cnorm(d, &yprev.data[0], &yprev.data[1]);
        for (int k = 0; k < d->pred_len; ++k) {
            Y.data[k * 2] = tg[k * 2]; Y.data[k * 2 + 1] = tg[k * 2 + 1];
            cnorm(d, &Y.data[k * 2], &Y.data[k * 2 + 1]);
        }
        maybe_drop_text(d, xtext);
        return;
    }
    int r0 = d->start[s];
    for (int t = 0; t < d->in_len; ++t) {
        memcpy(&xnum.data[t * d->d_num],  &d->num[(size_t)(r0 + t) * d->d_num], d->d_num * sizeof(float));
        memcpy(&xtext.data[t * d->d_text], &d->emb[(size_t)(r0 + t) * d->d_text], d->d_text * sizeof(float));
    }
    int last = r0 + d->in_len - 1;                    /* last observed coord = decoder seed */
    yprev.data[0] = d->lat[last]; yprev.data[1] = d->lon[last];
    cnorm(d, &yprev.data[0], &yprev.data[1]);
    for (int k = 0; k < d->pred_len; ++k) {
        Y.data[k * 2 + 0] = d->lat[r0 + d->in_len + k];
        Y.data[k * 2 + 1] = d->lon[r0 + d->in_len + k];
        cnorm(d, &Y.data[k * 2], &Y.data[k * 2 + 1]);
    }
    maybe_drop_text(d, xtext);
}

Split dataset_split3(Dataset *d, float val_frac, float test_frac, unsigned long seed) {
    Split sp; memset(&sp, 0, sizeof sp);
    if (d->prewindowed || d->n_storms <= 0 || d->start == NULL) {
        /* No storm/overlap info (.tfb): fall back to a sample-level split.
         * Windows may overlap, so this is not fully leakage-safe — prefer CSV. */
        int N = d->n_samples;
        int *idx = (int *)malloc((size_t)(N ? N : 1) * sizeof(int));
        for (int i = 0; i < N; ++i) idx[i] = i;
        unsigned long rng = seed ? seed : 1;
        for (int i = N - 1; i > 0; --i) {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            int j = (int)(rng % (unsigned long)(i + 1));
            int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        }
        int nv = (int)(val_frac * N), nte = (int)(test_frac * N);
        if (nv + nte > N) nte = N - nv;
        sp.n_val = nv; sp.n_test = nte; sp.n_train = N - nv - nte;
        sp.val   = (int *)malloc((size_t)(sp.n_val   ? sp.n_val   : 1) * sizeof(int));
        sp.test  = (int *)malloc((size_t)(sp.n_test  ? sp.n_test  : 1) * sizeof(int));
        sp.train = (int *)malloc((size_t)(sp.n_train ? sp.n_train : 1) * sizeof(int));
        for (int i = 0; i < nv; ++i)             sp.val[i]            = idx[i];
        for (int i = 0; i < nte; ++i)            sp.test[i]           = idx[nv + i];
        for (int i = nv + nte; i < N; ++i)       sp.train[i - nv - nte] = idx[i];
        free(idx);
        return sp;
    }
    int S = d->n_storms;
    /* shuffle storm ids deterministically, then slice by fraction */
    int *order = (int *)malloc((size_t)(S ? S : 1) * sizeof(int));
    for (int i = 0; i < S; ++i) order[i] = i;
    unsigned long rng = seed ? seed : 1;
    for (int i = S - 1; i > 0; --i) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        int j = (int)(rng % (unsigned long)(i + 1));
        int t = order[i]; order[i] = order[j]; order[j] = t;
    }
    int nv = (int)(val_frac * S), nte = (int)(test_frac * S);
    if (nv + nte > S) { nte = S - nv; if (nte < 0) nte = 0; }
    free(d->storm_split);
    d->storm_split = (int *)malloc((size_t)(S ? S : 1) * sizeof(int));
    for (int i = 0; i < S; ++i) {
        int label = (i < nv) ? 1 : (i < nv + nte) ? 2 : 0;   /* val / test / train */
        d->storm_split[order[i]] = label;
    }
    free(order);
    /* bucket samples by their storm's label */
    for (int s = 0; s < d->n_samples; ++s) {
        int lab = d->storm_split[d->gid[d->start[s]]];
        if      (lab == 0) sp.n_train++;
        else if (lab == 1) sp.n_val++;
        else               sp.n_test++;
    }
    sp.train = (int *)malloc((size_t)(sp.n_train ? sp.n_train : 1) * sizeof(int));
    sp.val   = (int *)malloc((size_t)(sp.n_val   ? sp.n_val   : 1) * sizeof(int));
    sp.test  = (int *)malloc((size_t)(sp.n_test  ? sp.n_test  : 1) * sizeof(int));
    int it = 0, iv = 0, ie = 0;
    for (int s = 0; s < d->n_samples; ++s) {
        int lab = d->storm_split[d->gid[d->start[s]]];
        if      (lab == 0) sp.train[it++] = s;
        else if (lab == 1) sp.val[iv++]   = s;
        else               sp.test[ie++]  = s;
    }
    return sp;
}

void split_free(Split *s) {
    free(s->train); free(s->val); free(s->test);
    memset(s, 0, sizeof *s);
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
    free(d->win_in); free(d->win_tg); free(d->win_seed); free(d->storm_split);
    memset(d, 0, sizeof(*d));
}
