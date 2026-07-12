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

/* ---- .npy (v1.0/2.0) 2-D float32 loader -----------------------------
 * NumPy's .npy binary format is: a 6-byte magic string "\x93NUMPY", one
 * version-major byte, one version-minor byte, a little-endian header-length
 * field (2 bytes in v1.x, 4 bytes in v2.x), then an ASCII Python-dict header
 * padded to a 64-byte boundary, then the raw array bytes. The header dict
 * carries three keys we care about: 'descr' (dtype string), 'fortran_order'
 * (memory layout), and 'shape' (a tuple). We only support the exact case the
 * embedding chunks are saved in: 2-D, C-order, little-endian float32. */
float *npy_load_2d(const char *path, int *rows, int *cols) {
    FILE *f = fopen(path, "rb");
    if (!f) { die("cannot open %s", path); }
    unsigned char h[8];
    /* h[0]=0x93 magic byte, h[1..5]="NUMPY", h[6]=version major, h[7]=minor. */
    if (fread(h, 1, 8, f) != 8 || h[0] != 0x93 || memcmp(h + 1, "NUMPY", 5)) {
        die("%s: not a .npy file", path);
    }
    /* Header length is 2 bytes for format v1.x, 4 bytes for v2.x; both stored
     * little-endian, hence the explicit byte-shift assembly (endian-safe). */
    unsigned int hlen;
    if (h[6] == 1) { unsigned char b[2]; if(fread(b,1,2,f)!=2)die("%s: unexpected end of file", path); hlen = b[0] | (b[1] << 8); }
    else           { unsigned char b[4]; if(fread(b,1,4,f)!=4)die("%s: unexpected end of file", path); hlen = b[0] | (b[1]<<8) | (b[2]<<16) | ((unsigned)b[3]<<24); }
    char *hdr = (char *)malloc(hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) die("%s: unexpected end of file", path);
    hdr[hlen] = 0;
    /* dtype: require little-endian (or not-applicable) float32. The reader is
     * row-major and host-endian, so reject big-endian and non-f4 explicitly
     * instead of silently misreading. */
    /* Locate 'descr' and read the quoted dtype string that follows the colon,
     * e.g. "'descr': '<f4'". We copy chars until the closing quote into dt. */
    char *descr = strstr(hdr, "'descr'");
    if (!descr) die("%s: .npy header missing 'descr'", path);
    char *q = strchr(descr, ':'); if (q) q = strchr(q, '\'');
    if (!q) die("%s: malformed .npy descr", path);
    char dt[8] = {0}; int di = 0; ++q;
    while (*q && *q != '\'' && di < 7) dt[di++] = *q++;
    /* The leading char is the byte-order flag: '<' little-endian, '|' N/A (for
     * single-byte types), '=' native. We only read host-endian float32, so we
     * accept those three and reject big-endian ('>f4') or any non-f4 dtype
     * rather than silently misreading the bytes. */
    if (!(strcmp(dt, "<f4") == 0 || strcmp(dt, "|f4") == 0 || strcmp(dt, "=f4") == 0))
        die("%s: expected little-endian float32 ('<f4'), got '%s'", path, dt);
    /* fortran_order must be False: our indexing below assumes C-order (row-
     * major). A Fortran/column-major array would need a transpose we don't do. */
    char *fo = strstr(hdr, "'fortran_order'");
    if (fo && strstr(fo, "True")) die("%s: fortran_order=True is not supported (save as C-order)", path);
    /* Parse the 2-D shape tuple "(rows, cols)". A 1-D or 3-D array will not
     * match "(%d, %d" and is rejected. */
    char *s = strstr(hdr, "'shape'");
    if (!s) die("%s: .npy header missing 'shape'", path);
    s = strchr(s, '(');
    int r = 0, c = 0;
    if (!s || sscanf(s, "(%d, %d", &r, &c) != 2) die("%s: bad or non-2D shape", path);
    if (r <= 0 || c <= 0) die("%s: non-positive shape (%d,%d)", path, r, c);
    free(hdr);
    /* The remaining file bytes are exactly r*c float32 values in row-major
     * order; read them straight into a flat buffer the caller owns. */
    float *data = (float *)malloc((size_t)r * c * sizeof(float));
    if (fread(data, sizeof(float), (size_t)r * c, f) != (size_t)r * c) die("%s: unexpected end of file", path);
    fclose(f);
    *rows = r; *cols = c;
    return data;
}

/* ---- small parsing helpers ------------------------------------------ */
/* Strip leading and trailing ASCII whitespace, in place. Returns a pointer
 * into `s` (advanced past leading spaces); trailing spaces are overwritten
 * with NUL. HURDAT2 fields are space-padded, so nearly every field is trimmed. */
static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) ++s;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = 0;
    return s;
}
/* Parse a HURDAT2 latitude/longitude like "27.0N" or "88.5W" into a signed
 * float degree value. The trailing hemisphere letter is stripped; S and W
 * (southern / western hemisphere) negate the magnitude, N and E stay positive.
 * An empty field parses to 0. Works on a local copy so the source is untouched. */
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

/* numerical feature columns (0-based) in the HURDAT2 CSV: max_wind (col 7),
 * min_pressure (col 8), then the twelve 34/50/64-kt wind radii (cols 9..20).
 * These 14 columns become d->num row [max_wind, min_pressure, r34_ne, ...].
 * The first element (index 0, max_wind) is the one dataset_build_neighbors
 * reads for its Δmax_wind neighbour feature. */
static const int NUMCOL[14] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
#define LATCOL 5     /* latitude column  */
#define LONCOL 6     /* longitude column */
#define IDCOL  0     /* storm id column (a change signals a new storm)      */
#define NFIELD 22    /* max CSV fields we split a line into                 */

/* Parse the HURDAT2 CSV and the MiniLM embedding chunks into a Dataset.
 * Records are read in file order (which HURDAT2 keeps chronological within a
 * storm), consecutive rows sharing an id are grouped into one storm, and
 * sliding windows are enumerated at the end. Everything stays RAW here; the
 * caller then does dataset_split3 + dataset_standardize for leakage-safe
 * normalization. `in_len` is the history/encoder length, `pred_len` the
 * forecast horizon. */
Dataset dataset_load(const char *csv, const char *embdir, int in_len, int pred_len) {
    Dataset d; memset(&d, 0, sizeof(d));
    d.d_num = 14; d.d_text = 384; d.in_len = in_len; d.pred_len = pred_len;

    /* ---- CSV ---- */
    FILE *f = fopen(csv, "r");
    if (!f) { die("cannot open %s", csv); }
    /* Grow-on-demand arrays: start with cap rows and double when full. The
     * five per-record arrays grow together and stay index-aligned. */
    int cap = 4096; d.num = malloc((size_t)cap * d.d_num * sizeof(float));
    d.lat = malloc(cap * sizeof(float)); d.lon = malloc(cap * sizeof(float));
    d.gid = malloc(cap * sizeof(int)); d.tkey = malloc(cap * sizeof(long));
    char line[4096]; int n = 0, gid = -1; char prev_id[64] = "";
    int lineno = 0;
    while (fgets(line, sizeof(line), f)) {
        if (lineno++ == 0) continue;                 /* skip the CSV header row */
        if (!strchr(line, ',')) continue;            /* skip blank/garbage lines */
        /* Split the line into up to NFIELD comma-separated fields in place by
         * replacing each comma with a NUL. Portable substitute for strsep. */
        char *fields[NFIELD]; int nf = 0;
        char *p = line;                              /* portable comma split (no strsep) */
        while (nf < NFIELD && p) {
            fields[nf++] = p;
            char *comma = strchr(p, ',');
            if (comma) { *comma = 0; p = comma + 1; } else p = NULL;
        }
        if (nf < 21) continue;                       /* need through col 20 (last radius) */
        if (n == cap) {                              /* full: double every parallel array */
            cap *= 2;
            d.num = realloc(d.num, (size_t)cap * d.d_num * sizeof(float));
            d.lat = realloc(d.lat, cap * sizeof(float));
            d.lon = realloc(d.lon, cap * sizeof(float));
            d.gid = realloc(d.gid, cap * sizeof(int));
            d.tkey = realloc(d.tkey, cap * sizeof(long));
        }
        /* New storm whenever the id string changes from the previous row.
         * gid starts at -1 so the first record gets gid 0. This relies on all
         * records of a storm being contiguous in the file (HURDAT2 property). */
        char *id = trim(fields[IDCOL]);
        if (strcmp(id, prev_id) != 0) { ++gid; strncpy(prev_id, id, 63); prev_id[63] = 0; }
        d.gid[n] = gid;
        /* Timestamp key packs date (col 1, e.g. YYYYMMDD) and time (col 2,
         * e.g. HHMM) into one comparable long: date*10000 + time. Records with
         * equal tkey are simultaneous -> candidate co-active neighbours. */
        d.tkey[n] = atol(trim(fields[1])) * 10000L + atol(trim(fields[2]));  /* date*1e4 + time */
        d.lat[n] = parse_latlon(fields[LATCOL]);
        d.lon[n] = parse_latlon(fields[LONCOL]);
        /* Copy the 14 numeric feature columns (RAW physical units) into row n. */
        for (int j = 0; j < d.d_num; ++j) d.num[n * d.d_num + j] = (float)atof(trim(fields[NUMCOL[j]]));
        ++n;
    }
    fclose(f);
    d.n_records = n;
    d.n_storms = gid + 1;             /* gids are 0..gid, so count is gid+1 */

    /* ---- embeddings ----
     * The MiniLM text embeddings were precomputed offline and saved as one or
     * more emb_chunk_*.npy files, each [rows, 384]. Concatenated in sorted name
     * order they must line up 1:1 with the CSV records (row i of emb <-> record
     * i), so the final `got` count must equal `n` or the data is misaligned.
     *
     * `--emb=none` is the TEXT-FREE path: no embeddings exist (e.g. a CSV
     * converted straight from raw HURDAT2 by tools/hurdat2_to_csv.py). The
     * language branch is fed all-zero vectors — exactly what the --no_text
     * ablation feeds, which FINDINGS §2/§6 measured as marginally BETTER than
     * real text — and no_text is set so downstream code knows. Explicit opt-in
     * ("none"), not a fallback: a typo'd embedding path still dies loudly. */
    if (strcmp(embdir, "none") == 0) {
        d.emb = calloc((size_t)n * d.d_text, sizeof(float));
        d.no_text = 1;
        printf("text-free mode (--emb=none): language branch fed zeros (== --no_text)\n");
        goto windows;
    }
    char **names; int nc = list_emb_chunks(embdir, &names);
    if (nc <= 0) { die("no emb_chunk_*.npy in %s (use --emb=none for text-free data)", embdir); }
    d.emb = malloc((size_t)n * d.d_text * sizeof(float));
    int got = 0;
    for (int i = 0; i < nc; ++i) {
        char path[1024]; snprintf(path, sizeof(path), "%s/%s", embdir, names[i]);
        int r, c; float *chunk = npy_load_2d(path, &r, &c);
        assert(c == d.d_text);                        /* each chunk must be 384-wide */
        if (got + r > n) r = n - got;                 /* guard against overrun past n */
        memcpy(&d.emb[(size_t)got * d.d_text], chunk, (size_t)r * d.d_text * sizeof(float));
        got += r; free(chunk); free(names[i]);
    }
    free(names);
    if (got != n) { die("embeddings (%d) != records (%d)", got, n); }

windows:
    /* Features/coords are left RAW here; dataset_split3 + dataset_standardize
     * fit and apply normalization on the TRAIN storms only (no leakage).
     * Until then, coordinate stats are identity so dataset_get returns degrees. */
    d.cmean[0] = d.cmean[1] = 0.0f; d.cstd[0] = d.cstd[1] = 1.0f;

    /* ---- sliding windows within a storm ----
     * A sample is a contiguous block of need = in_len + pred_len records: the
     * first in_len are history/input, the last pred_len are the forecast
     * target. We record only the START index of each valid window in d.start.
     * A window is valid only if it lies entirely inside ONE storm, tested by
     * checking that the first and last record of the block share a gid (since
     * gids are contiguous, equal endpoints imply the whole block is one storm).
     * This is what prevents a window from spanning two different storms. */
    int need = in_len + pred_len;
    d.start = malloc((size_t)n * sizeof(int)); d.n_samples = 0;
    for (int i = 0; i + need <= n; ++i)
        if (d.gid[i] == d.gid[i + need - 1]) d.start[d.n_samples++] = i;

    return d;
}

/* Fit a per-column z-score over the records flagged in `use` (or all records
 * if use==NULL), then apply it to ALL records in place. Column j is centered by
 * its mean and divided by its (population) std; the stats are stored in
 * d->mean[j]/d->std[j] so the exact same transform can later be reproduced on
 * val/test data (and saved in a checkpoint). Two passes per column: pass 1
 * accumulates the mean, pass 2 the variance. A near-zero std is clamped to 1 to
 * avoid divide-by-zero / exploding a constant column. Note the FIT uses only
 * `use`-flagged (train) rows, but the APPLY loop touches every row — that is
 * how train-only statistics get pushed onto val/test without leaking. */
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

/* Append 4 motion features to every record, growing d->d_num from `old` to
 * old+4. The new columns, in order, are:
 *   [old+0] lat        — absolute position (RAW degrees)
 *   [old+1] lon        — absolute position (RAW degrees)
 *   [old+2] dlat       — velocity: lat[i]-lat[i-1] within the same storm
 *   [old+3] dlon       — velocity: lon[i]-lon[i-1] within the same storm
 * Velocity is 0 at the FIRST record of each storm (no predecessor in-storm),
 * detected by `same` (previous record shares this record's gid). Injecting
 * position+velocity gives the encoder the trajectory signal it otherwise never
 * sees (the CSV features are intensity/size, not where the storm is heading).
 * Must be called BEFORE dataset_standardize so these columns get z-scored too.
 * These raw values are still raw here — standardization happens later. No-op on
 * the .tfb path, which carries no per-record coordinate history. */
void dataset_add_motion(Dataset *d) {
    if (d->prewindowed) return;                     /* no coord history in .tfb */
    int old = d->d_num, nd = old + 4;
    float *nm = (float *)malloc((size_t)d->n_records * nd * sizeof(float));
    for (int i = 0; i < d->n_records; ++i) {
        /* copy the existing `old` columns, then write the 4 motion columns */
        memcpy(&nm[(size_t)i * nd], &d->num[(size_t)i * old], (size_t)old * sizeof(float));
        int same = (i > 0 && d->gid[i] == d->gid[i - 1]);   /* velocity within a storm */
        nm[(size_t)i * nd + old + 0] = d->lat[i];
        nm[(size_t)i * nd + old + 1] = d->lon[i];
        nm[(size_t)i * nd + old + 2] = same ? d->lat[i] - d->lat[i - 1] : 0.0f;
        nm[(size_t)i * nd + old + 3] = same ? d->lon[i] - d->lon[i - 1] : 0.0f;
    }
    free(d->num); d->num = nm; d->d_num = nd;
}

/* Append 7 second-order physics features to every record, growing d->d_num by
 * 7. The new columns, in order (all RAW here; standardized later):
 *   [old+0] alat   — acceleration:  Δlat_i − Δlat_{i-1}  (0 unless i-2 in-storm)
 *   [old+1] alon   — acceleration:  Δlon_i − Δlon_{i-1}
 *   [old+2] speed  — translation speed |v| = √(Δlat² + Δlon²), degrees/step
 *   [old+3] ulat   — heading unit vector, lat component  (v/|v|; 0 when |v|≈0)
 *   [old+4] ulon   — heading unit vector, lon component
 *   [old+5] dsin   — sin(2π·day-of-year/365.25)   seasonal phase
 *   [old+6] dcos   — cos(2π·day-of-year/365.25)
 * Rationale: recurvature is what the cv decoder's correction must learn, and it
 * is literally a change of heading — so hand the model acceleration, speed, and
 * heading explicitly instead of hoping attention derives them from position
 * pairs; steering flow is seasonal and the (sin,cos) phase encodes that without
 * a year-end discontinuity. Like dataset_add_motion this must run BEFORE
 * dataset_standardize, works within storm boundaries (velocities do not reach
 * across storms), and is a no-op on the .tfb path (no coordinate history). */
void dataset_add_physics(Dataset *d) {
    if (d->prewindowed) return;
    int old = d->d_num, nd = old + 7;
    float *nm = (float *)malloc((size_t)d->n_records * nd * sizeof(float));
    for (int i = 0; i < d->n_records; ++i) {
        memcpy(&nm[(size_t)i * nd], &d->num[(size_t)i * old], (size_t)old * sizeof(float));
        float *o = &nm[(size_t)i * nd + old];
        int s1 = (i > 0 && d->gid[i] == d->gid[i - 1]);   /* i-1 in the same storm */
        int s2 = (i > 1 && d->gid[i] == d->gid[i - 2]);   /* i-2 too (gids contiguous) */
        float vlat = s1 ? d->lat[i] - d->lat[i - 1] : 0.0f;
        float vlon = s1 ? d->lon[i] - d->lon[i - 1] : 0.0f;
        float plat = s2 ? d->lat[i - 1] - d->lat[i - 2] : 0.0f;
        float plon = s2 ? d->lon[i - 1] - d->lon[i - 2] : 0.0f;
        o[0] = (s1 && s2) ? vlat - plat : 0.0f;           /* acceleration */
        o[1] = (s1 && s2) ? vlon - plon : 0.0f;
        float speed = sqrtf(vlat * vlat + vlon * vlon);
        o[2] = speed;
        o[3] = (speed > 1e-6f) ? vlat / speed : 0.0f;     /* heading unit vector */
        o[4] = (speed > 1e-6f) ? vlon / speed : 0.0f;
        /* tkey = YYYYMMDD·1e4 + HHMM: recover month/day, approximate the
         * day-of-year (month lengths averaged — phase accuracy of ±2 days is
         * plenty for a seasonal signal), and encode it as a continuous angle. */
        long date = d->tkey ? d->tkey[i] / 10000L : 0;
        int month = (int)((date / 100) % 100), day = (int)(date % 100);
        double doy = (month - 1) * 30.44 + day;
        double ang = 2.0 * 3.14159265358979323846 * doy / 365.25;
        o[5] = (float)sin(ang);
        o[6] = (float)cos(ang);
    }
    free(d->num); d->num = nm; d->d_num = nd;
}

/* ---- co-active spatial neighbours -----------------------------------
 * Multiple storms can be alive at the same moment (same tkey). For each record
 * we record up to TF_NBR_K other storms active at that instant, each as a
 * RELATIVE vector so the signal is translation-invariant (about the target
 * storm's frame, not absolute geography). This is the real multi-node spatial
 * input used by the --co_spatial ablation. */
typedef struct { long key; int idx; } TKPair;  /* (timestamp, original record index) */
/* Order TKPairs by timestamp key so equal-timestamp records become contiguous. */
static int tk_cmp(const void *a, const void *b) {
    long ka = ((const TKPair *)a)->key, kb = ((const TKPair *)b)->key;
    return (ka < kb) ? -1 : (ka > kb) ? 1 : 0;
}
/* Build the neighbour tables. Strategy: sort record indices by timestamp, walk
 * the sorted list in equal-key runs (each run = all storms active at one
 * moment), and for every record in a run copy up to K OTHER-storm records into
 * its neighbour slots. Because we sort once and scan runs, this is O(n log n)
 * plus O(run^2) within each shared-timestamp group. */
void dataset_build_neighbors(Dataset *d) {
    if (d->prewindowed || !d->tkey) return;
    int n = d->n_records;
    free(d->nbr); free(d->nbr_cnt);
    /* nbr is zero-initialized so unused slots (fewer than K neighbours) read 0. */
    d->nbr = (float *)calloc((size_t)n * TF_NBR_K * TF_NBR_NF, sizeof(float));
    d->nbr_cnt = (int *)calloc((size_t)n, sizeof(int));
    TKPair *pr = (TKPair *)malloc((size_t)n * sizeof(TKPair));
    for (int i = 0; i < n; ++i) { pr[i].key = d->tkey[i]; pr[i].idx = i; }
    qsort(pr, (size_t)n, sizeof(TKPair), tk_cmp);
    int g0 = 0;
    while (g0 < n) {                                    /* one group per shared timestamp */
        int g1 = g0;                                    /* [g0,g1) = records with this tkey */
        while (g1 < n && pr[g1].key == pr[g0].key) ++g1;
        for (int a = g0; a < g1; ++a) {
            int i = pr[a].idx, cnt = 0;                  /* i = the "center" record */
            for (int b = g0; b < g1 && cnt < TF_NBR_K; ++b) {
                int j = pr[b].idx;                       /* j = candidate neighbour */
                if (j == i || d->gid[j] == d->gid[i]) continue;   /* skip self / same storm */
                /* Write one neighbour vector into slot `cnt` of record i, as
                 * (target - center) differences: relative position + relative
                 * intensity. Relative form keeps it invariant to where on the
                 * globe the pair sits. */
                float *o = &d->nbr[((size_t)i * TF_NBR_K + cnt) * TF_NBR_NF];
                o[0] = d->lat[j] - d->lat[i];                     /* relative lat */
                o[1] = d->lon[j] - d->lon[i];                     /* relative lon */
                o[2] = d->num[(size_t)j * d->d_num] - d->num[(size_t)i * d->d_num]; /* Δmax_wind (col 0) */
                ++cnt;
            }
            d->nbr_cnt[i] = cnt;                         /* how many slots were filled */
        }
        g0 = g1;
    }
    free(pr);
}
/* Fetch sample s's neighbours as of its seed (last observed / in_len-1)
 * timestep — the same instant the decoder starts rolling out from. Copies the
 * fixed K*NF block into nbrmat (unfilled slots are zero) and returns the valid
 * count in *cnt. Zero on the .tfb path (no per-record neighbour table). */
void dataset_neighbors(const Dataset *d, int s, Mat nbrmat, int *cnt) {
    if (d->prewindowed || !d->nbr) { *cnt = 0; return; }
    int rec = d->start[s] + d->in_len - 1;
    *cnt = d->nbr_cnt[rec];
    memcpy(nbrmat.data, &d->nbr[(size_t)rec * TF_NBR_K * TF_NBR_NF],
           (size_t)TF_NBR_K * TF_NBR_NF * sizeof(float));
}

/* Seed velocity = the last observed step's displacement (used by the --cv
 * constant-velocity option to warm-start the decoder with real motion).
 * Returned in NORMALIZED coordinate units so it can be added directly to the
 * normalized seed position the decoder rolls out from.
 *
 * Why only the std divides and not the mean: normalizing a position p is
 * (p - cmean)/cstd, but here we output a DIFFERENCE of two positions,
 *   (p_last - p_prev) normalized = ((p_last-cmean)/cstd) - ((p_prev-cmean)/cstd)
 *                                = (p_last - p_prev)/cstd,
 * so the additive cmean cancels exactly and we divide the raw displacement by
 * cstd alone. Needs at least 2 history steps (in_len >= 2). Zero on the .tfb
 * path, which stores no per-record coordinate history to difference. */
void dataset_seed_velocity(const Dataset *d, int s, Mat vout) {
    vout.data[0] = vout.data[1] = 0.0f;
    if (d->prewindowed || d->in_len < 2) return;
    int last = d->start[s] + d->in_len - 1;           /* seed record */
    vout.data[0] = (d->lat[last] - d->lat[last - 1]) / d->cstd[0];
    vout.data[1] = (d->lon[last] - d->lon[last - 1]) / d->cstd[1];
}

/* Fit + apply the leakage-safe normalization. The `use` mask selects TRAIN
 * records only (storm_split label 0) when a split exists; before any split it
 * falls back to all records. Both the feature z-score and the coordinate
 * (lat,lon) z-score statistics are fit exclusively on those train records so
 * nothing about val/test distribution leaks into the transform. Features are
 * z-scored in place; coordinate stats are only STORED (cmean/cstd) and applied
 * lazily by cnorm inside dataset_get. No-op on the .tfb path, which standardizes
 * itself inside dataset_load_bin. */
void dataset_standardize(Dataset *d) {
    if (d->prewindowed) return;                    /* bin path standardizes itself */
    char *use = (char *)malloc((size_t)d->n_records);
    int have_split = (d->storm_split != NULL);
    for (int i = 0; i < d->n_records; ++i)
        use[i] = have_split ? (d->storm_split[d->gid[i]] == 0) : 1;   /* train storms */
    fit_apply_features(d, use);
    /* coordinate stats on the same train records: mean then variance, per axis,
     * with the same 1e-6 std-clamp used for features. */
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

/* Apply EXTERNALLY-supplied statistics instead of fitting fresh ones — used at
 * inference to reproduce exactly the transform a checkpoint was trained with.
 * Feature columns [0, min(d_num,n)) are z-scored in place using the passed
 * mean/std (std clamped away from ~0); the stats are also cached in d->mean/std.
 * If cmean/cstd are non-NULL the coordinate stats are set too; pass NULL to
 * leave coordinates raw (identity {0,1}). The input d->num must still be RAW. */
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

/* Load a pre-windowed .tfb dataset (already sliced into samples offline).
 * Layout on disk:
 *   [4 bytes]  magic: "TFB1" (legacy) or "TFB2" (carries a true seed coord)
 *   [5 ints]   header: n_samples, in_len, feat (=d_num+d_text), pred_len, out
 *   then per sample, tightly packed float32:
 *     in_sz  = in_len * feat   floats  (input window: numeric ++ text per step)
 *     tg_sz  = pred_len * out  floats  (target coords, out==2 -> lat,lon)
 *     [TFB2] 2 floats          the TRUE last-observed seed coordinate
 * The seed matters for leakage: TFB1 has no stored seed, so dataset_get falls
 * back to seeding the decoder with the FIRST target — a mild label leak. TFB2
 * stores the genuine last-observed position, matching the CSV path. This loader
 * sets prewindowed=1 and leaves coordinate stats identity (coords stay raw;
 * cnorm becomes a no-op), then standardizes only the numeric feature columns. */
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
    /* feat is the combined per-step width; the first 14 are numeric, the rest
     * are the 384-wide text embedding (d_text = feat - 14). */
    d.d_num = 14; d.d_text = feat - d.d_num; d.prewindowed = 1;
    assert(out == 2 && d.d_text > 0);                 /* targets are (lat,lon) */
    d.cmean[0] = d.cmean[1] = 0.0f; d.cstd[0] = d.cstd[1] = 1.0f;   /* coords raw */
    size_t in_sz = (size_t)d.in_len * feat, tg_sz = (size_t)d.pred_len * out;
    d.win_in = malloc((size_t)d.n_samples * in_sz * sizeof(float));
    d.win_tg = malloc((size_t)d.n_samples * tg_sz * sizeof(float));
    if (has_seed) d.win_seed = malloc((size_t)d.n_samples * 2 * sizeof(float));
    /* Read each sample's three blocks contiguously in the on-disk order. */
    for (int s = 0; s < d.n_samples; ++s) {
        if (fread(&d.win_in[s * in_sz], sizeof(float), in_sz, f) != in_sz) die("%s: unexpected end of file", path);
        if (fread(&d.win_tg[s * tg_sz], sizeof(float), tg_sz, f) != tg_sz) die("%s: unexpected end of file", path);
        if (has_seed && fread(&d.win_seed[s * 2], sizeof(float), 2, f) != 2) die("%s: unexpected end of file", path);
    }
    fclose(f);
    /* Standardize the numerical feature columns (0..d_num-1) across ALL samples
     * and ALL timesteps. Because .tfb has no storm split, this fits over the
     * whole file — the offline windowing step is expected to have split first.
     * Text columns and target coords are left as-is. Two passes (mean, then
     * variance) mirror fit_apply_features, with the same 1e-6 std clamp. */
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

/* Forward coordinate normalization: (raw degrees) -> (z-scored). With identity
 * stats {0,1} (the .tfb / pre-standardize state) this is a pass-through, so
 * dataset_get returns raw degrees until dataset_standardize sets real stats. */
static void cnorm(const Dataset *d, float *lat, float *lon) {
    *lat = (*lat - d->cmean[0]) / d->cstd[0];
    *lon = (*lon - d->cmean[1]) / d->cstd[1];
}

/* Inverse of cnorm: (z-scored) -> (raw degrees). The public denorm used to turn
 * a model's normalized (lat,lon) prediction, or a normalized target, back into
 * physical degrees for error reporting/plotting. In place, length-2 buffer. */
void dataset_denorm(const Dataset *d, float *latlon) {
    latlon[0] = latlon[0] * d->cstd[0] + d->cmean[0];
    latlon[1] = latlon[1] * d->cstd[1] + d->cmean[1];
}

/* Zero the text branch in place (numbers-only ablation, d->no_text). */
static void maybe_drop_text(const Dataset *d, Mat xtext) {
    if (d->no_text) memset(xtext.data, 0, (size_t)xtext.rows * xtext.cols * sizeof(float));
}

/* Materialise sample s into caller-provided matrices:
 *   xnum  [in_len, d_num]   — numeric input per history step (already z-scored)
 *   xtext [in_len, d_text]  — text embedding per history step
 *   yprev [1, 2]            — decoder seed coordinate (NORMALIZED)
 *   Y     [pred_len, 2]     — target coordinates (NORMALIZED)
 * Coordinates handed out are normalized via cnorm; callers use dataset_denorm to
 * get degrees back. There are two source layouts — pre-windowed (.tfb) tensors,
 * or the per-record CSV arrays sliced by d->start[s] — but the OUTPUT contract
 * is identical, so the model code is oblivious to which loader was used. */
void dataset_get(const Dataset *d, int s, Mat xnum, Mat xtext, Mat yprev, Mat Y) {
    if (d->prewindowed) {
        /* .tfb path: input window is stored interleaved as [in_len, feat] with
         * feat = d_num + d_text; split each step back into numeric ++ text. */
        const int F = d->d_num + d->d_text;
        const float *in = &d->win_in[(size_t)s * d->in_len * F];
        for (int t = 0; t < d->in_len; ++t) {
            memcpy(&xnum.data[t * d->d_num],  &in[(size_t)t * F], d->d_num * sizeof(float));
            memcpy(&xtext.data[t * d->d_text], &in[(size_t)t * F + d->d_num], d->d_text * sizeof(float));
        }
        const float *tg = &d->win_tg[(size_t)s * d->pred_len * 2];
        /* Seed: TFB2 stores the true last-observed coordinate; legacy TFB1 has
         * none, so it falls back to tg[0] — the first TARGET — which is a mild
         * label leak (the decoder starts from a value it is meant to predict). */
        if (d->win_seed) { yprev.data[0] = d->win_seed[s * 2]; yprev.data[1] = d->win_seed[s * 2 + 1]; }
        else             { yprev.data[0] = tg[0];              yprev.data[1] = tg[1]; }
        cnorm(d, &yprev.data[0], &yprev.data[1]);
        /* Copy + normalize each of the pred_len target (lat,lon) pairs. */
        for (int k = 0; k < d->pred_len; ++k) {
            Y.data[k * 2] = tg[k * 2]; Y.data[k * 2 + 1] = tg[k * 2 + 1];
            cnorm(d, &Y.data[k * 2], &Y.data[k * 2 + 1]);
        }
        maybe_drop_text(d, xtext);
        return;
    }
    /* CSV path: records [r0 .. r0+in_len-1] are history, the next pred_len are
     * targets. d->start[s] guarantees the whole span lies in one storm. */
    int r0 = d->start[s];
    for (int t = 0; t < d->in_len; ++t) {
        memcpy(&xnum.data[t * d->d_num],  &d->num[(size_t)(r0 + t) * d->d_num], d->d_num * sizeof(float));
        memcpy(&xtext.data[t * d->d_text], &d->emb[(size_t)(r0 + t) * d->d_text], d->d_text * sizeof(float));
    }
    int last = r0 + d->in_len - 1;                    /* last observed coord = decoder seed */
    yprev.data[0] = d->lat[last]; yprev.data[1] = d->lon[last];
    cnorm(d, &yprev.data[0], &yprev.data[1]);
    /* Targets are the pred_len records immediately after the history window. */
    for (int k = 0; k < d->pred_len; ++k) {
        Y.data[k * 2 + 0] = d->lat[r0 + d->in_len + k];
        Y.data[k * 2 + 1] = d->lon[r0 + d->in_len + k];
        cnorm(d, &Y.data[k * 2], &Y.data[k * 2 + 1]);
    }
    maybe_drop_text(d, xtext);
}

/* Deterministic 3-way split. The leakage-safe branch works at STORM
 * granularity: it shuffles storm ids with a fixed-seed xorshift PRNG, slices
 * the shuffled order into val/test/train by fraction, records each storm's
 * label in d->storm_split, and finally buckets SAMPLES by their storm's label.
 * Because whole storms move together, two overlapping windows (which share
 * records) always land in the same split — no window straddles a boundary, and
 * no train record can appear inside a val/test window. The same seed always
 * yields the same partition (reproducibility). */
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
    /* Fisher-Yates shuffle of storm ids with an xorshift PRNG (fixed seed =>
     * deterministic), then slice the shuffled order by fraction. */
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
    /* First nv shuffled storms -> val(1), next nte -> test(2), rest -> train(0).
     * order[i] maps shuffled position i back to the real storm id. */
    for (int i = 0; i < S; ++i) {
        int label = (i < nv) ? 1 : (i < nv + nte) ? 2 : 0;   /* val / test / train */
        d->storm_split[order[i]] = label;
    }
    free(order);
    /* First pass: count samples per split so we can size the index arrays.
     * A sample's split = the split of the storm owning its first record. */
    for (int s = 0; s < d->n_samples; ++s) {
        int lab = d->storm_split[d->gid[d->start[s]]];
        if      (lab == 0) sp.n_train++;
        else if (lab == 1) sp.n_val++;
        else               sp.n_test++;
    }
    sp.train = (int *)malloc((size_t)(sp.n_train ? sp.n_train : 1) * sizeof(int));
    sp.val   = (int *)malloc((size_t)(sp.n_val   ? sp.n_val   : 1) * sizeof(int));
    sp.test  = (int *)malloc((size_t)(sp.n_test  ? sp.n_test  : 1) * sizeof(int));
    /* Second pass: fill each split's sample-index array (indices into d->start). */
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

/* Legacy 2-way SAMPLE-level split: shuffle sample indices and cut off the first
 * val_frac as validation, the rest as train. NOT leakage-safe when windows
 * overlap (adjacent windows share records and can land on opposite sides), so
 * it is kept only as a utility; training uses dataset_split3. */
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

/* Release every owned allocation (both loader paths) and zero the struct so a
 * double-free is a harmless no-op. free(NULL) is fine, so unused-path pointers
 * (which stay NULL from the initial memset) need no special handling. */
void dataset_free(Dataset *d) {
    free(d->num); free(d->emb); free(d->lat); free(d->lon); free(d->gid); free(d->start);
    free(d->win_in); free(d->win_tg); free(d->win_seed); free(d->storm_split);
    free(d->tkey); free(d->nbr); free(d->nbr_cnt);
    memset(d, 0, sizeof(*d));
}
