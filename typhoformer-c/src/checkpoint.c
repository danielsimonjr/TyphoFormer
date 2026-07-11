/*
 * checkpoint.c — model weight serialization (see checkpoint.h for the format).
 */
#include "checkpoint.h"

#include <stdio.h>
#include <string.h>

#define MAGIC1 "TFW1"   /* legacy: no stats block                 */
#define MAGIC2 "TFW2"   /* config + feature stats + params        */
#define MAGIC3 "TFW3"   /* + coordinate (lat/lon) stats           */

static int is_magic(const char *m, const char *w) { return memcmp(m, w, 4) == 0; }

void checkpoint_save3(const char *path, Config c, const float *mean, const float *std,
                      int n_stats, const float *cmean, const float *cstd,
                      const ParamList *pl) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    int hc[9] = {c.d_num, c.d_text, c.d_model, c.out_dim,
                 c.in_len, c.pred_len, c.d_ff, c.n_heads, c.n_layers};
    int ns = (mean && std) ? n_stats : 0;
    int has_coord = (cmean && cstd) ? 1 : 0;
    fwrite(MAGIC3, 1, 4, f);
    fwrite(hc, sizeof(int), 9, f);
    fwrite(&ns, sizeof(int), 1, f);
    if (ns) { fwrite(mean, sizeof(float), ns, f); fwrite(std, sizeof(float), ns, f); }
    fwrite(&has_coord, sizeof(int), 1, f);
    if (has_coord) { fwrite(cmean, sizeof(float), 2, f); fwrite(cstd, sizeof(float), 2, f); }
    for (int p = 0; p < pl->count; ++p)
        fwrite(pl->item[p].v, sizeof(float), pl->item[p].n, f);
    fclose(f);
}

void checkpoint_save2(const char *path, Config c, const float *mean, const float *std,
                      int n_stats, const ParamList *pl) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    int hc[9] = {c.d_num, c.d_text, c.d_model, c.out_dim,
                 c.in_len, c.pred_len, c.d_ff, c.n_heads, c.n_layers};
    int ns = (mean && std) ? n_stats : 0;
    fwrite(MAGIC2, 1, 4, f);
    fwrite(hc, sizeof(int), 9, f);
    fwrite(&ns, sizeof(int), 1, f);
    if (ns) { fwrite(mean, sizeof(float), ns, f); fwrite(std, sizeof(float), ns, f); }
    for (int p = 0; p < pl->count; ++p)
        fwrite(pl->item[p].v, sizeof(float), pl->item[p].n, f);
    fclose(f);
}

void checkpoint_save(const char *path, Config c, const ParamList *pl) {
    checkpoint_save2(path, c, NULL, NULL, 0, pl);
}

Config checkpoint_load_config(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    char magic[4]; int h[9];
    if (fread(magic, 1, 4, f) != 4 ||
        (!is_magic(magic, MAGIC1) && !is_magic(magic, MAGIC2) && !is_magic(magic, MAGIC3)))
        die("%s: not a TyphoFormer checkpoint", path);
    if (fread(h, sizeof(int), 9, f) != 9) die("%s: unexpected end of file", path);
    fclose(f);
    Config c = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};
    return c;
}

/* Position `f` at the first parameter float, handling all formats. */
static void seek_to_params(FILE *f, const char *path) {
    char magic[4];
    if (fread(magic, 1, 4, f) != 4) die("%s: unexpected end of file", path);
    if (fseek(f, 9 * (long)sizeof(int), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
    if (is_magic(magic, MAGIC2) || is_magic(magic, MAGIC3)) {
        int ns; if (fread(&ns, sizeof(int), 1, f) != 1) die("%s: unexpected end of file", path);
        if (fseek(f, 2L * ns * (long)sizeof(float), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
    }
    if (is_magic(magic, MAGIC3)) {
        int hc; if (fread(&hc, sizeof(int), 1, f) != 1) die("%s: unexpected end of file", path);
        if (hc && fseek(f, 4L * (long)sizeof(float), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
    }
}

void checkpoint_load_params(const char *path, ParamList *pl) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    seek_to_params(f, path);
    for (int p = 0; p < pl->count; ++p)
        if (fread(pl->item[p].v, sizeof(float), pl->item[p].n, f) != (size_t)pl->item[p].n)
            die("%s: parameter size mismatch (wrong config?)", path);
    fclose(f);
}

int checkpoint_load_stats(const char *path, float *mean, float *std) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || (!is_magic(magic, MAGIC2) && !is_magic(magic, MAGIC3))) {
        fclose(f); return 0;
    }
    if (fseek(f, 9 * (long)sizeof(int), SEEK_CUR) != 0) { fclose(f); return 0; }
    int ns;
    if (fread(&ns, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (ns > 0) {
        if (fread(mean, sizeof(float), ns, f) != (size_t)ns ||
            fread(std,  sizeof(float), ns, f) != (size_t)ns)
            die("%s: unexpected end of file", path);
    }
    fclose(f);
    return ns;
}

int checkpoint_load_coord_stats(const char *path, float *cmean, float *cstd) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || !is_magic(magic, MAGIC3)) { fclose(f); return 0; }
    if (fseek(f, 9 * (long)sizeof(int), SEEK_CUR) != 0) { fclose(f); return 0; }
    int ns;
    if (fread(&ns, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (ns > 0 && fseek(f, 2L * ns * (long)sizeof(float), SEEK_CUR) != 0) { fclose(f); return 0; }
    int has_coord;
    if (fread(&has_coord, sizeof(int), 1, f) != 1 || !has_coord) { fclose(f); return 0; }
    if (fread(cmean, sizeof(float), 2, f) != 2 || fread(cstd, sizeof(float), 2, f) != 2)
        die("%s: unexpected end of file", path);
    fclose(f);
    return 1;
}

/* ---- optimizer-state sidecar (Adam moments) for resuming training -------- */
#define OMAGIC "TFO1"

void checkpoint_save_optim(const char *path, const Adam *a, int epoch) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    long n = a->n, t = a->t;
    fwrite(OMAGIC, 1, 4, f);
    fwrite(&n, sizeof(long), 1, f);
    fwrite(&t, sizeof(long), 1, f);
    fwrite(&epoch, sizeof(int), 1, f);
    fwrite(&a->lr, sizeof(float), 1, f);
    fwrite(a->fm, sizeof(float), (size_t)n, f);
    fwrite(a->sm, sizeof(float), (size_t)n, f);
    fclose(f);
}

int checkpoint_load_optim(const char *path, Adam *a, int *epoch) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[4]; long n, t; int ep; float lr;
    if (fread(magic, 1, 4, f) != 4 || !is_magic(magic, OMAGIC)) { fclose(f); return 0; }
    if (fread(&n, sizeof(long), 1, f) != 1 || n != a->n) { fclose(f); return 0; }
    if (fread(&t, sizeof(long), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&ep, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&lr, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(a->fm, sizeof(float), (size_t)n, f) != (size_t)n ||
        fread(a->sm, sizeof(float), (size_t)n, f) != (size_t)n) { fclose(f); return 0; }
    a->t = t; a->lr = lr; if (epoch) *epoch = ep;
    fclose(f);
    return 1;
}
