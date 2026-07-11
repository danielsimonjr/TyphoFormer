/*
 * checkpoint.c — model weight serialization (see checkpoint.h for the format).
 */
#include "checkpoint.h"

#include <stdio.h>
#include <string.h>

#define MAGIC1 "TFW1"   /* legacy: no stats block */
#define MAGIC2 "TFW2"   /* current: config + stats + params */

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
        (memcmp(magic, MAGIC1, 4) && memcmp(magic, MAGIC2, 4)))
        die("%s: not a TyphoFormer checkpoint", path);
    if (fread(h, sizeof(int), 9, f) != 9) die("%s: unexpected end of file", path);
    fclose(f);
    Config c = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};
    return c;
}

/* Position `f` at the first parameter float, handling both formats. */
static void seek_to_params(FILE *f, const char *path) {
    char magic[4];
    if (fread(magic, 1, 4, f) != 4) die("%s: unexpected end of file", path);
    if (fseek(f, 9 * (long)sizeof(int), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
    if (!memcmp(magic, MAGIC2, 4)) {
        int ns; if (fread(&ns, sizeof(int), 1, f) != 1) die("%s: unexpected end of file", path);
        if (fseek(f, 2L * ns * (long)sizeof(float), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
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
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, MAGIC2, 4)) { fclose(f); return 0; }
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
