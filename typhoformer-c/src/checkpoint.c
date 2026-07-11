/*
 * checkpoint.c — model weight serialization (see checkpoint.h for the format).
 */
#include "checkpoint.h"

#include <stdio.h>
#include <string.h>

#define CKPT_MAGIC "TFW1"
#define CKPT_HDR_BYTES (4 + 9 * (long)sizeof(int))

void checkpoint_save(const char *path, Config c, const ParamList *pl) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    int h[9] = {c.d_num, c.d_text, c.d_model, c.out_dim,
                c.in_len, c.pred_len, c.d_ff, c.n_heads, c.n_layers};
    fwrite(CKPT_MAGIC, 1, 4, f);
    fwrite(h, sizeof(int), 9, f);
    for (int p = 0; p < pl->count; ++p)
        fwrite(pl->item[p].v, sizeof(float), pl->item[p].n, f);
    fclose(f);
}

Config checkpoint_load_config(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    char magic[4]; int h[9];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, CKPT_MAGIC, 4))
        die("%s: not a TyphoFormer checkpoint", path);
    if (fread(h, sizeof(int), 9, f) != 9) die("%s: unexpected end of file", path);
    fclose(f);
    Config c = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};
    return c;
}

void checkpoint_load_params(const char *path, ParamList *pl) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    fseek(f, CKPT_HDR_BYTES, SEEK_SET);
    for (int p = 0; p < pl->count; ++p)
        if (fread(pl->item[p].v, sizeof(float), pl->item[p].n, f) != (size_t)pl->item[p].n)
            die("%s: parameter size mismatch (wrong config?)", path);
    fclose(f);
}
