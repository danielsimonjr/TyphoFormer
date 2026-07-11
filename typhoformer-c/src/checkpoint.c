/*
 * checkpoint.c — model weight serialization (see checkpoint.h for the format).
 *
 * On-disk design overview
 * -----------------------
 * A checkpoint is a flat, little-endian binary blob: a 4-byte ASCII "magic"
 * that names the format version, a fixed 9-int32 config header, optional
 * normalization statistics, and finally the raw model parameters concatenated
 * in registration order. There is no length prefix or per-parameter tag on the
 * parameter blob — the reader recomputes every tensor's element count from the
 * Config it was given, then reads exactly that many floats. That is why loading
 * with the wrong Config is caught as a "parameter size mismatch": the byte
 * stream only lines up when the reader's model geometry matches the writer's.
 *
 * Three format versions coexist, each a strict superset of the previous one, so
 * a newer reader can transparently load an older file:
 *   TFW1  magic + config                      (no stats at all)
 *   TFW2  TFW1 + per-feature (mean,std) block  (input feature normalization)
 *   TFW3  TFW2 + coordinate (lat/lon) block    (target/position normalization)
 * The "magic" is what lets the reader know which optional blocks to skip when
 * seeking to the parameters (see seek_to_params below).
 */
#include "checkpoint.h"

#include <stdio.h>
#include <string.h>

/* The 4-byte format tags, written verbatim as the first 4 bytes of the file.
 * They are ASCII so a `file`/hexdump immediately reveals the version, and their
 * ordering (1 < 2 < 3) mirrors how many optional blocks follow the header. */
#define MAGIC1 "TFW1"   /* legacy: no stats block                 */
#define MAGIC2 "TFW2"   /* config + feature stats + params        */
#define MAGIC3 "TFW3"   /* + coordinate (lat/lon) stats           */

/* Compare a just-read 4-byte tag against one of the MAGIC constants. Uses a
 * fixed length of 4 (not strcmp) because the tags are not NUL-terminated on
 * disk — they are exactly 4 raw bytes. */
static int is_magic(const char *m, const char *w) { return memcmp(m, w, 4) == 0; }

/* Write a full TFW3 checkpoint. Byte layout, in exact write order:
 *     "TFW3"                                     4 bytes
 *     hc[9] int32                                the Config header (see below)
 *     ns    int32                                # feature stats (0 or d_num)
 *     mean[ns], std[ns] float32                  present iff ns > 0
 *     has_coord int32                            0 or 1
 *     cmean[2], cstd[2]  float32                 present iff has_coord == 1
 *     params...          float32                 pl in registration order
 * `mean`/`std` may be NULL to omit the feature block (ns is forced to 0), and
 * likewise `cmean`/`cstd` may be NULL to omit the coordinate block. The header
 * is always written; the two stats blocks are independently optional, and the
 * has_coord flag is the reader's signal for whether the coord block exists. */
void checkpoint_save3(const char *path, Config c, const float *mean, const float *std,
                      int n_stats, const float *cmean, const float *cstd,
                      const ParamList *pl) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); return; }
    /* The 9-int32 config header. Field ORDER is load-bearing: checkpoint_load_config
     * reconstructs the Config positionally from exactly this sequence, so it must
     * never be reordered without bumping the format. */
    int hc[9] = {c.d_num, c.d_text, c.d_model, c.out_dim,
                 c.in_len, c.pred_len, c.d_ff, c.n_heads, c.n_layers};
    /* If either stats pointer is NULL, write ns=0 so the reader skips the block. */
    int ns = (mean && std) ? n_stats : 0;
    int has_coord = (cmean && cstd) ? 1 : 0;
    fwrite(MAGIC3, 1, 4, f);                 /* [0]  format tag                */
    fwrite(hc, sizeof(int), 9, f);           /* [4]  9 config int32s           */
    fwrite(&ns, sizeof(int), 1, f);          /* [40] feature-stats count       */
    if (ns) { fwrite(mean, sizeof(float), ns, f); fwrite(std, sizeof(float), ns, f); }  /* [44] mean[], then std[] */
    fwrite(&has_coord, sizeof(int), 1, f);   /* coord-block present flag       */
    if (has_coord) { fwrite(cmean, sizeof(float), 2, f); fwrite(cstd, sizeof(float), 2, f); }  /* lat/lon mean, then std */
    /* Parameters: each registered tensor's raw floats, back to back, in the
     * SAME order model_new registered them into the ParamList. No separators. */
    for (int p = 0; p < pl->count; ++p)
        fwrite(pl->item[p].v, sizeof(float), pl->item[p].n, f);
    fclose(f);
}

/* Write a TFW2 checkpoint: identical to TFW3 but WITHOUT the coordinate block.
 * Layout: "TFW2", hc[9], ns, mean[ns]/std[ns], then params. Because TFW3 is a
 * superset, a TFW3-aware reader distinguishes the two purely by the magic tag
 * and simply stops looking for a coord block when it sees "TFW2". */
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

/* Convenience: write a bare TFW2 with no stats at all (ns=0). This is the
 * smallest valid checkpoint — just magic + header + a zero count + params. */
void checkpoint_save(const char *path, Config c, const ParamList *pl) {
    checkpoint_save2(path, c, NULL, NULL, 0, pl);
}

/* Read just the config header. The 9 int32s occupy the same slots in every
 * format version (they follow immediately after the 4-byte magic), so this only
 * needs to validate the magic and read the header — it never touches stats or
 * params. Callers use the returned Config to allocate a model of the right shape
 * before calling checkpoint_load_params, which is what makes the size-mismatch
 * guard meaningful. */
Config checkpoint_load_config(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    char magic[4]; int h[9];
    /* Accept any of the three known magics; anything else is not our file. */
    if (fread(magic, 1, 4, f) != 4 ||
        (!is_magic(magic, MAGIC1) && !is_magic(magic, MAGIC2) && !is_magic(magic, MAGIC3)))
        die("%s: not a TyphoFormer checkpoint", path);
    if (fread(h, sizeof(int), 9, f) != 9) die("%s: unexpected end of file", path);
    fclose(f);
    /* Positional reconstruction — must mirror the hc[9] write order above. */
    Config c = {h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]};
    return c;
}

/* Position `f` at the first parameter float, handling all formats.
 *
 * Rather than parse the optional blocks, this skips over them by seeking, using
 * the magic to decide which blocks exist and the counts embedded in the file to
 * decide how far to jump:
 *   - always skip the 9 int32 config header,
 *   - for TFW2/TFW3, read ns and skip 2*ns floats (the mean[] and std[] arrays),
 *   - for TFW3 only, read has_coord and, if set, skip 4 floats (cmean[2],cstd[2]).
 * After this returns, the file offset sits exactly on the first parameter byte,
 * regardless of which format version produced the file. */
static void seek_to_params(FILE *f, const char *path) {
    char magic[4];
    if (fread(magic, 1, 4, f) != 4) die("%s: unexpected end of file", path);
    if (fseek(f, 9 * (long)sizeof(int), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
    /* TFW2 and TFW3 both carry a feature-stats block: an int count ns followed
     * by ns means and ns stds, i.e. 2*ns floats to jump over. */
    if (is_magic(magic, MAGIC2) || is_magic(magic, MAGIC3)) {
        int ns; if (fread(&ns, sizeof(int), 1, f) != 1) die("%s: unexpected end of file", path);
        if (fseek(f, 2L * ns * (long)sizeof(float), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
    }
    /* Only TFW3 carries the coordinate block. `hc` here is the has_coord flag;
     * when set, 4 floats follow (cmean[2] then cstd[2]). */
    if (is_magic(magic, MAGIC3)) {
        int hc; if (fread(&hc, sizeof(int), 1, f) != 1) die("%s: unexpected end of file", path);
        if (hc && fseek(f, 4L * (long)sizeof(float), SEEK_CUR) != 0) die("%s: bad checkpoint", path);
    }
}

/* Read parameters straight into a preconstructed ParamList (built by model_new
 * from a Config). Because the on-disk parameter blob is untagged, correctness
 * relies entirely on `pl` having the exact same per-tensor element counts, in
 * the same order, as the model that wrote the file.
 *
 * The size-mismatch GUARD: for each tensor we demand fread return exactly n
 * elements. If the reader's config differs from the writer's in any way that
 * changes tensor sizes — a different d_model, a toggled architecture flag that
 * adds/removes a layer, a different n_layers — then either a tensor runs short
 * (fread returns < n, tripping the die here) or the total drifts so a later
 * tensor misaligns. This is the mechanism that forces EVAL-time flags to match
 * TRAINING-time flags: you cannot silently load weights into a differently
 * shaped model and get quietly-wrong results; the byte accounting refuses. */
void checkpoint_load_params(const char *path, ParamList *pl) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    seek_to_params(f, path);
    for (int p = 0; p < pl->count; ++p)
        if (fread(pl->item[p].v, sizeof(float), pl->item[p].n, f) != (size_t)pl->item[p].n)
            die("%s: parameter size mismatch (wrong config?)", path);
    fclose(f);
}

/* Load the feature-normalization stats (per-input-feature mean/std) written by
 * checkpoint_save2/3. Returns ns, the number of stats read (== d_num when
 * present). A legacy TFW1 file has no stats block, so this returns 0 without
 * touching the caller's buffers — callers treat 0 as "no normalization stored".
 * Seek path: skip the 9-int header, read ns, then read mean[ns] and std[ns]. */
int checkpoint_load_stats(const char *path, float *mean, float *std) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    char magic[4];
    /* Only TFW2/TFW3 carry stats; anything else -> 0 (no stats available). */
    if (fread(magic, 1, 4, f) != 4 || (!is_magic(magic, MAGIC2) && !is_magic(magic, MAGIC3))) {
        fclose(f); return 0;
    }
    if (fseek(f, 9 * (long)sizeof(int), SEEK_CUR) != 0) { fclose(f); return 0; }
    int ns;
    if (fread(&ns, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (ns > 0) {
        /* mean[] and std[] are stored contiguously, means first then stds. */
        if (fread(mean, sizeof(float), ns, f) != (size_t)ns ||
            fread(std,  sizeof(float), ns, f) != (size_t)ns)
            die("%s: unexpected end of file", path);
    }
    fclose(f);
    return ns;
}

/* Load the coordinate (lat/lon) normalization stats, which only TFW3 carries.
 * Returns 1 if the coord block is present and read, else 0 (TFW1/TFW2, or a
 * TFW3 that was written with has_coord=0). To reach the coord block we must
 * first skip the feature-stats block, whose size is data-dependent: read its ns
 * and jump 2*ns floats, THEN read the has_coord flag and, if set, the 4 coord
 * floats (cmean[2] = lat/lon means, cstd[2] = lat/lon stds). */
int checkpoint_load_coord_stats(const char *path, float *cmean, float *cstd) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || !is_magic(magic, MAGIC3)) { fclose(f); return 0; }
    if (fseek(f, 9 * (long)sizeof(int), SEEK_CUR) != 0) { fclose(f); return 0; }
    int ns;
    if (fread(&ns, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    /* Skip the feature-stats payload (mean[ns] + std[ns]) to reach has_coord. */
    if (ns > 0 && fseek(f, 2L * ns * (long)sizeof(float), SEEK_CUR) != 0) { fclose(f); return 0; }
    int has_coord;
    if (fread(&has_coord, sizeof(int), 1, f) != 1 || !has_coord) { fclose(f); return 0; }
    if (fread(cmean, sizeof(float), 2, f) != 2 || fread(cstd, sizeof(float), 2, f) != 2)
        die("%s: unexpected end of file", path);
    fclose(f);
    return 1;
}

/* ---- optimizer-state sidecar (Adam moments) for resuming training --------
 *
 * Adam's per-parameter running moments (first moment fm, second moment sm) plus
 * the step counter t and learning rate are what let training RESUME exactly
 * where it left off. They are large (two floats per parameter) and useless for
 * inference, so they live in a SEPARATE ".opt" file rather than bloating the
 * weight checkpoint. The sidecar has its own magic, "TFO1".
 *
 * On-disk layout of the sidecar:
 *     "TFO1"           4 bytes
 *     n     int64      parameter count (must match the Adam being loaded)
 *     t     int64      Adam step counter (bias-correction timestep)
 *     epoch int32      training epoch to resume from
 *     lr    float32    learning rate at save time
 *     fm[n] float32    first-moment (mean of gradients) buffer
 *     sm[n] float32    second-moment (mean of squared gradients) buffer     */
#define OMAGIC "TFO1"

/* Serialize the Adam optimizer state. `n`/`t` are copied to `long` locals so
 * the on-disk width is fixed regardless of the Adam struct's field types. */
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

/* Restore Adam state into an already-allocated Adam `a`. Returns 1 on a clean
 * load, 0 on any mismatch or absence — the many early-return-0 paths make this
 * best-effort: a missing sidecar, a foreign magic, or an `n` that disagrees with
 * the current model's parameter count all cause training to simply start fresh
 * (moments left as-is) rather than abort. Only a truncated-but-otherwise-valid
 * moment payload is a hard error. Note fm/sm are read in place, so `a->fm`/`a->sm`
 * must already be sized to n. */
int checkpoint_load_optim(const char *path, Adam *a, int *epoch) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[4]; long n, t; int ep; float lr;
    if (fread(magic, 1, 4, f) != 4 || !is_magic(magic, OMAGIC)) { fclose(f); return 0; }
    /* Guard: the saved parameter count must equal this optimizer's — otherwise
     * the moment buffers would not correspond to the current parameters. */
    if (fread(&n, sizeof(long), 1, f) != 1 || n != a->n) { fclose(f); return 0; }
    if (fread(&t, sizeof(long), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&ep, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
    if (fread(&lr, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
    if (fread(a->fm, sizeof(float), (size_t)n, f) != (size_t)n ||
        fread(a->sm, sizeof(float), (size_t)n, f) != (size_t)n) { fclose(f); return 0; }
    /* Commit the scalar state only after the buffers loaded successfully. */
    a->t = t; a->lr = lr; if (epoch) *epoch = ep;
    fclose(f);
    return 1;
}
