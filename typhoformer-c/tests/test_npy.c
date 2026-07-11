/*
 * test_npy.c — exercise two data-layer primitives independent of the real
 * dataset files:
 *   1. npy_load_2d() against a hand-written .npy v1.0 file (round-trips shape
 *      and values, including a non-64-aligned header that forces padding).
 *   2. dataset_split() determinism: same seed -> same split; train/val are
 *      disjoint and together cover every sample exactly once.
 */
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
/* Returns 1 if npy_load_2d accepts the file. Runs in a child because a rejected
 * file calls die()/exit(1); the child's stderr is silenced. */
static int loads_ok(const char *path) {
    pid_t pid = fork();
    if (pid == 0) {
        FILE *n = freopen("/dev/null", "w", stderr); (void)n;
        int rr, cc; float *d = npy_load_2d(path, &rr, &cc); free(d); _exit(0);
    }
    int st; waitpid(pid, &st, 0);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0;
}
#endif

#define NPY "._test_arr.tmp.npy"

/* Write a minimal v1.0 .npy with a caller-chosen descr/fortran flag. */
static void write_npy_hdr(const char *path, const char *descr, const char *fortran,
                          int r, int c, const float *data, int nbytes_elem) {
    char dict[128];
    snprintf(dict, sizeof dict,
             "{'descr': '%s', 'fortran_order': %s, 'shape': (%d, %d), }", descr, fortran, r, c);
    int base = 10 + (int)strlen(dict) + 1;
    int pad = (64 - (base % 64)) % 64;
    int hlen = (int)strlen(dict) + 1 + pad;
    FILE *f = fopen(path, "wb");
    if (!f) { die("cannot write %s", path); }
    unsigned char pre[10] = {0x93, 'N','U','M','P','Y', 1, 0,
                             (unsigned char)(hlen & 0xff), (unsigned char)(hlen >> 8)};
    fwrite(pre, 1, 10, f);
    fwrite(dict, 1, strlen(dict), f);
    for (int i = 0; i < pad; ++i) fputc(' ', f);
    fputc('\n', f);
    if (data) fwrite(data, (size_t)nbytes_elem, (size_t)r * c, f);
    fclose(f);
}

/* Write a minimal float32 v1.0 .npy with the given shape and row-major data. */
static void write_npy(const char *path, int r, int c, const float *data) {
    char dict[128];
    snprintf(dict, sizeof dict,
             "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", r, c);
    /* total header (10 preamble bytes + dict + newline) must be a multiple of 64 */
    int base = 10 + (int)strlen(dict) + 1;
    int pad = (64 - (base % 64)) % 64;
    int hlen = (int)strlen(dict) + 1 + pad;
    FILE *f = fopen(path, "wb");
    if (!f) { die("cannot write %s", path); }
    unsigned char pre[10] = {0x93, 'N','U','M','P','Y', 1, 0,
                             (unsigned char)(hlen & 0xff), (unsigned char)(hlen >> 8)};
    fwrite(pre, 1, 10, f);
    fwrite(dict, 1, strlen(dict), f);
    for (int i = 0; i < pad; ++i) fputc(' ', f);
    fputc('\n', f);
    fwrite(data, sizeof(float), (size_t)r * c, f);
    fclose(f);
}

int main(void) {
    int fail = 0;

    /* ---- 1. .npy round-trip ---- */
    const int R = 3, C = 4;
    float src[12];
    for (int i = 0; i < R * C; ++i) src[i] = (float)i * 0.25f - 1.0f;
    write_npy(NPY, R, C, src);
    int rr = 0, cc = 0;
    float *got = npy_load_2d(NPY, &rr, &cc);
    if (rr != R || cc != C) { printf("FAIL: shape %dx%d != %dx%d\n", rr, cc, R, C); fail = 1; }
    for (int i = 0; i < R * C && !fail; ++i)
        if (got[i] != src[i]) { printf("FAIL: npy value[%d] %g != %g\n", i, got[i], src[i]); fail = 1; }
    free(got);
    remove(NPY);

#ifndef _WIN32
    /* ---- 1b. dtype / fortran_order validation (must be rejected) ---- */
    write_npy_hdr(NPY, "<f4", "False", R, C, src, 4);
    if (!loads_ok(NPY)) { printf("FAIL: valid <f4 npy rejected\n"); fail = 1; }
    write_npy_hdr(NPY, ">f4", "False", R, C, NULL, 4);   /* big-endian */
    if (loads_ok(NPY))  { printf("FAIL: big-endian >f4 accepted\n"); fail = 1; }
    write_npy_hdr(NPY, "<f8", "False", R, C, NULL, 8);   /* float64 */
    if (loads_ok(NPY))  { printf("FAIL: float64 <f8 accepted\n"); fail = 1; }
    write_npy_hdr(NPY, "<f4", "True", R, C, NULL, 4);    /* fortran order */
    if (loads_ok(NPY))  { printf("FAIL: fortran_order=True accepted\n"); fail = 1; }
    remove(NPY);
#endif

    /* ---- 2. split determinism + partition ---- */
    Dataset d; memset(&d, 0, sizeof d);
    d.n_samples = 100;    /* only n_samples is needed by dataset_split */
    int *tr1, *va1, *tr2, *va2, ntr1, nva1, ntr2, nva2;
    dataset_split(&d, 0.2f, 99, &tr1, &ntr1, &va1, &nva1);
    dataset_split(&d, 0.2f, 99, &tr2, &ntr2, &va2, &nva2);
    if (ntr1 != 80 || nva1 != 20) { printf("FAIL: split sizes %d/%d\n", ntr1, nva1); fail = 1; }
    if (ntr1 != ntr2 || nva1 != nva2) { printf("FAIL: split size not deterministic\n"); fail = 1; }
    for (int i = 0; i < ntr1 && !fail; ++i) if (tr1[i] != tr2[i]) { printf("FAIL: train not deterministic\n"); fail = 1; }
    for (int i = 0; i < nva1 && !fail; ++i) if (va1[i] != va2[i]) { printf("FAIL: val not deterministic\n"); fail = 1; }

    /* every index appears exactly once across train+val */
    int *seen = (int *)calloc(d.n_samples, sizeof(int));
    for (int i = 0; i < ntr1; ++i) if (tr1[i] >= 0 && tr1[i] < d.n_samples) seen[tr1[i]]++;
    for (int i = 0; i < nva1; ++i) if (va1[i] >= 0 && va1[i] < d.n_samples) seen[va1[i]]++;
    for (int i = 0; i < d.n_samples && !fail; ++i)
        if (seen[i] != 1) { printf("FAIL: index %d covered %d times\n", i, seen[i]); fail = 1; }
    free(seen); free(tr1); free(va1); free(tr2); free(va2);

    printf(fail ? "FAIL: npy/split tests\n" : "npy round-trip + split determinism check passed\n");
    return fail;
}
