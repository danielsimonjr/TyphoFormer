/*
 * module.c — Module adapters and the Sequential container.
 */
#include "module.h"

#include <assert.h>
#include <stdlib.h>

/* ---- Block adapter ---------------------------------------------------
 * These three thin trampolines are the glue that turns the concrete Block API
 * into the generic Module vtable signature. Each casts the opaque `void *self`
 * back to `Block *` and forwards to the real block_* function. block_del also
 * free()s the heap Block itself (block_free releases the Block's internals; the
 * outer free releases the malloc'd struct from module_block). */
static void block_fwd(void *s, const Mat x, Mat y) { block_forward((Block *)s, x, y); }
static void block_bwd(void *s, const Mat dy, Mat dx) { block_backward((Block *)s, dy, dx); }
static void block_del(void *s) { block_free((Block *)s); free(s); }

/* Construct a Block on the heap (so its address is stable for the lifetime of
 * the Module) and package it with the trampolines above into a Module value.
 * The Block registers its parameters into `pl` inside block_new, exactly as a
 * directly-instantiated Block would — going through the Module seam changes
 * nothing about parameter registration or gradients. */
Module module_block(int d_model, int n_heads, int ff_dim, int self_only,
                    ParamList *pl, const char *name) {
    Block *b = (Block *)malloc(sizeof(Block));
    *b = block_new(d_model, n_heads, ff_dim, self_only, pl, name);
    Module m = { b, name, block_fwd, block_bwd, block_del };
    return m;
}

/* ---- Sequential ------------------------------------------------------ */
/* Lazily (re)allocate a scratch Mat to shape [r,c], reusing it if it already
 * has that shape. Keeps the ping-pong buffers sized to [T,D] without allocating
 * on every forward/backward call. */
static void ensure(Mat *m, int r, int c) {
    if (m->data == NULL || m->rows != r || m->cols != c) { if (m->data) mat_free(m); *m = mat_new(r, c); }
}

/* Initialize an empty chain. No modules yet; the four scratch Mats (a/b for the
 * forward ping-pong, da/db for the backward ping-pong) start null and are
 * allocated on first use. */
void seq_init(Sequential *s, int T, int D) {
    s->mod = NULL; s->n = s->cap = 0; s->T = T; s->D = D;
    s->a = (Mat){0,0,NULL}; s->b = (Mat){0,0,NULL}; s->da = (Mat){0,0,NULL}; s->db = (Mat){0,0,NULL};
}
/* Append a Module, growing the backing array geometrically (4, 8, 16, ...) so
 * amortized insertion is O(1). Sequential takes ownership of the Module and
 * will free() it in seq_free. */
void seq_add(Sequential *s, Module m) {
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 4;
        s->mod = (Module *)realloc(s->mod, (size_t)s->cap * sizeof(Module)); assert(s->mod); }
    s->mod[s->n++] = m;
}
/* Forward pass through the chain, module 0 .. n-1. An empty chain is the
 * identity (copy x to y). Otherwise we ping-pong between the two scratch buffers
 * `a` and `b`: `cur` holds the current activation, `nxt` receives the module's
 * output, then we swap. The LAST module writes directly into the caller's `y`
 * (out = y) so the final result lands where the caller expects, no extra copy. */
void seq_forward(Sequential *s, const Mat x, Mat y) {
    if (s->n == 0) { mat_copy(y, x); return; }
    ensure(&s->a, s->T, s->D); ensure(&s->b, s->T, s->D);
    Mat cur = s->a, nxt = s->b, tmp;
    mat_copy(cur, x);
    for (int i = 0; i < s->n; ++i) {
        Mat out = (i == s->n - 1) ? y : nxt;      /* last module targets caller's y */
        s->mod[i].forward(s->mod[i].self, cur, out);
        if (i < s->n - 1) { tmp = cur; cur = out; nxt = tmp; }   /* swap cur<->nxt */
    }
}
/* Backward pass: the chain rule runs modules in REVERSE (n-1 .. 0), threading
 * the gradient backward. Same ping-pong idea with its own buffers da/db, and the
 * FIRST module (i==0) writes the final input-gradient straight into the caller's
 * `dx`. Each module's backward also accumulates its own parameter gradients as a
 * side effect, so after seq_backward the whole chain's grads are populated. */
void seq_backward(Sequential *s, const Mat dy, Mat dx) {
    if (s->n == 0) { mat_copy(dx, dy); return; }
    ensure(&s->da, s->T, s->D); ensure(&s->db, s->T, s->D);
    Mat cur = s->da, nxt = s->db, tmp;
    mat_copy(cur, dy);
    for (int i = s->n - 1; i >= 0; --i) {
        Mat out = (i == 0) ? dx : nxt;            /* module 0 targets caller's dx */
        s->mod[i].backward(s->mod[i].self, cur, out);
        if (i > 0) { tmp = cur; cur = out; nxt = tmp; }          /* swap cur<->nxt */
    }
}
/* Tear down: free each owned Module via its vtable free() slot, then the module
 * array and the four scratch buffers. */
void seq_free(Sequential *s) {
    for (int i = 0; i < s->n; ++i) s->mod[i].free(s->mod[i].self);
    free(s->mod);
    mat_free(&s->a); mat_free(&s->b); mat_free(&s->da); mat_free(&s->db);
}
