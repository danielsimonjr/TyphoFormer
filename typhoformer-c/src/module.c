/*
 * module.c — Module adapters and the Sequential container.
 */
#include "module.h"

#include <assert.h>
#include <stdlib.h>

/* ---- Block adapter --------------------------------------------------- */
static void block_fwd(void *s, const Mat x, Mat y) { block_forward((Block *)s, x, y); }
static void block_bwd(void *s, const Mat dy, Mat dx) { block_backward((Block *)s, dy, dx); }
static void block_del(void *s) { block_free((Block *)s); free(s); }

Module module_block(int d_model, int n_heads, int ff_dim, int self_only,
                    ParamList *pl, const char *name) {
    Block *b = (Block *)malloc(sizeof(Block));
    *b = block_new(d_model, n_heads, ff_dim, self_only, pl, name);
    Module m = { b, name, block_fwd, block_bwd, block_del };
    return m;
}

/* ---- Sequential ------------------------------------------------------ */
static void ensure(Mat *m, int r, int c) {
    if (m->data == NULL || m->rows != r || m->cols != c) { if (m->data) mat_free(m); *m = mat_new(r, c); }
}

void seq_init(Sequential *s, int T, int D) {
    s->mod = NULL; s->n = s->cap = 0; s->T = T; s->D = D;
    s->a = (Mat){0,0,NULL}; s->b = (Mat){0,0,NULL}; s->da = (Mat){0,0,NULL}; s->db = (Mat){0,0,NULL};
}
void seq_add(Sequential *s, Module m) {
    if (s->n == s->cap) { s->cap = s->cap ? s->cap * 2 : 4;
        s->mod = (Module *)realloc(s->mod, (size_t)s->cap * sizeof(Module)); assert(s->mod); }
    s->mod[s->n++] = m;
}
void seq_forward(Sequential *s, const Mat x, Mat y) {
    if (s->n == 0) { mat_copy(y, x); return; }
    ensure(&s->a, s->T, s->D); ensure(&s->b, s->T, s->D);
    Mat cur = s->a, nxt = s->b, tmp;
    mat_copy(cur, x);
    for (int i = 0; i < s->n; ++i) {
        Mat out = (i == s->n - 1) ? y : nxt;
        s->mod[i].forward(s->mod[i].self, cur, out);
        if (i < s->n - 1) { tmp = cur; cur = out; nxt = tmp; }
    }
}
void seq_backward(Sequential *s, const Mat dy, Mat dx) {
    if (s->n == 0) { mat_copy(dx, dy); return; }
    ensure(&s->da, s->T, s->D); ensure(&s->db, s->T, s->D);
    Mat cur = s->da, nxt = s->db, tmp;
    mat_copy(cur, dy);
    for (int i = s->n - 1; i >= 0; --i) {
        Mat out = (i == 0) ? dx : nxt;
        s->mod[i].backward(s->mod[i].self, cur, out);
        if (i > 0) { tmp = cur; cur = out; nxt = tmp; }
    }
}
void seq_free(Sequential *s) {
    for (int i = 0; i < s->n; ++i) s->mod[i].free(s->mod[i].self);
    free(s->mod);
    mat_free(&s->a); mat_free(&s->b); mat_free(&s->da); mat_free(&s->db);
}
