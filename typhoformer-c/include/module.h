/*
 * module.h — a minimal polymorphic "Module" interface and a Sequential
 * container, so new encoders/decoders/blocks can be plugged in without editing
 * the core model. A Module is a vtable over any layer with a uniform
 * [*, D] -> [*, D] shape (the transformer-block convention).
 *
 * The core Model uses concrete types (for clarity and zero indirection); this
 * interface is the *extension seam*. Anything you build behind it is
 * gradient-checkable exactly like the built-in layers (see tests/test_module.c).
 */
#ifndef TYPHOFORMER_MODULE_H
#define TYPHOFORMER_MODULE_H

#include "nn.h"

typedef struct Module {
    void       *self;                                   /* owned layer state */
    const char *name;
    void      (*forward )(void *self, const Mat x, Mat y);
    void      (*backward)(void *self, const Mat dy, Mat dx);
    void      (*free    )(void *self);
} Module;

/* Adapter: wrap a transformer Block as a Module (heap-allocates the Block). */
Module module_block(int d_model, int n_heads, int ff_dim, int self_only,
                    ParamList *pl, const char *name);

/* Sequential: a chain of same-width (D) Modules. */
typedef struct {
    Module *mod; int n, cap;
    int     T, D;                                       /* work-buffer shape */
    Mat     a, b, da, db;                               /* ping-pong buffers */
} Sequential;

void seq_init(Sequential *s, int T, int D);
void seq_add(Sequential *s, Module m);
void seq_forward (Sequential *s, const Mat x, Mat y);
void seq_backward(Sequential *s, const Mat dy, Mat dx);
void seq_free(Sequential *s);

#endif /* TYPHOFORMER_MODULE_H */
