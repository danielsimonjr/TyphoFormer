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

/* A Module is a manual C "vtable": a bundle of an opaque state pointer (`self`)
 * and function pointers that operate on it. This is how C does runtime
 * polymorphism — the caller invokes m.forward(m.self, ...) without knowing or
 * caring what concrete layer `self` points at. Each concrete layer supplies its
 * own three functions; the trio below is the entire contract every Module must
 * honor:
 *   forward (x -> y)   compute the layer output, [T,D] -> [T,D];
 *   backward(dy -> dx) given upstream grad dy, produce input grad dx AND
 *                      accumulate this layer's parameter grads (into the pl the
 *                      layer was built with) as a side effect;
 *   free(self)         release everything the state pointer owns.
 * `self` is "owned" — whoever adds the Module (e.g. Sequential) is responsible
 * for calling free() on teardown. */
typedef struct Module {
    void       *self;                                   /* owned layer state */
    const char *name;
    void      (*forward )(void *self, const Mat x, Mat y);
    void      (*backward)(void *self, const Mat dy, Mat dx);
    void      (*free    )(void *self);
} Module;

/* Adapter: wrap a transformer Block as a Module (heap-allocates the Block).
 * This is the canonical example of the seam — it binds the generic Module
 * function-pointer slots to the concrete block_forward/backward/free, so a Block
 * can be dropped into a Sequential (or any Module consumer) unchanged. */
Module module_block(int d_model, int n_heads, int ff_dim, int self_only,
                    ParamList *pl, const char *name);

/* Sequential: a chain of same-width (D) Modules, run in order on forward and in
 * reverse on backward (the standard chain-rule composition). Because every
 * Module is width-preserving ([T,D] -> [T,D]), the whole chain is too, and the
 * container needs only a couple of [T,D] scratch tensors to shuttle activations
 * between adjacent modules (see the ping-pong buffers below). */
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
