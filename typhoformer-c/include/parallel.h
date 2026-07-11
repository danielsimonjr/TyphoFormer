/*
 * parallel.h — data-parallel minibatch training across CPU cores (pthreads).
 *
 * The per-sample model operates on [T, D] tensors and its layers keep private
 * scratch/cache buffers, so a single Model is NOT safe to call from two threads
 * at once. Instead of threading one model, we replicate: N worker replicas, each
 * a full Model with identical *parameter values* but its own caches and its own
 * gradient buffers. A minibatch is split across the replicas; every worker runs
 * forward+backward on its shard; the gradients are then summed into a master
 * ParamList and the optimizer takes a single step. This is textbook synchronous
 * data-parallel SGD.
 *
 * Numerical note: the reduced gradient equals the serial gradient up to
 * floating-point summation order (the batch is summed in a different order), so
 * results match to ~1e-5 relative, not bit-for-bit. tests/test_parallel.c
 * pins this equivalence. With --threads=1 the driver uses the original serial
 * path, so the golden regression stays byte-identical.
 */
#ifndef TYPHOFORMER_PARALLEL_H
#define TYPHOFORMER_PARALLEL_H

#include "data.h"
#include "model.h"

typedef struct ParTrainer ParTrainer;

/* Build `n_workers` replicas from cfg. Replicas are zero-initialised; call
 * partrainer_broadcast() before the first step to load the master's weights. */
ParTrainer *partrainer_new(const Config *cfg, int n_workers);
void        partrainer_free(ParTrainer *pt);

int  partrainer_workers(const ParTrainer *pt);

/* Copy master parameter values into every replica (call once per optimizer
 * step, before dispatching the batch). */
void partrainer_broadcast(ParTrainer *pt, const ParamList *master);

/* Run one minibatch: samples idx[b .. b+bs) are distributed across the workers,
 * each accumulating raw gradients into its replica. The per-sample gradients are
 * summed and scaled by 1/bs into `master->g` (master is zeroed internally).
 * Returns the mean batch loss. `master` must have the same layout as the
 * replicas (same Config) — typically the ParamList of your master Model. */
double partrainer_step_grads(ParTrainer *pt, ParamList *master,
                             const Dataset *ds, const int *idx,
                             int b, int bs, float lambda);

#endif /* TYPHOFORMER_PARALLEL_H */
