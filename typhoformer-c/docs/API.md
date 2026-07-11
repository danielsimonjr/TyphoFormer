# API Reference & Memory Model

Everything you need to call, own, and reason about the code from another C
program. Read the **Memory model**, **Ownership & lifetime**, and
**Concurrency** sections first — they govern every call below.

---

## Memory model

- **`Mat` is a value type** holding a pointer:
  ```c
  typedef struct { int rows, cols; float *data; } Mat;
  ```
  Copying a `Mat` copies the *handle*, not the buffer — two `Mat`s can alias the
  same `data`. Data is a flat, **row-major** `float` array of `rows*cols`
  elements: element `(i,j)` is `m.data[i*m.cols + j]`.

- **Allocation.** `mat_new(r,c)` calloc's (zero-initialised); `mat_free(&m)`
  frees and nulls. Modules allocate their own parameter, cache, and scratch
  `Mat`s in their `*_new` constructor and release them in `*_free`.

- **The `ensure(&m, r, c)` pattern** (internal): grow a cached/scratch `Mat` to a
  shape lazily, reallocating only when the shape changes. Because all shapes are
  fixed after the first call, steady-state training does **no** allocation.

- **Persistent scratch.** Every module keeps its backward temporaries as fields
  (`s_*`) and reuses them across calls. Consequence: a module's buffers are
  overwritten by each `*_forward`/`*_backward`; do not hold pointers into them
  across calls.

- **Caches couple forward→backward.** `*_forward` stores what `*_backward` needs
  (inputs, activations, softmax weights). You must call `forward` before
  `backward`, and you must not run a second `forward` in between if you still
  intend to `backward` the first (finite-difference gradient checks re-run
  `forward` only *after* the analytic backward, which is why they are safe).

---

## Ownership & lifetime

| Rule | Detail |
|:--|:--|
| Constructor owns | `*_new` allocates all internal buffers and **registers parameters** into the passed `ParamList`. |
| Destructor frees | call `*_free` exactly once; it frees internals. Free in reverse dependency order isn't required (modules are independent). |
| `ParamList` does **not** own parameter memory | it stores *pointers* into module buffers. Free the `ParamList` (`plist_free`) to release its index array, and free the modules to release the actual weights. Never free a module while its params are still being stepped by an optimizer. |
| Caller owns I/O `Mat`s | you allocate the `Mat`s you pass to `*_forward`/`*_backward` (inputs, outputs, upstream/downstream grads) and free them yourself. |
| Datasets | `dataset_*` returns a `Dataset` by value; free with `dataset_free`. |

---

## Concurrency

- **Not thread-safe at the instance level.** A `Model` (and every sub-module)
  holds mutable caches and scratch, so a single instance may be used by **one
  thread at a time**.
- **Independent instances are independent.** Two separate `Model`s with separate
  `ParamList`s can run on two threads — *except* they share one process-global
  RNG (`nn_seed`/`nn_uniform`), which is only touched during construction and
  shuffling. Seed and build models on one thread, then run them on their own
  threads.
- **Multicore training uses exactly this pattern.** `ParTrainer`
  ([`parallel.h`](../include/parallel.h)) owns N replica `Model`s, hands each
  worker its own instance and scratch, and only touches the shared master
  `ParamList` *before* dispatch (broadcast) and *after* join (reduce) — never
  concurrently. So there are no locks and no data races (ThreadSanitizer-clean).
  Because the RNG is only used at construction/shuffle time on the main thread,
  the parallel region touches it not at all.
- **Determinism.** With a fixed `nn_seed`, initialization, shuffling, and the
  whole run are reproducible. There is no wall-clock or `rand()` dependence.
  Multicore reduction reorders floating-point sums, so `--threads>1` matches the
  serial result to ≈1e-7 rather than bit-for-bit; `--threads=1` is exact.
- **Fatal errors.** Unrecoverable I/O/format problems call `die()` (prints to
  `stderr`, `exit(1)`). Library users who cannot tolerate `exit` should validate
  inputs (file existence, header magic) before calling the loaders.

---

## `tensor.h` — matrix kernels

```c
_Noreturn void die(const char *fmt, ...);         // printf-style fatal error

Mat  mat_new(int rows, int cols);                 // zero-initialised
void mat_free(Mat *m);
void mat_zero(Mat m);
void mat_copy(Mat dst, const Mat src);            // shapes must match

void mat_matmul   (const Mat A, const Mat B, Mat out); // out = A  B      [m,k][k,n]→[m,n]
void mat_matmul_bt(const Mat A, const Mat B, Mat out); // out = A  Bᵀ     [m,k][n,k]→[m,n]
void mat_matmul_atb(const Mat A, const Mat B, Mat out);// out = Aᵀ B      [k,m][k,n]→[m,n]

void mat_add_bias(Mat m, const float *bias);      // m[i,:] += bias
void mat_colsum  (const Mat m, float *out);       // out[j] = Σ_i m[i,j]
void mat_relu(Mat m);   void mat_sigmoid(Mat m);  // in place
void mat_scale(Mat m, float s);                   // m *= s
void mat_axpy(Mat y, float a, const Mat x);       // y += a·x
```

The three matmul variants are chosen so a linear layer's forward and both
gradients need no transposes (see [ARCHITECTURE.md](ARCHITECTURE.md) §3.1).

---

## `nn.h` — generic layers

### RNG
```c
void  nn_seed(unsigned long s);          // seed the process-global xorshift RNG
float nn_uniform(float lo, float hi);    // U[lo,hi)
```

### ParamList — the optimizer/checkpoint bridge
```c
typedef struct { float *v; float *g; int n; const char *name; } Param;
typedef struct { Param *item; int count, cap; } ParamList;

void plist_init(ParamList *pl);
void plist_add (ParamList *pl, float *v, float *g, int n, const char *name);
void plist_zero_grad(ParamList *pl);     // memset every .g to 0
long plist_num_params(const ParamList *pl);
void plist_free(ParamList *pl);
```
Each `Param` points at a value buffer `v` and its gradient buffer `g` living
inside a module. This is what lets the optimizer and the checkpoint I/O be
architecture-agnostic: they just walk the list. **The registration order (the
order modules are constructed in `model_new`) defines the checkpoint layout** —
see [INTEGRATION.md](INTEGRATION.md).

### Layers (each: `*_new` registers params; `*_forward`; `*_backward` accumulates grads; `*_free`)
```c
Linear    linear_new(int in, int out, ParamList *pl, const char *name);
void      linear_forward (Linear *l, const Mat x, Mat y);       // caches x
void      linear_backward(Linear *l, const Mat dy, Mat dx);     // dx may be {0,0,NULL}

LayerNorm layernorm_new(int dim, ParamList *pl, const char *name);
void      layernorm_forward (LayerNorm *ln, const Mat x, Mat y);
void      layernorm_backward(LayerNorm *ln, const Mat dy, Mat dx);

FFN       ffn_new(int d, int f, ParamList *pl, const char *name);   // D→F→D
void      ffn_forward (FFN *ff, const Mat x, Mat y);
void      ffn_backward(FFN *ff, const Mat dy, Mat dx);

MHA       mha_new(int d_model, int n_heads, int self_only, ParamList *pl, const char *name);
void      mha_forward (MHA *m, const Mat x, Mat y);   // self-attention over rows of x
void      mha_backward(MHA *m, const Mat dy, Mat dx);

Block     block_new(int d_model, int n_heads, int ff_dim, int self_only, ParamList *pl, const char *name);
void      block_forward (Block *b, const Mat x, Mat y);
void      block_backward(Block *b, const Mat dy, Mat dx);
```
Passing `dx = (Mat){0,0,NULL}` to `linear_backward` skips the input-gradient
computation (used when the input is data, e.g. inside PGF).

---

## `model.h` — the model

```c
typedef struct { int d_num, d_text, d_model, out_dim,
                     in_len, pred_len, d_ff, n_heads, n_layers; } Config;
Config config_default(void);   // the paper config (d_model=256, d_ff=1024, 3 layers, 4 heads)

Model model_new(const Config *c, ParamList *pl);
void  model_forward (Model *m, const Mat xnum, const Mat xtext, const Mat yprev); // fills m->pred, m->pgf.gate
void  model_backward(Model *m, const Mat dpred, const Mat dgate_pen);
void  model_free(Model *m);

// L = MSE(pred,Y) + lambda·mean(relu(0.6−gate)²).
// If dpred / dgate_pen are non-NULL Mats they receive the upstream gradients.
double model_loss(const Mat pred, const Mat Y, const Mat gate, float lambda,
                  Mat dpred, Mat dgate_pen);
```
After `model_forward`, read predictions from `m->pred` `[pred_len,2]` and the
gate from `m->pgf.gate` `[T,d_model]`. The sub-modules (`PGF`, `Encoder`,
`Decoder`, `TimeMix`) have the same `*_new/forward/backward/free` shape and can
be used standalone.

---

## `data.h` — datasets

```c
typedef struct { int n_records, d_num, d_text, in_len, pred_len;
                 float *num, *emb, *lat, *lon; int *gid;
                 float mean[64], std[64]; int *start, n_samples;
                 int prewindowed; float *win_in, *win_tg; } Dataset;

Dataset dataset_load    (const char *csv, const char *embdir, int in_len, int pred_len);
Dataset dataset_load_bin(const char *path);                 // pre-windowed .tfb
void    dataset_get(const Dataset *d, int s, Mat xnum, Mat xtext, Mat yprev, Mat Y);
void    dataset_split(const Dataset *d, float val_frac, unsigned long seed,
                      int **train, int *n_train, int **val, int *n_val);
void    dataset_free(Dataset *d);
float  *npy_load_2d(const char *path, int *rows, int *cols);  // malloc'd; caller frees
```
`dataset_get` fills caller-allocated `Mat`s (`xnum[T,14]`, `xtext[T,384]`,
`yprev[1,2]`, `Y[pred_len,2]`). The index arrays from `dataset_split` are
`malloc`'d — the caller frees them.

---

## `optim.h` — Adam

```c
typedef struct { float lr,b1,b2,eps,wd; float *fm,*sm; long n,t; } Adam;

Adam adam_new(const ParamList *pl, float lr, float wd);
void adam_step(Adam *a, ParamList *pl);   // one update over all registered params
void adam_free(Adam *a);
```
`adam_new` sizes its moment buffers from `plist_num_params`. Zero grads with
`plist_zero_grad` at the start of each batch, accumulate over the batch, then
`adam_step`.

---

## Minimal call sequence

```c
nn_seed(1234);
ParamList pl; plist_init(&pl);
Config c = config_default();
Model m = model_new(&c, &pl);
Adam opt = adam_new(&pl, 1e-3f, 1e-5f);

Mat dpred = mat_new(c.pred_len, 2), dgate = mat_new(c.in_len, c.d_model);
// per batch:
plist_zero_grad(&pl);
for (each sample) {
    model_forward(&m, xnum, xtext, yprev);
    model_loss(m.pred, Y, m.pgf.gate, 0.1f, dpred, dgate);
    // scale dpred,dgate by 1/batch for the mean
    model_backward(&m, dpred, dgate);
}
adam_step(&opt, &pl);
// teardown: adam_free(&opt); model_free(&m); plist_free(&pl);
```
