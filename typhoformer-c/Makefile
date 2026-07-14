# TyphoFormer-C — pure C, standard library only.
CC      ?= gcc
OPT     ?= -O3
# -pthread: data-parallel multicore training (src/parallel.c). Portable; on
# platforms without pthreads, drop it and build --threads=1 only.
CFLAGS  := -std=c11 $(OPT) -Wall -Wextra -Wpedantic -Iinclude -pthread
LDLIBS  := -lm -pthread

# Auto-generated header dependencies (-MMD writes a .d beside each .o; -MP adds
# phony targets so a deleted header doesn't break the build). This makes editing
# a header — e.g. changing a struct's layout — recompile every dependent .o,
# instead of silently linking objects built against the old definition.
CFLAGS  += -MMD -MP

# Opt-in native SIMD (non-portable binary; source stays pure C):  make NATIVE=1
ifeq ($(NATIVE),1)
CFLAGS  += -march=native -funroll-loops
endif

# Portable SIMD tier between plain -O3 and NATIVE:  make MARCH=x86-64-v3
# (x86-64-v3 = AVX2+FMA, runs on any x86-64 CPU from ~2015 on; use any -march
# value your toolchain accepts). Ignored when NATIVE=1 is set.
ifneq ($(MARCH),)
ifneq ($(NATIVE),1)
CFLAGS  += -march=$(MARCH)
endif
endif

# Sanitizer build for tests:  make SAN=1 test
ifeq ($(SAN),1)
CFLAGS  += -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer
LDLIBS  += -fsanitize=address,undefined
endif

# OpenCL backend: swap src/tensor.c for the OpenCL implementation and link the
# ICD loader.  make OPENCL=1 [test-opencl].  Runs on any OpenCL device, incl. a
# CPU via POCL (pocl-opencl-icd).
ifeq ($(OPENCL),1)
LDLIBS  += -lOpenCL
TENSOR_OBJ := backends/opencl/tensor_opencl.o
else ifeq ($(CUDA),1)
# CUDA backend: same seam, swap in the .cu implementation and link cudart.
#   make CUDA=1 test-cuda        # needs the CUDA toolkit AND a real NVIDIA GPU
# CUDA_ARCH must match the device: sm_61 is Pascal (e.g. Quadro P1000). Note that
# CUDA 13.0 REMOVED Maxwell/Pascal/Volta, so sm_61 needs a 12.x toolkit.
NVCC      ?= nvcc
CUDA_ARCH ?= sm_61
CUDA_HOME ?= /usr/local/cuda
LDLIBS    += -L$(CUDA_HOME)/lib64 -lcudart
TENSOR_OBJ := backends/cuda/tensor_cuda.o
else
TENSOR_OBJ := src/tensor.o
endif

SRC     := $(filter-out src/tensor.c,$(wildcard src/*.c))
OBJ     := $(SRC:.c=.o) $(TENSOR_OBJ)

TESTS   := tests/test_tensor tests/test_nn tests/test_model tests/test_data \
           tests/test_golden tests/test_module tests/test_parallel \
           tests/test_checkpoint tests/test_npy tests/test_dropout tests/test_gru \
           tests/test_xattn
BUILD   := $(filter-out src/train.o,$(OBJ))

.PHONY: all lib test test-san test-valgrind test-opencl test-cuda clean

all: typhoformer

# Main training binary links all objects (including src/train.o's main).
typhoformer: $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Static library for embedding the model in another application (no main).
lib: libtyphoformer.a
libtyphoformer.a: $(BUILD)
	ar rcs $@ $^

# Unit tests link against the library objects (excluding the main).
tests/%: tests/%.c $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(TESTS)
	@for t in $(TESTS); do echo "== running $$t =="; ./$$t || exit 1; done

# Rebuild with AddressSanitizer + UBSan and run the tests.
test-san:
	$(MAKE) clean
	$(MAKE) SAN=1 test

# Run every test under valgrind (memcheck). Requires valgrind installed.
VALGRIND ?= valgrind --error-exitcode=1 --leak-check=full --errors-for-leak-kinds=definite
test-valgrind: $(TESTS)
	@for t in $(TESTS); do echo "== valgrind $$t =="; $(VALGRIND) ./$$t || exit 1; done

# Backend correctness cross-check: every kernel vs an inline CPU reference.
# The source (backends/test_backend.c) is backend-AGNOSTIC — it only speaks the
# tensor.h seam — so the SAME check verifies whichever backend is linked in.
backends/opencl/test_opencl: backends/test_backend.c $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

backends/cuda/test_cuda: backends/test_backend.c $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Verify the OpenCL backend: cross-check the kernels, then run the full-model
# gradient check *through* OpenCL. Must be invoked as `make OPENCL=1 test-opencl`.
test-opencl: backends/opencl/test_opencl tests/test_tensor tests/test_model
	@echo "== OpenCL kernel cross-check =="; ./backends/opencl/test_opencl || exit 1
	@echo "== tensor gradient check (through OpenCL) =="; ./tests/test_tensor || exit 1
	@echo "== full-model gradient check (through OpenCL) =="; ./tests/test_model || exit 1

# Verify the CUDA backend on a REAL GPU: same three gates as OpenCL, run through
# the .cu kernels. Must be invoked as `make CUDA=1 test-cuda`. CI only *compiles*
# the CUDA backend (no GPU on the runner); this target is what actually executes it.
test-cuda: backends/cuda/test_cuda tests/test_tensor tests/test_model
	@echo "== CUDA kernel cross-check =="; ./backends/cuda/test_cuda || exit 1
	@echo "== tensor gradient check (through CUDA) =="; ./tests/test_tensor || exit 1
	@echo "== full-model gradient check (through CUDA) =="; ./tests/test_model || exit 1

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

backends/opencl/%.o: backends/opencl/%.c
	$(CC) $(CFLAGS) -c $< -o $@

backends/cuda/%.o: backends/cuda/%.cu
	$(NVCC) -O3 -std=c++14 -arch=$(CUDA_ARCH) -Iinclude -c $< -o $@

clean:
	rm -f src/*.o src/*.d backends/opencl/*.o backends/opencl/*.d \
	      backends/opencl/test_opencl backends/cuda/*.o backends/cuda/test_cuda \
	      backends/cuda/libtyphoformer_cuda.a tests/*.d \
	      $(TESTS) typhoformer libtyphoformer.a *.ckpt *.tfb

# Pull in the auto-generated header dependencies (no-op on the first build).
-include $(OBJ:.o=.d) $(TESTS:=.d) $(TENSOR_OBJ:.o=.d)
