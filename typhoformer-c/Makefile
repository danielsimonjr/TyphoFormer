# TyphoFormer-C — pure C, standard library only.
CC      ?= gcc
OPT     ?= -O3
# -pthread: data-parallel multicore training (src/parallel.c). Portable; on
# platforms without pthreads, drop it and build --threads=1 only.
CFLAGS  := -std=c11 $(OPT) -Wall -Wextra -Wpedantic -Iinclude -pthread
LDLIBS  := -lm -pthread

# Opt-in native SIMD (non-portable binary; source stays pure C):  make NATIVE=1
ifeq ($(NATIVE),1)
CFLAGS  += -march=native -funroll-loops
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
else
TENSOR_OBJ := src/tensor.o
endif

SRC     := $(filter-out src/tensor.c,$(wildcard src/*.c))
OBJ     := $(SRC:.c=.o) $(TENSOR_OBJ)

TESTS   := tests/test_tensor tests/test_nn tests/test_model tests/test_data \
           tests/test_golden tests/test_module tests/test_parallel \
           tests/test_checkpoint tests/test_npy
BUILD   := $(filter-out src/train.o,$(OBJ))

.PHONY: all lib test test-san test-valgrind test-opencl clean

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

# OpenCL correctness cross-check: kernels vs a CPU reference (needs OPENCL=1).
backends/opencl/test_opencl: backends/opencl/test_opencl.c $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Verify the OpenCL backend: cross-check the kernels, then run the full-model
# gradient check *through* OpenCL. Must be invoked as `make OPENCL=1 test-opencl`.
test-opencl: backends/opencl/test_opencl tests/test_tensor tests/test_model
	@echo "== OpenCL kernel cross-check =="; ./backends/opencl/test_opencl || exit 1
	@echo "== tensor gradient check (through OpenCL) =="; ./tests/test_tensor || exit 1
	@echo "== full-model gradient check (through OpenCL) =="; ./tests/test_model || exit 1

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

backends/opencl/%.o: backends/opencl/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o backends/opencl/*.o backends/opencl/test_opencl \
	      $(TESTS) typhoformer libtyphoformer.a *.ckpt *.tfb
