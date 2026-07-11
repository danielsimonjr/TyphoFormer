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

SRC     := $(wildcard src/*.c)
OBJ     := $(SRC:.c=.o)

TESTS   := tests/test_tensor tests/test_nn tests/test_model tests/test_data \
           tests/test_golden tests/test_module tests/test_parallel \
           tests/test_checkpoint tests/test_npy
BUILD   := $(filter-out src/train.o,$(OBJ))

.PHONY: all lib test test-san test-valgrind clean

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

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TESTS) typhoformer libtyphoformer.a *.ckpt *.tfb
