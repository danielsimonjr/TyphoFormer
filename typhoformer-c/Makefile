# TyphoFormer-C — pure C, standard library only.
CC      ?= gcc
OPT     ?= -O3
CFLAGS  := -std=c11 $(OPT) -Wall -Wextra -Wpedantic -Iinclude
LDLIBS  := -lm

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

TESTS   := tests/test_tensor tests/test_nn tests/test_model tests/test_data
BUILD   := $(filter-out src/train.o,$(OBJ))

.PHONY: all test clean

all: typhoformer

# Main training binary links all objects (including src/train.o's main).
typhoformer: $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

# Unit tests link against the library objects (excluding the main).
tests/%: tests/%.c $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(TESTS)
	@for t in $(TESTS); do echo "== running $$t =="; ./$$t || exit 1; done

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TESTS) typhoformer typhoformer_weights.bin
