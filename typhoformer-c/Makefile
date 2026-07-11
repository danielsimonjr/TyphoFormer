# TyphoFormer-C — pure C, standard library only.
CC      ?= gcc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDLIBS  := -lm

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
