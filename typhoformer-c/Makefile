# TyphoFormer-C — pure C, standard library only.
CC      ?= gcc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDLIBS  := -lm

SRC     := $(wildcard src/*.c)
OBJ     := $(SRC:.c=.o)

TESTS   := tests/test_tensor tests/test_nn tests/test_model
BUILD   := $(filter-out src/main.o,$(OBJ))

.PHONY: all test clean

all: $(OBJ)

# Unit tests link against the library objects (excluding main.o).
tests/%: tests/%.c $(BUILD)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

test: $(TESTS)
	@for t in $(TESTS); do echo "== running $$t =="; ./$$t || exit 1; done

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TESTS)
