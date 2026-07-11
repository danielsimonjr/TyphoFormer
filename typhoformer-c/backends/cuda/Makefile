# Build the CUDA reference backend into a static library.
# Requires the CUDA toolkit (nvcc). Not invoked by the top-level Makefile.
#
#   make            # -> libtyphoformer_cuda.a
#   make clean
#
# To build the whole model on GPU, link this archive in place of tensor.o:
#   nvcc <other .o files> libtyphoformer_cuda.a -o typhoformer-gpu

NVCC    ?= nvcc
NVFLAGS ?= -O3 -std=c++14 -I../../include
AR      ?= ar

all: libtyphoformer_cuda.a

tensor_cuda.o: tensor_cuda.cu ../../include/tensor.h
	$(NVCC) $(NVFLAGS) -c $< -o $@

libtyphoformer_cuda.a: tensor_cuda.o
	$(AR) rcs $@ $^

clean:
	rm -f tensor_cuda.o libtyphoformer_cuda.a

.PHONY: all clean
