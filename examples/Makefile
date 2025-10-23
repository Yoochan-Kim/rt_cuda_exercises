NVCC ?= nvcc
NVCCFLAGS ?=

CUDA_SOURCES := gpu_skeleton.cu \
                hello_world.cu   \
                add_numbers.cu   \
                square_threads.cu \
                square_threads_mem.cu \
                grid_block_2d.cu \
                shared_tile_example.cu

BINARIES := $(CUDA_SOURCES:.cu=)

.PHONY: all clean

all: $(BINARIES)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(BINARIES)
