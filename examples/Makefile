NVCC ?= nvcc
NVCCFLAGS ?=

CUDA_SOURCES := gpu_skeleton.cu \
                hello_world.cu   \
                add_numbers.cu   \
                square_threads.cu \
                square_threads_mem.cu \
                grid_block_2d.cu \
                shared_tile_example.cu

BIN_DIR := bin
BINARIES := $(addprefix $(BIN_DIR)/, $(CUDA_SOURCES:.cu=))

.PHONY: all clean

all: $(BINARIES)

$(BIN_DIR)/%: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)
