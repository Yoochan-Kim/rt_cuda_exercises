NVCC ?= nvcc
NVCCFLAGS ?=

CUDA_SOURCES := warp_indexing_1d.cu \
                warp_indexing_2d.cu \
                divergent_kernel.cu \
                work_splitting_kernel.cu \
                stream_compaction.cu \
                memory_patterns.cu \
                data_based_divergence_kernel.cu \
                index_based_divergence_kernel.cu \
                early_exit_kernel.cu

BIN_DIR := bin
BINARIES := $(addprefix $(BIN_DIR)/, $(CUDA_SOURCES:.cu=))

.PHONY: all clean

all: $(BINARIES)

$(BIN_DIR)/%: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(BIN_DIR)
