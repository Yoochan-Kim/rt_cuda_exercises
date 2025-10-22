NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -lineinfo
INCLUDE_DIR := include
BIN_DIR := bin

# SKIP_CPU flag (0 or 1), default to 0 (CPU validation enabled)
SKIP_CPU ?= 0
ifeq ($(SKIP_CPU),1)
NVCCFLAGS += -DSKIP_CPU
endif

EXERCISE_DIRS := $(wildcard exercises/stage*)
ANSWER_DIRS := $(wildcard answers/stage*)
STAGES := $(sort $(patsubst exercises/stage%,%,$(EXERCISE_DIRS)) $(patsubst answers/stage%,%,$(ANSWER_DIRS)))

EXERCISE_BINS := $(addprefix $(BIN_DIR)/stage,$(addsuffix _exercise,$(STAGES)))
ANSWER_BINS := $(addprefix $(BIN_DIR)/stage,$(addsuffix _answer,$(STAGES)))

# Define phony targets explicitly
STAGE_TARGETS := $(addprefix stage,$(STAGES))
EXERCISE_TARGETS := $(addprefix stage,$(addsuffix _exercise,$(STAGES)))
ANSWER_TARGETS := $(addprefix stage,$(addsuffix _answer,$(STAGES)))

.PHONY: all clean answers $(STAGE_TARGETS) $(EXERCISE_TARGETS) $(ANSWER_TARGETS)

all: $(EXERCISE_BINS) $(ANSWER_BINS)

answers: $(ANSWER_BINS)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

# Pattern rules for building binaries
$(BIN_DIR)/stage%_exercise: exercises/stage%/main.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) $< -o $@

$(BIN_DIR)/stage%_answer: answers/stage%/main.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCLUDE_DIR) $< -o $@

# Generate explicit rules for each stage using eval
define STAGE_RULES
stage$(1)_exercise: $(BIN_DIR)/stage$(1)_exercise
stage$(1)_answer: $(BIN_DIR)/stage$(1)_answer
stage$(1): stage$(1)_exercise stage$(1)_answer
endef

$(foreach stage,$(STAGES),$(eval $(call STAGE_RULES,$(stage))))

clean:
	@rm -f $(EXERCISE_BINS) $(ANSWER_BINS)
