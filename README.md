# CUDA Stduy

A progressive CUDA programming tutorial with hands-on exercises and automated verification.

## Repository Structure

```
cuda_study/
├── include/               # Shared header files
│   ├── cuda_utils.cuh     # CUDA error checking utilities
│   ├── matrix.h           # Matrix structure definitions
│   └── matrix_with_stride.h  # Matrix with stride support
├── exercises/             # Exercise implementations (TODO sections for students)
│   ├── stage00/           # Hello World
│   ├── stage01/           # Memory allocation & kernel launch
│   ├── stage02/           # Vector addition
│   ├── stage03/           # Matrix addition (single block)
│   ├── stage04/           # Matrix addition (multiple blocks)
│   ├── stage05/           # Matrix multiplication (naive)
│   └── stage06/           # Matrix multiplication (shared memory)
├── answers/               # Complete reference implementations
├── tools/                 # Verification and build utilities
│   └── stages/            # Stage configurations
├── bin/                   # Compiled binaries (generated)
├── references/            # Golden reference outputs (generated, git-ignored)
├── Makefile               # Build system
├── run.py                 # Verification runner
└── generate_reference.py  # Golden reference generator
```

## Prerequisites

- NVIDIA CUDA Toolkit (nvcc compiler)
- Python 3.x
- CUDA-capable GPU (for running exercises)
- numpy (for generating golden references)

## Usage

### Basic Verification

Verify all exercises:

```bash
python run.py
```

Verify a specific stage:

```bash
python run.py --stage 00
```

Verify answer implementation:

```bash
python run.py --answer --stage 01
```

Show verbose output (GPU stream output):

```bash
python run.py --verbose
```

### SKIP_CPU Mode

For stages with CPU validation (stage 02-06), you can skip CPU computation and directly compare GPU output with pre-generated golden references:

1. **Generate golden references** (one-time setup):
   ```bash
   python generate_reference.py
   ```
   This creates reference files in `references/` directory (git-ignored).

2. **Run verification in SKIP_CPU mode**:
   ```bash
   python run.py --answer --skip_cpu
   ```

**Note**:
- Stage 00 and 01 don't have CPU validation, so `--skip_cpu` has no effect
- Golden references are generated using numpy and stored as pickled Python lists

### Learning Path

1. **Stage 00**: Hello World - Basic CUDA kernel launch
2. **Stage 01**: Vector Addition - 1D thread indexing
3. **Stage 02**: Vector Addition - 2D grid and block indexing
4. **Stage 03**: Matrix Addition - 2D indexing without blocks
5. **Stage 04**: Matrix Addition - Block-based 2D computation
6. **Stage 05**: Matrix Multiplication - Naive implementation with global memory
7. **Stage 06**: Matrix Multiplication - Shared memory tiling with __syncthreads()

## Development Workflow

1. Read the document in `docs/stageXX.md` and TODO comments in `exercises/stageXX/todo.cu`
2. Implement the required functionality
3. Run `python run.py` (automatically builds and verifies all stages)
   - Optional: `python run.py --stage XX` to verify a specific stage
   - Optional: `python run.py --answer --stage XX` to verify answer implementation
4. Compare with reference implementation in `answers/stageXX/todo.cu` if needed

## Notes

- Matrix struct uses `stride` field (stage06+) for sub-matrix support
- All timing measurements exclude data transfer overhead
- Verification normalizes output to ignore performance variations
