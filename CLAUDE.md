# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Build

```bash
# Standard CPU build
cmake -B build
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# CUDA (NVIDIA GPU)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# HIP (AMD GPU)
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx1100
cmake --build build --config Release -j $(nproc)

# Vulkan
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)
```

## Tests

```bash
# Build with tests enabled (default when building standalone)
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release -j $(nproc)

# Run all tests
cd build && ctest --output-on-failure

# Run a specific test binary (e.g., test-backend-ops)
./build/tests/test-backend-ops

# Run CI locally
bash ./ci/run.sh ./tmp/results ./tmp/mnt
# With CUDA: GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

For model format changes: run `test-backend-ops` to verify backend consistency. For perplexity/performance regressions: use `llama-perplexity` and `llama-bench` tools.

## Architecture

llama.cpp is structured in layers, each with a clear role:

### Layer 1: ggml tensor library (`ggml/`)
The foundation. Provides tensor operations, quantization, memory allocation, and hardware backends. Key files:
- `ggml/src/ggml.c` — core tensor ops, graph execution
- `ggml/src/ggml-backend.cpp` — backend abstraction (CPU, CUDA, Metal, etc.)
- `ggml/src/ggml-cpu/` — CPU-optimized kernels (AVX, NEON, etc.)
- `ggml/src/ggml-cuda/` — CUDA kernels
- `ggml/src/ggml-metal/` — Metal (Apple GPU) kernels
- Other backends: `ggml-hip/`, `ggml-vulkan/`, `ggml-sycl/`, `ggml-opencl/`

**Critical matrix multiply convention**: `ggml_mul_mat(ctx, A, B)` computes `C = B * A^T` (transposed from convention). Tensors use row-major order; dimension 0 = columns, 1 = rows, 2 = matrices.

### Layer 2: llama library (`src/`, `include/llama.h`)
The public C API lives in `include/llama.h`. Implementation in `src/`:
- `llama-model.cpp` / `llama-model-loader.cpp` — model loading from GGUF, architecture dispatch
- `llama-arch.cpp` — architecture definitions (LLaMA, Mistral, Falcon, etc.)
- `llama-context.cpp` — inference context, `llama_decode()` entrypoint
- `llama-graph.cpp` — computation graph construction per architecture
- `llama-kv-cache.cpp` — KV cache management; variants for sliding window (`-iswa`) and DualShiftAttention (`-dsa`)
- `llama-batch.cpp` — batched token input management
- `llama-sampler.cpp` — sampling logic (temperature, top-k, top-p, etc.)
- `llama-vocab.cpp` — tokenizer (BPE, SPM, WordPiece)
- `llama-adapter.cpp` — LoRA adapter support
- `llama-quant.cpp` — quantization routines

### Layer 3: common utilities (`common/`)
Shared helpers used by all tools but not part of the public API:
- `arg.cpp` — CLI argument parsing shared by all tools
- `chat.cpp`, `chat-peg-parser.cpp` — chat template parsing and formatting
- `common.cpp` — sampling params, model loading helpers
- `download.cpp` — HuggingFace model download support

### Layer 4: tools (`tools/`)
CLI binaries built on top of the library:
- `tools/server/` — OpenAI-compatible HTTP server (`llama-server`)
- `tools/llama-bench/` — performance benchmarking
- `tools/perplexity/` — perplexity evaluation
- `tools/quantize/` — model quantization
- `tools/imatrix/` — importance matrix computation
- `tools/rpc/` — RPC backend for offloading to remote GPU

### Model conversion (`conversion/`, `convert_hf_to_gguf.py`)
Python scripts for converting HuggingFace models to GGUF format. `gguf-py/` contains the Python GGUF library used by the converters. See `docs/development/HOWTO-add-model.md` for adding new model architectures.

## Server architecture

The `tools/server/` HTTP server uses a slot-based concurrency model:
- `server_context` — holds `llama_context` and all active slots
- `server_slot` — one "sequence" of parallel inference
- `server_queue` / `server_response` — thread-safe queues between HTTP workers and inference context
- Supports inference mode (single model) and router mode (multiple backends)

## Coding style

- 4-space indentation; brackets on same line; `void * ptr`, `int & a`
- No emdash `—`, unicode arrows `->`, or other non-ASCII characters in code/comments
- Use sized integer types (`int32_t`) in the public API
- Prefer basic `for` loops over fancy STL constructs; avoid templates
- Vertical alignment for readability
- Keep comments concise; avoid restating what the code already says
- Use `clang-format` (v15+) when uncertain about formatting
- Structs declared as `struct foo {}`, not `typedef struct`
