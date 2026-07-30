# LPU Backend Implementation Guide

**Status**: Implementation in progress  
**Last Updated**: 2026-06-29  
**Audience**: Developers implementing and maintaining the LPU backend in llama.cpp

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Operation Delegation Strategy](#operation-delegation-strategy)
5. [MoE Kernel Design](#moe-kernel-design)
6. [Build Instructions](#build-instructions)
7. [Testing Strategy](#testing-strategy)
8. [Debugging and Performance](#debugging-and-performance)

---

## Overview

The LPU (Learning Processing Unit) backend integrates LPU hardware acceleration into llama.cpp's ggml tensor library. The backend provides:

- **MoE (Mixture of Experts) acceleration** — fast expert selection and routing via `mul_mat_id`
- **Dense FFN operations** — accelerated matrix multiply and element-wise ops
- **Host-memory stub mode** — development/testing without real LPU hardware

### Key Design Principles

1. **Delegate to CPU for correctness** — LPU ops call internal ggml CPU functions (`ggml_compute_forward_*`), ensuring bit-for-bit correctness with CPU baseline
2. **MoE optimization via per-expert batching** — multiple tokens sharing an expert's weights are processed in a single GEMM (Gather-GEMM-Scatter pattern)
3. **Minimal hardware dependencies** — stub backend uses `std::malloc` as device memory; real hardware kernels plugged in later via `lpu_ops_compute` hooks
4. **Transparent op dispatch** — backend auto-registers in ggml's op-dispatch table; llama.cpp calls LPU ops without code changes

---

## Architecture

### Layer Stack

```
┌─────────────────────────────────────────────────────┐
│  llama.cpp application layer                        │
│  (inference context, batch processing)              │
└────────────────────┬────────────────────────────────┘
                     │
                     ↓ ggml_backend_* API
┌─────────────────────────────────────────────────────┐
│  ggml backend abstraction layer                     │
│  ├─ backend registry                               │
│  ├─ device enumeration                             │
│  └─ op dispatch (supports_op → compute)            │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ↓           ↓           ↓
    ┌─────────┐ ┌─────────┐ ┌──────────┐
    │ CPU0    │ │ CUDA0   │ │ LPU0     │
    │ backend │ │ backend │ │ backend  │
    └────┬────┘ └────┬────┘ └────┬─────┘
         │           │           │
         └───────────┼───────────┘
                     ↓ ggml_backend_compute API
        ┌────────────────────────────────┐
        │  Individual op kernels         │
        │  ├─ mul_mat                    │
        │  ├─ mul_mat_id (MoE)           │
        │  ├─ rms_norm, add, mul, etc.   │
        │  └─ [stub delegates to CPU]    │
        └────────────────────────────────┘
```

### File Organization

```
ggml/src/ggml-lpu/
├── ggml-lpu.cpp          -- backend lifecycle (init, device enum)
├── ggml-lpu.h (public)   -- C API (in ggml/include/)
├── lpu_ops.cpp           -- operation implementations (delegates/stubs)
├── lpu_ops.h             -- operation headers & statistics
└── common.h              -- internal utilities

tests/
├── test-lpu-backend.cpp  -- dedicated LPU correctness tests
└── test-backend-ops.cpp  -- generic backend tester (auto-discovers LPU)
```

---

## Core Components

### 1. Backend Registration (`ggml-lpu.cpp`)

**Responsibility**: Lifecycle management, device enumeration, buffer allocation.

#### Entry Point

```cpp
ggml_backend_reg_t ggml_backend_lpu_reg(void)
```

- Called once at startup by `ggml_backend_registry`
- Returns a registry object containing device count and factory function
- No real initialization; stub returns 1 virtual device ("LPU0")

#### Device Enumeration

```cpp
int ggml_backend_lpu_get_device_count(void)     // → 1 (stub)
ggml_backend_dev_t ggml_backend_lpu_get_device(int i)
```

#### Buffer Type

```cpp
ggml_backend_buffer_type_t ggml_backend_lpu_buffer_type(int device)
```

Returns a buffer type with interface:

| Method | Implementation |
|--------|-----------------|
| `alloc(size)` | `std::malloc(size)` (stub: host memory) |
| `free(ptr)` | `std::free(ptr)` |
| `is_host()` | **`true`** (stub buffers are host-accessible) |
| `get_tensor` / `set_tensor` | `memcpy` (stub: no-op since buffer is host) |

**Critical**: `is_host = true` lets the backend comparison harness (`test-backend-ops`) directly memcmp LPU outputs against CPU golden without expensive round-trip via `get_tensor`.

#### Backend Creation

```cpp
ggml_backend_t ggml_backend_lpu_init(int device)
```

Returns a backend object with:

- **`supports_op(op_type, src_tensors)`** — returns true for MoE/FFN ops with F32/F16 inputs
- **`compute(compute_params, tensors)`** — dispatches op to per-op handler
- Device callbacks for synchronization (stub: no-op)

### 2. Operation Implementations (`lpu_ops.cpp`)

**Responsibility**: Compute kernels for each supported op.

#### Delegation Pattern (Most Ops)

All ops except `mul_mat_id` delegate to internal ggml CPU functions:

```cpp
void lpu_op_mul_mat(ggml_tensor * dst) {
    auto params = make_single_thread_params();
    ggml_compute_forward_mul_mat(&params, dst);
}
```

**Why delegation?**

1. **Correctness**: Bit-for-bit CPU equivalence (stub phase)
2. **Simplicity**: No custom math bugs (broadcasting, strides, masks all handled by ggml CPU)
3. **Maintainability**: Op logic lives in one place (`ggml-cpu`)
4. **Hardware swap**: Real LPU kernels replace `ggml_compute_forward_*` calls via the `ggml-backend-reg` hook system

#### Supported Ops and Delegates

| Op | Delegate Function | Status |
|----|-------------------|--------|
| `MUL_MAT` | `ggml_compute_forward_mul_mat` | Delegated |
| `MUL_MAT_ID` (MoE) | Custom implementation (see below) | Hand-rolled |
| `RMS_NORM` | `ggml_compute_forward_rms_norm` | Delegated |
| `NORM` | `ggml_compute_forward_norm` | Delegated |
| `ADD` | `ggml_compute_forward_add` | Delegated |
| `MUL` | `ggml_compute_forward_mul` | Delegated |
| `SCALE` | `ggml_compute_forward_scale` | Delegated |
| `SOFT_MAX` | `ggml_compute_forward_soft_max` | Delegated (includes attention masks) |
| `UNARY` (silu, gelu, etc.) | `ggml_compute_forward_unary` | Delegated |
| `GLU` (swiglu, geglu) | `ggml_compute_forward_glu` | Delegated |
| `GET_ROWS` (gather) | `ggml_compute_forward_get_rows` | Delegated |
| `ARGSORT` (top-k) | `ggml_compute_forward_argsort` | Delegated |

#### Single-Thread Params Helper

```cpp
static struct ggml_compute_params make_single_thread_params() {
    struct ggml_compute_params p{};
    p.ith = 0;      // thread index (always 0 for stub)
    p.nth = 1;      // num threads (always 1 for stub)
    p.type = GGML_TASK_COMPUTE;
    p.wdata = nullptr;  // no work buffer (single-threaded)
    p.wsize = 0;
    return p;
}
```

**Include headers**:
```cpp
#include "../ggml-cpu/ops.h"             // ggml_compute_forward_* declarations
#include "../ggml-cpu/ggml-cpu-impl.h"   // struct ggml_compute_params
```

### 3. Operation Statistics

**Purpose**: Observability — prove LPU code paths are executed in tests.

#### Stats Structure (`lpu_ops.h`)

```cpp
struct lpu_op_stats {
    std::atomic<int64_t> mul_mat{0};
    std::atomic<int64_t> mul_mat_id{0};
    std::atomic<int64_t> rms_norm{0};
    std::atomic<int64_t> unary{0};      // silu, gelu, ...
    std::atomic<int64_t> glu{0};        // swiglu, geglu
    std::atomic<int64_t> add{0};
    std::atomic<int64_t> mul{0};
    std::atomic<int64_t> soft_max{0};
    std::atomic<int64_t> get_rows{0};
    std::atomic<int64_t> argsort{0};
    std::atomic<int64_t> norm{0};
    std::atomic<int64_t> scale{0};
    std::atomic<int64_t> total{0};
};

extern lpu_op_stats g_lpu_op_stats;
void lpu_reset_op_stats();
void lpu_print_op_stats();
```

#### Usage in Op Implementations

```cpp
void lpu_op_mul_mat(ggml_tensor * dst) {
    g_lpu_op_stats.mul_mat++;
    g_lpu_op_stats.total++;
    
    auto params = make_single_thread_params();
    ggml_compute_forward_mul_mat(&params, dst);
}
```

---

## Operation Delegation Strategy

### Why Delegate to CPU?

The LPU backend is currently in **stub mode** — it has no real hardware. Delegating to ggml's internal CPU functions achieves:

1. **Correctness validation** — tests compare LPU output against CPU golden; delegation guarantees they match
2. **Zero custom math** — avoids reimplementing broadcasting, striding, quantization, masking, etc. per op
3. **Rapid iteration** — swap out `ggml_compute_forward_*` calls later when real kernels are ready; rest of code unchanged
4. **Integration with LLM inference** — llama.cpp can run full models on LPU immediately (using CPU kernels as fallback)

### Hardware Integration Path

When real LPU SDK becomes available:

1. **Create `lpu-kernels/`** — new directory with LPU-specific implementations (C/assembly/LPU ISA)
2. **Update `supports_op()`** — add hardware-specific type/size checks (e.g., only F32/F16, dims ≥ N)
3. **Replace delegates** — swap `ggml_compute_forward_*` calls with `lpu_compute_forward_*` (or use the ggml-backend hook system)
4. **Performance tuning** — profile, optimize memory layout, kernel fusion, etc.

---

## MoE Kernel Design

### The Gather-GEMM-Scatter Pattern

The `mul_mat_id` (expert selection) operation is the core of MoE acceleration. Unlike dense matmul, MoE faces a unique constraint:

- **Dense GEMM**: All rows participate; easy to batch
- **MoE routing**: Only **top-K selected experts** are active; which experts varies per token

**Naive approach** (per-token-per-expert): Dispatch N×K independent small GEMMs → high kernel launch overhead, low compute utilization.

**Optimized approach** (per-expert batching): Group all tokens hitting the same expert, compute one large GEMM → amortized kernel launch, high SIMD utilization.

### Algorithm Overview

```
Input:
  ids     [n_exp_used, n_tokens]  int32   -- routing: ids[slot, token] = expert_id
  as      [cols, rows, n_expert]  float   -- stacked expert weights
  b       [cols, 1, n_tokens]     float   -- token hidden states (broadcasted)
  
Output:
  dst     [rows, n_exp_used, n_tokens]  float

Algorithm:
  1. Build routing table: for each (token, slot, expert_id) → register "hit"
  2. For each active expert:
     a. Gather: extract weights W[cols, rows] from as[:,:,expert_id]
     b. Gather: collect all token embeddings hitting this expert into inp[cols, n_hits]
     c. GEMM: out = W^T @ inp  [rows, n_hits]
     d. Scatter: place results back into dst[rows, slot, token] for each (token, slot) that hit
```

### Correctness Guarantees

The implementation maintains these invariants:

1. **Per-expert independence**: Experts are processed in order; each writes disjoint output slots
2. **Output shape**: `dst[r, slot, token]` ← result of `as[:,:,ids[slot,token]] @ b[:,0,token]`
3. **Broadcasting**: Input `b` can have shape `[cols, 1, n_tokens]` (broadcast on slot) or `[cols, n_exp_used, n_tokens]` (no broadcast)
4. **Strides**: Correctly handles contiguous and non-contiguous tensor layouts via `nb[]` fields

### Worked Example: 3 Tokens, 4 Experts, Top-2

**Scenario**:
- 3 tokens, each selecting top-2 experts
- 4 total experts available
- `n_exp_used=2, n_tokens=3, n_expert=4, cols=16, rows=32`

**Routing IDs** (row-major layout):
```
ids[0,:] = [2, 3, 0, 3, 1, 2]  (token0 picks experts 2,3; token1 picks 0,3; token2 picks 1,2)
```

**Expert 2 processing** (hit by token 0 slot 0, token 2 slot 1):
```
① Gather weights:
   W ← as[:,:,2]  [16×32]

② Gather inputs (2 tokens):
   inp[0,:] ← b[:,0,0]  (token 0)
   inp[1,:] ← b[:,0,2]  (token 2)
   inp is [16×2] (2 columns = 2 tokens)

③ GEMM (single large matmul):
   out = W @ inp  [32×2]
   out[:,0] = as[:,:,2] @ b[:,0,0]   (token 0's result)
   out[:,1] = as[:,:,2] @ b[:,0,2]   (token 2's result)
   ← same W reused for both tokens!

④ Scatter results:
   dst[:,0,0] ← out[:,0]    (token 0, slot 0)
   dst[:,1,2] ← out[:,1]    (token 2, slot 1)
```

**Efficiency gain**:
- Naive: 2 separate GEMMs (W loaded twice, kernel dispatched twice)
- Optimized: 1 GEMM (W loaded once, kernel dispatched once, larger N → better utilization)

---

## Build Instructions

### Prerequisites

The LPU backend requires:

1. **ggml and llama.cpp source** — standard llama.cpp checkout
2. **LPU SDK** (optional, stub mode only needs dummy stubs)
3. **CMake 3.15+**
4. **C++17 compiler**

### Building with LPU Backend

#### Option 1: Stub Backend (No Real Hardware)

```bash
# Create minimal stub libraries
mkdir -p /tmp/lpu_stub/lib /tmp/lpu_stub/include
cc -shared -o /tmp/lpu_stub/lib/liblpu_runtime.so -x c /dev/null
cc -shared -o /tmp/lpu_stub/lib/liblpu_nn.so      -x c /dev/null

# Build llama.cpp with LPU backend
cd llama.cpp
cmake -B build \
    -DGGML_LPU=ON \
    -DLPU_INSTALL_DIR=/tmp/lpu_stub \
    -DCMAKE_BUILD_TYPE=Release \
    -j$(nproc)

cmake --build build -j$(nproc)
```

#### Option 2: Real LPU Hardware

```bash
cmake -B build \
    -DGGML_LPU=ON \
    -DLPU_INSTALL_DIR=/path/to/actual/lpu/sdk \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

#### Build Targets

```bash
# Build everything
cmake --build build

# Build just the LPU backend + tests
cmake --build build --target ggml-lpu test-lpu-backend test-backend-ops

# Install
cmake --install build --prefix /usr/local
```

### Verification

```bash
# List backends
./build/bin/llama-server --version

# Check LPU device is available
./build/tests/test-backend-ops test -b LPU0
```

---

## Testing Strategy

### Test Coverage

`tests/test-lpu-backend.cpp` contains six independent tests. Each test:

1. Runs the same graph on CPU (golden reference) and on LPU
2. Compares outputs element-by-element with `max-abs-diff <= 1e-4`
3. Asserts that the relevant LPU op counters are `> 0` after the LPU run

| # | Function | Ops exercised | Tolerance |
|---|----------|---------------|-----------|
| 1 | `test_ffn_block` | `rms_norm`, `mul_mat` (×2), `unary`(silu), `add` | 1e-4 |
| 2 | `test_moe_block` | `mul_mat_id` (broadcast b, 4 experts, top-2, 3 tokens), `mul`, `add` | 1e-4 |
| 3 | `test_attention_ops` | `argsort` (descending), `get_rows`, `soft_max` + F16 causal mask | 1e-4 (exact for argsort) |
| 4 | `test_activations` | `unary`(gelu, relu), `glu`(swiglu, geglu) | 1e-4 |
| 5 | `test_norm_and_scale` | `norm` (layer norm), `rms_norm` (isolated), `scale` | 1e-4 |
| 6 | `test_moe_large` | `mul_mat_id` (**non-broadcast** b, 8 experts, top-4, 8 tokens), `mul`, `add` | 1e-4 |

> Tests 2 and 6 together cover both b-broadcasting modes of `mul_mat_id`:
> Test 2 uses `b = [cols, 1, n_tokens]` (broadcast slot dimension);
> Test 6 uses `b = [cols, n_exp_used, n_tokens]` (no broadcast — each slot has its own input vector).

### 1. Quick Start

```bash
# Step 1 — create stub SDK (skip if you have real LPU hardware)
mkdir -p /tmp/lpu_stub/{lib,include}
cc -shared -o /tmp/lpu_stub/lib/liblpu_runtime.so -x c /dev/null
cc -shared -o /tmp/lpu_stub/lib/liblpu_nn.so      -x c /dev/null

# Step 2 — configure
cmake -B build \
    -DGGML_LPU=ON \
    -DLPU_INSTALL_DIR=/tmp/lpu_stub \
    -DCMAKE_BUILD_TYPE=Release

# Step 3 — build the test binary
cmake --build build --target test-lpu-backend -j$(nproc)

# Step 4 — run
./build/tests/test-lpu-backend
```

### 2. Pass Criteria

A run is **passing** when all of the following hold:

- Every output line reads `PASS  …` (no `FAIL` lines)
- `max-abs-diff` for every F32 graph is `<= 1e-4`
- `argsort` output is **exactly equal** to CPU (integer sort, no floating point)
- All per-op counters checked via `LPU_CHECK` are `> 0` after each LPU run
- Program exits with code **0**
- Final summary line: `=== Results: N passed, 0 failed ===`

### 3. Expected Output (abbreviated)

```
=== test-lpu-backend ===

Registered backends (2):
  [0] LPU0
  [1] CPU0
PASS  LPU device is registered
PASS  CPU backend initialised
PASS  LPU backend initialised

--- Test 1: FFN block ---
[LPU op stats]  mul_mat:2  rms_norm:1  unary:1  add:1  total:5
  FFN max-abs-diff CPU vs LPU: 0.000000e+00  (tol 1e-04)
PASS  FFN output matches CPU golden (max_abs_diff=0.00e+00)
PASS  LPU MUL_MAT was called (2)
PASS  LPU RMS_NORM was called (1)
PASS  LPU UNARY(silu) was called (1)
PASS  LPU ADD was called (1)
PASS  LPU total ops > 0 (5)

--- Test 2: MoE block ---
[LPU op stats]  mul_mat_id:1  mul:1  add:1  total:3
  MoE max-abs-diff CPU vs LPU: 0.000000e+00  (tol 1e-04)
PASS  MoE output matches CPU golden (max_abs_diff=0.00e+00)
PASS  LPU MUL_MAT_ID was called (1)
...

--- Test 3: Attention ops (argsort, get_rows, soft_max) ---
PASS  argsort output matches CPU golden
PASS  LPU ARGSORT was called (1)
  get_rows max-abs-diff: 0.000000e+00  (tol 1e-04)
PASS  get_rows output matches CPU golden (max_abs_diff=0.00e+00)
PASS  LPU GET_ROWS was called (1)
  soft_max+mask max-abs-diff: X.XXXXXXe-XX  (tol 1e-04)
PASS  soft_max+mask output matches CPU golden (max_abs_diff=X.XXe-XX)
PASS  LPU SOFT_MAX was called (1)

--- Test 4: Activations (gelu, relu, swiglu, geglu) ---
...
--- Test 5: Norm and Scale ops ---
...
--- Test 6: MoE large (non-broadcast, 8 experts, top-4, 8 tokens) ---
...

=== Results: N passed, 0 failed ===
```

### 4. Integration Tests: `test-backend-ops`

The existing generic harness auto-discovers LPU0 and tests every op declared in `supports_op()`:

```bash
cmake --build build --target test-backend-ops -j$(nproc)

# All ops supported by LPU0
./build/tests/test-backend-ops test -b LPU0

# Specific op (useful for isolating failures)
./build/tests/test-backend-ops test -b LPU0 -o MUL_MAT_ID
./build/tests/test-backend-ops test -b LPU0 -o RMS_NORM
./build/tests/test-backend-ops test -b LPU0 -o SOFT_MAX

# Performance numbers (wall-clock, GFLOPS)
./build/tests/test-backend-ops perf -b LPU0
```

The harness compares every LPU output against CPU golden and reports `OK` / `FAILED` per variant.
Default thresholds: NMSE < `1e-7` for F32 single-thread.

### 5. CTest Integration

```bash
# Run the LPU-labelled tests only
cd build && ctest -L lpu --output-on-failure -V

# Run by name
ctest -R test-lpu-backend --output-on-failure
```

Expected:
```
Test #N: test-lpu-backend .............   Passed    X.XX sec
100% tests passed, 0 tests failed out of 1
```

### 6. Failure Diagnosis

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `LPU device not found` | `-DGGML_LPU=ON` missing or link error | Rebuild with flag; verify `ggml-backend-reg.cpp` compiled |
| `max-abs-diff > 0` | CPU delegate not called (wrong include path) | Check `compile_commands.json`; verify `../ggml-cpu/ops.h` resolves |
| op counter `= 0` | Graph fell back to CPU (type guard in `supports_op`) | Add the offending type to `ggml_backend_lpu_device_supports_op()` in `ggml-lpu.cpp` |
| Link error `lpu_reset_op_stats` | `ggml-lpu` not in link line | Verify `target_link_libraries(test-lpu-backend PRIVATE ggml-lpu)` in `tests/CMakeLists.txt` |
| `find_library` failure | `LPU_INSTALL_DIR` wrong or empty | Use stub SDK creation commands above |
| `soft_max+mask` diff large | F16 mask precision | Expected small diff (`< 1e-3`); tolerance is `1e-4` — check mask values |

---

## Debugging and Performance

### Operation Call Tracing

Enable detailed op logging via `lpu_print_op_stats()`:

```cpp
// In test or application code
#include "ggml-lpu.h"

lpu_reset_op_stats();
// ... run inference ...
lpu_print_op_stats();
```

**Output**:
```
[LPU op stats]
  mul_mat    : 1024
  mul_mat_id : 256
  rms_norm   : 512
  add        : 2048
  mul        : 1024
  total      : 5120
```

### Memory Profiling

The stub backend uses `std::malloc` for device allocation. To profile memory usage:

```bash
# With memory tracking
GGML_BACKEND=LPU0 valgrind --tool=massif ./build/bin/llama-server \
    -m model.gguf -n 32
```

### Performance Analysis

Stub backend performance is bounded by host-memory bandwidth (`std::malloc`/`memcpy` overhead). Once real LPU kernels are integrated:

1. **Measure kernel times**: Profile individual ops via `test-backend-ops perf`
2. **Identify bottlenecks**: Matrix dimensions, memory layout, quantization costs
3. **Optimize layout**: Transpose weights, batch operations, etc.
4. **Kernel fusion**: Combine ops where possible (e.g., `mul_mat + add` → fused op)

### Debugging Correctness Issues

#### Step 1: Isolate the Op

Run a single op via `test-backend-ops`:

```bash
./build/tests/test-backend-ops test -b LPU0 -o RMS_NORM
```

#### Step 2: Compare CPU vs LPU Side-by-Side

In a custom test:

```cpp
ggml_tensor * cpu_out = run_on_backend(graph, cpu_backend);
ggml_tensor * lpu_out = run_on_backend(graph, lpu_backend);

float max_diff = compare_tensors(cpu_out, lpu_out);
printf("Max absolute difference: %e\n", max_diff);

if (max_diff > 1e-5) {
    printf("FAIL: op computation mismatch\n");
    dump_tensor(cpu_out, "cpu_output.txt");
    dump_tensor(lpu_out, "lpu_output.txt");
}
```

#### Step 3: Check Op Parameters

Verify the tensor properties match expectations:

```cpp
printf("dst: shape=[%d,%d,%d] strides=[%d,%d,%d] type=%s\n",
    dst->ne[0], dst->ne[1], dst->ne[2],
    dst->nb[0], dst->nb[1], dst->nb[2],
    ggml_type_name(dst->type));
```

#### Step 4: Trace Delegation

Add logging to the op implementation:

```cpp
void lpu_op_mul_mat(ggml_tensor * dst) {
    fprintf(stderr, "[LPU] lpu_op_mul_mat: shape=[%d,%d] type=%s\n",
        dst->ne[0], dst->ne[1], ggml_type_name(dst->type));
    
    auto params = make_single_thread_params();
    ggml_compute_forward_mul_mat(&params, dst);
    
    g_lpu_op_stats.mul_mat++;
    g_lpu_op_stats.total++;
}
```

---

## Implementation Checklist

### Phase 1: Core Backend (Completed)

- [x] `ggml-lpu.cpp` — backend registration and lifecycle
- [x] `lpu_ops.h` — op declarations and stats structure
- [x] `ggml-lpu.h` — public C API
- [x] CMake integration (`ggml/CMakeLists.txt`, `ggml/src/CMakeLists.txt`)

### Phase 2: Operation Delegation (Completed)

- [x] `is_host = true` in buffer type interface (stub alloc is host memory)
- [x] CPU delegates in `lpu_ops.cpp` for all ops except `mul_mat_id`
  - [x] `mul_mat` → `ggml_compute_forward_mul_mat` (with work buffer for quantized)
  - [x] `rms_norm`, `norm`, `add`, `mul`, `scale`, `soft_max`, `unary`, `glu`, `get_rows`, `argsort`
- [x] Hand-rolled `lpu_op_mul_mat_id` (gather-GEMM-scatter, supports broadcast and non-broadcast b)
- [x] Per-op call counters (`g_lpu_op_stats`) with `lpu_reset_op_stats()` / `lpu_print_op_stats()`

### Phase 3: Testing & Validation (Completed)

- [x] `tests/test-lpu-backend.cpp` — 6 tests covering FFN, MoE (basic + large), attention ops, activations, norm+scale
- [x] `tests/CMakeLists.txt` — registered under `if (GGML_LPU)` guard at lines 246–254
- [x] Verify against `test-backend-ops` harness
- [x] Op call counters prove LPU code paths are exercised

### Phase 4: Hardware Integration (Future)

- [ ] Create `lpu-kernels/` with real LPU implementations
- [ ] Update `supports_op()` with hardware type/size constraints
- [ ] Swap `ggml_compute_forward_*` calls with LPU kernels
- [ ] Performance tuning and kernel fusion

---

## References

### Related Documentation

- [AGENTS.md](../AGENTS.md) — Architecture and agent guidelines
- [CLAUDE.md](../CLAUDE.md) — Project build and coding style
- [ggml-backend documentation](../ggml/README.md) — Backend abstraction layer

### LPU SDK Integration

When LPU SDK becomes available, refer to:
- LPU SDK documentation (provided by hardware vendor)
- Reference implementations (typically in `lpu-sdk/examples/`)
- Memory layout and alignment requirements
- Profiling tools and performance counters

### GGML Backend Examples

Reference implementations for comparison:
- `ggml/src/ggml-cuda/` — NVIDIA GPU backend
- `ggml/src/ggml-metal/` — Apple GPU backend
- `ggml/src/ggml-cann/` — Huawei CANN backend

---

## Common Issues and Solutions

### Build Fails: "lpu_runtime not found"

**Cause**: `LPU_INSTALL_DIR` points to wrong or non-existent directory.

**Solution**:
```bash
# Create stub SDK
mkdir -p /tmp/lpu_stub/{lib,include}
cc -shared -o /tmp/lpu_stub/lib/liblpu_runtime.so -x c /dev/null

# Rebuild
cmake -B build -DLPU_INSTALL_DIR=/tmp/lpu_stub
```

### test-backend-ops Reports "LPU device not found"

**Cause**: Backend not registered; `-DGGML_LPU=ON` missing or linking error.

**Solution**:
```bash
# Verify build flag
cmake -B build -DGGML_LPU=ON
cmake --build build

# Check compilation
grep "GGML_LPU" build/compile_commands.json
```

### test-lpu-backend Shows "max-abs-diff > 1e-5"

**Cause**: Op delegate not returning correct result; include path or function call issue.

**Solution**:
```cpp
// Add logging to op implementation
fprintf(stderr, "[DEBUG] calling ggml_compute_forward_rms_norm\n");
ggml_compute_forward_rms_norm(&params, dst);

// Rebuild and re-run
cmake --build build && ./build/tests/test-lpu-backend
```

### Linking Error: "undefined reference to lpu_reset_op_stats"

**Cause**: `ggml-lpu` library not linked to test executable.

**Solution**: Update `tests/CMakeLists.txt`:
```cmake
target_link_libraries(test-lpu-backend PRIVATE
    llama llama-common ggml ggml-lpu)
```

---

## Contributors

This guide is maintained by the llama.cpp LPU backend team. For questions or corrections, please refer to [AGENTS.md](../AGENTS.md) for architecture and workflow guidance.

---

**Last Updated**: 2026-06-29  
**Version**: 1.0 (Draft)
