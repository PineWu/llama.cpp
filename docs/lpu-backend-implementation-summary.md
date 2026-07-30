# LPU Backend Implementation & Test Summary

**Date**: 2026-06-30
**Scope**: Summary of the LPU (Learning Processing Unit) ggml backend changes and the accompanying test suite — covering key code logic, the bugs found and fixed during validation, and detailed annotations.

---

## Table of Contents

1. [Background](#1-background)
2. [Change Overview](#2-change-overview)
3. [Architecture](#3-architecture)
4. [Core Code Logic](#4-core-code-logic)
5. [Test Suite](#5-test-suite)
6. [Bugs Found & Fixed During Validation](#6-bugs-found--fixed-during-validation)
7. [How to Build & Run](#7-how-to-build--run)
8. [Final Test Results](#8-final-test-results)

---

## 1. Background

llama.cpp's ggml library supports pluggable hardware backends (CPU, CUDA, Metal, Vulkan, …). The **LPU backend** integrates LPU hardware into this framework, focused on two workloads:

- **Dense FFN** — matrix multiply + normalization + activation
- **MoE (Mixture of Experts)** — expert-routing GEMM (`mul_mat_id`)

In the current phase there is **no real LPU hardware/SDK yet**. The backend therefore runs in **stub mode**: device memory is `std::malloc` (host memory), and compute kernels **delegate to ggml's internal CPU functions**. This gives bit-for-bit CPU equivalence while letting the full op-dispatch, buffer-management, and test infrastructure be built and validated ahead of the real SDK.

The deliverable for this phase: make the stub ops **numerically correct** and prove correctness with a dedicated test suite comparing LPU output against the CPU golden reference.

---

## 2. Change Overview

### Files modified

| File | Role | Key changes |
|------|------|-------------|
| [ggml/include/ggml-lpu.h](../ggml/include/ggml-lpu.h) | Public C API | Backend lifecycle, device queries, buffer types |
| [ggml/src/ggml-lpu/ggml-lpu.cpp](../ggml/src/ggml-lpu/ggml-lpu.cpp) | Backend infra | 5 vtable layers; **fixed `set_tensor`/`get_tensor` to use `tensor->data`**; **fixed `guid` return** |
| [ggml/src/ggml-lpu/lpu_ops.cpp](../ggml/src/ggml-lpu/lpu_ops.cpp) | Op kernels | CPU delegation; **added threadpool to `compute_params`**; **fixed `cpu_sgemm` output layout**; op-call counters |
| [ggml/src/ggml-lpu/lpu_ops.h](../ggml/src/ggml-lpu/lpu_ops.h) | Op headers | `lpu_op_stats` struct + `g_lpu_op_stats` |
| [tests/test-lpu-backend.cpp](../tests/test-lpu-backend.cpp) | Tests | 6 tests (FFN, MoE×2, attention, activations, norm+scale); per-backend graph allocation |
| [tests/CMakeLists.txt](../tests/CMakeLists.txt) | Build | `test-lpu-backend` registered under `if (GGML_LPU)` |
| [docs/lpu-backend-guide.md](lpu-backend-guide.md) | Docs | Architecture, build, testing reference |

### What this phase added vs. what already existed

- **Already existed (skeleton)**: backend registration, buffer type, op dispatcher, 12 op stubs, op-call counters, CMake wiring, Tests 1–2.
- **Added this phase**: Tests 3–6, the in-file documentation header, the guide's testing section, and — critically — **5 real bugs** that prevented the stub from producing correct results, found and fixed by running the tests.

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────┐
│  llama.cpp (inference context, batch processing)    │
└────────────────────┬────────────────────────────────┘
                     │ ggml_backend_* API
┌────────────────────┴────────────────────────────────┐
│  ggml backend abstraction                           │
│  registry → device enum → supports_op → compute     │
└────────────────────┬────────────────────────────────┘
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      CPU0        CUDA0       LPU0   ◄── this backend
         │           │           │
         └───────────┼───────────┘
                     ▼
        per-op kernels (lpu_ops.cpp)
        ├─ delegate to ggml_compute_forward_* (CPU)
        └─ mul_mat_id: hand-rolled gather-GEMM-scatter
```

### Why delegate to CPU?

Because stub buffers **are** host memory, calling ggml's internal CPU compute functions gives bit-exact results with **zero custom math** — no need to reimplement broadcasting, striding, masking, or quantization per op. When the real LPU SDK arrives, each `ggml_compute_forward_*` call is swapped for an `lpu_*` kernel; the surrounding dispatch and buffer code stays unchanged.

---

## 4. Core Code Logic

### 4.1 Op delegation pattern (`lpu_ops.cpp`)

Every supported op (except `mul_mat_id`) follows one shape: increment a counter, build single-threaded compute params, call the CPU function.

```cpp
ggml_status lpu_op_rms_norm(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.rms_norm++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_rms_norm(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}
```

**Supported ops and their delegates:**

| Op | Delegate | Notes |
|----|----------|-------|
| `MUL_MAT` | `ggml_compute_forward_mul_mat` | Allocates a work buffer for quantized→float conversion |
| `RMS_NORM` / `NORM` | `ggml_compute_forward_rms_norm` / `_norm` | |
| `ADD` / `MUL` | `_add` / `_mul` | `mul` is in `binary-ops.h`, **not** `ops.h` |
| `SCALE` / `SOFT_MAX` | `_scale` / `_soft_max` | soft_max honors `dst->src[1]` mask |
| `UNARY` (silu/gelu/relu) | `_unary` | |
| `GLU` (swiglu/geglu) | `_glu` | |
| `GET_ROWS` / `ARGSORT` | `_get_rows` / `_argsort` | |
| `MUL_MAT_ID` | **hand-rolled** (see 4.3) | CPU version is `static`, not exported |
| `RESHAPE/VIEW/PERMUTE/TRANSPOSE/CONT` | no-op | metadata-only |

### 4.2 Single-threaded compute params (`lpu_ops.cpp:78-106`)

```cpp
static struct ggml_threadpool * s_lpu_threadpool = nullptr;

static struct ggml_threadpool * get_lpu_threadpool() {
    if (!s_lpu_threadpool) {
        struct ggml_threadpool_params tpp = ggml_threadpool_params_default(1);
        s_lpu_threadpool = ggml_threadpool_new(&tpp);
    }
    return s_lpu_threadpool;
}

static struct ggml_compute_params single_thread_params(void * wdata = nullptr,
                                                        size_t wsize = 0) {
    struct ggml_threadpool * tp = get_lpu_threadpool();
    ggml_threadpool_chunk_set(tp, 0);   // reset work-stealing counter (see bug #3)

    struct ggml_compute_params p;
    memset(&p, 0, sizeof(p));
    p.ith        = 0;
    p.nth        = 1;
    p.wdata      = wdata;
    p.wsize      = wsize;
    p.threadpool = tp;                  // must be non-null (see bug #2)
    return p;
}
```

**Why a real threadpool?** `ggml_compute_forward_mul_mat` calls into the llamafile sgemm path, which dereferences `params->threadpool` to do work-stealing (`ggml_threadpool_chunk_add`). A null pointer there is a SEGV.

**Why reset the chunk counter?** The same work-stealing counter is reused across op calls. Leftover values from a previous op make the next op skip chunks — producing wrong results (see bug #4).

### 4.3 MoE kernel: `lpu_op_mul_mat_id` (gather-GEMM-scatter)

This is the most involved op. It cannot delegate to CPU because `ggml_compute_forward_mul_mat_id` is `static` in `ggml-cpu.c`. The hand-rolled implementation uses **per-expert batching**: group all tokens hitting the same expert, run one GEMM, scatter results back.

**Tensor contract:**
```
as   [cols, rows, n_expert]          stacked expert weights (F32/F16)
b    [cols, n_exp_used|1, n_tokens]  token inputs (F32)
ids  [n_exp_used, n_tokens]          int32 routing: ids[slot, token] = expert_id
dst  [rows, n_exp_used, n_tokens]    outputs (F32)

semantics: dst[:, e, t] = as[:, :, ids[e,t]] @ b[:, e % b_ne1, t]
```

**Algorithm:**

```cpp
// 1. Build routing table: expert_id -> [(slot, token), ...]
std::vector<std::vector<std::pair<int,int>>> expert_tokens(n_expert);
for (t, e) in (n_tokens × n_exp_used):
    ex = ids_data[t * n_exp_used + e]
    expert_tokens[ex].emplace_back(e, t)

// 2. Zero output, then process each active expert
memset(dst, 0, ...)
for ex in 0..n_expert:
    if expert_tokens[ex].empty(): continue

    // (a) Gather expert weights  W[rows, cols]
    W = as[:, :, ex]                      // F16→F32 if needed

    // (b) Gather all token inputs hitting this expert  inp[n_hits, cols]
    for h, (slot, token) in hits:
        inp[h] = b[:, slot % b_ne1, token]

    // (c) GEMM: out[n_hits, rows] = inp[n_hits, cols] @ W[cols, rows]
    cpu_sgemm(W, inp, out, rows, n_hits, cols)

    // (d) Scatter: write each hit's result back to dst[:, slot, token]
    for h, (slot, token) in hits:
        dst[token*n_exp_used*rows + slot*rows .. ] = out[h*rows .. ]
```

**Why per-expert batching over per-(token,slot) GEMM?**

| | Per-(token,slot) | Per-expert (this impl) |
|---|---|---|
| Weight reads | `n_tokens × n_exp_used` | `n_active_experts` (≤ n_exp_used) |
| Kernel launches | many small GEMMs | one GEMM per active expert |
| Cache locality | poor (tokens scattered) | good (same-expert tokens contiguous) |

When expert 2 is hit by 2 tokens, its weights load **once** and are shared — the bandwidth win is the dominant factor at production MoE scale.

### 4.4 The `cpu_sgemm` helper (the subtle bug — see 6.5)

```cpp
// out[n, m] = sum_k W[m, k] * inp[n, k]    (ggml convention: C = B * A^T)
//
// Layout: W [M=rows, K=cols], inp [N=n_hits, K=cols],
//         out [N=n_hits, M=rows]  ← each hit's output row is contiguous
static void cpu_sgemm(const float * A, const float * B,
                      float * C, int M, int N, int K) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[n * K + k];
            C[n * M + m] = acc;          // ← [n, m], NOT [m, n]
        }
    }
}
```

The output is laid out as `[n_hits, rows]` so that `out + h*rows` is one token's full output row — matching the scatter loop's `memcpy(out + h*rows, ...)`.

### 4.5 Buffer type: `set_tensor`/`get_tensor` (`ggml-lpu.cpp:106-117`)

```cpp
/* set_tensor */ [](buf, tensor, data, offset, size) {
    // Write to tensor->data + offset, NOT buf base + offset.
    lpu_h2d_impl((char *)tensor->data + offset, data, size, nullptr);
},
/* get_tensor */ [](buf, tensor, data, offset, size) {
    lpu_d2h_impl(data, (const char *)tensor->data + offset, size, nullptr);
},
```

`tensor->data` already points to this tensor's slot within the buffer (set by the allocator). The `offset` parameter is relative to that slot, so the copy must be anchored at `tensor->data`, not the buffer base.

### 4.6 Op-call statistics (`lpu_ops.h`)

```cpp
struct lpu_op_stats {
    long long mul_mat, mul_mat_id, rms_norm, norm, unary, glu,
              add, mul, scale, soft_max, get_rows, argsort, total;
};
extern lpu_op_stats g_lpu_op_stats;
void lpu_reset_op_stats();
void lpu_print_op_stats();
```

Each `lpu_op_*` increments its counter + `total` before computing. Tests reset, run, then assert `counter > 0` — this **proves the LPU code path ran** rather than silently falling back to CPU.

---

## 5. Test Suite

[tests/test-lpu-backend.cpp](../tests/test-lpu-backend.cpp) contains 6 independent tests. Every test:

1. Generates random input data in `std::vector`s
2. Builds the **same graph twice** — once on CPU (golden), once on LPU — each in its own `ggml_context` + `ggml_gallocr`
3. Compares outputs element-by-element (`max-abs-diff <= 1e-4`)
4. Asserts the relevant op counters are `> 0`

### 5.1 The per-backend run helper (`test-lpu-backend.cpp:105-149`)

```cpp
static void run_single_backend(backend, build_fn, user_data,
                               entries, result_f32, result_i32, is_int_out) {
    ggml_context * ctx = ggml_init({128MB, nullptr, /*no_alloc=*/true});
    ggml_tensor * out = build_fn(ctx, user_data);     // build_fn fills tensor slots
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // gallocr (not ggml_backend_alloc_ctx_tensors): handles view ops correctly
    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    for (entry in entries)                              // upload inputs
        ggml_backend_tensor_set(*entry.tensor_slot, entry.data, 0, entry.nbytes);

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
}
```

**Key design choices (each was a hard-won lesson):**

- **Separate context per backend call** — sharing one context across CPU+LPU runs leaves `tensor->buffer` dangling after the first free, causing use-after-free.
- **`no_alloc=true` + `ggml_gallocr`** — `ggml_backend_alloc_ctx_tensors` underestimates peak size for graphs with `view` ops and aborts; `gallocr` sizes from the actual graph.
- **Build-function callback** — `build_fn(ctx, user_data)` rebuilds the identical graph for each backend, so the comparison is apples-to-apples without duplicating setup code.

### 5.2 Test coverage matrix

| # | Function | Ops exercised | Config | Tolerance |
|---|----------|---------------|--------|-----------|
| 1 | `test_ffn_block` | `rms_norm`, `mul_mat`×2, `unary`(silu), `add` | hidden=64, ffn=128, batch=4 | 1e-4 |
| 2 | `test_moe_block` | `mul_mat_id` (**broadcast b**), `mul`, `add` | 4 experts, top-2, 3 tokens | 1e-4 |
| 3 | `test_attention_ops` | `argsort`, `get_rows`, `soft_max`+F16 mask | 8 experts / vocab 16 / seq 8×8×2 | exact / 1e-4 |
| 4 | `test_activations` | `unary`(gelu, relu), `glu`(swiglu, geglu) | hidden=64, batch=4 | 1e-4 |
| 5 | `test_norm_and_scale` | `norm`, `rms_norm`, `scale` | hidden=128, batch=8 | 1e-4 |
| 6 | `test_moe_large` | `mul_mat_id` (**non-broadcast b**), `mul`, `add`×3 | 8 experts, top-4, 8 tokens | 1e-4 |

Tests 2 and 6 together cover **both `b`-broadcasting modes** of `mul_mat_id`:
- Test 2: `b = [cols, 1, n_tokens]` → `b_ne1=1`, each slot reads the same input (`slot % 1 = 0`)
- Test 6: `b = [cols, n_exp_used, n_tokens]` → `b_ne1=n_exp_used`, each slot reads its own input

### 5.3 Graph: MoE block (Test 2/6)

```
expert_w [cols, rows, n_expert]            inp [cols, b_ne1, n_tokens]
    └─ mul_mat_id(expert_w, inp, ids) ──► moe [rows, n_exp_used, n_tokens]
ids [n_exp_used, n_tokens]                     │
                                               ▼
routing_w [1, n_exp_used, n_tokens] ───► mul ─► weighted [rows, n_exp_used, n_tokens]
                                               │
                          ┌─ view slot0 [rows, n_tokens] ─┐
                          └─ view slot1 [rows, n_tokens] ─┴─► add ─► agg [rows, n_tokens]
```

---

## 6. Bugs Found & Fixed During Validation

Running the tests surfaced **5 real bugs** in the backend. Each was located by isolating the failing op and diffing CPU vs LPU output.

### Bug 1: `ggml_guid_t` return type mismatch (compile error)

**Symptom**: `ggml-lpu.cpp` failed to compile: `cannot convert 'uint8_t*' to 'ggml_guid_t' (aka 'unsigned char (*)[16]')`.

**Root cause**: `ggml_guid_t` is a pointer to a 16-byte array, not a plain pointer. Returning `guid` (decay to `uint8_t*`) is the wrong type.

**Fix** ([ggml-lpu.cpp:488](../ggml/src/ggml-lpu/ggml-lpu.cpp#L488)):
```cpp
return &guid;   // was: return guid;
```

### Bug 2: `ggml_compute_forward_mul` not declared (compile error)

**Symptom**: `'ggml_compute_forward_mul' was not declared in this scope`.

**Root cause**: `ggml_compute_forward_add` lives in `ops.h`, but `ggml_compute_forward_mul/div/sub` were split into a separate header `binary-ops.h`.

**Fix** ([lpu_ops.cpp:17-18](../ggml/src/ggml-lpu/lpu_ops.cpp#L17-L18)):
```cpp
#include "../ggml-cpu/ops.h"
#include "../ggml-cpu/binary-ops.h"   // added
```

### Bug 3: Null `threadpool` → SEGV in sgemm

**Symptom**: Test 1 (FFN) crashed with `SEGV at 0x100` inside `ggml_threadpool_chunk_set`, called from the llamafile sgemm path.

**Root cause**: `single_thread_params()` left `p.threadpool = nullptr`. The CPU mul_mat path dereferences it for work-stealing.

**Fix** ([lpu_ops.cpp:78-86](../ggml/src/ggml-lpu/lpu_ops.cpp#L78-L86)): lazily create a single-thread `ggml_threadpool` and assign it to every `compute_params`.

### Bug 4: Stale chunk counter → FFN diff 0.52

**Symptom**: After bug 3, FFN produced `max-abs-diff = 0.524` (wrong, not a crash).

**Root cause**: The threadpool's `current_chunk` work-stealing counter persists across op calls. The first `mul_mat` left it at a non-zero value; the second `mul_mat` then started reading from that offset and skipped early chunks.

**Diagnosis**: diff was large but no crash → looked like skipped work, not memory corruption. Confirmed by the pattern: every op after the first was affected.

**Fix** ([lpu_ops.cpp:96](../ggml/src/ggml-lpu/lpu_ops.cpp#L96)): reset the counter before each op:
```cpp
ggml_threadpool_chunk_set(tp, 0);
```
After this, FFN went to `max-abs-diff = 0.0`.

### Bug 5: `set_tensor` wrote to buffer base, not tensor slot → wrong MoE

**Symptom**: MoE tests crashed with `GGML_ASSERT(ex >= 0 && ex < n_expert)` — the `ids` tensor read as garbage.

**Root cause**: The `set_tensor` callback wrote to `buf_ctx->data + offset` (the buffer base). But `offset` passed in is `0` and `tensor->data` already points to the tensor's allocated slot. So **all inputs were written to offset 0 of the buffer, overwriting each other** — only the last-uploaded tensor's data survived, at the wrong location. The `ids` tensor ended up with garbage from another tensor's bytes.

**Fix** ([ggml-lpu.cpp:106-117](../ggml/src/ggml-lpu/ggml-lpu.cpp#L106-L117)): anchor the copy at `tensor->data`:
```cpp
lpu_h2d_impl((char *)tensor->data + offset, data, size, nullptr);   // was: ctx->data + offset
```
This is the bug that had been masking correctness for all ops — once fixed, argsort/get_rows/soft_max/activations/norm all went to exact match.

### Bug 6: `cpu_sgemm` output layout → MoE diff 0.44 / 1.11

**Symptom**: After bug 5, only the two MoE tests failed (`0.442` and `1.112`). Everything else passed exactly.

**Diagnosis** (this is the key debugging story):

1. Reproduced Test 2 standalone: 124/192 elements differed.
2. Stripped the graph to **just `mul_mat_id`** (no `mul`/`view`/`add`) — still failed → bug is in `lpu_op_mul_mat_id`, not the surrounding ops.
3. Printed per-slot, per-token rows. Discovered: **token 1 (single-hit experts) was correct; only token 0 (where two tokens shared an expert) was wrong.** This ruled out a layout/indexing bug and pointed at **output-buffer aliasing** when `n_hits > 1`.
4. Confirmed: LPU row `r1` held CPU row `r0`'s previous value — classic overlapping writes.

**Root cause**: `cpu_sgemm` wrote `C[m * N + n]` (i.e. `[rows, n_hits]` row-major), but the scatter loop read `out[h * rows .. h*rows+rows)` — expecting `[n_hits, rows]` layout. With `n_hits > 1`, result columns interleaved and overwrote each other.

**Fix** ([lpu_ops.cpp:117-128](../ggml/src/ggml-lpu/lpu_ops.cpp#L117-L128)): write `C[n * M + m]` so each hit's output row is contiguous, matching the scatter loop:
```cpp
C[(int64_t)n * M + m] = acc;   // was: C[m * N + n]
```

After this, Test 2 → `0.0` and Test 6 → `1.19e-7` (F32 round-off, well under tolerance). **All 32 assertions passed.**

### Bug summary table

| # | Bug | Type | How found | Fix location |
|---|-----|------|-----------|--------------|
| 1 | `guid` return type | compile | build | `ggml-lpu.cpp:488` |
| 2 | `_mul` undeclared | compile | build | `lpu_ops.cpp:17` |
| 3 | null threadpool | SEGV | run test 1 | `lpu_ops.cpp:78-86` |
| 4 | stale chunk counter | wrong result (0.52) | run test 1 | `lpu_ops.cpp:96` |
| 5 | `set_tensor` base offset | wrong result / assert | run test 2 | `ggml-lpu.cpp:106-117` |
| 6 | `cpu_sgemm` layout | wrong result (0.44/1.11) | isolate + diff | `lpu_ops.cpp:117-128` |

Bugs 1–2 were trivial. **Bugs 3–6 were silent correctness bugs** that only surfaced because the tests compared against a golden reference — exactly what the test suite was built to catch.

---

## 7. How to Build & Run

### Stub SDK (no real hardware needed)

```bash
mkdir -p /tmp/lpu_stub/{lib,include}
cc -shared -o /tmp/lpu_stub/lib/liblpu_runtime.so -x c /dev/null
cc -shared -o /tmp/lpu_stub/lib/liblpu_nn.so      -x c /dev/null

cmake -B build -DGGML_LPU=ON -DLPU_INSTALL_DIR=/tmp/lpu_stub -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test-lpu-backend -j$(nproc)
```

### Run

```bash
./build/bin/test-lpu-backend
# or via CTest:
cd build && ctest -R test-lpu-backend --output-on-failure -V
```

### Pass criteria

- Every `LPU_CHECK` line prints `PASS`
- `max-abs-diff <= 1e-4` for all F32 graphs (`argsort` must match exactly)
- All op counters checked are `> 0`
- Exit code `0`

---

## 8. Final Test Results

```
=== test-lpu-backend ===
Registered backends (2):
  [0] LPU0
  [1] CPU
PASS  LPU device is registered
PASS  CPU backend initialised
PASS  LPU backend initialised

--- Test 1: FFN block ---
  FFN max-abs-diff: 0.000000e+00  (tol 1e-04)        PASS
--- Test 2: MoE block (broadcast b, 4 experts, top-2, 3 tokens) ---
  MoE(broadcast) max-abs-diff: 0.000000e+00  (tol 1e-04)   PASS
--- Test 3: Attention ops (argsort, get_rows, soft_max+mask) ---
  argsort: exact match                          PASS
  get_rows max-abs-diff: 0.00e+00               PASS
  soft_max+mask max-abs-diff: 0.00e+00          PASS
--- Test 4: Activations (gelu, relu, swiglu, geglu) ---
  gelu / relu / swiglu / geglu: 0.00e+00        PASS (×4)
--- Test 5: Norm and Scale ops ---
  norm / rms_norm / scale: 0.00e+00             PASS (×3)
--- Test 6: MoE large (non-broadcast b, 8 experts, top-4, 8 tokens) ---
  MoE-large(non-broadcast) max-abs-diff: 1.19e-07  (tol 1e-04)   PASS

=== Results: 32 passed, 0 failed ===
```

**32/32 assertions pass, exit code 0.** All ops produce CPU-equivalent output; the single non-zero diff (1.19e-7 in Test 6) is F32 accumulation-order round-off in the larger MoE GEMM, four orders of magnitude under tolerance.

---

## Appendix: Debugging methodology (for future op bugs)

The MoE bug (Bug 6) was the hardest. The process that found it:

1. **Reproduce in isolation** — copy the exact test into a standalone `.cpp` to iterate fast without rebuilding the whole test binary.
2. **Reduce the graph** — strip `mul`/`view`/`add` until only the suspect op remains. If it still fails, the bug is in that op.
3. **Diff by position** — print CPU vs LPU per-element and look for structure: which tokens/rows/slots are wrong? A regular pattern (every other row, only multi-hit cases) narrows the cause dramatically.
4. **Test hypotheses cheaply** — e.g. "is it a layout swap?" → check if `lpu[slot,token]` matches `cpu[token,slot]` with a one-line loop, no code changes needed.
5. **Single-token cases are diagnostic** — if `n_hits=1` is always right and `n_hits>1` is wrong, the bug is in batching/buffer-aliasing, not in the per-element math.

This methodology — isolate, reduce, diff, hypothesize — is what turned a 0.44 max-diff failure into a 6-line fix.
