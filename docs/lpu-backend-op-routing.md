# LPU Backend Op Routing, Input Distribution & Result Collection

**Date**: 2026-06-30
**Question**: How does llama.cpp route a specific op to the LPU backend? How are inputs distributed and results collected?

This document traces the complete data path from "a `mul_mat_id` node appears in the graph" to "the result tensor is filled on the LPU backend", covering the **scheduler → split → copy → compute → recycle** pipeline.

---

## 1. The Big Picture

llama.cpp's compute graph is **not** executed op-by-op on a single backend. Instead, a **scheduler** (`ggml_backend_sched`) partitions the graph into **splits**, each assigned to one backend. An op ends up on the LPU when two things agree:

1. The LPU's `supports_op()` returns true for that op
2. The tensor-placement constraints make LPU the best (or only) choice

The data flow:

```
                ┌─────────────────────────────────────────────┐
                │  ggml_cgraph (built by llama-graph.cpp)     │
                │  nodes: ... rms_norm, mul_mat_id, mul ...   │
                └──────────────────────┬──────────────────────┘
                                       │
                       ┌───────────────▼────────────────┐
                       │  ggml_backend_sched (scheduler)│
                       │  - assign each node a backend  │
                       │  - split graph by backend      │
                       │  - insert tensor copies        │
                       └───────────────────────┬────────┘
                                               │
            ┌──────────────────────────────────┼──────────────────────────────────┐
            ▼                                  ▼                                  ▼
    ┌──────────────┐                  ┌──────────────┐                  ┌──────────────┐
    │  CPU split   │                  │  LPU split   │                  │  CPU split   │
    │ (nodes that  │  ──copy inputs──►│ (nodes that  │  ──copy back────►│ (next nodes) │
    │  CPU owns)   │                  │  LPU owns)   │  ──in results──► │              │
    └──────────────┘                  └──────────────┘                  └──────────────┘
                                               │
                                               ▼
                                    ggml_backend_lpu_graph_compute
                                      → per-node lpu_op_* dispatch
                                      → ggml_compute_forward_* (CPU delegate)
                                      → writes into dst->data (LPU buffer)
```

---

## 2. Step 1: Building the Graph

The inference context (`llama-context.cpp`) builds a `ggml_cgraph` representing one forward pass. Each node is a `ggml_tensor` with:
- `op` — the operation (e.g. `GGML_OP_MUL_MAT_ID`)
- `src[0..3]` — input tensors
- `ne[]`/`nb[]` — shape and strides
- `buffer` — **initially NULL** (no memory yet, just metadata)
- `data` — initially NULL

The graph is just a dependency DAG at this point; **no backend is chosen yet**. Building is backend-agnostic.

---

## 3. Step 2: Scheduler Assigns Each Node to a Backend

When the context runs the graph, it calls `ggml_backend_graph_compute(sched, graph)` on the scheduler. The scheduler's job: decide a backend for every node.

### 3.1 The assignment function: `ggml_backend_sched_backend_id_from_cur`

([ggml/src/ggml-backend.cpp:878](../ggml/src/ggml-backend.cpp#L878))

For each tensor node, the scheduler tries these rules **in priority order**:

#### Rule A — Pre-allocated tensor follows its buffer
```cpp
int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
```
If the tensor already has a `buffer` (e.g. a weight loaded at startup into an LPU buffer), the scheduler finds a backend that:
- supports the buffer **type** (`ggml_backend_supports_buft`)
- supports the **op** (`ggml_backend_supports_op`)

→ This is how **weights placed on LPU memory drag their consuming ops onto the LPU**.

#### Rule B — View follows its source
```cpp
if (tensor->view_src != NULL)
    cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
```
A `view` tensor inherits the backend of the tensor it views into.

#### Rule C — Graph input → CPU
```cpp
if (tensor->flags & GGML_TENSOR_FLAG_INPUT)
    return sched->n_backends - 1;   // last backend (assumed CPU)
```
User-supplied inputs (token IDs, activations) start on CPU.

#### Rule D — Weight-driven placement with offload  ← **the key rule for LPU**
```cpp
for (src in tensor->src[]):
    if (src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS):
        src_backend_id = backend_from_buffer(src, tensor);
        // if the weight is on CPU but a higher-prio backend wants to offload:
        if (sched->op_offload && src_backend_id == CPU && ggml_backend_buffer_is_host(src->buffer)):
            for (b in higher-priority backends):   // LPU before CPU
                if (supports_op(b, tensor) && offload_op(b, tensor)):
                    return b;   // ← op goes to LPU even though weights are on host
        return src_backend_id;
```

This is exactly the LPU's entry point. The LPU backend sets a **low offload threshold** via `offload_op()` ([ggml-lpu.cpp:416](../ggml/src/ggml-lpu/ggml-lpu.cpp#L416)):

```cpp
static bool ggml_backend_lpu_device_offload_op(dev, op) {
    int threshold = ctx->op_offload_min_batch_size;   // default 32
    switch (op->op) {
        case GGML_OP_MUL_MAT:    return op->ne[1] >= threshold;  // dense GEMM batch
        case GGML_OP_MUL_MAT_ID: return op->ne[2] >= threshold;  // MoE token count
        default: return false;
    }
}
```

So: **a `mul_mat_id` with ≥32 tokens, where the expert weights sit in a host (CPU) buffer marked WEIGHTS, gets pulled onto the LPU** — even though the weights haven't been copied to LPU memory yet. The scheduler will then insert the copy.

### 3.2 What `supports_op` admits ([ggml-lpu.cpp:331](../ggml/src/ggml-lpu/ggml-lpu.cpp#L331))

`offload_op` being true isn't enough; `supports_op` must also accept the op:

| Op | Accepted when |
|----|---------------|
| `MUL_MAT_ID`, `ARGSORT`, `GET_ROWS`, `MUL_MAT` | `src[0]->type` is F32 or F16 |
| `UNARY` | silu / gelu / relu / sigmoid / tanh |
| `GLU` | swiglu / geglu |
| `RMS_NORM`, `NORM`, `MUL`, `ADD`, `SCALE`, `SOFT_MAX` | always |
| `RESHAPE/VIEW/PERMUTE/TRANSPOSE/CONT` | always (metadata-only, no compute) |
| anything else | false → op stays on CPU |

### 3.3 Buffer-type compatibility

`ggml_backend_lpu_device_supports_buft` ([ggml-lpu.cpp:409](../ggml/src/ggml-lpu/ggml-lpu.cpp#L409)) accepts:
- LPU device buffer types (`ggml_backend_buft_is_lpu`)
- host buffers (`ggml_backend_buft_is_host`)

Accepting host buffers is essential — it's what lets the LPU consume weights that still live in CPU memory during the offload case (Rule D).

---

## 4. Step 3: The Scheduler Splits the Graph

Once every node has a backend id, `ggml_backend_sched_split_graph` ([ggml-backend.cpp:1014](../ggml/src/ggml-backend.cpp#L1014)) walks the node list and groups **maximal contiguous runs** of nodes on the same backend into **splits**.

```
nodes:  [rms_norm(CPU)] [mul_mat_id(LPU)] [mul(LPU)] [add(CPU)]
                 ▲              ▲▲           ▲            ▲
        split 0: CPU      split 1: LPU   split 2: CPU
```

For each split, the scheduler records:
- `backend_id` — which backend runs it
- `i_start, i_end` — node range
- `inputs[]` — **tensors produced by earlier splits that this split needs** (these are the cross-backend boundaries that require copies)
- `node_start[]` — per-node copy targets if an input needs to be mirrored

The split's nodes are packaged into a **sub-graph** (`split->graph`) that the backend executes as a unit.

### Why splits matter for data movement

The boundaries between splits are exactly where **tensor copies across backends** happen. If split 0 (CPU) produces tensor `T` and split 1 (LPU) consumes `T`, then `T` must be copied CPU→LPU before split 1 runs. The scheduler's `inputs[]` list is precisely that set of tensors.

---

## 5. Step 4: Input Distribution — Copying Tensors to the LPU

This is `ggml_backend_sched_compute_splits` ([ggml-backend.cpp:1541](../ggml/src/ggml-backend.cpp#L1541)). For each split, before computing, it copies every input tensor to the split's backend.

### 5.1 The general copy path

```cpp
for (input_id in split->n_inputs):
    input       = split->inputs[input_id];            // original tensor (on producer backend)
    input_cpy   = tensor_copy(input, split_backend_id, copy_id);  // mirror on LPU
    input_backend = backend_of(input);

    if (input is a user INPUT flag):
        ggml_backend_tensor_copy(input, input_cpy);   // sync copy now
    else:
        // wait for producer, then async or sync copy
        if (can cpy_tensor_async):
            cpy_tensor_async(input_backend, split_backend, input, input_cpy);
        else:
            sync(input_backend);
            ggml_backend_tensor_copy(input, input_cpy);
```

`tensor_copy(input, lpu_id, ...)` returns the **mirror tensor** allocated in the LPU buffer. The scheduler maintains these mirrors in `hv_tensor_copies[]` keyed by (tensor hash, backend, copy slot), so repeated uses reuse the same copy.

### 5.2 The MoE expert-only optimization (LPU-relevant)

For `MUL_MAT_ID` specifically, the scheduler has a **special case** ([ggml-backend.cpp:1576-1660](../ggml/src/ggml-backend.cpp#L1576)):

When the input is the expert-weight tensor (`src[0]` of `mul_mat_id`) and it's a host-side WEIGHTS buffer, instead of copying **all** experts, the scheduler:

1. Reads the `ids` tensor (`src[2]`) to see which experts are actually used this step
2. Builds a bitset `used_ids[0..n_expert-1]`
3. Copies **only the used experts**, grouping consecutive ones into batched copies:
   ```cpp
   copy_experts(first_id, last_id):
       expert_offset = first_id * expert_size;
       size = (last_id - first_id + 1) * expert_size;
       ggml_backend_tensor_set_async(split_backend, input_cpy,
           (uint8_t*)input->data + expert_offset, expert_offset, size + padding);
   ```

For a 256-expert MoE with top-8 routing and 32 tokens, this copies ~8 experts instead of 256 — a 32× bandwidth saving. This is the scheduler being MoE-aware because `MUL_MAT_ID` is the op that benefits most.

### 5.3 What the copy physically does (stub backend)

In stub mode, "copy to LPU" is a `memcpy` because LPU buffers are host memory. The LPU buffer's `set_tensor` callback ([ggml-lpu.cpp:106](../ggml/src/ggml-lpu/ggml-lpu.cpp#L106)) is what runs:

```cpp
set_tensor(buf, tensor, data, offset, size):
    lpu_h2d_impl((char*)tensor->data + offset, data, size, nullptr);
    // stub: lpu_h2d_impl = memcpy
```

`tensor->data` points to the mirror tensor's slot in the LPU buffer; `data` is the source on the producer backend. With real hardware, `lpu_h2d_impl` becomes a DMA transfer to device memory.

### 5.4 The upload contract

After Step 5, for every LPU-split node:
- All its `src[]` inputs that live on another backend have a **mirror copy** in the LPU buffer
- `node->src[i]->data` pointers are rewritten (via the scheduler's graph copy) to point at the mirror tensors
- The node is ready to compute entirely from LPU-local memory

---

## 6. Step 5: Compute — `ggml_backend_lpu_graph_compute`

Now the LPU backend actually runs its split. The entry point is `ggml_backend_lpu_graph_compute` ([ggml-lpu.cpp:203](../ggml/src/ggml-lpu/ggml-lpu.cpp#L203)):

```cpp
static ggml_status ggml_backend_lpu_graph_compute(backend, cgraph) {
    auto * ctx = backend->context;
    lpu_set_device_impl(ctx->device);

    for (i in 0..cgraph->n_nodes):
        node = cgraph->nodes[i];
        if (empty(node) || node->op == GGML_OP_NONE) continue;

        status = ggml_backend_lpu_compute_forward(ctx, node);   // ← per-op dispatch
        if (status != SUCCESS) return status;

    lpu_stream_sync_impl(ctx->stream);   // ensure async work done
    return SUCCESS;
}
```

This is a **simple sequential loop** over the split's nodes — no further scheduling, no re-dispatch. Each node goes to `ggml_backend_lpu_compute_forward` ([lpu_ops.cpp:134](../ggml/src/ggml-lpu/lpu_ops.cpp#L134)), the big `switch`:

```cpp
switch (dst->op) {
    case GGML_OP_MUL_MAT:    return lpu_op_mul_mat(ctx, dst);
    case GGML_OP_MUL_MAT_ID: return lpu_op_mul_mat_id(ctx, dst);
    case GGML_OP_RMS_NORM:   return lpu_op_rms_norm(ctx, dst);
    /* ... etc ... */
    case GGML_OP_VIEW: case GGML_OP_RESHAPE: ... return SUCCESS;  // metadata-only
    default: return FAILED;
}
```

Each `lpu_op_*` reads its inputs from `dst->src[i]->data` (now pointing at LPU-local mirrors), computes, and **writes the result into `dst->data`** — which is in the LPU buffer, because the scheduler allocated `dst` on the LPU too (it's a node in the LPU split).

### The delegation to CPU

For everything except `mul_mat_id`, the "compute" is a call into ggml's internal CPU function:

```cpp
ggml_status lpu_op_rms_norm(ctx, dst) {
    g_lpu_op_stats.rms_norm++;
    auto p = single_thread_params();
    ggml_compute_forward_rms_norm(&p, dst);   // CPU kernel, but reads/writes LPU buffer
    return SUCCESS;
}
```

Since LPU stub buffers are host memory, the CPU kernel operates on the same bytes the LPU "owns" — no extra copy. The result lands directly in `dst->data` (LPU buffer).

For `mul_mat_id`, the hand-rolled gather-GEMM-scatter ([lpu_ops.cpp:204](../ggml/src/ggml-lpu/lpu_ops.cpp#L204)) does the same: reads `as`/`b`/`ids` from LPU-local memory, writes `dst` into LPU-local memory.

---

## 7. Step 6: Result Collection — Copying Back Across Splits

After the LPU split finishes, control returns to the scheduler loop. The next split (typically CPU again, e.g. the residual `add` after MoE) may consume the LPU's output.

The same input-copy machinery runs, in reverse:

1. The next split's `inputs[]` includes the LPU-produced tensor `T`
2. The scheduler finds `T`'s backend = LPU, and the next split's backend = CPU
3. It creates a mirror `T_cpy` in the **CPU** buffer
4. `ggml_backend_tensor_copy(T, T_cpy)` runs — which calls the LPU buffer's `get_tensor`:
   ```cpp
   get_tensor(buf, tensor, data, offset, size):
       lpu_d2h_impl(data, (const char*)tensor->data + offset, size, nullptr);
       // stub: lpu_d2h_impl = memcpy
   ```
5. The next split's node now has `src[i]->data` pointing at `T_cpy` (CPU), and computes normally

### Synchronization

The scheduler uses **events** (`sched->events[backend][copy]`) to avoid overwriting a copy before the consumer finishes. In stub mode everything is synchronous (`lpu_stream_sync_impl` is a no-op), but the structure supports real async pipelining:
- After a split computes, it records an event
- Before the next split reuses a copy slot, it waits on that event

### Final output

The very last split's output tensor (e.g. logits) is the graph's terminal node. The inference context reads it back to host with `ggml_backend_tensor_get`, completing the round trip.

---

## 8. End-to-End Trace: A MoE Layer on LPU

Concrete walkthrough for one MoE layer with 32 tokens, expert weights on host:

```
1. graph built:
   embed ─► rms_norm ─► mul_mat(gate,host) ─► argsort ─► mul_mat_id(expert_w,host, ids)
                                                          │
                                                          ▼
                                                        mul(routing_w)
                                                          │
                                                          ▼
                                                        add(residual)

2. scheduler assigns (Rule D, offload_op=true for ≥32 tokens):
   - rms_norm          → CPU   (input on CPU)
   - mul_mat(gate)     → LPU   (offload, weight on host, batch≥32)
   - argsort           → LPU   (follows mul_mat output, supports_op=true)
   - mul_mat_id        → LPU   (offload, expert_w on host, ne[2]=32≥32)  ★
   - mul(routing_w)    → LPU   (follows mul_mat_id)
   - add(residual)     → CPU   (residual on CPU)

3. splits:
   split 0 (CPU):  rms_norm
   split 1 (LPU):  mul_mat, argsort, mul_mat_id, mul
   split 2 (CPU):  add

4. compute split 1:
   a. copy inputs to LPU:
      - gate weights:        full copy host→LPU          (set_tensor)
      - expert weights:      ★ PARTIAL copy — only used experts (MoE opt)
      - routing_w:           full copy host→LPU
      - rms_norm output:     copy CPU→LPU (split boundary)
   b. lpu_graph_compute loops:
      - lpu_op_mul_mat      → ggml_compute_forward_mul_mat   (CPU delegate)
      - lpu_op_argsort      → ggml_compute_forward_argsort
      - lpu_op_mul_mat_id   → gather-GEMM-scatter (hand-rolled)
      - lpu_op_mul          → ggml_compute_forward_mul
      each writes dst->data in LPU buffer
   c. lpu_stream_sync

5. compute split 2:
   - copy mul output LPU→CPU  (get_tensor / d2h)
   - CPU add reads it, adds residual → final logits

6. context reads logits back to host
```

The **only** LPU-specific code in this whole flow is:
- `supports_op` / `offload_op` / `supports_buft` (decision)
- `set_tensor` / `get_tensor` / `cpy_tensor` (data movement)
- `graph_compute` → `compute_forward` → `lpu_op_*` (compute)

Everything else — splitting, mirroring, synchronization, the MoE expert-partial-copy — is generic scheduler code in `ggml-backend.cpp` that works for any backend.

---

## 9. Summary Table: Who Does What

| Phase | Component | File | LPU-specific? |
|-------|-----------|------|---------------|
| Build graph | llama context | `llama-context.cpp`, `llama-graph.cpp` | No |
| Assign node→backend | scheduler | `ggml-backend.cpp:878` | No (calls LPU's `supports_op`/`offload_op`) |
| Decide LPU eligibility | `supports_op`, `offload_op` | `ggml-lpu.cpp:331,416` | **Yes** |
| Split graph | scheduler | `ggml-backend.cpp:1014` | No |
| Copy inputs to LPU | scheduler + buffer callbacks | `ggml-backend.cpp:1541`, `ggml-lpu.cpp:106` | Callback yes, orchestration no |
| MoE expert-only copy | scheduler (special case) | `ggml-backend.cpp:1576` | No (triggered by `MUL_MAT_ID`) |
| Run LPU split | `graph_compute` | `ggml-lpu.cpp:203` | **Yes** |
| Per-op dispatch | `compute_forward` switch | `lpu_ops.cpp:134` | **Yes** |
| Actual math | `lpu_op_*` → CPU delegate | `lpu_ops.cpp` | **Yes** (swappable for real kernels) |
| Copy results back | scheduler + `get_tensor` | `ggml-backend.cpp`, `ggml-lpu.cpp:113` | Callback yes |
| Sync | events / `synchronize` | `ggml-backend.cpp`, `ggml-lpu.cpp` | Minimal |

The LPU backend is a **thin plugin**: ~5 decision/callback functions tell the generic scheduler how to use it, and ~12 op kernels do the work. The scheduler handles all cross-backend data movement, split management, and synchronization — the backend just needs to honestly report what it supports and correctly move bytes in/out of its buffers.

---

## 10. Key Insight: Placement Follows Weights + Offload Policy

The single most important rule for understanding LPU routing:

> **An op runs on the LPU when (a) the LPU admits the op via `supports_op`, AND (b) either the op's weight tensor lives in an LPU buffer, OR the weight is on host and `offload_op` says the batch is large enough to justify pulling the op (and copying the weight) to the LPU.**

This means LPU placement is **weight-driven**, not op-driven. You don't write "run this mul_mat on LPU"; you place expert/FFN weights where you want them (or mark them for offload), and the scheduler routes the consuming ops accordingly. The `offload_op` threshold (`GGML_LPU_OFFLOAD_MIN_BATCH`, default 32) is the knob that controls how eagerly small batches get pulled to the LPU versus staying on CPU.
