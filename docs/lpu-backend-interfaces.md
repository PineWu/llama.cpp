# llama.cpp Backend Interface Structures

**Date**: 2026-06-30
**Scope**: Catalog of every interface structure (the `..._i` vtables) in llama.cpp's ggml backend abstraction, with detailed explanations of each method.

llama.cpp's multi-backend system is built on **interface structures** — C structs of function pointers that act as vtables. Every backend (CPU, CUDA, Metal, Vulkan, LPU, ...) implements the same set of these interfaces; the core engine calls through them without knowing which backend is running.

This document lists all of them, shows where they are defined, and explains what each function does.

---

## Table of Contents

1. [The Interface Hierarchy](#1-the-interface-hierarchy)
2. [`ggml_backend_reg_i` — Backend Registry](#2-ggml_backend_reg_i--backend-registry)
3. [`ggml_backend_device_i` — Device](#3-ggml_backend_device_i--device)
4. [`ggml_backend_i` — Backend (Compute Stream)](#4-ggml_backend_i--backend-compute-stream)
5. [`ggml_backend_buffer_type_i` — Buffer Type (Allocator)](#5-ggml_backend_buffer_type_i--buffer-type-allocator)
6. [`ggml_backend_buffer_i` — Buffer (Allocated Memory)](#6-ggml_backend_buffer_i--buffer-allocated-memory)
7. [Non-Interface Supporting Structures](#7-non-interface-supporting-structures)
8. [How They Compose: The LPU Example](#8-how-they-compose-the-lpu-example)
9. [Call Flow Summary](#9-call-flow-summary)

---

## 1. The Interface Hierarchy

All interfaces live in [ggml/src/ggml-backend-impl.h](../ggml/src/ggml-backend-impl.h). They form a five-layer hierarchy from most-abstract (registry) to most-concrete (allocated buffer):

```
ggml_backend_reg_i          ← registry: "this DLL/backend exists, here are its devices"
   │
   ▼  get_device(i)
ggml_backend_device_i       ← device: "LPU0 exists, supports these ops, has this memory"
   │
   ▼  init_backend()
ggml_backend_i              ← backend/stream: "compute this graph for me"
   │
   ▼  uses buffer types from the device
ggml_backend_buffer_type_i  ← buffer type: "how to allocate memory of a given kind"
   │
   ▼  alloc_buffer(size)
ggml_backend_buffer_i       ← buffer: "a concrete block of memory; here's how to read/write tensors in it"
```

Each layer holds a handle to the next:
- `ggml_backend_reg` holds devices
- `ggml_backend_device` creates a backend
- `ggml_backend` uses buffer types
- `ggml_backend_buffer_type` creates buffers
- `ggml_backend_buffer` holds the actual bytes

Each concrete struct (`ggml_backend_reg`, `ggml_backend_device`, `ggml_backend`, `ggml_backend_buffer_type`, `ggml_backend_buffer`) is `{ iface, ...handle fields..., context }` — i.e. the vtable plus a context pointer for backend-private state.

---

## 1.5 Interface (`_i`) vs Concrete (`_t`) — The Vtable / Instance Split

This is the single most important structural relationship to understand. For every layer there are **two** types: an interface struct (the vtable, `..._i`) and a concrete struct (the instance, `...`/`..._t`). They are deliberately decoupled — exactly like C++'s separation of an abstract base class from a derived object.

### The naming convention

| Interface (vtable) | Concrete (instance) | Public opaque handle |
|--------------------|---------------------|----------------------|
| `ggml_backend_reg_i` | `struct ggml_backend_reg` | `ggml_backend_reg_t` |
| `ggml_backend_device_i` | `struct ggml_backend_device` | `ggml_backend_dev_t` |
| `ggml_backend_i` | `struct ggml_backend` | `ggml_backend_t` |
| `ggml_backend_buffer_type_i` | `struct ggml_backend_buffer_type` | `ggml_backend_buffer_type_t` |
| `ggml_backend_buffer_i` | `struct ggml_backend_buffer` | `ggml_backend_buffer_t` |
| (`ggml_backend_event` has no `_i`) | `struct ggml_backend_event` | `ggml_backend_event_t` |

- The **`_i`** suffix marks an **interface** — a struct of function pointers. It has no data, only the vtable.
- The **unsuffixed** `struct ggml_backend_*` is the **concrete instance** — it *contains* an `_i` as its first meaningful field, plus instance data.
- The **`_t`** suffix is the **public opaque handle** — a pointer the rest of the codebase uses. Callers never see the concrete struct's fields; they only pass the handle around and call functions like `ggml_backend_graph_compute(handle, ...)` which internally dereference `handle->iface.method(...)`.

### The general shape

Every concrete struct follows the same skeleton:

```c
struct ggml_backend_<layer> {
    /* 1. vtable          */ struct ggml_backend_<layer>_i  iface;
    /* 2. linking handles */ <one or more parent/peer handles>
    /* 3. private state   */ void * context;
    /* 4. (layer-specific fields, e.g. size, guid, api_version) */
};
```

- **`iface`** — points to a `static const` vtable the backend fills in once. Many instances share one vtable.
- **linking handles** — back-pointers that let a layer find its creator/owner (e.g. a `ggml_backend` holds a `device` back-pointer; a `ggml_backend_buffer` holds its `buft`).
- **`context`** — the backend's private scratch (device handle, stream, allocator state). This is how the same vtable methods behave differently per instance.

### Pair-by-pair detail

#### `ggml_backend_reg_i`  ⇄  `ggml_backend_reg`

```c
// Interface (vtable) — defined in ggml-backend-impl.h:214
struct ggml_backend_reg_i {
    const char * (*get_name)(ggml_backend_reg_t reg);
    size_t       (*get_device_count)(ggml_backend_reg_t reg);
    ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t index);
    void *       (*get_proc_address)(ggml_backend_reg_t reg, const char * name);
};

// Concrete instance — defined in ggml-backend-impl.h:226
struct ggml_backend_reg {
    int   api_version;                     // ABI version for dynamic loading
    struct ggml_backend_reg_i  iface;      // ← the vtable
    void * context;                        // backend-private registry state
};
```

- **`iface`** holds the function pointers; e.g. for LPU it is `ggml_backend_lpu_reg_interface` ([ggml-lpu.cpp:471](../ggml/src/ggml-lpu/ggml-lpu.cpp#L471)).
- **`api_version`** lets the loader reject a `.so` built against an incompatible backend ABI (`GGML_BACKEND_API_VERSION`).
- **`context`** stores registry-private data (e.g. the loaded `dlopen` handle, SDK init state).
- `ggml_backend_reg_t` is `typedef struct ggml_backend_reg *` — callers only ever hold a pointer.

#### `ggml_backend_device_i`  ⇄  `ggml_backend_device`

```c
// Interface — ggml-backend-impl.h:160
struct ggml_backend_device_i { /* supports_op, offload_op, init_backend, get_buffer_type, ... */ };

// Concrete — ggml-backend-impl.h:204
struct ggml_backend_device {
    struct ggml_backend_device_i  iface;   // ← the vtable
    ggml_backend_reg_t            reg;     // ← back-pointer to owning registry
    void *                        context; // device-private state (mem size, dev index)
};
```

- The **back-pointer `reg`** lets code walk up the hierarchy (given a device, find which backend family it belongs to).
- **`context`** holds per-device facts the methods need (e.g. LPU stores `op_offload_min_batch_size` and device index here).
- For LPU, `iface = ggml_backend_lpu_device_interface` ([ggml-lpu.cpp:433](../ggml/src/ggml-lpu/ggml-lpu.cpp#L433)).

#### `ggml_backend_i`  ⇄  `ggml_backend`

```c
// Interface — ggml-backend-impl.h:105
struct ggml_backend_i { /* graph_compute, set/get_tensor_async, synchronize, ... */ };

// Concrete — ggml-backend-impl.h:142
struct ggml_backend {
    ggml_guid_t           guid;     // 16-byte type ID for ggml_backend_is_* checks
    struct ggml_backend_i iface;    // ← the vtable
    ggml_backend_dev_t    device;   // ← back-pointer to the device that created it
    void *                context;  // stream/command-queue private state
};
```

- **`guid`** is the one field without an `_i` analogue — it's instance identity, not behavior. `ggml_backend_is_lpu(backend)` compares `backend->guid == ggml_backend_lpu_guid()`.
- **`device`** back-pointer lets `ggml_backend_supports_op(backend, op)` delegate to `backend->device->iface.supports_op(...)` ([ggml-backend.cpp:455](../ggml/src/ggml-backend.cpp#L455)).
- **`context`** for LPU holds the `ggml_backend_lpu_context` (device index, stream, etc.).

#### `ggml_backend_buffer_type_i`  ⇄  `ggml_backend_buffer_type`

```c
// Interface — ggml-backend-impl.h:17
struct ggml_backend_buffer_type_i { /* alloc_buffer, get_alignment, is_host, ... */ };

// Concrete — ggml-backend-impl.h:31
struct ggml_backend_buffer_type {
    struct ggml_backend_buffer_type_i  iface;  // ← the vtable
    ggml_backend_dev_t                 device; // ← owning device
    void *                             context;// allocator-private state
};
```

- **`device`** back-pointer ties an allocator to the device whose memory it manages.
- A single device may expose **multiple** `buffer_type` instances sharing one vtable but different `context` (e.g. one for device memory, one for pinned host memory).
- LPU: `iface = ggml_backend_lpu_buffer_type_interface` ([ggml-lpu.cpp:151](../ggml/src/ggml-lpu/ggml-lpu.cpp#L151)).

#### `ggml_backend_buffer_i`  ⇄  `ggml_backend_buffer`

```c
// Interface — ggml-backend-impl.h:41
struct ggml_backend_buffer_i { /* set_tensor, get_tensor, clear, get_base, ... */ };

// Concrete — ggml-backend-impl.h:64
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i    iface;   // ← the vtable
    ggml_backend_buffer_type_t      buft;    // ← back-pointer to its type
    void *                          context; // the actual memory + private state
    size_t                          size;    // allocated bytes
    enum ggml_backend_buffer_usage  usage;   // WEIGHTS vs SCRATCH hint
};
```

- **`buft`** back-pointer lets code ask "what kind of memory is this?" (`buffer->buft->iface.is_host()`).
- **`context`** holds the pointer to the allocated bytes plus any layout metadata; `get_base()` returns it.
- **`size`** and **`usage`** are instance data with no `_i` analogue — `usage` in particular drives scheduler decisions (only WEIGHTS buffers get the MoE expert-partial-copy fast path).
- The buffer vtable is **not** a `static const` global — it is constructed inside `alloc_buffer` and passed to `ggml_backend_buffer_init()` ([ggml-lpu.cpp:137](../ggml/src/ggml-lpu/ggml-lpu.cpp#L137)), because the closures may capture per-allocation state (e.g. the `std::malloc`'d pointer in `context`).

### Why the split? (the design rationale)

1. **Many instances, one vtable.** There may be 4 GPU devices, each a separate `ggml_backend_device` instance, but all sharing one `ggml_backend_device_i` vtable. The vtable is `static const`; the instances differ only in `context`. This is plain C manual vtabling — no per-instance function-pointer duplication.

2. **Opaque public API.** The public header ([ggml-backend.h:24-30](../ggml/include/ggml-backend.h#L24)) only exposes `typedef struct ggml_backend_X * ggml_backend_X_t;` — pointers to incomplete types. Callers cannot touch `iface` or `context` directly; they must go through the wrapper functions (`ggml_backend_graph_compute`, `ggml_backend_tensor_set`, ...). This keeps the ABI stable: the concrete struct's layout can change as long as the wrapper signatures don't.

3. **Bidirectional navigation.** The back-pointers (`device` in a backend, `reg` in a device, `buft` in a buffer) let any layer walk up to its owner. This is essential for the scheduler: given a tensor's `buffer`, it asks `buffer->buft->device` to find the device, then `device->iface.supports_op` to decide routing — all without the caller knowing the concrete types.

4. **Separation of "what" from "how much".** The `_i` describes *behavior* (what ops are supported, how to copy); the concrete struct adds *instance data* (how much memory, which usage flag, which guid). Keeping behavior in a shared vtable and data per-instance is the textbook OOP-in-C pattern.

### The wiring at backend construction (LPU example)

This shows how the five pairs connect at runtime, in [ggml-lpu.cpp](../ggml/src/ggml-lpu/ggml-lpu.cpp):

```cpp
// ggml_backend_lpu_reg() builds the registry instance:
struct ggml_backend_reg reg;
reg.api_version = GGML_BACKEND_API_VERSION;
reg.iface       = ggml_backend_lpu_reg_interface;   // vtable → reg_i
reg.context     = /* registry-private state */;
//   reg holds nothing else; devices are created on demand via reg.iface.get_device()

// reg.iface.get_device(0) builds a device instance:
struct ggml_backend_device dev;
dev.iface    = ggml_backend_lpu_device_interface;    // vtable → device_i
dev.reg      = &reg;                                  // back-pointer ↑ to registry
dev.context  = /* device index, offload threshold */;

// dev.iface.init_backend() builds a backend instance:
struct ggml_backend backend;
backend.guid    = ggml_backend_lpu_guid();            // instance identity
backend.iface   = ggml_backend_lpu_interface;         // vtable → backend_i
backend.device  = &dev;                               // back-pointer ↑ to device
backend.context = /* stream, device idx */;

// dev.iface.get_buffer_type() builds a buffer-type instance:
struct ggml_backend_buffer_type bt;
bt.iface   = ggml_backend_lpu_buffer_type_interface;  // vtable → buffer_type_i
bt.device  = &dev;                                    // back-pointer ↑ to device
bt.context = /* allocator state */;

// bt.iface.alloc_buffer(size) builds a buffer instance:
struct ggml_backend_buffer buf;
buf.iface   = lpu_buf_iface;                          // vtable → buffer_i
buf.buft    = &bt;                                    // back-pointer ↑ to buffer_type
buf.context = /* the malloc'd pointer */;
buf.size    = size;
buf.usage   = GGML_BACKEND_BUFFER_USAGE_ANY;
//   buf is returned wrapped via ggml_backend_buffer_init(&bt, lpu_buf_iface, ctx, size)
```

Each instance holds its vtable (`iface`) plus a back-pointer to the layer above, forming a doubly-linked chain `buffer ⇄ buffer_type ⇄ device ⇄ reg`. The scheduler and engine code walk this chain through the opaque `_t` handles, calling `iface.method(self, ...)` at each step.

### One-picture summary

```
       ┌─────────────────────────────────────────────────────────────┐
       │  ggml_backend_reg          (instance)                       │
       │    .api_version, .iface ──► ggml_backend_reg_i  (vtable)    │
       │    .context                                                 │
       └──────────────▲──────────────────────────────────────────────┘
                      │ .reg (back-pointer)
       ┌──────────────┴──────────────────────────────────────────────┐
       │  ggml_backend_device       (instance)                       │
       │    .iface ──► ggml_backend_device_i  (vtable)               │
       │    .context                                                 │
       └──────────────▲──────────────────────────────────────────────┘
                      │ .device (back-pointer)
       ┌──────────────┴──────────────┐  ┌───────────────────────────┐
       │  ggml_backend (instance)    │  │  ggml_backend_buffer_type │
       │  .guid, .context            │  │    (instance)             │
       │  .iface ► ggml_backend_i    │  │  .iface ► ..._buffer_type_i│
       │  .device ────────────────►──┘  │  .device ──────────────►──┤ (back to device)
       └─────────────────────────────┘  │  .context                 │
                                         └────────────▲──────────────┘
                                                      │ .buft (back-pointer)
                                         ┌────────────┴──────────────┐
                                         │  ggml_backend_buffer      │
                                         │    (instance)             │
                                         │  .iface ► ..._buffer_i    │
                                         │  .context, .size, .usage  │
                                         └───────────────────────────┘
```

Read top-down: a registry owns devices; a device creates backends and buffer types; a buffer type creates buffers. Read bottom-up: a buffer knows its type, a type knows its device, a device knows its registry — via the back-pointer fields. The `_i` vtables hang off each instance's `iface` field and are shared across all instances of the same backend.

---

## 2. `ggml_backend_reg_i` — Backend Registry

**Defined**: [ggml-backend-impl.h:214-224](../ggml/src/ggml-backend-impl.h#L214)

A registry represents **one loaded backend** (one `.so`/`.a` for a hardware family). At process startup, `ggml-backend-reg.cpp` discovers all registries and calls `get_device_count` / `get_device` to enumerate devices.

```c
struct ggml_backend_reg_i {
    const char * (*get_name)         (ggml_backend_reg_t reg);
    size_t       (*get_device_count) (ggml_backend_reg_t reg);
    ggml_backend_dev_t (*get_device) (ggml_backend_reg_t reg, size_t index);
    void *       (*get_proc_address) (ggml_backend_reg_t reg, const char * name);  // optional
};
```

| Method | Purpose |
|--------|---------|
| `get_name` | Human-readable backend family name, e.g. `"LPU"`, `"CUDA"`. |
| `get_device_count` | How many devices this backend exposes (e.g. 1 CPU, 4 GPUs). |
| `get_device` | Returns the `ggml_backend_dev_t` handle for device `index`. |
| `get_proc_address` | Optional. Extension hook — backends can expose non-standard functions by name (used for backend-specific APIs not in the core interface). |

**LPU impl**: [ggml-lpu.cpp:471](../ggml/src/ggml-lpu/ggml-lpu.cpp#L471) — returns name `"LPU"`, count `1`, the single `LPU0` device.

---

## 3. `ggml_backend_device_i` — Device

**Defined**: [ggml-backend-impl.h:160-202](../ggml/src/ggml-backend-impl.h#L160)

A device is a physical or logical unit (one GPU, the CPU, one LPU chip). It is the **decision layer**: it reports capabilities and creates the compute backend. This is where `supports_op` lives — the function the scheduler queries to route ops.

```c
struct ggml_backend_device_i {
    const char * (*get_name)        (ggml_backend_dev_t dev);
    const char * (*get_description) (ggml_backend_dev_t dev);
    void         (*get_memory)      (ggml_backend_dev_t dev, size_t * free, size_t * total);
    enum ggml_backend_dev_type (*get_type)(ggml_backend_dev_t dev);
    void         (*get_props)       (ggml_backend_dev_t dev, struct ggml_backend_dev_props * props);
    ggml_backend_t (*init_backend)  (ggml_backend_dev_t dev, const char * params);
    ggml_backend_buffer_type_t (*get_buffer_type)    (ggml_backend_dev_t dev);
    ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);  // optional
    ggml_backend_buffer_t (*buffer_from_host_ptr)    (ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);  // optional
    bool (*supports_op)   (ggml_backend_dev_t dev, const struct ggml_tensor * op);
    bool (*supports_buft) (ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
    bool (*offload_op)    (ggml_backend_dev_t dev, const struct ggml_tensor * op);  // optional
    ggml_backend_event_t (*event_new)        (ggml_backend_dev_t dev);  // optional
    void                 (*event_free)       (ggml_backend_dev_t dev, ggml_backend_event_t event);  // optional
    void                 (*event_synchronize)(ggml_backend_dev_t dev, ggml_backend_event_t event);  // optional
};
```

| Method | Purpose |
|--------|---------|
| `get_name` | Short identifier: `"LPU0"`, `"CUDA0"`, `"CPU"`. |
| `get_description` | Longer description: model name, e.g. `"LPU-Gen1"`, `"NVIDIA GeForce RTX 4090"`. |
| `get_memory` | Free/total device memory in bytes (0,0 if not reportable). |
| `get_type` | Device class: `GGML_BACKEND_DEVICE_TYPE_CPU/GPU/1_2/2_0/...`. The scheduler uses this to pick a default backend (GPU preferred). |
| `get_props` | Fills a `ggml_backend_dev_props` struct bundling name/desc/type/memory/caps — convenience for one-shot queries. |
| `init_backend` | **Creates the compute backend** (`ggml_backend_t`) for this device. The backend is the "stream" that runs graphs. |
| `get_buffer_type` | The device's preferred device-memory buffer type. Tensors allocated here live in device memory. |
| `get_host_buffer_type` | Optional pinned-host-memory buffer type for fast H2D/D2H transfers. NULL if unsupported. |
| `buffer_from_host_ptr` | Optional. Wrap an existing host pointer (e.g. mmap'd model file) as a buffer. Used by memory-mapped model loading. |
| `supports_op` | **The routing decision.** Returns true if this device can compute `op` given its op type, source tensor types/shapes. The scheduler calls this to assign nodes to backends. |
| `supports_buft` | Returns true if this device can consume tensors stored in `buft` (e.g. LPU accepts its own buft + host buft). Lets a device read host-resident weights. |
| `offload_op` | Optional. Returns true if this device **wants** to pull an op onto itself even though the weights are in an incompatible buffer — used for expensive ops (large GEMMs) worth the copy cost. This is the LPU's `MUL_MAT`/`MUL_MAT_ID` batch-threshold hook. |
| `event_new` / `event_free` / `event_synchronize` | Optional synchronization primitives for async pipelines (see `ggml_backend_event`). |

**LPU impl**: [ggml-lpu.cpp:433](../ggml/src/ggml-lpu/ggml-lpu.cpp#L433). `supports_op` is the big switch admitting MoE/FFN ops ([ggml-lpu.cpp:331](../ggml/src/ggml-lpu/ggml-lpu.cpp#L331)); `offload_op` returns true for GEMMs ≥ `op_offload_min_batch_size` ([ggml-lpu.cpp:416](../ggml/src/ggml-lpu/ggml-lpu.cpp#L416)).

---

## 4. `ggml_backend_i` — Backend (Compute Stream)

**Defined**: [ggml-backend-impl.h:105-140](../ggml/src/ggml-backend-impl.h#L105)

The backend is the **execution handle** — think "a CUDA stream" or "an LPU command queue". Once the device creates it via `init_backend`, the engine calls `graph_compute` to run a (sub)graph on it. It also owns async tensor I/O.

```c
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void (*free)(ggml_backend_t backend);

    void (*set_tensor_async)   (ggml_backend_t, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor_async)   (ggml_backend_t, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    void (*set_tensor_2d_async)(...);  // optional
    void (*get_tensor_2d_async)(...);  // optional
    bool (*cpy_tensor_async)   (ggml_backend_t src, ggml_backend_t dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

    void (*synchronize)(ggml_backend_t backend);  // optional, required if async ops supported

    ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t, const struct ggml_cgraph *);  // optional, unused
    void (*graph_plan_free)   (ggml_backend_t, ggml_backend_graph_plan_t);                          // optional
    void (*graph_plan_update) (ggml_backend_t, ggml_backend_graph_plan_t, const struct ggml_cgraph *); // optional
    enum ggml_status (*graph_plan_compute)(ggml_backend_t, ggml_backend_graph_plan_t);              // optional

    enum ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);

    void (*event_record)(ggml_backend_t, ggml_backend_event_t event);  // optional
    void (*event_wait)  (ggml_backend_t, ggml_backend_event_t event);  // optional
    void (*graph_optimize)(ggml_backend_t, struct ggml_cgraph *);      // optional
};
```

| Method | Purpose |
|--------|---------|
| `get_name` | Name of this backend instance. |
| `free` | Tear down the backend/stream. |
| `set_tensor_async` | Schedule a host→device write: copy `data` into `tensor` at `offset`. Non-blocking if the backend supports async. |
| `get_tensor_async` | Schedule a device→host read: copy `tensor`'s data out to `data`. |
| `set/get_tensor_2d_async` | Optional batched 2D strided copies — used for the MoE expert-partial-copy optimization. |
| `cpy_tensor_async` | Optional direct device→device copy between two backends (e.g. GPU↔GPU via NCCL/P2P). Returns false if not supported, then the scheduler falls back to get+set. |
| `synchronize` | Block until all queued async work on this stream is done. **Required if any async method is implemented.** |
| `graph_plan_create/free/update/compute` | Optional graph-planning API (pre-allocate/plan a graph for repeated execution). Currently unused by the scheduler; backends may use internally. |
| `graph_compute` | **The core entry point.** Execute every node in `cgraph` on this backend. Always async if supported; caller follows with `synchronize` or relies on events. |
| `event_record` / `event_wait` | Optional event-based sync between streams (cheaper than full synchronize for pipelining). |
| `graph_optimize` | Optional graph reordering/optimization pass before compute. |

**LPU impl**: [ggml-lpu.cpp:227](../ggml/src/ggml-lpu/ggml-lpu.cpp#L227). `graph_compute` is the simple node loop ([ggml-lpu.cpp:203](../ggml/src/ggml-lpu/ggml-lpu.cpp#L203)) that dispatches each node to `lpu_op_*`. Stub mode: async methods are sync `memcpy`, `synchronize` is a no-op.

---

## 5. `ggml_backend_buffer_type_i` — Buffer Type (Allocator)

**Defined**: [ggml-backend-impl.h:17-29](../ggml/src/ggml-backend-impl.h#L17)

A buffer type is a **memory allocator description** — it says "this is how you allocate device memory of a particular kind" (device memory, pinned host memory, unified memory, ...). One device may expose several buffer types (e.g. a GPU exposes device-memory buft + pinned-host buft). The scheduler picks tensor locations by matching buffer types to device `supports_buft`.

```c
struct ggml_backend_buffer_type_i {
    const char *          (*get_name)       (ggml_backend_buffer_type_t buft);
    ggml_backend_buffer_t (*alloc_buffer)   (ggml_backend_buffer_type_t buft, size_t size);
    size_t                (*get_alignment)  (ggml_backend_buffer_type_t buft);
    size_t                (*get_max_size)   (ggml_backend_buffer_type_t buft);           // optional
    size_t                (*get_alloc_size) (ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor);  // optional
    bool                  (*is_host)        (ggml_backend_buffer_type_t buft);           // optional
};
```

| Method | Purpose |
|--------|---------|
| `get_name` | Name for debugging, e.g. `"LPU0"`, `"CPU"`. |
| `alloc_buffer` | Allocate `size` bytes of this buffer type, returning a `ggml_backend_buffer_t`. This is the actual `malloc`/`cudaMalloc`/`lpu_malloc`. |
| `get_alignment` | Required alignment for tensor base addresses within the buffer (e.g. 128 for LPU, 256 for CUDA). The allocator pads tensor offsets to this. |
| `get_max_size` | Optional. Max allocatable size (defaults to `SIZE_MAX`). Lets a device report "I have 8 GiB". |
| `get_alloc_size` | Optional. Override the per-tensor allocation size (default `ggml_nbytes`). Useful for backends needing extra padding/extras per tensor. |
| `is_host` | Optional. **Important.** Returns true if the buffer is host-accessible memory with standard ggml layout. When true, the scheduler/test harness can `memcmp` outputs directly and skip get/set copies. The LPU stub sets this true (its buffers are `std::malloc`). |

**LPU impl**: [ggml-lpu.cpp:151](../ggml/src/ggml-lpu/ggml-lpu.cpp#L151). `alloc_buffer` calls `std::malloc` (stub), `is_host` returns true ([ggml-lpu.cpp:144](../ggml/src/ggml-lpu/ggml-lpu.cpp#L144)).

---

## 6. `ggml_backend_buffer_i` — Buffer (Allocated Memory)

**Defined**: [ggml-backend-impl.h:41-62](../ggml/src/ggml-backend-impl.h#L41)

A buffer is a **concrete block of allocated memory**. While the buffer type knows *how* to allocate, the buffer knows *how to move bytes in and out of one specific allocation* and *how to lay tensors out within it*. This is the interface the user selected in the IDE — the one whose `set_tensor`/`get_tensor` callbacks the scheduler calls to distribute inputs and collect results.

```c
struct ggml_backend_buffer_i {
    void         (*free_buffer)  (ggml_backend_buffer_t buffer);                          // optional
    void *       (*get_base)     (ggml_backend_buffer_t buffer);
    enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);  // optional
    void         (*memset_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
    void         (*set_tensor)   (ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    void         (*set_tensor_2d)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);  // optional
    void         (*get_tensor_2d)(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);  // optional
    bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);  // optional
    void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
    void         (*reset)        (ggml_backend_buffer_t buffer);  // optional
};
```

| Method | Purpose |
|--------|---------|
| `free_buffer` | Optional. Free the underlying memory (called when the buffer handle is released). |
| `get_base` | Returns the base host/device pointer of the whole buffer. Used to compute absolute addresses. |
| `init_tensor` | Optional. Called when a tensor is first placed in this buffer — lets the backend attach per-tensor extras (e.g. packed weight layouts). |
| `memset_tensor` | Fill a region of a tensor with a byte value (used to zero outputs). |
| `set_tensor` | **Host→device write.** Copy `data` (host) into `tensor` at byte `offset`, `size` bytes. `tensor->data` points to the tensor's slot in the buffer; the impl writes at `tensor->data + offset`. **This is how inputs reach the LPU.** |
| `get_tensor` | **Device→host read.** Copy `tensor`'s data out to `data`. **This is how results leave the LPU.** |
| `set_tensor_2d` / `get_tensor_2d` | Optional batched 2D strided copies — used by the scheduler's MoE expert-partial-copy fast path. |
| `cpy_tensor` | Optional direct buffer→buffer copy (same backend, or cross-backend if both sides cooperate). Returns false to fall back to get+set. |
| `clear` | Fill the entire buffer with a value (typically zeroing compute scratch). |
| `reset` | Optional. Reset any internal state accumulated during tensor init (e.g. clear tensor-extras bookkeeping). |

**Critical correctness note** (from the LPU debugging): `set_tensor`/`get_tensor` receive `offset` relative to `tensor->data`, not the buffer base. The implementation must be `memcpy((char*)tensor->data + offset, data, size)` — **not** `buf_base + offset`. Getting this wrong silently corrupts every input.

**LPU impl**: [ggml-lpu.cpp:97-135](../ggml/src/ggml-lpu/ggml-lpu.cpp#L97). `set_tensor`/`get_tensor` are the fixed `lpu_h2d_impl`/`lpu_d2h_impl` (stub: `memcpy`) anchored at `tensor->data`.

---

## 7. Non-Interface Supporting Structures

These are not `_i` vtables but are part of the backend system. Included for completeness.

### `struct ggml_backend_buffer` ([impl.h:64](../ggml/src/ggml-backend-impl.h#L64))
```c
struct ggml_backend_buffer {
    struct ggml_backend_buffer_i  iface;
    ggml_backend_buffer_type_t    buft;
    void * context;
    size_t size;
    enum ggml_backend_buffer_usage usage;   // WEIGHTS vs SCRATCH
};
```
Concrete buffer handle: vtable + back-ref to its type + private context + size + usage hint (weights get special scheduler treatment, e.g. partial MoE copy).

### `struct ggml_backend` ([impl.h:142](../ggml/src/ggml-backend-impl.h#L142))
```c
struct ggml_backend {
    ggml_guid_t guid;
    struct ggml_backend_i iface;
    ggml_backend_dev_t device;
    void * context;
};
```
Concrete backend handle: GUID (for `ggml_backend_is_*` type checks) + vtable + owning device + private context.

### `struct ggml_backend_device` ([impl.h:204](../ggml/src/ggml-backend-impl.h#L204))
```c
struct ggml_backend_device {
    struct ggml_backend_device_i iface;
    ggml_backend_reg_t reg;
    void * context;
};
```
Concrete device handle: vtable + owning registry + private context.

### `struct ggml_backend_reg` ([impl.h:226](../ggml/src/ggml-backend-impl.h#L226))
```c
struct ggml_backend_reg {
    int api_version;   // GGML_BACKEND_API_VERSION
    struct ggml_backend_reg_i iface;
    void * context;
};
```
Concrete registry handle: API version (for ABI compatibility across dynamically-loaded backends) + vtable + private context.

### `struct ggml_backend_event` ([impl.h:149](../ggml/src/ggml-backend-impl.h#L149))
```c
struct ggml_backend_event {
    struct ggml_backend_device * device;
    void * context;
};
```
Synchronization primitive. Has no vtable of its own — operations go through the device's `event_new`/`event_free`/`event_synchronize` and the backend's `event_record`/`event_wait`. Used by the scheduler to pipeline splits without full synchronizes.

### `struct ggml_backend_sched` ([ggml-backend.cpp:774](../ggml/src/ggml-backend.cpp#L774))
The scheduler. Not an interface — it's a concrete struct the engine owns. It holds the array of backends, buffer types, the computed splits, tensor-copy mirrors, and events. It **drives** all the interfaces above. (See [docs/lpu-backend-op-routing.md](lpu-backend-op-routing.md) for how it routes ops.)

### `struct ggml_backend_sched_split` ([ggml-backend.cpp:764](../ggml/src/ggml-backend.cpp#L764))
One contiguous run of graph nodes assigned to one backend, plus the list of cross-boundary input tensors that need copying. The scheduler produces an array of these and executes them in order.

### Callback typedefs (not interfaces, but related)
- `ggml_backend_eval_callback` ([ggml-backend.h:420](../ggml/include/ggml-backend.h#L420)) — per-node callback during `ggml_backend_compare_graph_backend`; used by tests to compare CPU vs LPU node-by-node.
- `ggml_backend_comm_init_t` / `ggml_backend_comm_free_t` / `ggml_backend_comm_allreduce_tensor_t` ([ggml-backend.h:206-208](../ggml/include/ggml-backend.h#L206)) — multi-GPU tensor-parallel communication hooks (NCCL/RCCL allreduce).

---

## 8. How They Compose: The LPU Example

The LPU backend ([ggml-lpu.cpp](../ggml/src/ggml-lpu/ggml-lpu.cpp)) instantiates all five interfaces:

```cpp
// Layer 1: registry — "the LPU backend exists, 1 device"
static const ggml_backend_reg_i ggml_backend_lpu_reg_interface = {
    /* get_name */ ..., /* get_device_count */ ..., /* get_device */ ..., /* get_proc_address */ nullptr,
};

// Layer 2: device — "LPU0, supports these ops, offload large GEMMs"
static const ggml_backend_device_i ggml_backend_lpu_device_interface = {
    /* get_name */ ..., /* get_description */ ..., /* get_memory */ ...,
    /* get_type */ ..., /* get_props */ ..., /* init_backend */ ...,
    /* get_buffer_type */ ..., /* get_host_buffer_type */ nullptr,
    /* buffer_from_host_ptr */ nullptr,
    /* supports_op */ ...,        // ← the big switch (MoE/FFN ops)
    /* supports_buft */ ...,      // ← accepts LPU + host buffers
    /* offload_op */ ...,         // ← batch-threshold gate
    /* event_new/free/synchronize */ nullptr, nullptr, nullptr,
};

// Layer 3: backend/stream — "run this graph node-by-node"
static const ggml_backend_i ggml_backend_lpu_interface = {
    /* get_name */ ..., /* free */ ...,
    /* set_tensor_async/get_tensor_async/... */ ...,   // stub: sync memcpy
    /* synchronize */ ...,
    /* graph_plan_* */ nullptr, nullptr, nullptr, nullptr,
    /* graph_compute */ ggml_backend_lpu_graph_compute,  // ← the node loop
    /* event_record/wait */ nullptr, nullptr,
    /* graph_optimize */ nullptr,
};

// Layer 4: buffer type — "allocate host-malloc'd memory, 128-byte align, is_host=true"
static const ggml_backend_buffer_type_i ggml_backend_lpu_buffer_type_interface = {
    /* get_name */ ..., /* alloc_buffer */ ..., /* get_alignment */ ...,
    /* get_max_size */ nullptr, /* get_alloc_size */ nullptr, /* is_host */ []{ return true; },
};

// Layer 5: buffer — "read/write tensors via memcpy at tensor->data+offset"
// (defined inline in ggml_backend_lpu_buffer_type_alloc_buffer, ggml-lpu.cpp:97)
static const ggml_backend_buffer_i lpu_buf_iface = {
    /* free_buffer */ ..., /* get_base */ ..., /* init_tensor */ nullptr,
    /* memset_tensor */ ..., /* set_tensor */ ..., /* get_tensor */ ...,
    /* set_tensor_2d/get_tensor_2d */ nullptr, nullptr,
    /* cpy_tensor */ ..., /* clear */ ..., /* reset */ nullptr,
};
```

Each layer's struct instance is assigned into the concrete handle's `iface` field:
```cpp
reg.iface   = ggml_backend_lpu_reg_interface;     // ggml-lpu.cpp:569
dev->iface  = ggml_backend_lpu_device_interface;  // ggml-lpu.cpp:559
backend->iface = ggml_backend_lpu_interface;      // ggml-lpu.cpp:295
bt.iface    = ggml_backend_lpu_buffer_type_interface;  // ggml-lpu.cpp:323
// buffer iface is passed to ggml_backend_buffer_init()  // ggml-lpu.cpp:137
```

---

## 9. Call Flow Summary

A complete op execution, mapped to the interfaces:

```
ggml_backend_graph_compute(sched, graph)
  │  (ggml-backend.cpp scheduler)
  ▼
for each split:
  │
  │ 1. Copy inputs to split backend
  │    ggml_backend_tensor_set(src_mirror)        ← backend_i.set_tensor_async
  │    or ggml_backend_tensor_copy(src, mirror)   ← buffer_i.set_tensor / cpy_tensor
  │
  │ 2. Run the split's sub-graph
  │    ggml_backend_graph_compute_async(split_backend, &split->graph)
  │       │  ← backend_i.graph_compute
  │       ▼
  │       (LPU) ggml_backend_lpu_graph_compute
  │         for node in graph:
  │           ggml_backend_lpu_compute_forward(node)   ← per-op dispatch
  │             lpu_op_* → ggml_compute_forward_*      ← actual math
  │             writes dst->data in LPU buffer
  │
  │ 3. (Next split) copy outputs back
  │    ggml_backend_tensor_get(dst)               ← buffer_i.get_tensor
  │
  ▼
done; final output read to host
```

The interface boundaries:
- **`device_i`** answers "can/should this op run here?" — called by the scheduler at graph-split time.
- **`buffer_type_i`** answers "how do I allocate memory for this?" — called when placing tensors.
- **`buffer_i`** answers "how do I move bytes in/out?" — called at split boundaries and for I/O.
- **`backend_i`** answers "how do I execute?" — called to run each split's sub-graph.
- **`reg_i`** answers "what devices exist?" — called once at startup.

Everything else in the engine (graph building, scheduling, split management, async pipelining) is **backend-agnostic** and lives in `ggml-backend.cpp`. A new backend like LPU only needs to implement these five interfaces honestly; the framework does the rest.
