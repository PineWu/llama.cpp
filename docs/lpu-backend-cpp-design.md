# C++ Redesign of the Backend Interface/Implementation Separation

**Date**: 2026-06-30
**Scope**: How to redesign ggml's C vtable pattern (`_i` interface struct + concrete struct + opaque `_t` handle) in idiomatic C++, with reference code.

The C design (see [docs/lpu-backend-interfaces.md](lpu-backend-interfaces.md)) hand-rolls vtables as structs of function pointers. C++ has first-class support for this via **abstract base classes** — the compiler generates the vtable, and inheritance provides the instance/vtable relationship automatically. This document shows the direct translation.

---

## Table of Contents

1. [The C → C++ Mapping](#1-the-c--c-mapping)
2. [The Five Abstract Interfaces](#2-the-five-abstract-interfaces)
3. [LPU Concrete Implementation](#3-lpu-concrete-implementation)
4. [Factory & Ownership Model](#4-factory--ownership-model)
5. [Polymorphic Scheduler Usage](#5-polymorphic-scheduler-usage)
6. [ABI-Stable Plugin Boundary](#6-abi-stable-plugin-boundary)
7. [C vs C++ Comparison](#7-c-vs-c-comparison)
8. [Design Rationale & Caveats](#8-design-rationale--caveats)

---

## 1. The C → C++ Mapping

| C concept | C++ equivalent | Why |
|-----------|----------------|-----|
| `struct ggml_backend_X_i` (function-pointer vtable) | `class IBackendX` with **pure virtual** methods | Compiler generates the vtable; no hand-written function pointers |
| `struct ggml_backend_X` (instance: `iface` + `context` + handles) | `class BackendXImpl : public IBackendX` with **member variables** | Members replace `context`; inheritance replaces the `iface` field |
| `typedef struct ggml_backend_X * ggml_backend_X_t` (opaque handle) | `IBackendX*` or `std::unique_ptr<IBackendX>` | The abstract base *is* the opaque type |
| `init()` / `free()` functions | **constructor / destructor** (RAII) | Exceptions-safe, impossible to forget |
| `context` (void* private state) | **private members** of the derived class | Type-safe, no casting |
| Back-pointers (`device`, `reg`, `buft` fields) | **raw non-owning pointers** (`IBackendDevice* dev_`) | Non-owning → raw pointer; ownership → smart pointer |
| `static const ..._i` vtable shared across instances | **one vtable per class, automatic** | All instances of `LpuBackend` share the compiler-generated vtable |
| Free wrapper `ggml_backend_graph_compute(h, g)` calling `h->iface.graph_compute(h, g)` | **virtual member function** `backend->graph_compute(g)` | The `->` already dispatches virtually; no wrapper needed |
| `guid` field for `is_lpu()` type checks | **`dynamic_cast`** or **visitor** | RTTI replaces hand-rolled type IDs |

### The single key insight

In C, you write **two** types per layer (the `_i` vtable and the concrete struct) and wire them together with an `iface` field. In C++, you write **one** abstract class (the interface) and **one** derived class (the implementation); the vtable is implicit and the `iface` wiring is automatic through inheritance. "Many instances share one vtable" — a property you manually ensure in C with `static const` — is the default in C++.

---

## 2. The Five Abstract Interfaces

These directly replace the five `_i` structs. Each pure-virtual method corresponds to a function pointer in the C version.

```cpp
// ggml-fwd.hpp — shared forward declarations (the C types stay as-is)
struct ggml_tensor;
struct ggml_cgraph;
enum class ggml_status { success, failed };
enum class dev_type    { cpu, gpu };
enum class buf_usage   { any, weights, scratch };

// =====================================================================
// Layer 1: Registry  (replaces ggml_backend_reg_i)
// =====================================================================
class IBackendRegistry {
public:
    virtual ~IBackendRegistry() = default;
    virtual const char*      name()                        const = 0;
    virtual size_t           device_count()                const = 0;
    virtual class IBackendDevice* device(size_t index)          = 0;
    // get_proc_address extension hook omitted for clarity
};

// =====================================================================
// Layer 2: Device  (replaces ggml_backend_device_i)  — the decision layer
// =====================================================================
class IBackendDevice {
public:
    virtual ~IBackendDevice() = default;

    // identity & capability queries
    virtual const char*  name()        const = 0;
    virtual const char*  description() const = 0;
    virtual dev_type     type()        const = 0;
    virtual void         memory(size_t* free, size_t* total) const = 0;

    // factories (replace init_backend / get_buffer_type)
    virtual std::unique_ptr<class IBackend>     create_backend()       = 0;
    virtual class IBufferType*                  buffer_type()          = 0;
    virtual class IBufferType*                  host_buffer_type()     { return nullptr; }      // optional
    virtual std::unique_ptr<class IBuffer>      buffer_from_host_ptr(
        void* ptr, size_t size, size_t max_tensor_size)                                        { return nullptr; } // optional

    // routing decisions — the scheduler calls these
    virtual bool supports_op  (const ggml_tensor* op)              const = 0;
    virtual bool supports_buft(const IBufferType*  buft)           const = 0;
    virtual bool offload_op   (const ggml_tensor* op)              const { return false; }     // optional

    // optional events
    virtual class IBackendEvent* event_new()         { return nullptr; }
    virtual void                 event_free(IBackendEvent*)           {}
    virtual void                 event_synchronize(IBackendEvent*)     {}

    // back-pointer to owning registry
    virtual IBackendRegistry* registry() = 0;
};

// =====================================================================
// Layer 3: Backend / compute stream  (replaces ggml_backend_i)
// =====================================================================
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual const char* name() const = 0;

    // tensor I/O (set_tensor / get_tensor; async variants fold into these)
    virtual void set_tensor(ggml_tensor* t, const void* data, size_t off, size_t size)             = 0;
    void         set_tensor(ggml_tensor* t, const void* data, size_t size) { set_tensor(t, data, 0, size); }
    virtual void get_tensor(const ggml_tensor* t, void* data, size_t off, size_t size) const       = 0;
    void         get_tensor(const ggml_tensor* t, void* data, size_t size) const { get_tensor(t, data, 0, size); }

    virtual bool cpy_tensor_async(IBackend* src_backend, const ggml_tensor* src, ggml_tensor* dst) {
        return false; // default: not supported → caller falls back to get+set
    }
    virtual void synchronize() {}  // required only if async ops supported

    // the core entry point
    virtual ggml_status graph_compute(ggml_cgraph* g) = 0;

    // optional events / graph optimization
    virtual void event_record(IBackendEvent*) {}
    virtual void event_wait  (IBackendEvent*) {}
    virtual void graph_optimize(ggml_cgraph*) {}

    // back-pointer to the device that created this backend
    virtual IBackendDevice* device() = 0;
};

// =====================================================================
// Layer 4: Buffer type / allocator  (replaces ggml_backend_buffer_type_i)
// =====================================================================
class IBufferType {
public:
    virtual ~IBufferType() = default;
    virtual const char*            name()                       const = 0;
    virtual std::unique_ptr<class IBuffer> alloc_buffer(size_t size) = 0;
    virtual size_t                 alignment()                  const = 0;
    virtual size_t                 max_size()                   const { return SIZE_MAX; }   // optional
    virtual size_t                 alloc_size(const ggml_tensor* t) const { return /* ggml_nbytes(t) */ 0; } // optional
    virtual bool                   is_host()                    const { return false; }      // optional

    virtual IBackendDevice* device() = 0;  // back-pointer
};

// =====================================================================
// Layer 5: Buffer / allocated memory  (replaces ggml_backend_buffer_i)
// =====================================================================
class IBuffer {
public:
    virtual ~IBuffer() = default;
    virtual void*      base()       = 0;
    virtual size_t     size()       const = 0;
    virtual buf_usage  usage()      const = 0;
    virtual void       set_usage(buf_usage u) = 0;

    // the data-movement callbacks the scheduler calls at split boundaries
    virtual void memset_tensor(ggml_tensor* t, uint8_t value, size_t off, size_t size)       = 0;
    virtual void set_tensor  (ggml_tensor* t, const void* data, size_t off, size_t size)     = 0;
    virtual void get_tensor  (const ggml_tensor* t, void* data, size_t off, size_t size) const = 0;
    virtual bool cpy_tensor  (const ggml_tensor* src, ggml_tensor* dst) { return false; }    // optional
    virtual void clear       (uint8_t value)                                            = 0;
    virtual void reset       () {}                                                       // optional

    virtual IBufferType* buffer_type() = 0;  // back-pointer
};

// =====================================================================
// Event (no _i in C; operations go through device/backend) — kept minimal
// =====================================================================
class IBackendEvent {
public:
    virtual ~IBackendEvent() = default;
    virtual void record  (IBackend* backend)   = 0;
    virtual void wait    (IBackend* backend)   = 0;
    virtual void synchronize()                  = 0;
};
```

**Design notes on the interfaces:**

- **`virtual ~IClass() = default;`** in every interface — mandatory so a derived object can be deleted through a base pointer. Without it, deleting `IBackend*` that actually points to `LpuBackend` is undefined behavior.
- **Pure virtual (`= 0`)** for required methods; **virtual with a default body** for optional ones (e.g. `offload_op` returns false, `synchronize` is a no-op). This mirrors C's "optional function pointer may be NULL" — but is type-safe.
- **Overload sets** (`set_tensor(t,data,size)` calling `set_tensor(t,data,0,size)`) replace C's default-argument conventions without breaking the vtable layout.
- **No `context` void*** — private members of the derived class replace it, with full type safety.

---

## 3. LPU Concrete Implementation

Each C struct+context pair becomes one derived class. The `context` fields become private members; the `iface` assignment disappears (inheritance handles it).

```cpp
// lpu-backend.hpp
class LpuRegistry   : public IBackendRegistry;
class LpuDevice     : public IBackendDevice;
class LpuBackend    : public IBackend;
class LpuBufferType : public IBufferType;
class LpuBuffer     : public IBuffer;

// =====================================================================
// Registry — owns the devices
// =====================================================================
class LpuRegistry : public IBackendRegistry {
    std::vector<std::unique_ptr<LpuDevice>> devices_;
public:
    LpuRegistry() {
        // stub: one virtual device. Real hardware would enumerate here.
        devices_.push_back(std::make_unique<LpuDevice>(this, 0, "LPU0", "LPU-Gen1"));
    }
    const char* name() const override { return "LPU"; }
    size_t      device_count() const override { return devices_.size(); }
    IBackendDevice* device(size_t i) override { return devices_.at(i).get(); }
};

// =====================================================================
// Device — the decision layer
// =====================================================================
class LpuDevice : public IBackendDevice {
    IBackendRegistry*  reg_;                 // non-owning back-pointer
    int                index_;
    std::string        name_, desc_;
    LpuBufferType      buft_;                // device owns its buffer type
    int                offload_threshold_ = 32;
public:
    LpuDevice(IBackendRegistry* reg, int idx, std::string n, std::string d);

    const char*  name()        const override { return name_.c_str(); }
    const char*  description() const override { return desc_.c_str(); }
    dev_type     type()        const override { return dev_type::gpu; }
    void         memory(size_t* f, size_t* t) const override { *f = *t = 0; }

    // factories
    std::unique_ptr<IBackend> create_backend() override {
        return std::make_unique<LpuBackend>(this, index_);
    }
    IBufferType* buffer_type()      override { return &buft_; }

    // routing decisions (the big switch — abbreviated)
    bool supports_op(const ggml_tensor* op) const override;
    bool supports_buft(const IBufferType* b) const override;
    bool offload_op(const ggml_tensor* op)   const override;

    IBackendRegistry* registry() override { return reg_; }

    int offload_threshold() const { return offload_threshold_; }
};

// =====================================================================
// Backend / compute stream — runs graphs node-by-node
// =====================================================================
class LpuBackend : public IBackend {
    LpuDevice*  dev_;            // non-owning back-pointer
    int         device_idx_;
    // stream_, device_handle_, ... would live here with real hardware
public:
    explicit LpuBackend(LpuDevice* dev, int idx) : dev_(dev), device_idx_(idx) {}
    ~LpuBackend() override { /* lpu_stream_destroy(stream_); */ }

    const char* name() const override { return "LPU0"; }

    void set_tensor(ggml_tensor* t, const void* data, size_t off, size_t size) override {
        // stub: host memory, so plain memcpy anchored at tensor->data
        std::memcpy(static_cast<char*>(t->data) + off, data, size);
    }
    void get_tensor(const ggml_tensor* t, void* data, size_t off, size_t size) const override {
        std::memcpy(data, static_cast<const char*>(t->data) + off, size);
    }
    // cpy_tensor_async, synchronize, etc. default-implemented (stub: no-op/sync)

    ggml_status graph_compute(ggml_cgraph* g) override;

    IBackendDevice* device() override { return dev_; }

private:
    // per-op dispatch — the equivalent of ggml_backend_lpu_compute_forward
    ggml_status compute_forward(ggml_tensor* node);
};

// =====================================================================
// Buffer type — allocator description
// =====================================================================
class LpuBufferType : public IBufferType {
    LpuDevice* dev_;            // non-owning back-pointer
public:
    explicit LpuBufferType(LpuDevice* dev) : dev_(dev) {}
    const char* name() const override { return "LPU0"; }

    std::unique_ptr<IBuffer> alloc_buffer(size_t size) override {
        return std::make_unique<LpuBuffer>(this, size);
    }
    size_t alignment() const override { return 128; }       // LPU_BUFFER_ALIGNMENT
    bool   is_host()    const override { return true; }      // stub: std::malloc is host memory

    IBackendDevice* device() override { return dev_; }
};

// =====================================================================
// Buffer — a concrete block of allocated memory
// =====================================================================
class LpuBuffer : public IBuffer {
    LpuBufferType*       buft_;            // non-owning back-pointer
    std::vector<uint8_t> data_;            // the actual bytes (stub: host vector)
    buf_usage            usage_ = buf_usage::any;
public:
    LpuBuffer(LpuBufferType* bt, size_t size) : buft_(bt), data_(size) {}

    void*     base()         override { return data_.data(); }
    size_t    size()    const override { return data_.size(); }
    buf_usage usage()   const override { return usage_; }
    void      set_usage(buf_usage u) override { usage_ = u; }

    void memset_tensor(ggml_tensor* t, uint8_t v, size_t off, size_t size) override {
        std::memset(static_cast<char*>(t->data) + off, v, size);
    }
    void set_tensor(ggml_tensor* t, const void* data, size_t off, size_t size) override {
        // anchored at tensor->data, NOT buffer base — the bug we fixed in C
        std::memcpy(static_cast<char*>(t->data) + off, data, size);
    }
    void get_tensor(const ggml_tensor* t, void* data, size_t off, size_t size) const override {
        std::memcpy(data, static_cast<const char*>(t->data) + off, size);
    }
    void clear(uint8_t v) override { std::fill(data_.begin(), data_.end(), v); }

    IBufferType* buffer_type() override { return buft_; }
};
```

### The `supports_op` / `offload_op` implementations

Direct translations of the C versions:

```cpp
bool LpuDevice::supports_op(const ggml_tensor* op) const {
    switch (op->op) {
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ARGSORT:
        case GGML_OP_GET_ROWS:
        case GGML_OP_MUL_MAT:
            return op->src[0]->type == GGML_TYPE_F32 ||
                   op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_UNARY: { /* silu/gelu/relu/... */ return true; }
        case GGML_OP_GLU:   { /* swiglu/geglu */       return true; }
        case GGML_OP_RMS_NORM: case GGML_OP_NORM:
        case GGML_OP_MUL: case GGML_OP_ADD: case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_RESHAPE: case GGML_OP_VIEW: case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE: case GGML_OP_CONT:
            return true;
        default:
            return false;
    }
}

bool LpuDevice::supports_buft(const IBufferType* b) const {
    // accept our own buffer type and host buffers
    return dynamic_cast<const LpuBufferType*>(b) != nullptr || b->is_host();
}

bool LpuDevice::offload_op(const ggml_tensor* op) const {
    const int threshold = offload_threshold_;
    switch (op->op) {
        case GGML_OP_MUL_MAT:    return op->ne[1] >= threshold;  // dense GEMM batch
        case GGML_OP_MUL_MAT_ID: return op->ne[2] >= threshold;  // MoE token count
        default: return false;
    }
}
```

### The `graph_compute` implementation

```cpp
ggml_status LpuBackend::graph_compute(ggml_cgraph* g) {
    // lpu_set_device(device_idx_);
    for (int i = 0; i < g->n_nodes; i++) {
        ggml_tensor* node = g->nodes[i];
        if (node->op == GGML_OP_NONE) continue;   // skip leaves
        if (compute_forward(node) != ggml_status::success) return ggml_status::failed;
    }
    // lpu_stream_sync(stream_);
    return ggml_status::success;
}

ggml_status LpuBackend::compute_forward(ggml_tensor* node) {
    // the dispatch switch — equivalent to ggml_backend_lpu_compute_forward
    switch (node->op) {
        case GGML_OP_MUL_MAT:    return lpu_op_mul_mat(node);
        case GGML_OP_MUL_MAT_ID: return lpu_op_mul_mat_id(node);
        case GGML_OP_RMS_NORM:   return lpu_op_rms_norm(node);
        // ... etc ...
        case GGML_OP_VIEW: case GGML_OP_RESHAPE: return ggml_status::success; // metadata-only
        default: return ggml_status::failed;
    }
}
```

Note: `compute_forward` is a **private non-virtual** method — it's an implementation detail of `LpuBackend`, not part of the interface. The C version exposed it via `lpu_op_*` free functions; here it's encapsulated.

---

## 4. Factory & Ownership Model

The C code uses ad-hoc ownership (who frees what is documented in comments). C++ makes it explicit through smart pointers:

| Object | Owner | Lifetime |
|--------|-------|----------|
| `IBackendRegistry` | process / `main` | static or stack |
| `IBackendDevice` | `IBackendRegistry` (via `unique_ptr`) | until registry destroyed |
| `IBackend` | caller (via `unique_ptr`) | until released |
| `IBufferType` | `IBackendDevice` (member) | until device destroyed |
| `IBuffer` | caller (via `unique_ptr`) | until released |
| back-pointers (`dev_`, `reg_`, `buft_`) | **non-owning raw pointers** | valid as long as owner alive |

```cpp
// Construction shows the ownership chain clearly:
LpuRegistry registry;                                      // stack-owned
IBackendDevice* dev = registry.device(0);                  // borrowed
auto backend     = dev->create_backend();                  // unique_ptr<IBackend>
IBufferType*  bt = dev->buffer_type();                     // borrowed
auto buf          = bt->alloc_buffer(1 << 20);             // unique_ptr<IBuffer>

// No manual free needed — destructors handle everything when scopes exit.
// Contrast with C: ggml_backend_free(backend); ggml_backend_buffer_free(buf); ...
```

**Why raw pointers for back-references?** A back-pointer is non-owning by definition (the owner is the layer above). Using `shared_ptr` would create cycles (registry→device→registry) and leak. Raw pointer + lifetime guarantee ("the device outlives all its backends") is the correct model. This is a well-established C++ idiom (see `std::enable_shared_from_this` caveats, or the "observer pointer" pattern).

---

## 5. Polymorphic Scheduler Usage

The scheduler holds `IBackend*` pointers and dispatches virtually — it has no knowledge of `LpuBackend` vs `CpuBackend`. This is the whole point of the separation.

```cpp
class Scheduler {
    std::vector<IBackend*>            backends_;   // borrowed
    std::vector<IBufferType*>         bufts_;      // borrowed
public:
    void add(IBackend* b, IBufferType* bt) {
        backends_.push_back(b);
        bufts_.push_back(bt);
    }

    // The routing decision — equivalent to ggml_backend_sched_backend_id_from_cur
    IBackend* pick_backend(const ggml_tensor* op) const {
        // Rule D: weight-driven placement with offload
        for (int i = 0; i < (int)backends_.size() - 1; i++) {  // skip last (CPU)
            IBackendDevice* dev = backends_[i]->device();
            if (dev->supports_op(op) && dev->offload_op(op))
                return backends_[i];
        }
        return backends_.back();  // CPU fallback
    }

    // Cross-backend tensor copy at split boundaries
    void copy_tensor(IBackend* src_be, IBackend* dst_be,
                     const ggml_tensor* src, ggml_tensor* dst) {
        // try device-to-device first
        if (dst_be->cpy_tensor_async(src_be, src, dst)) return;
        // fall back to get + set
        std::vector<uint8_t> tmp(/* ggml_nbytes(src) */ 0);
        src_be->get_tensor(src, tmp.data(), tmp.size());
        dst_be->set_tensor(dst, tmp.data(), tmp.size());
    }

    ggml_status run(ggml_cgraph* g) {
        // ... split graph by pick_backend(), copy cross-boundary tensors,
        //     call backend->graph_compute(&split) for each split ...
        return ggml_status::success;
    }
};
```

The scheduler code is **identical regardless of which backends are registered** — pure polymorphism through the abstract interfaces. Adding a new backend (e.g. `MetalBackend`) requires zero changes to `Scheduler`.

---

## 6. ABI-Stable Plugin Boundary

If backends are loaded as `.so`/`.dll` plugins (like ggml's `GGML_BACKEND_DL` mode), **C++ virtual tables across `.so` boundaries are not portable** — different compilers/flags produce different vtable layouts. The standard solution is an **extern "C" factory** that returns the abstract interface, with the concrete class hidden in the `.so`:

```cpp
// ===== in lpu-backend.so =====
extern "C" {
    // The only symbol the host looks up (via dlsym). C linkage → stable ABI.
    IBackendRegistry* ggml_backend_lpu_reg() {
        static LpuRegistry instance;   // constructed once, destroyed at exit
        return &instance;
    }
}

// ===== in the host (ggml-backend-reg.cpp equivalent) =====
using reg_fn_t = IBackendRegistry*();
void load_backend_plugin(const char* path) {
    void* h = dlopen(path, RTLD_NOW);
    auto* fn = reinterpret_cast<reg_fn_t*>(dlsym(h, "ggml_backend_lpu_reg"));
    IBackendRegistry* reg = fn();          // get the abstract interface
    registries_.push_back(reg);            // use polymorphically
}
```

The host never names `LpuRegistry` directly — it only sees `IBackendRegistry*`. As long as the **abstract class definition** (the `IBackend*` headers) is compiled identically on both sides, the vtable layout matches. For full safety, the interface headers are kept free of STL types in their method signatures (use `const char*`, raw pointers, `size_t`), so the ABI doesn't depend on libc++/libstdc++ version.

> **This is the pImpl + abstract-factory pattern.** The abstract class is the "Pimpl interface"; the factory is the C-linkage constructor. It's how Qt plugins, WebKit ports, and many game engines do runtime polymorphism across DSOs.

---

## 7. C vs C++ Comparison

| Aspect | C (ggml today) | C++ redesign |
|--------|----------------|--------------|
| Define an interface | Hand-write `struct X_i { fn_ptr; ... }` | Write `class IX { virtual ... = 0; }` |
| Implement an interface | `static const X_i iface = { fn1, fn2, ... }` + assign to `inst.iface` | `class Impl : public IX { ... override ... }` |
| Per-instance state | `void* context` + casting | Private member variables (type-safe) |
| Vtable sharing | Manual: `static const` vtable, many instances | Automatic: one vtable per class |
| Method call | `inst->iface.method(inst, args)` | `inst->method(args)` (virtual dispatch) |
| Memory management | `init()` / `free()` pairs, manual | RAII: constructors/destructors, `unique_ptr` |
| Optional methods | NULL function pointer + caller checks | Default virtual body, override if needed |
| Type identification | `guid` field + `is_lpu()` compare | `dynamic_cast<LpuBackend*>(b)` or visitor |
| Back-pointers | Struct fields | Raw non-owning pointers |
| Cross-DSO ABI | Naturally stable (C ABI) | Needs `extern "C"` factory + shared interface header |
| Boilerplate per layer | ~2 types + wiring code | 1 abstract class + 1 derived class |
| Error-prone? | Yes — `set_tensor` base-offset bug, null vtable slots, missing `free` | Less — compiler enforces override, RAII prevents leaks |

### The bug that doesn't happen in C++

Recall the C `set_tensor` bug (writing to `buf base + offset` instead of `tensor->data + offset`). In C++ the buffer's `set_tensor` is a member with `tensor` as a typed parameter — there's no `buf->context->data` to mistakenly use as the base, and the method naturally anchors at `tensor->data`. The type system steers you toward the correct implementation.

---

## 8. Design Rationale & Caveats

### Why abstract classes over CRTP or concepts?

- **CRTP** (`template<class D> class IBackend { ... static_cast<D*>(this)... }`) is **static** polymorphism — the backend type must be known at compile time. The scheduler couldn't hold a heterogeneous `vector<IBackend*>`. Rejected for this use case.
- **C++20 concepts + templates** have the same problem — no runtime polymorphism.
- **Abstract base classes** are **dynamic** polymorphism — exactly what a plugin/backend system needs: a `vector<IBackend*>` of unknown concrete types.

### Why `unique_ptr` for factory returns (not `shared_ptr`)?

`create_backend()` and `alloc_buffer()` transfer **sole ownership** to the caller. `unique_ptr` expresses this precisely and has zero overhead vs. a raw pointer. `shared_ptr` would imply shared ownership (and add a control block) where there is none. If the caller wants to share, it can `std::move` into a `shared_ptr` themselves.

### Why raw pointers for back-references (not `shared_ptr`/`weak_ptr`)?

Back-references form a strict ownership tree (registry → device → backend). Using `shared_ptr` for `device_->backends` would create a cycle (device owns backend, backend points back to device) → memory leak. `weak_ptr` would work but adds overhead for no benefit when the lifetime is already strictly hierarchical. Raw pointer + documented invariant ("a device outlives all backends it creates") is the idiomatic C++ solution.

### When would you keep the C style?

- **Cross-compiler ABI is required** and you can't pin the C++ compiler/STL (rare in practice; `extern "C"` factory usually suffices).
- **C compatibility** — ggml's public API is C so it can be called from Rust/Python/Go bindings. The C vtable pattern is a deliberate choice for that ecosystem.
- **Zero-overhead virtual dispatch concerns** — irrelevant here; graph_compute is a heavy operation, the virtual call cost is negligible.

### What's gained by the C++ redesign

1. **~40% less code per layer** — no `iface` field, no `static const` vtable definition, no `iface.method(self, ...)` wrappers.
2. **Type safety** — `context` becomes members; no `void*` casting; `dynamic_cast` replaces `guid` checks.
3. **RAII** — impossible to leak (forget `free`) or use-after-free (use after `free`); exceptions unwind correctly.
4. **Compiler-checked `override`** — if the interface changes, derived classes that don't match fail to compile (in C, a stale function pointer silently stays NULL or points to a wrong-typed function).
5. **Encapsulation** — `compute_forward` is private; the C version's `lpu_op_*` are extern and callable by anyone.

### What's lost

1. **C ABI** — the redesign can't be consumed by C/Rust/Go directly without a C wrapper.
2. **Manual control** — you can't easily swap a vtable at runtime (C can reassign `inst->iface`). Rarely needed.
3. **Build complexity** — C++ requires matching compiler/STL across the plugin boundary for the interface headers.

---

## Summary

The C++ redesign collapses the C pattern's **two types per layer** (`_i` + concrete struct) into **one abstract class + one derived class**, with the compiler taking over vtable generation and wiring. Ownership becomes explicit through smart pointers; lifetime becomes automatic through RAII; type identity becomes `dynamic_cast`. The five-layer hierarchy (registry → device → backend → buffer_type → buffer) and the back-pointer chains map directly, and the scheduler's polymorphic code is unchanged in spirit — it just calls `backend->graph_compute(g)` instead of `backend->iface.graph_compute(backend, g)`.

For ggml specifically, the C design is the right choice because of its C ABI requirement. But for a greenfield C++ project, the abstract-class design shown here is strictly simpler and safer while preserving every capability.
