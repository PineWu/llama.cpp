// ggml-lpu.cpp  -- LPU ggml backend implementation
//
// This file implements the four vtable layers required by the ggml backend
// abstraction (ggml/src/ggml-backend-impl.h):
//
//   1. ggml_backend_buffer_type_i  -- memory allocator configuration
//   2. ggml_backend_buffer_i       -- concrete allocated memory region
//   3. ggml_backend_i              -- compute stream / execution
//   4. ggml_backend_device_i       -- physical device properties & op filter
//   5. ggml_backend_reg_i          -- device registry (entry point)
//
// LPU SDK calls are wrapped in thin helpers; the placeholders (lpu_malloc,
// lpu_free, etc.) must be replaced with actual LPU SDK calls once the SDK
// headers and libraries are available.

#include "ggml-lpu.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "common.h"
#include "lpu_ops.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LPU SDK stubs
//
// TODO: Replace all functions in this block with real LPU SDK calls.
//       The signatures are intentional placeholders to make the file
//       compile while the SDK integration is being finalised.
// ---------------------------------------------------------------------------

static int     lpu_device_count_impl()                          { return 1; }
static const char * lpu_device_name_impl(int /*dev*/)           { return "LPU-Gen1"; }
static lpu_status_t lpu_init_impl()                             { return LPU_SUCCESS; }
static lpu_status_t lpu_set_device_impl(int /*dev*/)            { return LPU_SUCCESS; }
static lpu_status_t lpu_stream_create_impl(lpu_stream_t * s)    { *s = nullptr; return LPU_SUCCESS; }
static lpu_status_t lpu_stream_destroy_impl(lpu_stream_t /*s*/) { return LPU_SUCCESS; }
static lpu_status_t lpu_stream_sync_impl(lpu_stream_t /*s*/)    { return LPU_SUCCESS; }
static lpu_status_t lpu_malloc_impl(void ** ptr, size_t size)   { *ptr = std::malloc(size); return *ptr ? LPU_SUCCESS : -1; }
static lpu_status_t lpu_free_impl(void * ptr)                   { std::free(ptr); return LPU_SUCCESS; }
static lpu_status_t lpu_memset_impl(void * dst, int val, size_t size, lpu_stream_t /*s*/) { std::memset(dst, val, size); return LPU_SUCCESS; }
static lpu_status_t lpu_h2d_impl(void * dst, const void * src, size_t size, lpu_stream_t /*s*/) { std::memcpy(dst, src, size); return LPU_SUCCESS; }
static lpu_status_t lpu_d2h_impl(void * dst, const void * src, size_t size, lpu_stream_t /*s*/) { std::memcpy(dst, src, size); return LPU_SUCCESS; }
static lpu_status_t lpu_d2d_impl(void * dst, const void * src, size_t size, lpu_stream_t /*s*/) { std::memcpy(dst, src, size); return LPU_SUCCESS; }
static void lpu_mem_info_impl(int /*dev*/, size_t * free_out, size_t * total_out) {
    *free_out  = (size_t)8 << 30;  // 8 GiB placeholder
    *total_out = (size_t)8 << 30;
}

#define LPU_CHECK(call)  do { \
    lpu_status_t _st = (call); \
    if (_st != LPU_SUCCESS) { \
        GGML_LOG_ERROR("LPU error %d at %s:%d\n", (int)_st, __FILE__, __LINE__); \
        GGML_ABORT("LPU call failed"); \
    } \
} while (0)

// Required alignment for LPU device memory (adjust to hardware requirement)
static constexpr size_t LPU_BUFFER_ALIGNMENT = 128;

// Name prefix used in all log / debug output
static constexpr const char * LPU_BACKEND_NAME = "LPU";

// ============================================================================
// 1.  Buffer type interface
// ============================================================================

static const char * ggml_backend_lpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = static_cast<ggml_backend_lpu_buffer_type_context *>(buft->context);
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_lpu_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {

    auto * bt_ctx = static_cast<ggml_backend_lpu_buffer_type_context *>(buft->context);

    auto * buf_ctx = new ggml_backend_lpu_buffer_context;
    buf_ctx->device = bt_ctx->device;
    buf_ctx->data   = nullptr;

    LPU_CHECK(lpu_set_device_impl(bt_ctx->device));
    LPU_CHECK(lpu_malloc_impl(&buf_ctx->data, size));

    static const ggml_backend_buffer_i lpu_buf_iface = {
        /* free_buffer  */ [](ggml_backend_buffer_t buf) {
            auto * ctx = static_cast<ggml_backend_lpu_buffer_context *>(buf->context);
            lpu_free_impl(ctx->data);
            delete ctx;
        },
        /* get_base     */ [](ggml_backend_buffer_t buf) -> void * {
            return static_cast<ggml_backend_lpu_buffer_context *>(buf->context)->data;
        },
        /* init_tensor  */ nullptr,
        /* memset_tensor*/ [](ggml_backend_buffer_t buf, ggml_tensor * tensor,
                               uint8_t value, size_t offset, size_t size) {
            auto * ctx = static_cast<ggml_backend_lpu_buffer_context *>(buf->context);
            lpu_memset_impl(static_cast<char *>(ctx->data) + offset, value, size, nullptr);
            GGML_UNUSED(tensor);
        },
        /* set_tensor   */ [](ggml_backend_buffer_t buf, ggml_tensor * tensor,
                               const void * data, size_t offset, size_t size) {
            // tensor->data already points to this tensor's allocated region
            // within the buffer. Write relative to that, not to buf base.
            lpu_h2d_impl(static_cast<char *>(tensor->data) + offset, data, size, nullptr);
            GGML_UNUSED(buf);
        },
        /* get_tensor   */ [](ggml_backend_buffer_t buf, const ggml_tensor * tensor,
                               void * data, size_t offset, size_t size) {
            lpu_d2h_impl(data, static_cast<const char *>(tensor->data) + offset, size, nullptr);
            GGML_UNUSED(buf);
        },
        /* set_tensor_2d*/ nullptr,
        /* get_tensor_2d*/ nullptr,
        /* cpy_tensor   */ [](ggml_backend_buffer_t buf, const ggml_tensor * src, ggml_tensor * dst) -> bool {
            if (!ggml_backend_buft_is_lpu(dst->buffer->buft)) return false;
            auto * ctx_dst = static_cast<ggml_backend_lpu_buffer_context *>(buf->context);
            auto * ctx_src = static_cast<ggml_backend_lpu_buffer_context *>(src->buffer->context);
            size_t nbytes  = ggml_nbytes(src);
            lpu_d2d_impl(static_cast<char *>(ctx_dst->data) + dst->view_offs,
                         static_cast<const char *>(ctx_src->data) + src->view_offs,
                         nbytes, nullptr);
            return true;
        },
        /* clear         */ [](ggml_backend_buffer_t buf, uint8_t value) {
            auto * ctx = static_cast<ggml_backend_lpu_buffer_context *>(buf->context);
            lpu_memset_impl(ctx->data, value, buf->size, nullptr);
        },
        /* reset         */ nullptr,
    };

    return ggml_backend_buffer_init(buft, lpu_buf_iface, buf_ctx, size);
}

static size_t ggml_backend_lpu_buffer_type_get_alignment(ggml_backend_buffer_type_t /*buft*/) {
    return LPU_BUFFER_ALIGNMENT;
}

static bool ggml_backend_lpu_buffer_type_is_host(ggml_backend_buffer_type_t /*buft*/) {
    // The stub allocator uses std::malloc (host memory).
    // Returning true lets the test harness compare outputs directly without
    // going through get_tensor/set_tensor copies.
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_lpu_buffer_type_interface = {
    /* get_name       */ ggml_backend_lpu_buffer_type_get_name,
    /* alloc_buffer   */ ggml_backend_lpu_buffer_type_alloc_buffer,
    /* get_alignment  */ ggml_backend_lpu_buffer_type_get_alignment,
    /* get_max_size   */ nullptr,   // SIZE_MAX default
    /* get_alloc_size */ nullptr,   // element-size default
    /* is_host        */ ggml_backend_lpu_buffer_type_is_host,
};

// ============================================================================
// 2.  Backend (compute stream) interface
// ============================================================================

static const char * ggml_backend_lpu_get_name(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_lpu_context *>(backend->context);
    return ctx->name.c_str();
}

static void ggml_backend_lpu_free(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_lpu_context *>(backend->context);
    lpu_stream_destroy_impl(ctx->stream);
    delete ctx;
    delete backend;
}

static void ggml_backend_lpu_set_tensor_async(ggml_backend_t backend,
        ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<ggml_backend_lpu_context *>(backend->context);
    lpu_h2d_impl(static_cast<char *>(tensor->data) + offset, data, size, ctx->stream);
}

static void ggml_backend_lpu_get_tensor_async(ggml_backend_t backend,
        const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<ggml_backend_lpu_context *>(backend->context);
    lpu_d2h_impl(data, static_cast<const char *>(tensor->data) + offset, size, ctx->stream);
}

static bool ggml_backend_lpu_cpy_tensor_async(ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_lpu(backend_dst)) return false;
    auto * ctx_dst = static_cast<ggml_backend_lpu_context *>(backend_dst->context);
    GGML_UNUSED(backend_src);
    lpu_d2d_impl(dst->data, src->data, ggml_nbytes(src), ctx_dst->stream);
    return true;
}

static void ggml_backend_lpu_synchronize(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_lpu_context *>(backend->context);
    lpu_stream_sync_impl(ctx->stream);
}

static enum ggml_status ggml_backend_lpu_graph_compute(ggml_backend_t backend,
        ggml_cgraph * cgraph) {
    auto * ctx = static_cast<ggml_backend_lpu_context *>(backend->context);
    lpu_set_device_impl(ctx->device);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE) {
            continue;
        }

        enum ggml_status status = ggml_backend_lpu_compute_forward(ctx, node);
        if (status != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("LPU: compute_forward failed for op %s\n",
                           ggml_op_name(node->op));
            return status;
        }
    }

    // Ensure all async work is complete before returning
    lpu_stream_sync_impl(ctx->stream);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_lpu_interface = {
    /* get_name             */ ggml_backend_lpu_get_name,
    /* free                 */ ggml_backend_lpu_free,
    /* set_tensor_async     */ ggml_backend_lpu_set_tensor_async,
    /* get_tensor_async     */ ggml_backend_lpu_get_tensor_async,
    /* set_tensor_2d_async  */ nullptr,
    /* get_tensor_2d_async  */ nullptr,
    /* cpy_tensor_async     */ ggml_backend_lpu_cpy_tensor_async,
    /* synchronize          */ ggml_backend_lpu_synchronize,
    /* graph_plan_create    */ nullptr,
    /* graph_plan_free      */ nullptr,
    /* graph_plan_update    */ nullptr,
    /* graph_plan_compute   */ nullptr,
    /* graph_compute        */ ggml_backend_lpu_graph_compute,
    /* event_record         */ nullptr,
    /* event_wait           */ nullptr,
    /* graph_optimize       */ nullptr,
};

// ============================================================================
// 3.  Device interface
// ============================================================================

static const char * ggml_backend_lpu_device_get_name(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_lpu_device_context *>(dev->context)->name.c_str();
}

static const char * ggml_backend_lpu_device_get_description(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_lpu_device_context *>(dev->context)->description.c_str();
}

static void ggml_backend_lpu_device_get_memory(ggml_backend_dev_t dev,
        size_t * free_out, size_t * total_out) {
    auto * ctx = static_cast<ggml_backend_lpu_device_context *>(dev->context);
    lpu_mem_info_impl(ctx->device, free_out, total_out);
}

static enum ggml_backend_dev_type ggml_backend_lpu_device_get_type(ggml_backend_dev_t /*dev*/) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_lpu_device_get_props(ggml_backend_dev_t dev,
        struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_lpu_device_get_name(dev);
    props->description = ggml_backend_lpu_device_get_description(dev);
    props->type        = ggml_backend_lpu_device_get_type(dev);
    ggml_backend_lpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* async               */ true,
        /* host_buffer         */ false,
        /* buffer_from_host_ptr*/ false,
        /* events              */ false,
    };
}

static ggml_backend_t ggml_backend_lpu_init_backend(ggml_backend_dev_t dev,
        const char * /*params*/) {
    auto * dev_ctx = static_cast<ggml_backend_lpu_device_context *>(dev->context);

    lpu_set_device_impl(dev_ctx->device);

    auto * ctx = new ggml_backend_lpu_context;
    ctx->device = dev_ctx->device;
    ctx->name   = dev_ctx->name;
    LPU_CHECK(lpu_stream_create_impl(&ctx->stream));

    auto * backend  = new ggml_backend;
    backend->guid   = ggml_backend_lpu_guid();
    backend->iface  = ggml_backend_lpu_interface;
    backend->device = dev;
    backend->context = ctx;
    return backend;
}

// Buffer type is created lazily; one per device
static ggml_backend_buffer_type_t ggml_backend_lpu_device_get_buffer_type(
        ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_backend_lpu_device_context *>(dev->context);

    // One static buffer-type per device (devices are small in number)
    static std::vector<ggml_backend_buffer_type> bt_vec;
    static std::vector<ggml_backend_lpu_buffer_type_context> bt_ctx_vec;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    if ((int)bt_vec.size() <= dev_ctx->device) {
        bt_vec.resize(dev_ctx->device + 1);
        bt_ctx_vec.resize(dev_ctx->device + 1);
    }

    auto & bt     = bt_vec[dev_ctx->device];
    auto & bt_ctx = bt_ctx_vec[dev_ctx->device];

    if (bt.context == nullptr) {
        bt_ctx.device = dev_ctx->device;
        bt_ctx.name   = dev_ctx->name;
        bt.iface      = ggml_backend_lpu_buffer_type_interface;
        bt.device     = dev;
        bt.context    = &bt_ctx;
    }

    return &bt;
}

static bool ggml_backend_lpu_device_supports_op(ggml_backend_dev_t /*dev*/,
        const struct ggml_tensor * op) {
    // ----------------------------------------------------------------
    // Supported ops:  MoE core (P0) + FFN auxiliaries (P1) + metadata (P2)
    // ----------------------------------------------------------------
    switch (op->op) {

        // ── MoE core ──────────────────────────────────────────────────
        case GGML_OP_MUL_MAT_ID:
            // Sparse expert GEMM: start with F32/F16; add Q types once SDK is ready
            return op->src[0]->type == GGML_TYPE_F32 ||
                   op->src[0]->type == GGML_TYPE_F16;

        case GGML_OP_ARGSORT:
            // top-k expert score selection
            return op->src[0]->type == GGML_TYPE_F32 ||
                   op->src[0]->type == GGML_TYPE_F16;

        case GGML_OP_GET_ROWS:
            // weight gather (token-embed, MoE routing probability extract)
            return op->src[0]->type == GGML_TYPE_F32 ||
                   op->src[0]->type == GGML_TYPE_F16;

        // ── Dense FFN ─────────────────────────────────────────────────
        case GGML_OP_MUL_MAT:
            return op->src[0]->type == GGML_TYPE_F32 ||
                   op->src[0]->type == GGML_TYPE_F16;

        // ── Activations ───────────────────────────────────────────────
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }

        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_GEGLU:
                    return true;
                default:
                    return false;
            }

        // ── Norms ─────────────────────────────────────────────────────
        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
            return true;

        // ── Element-wise ──────────────────────────────────────────────
        case GGML_OP_MUL:
        case GGML_OP_ADD:
        case GGML_OP_SCALE:
            return true;

        // ── Attention helpers ─────────────────────────────────────────
        case GGML_OP_SOFT_MAX:
            return true;

        // ── Metadata-only (no compute; prevents spurious H2D copies) ──
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:
            return true;

        default:
            return false;
    }
}

static bool ggml_backend_lpu_device_supports_buft(ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft) {
    // Accept our own device buffer types and host (CPU) buffers
    return ggml_backend_buft_is_lpu(buft) || ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static bool ggml_backend_lpu_device_offload_op(ggml_backend_dev_t dev,
        const struct ggml_tensor * op) {
    // Pull the op to LPU even when weights are still on the host,
    // provided the batch is large enough to amortise the H2D cost.
    auto * ctx  = static_cast<ggml_backend_lpu_device_context *>(dev->context);
    const int threshold = ctx->op_offload_min_batch_size;

    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1] >= threshold;   // batch dimension for dense GEMM
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2] >= threshold;   // token dimension for MoE GEMM
        default:
            return false;
    }
}

static const ggml_backend_device_i ggml_backend_lpu_device_interface = {
    /* get_name              */ ggml_backend_lpu_device_get_name,
    /* get_description       */ ggml_backend_lpu_device_get_description,
    /* get_memory            */ ggml_backend_lpu_device_get_memory,
    /* get_type              */ ggml_backend_lpu_device_get_type,
    /* get_props             */ ggml_backend_lpu_device_get_props,
    /* init_backend          */ ggml_backend_lpu_init_backend,
    /* get_buffer_type       */ ggml_backend_lpu_device_get_buffer_type,
    /* get_host_buffer_type  */ nullptr,          // no pinned host memory yet
    /* buffer_from_host_ptr  */ nullptr,          // no mmap support
    /* supports_op           */ ggml_backend_lpu_device_supports_op,
    /* supports_buft         */ ggml_backend_lpu_device_supports_buft,
    /* offload_op            */ ggml_backend_lpu_device_offload_op,
    /* event_new             */ nullptr,
    /* event_free            */ nullptr,
    /* event_synchronize     */ nullptr,
};

// ============================================================================
// 4.  Registry interface
// ============================================================================

static const char * ggml_backend_lpu_reg_get_name(ggml_backend_reg_t /*reg*/) {
    return LPU_BACKEND_NAME;
}

static size_t ggml_backend_lpu_reg_get_device_count(ggml_backend_reg_t reg) {
    auto * ctx = static_cast<ggml_backend_lpu_reg_context *>(reg->context);
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_lpu_reg_get_device(
        ggml_backend_reg_t reg, size_t index) {
    auto * ctx = static_cast<ggml_backend_lpu_reg_context *>(reg->context);
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static const ggml_backend_reg_i ggml_backend_lpu_reg_interface = {
    /* get_name         */ ggml_backend_lpu_reg_get_name,
    /* get_device_count */ ggml_backend_lpu_reg_get_device_count,
    /* get_device       */ ggml_backend_lpu_reg_get_device,
    /* get_proc_address */ nullptr,   // no extended API exports yet
};

// ============================================================================
// 5.  GUID helper
// ============================================================================

ggml_guid_t ggml_backend_lpu_guid() {
    // A randomly-chosen 16-byte identifier for the LPU backend type.
    // Must differ from all other registered backends.
    static ggml_guid guid = { 0x4c, 0x50, 0x55, 0x62, 0x61, 0x63, 0x6b,
                               0x65, 0x6e, 0x64, 0x00, 0x00, 0x00, 0x00,
                               0x00, 0x01 };
    return &guid;
}

// ============================================================================
// 6.  Public API implementation
// ============================================================================

bool ggml_backend_is_lpu(ggml_backend_t backend) {
    return backend && backend->iface.get_name == ggml_backend_lpu_get_name;
}

bool ggml_backend_buft_is_lpu(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == ggml_backend_lpu_buffer_type_get_name;
}

int ggml_backend_lpu_get_device_count() {
    return lpu_device_count_impl();
}

ggml_backend_dev_t ggml_backend_lpu_get_device(int device) {
    return ggml_backend_lpu_reg()->iface.get_device(ggml_backend_lpu_reg(), device);
}

ggml_backend_t ggml_backend_lpu_init(int device) {
    ggml_backend_dev_t dev = ggml_backend_lpu_get_device(device);
    return dev ? ggml_backend_lpu_init_backend(dev, nullptr) : nullptr;
}

ggml_backend_buffer_type_t ggml_backend_lpu_buffer_type(int device) {
    ggml_backend_dev_t dev = ggml_backend_lpu_get_device(device);
    return dev ? ggml_backend_lpu_device_get_buffer_type(dev) : nullptr;
}

ggml_backend_buffer_type_t ggml_backend_lpu_host_buffer_type() {
    // Pinned host memory is not yet implemented; return nullptr.
    return nullptr;
}

// ============================================================================
// 7.  Registry entry point  (called once by ggml_backend_registry ctor)
// ============================================================================

ggml_backend_reg_t ggml_backend_lpu_reg() {
    static ggml_backend_reg reg{};
    static bool initialized = false;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    if (!initialized) {
        // SDK-level initialisation (must happen once per process)
        LPU_CHECK(lpu_init_impl());

        auto * ctx = new ggml_backend_lpu_reg_context;

        // Read optional env override for the offload batch threshold
        const int min_batch = []() -> int {
            const char * v = getenv("GGML_LPU_OFFLOAD_MIN_BATCH");
            return v ? std::atoi(v) : 32;
        }();

        const int n_dev = lpu_device_count_impl();
        GGML_LOG_INFO("LPU: found %d device(s)\n", n_dev);

        for (int i = 0; i < n_dev; i++) {
            auto * dev_ctx = new ggml_backend_lpu_device_context;
            dev_ctx->device                   = i;
            dev_ctx->name                     = std::string(LPU_BACKEND_NAME) + std::to_string(i);
            dev_ctx->description              = lpu_device_name_impl(i);
            dev_ctx->op_offload_min_batch_size = min_batch;

            auto * dev  = new ggml_backend_device;
            dev->iface   = ggml_backend_lpu_device_interface;
            dev->reg     = &reg;
            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
            GGML_LOG_INFO("LPU:   device %d: %s (%s)\n",
                          i, dev_ctx->name.c_str(), dev_ctx->description.c_str());
        }

        reg.api_version = GGML_BACKEND_API_VERSION;
        reg.iface       = ggml_backend_lpu_reg_interface;
        reg.context     = ctx;

        initialized = true;
    }
    return &reg;
}

// Allow this backend to be dlopen-ed as a standalone shared library
GGML_BACKEND_DL_IMPL(ggml_backend_lpu_reg)
