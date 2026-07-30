#pragma once

// common.h  -- internal context / helper types for the LPU ggml backend
//
// This header is NOT part of the public API; only ggml-lpu.cpp and
// lpu_ops.cpp should include it.

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// LPU SDK forward declarations
//
// Replace the typedefs below with the real LPU SDK types once the SDK
// headers are available (e.g. #include "lpu_runtime.h").
// ---------------------------------------------------------------------------
typedef void * lpu_stream_t;    // async execution stream handle
typedef int    lpu_status_t;    // status/error code returned by SDK calls

#define LPU_SUCCESS 0

// ---------------------------------------------------------------------------
// Registry context  (one per process, holds all discovered devices)
// ---------------------------------------------------------------------------
struct ggml_backend_lpu_reg_context {
    std::vector<struct ggml_backend_device *> devices;
};

// ---------------------------------------------------------------------------
// Device context  (one per physical LPU device)
// ---------------------------------------------------------------------------
struct ggml_backend_lpu_device_context {
    int         device;      // 0-based physical device index
    std::string name;        // "LPU0", "LPU1", ...
    std::string description; // human-readable chip model from LPU SDK

    // Batch-size threshold below which MUL_MAT / MUL_MAT_ID are NOT offloaded
    // even if the weights happen to be on the host.
    // Controlled by env var GGML_LPU_OFFLOAD_MIN_BATCH (default 32).
    int op_offload_min_batch_size;
};

// ---------------------------------------------------------------------------
// Backend (compute stream) context  (one per ggml_backend_t instance)
// ---------------------------------------------------------------------------
struct ggml_backend_lpu_context {
    int          device;
    lpu_stream_t stream;   // async stream used for all kernel launches
    std::string  name;     // "LPU0", "LPU1", ... (cached for get_name())
};

// ---------------------------------------------------------------------------
// Device buffer context  (one per ggml_backend_buffer_t)
// ---------------------------------------------------------------------------
struct ggml_backend_lpu_buffer_context {
    int   device;
    void* data;    // pointer to the LPU device memory allocation
};

// ---------------------------------------------------------------------------
// Buffer-type context  (one per device, created lazily)
// ---------------------------------------------------------------------------
struct ggml_backend_lpu_buffer_type_context {
    int         device;
    std::string name;   // "LPU0", "LPU1", ...
};
