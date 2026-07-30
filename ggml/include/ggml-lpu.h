#pragma once

// ggml-lpu.h  -- public C API for the LPU ggml backend
//
// This header is installed to the public include path (ggml/include/).
// It follows the same pattern as ggml-cann.h and ggml-zendnn.h.

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

/// Returns (and lazily creates) the singleton backend registry for all LPU
/// devices found in the system.  Called by ggml_backend_registry during startup.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_lpu_reg(void);

/// Creates and returns a compute backend for the device at index \p device.
/// The caller is responsible for calling ggml_backend_free() when done.
GGML_BACKEND_API ggml_backend_t ggml_backend_lpu_init(int device);

/// Returns true if \p backend is an LPU backend.
GGML_BACKEND_API bool ggml_backend_is_lpu(ggml_backend_t backend);

// ---------------------------------------------------------------------------
// Device queries
// ---------------------------------------------------------------------------

/// Returns the number of LPU devices available on the system.
GGML_BACKEND_API int ggml_backend_lpu_get_device_count(void);

/// Returns the ggml_backend_dev_t handle for the device at index \p device.
GGML_BACKEND_API ggml_backend_dev_t ggml_backend_lpu_get_device(int device);

// ---------------------------------------------------------------------------
// Buffer types
// ---------------------------------------------------------------------------

/// Returns the device-side buffer type for the given LPU device.
/// Tensors allocated with this type reside in LPU device memory.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_lpu_buffer_type(int device);

/// Returns a host-side pinned-memory buffer type suitable for fast H2D/D2H
/// transfers.  Returns NULL if the LPU SDK does not support pinned memory.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_lpu_host_buffer_type(void);

/// Returns true if \p buft is an LPU device buffer type.
GGML_BACKEND_API bool ggml_backend_buft_is_lpu(ggml_backend_buffer_type_t buft);

// ---------------------------------------------------------------------------
// Internals (used by ggml-lpu.cpp)
// ---------------------------------------------------------------------------

/// Returns the unique GUID that identifies the LPU backend type.
GGML_BACKEND_API ggml_guid_t ggml_backend_lpu_guid(void);

#ifdef __cplusplus
}
#endif
