#pragma once

// lpu_ops.h  -- declarations for per-op LPU dispatch functions
//
// Each function takes the per-backend context and the destination tensor
// (which carries src[0], src[1], ... as inputs and stores op parameters).
// Returns GGML_STATUS_SUCCESS on success, GGML_STATUS_FAILED on error.

#include "ggml.h"
#include "common.h"

#include <cstdint>

// ---------------------------------------------------------------------------
// Op call statistics  (used to prove LPU code paths are actually exercised)
// ---------------------------------------------------------------------------

// Simple counter struct (not thread-safe; good enough for single-threaded tests).
// We avoid std::atomic here to stay free of C++ STL header issues when the
// file is included from plain C translation units.  The .cpp side uses volatile
// for the increment so the compiler does not optimise them away.
struct lpu_op_stats {
    long long mul_mat;
    long long mul_mat_id;
    long long rms_norm;
    long long norm;
    long long unary;    // silu, gelu, relu, ...
    long long glu;      // swiglu, geglu, ...
    long long add;
    long long mul;
    long long scale;
    long long soft_max;
    long long get_rows;
    long long argsort;
    long long total;    // sum of all the above
};

extern lpu_op_stats g_lpu_op_stats;  // defined in lpu_ops.cpp

/// Reset all counters to zero.
void lpu_reset_op_stats();

/// Print a human-readable summary to stderr.
void lpu_print_op_stats();

// ---------------------------------------------------------------------------
// Top-level dispatcher  (called from graph_compute for every node)
// ---------------------------------------------------------------------------
ggml_status ggml_backend_lpu_compute_forward(
        ggml_backend_lpu_context * ctx,
        ggml_tensor              * dst);

// ---------------------------------------------------------------------------
// MoE core ops
// ---------------------------------------------------------------------------

/// Indirect matrix multiply: c[:, e, t] = as[:,:,ids[e,t]] @ b[:, e%r, t]
ggml_status lpu_op_mul_mat_id(ggml_backend_lpu_context * ctx, ggml_tensor * dst);

/// Standard dense matrix multiply: C = B * A^T  (ggml convention)
ggml_status lpu_op_mul_mat   (ggml_backend_lpu_context * ctx, ggml_tensor * dst);

/// Gather rows: dst[i] = src0[src1[i]]
ggml_status lpu_op_get_rows  (ggml_backend_lpu_context * ctx, ggml_tensor * dst);

/// Top-k argsort: returns indices of the k largest elements per row
ggml_status lpu_op_argsort   (ggml_backend_lpu_context * ctx, ggml_tensor * dst);

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------
ggml_status lpu_op_rms_norm  (ggml_backend_lpu_context * ctx, ggml_tensor * dst);
ggml_status lpu_op_norm      (ggml_backend_lpu_context * ctx, ggml_tensor * dst);

// ---------------------------------------------------------------------------
// Element-wise ops
// ---------------------------------------------------------------------------
ggml_status lpu_op_mul       (ggml_backend_lpu_context * ctx, ggml_tensor * dst);
ggml_status lpu_op_add       (ggml_backend_lpu_context * ctx, ggml_tensor * dst);
ggml_status lpu_op_scale     (ggml_backend_lpu_context * ctx, ggml_tensor * dst);

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------
ggml_status lpu_op_unary     (ggml_backend_lpu_context * ctx, ggml_tensor * dst);

/// Gated Linear Units (SwiGLU, GeGLU, ...)
ggml_status lpu_op_glu       (ggml_backend_lpu_context * ctx, ggml_tensor * dst);

// ---------------------------------------------------------------------------
// Attention helper
// ---------------------------------------------------------------------------
ggml_status lpu_op_soft_max  (ggml_backend_lpu_context * ctx, ggml_tensor * dst);
