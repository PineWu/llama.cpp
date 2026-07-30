// lpu_ops.cpp  -- MoE / FFN op implementations for the LPU ggml backend
//
// Strategy: delegate to the ggml CPU compute functions from ggml-cpu/ops.h.
// Since LPU stub buffers are std::malloc (host memory), this gives
// bit-for-bit CPU equivalence with no custom math.
//
// Exception: lpu_op_mul_mat_id uses a hand-rolled gather-GEMM-scatter loop
// because ggml_compute_forward_mul_mat_id is declared `static` in ggml-cpu.c
// and is not exported from any header.

#include "lpu_ops.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-cpu.h"      // ggml_threadpool_new / ggml_threadpool_free

// Internal CPU compute functions (declared in ggml-cpu/ops.h)
#include "../ggml-cpu/ops.h"
// ggml_compute_forward_mul/div/sub are in binary-ops.h (separate from ops.h)
#include "../ggml-cpu/binary-ops.h"
// struct ggml_compute_params
#include "../ggml-cpu/ggml-cpu-impl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Global op-call statistics
// ---------------------------------------------------------------------------

lpu_op_stats g_lpu_op_stats = {};

void lpu_reset_op_stats() {
    g_lpu_op_stats = {};
}

void lpu_print_op_stats() {
    fprintf(stderr,
        "[LPU op stats]\n"
        "  mul_mat    : %lld\n"
        "  mul_mat_id : %lld\n"
        "  rms_norm   : %lld\n"
        "  norm       : %lld\n"
        "  unary      : %lld\n"
        "  glu        : %lld\n"
        "  add        : %lld\n"
        "  mul        : %lld\n"
        "  scale      : %lld\n"
        "  soft_max   : %lld\n"
        "  get_rows   : %lld\n"
        "  argsort    : %lld\n"
        "  total      : %lld\n",
        g_lpu_op_stats.mul_mat,
        g_lpu_op_stats.mul_mat_id,
        g_lpu_op_stats.rms_norm,
        g_lpu_op_stats.norm,
        g_lpu_op_stats.unary,
        g_lpu_op_stats.glu,
        g_lpu_op_stats.add,
        g_lpu_op_stats.mul,
        g_lpu_op_stats.scale,
        g_lpu_op_stats.soft_max,
        g_lpu_op_stats.get_rows,
        g_lpu_op_stats.argsort,
        g_lpu_op_stats.total);
}

// ---------------------------------------------------------------------------
// Single-threaded ggml_compute_params helper
// ---------------------------------------------------------------------------
// ggml_compute_forward_mul_mat (and some others) write to params->threadpool,
// so we must provide a real single-thread threadpool, not null.
// We lazily create it once and reuse across all op calls.

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
    // Reset the work-stealing chunk counter to 0 before every op.
    // ggml_compute_forward_* (especially the llamafile sgemm path) uses
    // ggml_threadpool_chunk_add() to fetch the next work unit atomically.
    // Without this reset the counter is left at its end-of-previous-op value,
    // causing subsequent ops to process zero chunks (wrong results / no-op).
    ggml_threadpool_chunk_set(tp, 0);

    struct ggml_compute_params p;
    memset(&p, 0, sizeof(p));
    p.ith        = 0;
    p.nth        = 1;
    p.wdata      = wdata;
    p.wsize      = wsize;
    p.threadpool = tp;
    return p;
}

// ---------------------------------------------------------------------------
// Helper: simple host-side F32 matrix multiply used by mul_mat_id
//   out[n, m] = sum_k W[m, k] * inp[n, k]      (ggml convention: C = B * A^T)
//
// Layout: W is [M=rows, K=cols] row-major, inp is [N=n_hits, K=cols] row-major,
// out is [N=n_hits, M=rows] row-major so that the n_hits result vectors are
// each contiguous (out + h*rows is one token's full output row). This matches
// the scatter loop below, which memcpys out[h*rows .. h*rows+rows) into dst.
// ---------------------------------------------------------------------------
static void cpu_sgemm(const float * A, const float * B,
                      float * C, int M, int N, int K) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[(int64_t)m * K + k] * B[(int64_t)n * K + k];
            }
            C[(int64_t)n * M + m] = acc;
        }
    }
}

// ============================================================================
// Top-level dispatcher
// ============================================================================

ggml_status ggml_backend_lpu_compute_forward(
        ggml_backend_lpu_context * ctx,
        ggml_tensor              * dst) {

    switch (dst->op) {
        case GGML_OP_MUL_MAT:    return lpu_op_mul_mat(ctx, dst);
        case GGML_OP_MUL_MAT_ID: return lpu_op_mul_mat_id(ctx, dst);
        case GGML_OP_GET_ROWS:   return lpu_op_get_rows(ctx, dst);
        case GGML_OP_ARGSORT:    return lpu_op_argsort(ctx, dst);
        case GGML_OP_RMS_NORM:   return lpu_op_rms_norm(ctx, dst);
        case GGML_OP_NORM:       return lpu_op_norm(ctx, dst);
        case GGML_OP_MUL:        return lpu_op_mul(ctx, dst);
        case GGML_OP_ADD:        return lpu_op_add(ctx, dst);
        case GGML_OP_SCALE:      return lpu_op_scale(ctx, dst);
        case GGML_OP_SOFT_MAX:   return lpu_op_soft_max(ctx, dst);
        case GGML_OP_UNARY:      return lpu_op_unary(ctx, dst);
        case GGML_OP_GLU:        return lpu_op_glu(ctx, dst);

        // Metadata-only ops — no data movement needed
        case GGML_OP_CONT:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return GGML_STATUS_SUCCESS;

        default:
            GGML_LOG_ERROR("LPU: unsupported op %s\n", ggml_op_name(dst->op));
            return GGML_STATUS_FAILED;
    }
}

// ============================================================================
// Dense matrix multiply — delegate to CPU
// ggml_compute_forward_mul_mat may need a work buffer for quantised inputs.
// We allocate one sized to the worst-case requirement.
// ============================================================================
ggml_status lpu_op_mul_mat(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.mul_mat++;
    g_lpu_op_stats.total++;

    // Work buffer: CPU mul_mat uses wdata for type-converted rows.
    // Size formula from ggml-cpu.c: nth * (ne10 * type_size(vec_dot_type))
    // We conservatively allocate ne10 * sizeof(float) * 4 which is always enough
    // for a single thread (nth=1).
    const ggml_tensor * src1 = dst->src[1];
    const size_t wsize = (size_t)src1->ne[0] * sizeof(float) * 4;
    std::vector<char> wbuf(wsize);

    auto p = single_thread_params(wbuf.data(), wsize);
    ggml_compute_forward_mul_mat(&p, dst);

    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// MoE expert-routing GEMM
//
// ggml_compute_forward_mul_mat_id is `static` in ggml-cpu.c and cannot be
// called from here.  We implement the equivalent gather-GEMM-scatter loop.
//
// Tensors:
//   src[0] = as  [cols, rows, n_expert]        stacked expert weights (F32/F16)
//   src[1] = b   [cols, n_exp_used|1, n_tokens] inputs (F32)
//   src[2] = ids [n_exp_used, n_tokens]         int32 routing indices
//   dst    = c   [rows, n_exp_used, n_tokens]   outputs (F32)
//
// Semantics: c[:, e, t] = as[:, :, ids[e,t]] @ b[:, e % r, t]
// ============================================================================
ggml_status lpu_op_mul_mat_id(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.mul_mat_id++;
    g_lpu_op_stats.total++;

    const ggml_tensor * as  = dst->src[0];
    const ggml_tensor * b   = dst->src[1];
    const ggml_tensor * ids = dst->src[2];

    const int64_t cols       = as->ne[0];
    const int64_t rows       = as->ne[1];
    const int64_t n_expert   = as->ne[2];
    const int64_t n_exp_used = ids->ne[0];
    const int64_t n_tokens   = ids->ne[1];
    const int64_t b_ne1      = b->ne[1];   // 1 = broadcast, n_exp_used = no broadcast

    // Read routing IDs (small tensor, always contiguous int32)
    const int32_t * ids_data = static_cast<const int32_t *>(ids->data);

    // Build routing table: expert_id -> list of (slot, token)
    std::vector<std::vector<std::pair<int,int>>> expert_tokens(n_expert);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t e = 0; e < n_exp_used; e++) {
            int32_t ex = ids_data[t * n_exp_used + e];
            GGML_ASSERT(ex >= 0 && ex < n_expert);
            expert_tokens[ex].emplace_back((int)e, (int)t);
        }
    }

    // Zero output
    std::memset(dst->data, 0, ggml_nbytes(dst));

    // Process each active expert
    for (int64_t ex = 0; ex < n_expert; ex++) {
        const auto & hits = expert_tokens[ex];
        if (hits.empty()) continue;

        const int n_hits = (int)hits.size();

        // Extract weight matrix for this expert [cols x rows], convert to F32
        std::vector<float> W(cols * rows);
        {
            const char * src = static_cast<const char *>(as->data) +
                               (size_t)ex * as->nb[2];
            if (as->type == GGML_TYPE_F16) {
                const ggml_fp16_t * w16 = reinterpret_cast<const ggml_fp16_t *>(src);
                for (int64_t i = 0; i < cols * rows; i++) {
                    W[i] = ggml_fp16_to_fp32(w16[i]);
                }
            } else {
                // F32
                std::memcpy(W.data(), src, (size_t)cols * rows * sizeof(float));
            }
        }

        // Gather input vectors for this expert [cols x n_hits]
        std::vector<float> inp(cols * n_hits);
        for (int h = 0; h < n_hits; h++) {
            int slot  = hits[h].first;
            int token = hits[h].second;
            // b is [cols, b_ne1, n_tokens]; row index for (slot%b_ne1, token)
            const size_t b_row_offset =
                (size_t)(slot % b_ne1) * b->nb[1] +
                (size_t)token         * b->nb[2];
            const float * src_row = reinterpret_cast<const float *>(
                static_cast<const char *>(b->data) + b_row_offset);
            std::memcpy(inp.data() + (size_t)h * cols,
                        src_row,
                        (size_t)cols * sizeof(float));
        }

        // GEMM: out[rows, n_hits] = W[rows, cols] * inp[n_hits, cols]^T
        // (cpu_sgemm: C[M, N] = A[M, K] * B[N, K]^T  with M=rows, N=n_hits, K=cols)
        std::vector<float> out(rows * n_hits);
        cpu_sgemm(W.data(), inp.data(), out.data(), (int)rows, n_hits, (int)cols);

        // Scatter output into dst at c[:, slot, token]
        // dst is [rows, n_exp_used, n_tokens], contiguous
        for (int h = 0; h < n_hits; h++) {
            int slot  = hits[h].first;
            int token = hits[h].second;
            float * dst_row = static_cast<float *>(dst->data) +
                              ((int64_t)token * n_exp_used + slot) * rows;
            std::memcpy(dst_row,
                        out.data() + (size_t)h * rows,
                        (size_t)rows * sizeof(float));
        }
    }

    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Remaining ops — all delegate to ggml CPU compute functions
// ============================================================================

ggml_status lpu_op_get_rows(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.get_rows++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_get_rows(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_argsort(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.argsort++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_argsort(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_rms_norm(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.rms_norm++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_rms_norm(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_norm(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.norm++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_norm(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_mul(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.mul++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_mul(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_add(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.add++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_add(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_scale(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.scale++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_scale(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_soft_max(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.soft_max++;
    g_lpu_op_stats.total++;
    // soft_max may need a work buffer for the row-wise exp accumulation.
    const size_t wsize = (size_t)dst->ne[0] * sizeof(float);
    std::vector<char> wbuf(wsize);
    auto p = single_thread_params(wbuf.data(), wsize);
    ggml_compute_forward_soft_max(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_unary(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.unary++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_unary(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}

ggml_status lpu_op_glu(ggml_backend_lpu_context * ctx, ggml_tensor * dst) {
    g_lpu_op_stats.glu++;
    g_lpu_op_stats.total++;
    auto p = single_thread_params();
    ggml_compute_forward_glu(&p, dst);
    GGML_UNUSED(ctx);
    return GGML_STATUS_SUCCESS;
}
