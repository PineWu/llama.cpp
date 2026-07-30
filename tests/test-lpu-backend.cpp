// test-lpu-backend.cpp
//
// Validates the LPU ggml backend by running sub-graphs against the CPU golden
// reference and checking that all outputs match within tolerance, and that the
// LPU op-call counters are non-zero (proving LPU code paths were exercised).
//
// TESTS INCLUDED
//   Test 1: FFN block       -- rms_norm, mul_mat(x2), silu(unary), add
//   Test 2: MoE block       -- mul_mat_id, mul, add (4 experts, top-2, 3 tokens)
//   Test 3: Attention ops   -- argsort, get_rows, soft_max (with mask)
//   Test 4: Activations     -- gelu, relu (unary), swiglu, geglu (glu)
//   Test 5: Norm + Scale    -- norm (layer norm), rms_norm (isolated), scale
//   Test 6: MoE large       -- mul_mat_id non-broadcast b (8 experts, top-4, 8 tokens)
//
// BUILD
//   mkdir -p /tmp/lpu_stub/{lib,include}
//   cc -shared -o /tmp/lpu_stub/lib/liblpu_runtime.so -x c /dev/null
//   cc -shared -o /tmp/lpu_stub/lib/liblpu_nn.so      -x c /dev/null
//   cmake -B build -DGGML_LPU=ON -DLPU_INSTALL_DIR=/tmp/lpu_stub -DCMAKE_BUILD_TYPE=Release
//   cmake --build build --target test-lpu-backend -j$(nproc)
//
// RUN
//   ./build/bin/test-lpu-backend
//   cd build && ctest -R test-lpu-backend --output-on-failure -V
//
// PASS CRITERIA
//   All LPU_CHECK lines print "PASS", max-abs-diff <= 1e-4, exit code 0.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

// Access LPU op-call statistics (linked via ggml-lpu)
#include "../ggml/src/ggml-lpu/lpu_ops.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny test framework
// ---------------------------------------------------------------------------

static int g_n_pass = 0;
static int g_n_fail = 0;

#define LPU_CHECK(cond, msg, ...)                                       \
    do {                                                                 \
        if (!(cond)) {                                                   \
            fprintf(stderr, "FAIL [%s:%d] " msg "\n",                   \
                    __FILE__, __LINE__, ##__VA_ARGS__);                  \
            g_n_fail++;                                                  \
        } else {                                                         \
            fprintf(stderr, "PASS  " msg "\n", ##__VA_ARGS__);          \
            g_n_pass++;                                                  \
        }                                                                \
    } while (0)

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

static void rand_f32(std::vector<float> & v, float lo = -1.0f, float hi = 1.0f) {
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
    }
}

static void rand_ids(std::vector<int32_t> & v, int n_tokens, int n_exp_used, int n_expert) {
    v.resize((size_t)n_tokens * n_exp_used);
    for (int tok = 0; tok < n_tokens; tok++) {
        for (int e = 0; e < n_exp_used; e++) {
            v[(size_t)tok * n_exp_used + e] = (int32_t)((tok * n_exp_used + e) % n_expert);
        }
    }
}

// ---------------------------------------------------------------------------
// Key design: every backend call gets its own ggml_context + buffer.
// The graph is rebuilt identically for each backend; no context is shared.
//
// run_single_backend:
//   build_fn  - builds the graph and returns the output node
//   backend   - which backend to run on
//   uploads   - (tensor, data) pairs to upload before compute
//   out_bytes - set to nbytes of the output tensor (for get)
//   result    - output vector (F32) filled on return
//
// We use a callback-style "build function" so the graph can be recreated
// independently for CPU and LPU without duplicating setup code at call sites.
// ---------------------------------------------------------------------------

struct upload_entry {
    // pointer to tensor slot inside the built graph; set by build_fn
    ggml_tensor ** tensor_slot;
    const void   * data;
    size_t         nbytes;
};

typedef ggml_tensor * (*build_fn_t)(ggml_context * ctx, void * user_data);

static void run_single_backend(
        ggml_backend_t backend,
        build_fn_t     build_fn,
        void         * user_data,
        // Array of (tensor_slot, raw_data, nbytes) filled by build_fn
        std::vector<upload_entry> & entries,
        std::vector<float>        & result_f32,
        std::vector<int32_t>      & result_i32,
        bool                        is_int_out = false) {

    struct ggml_init_params ip = { 128 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    assert(ctx);

    // entries[*].tensor_slot will be written by build_fn
    ggml_tensor * out = build_fn(ctx, user_data);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // Use gallocr (not ggml_backend_alloc_ctx_tensors) so graphs containing
    // view ops allocate correctly — view ops alias existing storage and the
    // context-based allocator may underestimate peak buffer size.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, gf);

    for (size_t i = 0; i < entries.size(); i++) {
        ggml_backend_tensor_set(*entries[i].tensor_slot, entries[i].data,
                                0, entries[i].nbytes);
    }

    ggml_backend_graph_compute(backend, gf);

    if (is_int_out) {
        result_i32.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, result_i32.data(), 0, ggml_nbytes(out));
    } else {
        result_f32.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, result_f32.data(), 0, ggml_nbytes(out));
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
}

static float max_diff_f32(const std::vector<float> & a, const std::vector<float> & b) {
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

// Backend helpers
static ggml_backend_t find_lpu_backend() {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strncmp(ggml_backend_dev_name(dev), "LPU", 3) == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

static ggml_backend_t find_cpu_backend() {
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

// ===========================================================================
// Test 1: FFN block
//   input [64, batch] -> rms_norm -> mul_mat(gate) -> silu
//                     -> mul_mat(down) -> add(residual)
// ===========================================================================

struct ffn_data {
    std::vector<float> d_inp, d_gw, d_dw, d_res;
    ggml_tensor * t_inp, * t_gw, * t_dw, * t_res;
};

static ggml_tensor * build_ffn(ggml_context * ctx, void * ud_) {
    ffn_data * ud = (ffn_data *)ud_;
    const int hidden = 64, ffn_dim = 128, batch = 4;
    ud->t_inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden,  batch);
    ud->t_gw  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden,  ffn_dim);
    ud->t_dw  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_dim, hidden);
    ud->t_res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden,  batch);
    return ggml_add(ctx,
               ggml_mul_mat(ctx, ud->t_dw,
                   ggml_silu(ctx,
                       ggml_mul_mat(ctx, ud->t_gw,
                           ggml_rms_norm(ctx, ud->t_inp, 1e-6f)))),
               ud->t_res);
}

static bool test_ffn_block(ggml_backend_t cpu, ggml_backend_t lpu) {
    fprintf(stderr, "\n--- Test 1: FFN block ---\n");
    const float tol = 1e-4f;

    ffn_data ud;
    srand(42);
    ud.d_inp.resize(64*4);  ud.d_gw.resize(64*128);
    ud.d_dw .resize(128*64); ud.d_res.resize(64*4);
    rand_f32(ud.d_inp); rand_f32(ud.d_gw, -0.1f, 0.1f);
    rand_f32(ud.d_dw, -0.1f, 0.1f); rand_f32(ud.d_res);

    std::vector<upload_entry> ents = {
        { &ud.t_inp, ud.d_inp.data(), ud.d_inp.size()*4 },
        { &ud.t_gw,  ud.d_gw.data(),  ud.d_gw.size()*4 },
        { &ud.t_dw,  ud.d_dw.data(),  ud.d_dw.size()*4 },
        { &ud.t_res, ud.d_res.data(), ud.d_res.size()*4 },
    };

    std::vector<float>   cpu_out, lpu_out;
    std::vector<int32_t> dummy_i;

    run_single_backend(cpu, build_ffn, &ud, ents, cpu_out, dummy_i);

    lpu_reset_op_stats();
    run_single_backend(lpu, build_ffn, &ud, ents, lpu_out, dummy_i);
    lpu_print_op_stats();

    float mx = max_diff_f32(cpu_out, lpu_out);
    fprintf(stderr, "  FFN max-abs-diff: %e  (tol %e)\n", mx, tol);
    bool pass = mx <= tol;
    LPU_CHECK(pass,                        "FFN output matches CPU golden (max_abs_diff=%.2e)", mx);
    LPU_CHECK(g_lpu_op_stats.mul_mat  > 0, "LPU MUL_MAT was called (%lld)",     g_lpu_op_stats.mul_mat);
    LPU_CHECK(g_lpu_op_stats.rms_norm > 0, "LPU RMS_NORM was called (%lld)",    g_lpu_op_stats.rms_norm);
    LPU_CHECK(g_lpu_op_stats.unary    > 0, "LPU UNARY(silu) was called (%lld)", g_lpu_op_stats.unary);
    LPU_CHECK(g_lpu_op_stats.add      > 0, "LPU ADD was called (%lld)",         g_lpu_op_stats.add);
    LPU_CHECK(g_lpu_op_stats.total    > 0, "LPU total ops > 0 (%lld)",          g_lpu_op_stats.total);
    return pass;
}

// ===========================================================================
// Test 2: MoE block (broadcast b, 4 experts, top-2, 3 tokens)
// ===========================================================================

struct moe_data {
    int cols, rows, n_expert, n_exp_used, n_tokens;
    bool broadcast_b;  // b_ne1 = 1 (broadcast) vs n_exp_used (non-broadcast)
    std::vector<float>   d_ew, d_inp, d_rw;
    std::vector<int32_t> d_ids;
    ggml_tensor * t_ew, * t_inp, * t_ids, * t_rw;
};

static ggml_tensor * build_moe(ggml_context * ctx, void * ud_) {
    moe_data * ud = (moe_data *)ud_;
    int ne1 = ud->broadcast_b ? 1 : ud->n_exp_used;
    ud->t_ew  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ud->cols, ud->rows, ud->n_expert);
    ud->t_inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ud->cols, ne1, ud->n_tokens);
    ud->t_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ud->n_exp_used, ud->n_tokens);
    ud->t_rw  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, ud->n_exp_used, ud->n_tokens);

    ggml_tensor * moe = ggml_mul_mat_id(ctx, ud->t_ew, ud->t_inp, ud->t_ids);
    ggml_tensor * wgt = ggml_mul(ctx, moe, ud->t_rw);

    // aggregate all n_exp_used slots by summing views
    ggml_tensor * agg = ggml_view_2d(ctx, wgt, ud->rows, ud->n_tokens, wgt->nb[2], 0);
    for (int e = 1; e < ud->n_exp_used; e++) {
        ggml_tensor * se = ggml_view_2d(ctx, wgt, ud->rows, ud->n_tokens, wgt->nb[2],
                                        (size_t)e * ud->rows * sizeof(float));
        agg = ggml_add(ctx, agg, se);
    }
    return agg;
}

static bool run_moe_test(ggml_backend_t cpu, ggml_backend_t lpu, moe_data & ud,
                          const char * label) {
    const float tol = 1e-4f;

    std::vector<upload_entry> ents = {
        { &ud.t_ew,  ud.d_ew.data(),  ud.d_ew.size()*4  },
        { &ud.t_inp, ud.d_inp.data(), ud.d_inp.size()*4  },
        { &ud.t_ids, ud.d_ids.data(), ud.d_ids.size()*4  },
        { &ud.t_rw,  ud.d_rw.data(),  ud.d_rw.size()*4  },
    };

    std::vector<float>   cpu_out, lpu_out;
    std::vector<int32_t> dummy_i;
    run_single_backend(cpu, build_moe, &ud, ents, cpu_out, dummy_i);

    lpu_reset_op_stats();
    run_single_backend(lpu, build_moe, &ud, ents, lpu_out, dummy_i);
    lpu_print_op_stats();

    float mx = max_diff_f32(cpu_out, lpu_out);
    fprintf(stderr, "  %s max-abs-diff: %e  (tol %e)\n", label, mx, tol);
    bool pass = mx <= tol;
    LPU_CHECK(pass,                           "%s matches CPU golden (max_abs_diff=%.2e)", label, mx);
    LPU_CHECK(g_lpu_op_stats.mul_mat_id > 0,  "LPU MUL_MAT_ID was called (%lld)", g_lpu_op_stats.mul_mat_id);
    LPU_CHECK(g_lpu_op_stats.mul        > 0,  "LPU MUL was called (%lld)",         g_lpu_op_stats.mul);
    LPU_CHECK(g_lpu_op_stats.add        > 0,  "LPU ADD was called (%lld)",         g_lpu_op_stats.add);
    LPU_CHECK(g_lpu_op_stats.total      > 0,  "LPU total ops > 0 (%lld)",          g_lpu_op_stats.total);
    return pass;
}

static bool test_moe_block(ggml_backend_t cpu, ggml_backend_t lpu) {
    fprintf(stderr, "\n--- Test 2: MoE block (broadcast b, 4 experts, top-2, 3 tokens) ---\n");
    moe_data ud;
    ud.cols = 16; ud.rows = 32; ud.n_expert = 4; ud.n_exp_used = 2;
    ud.n_tokens = 3; ud.broadcast_b = true;
    srand(123);
    ud.d_ew .resize((size_t)ud.cols*ud.rows*ud.n_expert); rand_f32(ud.d_ew, -0.1f, 0.1f);
    ud.d_inp.resize((size_t)ud.cols*1*ud.n_tokens);        rand_f32(ud.d_inp);
    ud.d_rw .resize((size_t)1*ud.n_exp_used*ud.n_tokens);  rand_f32(ud.d_rw, 0.3f, 0.7f);
    rand_ids(ud.d_ids, ud.n_tokens, ud.n_exp_used, ud.n_expert);
    return run_moe_test(cpu, lpu, ud, "MoE(broadcast)");
}

// ===========================================================================
// Test 3: Attention ops — argsort, get_rows, soft_max+mask
// ===========================================================================

struct argsort_data {
    int N, T;
    std::vector<float> d_logits;
    ggml_tensor * t_logits;
};
static ggml_tensor * build_argsort(ggml_context * ctx, void * ud_) {
    argsort_data * ud = (argsort_data *)ud_;
    ud->t_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ud->N, ud->T);
    return ggml_argsort(ctx, ud->t_logits, GGML_SORT_ORDER_DESC);
}

struct getrows_data {
    int D, V, T;
    std::vector<float>   d_tab;
    std::vector<int32_t> d_ids;
    ggml_tensor * t_tab, * t_ids;
};
static ggml_tensor * build_getrows(ggml_context * ctx, void * ud_) {
    getrows_data * ud = (getrows_data *)ud_;
    ud->t_tab  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ud->D, ud->V);
    ud->t_ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ud->T);
    return ggml_get_rows(ctx, ud->t_tab, ud->t_ids);
}

struct softmax_data {
    int S, H;
    float scale_val;
    std::vector<float>        d_sc;
    std::vector<ggml_fp16_t>  d_mk;
    ggml_tensor * t_sc, * t_mk;
};
static ggml_tensor * build_softmax(ggml_context * ctx, void * ud_) {
    softmax_data * ud = (softmax_data *)ud_;
    ud->t_sc = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ud->S, ud->S, ud->H);
    ud->t_mk = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, ud->S, ud->S);
    return ggml_soft_max_ext(ctx, ud->t_sc, ud->t_mk, ud->scale_val, 0.0f);
}

static bool test_attention_ops(ggml_backend_t cpu, ggml_backend_t lpu) {
    fprintf(stderr, "\n--- Test 3: Attention ops (argsort, get_rows, soft_max+mask) ---\n");
    bool all_pass = true;
    const float tol = 1e-4f;

    // argsort (integer output: compare exactly)
    {
        argsort_data ud; ud.N = 8; ud.T = 4;
        srand(7);
        ud.d_logits.resize((size_t)ud.N*ud.T); rand_f32(ud.d_logits);
        std::vector<upload_entry> ents = {{ &ud.t_logits, ud.d_logits.data(), ud.d_logits.size()*4 }};

        std::vector<float>   dummy_f;
        std::vector<int32_t> cpu_ids, lpu_ids;
        run_single_backend(cpu, build_argsort, &ud, ents, dummy_f, cpu_ids, true);

        lpu_reset_op_stats();
        run_single_backend(lpu, build_argsort, &ud, ents, dummy_f, lpu_ids, true);
        lpu_print_op_stats();

        bool match = (cpu_ids == lpu_ids);
        fprintf(stderr, "  argsort: %s\n", match ? "exact match" : "MISMATCH");
        LPU_CHECK(match, "argsort matches CPU golden (exact)");
        LPU_CHECK(g_lpu_op_stats.argsort > 0, "LPU ARGSORT was called (%lld)", g_lpu_op_stats.argsort);
        all_pass = all_pass && match;
    }

    // get_rows
    {
        getrows_data ud; ud.D = 32; ud.V = 16; ud.T = 4;
        srand(11);
        ud.d_tab.resize((size_t)ud.D*ud.V); rand_f32(ud.d_tab);
        ud.d_ids.resize(ud.T);
        for (int i = 0; i < ud.T; i++) ud.d_ids[i] = (int32_t)(rand() % ud.V);
        std::vector<upload_entry> ents = {
            { &ud.t_tab, ud.d_tab.data(), ud.d_tab.size()*4 },
            { &ud.t_ids, ud.d_ids.data(), ud.d_ids.size()*4 },
        };

        std::vector<float>   cpu_out, lpu_out;
        std::vector<int32_t> dummy_i;
        run_single_backend(cpu, build_getrows, &ud, ents, cpu_out, dummy_i);
        lpu_reset_op_stats();
        run_single_backend(lpu, build_getrows, &ud, ents, lpu_out, dummy_i);
        lpu_print_op_stats();

        float mx = max_diff_f32(cpu_out, lpu_out);
        fprintf(stderr, "  get_rows max-abs-diff: %e  (tol %e)\n", mx, tol);
        bool pass = mx <= tol;
        LPU_CHECK(pass, "get_rows matches CPU golden (max_abs_diff=%.2e)", mx);
        LPU_CHECK(g_lpu_op_stats.get_rows > 0, "LPU GET_ROWS was called (%lld)", g_lpu_op_stats.get_rows);
        all_pass = all_pass && pass;
    }

    // soft_max with F16 causal mask
    {
        softmax_data ud; ud.S = 8; ud.H = 2;
        ud.scale_val = 1.0f / sqrtf((float)ud.S);
        srand(13);
        ud.d_sc.resize((size_t)ud.S*ud.S*ud.H); rand_f32(ud.d_sc, -2.0f, 2.0f);
        ud.d_mk.resize((size_t)ud.S*ud.S);
        for (int i = 0; i < ud.S; i++)
            for (int j = 0; j < ud.S; j++)
                ud.d_mk[(size_t)i*ud.S+j] = ggml_fp32_to_fp16(j <= i ? 0.0f : -1e9f);

        std::vector<upload_entry> ents = {
            { &ud.t_sc, ud.d_sc.data(), ud.d_sc.size()*4           },
            { &ud.t_mk, ud.d_mk.data(), ud.d_mk.size()*sizeof(ggml_fp16_t) },
        };

        std::vector<float>   cpu_out, lpu_out;
        std::vector<int32_t> dummy_i;
        run_single_backend(cpu, build_softmax, &ud, ents, cpu_out, dummy_i);
        lpu_reset_op_stats();
        run_single_backend(lpu, build_softmax, &ud, ents, lpu_out, dummy_i);
        lpu_print_op_stats();

        float mx = max_diff_f32(cpu_out, lpu_out);
        fprintf(stderr, "  soft_max+mask max-abs-diff: %e  (tol %e)\n", mx, tol);
        bool pass = mx <= tol;
        LPU_CHECK(pass, "soft_max+mask matches CPU golden (max_abs_diff=%.2e)", mx);
        LPU_CHECK(g_lpu_op_stats.soft_max > 0, "LPU SOFT_MAX was called (%lld)", g_lpu_op_stats.soft_max);
        all_pass = all_pass && pass;
    }
    return all_pass;
}

// ===========================================================================
// Test 4: Activation variants — gelu, relu, swiglu, geglu
// ===========================================================================

struct act_data {
    int dim0, batch;
    const char * name;
    std::vector<float> d_x;
    ggml_tensor * t_x;
};
static ggml_tensor * build_act(ggml_context * ctx, void * ud_) {
    act_data * ud = (act_data *)ud_;
    ud->t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ud->dim0, ud->batch);
    if      (strcmp(ud->name,"gelu")   == 0) return ggml_gelu(ctx, ud->t_x);
    else if (strcmp(ud->name,"relu")   == 0) return ggml_relu(ctx, ud->t_x);
    else if (strcmp(ud->name,"swiglu") == 0) return ggml_swiglu(ctx, ud->t_x);
    else                                      return ggml_geglu(ctx, ud->t_x);
}

static bool test_activations(ggml_backend_t cpu, ggml_backend_t lpu) {
    fprintf(stderr, "\n--- Test 4: Activations (gelu, relu, swiglu, geglu) ---\n");
    const float tol = 1e-4f;
    const int hidden = 64, batch = 4;
    bool all_pass = true;

    const char * names[] = { "gelu", "relu", "swiglu", "geglu" };
    const int    dims[]  = { hidden, hidden, 2*hidden, 2*hidden };

    for (int ci = 0; ci < 4; ci++) {
        act_data ud;
        ud.name = names[ci]; ud.dim0 = dims[ci]; ud.batch = batch;
        srand(17 + ci);
        ud.d_x.resize((size_t)ud.dim0 * ud.batch); rand_f32(ud.d_x);
        std::vector<upload_entry> ents = {{ &ud.t_x, ud.d_x.data(), ud.d_x.size()*4 }};

        std::vector<float>   cpu_out, lpu_out;
        std::vector<int32_t> dummy_i;
        run_single_backend(cpu, build_act, &ud, ents, cpu_out, dummy_i);
        lpu_reset_op_stats();
        run_single_backend(lpu, build_act, &ud, ents, lpu_out, dummy_i);
        lpu_print_op_stats();

        float mx = max_diff_f32(cpu_out, lpu_out);
        fprintf(stderr, "  %s max-abs-diff: %e  (tol %e)\n", ud.name, mx, tol);
        bool pass = mx <= tol;
        LPU_CHECK(pass, "%s matches CPU golden (max_abs_diff=%.2e)", ud.name, mx);
        all_pass = all_pass && pass;
    }
    return all_pass;
}

// ===========================================================================
// Test 5: Norm + Scale
// ===========================================================================

struct norm_data {
    int hidden, batch;
    const char * name;
    std::vector<float> d_x;
    ggml_tensor * t_x;
};
static ggml_tensor * build_norm(ggml_context * ctx, void * ud_) {
    norm_data * ud = (norm_data *)ud_;
    ud->t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ud->hidden, ud->batch);
    if      (strcmp(ud->name,"norm")     == 0) return ggml_norm(ctx, ud->t_x, 1e-5f);
    else if (strcmp(ud->name,"rms_norm") == 0) return ggml_rms_norm(ctx, ud->t_x, 1e-6f);
    else                                        return ggml_scale(ctx, ud->t_x, 2.5f);
}

static bool test_norm_and_scale(ggml_backend_t cpu, ggml_backend_t lpu) {
    fprintf(stderr, "\n--- Test 5: Norm and Scale ops ---\n");
    const float tol = 1e-4f;
    bool all_pass = true;

    srand(31);
    std::vector<float> d_x((size_t)128 * 8); rand_f32(d_x);

    const char * names[] = { "norm", "rms_norm", "scale" };
    for (int ci = 0; ci < 3; ci++) {
        norm_data ud;
        ud.name = names[ci]; ud.hidden = 128; ud.batch = 8;
        ud.d_x = d_x;  // same data for all three

        std::vector<upload_entry> ents = {{ &ud.t_x, ud.d_x.data(), ud.d_x.size()*4 }};
        std::vector<float>   cpu_out, lpu_out;
        std::vector<int32_t> dummy_i;
        run_single_backend(cpu, build_norm, &ud, ents, cpu_out, dummy_i);
        lpu_reset_op_stats();
        run_single_backend(lpu, build_norm, &ud, ents, lpu_out, dummy_i);
        lpu_print_op_stats();

        float mx = max_diff_f32(cpu_out, lpu_out);
        fprintf(stderr, "  %s max-abs-diff: %e  (tol %e)\n", ud.name, mx, tol);
        bool pass = mx <= tol;
        LPU_CHECK(pass, "%s matches CPU golden (max_abs_diff=%.2e)", ud.name, mx);
        all_pass = all_pass && pass;
    }
    return all_pass;
}

// ===========================================================================
// Test 6: MoE large — non-broadcast b, 8 experts, top-4, 8 tokens
// ===========================================================================

static bool test_moe_large(ggml_backend_t cpu, ggml_backend_t lpu) {
    fprintf(stderr, "\n--- Test 6: MoE large (non-broadcast b, 8 experts, top-4, 8 tokens) ---\n");
    moe_data ud;
    ud.cols = 32; ud.rows = 64; ud.n_expert = 8; ud.n_exp_used = 4;
    ud.n_tokens = 8; ud.broadcast_b = false;
    srand(199);
    ud.d_ew .resize((size_t)ud.cols*ud.rows*ud.n_expert);       rand_f32(ud.d_ew, -0.1f, 0.1f);
    ud.d_inp.resize((size_t)ud.cols*ud.n_exp_used*ud.n_tokens); rand_f32(ud.d_inp);
    ud.d_rw .resize((size_t)1*ud.n_exp_used*ud.n_tokens);        rand_f32(ud.d_rw, 0.1f, 0.9f);
    rand_ids(ud.d_ids, ud.n_tokens, ud.n_exp_used, ud.n_expert);
    return run_moe_test(cpu, lpu, ud, "MoE-large(non-broadcast)");
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    fprintf(stderr, "=== test-lpu-backend ===\n\n");
    ggml_backend_load_all();

    bool lpu_found = false;
    fprintf(stderr, "Registered backends (%zu):\n", ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        fprintf(stderr, "  [%zu] %s\n", i, ggml_backend_dev_name(dev));
        if (strncmp(ggml_backend_dev_name(dev), "LPU", 3) == 0) lpu_found = true;
    }
    LPU_CHECK(lpu_found, "LPU device is registered");
    if (!lpu_found) {
        fprintf(stderr, "\nLPU device not found; skipping.\n");
        fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", g_n_pass, g_n_fail);
        return g_n_fail > 0 ? 1 : 0;
    }

    ggml_backend_t cpu = find_cpu_backend();
    ggml_backend_t lpu = find_lpu_backend();
    LPU_CHECK(cpu != nullptr, "CPU backend initialised");
    LPU_CHECK(lpu != nullptr, "LPU backend initialised");
    if (!cpu || !lpu) { fprintf(stderr, "Backend init failed.\n"); return 1; }

    fprintf(stderr, "\nCPU backend: %s\n", ggml_backend_name(cpu));
    fprintf(stderr, "LPU backend: %s\n", ggml_backend_name(lpu));

    test_ffn_block     (cpu, lpu);
    test_moe_block     (cpu, lpu);
    test_attention_ops (cpu, lpu);
    test_activations   (cpu, lpu);
    test_norm_and_scale(cpu, lpu);
    test_moe_large     (cpu, lpu);

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", g_n_pass, g_n_fail);
    ggml_backend_free(cpu);
    ggml_backend_free(lpu);
    return g_n_fail > 0 ? 1 : 0;
}
