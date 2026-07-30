// lpu_sdk.cpp -- LPU SDK 子算子 stub 实现
//
// 策略：手写参考实现（方案 A）
//   每个子算子使用裸指针直接做数学运算，风格与 lpu_ops.cpp 中的 cpu_sgemm 一致。
//   数值结果与 ggml CPU 后端等价，可直接用 test-lpu-backend.cpp 验证正确性。
//
// TODO: 用真实 LPU SDK 调用替换每个函数体，保持接口签名不变。

#include "lpu_sdk.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// 内部辅助：将任意 ggml_type 缓冲区的第 row 行读出为 F32 向量
// ---------------------------------------------------------------------------
static void read_row_f32(const void * src, ggml_type dtype,
                         int64_t row, int64_t dim, float * dst) {
    if (dtype == GGML_TYPE_F32) {
        const float * p = static_cast<const float *>(src) + row * dim;
        std::memcpy(dst, p, (size_t)dim * sizeof(float));
    } else if (dtype == GGML_TYPE_F16) {
        const ggml_fp16_t * p = static_cast<const ggml_fp16_t *>(src) + row * dim;
        ggml_fp16_to_fp32_row(p, dst, dim);
    } else {
        // 其他类型暂不支持，填零并记录
        GGML_LOG_ERROR("lpu_sdk: unsupported dtype %d in read_row_f32\n", (int)dtype);
        std::memset(dst, 0, (size_t)dim * sizeof(float));
    }
}

// 将 F32 向量写回目标缓冲区指定行
static void write_row_f32(void * dst, ggml_type dtype,
                          int64_t row, int64_t dim, const float * src) {
    if (dtype == GGML_TYPE_F32) {
        float * p = static_cast<float *>(dst) + row * dim;
        std::memcpy(p, src, (size_t)dim * sizeof(float));
    } else if (dtype == GGML_TYPE_F16) {
        ggml_fp16_t * p = static_cast<ggml_fp16_t *>(dst) + row * dim;
        ggml_fp32_to_fp16_row(src, p, dim);
    } else {
        GGML_LOG_ERROR("lpu_sdk: unsupported dtype %d in write_row_f32\n", (int)dtype);
    }
}

// ============================================================================
// 2.1  lpu_norm
// ============================================================================
lpu_status_t lpu_norm(
        lpu_stream_t     stream,
        lpu_norm_type_t  norm_type,
        float            eps,
        int64_t          n_tokens,
        int64_t          hidden_dim,
        ggml_type        input_dtype,
        ggml_type        output_dtype,
        const void *     input,
        const void *     weight,
        const void *     bias,
        void *           output) {

    (void)stream;

    // weight と bias は常に F32 と仮定（ggml の norm weight は F32）
    const float * w = static_cast<const float *>(weight);
    const float * b = static_cast<const float *>(bias);  // NULL OK

    std::vector<float> row(hidden_dim);

    for (int64_t t = 0; t < n_tokens; t++) {
        read_row_f32(input, input_dtype, t, hidden_dim, row.data());

        if (norm_type == LPU_NORM_RMS) {
            // RMS Norm: output = input / rms(input) * weight
            float sum_sq = 0.0f;
            for (int64_t i = 0; i < hidden_dim; i++) {
                sum_sq += row[i] * row[i];
            }
            const float rms_inv = 1.0f / std::sqrt(sum_sq / (float)hidden_dim + eps);
            for (int64_t i = 0; i < hidden_dim; i++) {
                row[i] = row[i] * rms_inv * w[i];
            }
        } else {
            // LayerNorm: output = (input - mean) / std * weight + bias
            float mean = 0.0f;
            for (int64_t i = 0; i < hidden_dim; i++) mean += row[i];
            mean /= (float)hidden_dim;

            float var = 0.0f;
            for (int64_t i = 0; i < hidden_dim; i++) {
                float d = row[i] - mean;
                var += d * d;
            }
            const float std_inv = 1.0f / std::sqrt(var / (float)hidden_dim + eps);

            for (int64_t i = 0; i < hidden_dim; i++) {
                row[i] = (row[i] - mean) * std_inv * w[i];
                if (b) row[i] += b[i];
            }
        }

        write_row_f32(output, output_dtype, t, hidden_dim, row.data());
    }

    return LPU_SUCCESS;
}

// ============================================================================
// 2.2  lpu_gemm
//
// output = input × weight^T + bias
// input  : [n_tokens, in_dim]   (行主序)
// weight : [out_dim, in_dim]    (行主序，转置后与 input 相乘)
// output : [n_tokens, out_dim]
// ============================================================================
lpu_status_t lpu_gemm(
        lpu_stream_t  stream,
        int64_t       n_tokens,
        int64_t       in_dim,
        int64_t       out_dim,
        ggml_type     input_dtype,
        ggml_type     weight_dtype,
        ggml_type     output_dtype,
        const void *  input,
        const void *  weight,
        const void *  bias,
        void *        output) {

    (void)stream;

    const float * b = static_cast<const float *>(bias);  // NULL OK

    // 将权重全部提前转换为 F32（权重矩阵相对较小，一次性转换）
    std::vector<float> W((size_t)out_dim * in_dim);
    for (int64_t o = 0; o < out_dim; o++) {
        read_row_f32(weight, weight_dtype, o, in_dim, W.data() + o * in_dim);
    }

    std::vector<float> in_row(in_dim);
    std::vector<float> out_row(out_dim);

    for (int64_t t = 0; t < n_tokens; t++) {
        read_row_f32(input, input_dtype, t, in_dim, in_row.data());

        // out_row[o] = dot(W[o], in_row)
        for (int64_t o = 0; o < out_dim; o++) {
            float acc = b ? b[o] : 0.0f;
            const float * w_row = W.data() + o * in_dim;
            for (int64_t k = 0; k < in_dim; k++) {
                acc += w_row[k] * in_row[k];
            }
            out_row[o] = acc;
        }

        write_row_f32(output, output_dtype, t, out_dim, out_row.data());
    }

    return LPU_SUCCESS;
}

// ============================================================================
// 2.3  lpu_activate
// ============================================================================

static float apply_act(lpu_act_type_t act_type, float x) {
    switch (act_type) {
        case LPU_ACT_SILU:
            return x / (1.0f + std::exp(-x));
        case LPU_ACT_GELU:
            // 近似 GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
        case LPU_ACT_RELU:
            return x > 0.0f ? x : 0.0f;
        case LPU_ACT_TANH:
            return std::tanh(x);
        case LPU_ACT_SIGMOID:
            return 1.0f / (1.0f + std::exp(-x));
        default:
            GGML_LOG_ERROR("lpu_sdk: unsupported act_type %d\n", (int)act_type);
            return x;
    }
}

lpu_status_t lpu_activate(
        lpu_stream_t    stream,
        lpu_act_type_t  act_type,
        int64_t         n_tokens,
        int64_t         dim,
        ggml_type       input_dtype,
        ggml_type       output_dtype,
        const void *    input,
        void *          output) {

    (void)stream;

    std::vector<float> row(dim);

    for (int64_t t = 0; t < n_tokens; t++) {
        read_row_f32(input, input_dtype, t, dim, row.data());
        for (int64_t i = 0; i < dim; i++) {
            row[i] = apply_act(act_type, row[i]);
        }
        write_row_f32(output, output_dtype, t, dim, row.data());
    }

    return LPU_SUCCESS;
}

// ============================================================================
// 2.4  lpu_activate_gated
//
// output[i] = act(gate[i]) * up[i]
// ============================================================================
lpu_status_t lpu_activate_gated(
        lpu_stream_t    stream,
        lpu_act_type_t  act_type,
        int64_t         n_tokens,
        int64_t         dim,
        ggml_type       input_dtype,
        ggml_type       output_dtype,
        const void *    gate,
        const void *    up,
        void *          output) {

    (void)stream;

    std::vector<float> gate_row(dim);
    std::vector<float> up_row(dim);

    for (int64_t t = 0; t < n_tokens; t++) {
        read_row_f32(gate, input_dtype, t, dim, gate_row.data());
        read_row_f32(up,   input_dtype, t, dim, up_row.data());
        for (int64_t i = 0; i < dim; i++) {
            gate_row[i] = apply_act(act_type, gate_row[i]) * up_row[i];
        }
        write_row_f32(output, output_dtype, t, dim, gate_row.data());
    }

    return LPU_SUCCESS;
}

// ============================================================================
// 2.5  lpu_add
//
// output[i] = a[i] + b[i]
// ============================================================================
lpu_status_t lpu_add(
        lpu_stream_t  stream,
        int64_t       n_tokens,
        int64_t       dim,
        ggml_type     input_dtype,
        ggml_type     output_dtype,
        const void *  a,
        const void *  b,
        void *        output) {

    (void)stream;

    std::vector<float> a_row(dim);
    std::vector<float> b_row(dim);

    for (int64_t t = 0; t < n_tokens; t++) {
        read_row_f32(a, input_dtype, t, dim, a_row.data());
        read_row_f32(b, input_dtype, t, dim, b_row.data());
        for (int64_t i = 0; i < dim; i++) {
            a_row[i] += b_row[i];
        }
        write_row_f32(output, output_dtype, t, dim, a_row.data());
    }

    return LPU_SUCCESS;
}

// ============================================================================
// 顶层融合接口：lpu_ffn
//
// 标准 FFN (is_gated=0):
//   lpu_norm → lpu_gemm(gate) → lpu_activate → lpu_gemm(down) → lpu_add
//
// 门控 FFN / SwiGLU (is_gated=1):
//   lpu_norm → lpu_gemm(gate) + lpu_gemm(up) → lpu_activate_gated → lpu_gemm(down) → lpu_add
// ============================================================================
lpu_status_t lpu_ffn(
        lpu_stream_t        stream,
        lpu_ffn_desc_t      desc,
        const void *        w_norm,
        const void *        w_gate,
        const void *        w_up,
        const void *        w_down,
        const void *        residual,
        const void *        input,
        void *              output) {

    const int64_t T  = desc.n_tokens;
    const int64_t H  = desc.hidden_dim;
    const int64_t I  = desc.intermediate_dim;
    const ggml_type wt = desc.weight_dtype;
    const ggml_type at = desc.act_dtype;

    // ── 中间缓冲区（调用方分配约定：此处由 lpu_ffn 本身分配，因为它是融合入口）
    std::vector<float> buf_normed((size_t)T * H);
    std::vector<float> buf_gate  ((size_t)T * I);
    std::vector<float> buf_up    (desc.is_gated ? (size_t)T * I : 0);
    std::vector<float> buf_act   ((size_t)T * I);
    std::vector<float> buf_down  ((size_t)T * H);

    lpu_status_t st;

    // 1. 归一化
    st = lpu_norm(stream, desc.norm_type, desc.norm_eps,
                  T, H, at, GGML_TYPE_F32,
                  input, w_norm, /*bias*/nullptr,
                  buf_normed.data());
    if (st != LPU_SUCCESS) return st;

    // 2. gate 投影
    st = lpu_gemm(stream, T, H, I,
                  GGML_TYPE_F32, wt, GGML_TYPE_F32,
                  buf_normed.data(), w_gate, /*bias*/nullptr,
                  buf_gate.data());
    if (st != LPU_SUCCESS) return st;

    // 3. up 投影（门控路径）
    if (desc.is_gated) {
        assert(w_up != nullptr);
        st = lpu_gemm(stream, T, H, I,
                      GGML_TYPE_F32, wt, GGML_TYPE_F32,
                      buf_normed.data(), w_up, /*bias*/nullptr,
                      buf_up.data());
        if (st != LPU_SUCCESS) return st;
    }

    // 4. 激活
    if (desc.is_gated) {
        st = lpu_activate_gated(stream, desc.act_type,
                                T, I,
                                GGML_TYPE_F32, GGML_TYPE_F32,
                                buf_gate.data(), buf_up.data(),
                                buf_act.data());
    } else {
        st = lpu_activate(stream, desc.act_type,
                          T, I,
                          GGML_TYPE_F32, GGML_TYPE_F32,
                          buf_gate.data(),
                          buf_act.data());
    }
    if (st != LPU_SUCCESS) return st;

    // 5. down 投影
    st = lpu_gemm(stream, T, I, H,
                  GGML_TYPE_F32, wt, GGML_TYPE_F32,
                  buf_act.data(), w_down, /*bias*/nullptr,
                  buf_down.data());
    if (st != LPU_SUCCESS) return st;

    // 6. 残差加法 → 写入最终输出
    st = lpu_add(stream, T, H,
                 GGML_TYPE_F32, at,
                 buf_down.data(), residual,
                 output);
    return st;
}
