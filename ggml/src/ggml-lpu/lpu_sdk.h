#pragma once

// lpu_sdk.h -- LPU SDK 子算子接口声明
//
// 本文件定义 lpu_ffn() 内部实现所使用的子算子原语。
// 每个接口对应 FFN 数据流中的一个计算步骤：
//   lpu_norm → lpu_gemm → lpu_activate / lpu_activate_gated → lpu_gemm → lpu_add
//
// 接口设计原则：
//   - 无状态纯函数：中间缓冲区由调用方分配，作为指针参数传入
//   - dtype 使用 ggml_type，直接与上层 ggml 张量对齐
//   - 可选参数（bias、norm bias）传 NULL 表示不使用
//
// 详细设计见 docs/lpu-ffn-internal-ops.md

#include "common.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// 枚举类型
// ---------------------------------------------------------------------------

typedef enum {
    LPU_NORM_RMS   = 0,   // RMS Norm（LLaMA 等）
    LPU_NORM_LAYER = 1,   // LayerNorm（GPT-2 等）
} lpu_norm_type_t;

typedef enum {
    LPU_ACT_SILU    = 0,
    LPU_ACT_GELU    = 1,
    LPU_ACT_RELU    = 2,
    LPU_ACT_TANH    = 3,
    LPU_ACT_SIGMOID = 4,
} lpu_act_type_t;

// ---------------------------------------------------------------------------
// 2.1  归一化：lpu_norm
//
// RMS Norm:  output[i] = input[i] / rms(input_row) * weight[i]
// LayerNorm: output[i] = (input[i] - mean) / std * weight[i] + bias[i]
//
// input  : [n_tokens, hidden_dim]
// weight : [hidden_dim]
// bias   : [hidden_dim]，RMS Norm 时传 NULL
// output : [n_tokens, hidden_dim]
// ---------------------------------------------------------------------------
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
    void *           output
);

// ---------------------------------------------------------------------------
// 2.2  矩阵乘法：lpu_gemm
//
// output = input × weight^T + bias   (bias 为 NULL 时省略加法)
//
// input  : [n_tokens, in_dim]
// weight : [out_dim, in_dim]  (ggml 转置约定：C = B * A^T)
// bias   : [out_dim]，无 bias 时传 NULL
// output : [n_tokens, out_dim]
// ---------------------------------------------------------------------------
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
    void *        output
);

// ---------------------------------------------------------------------------
// 2.3  标准激活：lpu_activate
//
// output[i] = act(input[i])，逐元素
//
// input  : [n_tokens, dim]
// output : [n_tokens, dim]
// ---------------------------------------------------------------------------
lpu_status_t lpu_activate(
    lpu_stream_t    stream,
    lpu_act_type_t  act_type,
    int64_t         n_tokens,
    int64_t         dim,
    ggml_type       input_dtype,
    ggml_type       output_dtype,
    const void *    input,
    void *          output
);

// ---------------------------------------------------------------------------
// 2.4  门控激活：lpu_activate_gated
//
// output[i] = act(gate[i]) * up[i]，逐元素乘
//   SwiGLU: act_type = LPU_ACT_SILU
//   GeGLU:  act_type = LPU_ACT_GELU
//
// gate   : [n_tokens, dim]
// up     : [n_tokens, dim]
// output : [n_tokens, dim]
// ---------------------------------------------------------------------------
lpu_status_t lpu_activate_gated(
    lpu_stream_t    stream,
    lpu_act_type_t  act_type,
    int64_t         n_tokens,
    int64_t         dim,
    ggml_type       input_dtype,
    ggml_type       output_dtype,
    const void *    gate,
    const void *    up,
    void *          output
);

// ---------------------------------------------------------------------------
// 2.5  残差加法：lpu_add
//
// output[i] = a[i] + b[i]，逐元素
//
// a      : [n_tokens, dim]  (down_proj 输出)
// b      : [n_tokens, dim]  (残差 x)
// output : [n_tokens, dim]
// ---------------------------------------------------------------------------
lpu_status_t lpu_add(
    lpu_stream_t  stream,
    int64_t       n_tokens,
    int64_t       dim,
    ggml_type     input_dtype,
    ggml_type     output_dtype,
    const void *  a,
    const void *  b,
    void *        output
);

// ---------------------------------------------------------------------------
// 顶层融合接口：lpu_ffn
//
// 融合范围：rms_norm/norm → gate_proj → [up_proj →] act → down_proj → add(residual)
//
// 标准 FFN (is_gated=false):
//   lpu_norm → lpu_gemm(gate) → lpu_activate → lpu_gemm(down) → lpu_add
//
// 门控 FFN / SwiGLU (is_gated=true):
//   lpu_norm → lpu_gemm(gate) + lpu_gemm(up) → lpu_activate_gated → lpu_gemm(down) → lpu_add
// ---------------------------------------------------------------------------

typedef struct {
    int64_t   hidden_dim;        // 隐层维度
    int64_t   intermediate_dim;  // 中间层维度
    lpu_act_type_t  act_type;    // 激活函数类型
    int       is_gated;          // 1 = SwiGLU/GeGLU 双路；0 = 单路
    lpu_norm_type_t norm_type;   // 归一化类型
    float     norm_eps;          // 归一化 epsilon
    ggml_type weight_dtype;      // 权重数据类型
    ggml_type act_dtype;         // 激活数据类型
    int64_t   n_tokens;          // token 数（batch size）
} lpu_ffn_desc_t;

// w_up    : SwiGLU/GeGLU 时传 up_proj 权重指针；标准 FFN 时传 NULL
// residual: 进入 FFN 前的原始激活 x（加到 down_proj 输出上）
// input   : 与 residual 相同指针（norm 的输入）
// output  : 最终输出，对应 add 节点的 dst
lpu_status_t lpu_ffn(
    lpu_stream_t        stream,
    lpu_ffn_desc_t      desc,
    const void *        w_norm,
    const void *        w_gate,
    const void *        w_up,
    const void *        w_down,
    const void *        residual,
    const void *        input,
    void *              output
);

#ifdef __cplusplus
}
#endif
