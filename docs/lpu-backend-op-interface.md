# LPU 后端算子接口设计

**日期**: 2026-07-30
**范围**: LPU 后端需要定义和实现的算子接口——基于对硬件架构、融合策略和调度边界的逐步分析。

---

## 1. 硬件定位与融合策略

LPU 是一个**数据流引擎**，其设计目标是以整个 FFN 或 MoE 模块为单位进行融合计算，而非逐个算子分发。

这与当前代码脚手架（`lpu_op_*` 逐节点循环）的模型完全不同。真正的加速接口是两个**融合 SDK 调用**，而非十几个单独的算子函数。

### 融合策略选择

融合在 **`ggml_backend_lpu_graph_compute` 内部**（进入算子循环之前）通过模式匹配实现，而非通过 `graph_optimize` 重写图节点。原因：

- 实现自包含于 LPU 后端，不引入新的合成 op 类型
- 匹配失败时可直接回退到逐 op 的 CPU 代理存根
- 调试路径清晰：匹配命中/未命中均可通过日志追踪

---

## 2. 融合块的边界定义

### 2.1 密集 FFN 块

融合范围：**rms_norm/norm → 投影 → 激活 → 下投影 → add(残差)**

```
x ──► rms_norm ──► gate_proj ──► act ──► down_proj ──► add(x) ──► output
                        │                    ▲
                    up_proj ─────────────────┘  (SwiGLU/GeGLU 时存在)
```

- 残差张量 `x` 既是 `rms_norm` 的输入，也是末尾 `add` 的第二操作数
- 输出写入单独的目标张量（`add` 节点的 `dst`），不原地修改 `x`
- 标准 FFN（单路激活）与门控 FFN（SwiGLU/GeGLU）通过描述符中的标志区分，共用同一个 SDK 调用

### 2.2 MoE FFN 块

融合范围：**rms_norm/norm → 门控投影 → softmax → top-k → mul_mat_id → mul(路由权重) → add(残差)**

```
x ──► rms_norm ──► gate_proj ──► softmax ──► top-k
                                                │
                              expert_W[ids] ◄──┘
                                  │
                              mul(routing_w) ──► add(x) ──► output
```

- 整个路由 + 专家 GEMM + 路由权重加权合并为**一次** SDK 调用
- 输出同样写入单独目标张量

---

## 3. 两个融合 SDK 调用接口

### 3.1 密集 FFN

```c
typedef struct {
    int64_t  hidden_dim;          // 隐层维度
    int64_t  intermediate_dim;    // 中间层维度
    int      act_type;            // 激活类型: SILU / GELU / RELU / ...
    bool     is_gated;            // true = SwiGLU/GeGLU 双路; false = 单路
    int      norm_type;           // 归一化类型: RMS_NORM / NORM(LayerNorm)
    float    norm_eps;            // 归一化 epsilon
    int      weight_dtype;        // 权重数据类型: F32 / F16 / ...
    int      act_dtype;           // 激活数据类型
} lpu_ffn_desc_t;

lpu_status_t lpu_ffn(
    lpu_stream_t        stream,
    lpu_ffn_desc_t      desc,
    const void *        w_norm,     // 归一化层权重
    const void *        w_gate,     // gate 投影权重 [intermediate_dim, hidden_dim]
    const void *        w_up,       // up 投影权重（非门控时为 NULL）
    const void *        w_down,     // down 投影权重 [hidden_dim, intermediate_dim]
    const void *        residual,   // 残差张量 x（加到输出末尾）
    const void *        input,      // 归一化后的激活输入
    void *              output      // 输出张量（对应 add 节点的 dst）
);
```

### 3.2 MoE FFN

```c
typedef struct {
    int64_t  hidden_dim;
    int64_t  intermediate_dim;
    int64_t  n_expert;            // 专家总数
    int64_t  n_expert_used;       // 每 token 激活的专家数（top-k）
    int      act_type;
    int      norm_type;
    float    norm_eps;
    int      weight_dtype;
    int      act_dtype;
} lpu_moe_desc_t;

lpu_status_t lpu_moe(
    lpu_stream_t        stream,
    lpu_moe_desc_t      desc,
    const void *        w_norm,       // 归一化层权重
    const void *        w_gate_proj,  // 门控投影权重（用于路由打分）
    const void *        w_expert,     // 堆叠专家权重 [cols, rows, n_expert]
    const void *        residual,     // 残差张量 x
    const void *        input,        // 归一化后的激活输入
    void *              output        // 输出张量
);
```

---

## 4. `supports_op` 最终清单

| Op | 类别 | 保留原因 |
|----|------|---------|
| `GGML_OP_RMS_NORM` | 可融合上下文 op | 模式匹配器需在 LPU split 中看到前置归一化 |
| `GGML_OP_NORM` | 可融合上下文 op | 同上，用于 LayerNorm 架构（GPT-2 等） |
| `GGML_OP_ADD` | 可融合上下文 op | 模式匹配器需看到残差 add 节点 |
| `GGML_OP_MUL` | 可融合上下文 op | 模式匹配器需看到 MoE 路由权重乘法 |
| `GGML_OP_RESHAPE` / `VIEW` / `PERMUTE` / `TRANSPOSE` / `CONT` | 元数据 op | 阻止调度器插入不必要的 H2D/D2H 拷贝 |

| Op | 状态 | 移除原因 |
|----|------|---------|
| `GGML_OP_MUL_MAT` | **移除** | 融合块内已消耗；独立路由带来拷贝开销而无硬件收益 |
| `GGML_OP_MUL_MAT_ID` | **移除** | 同上，融合块外出现属于异常路径 |
| `GGML_OP_GET_ROWS` | **移除** | 嵌入层查找是内存带宽瓶颈，CPU 处理即可 |
| `GGML_OP_ARGSORT` | **移除** | 仅出现在 MoE 路由内（已融合） |
| `GGML_OP_UNARY` | **移除** | 仅出现在 FFN 激活内（已融合） |
| `GGML_OP_GLU` | **移除** | 仅出现在门控 FFN 内（已融合） |
| `GGML_OP_SOFT_MAX` | **移除** | MoE 路由内已融合；attention 留在 CPU |
| `GGML_OP_SCALE` | **移除** | 仅出现在 attention 缩放内（CPU） |

---

## 5. `graph_compute` 执行结构

```
ggml_backend_lpu_graph_compute(backend, cgraph):
    i = 0
    while i < cgraph->n_nodes:
        if match_moe_pattern(cgraph->nodes, i, &match):
            lpu_moe(ctx->stream, match.desc, ...)
            i += match.n_nodes
        elif match_ffn_pattern(cgraph->nodes, i, &match):
            lpu_ffn(ctx->stream, match.desc, ...)
            i += match.n_nodes
        else:
            // 回退：context op 出现在未匹配模式外，或元数据 op
            ggml_backend_lpu_compute_forward(ctx, cgraph->nodes[i])
            i += 1
    lpu_stream_sync(ctx->stream)
```

模式匹配器输出一个 `lpu_match_t` 结构，携带：
- 匹配消耗的节点数 `n_nodes`
- 填好的描述符 `desc`
- 各权重张量指针（从 `src[]` 读取）
- 残差和输入张量指针
- 输出张量指针（`add` 节点的 `dst`）

---

## 6. 保留的 CPU 代理存根

以下存根**仅在回退路径**中被调用（context op 出现在未匹配的融合模式之外），不对应任何 LPU 硬件调用：

| 函数 | 对应 op |
|------|---------|
| `lpu_op_rms_norm` | `GGML_OP_RMS_NORM` |
| `lpu_op_norm` | `GGML_OP_NORM` |
| `lpu_op_add` | `GGML_OP_ADD` |
| `lpu_op_mul` | `GGML_OP_MUL` |

元数据 op（`RESHAPE`、`VIEW`、`PERMUTE`、`TRANSPOSE`、`CONT`）保持直接返回 `GGML_STATUS_SUCCESS`，无任何计算。

以下存根可以**删除**（op 已从 `supports_op` 移除，调度器不会再路由到 LPU）：

`lpu_op_mul_mat`、`lpu_op_mul_mat_id`、`lpu_op_get_rows`、`lpu_op_argsort`、`lpu_op_unary`、`lpu_op_glu`、`lpu_op_soft_max`、`lpu_op_scale`

---

## 7. 注意事项

**offload_op 需同步更新**：`ggml_backend_lpu_device_offload_op` 当前基于 `MUL_MAT` / `MUL_MAT_ID` 的 batch size 阈值触发。这两个 op 从 `supports_op` 移除后，`offload_op` 永远不会被这两个 op 触发——它的新触发点应改为 LPU split 中的 context op（`RMS_NORM` / `NORM`），以 token 数作为 offload 阈值判断标准。

**模式匹配的健壮性**：llama.cpp 的不同模型架构（LLaMA、Mixtral、Qwen 等）在 FFN/MoE 子图上存在细微差异（例如是否有 bias、norm 位置、专家数量）。模式匹配器应优先匹配结构形状而非节点顺序，并在匹配失败时静默回退到逐 op 执行，而非报错中止。
