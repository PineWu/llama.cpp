# LPU SDK FFN 内部子算子接口设计

**日期**: 2026-07-30
**范围**: `lpu_ffn()` 黑盒接口内部实现时，各计算步骤对应的子算子接口定义。

上层融合接口 `lpu_ffn()` 的设计见 [lpu-backend-op-interface.md](lpu-backend-op-interface.md)。

---

## 1. 设计原则

- **方案 A：调用方分配中间缓冲区**。每个子算子是无状态纯函数，输入输出指针全部由调用方（`lpu_ffn()` 实现）传入，不在接口内部分配内存。
- **所有接口分离 `input_dtype` / `output_dtype`**，为将来在任意阶段插入精度转换预留空间。
- **可选参数传 `NULL`** 表示不使用（`bias`、`norm` 的 `bias`），接口签名统一，不按类型分叉。
- **激活函数分两个独立接口**：标准激活（单输入）与门控激活（双输入），语义清晰，不靠 `NULL` 隐式区分。

---

## 2. 子算子接口定义

### 2.1 归一化：`lpu_norm`

```c
// norm_type 枚举
typedef enum {
    LPU_NORM_RMS   = 0,   // RMS Norm（LLaMA 等）
    LPU_NORM_LAYER = 1,   // LayerNorm（GPT-2 等）
} lpu_norm_type_t;

lpu_status_t lpu_norm(
    lpu_stream_t     stream,
    lpu_norm_type_t  norm_type,    // 归一化类型
    float            eps,          // 数值稳定 epsilon
    int64_t          n_tokens,     // token 数（batch size）
    int64_t          hidden_dim,   // 隐层维度
    int              input_dtype,  // 输入数据类型
    int              output_dtype, // 输出数据类型
    const void *     input,        // [n_tokens, hidden_dim]
    const void *     weight,       // [hidden_dim]  缩放向量
    const void *     bias,         // [hidden_dim]  偏置向量；RMS Norm 时传 NULL
    void *           output        // [n_tokens, hidden_dim]
);
```

**语义**：
- RMS Norm：`output[i] = input[i] / rms(input_row) * weight[i]`
- LayerNorm：`output[i] = (input[i] - mean(input_row)) / std(input_row) * weight[i] + bias[i]`

---

### 2.2 矩阵乘法：`lpu_gemm`

```c
lpu_status_t lpu_gemm(
    lpu_stream_t  stream,
    int64_t       n_tokens,      // token 数（M 维度）
    int64_t       in_dim,        // 输入维度（K 维度）
    int64_t       out_dim,       // 输出维度（N 维度）
    int           input_dtype,   // 输入激活数据类型
    int           weight_dtype,  // 权重数据类型（可与输入不同，如 F16 权重 + F32 激活）
    int           output_dtype,  // 输出数据类型
    const void *  input,         // [n_tokens, in_dim]
    const void *  weight,        // [out_dim, in_dim]（ggml 转置约定：C = B * A^T）
    const void *  bias,          // [out_dim]；无 bias 时传 NULL
    void *        output         // [n_tokens, out_dim]
);
```

**语义**：`output = input × weight^T + bias`（bias 为 NULL 时省略加法）

**在 FFN 中的调用位置**：
- `gate_proj`：`in_dim = hidden_dim`，`out_dim = intermediate_dim`
- `up_proj`（SwiGLU 时）：同 `gate_proj`
- `down_proj`：`in_dim = intermediate_dim`，`out_dim = hidden_dim`

---

### 2.3 标准激活：`lpu_activate`

```c
// act_type 枚举（供 lpu_activate 和 lpu_activate_gated 共用）
typedef enum {
    LPU_ACT_SILU    = 0,
    LPU_ACT_GELU    = 1,
    LPU_ACT_RELU    = 2,
    LPU_ACT_TANH    = 3,
    LPU_ACT_SIGMOID = 4,
} lpu_act_type_t;

lpu_status_t lpu_activate(
    lpu_stream_t    stream,
    lpu_act_type_t  act_type,     // 激活函数类型
    int64_t         n_tokens,
    int64_t         dim,
    int             input_dtype,
    int             output_dtype,
    const void *    input,        // [n_tokens, dim]
    void *          output        // [n_tokens, dim]
);
```

**语义**：`output[i] = act(input[i])`，逐元素

---

### 2.4 门控激活：`lpu_activate_gated`

```c
lpu_status_t lpu_activate_gated(
    lpu_stream_t    stream,
    lpu_act_type_t  act_type,     // 作用于 gate 分量的激活函数（SiLU / GELU）
    int64_t         n_tokens,
    int64_t         dim,
    int             input_dtype,  // gate 和 up 的输入类型（两者相同）
    int             output_dtype,
    const void *    gate,         // [n_tokens, dim]  门控分量
    const void *    up,           // [n_tokens, dim]  线性分量
    void *          output        // [n_tokens, dim]  = act(gate) ⊙ up
);
```

**语义**：`output[i] = act(gate[i]) * up[i]`，逐元素乘
- SwiGLU：`act_type = LPU_ACT_SILU`
- GeGLU：`act_type = LPU_ACT_GELU`

---

### 2.5 残差加法：`lpu_add`

```c
lpu_status_t lpu_add(
    lpu_stream_t  stream,
    int64_t       n_tokens,
    int64_t       dim,
    int           input_dtype,   // a 和 b 的输入类型（两者相同）
    int           output_dtype,
    const void *  a,             // [n_tokens, dim]  第一操作数（down_proj 输出）
    const void *  b,             // [n_tokens, dim]  第二操作数（残差 x）
    void *        output         // [n_tokens, dim]
);
```

**语义**：`output[i] = a[i] + b[i]`，逐元素

> **注**：当前作为独立接口。后续若性能分析表明 down_proj + 残差加的内存往返是瓶颈，可考虑将 `b`（残差）融入 `lpu_gemm` 的可选参数，消除 down_proj 结果的中间落盘。

---

## 3. `lpu_ffn()` 内部调用序列

### 3.1 中间缓冲区分配（调用方负责）

| 缓冲区 | 形状 | 用途 |
|--------|------|------|
| `buf_normed` | `[n_tokens, hidden_dim]` | norm 输出 |
| `buf_gate` | `[n_tokens, intermediate_dim]` | gate_proj 输出 |
| `buf_up` | `[n_tokens, intermediate_dim]` | up_proj 输出（SwiGLU 时） |
| `buf_act` | `[n_tokens, intermediate_dim]` | 激活输出 |
| `buf_down` | `[n_tokens, hidden_dim]` | down_proj 输出 |

### 3.2 标准 FFN（`is_gated = false`）

```
lpu_norm(input → buf_normed)
lpu_gemm(buf_normed → buf_gate)          // gate_proj
lpu_activate(buf_gate → buf_act)
lpu_gemm(buf_act → buf_down)             // down_proj
lpu_add(buf_down + residual → output)
```

### 3.3 门控 FFN / SwiGLU（`is_gated = true`）

```
lpu_norm(input → buf_normed)
lpu_gemm(buf_normed → buf_gate)          // gate_proj
lpu_gemm(buf_normed → buf_up)            // up_proj（与 gate_proj 并行或顺序，由实现决定）
lpu_activate_gated(buf_gate, buf_up → buf_act)
lpu_gemm(buf_act → buf_down)             // down_proj
lpu_add(buf_down + residual → output)
```

---

## 4. 接口间的数据流图

```
input (残差 x)
  │
  ├──────────────────────────────────────────────► lpu_add ──► output
  │                                                  ▲
  ▼                                                  │
lpu_norm                                         buf_down
  │                                                  ▲
  ▼                                                  │
buf_normed ──► lpu_gemm(gate) ──► buf_gate       lpu_gemm(down)
  │                                   │               ▲
  │            lpu_gemm(up) ◄─────────┤           buf_act
  │                  │                │               ▲
  │            buf_up│            buf_gate    lpu_activate / lpu_activate_gated
  │                  └────────────────┘               │
  └────────────────────────────────────────────────►──┘
                  (SwiGLU: gate + up → act_gated)
                  (标准:   gate only → activate)
```

---

## 5. 与上层 `lpu_ffn_desc_t` 的对应关系

| `lpu_ffn_desc_t` 字段 | 传递给子算子的位置 |
|----------------------|------------------|
| `hidden_dim` | `lpu_norm.hidden_dim`，`lpu_gemm.in_dim`（gate/up），`lpu_gemm.out_dim`（down），`lpu_add.dim` |
| `intermediate_dim` | `lpu_gemm.out_dim`（gate/up），`lpu_gemm.in_dim`（down），`lpu_activate*.dim` |
| `act_type` | `lpu_activate.act_type` / `lpu_activate_gated.act_type` |
| `is_gated` | 选择调用 `lpu_activate` 还是 `lpu_activate_gated`，是否调用 up_proj GEMM |
| `norm_type` | `lpu_norm.norm_type` |
| `norm_eps` | `lpu_norm.eps` |
| `weight_dtype` | `lpu_gemm.weight_dtype` |
| `act_dtype` | 各子算子的 `input_dtype` / `output_dtype` |
