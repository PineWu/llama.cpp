# llama.cpp MoE 模型推理全流程分析

> 本文合并自三份 MoE 分析报告,系统梳理 llama.cpp 对 Mixture-of-Experts (MoE) 模型推理的完整支持,聚焦 **top-k 专家选择 → 选定专家执行 → 加权聚合** 三步流水,以 DeepSeek-V2/V3 (deepseek2 架构) 为主要示例,CUDA 后端为主线、CPU 作对照。
> 分析对象: llama.cpp 仓库 (master 分支)
> 整理日期: 2026-06-23

## 目录

- [第一部分:MoE 推理流程总览](#第一部分moe-推理流程总览)
  - [1. MoE 推理背景与 llama.cpp 的设计](#1-moe-推理背景与-llamacpp-的设计)
  - [2. MoE 超参与权重张量](#2-moe-超参与权重张量)
  - [3. DeepSeek2 MoE 层调用入口](#3-deepseek2-moe-层调用入口)
  - [4. `build_moe_ffn` 三步流水详解](#4-build_moe_ffn-三步流水详解)
  - [5. 完整数据流图 (DeepSeek-V3 一层)](#5-完整数据流图-deepseek-v3-一层)
  - [6. 其他 MoE 模型的差异](#6-其他-moe-模型的差异)
  - [7. 性能与设计要点](#7-性能与设计要点)
  - [8. 三步流水对应代码](#8-三步流水对应代码)
- [第二部分:专家激活实现细节](#第二部分专家激活实现细节)
  - [9. 问题定位](#9-问题定位)
  - [10. 数据契约:`ids` 与 `mul_mat_id`](#10-数据契约ids-与-mul_mat_id)
  - [11. CUDA 后端:专家激活四段流水](#11-cuda-后端专家激活四段流水)
  - [12. CPU 后端对照](#12-cpu-后端对照)
  - [13. 输出布局如何喂给加权聚合](#13-输出布局如何喂给加权聚合)
  - [14. 端到端数值示例 (3 token)](#14-端到端数值示例-3-token)
  - [15. 关键设计洞察](#15-关键设计洞察)
- [第三部分:多 token 批处理的路由与高效计算](#第三部分多-token-批处理的路由与高效计算)
  - [16. 场景精确定义](#16-场景精确定义)
  - [17. 按 batch 大小选执行路径(三档)](#17-按-batch-大小选执行路径三档)
  - [18. `mm_ids_helper` 解决路由问题](#18-mm_ids_helper-解决路由问题档位-c-详解)
  - [19. 负载不均如何被天然吸收](#19-负载不均如何被天然吸收)
  - [20. 端到端数值示例 (16 token)](#20-端到端数值示例-16-token)
  - [21. 三档的工程权衡](#21-三档的工程权衡)
- [第四部分:其他硬件后端是否需要按专家重排 token](#第四部分其他硬件后端是否需要按专家重排-token)
  - [22. 结论先行](#22-结论先行)
  - [23. 需要重排的后端:机制差异详解](#23-需要重排的后端机制差异详解)
  - [24. 不做显式重排的后端:替代方案](#24-不做显式重排的后端替代方案)
  - [25. 为什么有差异:重排的本质与触发条件](#25-为什么有差异重排的本质与触发条件)
  - [26. 权重搬运总量的跨后端对比(带宽维度)](#26-权重搬运总量的跨后端对比带宽维度)
  - [27. `mm_ids_helper` 重排细节详解(16 token x 8 专家实例)](#27-mm_ids_helper-重排细节详解16-token-x-8-专家实例)
  - [28. 小 batch(decode)的统一例外](#28-小-batchdecode的统一例外)
  - [29. 总结](#29-总结)
- [附录:关键文件速查](#附录关键文件速查)

---

# 第一部分:MoE 推理流程总览

## 1. MoE 推理背景与 llama.cpp 的设计

MoE (混合专家) 层用一个轻量 **路由器 (router/gate)** 为每个 token 选出 top-k 个专家 FFN,只计算被选中的专家,再把它们的输出按路由权重加权求和。好处是: 参数量大 (专家多) 但单次计算量小 (每 token 只激活 k 个),实现"容量-算力解耦"。

llama.cpp 的实现要点:
1. **图构建层** ([src/llama-graph.cpp](src/llama-graph.cpp) `build_moe_ffn`): 一套通用 MoE 图构建函数,所有 MoE 模型复用,支持 shared expert、group-limited routing、多种 gating、权重归一化等。
2. **核心算子 `ggml_mul_mat_id`**: "按专家索引做矩阵乘" - 只对被选中的专家执行 GEMM,是稀疏计算的关键。
3. **每架构一个文件** ([src/models/](src/models/)): 只负责声明专家权重张量、调 `build_moe_ffn`、(可选)拼接 shared expert。

下面以 DeepSeek-V2/V3 ([src/models/deepseek2.cpp](src/models/deepseek2.cpp)) 为主线,它同时含 shared expert 与 group-limited routing,最能体现全貌。

## 2. MoE 超参与权重张量

### 2.1 超参 ([src/llama-hparams.h:53-101](src/llama-hparams.h#L53))

| 字段 | 含义 | DeepSeek-V3 典型值 |
|------|------|--------------------|
| `n_expert` | 专家总数 | 256 |
| `n_expert_used` | 每 token 选中专家数 (top-k) | 8 |
| `n_expert_shared` | shared expert 数 (恒激活) | 1 |
| `n_ff_exp` | 路由专家 FFN 维度 | 2048 |
| `n_ff_shexp` | shared expert FFN 维度 | 2048 |
| `n_expert_groups` | 专家分组数 (group-limited routing) | 8 |
| `n_group_used` | 每 token 选中的组数 | 4 |
| `n_group_experts` | 每组专家数 | 32 |
| `expert_group_scale` | 组路由缩放 | 0.05 |
| `expert_weights_scale` | 权重后缩放 (sigmoid gating 用,如 16.0) | 16.0 |
| `expert_weights_norm` | 是否把权重归一化为和=1 | false |
| `expert_gating_func` | gating 类型 (softmax/sigmoid/softmax_weight) | sigmoid |
| `moe_every_n_layers` | 每 N 层一个 MoE (0=层全 MoE) | - |
| `n_layer_dense_lead` | 前几个 dense 层不接 MoE | 3 |

gating 枚举 ([src/llama-hparams.h:12-17](src/llama-hparams.h#L12)):

```c
enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE            = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX         = 1,  // 全 softmax 路由
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID         = 2,  // 各专家独立 sigmoid (DeepSeek-V3)
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT  = 3,  // 选完再 softmax
};
```

### 2.2 专家权重张量布局 (DeepSeek2)

每个 MoE 层的权重 ([src/models/deepseek2.cpp](src/models/deepseek2.cpp) `load_arch_tensors`):

| 张量 | 形状 | 说明 |
|------|------|------|
| `ffn_gate_inp` | `[n_embd, n_expert]` | 路由器权重,把 token 投影成专家 logits |
| `ffn_gate_up_exps` | `[n_embd, n_ff_exp*2, n_expert]` | 合并的 gate+up 专家权重 (DeepSeek 把两路合并省一次 mul_mat_id) |
| `ffn_down_exps` | `[n_ff_exp, n_embd, n_expert]` | down 投影专家权重 |
| `ffn_exp_probs_b` | `[n_expert]` | 路由偏置 (可选) |
| `ffn_gate_shexp` / `ffn_up_shexp` / `ffn_down_shexp` | shared expert 三路权重 | 恒激活的共享专家 |

关键:**所有专家权重堆叠在第 3 维**,形成 `[in, out, n_expert]` 的三维张量,这正是 `ggml_mul_mat_id` 的输入格式 - 一次调用即可让不同 token 用不同专家。

---

## 2.3 权重打包与间接寻址:为何一次调用即可让不同 token 用不同专家

上文的"一次调用即可让不同 token 用不同专家"是 `ggml_mul_mat_id` 的核心机制。它由三个设计叠加而成:**权重打包成三维张量 + `ids` 作为间接索引表 + 单个图节点把路由推进内核**。逐步拆开如下。

### (a) 权重打包:把 N 个专家"摞"成一个三维张量

朴素想象是"每个专家一个独立权重张量 `W_0, W_1, ..., W_255`"。llama.cpp 不是这样——它把全部 N 个专家的权重矩阵**沿第 3 维物理堆叠**成一个张量:

```
as : [cols, rows, n_expert]
        ↑     ↑     ↑
      输入维  输出维  专家号(第3维 = channel)
```

DeepSeek-V3 down 投影:`as = ffn_down_exps [n_ff_exp=2048, n_embd=7168, n_expert=256]`。即:

- `as[:, :, 0]`   = 专家 0 的权重矩阵,形状 `[2048, 7168]`
- `as[:, :, 1]`   = 专家 1 的权重矩阵
- ...
- `as[:, :, 42]`  = 专家 42 的权重矩阵
- ...
- `as[:, :, 255]` = 专家 255 的权重矩阵

它们在内存里**连续摆放**(第 3 维步长 `nb02` = 一个专家权重的字节数)。这是模型加载时一次性摆好的布局,**推理时不再搬运、不复制**。所有 256 个专家始终"都在那儿",物理上是同一个大张量的不同切片。

> 注意:这里的"第 3 维"指 ggml 的 `ne[2]`(0-indexed 的第 2 维)。上文 2.2 节说的"第 3 维"就是它。

### (b) `ids` 是"间接索引表"(indirection table),不是数据

`ids : [n_expert_used(k), n_tokens]`,int32。它**不是权重也不是输入**,而是一张"查表指令":

```
ids[e, t] = 专家号   // 第 t 个 token 的第 e 个被选专家,用 as 的第几个切片
```

比如 `ids[0, 5] = 42` 的含义是:"token 5 的第 0 个专家槽,请去用 `as[:, :, 42]` 这片权重"。

这就是 op 的语义定义(直接引自 [ggml/src/ggml.c:3283](ggml/src/ggml.c#L3283)):

```
c[:, e, t] = as[:, :, ids[e,t]] @ b[:, e, t]
                  ^^^^^^^^^^^
                  关键:ids[e,t] 是个整数,用来索引 as 的第 3 维
```

`as[:, :, ids[e,t]]` 这一步就是"间接寻址":用 `ids[e,t]` 的整数值去选 `as` 的某一片。`ids[e,t]` 不同 → 选不同的切片 → 不同的专家权重。

### (c) 为什么是"一次调用":一个图节点,把路由推进内核

这是最关键的一点。对比两种实现 MoE 的方式。

**朴素方式(不可行)**:在建图时(CPU 上搭 `ggml_cgraph`)并不知道每个 token 该用哪个专家——因为路由结果 `ids` 是**运行时**才算出的(取决于输入)。所以没法在建图时写:

```
for t in tokens:
    for e in k:
        out = mul_mat(W[ids[e,t]], x[t])   # ← ids[e,t] 此时还不知道!
```

若硬要支持不同 token 用不同专家,只能:
- 要么为每个专家建一个 `mul_mat` 节点(N=256 个节点),把所有 token 都喂进去——**稠密计算,违背 MoE 稀疏性**(256 个专家全算一遍);
- 要么建 N 个分支按 token 路由——但路由是数据相关的,图是静态的,做不到。

**llama.cpp 的方式:`mul_mat_id` 单节点 + 运行时间接寻址**:

建图时只建**一个** `ggml_mul_mat_id(as, b, ids)` 节点。`ids` 作为该节点的第三个输入张量(src[2]),其**具体数值要到运行时才填**(由前面的 `argsort_top_k` 算出)。这个节点的后端内核在执行时,才去读 `ids` 的值,据此决定每个 (token, 槽) 用哪片 `as`。

```
建图时(CPU):  graph 里就一个节点  mul_mat_id(as, b, ids)
                ↑ as/b 是权重/输入张量,ids 是"待填的索引张量"
运行时(GPU): 内核读 ids[e,t] 的整数值
              → 算出权重偏移 = ids[e,t] * nb02
              → 从 as + 该偏移 取出专家 ids[e,t] 的权重
              → 对 token t 的输入做 GEMM
              → 结果写到 c[:, e, t]
```

所以"一次调用"指的是:**图层面只有一个 op 节点、一次 kernel 调度**(具体到 CUDA 是一组 block),而不是"对每个专家调一次 mul_mat"。路由选择(哪个 token 用哪个专家)被**推迟到内核内部**用 `ids` 间接寻址完成,不再体现在图结构上。

### (d) 具体 indexing 数学(以 down 投影为例)

```
as  : [2048, 7168, 256]     nb02 = 2048*7168*sizeof(元素)  ← 一个专家权重的字节跨度
b   : [2048, 8, n_tokens]   b[:, e, t] = token t 在槽 e 的输入向量 [2048]
ids : [8, n_tokens]          ids[e, t] ∈ [0, 255]
c   : [7168, 8, n_tokens]    c[:, e, t] = 输出向量 [7168]
```

对每个 `(e, t)` 对,内核做的事:

```
expert_id = ids[e, t]                              // 读间接索引,如 42
W = (char*)as->data + expert_id * nb02             // 取出 as[:,:,42] 这片权重
x = b[:, e, t]                                     // token t 在槽 e 的输入
c[:, e, t] = W^T @ x                               // 该专家的 FFN 输出
```

因为 `ids[:, t1]` 和 `ids[:, t2]` 通常不同(不同 token 激活不同专家),所以:
- token t1 用 `as[:,:,42]`、`as[:,:,7]`、...
- token t2 用 `as[:,:,199]`、`as[:,:,150]`、...

**同一个 `as` 张量,不同 token 通过 `ids` 的不同取值访问不同切片**——这就是"不同 token 用不同专家"的物理实现。

### (e) 配合 `mm_ids_helper` 实现高效(避免零散访存)

如果内核真的按 `(e,t)` 顺序逐个去 `as` 里取切片,会因专家交错导致 cache 颠簸(第 11 节详述的核心难点)。所以实际 CUDA 实现做了重排:

1. `mm_ids_helper` 扫描 `ids`,把"同一专家被哪些 (token,槽) 命中"统计成 CSR 式分段(`expert_bounds`)。
2. MMQ 内核的 grid 第 z 维 = `n_expert = 256`,**每个专家一组 block**;该组只从 `as` 取自己那一片(`as + e*nb02`),对紧凑缓冲里分到它的 token 做 GEMM。
3. 未被任何 token 命中的专家(`expert_bounds[e]==expert_bounds[e+1]`)→ 该组无工作量 → 零计算。

这把"间接寻址"组织成了"按专家分组的连续 GEMM",既保留了"单节点、稀疏、按 ids 选专家"的语义,又保证了访存连续。

### 小结

> "**所有专家权重堆叠在第 3 维**"提供了"所有专家都在一个张量里、按切片寻址"的物理布局;"**`ids` 是间接索引表**"让运行时能用一个整数选出某片专家权重;"**单个 `mul_mat_id` 节点**"把"哪个 token 用哪个专家"这个数据相关的路由决策,从静态的图结构里移到了内核内部的间接寻址——所以一次 op 调用 + 一组 kernel,就能让每个 token 各自走自己的专家,且只算被选中的那些(稀疏)。

---

## 3. DeepSeek2 MoE 层调用入口

[src/models/deepseek2.cpp:385-414](src/models/deepseek2.cpp#L385) 在每层 FFN 处:

```cpp
// MoE 分支 (dense_lead 层之后)
ggml_tensor * moe_out = build_moe_ffn(cur,
    model.layers[il].ffn_gate_inp,      // 路由器
    model.layers[il].ffn_up_exps,
    model.layers[il].ffn_gate_exps,
    model.layers[il].ffn_down_exps,
    model.layers[il].ffn_exp_probs_b,    // 路由偏置
    n_expert, n_expert_used,
    LLM_FFN_SILU, hparams.expert_weights_norm,
    hparams.expert_weights_scale,
    (llama_expert_gating_func_type) hparams.expert_gating_func,
    il,
    nullptr,
    model.layers[il].ffn_gate_up_exps);  // 合并 gate+up 路径
cb(moe_out, "ffn_moe_out", il);

// FFN shared expert (恒激活,无路由)
{
    ggml_tensor * ffn_shexp = build_ffn(cur,
        model.layers[il].ffn_up_shexp, NULL, NULL,
        model.layers[il].ffn_gate_shexp, NULL, NULL,
        model.layers[il].ffn_down_shexp, NULL, NULL,
        NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);   // 路由专家 + shared 专家 相加
    cb(cur, "ffn_out", il);
}
cur = ggml_add(ctx0, cur, ffn_inp);             // 残差
```

DeepSeek 的设计:**路由专家 (top-k, 稀疏) + shared expert (恒定, 稠密) 并行计算后相加**。shared expert 承担"所有 token 都需要的通用计算",路由专家只补充特化部分。

## 4. `build_moe_ffn` 三步流水详解

> 核心函数: [src/llama-graph.cpp:1441-1795](src/llama-graph.cpp#L1441)

这是所有 MoE 模型共用的图构建器。下面按 **①路由+选专家 → ②执行专家FFN → ③加权聚合** 三步拆解,标注每步对应的 ggml 算子与张量形状。

### 4.0 输入

```
cur       : [n_embd, n_tokens]               输入隐状态
gate_inp  : [n_embd, n_expert]               路由器权重
up_exps   : [n_embd, n_ff_exp, n_expert]     up 投影专家 (或用合并的 gate_up_exps)
gate_exps : [n_embd, n_ff_exp, n_expert]     gate 投影专家
down_exps : [n_ff_exp, n_embd, n_expert]     down 投影专家
```

### 4.1 第①步: 路由 + top-k 专家选择

#### (a) 路由 logits 与 gating ([llama-graph.cpp:1478-1517](src/llama-graph.cpp#L1478))

```c
logits = build_lora_mm(gate_inp, cur);        // [n_expert, n_tokens]  路由线性层
if (gate_inp_b) logits = ggml_add(ctx0, logits, gate_inp_b);
if (exp_probs_b) selection_probs = ggml_add(ctx0, probs, exp_probs_b);  // 加路由偏置

switch (gating_op) {
    case SOFTMAX:        probs = ggml_soft_max(ctx0, logits);   // [n_expert, n_tokens]
    case SIGMOID:        probs = ggml_sigmoid(ctx0, logits);    // DeepSeek-V3
    case SOFTMAX_WEIGHT: probs = logits;                        // 选完再 softmax
}
```

- `selection_probs` 用于 **选哪些专家** (排序依据);
- `probs` 用于 **加权** (最终乘到专家输出上)。
- DeepSeek-V3 用 sigmoid: 每个专家独立概率,选 top-k 后再乘 `expert_weights_scale` 缩放 (如 ×16)。

#### (b) Group-limited routing (DeepSeek-V3 专用, [llama-graph.cpp:1523-1545](src/llama-graph.cpp#L1523))

```c
if (hparams.n_expert_groups > 1 && n_tokens > 0) {
    const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;  // 256/8 = 32
    // 重排成 [n_exp_per_group, n_expert_groups, n_tokens]
    ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, ...);
    // 每组取 top-2 作为组得分
    ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2);
    group_scores = ggml_get_rows(...);  // 取出得分
    group_scores = ggml_sum_rows(...);  // 组内求和 -> [n_expert_groups, n_tokens]
    // 选 top n_group_used 个组 (如 4/8)
    ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used);
    // 把未选中组置 -INF,只在选中组里继续选专家
    selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY),
                                    selection_probs, expert_groups);
}
```

作用: **先选组,再在选中组里选专家**,避免某 token 的专家全集中在一个区域,促进负载均衡。这是 DeepSeek 的 auxiliary-loss-free 负载均衡策略的推理侧实现。

#### (c) top-k 选专家 ([llama-graph.cpp:1547](src/llama-graph.cpp#L1547))

```c
ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
// [n_expert_used, n_tokens]  int32, 每个位置是专家索引
```

`ggml_argsort_top_k` ([ggml/include/ggml.h:2369](ggml/include/ggml.h#L2369)) 返回 top-k 的 **索引** (而非值),这正是 `ggml_mul_mat_id` 需要的 `ids` 张量。

#### (d) 提取被选专家的权重 ([llama-graph.cpp:1560-1584](src/llama-graph.cpp#L1560))

```c
ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]

if (gating_op == SOFTMAX_WEIGHT) {            // 仅在选中专家上 softmax
    weights = ggml_reshape_2d(...); weights = ggml_soft_max(ctx0, weights); weights = ggml_reshape_3d(...);
}
if (norm_w) {                                  // 归一化为和=1 (DBRX/Grok)
    ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights);
    weights = ggml_div(ctx0, weights, weights_sum);
}
if (w_scale != 0.0f && w_scale != 1.0f) {      // sigmoid gating 的缩放 (DeepSeek-V3: ×16)
    weights = ggml_scale(ctx0, weights, w_scale);
}
```

至此 `selected_experts` (谁被选) 和 `weights` (各被选专家的权重) 都准备好了。

### 4.2 第②步: 执行被选专家 FFN (核心 `ggml_mul_mat_id`)

#### (a) up/gate 投影 ([llama-graph.cpp:1606-1668](src/llama-graph.cpp#L1606))

DeepSeek 用合并的 `gate_up_exps` 一次算出 gate+up (省一次 mul_mat_id):

```c
if (gate_up_exps) {
    ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts);
    // [n_ff*2, n_expert_used, n_tokens]
    // 拆成 gate 和 up 两个 view
    gate = ggml_view_3d(ctx0, gate_up, n_ff, ...);   // [n_ff, n_expert_used, n_tokens]
    up   = ggml_view_3d(ctx0, gate_up, n_ff, ...);
}
```

`build_lora_mm_id` ([llama-graph.cpp:1098](src/llama-graph.cpp#L1098)) 包装 `ggml_mul_mat_id`,可叠加 LoRA。**这是 MoE 稀疏计算的核心**:

```c
// ggml/include/ggml.h:1434
ggml_tensor * ggml_mul_mat_id(ctx, as, b, ids);
//   as  : [cols, rows, n_expert]   全部专家权重堆叠
//   b   : [cols, n_expert_used, n_tokens]   输入(按 token 广播到 n_expert_used)
//   ids : [n_expert_used, n_tokens] (int32)   每个 token 选了哪些专家
//   out : [rows, n_expert_used, n_tokens]   每个被选专家对该 token 的输出
//
// 语义: out[:, e, t] = as[:, :, ids[e,t]] @ b[:, e, t]
```

关键点:
- `as` 是**全部** n_expert 个专家权重,但只有被 `ids` 点名的专家会被实际计算 - 稀疏执行。
- 输出第 1 维是 `n_expert_used` (k),不是 n_expert - 每个 token 得到 k 个专家结果。

#### (b) 激活与 down 投影 ([llama-graph.cpp:1671-1745](src/llama-graph.cpp#L1671))

```c
// SwiGLU: cur = silu(gate) * up
cur = ggml_swiglu_split(ctx0, gate, up);   // [n_ff, n_expert_used, n_tokens]
// down 投影,仍是 mul_mat_id
experts = build_lora_mm_id(down_exps, cur, selected_experts);
// [n_embd, n_expert_used, n_tokens]
if (down_exps_b) experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);  // 按专家加偏置
```

`ggml_add_id` ([ggml/include/ggml.h:905](ggml/include/ggml.h#L905)) 是 `mul_mat_id` 的"加法版":按 `ids` 选对应专家的偏置加上去。

此时 `experts[ :, e, t]` = 第 t 个 token 的第 e 个被选专家的完整 FFN 输出。

### 4.3 第③步: 加权聚合得到最终结果

#### (a) 乘权重 ([llama-graph.cpp:1757-1760](src/llama-graph.cpp#L1757))

```c
if (!weight_before_ffn) {   // 标准路径:FFN 后乘权重
    experts = ggml_mul(ctx0, experts, weights);   // [n_embd, n_expert_used, n_tokens]
    // weights 形状 [1, n_expert_used, n_tokens] 自动广播到 n_embd
}
```

Llama4 走 `weight_before_ffn` 路径 ([llama-graph.cpp:1467](src/llama-graph.cpp#L1467)):先把输入 repeat 到 `[n_embd, n_expert_used, n_tokens]` 乘 sigmoid 权重,再做 FFN。

#### (b) 求和聚合 ([llama-graph.cpp:1764-1785](src/llama-graph.cpp#L1764))

```c
ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
// 把 [n_embd, n_expert_used, n_tokens] 沿第 1 维切成 k 个 [n_embd, n_tokens] 视图 (零拷贝)
for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
    cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
    ggml_build_forward_expand(gf, cur_experts[i]);
}
// 逐个累加: out = expert_0 + expert_1 + ... + expert_{k-1}
ggml_tensor * moe_out = cur_experts[0];
for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
}
```

因为前面已乘上各自权重,这里直接 `ggml_add` 累加即得加权平均。结果 `[n_embd, n_tokens]` 回到主流程,再与 shared expert 相加 ([deepseek2.cpp:412](src/models/deepseek2.cpp#L412))。

## 5. 完整数据流图 (DeepSeek-V3 一层)

```
输入 cur [n_embd, n_tokens]
   │
   ├──── 路由 ────────────────────────────────────────────┐
   │  gate_inp @ cur -> logits [n_expert, n_tokens]        │
   │  + exp_probs_b (路由偏置)                              │
   │  sigmoid -> probs [n_expert, n_tokens]  (selection_probs 同步) │
   │                                                        │
   │  Group-limited (n_expert_groups=8):                    │
   │    reshape -> [32, 8, n_tokens]                        │
   │    argsort_top_k(,2) -> 每组 top2 得分组得分            │
   │    argsort_top_k(, n_group_used=4) -> 选 4 组          │
   │    未选中组置 -INF                                      │
   │                                                        │
   │  selected_experts = argsort_top_k(selection_probs, k=8)  [8, n_tokens] (int32) ★ids
   │  weights = get_rows(probs, selected_experts) [1, 8, n_tokens]
   │  weights = scale(weights, 16.0)  (sigmoid ×缩放)       │
   │                                                        │
   ├──── 路由专家 FFN (稀疏) ──────────────────────────────┐│
   │  gate_up = mul_mat_id(gate_up_exps, cur, ids)         ││
   │            [n_ff*2, 8, n_tokens]  只算 8 个被选专家     ││
   │  拆 gate / up  (view)                                  ││
   │  cur' = swiglu_split(gate, up) [n_ff, 8, n_tokens]    ││
   │  experts = mul_mat_id(down_exps, cur', ids)            ││
   │           [n_embd, 8, n_tokens]                        ││
   │  experts = experts * weights  (广播)  [n_embd, 8, n_tokens] │
   │  切 8 个 view,逐个 ggml_add 累加                       ││
   │  -> moe_out [n_embd, n_tokens]                         ││
   │                                                        ││
   ├──── shared expert FFN (稠密,恒激活) ─────────────────┐ ││
   │  ffn_shexp = build_ffn(cur, gate_shexp, up_shexp, down_shexp) ││
   │              [n_embd, n_tokens]                       │ ││
   │                                                        │ │
   └──► cur = moe_out + ffn_shexp  [n_embd, n_tokens] ◄────┘─┘
        cur = cur + ffn_inp  (残差)
        cur = build_cvec(cur, il)  (控制向量,可选)
        -> 下一层
```

## 6. 其他 MoE 模型的差异

`build_moe_ffn` 的参数表 ([src/llama-graph.h:893-911](src/llama-graph.h#L893)) 覆盖了所有变体,各模型只改调用参数:

| 模型 | 文件 | shared expert | group routing | gating | 权重归一 | 特点 |
|------|------|:---:|:---:|--------|:---:|------|
| **DeepSeek-V2/V3** | [deepseek2.cpp](src/models/deepseek2.cpp) | ✓ | ✓ | sigmoid | ✗ | 合并 gate_up, 路由偏置 |
| **Qwen2-MoE** | [qwen2moe.cpp](src/models/qwen2moe.cpp) | ✓ | ✗ | softmax | ✗ | shared expert 独立 gate bias |
| **Qwen3-MoE** | [qwen3moe.cpp](src/models/qwen3moe.cpp) | ✓ | ✗ | softmax | ✗ | 同上 |
| **DBRX** | [dbrx.cpp](src/models/dbrx.cpp) | ✗ | ✗ | softmax | ✓ | 无 shared, 权重归一 |
| **Grok** | [grok.cpp](src/models/grok.cpp) | ✗ | ✗ | softmax | ✓ | GELU, 可选 dense FFN |
| **BailingMoE** | [bailingmoe.cpp](src/models/bailingmoe.cpp) | ✓ | ✗ | softmax | 可选 | dense_lead 层 |
| **Llama4** | [llama4.cpp](src/models/llama4.cpp) | ✗ | ✗ | sigmoid | ✗ | weight_before_ffn (先乘权重再 FFN) |

差异只在: 是否有 shared expert (调用方在 `build_moe_ffn` 后 `ggml_add` 一个 `build_ffn`)、是否开 group routing、gating 类型、权重是否归一、gate/up 是否合并。**核心三步流水完全复用 `build_moe_ffn`**。

## 7. 性能与设计要点

1. **稀疏执行是关键**: `ggml_mul_mat_id` 只计算被 `ids` 点名的专家。256 选 8 意味着 ~97% 专家权重每次推理都不动 - 但这些权重仍占显存 (需量化/卸载)。

2. **合并 gate+up**: DeepSeek 把 gate_proj 和 up_proj 合并成一个 `[n_embd, n_ff*2, n_expert]` 张量,一次 `mul_mat_id` 算两路,省一次稀疏 GEMM 的 launch 与重排开销。

3. **shared expert 稠密算**: 它对每个 token 都算,所以用普通 `build_ffn` (即两次普通 `mul_mat`),不走 mul_mat_id。与路由专家相加,提供共享计算基底。

4. **group-limited routing 只影响选谁**: 它改的是 `selection_probs`(置 -INF 屏蔽未选组),不改变后续 `mul_mat_id` 流程 - 仍是选 k 个专家稀疏执行。

5. **加权聚合用 view + add**: 切 k 个零拷贝视图再 `ggml_add` 累加,避免引入新的融合算子;warmup 时用 `hparams.n_expert_used` 而非局部 `n_expert_used` 以免生成过多 add 节点 ([llama-graph.cpp:1776-1778](src/llama-graph.cpp#L1776), 见 [PR #14753](https://github.com/ggml-org/llama.cpp/pull/14753))。

6. **CUDA 的 expert 重排**: `mm_ids_helper` 把交错布局重排成"按专家紧凑",使 GEMM 访存连续、warp 不发散,是 MoE 在 GPU 上高效的核心 ([mmid.cu](ggml/src/ggml-cuda/mmid.cu))。

7. **与后端调度协同**: 专家权重 `[n_embd, n_ff, n_expert]` 是普通权重张量,遵循 [docs/op-scheduling-analysis.md](op-scheduling-analysis.md) 的层卸载规则 - 可整体放 GPU,或 split 到多卡。`mul_mat_id` 本身被 CUDA/Metal/Vulkan/CPU 各自实现 `supports_op` 认领。

## 8. 三步流水对应代码

| 步骤 | 操作 | 关键算子 | 代码位置 |
|------|------|----------|----------|
| ① 路由 | gate_inp @ cur → logits | `ggml_mul_mat` | [llama-graph.cpp:1479](src/llama-graph.cpp#L1479) |
| ① gating | softmax/sigmoid → probs | `ggml_soft_max` / `ggml_sigmoid` | [llama-graph.cpp:1487-1491](src/llama-graph.cpp#L1487) |
| ① group routing | 选组、屏蔽未选组 | `ggml_argsort_top_k`, `ggml_set_rows`, `ggml_fill` | [llama-graph.cpp:1523-1545](src/llama-graph.cpp#L1523) |
| ① top-k 选专家 | 得 ids | `ggml_argsort_top_k` | [llama-graph.cpp:1547](src/llama-graph.cpp#L1547) |
| ① 取权重 | 按 ids gather probs | `ggml_get_rows` | [llama-graph.cpp:1560](src/llama-graph.cpp#L1560) |
| ② 专家 up/gate | 稀疏 GEMM | `ggml_mul_mat_id` ★ | [llama-graph.cpp:1608,1632,1650](src/llama-graph.cpp#L1608) |
| ② 激活 | SwiGLU | `ggml_swiglu_split` | [llama-graph.cpp:1671](src/llama-graph.cpp#L1671) |
| ② 专家 down | 稀疏 GEMM | `ggml_mul_mat_id` | [llama-graph.cpp:1740](src/llama-graph.cpp#L1740) |
| ③ 乘权重 | 广播乘 | `ggml_mul` | [llama-graph.cpp:1757](src/llama-graph.cpp#L1757) |
| ③ 聚合 | k 个 view 累加 | `ggml_view_2d` + `ggml_add` | [llama-graph.cpp:1769-1785](src/llama-graph.cpp#L1769) |
| + shared expert | 稠密 FFN | `build_ffn` (普通 `mul_mat`) | [deepseek2.cpp:404-411](src/models/deepseek2.cpp#L404) |
| 合并 | 路由+shared 相加 | `ggml_add` | [deepseek2.cpp:412](src/models/deepseek2.cpp#L412) |

**核心洞察**: llama.cpp 用一个图构建函数 `build_moe_ffn` + 一个稀疏算子 `ggml_mul_mat_id` 统一了所有 MoE 模型。top-k 选择产生 `ids` 索引张量 → `mul_mat_id` 按 `ids` 只算被选专家 → 各专家输出乘权重后 `ggml_add` 累加。DeepSeek 的 shared expert 与 group-limited routing 是在此基础上的两个正交扩展,前者多加一个稠密 FFN,后者在选择前加一步组筛选。

---

# 第二部分:专家激活实现细节

> 承接第一部分,聚焦回答一个具体问题:**路由器选出 top-k 专家后(`ids` 张量),引擎到底如何用这个索引去"激活"(实际计算)被选中的专家,并把结果摆好供加权聚合?** 以 DeepSeek-V3 + CUDA 为主线,CPU 作对照。

## 9. 问题定位

第一部分已说明三步流水:路由选专家 → 执行专家 FFN → 加权聚合。其中"执行专家 FFN"这一步,核心是算子 `ggml_mul_mat_id`。但"激活"具体怎么发生,关键在于一个看似平凡的转换:

```
top-k 输出: ids[e, t] = expert_id      形状 [n_expert_used(k), n_tokens]
                ↓ (这是个交错布局:同一专家的 token 散落各处)
专家激活需要: 把同一专家的所有 token 聚到一起 -> 只对这些 token 算该专家的 GEMM -> 结果再散回原位
```

**核心难点**: `ids` 是按"token 为主序、每 token k 个专家"交错的;而 GEMM 要按专家为主序、token 连续才高效(否则 warp 发散、访存跳跃)。llama.cpp 的解法是:在 CUDA 上用一个专门的 **索引重排内核 `mm_ids_helper`** 把 `ids` 转成三个辅助张量,再驱动一次"按专家紧凑"的 GEMM。CPU 上则用 `ith==0` 单线程建查找表 + 多线程按专家分组循环。

## 10. 数据契约:`ids` 与 `mul_mat_id`

### 10.1 top-k 产出 `ids`

[src/llama-graph.cpp:1547](src/llama-graph.cpp#L1547):

```c
ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
// 形状 [n_expert_used, n_tokens], dtype int32
// selected_experts[e, t] = 第 t 个 token 选中的第 e 个专家的全局编号 (0..n_expert-1)
```

DeepSeek-V3: `n_expert=256`, `n_expert_used=8`。所以 `ids` 是 `[8, n_tokens]` 的 int32 张量,每个 token 一列 8 个专家编号。这就是后面所有操作的"指挥棒"。

### 10.2 `ggml_mul_mat_id` 的输入输出契约

> 定义见 [ggml/src/ggml.c:3277-3315](ggml/src/ggml.c#L3277),签名见 [ggml/include/ggml.h:1434](ggml/include/ggml.h#L1434)

```
c = ggml_mul_mat_id(ctx, as, b, ids)

  as  : [cols, rows, n_expert]        全部专家权重堆叠 (第3维索引=专家号)
  b   : [cols, n_expert_used, n_tokens]  输入 (按 token 广播到 n_expert_used)
  ids : [n_expert_used, n_tokens] (int32)  top-k 专家索引
  c   : [rows, n_expert_used, n_tokens]   每个被选专家对每个 token 的输出

  语义:  c[:, e, t] = as[:, :, ids[e,t]] @ b[:, e, t]
```

DeepSeek-V3 down 投影为例:
- `as` = `ffn_down_exps` `[n_ff_exp(2048), n_embd(7168), 256]`
- `b` = SwiGLU 后的中间态 `[2048, 8, n_tokens]`
- `ids` = `[8, n_tokens]`
- `c` = `[7168, 8, n_tokens]` — 每 token 8 个专家各一个 `[7168]` 输出,沿第 1 维(8)叠好

注意 **`c` 的第 1 维是 `n_expert_used` 而非 `n_expert`**:只算被选中的 k 个,不碰其余 248 个专家。这是稀疏性的体现。

### 10.3 为什么不能直接遍历 ids 做 GEMM

朴素做法:对每个 `(e, t)`,取 `as[:,:,ids[e,t]] @ b[:,e,t]`,写进 `c[:,e,t]`。问题是:
- 同一专家(比如 expert=42)被很多 token 选中,这些 token 在 `ids` 里**不连续**(交错)。
- 如果一个线程块处理一个 token,它要去访问 expert 42 的权重;相邻线程块可能访问 expert 7 的权重 -> **L2 cache 颠簸、warp 发散**。
- 量化 GEMM (MMQ) 需要先把输入量化成 Q8_1 block,若按 token 交错量化,cache 利用率差。

解法: **先按专家重排 token,让"同一专家的所有 token"在内存里连续,再做一次普通 GEMM**。这就是 `mm_ids_helper` 干的事。

## 11. CUDA 后端:专家激活四段流水

> 入口 `ggml_cuda_mul_mat_id` ([ggml/src/ggml-cuda/ggml-cuda.cu:2632-2788](ggml/src/ggml-cuda/ggml-cuda.cu#L2632))

按权重类型与 batch 选路径:

```c
if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
    if (ggml_is_quantized(src0->type)) {
        if (ne2 <= get_mmvq_mmid_max_batch(...)) {
            ggml_cuda_mul_mat_vec_q(...);   // 1. 量化-向量化 (decode 主路径)
            return;
        }
    } ...
}
if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
    ggml_cuda_mul_mat_q(...);              // 2. 量化-MMQ (prefill 较大 batch)
    return;
}
if (ggml_cuda_should_use_mmf(...)) {
    ggml_cuda_mul_mat_f(...);              // 3. 浮点
    return;
}
// 4. fallback: host 端按专家循环 (见 11.5)
```

`ne2` 是 `b` 的第 2 维,即 `n_tokens`(注意此处 batch 维在 ne2,因为输入 reshape 过)。decode 时 `n_tokens` 小 -> 走 1 向量化;prefill 时大 -> 走 2 MMQ。三条量化/浮点路径**共用同一个重排机制**,下面以 **2 MMQ 路径** 为代表详解(它最能体现"专家激活"全貌,且 prefill 是算力密集处)。

### 11.1 第1段:索引重排内核 `mm_ids_helper`

> [ggml/src/ggml-cuda/mmid.cu](ggml/src/ggml-cuda/mmid.cu),由 [mmq.cu:181](ggml/src/ggml-cuda/mmq.cu#L181) 调用

输入 `ids [k, n_tokens]`,输出三个辅助张量:

```c
ggml_cuda_launch_mm_ids_helper(
    ids, ids_src1, ids_dst, expert_bounds,
    n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, stream);
```

- **`ids_src1[i]`**: 输入 `b` 的 gather 索引。`b` 被重排后的第 i 行 = 原始 `b` 的第 `ids_src1[i]` 行。效果:把同一专家的 token 连续摆放。
- **`ids_dst[i]`**: 输出 `c` 的 scatter 索引。紧凑结果第 i 行要写回原始 `c` 的第 `ids_dst[i]` 行。
- **`expert_bounds[e]`**: 专家 e 在紧凑缓冲里的起始行号(类比 CSR 的 `row_ptr`)。

#### 内核实现 ([mmid.cu:28-116](ggml/src/ggml-cuda/mmid.cu#L28))

**每专家一个 block**:`blockIdx.x = expert`(0..n_experts-1),block 内一个 warp 处理。对每个 token `it`:

```c
const int expert = blockIdx.x;
// 扫描该 token 选中的 k 个专家,看本专家(expert)是否被选中、选在第几位 iex_used
for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
    const int expert_used = ids[it*si1 + iex];      // 该 token 第 iex 个专家号
    nex_prev += expert_used < expert;                 // 累计:比我编号小的专家占了多少 token
    if (expert_used == expert) iex_used = iex;        // 我被选中,记下位置
}
if (iex_used != -1) store[it_compact] = {it, iex_used};  // 存 (token号, 选中槽位)
if (warp_reduce_any(iex_used != -1)) it_compact++;        // 本专家又多分到一个 token
```

`n_expert_used_template` 特化版([mmid.cu:61-93](ggml/src/ggml-cuda/mmid.cu#L61))对 k in {2,4,6,8,16,32} 用 warp 内 `__shfl_up_sync` 做前缀和扫描,比通用版快([mmid.cu:141-163](ggml/src/ggml-cuda/mmid.cu#L141) 的 switch 分派)。DeepSeek-V3 k=8 走特化版。

收尾把 `store` 写成两个索引数组:

```c
ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;  // 输入 gather
ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;                 // 输出 scatter
expert_bounds[expert]    = nex_prev;                                    // 本专家起始
expert_bounds[gridDim.x] = nex_prev + it_compact;                       // 末尾边界
```

`mm_ids_helper_store` 用 22 bit 存 token 号 + 10 bit 存 iex_used,合在一个 uint32 里省 shared memory([mmid.cu:5-19](ggml/src/ggml-cuda/mmid.cu#L5)),并断言 `n_tokens < 2^22`。

**结果**:交错 `ids` 被转成"按专家分段连续"的紧凑布局,每段边界记录在 `expert_bounds`。

### 11.2 第2段:按重排索引量化输入

> [mmq.cu:186-207](ggml/src/ggml-cuda/mmq.cu#L186)

MMQ 要求输入是 Q8_1,所以量化时**直接按 `ids_src1` 重排**:

```c
const int64_t ne_get_rows = ne12 * n_expert_used;   // = k * n_tokens 紧凑行数
// 量化 src1,行序按 ids_src1 重排
quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type,
                       ne10, s11, s12, s13, ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
```

`quantize_mmq_q8_1_cuda` 内部按 `ids_src1[i]` 去原始 `src1` 取第 i 行并量化。**一步完成"量化 + 重排"**,省一次额外 gather。

注意 `ne11_flat = ne12 * n_expert_used`(即 `n_tokens * k`):量化后的紧凑缓冲把"专家 x token"维度铺平成一个连续的行集合,每段对应一个专家。

### 11.3 第3段:一次 MMQ GEMM 激活所有专家(关键)

> [mmq.cu:215-222](ggml/src/ggml-cuda/mmq.cu#L215)

```c
const mmq_args args = {
    src0_d, src0->type, src1_q8_1.get(),   // as=全部专家权重, b=重排量化后的输入
    ids_dst.get(), expert_bounds.get(),      // scatter 索引 + 专家边界
    dst_d,
    ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
    ne02, ne02, s02, s12, s2,                 // ne02(=n_expert) 作为 grid 的 z 维
    ...
};
ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
```

这里 `ids_dst` 和 `expert_bounds` 进了 MMQ 内核。MMQ 内核 ([mmq.cu](ggml/src/ggml-cuda/mmq.cu) 的 `mul_mat_q` 模板)的 grid 用 `ne02`(n_expert)做 z 维,**每个专家一组 block**:

- `expert_bounds[e]` 告诉 block 组 e 去紧凑缓冲的哪段行取输入(`[expert_bounds[e], expert_bounds[e+1])`);
- 该 block 组从 `src0` 第 e 层取该专家权重(`src0_d + e*nb02`,即 `as[:,:,e]`);
- 算出的结果按 `ids_dst[i]` **scatter** 写回原始 `c` 的对应位置。

**这就是"专家激活"的物理时刻**:每个专家一组 CUDA block,只对该专家分到的 token 做 GEMM,未被选中的专家根本没有 block 启动(或 `expert_bounds[e]==expert_bounds[e+1]` 时该专家段长度为 0,等价跳过)。

稀疏性体现在两处:
1. 248/256 个专家若本 batch 没被任何 token 选中 -> 该专家段为空 -> 零计算。
2. 被选中的专家也只算分到它的那几个 token,不是全部 token。

### 11.4 第4段:输出回写

`ids_dst` 让 MMQ 内核直接把结果写到正确位置 `c[:, e, t]`,无需额外 scatter pass。最终 `c [rows, n_expert_used, n_tokens]` 就位,第 1 维(8)正是 k 个专家槽,后续加权聚合直接沿它切 view。

### 11.5 fallback 路径:host 端按专家循环

> [ggml-cuda.cu:2660-2788](ggml/src/ggml-cuda/ggml-cuda.cu#L2660)

当上面三条特化路径都不命中(如不支持 CUDA Graph 的情形)时走 fallback:

```c
cudaMemcpyAsync(ids_host, ids->data, ..., DeviceToHost, stream);
cudaStreamSynchronize(stream);                       // 阻塞,打断流水线

// host 端按专家分组:tokens_per_expert[e], ids_to_sorted[], ids_from_sorted[]
for (i02 in experts) for (i12 in tokens) for (iex in k)
    if ids[iex, i12] == i02: 记录 (i12, iex) 属于专家 i02;

get_rows_cuda(src1, ids_to_sorted, src1_sorted, ...);   // GPU 上 gather 输入

for (i02 in experts) {
    if (tokens_per_expert[i02] == 0) continue;            // 跳过空专家
    // 切出该专家权重 src0_slice = as[:,:,i02] (view,零拷贝)
    // 切出该专家分到的 token 块 src1_slice (紧凑缓冲里的连续段)
    ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);  // 普通 GEMM
}

get_rows_cuda(dst_sorted, ids_from_sorted, dst, ...);    // GPU 上 scatter 回去
```

逻辑等价于前三段,但重排/gather/scatter 用 `get_rows_cuda` ([getrows.cu:231](ggml/src/ggml-cuda/getrows.cu#L231)),且需要 host 端读 `ids` 建分组(故要 sync)。主路径(mmq/mmvq/mmf)把这件事全放进 GPU 内核,免 sync,所以快。

### 11.6 向量化路径 (decode 主用)

decode 时 `n_tokens` 很小(常为 1),走 `ggml_cuda_mul_mat_vec_q` ([mmvq.cu:1124](ggml/src/ggml-cuda/mmvq.cu#L1124))。它内部同样调 `mm_ids_helper` 重排,但 GEMM 是"一 token 一线程累积 k 个专家"的向量化内核([mmvq.cu:1206-1221](ggml/src/ggml-cuda/mmvq.cu#L1206) 调 `mul_mat_vec_q_switch_type`),适合 k 小、token 少的 decode 场景。这是 DeepSeek-V3 本地推理最常走的路径。

## 12. CPU 后端对照

> [ggml/src/ggml-cpu/ggml-cpu.c:1525-1644](ggml/src/ggml-cpu/ggml-cpu.c#L1525)

CPU 没有 GPU 那种 grid/warp,改用**单线程建查找表 + 多线程按专家切 chunk** 的方式,思想一致:把交错 `ids` 转成"按专家分组"。

### 12.1 建查找表 (ith==0 单线程, [ggml-cpu.c:1613-1628](ggml/src/ggml-cpu/ggml-cpu.c#L1613))

```c
if (ith == 0) {
    memset(matrix_row_counts, 0, n_as*sizeof(int64_t));   // n_as = n_expert
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {   // 遍历 token
        for (int id = 0; id < n_ids; ++id) {              // 遍历 k 个专家槽
            const int32_t i02 = ids[id, iid1];            // 该 token 该槽的专家号
            MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = {id, iid1};  // 记 (槽,token)
            matrix_row_counts[i02] += 1;
        }
    }
}
```

`matrix_rows[e]` 是专家 e 分到的 `(slot, token)` 列表,`matrix_row_counts[e]` 是其长度。这等价于 CUDA 的 `expert_bounds` + 紧凑缓冲,只是用 CSR 式的"每专家一个变长数组"而非连续平铺。

### 12.2 按专家循环 + chunk 并行 ([ggml-cpu.c:1638-1644](ggml/src/ggml-cpu/ggml-cpu.c#L1638) 后续)

```c
for (int cur_a = 0; cur_a < n_as; ++cur_a) {       // 遍历每个专家
    const int64_t cne1 = matrix_row_counts[cur_a];
    if (cne1 == 0) continue;                        // 空专家跳过

    // 把该专家权重 src0[:,:,cur_a] 与分到的 token 块切出
    // 多线程按 16x16 chunk 并行
    ggml_compute_forward_mul_mat_id_one_chunk(
        dst, src0, src1, ids, cur_a, ir0_start, ir0_end, ir1_start, ir1_end,
        src0_cur, matrix_rows, row_size, src1_cont, wdata);
}
```

`_one_chunk` ([ggml-cpu.c:1454-1515](ggml/src/ggml-cpu/ggml-cpu.c#L1454)) 对该专家分到的每个 `(slot, token)`:

```c
struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, _i12);
const int id  = row_mapping.i1;   // 选中槽 (即 e)
const int i12 = row_mapping.i2;   // token
// 取该 token 输入 src1[:, id%ne11, i12],取专家权重 src0[:, :, cur_a]
// 调用量化点积内核 vec_dot() 累加 (复用普通 mul_mat 的内核)
vec_dot(ne00, &tmp[ir0], 0, src0_cur + ir0*nb01, 0, src1_col, 0, 1);
// 写回 dst[:, id, i12]   (第1维是 e,与 CUDA 的 c[:, e, t] 布局一致)
memcpy(&dst_col[iir0], tmp, ...);
```

**稀疏性同样体现在**: `if (cne1 == 0) continue` — 没被选中的专家直接跳过,`vec_dot` 只对被选专家分到的 token 跑。

### 12.3 CPU vs CUDA 对比

| 维度 | CUDA (mmq/mmvq) | CPU |
|------|-----------------|-----|
| 重排 | `mm_ids_helper` 内核,产出 ids_src1/ids_dst/expert_bounds | `ith==0` 建 matrix_rows 查找表 |
| 分组粒度 | 专家 = grid z 维,block 组 | 专家 = 外层 for 循环 |
| 空专家处理 | expert_bounds 段长 0,无 block | `if (cne1==0) continue` |
| GEMM 内核 | MMQ (量化) / 向量化 / MMF | `vec_dot` 复用普通 mul_mat 的点积内核 |
| 输出布局 | `c[rows, k, n_tokens]` | 同 |
| 同步开销 | 全 GPU,免 sync (fallback 除外) | 线程池 barrier |

两者**语义完全一致**:都是"按专家分组 -> 跳过空专家 -> 只算被选 token -> 结果写到 `c[:, e, t]` 第 1 维"。差异只在分组与并行的实现方式。

## 13. 输出布局如何喂给加权聚合

激活后 `c [rows, n_expert_used, n_tokens]`(代码里叫 `experts`)。回到 [src/llama-graph.cpp:1757-1785](src/llama-graph.cpp#L1757):

```c
if (!weight_before_ffn) {
    experts = ggml_mul(ctx0, experts, weights);   // weights [1, k, n_tokens] 广播乘
}
// 沿第1维(k)切 k 个零拷贝 view
for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
    cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
}
// 累加
ggml_tensor * moe_out = cur_experts[0];
for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
    moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
}
```

关键:**`c` 的第 1 维恰好是 `n_expert_used`(k)**,所以 `ggml_view_2d` 沿第 1 维切得到 k 个 `[n_embd, n_tokens]` 切片,每个切片是一个专家对所有 token 的(已加权)输出。`ggml_add` 逐个累加即得加权平均。

这个布局是 `mul_mat_id` 在 [ggml/src/ggml.c:3277](ggml/src/ggml.c#L3277) 定义的:`c` 形状 `[rows, n_expert_used, n_tokens]` — **第 1 维放 k 个专家槽,正是为了让下游用一个简单的 view+add 循环完成聚合**,无需引入新的融合算子。

## 14. 端到端数值示例 (3 token)

设 batch=3 token,n_expert=256,n_expert_used=8。top-k 选出的 `ids`(部分,示意; slot 为行、token 为列, 物理内存连续方向为纵向 slot, 见第 27.1.1 节):

```
ids [8, 3]:       token0      token1      token2
  slot0:          42          42          199
  slot1:          7           199         42
  slot2:          150         7           7
  ... (8 行)
```

**第1段 `mm_ids_helper`**(以 expert=42 这个 block 为例):扫描发现 token0(slot0)、token1(无)、token2(slot1)选中了 42。`nex_prev` = 比 42 小的专家累计占的 token 数(设为 5)。产出:

```
ids_src1[5]   = token0 的输入行索引 + slot0   // 输入 gather
ids_src1[6]   = token2 的输入行索引 + slot1
ids_dst [5]   = 0*8 + 0 = 0                   // 写回 c[:, 0, 0]
ids_dst [6]   = 2*8 + 1 = 17                  // 写回 c[:, 1, 2]
expert_bounds[42] = 5
expert_bounds[43] = 7   (= 5 + 2, 本专家分到 2 个 token)
```

**第2段量化**:按 `ids_src1` 把 token0、token2 的输入行(对应 slot)取出来量化成 Q8_1,连续摆在紧凑缓冲的第 5、6 行。

**第3段 MMQ**:expert=42 的 block 组从 `src0[:,:,42]` 取权重,对紧凑缓冲 `[5,7)` 这段(2 行)做 GEMM,结果按 `ids_dst` 写到 `c[:,0,0]` 和 `c[:,1,2]`。

**未被任何 token 选中的专家**(如 250):`expert_bounds[250]==expert_bounds[251]` -> 段长 0 -> 该专家 block 组无工作量。

**第4段聚合**:`c[:, e, t]` 8 个槽各自乘权重,view 切 8 片累加 -> `[n_embd, 3]`。

整个过程中,**只有被点名的专家被计算**,且同一专家的多个 token 连续处理 — 这就是 top-k 结果驱动专家激活的全部物理含义。

## 15. 关键设计洞察

1. **`ids` 是交错布局,GEMM 要连续布局** — 这是 MoE 高效执行的核心矛盾。llama.cpp 用 `mm_ids_helper` 把交错索引转成三个辅助张量(ids_src1/ids_dst/expert_bounds),化"稀疏按需计算"为"一次连续 GEMM + gather/scatter"。

2. **空专家天然跳过** — `expert_bounds` 段长为 0 的专家不启动 block(CPU 侧 `if (cne1==0) continue`)。256 选 8 时,平均每 batch 约有大量专家未被命中,直接零计算。

3. **量化与重排合一** — `quantize_mmq_q8_1_cuda` 按 `ids_src1` 边取边量化,省一次额外 gather,cache 友好。

4. **输出布局为聚合而设计** — `c` 第 1 维 = `n_expert_used`,让加权聚合退化成 `view_2d` + `ggml_add` 循环,无需融合算子。

5. **三条特化路径 + 一条 fallback** — decode(小 batch)走 mmvq 向量化,prefill(大 batch)走 mmq,浮点走 mmf,都不支持时 fallback 用 host 端循环 + `get_rows_cuda`。前四段共用 `mm_ids_helper`,后三段共用紧凑 GEMM 思想。

6. **k 的特化** — `mm_ids_helper` 对 k in {2,4,6,8,16,32} 用 warp shuffle 前缀和特化版,DeepSeek-V3 的 k=8 直接受益。通用 k 走循环版。

7. **与后端调度协同** — `mul_mat_id` 被 CUDA/Metal/Vulkan/CPU 各自实现 `supports_op` 认领;专家权重 `as [in,out,n_expert]` 是普通权重,遵循 [docs/op-scheduling-analysis.md](op-scheduling-analysis.md) 的层卸载规则,可整体放 GPU。注意 [ggml-cuda.cu:2646](ggml/src/ggml-cuda/ggml-cuda.cu#L2646) 断言:`mul_mat_id does not support split buffers` — MoE 专家权重不能跨多卡行切分(需整层放单卡)。

---

# 第三部分:多 token 批处理的路由与高效计算

> 场景: **Decode 阶段,16 个并发 token(连续批处理),每个 token 经路由器激活不同的 top-k 专家组合**。深入分析 llama.cpp 如何解决"token ↔ 专家"的路由问题并高效计算。以 DeepSeek-V3 + CUDA、量化权重(如 Q4_K)、k=8、n_expert=256 为例。承接第一、二部分。

## 16. 场景精确定义

连续批处理 (continuous batching) 下,16 个序列同时各解码 1 个 token,构成一个 ubatch:`n_tokens = 16`。每 token 路由到 8 个专家(共 256 个专家池),不同 token 的专家组合**各不相同**。

一次 MoE down 投影的 `ggml_mul_mat_id` 调用:

```
as  = ffn_down_exps [n_ff_exp(2048), n_embd(7168), n_expert(256)]   全部专家权重
b   = cur           [n_embd(7168), 1, n_tokens(16)]                  输入(ne11=1 广播)
ids = selected_experts [n_expert_used(8), n_tokens(16)] int32        top-k 索引
c   =               [n_embd(7168), n_expert_used(8), n_tokens(16)]    输出
```

关键维度(后续路径判断用):
- `ne2 = c 的 ne[2] = n_tokens = 16` (dst 张量第 2 维)
- `ne12 = b 的 ne[2] = n_tokens = 16`
- `ne1 = c 的 ne[1] = n_expert_used = k = 8` (即 `nchannels_dst`)
- `ne02 = as 的 ne[2] = n_expert = 256`

**路由问题**:16 token × 8 专家 = 128 个 (token, 专家) 分配,但它们**交错**躺在 `ids [8,16]` 里(每个 token 一列 8 个专家号,同专家的 token 散落各处)。不同 token 激活不同专家 → **负载不均**:热门专家可能被 5 个 token 命中,冷门专家 0 命中。朴素地"对每个 (token,expert) 做一次小 GEMM"会有 128 次碎 kernel launch、warp 发散、L2 cache 颠簸。

llama.cpp 的解法分两层:**①按 batch 大小选执行路径;②用 `mm_ids_helper` 把交错路由重排成专家分组,做一次连续 GEMM**。

## 17. 按 batch 大小选执行路径(三档)

> 分派器 `ggml_cuda_mul_mat_id` ([ggml-cuda.cu:2632](ggml/src/ggml-cuda/ggml-cuda.cu#L2632))

```c
const int cc = ...;
if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
    static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);   // 都是 8
    if (ne2 <= MMVQ_MAX_BATCH_SIZE) {           // n_tokens <= 8 ?
        if (ggml_is_quantized(src0->type)) {
            if (ne2 <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                ggml_cuda_mul_mat_vec_q(...);   // 档位 B
                return;
            }
        } ...
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
        ggml_cuda_mul_mat_q(...);                // 档位 C  ← 16 token 走这里
        return;
    }
    if (ggml_cuda_should_use_mmf(...)) { ggml_cuda_mul_mat_f(...); return; }
}
// 档位 D: fallback (host 端按专家循环)
```

| 档位 | n_tokens 范围 | 条件 | 内核 | 文件 |
|------|---------------|------|------|------|
| **A** | = 1 | decode 单序列 | `mul_mat_vec_q<ncols_dst=1>` 向量化 | [mmvq.cu:476](ggml/src/ggml-cuda/mmvq.cu#L476) |
| **B** | 2..8 | `ne2 <= MMVQ_MAX_BATCH_SIZE(8)` 且量化 | **专用 MoE 多 token 内核** `mul_mat_vec_q_moe` | [mmvq.cu:684](ggml/src/ggml-cuda/mmvq.cu#L684) |
| **C** | > 8 (本场景 16) | `should_use_mmq` 返回 true | **`mm_ids_helper` 重排 + 紧凑 MMQ GEMM** | [mmq.cu:164-223](ggml/src/ggml-cuda/mmq.cu#L164) |
| D | 任意 | 上面都不命中 | host 端按专家循环 + `get_rows_cuda` | [ggml-cuda.cu:2660](ggml/src/ggml-cuda/ggml-cuda.cu#L2660) |

**对本场景 (n_tokens=16)**:`ne2=16 ≤ 8` 为假 → 跳过档位 B → `should_use_mmq` 判定。`should_use_mmq` ([mmq.cu:267](ggml/src/ggml-cuda/mmq.cu#L267)) 对量化类型且 `ne11(=16) < MMQ_DP4A_MAX_BATCH_SIZE(64)` 返回 true(现代 GPU 有 tensor core 时 `!mma || 16<64` = true;Pascal 无 mma 时 `true || ...` = true)→ **走档位 C**。

> 注意档位 B 的存在很关键:`MMVQ_MAX_BATCH_SIZE = 8` 是个**架构相关上限** ([mmvq.cuh:3](ggml/src/ggml-cuda/mmvq.cuh#L3)),因为向量化内核把所有 token 塞进一个 block 的 `threadIdx.y` 维(block 形状 `(warp_size, ncols_dst)`,ncols_dst 即 n_tokens),超过 8 会撑爆寄存器/launch bounds。所以 16 token 必须改走"按专家分组"的 MMQ 路径。**这正是 16 token 场景与日常单 token decode 的本质区别**。

## 18. `mm_ids_helper` 解决路由问题(档位 C 详解)

### 18.1 问题本质:交错 vs 连续

`ids [8, 16]` 的布局(每列一个 token,8 行是 8 个专家槽)。下图按 **slot 为行、token 为列**绘制;此朝向下物理内存连续方向是图中**纵向**(ne[0]=slot, 同 token 的 8 个 slot 逐元素相邻), 而**横向**是跨 token 步进维(ne[1])。详见第 27.1.1 节。

```
         <---- ne[1] = token (步进维, 跨 token 步进 k=8 个 int32) ---->
         t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15
slot0:   42   7 199  42   7 150  42  199  42   7  42  199   7  42 199   7   |
slot1:    7 199  42   7 199  42  199  42   7 199 199   7  42 199  42  42   | ne[0] = slot
... (8 行)                                                                 | (连续维, 内存中
                                                                          |  同 token 的 8 个
                                                                          |  slot 逐元素相邻)
                                                                            v
  内存连续方向: 沿一列向下 (固定 token, 变 slot)  ->  而非沿一行向右 (那会跨 token)
```

若直接遍历:`for e in 8: for t in 16: as[:,:,ids[e,t]] @ b[:,t]` → 128 次碎点积,相邻线程访问不同专家权重 → cache 颠簸。

### 18.2 重排内核 `mm_ids_helper`

> [mmid.cu:28-116](ggml/src/ggml-cuda/mmid.cu#L28),由 [mmq.cu:181](ggml/src/ggml-cuda/mmq.cu#L181) 调用

**每个专家一个 block**(`blockIdx.x = expert`,0..255),扫描 16 个 token × 8 槽,看本专家被哪些 (token, 槽) 命中,产出三个辅助张量:

- **`expert_bounds[e]`**:专家 e 在紧凑缓冲里的起始行(CSR 的 `row_ptr`)。`expert_bounds[256]` 存总行数。
- **`ids_src1[i]`**:输入 `b` 的 gather 索引 — 紧凑缓冲第 i 行 = 原始 `b` 第 `ids_src1[i]` 行。
- **`ids_dst[i]`**:输出 `c` 的 scatter 索引 — 紧凑结果第 i 行写回 `c` 第 `ids_dst[i]` 行。

k=8 走特化版([mmid.cu:61-93](ggml/src/ggml-cuda/mmid.cu#L61)),用 warp `__shfl_up_sync` 前缀和算 `expert_bounds`,比通用循环快。

**核心效果**:128 个 (token,专家) 分配被重排成"按专家分段连续"的紧凑布局。例如 expert=42 命中了 t0/t3/t6/t8/t10/t13(6 个),它就占紧凑缓冲的连续 6 行,边界记在 `expert_bounds[42]` 和 `expert_bounds[43]`。

### 18.3 一步完成"量化 + 重排"

> [mmq.cu:200-205](ggml/src/ggml-cuda/mmq.cu#L200)

```c
const int64_t ne_get_rows = ne12 * n_expert_used;   // = 16 * 8 = 128 紧凑行数
quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), ...);
```

`quantize_mmq_q8_1_cuda` 内部按 `ids_src1[i]` 取原始输入第 i 行并量化成 Q8_1。**gather 与量化合一**,省一次额外搬运。量化后紧凑缓冲 `ne11_flat = 128`(把"专家×token"铺平)。

### 18.4 一次 MMQ GEMM,每专家一组 block(激活时刻)

> [mmq.cu:215-222](ggml/src/ggml-cuda/mmq.cu#L215)

```c
const mmq_args args = {
    src0_d, src0->type, src1_q8_1.get(),   // as=256 专家权重, b=重排量化输入
    ids_dst.get(), expert_bounds.get(),      // scatter 索引 + 专家边界
    dst_d,
    ne00, ne01, ne_get_rows(128), s01, ne_get_rows, s1,
    ne02(256), ne02(256), s02, s12, s2,      // ★ ne02 作为 grid 的 z 维 = 256 个专家
    ...
};
ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
```

MMQ 内核 grid 用 `ne02 = 256` 做 z 维,**每个专家一组 block**:
- `expert_bounds[e]` 告诉 block 组 e 去紧凑缓冲 `[expert_bounds[e], expert_bounds[e+1])` 段取输入(只有该专家分到的 token);
- 该 block 组从 `src0` 第 e 层取权重 `src0_d + e*nb02`(即 `as[:,:,e]`);
- 结果按 `ids_dst[i]` scatter 写回 `c[:, e, t]`。

**这就是路由问题的解**:不是"对每个 (token,expert) 算一次",而是"对每个专家算一次(只算分到它的那些 token)"。128 个分配被 256 个专家 block 并行消化,每个 block 的工作量 = 该专家分到的 token 数。

## 19. 负载不均如何被天然吸收

16 token × k=8 = 128 个分配,散在 256 个专家上。实际命中分布(示意,假设):

```
expert  42: 命中 t0,t3,t6,t8,t10,t13       → 6 token  → block 组做 6 行 GEMM
expert   7: 命中 t1,t4,t9,t12,t15           → 5 token  → 5 行
expert 199: 命中 t2,t7,t11,t14              → 4 token  → 4 行
expert 150: 命中 t5                          → 1 token  → 1 行
expert   0,1,2,...(约 244 个): 0 命中       → 0 token  → expert_bounds 段长 0 → 无 block 工作
```

**关键性质**:段长 = 该专家分到的 token 数,**无需 padding**。
- 热门专家(6 token):做 6 行 GEMM,block 多一点工作量。
- 冷门专家(0 token):`expert_bounds[e] == expert_bounds[e+1]` → 该专家 block 组的循环范围为空 → 零计算、零访存。
- 不需要把所有专家都对齐到最大 token 数(那会浪费算力)。

这与档位 B 的专用 MoE 内核形成对比:档位 B 的 `mul_mat_vec_q_moe` ([mmvq.cu:684](ggml/src/ggml-cuda/mmvq.cu#L684)) 用 `blockIdx.y = nchannels_dst = k = 8` 个 block,每 block 处理"一个专家槽对所有 token",block 内 `threadIdx.y = token_idx` 各算各——它把负载均衡粒度放在"槽×token"上,适合 token 少(≤8)时每 warp 独立、无 shared memory 归约。但 token 一多(16)就超出 `MMVQ_MAX_BATCH_SIZE=8` 的 launch bounds 上限,必须改走档位 C 的"专家分组"思路。

**所以 16 token 选 MMQ 不是偶然**:它正是用"专家维度做并行、token 维度做紧凑分段"来扛住负载不均,而档位 B 的"token 维度做并行"在 token 多时会撑爆寄存器。

## 20. 端到端数值示例 (16 token)

**输入**:`ids [8,16]`(如第 18.1 节上表, slot 为行、token 为列; 物理内存连续方向为纵向 slot, 见第 27.1.1 节),`b [7168, 1, 16]`,`as [2048,7168,256]`(down 投影)。

**Step 1 — 路径判断**:分派器 `ne2=16 ≤ 8`?否 → `should_use_mmq(Q4_K, cc, ne12=16, 256)` = true → 档位 C。

**Step 2 — `mm_ids_helper`**:256 个 block 并行扫描。以 expert=42 为例,它命中 (t0,slot0)、(t3,slot0)、(t6,slot0)、(t8,slot0)、(t10,slot0)、(t13,slot0)(假设都在 slot0)。设比 42 小的专家累计占 30 行,则:

```
expert_bounds[42] = 30
expert_bounds[43] = 36        (= 30 + 6, 42 分到 6 个 token)
ids_src1[30..35] = t0,t3,t6,t8,t10,t13 的输入行索引 + slot0
ids_dst [30..35] = 0*8+0, 3*8+0, 6*8+0, 8*8+0, 10*8+0, 13*8+0
                 = 0, 24, 48, 64, 80, 104    (写回 c[:, slot0, t] 的偏移)
```

未被命中的 expert=250:`expert_bounds[250]==expert_bounds[251]` → 段长 0 → 无工作。

**Step 3 — 量化+重排**:按 `ids_src1[30..35]` 把 t0/t3/t6/t8/t10/t13 的输入行(各 2048 元素)量化成 Q8_1,连续摆在紧凑缓冲第 30..35 行。

**Step 4 — MMQ GEMM**:expert=42 的 block 组从 `src0[:,:,42]`(形状 `[2048,7168]`)取权重,对紧凑缓冲 `[30,36)` 段(6 行 × 7168)做 GEMM,输出 `[7168, 6]`,按 `ids_dst` scatter 写到 `c[:, 0, t0/t3/t6/t8/t10/t13]`。

**Step 5 — 加权聚合**([llama-graph.cpp:1757-1785](src/llama-graph.cpp#L1757)):`c [7168, 8, 16]` 各槽乘权重 → `ggml_view_2d` 切 8 片 → `ggml_add` 累加 → `[7168, 16]`。

整个过程中:
- **只有被命中的 ~12 个专家被计算**(244 个空专家零计算)。
- 每个专家只算分到它的 token(42 算 6 个,150 算 1 个),不 padding。
- 同一专家的 token 连续处理 → GEMM 访存连续、warp 不发散、cache 命中高。

## 21. 三档的工程权衡

| 维度 | 档位 A (n=1) | 档位 B (n=2..8) | 档位 C (n>8,本场景) |
|------|-------------|-----------------|---------------------|
| 并行主轴 | 行×专家 | 槽(专家)×token,warp 各管一 token | **专家(grid z),token 紧凑分段** |
| 路由处理 | ids 直接索引 | ids 直接索引(`channel_x = ids[...]`) | **`mm_ids_helper` 重排成 CSR** |
| 负载不均 | 无(单 token) | 每 warp 独立,自然吸收 | **段长=token 数,空专家跳过** |
| 量化输入 | 原位量化 | 原位量化 | **量化+重排合一** |
| 输出回写 | 直接写 | 直接写 | **ids_dst scatter** |
| 上限 | 无 | `MMVQ_MAX_BATCH_SIZE=8`(launch bounds) | 无(适合大 batch,也用于 prefill) |
| 同步 | 全 GPU | 全 GPU | 全 GPU(免 sync) |

**核心洞察**:
1. **路由问题的本质是"交错 ids → 连续 GEMM"的转换**。小 batch(≤8)token 少,可以用"每 warp 一 token"的向量化内核直接按 ids 索引,无需重排;大 batch(>8)token 多,必须先用 `mm_ids_helper` 把路由重排成专家分组,否则寄存器/launch bounds 撑不住、cache 也颠簸。
2. **`MMVQ_MAX_BATCH_SIZE=8` 是分水岭**。16 token 正好落在"大 batch"侧,触发 MMQ 路径。这不是硬编码的魔数,而是向量化内核 block 形状 `(warp_size, ncols_dst)` 的 launch bounds 上限([mmvq.cu:683](ggml/src/ggml-cuda/mmvq.cu#L683) `__launch_bounds__(get_mmvq_mmid_max_batch_for_device<type>()*warp_size, 1)`)。
3. **负载不均靠"段长=token 数"吸收**,无需 padding。这是 CSR 式分段的天然优势——空专家零成本,热专家多干活,各 block 工作量天然与命中数成正比。
4. **gather/scatter 一步合一**:输入量化按 `ids_src1` 边取边量化,输出按 `ids_dst` 边算边写,避免单独的重排 pass。
5. **专家维度做并行**(grid z = 256),天然适合 GPU 的大量 SM——即使只有 ~12 个专家有活干,256 个 block 组也足够铺满现代 GPU(空 block 组几乎零开销)。

---

# 附录:关键文件速查

| 关注点 | 文件 |
|--------|------|
| MoE 图构建 (核心) | [src/llama-graph.cpp:1441-1795](src/llama-graph.cpp#L1441) |
| MoE 超参 | [src/llama-hparams.h:53-101](src/llama-hparams.h#L53) |
| gating 枚举 | [src/llama-hparams.h:12-17](src/llama-hparams.h#L12) |
| DeepSeek2 调用 | [src/models/deepseek2.cpp:385-414](src/models/deepseek2.cpp#L385) |
| 其他 MoE 模型 | [src/models/qwen2moe.cpp](src/models/qwen2moe.cpp), [qwen3moe.cpp](src/models/qwen3moe.cpp), [dbrx.cpp](src/models/dbrx.cpp), [grok.cpp](src/models/grok.cpp), [bailingmoe.cpp](src/models/bailingmoe.cpp) |
| top-k 产出 ids | [src/llama-graph.cpp:1547](src/llama-graph.cpp#L1547) |
| mul_mat_id 算子定义 | [ggml/include/ggml.h:1434](ggml/include/ggml.h#L1434), [ggml/src/ggml.c:3277](ggml/src/ggml.c#L3277) |
| argsort_top_k | [ggml/include/ggml.h:2369](ggml/include/ggml.h#L2369) |
| CUDA mul_mat_id 入口 | [ggml/src/ggml-cuda/ggml-cuda.cu:2632](ggml/src/ggml-cuda/ggml-cuda.cu#L2632) |
| 索引重排内核 `mm_ids_helper` | [ggml/src/ggml-cuda/mmid.cu:28](ggml/src/ggml-cuda/mmid.cu#L28) |
| 重排启动器 | [ggml/src/ggml-cuda/mmid.cu:138](ggml/src/ggml-cuda/mmid.cu#L138) `ggml_cuda_launch_mm_ids_helper` |
| k 特化分派 | [ggml/src/ggml-cuda/mmid.cu:141-163](ggml/src/ggml-cuda/mmid.cu#L141) |
| 量化+重排 | [ggml/src/ggml-cuda/mmq.cu:200-205](ggml/src/ggml-cuda/mmq.cu#L200) `quantize_mmq_q8_1_cuda` |
| MMQ GEMM (激活) | [ggml/src/ggml-cuda/mmq.cu:215-222](ggml/src/ggml-cuda/mmq.cu#L215) `ggml_cuda_mul_mat_q_switch_type` |
| `MMVQ_MAX_BATCH_SIZE=8` 阈值 | [ggml/src/ggml-cuda/mmvq.cuh:3](ggml/src/ggml-cuda/mmvq.cuh#L3) |
| 档位 B 专用 MoE 内核 (n≤8) | [ggml/src/ggml-cuda/mmvq.cu:684](ggml/src/ggml-cuda/mmvq.cu#L684) `mul_mat_vec_q_moe` |
| `should_use_mmq` 判定 | [ggml/src/ggml-cuda/mmq.cu:267](ggml/src/ggml-cuda/mmq.cu#L267) |
| 向量化路径 (decode, n=1) | [ggml/src/ggml-cuda/mmvq.cu:1124](ggml/src/ggml-cuda/mmvq.cu#L1124) |
| 浮点路径 | [ggml/src/ggml-cuda/mmf.cu:13](ggml/src/ggml-cuda/mmf.cu#L13) |
| fallback (host 循环) | [ggml/src/ggml-cuda/ggml-cuda.cu:2660-2788](ggml/src/ggml-cuda/ggml-cuda.cu#L2660) |
| gather/scatter 原语 | [ggml/src/ggml-cuda/getrows.cu:231](ggml/src/ggml-cuda/getrows.cu#L231) `get_rows_cuda` |
| CPU mul_mat_id 入口 | [ggml/src/ggml-cpu/ggml-cpu.c:1525](ggml/src/ggml-cpu/ggml-cpu.c#L1525) |
| CPU 查找表 (ith==0) | [ggml/src/ggml-cpu/ggml-cpu.c:1613](ggml/src/ggml-cpu/ggml-cpu.c#L1613) |
| CPU 专家循环 | [ggml/src/ggml-cpu/ggml-cpu.c:1638](ggml/src/ggml-cpu/ggml-cpu.c#L1638) |
| CPU 单 chunk 点积 | [ggml/src/ggml-cpu/ggml-cpu.c:1454](ggml/src/ggml-cpu/ggml-cpu.c#L1454) `vec_dot` |
| 加权聚合 (view+add) | [src/llama-graph.cpp:1757-1785](src/llama-graph.cpp#L1757) |
| 后端调度协同 | [docs/op-scheduling-analysis.md](docs/op-scheduling-analysis.md) |
| `mul_mat_id` 不支持 split buffer 断言 | [ggml/src/ggml-cuda/ggml-cuda.cu:2646](ggml/src/ggml-cuda/ggml-cuda.cu#L2646) |

---

# 第四部分:其他硬件后端是否需要按专家重排 token

> 承接前三部分。第二、三部分以 CUDA 为主线,说明 MoE 专家激活的核心矛盾是 **`ids` 交错布局(token 主序)与 GEMM 期望的连续布局(专家主序)的不匹配**,CUDA 用 `mm_ids_helper` 把交错 `ids` 重排成"按专家分段连续"再驱动一次 GEMM。本部分回答:**其他硬件后端(Metal / Vulkan / SYCL / CANN / OpenCL / Hexagon / ZenDNN / WebGPU / CPU)是否也需要做同样的"按专家重排"?如果不需要,它们各自采用什么方案支持 MoE?**
> 分析对象: llama.cpp 仓库 (master 分支),整理日期: 2026-06-23

## 22. 结论先行

**绝大多数后端都需要(并实现了)按专家重排 token**,只是重排的**发生位置**与**实现机制**各不相同;仅有 **Vulkan 与 CANN** 走了本质不同的路线(一个内联过滤、一个根本不重排)。具体:

| 后端 | 是否按专家重排 token | 重排发生位置 / 机制 | GEMM 组织方式 |
|------|:---:|------|------|
| **CUDA** | ✓(大 batch) | GPU 内核 `mm_ids_helper` → ids_src1/ids_dst/expert_bounds | grid z = n_expert,每专家一组 block |
| **CPU** | ✓ | `ith==0` 建 `matrix_rows` 查找表(每专家一个变长数组) | 外层 for 遍历专家,内层 chunk 并行 `vec_dot` |
| **Metal** | ✓(大 batch) | GPU 内核 `kernel_mul_mm_id_map0` → `tpe` + 重排 `ids` | grid z = ne02(专家),每专家一组 threadgroup |
| **SYCL** | ✓(多 token) | **host 端**计数排序 `mmid_counting_sort_rows` → expert_row_offsets + routed_row_src,再 copy 内核搬运 | host 端 for 遍历专家,每专家一次 `ggml_sycl_mul_mat` |
| **OpenCL** | ✓ | 四步 GPU 内核 histogram→scan→fill→scatter(`moe_router` 流水) | 重排后喂 moe gemv/gemm 内核 |
| **ZenDNN** | ✓ | `ith==0` 建 `matrix_rows`(同 CPU) | 每专家 gather + ZenDNN 批量 GEMM + scatter |
| **WebGPU** | ✓ | 两个 WGSL dispatch:`mul_mat_id_gather`(每专家一 workgroup,原子 gather) | `mul_mat_id` 每 expert 分块 GEMM |
| **Vulkan** | △(概念上分组,但不显式重排) | **不预处理**;GEMM 内核内**内联过滤** ids,shared memory 建 `row_ids` 映射回原坐标 | grid z = n_as(专家),每专家一 workgroup,边过滤边算 |
| **CANN** | ✗(完全不重排 token) | 不重排 token,改在**权重侧**用 `aclnnIndexSelect` 抽出 k 个被选专家权重 | 厂商批量算子:`aclnnBatchMatMul`(FP)/ `aclnnWeightQuantBatchMatmulV2`(量化) |
| **Hexagon** | ?(闭源) | 委托给高通 HTP DSP 运行时,`HTP_OP_MUL_MAT_ID` | 黑盒,源码不可见 |

> ✓ = 显式按专家重排 token;✗ = 不重排;△ = 不显式重排但内核内按专家分组;? = 闭源不可见。

**一句话总结**: **重排是"自研 GEMM 内核"后端的通用范式**(CUDA/CPU/Metal/SYCL/OpenCL/ZenDNN/WebGPU 都是先把同专家 token 聚到一起再做连续 GEMM,差别只在重排放 GPU 内核 / host CPU / 查找表);**不重排只发生在"依赖厂商批量矩阵乘库"的后端**(CANN 用 BatchMatMul 把 k 个专家当成 batch 维一次算,Vulkan 用内联过滤省掉重排 pass)。

## 23. 需要重排的后端:机制差异详解

这七个后端**语义一致**(都是"按专家分组 → 跳过空专家 → 只算被选 token → 写回 `c[:, e, t]`"),但重排的**落点**分三类。

### 23.1 GPU 内核重排(重排结果物化到显存)

**CUDA** 已详述(第 11/18 节):`mm_ids_helper` 每 expert 一个 block,产 `ids_src1`(输入 gather)/`ids_dst`(输出 scatter)/`expert_bounds`(CSR 边界),量化时边 gather 边量化,M MQ 内核 grid z = n_expert。

**Metal** 与 CUDA 思路几乎对称,但用**两个串行内核**实现([ggml-metal-ops.cpp:2282](ggml/src/ggml-metal/ggml-metal-ops.cpp#L2282) 入口):

1. 重排内核 `kernel_mul_mm_id_map0`([ggml-metal.metal:9761](ggml/src/ggml-metal/ggml-metal.metal#L9761)):dispatch `(1,1,1)` threadgroup + `ne02` 个线程(**每线程 = 一个专家**)。线程 `ide` 扫描所有 token 的 k 个槽,凡 `ids[t][i]==ide` 就把 `(token, slot)` 编进自己的段,产出 `tpe[ide]`(本专家 token 数,等价 CUDA 的 expert_bounds 段长)和重排后的 `ids` 数组。dispatch 见 [ggml-metal-ops.cpp:2367](ggml/src/ggml-metal/ggml-metal-ops.cpp#L2367)。
2. GEMM 内核 `kernel_mul_mm_id`([ggml-metal.metal:9827](ggml/src/ggml-metal/ggml-metal.metal#L9827)):dispatch `(ceil(ne21/32), ceil(ne01/64), ne02)`([ggml-metal-ops.cpp:2407](ggml/src/ggml-metal/ggml-metal-ops.cpp#L2407)),**第 3 维 = ne02 = 专家数**,即每专家一组 threadgroup。`tgpig.z = im` 即专家号,通过 `ids_i32[im*ne21 + r1 + lr1]` 取本专家分到的 (token,slot) 再解码回原坐标 `i11/i12` 访问 src1/dst([ggml-metal.metal:9878-9895](ggml/src/ggml-metal/ggml-metal.metal#L9878))。空专家(`tpe==0`)直接 return 跳过。

阈值 `ne21_mm_id_min = 32`([ggml-metal-ops.cpp:2320](ggml/src/ggml-metal/ggml-metal-ops.cpp#L2320)):token 数 ≥ 32 且有 simdgroup matrix 才走"重排 + 矩阵乘";否则走 `kernel_mul_mv_id`([ggml-metal.metal:10367](ggml/src/ggml-metal/ggml-metal.metal#L10367))的向量化路径(每 (token,expert) 一份,不重排)——对应 CUDA 的档位 A/B。

**OpenCL** 用**四步流水**把重排做成了完整的 `moe_router` 管线([ggml-opencl.cpp:14746](ggml/src/ggml-opencl/ggml-opencl.cpp#L14746)),四个内核 `kernel_moe_histogram / scan / fill / scatter`([ggml-opencl.cpp:662](ggml/src/ggml-opencl/ggml-opencl.cpp#L662)):

1. **histogram**:统计每专家被多少 token 命中(全局尺寸 `(ceil(ne21/64)*64, ne20, 1)`);
2. **scan**:前缀和算每专家的 tile 偏移 `tile_offset`(单线程);
3. **fill**:按 tile 结构预填 `post_router` 缓冲;
4. **scatter**:把交错 router ids 散射进 `post_router`(专家主序)并建 `emap`。

产出 `prealloc_post_router`(重排后 ids)、`prealloc_emap`、`prealloc_act_trans`(重排后激活)([ggml-opencl.cpp:439,452,453](ggml/src/ggml-opencl/ggml-opencl.cpp#L439))。随后按量化类型分派 `gemv_moe_*` / `gemm_moe_*` 内核消费这份专家主序数据。这是所有后端里**最显式的重排流水**。

**WebGPU** 也是两段 dispatch([ggml-webgpu.cpp:1626](ggml/src/ggml-webgpu/ggml-webgpu.cpp#L1626)):先 `mul_mat_id_gather.wgsl`([ggml-webgpu.cpp:1649](ggml/src/ggml-webgpu/ggml-webgpu.cpp#L1649),每专家一 workgroup,线程遍历所有 token,命中即原子计数并写 `(expert_used_idx, token_idx)`),把 token 物化成专家主序;再 `mul_mat_id.wgsl`([ggml-webgpu.cpp:1650](ggml/src/ggml-webgpu/ggml-webgpu.cpp#L1650))做每专家分块 GEMM。小 batch 走 `mul_mat_id_vec`([ggml-webgpu.cpp:1568](ggml/src/ggml-webgpu/ggml-webgpu.cpp#L1568))。

### 23.2 host 端重排(在 CPU 上规划,再用 copy 内核搬到显存)

**SYCL** 是唯一把重排**放在 host CPU**上做的 GPU 后端([ggml-sycl.cpp:4251](ggml/src/ggml-sycl/ggml-sycl.cpp#L4251))。多 token 路径(`ne12>1`,[ggml-sycl.cpp:4321](ggml/src/ggml-sycl/ggml-sycl.cpp#L4321)):

1. **阻塞**地把 `ids` 拷回 host(`stream->memcpy` + `wait()`,[ggml-sycl.cpp:4275-4277](ggml/src/ggml-sycl/ggml-sycl.cpp#L4275))——这是 SYCL 特有的 sync 点(CUDA 主路径免 sync);
2. host 端计数排序 `mmid_counting_sort_rows`([ggml-sycl.cpp:4212](ggml/src/ggml-sycl/ggml-sycl.cpp#L4212)):算 `expert_row_offsets`(每专家在紧凑缓冲的起始,等价 expert_bounds)和 `routed_row_src`(每个紧凑行的来源 `(slot,token)`,等价 ids_src1/ids_dst);
3. `k_copy_src1_to_contiguous`([ggml-sycl.cpp:4123](ggml/src/ggml-sycl/ggml-sycl.cpp#L4123))把输入按映射 gather 成专家主序紧凑缓冲;
4. **host 端 for 遍历专家**([ggml-sycl.cpp:4367](ggml/src/ggml-sycl/ggml-sycl.cpp#L4367)),每专家切出自己那段紧凑输入 + `src0[:,:,i02]` 权重,调一次普通 `ggml_sycl_mul_mat`(注意:**不是** grid z = n_expert 的单内核,而是 N 次普通 GEMM 调用);
5. `k_copy_dst_from_contiguous`([ggml-sycl.cpp:4146](ggml/src/ggml-sycl/ggml-sycl.cpp#L4146))把结果 scatter 回原布局。

单 token(`ne12==1`)走 `ggml_sycl_mul_mat_id_mmvq_fused`([ggml-sycl.cpp:4166](ggml/src/ggml-sycl/ggml-sycl.cpp#L4166))融合向量化路径,免重排。同样断言 `mul_mat_id does not support split buffers`([ggml-sycl.cpp:4256](ggml/src/ggml-sycl/ggml-sycl.cpp#L4256))。

SYCL 与 CUDA 的关键差异:**重排在 host**(需 sync 读 ids)+ **专家维度用 host 循环**而非 GPU grid z。这是 dpct 从 CUDA 移植时的工程取舍——牺牲一点 host 开销换取实现简单,MoE 每 batch 命中专家有限故 host 循环开销可接受。

### 23.3 查找表 + 即时 gather(CPU 系)

**CPU** 已详述(第 12 节):`ith==0` 建 `matrix_rows[e]`(每专家的 `(slot,token)` 列表)+ `matrix_row_counts[e]`,外层 for 遍历专家,`if (cne1==0) continue` 跳空,内层 chunk 并行 `vec_dot`。

**ZenDNN** 与 CPU **逐字相同**的查找表结构([ggml-zendnn.cpp:226](ggml/src/ggml-zendnn/ggml-zendnn.cpp#L226) 入口):`ith==0` 建 `matrix_rows` / `matrix_row_counts`([ggml-zendnn.cpp:262-274](ggml/src/ggml-zendnn/ggml-zendnn.cpp#L262)),每专家 gather 输入行进连续缓冲,调 ZenDNN 批量 GEMM,再 scatter 回去。`supports_op` 限定 ≤32 专家([ggml-zendnn.cpp:604](ggml/src/ggml-zendnn/ggml-zendnn.cpp#L604))。差别只是 GEMM 内核换成 ZenDNN 库而非 `vec_dot`。

## 24. 不做显式重排的后端:替代方案

### 24.1 Vulkan:GEMM 内核内"内联过滤",不物化重排缓冲

Vulkan 是所有自研 GEMM 后端里**唯一不预处理重排**的([ggml-vulkan.cpp:9586](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9586) 入口 → [ggml_vk_mul_mat_id_q_f16](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9053) → [ggml_vk_matmul_id](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7934))。它的 GEMM 内核 `mul_mmq.comp` 编译时带 `#define MUL_MAT_ID`,dispatch 形状是 `{ m, nei1, n_as }`([ggml-vulkan.cpp:7947](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7947)),**第 3 维 `n_as` = 专家数**——所以仍是"每专家一组 workgroup"。

区别在于:每个专家 workgroup **不读预重排缓冲**,而是在内核启动时**当场扫描整张交错 `ids`**,把命中本专家的 `(ii0, ii1)`(槽,token)记进 shared memory 的 `row_ids[BN]`(`mul_mm_id_funcs.glsl` 的 `load_row_ids`,用 `subgroupBallot` 紧凑化命中位),GEMM 累加时再按 `row_ids` 映射回原始坐标取 src1、写 dst。另有一个 `count_experts.comp`([ggml-vulkan.cpp:5105](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5105),[9187](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9187))预跑一遍统计每专家命中数,**仅用于跳过空 workgroup**,不产生重排数据。

**本质**:Vulkan 把 CUDA 的"重排 pass + GEMM pass"**融合进了单个 GEMM 内核**(每 expert workgroup 自己做一遍 O(k·n_tokens) 的 ids 扫描)。语义上仍是专家主序计算(每专家只算自己的 token),但省掉了重排缓冲的分配与一次显式搬运;代价是每个 workgroup 都要重扫一遍整张 ids(总扫描量 = n_expert × k × n_tokens,大于 CUDA 的单次扫描)。这是 Vulkan shader 模型(无动态分配、shared memory 有限)下的合理取舍。

### 24.2 CANN:完全不重排 token,改在"权重侧"选专家 + 厂商批量 GEMM

CANN 是**唯一彻底不重排 token** 的后端,方案最特殊([ggml-cann.cpp:1918](ggml/src/ggml-cann/ggml-cann.cpp#L1918) 入口 → [aclnn_ops.cpp:3841](ggml/src/ggml-cann/aclnn_ops.cpp#L3841) 分派)。核心思路:**不动 token,动权重**——用昇腾 ACL 的 `aclnnIndexSelect` 把 k 个被选专家的权重从 `[in, out, n_expert]` 里抽出来,再用厂商批量矩阵乘一次算完。

- **浮点路径** `ggml_cann_mul_mat_id_fp`([aclnn_ops.cpp:3571](ggml/src/ggml-cann/aclnn_ops.cpp#L3571)):对每个 batch,先 `aclnnIndexSelect`(沿专家维 0)抽出 k 个专家权重得 `[D, M, K]`(K=k),转置后与输入 `[D, 1, K]` 做一次 `aclnnBatchMatMul`([aclnn_ops.cpp:3617](ggml/src/ggml-cann/aclnn_ops.cpp#L3617))——**把 k 个专家当成 batch 维,一次批量乘算完所有 k 个专家**,根本不需要把同专家 token 聚拢。
- **量化路径** `ggml_cann_mul_mat_id_quant`([aclnn_ops.cpp:3648](ggml/src/ggml-cann/aclnn_ops.cpp#L3648)):同样先 `IndexSelect` 抽出 k 份权重和 scale,但量化算子 `aclnnWeightQuantBatchMatmulV2` 不支持 batch,故退化为**每专家一次**调用([aclnn_ops.cpp:3816](ggml/src/ggml-cann/aclnn_ops.cpp#L3816))——仍是"每专家一次 GEMM",但 token 不重排(广播或按 slot 取)。

**为什么 CANN 能不重排**:它依赖昇腾硬件的 `BatchMatMul` 把"专家"维度直接当 batch 维原生处理——权重张量 `[in, out, n_expert]` 本身就是 batch=K 的批量矩阵,选 k 个专家 = 选 batch 的 k 个切片,一次批量乘即可。token 是否在内存里按专家连续**对这种厂商库批量乘无关紧要**,因为 batch 维的各切片本就独立索引、无需连续。这正是"自研 GEMM 内核要连续访存"与"厂商库 batched-matmul 原生支持 batch 维"的根本差异。

### 24.3 Hexagon:委托闭源 HTP 运行时

Hexagon 把 `GGML_OP_MUL_MAT_ID` 映射为 `HTP_OP_MUL_MAT_ID`([ggml-hexagon.cpp:3134](ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3134)),实际计算交给高通 HTP DSP 专有运行时,源码不可见。`supports_op` 仅校验类型(Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4、repacked 权重、F32 输入输出, [ggml-hexagon.cpp:2639](ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2639))。重排与否由 HTP 内部决定,对外是黑盒。

## 25. 为什么有差异:重排的本质与触发条件

把所有后端摆在一起,能提炼出一条清晰规律:

**重排 token 是"自研 GEMM 内核"后端的刚需,而非 MoE 的固有要求。**

1. **自研 GEMM 内核(MMQ/vec_dot/simdgroup-matrix)追求连续访存与无 warp 发散**:这类内核按行/分块取权重与输入,若同专家 token 散落各处,相邻线程访问不同专家权重 → cache 颠簸、warp 发散(第二部分第 10.3 节)。故 CUDA/CPU/Metal/SYCL/OpenCL/ZenDNN/WebGPU **全部**先做"按专家分组"再 GEMM。差别仅在重排落点:
   - GPU 内核物化(CUDA `mm_ids_helper`、Metal `map0`、OpenCL 四步流水、WebGPU gather)→ 适合大 batch,一次重排多次复用;
   - host CPU 规划 + copy 内核搬运(SYCL 计数排序)→ 实现简单但需 sync;
   - 查找表 + 即时 gather(CPU、ZenDNN `matrix_rows`)→ 无额外显存,适合 CPU 内存模型。

2. **厂商批量矩阵乘库原生支持 batch 维时,无需重排**:CANN 的 `aclnnBatchMatMul` 把 k 个专家当 batch 切片一次算完,token 连续性无关紧要 → 完全不重排。Vulkan 介于两者之间:仍是自研内核(要分组),但把"分组"融合进内核内联过滤、不物化重排缓冲。

3. **重排与否不影响语义,只影响性能与实现**:所有后端的 `mul_mat_id` 输出契约都是 `c[rows, n_expert_used, n_tokens]`(第 1 维 = k),下游加权聚合(view+add)对布局无感。所以"是否重排"是各后端根据自身算子库能力做的工程选择,MoE 的数学语义在三步流水层面完全一致。

## 26. 权重搬运总量的跨后端对比(带宽维度)

> 承接第 25 节"重排本质"。本节从**权重读取带宽**这一常被忽略的维度,定量对比"重排 / 不重排"方案在专家权重搬运量上的差异。先澄清一个常见误解: **"重排 token"本身完全不搬运专家权重** - 重排只动 `ids` 和输入 `src1`,专家权重 `as [in,out,n_expert]` 始终原地不动。权重搬运只发生在 GEMM 执行时。真正决定权重搬运量的不是"是否重排",而是两个更底层的属性: **(1) GEMM 是否按专家分组(能否合并"同一专家被多 token 选中") (2) 是否多一次权重的物化拷贝**。

### 26.1 记号与三类方案

设: `W` = 单个专家权重的字节数, `n_active` = 本 batch 被命中的活跃专家数, `n_tokens * k` = 总分配数(如 16*8=128)。

| 方案 | 代表后端 | 权重读取量(算法下界,忽略 cache) | 合并同专家多 token? | 额外物化拷贝? |
|------|----------|------|:---:|:---:|
| **重排 + 按专家 GEMM** | CUDA / CPU / Metal / OpenCL / ZenDNN / WebGPU / SYCL | `n_active * W` | 是 | 否 |
| **不重排,内核内联分组** | Vulkan | `n_active * W` | 是 | 否 |
| **不重排,逐 token + 厂商批量乘** | CANN | `~ 2 * n_tokens * k * W` | 否 | 是(IndexSelect) |

### 26.2 重排方案: `n_active * W`(达到下界)

重排后每个活跃专家 e 分到一段连续 token, **一次 GEMM** 处理它分到的所有 token。GEMM 内核从 `as + e*nb02` 读该专家权重一次, 在 shared memory / cache 里 tile 复用给该专家的全部 token。

- 同一专家被 6 个 token 选中 -> 权重**只读 1 次**, 6 个 token 共享。
- 重排本身只 gather 输入 `src1`, **不碰权重**, 零额外权重搬运。
- 权重读取 = 活跃专家数 * 单专家权重, 这是 MoE 稀疏性的理论下界。

### 26.3 Vulkan 内联过滤: `n_active * W`(与重排相同)

Vulkan 不显式重排, 但 GEMM 内核 dispatch 的第 3 维 `n_as = n_expert`([ggml-vulkan.cpp:7947](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7947)), **仍是每专家一组 workgroup**。非空 workgroup 加载该专家权重 tile, 对内联过滤出的 token 复用。

- 权重读取量 = `n_active * W`, **与重排方案完全相同**。
- 代价不在权重侧, 而在: (1) 每 workgroup 冗余扫描整张 `ids`(总扫描量 `n_expert * k * n_tokens`, 大于重排的单次 `k * n_tokens`); (2) 输入 `src1` 按 `row_ids` 随机访问而非连续 gather。这两项都不涉及权重搬运。

> **结论**: Vulkan 的"不重排"在权重搬运上不比重排多花 - 它把重排融进了 GEMM, 权重侧仍是"每活跃专家读 1 次"。

### 26.4 CANN: `~ 2 * n_tokens * k * W`(明显更高)

CANN 是唯一权重搬运量质变的后端。FP 路径逐 token 循环([aclnn_ops.cpp:3586](ggml/src/ggml-cann/aclnn_ops.cpp#L3586)):

```
for i in n_tokens:                       # 逐 token, 不合并
    IndexSelect(as, ids[:,i]) -> select_export [in,out,k]   # 读 k 个专家权重, 写出
    BatchMatMul(active, select_export) -> dst               # 再读 k 个专家权重
```

两个放大因素叠加:

- **不合并同专家多 token**: 处理是 token 粒度。若专家 42 被 6 个 token 选中, 它的权重被 **6 个 token 各自的 IndexSelect 重复读 6 次**(重排方案只读 1 次)。
- **IndexSelect 多一次物化拷贝**: 每个 (token, slot) 的权重被读 2 次 - IndexSelect 从大张量读出并写入 `select_export`, BatchMatMul 再从 `select_export` 读一次。

量化路径([aclnn_ops.cpp:3648](ggml/src/ggml-cann/aclnn_ops.cpp#L3648))同样逐 token + IndexSelect + 逐 expert `WeightQuantBatchMatmulV2`, 权重读取特性一致。

### 26.5 量化对比(16 token, k=8, 假设约 12 活跃专家)

- 重排 / Vulkan: `12 * W`
- CANN: `2 * 16 * 8 * W = 256 * W`, 约为重排方案的 **21 倍**(算法下界, 实际被 NPU L1/L2 cache 缓解)

即便负载完全均衡(128 个分配落 128 个不同专家, `n_active=128`):
- 重排 / Vulkan: `128 * W`
- CANN: `256 * W`, 仍是 **2 倍**(IndexSelect 那次物化拷贝无法省)。

### 26.6 两点二次影响

1. **读取连续性**: 重排方案和 Vulkan 的 GEMM 读 `as[:,:,e]` 是**连续大块读**, 带宽利用率高; CANN 的 `IndexSelect` 是按 `ids` **跳跃 gather** 读, 有效带宽通常低于连续读(除非硬件有 gather 优化)。所以 CANN 即便"读同样字节", 实际耗时也更长。
2. **cache 兜底**: CANN 的重复读取在 NPU cache 充足、`n_tokens` 小(decode)时可能被 cache 命中吸收, 差异不明显; 但 **prefill 大 batch(n_tokens=4096)时, 同专家权重的重复读取会突破 cache, 权重搬运成为瓶颈**。这正是重排方案为 prefill 设计的根本原因(见第三部分第 21 节)。

### 26.7 小结

- **重排 vs Vulkan(不重排但内联分组): 权重搬运量相同**, 都是 `n_active * W`。Vulkan 省掉的是重排缓冲与输入 gather, 不是权重读取。
- **重排 vs CANN(不重排且不分组): 权重搬运量差异巨大**。CANN 因"逐 token 不合并 + IndexSelect 物化"导致权重被重复读, 算法下界是重排的 2 至 20+ 倍。CANN 用带宽换的是"无需自研重排内核 + 直接调用昇腾高效 BatchMatMul 的算力利用率" - 这是**带宽效率 vs 算力效率**的权衡, 在小 batch + cache 充足时划算, 大 batch 时权重带宽吃亏。
- **根源**: 权重搬运量取决于"GEMM 能否按专家合并多 token", 而非"是否重排 token"。重排只是实现"按专家合并"的手段之一; Vulkan 用内核内联达到同样合并; CANN 放弃合并改用厂商 batched-matmul。

## 27. `mm_ids_helper` 重排细节详解(16 token x 8 专家实例)

> 承接第二、三部分。本节用 16 个 token、每 token 激活 8 个专家(`n_expert=256`, `k=8`)的具体数值, 把 CUDA `mm_ids_helper` 的重排细节逐步拆透, 并可视化输入(ids)与输出(ids_src1 / ids_dst / expert_bounds)的布局, 以及它们如何驱动后续 GEMM。内核源码: [mmid.cu:28-116](ggml/src/ggml-cuda/mmid.cu#L28), 启动器: [mmid.cu:138-164](ggml/src/ggml-cuda/mmid.cu#L138), 调用点: [mmq.cu:181](ggml/src/ggml-cuda/mmq.cu#L181)。

### 27.0 前置: `mul_mat_id` 的张量契约与 `b`/`c` 形状

`mm_ids_helper` 处理的是 `mul_mat_id` 的三个输入之一 `ids`, 但它产出的 `ids_src1`/`ids_dst` 是为了**重组输入 `b` 和输出 `c` 的行序**。所以先要把 `b`/`c` 的形状和含义讲清楚, 否则后面 27.3 的两个公式无法理解。契约定义见 [ggml.c:3277-3315](ggml/src/ggml.c#L3277):

```
as  : [cols, rows, n_expert]            全部专家权重堆叠(第3维=专家号)
b   : [cols, n_expert_used, n_tokens]   输入(n_expert_used 可广播为 1)
ids : [n_expert_used, n_tokens] (i32)   top-k 索引
c   : [rows, n_expert_used, n_tokens]   输出

语义:  c[:, e, t] = as[:, :, ids[e,t]] @ b[:, e % r, t]    (r = b->ne[1], 广播因子)
输出形状: c->ne = { as->ne[1], ids->ne[0], b->ne[2] } = {rows, k, n_tokens}   [ggml.c:3302]
```

每维含义:

| 张量 | 维度 | 物理含义 |
|------|------|----------|
| `as` | `ne[0]=cols` | GEMM 的 K 维(权重输入侧, 必须与 `b` 的 `ne[0]` 相等, [ggml.c:3303](ggml/src/ggml.c#L3303)) |
| `as` | `ne[1]=rows` | GEMM 的 M 维(权重输出侧 = `c` 的 `ne[0]`) |
| `as` | `ne[2]=n_expert` | 专家号(256), 选哪片权重靠 `ids` 间接寻址 |
| `b` | `ne[0]=cols` | 输入向量的长度(K 维, 与 `as` 的 cols 相等) |
| `b` | `ne[1]=n_expert_used`(或 1) | **每 token 的输入槽**。=k 时每槽一个独立输入; =1 时一个输入广播给 k 个专家 |
| `b` | `ne[2]=n_tokens` | token 维(16) |
| `c` | `ne[0]=rows` | 输出向量长度(M 维) |
| `c` | `ne[1]=k` | **每 token 的 k 个专家输出槽** - 第 1 维放 k, 正是供下游加权聚合 view+add 切片(第 13 节) |
| `c` | `ne[2]=n_tokens` | token 维(16) |

#### 27.0.1 为什么 `b` 在 down 投影是 `[cols, 8, 16]` 而不是 `[cols, 1, 16]`

关键: **`b` 的 `ne[1]`(输入槽)取决于"同一 token 的 k 个专家是否共用同一输入"**。MoE 的两次 `mul_mat_id`(up/gate 投影 与 down 投影)在此不同:

**up/gate 投影(第一次 mul_mat_id)** - [llama-graph.cpp:1608](src/llama-graph.cpp#L1608):

```c
gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts);
//                                          ^^^ b = cur
```

`cur` 是 MoE 层的输入隐状态, 形状 `[n_embd, n_tokens] = [7168, 16]`(3 维化即 `ne[1]=1`, 因为它本来就只有每 token 一个隐向量)。这里 **k 个被选专家吃的输入是同一个 `cur[:, t]`** - token t 无论路由到哪 8 个专家, 喂进去的都是它自己那一个 `[7168]` 隐向量。所以 `b = cur` 的 `ne[1]=1`, **广播**给 k=8 个专家。契约里 `ids->ne[0] % b->ne[1] == 0` 即 `8 % 1 == 0`([ggml.c:3304](ggml/src/ggml.c#L3304))允许这种广播。这种情况下 `b` 是 `[cols=7168, 1, 16]`, 确实只占 16 行, 省空间。

**down 投影(第二次 mul_mat_id)** - [llama-graph.cpp:1740](src/llama-graph.cpp#L1740):

```c
experts = build_lora_mm_id(down_exps, cur, selected_experts);
//                                     ^^^ b = swiglu 后的 cur
```

这里的 `cur` 是上一段 `gate_up` 经过 SwiGLU 激活(`ggml_swiglu_split`, [llama-graph.cpp:1697](src/llama-graph.cpp#L1697))后的结果, 形状 `[n_ff, k=8, n_tokens=16]`。**为什么这时 `ne[1]` 变成了 8 而不是 1?** 因为 SwiGLU 的输出**每个专家槽各不相同**:

- up 投影时, `gate_up = mul_mat_id(gate_up_exps, cur, ids)` 输出 `[n_ff*2, 8, 16]`([llama-graph.cpp:1608](src/llama-graph.cpp#L1608)) - 注意输出第 1 维是 k=8。即 **token t 的 8 个被选专家各算出了一个独立的 `[n_ff*2]` 向量**(因为 8 个专家权重 `as[:,:,ids[0..7,t]]` 各不相同)。
- 拆成 gate/up 两路 `[n_ff, 8, 16]`, SwiGLU 逐元素 `silu(gate)*up` 后仍是 `[n_ff, 8, 16]`([ggml.c:3036](ggml/src/ggml.c#L3036) `ggml_glu_impl` 输出 `ne = a->ne` 即保持 `[n_ff, 8, 16]`)。
- 所以 down 投影的输入 `b` 第 1 维是 **k=8 个互不相同的中间向量** - token t 的第 0 个专家要用它 up 投影的 slot0 中间态, 第 1 个专家要用 slot1 中间态, ... 第 7 个专家用 slot7 中间态。**这不是冗余复制, 而是 8 个专家各自的真实中间结果, 不能合并成 1。**

所以 down 投影 `b = [n_ff, 8, 16]` 是**必需的**, 不是浪费:

- 若强行用 `[n_ff, 1, 16]`(广播), 语义会变成"token t 的 8 个专家都吃同一个中间态" - 但它们 up 投影时算出的中间态本来就不同, 广播会把 8 个专家的不同输入强行捏成一个, **算错**。
- 8 个专家的中间态在内存里是 token t 的连续 8 行(第 27.1.1 节: `ne[0]=n_ff` 连续、`ne[1]=8` 是 k 槽、`ne[2]=16` 是 token), 这正是 `ids_src1 = it*8 + slot` 要 gather 的对象。

> **一句话**: up 投影 `b=[cols,1,16]`(广播, k 个专家共用 token 输入, 省空间)是因为输入本来就一个; down 投影 `b=[cols,8,16]`(不广播, 每专家一个中间态)是因为前一步 up 投影已经为每个专家产出了不同的中间结果。`ne[1]` 是否为 1 取决于"同 token 的 k 个专家输入是否相同", 由数据流决定, 不是能随意省的 - 第 27.3 节的 `ids_src1` 公式 `it*sis1 + iex%nchannels_y` 里 `sis1 = b->ne[1]`(=1 或 =k)正是为此自适应: up 投影 `sis1=1` 只取 token 号, down 投影 `sis1=8` 取 token*8+slot。

### 27.1 输入: 交错的 `ids` 张量

`ids` 由 [llama-graph.cpp:1547](src/llama-graph.cpp#L1547) 的 `ggml_argsort_top_k` 产出, 形状 `[k=8, n_tokens=16]`, dtype int32。在文档里称作"token 主序、每 token 一列 8 个专家号",但这个词指的是**逻辑组织语义**(以 token 为单位、每 token 带 k 个专家号),**不是物理内存连续**。物理布局需要单独澄清, 见 27.1.1。

下图按 **slot 为行、token 为列**绘制(与第二、三部分保持一致)。注意此朝向下, 物理内存的连续方向是图中**沿一列从上到下**(固定 token、变 slot), 而非"沿一行从左到右"(那是跨 token 步进)。轴标注见下方:

```
         <--------------- ne[1] = token (步进维, 跨 token 步进 k=8 个 int32) --------------->
         t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15
slot0:    0   0   0   0   1   0   0   2   0   1   0   0   4   0   2   1   |
slot1:    1   1   1   2   3   2   1   4   1   3   2   1   5   1   4   3   |
slot2:    2   2   4   4   5   4   3   6   2   5   4   2   6   2   6   5   | ne[0] = slot
slot3:    3   3   5   6   7   6   5   8   3   7   6   3   7   3   8   7   | (连续维, 内存中
slot4:    4   8   8   8   9   8   7  10   4   9   8   8   8   4  10   9   |  同 token 的 8 个
slot5:    5   9   9  10  11  10   9  12   5  11  10   9   9   5  12  11   |  slot 逐元素相邻)
slot6:    6  10  12  12  13  12  11  14   6  13  12  10  10   6  14  13   |
slot7:    7  11  13  14  15  14  15  15   7  15  14  11  11   7  15  15   |
                                                                            v
  内存连续方向: 沿一列向下 (固定 token, 变 slot)  ->  而非沿一行向右 (那会跨 token)
```

> **读图提示**: 因本图把 slot 画成行、token 画成列, 其"常规行优先读法"(沿一行从左到右)对应的是**跨 token**(ne[1] 步进维), 恰好**不是**内存连续方向; 真正连续的是图中**纵向**(ne[0]=slot)。这与下文 27.1.1 给出的物理内存顺序一致。

#### 27.1.1 物理内存布局澄清:`ne[0]=k` 连续,token 是步进维

`ids` 的维度与步进定义(一路追到源码):

- 路由输入 `selection_probs` 形状 `[n_expert, n_tokens] = [256, 16]`([llama-graph.cpp:1487](src/llama-graph.cpp#L1487)),即 `ne[0]=n_expert`, `ne[1]=n_tokens`。
- `ggml_argsort_top_k` 输出 `ne[0]=k=8`, `ne[1]=n_tokens=16`([ggml.c:5281](ggml/src/ggml.c#L5281) `ggml_view_4d(..., k, result->ne[1], ...)`)。
- CUDA 入口读 `n_expert_used = ids->ne[0]`([mmq.cu:168](ggml/src/ggml-cuda/mmq.cu#L168)),并断言 `ids->nb[0] == ggml_element_size(ids)`([mmq.cu:177](ggml/src/ggml-cuda/mmq.cu#L177)),`si1 = ids->nb[1] / ggml_element_size(ids)`([mmq.cu:178](ggml/src/ggml-cuda/mmq.cu#L178))。

关键在断言 `ids->nb[0] == 4`(int32 元素大小)。ggml 的布局约定借鉴 NumPy C-order(row-major): **`ne[0]` 是最内层、物理逐元素连续的维度**, `nb[0] = element_size`; 越外层维度步进越大。断言强制 `ne[0]`(专家槽 k)这一维在内存里逐元素连续, 而 token(`ne[1]`)是步进维, 步进 `si1 = nb[1]/4 = k = 8` 个 int32。

因此 `ids` 的**物理内存**是:

```
ne[0] = k = 8 (专家槽, 最内层, 连续, 单位步进 1 个 int32)
ne[1] = n_tokens = 16 (token, 外层, 步进 si1 = k = 8 个 int32)

内存顺序:
[t0_s0, t0_s1, t0_s2, ..., t0_s7,  t1_s0, t1_s1, ..., t1_s7,  ..., t15_s0, ..., t15_s7]
 ^------ 同一个 token 的 8 个专家槽: 连续 8 个 int32 ------^
```

内核访问 `ids[it*si1 + iex]`([mmid.cu:46](ggml/src/ggml-cuda/mmid.cu#L46))正好印证: `iex` 是单位步进的连续维, `it*si1` 是跨 token 的步进维(`si1=k=8`)。

> **结论(纠偏)**: 不是 token 维度在内存里连续, 而是**专家槽维度(ne[0]=k)连续、token 维度(ne[1])步进**。文档里的"token 主序"指逻辑组织语义("以 token 为单位、每 token 带 k 个专家号"), 在 C-order 下表现为"一行 = 一个 token 的 k 个专家号": **行内(同 token 的 k 个槽)连续, 行间(跨 token)步进**。同理 `b = [cols, k, n_tokens]` 的 `cols`(ne[0])最内层连续、`token`(ne[2])最外层步进, 铺平成 `n_tokens*k` 行时行号 `it*k + iex` 里 `iex` 连续、`it` 步进 - 所以 `ids_src1 = it*k + iex` 取的也是"同 token 的 k 行连续"。这就是 `mm_ids_helper` 要解决的"交错"的物理含义: 同专家的命中**跨 token 步进散落**(每隔 k 个元素才出现一次), 而非逐元素连续。

### 27.2 内核执行模型: 每专家一个 block, 一个 warp 扫描

```c
const dim3 num_blocks(n_experts, 1, 1);   // = 256 个 block, 一个专家一个 block   [mmid.cu:130]
const dim3 block_size(warp_size, 1, 1);   // 每 block 一个 warp(32 线程)         [mmid.cu:131]
mm_ids_helper<8><<<num_blocks, block_size, ...>>>            // k=8 走特化模板    [mmid.cu:151-152]
```

每个 block 内: `expert = blockIdx.x`(本 block 负责的专家号), 一个 warp 协作扫描所有 16 个 token 的 8 个 slot。k=8 走特化实现([mmid.cu:61-93](ggml/src/ggml-cuda/mmid.cu#L61)): warp 的 32 线程被分成 `warp_size/neu_padded = 32/8 = 4` 组, 每组 8 线程同时处理一个 token 的 8 个 slot(`threadIdx.x % 8 = iex`, `threadIdx.x / 8 = token-in-warp`), 用 `__shfl_up_sync` 在组间做前缀和扫描, 比 k=0 通用版([mmid.cu:41-60](ggml/src/ggml-cuda/mmid.cu#L41))的逐 token 循环快。

### 27.3 每个 block 计算的两个量

对专家 `expert`, warp 扫描所有 `(token, slot)` 时维护两个计数器:

1. **`nex_prev`**([mmid.cu:38](ggml/src/ggml-cuda/mmid.cu#L38)): **比本专家编号小的专家**累计占多少个紧凑行。这是本专家在紧凑缓冲里的**起始偏移**(CSR 的 `row_ptr`)。算法上对每个 `(token,slot)` 累加 `nex_prev += (expert_used < expert)`, 最后 `warp_reduce_sum` 得全 block 一致的值。
2. **`it_compact`**([mmid.cu:39](ggml/src/ggml-cuda/mmid.cu#L39)): 本专家**分到多少个 token**(命中计数)。每发现一个命中(`expert_used == expert`)就 `it_compact++`(用 `warp_reduce_any` 跨 slot 聚合)。

命中的 `(token, slot)` 暂存进 shared memory 的 `store[]`([mmid.cu:54,88](ggml/src/ggml-cuda/mmid.cu#L54)), 用 22 bit 存 token 号 + 10 bit 存 slot([mmid.cu:5-19](ggml/src/ggml-cuda/mmid.cu#L5), 故断言 `n_tokens < 2^22`, `k < 2^10`, [mmid.cu:122-123](ggml/src/ggml-cuda/mmid.cu#L122))。

扫描完后写出([mmid.cu:97-103](ggml/src/ggml-cuda/mmid.cu#L97)):

```c
for (itc in [0, it_compact)):   // 本专家分到的每个命中
    it       = store[itc].it();        // token 号
    iex_used = store[itc].iex_used();  // 命中的 slot 号
    ids_src1[nex_prev + itc] = it*sis1          + iex_used % nchannels_y;  // 输入 gather 索引
    ids_dst [nex_prev + itc] = it*n_expert_used + iex_used;                 // 输出 scatter 索引
expert_bounds[expert] = nex_prev;                  // 本专家起始(仅 thread 0 写)   [mmid.cu:109]
// 最后一个 block 还写: expert_bounds[gridDim.x] = nex_prev + it_compact;  // 总长度  [mmid.cu:115]
```

**两个公式的含义**(以 down 投影为例, `b = [cols, k, n_tokens]`, 故 `nchannels_y = ne11 = k = 8`, `sis1 = nb12/nb11 = k = 8`):

- `ids_src1 = it*sis1 + iex_used%nchannels_y = it*8 + slot`: 这是**输入 `b` 在铺平后的行号**。`b` 形状 `[cols, 8, 16]` 铺平成 128 行, 第 `(it, slot)` 个向量在行 `it*8 + slot`。`iex_used % nchannels_y` 对 down 投影(`nchannels_y=8=k`)就是 `slot` 本身; 对 up 投影(`b=[cols,1,16]` 广播, `nchannels_y=1`)则恒为 0, 即只取 token 号(同一输入向量被该 token 的 k 个专家复用)。
- `ids_dst = it*k + iex_used`: 这是**输出 `c` 在铺平后的行号**。`c = [rows, k, n_tokens]` 铺平成 128 行, 第 `(slot, it)` 个输出向量在行 `it*k + slot`。GEMM 算出的紧凑结果第 `nex_prev+itc` 行, 按 `ids_dst` scatter 写回 `c` 的对应位置。

### 27.4 输出之一: `expert_bounds`(CSR row_ptr, 长度 n_experts+1)

对上面那张 `ids` 表, `mm_ids_helper` 算出每专家命中数, 累加得 `expert_bounds`(本例 16 个专家非零, 240 个为零):

```
expert  :   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16..255
count   :  10  10  10   9  10   9   9   8   9   8   8   7   6   4   5   6     0
bounds  :   0  10  20  30  39  49  58  67  75  84  92 100 107 113 117 122  ... 128
            ↑ start of expert e                              expert_bounds[256]=128 (total)
```

语义: 紧凑缓冲的行区间 `[expert_bounds[e], expert_bounds[e+1])` 归专家 e。空专家(如 16..255)该区间长度为 0 -> GEMM 里该专家 block 组**零工作量**。这是负载不均被天然吸收的物理实现(第 19 节): 热门专家(expert 0 占 10 行)多干活, 冷门专家(0 行)零成本, **无需 padding**。

### 27.5 输出之二、三: `ids_src1` / `ids_dst`(各 128 个 int32)

紧凑缓冲共 128 行(= 16 token x 8 专家)。`ids_src1[i]` = 第 i 行该从原始输入 `b` 的哪行 gather; `ids_dst[i]` = 第 i 行算完该 scatter 到原始输出 `c` 的哪行。两者都按**专家主序**排好(专家 0 的 10 行连续在前, 接着专家 1 的 10 行, ...)。以 expert=0 的 10 个命中为例:

```
expert 0 (bounds 0..10, n=10):
  compact_row  token  slot   ids_src1 = token*8+slot   ids_dst = token*8+slot
     r0          0      0         0                       0
     r1          1      0         8                       8
     r2          2      0        16                      16
     r3          3      0        24                      24
     r4          5      0        40                      40
     r5          6      0        48                      48
     r6          8      0        64                      64
     r7         10      0        80                      80
     r8         11      0        88                      88
     r9         13      0       104                     104
```

(down 投影 `nchannels_y=k=8`, 故 `ids_src1 == ids_dst`。) 整张 `ids_src1`(128 项, 按专家分段):

```
[0,8,16,24,40,48,64,80,88,104, | 1,9,17,32,49,65,72,89,105,120, | 2,10,25,41,56,66,81,90,106,112, | 3,11,33,50,67,73,91,107,121, | 4,18,26,42,57,68,82,96,108,113, | 5,19,34,51,69,74,97,109,122, | 6,27,43,58,70,83,98,110,114, | 7,35,52,71,75,99,111,123, | 12,20,28,44,59,84,92,100,115, | 13,21,36,53,76,93,101,124, | 14,29,45,60,85,94,102,116, | 15,37,54,77,95,103,125, | 22,30,46,61,86,117, | 23,38,78,126, | 31,47,62,87,118, | 39,55,63,79,119,127]
 ^--- expert 0 (10) ---^        ^--- expert 1 (10) ---^        ^--- expert 2 (10) ---^      ... 每段长度 = 该专家命中数 ...       ^ exp15(6)^
```

每个值都是 0..127 之间的原始行号, 但**顺序被重排成专家主序**。`ids_dst` 在本例与之相同。

### 27.6 输入输出布局可视化

> 本节图示以 **down 投影**为例(`b=[cols,8,16]`, 每专家一个独立中间态, 见 27.0.1)。up 投影 `b=[cols,1,16]` 广播时图改成"每 token 仅 1 行", `ids_src1` 退化为纯 token 号。

**输入 `b`(原始, token 主序)** - `[cols, k=8, n_tokens=16]`, 铺平 128 行, 每行一个 `[cols]` 向量:

```
行号:  token  slot
  0..7 :  t0   slot0..7   (t0 的 8 个专家输入, 8 行连续)
  8..15:  t1   slot0..7
 16..23:  t2   slot0..7
 ...
120..127: t15  slot0..7
```

```
b (token-major, 128 rows):
 [t0s0][t0s1][t0s2]...[t0s7] [t1s0][t1s1]...[t1s7] ... [t15s0]...[t15s7]
  ^--- t0 的 8 行 ---^        ^--- t1 的 8 行 ---^
```

**紧凑缓冲(重排后, 专家主序)** - 量化时由 `quantize_mmq_q8_1_cuda` 按 `ids_src1` 边 gather 边量化([mmq.cu:200-205](ggml/src/ggml-cuda/mmq.cu#L200)), 仍是 128 行, 但顺序变成:

```
compact buffer (expert-major, 128 rows):
 [expert0 的 10 行][expert1 的 10 行][expert2 的 10 行][exp3 的 9 行]...[exp15 的 6 行]
  r0..r9            r10..r19          r20..r29          r30..r38       ... r122..r127
  ^-- 这些行来自 b 的 t0/t1/t2/t3/t5/t6/t8/t10/t11/t13 的 slot0 (按 ids_src1 gather)
```

可视化对照(行 -> 来源):

```
原始 b (token-major)                 紧凑缓冲 (expert-major, 按 ids_src1 重排)
t0s0 r0    t0s1 r1    ... t0s7 r7     expert0: r0=t0s0  r1=t1s0  r2=t2s0  r3=t3s0  r4=t5s0  r5=t6s0  r6=t8s0  r7=t10s0  r8=t11s0  r9=t13s0
t1s0 r8    t1s1 r9    ... t1s7 r15    expert1: r10=t0s1 r11=t1s1 r12=t2s1 r13=t4s0 r14=t6s1 r15=t8s1 r16=t9s0 r17=t11s1 r18=t13s1 r19=t15s0
t2s0 r16   ...                        expert2: r20=t0s2 r21=t1s2 r22=t3s1 r23=t5s1 r24=t7s0 r25=t8s2 r26=t10s1 r27=t11s2 r28=t13s2 r29=t14s0
...                                   ...
t15s0 r120 ... t15s7 r127             expert15:r122=t4s7 r123=t6s7 r124=t7s7 r125=t9s7 r126=t14s7 r127=t15s7
```

**关键性质**: 同一专家的输入向量现在**物理连续**(expert0 的 10 行紧挨着), GEMM 读它们时 cache 连续、warp 不发散; 而同一 token 的输入被拆散到不同专家段(t0 的 8 个 slot 散在 expert0..7 各段开头) - 但这无所谓, 因为 GEMM 是**按专家**组织的, 不需要同 token 连续。

### 27.7 下游 GEMM 如何消费这三个张量

紧凑缓冲就位后, 一次 MMQ/MMF GEMM 完成全部激活。以浮点路径为例([mmf.cuh:343-561](ggml/src/ggml-cuda/mmf.cuh#L343)), 内核的 grid 用 `blockIdx.y = expert_idx`(**每专家一组 block**):

```c
const int expert_idx   = blockIdx.y;
const int expert_start = expert_bounds[expert_idx];      // 本专家紧凑段起点  [mmf.cuh:346]
const int expert_end   = expert_bounds[expert_idx + 1];  // 终点               [mmf.cuh:347]
const int ncols_expert = expert_end - expert_start;      // 本专家分到的 token 数
if (ncols_expert == 0) return;                           // 空专家直接返回(跳过)
const int32_t * ids_src_expert = ids_src_compact + expert_start;  // 本专家的 gather 索引段  [mmf.cuh:369]
const int32_t * ids_dst_expert = ids_dst_compact + expert_start;  // 本专家的 scatter 索引段 [mmf.cuh:370]

// 取权重: 本专家的权重切片 src0[:,:,expert_idx]  (channel_x = expert_idx)        [mmf.cuh:360,365]
x += channel_x * stride_channel_x;

// 取输入: 按 ids_src_expert[global_j] 从紧凑缓冲(已 gather)取, 再解码回 (token, channel) [mmf.cuh:404-409]
const int src_entry = ids_src_expert[global_j];
token   = src_entry / sis1;          // = it
channel = src_entry % sis1;          // = slot (down) 或 0 (up)
val = y[channel*stride_channel_y + token*stride_col_y + col];

// 写输出: 按 ids_dst_expert[global_j] scatter 回 c                         [mmf.cuh:550-557]
const int dst_entry = ids_dst_expert[global_j];
token = dst_entry / nch_fd;          // = it
slot  = dst_entry % nch_fd;          // = iex_used
dst[slot * stride_channel_dst + token * stride_col_dst + row0 + ...] = sum;
```

**这就是"专家激活"的完整闭环**: 每个专家 block 组只读自己那段紧凑输入(连续)、自己那片权重(`as[:,:,expert_idx]`, 一次读入复用给该专家全部 token), 算完按 `ids_dst` 直接 scatter 回原始 `c` 的 `(slot, token)` 位置 - **无需单独的 scatter pass**。空专家(`ncols_expert==0`)零计算零访存。

### 27.8 数值规模小结

- `ids`: 16x8 = 128 个 int32(交错, token 主序)。
- `ids_src1` / `ids_dst`: 各 128 个 int32(重排后, 专家主序)。
- `expert_bounds`: 257 个 int32(n_experts+1, CSR 边界)。
- 紧凑输入缓冲: 128 行 x `[cols]`(量化后 Q8_1)。
- 重排本身只动这 128 行**输入**(gather + 量化合一), **完全不搬运专家权重**; 权重搬运只发生在 GEMM 里、每活跃专家读 1 次(见第 26 节)。
- 256 个 block 中 240 个空专家 block 几乎零开销(`ncols_expert==0` 立即返回), 只有 16 个活跃专家 block 实际工作。

**`mm_ids_helper` 的全部价值**: 用一个 O(n_experts x n_tokens x k) 的轻量内核, 把"交错 ids"转换成"专家主序 CSR + gather/scatter 索引", 让后续 GEMM 能以"每专家一次连续 GEMM + scatter 回写"的方式执行 - 既有稀疏性(只算活跃专家)、又有连续性(同专家 token 紧凑), 这是 CUDA MoE 高效的核心(第二部分第 15 节)。

## 28. 小 batch(decode)的统一例外

几乎所有后端都为 **token 数很少的 decode 场景**单独准备了一条**不重排的向量化路径**,因为 token 少时重排开销不划算、且向量化内核更直接:

| 后端 | 小 batch 路径 | 阈值/条件 |
|------|------|------|
| CUDA | `mul_mat_vec_q` / `mul_mat_vec_q_moe` | `n_tokens ≤ MMVQ_MAX_BATCH_SIZE = 8` |
| Metal | `kernel_mul_mv_id`(每 (token,expert) 一份) | `ne21 < 32` 或无 simdgroup matrix |
| SYCL | `ggml_sycl_mul_mat_id_mmvq_fused` | `ne12 == 1` |
| OpenCL | `gemv_moe_*` 内核 | gemv vs gemm 按 batch 分派 |
| WebGPU | `mul_mat_id_vec` | 小 batch |
| CPU | (本身即向量化 `vec_dot`,无单独阈值) | — |

这印证了第三部分第 21 节的洞察:**重排是为"大 batch / prefill"服务的**;decode(单 token 或 ≤8 token)时各后端普遍走"直接按 ids 索引"的向量化路径,绕开重排。

## 29. 总结

1. **重排 token 按专家主序,是 llama.cpp 跨后端支持 MoE 的主流范式**:CUDA、CPU、Metal、SYCL、OpenCL、ZenDNN、WebGPU 七个后端都做,差别仅在重排放 GPU 内核 / host CPU / 查找表,以及 GEMM 用 grid-z 专家分组还是 host for 循环。
2. **Vulkan 是自研内核里的特例**:不显式重排,而在 GEMM 内核内内联过滤 ids + shared memory 映射回原坐标,概念上仍是专家分组计算,只是把重排融进了 GEMM。
3. **CANN 是唯一完全不重排 token 的后端**:改在权重侧用 `aclnnIndexSelect` 抽出 k 个被选专家权重,再用昇腾 `aclnnBatchMatMul`(FP,一次算完 k 专家)/ `aclnnWeightQuantBatchMatmulV2`(量化,每专家一次)完成。这依赖厂商批量矩阵乘原生支持 batch 维,故 token 连续性不再重要。
4. **Hexagon 委托闭源 HTP**,不可见。
5. **差异的根源**:重排是"自研 GEMM 内核为连续访存/无发散"的刚需;一旦后端能用厂商库把"专家"当 batch 维原生处理(如 CANN BatchMatMul),重排即可省略。语义层面所有后端等价,均为 `c[:, e, t] = as[:, :, ids[e,t]] @ b[:, e, t]`。
6. **小 batch(decode)统一绕过重排**:各后端均有向量化直索引路径(CUDA `MMVQ_MAX_BATCH_SIZE=8`、Metal `ne21<32`、SYCL `ne12==1` 等),重排主要为 prefill/大 batch 服务。
7. **权重搬运量取决于"GEMM 能否按专家合并多 token",而非"是否重排"**(第 26 节):重排方案与 Vulkan 内联过滤都达到 `n_active * W` 的理论下界(每活跃专家权重只读 1 次,同专家多 token 共享);CANN 因"逐 token 不合并 + IndexSelect 物化拷贝",权重被重复读,算法下界是 `2 * n_tokens * k * W`,约为重排的 2 至 20+ 倍。这是**带宽效率 vs 算力效率**的权衡 - CANN 用权重带宽换"无需自研重排内核 + 昇腾 BatchMatMul 的算力利用率",decode 时被 NPU cache 吸收划算,prefill 大 batch 突破 cache 后权重带宽吃亏。重排只是实现"按专家合并"的手段之一,Vulkan 用内核内联达到同样合并,CANN 放弃合并改用厂商 batched-matmul。

## 附录补充:其他后端关键文件速查

| 后端 | 关注点 | 文件 |
|------|------|------|
| Metal | mul_mat_id 入口 | [ggml-metal-ops.cpp:2282](ggml/src/ggml-metal/ggml-metal-ops.cpp#L2282) |
| Metal | 重排内核 `map0` | [ggml-metal.metal:9761](ggml/src/ggml-metal/ggml-metal.metal#L9761) |
| Metal | GEMM 内核(grid z=ne02) | [ggml-metal.metal:9827](ggml/src/ggml-metal/ggml-metal.metal#L9827) |
| Metal | 向量化路径(decode) | [ggml-metal.metal:10367](ggml/src/ggml-metal/ggml-metal.metal#L10367) |
| SYCL | mul_mat_id 入口 | [ggml-sycl.cpp:4251](ggml/src/ggml-sycl/ggml-sycl.cpp#L4251) |
| SYCL | host 计数排序 | [ggml-sycl.cpp:4212](ggml/src/ggml-sycl/ggml-sycl.cpp#L4212) `mmid_counting_sort_rows` |
| SYCL | gather/scatter copy 内核 | [ggml-sycl.cpp:4123](ggml/src/ggml-sycl/ggml-sycl.cpp#L4123), [4146](ggml/src/ggml-sycl/ggml-sycl.cpp#L4146) |
| SYCL | 专家循环(host for) | [ggml-sycl.cpp:4367](ggml/src/ggml-sycl/ggml-sycl.cpp#L4367) |
| SYCL | 融合向量化(decode) | [ggml-sycl.cpp:4166](ggml/src/ggml-sycl/ggml-sycl.cpp#L4166) |
| Vulkan | mul_mat_id 入口 | [ggml-vulkan.cpp:9586](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9586) |
| Vulkan | dispatch {m, nei1, n_as} | [ggml-vulkan.cpp:7947](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7947) |
| Vulkan | 内联过滤 shader | [vulkan-shaders/mul_mm_id_funcs.glsl](ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_id_funcs.glsl), [mul_mmq.comp](ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp) |
| Vulkan | count_experts(跳空 workgroup) | [ggml-vulkan.cpp:5105](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5105) |
| CANN | mul_mat_id 入口 | [ggml-cann.cpp:1918](ggml/src/ggml-cann/ggml-cann.cpp#L1918) |
| CANN | 分派 | [aclnn_ops.cpp:3841](ggml/src/ggml-cann/aclnn_ops.cpp#L3841) |
| CANN | FP: IndexSelect + BatchMatMul | [aclnn_ops.cpp:3571](ggml/src/ggml-cann/aclnn_ops.cpp#L3571) |
| CANN | 量化: 每专家 WeightQuantBatchMatmulV2 | [aclnn_ops.cpp:3648](ggml/src/ggml-cann/aclnn_ops.cpp#L3648) |
| OpenCL | moe_router 重排流水 | [ggml-opencl.cpp:14746](ggml/src/ggml-opencl/ggml-opencl.cpp#L14746) |
| OpenCL | histogram/scan/fill/scatter 内核 | [ggml-opencl.cpp:662](ggml/src/ggml-opencl/ggml-opencl.cpp#L662) |
| WebGPU | mul_mat_id(gather+main) | [ggml-webgpu.cpp:1626](ggml/src/ggml-webgpu/ggml-webgpu.cpp#L1626) |
| WebGPU | 向量化路径 | [ggml-webgpu.cpp:1568](ggml/src/ggml-webgpu/ggml-webgpu.cpp#L1568) |
| Hexagon | supports + HTP 委托 | [ggml-hexagon.cpp:2639](ggml/src/ggml-hexagon/ggml-hexagon.cpp#L2639), [3134](ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3134) |
| ZenDNN | mul_mat_id(matrix_rows) | [ggml-zendnn.cpp:226](ggml/src/ggml-zendnn/ggml-zendnn.cpp#L226) |
