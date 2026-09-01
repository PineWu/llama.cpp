# llama.cpp 算子级后端调度与层卸载分析

> 承接 [docs/backend-architecture.md](backend-architecture.md),深入分析 llama.cpp 如何实现"把模型的一部分层/算子放到 GPU、另一部分留在 CPU"这种混合后端执行,以及具体到单个 `mul_mat`/FFN 算子是如何被决定执行设备、如何在设备边界搬运数据的。
> 分析对象: llama.cpp 仓库 (master 分支)
> 整理日期: 2026-06-22

---

## 0. 问题与全景

LLM 推理常有这样的需求:显存装不下整个 70B 模型,但能装下前 N 层;或者干脆没有 GPU,只有 CPU + 一块弱加速器。llama.cpp 用两个正交机制解决:

| 机制 | 粒度 | 决策时机 | 由谁决定 | 控制参数 |
|------|------|----------|----------|----------|
| **层卸载 (layer offload)** | 整层权重放哪 | 模型加载时 | llama 高层 | `n_gpu_layers`, `tensor_split` |
| **算子分派 (op placement)** | 单个图节点跑哪 | 每次 decode 建图时 | ggml 调度器 | `supports_op` / `offload_op` / `op_offload` |
| **手动钉选 (manual pin)** | 特定节点固定后端 | 建图时 | 模型图代码 | `ggml_backend_sched_set_tensor_backend` |

三者协作: **层卸载决定权重张量物理落在哪个设备的缓冲 → 算子分派依据"权重在哪算子就在哪"的默认规则,但允许 `offload_op` 覆盖 → 极少数特殊算子(如不卸载 KQV)用手动钉选强制走 CPU**。下文逐一拆解。

---

## 1. 层卸载: `n_gpu_layers` 如何把权重分发到不同设备

### 1.1 参数定义与规范化

`n_gpu_layers` 定义在 [include/llama.h](include/llama.h) 的 `llama_model_params`:

```c
int32_t n_gpu_layers;   // number of layers to store in VRAM, negative = all layers
const float * tensor_split; // size: llama_max_devices(), per-device split proportions
```

规范化在 [src/llama-model.cpp:1651-1653](src/llama-model.cpp#L1651):

```c
uint32_t llama_model::n_gpu_layers() const {
    return params.n_gpu_layers >= 0 ? params.n_gpu_layers : hparams.n_layer_all + 1;
}
```

负值表示"全部层 + 输出层都放 GPU"。

### 1.2 核心: 按层索引决定设备

[src/llama-model.cpp:1267-1292](src/llama-model.cpp#L1267) 是层卸载的心脏:

```c
const int i_gpu_start   = std::max(n_layer_all + 1 - n_gpu_layers, 0); // GPU 起始层索引
const int act_gpu_layers = devices.empty() ? 0 : std::min(n_gpu_layers, n_layer_all + 1);

auto get_layer_buft_list = [&](int il) -> layer_dev {
    // 第 il 层的索引 < i_gpu_start -> CPU;或超出实际可卸载层数 -> CPU
    if (il < i_gpu_start || (il - i_gpu_start) >= act_gpu_layers) {
        return {cpu_dev, &pimpl->cpu_buft_list};
    }
    // 在多 GPU 间按 tensor_split 比例分配这一层给哪张卡
    const int layer_gpu = std::upper_bound(
        splits.begin(), splits.begin() + n_devices(),
        float(il - i_gpu_start) / act_gpu_layers) - splits.begin();
    // ... 返回对应 GPU dev 及其 buft_list
    return {dev, &pimpl->gpu_buft_list.at(dev)};
};

pimpl->dev_input  = {cpu_dev, &pimpl->cpu_buft_list};           // 输入嵌入始终 CPU
for (int il = 0; il < n_layer_all; ++il) {
    pimpl->dev_layer[il] = get_layer_buft_list(il);             // 每层定设备
}
pimpl->dev_output = get_layer_buft_list(n_layer_all);           // 输出头随最后一层
```

关键设计:
- **按层号切分,后 N 层上 GPU,前面留 CPU**(注意是把靠后的层放 GPU,因为靠后层的权重通常最先被频繁访问)。
- `n_layer_all + 1` 多出来的那一层是 **输出投影 (output/tok_embd)**,与最后一层共享设备。
- `splits` 是 `tensor_split` 规范化成的累计分布 (0.0→1.0),用 `upper_bound` 把每层映射到一张卡 - 这是多卡 **层间并行** (pipeline) 的分配。

### 1.3 张量层分类: 决定用哪个 buft_list

[src/llama-arch.h:571-575](src/llama-arch.h#L571) 把每个张量按位置分三类:

```c
enum llm_tensor_layer {
    LLM_TENSOR_LAYER_INPUT,     // 输入嵌入等,始终 CPU
    LLM_TENSOR_LAYER_REPEATING, // 每层重复的权重,按 n_gpu_layers 决定
    LLM_TENSOR_LAYER_OUTPUT,    // 输出头,随最后一层
};
```

[src/llama-model-loader.cpp:1127-1148](src/llama-model-loader.cpp#L1127) 在 `create_tensor` 里据此选 buft_list:

```c
switch (info.layer) {
    case LLM_TENSOR_LAYER_INPUT:    buft_list = buft_list_input;  break;   // CPU
    case LLM_TENSOR_LAYER_OUTPUT:   buft_list = buft_list_output; break;
    case LLM_TENSOR_LAYER_REPEATING:
        GGML_ASSERT(buft_list_layer != nullptr);
        buft_list = buft_list_layer; break;   // 已按层号定好 CPU/GPU
}
```

随后 [src/llama-model-loader.cpp:1033](src/llama-model-loader.cpp#L1033) 的 `select_weight_buft()` 在该 buft_list 里挑具体缓冲类型 (如 Q4_K 权重选对应的量化 buffer type),最终 `ml.create_tensor()` 把张量分配到该后端的缓冲。

**结果**: 模型加载完成后,每个权重张量都已物理落在某设备缓冲里 (`tensor->buffer->buft` 指向某后端)。这是后续算子分派的"地基"。

### 1.4 多卡: `tensor_split` 与 split buffer

[src/llama-model.cpp:923-946](src/llama-model.cpp#L923) `make_gpu_buft_list()`:当 `split_mode == LLAMA_SPLIT_MODE_ROW` 时,调用后端导出的 `ggml_backend_split_buffer_type_fn(dev_index, tensor_split)` 创建 **行切分缓冲类型** - 同一个权重张量按行切成多段,分布在多卡上 ([src/llama-model.cpp:1235-1265](src/llama-model.cpp#L1235) 把 `tensor_split` 规范化为累计分布)。

CUDA 侧 [ggml/src/ggml-cuda/ggml-cuda.cu:1438](ggml/src/ggml-cuda/ggml-cuda.cu#L1438) `ggml_backend_cuda_split_buffer_type` 实现:每个张量在每卡上只占 `row_low..row_high` 行,`MUL_MAT` 时每卡算自己那部分行,再经 [allreduce.cu](ggml/src/ggml-cuda/allreduce.cu) 聚合。这是 **层内并行** (张量并行),与 1.2 的层间 pipeline 正交。详见 [docs/multi-gpu.md](docs/multi-gpu.md)。

---

## 2. 算子分派: 调度器如何决定每个节点跑在哪

> 核心文件: [ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp)

`ggml_backend_sched` 把 N 个后端按"注册顺序 = 优先级"排成数组,**最后一个必须是 CPU** (构造时断言 [ggml-backend.cpp:1736](ggml/src/ggml-backend.cpp#L1736))。每次 `ggml_backend_sched_graph_compute()` 先调 `ggml_backend_sched_split_graph()` 给每个节点定后端,再按"split"分组执行。

### 2.1 单节点决策: `ggml_backend_sched_backend_id_from_cur`

> [ggml/src/ggml-backend.cpp:878-933](ggml/src/ggml-backend.cpp#L878)

这是"这个算子去哪"的五步决策树:

```
对每个待决策张量 tensor (它是一个 op 节点):
1. 已预分配目标缓冲?  -> 用该缓冲所属后端 (要求 supports_buft + supports_op)   [cause "1.dst"]
2. 是 view?           -> 用 view_src 的缓冲后端                                [cause "1.vsrc"]
3. 缓冲已分配却无后端支持? -> ABORT (配置错误)
4. 是图输入(INPUT)?   -> 丢给最后一个后端 (CPU 兜底)                          [cause "1.inp"]
5. 有权重类源 src (buffer->usage == WEIGHTS)?
   a. 取 src 的后端 src_backend_id
   b. 若 src 在 CPU 但更高优先级后端愿意 offload_op -> 升到 GPU              [cause "1.off"]
   c. 否则 -> 就在权重所在后端跑                                              [cause "1.wgt%d"]
```

**第 5 步是最关键的默认规则**: 算子跟随它的权重张量 - 权重在 GPU,`mul_mat` 就在 GPU;权重在 CPU,就在 CPU。这正是第 1 节层卸载的延伸: **层卸载决定权重位置,权重位置决定算子位置**。

`ggml_backend_sched_backend_from_buffer()` ([ggml-backend.cpp:845-865](ggml/src/ggml-backend.cpp#L845)) 是步骤 1/2 的底层:遍历后端数组,返回第一个同时满足 `supports_buft` 且 `supports_op` 的(下标最小=优先级最高)。

> 注意 [ggml-backend.cpp:908-930](ggml/src/ggml-backend.cpp#L908) 有个小优化: ROPE 算子不依据 freqs 权重选后端 (freqs 太小,无意义),用专门的跳过。

### 2.2 `offload_op`: 权重在 CPU 但算子想去 GPU

第 5b 步的覆盖机制。CUDA 在 [ggml/src/ggml-cuda/ggml-cuda.cu:5458-5462](ggml/src/ggml-cuda/ggml-cuda.cu#L5458) 实现:

```c
static bool ggml_backend_cuda_device_offload_op(dev, op) {
    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}
```

`get_op_batch_size()` ([ggml-cuda.cu:5443-5456](ggml/src/ggml-cuda/ggml-cuda.cu#L5443)): `MUL_MAT` 用 `ne[1]`,`MUL_MAT_ID`/`ROPE` 用 `ne[2]`,`GET_ROWS` 恒 0(永不卸载)。

调度器侧 ([ggml-backend.cpp:919-925](ggml/src/ggml-backend.cpp#L919)):

```c
if (sched->op_offload && src_backend_id == n_backends - 1  // 权重在 CPU
    && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {              // 试更高优先级后端
        if (ggml_backend_supports_op(backends[b], tensor)
            && ggml_backend_offload_op(backends[b], tensor)) {
            SET_CAUSE(tensor, "1.off");
            return b;                                         // 升级到 GPU
        }
    }
}
```

效果: **即使某层权重仍在 CPU(显存不够没卸载),只要 batch 够大,CUDA 仍会把这次 `mul_mat` 拉到 GPU 跑** - 权重会在算子执行时按需 H2D 上传。这是"权重在 CPU、计算在 GPU"的细粒度折中,靠 `cparams.op_offload` 开关 ([src/llama-context.cpp:199](src/llama-context.cpp#L199))。

### 2.3 四趟扫描: 让相邻算子聚成连续块

`ggml_backend_sched_split_graph()` ([ggml-backend.cpp:1014-1240](ggml/src/ggml-backend.cpp#L1014)) 在单节点决策之上做图级优化:

| 趟 | 行号 | 作用 |
|----|------|------|
| **Pass 1** | 1035-1070 | 对所有 leaf 和 node 跑 2.1 的决策树,得到初步后端分配(用户已 `set_tensor_backend` 的不覆盖) |
| **Pass 2** | 1072-1150 | **贪婪扩展 GPU 区域**:向前向后把未分配且 GPU 支持的相邻节点并入同一 GPU 后端,减少跨设备拷贝。CPU 不触发扩展(只在 GPU 边界停) |
| **Pass 3** | 1152-1211 | 未分配节点选"支持它且支持最多输入缓冲"的后端;已分配节点尝试升到更高优先级后端(要求 buft 相同 + 全部输入缓冲兼容) |
| **Pass 4** | 1213-1243 | 收尾:view 跟随 view_src;剩余源跟节点同后端;最后兜底找第一个支持的后端 |

Pass 2 的核心逻辑 ([ggml-backend.cpp:1078-1097](ggml/src/ggml-backend.cpp#L1078)):

```c
// expand gpu down
int cur_backend_id = -1;
for (i in nodes) {
    if (*node_backend_id != -1) {
        if (*node_backend_id == n_backends - 1) cur_backend_id = -1; // CPU 不扩散
        else cur_backend_id = *node_backend_id;                       // GPU 继续扩散
    } else if (cur_backend_id != -1) {
        ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
    }
}
```

注释 [ggml-backend.cpp:1074](ggml/src/ggml-backend.cpp#L1074) 点明意图: *"expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)"* - 让 GPU 区域尽量连续成块。

---

## 3. 设备边界: 如何插入数据拷贝

分派完成后,相邻节点可能在不同后端。调度器在 Pass 5 ([ggml-backend.cpp:1245-1376](ggml/src/ggml-backend.cpp#L1245)) 检测边界并插入 **COPY 张量** (不是算子,是 `ggml_dup_tensor_layout` 出来的拷贝依赖):

```c
// 当 src 的后端 != 当前 split 的后端,且当前后端不支持 src 的缓冲类型时:
if (src_backend_id != cur_backend_id &&
    !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
    // 在当前后端建一份 src 的拷贝
    struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
    tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
    split->inputs[split->n_inputs++] = src;       // 记为 split 输入
    node->src[j] = tensor_copy;                   // 把节点的源换成拷贝
}
```

实际搬运发生在执行期 `ggml_backend_sched_compute_splits()` ([ggml-backend.cpp:1541-1725](ggml/src/ggml-backend.cpp#L1541)):

```c
for (input_id in split->inputs) {
    if (input->flags & GGML_TENSOR_FLAG_INPUT) {
        // 用户输入: 阻塞拷贝
        ggml_backend_synchronize(split_backend);
        ggml_backend_tensor_copy(input, input_cpy);
    } else {
        // 中间张量: 优先异步拷贝
        if (!split_backend->iface.cpy_tensor_async ||
            !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
            // 回退阻塞拷贝
            ggml_backend_synchronize(input_backend);
            ggml_backend_synchronize(split_backend);
            ggml_backend_tensor_copy(input, input_cpy);
        }
    }
}
```

要点:
- `cpy_tensor_async` (CUDA/Metal 等实现) 让数据搬运与计算重叠 ([ggml-cuda.cu](ggml/src/ggml-cuda/ggml-cuda.cu) 的 pinned host buffer 支撑异步 H2D/D2H)。
- `n_copies` 份拷贝做 **流水线并行** (`pipeline_parallel`, [src/llama-context.cpp:203](src/llama-context.cpp#L203)):一个 split 在算时,下一个 split 的输入已在搬运。
- 模型代码完全无感 - 它只搭 `ggml_cgraph`,边界拷贝由调度器自动插。

---

## 4. 手动钉选: 极少数算子强制走 CPU

公开 API ([ggml/include/ggml-backend.h:334-335](ggml/include/ggml-backend.h#L334)):

```c
void           ggml_backend_sched_set_tensor_backend(sched, node, backend);  // 用户指定,cause "usr"
ggml_backend_t ggml_backend_sched_get_tensor_backend(sched, node);
```

llama.cpp 内部确实用了 - 典型场景是 **不卸载 KQV (key-query-value 投影)**。在 [src/llama-graph.cpp:2157](src/llama-graph.cpp#L2157) 无缓存注意力路径里:

```c
cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);

if (!cparams.offload_kqv) {
    // all nodes between the KV store and the attention output are run on the CPU
    ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
}
```

意图: 当 `offload_kqv=false` 时,把从 KV 存储到注意力输出之间的所有节点强制钉在 CPU - 因为这部分涉及对 KV 缓存的频繁访问,放 CPU 有时更划算 (省显存/省 PCIE)。注意它只钉 `cur` 这一个节点,但 Pass 2 的 GPU 扩展会停在这里,后续相关算子也跟着留 CPU。

`ggml_backend_sched_set_tensor_backend` 实现 ([ggml-backend.cpp:1960-1967](ggml/src/ggml-backend.cpp#L1960)):

```c
void ggml_backend_sched_set_tensor_backend(sched, node, backend) {
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");   // 标记用户指定,Pass 1 不覆盖
    sched->is_reset = false;
}
```

这是三个机制里唯一需要模型代码主动参与的 - 绝大多数算子靠前两个机制自动就位。

---

## 5. 预留与分配: 避免 decode 时反复分配

> [src/llama-context.cpp:427](src/llama-context.cpp#L427) `sched_reserve()` + [ggml-backend.cpp:1847-1881](ggml/src/ggml-backend.cpp#L1847)

每次 decode 都要重建图,但缓冲不能每次重分配 (太慢)。做法:

1. **建最坏情况图**: [src/llama-context.cpp:468-635](src/llama-context.cpp#L468) 用最大 token/seq/output 数构建多个 reserve graph (FA 检查、prompt 阶段、生成阶段)。
2. **调度器预留**: `ggml_backend_sched_reserve(sched, measure_graph)` ([ggml-backend.cpp:1847](ggml/src/ggml-backend.cpp#L1847)) 把该图走一遍 split 逻辑,然后用 `ggml_gallocr_reserve_n()` ([ggml/src/ggml-alloc.c](ggml/src/ggml-alloc.c)) 在每个后端按最坏情况**预分配**缓冲。
3. **增量复用**: 实际 decode 时 `ggml_backend_sched_alloc_splits()` ([ggml-backend.cpp:1489-1539](ggml/src/ggml-backend.cpp#L1489)) 比较新旧 `node_backend_ids`,只有当 buft 变化才重新 reserve,否则直接复用已分配缓冲。

```c
// [ggml-backend.cpp:1489] ggml_backend_sched_alloc_splits
bool backend_ids_changed = /* 检查 node_backend_ids 是否变 */;
if (!backend_ids_changed && ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
    return true;   // 复用,零分配
}
// 否则:重新 reserve + alloc
ggml_gallocr_reserve_n(...);
return ggml_gallocr_alloc_graph(...);
```

`graph_compute()` ([src/llama-context.cpp:240](src/llama-context.cpp)) 里还有一个图复用开关 `graph_reuse_disable`: 当图拓扑没变时直接复用上次的 split 结果,跳过整个 split_graph。这让 decode 循环里只有真正变化时才重算分派。

---

## 6. 端到端示例: 一个混合 CPU/GPU 的 FFN 层

假设 `n_gpu_layers=20`,模型 32 层,层 12-31 在 GPU,层 0-11 在 CPU。decode 第 15 层的 FFN 时:

```
build_arch_graph (src/models/llama.cpp)
  产出图节点:  norm_rms -> gate_proj(mul_mat) -> silu -> up_proj(mul_mat) -> mul -> down_proj(mul_mat)
      权重 gate_proj.weight 物理在 GPU split buffer (层卸载决定)
      权重 norm.weight    也在 GPU

ggml_backend_sched_split_graph (ggml-backend.cpp)
  Pass 1: 每个 mul_mat 的 src[0] 是权重(GPU),走第5步 -> 后端 = GPU  [cause "1.wgt0"]
          norm 算子: src 是激活(无权重),但相邻 mul_mat 已是 GPU
  Pass 2: GPU 区域向前扩展,把 norm 也并入 GPU
          结果: 整个 FFN 连续块都在 GPU,无需跨设备拷贝
  (若上一层层 11 在 CPU):  边界检测 -> 在层11->12 之间插入 COPY
     执行时: cpy_tensor_async 把层11输出从 CPU pinned buffer 异步拷到 GPU

ggml_backend_sched_compute_splits (ggml-backend.cpp:1541)
  对 GPU split: ggml_backend_cuda_graph_compute -> 每节点 launch kernel (或整图走 CUDA Graph)
  对 CPU split: ggml_backend_cpu_graph_compute -> threadpool
```

若显存只够放权重但 `op_offload=true` 且 batch 够大:层卸载没把权重放 GPU(仍在 CPU),但 `offload_op` 让 `mul_mat` 升级到 GPU,执行时权重按需 H2D。

若该层是 attention 且 `offload_kqv=false`:即便整层权重在 GPU,`set_tensor_backend` 把 KQV 钉 CPU,Pass 2 扩展在此停止,attention 输出经 COPY 回流 CPU。

---

## 7. 三机制协作总结

```
                    权重物理位置                      算子默认后端              覆盖
层卸载 ──────────────────────────────►  (n_gpu_layers/tensor_split)
   决定每个权重张量 -> buffer -> device    │
                                           ▼
                                默认: 算子跟权重走 (决策树第5步 "1.wgt")
                                           │
offload_op ──────────────────────────────► 覆盖: 权重在CPU但batch够大 -> 升GPU ("1.off")
   (op_offload 开关 + CUDA get_op_batch_size)
                                           │
手动钉选 ─────────────────────────────────► 覆盖: set_tensor_backend 强制某节点 ("usr")
   (offload_kqv 等,仅极少数算子)            │
                                           ▼
                              四趟扫描: 让相邻同后端算子聚成连续 split
                                           │
                                           ▼
                              split 边界自动插入 COPY (async cpy_tensor_async)
                                           │
                                           ▼
                              各后端并行执行各 split,流水线搬运
```

**关键洞察**:
1. llama.cpp 不需要模型代码写"这个算子放哪" - **权重在哪算子就在哪**是默认规则,层卸载一处决定全局。
2. `offload_op` 是对默认规则的细粒度修正,实现"权重 CPU、计算 GPU"的折中。
3. 跨设备数据搬运完全自动 (COPY 张量 + async copy),模型图代码零感知。
4. 调度器的四趟扫描 + 预留复用,让混合后端在 decode 热路径上几乎零开销。

---

## 附: 关键文件速查

| 关注点 | 文件 |
|--------|------|
| n_gpu_layers / tensor_split 解析 | [src/llama-model.cpp:1211-1292](src/llama-model.cpp#L1211) |
| 张量层分类 (INPUT/REPEATING/OUTPUT) | [src/llama-arch.h:571](src/llama-arch.h#L571), [src/llama-model-loader.cpp:1127](src/llama-model-loader.cpp#L1127) |
| create_tensor 选 buft | [src/llama-model-loader.cpp:1033](src/llama-model-loader.cpp#L1033) |
| 单节点决策树 | [ggml/src/ggml-backend.cpp:878-933](ggml/src/ggml-backend.cpp#L878) |
| offload_op (CUDA) | [ggml/src/ggml-cuda/ggml-cuda.cu:5443-5462](ggml/src/ggml-cuda/ggml-cuda.cu#L5443) |
| 四趟 split 扫描 | [ggml/src/ggml-backend.cpp:1014-1243](ggml/src/ggml-backend.cpp#L1014) |
| COPY 插入与执行 | [ggml/src/ggml-backend.cpp:1245-1376](ggml/src/ggml-backend.cpp#L1245), [1541-1725](ggml/src/ggml-backend.cpp#L1541) |
| 手动钉选 API | [ggml/include/ggml-backend.h:334](ggml/include/ggml-backend.h#L334), [ggml/src/ggml-backend.cpp:1960](ggml/src/ggml-backend.cpp#L1960) |
| KQV 不卸载实例 | [src/llama-graph.cpp:2157](src/llama-graph.cpp#L2157) |
| 调度器预留/复用 | [src/llama-context.cpp:427](src/llama-context.cpp#L427), [ggml/src/ggml-backend.cpp:1489-1881](ggml/src/ggml-backend.cpp#L1489) |
| 多卡 split buffer | [src/llama-model.cpp:923](src/llama-model.cpp#L923), [ggml/src/ggml-cuda/ggml-cuda.cu:1438](ggml/src/ggml-cuda/ggml-cuda.cu#L1438) |
| 多卡使用指南 | [docs/multi-gpu.md](docs/multi-gpu.md) |
