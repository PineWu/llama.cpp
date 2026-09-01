# llama.cpp 代码架构分析报告

> 基于 LLM 推理基础背景的源码级架构梳理
> 分析对象: llama.cpp 仓库 (master 分支)
> 整理日期: 2026-06-22

---

## 0. 背景与定位

llama.cpp 是一个面向 **大语言模型 (LLM) 本地推理** 的 C/C++ 高性能引擎,由 Georgi Gerganov 发起。它的核心目标是: 让任意人能在任意设备 (从树莓派/手机到多卡 GPU 服务器) 上以最低依赖、最高效率运行主流开源大模型。

从 LLM 推理的角度,一次完整推理需要解决以下问题:

1. **模型存储与加载** - 如何把数十 GB 的权重高效存盘并按需映射进内存 (GGUF 格式 + mmap)。
2. **前向计算图构建** - 把 Transformer/Mamba/RWKV 等架构翻译成一组可调度算子 (graph)。
3. **异构后端执行** - 在 CPU/CUDA/Metal/Vulkan/SYCL 等后端上执行该图,并支持多卡切分。
4. **低比特量化** - 把权重压到 4bit/2bit/1bit 以换取显存与带宽 (k-quants/i-quants)。
5. **自回归生成控制** - KV 缓存管理、采样 (top-k/p, mirostat, grammar)、批处理。
6. **多模型/多请求服务化** - 通过 HTTP server 提供连续批处理 (continuous batching)。

llama.cpp 的代码分层正是围绕这六个问题展开的。下面分层给出整体框架、核心设计思想、功能模块与对应代码文件路径。

---

## 1. 整体框架 (分层架构)

llama.cpp 采用清晰的四层结构,自底向上依次为:

```
+------------------------------------------------------------------+
| Layer 4: 应用层 (tools/, examples/)                                |
|   cli / server(OpenAI,Anthropic 兼容) / quantize / llama-bench ... |
+------------------------------------------------------------------+
| Layer 3: llama 高层 C++ 库 (src/, include/llama.h)                |
|   model / context / graph / sampler / grammar / vocab / adapter   |
|   memory(KV cache, recurrent, hybrid) / chat template              |
+------------------------------------------------------------------+
| Layer 2: ggml 后端抽象 (ggml/src/ggml-backend*, ggml/src/ggml-alloc) |
|   backend registry / device / buffer / scheduler (多卡切分)        |
+------------------------------------------------------------------+
| Layer 1: ggml 核心张量库 (ggml/include/ggml.h, ggml/src/ggml.c)    |
|   ggml_tensor / ggml_cgraph / 算子 (mul_mat, rope, flash_attn...)  |
|   量化内核 (ggml-quants) / GGUF 序列化 (gguf) / 训练 (ggml-opt)     |
+------------------------------------------------------------------+
        |                    |                  |               |
   ggml-cpu           ggml-cuda/hip       ggml-metal      ggml-vulkan/sycl/cann/rpc ...
```

**目录映射**:

| 层 | 关键目录 | 作用 |
|----|---------|------|
| Layer 1 核心张量库 | [ggml/include/](ggml/include/), [ggml/src/](ggml/src/) | 张量定义、算子、量化、GGUF、内存分配、训练 |
| Layer 1 硬件后端 | [ggml/src/ggml-cpu/](ggml/src/ggml-cpu/), [ggml/src/ggml-cuda/](ggml/src/ggml-cuda/), [ggml/src/ggml-metal/](ggml/src/ggml-metal/), [ggml/src/ggml-vulkan/](ggml/src/ggml-vulkan/), ... | 各后端算子实现 |
| Layer 2 后端抽象 | [ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp), [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp) | 设备注册、调度器 |
| Layer 3 llama 库 | [src/](src/), [include/llama.h](include/llama.h) | 模型/上下文/图/采样/词汇表/内存/适配器 |
| Layer 3 各架构图构建 | [src/models/](src/models/) (132 个文件) | 每个模型架构一个文件,定义计算图 |
| 共享工具库 | [common/](common/) | CLI 参数、采样、chat 模板、Jinja、下载 |
| Layer 4 应用 | [tools/](tools/), [examples/](examples/) | server/cli/quantize/bench ... |
| Python 转换 | [convert_hf_to_gguf.py](convert_hf_to_gguf.py), [gguf-py/](gguf-py/) | HF 模型转 GGUF |

---

## 2. 核心设计思想

### 2.1 "先建图,后执行" (Lazy Graph Execution)

这是整个引擎的灵魂。ggml 不像 PyTorch 那样即时求值,而是:

1. 在 CPU 上用 `ggml_mul_mat()`、`ggml_add()`、`ggml_rope()` 等 API **搭出一棵算子 DAG** (每个 `ggml_tensor` 节点记录 `op` 和 `src[]`),此时不分配数据、不计算。
2. 用 `ggml_new_graph()` + `ggml_build_forward_expand()` 把这棵 DAG 收集成 `ggml_cgraph`。
3. 把图交给后端 `ggml_backend_graph_compute(backend, graph)` 执行,数据才真正在 CPU/GPU 上流动。

好处: 图结构是后端无关的,同一份图可被任意后端执行;调度器可做内存复用 (中间张量生命周期分析) 与跨设备切分。详见 [ggml/include/ggml.h](ggml/include/ggml.h) (`ggml_tensor` 约 666 行, `ggml_cgraph` 约 2712 行) 与 [ggml/src/ggml.c](ggml/src/ggml.c)。

### 2.2 后端抽象与动态注册 (Pluggable Backend)

ggml 把"硬件设备"抽象成统一接口 `ggml_backend_dev_t`,每个后端实现 `supports_op()` 决定能否执行某算子。后端可 **动态加载** (dlopen),通过 [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp) 注册,`ggml_backend_load_all()` 扫描加载所有可用后端。这使引擎能在编译期不知道目标硬件的情况下,运行时按需启用 CUDA/Metal/Vulkan。

调度器 [ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp) 中的 `ggml_backend_sched_*` 负责把一个图自动切分到多设备 (如权重在 CPU、矩阵乘在 GPU),并在边界插入自动 tensor 拷贝与同步。这是 llama.cpp 多卡推理的基础。

### 2.3 数据导向的量化设计 (Block-Quantization)

量化不是全局标量缩放,而是 **分块 (block)** 存储: 每个固定大小 (如 32 或 256) 的元素共享一个 scale (+可选 min)。这使得反量化可以 SIMD/GPU 并行,且精度损失可控。所有量化格式定义在 [ggml/src/ggml-common.h](ggml/src/ggml-common.h) (`block_q4_0`, `block_q4_K` ...),参考实现与 `quantize_row_*` / `dequantize_row_*` 在 [ggml/src/ggml-quants.c](ggml/src/ggml-quants.c)。支持从 F32/F16/BF16 到 Q8_0、k-quants (Q2_K~Q8_K)、重要性感知 i-quants (IQ1/IQ2/IQ3/IQ4)、乃至 MXFP4/NVFP4/TQ1_0 等共数十种格式。

### 2.4 统一内存接口 (Unified Memory Abstraction)

为同时支持 Transformer (KV cache)、Mamba/RWKV (recurrent state)、混合架构 (Jamba/Granite),llama.cpp 抽象出 `llama_memory_i` 接口 ([src/llama-memory.h](src/llama-memory.h)),下挂 `llama_kv_cache`、`llama_memory_recurrent`、`llama_memory_hybrid` 等实现。同一套序列操作 (`seq_rm/cp/keep/add/div`) 对所有缓存类型统一,上层 context 无需关心底层是 attention 还是 SSM。

### 2.5 每架构一文件的可扩展模型系统

[src/llama-arch.h](src/llama-arch.h) 用一个 `llm_arch` 枚举登记所有支持架构 (132+ 个,覆盖 Llama/Qwen/DeepSeek/Gemma/Mamba/RWKV/BERT ...),并用 `LLM_TN`/`LLM_KV` 生成架构相关的张量名。[src/llama-model.cpp](src/llama-model.cpp) 的 `build_arch_graph()` 用一个巨大的 `switch(arch)` 分派到具体实现。每个架构在 [src/models/](src/models/) 下一个文件 (如 [src/models/llama.cpp](src/models/llama.cpp), [src/models/qwen3.cpp](src/models/qwen3.cpp)),实现:

- `load_arch_hparams()` - 解析该架构特有超参
- `load_arch_tensors()` - 声明/创建该架构的张量
- `struct graph : public llm_graph_context` - 搭建该架构的前向计算图

新模型只需新增一个文件 + 在 arch 枚举登记,无需改动核心。详见 [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md)。

### 2.6 C ABI 作为稳定边界

[include/llama.h](include/llama.h) 是稳定的 C 接口 (LLAMA_API),所有上层 (Swift/Android/Python 绑定、server、cli) 都只依赖它。内部 C++ 实现可自由重构,只要 ABI 不变即可。这是项目能长期演进而不破坏生态的关键。

### 2.7 mmap + 零拷贝加载

权重默认通过内存映射 (mmap) 加载 ([src/llama-mmap.cpp](src/llama-mmap.cpp)),配合 `use_mmap` 标志,模型文件直接映射进进程地址空间,按页按需读入,启动快、可被多进程共享。配合量化,使在消费级硬件上运行大模型成为可能。

---

## 3. 主要功能模块与代码文件路径

### 3.1 Layer 1: ggml 核心张量库

| 模块 | 文件 | 说明 |
|------|------|------|
| 张量定义与算子声明 | [ggml/include/ggml.h](ggml/include/ggml.h) | `ggml_tensor`, `ggml_op` (mul_mat, rope, flash_attn_ext, ssm_conv, ssm_scan ...), `ggml_cgraph` |
| 核心实现 | [ggml/src/ggml.c](ggml/src/ggml.c) | 图构建/拓扑序/参考实现 (~7800 行) |
| 量化格式定义 | [ggml/src/ggml-common.h](ggml/src/ggml-common.h) | `block_q4_0` 等所有 block 结构 |
| 量化参考内核 | [ggml/src/ggml-quants.c](ggml/src/ggml-quants.c), [ggml/src/ggml-quants.h](ggml/src/ggml-quants.h) | `quantize_row_*` / `dequantize_row_*` / `vec_dot_*` |
| 内存分配器 | [ggml/src/ggml-alloc.c](ggml/src/ggml-alloc.c), [ggml/include/ggml-alloc.h](ggml/include/ggml-alloc.h) | `ggml_tallocr` (单张量) + `ggml_gallocr` (整图复用分配) |
| 训练/优化层 | [ggml/src/ggml-opt.cpp](ggml/src/ggml-opt.cpp), [ggml/include/ggml-opt.h](ggml/include/ggml-opt.h) | `ggml_opt_dataset`, AdamW/SGD, 前向+反向+优化步 |
| GGUF 格式 | [ggml/src/gguf.cpp](ggml/src/gguf.cpp), [ggml/include/gguf.h](ggml/include/gguf.h) | 模型文件读写 (magic + KV metadata + 张量表 + 数据) |
| 线程池 | [ggml/src/ggml-threading.cpp](ggml/src/ggml-threading.cpp) | CPU 多线程 |

**ggml_tensor 结构** (核心数据结构):

```c
struct ggml_tensor {
    enum ggml_type  type;                 // F32/F16/Q4_K ...
    int64_t  ne[GGML_MAX_DIMS];           // 各维元素数 (最多 4 维)
    size_t   nb[GGML_MAX_DIMS];           // 各维字节步长 (支持非连续)
    enum ggml_op op;                      // 生成该张量的算子
    int32_t  op_params[...];              // 算子参数
    struct ggml_tensor * src[GGML_MAX_SRC]; // 源张量 (DAG 边, 最多 10)
    void *   data;                        // 实际数据指针
    struct ggml_backend_buffer * buffer;  // 所属后端缓冲区
};
```

### 3.2 Layer 1: 硬件后端 (Hardware Backends)

每个后端一个 `ggml/src/ggml-<name>/` 目录,实现 [ggml/src/ggml-backend-impl.h](ggml/src/ggml-backend-impl.h) 中的 `ggml_backend_device_i` 接口 (`get_name`, `supports_op`, `get_buffer_type`, `init_backend` ...):

| 后端 | 目录 | 覆盖硬件 |
|------|------|----------|
| CPU | [ggml/src/ggml-cpu/](ggml/src/ggml-cpu/) | x86/ARM/RISC-V, SIMD (AVX/NEON), 详见 [ggml/include/ggml-cpu.h](ggml/include/ggml-cpu.h) |
| CUDA | [ggml/src/ggml-cuda/](ggml/src/ggml-cuda/) | NVIDIA GPU (100+ .cu 内核) |
| HIP | [ggml/src/ggml-hip/](ggml/src/ggml-hip/) | AMD GPU |
| Metal | [ggml/src/ggml-metal/](ggml/src/ggml-metal/) | Apple macOS/iOS GPU |
| Vulkan | [ggml/src/ggml-vulkan/](ggml/src/ggml-vulkan/) | 跨平台 GPU |
| SYCL | [ggml/src/ggml-sycl/](ggml/src/ggml-sycl/) | Intel oneAPI |
| CANN | [ggml/src/ggml-cann/](ggml/src/ggml-cann/) | 华为昇腾 NPU |
| RPC | [ggml/src/ggml-rpc/](ggml/src/ggml-rpc/) | 远程后端 (分布式) |
| 其他 | ggml-blas, ggml-opencl, ggml-openvino, ggml-musa, ggml-hexagon, ggml-webgpu, ggml-zdnn, ggml-zendnn | 各类加速器 |

### 3.3 Layer 2: 后端抽象与调度

| 模块 | 文件 | 说明 |
|------|------|------|
| 后端接口 | [ggml/include/ggml-backend.h](ggml/include/ggml-backend.h) | `ggml_backend_t`, `ggml_backend_dev_t`, `ggml_backend_buffer_t`, `ggml_backend_buffer_type_t` |
| 后端实现 + 调度器 | [ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp) | `ggml_backend_sched_new/split_graph/graph_compute` 多卡切分 |
| 后端注册表 | [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp) | `ggml_backend_dev_count/get`, `ggml_backend_init_best`, `ggml_backend_load_all` |
| 动态加载 | [ggml/src/ggml-backend-dl.cpp](ggml/src/ggml-backend-dl.cpp) | dlopen 加载后端共享库 |
| 内部接口 | [ggml/src/ggml-backend-impl.h](ggml/src/ggml-backend-impl.h) | `ggml_backend_device_i` (各后端实现的虚表) |

### 3.4 Layer 3: llama 高层库 (模型与执行)

| 模块 | 文件 | 说明 |
|------|------|------|
| 公共 C API | [include/llama.h](include/llama.h) | 稳定 ABI: model/context/batch/encode/decode/sampler/vocab/state/adapter/memory/chat |
| API 实现入口 | [src/llama.cpp](src/llama.cpp) | C API 转发到内部 C++ 对象 |
| 架构枚举与张量命名 | [src/llama-arch.h](src/llama-arch.h), [src/llama-arch.cpp](src/llama-arch.cpp) | `llm_arch` (132+), `LLM_TN`/`LLM_KV`, arch 分类判定 |
| 模型加载器 | [src/llama-model-loader.h](src/llama-model-loader.h), [src/llama-model-loader.cpp](src/llama-model-loader.cpp) | 打开 GGUF、解析 metadata、张量映射、多文件/split、mmap/direct-io |
| 模型保存器 | [src/llama-model-saver.h](src/llama-model-saver.h), [src/llama-model-saver.cpp](src/llama-model-saver.cpp) | 写回 GGUF |
| 模型表示与图分派 | [src/llama-model.h](src/llama-model.h), [src/llama-model.cpp](src/llama-model.cpp) | `llama_model` 结构、`llama_layer`、`build_arch_graph()` switch 分派 |
| 超参 | [src/llama-hparams.h](src/llama-hparams.h), [src/llama-hparams.cpp](src/llama-hparams.cpp) | 从 GGUF 解析的 `llama_hparams` (n_embd, n_head, rope, MoE, SWA ...) |
| 图构建基类 | [src/llama-graph.h](src/llama-graph.h), [src/llama-graph.cpp](src/llama-graph.cpp) | `llm_graph_context`: 通用 attention/ffn/rope/build_inp_* 工具 |
| 各架构图实现 | [src/models/](src/models/) (132 文件) | 每架构一个 `llama_model_<arch>`,内嵌 `struct graph` |
| 图构建基类 (SSM) | [src/models/models.h](src/models/models.h) | `llm_build_mamba_base`, `llm_build_delta_net_base`, `llm_build_rwkv6/7_base` |
| 上下文/执行 | [src/llama-context.h](src/llama-context.h), [src/llama-context.cpp](src/llama-context.cpp) | `llama_context`: `encode()`/`decode()`/`graph_compute()`, 调度器预留, 图复用 |
| 批处理 | [src/llama-batch.h](src/llama-batch.h), [src/llama-batch.cpp](src/llama-batch.cpp) | `llama_batch` (公开) -> `llama_ubatch` (微批), `llama_batch_allocr` 切分 |
| 上下文参数 | [src/llama-cparams.h](src/llama-cparams.h), [src/llama-cparams.cpp](src/llama-cparams.cpp) | `n_ctx/n_batch/n_ubatch/n_threads/flash_attn/causal_attn` |
| mmap | [src/llama-mmap.h](src/llama-mmap.h), [src/llama-mmap.cpp](src/llama-mmap.cpp) | 内存映射文件 |
| 内部工具 | [src/llama-impl.h](src/llama-impl.h), [src/llama-impl.cpp](src/llama-impl.cpp) | 日志、错误、内部宏 |
| 量化 | [src/llama-quant.h](src/llama-quant.h), [src/llama-quant.cpp](src/llama-quant.cpp) | 模型级量化封装 |

**模型图构建约定** (以 Llama 为例, [src/models/llama.cpp](src/models/llama.cpp)):

```cpp
struct llama_model_llama : public llama_model_base {
    void load_arch_hparams(llama_model_loader & ml) override;  // 解析该架构超参
    void load_arch_tensors(llama_model_loader & ml) override;   // 创建 tok_embd/层权重
    struct graph : public llm_graph_context {                   // 搭建前向图
        graph(const llama_model & model, const llm_graph_params & params);
    };
    std::unique_ptr<llm_graph_context> build_arch_graph(...) const override;
};
```

### 3.5 Layer 3: 内存子系统 (Memory / KV Cache)

| 模块 | 文件 | 说明 |
|------|------|------|
| 内存抽象接口 | [src/llama-memory.h](src/llama-memory.h), [src/llama-memory.cpp](src/llama-memory.cpp) | `llama_memory_i`: `init_batch/full/update`, `clear/seq_rm/cp/keep/add/div` |
| KV 缓存单元 | [src/llama-kv-cells.h](src/llama-kv-cells.h) | 缓存槽位分配 |
| 标准 KV 缓存 | [src/llama-kv-cache.h](src/llama-kv-cache.h), [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp) | Transformer K/V 张量、槽管理、支持 Q8 等类型 |
| DSA (双空间注意力) | [src/llama-kv-cache-dsa.h](src/llama-kv-cache-dsa.h), [.cpp](src/llama-kv-cache-dsa.cpp) | Qwen3 等索引式注意力 |
| iSWA (集成滑动窗) | [src/llama-kv-cache-iswa.h](src/llama-kv-cache-iswa.h), [.cpp](src/llama-kv-cache-iswa.cpp) | 全局+滑动窗混合 |
| 循环状态 | [src/llama-memory-recurrent.h](src/llama-memory-recurrent.h), [.cpp](src/llama-memory-recurrent.cpp) | Mamba/RWKV 的 SSM 状态 |
| 混合架构内存 | [src/llama-memory-hybrid.h](src/llama-memory-hybrid.h), [.cpp](src/llama-memory-hybrid.cpp) | 同时含 KV + 循环状态 (Jamba/Granite) |
| 混合 + iSWA | [src/llama-memory-hybrid-iswa.h](src/llama-memory-hybrid-iswa.h), [.cpp](src/llama-memory-hybrid-iswa.cpp) | 上述组合 |
| IO 抽象 | [src/llama-io.h](src/llama-io.h), [src/llama-io.cpp](src/llama-io.cpp) | 状态读写序列化 |

### 3.6 Layer 3: 采样与约束生成

| 模块 | 文件 | 说明 |
|------|------|------|
| 采样器框架 | [src/llama-sampler.h](src/llama-sampler.h), [src/llama-sampler.cpp](src/llama-sampler.cpp) | `llama_sampler_i` 接口 + `llama_sampler_chain` 链式组合 |
| 内置采样器 | (同上) | greedy/dist/top-k/top-p/min-p/typical/temp/ext/xtc/top_n_sigma/mirostat(v1/v2)/penalties/dry/adaptive_p/logit_bias/infill |
| 语法约束 (GBNF) | [src/llama-grammar.h](src/llama-grammar.h), [src/llama-grammar.cpp](src/llama-grammar.cpp) | `llama_gretype`, 语法规则栈, 候选过滤 |

采样器接口设计为可链式组合,且支持后端硬件加速 (`backend_init/apply/accept`)。

### 3.7 Layer 3: 词汇表与对话模板

| 模块 | 文件 | 说明 |
|------|------|------|
| 词汇表/分词器 | [src/llama-vocab.h](src/llama-vocab.h), [src/llama-vocab.cpp](src/llama-vocab.cpp) | 6 种分词: SPM/BPE/WPM/UGM/RWKV/PLAMO2, 55+ 预分词变体 |
| Unicode | [src/unicode.h](src/unicode.h), [src/unicode.cpp](src/unicode.cpp), [src/unicode-data.h](src/unicode-data.h) | UTF-8 规范化 |
| 对话模板 | [src/llama-chat.h](src/llama-chat.h), [src/llama-chat.cpp](src/llama-chat.cpp) | 60+ 内置模板, `llm_chat_detect_template/apply_template` |

### 3.8 Layer 3: 适配器 (LoRA / 控制向量)

| 模块 | 文件 | 说明 |
|------|------|------|
| 适配器 | [src/llama-adapter.h](src/llama-adapter.h), [src/llama-adapter.cpp](src/llama-adapter.cpp) | `llama_adapter_lora` (A/B 秩分解, 支持 aLoRA), `llama_adapter_cvec` (控制向量) |

### 3.9 共享工具库 (common/)

为所有上层应用提供统一的基础设施,避免重复:

| 模块 | 文件 | 说明 |
|------|------|------|
| CLI 参数框架 | [common/arg.h](common/arg.h), [common/arg.cpp](common/arg.cpp) | 命令行标志、handler、环境变量、preset |
| 核心工具 | [common/common.h](common/common.h), [common/common.cpp](common/common.cpp) | model 加载、设备管理、采样初始化、adapter 加载 |
| 采样封装 | common/sampling.cpp/.h | 统一采样链 (DRY/top-k/p/mirostat/grammar) |
| 对话模板检测 | [common/chat.h](common/chat.h), [common/chat.cpp](common/chat.cpp) | 模板检测、消息格式化、工具调用提取 |
| PEG 解析器 | [common/peg-parser.cpp](common/peg-parser.cpp), [common/chat-peg-parser.cpp](common/chat-peg-parser.cpp) | 替代正则的模板解析 |
| 自动解析器 | [common/chat-auto-parser.cpp](common/chat-auto-parser.cpp), helpers | 自动检测模型特性 |
| Jinja 引擎 | [common/jinja/](common/jinja/) | 完整 Jinja2 实现 (lexer/parser/runtime), 详见 [common/jinja/README.md](common/jinja/README.md) |
| JSON Schema -> GBNF | [common/json-schema-to-grammar.cpp](common/json-schema-to-grammar.cpp) | 结构化输出约束 |
| llguidance 集成 | [common/llguidance.cpp](common/llguidance.cpp) | 详见 [docs/llguidance.md](docs/llguidance.md) |
| ngram 缓存 | [common/ngram-cache.cpp](common/ngram-cache.cpp), ngram-map/mod | 投机解码 lookup 加速 |
| HF 下载 | [common/download.cpp](common/download.cpp), [common/hf-cache.cpp](common/hf-cache.cpp) | 从 HuggingFace 下载并缓存 |
| imatrix 加载 | [common/imatrix-loader.cpp](common/imatrix-loader.cpp) | 校准量化用重要性矩阵 |
| 内存自适应 | [common/fit.cpp](common/fit.cpp) | 自动调 batch 适配显存 |
| 控制台/日志 | [common/console.cpp](common/console.cpp), [common/log.cpp](common/log.cpp) | 彩色输出/spinner/进度/分级日志 |

### 3.10 Layer 4: 应用与工具 (tools/)

| 工具 | 主文件 | 用途 |
|------|--------|------|
| HTTP server | [tools/server/server.cpp](tools/server/server.cpp), [server-http.cpp](tools/server/server-http.cpp), [server-context.cpp](tools/server/server-context.cpp) | OpenAI/Anthropic 兼容 API, 连续批处理, 多 slot |
| CLI 对话 | [tools/cli/main.cpp](tools/cli/main.cpp), [tools/cli/cli.cpp](tools/cli/cli.cpp) | 交互式 REPL |
| quantize | [tools/quantize/quantize.cpp](tools/quantize/quantize.cpp) | 模型量化转换 |
| llama-bench | tools/llama-bench/llama-bench.cpp | 跨配置性能基准 |
| perplexity | tools/perplexity/perplexity.cpp | 困惑度评估 |
| imatrix | tools/imatrix/imatrix.cpp | 生成重要性矩阵 |
| completion | tools/completion/completion.cpp | 文本补全 |
| tokenize | tools/tokenize/tokenize.cpp | 分词调试 |
| gguf-split | tools/gguf-split/gguf-split.cpp | GGUF 拆分/合并 |
| export-lora | tools/export-lora/export-lora.cpp | LoRA 导出为独立模型 |
| cvector-generator | tools/cvector-generator/cvector-generator.cpp | PCA 控制向量生成 |
| rpc-server | tools/rpc/rpc-server.cpp | TCP 暴露远程加速器 |
| tts | tools/tts/tts.cpp | OuteTTS 语音合成 |
| mtmd | [tools/mtmd/](tools/mtmd/) | 多模态 (图像/音频) |
| parser | tools/parser/debug-template-parser.cpp | 模板/语法调试 |
| batched-bench | tools/batched-bench/main.cpp | 批处理基准 |
| fit-params | tools/fit-params/fit-params.cpp | 自动调参适配显存 |

**server 核心特性**: 基于 cpp-httplib;slot 抽象实现并行请求与连续批处理 (`-np/--parallel`、`-cb/--cont-batching`);任务队列 `server_queue` 含主队列与延迟队列;端点覆盖 `/completion`、`/v1/chat/completions`、`/v1/embeddings`、`/v1/rerank`、`/v1/messages` (Anthropic)、`/tokenize`、`/slots`、`/lora-adapters`、`/metrics` 等。详见 [tools/server/README.md](tools/server/README.md) 与 [tools/server/README-dev.md](tools/server/README-dev.md)。

### 3.11 Python 转换工具链

| 脚本/包 | 路径 | 用途 |
|--------|------|------|
| HF -> GGUF 主转换器 | [convert_hf_to_gguf.py](convert_hf_to_gguf.py) | HuggingFace 模型/仓库 -> GGUF, 支持多种输出精度, 懒求值, 大模型 split |
| LoRA -> GGUF | [convert_lora_to_gguf.py](convert_lora_to_gguf.py) | LoRA 适配器转换 |
| 旧 GGML -> GGUF | [convert_llama_ggml_to_gguf.py](convert_llama_ggml_to_gguf.py) | 历史格式迁移 |
| GGUF Python 包 | [gguf-py/gguf/](gguf-py/gguf/) | `GGUFReader/Writer`, `constants.py`, `tensor_mapping.py` (HF->GGUF 张量映射, 50+ 架构), `quants.py`, `vocab.py`, `lazy.py` |

`convert_hf_to_gguf.py` 导入并使用 `gguf-py` 包完成实际序列化,架构特定逻辑在转换脚本中实现。

---

## 4. 端到端推理数据流 (串起各层)

以 server 一次 `/v1/chat/completions` 请求为例:

1. **请求接入** ([tools/server/server-http.cpp](tools/server/server-http.cpp)): httplib 收到请求,经 `server_queue` 调度分配给空闲 slot。
2. **模板与分词** ([common/chat.cpp](common/chat.cpp) + [src/llama-vocab.cpp](src/llama-vocab.cpp)): 把消息数组套用 chat 模板 -> `llama_tokenize()` 得到 token id 序列。
3. **图构建** ([src/llama-model.cpp](src/llama-model.cpp) -> [src/models/<arch>.cpp](src/models/)): `build_arch_graph()` 按架构搭出 `ggml_cgraph` (含 attention/ffn/rope/输出投影)。
4. **批切分** ([src/llama-batch.cpp](src/llama-batch.cpp)): prompt 拆成若干 `llama_ubatch`。
5. **Prefill** ([src/llama-context.cpp](src/llama-context.cpp) `encode`/`decode`): 每个微批走 `graph_compute()` -> 后端调度器 ([ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp)) 把图分到 CPU/GPU 执行,中间张量由 `ggml_gallocr` 复用分配 ([ggml/src/ggml-alloc.c](ggml/src/ggml-alloc.c))。
6. **KV 写入** ([src/llama-kv-cache.cpp](src/llama-kv-cache.cpp)): 计算出的 K/V 写入缓存,供后续 token 复用。
7. **采样** ([src/llama-sampler.cpp](src/llama-sampler.cpp) + 可选 [src/llama-grammar.cpp](src/llama-grammar.cpp)): 对末位 logit 跑采样链 -> 得到下一个 token。
8. **Decode 循环**: 把新 token 喂回 step 3-7,自回归生成至 EOS 或上限;流式返回给客户端。
9. **状态保存/恢复** ([src/llama-context.cpp](src/llama-context.cpp) `state_*`): 可把 KV/序列状态存盘,供会话续接。

---

## 5. 关键数据结构与对象关系

```
llama_model  (src/llama-model.h)
 |- llama_hparams        (src/llama-hparams.h)   <- GGUF metadata
 |- llama_vocab          (src/llama-vocab.h)
 |- vector<llama_layer>  (src/llama-model.h)      <- 每层权重张量指针
 |- build_arch_graph()   -> 分派到 src/models/<arch>.cpp::graph
        |
        v
llama_context (src/llama-context.h)
 |- llama_memory_i*      (src/llama-memory.h)     <- kv_cache / recurrent / hybrid
 |- llama_batch_allocr   (src/llama-batch.h)
 |- ggml_backend_sched   (ggml-backend.cpp)        <- 多后端图执行
 |- graph_compute() -> ggml_backend_graph_compute()
        |
        v
ggml_cgraph (ggml.h)  [mul_mat, rope, flash_attn_ext, ssm_scan, ...]
        |
        v
ggml_backend_dev (CPU/CUDA/Metal/...)  ->  命中 quantize/dequantize 内核 (ggml-quants)
```

---

## 6. 如何阅读这份代码 (建议路径)

1. **入口与稳定接口**: 先读 [include/llama.h](include/llama.h) 建立全局心智模型,再看 [examples/simple/](examples/simple/) 与 [tools/cli/cli.cpp](tools/cli/cli.cpp) 理解典型调用流程。
2. **张量与图**: 读 [ggml/include/ggml.h](ggml/include/ggml.h) 的 `ggml_tensor`/`ggml_op`/`ggml_cgraph`,配合 [examples/simple/](examples/simple/) 中如何搭图并执行。
3. **后端**: 从 [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp) 看注册机制,挑一个后端 (推荐 [ggml/src/ggml-cpu/](ggml/src/ggml-cpu/)) 理解 `supports_op` 与算子实现。
4. **模型系统**: 读 [src/llama-arch.h](src/llama-arch.h) 的架构枚举,再读 [src/models/llama.cpp](src/models/llama.cpp) 作为最简单的样例,理解 `load_arch_hparams`/`load_arch_tensors`/`struct graph` 三段式。新增模型参考 [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md)。
5. **采样与约束**: 读 [src/llama-sampler.h](src/llama-sampler.h) 接口与 [src/llama-grammar.cpp](src/llama-grammar.cpp),配合 [common/json-schema-to-grammar.cpp](common/json-schema-to-grammar.cpp) 理解结构化输出。
6. **服务化**: 读 [tools/server/README.md](tools/server/README.md) 与 [tools/server/server-context.cpp](tools/server/server-context.cpp) 理解 slot 与连续批处理。

---

## 7. 参考文档 (仓库内)

- 构建: [docs/build.md](docs/build.md), [docs/install.md](docs/install.md)
- 多卡: [docs/multi-gpu.md](docs/multi-gpu.md)
- server: [tools/server/README.md](tools/server/README.md), [tools/server/README-dev.md](tools/server/README-dev.md)
- 新增模型: [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md)
- 解析器: [docs/development/parsing.md](docs/development/parsing.md), [docs/autoparser.md](docs/autoparser.md)
- Jinja: [common/jinja/README.md](common/jinja/README.md)
- 投机解码: [docs/speculative.md](docs/speculative.md)
- 多模态: [docs/multimodal.md](docs/multimodal.md)
- 算子清单: [docs/ops.md](docs/ops.md), [docs/ops/](docs/ops/)
- 函数调用: [docs/function-calling.md](docs/function-calling.md)
- llguidance: [docs/llguidance.md](docs/llguidance.md)

---

## 附: 模块速查表

| 关注点 | 核心文件 |
|--------|---------|
| 张量/算子/图 | [ggml/include/ggml.h](ggml/include/ggml.h), [ggml/src/ggml.c](ggml/src/ggml.c) |
| 量化 | [ggml/src/ggml-quants.c](ggml/src/ggml-quants.c), [ggml/src/ggml-common.h](ggml/src/ggml-common.h) |
| 后端抽象/调度 | [ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp), [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp) |
| 内存分配 | [ggml/src/ggml-alloc.c](ggml/src/ggml-alloc.c) |
| GGUF 格式 | [ggml/src/gguf.cpp](ggml/src/gguf.cpp), [gguf-py/](gguf-py/) |
| 训练 | [ggml/src/ggml-opt.cpp](ggml/src/ggml-opt.cpp) |
| 模型加载 | [src/llama-model-loader.cpp](src/llama-model-loader.cpp) |
| 架构登记 | [src/llama-arch.h](src/llama-arch.h) |
| 各架构图 | [src/models/](src/models/) |
| 上下文/执行 | [src/llama-context.cpp](src/llama-context.cpp) |
| KV/内存 | [src/llama-memory.h](src/llama-memory.h), [src/llama-kv-cache.cpp](src/llama-kv-cache.cpp) |
| 采样 | [src/llama-sampler.cpp](src/llama-sampler.cpp) |
| 语法约束 | [src/llama-grammar.cpp](src/llama-grammar.cpp) |
| 词汇表/分词 | [src/llama-vocab.cpp](src/llama-vocab.cpp) |
| LoRA/控制向量 | [src/llama-adapter.cpp](src/llama-adapter.cpp) |
| 公共 API | [include/llama.h](include/llama.h) |
| CLI 参数/采样/chat | [common/](common/) |
| HTTP server | [tools/server/](tools/server/) |
| 模型转换 | [convert_hf_to_gguf.py](convert_hf_to_gguf.py) |

---

## 8. 内存分配接口体系:与 `ggml_gallocr` 相似的分配器对比

前面提到 `ggml_gallocr` 是"整图复用分配器"。实际上 llama.cpp 的分配需求是分层的(从一块裸显存,到一组张量,到一张计算图,到跨多后端的图),ggml 暴露了一套**层次化分配接口**,`ggml_gallocr` 只是其中一层。下表先给全景,再逐个对比。

### 8.1 分配接口全景

| 接口 | 声明位置 | 分配对象 | 是否做生命周期/复用分析 | 缓冲区来源 |
|------|----------|----------|--------------------------|------------|
| `ggml_backend_alloc_buffer` | [ggml-backend.h:82](ggml/include/ggml-backend.h#L82) | 一块裸 buffer(按大小) | 否 | 后端直接分配 |
| `ggml_backend_buft_alloc_buffer` | [ggml-backend.h:38](ggml/include/ggml-backend.h#L38) | 一块裸 buffer(按 buft) | 否 | 指定 buft 分配 |
| `ggml_tallocr` | [ggml-alloc.h:14-22](ggml/include/ggml-alloc.h#L14) | 单个张量 | 否(bump 线性递增) | 调用方预先备好 buffer |
| `ggml_backend_alloc_ctx_tensors[_from_buft]` | [ggml-alloc.h:80-82](ggml/include/ggml-alloc.h#L80) | 一组 ctx 里的张量 | 否(批量铺排,不回收) | 内部新建 buffer |
| `ggml_gallocr` | [ggml-alloc.h:46-74](ggml/include/ggml-alloc.h#L46) | 整张 cgraph 的所有节点 | **是(按图拓扑分析生命周期)** | 指定一种/多种 buft |
| `ggml_backend_sched_alloc_graph` | [ggml-backend.h:341](ggml/include/ggml-backend.h#L341) | 整张图(跨多后端) | **是(在 gallocr 之上 + 调度)** | 每后端一个 gallocr |
| `ggml_backend_graph_copy` | [ggml-backend.h:417](ggml/include/ggml-backend.h#L417) | 图的静态副本(跨后端拷贝) | 否(静态铺排) | 目标后端新建 buffer |

层次关系:

```
裸显存
  └── ggml_backend_alloc_buffer / buft_alloc_buffer        (拿一块大 buffer)
        └── ggml_tallocr                                     (在 buffer 里线性分张量)
        └── ggml_backend_alloc_ctx_tensors                   (把一组张量铺进新 buffer)
        └── ggml_gallocr                                     (按图拓扑复用分张量,可跨多 buft)
              └── ggml_backend_sched                         (多后端 + gallocr + 自动拷贝)
                    └── ggml_backend_graph_copy              (把图静态复制到另一后端)
```

### 8.2 关键差异维度

| 维度 | `ggml_tallocr` | `ggml_backend_alloc_ctx_tensors` | `ggml_gallocr` | `ggml_backend_sched` |
|------|-----------------|----------------------------------|----------------|----------------------|
| **输入** | 单个张量 | 一个 `ggml_context` 里的张量集合 | 一整张 `ggml_cgraph` | 一整张图 + 多后端 |
| **分配策略** | bump(线性递增 offset,不回收) | 遍历 ctx 张量连续铺排,不回收 | **按图拓扑序分配 + 释放**,中间张量生命周期结束即回收复用 | 在 gallocr 之上 + 跨后端切图 + 边界拷贝 |
| **生命周期分析** | 无 | 无 | **有**(基于 `hash_node` 的 `n_children/n_views`,见 [ggml-alloc.c:456](ggml/src/ggml-alloc.c#L456)) | 复用 gallocr |
| **复用内存** | 否 | 否 | **是**(同一段 buffer 可被多个不重叠生命周期的张量复用) | 是 |
| **多后端** | 否(单 buffer) | 否(单 buft) | 是(`new_n` 支持多种 buft,每节点定 buft) | **是**(核心卖点) |
| **buffer 归属** | 调用方自备 | 接口内部新建并返回 | 接口内部按 buft 新建/复用 | 接口内部 |
| **需要 reserve** | 否 | 否 | 可选(单 buffer 自动 realloc;多 buffer 需 `reserve_n`) | 是(`sched_reserve` 用最坏图) |
| **典型开销** | 极低 | 低(一次遍历) | 中(建 hash + 拓扑序遍历) | 高(切图 + 每后端 gallocr) |
| **是否触动图** | 否 | 否 | 是(给节点写 `buffer/data`) | 是 |

### 8.3 各接口实现要点

#### `ggml_tallocr` — bump allocator([ggml-alloc.c](ggml/src/ggml-alloc.c))

```c
enum ggml_status ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor) {
    size_t size = GGML_PAD(ggml_backend_buffer_get_alloc_size(talloc->buffer, tensor), talloc->alignment);
    // 越界则 ABORT(不会扩容)
    void * addr = (char *)base + talloc->offset;
    talloc->offset += size;                          // 只前进,不回收
    return ggml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}
```

最简单的"指针递增"。无生命周期分析、无复用、不能扩容。调用方必须保证 buffer 够大。**用途**:对一块已知大小的 buffer 做确定性顺序分配——典型是 llama.cpp 里"复用上次的输出 buffer 装新拷贝"(见 [src/llama-context.cpp:2646](src/llama-context.cpp#L2646) 的 `mbuf_cur` 拷贝区)。

#### `ggml_backend_alloc_ctx_tensors[_from_buft]` — 批量静态分配([ggml-alloc.c](ggml/src/ggml-alloc.c))

遍历一个 `ggml_context` 里**所有张量**,算总大小,新建一个 buffer,把它们连续铺进去。**不分析图、不回收**——张量是静态的(权重、KV cache 槽),生命周期等于整个 context,没必要复用。

- `_from_buft_size` 只算大小不分配(用于预留)
- `_from_buft` 实际分配并返回 buffer

**用途**:模型权重([src/llama-model.cpp:1544](src/llama-model.cpp#L1544))、KV cache 张量([src/llama-kv-cache.cpp:297](src/llama-kv-cache.cpp#L297))、LoRA 权重([src/llama-adapter.cpp:82](src/llama-adapter.cpp#L82))。这些是"一次分配、长期持有"的,不需要图级复用。

#### `ggml_gallocr` — 图级复用分配器(本节主角)

核心是**按图拓扑序做生命周期分析**([ggml-alloc.c:717](ggml/src/ggml-alloc.c#L717) `ggml_gallocr_alloc_graph_impl`):

1. 遍历图节点,对每个张量在 `hash_node` 里记 `n_children`(还有多少下游用它)、`n_views`(被 view 引用次数)。
2. 分配节点时 `ggml_dyn_tallocr_alloc` 在 buffer 里找 best-fit 空闲块([ggml-alloc.c:201](ggml/src/ggml-alloc.c#L201),支持 chunk + 空闲块链表)。
3. 当一个张量的 `n_children` 归零(且非 OUTPUT/INPUT)→ `ggml_gallocr_free_node` 释放回空闲块池,供后续张量复用([ggml-alloc.c:690](ggml/src/ggml-alloc.c#L690))。
4. `GGML_TENSOR_FLAG_INPUT`:图输入,分配在 buffer 开头且**不重叠**(避免被中间张量覆盖)。
5. `GGML_TENSOR_FLAG_OUTPUT`:图输出,**永不释放、永不被覆盖**([ggml-alloc.c:692](ggml/src/ggml-alloc.c#L692))。
6. view:复用 view_src 的地址(零拷贝),见 [ggml-alloc.c:660](ggml/src/ggml-alloc.c#L660)。

底层 `ggml_dyn_tallocr` 是一个**带空闲块回收的 arena**:维护 `chunks[]`,每 chunk 有 `free_blocks`(best-fit)和 `allocated_tensors`(调试用)。这跟 `ggml_tallocr` 的纯 bump 不同——它能"挖洞回收"。

关键 API 三档:
- `ggml_gallocr_alloc_graph`:单 buffer 自动(必要时 realloc);**多 buffer 不行**。
- `ggml_gallocr_reserve[_n]`:用最坏情况图**预分配**,避免热路径 realloc([ggml-alloc.c:961](ggml/src/ggml-alloc.c#L961))。
- `ggml_gallocr_reserve_n`:多 buffer 时**必须**先 reserve 指定每节点落哪个 buffer,再 alloc。

**用途**:计算图的中间张量(激活值)。一次 decode 里 n_tokens 变化,中间张量大小变,但图拓扑不变 → reserve 一次最坏情况,后续 alloc_graph 直接复用,热路径零 realloc。

#### `ggml_backend_sched_alloc_graph` — 多后端 + gallocr([ggml-backend.cpp](ggml/src/ggml-backend.cpp))

在 `ggml_gallocr` 之上加了一层:**先切图**(按 `supports_op` 把节点分到各后端),**再对每个后端用 gallocr 分配**,并在边界插入自动拷贝。它内部持有一个 `ggml_gallocr`(多 buft 版本),`sched_reserve` 转发到 `ggml_gallocr_reserve_n`。

`ggml_backend_sched_alloc_graph` 与 `ggml_gallocr_alloc_graph` 的关系:
- 后者只管"在一个或多个已知 buft 上铺张量",**不管切图**。
- 前者**先决定每节点去哪个后端**(切图),再调用后者分配,还要处理跨后端张量拷贝。
- `sched_alloc_graph` 会比较新旧 `node_backend_ids`,只有 buft 变化才重新 reserve([ggml-backend.cpp](ggml/src/ggml-backend.cpp) `ggml_backend_sched_alloc_splits`),否则复用——这是 decode 热路径不反复分配的关键。

**用途**:实际推理(`llama_decode`)的中间张量分配,支持 CPU+GPU 混合、多卡。详见 [docs/op-scheduling-analysis.md](op-scheduling-analysis.md)。

#### `ggml_backend_graph_copy` — 图静态复制([ggml-backend.h:417](ggml/include/ggml-backend.h#L417))

把一张图整体复制到另一个后端,产出 `ctx_allocated`(已分配)+ `ctx_unallocated`(未分配)+ 新图。不做生命周期复用(静态铺排)。**用途**:调试、后端对比(`ggml_backend_compare_graph_backend`)、把图固化到某后端。

### 8.4 各接口适用场景

| 场景 | 选哪个 | 理由 |
|------|--------|------|
| 模型权重、KV cache 张量等"一次分配长期持有" | `ggml_backend_alloc_ctx_tensors_from_buft` | 静态张量,无需图级复用;接口内部自动算总大小并新建 buffer |
| LoRA 权重、控制向量等小批量静态张量 | `ggml_backend_alloc_ctx_tensors` 或 `ggml_tallocr` | 同上;`tallocr` 适合自备 buffer 的确定性顺序分配 |
| 复用一块已分配 buffer 装新数据(如拷贝缓冲) | `ggml_tallocr` | bump 分配极轻量,确定性偏移,适合已知大小的临时区 |
| 单后端计算图的中间张量(简单场景) | `ggml_gallocr` | 图级生命周期复用,省显存;单 buffer 自动 realloc |
| 单后端 + 变 batch(如纯 GPU 推理) | `ggml_gallocr` + `reserve` | reserve 最坏情况,decode 热路径零 realloc |
| 多后端 / CPU+GPU 混合 / 多卡 | `ggml_backend_sched_alloc_graph` + `sched_reserve` | 唯一能切图 + 跨后端分配 + 边界拷贝的接口 |
| 调试 / 后端对比 / 图固化 | `ggml_backend_graph_copy` | 静态复制整图到目标后端,便于对比或固化 |
| 只需要一块裸 buffer(自己管理) | `ggml_backend_alloc_buffer` / `buft_alloc_buffer` | 最底层,拿一块大 buffer 自己分配 |

### 8.5 一句话记忆

- `tallocr`:**线性 bump**,单 buffer 顺序分张量,不回收。
- `alloc_ctx_tensors`:**批量静态**,把一组张量铺进新 buffer,不回收。
- `gallocr`:**图级复用**,按拓扑分析生命周期,中间张量回收复用,是省显存的核心。
- `sched_alloc_graph`:**多后端 gallocr**,先切图再分配,推理实际走的入口。
- `graph_copy`:**静态复制**,调试/固化用。

**核心区分**:`tallocr` / `alloc_ctx_tensors` 用于**静态张量**(权重、KV、LoRA,生命周期=整个模型),不需要复用;`gallocr` / `sched` 用于**图中间张量**(激活值,生命周期=几条算子),需要复用以省显存。`gallocr` 是分水岭——它之上的 `sched` 加了切图,它之下的接口都没有生命周期分析。
