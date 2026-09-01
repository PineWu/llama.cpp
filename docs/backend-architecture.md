# llama.cpp 硬件后端接入架构分析

> 承接 [docs/architecture-analysis.md](architecture-analysis.md) 的 Layer 1/2,深入分析 llama.cpp 如何让各种异构硬件 (CPU/CUDA/Metal/Vulkan/SYCL/CANN/HIP/RPC ...) 接入推理引擎。
> 分析对象: llama.cpp 仓库 (master 分支)
> 整理日期: 2026-06-22

---

## 0. 问题背景

一个 LLM 推理引擎要跑在多种硬件上,核心矛盾是: **算子实现与硬件强耦合** (矩阵乘在 CPU 上是 AVX/NEON 汇编,在 NVIDIA GPU 上是 CUDA kernel,在 Apple 上是 Metal shader),但 **上层模型逻辑又必须与硬件无关** (同一个 Transformer 图应当能在任意后端执行)。

llama.cpp 的解法是 ggml 的 **后端抽象层 (Backend Abstraction)**:

1. 定义一套统一的 C 风格"虚函数表"接口 (vtable):设备接口 `ggml_backend_device_i`、流接口 `ggml_backend_i`、缓冲区接口 `ggml_backend_buffer_type_i`、注册表接口 `ggml_backend_reg_i`。
2. 每个硬件后端实现这套 vtable,通过一个 `ggml_backend_<name>_reg()` 函数把自己登记进全局注册表。
3. 一个跨设备调度器 `ggml_backend_sched` 在运行时根据每个后端的 `supports_op()` 回答,把同一张计算图 (`ggml_cgraph`) 的不同节点分派到不同设备,并在边界自动插入 tensor 拷贝与同步。

这样,新增一个硬件后端 = 实现一套 vtable + 登记一次,**核心引擎完全不用改**。下面逐层拆解。

---

## 1. 四层接口契约 (Backend Interface Contract)

接口全部定义在 [ggml/src/ggml-backend-impl.h](ggml/src/ggml-backend-impl.h) (内部头) 与 [ggml/include/ggml-backend.h](ggml/include/ggml-backend.h) (公开头)。这是一组 C 函数指针结构体 (vtable),构成了后端接入的"合同"。

### 1.1 注册表接口 `ggml_backend_reg_i` (一个后端家族)

> 文件: [ggml/src/ggml-backend-impl.h](ggml/src/ggml-backend-impl.h)

```c
struct ggml_backend_reg {
    int   api_version;            // 必须等于 GGML_BACKEND_API_VERSION (=2)
    struct ggml_backend_reg_i iface;
    void * context;
};

struct ggml_backend_reg_i {
    const char * (*get_name)      (ggml_backend_reg_t reg);            // 家族名,如 "CUDA"
    size_t       (*get_device_count)(ggml_backend_reg_t reg);          // 该家族暴露几个设备
    ggml_backend_dev_t (*get_device)(ggml_backend_reg_t reg, size_t i);// 取第 i 个设备
    void *       (*get_proc_address)(ggml_backend_reg_t reg, const char * name); // 可选:导出扩展函数
};
```

`get_proc_address` 是后端扩展机制的入口 - 例如 CUDA 通过它导出多卡通信函数 `ggml_backend_cuda_comm_*` (allreduce)。

### 1.2 设备接口 `ggml_backend_device_i` (一个物理设备)

> 文件: [ggml/src/ggml-backend-impl.h](ggml/src/ggml-backend-impl.h)

这是后端接入 **最核心** 的契约,决定"这个设备是什么、能做什么、内存怎么分"。

```c
struct ggml_backend_device {
    struct ggml_backend_device_i iface;
    ggml_backend_reg_t reg;       // 指回所属注册表
    void * context;
};

struct ggml_backend_device_i {
    const char * (*get_name)        (dev);  // 设备名,如 "CUDA0"
    const char * (*get_description) (dev);  // 人类可读描述,如 "NVIDIA GeForce RTX 4090"
    void         (*get_memory)      (dev, size_t *free, size_t *total); // 查询显存
    enum ggml_backend_dev_type (*get_type)(dev);  // CPU / GPU / IGPU / ACCEL / META
    void         (*get_props)       (dev, struct ggml_backend_dev_props * props); // 填充完整属性
    ggml_backend_t            (*init_backend)(dev, const char * params);        // 实例化一个计算流
    ggml_backend_buffer_type_t (*get_buffer_type)(dev);       // 该设备的首选缓冲区类型
    ggml_backend_buffer_type_t (*get_host_buffer_type)(dev);  // 可选: 用于 H2D/D2H 的 pinned host 缓冲
    ggml_backend_buffer_t (*buffer_from_host_ptr)(dev, void *ptr, size_t size, size_t max_tensor_size); // 可选:从主机指针建缓冲(mmap)
    bool (*supports_op)(dev, const struct ggml_tensor * op);  // ★能否执行该算子(调度器核心判据)
    bool (*supports_buft)(dev, ggml_backend_buffer_type_t);    // 能否消费该缓冲类型里的张量
    bool (*offload_op)(dev, const struct ggml_tensor * op);   // 可选:即便权重不在本设备也愿不愿意跑(密集 matmul 下卸载)
    ggml_backend_event_t (*event_new)(dev);                    // 可选:同步事件
    void (*event_free)(dev, event);
    void (*event_synchronize)(dev, event);
};
```

设备类型枚举 ([ggml/include/ggml-backend.h](ggml/include/ggml-backend.h)):

```c
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,    // CPU,系统内存
    GGML_BACKEND_DEVICE_TYPE_GPU,    // 独显,专用显存
    GGML_BACKEND_DEVICE_TYPE_IGPU,   // 集显,共享主机内存
    GGML_BACKEND_DEVICE_TYPE_ACCEL,  // 加速器 (BLAS/AMX)
    GGML_BACKEND_DEVICE_TYPE_META,   // 元设备,包装多设备做张量并行
};
```

设备属性结构 `ggml_backend_dev_props` 包含 name/description/memory_free/memory_total/type/device_id(PCI bus id) 以及能力位 `caps.{async, host_buffer, buffer_from_host_ptr, events}`。

### 1.3 流接口 `ggml_backend_i` (一个计算流实例)

`init_backend()` 返回一个 `ggml_backend_t`,代表该设备上的一个执行流,负责真正的图执行与异步拷贝:

```c
struct ggml_backend {
    ggml_guid_t guid;              // 16 字节唯一标识后端类型
    struct ggml_backend_i iface;
    ggml_backend_dev_t device;
    void * context;
};

struct ggml_backend_i {
    const char * (*get_name)(backend);
    void (*free)(backend);
    // 异步拷贝(可选):用于跨后端、跨设备数据搬运
    void (*set_tensor_async)(backend, tensor, data, offset, size);
    void (*get_tensor_async)(backend, tensor, data, offset, size);
    void (*set_tensor_2d_async) / (*get_tensor_2d_async)(...);
    bool (*cpy_tensor_async)(backend_src, backend_dst, src, dst);
    void (*synchronize)(backend);
    // 图执行:plan 系列是可选的预编译加速路径,graph_compute 是必选
    ggml_backend_graph_plan_t (*graph_plan_create)(backend, cgraph);
    void (*graph_plan_free / update)(...);
    enum ggml_status (*graph_plan_compute)(backend, plan);
    enum ggml_status (*graph_compute)(backend, cgraph);   // ★核心:执行整张图
    void (*event_record / event_wait)(backend, event);
    void (*graph_optimize)(backend, cgraph);              // 可选:图拓扑优化
};
```

### 1.4 缓冲区类型接口 `ggml_backend_buffer_type_i`

决定张量数据物理放在哪、如何分配:

```c
struct ggml_backend_buffer_type_i {
    const char * (*get_name)(buft);
    ggml_backend_buffer_t (*alloc_buffer)(buft, size_t size);
    size_t (*get_alignment)(buft);          // 张量对齐要求
    size_t (*get_max_size)(buft);          // 可选,默认 SIZE_MAX
    size_t (*get_alloc_size)(buft, tensor);// 可选:含 padding 的实际占用
    bool   (*is_host)(buft);              // 可选:是否主机可见(可直接访问)
};
```

四种接口的层次关系:

```
ggml_backend_reg  (家族: "CUDA")          -- get_device_count / get_device
   └── ggml_backend_device (设备: CUDA0)  -- supports_op ★ / get_buffer_type / init_backend
          ├── ggml_backend_buffer_type    -- alloc_buffer / get_alignment
          │      └── ggml_backend_buffer   -- 实际内存块,set/get_tensor_data
          └── ggml_backend (流)            -- graph_compute ★ / set/get_tensor_async
```

---

## 2. 后端注册机制 (如何把后端"插进来")

### 2.1 静态 (编译期) 注册

> 文件: [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp)

全局注册表 `ggml_backend_registry` 的构造函数里,用一串 `#ifdef GGML_USE_<NAME>` 条件编译,把每个编译进来的后端登记进去 ([ggml-backend-reg.cpp:116-165](ggml/src/ggml-backend-reg.cpp#L116)):

```cpp
ggml_backend_registry() {
#ifdef GGML_USE_CUDA
    register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
    register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
    register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
    if (getenv("GGML_DISABLE_VULKAN") == nullptr)
        register_backend(ggml_backend_vk_reg());
#endif
#ifdef GGML_USE_CANN
    register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_RPC
    register_backend(ggml_backend_rpc_reg());
#endif
    // ... WEBGPU, ZDNN, OPENCL, HEXAGON, BLAS, OPENVINO, ZENDNN, VIRTGPU
#ifdef GGML_USE_CPU
    register_backend(ggml_backend_cpu_reg());   // CPU 始终兜底,放最后
#endif
}
```

注意 **CPU 永远最后注册** - 注册顺序就是优先级,调度器把它当 fallback。

每个后端的 `ggml_backend_<name>_reg()` 返回一个静态 `ggml_backend_reg` 结构(它的 `iface` 是该家族的 vtable)。以 CPU 为例 ([ggml/src/ggml-cpu/ggml-cpu.cpp:690-703](ggml/src/ggml-cpu/ggml-cpu.cpp#L690)):

```c
ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    static struct ggml_backend_reg reg = {
        /*.api_version =*/ GGML_BACKEND_API_VERSION,
        /*.iface       =*/ ggml_backend_cpu_reg_i,
        /*.context     =*/ nullptr,
    };
    return &reg;
}
GGML_BACKEND_DL_IMPL(ggml_backend_cpu_reg)   // 同一个函数,按动态库方式导出
```

### 2.2 动态 (dlopen) 加载

> 文件: [ggml/src/ggml-backend-dl.cpp](ggml/src/ggml-backend-dl.cpp), 宏定义 [ggml/src/ggml-backend-impl.h:235-271](ggml/src/ggml-backend-impl.h#L235)

当以 `GGML_BACKEND_DL=ON` 编译时,每个后端编译成独立 `.so/.dylib/.dll` (`MODULE` 库),运行时由 `ggml_backend_load_all_from_path()` 扫描 `libggml-<name>-*.so` 并 dlopen。加载时查找两个符号:

- `ggml_backend_init` (必须) - 返回 `ggml_backend_reg_t`,由宏 `GGML_BACKEND_DL_IMPL(reg_fn)` 生成:
  ```c
  GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)
  // 展开为:
  GGML_BACKEND_API ggml_backend_reg_t ggml_backend_init(void) {
      return ggml_backend_cuda_reg();
  }
  ```
- `ggml_backend_score` (可选) - 返回 0 表示当前环境不可用、>0 表示可用性分数,用于"挑最好的后端"。

跨平台封装:Linux/macOS 用 `dlopen(RTLD_NOW|RTLD_LOCAL)` + `dlsym`,Windows 用 `LoadLibraryW` + `GetProcAddress`。

`ggml_backend_init_best()` 按优先级 GPU > IGPU > CPU 自动选最佳设备;`GGML_BACKEND_PATH` 环境变量可指定额外后端目录。

### 2.3 公开查询 API

> 文件: [ggml/include/ggml-backend.h](ggml/include/ggml-backend.h)

```c
size_t              ggml_backend_reg_count(void);
ggml_backend_reg_t   ggml_backend_reg_get(size_t i);
ggml_backend_reg_t   ggml_backend_reg_by_name(const char * name);
size_t              ggml_backend_dev_count(void);          // 扁平化所有设备数
ggml_backend_dev_t   ggml_backend_dev_get(size_t i);
ggml_backend_dev_t   ggml_backend_dev_by_name(const char * name);
ggml_backend_dev_t   ggml_backend_dev_by_type(enum ggml_backend_dev_type);
ggml_backend_t       ggml_backend_init_by_name(const char * name, const char * params);
ggml_backend_t       ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
ggml_backend_t       ggml_backend_init_best(void);
void                 ggml_backend_load_all(void);
```

llama.cpp 上层 ([src/llama-model-loader.cpp](src/llama-model-loader.cpp) 加载权重、[src/llama-context.cpp](src/llama-context.cpp) 构造调度器) 只依赖这套公开 API,完全不感知具体后端。

---

## 3. 调度器:如何把一张图分到多设备

> 文件: [ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp)

`ggml_backend_sched` 把 N 个后端组合起来执行一张图。核心是 `ggml_backend_sched_split_graph()` ([ggml-backend.cpp:1014-1240](ggml/src/ggml-backend.cpp#L1014)),分四趟扫描图节点:

| 趟 | 作用 |
|----|------|
| Pass 1 | 按已分配张量的缓冲位置定节点后端:预分配的目标/视图用其缓冲所属后端;图输入归最后(CPU);有权重的算子优先在权重所在后端跑,除非更高优先级后端 `offload_op()` 愿意抢 |
| Pass 2 | 向上向下扩展 GPU 后端覆盖范围,把相邻算子批在一起(减少跨设备拷贝) |
| Pass 3 | 把未分配节点交给"支持该算子且支持其输入缓冲类型"的后端;能升到更高优先级就升 |
| Pass 4 | 收尾:剩余源/后端按目标反推 |

判据函数:

- `ggml_backend_supports_op(backend, op)` -> 调 `device->iface.supports_op(device, op)` ([ggml-backend.cpp:455](ggml/src/ggml-backend.cpp#L455)) - **这是后端接入的关键回调**,决定每个算子落在哪。
- `ggml_backend_supports_buft(backend, buft)` - 数据在这类缓冲里时本后端能不能读。
- `ggml_backend_offload_op(backend, op)` - 是否愿意把密集算子卸载到本设备(即使输入还在主机)。

优先级模型:**注册顺序 = 优先级**(数组下标越小越高),最后一个是 CPU fallback。调度器在节点边界自动插入 `cpy_tensor_async` + `event` 同步,所以模型代码完全不用关心"这个 mul_mat 到底在 CPU 还是 GPU 跑"。

---

## 4. 各后端实现剖析 (统一的接入范式)

所有后端都遵循同一个五件套范式:**注册函数 + 设备 vtable + 流 vtable + 缓冲类型 + 图执行**。差异只在第五件套(图怎么算)和设备发现方式上。下表先给全景,再展开重点。

### 4.1 后端全景对照

| 后端 | 主文件 | 注册函数 | 设备发现 | 图执行方式 | 特色缓冲 |
|------|--------|---------|----------|-----------|---------|
| **CPU** | [ggml-cpu.cpp](ggml/src/ggml-cpu/ggml-cpu.cpp) | `ggml_backend_cpu_reg` | 固定 1 个 | plan + threadpool 多线程 | host buffer,可 mmap host ptr |
| **CUDA** | [ggml-cuda.cu](ggml/src/ggml-cuda/ggml-cuda.cu) | `ggml_backend_cuda_reg` | `cudaGetDeviceCount` | stream + CUDA Graph | device / **split** / pinned host |
| **HIP** | [ggml/src/ggml-hip/](ggml/src/ggml-hip/) | `ggml_backend_hip_reg` | hip 设备数 | 同 CUDA(移植) | 同 CUDA |
| **Metal** | [ggml-metal.cpp](ggml/src/ggml-metal/ggml-metal.cpp) | `ggml_backend_metal_reg` | MTLDevice | command buffer + encoder | shared / private / mapped |
| **Vulkan** | [ggml-vulkan.cpp](ggml/src/ggml-vulkan/ggml-vulkan.cpp) | `ggml_backend_vk_reg` | `vkEnumeratePhysicalDevices` | command buffer + descriptor set | device-local / host-visible |
| **SYCL** | [ggml-sycl.cpp](ggml/src/ggml-sycl/ggml-sycl.cpp) | `ggml_backend_sycl_reg` | `dpct::dev_mgr` | SYCL queue 提交 | device + host staging |
| **CANN** | [ggml-cann.cpp](ggml/src/ggml-cann/ggml-cann.cpp) | `ggml_backend_cann_reg` | `aclrtGetDeviceCount` | ACLNN 算子库 | ACL 设备缓冲 |
| **RPC** | [ggml-rpc.cpp](ggml/src/ggml-rpc/ggml-rpc.cpp) | `ggml_backend_rpc_reg` | 远程查询 | 序列化 cgraph 走 socket | 远程 opaque 指针 |
| 其他 | BLAS, OpenCL, OpenVINO, Hexagon, MUSA, WebGPU, zDNN, ZenDNN, VirtGPU | 同范式 | 各 SDK | 各 SDK | - |

### 4.2 CPU 后端 ([ggml/src/ggml-cpu/](ggml/src/ggml-cpu/))

- **设备 vtable**: [ggml-cpu.cpp:481-497](ggml/src/ggml-cpu/ggml-cpu.cpp#L481),`get_type` 返回 `GGML_BACKEND_DEVICE_TYPE_CPU`,`buffer_from_host_ptr` 支持 mmap(把 GGUF 权重直接映射进进程地址空间)。
- **`supports_op`**: [ggml-cpu.cpp:423-474](ggml/src/ggml-cpu/ggml-cpu.cpp#L423)。CPU 几乎支持所有算子,但对部分 i-quant 类型 (IQ1/IQ2/IQ3) 的 `CPY`/`SET_ROWS` 有限制,`MUL_MAT` 要求 src1 是 F32 或该量类型的 `vec_dot_type`。
- **图执行**: [ggml-cpu.cpp:170-191](ggml/src/ggml-cpu/ggml-cpu.cpp#L170) `ggml_backend_cpu_graph_compute` -> `ggml_graph_compute(cgraph, &cplan)` -> `ggml_compute_threadpool` 多线程按节点并行。这是 CPU 独有的 **plan 阶段** (`graph_plan_create/compute`),用于线程任务划分。
- **SIMD 内核**: 按 ISA 分目录 [ggml/src/ggml-cpu/arch/](ggml/src/ggml-cpu/arch/) (`x86/` AVX/AVX2/AVX-512/AMX, `arm/` NEON/SVE/SME, `riscv/` RVV, `wasm/`, `powerpc/`, `s390/`, `loongarch/`),主算子在 [ops.cpp](ggml/src/ggml-cpu/ops.cpp) (~389KB) 与 [vec.h](ggml/src/ggml-cpu/vec.h)。x86 特性检测靠 CPUID ([arch/x86/cpu-feats.cpp](ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp)),**运行时**按 CPU 支持的 ISA 选内核。
- **特点**: 无异步、无 pinned(CPU 自己就是主机);线程池可 pause/resume 供多上下文复用。

### 4.3 CUDA 后端 ([ggml/src/ggml-cuda/](ggml/src/ggml-cuda/))

- **初始化**: [ggml-cuda.cu](ggml/src/ggml-cuda/ggml-cuda.cu) `ggml_cuda_init()` ([~204-350 行](ggml/src/ggml-cuda/ggml-cuda.cu#L204)) - `cudaGetDeviceCount` + `cudaGetDeviceProperties`,记录每卡的 compute capability、显存、shared mem、SM 数、是否支持 cooperative launch 与虚拟内存管理。
- **设备 vtable**: [ggml-cuda.cu:5494-5510](ggml/src/ggml-cuda/ggml-cuda.cu#L5494)。`get_type` 按 `prop.integrated` 区分 GPU/IGPU;`get_host_buffer_type` 返回 pinned host 缓冲(`cudaHostRegister`,受 `GGML_CUDA_NO_PINNED` 控制);`offload_op` 按 batch size 决定是否把 matmul 卸载到 GPU。
- **`supports_op`**: [ggml-cuda.cu:5050+](ggml/src/ggml-cuda/ggml-cuda.cu#L5050)。CUDA 是"宽进"策略:支持 F32/F16/BF16 及几乎所有量化类型 (Q1_0~Q8_0、Q2_K~Q8_K、IQ1/IQ2/IQ3/IQ4、MXFP4/NVFP4) 的 `MUL_MAT`;限制主要是 split buffer 仅用于 `MUL_MAT`、源张量须在同一设备、F16 输入要求 F16 输出。`FLASH_ATTN_EXT` 在此声明支持(快注意力内核 [fattn.cu](ggml/src/ggml-cuda/fattn.cu))。
- **图执行**: [ggml-cuda.cu:4464-4521](ggml/src/ggml-cuda/ggml-cuda.cu#L4464) `ggml_backend_cuda_graph_compute`。每个图节点 launch 一个 kernel;预热稳定后用 **CUDA Graph** (`cudaStreamBeginCapture/EndCapture`) 捕获整张图以减少 launch 开销;支持 async `set/get/cpy_tensor` + event 同步。
- **内核组织**: 每算子一个 `.cu` + `.cuh` ([mul-mat.cu](ggml/src/ggml-cuda/mul-mat.cu), [rope.cu](ggml/src/ggml-cuda/rope.cu), [fattn.cu](ggml/src/ggml-cuda/fattn.cu), [norm.cu](ggml/src/ggml-cuda/norm.cu), [gla.cu](ggml/src/ggml-cuda/gla.cu)/[wkv.cu](ggml/src/ggml-cuda/wkv.cu) SSM 算子 ...)。量化 matmul 用模板按 `ggml_type` + GPU arch 实例化 ([mmq.cu](ggml/src/ggml-cuda/mmq.cu) `mul_mat_q_case<TYPE>()`)。
- **特色缓冲 - 关键**:
  - **device buffer** ([ggml-cuda.cu:830](ggml/src/ggml-cuda/ggml-cuda.cu#L830)) - 标准显存。
  - **split buffer** ([ggml-cuda.cu:1438](ggml/src/ggml-cuda/ggml-cuda.cu#L1438)) - ★张量并行核心:把一个 2D 权重张量按行切成多段,分布在多卡上,`MUL_MAT` 时每卡算自己那段行再 allreduce ([allreduce.cu](ggml/src/ggml-cuda/allreduce.cu))。这是 llama.cpp **单模型多卡** 推理的基础,详见 [docs/multi-gpu.md](docs/multi-gpu.md)。
  - **host pinned buffer** ([ggml-cuda.cu:1529](ggml/src/ggml-cuda/ggml-cuda.cu#L1529)) - 锁页内存,异步 H2D/D2H。
- **多卡通信**: 通过 `get_proc_address` 导出 `ggml_backend_cuda_comm_init/free/allreduce_tensor` ([ggml-cuda.cu:5597](ggml/src/ggml-cuda/ggml-cuda.cu#L5597))。

### 4.4 Metal 后端 ([ggml/src/ggml-metal/](ggml/src/ggml-metal/), macOS/iOS)

- **注册**: [ggml-metal.cpp:908](ggml/src/ggml-metal/ggml-metal.cpp#L908) `ggml_backend_metal_reg`,设备数默认 1,可被 `GGML_METAL_DEVICES` 覆盖。
- **设备发现**: `MTLDevice` 枚举,查名/显存 ([ggml-metal-device.m](ggml/src/ggml-metal/ggml-metal-device.m))。
- **图执行**: `ggml_metal_graph_compute` ([ggml-metal-context.m:438](ggml/src/ggml-metal/ggml-metal-context.m#L438)) - 用 Metal command buffer + compute encoder 派发 kernel。
- **内核**: Metal shader 源码可运行时编译,也可通过 `GGML_METAL_EMBED_LIBRARY` 把预编译 `.metallib` 嵌入二进制。
- **缓冲**: 三类 - `shared` (CPU-GPU 共享内存,UMA 架构友好)、`private` (设备独占)、`mapped` (映射主机指针)。`offload_op` 在 batch 够大时把 matmul 卸到 GPU。
- **特点**: 苹果统一内存架构下,权重可零拷贝在 CPU/GPU 间共享。

### 4.5 Vulkan 后端 ([ggml/src/ggml-vulkan/](ggml/src/ggml-vulkan/))

- **注册**: [ggml-vulkan.cpp:17153](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L17153) `ggml_backend_vk_reg`,实例由 `ggml_vk_instance_init()` 建立。
- **设备发现**: `vkEnumeratePhysicalDevices`,按 queue family 选,检测扩展 (VK_KHR_storage_buffer_read_only 等) 与可移植性;区分独显/集显。
- **内核**: SPIR-V 着色器 **预编译**在 [ggml/src/ggml-vulkan/vulkan-shaders/](ggml/src/ggml-vulkan/vulkan-shaders/) (`*.comp`),由 `vulkan-shaders-gen.cpp` 生成 C 头 `ggml-vulkan-shaders.hpp` 编进二进制,运行时按字符串表把 op 映射到对应 shader。
- **图执行**: [ggml-vulkan.cpp:15479](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L15479) `ggml_backend_vk_graph_compute` - 每算子一个 command buffer + descriptor set,按 ~100MB 或 ~100 节点批量提交队列,可选 query pool 做性能采样。
- **缓冲**: device-local(计算) + host-visible staging(上传下载)。
- **意义**: 跨平台 GPU 后端,Windows/Linux/Android 上不依赖厂商 SDK 即可跑任意 Vulkan GPU。

### 4.6 SYCL / CANN / 其他加速器

- **SYCL** ([ggml-sycl.cpp:5691](ggml/src/ggml-sycl/ggml-sycl.cpp#L5691)): Intel oneAPI,经 `dpct::dev_mgr` 发现设备,SYCL queue 提交内核;**基本是 CUDA 的移植**,多卡支持同样存在。`GGML_OP_OFFLOAD_MIN_BATCH` 控制卸载阈值。
- **CANN** ([ggml-cann.cpp:2985](ggml/src/ggml-cann/ggml-cann.cpp#L2985)): 华为昇腾,`aclrtGetDeviceCount` 发现设备,`aclInit` 初始化,图执行走 ACLNN 算子库;不支持 `buffer_from_host_ptr`。
- **HIP** ([ggml/src/ggml-hip/](ggml/src/ggml-hip/)): AMD,是 CUDA 源码的 hip 化移植,接口与 CUDA 几乎一一对应。
- **OpenCL / OpenVINO / Hexagon / MUSA(摩尔线程) / WebGPU / zDNN(IBM z) / ZenDNN(AMD CPU)**: 同一范式,各自 SDK。
- **BLAS** ([ggml/src/ggml-blas/](ggml/src/ggml-blas/)): 把 matmul 卸到 BLAS 库 (OpenBLAS/Intel MKL),作为 CPU 加速器 (`ACCEL` 类型)。

### 4.7 RPC 后端 ([ggml/src/ggml-rpc/](ggml/src/ggml-rpc/)) - 网络代理后端

这是最特殊的一个 - 它本身不做计算,而是把所有接口 **代理** 到远端的 RPC server (由 [tools/rpc/rpc-server.cpp](tools/rpc/rpc-server.cpp) 提供,背后是任意真实后端如 CUDA/Metal/CPU)。

- **注册**: [ggml-rpc.cpp:1902](ggml/src/ggml-rpc/ggml-rpc.cpp#L1902) `ggml_backend_rpc_reg`,无内置设备,通过 `ggml_backend_rpc_add_server(endpoint)` 动态加远端。
- **协议**: 自定义二进制 RPC (`enum rpc_cmd`: `RPC_CMD_ALLOC_BUFFER`/`SET_TENSOR`/`GET_TENSOR`/`GRAPH_COMPUTE`/`GET_DEVICE_MEMORY` ...),张量序列化为 `struct rpc_tensor` ([ggml-rpc.cpp:35-51](ggml/src/ggml-rpc/ggml-rpc.cpp#L35))。大块传输 (>10MB) 走 hash 去重。
- **代理点**: `supports_op` 暂返回 true(可缓存远端回答);`graph_compute` 把整张 cgraph 的张量数据 + op 结构发到远端执行;缓冲分配返回远端 opaque 指针。无 event(不支持异步)。
- **用途**: 把分散的机器/加速器组合成一个虚拟多卡环境,或在不装 CUDA 的机器上借用远端 GPU。

---

## 5. 接入新硬件后端的步骤 (范式总结)

新增一个硬件后端 `XYZ`,只需:

1. 新建目录 `ggml/src/ggml-xyz/`,实现:
   - `ggml_backend_xyz_reg()` 返回 `ggml_backend_reg` (含 `get_name/get_device_count/get_device`)。
   - `ggml_backend_xyz_device_i` 静态 vtable (实现 `get_name/get_memory/get_type/supports_op/get_buffer_type/init_backend/...`)。
   - `ggml_backend_xyz_interface` 流 vtable (实现 `graph_compute`,可选 async/event/plan)。
   - 一个或多个 `ggml_backend_buffer_type_i` 实现 (设备内存 + 可选 pinned host)。
   - 图执行主体:把 `ggml_cgraph` 的每个节点翻译成本平台的 kernel/shader launch。
2. 末尾加 `GGML_BACKEND_DL_IMPL(ggml_backend_xyz_reg)`,使其既能静态链接又能 dlopen。
3. 在 [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp) 注册块加 `#ifdef GGML_USE_XYZ / register_backend(ggml_backend_xyz_reg());`。
4. 在 [ggml/src/CMakeLists.txt](ggml/src/CMakeLists.txt) 加 `ggml_add_backend(XYZ)` 并建子目录的 `CMakeLists.txt`。

完成后,所有上层代码 (llama 模型图、server、cli) 无需任何改动即可使用新硬件 - 它们只通过公开 API `ggml_backend_init_best()` / `ggml_backend_dev_*` 交互,调度器自动用新后端的 `supports_op` 决定算子放置。

---

## 6. CMake 编译开关

> 文件: [ggml/CMakeLists.txt](ggml/CMakeLists.txt), [ggml/src/CMakeLists.txt](ggml/src/CMakeLists.txt)

`ggml_add_backend(<NAME>)` 宏 ([ggml/src/CMakeLists.txt:312-323](ggml/src/CMakeLists.txt#L312)) 统一处理:

```cmake
function(ggml_add_backend backend)
    string(TOUPPER "GGML_${backend}" backend_id)
    if (${backend_id})
        add_subdirectory(ggml-${backend_lower})
        if (NOT GGML_BACKEND_DL)
            target_compile_definitions(ggml PUBLIC GGML_USE_${backend})  # 静态:打上 GGML_USE_* 宏
        endif()
    endif()
endfunction()

ggml_add_backend(BLAS)  ggml_add_backend(CANN)  ggml_add_backend(CUDA)
ggml_add_backend(HIP)    ggml_add_backend(METAL) ggml_add_backend(MUSA)
ggml_add_backend(RPC)    ggml_add_backend(VirtGPU) ggml_add_backend(SYCL)
ggml_add_backend(Vulkan) ggml_add_backend(WebGPU)  ggml_add_backend(zDNN)
ggml_add_backend(OpenCL) ggml_add_backend(Hexagon) ggml_add_backend(ZenDNN)
ggml_add_backend(OPENVINO)
```

两种链接模式:
- **静态 (默认)**: 后端直接链进 `libggml`,编译期 `#define GGML_USE_<NAME>` 决定哪些后端在场。
- **动态 (`GGML_BACKEND_DL=ON`)**: 每个后端编成独立 MODULE 库,运行时按 `libggml-<name>-*.so` 模式 dlopen - 适合发版一个二进制、运行时按硬件挑后端。

主要开关默认值: `GGML_METAL`(macOS 默认 ON)、`GGML_VULKAN`(默认 OFF)、`GGML_SYCL`/`GGML_CANN`/`GGML_RPC`(默认 OFF)、CPU 始终 ON。

---

## 7. 端到端调用链 (后端如何被用起来)

以一次 `llama_decode()` 为例,追踪后端如何参与:

```
llama_decode(batch)                                  [src/llama-context.cpp]
  └─ graph_compute(cgraph)                           构建好的 ggml_cgraph
       └─ ggml_backend_sched_graph_compute(sched, gf)  [ggml/src/ggml-backend.cpp]
            ├─ ggml_backend_sched_split_graph()         四趟扫描,按 supports_op 给每个节点定后端
            ├─ 对每个 split 子图: ggml_backend_graph_compute(backend[i], subgf)
            │      └─ 调用 device->iface.init_backend 创建的流
            │         -> backend->iface.graph_compute   (CPU: threadpool / CUDA: kernel launch / Metal: command buffer ...)
            └─ 边界自动: ggml_backend_cpy_tensor_async + event_record/wait  跨设备搬运
```

权重加载侧 ([src/llama-model-loader.cpp](src/llama-model-loader.cpp) + [src/llama-context.cpp](src/llama-context.cpp) `sched_reserve`):每个张量按其 `ggml_backend_buffer_type` 落到对应设备的缓冲里。CPU mmap 直接零拷贝;GPU 则在首次访问时异步 H2D 上传;多卡时按 split 比例把权重切到各卡 split buffer。

---

## 8. 关键设计要点回顾

| 设计点 | 机制 | 价值 |
|--------|------|------|
| 后端无关 | 四层 vtable 契约 + `ggml_backend_reg` 登记 | 同一份模型图跑任意后端 |
| 可插拔 | 静态 `#ifdef` 注册 + 运行时 dlopen | 一份二进制适配多硬件 |
| 自动分派 | `ggml_backend_sched` 按 `supports_op` 切图 | CPU+GPU 混合、多卡无需改模型代码 |
| 多卡张量并行 | CUDA split buffer + allreduce | 大模型跨多 GPU |
| 零拷贝 | CPU mmap + Metal 统一内存 | 省显存、启动快 |
| 异步 | `set/get_tensor_async` + event | 计算与传输重叠 |
| 扩展性 | `get_proc_address` 导出后端特有函数 | 如 CUDA 多卡通信 |

---

## 附:后端相关文件速查

| 关注点 | 文件 |
|--------|------|
| 接口契约 (vtable) | [ggml/src/ggml-backend-impl.h](ggml/src/ggml-backend-impl.h) |
| 公开后端 API | [ggml/include/ggml-backend.h](ggml/include/ggml-backend.h) |
| 注册表 + 动态加载 | [ggml/src/ggml-backend-reg.cpp](ggml/src/ggml-backend-reg.cpp), [ggml/src/ggml-backend-dl.cpp](ggml/src/ggml-backend-dl.cpp) |
| 调度器 (多设备切图) | [ggml/src/ggml-backend.cpp](ggml/src/ggml-backend.cpp) |
| CPU 后端 | [ggml/src/ggml-cpu/](ggml/src/ggml-cpu/) |
| CUDA 后端 | [ggml/src/ggml-cuda/](ggml/src/ggml-cuda/) |
| Metal 后端 | [ggml/src/ggml-metal/](ggml/src/ggml-metal/) |
| Vulkan 后端 | [ggml/src/ggml-vulkan/](ggml/src/ggml-vulkan/) |
| SYCL / CANN / HIP | [ggml/src/ggml-sycl/](ggml/src/ggml-sycl/), [ggml/src/ggml-cann/](ggml/src/ggml-cann/), [ggml/src/ggml-hip/](ggml/src/ggml-hip/) |
| RPC 后端 | [ggml/src/ggml-rpc/](ggml/src/ggml-rpc/), [tools/rpc/rpc-server.cpp](tools/rpc/rpc-server.cpp) |
| 编译开关 | [ggml/CMakeLists.txt](ggml/CMakeLists.txt), [ggml/src/CMakeLists.txt](ggml/src/CMakeLists.txt) |
| 多卡使用指南 | [docs/multi-gpu.md](docs/multi-gpu.md) |
