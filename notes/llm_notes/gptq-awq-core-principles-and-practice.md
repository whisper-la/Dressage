# GPTQ 与 AWQ：核心原理及工程实践

> 面试导向速查文档  
> 更新日期：2026-07-28

## 1. 先用一句话说清楚

- **GPTQ**：利用校准数据估计权重之间的二阶相关性，逐个量化权重，并把当前量化误差补偿到尚未量化的权重上。
- **AWQ**：利用激活值判断哪些输入通道更重要，通过等价缩放保护重要通道，再对全部权重做低比特量化。

二者通常都属于：

- **PTQ（Post-Training Quantization，训练后量化）**：不重新完整训练模型，只需少量校准数据。
- **Weight-only Quantization（仅权重量化）**：常见配置是 **W4A16**，即权重 INT4，激活仍用 FP16/BF16。
- 目标是减少模型权重显存和内存带宽压力。理论上，FP16 权重变成 INT4 后，权重主体约缩小到原来的 1/4；实际还需保存 scale、zero-point 等元数据。

---

## 2. 必须掌握的量化基础

### 2.1 线性层与量化目标

线性层可以写成：

$$
Y=XW
$$

权重量化后：

$$
\hat{Y}=X\hat{W}
$$

真正需要控制的不是单纯的权重误差 $\|W-\hat{W}\|$，而是输出误差：

$$
\|XW-X\hat{W}\|_2^2
$$

这也是 GPTQ 和 AWQ 都需要校准数据的原因：同样大小的权重误差，出现在高频、强激活的通道上，比出现在几乎不被使用的通道上更危险。

### 2.2 均匀量化

将浮点数 $w$ 映射到 $b$ 位整数：

$$
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{w}{s}\right)+z,\ q_{\min},q_{\max}\right)
$$

反量化为：

$$
\hat{w}=s(q-z)
$$

其中：

- $s$：scale，决定量化间隔。
- $z$：zero-point；对称量化时通常为 0，非对称量化时可移动整数范围。
- `group_size`：多少个权重共享一组 scale/zero-point。常见值是 128。

`group_size` 越小，量化粒度越细，通常精度越好，但元数据和访存开销更大。`group_size` 越大，压缩和内核实现更简单，但不同分布的权重被迫共享参数，误差可能增大。

### 2.3 W4A16 不等于真正的 INT4 矩阵乘

工程上常见的 W4A16 内核会：

1. 以 INT4 形式存储权重；
2. 在计算过程中分块解包、反量化到 FP16/BF16；
3. 与 FP16/BF16 激活做融合计算。

因此，INT4 模型通常能明显省显存，但**不保证一定比 FP16 更快**。是否加速取决于 GPU、batch size、序列长度、量化格式、内核和服务框架。量化降低的是权重带宽；解包、反量化和不匹配的内核也会引入成本。

---

## 3. GPTQ

### 3.1 核心思想

GPTQ 来源于二阶量化思想。它把每个线性层的权重量化看成一个局部重构问题：

$$
\min_{\hat{W}}\ \|XW-X\hat{W}\|_2^2
$$

令 $\Delta W=W-\hat{W}$，则：

$$
\|X\Delta W\|_2^2
=
\operatorname{tr}\left(\Delta W^T X^TX\Delta W\right)
$$

所以输入激活的二阶统计量可作为 Hessian 的近似：

$$
H \approx 2X^TX
$$

具体实现中，$X$ 的存储方向可能不同，也会看到 $H=2XX^T$。关键不是转置写在哪边，而是：**Hessian 近似来自校准激活的协方差，描述不同输入通道的相关性和敏感度。**

### 3.2 为什么需要误差补偿

假设直接把某个权重 $w_q$ 量化为 $\hat{w}_q$，会产生误差。GPTQ 不接受这个误差原样传播，而是修改其余尚未量化的权重，使线性层输出尽量保持不变。

一种常见的简化表达为：

$$
\delta_q=
\frac{w_q-\hat{w}_q}{[H^{-1}]_{qq}}
$$

$$
W_{\text{remain}}
\leftarrow
W_{\text{remain}}
-
\delta_q[H^{-1}]_{q,\text{remain}}
$$

不同代码中的符号和索引写法会略有差异，但面试时要说清楚这个直觉：

> 当前权重舍入后产生的输出误差，会按照逆 Hessian 给出的相关性，传播并补偿到后续尚未量化的权重中。

这比逐元素 round-to-nearest 更精细，因为它考虑了输入通道之间的相关性。

### 3.3 算法流程

以一个 Transformer 线性层为例：

1. 用少量代表性文本跑前向，缓存该层输入激活 $X$。
2. 根据激活估计 Hessian 近似 $H$，并加入阻尼项保证数值稳定：

   $$
   H \leftarrow H+\lambda I
   $$

3. 对 $H^{-1}$ 做 Cholesky 等数值分解，便于高效、稳定地计算误差更新。
4. 按列或按分组依次量化权重。
5. 每量化一部分权重，就利用逆 Hessian 更新剩余浮点权重。
6. 采用 blockwise/lazy batch update 批量更新，避免逐元素更新造成极高开销。
7. 保存 INT4 权重、scale、zero-point、分组信息和格式元数据。

伪代码：

```text
for layer in transformer_layers:
    X = collect_calibration_activations(layer)
    H = estimate_second_order_statistics(X)
    H = H + damp * I
    H_inv_factor = stable_factorization(H)

    for block in split_columns(layer.weight):
        for column in block:
            q = quantize(column, scale, zero_point)
            error = normalized_quantization_error(column, q, H_inv_factor)
            compensate(unquantized_columns, error, H_inv_factor)
        apply_batched_update_to_remaining_blocks()

    save_packed_weights_and_metadata()
```

### 3.4 关键参数怎么解释

| 参数 | 作用 | 常见选择与影响 |
|---|---|---|
| `bits` | 权重位宽 | 最常见为 4；3/2 bit 更省空间但更难保精度、内核支持也更有限 |
| `group_size` | 每组共享量化参数的权重数 | 128 是常见起点；更小通常更准但开销更高 |
| `damp_percent` | 给 Hessian 对角线加阻尼 | 防止病态矩阵或数值不稳定；过大也会削弱二阶信息 |
| `desc_act` / act-order | 优先量化高激活或高敏感度通道 | 可能提高精度，但会改变权重排列并增加运行时复杂度 |
| `sym` | 是否对称量化 | 对称格式简单；具体精度和内核支持要实测 |
| `true_sequential` | 是否按模型真实依赖顺序逐层处理 | 通常更贴近误差传播，但校准更慢 |
| `calibration dataset` | 估计激活统计的数据 | 应覆盖真实请求的语言、长度、领域和提示模板 |

### 3.5 GPTQ 的优缺点

优点：

- 二阶误差补偿扎实，在 W4A16 场景中通常能获得较好的精度。
- 生态成熟，存在多种 GPTQ 权重格式和优化内核。
- 不需要完整训练，校准成本远低于 QAT。

缺点：

- 需要构造、分解 Hessian 近似，量化时间和内存开销通常高于简单 RTN。
- 实现和格式较多，例如 GPTQ、GPTQ v2、Marlin 兼容格式等；“都是 INT4”不代表可以被同一内核直接加载。
- 低位宽、异常值明显或校准集不匹配时，仍可能出现明显精度下降。

### 3.6 GPTQ 工程实践

#### 校准阶段

建议起点：

- 选取数百到数千条真实或近似真实请求。
- 覆盖线上主要语言、领域、prompt 模板和长度分布。
- 不要只用随机 token，也不要只用与生产完全无关的通用语料。
- 在固定版本中记录模型 revision、tokenizer、校准数据摘要、随机种子和量化参数。

Transformers 当前推荐通过 GPT-QModel 生态使用 GPTQ；AutoGPTQ 已不再由当前 Transformers 集成支持。下面的代码只表达典型配置，具体 API 应以锁定版本的文档为准：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "your-fp16-or-bf16-model"
tokenizer = AutoTokenizer.from_pretrained(model_id)

calibration_texts = [
    "与生产请求分布一致的示例一",
    "与生产请求分布一致的示例二",
]

quant_config = GPTQConfig(
    bits=4,
    dataset=calibration_texts,
    tokenizer=tokenizer,
    group_size=128,
    damp_percent=0.1,
    desc_act=False,
    sym=True,
    true_sequential=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quant_config,
)
model.save_pretrained("./model-gptq-w4-g128")
tokenizer.save_pretrained("./model-gptq-w4-g128")
```

#### 部署阶段

上线前依次确认：

1. 服务框架是否支持该模型架构。
2. 推理内核是否支持该 GPTQ 格式、位宽、group size、对称/非对称方式和 act-order。
3. GPU 架构是否匹配内核，例如 Marlin 等内核有自己的硬件与格式要求。
4. 单卡、张量并行、LoRA、长上下文、CUDA Graph 等组合是否经过实际测试。
5. 不只测 perplexity，还要测业务集质量、首 token 延迟、输出 token 延迟、吞吐、峰值显存和稳定性。

---

## 4. AWQ

### 4.1 核心观察

AWQ 的关键观察是：

> 权重的重要性不能只由权重绝对值判断，还要看与它相乘的激活有多大；少量高激活输入通道对输出质量尤其重要。

原论文发现，保护大约 1% 的显著权重可以显著降低量化误差。但这里有一个常见误解：

> AWQ 并不是简单地把这 1% 权重永久保留为 FP16、其余变成 INT4。它主要通过通道缩放降低重要权重的相对量化误差，最终仍可把全部权重存成规则的低比特格式，便于高效内核执行。

### 4.2 等价缩放为什么有效

设 $S=\operatorname{diag}(s)$ 是输入通道的缩放矩阵，则：

$$
XW=(XS^{-1})(SW)
$$

这是一种函数等价变换：一个输入通道在激活侧除以 $s_j$，相应权重通道乘以 $s_j$，浮点输出不变。

量化之后情况不同。若重要通道的权重先被放大，它们在共享量化网格中的相对舍入误差会减小：

$$
\hat{W}=Q(SW)
$$

AWQ 的目标可以直观写成：

$$
s^*=
\arg\min_s
\left\|
XW-(XS^{-1})Q(SW)
\right\|_2^2
$$

实际实现通常根据通道激活幅度生成缩放候选，例如：

$$
s_j \propto
\left(\operatorname{mean}|X_j|\right)^\alpha
$$

再搜索 $\alpha$ 或相关缩放参数，使层或 block 的输出重构误差最小。有些工程实现还会同时搜索 clipping ratio，减少离群值把量化范围拉得过宽的问题。

### 4.3 算法流程

1. 用少量代表性文本运行模型，收集各线性层输入激活统计。
2. 根据通道激活幅度识别显著通道。
3. 搜索通道缩放系数，使浮点输出与伪量化输出的差异最小。
4. 可选地搜索裁剪阈值，降低权重离群值的影响。
5. 将缩放等价地融合进相邻算子，例如 LayerNorm 和线性层。
6. 按 group 将全部权重量化并打包成 INT4。
7. 用目标推理内核验证数值一致性和真实性能。

伪代码：

```text
for block in transformer_blocks:
    activations = collect_calibration_activations(block)

    for linear_group in related_linear_layers(block):
        importance = channel_activation_statistics(activations)

        best_scale = search_scale(
            candidates_from(importance),
            objective=output_reconstruction_error
        )
        best_clip = optional_search_clipping_threshold()

        fuse_equivalent_scale_into_adjacent_ops(best_scale)
        quantize_and_pack_all_weights(bits=4, group_size=128)
```

### 4.4 AWQ 的优缺点

优点：

- 不需要 Hessian 求逆或反向传播，搜索和校准流程通常比 GPTQ轻。
- 直接利用激活统计判断通道重要性，对低比特 weight-only 量化有效。
- 全部权重仍可采用规则的低比特存储，适合融合内核和边缘设备。

缺点：

- 效果依赖校准数据是否能暴露真实的重要通道。
- 算子融合、权重布局、缩放吸收位置与模型结构有关，新模型架构可能需要额外适配。
- “AWQ 权重”仍有具体序列化格式和内核差异；不能只看模型名后缀就假设任意框架都能高效运行。

### 4.5 AWQ 工程实践

AWQ 的工程流程最好拆成四步：

```text
浮点模型
  → 校准并搜索 scale/clip
  → 生成并保存 W4 权重
  → 用目标运行时加载、压测和验收
```

加载一个已经按兼容格式保存的 AWQ 模型时，Transformers 通常能从 `config.json` 中的 `quantization_config` 识别量化方法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "your-awq-model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)
```

模型配置通常需要明确记录：

```json
{
  "quant_method": "awq",
  "bits": 4,
  "group_size": 128,
  "zero_point": true,
  "version": "目标运行时支持的权重布局"
}
```

注意：

- `version` 往往描述 GEMM、GEMV、ExLlama 或 Marlin 等具体布局/内核约定，而不是 AWQ 数学原理。
- 量化工具、Transformers 和推理运行时更新速度很快。AutoAWQ 可能固定或降级 Transformers 版本；生产中应使用隔离环境并锁定完整依赖。
- 如果目标是 vLLM 或 TensorRT-LLM，先从其当前版本支持矩阵反推量化格式，不要量化完成后才寻找可用内核。

---

## 5. GPTQ 与 AWQ 对比

| 维度 | GPTQ | AWQ |
|---|---|---|
| 核心依据 | 激活形成的二阶/Hessian 信息 | 激活幅度反映的通道重要性 |
| 核心动作 | 量化当前权重，并补偿剩余权重 | 搜索等价通道缩放，保护显著通道 |
| 是否需要校准集 | 需要 | 需要 |
| 是否反向传播 | 不需要 | 不需要 |
| 主要计算 | Hessian 近似、分解、逐步误差更新 | 激活统计、scale/clip 搜索 |
| 典型量化成本 | 通常较高 | 通常较低 |
| 常见配置 | W4A16，group size 128 | W4A16，group size 128 |
| 主要风险 | Hessian 数值稳定、量化耗时、格式兼容 | 校准覆盖不足、算子融合和结构适配 |
| 更适合的表述 | 二阶误差补偿型 PTQ | 激活感知的缩放型 PTQ |

不能脱离模型、任务和推理栈断言“GPTQ 一定比 AWQ 准”或“AWQ 一定更快”。实际结果还取决于：

- 模型结构和参数分布；
- 校准数据；
- 位宽、group size、是否对称、是否 act-order；
- GPU 架构和推理内核；
- batch size、输入/输出长度与并发；
- 权重格式转换是否损失了算法原本的布局和元数据。

---

## 6. 一套可落地的选型与验收流程

### 6.1 先定义目标

- 是显存放不下，还是吞吐不足？
- 目标是单请求低延迟，还是高并发吞吐？
- 能接受多少质量损失？
- 目标 GPU、CUDA、服务框架和模型结构是什么？

### 6.2 建立基线

至少保留三组结果：

1. FP16/BF16 原模型；
2. GPTQ W4A16；
3. AWQ W4A16。

不要只比较模型文件大小。

### 6.3 质量验收

- 通用指标：perplexity、MMLU 类知识题或项目既有评测。
- 业务指标：真实提示词、结构化输出成功率、代码执行率、检索问答正确率、工具调用成功率。
- 长上下文：不同输入长度下分别评测。
- 稳定性：重复生成、极端 prompt、空输入、最大长度和多语言样本。

### 6.4 性能验收

- 模型静态显存；
- 峰值显存；
- TTFT（Time To First Token）；
- TPOT（Time Per Output Token）或 ITL；
- 单请求 tokens/s；
- 不同并发下的总吞吐；
- P50/P95/P99 延迟；
- 启动和权重加载时间。

### 6.5 最小实验矩阵

| 变量 | 建议至少测试 |
|---|---|
| 算法 | GPTQ、AWQ |
| group size | 128；精度不足时再试 64 |
| 校准集 | 通用集、业务集或二者混合 |
| batch/并发 | 1、典型并发、压力并发 |
| 输入长度 | 短、典型、长 |
| 输出长度 | 短、典型、长 |
| 内核 | 目标框架默认内核、可用的优化内核 |

---

## 7. 常见工程坑

### 7.1 校准集“有数据”但不代表“有代表性”

如果线上是中文代码问答，却只用英文百科校准，GPTQ 估计的二阶统计和 AWQ 找到的重要通道都可能偏离实际分布。校准集不必很大，但必须像生产流量。

### 7.2 把量化算法、文件格式和推理内核当成同一件事

- GPTQ/AWQ：决定怎样生成低比特权重。
- checkpoint format：决定权重、scale、zero-point、排列方式怎样保存。
- Marlin、ExLlama、TinyChat 等 kernel/runtime：决定怎样执行。

算法相同，不代表格式相同；格式能读，不代表有高性能内核。

### 7.3 只看静态显存，不看 KV Cache

GPTQ/AWQ 主要压缩权重。长上下文或高并发场景中，KV Cache 可能成为主要显存开销。W4A16 不会自动把 KV Cache 也变成 4 bit。

### 7.4 认为 INT4 必然获得 4 倍吞吐

4 倍主要是权重主体的理论存储压缩比。端到端吞吐受反量化、计算单元利用率、调度、KV Cache、采样和通信影响。

### 7.5 量化完成后才考虑部署框架

正确顺序是：

```text
目标硬件与运行时
→ 支持的格式和内核
→ 量化参数
→ 生成权重
→ 质量与性能验收
```

---

## 8. 高频面试题与参考回答

### Q1：GPTQ 和 AWQ 的核心区别是什么？

GPTQ 使用校准激活构造 Hessian 近似，在逐步量化权重时，用逆 Hessian 将当前量化误差补偿到剩余权重；AWQ 使用激活统计判断重要输入通道，通过等价通道缩放降低显著权重的相对量化误差。前者是二阶误差补偿，后者是激活感知的缩放搜索。

### Q2：为什么二者都需要校准数据，却不属于训练？

校准数据只用于收集激活统计、估计敏感度或搜索量化参数，一般不计算训练损失、不做反向传播，也不更新全部模型参数，因此属于 PTQ，而不是 QAT 或微调。

### Q3：GPTQ 为什么要用 Hessian？

Hessian 近似描述线性层输出误差对权重扰动的敏感度及输入通道之间的相关性。量化一个权重后，GPTQ 可以利用逆 Hessian决定如何调整其余权重，把输出误差降到更低。

### Q4：GPTQ 的 dampening 是做什么的？

校准样本有限时，Hessian 近似可能奇异或条件数很差。给对角线加阻尼可以改善数值稳定性，使求逆或 Cholesky 分解更可靠。阻尼过大则会抹平真实的二阶结构。

### Q5：AWQ 所说的“保护 1% 重要权重”是不是混合精度？

原理观察是少量显著权重对质量非常关键，但常见 AWQ 做法不是把它们单独保留为 FP16。它通过通道缩放放大重要权重，降低它们在 INT4 网格中的相对误差，因此全部权重仍可规则地量化和打包。

### Q6：AWQ 为什么看激活，而不是只看权重大小？

线性层的贡献由激活和权重共同决定。某个权重本身不大，但如果对应输入通道经常出现很大的激活，它对输出仍可能非常重要。只看权重幅度会漏掉这种情况。

### Q7：为什么 group size 越小通常越准？

每个 group 独立估计量化范围。group 越小，组内权重分布更一致，少数离群值不容易拉大整组 scale；代价是 scale/zero-point 元数据更多，内核和访存开销可能上升。

### Q8：为什么量化后显存下降，速度却可能不升反降？

可能没有命中优化 INT4 内核，或者解包和反量化成本超过带宽收益；也可能瓶颈已经变成 KV Cache、attention、CPU 调度或多卡通信。因此必须在目标硬件、真实 batch 和长度分布上压测。

### Q9：GPTQ 和 AWQ 应该怎么选？

先由部署框架和硬件支持确定候选格式，再用同一校准集和业务评测对比质量与性能。一般可把 AWQ 作为校准较轻的候选，把 GPTQ 作为二阶重构能力较强的候选，但不能跳过实测。

### Q10：如何选择校准集？

重点不是盲目增加数量，而是覆盖生产分布：语言、领域、prompt 模板、输入长度、代码或数学内容以及工具调用格式。还应固定随机种子和数据版本，避免量化结果不可复现。

### Q11：GPTQ/AWQ 能解决长上下文的全部显存问题吗？

不能。它们主要降低模型权重显存。长上下文和高并发下，KV Cache 占用会快速上升，还需要分页 KV Cache、KV Cache 量化、并发调度等方案。

### Q12：线上量化方案如何验收？

先与 FP16/BF16 基线比较业务质量，再测 TTFT、TPOT、吞吐、P95/P99 延迟、峰值显存和稳定性；同时验证目标内核、并行方式、长上下文、LoRA 等组合。不能只用 perplexity 或模型文件大小做结论。

---

## 9. 30 秒回答模板

### GPTQ

> GPTQ 是一种训练后、仅权重量化方法。它用少量校准数据收集线性层输入，构造 Hessian 的近似来描述通道相关性。量化某个权重时，不只是简单舍入，还会利用逆 Hessian 把量化误差补偿到尚未量化的权重，所以在 4 bit 下通常能较好地保持输出。工程上重点关注校准集、group size、dampening、act-order，以及生成格式是否匹配 vLLM、Marlin 或其他目标内核。

### AWQ

> AWQ 也是训练后 weight-only 量化。它发现权重是否重要与对应激活强度有关，于是用校准激活找到显著输入通道，并通过等价缩放放大这些通道的权重，降低其相对量化误差。它通常不需要 Hessian 求逆或反向传播，校准更轻。工程上要关注 scale 和 clipping 搜索、group size、算子融合，以及 AWQ checkpoint 的具体布局是否被目标运行时高效支持。

### 二者对比

> GPTQ 的关键词是“二阶信息和误差补偿”，AWQ 的关键词是“激活感知和通道缩放”。二者都常用于 W4A16，也都需要代表性校准集。实际选型不能只看论文精度，还要从目标 GPU 和推理框架支持的内核出发，用同一业务集比较质量、显存、TTFT、TPOT 和吞吐。

---

## 10. 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [Hugging Face Transformers：GPTQ](https://huggingface.co/docs/transformers/quantization/gptq)
- [Hugging Face Transformers：AWQ](https://huggingface.co/docs/transformers/quantization/awq)
- [Hugging Face Transformers：Selecting a quantization method](https://huggingface.co/docs/transformers/main/quantization/selecting)
- [MIT Han Lab：llm-awq 官方实现](https://github.com/mit-han-lab/llm-awq)
- [vLLM：Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [NVIDIA TensorRT-LLM：Quantization](https://nvidia.github.io/TensorRT-LLM/features/quantization.html)

> 说明：量化库、checkpoint 格式和推理内核的兼容矩阵变化很快。本文的算法原理相对稳定；实际命令、依赖版本和硬件支持应以部署时的官方文档为准。
