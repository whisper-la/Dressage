# AWQ 量化算法：原理、数值推导与工程实现

> 本文系统介绍 Activation-aware Weight Quantization（AWQ）的动机、数学原理、通道缩放系数搜索、INT4 分组量化、缩放融合、权重裁剪、打包与推理实现。文中使用一个完整的 5 输入通道数值例子，逐步展示普通 INT4 量化为什么会失败，以及 AWQ 如何降低重要权重的有效量化误差。

## 1. AWQ 是什么

AWQ（Activation-aware Weight Quantization）是一类面向大语言模型的训练后权重量化方法。最常见的部署配置是 **W4A16**：

- 权重使用 INT4 存储和计算；
- 激活保持 FP16 或 BF16；
- 不需要重新训练模型，也不需要反向传播；
- 使用少量校准数据统计激活，并搜索量化参数；
- 通过专用 CUDA、Triton、Marlin 或 ExLlama 等内核执行量化推理。

AWQ 的核心观察是：

> 一个权重是否重要，不能只看权重自身的数值大小，还要看它所对应输入通道的激活幅值。即使某个权重很小，如果对应激活经常很大，它的量化误差仍然可能对输出造成很大影响。

因此，AWQ 的核心不是简单地“保留大权重”，而是：

1. 使用校准数据统计每个输入通道的激活幅值；
2. 根据激活统计构造逐通道缩放系数；
3. 放大高激活通道对应的权重；
4. 对缩放后的权重进行 INT4 量化；
5. 通过等价变换保持原始浮点计算不变；
6. 在校准集上搜索使输出重建误差最小的缩放方案。

AWQ 的代表性论文为 [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)，官方参考实现为 [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)。

---

## 2. 线性层、输入通道与权重列

Transformer 中绝大部分参数位于线性层。PyTorch 线性层写作：

```python
linear = torch.nn.Linear(in_features, out_features)
```

其权重形状为：

```text
weight.shape = [out_features, in_features]
```

给定输入：

$$
X\in\mathbb{R}^{B\times T\times C_{\text{in}}}
$$

以及权重：

$$
W\in\mathbb{R}^{C_{\text{out}}\times C_{\text{in}}}
$$

线性层计算为：

$$
Y=XW^T+b
$$

这里所说的第 $j$ 个“输入通道”，就是输入张量最后一维的第 $j$ 个特征：

$$
X[:,:,j]
$$

它对应权重矩阵的第 $j$ 列：

$$
W[:,j]
$$

因此，AWQ 对第 $j$ 个输入通道设置缩放系数 $s_j$ 时，需要同时处理：

$$
X[:,:,j]\leftarrow X[:,:,j]/s_j
$$

$$
W[:,j]\leftarrow W[:,j]s_j
$$

这里的“通道”不等于 Attention Head。Attention Head 是输出经过 reshape 后形成的逻辑分组；AWQ 输入通道是 Linear 输入张量最后一维中的单个特征。

### 2.1 不同线性层的通道含义

假设模型隐藏维度是 $d_{\text{model}}$，FFN 中间维度是 $d_{\text{ffn}}$：

| 线性层 | 典型权重形状 | AWQ 输入通道数 |
|---|---:|---:|
| `q_proj` | `[q_dim, d_model]` | `d_model` |
| `k_proj` | `[kv_dim, d_model]` | `d_model` |
| `v_proj` | `[kv_dim, d_model]` | `d_model` |
| `o_proj` | `[d_model, q_dim]` | `q_dim` |
| `gate_proj` | `[d_ffn, d_model]` | `d_model` |
| `up_proj` | `[d_ffn, d_model]` | `d_model` |
| `down_proj` | `[d_model, d_ffn]` | `d_ffn` |

采用 GQA 的模型中，Q、K、V 投影的输出维度可能不同，但只要它们共享同一个隐藏状态输入，其输入通道数仍然相同。

---

## 3. INT4 量化基础

### 3.1 对称 INT4 量化

为了使零点固定为 0，可以使用对称整数范围：

$$
q\in[-7,7]
$$

对一组浮点权重 $W_g$，先取最大绝对值：

$$
M_g=\max(|W_g|)
$$

将浮点范围对称化为：

$$
[-M_g,M_g]
$$

量化步长为：

$$
\Delta_g=\frac{M_g}{7}
$$

也可以写成：

$$
\Delta_g =\frac{M_g-(-M_g)}{7-(-7)} =\frac{2M_g}{14} =\frac{M_g}{7}
$$

所以“最大绝对值除以 7”和“对称最大值减最小值后除以 14”完全等价。

量化公式：

$$
q=\operatorname{clip} \left( \operatorname{round}\left(\frac{W_g}{\Delta_g}\right), -7,7 \right)
$$

反量化公式：

$$
\hat W_g=q\Delta_g
$$

一些实现使用 `[-8, 7]` 的有符号 INT4 存储范围。由于这个范围不是严格对称的，具体分母和裁剪边界会因实现而不同。本文为了让公式清楚，统一使用 `[-7, 7]`。

### 3.2 非对称 UINT4 量化

非对称量化常用整数范围：

$$
q\in[0,15]
$$

量化步长：

$$
\Delta_g=\frac{W_{\max,g}-W_{\min,g}}{15}
$$

零点：

$$
z_g=\operatorname{round} \left(-\frac{W_{\min,g}}{\Delta_g}\right)
$$

量化与反量化：

$$
q=\operatorname{clip} \left( \operatorname{round}(W_g/\Delta_g)+z_g, 0,15 \right)
$$

$$
\hat W_g=(q-z_g)\Delta_g
$$

AutoAWQ 中常见的 `zero_point=True` 配置对应带零点的非对称量化。

### 3.3 分组量化

大模型通常不会让整行权重共享同一个量化参数，而是沿权重矩阵最后一维分组。例如：

```python
q_group_size = 128
```

表示每 128 个连续权重共享一个量化步长和可选零点。对于：

```text
weight.shape = [4096, 4096]
```

每个输出行包含：

$$
4096/128=32
$$

个量化组。

需要区分两种 scale：

1. **AWQ 通道 scale $s_j$**：通常每个输入通道一个，通过校准数据和搜索获得；
2. **INT4 量化 scale $\Delta_g$**：每个量化 group 一个，根据该组权重范围计算。

两者作用不同，不能混为一谈。

---

## 4. 为什么权重大小不能代表重要性

忽略 bias 时，线性层第 $i$ 个输出为：

$$
y_i=\sum_j x_jw_{i,j}
$$

量化权重后：

$$
\hat y_i=\sum_j x_j\hat w_{i,j}
$$

因此输出误差为：

$$
\Delta y_i =\hat y_i-y_i =\sum_j x_j(\hat w_{i,j}-w_{i,j})
$$

令权重量化误差：

$$
e_{i,j}=\hat w_{i,j}-w_{i,j}
$$

则：

$$
\Delta y_i=\sum_jx_je_{i,j}
$$

即使某个权重的误差 $e_{i,j}$ 很小，只要对应输入通道 $|x_j|$ 很大，它对输出的影响仍然可能很大。这就是 AWQ 使用激活信息判断权重重要性的理论基础。

---

## 5. AWQ 的等价缩放

定义逐输入通道缩放向量：

$$
s=[s_0,s_1,\ldots,s_{C_{\text{in}}-1}],\qquad s_j>0
$$

以及对角矩阵：

$$
S=\operatorname{diag}(s)
$$

原始线性计算可以改写为：

$$
XW^T=(XS^{-1})(WS)^T
$$

逐通道写法为：

$$
X'_j=X_j/s_j
$$

$$
W'_{:,j}=W_{:,j}s_j
$$

浮点条件下：

$$
X'W'^T=XW^T
$$

两边的缩放严格抵消，不会改变模型函数。但是量化是非线性的，因此通常有：

$$
Q(WS)\neq Q(W)S
$$

AWQ 正是利用这一点：选择合适的 $S$，让缩放后的权重 $WS$ 更适合低比特量化。

### 5.1 为什么较大的 scale 能保护权重

设缩放后权重组的量化步长为 $\Delta_s$。缩放域中的舍入误差大致满足：

$$
|Q(w_js_j)-w_js_j|\leq\frac{\Delta_s}{2}
$$

映射回原始权重空间：

$$
\hat w_j=\frac{Q(w_js_j)}{s_j}
$$

因此有效权重误差上界为：

$$
|\hat w_j-w_j| \leq \frac{\Delta_s}{2s_j}
$$

在暂时假设 $\Delta_s$ 不变的情况下，$s_j$ 越大，该通道对应权重的有效量化误差越小。

第 $j$ 个通道造成的输出误差上界为：

$$
|\Delta y_j| \lesssim |x_j|\frac{\Delta_s}{2s_j}
$$

若令：

$$
s_j=x_j^\alpha
$$

则：

$$
|\Delta y_j| \lesssim \frac{\Delta_s}{2}|x_j|^{1-\alpha}
$$

当 $\alpha=0$ 时，没有缩放，误差影响近似随 $|x_j|$ 线性增长；当 $0<\alpha<1$ 时，高激活通道对误差的放大被部分抵消。

### 5.2 为什么 scale 不能无限增大

真实量化中，$\Delta_s$ 并非固定：

$$
\Delta_s=\frac{\max_j|w_js_j|}{7}
$$

如果某个通道的 $s_j$ 过大，$w_js_j$ 可能成为新的离群值，增大整个量化组的步长，从而伤害同组其他权重。因此，AWQ 需要在“保护高激活通道”和“控制整个 group 的动态范围”之间搜索平衡点。

---

## 6. AWQ scale 如何获得

AWQ scale 不是通过反向传播训练得到的，而是通过校准数据上的统计和网格搜索得到的。

### 6.1 收集输入激活

对模型运行少量校准文本，通过 forward hook 或模型包装器捕获目标线性层输入：

$$
X\in\mathbb{R}^{N\times C_{\text{in}}}
$$

这里 $N$ 合并了 batch 和 token 维度。

对每个输入通道计算绝对值均值：

$$
x_j=\frac{1}{N}\sum_{n=1}^{N}|X_{n,j}|
$$

PyTorch 形式：

```python
x_flat = input_feat.abs().reshape(-1, input_feat.shape[-1])
x_mean = x_flat.float().mean(dim=0)
```

如果线性层 `in_features=4096`，那么：

```text
x_mean.shape = [4096]
```

### 6.2 只使用激活统计的候选公式

简化的 AWQ 候选 scale 为：

$$
s_j(\alpha)=x_j^\alpha
$$

其中：

- $x_j$ 是第 $j$ 个通道的激活统计；
- $\alpha$ 是一个标量幂指数；
- $s_j$ 是最终作用于第 $j$ 个输入通道的缩放值。

注意：$\alpha$ 不是 scale 本身。一个候选 $\alpha$ 会生成一整条 scale 向量：

$$
\alpha \longrightarrow s(\alpha)=[x_0^\alpha,x_1^\alpha,\ldots,x_{C-1}^\alpha]
$$

### 6.3 AutoAWQ 的 duo scaling

AutoAWQ 的 `duo_scaling` 还会考虑权重幅值。其实现先在量化 group 内归一化权重：

$$
\bar W= \frac{|W|}{\max_{\text{group}}|W|+\epsilon}
$$

再沿输出通道计算每个输入通道的平均归一化权重幅值：

$$
w_j=\operatorname{mean}_i(\bar W_{i,j})
$$

候选 scale 近似为：

$$
s_j(\alpha) = \frac{x_j^\alpha}{w_j^{1-\alpha}+\epsilon}
$$

如果关闭 `duo_scaling`，则退化为：

$$
s_j(\alpha)=x_j^\alpha
$$

AutoAWQ 随后会归一化候选 scale：

$$
s\leftarrow \frac{s}{\sqrt{\max(s)\min(s)}}
$$

该操作保留各通道 scale 的相对比例，同时避免所有数值整体过大或过小。

### 6.4 网格搜索 alpha

AutoAWQ 的典型实现测试 20 个候选值：

$$
\alpha\in\{0,0.05,0.10,\ldots,0.95\}
$$

首先计算原始 FP16 模块输出：

$$
Y_{\text{ref}}=F(X;W)
$$

然后对每个候选 $\alpha$：

1. 生成 $s(\alpha)$；
2. 将相关线性层权重按列乘以 $s$；
3. 对 $Ws$ 执行 INT4 伪量化；
4. 计算候选量化输出；
5. 与 FP16 输出计算均方误差；
6. 恢复原始权重，测试下一个候选值。

目标函数为：

$$
\alpha^* = \arg\min_\alpha \left\| XW^T-(X/s_\alpha)Q(Ws_\alpha)^T \right\|_2^2
$$

AutoAWQ 实现中常使用融合后的等价形式：

$$
\left[\frac{Q(Ws)}{s}\right]X
$$

这样可以保持输入不变，在临时权重中模拟 `X/s` 的效果。

简化伪代码：

```python
@torch.no_grad()
def search_awq_scale(x, weight, x_mean, w_mean=None):
    reference = x @ weight.T

    best_error = float("inf")
    best_scale = None
    best_alpha = None

    for i in range(20):
        alpha = i / 20

        if w_mean is None:
            scale = x_mean.pow(alpha)
        else:
            scale = (
                x_mean.pow(alpha)
                / (w_mean.pow(1 - alpha) + 1e-4)
            )

        scale = scale.clamp(min=1e-4)
        scale = scale / torch.sqrt(scale.max() * scale.min())

        scaled_weight = weight * scale.view(1, -1)
        fake_quant_weight = pseudo_quantize_int4(scaled_weight)
        candidate = (x / scale) @ fake_quant_weight.T

        error = (candidate - reference).float().pow(2).mean()

        if error < best_error:
            best_error = error
            best_alpha = alpha
            best_scale = scale.clone()

    return best_alpha, best_scale
```

AutoAWQ 的实际搜索逻辑可参考其归档源码中的 [`quantizer.py`](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/quantizer.py)。

---

## 7. 完整 5 通道数值例子

下面使用一个具有 5 个输入通道、1 个输出通道的线性层，完整展示普通 INT4 与 AWQ INT4 的区别。为了便于手算，假设 5 个权重位于同一个对称 INT4 量化组中。

### 7.1 原始权重与输入

权重：

$$
W= \begin{bmatrix} 1 & 0.5 & 0.25 & 0.125 & 0.0625 \end{bmatrix}
$$

形状：

```text
W.shape = [1, 5]
```

假设校准统计得到的通道激活绝对值均值为：

$$
x_{\text{mean}} = [0.0625,0.25,1,4,16]
$$

为了便于展示，取一个与该统计相同的代表性输入：

$$
X=[0.0625,0.25,1,4,16]
$$

每个通道对原始输出的贡献如下：

| 通道 $j$ | 输入 $x_j$ | 权重 $w_j$ | 贡献 $x_jw_j$ |
|---:|---:|---:|---:|
| 0 | 0.0625 | 1.0000 | 0.0625 |
| 1 | 0.2500 | 0.5000 | 0.1250 |
| 2 | 1.0000 | 0.2500 | 0.2500 |
| 3 | 4.0000 | 0.1250 | 0.5000 |
| 4 | 16.0000 | 0.0625 | 1.0000 |

原始输出：

$$
Y=XW^T
$$

$$
Y=0.0625+0.125+0.25+0.5+1=1.9375
$$

第 4 个权重只有 0.0625，是绝对值最小的权重；但它对应的激活为 16，对输出贡献为 1，反而是最重要的通道。

### 7.2 普通对称 INT4 量化

最大绝对权重：

$$
M=1
$$

量化步长：

$$
\Delta=\frac{1}{7}\approx0.142857
$$

逐项量化：

| 通道 | 原始权重 | $w_j/\Delta$ | INT4 值 | 反量化权重 |
|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 7.000 | 7 | 1.0000 |
| 1 | 0.5000 | 3.500 | 4 | 0.5714 |
| 2 | 0.2500 | 1.750 | 2 | 0.2857 |
| 3 | 0.1250 | 0.875 | 1 | 0.1429 |
| 4 | 0.0625 | 0.438 | 0 | 0 |

反量化权重：

$$
\hat W=[1,0.5714,0.2857,0.1429,0]
$$

量化输出：

$$
\hat Y =0.0625\times1 +0.25\times0.5714 +1\times0.2857 +4\times0.1429 +16\times0
$$

$$
\hat Y\approx1.0625
$$

输出误差：

$$
\hat Y-Y=1.0625-1.9375=-0.875
$$

平方误差：

$$
L=0.875^2=0.765625
$$

误差最大的根源是：

$$
0.0625\rightarrow0
$$

它让输出直接损失：

$$
16\times0.0625=1
$$

### 7.3 构造候选 AWQ scale

使用简化公式：

$$
s=x_{\text{mean}}^\alpha
$$

选择候选：

$$
\alpha=0.5
$$

则：

$$
s= [\sqrt{0.0625},\sqrt{0.25},\sqrt1,\sqrt4,\sqrt{16}]
$$

$$
s=[0.25,0.5,1,2,4]
$$

激活越大的通道得到越大的相对 scale：

| 通道 | 激活统计 | AWQ scale |
|---:|---:|---:|
| 0 | 0.0625 | 0.25 |
| 1 | 0.25 | 0.5 |
| 2 | 1 | 1 |
| 3 | 4 | 2 |
| 4 | 16 | 4 |

### 7.4 缩放权重

按输入通道缩放权重：

$$
W'=W\odot s
$$

| 通道 | 原始权重 | AWQ scale | 缩放后权重 |
|---:|---:|---:|---:|
| 0 | 1.0000 | 0.25 | 0.25 |
| 1 | 0.5000 | 0.50 | 0.25 |
| 2 | 0.2500 | 1.00 | 0.25 |
| 3 | 0.1250 | 2.00 | 0.25 |
| 4 | 0.0625 | 4.00 | 0.25 |

因此：

$$
W'=[0.25,0.25,0.25,0.25,0.25]
$$

最小但重要的第 4 个权重被放大：

$$
0.0625\times4=0.25
$$

不再容易被 INT4 舍入为 0。

### 7.5 对输入反向缩放

为了保持浮点计算等价：

$$
X'=X/s
$$

| 通道 | 原始输入 | AWQ scale | 反向缩放输入 |
|---:|---:|---:|---:|
| 0 | 0.0625 | 0.25 | 0.25 |
| 1 | 0.25 | 0.50 | 0.50 |
| 2 | 1 | 1.00 | 1.00 |
| 3 | 4 | 2.00 | 2.00 |
| 4 | 16 | 4.00 | 4.00 |

所以：

$$
X'=[0.25,0.5,1,2,4]
$$

验证浮点等价性：

$$
X'W'^T =0.25\times0.25 +0.5\times0.25 +1\times0.25 +2\times0.25 +4\times0.25
$$

$$
X'W'^T=0.0625+0.125+0.25+0.5+1=1.9375
$$

与原始输出完全相同。

### 7.6 量化缩放后的权重

缩放后权重的最大绝对值为 0.25，因此新的量化步长为：

$$
\Delta'=\frac{0.25}{7}\approx0.035714
$$

每个权重的量化整数均为：

$$
q'_j=\operatorname{round}(0.25/0.035714)=7
$$

所以：

$$
Q(W')=[7,7,7,7,7]
$$

反量化后：

$$
\hat W'=[0.25,0.25,0.25,0.25,0.25]
$$

量化输出：

$$
\hat Y_{\text{AWQ}}=X'\hat W'^T=1.9375
$$

平方误差：

$$
L_{\text{AWQ}}=0
$$

结果对比：

| 方法 | 反量化权重 | 输出 | 平方误差 |
|---|---|---:|---:|
| 普通 INT4 | `[1, 0.5714, 0.2857, 0.1429, 0]` | 1.0625 | 0.765625 |
| AWQ INT4 | `[0.25, 0.25, 0.25, 0.25, 0.25]` | 1.9375 | 0 |

这是为了说明机制而构造的理想化例子。真实模型中通常无法达到零误差，但 AWQ 仍能通过相同机制降低重要通道对应权重的有效误差。

### 7.7 alpha 的搜索结果示意

对于同一组激活统计，不同 alpha 会产生不同 scale 和量化误差：

| $\alpha$ | 通道 scale $x^\alpha$ | 代表性输出 | 平方误差 |
|---:|---|---:|---:|
| 0 | `[1, 1, 1, 1, 1]` | 1.0625 | 0.765625 |
| 0.25 | `[0.5, 0.7071, 1, 1.4142, 2]` | 约 2.0214 | 约 0.0070 |
| 0.50 | `[0.25, 0.5, 1, 2, 4]` | 1.9375 | 0 |
| 0.75 | `[0.125, 0.3536, 1, 2.8284, 8]` | 约 1.9632 | 约 0.0007 |
| 1.00 | `[0.0625, 0.25, 1, 4, 16]` | 约 2.0000 | 约 0.0039 |

本例中 $\alpha=0.5$ 最优。真实 AutoAWQ 会在多条校准样本和更大的模块输出上计算 MSE，而不是只比较一个标量输出。

---

## 8. 搜索单位：不是每个权重，也不一定是每个 Linear

AutoAWQ 通常逐个 Transformer block 处理，但 scale 搜索的单位是“可以共享同一输入缩放的算子组”。

例如：

```text
RMSNorm
   ├── q_proj
   ├── k_proj
   └── v_proj
```

Q、K、V 共享同一个 RMSNorm 输出。如果将该输出改为 $X/s$，那么三个权重都必须使用同一组输入通道 scale：

$$
W_q'=W_qS,\quad W_k'=W_kS,\quad W_v'=W_vS
$$

类似地：

```text
RMSNorm
   ├── gate_proj
   └── up_proj
```

`gate_proj` 和 `up_proj` 通常也需要联合处理。

整体循环结构大致是：

```python
for decoder_block in model.layers:
    input_features = collect_inputs(decoder_block)
    scaling_groups = get_scaling_groups(decoder_block)

    for group in scaling_groups:
        best_scale = search_best_scale(
            previous_op=group.previous_op,
            linears=group.linears,
            input_features=group.input,
        )
        apply_scale(group, best_scale)
```

搜索内部再遍历 alpha：

```text
遍历 Transformer block
        ↓
遍历 block 内的缩放关系组
        ↓
遍历约 20 个 alpha 候选
```

因此它不是给每个权重搜索一次，也不一定给每个 Linear 独立搜索一次。

---

## 9. scale 融合

搜索阶段可以显式计算：

```python
x_scaled = x / scale
w_scaled = weight * scale
```

但如果推理时额外执行 `x / scale`，会增加一个逐元素算子和额外显存读写。AWQ 通常把该操作融合进前置算子。

### 9.1 LayerNorm/RMSNorm 到 Linear

设归一化层输出：

$$
x=\gamma\odot\hat x+\beta
$$

为了让输出变成：

$$
x'=x/s
$$

可以修改：

$$
\gamma'=\gamma/s
$$

如果存在 bias：

$$
\beta'=\beta/s
$$

同时把后续线性层权重按输入通道放大：

$$
W'=WS
$$

PyTorch 示意：

```python
@torch.no_grad()
def fuse_norm_linear(norm, linears, scales):
    norm.weight.div_(scales)

    if getattr(norm, "bias", None) is not None:
        norm.bias.div_(scales)

    for linear in linears:
        # Linear 权重形状为 [out_features, in_features]
        linear.weight.mul_(scales.view(1, -1))
```

### 9.2 Linear 到 Linear

设前一线性层输出：

$$
x=W_1h+b_1
$$

希望后续线性层收到 $x/s$，可以把前一层的输出通道除以 scale：

$$
W'_{1,j,:}=W_{1,j,:}/s_j
$$

$$
b'_{1,j}=b_{1,j}/s_j
$$

并把后一层的输入通道乘以 scale：

$$
W'_{2,:,j}=W_{2,:,j}s_j
$$

代码示意：

```python
@torch.no_grad()
def fuse_linear_linear(previous, following, scales):
    previous.weight.div_(scales.view(-1, 1))

    if previous.bias is not None:
        previous.bias.div_(scales)

    for linear in following:
        linear.weight.mul_(scales.view(1, -1))
```

### 9.3 不能盲目融合的情况

融合必须保持计算图等价，需要注意：

- 前一算子的输出是否同时进入残差分支；
- 是否有多个后续消费者共享同一个张量；
- 中间是否存在不能与缩放交换的非线性函数；
- bias 是否同时正确缩放；
- 当前 scale 对应输入通道还是输出通道；
- Q、K、V 等共享输入的层是否同时补偿。

如果一个张量还被未经补偿的残差或其他分支使用，直接修改它会改变模型函数。实际 AWQ 实现会针对不同模型结构维护可缩放算子关系。

---

## 10. 权重裁剪搜索

AWQ 实现通常还包含权重 clipping。其目的与通道 scale 不同：

- 通道 scale：利用激活统计保护重要通道；
- clipping：降低少量权重离群值对量化步长的影响。

普通对称量化使用：

$$
\Delta_g=\frac{\max(|W_g|)}{7}
$$

如果组内只有一个极端离群值，$\Delta_g$ 会被拉大，导致其余权重使用的量化级别过少。可以搜索裁剪比例 $r$：

$$
M'_g=rM_g
$$

$$
W'_g=\operatorname{clip}(W_g,-M'_g,M'_g)
$$

$$
\Delta'_g=\frac{M'_g}{7}
$$

然后使用校准输入比较：

$$
\left\|XW_g^T-XQ(W'_g)^T\right\|_2^2
$$

选择输出误差最小的裁剪范围。

裁剪过少无法消除离群值影响；裁剪过多会直接破坏大权重。因此同样需要搜索。

---

## 11. 从校准到量化模型的完整流程

```text
加载 FP16/BF16 模型
        ↓
准备少量代表性校准文本
        ↓
逐个 Transformer block 前向
        ↓
通过 hook 捕获各 Linear 输入激活
        ↓
构造模型结构对应的缩放算子组
        ↓
统计每个输入通道的 mean(|X|)
        ↓
可选：计算归一化权重通道统计
        ↓
网格搜索 alpha，生成候选通道 scale
        ↓
伪量化候选权重并计算模块输出 MSE
        ↓
选择最优 AWQ scale
        ↓
将 scale 融合进 Norm/前置 Linear 与目标 Linear
        ↓
搜索可选的权重 clipping 范围
        ↓
按 group 执行真正的 INT4 量化
        ↓
打包 qweight、qzeros、quant scales
        ↓
替换 nn.Linear 为量化 Linear 模块
        ↓
使用 CUDA/Triton 等 INT4 kernel 推理
```

---

## 12. PyTorch、Triton 和 CUDA 分别负责什么

AWQ 的离线量化过程主要由 Python/PyTorch 实现；Triton/CUDA 主要负责量化模型的高性能推理。

| 阶段 | 典型实现 |
|---|---|
| 模型加载与校准前向 | Transformers + PyTorch |
| 激活统计 | PyTorch Tensor 操作 |
| scale 网格搜索 | Python 循环 + PyTorch |
| INT4 伪量化 | PyTorch |
| 输出重建误差 | PyTorch |
| scale 融合 | Python + PyTorch 参数修改 |
| 权重转换与打包 | PyTorch 位运算或扩展代码 |
| INT4 GEMM/GEMV | CUDA、Triton、Marlin、ExLlama 等 |

需要注意，“用 PyTorch 实现”和“底层使用 CUDA”并不矛盾。如果 PyTorch Tensor 位于 GPU，`torch.matmul` 等操作仍然会调用 CUDA/cuBLAS 内核。

AWQ 专用推理内核的价值在于融合以下过程：

```text
读取打包 INT4 权重
        ↓
在寄存器或共享内存中解包
        ↓
应用 group scale 和 zero point
        ↓
与 FP16/BF16 激活相乘
        ↓
FP16/FP32 累加并输出
```

如果先把整个 INT4 权重反量化成 FP16 再调用普通矩阵乘法，会产生较大的显存流量，也会削弱低比特权重的性能优势。

AutoAWQ 的量化 Linear 会优先尝试编译好的 CUDA 扩展；不可用时可使用 Triton；再不可用时回退到先反量化、再调用 PyTorch 矩阵乘法的慢速路径。可参考其 [`WQLinear_GEMM`](https://github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py) 实现。

---

## 13. 权重打包与模型存储

INT4 只有 4 bit，而常规硬件最小寻址单位通常是字节，因此需要打包。例如两个无符号 INT4 值：

$$
q_0,q_1\in[0,15]
$$

可以打包到一个字节：

```python
packed = q0 | (q1 << 4)
```

量化模型通常保存：

- `qweight`：打包后的 INT4 权重；
- `scales`：每个量化 group 的 FP16/BF16 量化步长；
- `qzeros`：非对称量化的打包零点；
- bias：通常保持 FP16/BF16；
- 量化配置：bit 数、group size、kernel 版本、是否使用 zero point 等。

理论上，纯权重从 16 bit 降到 4 bit 可以缩小 4 倍。但实际模型还包含：

- 每组 scale 和 zero point；
- 未量化模块；
- embedding、LayerNorm、bias；
- CUDA kernel 工作区；
- 激活和 KV cache；
- 框架与内存分配开销。

因此，“模型权重文件大小”“模型加载后静态显存”和“生成时峰值显存”必须分别测量，不能混用。

---

## 14. 选择性量化与标准 W4A16 AWQ

标准 AWQ 通常量化模型中的主要 Linear 权重，激活保持 FP16/BF16。工程中也可以使用混合精度或选择性量化：

```text
敏感模块：保持 FP16/BF16
其他 Linear：使用 AWQ INT4
```

例如通过 `modules_to_not_convert` 排除特定模块。需要区分：

1. **层级敏感度策略**：决定哪些模块完全不量化；
2. **AWQ 通道 scale 搜索**：在被量化层内部重新分配各输入通道的有效量化精度；
3. **INT4 group quantization**：真正把权重映射和打包为 INT4。

“注意力权重保持 FP16”是一个容易产生歧义的说法：

- 如果指 softmax 后的 attention matrix，它属于运行时激活，W4A16 本来就不会把它量化为 INT4；
- 如果指 Q/K/V/O 投影参数，则表示这些 Linear 权重被排除在 INT4 量化之外；
- 如果只量化 FFN、保留全部注意力投影参数为 FP16，需要重新核算实际可达到的显存压缩比例。

---

## 15. 一个更接近工程代码的实现框架

下面的代码省略了模型结构适配、权重打包和高性能内核，但展示了 AWQ 的核心计算关系。

```python
import torch


@torch.no_grad()
def pseudo_quantize_symmetric_int4(weight, group_size=128):
    """返回伪量化后的浮点权重，用于搜索误差。"""
    original_shape = weight.shape
    assert original_shape[-1] % group_size == 0

    grouped = weight.reshape(-1, group_size)
    max_abs = grouped.abs().amax(dim=1, keepdim=True)
    quant_scale = max_abs.clamp(min=1e-5) / 7

    q = torch.round(grouped / quant_scale)
    q = torch.clamp(q, -7, 7)
    dequant = q * quant_scale

    return dequant.reshape(original_shape)


@torch.no_grad()
def search_awq_scale(
    x,
    weight,
    group_size=128,
    num_grid=20,
):
    """
    x:      [num_tokens, in_features]
    weight: [out_features, in_features]
    """
    x = x.reshape(-1, x.shape[-1])
    x_mean = x.abs().float().mean(dim=0).clamp(min=1e-5)

    reference = x @ weight.T
    best_error = float("inf")
    best_alpha = None
    best_scale = None

    for index in range(num_grid):
        alpha = index / num_grid
        channel_scale = x_mean.pow(alpha)

        channel_scale = channel_scale / torch.sqrt(
            channel_scale.max() * channel_scale.min()
        )

        scaled_weight = weight * channel_scale.view(1, -1)
        fake_quant_weight = pseudo_quantize_symmetric_int4(
            scaled_weight,
            group_size=group_size,
        )

        candidate = (
            x / channel_scale
        ) @ fake_quant_weight.T

        error = (
            candidate - reference
        ).float().pow(2).mean().item()

        if error < best_error:
            best_error = error
            best_alpha = alpha
            best_scale = channel_scale.clone()

    return {
        "alpha": best_alpha,
        "scale": best_scale,
        "mse": best_error,
    }
```

生产实现还需要处理：

- 非对称量化与 zero point；
- Q/K/V、gate/up 等多 Linear 联合搜索；
- 不同模型架构的 scale 融合规则；
- clipping 搜索；
- 校准样本分块，避免显存溢出；
- 不可整除 group size 的 padding；
- INT4 权重布局转换与打包；
- GEMM、GEMV、Marlin、ExLlama 等不同内核格式；
- 与 Transformers、vLLM 等推理框架的配置兼容。

---

## 16. AWQ 与 GPTQ 的核心差异

| 对比项 | AWQ | GPTQ |
|---|---|---|
| 主要依据 | 激活幅值与通道缩放 | 近似二阶信息/Hessian |
| 核心操作 | 搜索逐通道 scale、可选 clipping | 逐列量化并补偿量化误差 |
| 是否需要反向传播 | 否 | 否 |
| 校准复杂度 | 相对较低 | 通常更高 |
| 典型部署 | W4A16 | W4A16 |
| 工程重点 | scale 融合和规则 INT4 布局 | 量化顺序与误差补偿 |

两者都属于训练后权重量化，但误差建模和搜索方式不同。

---

## 17. 常见误区

### 误区 1：AWQ 只给少数敏感通道设置 scale

通常不是。AWQ 会为目标输入维度生成完整的通道 scale 向量。高激活通道获得更大的相对 scale，但其他通道也有对应缩放值。

### 误区 2：AWQ 直接量化激活

标准 W4A16 AWQ 量化的是权重，激活仍然是 FP16/BF16。激活统计用于判断哪些权重通道更重要。

### 误区 3：alpha 就是最终 scale

不是。alpha 是标量幂指数；最终 scale 是由 alpha、逐通道激活统计以及可选权重统计生成的向量。

### 误区 4：scale 越大越好

不是。过大的通道 scale 会产生新的权重离群值并增大量化 group 的量化步长，可能伤害其他通道。因此必须用输出误差搜索平衡点。

### 误区 5：AWQ 通道 scale 等于 INT4 quant scale

不是。前者是逐输入通道的等价变换参数，后者是逐权重 group 的整数映射步长。

### 误区 6：每个 Linear 都独立搜索 scale

不一定。共享同一输入和前置算子的多个 Linear 通常需要作为一个缩放关系组联合搜索。

### 误区 7：调用 AutoAWQ 等于自己实现了 Triton/CUDA kernel

不等于。AutoAWQ 的校准和量化搜索主要由 PyTorch 完成，推理阶段调用现成的 CUDA/Triton kernel。只有实际编写或修改了 INT4 解包、反量化、GEMM/GEMV 或调度逻辑，才适合表述为实现了量化推理内核。

---

## 18. 面试中的简洁回答

可以用下面这段话概括 AWQ：

> AWQ 是一种激活感知的训练后权重量化方法。它先用少量校准数据统计每个 Linear 输入通道的激活幅值，再通过网格搜索 alpha 生成逐通道 scale。对高激活通道，AWQ 相对放大其对应的权重列，并对输入做反向缩放，使浮点计算保持等价；随后对缩放后的权重做 INT4 分组量化，并以模块输出 MSE 选择最佳 scale。这样可以降低高激活通道对应权重的有效量化误差，同时保持规则的 INT4 权重布局，便于使用 CUDA 或 Triton kernel 加速推理。

如果继续追问 scale 的来源，可以回答：

> 简化公式是 $s_j=x_j^\alpha$，其中 $x_j$ 是校准集上第 $j$ 个输入通道的绝对激活均值，alpha 通过网格搜索得到。每个候选 alpha 会生成一整组通道 scale，然后对缩放权重做伪量化，计算量化模块输出和 FP16 输出之间的 MSE，选择误差最小的一组。AutoAWQ 的 duo scaling 还会同时考虑归一化权重幅值。

---

## 19. 总结

AWQ 的核心可以浓缩为以下公式：

$$
\boxed{ s^* = \arg\min_s \left\| XW^T-(X/s)Q(Ws)^T \right\|_2^2 }
$$

其本质是：

1. 激活越大的输入通道，对权重量化误差越敏感；
2. 根据激活统计，为高敏感通道设置更大的相对权重缩放；
3. 通过输入反向缩放保持浮点函数不变；
4. 使用校准集输出误差选择适当的缩放强度；
5. 将 scale 融合进相邻算子，避免额外运行时开销；
6. 最终使用规则的分组 INT4 权重和专用内核完成高效推理。

可以将 AWQ 理解为：

> 在有限的 INT4 表示能力下，利用真实激活分布重新分配量化精度，把更多有效精度留给最可能影响模型输出的权重通道。

---

## 参考资料

1. Lin et al., [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
2. MIT HAN Lab, [llm-awq](https://github.com/mit-han-lab/llm-awq)
3. Casper Hansen, [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)（该仓库已归档并停止维护）
4. Hugging Face, [Transformers AWQ documentation](https://huggingface.co/docs/transformers/quantization/awq)

