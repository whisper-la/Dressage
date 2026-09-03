# QServe W4A8 vs W8A8 Roofline 计算过程

本文整理 Qwen3-32B 在 A100-SXM4-80GB 上使用 QServe W4A8 的收益边界。核心问题是：W4A8 相比 W8A8 在 decode 阶段能少读一半权重，但在 prefill 阶段会引入 INT4 unpack / dequant 的额外开销。是否值得启用 W4A8，取决于 decode 省下的时间能否覆盖 prefill 多花的时间。

结论先行：

```text
当 B < 72：
  W4A8 decode linear 仍是 memory-bound，每个 decode step 约省 7.65 ms

当 72 <= B < 143：
  W4A8 decode linear 变成 compute-bound，但整体仍可能比 W8A8 快，收益随 batch 增大而递减

当 B >= 143：
  按本文模型，W4A8 的 decode 优势基本消失，W8A8 可能更合适
```

这里的 `72` 和 `143` 都不是经验拍脑袋，而是由 roofline 模型推出来的临界 batch size。

## 0. 先把计算思路讲清楚

这份计算只围绕 decode linear 做边界判断。原因是 LLM 在线服务里，decode 阶段是一 token 一 token 生成，batch 不大时每一步都要把每层 linear 的权重从 HBM 读出来，显存带宽经常是主要瓶颈。W4A8 和 W8A8 的最大差异正好在权重字节数：

```text
W8A8：每个权重约 1 byte
W4A8：每个权重约 0.5 byte
```

但 W4A8 不是白省这 0.5 byte。运行时需要把 INT4 权重 unpack，并结合 scale 做反量化或融合计算，所以它的有效算力会打折。文档把这个折损写成：

```text
alpha = 0.1
```

即认为 W4A8 的有效计算能力为理论峰值的 90%：

```text
F_effective = F * (1 - alpha)
```

于是整件事就变成三个问题：

```text
1. W4A8 在 prefill 里因为反量化多花多少时间？
2. W4A8 在 decode 里因为少读权重省多少时间？
3. 当 batch 变大时，decode 从 memory-bound 变成 compute-bound 的点在哪里？
```

`72` 回答第 3 个问题：W4A8 decode linear 到了 batch 约 72 后，从 memory-bound 进入 compute-bound。

`143` 回答另一个问题：batch 继续变大后，W4A8 的 decode 时间追上 W8A8，大约在 batch 143 附近优势消失。

## 1. 符号和硬件参数

请求侧参数：

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `S` | prefill 输入长度 |
| `D` | decode 输出长度 |
| `L` | Transformer 层数 |

硬件参数：

| 符号 | 含义 | A100-SXM4-80GB 取值 |
| --- | --- | --- |
| `F` | INT8/FP8 峰值计算能力 | `624 TFLOPS` |
| `BW` | HBM 显存带宽 | `2039 GB/s` |
| `alpha` | W4A8 反量化额外开销比例 | `0.1` |

因为 W4A8 需要处理 INT4 unpack、scale、dequant 等额外操作，文档用 `alpha=0.1` 表示 10% 性能折损。因此 W4A8 的有效计算能力为：

```text
F_effective = F * (1 - alpha)
            = 624T * 0.9
            = 561.6 TFLOPS
```

注意这里有两个容易混淆的量：

```text
FLOPS：某次计算需要完成的操作量
F：GPU 每秒可以完成的操作量，也就是计算能力
```

所以耗时的基本形式是：

```text
T = FLOPS / F
```

带宽侧也一样：

```text
T = Bytes / BW
```

如果 `Bytes` 用 byte，`BW` 用 byte/s，那么算出来的 `T` 是秒。转成毫秒需要再乘 `1000`。

例如：

```text
15.6 GB / 2039 GB/s = 0.00765 s = 7.65 ms
```

后面所有毫秒级常数都是这样来的。

## 2. Qwen3-32B 每层 Linear 的参数量

Qwen3-32B 使用如下结构参数：

| 参数 | 含义 | 取值 |
| --- | --- | --- |
| `H` | hidden size | `5120` |
| `n_q` | Q heads 数 | `64` |
| `n_kv` | KV heads 数 | `8` |
| `d` | head dim | `128` |
| `I` | FFN intermediate size | `25600` |
| `L` | layers | `64` |

每层主要 linear 包括：

```text
Q proj:    H -> n_q * d
K proj:    H -> n_kv * d
V proj:    H -> n_kv * d
O proj:    n_q * d -> H
Gate proj: H -> I
Up proj:   H -> I
Down proj: I -> H
```

逐个算权重参数量：

```text
Q proj:
  输入 H = 5120
  输出 n_q * d = 64 * 128 = 8192
  权重 = 5120 * 8192 = 41,943,040

K proj:
  输入 H = 5120
  输出 n_kv * d = 8 * 128 = 1024
  权重 = 5120 * 1024 = 5,242,880

V proj:
  和 K proj 一样
  权重 = 5,242,880

O proj:
  输入 n_q * d = 8192
  输出 H = 5120
  权重 = 8192 * 5120 = 41,943,040

Gate proj:
  输入 H = 5120
  输出 I = 25600
  权重 = 5120 * 25600 = 131,072,000

Up proj:
  和 Gate proj 一样
  权重 = 131,072,000

Down proj:
  输入 I = 25600
  输出 H = 5120
  权重 = 25600 * 5120 = 131,072,000
```

把 attention 里的四个 projection 合并：

```text
Q + O = 2 * H * (n_q * d)
K + V = 2 * H * (n_kv * d)

Q + K + V + O
  = 2 * H * (n_q * d + n_kv * d)
```

把 FFN 里的三个 projection 合并：

```text
Gate + Up + Down = 3 * H * I
```

每层权重参数量记为 `W`：

```text
W = 2 * H * (n_q * d + n_kv * d) + 3 * H * I
```

代入数值：

```text
n_q * d  = 64 * 128 = 8192
n_kv * d = 8 * 128  = 1024

W = 2 * 5120 * (8192 + 1024) + 3 * 5120 * 25600
  = 94,371,840 + 393,216,000
  = 487,587,840
```

所以 Qwen3-32B 每层 linear 权重约为：

```text
W = 487,587,840 params / layer
```

W8A8 下每个权重 1 byte，W4A8 下每个权重 0.5 byte：

```text
W8A8 权重读取 / 层 = W * 1.0 = 487.6 MB
W4A8 权重读取 / 层 = W * 0.5 = 243.8 MB
```

每层每 token 的激活读写量记为 `A`：

```text
A = 7 * H + 2 * (n_q * d + n_kv * d) + 3 * I
```

这个 `A` 是一个简化的 linear 激活流量估计。它把每个 linear 的输入和输出 activation 按元素数相加：

```text
Q proj:    H + n_q*d
K proj:    H + n_kv*d
V proj:    H + n_kv*d
O proj:    n_q*d + H
Gate proj: H + I
Up proj:   H + I
Down proj: I + H
```

把所有项相加：

```text
H 出现 7 次
n_q*d 出现 2 次
n_kv*d 出现 2 次
I 出现 3 次
```

所以：

```text
A = 7H + 2(n_q*d + n_kv*d) + 3I
```

代入：

```text
A = 7 * 5120 + 2 * (8192 + 1024) + 3 * 25600
  = 35,840 + 18,432 + 76,800
  = 131,072 bytes/token/layer
```

## 3. Prefill 阶段：W4A8 多花多少时间

Prefill 阶段一次处理 `B * S` 个 token，linear 部分的计算量为：

```text
FLOPS_prefill_linear = 2 * B * S * W * L
```

W8A8 耗时：

```text
T_prefill_W8A8 = FLOPS_prefill_linear / F
```

W4A8 耗时：

```text
T_prefill_W4A8 = FLOPS_prefill_linear / (F * (1 - alpha))
```

因此 W4A8 在 prefill 多花的时间为：

```text
Delta_T_prefill
  = T_prefill_W4A8 - T_prefill_W8A8
  = FLOPS_prefill_linear * (1 / (F * (1 - alpha)) - 1 / F)
```

这里的括号可以继续化简。先通分：

```text
1 / (F * (1 - alpha)) - 1 / F

= [F - F * (1 - alpha)] / [F * F * (1 - alpha)]

= [F * alpha] / [F * F * (1 - alpha)]

= alpha / [F * (1 - alpha)]
```

所以：

```text
Delta_T_prefill
  = 2 * B * S * W * L * alpha / (F * (1 - alpha))
```

代入 A100 和 Qwen3-32B 参数：

```text
Delta_T_prefill
  = 2 * B * S * 487,587,840 * 64 * 0.1 / 561.6T
  ≈ B * S * 0.0111 ms
```

逐步看这个常数：

```text
2 * W * L * alpha
  = 2 * 487,587,840 * 64 * 0.1
  = 6,241,124,352

F * (1 - alpha)
  = 624e12 * 0.9
  = 561.6e12

每个 batch-token 多花时间
  = 6,241,124,352 / 561.6e12 秒
  = 0.000011113 秒
  = 0.011113 ms
```

文档里取近似：

```text
Delta_T_prefill ≈ B * S * 0.0112 ms
```

这表示输入越长、batch 越大，W4A8 在 prefill 端需要偿还的反量化成本越高。

## 4. Decode 阶段：为什么小 batch 下 W4A8 快

Decode 每一步只生成 1 个 token。对 linear 来说，计算量是：

```text
FLOPS_decode_linear = 2 * B * W * L
```

但权重读取量不是随 `B` 等比例增长的。同一份权重可以被 batch 内的多个样本复用，所以小 batch 下 decode 的瓶颈通常是读权重。

当 W4A8 和 W8A8 都是 memory-bound 时：

```text
T_decode_W8A8 = (W + B * A) * L / BW
T_decode_W4A8 = (0.5 * W + B * A) * L / BW
```

二者相减：

```text
Delta_T_decode
  = T_decode_W8A8 - T_decode_W4A8
  = 0.5 * W * L / BW
```

代入数值：

```text
Delta_T_decode
  = 0.5 * 487,587,840 * 64 / 2039G
  ≈ 7.65 ms / decode step
```

逐步算：

```text
W4A8 比 W8A8 每层少读
  = 0.5 * W
  = 0.5 * 487,587,840
  = 243,793,920 bytes

64 层总共少读
  = 243,793,920 * 64
  = 15,602,810,880 bytes
  ≈ 15.60 GB

除以 A100 HBM 带宽
  = 15,602,810,880 / 2039e9 秒
  = 0.007652 秒
  = 7.652 ms
```

这就是文档中 `B < 72` 时每步固定节省 `7.65 ms` 的来源。

## 5. 72 是怎么推出来的

`72` 是 W4A8 decode linear 从 memory-bound 变为 compute-bound 的临界 batch size。

Roofline 的判断标准是算术强度：

```text
AI = FLOPS / Bytes
```

硬件的临界算术强度为：

```text
R = F_effective / BW
  = F * (1 - alpha) / BW
```

代入 A100：

```text
R = 624T * 0.9 / 2039G
  ≈ 275.43 FLOPs/byte
```

单位也可以拆开看：

```text
F_effective = 561.6 TFLOPS = 561.6e12 FLOPs/s
BW = 2039 GB/s = 2039e9 bytes/s

R = 561.6e12 / 2039e9
  = 275.43 FLOPs/byte
```

`R` 的意思是：如果一个 kernel 每读 1 byte 数据，能做超过约 275 次计算，就更可能被算力限制；如果每读 1 byte 只能做很少计算，就更可能被显存带宽限制。

W4A8 decode linear 的计算量和访存量为：

```text
FLOPS = 2 * B * W
Bytes = 0.5 * W + B * A
```

为什么 `FLOPS = 2 * B * W`：

```text
每个 linear 本质是矩阵乘：
  [B, in_dim] x [in_dim, out_dim]

乘加操作量约为：
  2 * B * in_dim * out_dim

而 in_dim * out_dim 就是这个 linear 的权重参数量。
把每层所有 linear 加起来，权重参数量就是 W。

所以每层 decode linear FLOPS:
  2 * B * W
```

为什么 `Bytes = 0.5 * W + B * A`：

```text
0.5 * W：
  W4A8 权重读取量，每个权重 0.5 byte

B * A：
  batch 内每个 token 的激活读写量，每个 token 约 A bytes
```

这里先不乘 `L`，因为判断 memory-bound / compute-bound 时每层都一样，乘不乘层数不会改变临界 batch。后面算绝对耗时时才乘 `L=64`。

因此：

```text
AI_W4A8 = 2 * B * W / (0.5 * W + B * A)
```

令 `AI_W4A8 = R`，得到临界点：

```text
2 * B * W / (0.5 * W + B * A) = R
```

展开：

```text
2 * B * W = R * (0.5 * W + B * A)
2BW = 0.5RW + RBA
B * (2W - RA) = 0.5RW
B = 0.5 * R * W / (2W - R * A)
```

把这个临界 batch 记为 `B1`：

```text
B1 = 0.5 * R * W / (2W - R * A)
```

代入：

```text
R = 275.43
W = 487,587,840
A = 131,072

B1 = 0.5 * 275.43 * 487,587,840
     / (2 * 487,587,840 - 275.43 * 131,072)
   ≈ 71.50
```

把分子分母拆开：

```text
分子：
  0.5 * R * W
  = 0.5 * 275.43 * 487,587,840
  ≈ 67,144,971,404

分母第一项：
  2 * W
  = 975,175,680

分母第二项：
  R * A
  = 275.43 * 131,072
  ≈ 36,101,620

分母：
  2W - RA
  ≈ 975,175,680 - 36,101,620
  ≈ 939,074,060

B1：
  67,144,971,404 / 939,074,060
  ≈ 71.50
```

工程上四舍五入：

```text
B1 ≈ 72
```

所以：

```text
B < 72：W4A8 decode linear 是 memory-bound
B >= 72：W4A8 decode linear 开始进入 compute-bound
```

直觉理解：batch 小的时候，计算量不够大，主要时间花在读权重；batch 变大后，同一份权重被更多样本复用，计算量按 `B` 增长，瓶颈逐渐从显存带宽转向计算。

## 6. 143 是怎么推出来的

`143` 是 W4A8 decode 优势消失的上界。也就是说，在这个点上：

```text
T_decode_W4A8 = T_decode_W8A8
```

当 `B >= B1` 后，W4A8 decode linear 已经 compute-bound，因此：

```text
T_decode_W4A8 = 2 * B * W * L / (F * (1 - alpha))
```

而在这段范围内，W8A8 仍按 memory-bound 估计：

```text
T_decode_W8A8 = (W + B * A) * L / BW
```

令两者相等：

```text
2 * B * W * L / (F * (1 - alpha))
  = (W + B * A) * L / BW
```

两边消去 `L`，并令：

```text
R = F * (1 - alpha) / BW
```

得到：

```text
2 * B * W = R * (W + B * A)
2BW = RW + RBA
B * (2W - RA) = RW
B = R * W / (2W - R * A)
```

这就是 W4A8 和 W8A8 decode 时间相等的 batch 上界，记为：

```text
B_upper = R * W / (2W - R * A)
```

而前面 `B1` 是：

```text
B1 = 0.5 * R * W / (2W - R * A)
```

所以：

```text
B_upper = 2 * B1
```

代入 `B1 ≈ 71.50`：

```text
B_upper ≈ 2 * 71.50
        ≈ 143.01
```

工程上取：

```text
B_upper ≈ 143
```

也可以不借助 `2 * B1`，直接代入：

```text
B_upper = R * W / (2W - R * A)

分子：
  R * W
  = 275.43 * 487,587,840
  ≈ 134,289,942,808

分母：
  2W - RA
  ≈ 939,074,060

B_upper:
  134,289,942,808 / 939,074,060
  ≈ 143.01
```

所以文档里给出三段：

```text
B < 72:
  W4A8 decode 节省稳定

72 <= B < 143:
  W4A8 decode 仍有收益，但收益递减

B >= 143:
  W4A8 decode 优势消失，W8A8 可能更快
```

## 7. 72 到 143 之间，收益为什么递减

在 `72 <= B < 143` 时：

```text
T_decode_W4A8 = 2 * B * W * L / (F * (1 - alpha))
T_decode_W8A8 = (W + B * A) * L / BW
```

所以每个 decode step 的节省为：

```text
Delta_T_decode
  = T_decode_W8A8 - T_decode_W4A8
  = L * [(W + B * A) / BW - 2 * B * W / (F * (1 - alpha))]
```

代入数值：

```text
W * L / BW ≈ 15.304 ms
B * A * L / BW ≈ B * 0.00411 ms
2 * B * W * L / (F * (1 - alpha)) ≈ B * 0.11113 ms
```

因此：

```text
Delta_T_decode
  ≈ 15.304 + B * 0.00411 - B * 0.11113
  ≈ 15.304 - B * 0.107 ms
```

这解释了为什么 batch 越大，W4A8 每步能省的时间越少。

几个例子：

| Batch | 每 decode step 约省时间 |
| --- | --- |
| `72` | `15.304 - 72 * 0.107 ≈ 7.60 ms` |
| `100` | `15.304 - 100 * 0.107 ≈ 4.60 ms` |
| `120` | `15.304 - 120 * 0.107 ≈ 2.46 ms` |
| `140` | `15.304 - 140 * 0.107 ≈ 0.32 ms` |
| `143` | 约等于 `0 ms` |

## 8. 什么时候 W4A8 划算

整体判断条件是：

```text
D * Delta_T_decode > Delta_T_prefill
```

也就是：

```text
decode 总共省下的时间 > prefill 多花的时间
```

### 8.1 当 B < 72

此时：

```text
Delta_T_decode ≈ 7.65 ms
Delta_T_prefill ≈ B * S * 0.0112 ms
```

所以：

```text
D * 7.65 > B * S * 0.0112
```

整理：

```text
D / S > B * 0.0112 / 7.65
D / S > B * 0.00146
```

例如 `S=1024`：

| Batch | D/S 门槛 | 最少输出 D |
| --- | --- | --- |
| `1` | `0.00146` | `2` |
| `8` | `0.0117` | `12` |
| `16` | `0.0234` | `24` |
| `32` | `0.0467` | `48` |
| `64` | `0.0934` | `96` |

这说明小 batch 下，W4A8 很容易划算。只要有一点 decode 长度，节省的权重带宽时间就能覆盖 prefill 端的反量化开销。

### 8.2 当 72 <= B < 143

此时：

```text
Delta_T_decode ≈ 15.304 - B * 0.107 ms
Delta_T_prefill ≈ B * S * 0.0112 ms
```

判断条件：

```text
D * (15.304 - B * 0.107) > B * S * 0.0112
```

整理：

```text
D / S > B * 0.0112 / (15.304 - B * 0.107)
```

例如 `S=1024`：

| Batch | 每步节省 | D/S 门槛 | 最少输出 D |
| --- | --- | --- | --- |
| `72` | `7.60 ms` | `0.106` | `109` |
| `80` | `6.74 ms` | `0.133` | `136` |
| `100` | `4.60 ms` | `0.243` | `249` |
| `120` | `2.46 ms` | `0.546` | `559` |
| `130` | `1.39 ms` | `1.044` | `1069` |
| `140` | `0.32 ms` | `4.84` | `4956` |

越接近 143，每步 decode 能省的时间越少，因此需要非常长的输出才能抵消 prefill 成本。

## 9. 通用公式

如果换 GPU 或模型，可以用下面这一组公式重新计算。

模型侧：

```text
W = 2 * H * (n_q * d + n_kv * d) + 3 * H * I
A = 7 * H + 2 * (n_q * d + n_kv * d) + 3 * I
```

硬件侧：

```text
R = F * (1 - alpha) / BW
```

W4A8 从 memory-bound 变为 compute-bound 的临界 batch：

```text
B1 = 0.5 * R * W / (2W - R * A)
```

W4A8 decode 优势消失的 batch 上界：

```text
B_upper = R * W / (2W - R * A)
        = 2 * B1
```

选择 W4A8 的 D/S 门槛：

```text
当 B < B1:
  D / S > 4 * alpha * B / R

当 B1 <= B < 2B1:
  D / S > 4 * alpha * B * B1 / [R * (2B1 - B)]

当 B >= 2B1:
  W4A8 不再推荐
```

## 10. 工程解释

这个模型背后的直觉是：

```text
小 batch decode：
  每一步都要读完整模型权重，显存带宽是瓶颈。
  W4A8 把权重从 1 byte/param 降到 0.5 byte/param，因此收益明显。

大 batch decode：
  同一份权重被更多样本复用，计算量按 batch 增长。
  W4A8 的 unpack/dequant 开销开始显现，收益逐渐下降。

prefill：
  B*S 通常较大，linear 更接近 compute-bound。
  W4A8 的额外反量化开销会让 prefill 比 W8A8 慢一点。
```

所以最终判断不是“W4A8 一定比 W8A8 快”，而是看请求形态：

```text
小 batch + 长输出：W4A8 更合适
大 batch + 短输出：W8A8 可能更合适
前缀缓存命中高：有效 S 变小，W4A8 更容易划算
```

对于 A100 + Qwen3-32B，本文模型给出的实用规则是：

```text
B < 72：
  大多数有正常输出长度的场景都适合 W4A8

72 <= B < 143：
  需要根据 D/S 判断，输出越长越适合 W4A8

B >= 143：
  W4A8 的 decode 优势基本消失
```

如果服务里有 radix cache 或 prefix cache，需要把 `S` 换成实际发生 prefill 的有效输入长度。前缀命中越多，`Delta_T_prefill` 越小，W4A8 越容易获得端到端收益。
