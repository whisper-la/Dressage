"""
常见 Transformer / RL 算法手撕题：最小 PyTorch 教学实现。

统一符号：
    B: batch size
    S/T: sequence length
    H: hidden size
    Nq: query head 数
    Nkv: key/value head 数
    D: head_dim = H / Nq
    F: MLP intermediate size
    P: TP / SP / CP world size

说明：
1. 所有 Attention 接口都采用 batch-first。
2. TP/SP/CP 代码强调数学与通信语义，不替代 Megatron/TransformerEngine
   的融合 kernel、异步流水和自定义 backward。
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

Tensor = torch.Tensor


# =============================================================================
# 10. Safe softmax
# =============================================================================


def safe_softmax(x, dim=-1):
    dtype = x.dtype
    x = x.float()

    max_x = x.max(dim=dim, keepdim=True).values
    max_x = torch.where(torch.isfinite(max_x), max_x, 0)

    exp_x = torch.where(
        torch.isfinite(x),
        torch.exp(x - max_x),
        0,
    )

    denominator = exp_x.sum(dim=dim, keepdim=True)
    denominator = denominator.clamp_min(1e-12)

    probabilities = exp_x / denominator
    return probabilities.to(dtype)


# =============================================================================
# 1. SDPA / MHA / GQA
# =============================================================================


def sdpa(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    """
    最简 self-attention。

    q/k/v: [B,N,S,D]
    score: [B,N,S,S]
    out:   [B,N,S,D]
    """
    s, d = q.size(-2), q.size(-1)
    score = q @ k.transpose(-2, -1) / math.sqrt(d)         # [B,N,S,S]
    if causal:
        mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
        score = score.masked_fill(~mask, float("-inf"))
    prob = torch.softmax(score, dim=-1)                    # [B,N,S,S]
    return prob @ v                                       # [B,N,S,D]


class MultiHeadAttention(nn.Module):
    """最小 self-MHA；输入输出均为 [B, S, H]。"""

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        b, s, h = x.shape                                  # x: [B,S,H]
        qkv = self.qkv(x)                                  # [B,S,3H]
        q, k, v = qkv.chunk(3, dim=-1)                    # 各 [B,S,H]

        # [B,S,H] -> [B,S,N,D] -> [B,N,S,D]
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        y = sdpa(q, k, v, causal)                          # [B,N,S,D]
        y = y.transpose(1, 2).contiguous().view(b, s, h)   # [B,S,H]
        return self.out_proj(y)                            # [B,S,H]


class GroupedQueryAttention(nn.Module):
    """
    GQA：Nq 个 Q heads，共享 Nkv 个 KV heads。
    Nkv=Nq 是 MHA；Nkv=1 是 MQA。
    """

    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_query_heads == 0
        assert num_query_heads % num_kv_heads == 0
        self.hidden_size = hidden_size
        self.nq = num_query_heads
        self.nkv = num_kv_heads
        self.d = hidden_size // num_query_heads
        self.q_proj = nn.Linear(hidden_size, self.nq * self.d, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.nkv * self.d, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.nkv * self.d, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        b, s, _ = x.shape                                  # [B,S,H]
        q = self.q_proj(x).view(b, s, self.nq, self.d).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.nkv, self.d).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.nkv, self.d).transpose(1, 2)
        # q:[B,Nq,S,D], k/v:[B,Nkv,S,D]

        repeat = self.nq // self.nkv
        k = k.repeat_interleave(repeat, dim=1)             # [B,Nq,S,D]
        v = v.repeat_interleave(repeat, dim=1)             # [B,Nq,S,D]
        y = sdpa(q, k, v, causal)                          # [B,Nq,S,D]
        y = y.transpose(1, 2).contiguous().view(b, s, self.hidden_size)
        return self.out_proj(y)                            # [B,S,H]


# =============================================================================
# 2. MHA with KV cache
# =============================================================================


KVCache = tuple[Tensor, Tensor]


class MHAWithKVCache(nn.Module):
    """
    自回归 MHA。

    cache：
        K_cache/V_cache: [B, N, Tpast, D]
    每次输入：
        x_new:           [B, Tnew, H]
    返回：
        y_new:           [B, Tnew, H]
        new_cache:       两个 [B, N, Tpast+Tnew, D]
    """

    def __init__(self, hidden_size: int, num_heads: int, bias: bool = False):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.h = hidden_size
        self.n = num_heads
        self.d = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        x_new: Tensor,
        cache: Optional[KVCache] = None,
    ) -> tuple[Tensor, KVCache]:
        b, t_new, _ = x_new.shape                          # [B,Tnew,H]
        q, k_new, v_new = self.qkv(x_new).chunk(3, dim=-1)

        q = q.view(b, t_new, self.n, self.d).transpose(1, 2)
        k_new = k_new.view(b, t_new, self.n, self.d).transpose(1, 2)
        v_new = v_new.view(b, t_new, self.n, self.d).transpose(1, 2)
        # q/k_new/v_new: [B,N,Tnew,D]

        if cache is None:
            k_all, v_all = k_new, v_new
        else:
            k_cache, v_cache = cache                       # [B,N,Tpast,D]
            k_all = torch.cat((k_cache, k_new), dim=2)     # [B,N,Ttotal,D]
            v_all = torch.cat((v_cache, v_new), dim=2)     # [B,N,Ttotal,D]

        # KV cache 中 Sq != Sk，所以在这里直接写带 past offset 的 attention。
        t_total = k_all.size(2)
        score = q @ k_all.transpose(-2, -1) / math.sqrt(self.d)
        # score: [B,N,Tnew,Ttotal]
        q_pos = torch.arange(t_new, device=x_new.device) + t_total - t_new
        k_pos = torch.arange(t_total, device=x_new.device)
        causal_mask = k_pos[None, :] <= q_pos[:, None]     # [Tnew,Ttotal]
        score = score.masked_fill(~causal_mask, float("-inf"))
        prob = torch.softmax(score, dim=-1)
        y = prob @ v_all                                   # [B,N,Tnew,D]
        y = y.transpose(1, 2).contiguous().view(b, t_new, self.h)
        return self.out_proj(y), (k_all, v_all)


# =============================================================================
# 3. FLOPs 计算
# =============================================================================


def transformer_block_flops(
    b: int,
    s: int,
    h: int,
    num_heads: int,
    ffn_hidden: Optional[int] = None,
    *,
    causal_ideal_triangle: bool = False,
) -> dict[str, int]:
    """
    返回一次 forward 的主项 FLOPs；1 次乘加按 2 FLOPs。

    SwiGLU: gate/up 两个 H->F，再 F->H。
    不含 RMSNorm、bias、dropout；softmax 单独给出近似值。
    """
    f = 8 * h // 3 if ffn_hidden is None else ffn_hidden
    qkv = 6 * b * s * h * h
    out_proj = 2 * b * s * h * h
    # QK^T 与 PV 各 2*B*S^2*H；理想 causal kernel 只算约一半三角区。
    attention_matmuls = (2 if causal_ideal_triangle else 4) * b * s * s * h
    attention_elements = (
        b * num_heads * s * (s + 1) // 2
        if causal_ideal_triangle
        else b * num_heads * s * s
    )
    softmax_approx = 5 * attention_elements
    mlp = 6 * b * s * h * f
    return {
        "qkv_projection": qkv,
        "attention_qk_and_pv": attention_matmuls,
        "output_projection": out_proj,
        "mlp_matmuls": mlp,
        "total_matmuls": qkv + attention_matmuls + out_proj + mlp,
        "softmax_approx": softmax_approx,
    }


# =============================================================================
# 4. 基于 MHA 的 Transformer block
# =============================================================================


class FeedForward(nn.Module):
    """SwiGLU MLP：[B,S,H] -> [B,S,F] -> [B,S,H]。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate(x))                       # [B,S,F]
        up = self.up(x)                                   # [B,S,F]
        return self.down(gate * up)                       # [B,S,H]


class RMSNorm(nn.Module):
    """只在最后一个 hidden 维度上做均方根归一化。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()

        rms = x.pow(2).mean(dim=-1, keepdim=True)          # [B,S,1]
        x = x * torch.rsqrt(rms + self.eps)                # [B,S,H]
        x = x * self.weight

        return x.to(dtype)


class TransformerBlock(nn.Module):
    """Pre-RMSNorm decoder block；输入输出 [B,S,H]。"""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), causal=True)      # [B,S,H]
        x = x + self.mlp(self.norm2(x))                    # [B,S,H]
        return x


# =============================================================================
# 5/6. TP 与 TP+SP Transformer block
# =============================================================================


def _group_size(group: Optional[dist.ProcessGroup]) -> int:
    return dist.get_world_size(group) if dist.is_initialized() else 1


def _group_rank(group: Optional[dist.ProcessGroup]) -> int:
    return dist.get_rank(group) if dist.is_initialized() else 0


def all_reduce_sum(x: Tensor, group: Optional[dist.ProcessGroup]) -> Tensor:
    """各 TP rank 的同 shape 张量求和。"""
    if _group_size(group) == 1:
        return x
    y = x.clone()
    dist.all_reduce(y, group=group)
    return y


class ColumnParallelLinear(nn.Module):
    """
    采用文章中的矩阵记法 Y=X@W。

    完整 W: [in, out]
    本地 W: [in, out/P]
    输入 x: [B,S,in]
    输出 y: [B,S,out/P]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.p = _group_size(group)
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features // self.p)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight


class RowParallelLinear(nn.Module):
    """
    采用文章中的矩阵记法 Y=X@W。

    完整 W:   [in, out]
    本地 W:   [in/P, out]
    本地输入: [B,S,in/P]
    partial:  [B,S,out]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.p = _group_size(group)
        self.weight = nn.Parameter(
            torch.empty(in_features // self.p, out_features)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x_local: Tensor) -> Tensor:
        """只返回本地 partial；通信由 Attention/MLP 决定。"""
        return x_local @ self.weight


class TPAttention(nn.Module):
    """只做 TP；输入输出都是 [B,S,H]。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.group = group
        self.p = _group_size(group)
        self.h = hidden_size
        self.local_heads = num_heads // self.p
        self.d = hidden_size // num_heads

        # 完整 Q/K/V 权重为 [H,H]；每卡保存 [H,H/P]。
        self.q_proj = ColumnParallelLinear(hidden_size, hidden_size, group)
        self.k_proj = ColumnParallelLinear(hidden_size, hidden_size, group)
        self.v_proj = ColumnParallelLinear(hidden_size, hidden_size, group)

        # 完整 O 权重为 [H,H]；每卡保存 [H/P,H]。
        self.o_proj = RowParallelLinear(hidden_size, hidden_size, group)

    def partial_output(self, x: Tensor) -> Tensor:
        """返回每个 rank 的 [B,S,H] partial output。"""
        b, s, _ = x.shape
        q = self.q_proj(x)                             # [B,S,H/P]
        k = self.k_proj(x)                             # [B,S,H/P]
        v = self.v_proj(x)                             # [B,S,H/P]

        q = q.view(b, s, self.local_heads, self.d).transpose(1, 2)
        k = k.view(b, s, self.local_heads, self.d).transpose(1, 2)
        v = v.view(b, s, self.local_heads, self.d).transpose(1, 2)
        # q/k/v: [B,N/P,S,D]

        y = sdpa(q, k, v, causal=True)                 # [B,N/P,S,D]
        y = y.transpose(1, 2).contiguous().view(b, s, self.h // self.p)
        return self.o_proj(y)                          # [B,S,H] partial

    def forward(self, x: Tensor) -> Tensor:
        partial = self.partial_output(x)               # [B,S,H]
        return all_reduce_sum(partial, self.group)     # [B,S,H]


class TPMLP(nn.Module):
    """TP SwiGLU；输入输出都是 [B,S,H]。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.group = group
        self.gate = ColumnParallelLinear(hidden_size, intermediate_size, group)
        self.up = ColumnParallelLinear(hidden_size, intermediate_size, group)
        self.down = RowParallelLinear(intermediate_size, hidden_size, group)

    def partial_output(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate(x))                  # [B,S,F/P]
        up = self.up(x)                              # [B,S,F/P]
        return self.down(gate * up)                  # [B,S,H] partial

    def forward(self, x: Tensor) -> Tensor:
        partial = self.partial_output(x)             # [B,S,H]
        return all_reduce_sum(partial, self.group)   # [B,S,H]


class TPBlock(nn.Module):
    """
    纯 TP Transformer block。
    输入输出均为 [B,S,H]；每个 TP rank 都保存完整 activation。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = TPAttention(hidden_size, num_heads, group)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = TPMLP(hidden_size, intermediate_size, group)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))             # [B,S,H]
        x = x + self.mlp(self.norm2(x))              # [B,S,H]
        return x


# -----------------------------------------------------------------------------
# 理解纯 TP 后，再增加 SP
# -----------------------------------------------------------------------------


def all_gather_sequence(
    x_local: Tensor,
    group: Optional[dist.ProcessGroup],
) -> Tensor:
    """[B,S/P,H] -> [B,S,H]。"""
    p = _group_size(group)
    if p == 1:
        return x_local
    parts = [torch.empty_like(x_local) for _ in range(p)]
    dist.all_gather(parts, x_local.contiguous(), group=group)
    return torch.cat(parts, dim=1)


def reduce_scatter_sequence(
    partial: Tensor,
    group: Optional[dist.ProcessGroup],
) -> Tensor:
    """
    [B,S,H] partial -> [B,S/P,H]。
    教学写法使用 all-reduce + chunk；生产实现使用 reduce-scatter。
    """
    p = _group_size(group)
    if p == 1:
        return partial
    rank = _group_rank(group)
    full = all_reduce_sum(partial, group)
    return full.chunk(p, dim=1)[rank].contiguous()


class TPSPAttention(TPAttention):
    """TP+SP Attention；输入输出都是 [B,S/P,H]。"""

    def forward(self, x_local: Tensor) -> Tensor:
        x = all_gather_sequence(x_local, self.group)       # [B,S,H]
        partial = self.partial_output(x)                   # [B,S,H]
        return reduce_scatter_sequence(partial, self.group)  # [B,S/P,H]


class TPSPMLP(TPMLP):
    """TP+SP MLP；输入输出都是 [B,S/P,H]。"""

    def forward(self, x_local: Tensor) -> Tensor:
        x = all_gather_sequence(x_local, self.group)       # [B,S,H]
        partial = self.partial_output(x)                   # [B,S,H]
        return reduce_scatter_sequence(partial, self.group)  # [B,S/P,H]


class TPSPBlock(nn.Module):
    """
    TP+SP Transformer block。
    输入输出均为 [B,S/P,H]；RMSNorm 与残差只处理本地 sequence chunk。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = TPSPAttention(hidden_size, num_heads, group)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = TPSPMLP(hidden_size, intermediate_size, group)

    def forward(self, x_local: Tensor) -> Tensor:
        x_local = x_local + self.attn(self.norm1(x_local))
        x_local = x_local + self.mlp(self.norm2(x_local))
        return x_local


# =============================================================================
# 7. Megatron CP 的 ring attention（教学版 forward）
# =============================================================================


def megatron_cp_indices(
    global_seqlen: int,
    cp_size: int,
    cp_rank: int,
    *,
    device: Optional[torch.device] = None,
) -> Tensor:
    """
    Megatron causal CP 的负载均衡切法。

    全序列先切成 2P 块，rank r 拿第 r 块和第 2P-r-1 块：
        global: [0,1,2,3,4,5,6,7], P=2
        rank0:  [0,1] + [6,7]
        rank1:  [2,3] + [4,5]
    每个 rank 都同时有“较早”和“较晚”的 query，三角计算更均衡。
    返回 positions: [S/P]。
    """
    if cp_size == 1:
        return torch.arange(global_seqlen, device=device)
    assert global_seqlen % (2 * cp_size) == 0
    chunks = torch.arange(global_seqlen, device=device).chunk(2 * cp_size)
    return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]))


def _global_peer(group: Optional[dist.ProcessGroup], group_rank: int) -> int:
    if group is None:
        return group_rank
    return dist.get_global_rank(group, group_rank)


def cp_ring_attention(
    q_local: Tensor,
    k_local: Tensor,
    v_local: Tensor,
    *,
    global_seqlen: Optional[int] = None,
    q_positions: Optional[Tensor] = None,
    group: Optional[dist.ProcessGroup] = None,
    causal: bool = True,
) -> Tensor:
    """
    Megatron CP / ring attention 的最小 forward。

    每个 rank：
        q_local/k_local/v_local: [B,N,S/P,D]
        固定本地 Q，把 K/V 沿 ring 轮转 P 次；
        每块结果用 online-softmax 状态 (m,l,acc) 精确合并。

    返回：
        out_local: [B,N,S/P,D]

    这是教学版：
        - 展示 P2P ring、causal 全局位置和 online merge；
        - 不含生产版 fused kernel、双缓冲细节、自定义 backward；
        - Megatron 生产实现会在 backward 反向轮转 dK/dV，并重算 attention。
    """
    p = _group_size(group)
    rank = _group_rank(group)
    b, n, s_local, d = q_local.shape
    s_global = p * s_local if global_seqlen is None else global_seqlen

    if q_positions is None:
        q_positions = megatron_cp_indices(s_global, p, rank, device=q_local.device)

    # online softmax 状态：m/l 为每个 query row 的最大值和 exp-sum。
    m = torch.full((b, n, s_local, 1), float("-inf"), device=q_local.device)
    l = torch.zeros((b, n, s_local, 1), device=q_local.device)
    acc = torch.zeros((b, n, s_local, d), device=q_local.device)
    qf = q_local.float()
    k_cur, v_cur = k_local.contiguous(), v_local.contiguous()

    for step in range(p):
        source_rank = (rank - step) % p
        k_positions = megatron_cp_indices(
            s_global, p, source_rank, device=q_local.device
        )

        # 先发起异步 P2P，再计算当前 KV block，以展示通信/计算重叠位置。
        works = []
        if p > 1 and step < p - 1:
            k_next, v_next = torch.empty_like(k_cur), torch.empty_like(v_cur)
            next_peer = _global_peer(group, (rank + 1) % p)
            prev_peer = _global_peer(group, (rank - 1) % p)
            ops = [
                dist.P2POp(dist.isend, k_cur, next_peer, group),
                dist.P2POp(dist.irecv, k_next, prev_peer, group),
                dist.P2POp(dist.isend, v_cur, next_peer, group),
                dist.P2POp(dist.irecv, v_next, prev_peer, group),
            ]
            works = dist.batch_isend_irecv(ops)

        scores = torch.matmul(qf, k_cur.float().transpose(-2, -1)) / math.sqrt(d)
        # scores: [B,N,Sq_local,Sk_local]
        if causal:
            keep = k_positions[None, :] <= q_positions[:, None]  # [Sq,Sk]
            scores = scores.masked_fill(~keep[None, None], float("-inf"))

        block_max = scores.amax(dim=-1, keepdim=True)       # [B,N,Slocal,1]
        new_m = torch.maximum(m, block_max)
        new_m_safe = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
        alpha = torch.where(torch.isfinite(m), torch.exp(m - new_m_safe), torch.zeros_like(m))
        p_block = torch.where(
            torch.isfinite(scores),
            torch.exp(scores - new_m_safe),
            torch.zeros_like(scores),
        )                                                   # [B,N,Slocal,Slocal]
        acc = alpha * acc + torch.matmul(p_block, v_cur.float())
        l = alpha * l + p_block.sum(dim=-1, keepdim=True)
        m = new_m

        for work in works:
            work.wait()
        if works:
            k_cur, v_cur = k_next, v_next

    out = torch.where(l > 0, acc / l.clamp_min(1e-30), torch.zeros_like(acc))
    return out.to(q_local.dtype)


# =============================================================================
# 8. PPO / GRPO / GSPO / CISPO
# =============================================================================


# -----------------------------------------------------------------------------
# 8.1 PPO：GAE + clipped policy/value loss
# -----------------------------------------------------------------------------


def _masked_mean(x, mask, dim=None):
    """只对 mask=1 的位置求平均；dim=None 表示对所有维度平均。"""
    mask = mask.to(x.dtype)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(1)


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """
    PPO 的 GAE。

    rewards: [B,T]，每个 token 的 reward
    values:  [B,T]，每个 token 的 value
    mask:    [B,T]，有效 token 为 1，padding 为 0

    每个样本的最后一个有效 token 视为终止状态，
    因此它的 next_value=0。
    """
    b, t = rewards.shape
    mask = mask.to(device=rewards.device, dtype=rewards.dtype)

    advantages = torch.zeros_like(rewards)                 # [B,T]
    gae = torch.zeros(b, device=rewards.device, dtype=rewards.dtype)  # [B]

    for i in reversed(range(t)):
        valid = mask[:, i]                                 # [B]

        if i < t - 1:
            next_valid = mask[:, i + 1]                    # [B]
            next_value = values[:, i + 1] * next_valid     # [B]
        else:
            next_valid = torch.zeros_like(valid)           # [B]
            next_value = torch.zeros_like(gae)             # [B]

        delta = rewards[:, i] + gamma * next_value - values[:, i]  # [B]
        gae = delta + gamma * gae_lambda * next_valid * gae        # [B]
        gae = gae * valid                                  # [B]
        advantages[:, i] = gae                             # [B,T] 的第 i 列

    returns = (advantages + values) * mask                 # [B,T]
    return advantages, returns


@torch.no_grad()
def compute_gae_multiturn(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """
    Multi-turn PPO 的 GAE。

    rewards/values/mask: [B,T]
    mask=1 表示模型生成的 action token；
    mask=0 可以是 prompt、user、tool observation 或 padding。

    同一行视为一个完整 episode，GAE 沿下一个 mask=1 的位置递推，
    只有最后一个有效 action 才视为终止。
    """
    b, _ = rewards.shape
    valid_mask = mask.bool()
    advantages = torch.zeros_like(rewards)                 # [B,T]

    for batch_idx in range(b):
        positions = torch.where(valid_mask[batch_idx])[0]  # 当前样本的 action 位置
        gae = rewards.new_zeros(())

        for j in reversed(range(positions.numel())):
            pos = positions[j]
            next_value = values[batch_idx, positions[j + 1]] if j < positions.numel() - 1 else 0.0
            delta = rewards[batch_idx, pos] + gamma * next_value - values[batch_idx, pos]
            gae = delta + gamma * gae_lambda * gae
            advantages[batch_idx, pos] = gae

    returns = (advantages + values) * mask                 # [B,T]
    return advantages, returns


def ppo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    clip_eps: float = 0.2,
    values: Optional[Tensor] = None,
    old_values: Optional[Tensor] = None,
    returns: Optional[Tensor] = None,
    value_clip_eps: float = 0.2,
    entropy: Optional[Tensor] = None,
    value_coef: float = 0.5,
    entropy_coef: float = 0.0,
) -> dict[str, Tensor]:
    """
    token-level PPO。

    log_probs/old_log_probs/advantages/mask: [B,T]
    values/old_values/returns:     [B,T]（可选）
    entropy:                       [B,T]（可选）

    单轮使用 compute_gae，多轮使用 compute_gae_multiturn；
    rollout 阶段计算 advantages/returns，PPO epoch 中复用。
    """
    ratio = torch.exp(log_probs - old_log_probs.detach())  # [B,T]
    adv = advantages.detach()
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -_masked_mean(torch.minimum(surr1, surr2), mask)

    value_loss = log_probs.new_zeros(())
    if values is not None:
        v_clip = old_values + (values - old_values).clamp(
            -value_clip_eps, value_clip_eps
        )
        vf1 = (values - returns) ** 2
        vf2 = (v_clip - returns) ** 2
        value_loss = 0.5 * _masked_mean(torch.maximum(vf1, vf2), mask)

    entropy_mean = (
        log_probs.new_zeros(())
        if entropy is None
        else _masked_mean(entropy, mask)
    )
    total = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
    }


# -----------------------------------------------------------------------------
# 8.2 GRPO：group-relative advantage + token-level clipping
# -----------------------------------------------------------------------------


def group_relative_advantage(rewards: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Slime 风格的组内相对 advantage；GRPO、GSPO、CISPO 共用。

    rewards: [B,G]，同一个 prompt 的 G 条 response 为一组。
    return:  [B,G]。
    """
    advantages = rewards - rewards.mean(dim=-1, keepdim=True)  # [B,G]
    advantages = advantages / (
        advantages.std(dim=-1, keepdim=True) + eps
    )
    return advantages                                          # [B,G]


def grpo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    mask: Tensor,
    *,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    reference_log_probs: Optional[Tensor] = None,
    kl_beta: float = 0.0,
) -> Tensor:
    """
    原始 GRPO 的 outcome-supervision 目标。

    log_probs/old_log_probs/mask: [B,G,T]
    rewards:            [B,G]
    advantage:          [B,G] -> [B,G,1]，一条 response 内所有 token 相同。
    ratio:              [B,G,T]，逐 token clipping。
    """
    advantage = group_relative_advantage(rewards).detach()
    advantage = advantage.unsqueeze(-1)                             # [B,G,1]
    ratio = torch.exp(log_probs - old_log_probs.detach())           # [B,G,T]
    obj1 = ratio * advantage
    obj2 = ratio.clamp(1 - epsilon_low, 1 + epsilon_high) * advantage
    objective = torch.minimum(obj1, obj2)

    # 原论文：先对每条 response 的 token 平均，再对 B*G 平均。
    per_seq = _masked_mean(objective, mask, dim=-1)                 # [B,G]
    valid_seq = mask.sum(dim=-1) > 0                                # [B,G]
    policy_loss = -_masked_mean(per_seq, valid_seq)                 # []

    # Slime 将 reference KL 作为独立 loss，不放进 clipping objective。
    if reference_log_probs is not None and kl_beta:
        log_reference_over_policy = reference_log_probs.detach() - log_probs
        kl = (
            torch.exp(log_reference_over_policy)
            - log_reference_over_policy
            - 1
        )                                                          # [B,G,T]
        kl_per_seq = _masked_mean(kl, mask, dim=-1)                 # [B,G]
        kl_loss = _masked_mean(kl_per_seq, valid_seq)               # []
        policy_loss = policy_loss + kl_beta * kl_loss

    return policy_loss


# -----------------------------------------------------------------------------
# 8.3 GSPO：sequence-level ratio/clipping
# 复用 8.2 的 group_relative_advantage
# -----------------------------------------------------------------------------


def gspo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    mask: Tensor,
    *,
    epsilon_low: float = 3e-4,
    epsilon_high: float = 4e-4,
) -> Tensor:
    """
    GSPO：长度归一化的 sequence ratio + sequence-level clip。

    log_probs/old_log_probs/mask: [B,G,T]
    rewards:            [B,G]
    seq_ratio:          [B,G]
    """
    lengths = mask.sum(dim=-1)                                      # [B,G]
    mean_log_ratio = ((log_probs - old_log_probs.detach()) * mask).sum(-1)
    mean_log_ratio = mean_log_ratio / lengths.clamp_min(1)          # [B,G]
    seq_ratio = torch.exp(mean_log_ratio)                           # [B,G]
    adv = group_relative_advantage(rewards).detach()                # [B,G]
    obj1 = seq_ratio * adv
    obj2 = seq_ratio.clamp(1 - epsilon_low, 1 + epsilon_high) * adv
    objective = torch.minimum(obj1, obj2)
    valid = lengths > 0
    return -(objective * valid).sum() / valid.sum().clamp_min(1)


# -----------------------------------------------------------------------------
# 8.4 CISPO：clipped IS weight
# 复用 8.2 的 group_relative_advantage
# -----------------------------------------------------------------------------


def cispo_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    rewards: Tensor,
    mask: Tensor,
    *,
    epsilon_low: Optional[float] = None,
    epsilon_high: float = 0.2,
) -> Tensor:
    """
    CISPO：clip IS weight，而不是把越界 token 的目标裁成常数。

    log_probs/old_log_probs/mask: [B,G,T]
    rewards:            [B,G]

    原论文实验不设下界；因此 epsilon_low=None 时 lower=0，只裁上界。
    clipped weight 被 detach，但 log_probs 不 detach，所以每个有效 token 仍有梯度。
    """
    adv = group_relative_advantage(rewards).detach().unsqueeze(-1)  # [B,G,1]
    ratio = torch.exp(log_probs - old_log_probs.detach())           # [B,G,T]
    lower = 0.0 if epsilon_low is None else 1.0 - epsilon_low
    weight = ratio.clamp(lower, 1.0 + epsilon_high).detach()        # [B,G,T]
    token_objective = weight * adv * log_probs                      # [B,G,T]

    # 论文：每个 prompt 内除以 G 条 response 的总有效 token 数，再平均 B。
    numerator = (token_objective * mask).sum(dim=(-2, -1))          # [B]
    denominator = mask.sum(dim=(-2, -1)).clamp_min(1)               # [B]
    return -(numerator / denominator).mean()


# =============================================================================
# 9. KL 散度：probs / log_probs / logits / sampled-token
# =============================================================================


def _reduce_kl(
    value: Tensor,
    reduction: Literal["none", "mean", "sum", "batchmean"],
    mask: Optional[Tensor],
) -> Tensor:
    if mask is not None:
        value = value * mask.to(value.dtype)
    if reduction == "none":
        return value
    if reduction == "sum":
        return value.sum()
    if reduction == "batchmean":
        return value.sum() / max(value.shape[0], 1)
    if reduction == "mean":
        if mask is None:
            return value.mean()
        return value.sum() / mask.sum().clamp_min(1)
    raise ValueError(f"未知 reduction: {reduction}")


def categorical_kl(
    p: Tensor,
    q: Tensor,
    *,
    input_format: Literal["probs", "log_probs", "logits"] = "logits",
    reduction: Literal["none", "mean", "sum", "batchmean"] = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    精确 categorical KL(P||Q)，最后一维 V 是完整词表。

    输入：
        p/q: [...,V]
    输出（reduction='none'）：
        [...]

    probs      -> 直接处理概率；
    log_probs  -> 输入已经是 log probability；
    logits     -> 内部做 log_softmax。
    """
    if input_format == "logits":
        log_probs_p, log_probs_q = F.log_softmax(p, -1), F.log_softmax(q, -1)
        probs_p = log_probs_p.exp()
        term = torch.where(
            probs_p > 0,
            probs_p * (log_probs_p - log_probs_q),
            torch.zeros_like(probs_p),
        )
    elif input_format == "log_probs":
        log_probs_p, log_probs_q = p, q
        probs_p = log_probs_p.exp()
        term = torch.where(
            probs_p > 0,
            probs_p * (log_probs_p - log_probs_q),
            torch.zeros_like(probs_p),
        )
    elif input_format == "probs":
        # xlogy(0,0)=0；若 p>0,q=0，则结果自然为 +inf。
        term = torch.special.xlogy(p, p) - torch.special.xlogy(p, q)
    else:
        raise ValueError(f"未知 input_format: {input_format}")

    kl = term.sum(dim=-1)                                   # [...]
    return _reduce_kl(kl, reduction, mask)


def sampled_token_kl(
    log_probs_p: Tensor,
    log_probs_q: Tensor,
    *,
    estimator: Literal["k1", "k3"] = "k3",
    reduction: Literal["none", "mean", "sum", "batchmean"] = "none",
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    只有“已采样 token 的 log_probs”时估计 KL(P||Q)。

    前提：token 样本来自 P。
        log_probs_p/log_probs_q: [...]，不是 [...,V]。

    k1 = log P(a)-log Q(a)：期望无偏，但单点可为负。
    k3 = Q(a)/P(a)-log(Q/P)-1：期望同为 KL，单点非负、方差通常更稳。
    """
    if estimator == "k1":
        value = log_probs_p - log_probs_q
    elif estimator == "k3":
        log_probs_ratio_q_over_p = log_probs_q - log_probs_p
        value = (
            torch.exp(log_probs_ratio_q_over_p)
            - log_probs_ratio_q_over_p
            - 1
        )
    else:
        raise ValueError(f"未知 estimator: {estimator}")
    return _reduce_kl(value, reduction, mask)


# =============================================================================
# 11. 分块矩阵乘法与 FlashAttention
# =============================================================================


# -----------------------------------------------------------------------------
# 11.1 简单分块矩阵乘法
# -----------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _block_matmul_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        m,
        n,
        k,
        BLOCK_SIZE: tl.constexpr,
    ):
        """一个 Triton program 计算一个 [BLOCK_SIZE,BLOCK_SIZE] 输出块。"""
        block_row = tl.program_id(axis=0)
        block_col = tl.program_id(axis=1)
        row_offsets = block_row * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_offsets = block_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        k_offsets = tl.arange(0, BLOCK_SIZE)

        accumulator = tl.zeros(
            (BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32
        )                                                    # [Bm,Bn]

        for tile_index in range(0, tl.cdiv(k, BLOCK_SIZE)):
            current_k = tile_index * BLOCK_SIZE + k_offsets
            a_ptrs = (
                a_ptr + row_offsets[:, None] * k + current_k[None, :]
            )                                                # [Bm,Bk]
            b_ptrs = (
                b_ptr + current_k[:, None] * n + col_offsets[None, :]
            )                                                # [Bk,Bn]

            # tl.load 得到的 A/B tile 会由 Triton 放入片上 SRAM 并复用。
            a_block = tl.load(
                a_ptrs,
                mask=(row_offsets[:, None] < m) & (current_k[None, :] < k),
                other=0.0,
            )                                                # [Bm,Bk]
            b_block = tl.load(
                b_ptrs,
                mask=(current_k[:, None] < k) & (col_offsets[None, :] < n),
                other=0.0,
            )                                                # [Bk,Bn]
            accumulator = tl.dot(a_block, b_block, accumulator)

        out_ptrs = (
            out_ptr + row_offsets[:, None] * n + col_offsets[None, :]
        )                                                    # [Bm,Bn]
        out_mask = (
            (row_offsets[:, None] < m) & (col_offsets[None, :] < n)
        )
        tl.store(out_ptrs, accumulator, mask=out_mask)


def block_matmul(
    a: Tensor,
    b: Tensor,
    block_size: int = 32,
) -> Tensor:
    """
    Triton tiled matmul。

    a:   [M,K]，CUDA contiguous tensor
    b:   [K,N]，CUDA contiguous tensor
    out: [M,N]
    """
    if triton is None:
        raise RuntimeError("block_matmul 需要 Triton 和 CUDA/ROCm 环境")

    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)  # [M,N]
    grid = (triton.cdiv(m, block_size), triton.cdiv(n, block_size))
    _block_matmul_kernel[grid](
        a,
        b,
        out,
        m,
        n,
        k,
        BLOCK_SIZE=block_size,
    )
    return out


# -----------------------------------------------------------------------------
# 11.2 简化版 FlashAttention
# -----------------------------------------------------------------------------


def flash_attention_simple(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    causal: bool = False,
    q_block_size: int = 64,
    kv_block_size: int = 64,
) -> Tensor:
    """
    简化版 FlashAttention：不显式保存 [B,N,Sq,Sk] 完整矩阵。

    q:   [B,N,Sq,D]
    k:   [B,N,Sk,D]
    v:   [B,N,Sk,Dv]
    out: [B,N,Sq,Dv]

    对每个 Q block，遍历 KV blocks，并在线维护：
        m   [B,N,Bq,1]：截至当前块的 row max
        l   [B,N,Bq,1]：截至当前块的 exp-sum
        acc [B,N,Bq,Dv]：未归一化的 P@V

    数学上是 exact attention；这里只实现 forward 逻辑，没有 dropout 和
    FlashAttention 生产版的 fused CUDA kernel / 重计算 backward。
    """
    b, n, sq, d = q.shape
    sk, dv = k.size(-2), v.size(-1)
    qf, kf, vf = q.float(), k.float(), v.float()
    output_blocks = []

    for qs in range(0, sq, q_block_size):
        qe = min(qs + q_block_size, sq)
        qb = qf[:, :, qs:qe]                              # [B,N,Bq,D]
        bq = qe - qs
        m = torch.full((b, n, bq, 1), float("-inf"), device=q.device)
        l = torch.zeros((b, n, bq, 1), device=q.device)
        acc = torch.zeros((b, n, bq, dv), device=q.device)

        for ks in range(0, sk, kv_block_size):
            ke = min(ks + kv_block_size, sk)
            kb = kf[:, :, ks:ke]                           # [B,N,Bk,D]
            vb = vf[:, :, ks:ke]                           # [B,N,Bk,Dv]
            scores = torch.matmul(qb, kb.transpose(-2, -1)) / math.sqrt(d)
            # scores: [B,N,Bq,Bk]

            if causal:
                # 与带 cache 的 causal 对齐：q[0] 的全局位置为 Sk-Sq。
                q_pos = torch.arange(qs, qe, device=q.device) + (sk - sq)
                k_pos = torch.arange(ks, ke, device=q.device)
                keep = k_pos[None, :] <= q_pos[:, None]    # [Bq,Bk]
                scores = scores.masked_fill(~keep[None, None], float("-inf"))

            block_max = scores.amax(dim=-1, keepdim=True)  # [B,N,Bq,1]
            new_m = torch.maximum(m, block_max)
            new_m_safe = torch.where(
                torch.isfinite(new_m), new_m, torch.zeros_like(new_m)
            )
            alpha = torch.where(
                torch.isfinite(m), torch.exp(m - new_m_safe), torch.zeros_like(m)
            )
            p_block = torch.where(
                torch.isfinite(scores),
                torch.exp(scores - new_m_safe),
                torch.zeros_like(scores),
            )
            acc = alpha * acc + torch.matmul(p_block, vb)  # [B,N,Bq,Dv]
            l = alpha * l + p_block.sum(dim=-1, keepdim=True)
            m = new_m

        out_block = torch.where(
            l > 0, acc / l.clamp_min(1e-30), torch.zeros_like(acc)
        )
        output_blocks.append(out_block)

    return torch.cat(output_blocks, dim=2).to(q.dtype)      # [B,N,Sq,Dv]


# =============================================================================
# 冒烟测试
# =============================================================================


def _smoke_test() -> None:
    torch.manual_seed(0)
    b, s, h, n = 2, 7, 16, 4
    d = h // n
    x = torch.randn(b, s, h)

    # SDPA 与 PyTorch 参考实现一致。
    q = torch.randn(b, n, s, d)
    k = torch.randn(b, n, s, d)
    v = torch.randn(b, n, s, d)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(sdpa(q, k, v, causal=True), ref, rtol=1e-5, atol=1e-6)

    # Triton 分块矩阵乘法与 torch.matmul 一致。
    if triton is not None and torch.cuda.is_available():
        mat_a = torch.randn(17, 19, device="cuda", dtype=torch.float16)
        mat_b = torch.randn(19, 23, device="cuda", dtype=torch.float16)
        blocked = block_matmul(mat_a, mat_b, block_size=16)
        torch.testing.assert_close(
            blocked, mat_a @ mat_b, rtol=1e-2, atol=1e-2
        )

    # FlashAttention 简版与普通 SDPA 一致。
    flash = flash_attention_simple(
        q, k, v, causal=True, q_block_size=3, kv_block_size=2
    )
    torch.testing.assert_close(flash, ref, rtol=1e-5, atol=1e-6)

    assert MultiHeadAttention(h, n)(x).shape == (b, s, h)
    assert GroupedQueryAttention(h, n, 2)(x).shape == (b, s, h)
    assert TransformerBlock(h, n, 4 * h)(x).shape == (b, s, h)
    assert TPBlock(h, n, 4 * h)(x).shape == (b, s, h)
    assert TPSPBlock(h, n, 4 * h)(x).shape == (b, s, h)

    # KV cache：分块 prefill/decode 等价于一次完整 causal forward。
    cached = MHAWithKVCache(h, n)
    y_full, _ = cached(x)
    y0, cache = cached(x[:, :4])
    y1, cache = cached(x[:, 4:], cache)
    torch.testing.assert_close(torch.cat((y0, y1), 1), y_full, rtol=1e-5, atol=1e-6)
    assert cache[0].shape == (b, n, s, d)

    # CP=1 时就是普通 attention。
    cp_out = cp_ring_attention(q, k, v, global_seqlen=s)
    torch.testing.assert_close(cp_out, ref, rtol=1e-5, atol=1e-6)

    # 三种 KL 输入格式结果一致。
    p_logits = torch.randn(3, 11)
    q_logits = torch.randn(3, 11)
    kl_logits = categorical_kl(p_logits, q_logits, input_format="logits")
    kl_logs = categorical_kl(
        p_logits.log_softmax(-1), q_logits.log_softmax(-1), input_format="log_probs"
    )
    kl_probs = categorical_kl(
        p_logits.softmax(-1), q_logits.softmax(-1), input_format="probs"
    )
    torch.testing.assert_close(kl_logits, kl_logs)
    torch.testing.assert_close(kl_logits, kl_probs)

    # safe_softmax 的全 mask 行不产生 NaN。
    z = torch.tensor([[0.0, 1.0], [float("-inf"), float("-inf")]])
    got = safe_softmax(z)
    assert torch.isfinite(got).all() and torch.equal(got[1], torch.zeros(2))

    # 8.1 PPO：先验证 GAE，再验证 policy/value loss 反向传播。
    bg, g, t = 2, 3, 5
    # 变长序列：有效长度 3 和 1，padding 不参与 GAE。
    gae_mask = torch.tensor([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    gae_adv, gae_returns = compute_gae(
        rewards=torch.ones(2, 3),
        values=torch.zeros(2, 3),
        mask=gae_mask,
        gamma=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(
        gae_adv,
        torch.tensor([
            [3.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
        ]),
    )
    torch.testing.assert_close(gae_returns, gae_adv)

    # Multi-turn：跨过 mask=0 的 user/tool token，连接下一个 action。
    multiturn_mask = torch.tensor([
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    ])
    multiturn_adv, multiturn_returns = compute_gae_multiturn(
        rewards=multiturn_mask.clone(),
        values=torch.zeros_like(multiturn_mask),
        mask=multiturn_mask,
        gamma=1.0,
        gae_lambda=1.0,
    )
    expected_multiturn_adv = torch.tensor([
        [0.0, 0.0, 0.0, 7.0, 6.0, 0.0, 0.0, 5.0, 4.0, 3.0, 0.0, 2.0, 1.0]
    ])
    torch.testing.assert_close(multiturn_adv, expected_multiturn_adv)
    torch.testing.assert_close(multiturn_returns, multiturn_adv)

    ppo_mask = torch.ones(bg, t)
    old_values = torch.randn(bg, t)
    advantages, returns = compute_gae(
        rewards=torch.randn(bg, t),
        values=old_values,
        mask=ppo_mask,
    )
    ppo_log_probs = torch.randn(bg, t, requires_grad=True)
    values = torch.randn(bg, t, requires_grad=True)
    ppo_out = ppo_loss(
        ppo_log_probs,
        torch.randn(bg, t),
        advantages,
        ppo_mask,
        values=values,
        old_values=old_values,
        returns=returns,
    )
    ppo_out["loss"].backward()
    assert torch.isfinite(ppo_log_probs.grad).all()
    assert torch.isfinite(values.grad).all()

    # 8.2-8.4 GRPO / GSPO / CISPO：loss 有限且可反向传播。
    rewards = torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0]])
    mask = torch.ones(bg, g, t)
    old_log_probs = torch.randn(bg, g, t)
    for loss_fn in (grpo_loss, gspo_loss, cispo_loss):
        current_log_probs = old_log_probs.clone().requires_grad_(True)
        loss = loss_fn(current_log_probs, old_log_probs, rewards, mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(current_log_probs.grad).all()

    print("All smoke tests passed.")


if __name__ == "__main__":
    _smoke_test()
