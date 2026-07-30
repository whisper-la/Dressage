from enum import KEEP
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

Tensor = torch.Tensor

def safe_softmax(x,dim=1):
    dtype = x.dtype
    x=x.float()
    
    # 取得最大值
    max_x = x.max(dim=dim,keepdim=True).values
    max_x = torch.where(torch.isfinite(max_x),max_x,0)

    # 求exp
    exp_x = torch.where(torch.isfinite(x),torch.exp(x - max_x),0)
    sum_exp_x = exp_x.sum(dim=dim,keepdim=True)
    sum_exp_x = sum_exp_x.clamp_min(1e-12)

    # 求概率
    prob = exp_x/sum_exp_x

    return prob.to(dtype)



def sdpa(q:Tensor,k:Tensor,v:Tensor,causal:bool=True)->Tensor:
    """
    最简 self-attention。

    q/k/v: [B,N,S,D]
    score: [B,N,S,S]
    out:   [B,N,S,D]
    """

    s,d = q.size(-2), q.size(-1)
    score = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        mask = torch.ones(s, s, dtype=torch.bool, device=q.device).tril()
        score = score.masked_fill(mask, float("-inf"))

    attention = safe_softmax(score)
    return attention @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int ,bias : bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x:Tensor,causal:bool=True)->Tensor:
        b,s,h = x.shape()
        qkv = self.qkv(x)
        q,k,v = qkv.chunk(3,dim=-1)


        # b,s,h -- b,n,s,d
        q= q.view(b,s,self.num_heads,self.head_dim).transpose(1,2)
        k= k.view(b,s,self.num_heads,self.head_dim).transpose(1,2)
        v= v.view(b,s,self.num_heads,self.head_dim).transpose(1,2)

        y=sdpa(q,k,v,causal)
        y = y.transpose(1,2).contiguous().view(b,s,h)

        return self.out_proj(y)



