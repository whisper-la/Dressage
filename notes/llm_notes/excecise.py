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

import torch

def ppo_loss(
    log_probs,        # [B, T] 当前 policy 的 log prob
    old_log_probs,    # [B, T] 采样时 policy 的 log prob
    advantages,       # [B, T] 或 [B]
    values,           # [B, T] 当前 value
    returns,          # [B, T] value target
    response_mask,    # [B, T]
    clip_eps=0.2,
    vf_coef=0.5,
    entropy=None,
    ent_coef=0.01,
):
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    ratio = torch.exp(log_probs - old_log_probs)

    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2)

    value_loss = (values - returns).pow(2)

    loss = policy_loss + vf_coef * value_loss

    if entropy is not None:
        loss = loss - ent_coef * entropy

    loss = loss * response_mask
    return loss.sum() / response_mask.sum()


import torch

def grpo_loss(
    log_probs,        # [B, G, T] 当前 policy 的 log prob
    old_log_probs,    # [B, G, T] 采样时 policy 的 log prob
    ref_log_probs,    # [B, G, T] reference model 的 log prob
    rewards,          # [B, G]
    response_mask,    # [B, G, T]
    clip_eps=0.2,
    beta=0.04,
    eps=1e-8,
):
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True)

    advantages = (rewards - group_mean) / (group_std + eps)
    advantages = advantages.detach().unsqueeze(-1)

    ratio = torch.exp(log_probs - old_log_probs)

    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2)

    log_ratio_ref = ref_log_probs - log_probs
    kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1

    loss = policy_loss + beta * kl

    loss = loss * response_mask
    return loss.sum() / response_mask.sum()