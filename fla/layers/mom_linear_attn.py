# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import RMSNorm
from fla.modules.feature_map import (DPFPFeatureMap, HadamardFeatureMap,
                                     HedgehogFeatureMap, T2RFeatureMap)
from fla.ops.linear_attn import (chunk_linear_attn, fused_chunk_linear_attn,
                                 fused_recurrent_linear_attn)

def transform(x: torch.Tensor, routing_mask: torch.Tensor, num_experts: int, selected_experts: torch.Tensor, capacity: float):
    '''
    transform the hidden_states into chunks by experts (expert_batch, selected_len, hidden_size)
    expert_batch may be close to experts * orginal_batch

    x: (batch_size, seq_len, hidden_size)
    routing_mask: (batch_size, seq_len, num_experts)
    '''
    if selected_experts.dim() == 3:
        topk = selected_experts.shape[2]
        x = x.repeat_interleave(topk, dim=1) 
        selected_experts = selected_experts.reshape(selected_experts.shape[0], -1)

    b, s, d = x.shape
    x_flat = x.reshape(b * s, d)  # [b*s, d]

    with torch.no_grad():
        batch_indices = torch.arange(b, device=x.device).unsqueeze(-1)
        batch_indices = batch_indices.expand(b, s).reshape(-1)
        
        experts_flat = selected_experts.reshape(-1)  # [b*s]

        combined = batch_indices * (experts_flat.max() + 1) + experts_flat
        sorted_indices = combined.argsort()

    x_sorted = x_flat[sorted_indices]  # [b*s, d]

    with torch.no_grad():
        batch_expert_tokens = routing_mask.sum(dim=1)
        offset = batch_expert_tokens.cumsum(dim=1)
        expert_batch_offset = offset.transpose(0,1)
        batch_offset = torch.arange(0, b*s, s, device=offset.device)
        expert_batch_offset += batch_offset
        flatten_offset = expert_batch_offset.transpose(0, 1).reshape(-1)
        lengths = torch.concat([flatten_offset[:1], flatten_offset[1:] - flatten_offset[:-1]], dim=0)
        max_len = lengths.max()
        capacity_len = int(s / topk * capacity)
        max_len = min(max_len, capacity_len)

        indices = torch.arange(max_len, device=flatten_offset.device).unsqueeze(0).expand(b*num_experts, -1) + torch.cat([torch.tensor([0], device=flatten_offset.device), flatten_offset[:-1]], dim=0).unsqueeze(1)
        # discard tokens exceed capacity and is far from now
        truncation_indices = indices + torch.max(torch.zeros((1,), dtype=batch_expert_tokens.dtype, device=batch_expert_tokens.device), batch_expert_tokens.reshape((-1,)) - max_len).unsqueeze(-1)
        mask = truncation_indices < flatten_offset.unsqueeze(-1)
        truncation_indices = torch.where(mask, truncation_indices, torch.zeros_like(truncation_indices))

    gathered_x = torch.gather(x_sorted, 0, truncation_indices.reshape(-1).unsqueeze(-1).expand(-1, d))
    ret_x = gathered_x.reshape(b * num_experts, -1, d)
    # with torch.no_grad():
    #     mask = mask.unsqueeze(-1)
    #     mask_x = mask.expand_as(ret_x).bitwise_not()
    #     ret_x[mask_x] = 0.0
    ret_x = ret_x * mask.unsqueeze(-1).expand_as(ret_x)
    pad_x = torch.zeros((b * num_experts, capacity_len-max_len, d), dtype=ret_x.dtype, device=ret_x.device)
    ret_x = torch.cat((ret_x, pad_x), dim=1).reshape((b, num_experts, capacity_len, d)).transpose(0, 1)

    return ret_x, truncation_indices, sorted_indices, max_len, mask
    
# @torch.jit.script
def reconstruct(re_x, indices: torch.Tensor, sorted_indices: torch.Tensor, batch_size: int, seq_len: int, topk: int, routing_weights: torch.Tensor, mask: torch.Tensor):
    re_x = re_x.transpose(0, 1).reshape((-1, re_x.shape[2], re_x.shape[3], re_x.shape[4]))
    b, s, k, h, d = batch_size, seq_len, topk, re_x.shape[2], re_x.shape[3]
    gathered_x = re_x.reshape((re_x.shape[0] * re_x.shape[1], re_x.shape[2], re_x.shape[3]))
    # with torch.no_grad():
    #     gathered_x[mask.reshape(-1).bitwise_not().unsqueeze(-1).unsqueeze(-1).expand_as(gathered_x)]
    # gathered_x = torch.where(mask.reshape(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gathered_x), gathered_x, torch.zeros_like(gathered_x))
    mask_expanded = mask.reshape(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gathered_x)
    gathered_x = gathered_x * mask_expanded

    assert (indices >= 0).all(), "Indices should be non-negative"

    resortd_x = torch.zeros((b * s * k, h, d) ,device=gathered_x.device, dtype=gathered_x.dtype).scatter_add_(
        0,
        indices.reshape(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, h, d),
        gathered_x,
    )
    assert (indices < resortd_x.size(0)).all(), "Indices should be less than resortd_x size"

    inverse_indices = sorted_indices.argsort()
    rearranged_x_flat = resortd_x[inverse_indices]
    restored_x = rearranged_x_flat.reshape((b, s * k, h, d))
    restored_x = restored_x.reshape(b, s, k, h, d) * routing_weights.reshape(b, s, k).unsqueeze(-1).unsqueeze(-1)
    restored_x = restored_x.sum(dim=2)
    return restored_x

class MomLinearAttention(nn.Module):
    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: str = 1024,
        expand_k: int = 1.0,
        expand_v: int = 1.0,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        feature_map: str = 'elementwise_product',
        tie_feature_map_qk: bool = False,
        output_norm: str = 'rmsnorm',
        norm_q: bool = False,
        norm_k: bool = False,
        # standard linear attention normalization
        do_feature_map_norm: bool = False,
        elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        num_experts: int = 8,
        topk: int = 2,
        capacity: float = 1.0,
        shared_mem: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.capacity = capacity
        self.shared_mem = shared_mem

        self.hidden_size = hidden_size
        self.mode = mode
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups

        assert mode in ['chunk', 'fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.do_feature_map_norm = do_feature_map_norm

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elementwise_product':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HadamardFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HadamardFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'dpfp':
            self.feature_map_q = DPFPFeatureMap(head_dim=self.head_qk_dim)
            self.feature_map_k = DPFPFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu
            self.feature_map_k = elu

        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()

        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported feature map `{feature_map}`.")

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        # self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        # self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        self.k_proj =  nn.ModuleList([nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False) for _ in range(self.num_experts + self.shared_mem)])
        self.v_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False) for _ in range(self.num_experts + self.shared_mem)])
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
 

        if output_norm == 'rmsnorm':
            self.norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
        elif output_norm == 'identity':
            self.norm = nn.Identity()
        else:
            raise NotImplementedError(f"Not supported output norm `{output_norm}`.")

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.norm_q = norm_q
        self.norm_k = norm_k

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, x):
        mode = self.mode
    
        # 🔍 topk gating
        router_logits = self.gate(x)  # (bsz, q_len, num_experts)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(scores, self.topk, dim=-1)  # (bsz, q_len, top_k_attn)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        if self.shared_mem:
            selected_experts = torch.cat((torch.full((selected_experts.shape[0], selected_experts.shape[1], 1), self.num_experts, device=selected_experts.device, dtype=selected_experts.dtype), selected_experts), dim=2)
            routing_weights = torch.cat((torch.full((routing_weights.shape[0], routing_weights.shape[1], 1), 1.0, device=routing_weights.device, dtype=routing_weights.dtype), routing_weights), dim=2)
        routing_weights = routing_weights.to(x.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros((routing_weights.shape[0], routing_weights.shape[1], self.num_experts + self.shared_mem), dtype=routing_weights.dtype, device=routing_weights.device).scatter(-1, selected_experts, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        batch_size, seq_len = x.shape[0], x.shape[1]
        x, indices, sorted_indices, max_len, mask = transform(x, routing_mask, self.num_experts + self.shared_mem, selected_experts, self.capacity)

        q = self.q_proj(x)
        k = torch.stack([k_expert(x[i]) for i, k_expert in enumerate(self.k_proj)], dim=0)
        v = torch.stack([v_expert(x[i]) for i, v_expert in enumerate(self.v_proj)], dim=0)

        # k = self.k_proj(x)
        # v = self.v_proj(x)

        q = rearrange(q, '... (h d) -> ... h d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v = (repeat(x, '... (h d) -> ... (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v))
        else:
            k, v = (rearrange(x, '... (h d) -> ... h d', h=self.num_kv_heads) for x in (k, v))

        q = self.feature_map_q(q)
        k = self.feature_map_k(k)

        if self.norm_q:
            q = q / (q.sum(-1, True) + 1e-4)
        if self.norm_k:
            k = k / (k.sum(-1, True) + 1e-4)

        if mode == 'chunk':
            o_list = [None for _ in range(self.num_experts + self.shared_mem)]
            for e in range(self.num_experts + self.shared_mem):
                o_e, final_state = chunk_linear_attn(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    normalize=self.do_feature_map_norm,
                    head_first=False
                )
                o_e = o_e[:,:max_len,:,:]
                o_list[e] = o_e
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk + self.shared_mem, routing_weights=routing_weights, mask=mask)

        elif mode == 'fused_chunk':
            o_list = [None for _ in range(self.num_experts + self.shared_mem)]
            for e in range(self.num_experts + self.shared_mem):
                o_e, final_state = fused_chunk_linear_attn(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    normalize=self.do_feature_map_norm,
                )
                o_e = o_e[:,:max_len,:,:]
                o_list[e] = o_e
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk + self.shared_mem, routing_weights=routing_weights, mask=mask)

        elif mode == 'fused_recurrent':
            o_list = [None for _ in range(self.num_experts + self.shared_mem)]
            for e in range(self.num_experts + self.shared_mem):
                o_e, final_state = fused_recurrent_linear_attn(
                    q=q,
                    k=k,
                    v=v,
                    normalize=self.do_feature_map_norm,
                )
                o_e = o_e[:,:max_len,:,:]
                o_list[e] = o_e
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk + self.shared_mem, routing_weights=routing_weights, mask=mask)

        else:
            raise NotImplementedError
        
        o = self.norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)
        return o, router_logits
