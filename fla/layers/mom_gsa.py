# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import time
import warnings
from typing import TYPE_CHECKING, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules import RMSNorm, ShortConvolution
from fla.modules.activations import swish
from fla.modules.feature_map import (ReLUFeatureMap, SwishFeatureMap,
                                     T2RFeatureMap)
from fla.modules.layernorm import rms_norm_linear
from fla.ops.gsa import chunk_gsa, fused_recurrent_gsa


if TYPE_CHECKING:
    from fla.models.utils import Cache

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
        # left pad
        truncation_indices = indices + batch_expert_tokens.reshape((-1,)).unsqueeze(-1) - max_len
        mask = torch.bitwise_and(truncation_indices < flatten_offset.unsqueeze(-1), truncation_indices >= 0)
        mask = torch.bitwise_and(mask, truncation_indices >= torch.cat((torch.zeros((1,), dtype=flatten_offset.dtype, device=flatten_offset.device), flatten_offset[:-1])).unsqueeze(-1))
        truncation_indices = torch.where(mask, truncation_indices, torch.zeros_like(truncation_indices))

    gathered_x = torch.gather(x_sorted, 0, truncation_indices.reshape(-1).unsqueeze(-1).expand(-1, d))
    ret_x = gathered_x.reshape(b * num_experts, -1, d)
    # with torch.no_grad():
    #     mask = mask.unsqueeze(-1)
    #     mask_x = mask.expand_as(ret_x).bitwise_not()
    #     ret_x[mask_x] = 0.0
    ret_x = ret_x * mask.unsqueeze(-1).expand_as(ret_x)
    pad_x = torch.zeros((b * num_experts, capacity_len-max_len, d), dtype=ret_x.dtype, device=ret_x.device)
    # left pad
    ret_x = torch.cat((pad_x, ret_x), dim=1).reshape((b, num_experts, capacity_len, d)).transpose(0, 1)
    # truncation_indices += capacity_len-max_len

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

# global_step = 0

class MomGatedSlotAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_slots: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        norm_first: bool = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        use_norm: bool = True,
        layer_idx: Optional[int] = None,
        scale: Optional[float] = 1.,
        num_experts: int = 8,
        topk: int = 2,
        capacity: float = 1.0,
        shared_mem: bool = False,
        **kwargs
    ) -> MomGatedSlotAttention:
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.capacity = capacity
        self.shared_mem = shared_mem

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.scale = scale

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots
        self.norm_first = norm_first

        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        if norm_first:
            self.norm = RMSNorm(self.hidden_size, eps=norm_eps)
        self.register_module('feature_map', None)
        if feature_map == 'swish':
            self.feature_map = SwishFeatureMap()
        elif feature_map == 'relu':
            self.feature_map = ReLUFeatureMap()
        elif feature_map == 't2r':
            self.feature_map = T2RFeatureMap(self.head_k_dim, self.head_k_dim)
        else:
            raise NotImplementedError(f"Feature map `{feature_map}` is not supported now.")

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj =  nn.ModuleList([nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False) for _ in range(self.num_experts + self.shared_mem)])
        self.v_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False) for _ in range(self.num_experts + self.shared_mem)])
        self.f_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False) for _ in range(self.num_experts + self.shared_mem)])
        # self.k_proj =  nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False)
        # self.f_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.num_slots, bias=False)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        self.g_norm = RMSNorm(self.hidden_size, elementwise_affine, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        if self.norm_first:
            hidden_states = self.norm(hidden_states)

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # 🔍 topk gating
        router_logits = self.gate(hidden_states)  # (bsz, q_len, num_experts)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(scores, self.topk, dim=-1)  # (bsz, q_len, top_k_attn)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        if self.shared_mem:
            selected_experts = torch.cat((torch.full((selected_experts.shape[0], selected_experts.shape[1], 1), self.num_experts, device=selected_experts.device, dtype=selected_experts.dtype), selected_experts), dim=2)
            routing_weights = torch.cat((torch.full((routing_weights.shape[0], routing_weights.shape[1], 1), 1.0, device=routing_weights.device, dtype=routing_weights.dtype), routing_weights), dim=2)
        routing_weights = routing_weights.to(hidden_states.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros((routing_weights.shape[0], routing_weights.shape[1], self.num_experts + self.shared_mem), dtype=routing_weights.dtype, device=routing_weights.device).scatter(-1, selected_experts, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        hidden_states, indices, sorted_indices, max_len, mask = transform(hidden_states, routing_mask, self.num_experts + self.shared_mem, selected_experts, self.capacity)

        q = self.q_proj(hidden_states)
        k = torch.stack([k_expert(hidden_states[i]) for i, k_expert in enumerate(self.k_proj)], dim=0)
        v = torch.stack([v_expert(hidden_states[i]) for i, v_expert in enumerate(self.v_proj)], dim=0)
        f = torch.stack([f_expert(hidden_states[i]) for i, f_expert in enumerate(self.f_proj)], dim=0)
        # k = self.k_proj(hidden_states)
        # v = self.v_proj(hidden_states)
        # f = self.f_proj(hidden_states)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            q, conv_state_q = self.q_conv1d(x=q,
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache)
            k, conv_state_k = self.k_conv1d(x=k,
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache)
            v, conv_state_v = self.v_conv1d(x=v,
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache)

        q = rearrange(q, 'e b t (h d) -> e b t h d', h=self.num_heads)
        k = rearrange(k, 'e b t (h d) -> e b t h d', h=self.num_kv_heads)
        v = rearrange(v, 'e b t (h d) -> e b t h d', h=self.num_kv_heads)
        # f = rearrange(f, 'e b t (h m) -> e b t h m', h=self.num_kv_heads)
        f = rearrange(f, 'e b t (h m) -> e b t h m', h=self.num_kv_heads)

        if self.feature_map is not None:
            q, k = map(lambda x: self.feature_map(x), (q, k))
        v = swish(v)

        f = F.logsigmoid(f) / self.gate_logit_normalizer
        s = (1 - f.exp()).to(f.dtype)
        # dealing with left-padding
        if attention_mask is not None:
            s = s.mul_(attention_mask[None, :, -s.shape[2]:, None, None])
            v = v.mul_(attention_mask[None, :, -v.shape[2]:, None, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else [None for _ in range(self.num_experts + self.shared_mem)]

        if mode == 'fused_recurrent':
            o_list = [None for _ in range(self.num_experts + self.shared_mem)]
            for e in range(self.num_experts + self.shared_mem):
                o_e, state_e = fused_recurrent_gsa(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    s=s[e],
                    g=f[e],
                    initial_state=recurrent_state[e],
                    output_final_state=use_cache,
                    scale=self.scale,
                    head_first=False
                )
                o_e = o_e[:,-max_len:,:,:]
                o_list[e] = o_e
                if len(state_e) == 1:
                    state_e = state_e[0]
                # recurrent_state[e] = state_e
                # only activated memory updates
                for token in range(state_e[0].shape[0]):
                    if q[e, token].any() and recurrent_state[e] is not None:
                        recurrent_state[e][0][token] = state_e[0][token]
                        recurrent_state[e][1][token] = state_e[1][token]
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk + self.shared_mem, routing_weights=routing_weights, mask=mask)
        elif mode == 'chunk':
            o_list = [None for _ in range(self.num_experts + self.shared_mem)]
            for e in range(self.num_experts + self.shared_mem):
                o_e, state_e = chunk_gsa(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    s=s[e],
                    g=f[e],
                    initial_state=recurrent_state[e],
                    output_final_state=use_cache,
                    scale=self.scale,
                    head_first=False
                )
                o_e = o_e[:,-max_len:,:,:]
                o_list[e] = o_e
                if len(state_e) == 1:
                    state_e = state_e[0]
                recurrent_state[e] = state_e
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk + self.shared_mem, routing_weights=routing_weights, mask=mask)

        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2]
            )

        o = rearrange(o, 'b t h d -> b t (h d)')
        o = rms_norm_linear(swish(o), self.g_norm.weight, self.g_norm.bias, self.o_proj.weight, self.o_proj.bias)

        return o, None, past_key_values, router_logits

    def state_size(self, *args, **kwargs) -> int:
        return 2 * self.num_slots * self.hidden_size
