# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                      fused_recurrent_gated_delta_rule)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1

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
        capacity_len = math.ceil(s / topk * capacity)
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


class MomGatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.
    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        num_experts: int = 8,
        topk: int = 2,
        capacity: float = 1.0,
        shared_mem: bool = False,
        single_expert: bool = False,
        **kwargs
    ) -> MomGatedDeltaNet:
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.capacity = capacity
        self.shared_mem = shared_mem
        self.single_expert = single_expert

        self.mode = mode

        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim * self.expand_v
        self.head_qk_dim = head_dim
        self.head_v_dim = head_dim * self.expand_v
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()

        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        if self.single_expert:
            self.shared_k = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.shared_v = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.shared_b = nn.Linear(hidden_size, self.num_heads, bias=False)
            self.shared_a = nn.Linear(hidden_size, self.num_heads, bias=False)
        else:
            self.k_proj =  nn.ModuleList([nn.Linear(self.hidden_size, self.key_dim, bias=False) for _ in range(self.num_experts)])
            self.v_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.value_dim, bias=False) for _ in range(self.num_experts)])
            self.b_proj =  nn.ModuleList([nn.Linear(self.hidden_size, self.num_heads, bias=False) for _ in range(self.num_experts)])
            self.a_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.num_heads, bias=False) for _ in range(self.num_experts)])
            if self.shared_mem:
                self.shared_k = nn.Linear(hidden_size, self.key_dim, bias=False)
                self.shared_v = nn.Linear(hidden_size, self.value_dim, bias=False)
                self.shared_b = nn.Linear(hidden_size, self.num_heads, bias=False)
                self.shared_a = nn.Linear(hidden_size, self.num_heads, bias=False)

        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
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
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # 🔍 topk gating
        router_logits = self.gate(hidden_states)  # (bsz, q_len, num_experts)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(scores, self.topk, dim=-1)  # (bsz, q_len, top_k_attn)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # if self.shared_mem:
        #     selected_experts = torch.cat((torch.full((selected_experts.shape[0], selected_experts.shape[1], 1), self.num_experts, device=selected_experts.device, dtype=selected_experts.dtype), selected_experts), dim=2)
        #     routing_weights = torch.cat((torch.full((routing_weights.shape[0], routing_weights.shape[1], 1), 1.0, device=routing_weights.device, dtype=routing_weights.dtype), routing_weights), dim=2)
        routing_weights = routing_weights.to(hidden_states.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros((routing_weights.shape[0], routing_weights.shape[1], self.num_experts), dtype=routing_weights.dtype, device=routing_weights.device).scatter(-1, selected_experts, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        if self.use_gate:
            o_g = self.g_proj(hidden_states)
        
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        shared_hidden_states = hidden_states
        hidden_states, indices, sorted_indices, max_len, mask = transform(hidden_states, routing_mask, self.num_experts, selected_experts, self.capacity)

        q = self.q_proj(hidden_states)
        if self.single_expert:
            k = self.shared_k(hidden_states)
            v = self.shared_v(hidden_states)
            beta = self.shared_b(hidden_states).sigmoid()
            g = -self.A_log.float().exp() * F.softplus(self.shared_a(hidden_states).float() + self.dt_bias)
        else:
            k = torch.stack([k_expert(hidden_states[i]) for i, k_expert in enumerate(self.k_proj)], dim=0)
            v = torch.stack([v_expert(hidden_states[i]) for i, v_expert in enumerate(self.v_proj)], dim=0)
            beta = torch.stack([b_expert(hidden_states[i]).sigmoid() for i, b_expert in enumerate(self.b_proj)], dim=0)
            g = torch.stack([-self.A_log.float().exp() * F.softplus(a_expert(hidden_states[i]).float() + self.dt_bias) for i, a_expert in enumerate(self.a_proj)], dim=0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = [None, None], [None, None], [None, None]
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[2]:].repeat_interleave(self.num_experts, 0) if attention_mask is not None else None
            seq_idx=kwargs.get('seq_idx', None)
            q, k, v = map(lambda x: rearrange(x, 'e b t d -> (e b) t d'), (q, k, v))
            q, conv_state_q[0] = self.q_conv1d(x=q,
                                            mask=conv_mask,
                                            cache=conv_state_q[0],
                                            output_final_state=use_cache,seq_idx=seq_idx)
            k, conv_state_k[0] = self.k_conv1d(x=k,
                                            mask=conv_mask,
                                            cache=conv_state_k[0],
                                            output_final_state=use_cache,seq_idx=seq_idx)
            v, conv_state_v[0] = self.v_conv1d(x=v,
                                            mask=conv_mask,
                                            cache=conv_state_v[0],
                                            output_final_state=use_cache,seq_idx=seq_idx)

            q, k, v = map(lambda x: rearrange(x, '(e b) t d -> e b t d', b=batch_size), (q, k, v))

        else:
            q, k, v = self.silu(q), self.silu(k), self.silu(v),

        q, k, v = map(lambda x: rearrange(x, 'e b t (h d) -> e b t h d', h=self.num_heads), (q, k, v))
        q = l2_norm(q)
        k = l2_norm(k)

        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[None, :, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[None, :, -g.shape[-2]:, None])

        recurrent_state = last_state['recurrent_state'] if last_state is not None else [None for _ in range(self.num_experts + self.shared_mem)]
        offsets = kwargs.get('offsets', None)
        if mode == 'chunk':
            o_list = [None for _ in range(self.num_experts)]
            for e in range(self.num_experts):
                o_e, state_e = chunk_gated_delta_rule(
                    q=q[e].to(dtype=torch.bfloat16),
                    k=k[e].to(dtype=torch.bfloat16),
                    v=v[e].to(dtype=torch.bfloat16),
                    g=g[e].to(dtype=torch.bfloat16),
                    beta=beta[e].to(dtype=torch.bfloat16),
                    initial_state=recurrent_state[e],
                    output_final_state=use_cache,
                    offsets=offsets,
                    head_first=False
                )
                o_e = o_e[:,-max_len:,:,:].to(dtype=q[e].dtype)
                o_list[e] = o_e
                recurrent_state[e] = state_e
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)

        elif mode == 'fused_recurrent':
            o_list = [None for _ in range(self.num_experts)]
            for e in range(self.num_experts):
                # only activated memory updates
                if not hidden_states[e, 0].any() and hidden_states.shape[1] == 1:
                    o_list[e] = torch.zeros_like(v[e,:,-max_len:,:,:])
                    continue
                o_e, state_e = fused_recurrent_gated_delta_rule(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    g=g[e],
                    beta=beta[e],
                    initial_state=recurrent_state[e],
                    output_final_state=use_cache,
                    offsets=offsets,
                    head_first=False
                )
                o_e = o_e[:,-max_len:,:,:]
                o_list[e] = o_e
                # recurrent_state[e] = state_e
                for batch in range(state_e.shape[0]):
                    if recurrent_state[e] is None:
                        recurrent_state[e] = state_e
                    elif hidden_states[e, batch].any():
                        recurrent_state[e][batch] = state_e[batch]
            o_list = torch.stack(o_list, dim=0)
        o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)

            # if q.shape[2] == 1:
            #     # single token
            #     flatten_q = q.flatten(0, 1)
            #     non_zero_mask = (flatten_q != 0).any(dim=(1,2,3))
            #     valid_ids = torch.where(non_zero_mask)[0]
            #     o_list, state_list = fused_recurrent_gated_delta_rule(
            #         q=q.flatten(0, 1)[non_zero_mask],
            #         k=k.flatten(0, 1)[non_zero_mask],
            #         v=v.flatten(0, 1)[non_zero_mask],
            #         g=g.flatten(0, 1)[non_zero_mask],
            #         beta=beta.flatten(0, 1)[non_zero_mask],
            #         initial_state=None if recurrent_state[0] is None else torch.cat(recurrent_state).flatten(0, 1)[:q.shape[0] * q.shape[1]][non_zero_mask],
            #         output_final_state=use_cache,
            #         offsets=offsets,
            #         head_first=False
            #     )
            #     o = torch.zeros_like(v).flatten(0, 1)
            #     o[valid_ids] = o_list
            #     o = o.reshape_as(v)[:,:,-max_len:,:,:]
            #     for e in range(self.num_experts):
            #         for batch in range(q.shape[1]):
            #             if recurrent_state[e] is None:
            #                 recurrent_state[e] = state_list[e*q.shape[1]:(e+1)*q.shape[1]]
            #             elif q[e, batch].any():
            #                 recurrent_state[e][batch] = state_list[e*q.shape[1] + batch]
            #     o = reconstruct(o, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)
            # else:
            #     o_list, state_list = fused_recurrent_gated_delta_rule(
            #         q=q.flatten(0, 1),
            #         k=k.flatten(0, 1),
            #         v=v.flatten(0, 1),
            #         g=g.flatten(0, 1),
            #         beta=beta.flatten(0, 1),
            #         initial_state=None if recurrent_state[0] is None else torch.cat(recurrent_state).flatten(0, 1),
            #         output_final_state=use_cache,
            #         offsets=offsets,
            #         head_first=False
            #     )
            #     o_list = o_list.reshape_as(v)[:,:,-max_len:,:,:]
            #     for e in range(self.num_experts):
            #         for batch in range(q.shape[1]):
            #             if recurrent_state[e] is None:
            #                 recurrent_state[e] = state_list[e*q.shape[1]:(e+1)*q.shape[1]]
            #             elif q[e, batch].any():
            #                 recurrent_state[e][batch] = state_list[e*q.shape[1] + batch]
            #     o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)

        if self.shared_mem:
            shared_o = self.shared_o(shared_hidden_states, attention_mask, recurrent_state, use_cache, conv_state_q, conv_state_k, conv_state_v)
            o += shared_o

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2]
            )

        if self.use_gate:
            o_g = rearrange(o_g, '... (h d) -> ... h d', h=self.num_heads)
            o = self.o_norm(o, o_g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values, router_logits


    def shared_o(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state = None,
        use_cache: Optional[bool] = False,
        conv_state_q = [None, None],
        conv_state_k = [None, None],
        conv_state_v = [None, None],
        **kwargs
    ) -> torch.Tensor:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        if self.use_short_conv:
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            seq_idx=kwargs.get('seq_idx', None)
            q, conv_state_q[1] = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_q[1],
                                            output_final_state=use_cache,seq_idx=seq_idx)
            k, conv_state_k[1] = self.k_conv1d(x=self.shared_k(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_k[1],
                                            output_final_state=use_cache,seq_idx=seq_idx)
            v, conv_state_v[1] = self.v_conv1d(x=self.shared_v(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_v[1],
                                            output_final_state=use_cache,seq_idx=seq_idx)
        else:
            q = self.silu(self.q_proj(hidden_states))
            k = self.silu(self.shared_k(hidden_states))
            v = self.silu(self.shared_v(hidden_states))

        q, k, v = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads), (q, k, v))
        q = l2_norm(q)
        k = l2_norm(k)
        beta = self.shared_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.shared_a(hidden_states).float() + self.dt_bias)

        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])

        offsets = kwargs.get('offsets', None)
        if mode == 'chunk':
            o, recurrent_state[-1] = chunk_gated_delta_rule(
                q=q.to(dtype=torch.bfloat16),
                k=k.to(dtype=torch.bfloat16),
                v=v.to(dtype=torch.bfloat16),
                g=g.to(dtype=torch.bfloat16),
                beta=beta.to(dtype=torch.bfloat16),
                initial_state=recurrent_state[-1],
                output_final_state=use_cache,
                offsets=offsets,
                head_first=False
            )
            o = o.to(dtype=q.dtype)
        elif mode == 'fused_recurrent':
            o, recurrent_state[-1] = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state[-1],
                output_final_state=use_cache,
                offsets=offsets,
                head_first=False
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        return o