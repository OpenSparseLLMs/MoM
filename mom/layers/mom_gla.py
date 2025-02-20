# -*- coding: utf-8 -*-
# Copyright (c) 2024, Songlin Yang, Yu Zhang


from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.activations import ACT2FN
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

if TYPE_CHECKING:
    from fla.models.utils import Cache

def transform(x: torch.Tensor, routing_mask: torch.Tensor, num_memories: int, selected_memories: torch.Tensor, capacity: float):
    '''
    Transform input sequences into memory-organized chunks with capacity constraints.
    
    Processes input sequences by routing tokens to designated memory states according to routing_mask,
    sorts tokens by memory assignments, handles token truncation/padding based on memory capacity,
    and returns memory-aligned tensors for parallel processing.

    Key operations:
    1. Expands input tensors when multiple memories are selected per token (top-k routing)
    2. Sorts tokens globally by (batch_idx, memory_idx) to group memory-assigned tokens
    3. Applies capacity-aware truncation (left-truncate oldest tokens when exceeding capacity)
    4. Pads memory chunks to uniform length for tensorization

    Args:
        x: Input hidden states
            Shape: (batch_size, seq_len, hidden_size)
        routing_mask: Binary mask indicating active memory assignments
            Shape: (batch_size, seq_len, num_memories)
        num_memories: Total number of memories per batch
        selected_memories: Memory indices assigned to each token. When using top-k routing,
            this contains k memory indices per token (k >= 1)
            Shape: (batch_size, seq_len) for k=1 or (batch_size, seq_len, topk) for k>1
        capacity: Scaling factor for memory capacity calculation. Actual capacity per memory is
            ceil(seq_len * capacity), maintaining proportional capacity to sequence length

    Returns:
        transformed_x: Memory-organized tensor with zero-padded capacity alignment
            Shape: (num_memories, batch_size, capacity_len, hidden_size)
        truncation_indices: Original indices used for gathering tokens after capacity truncation
            Shape: (batch*num_memories, max_len)
        sorted_indices: Sorting indices used to group tokens by memory assignments
            Shape: (batch_size*seq_len*topk)
        max_len: Maximum tokens per memory
        mask: Boolean mask indicating valid (non-padded) positions in transformed_x
            Shape: (batch*num_memories, max_len)
    '''
    if selected_memories.dim() == 3:
        # (batch, seq, topk)
        topk = selected_memories.shape[2]
        # x (batch, seq, hidden)
        x = x.repeat_interleave(topk, dim=1)
        # x (batch, seq * topk, hidden)
        # (batch, seq, topk)
        selected_memories = selected_memories.reshape(selected_memories.shape[0], -1)
        # (batch, seq * topk)

    b, s, d = x.shape
    x_flat = x.reshape(b * s, d)  # [b*s, d]

    with torch.no_grad():
        batch_indices = torch.arange(b, device=x.device).unsqueeze(-1)
        batch_indices = batch_indices.expand(b, s).reshape(-1)
        # (b * s)
        memories_flat = selected_memories.reshape(-1)  # [b*s]

        combined = batch_indices * (memories_flat.max() + 1) + memories_flat
        sorted_indices = combined.argsort()

    x_sorted = x_flat[sorted_indices]  # [b*s, d]
    # (b*s, hidden) -> (b, s, hidd)
    with torch.no_grad():
        # routing_mask (b, s, num_memories)
        batch_memory_tokens = routing_mask.sum(dim=1)
        # (b, num_memories)
        offset = batch_memory_tokens.cumsum(dim=1)
        memory_batch_offset = offset.transpose(0,1)
        batch_offset = torch.arange(0, b*s, s, device=offset.device)
        memory_batch_offset += batch_offset
        flatten_offset = memory_batch_offset.transpose(0, 1).reshape(-1)
        lengths = torch.concat([flatten_offset[:1], flatten_offset[1:] - flatten_offset[:-1]], dim=0)
        max_len = lengths.max()
        capacity_len = math.ceil(s / topk * capacity)
        max_len = min(max_len, capacity_len)

        indices = torch.arange(max_len, device=flatten_offset.device).unsqueeze(0).expand(b*num_memories, -1) + torch.cat([torch.tensor([0], device=flatten_offset.device), flatten_offset[:-1]], dim=0).unsqueeze(1)
        # discard tokens exceed capacity and is far from now
        # left pad
        truncation_indices = indices + batch_memory_tokens.reshape((-1,)).unsqueeze(-1) - max_len
        mask = torch.bitwise_and(truncation_indices < flatten_offset.unsqueeze(-1), truncation_indices >= 0)
        mask = torch.bitwise_and(mask, truncation_indices >= torch.cat((torch.zeros((1,), dtype=flatten_offset.dtype, device=flatten_offset.device), flatten_offset[:-1])).unsqueeze(-1))
        truncation_indices = torch.where(mask, truncation_indices, torch.zeros_like(truncation_indices))

    gathered_x = torch.gather(x_sorted, 0, truncation_indices.reshape(-1).unsqueeze(-1).expand(-1, d))
    transformed_x = gathered_x.reshape(b * num_memories, -1, d)
    transformed_x = transformed_x * mask.unsqueeze(-1).expand_as(transformed_x)
    pad_x = torch.zeros((b * num_memories, capacity_len-max_len, d), dtype=transformed_x.dtype, device=transformed_x.device)
    # left pad
    transformed_x = torch.cat((pad_x, transformed_x), dim=1).reshape((b, num_memories, capacity_len, d)).transpose(0, 1)
    # truncation_indices += capacity_len-max_len

    return transformed_x, truncation_indices, sorted_indices, max_len, mask
    # (num_memories, batch, seq, hidden)

# @torch.jit.script
def reconstruct(transformed_x, indices: torch.Tensor, sorted_indices: torch.Tensor, batch_size: int, seq_len: int, topk: int, routing_weights: torch.Tensor, mask: torch.Tensor):
    '''
    Reconstruct and mix transformed outputs back into the original input sequence shape.

    Key operations:
    1. Reshapes and transposes `transformed_x` to prepare for scattering.
    2. Applies the `mask` to zero out invalid positions.
    3. Uses `torch.scatter_add_` to scatter and sum the transformed outputs back to their original positions based on `indices`.
    4. Rearranges the scattered outputs using `sorted_indices` to ensure correct ordering.
    5. Applies the `routing_weights` to weight the outputs.
    6. Sums over the `topk` dimension to produce the final reconstructed output.

    Args:
        transformed_x (torch.Tensor):
            The transformed output tensor from memory units or experts.
            Shape: (num_memories, batch_size, capacity_len, hidden_size)
        indices (torch.Tensor):
            Indices used for scattering the transformed outputs back to their corresponding positions.
            Shape: (batch*num_memories, max_len)
        sorted_indices (torch.Tensor):
            Sorting indices used to rearrange the scattered outputs back into the original sequence order.
            Shape: (batch_size*seq_len*topk)
        batch_size (int):
            The size of the batch.
        seq_len (int):
            The length of the input sequence.
        topk (int):
            The number of top elements selected (`topk`) per token during the selection process.
        routing_weights (torch.Tensor):
            Routing weights assigned to the top-k selected outputs when reconstructing the final output.
            Shape: (batch_size, seq_len, topk)
        mask (torch.Tensor):
            Boolean mask indicating valid positions in the sequence.
            Shape: (batch*num_memories, max_len)

    Returns:
        restored_x (torch.Tensor):
            The reconstructed output tensor in the original input sequence shape.
            Shape: (batch_size, seq_len, hidden_size)
    '''
    transformed_x = transformed_x.transpose(0, 1).reshape((-1, transformed_x.shape[2], transformed_x.shape[3], transformed_x.shape[4]))
    b, s, k, h, d = batch_size, seq_len, topk, transformed_x.shape[2], transformed_x.shape[3]
    gathered_x = transformed_x.reshape((transformed_x.shape[0] * transformed_x.shape[1], transformed_x.shape[2], transformed_x.shape[3]))
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


class MomGatedLinearAttention(nn.Module):
    r"""
    The layer implementaion for [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635).  # noqa

    Args:
        mode (str, Optional):
            Which GLA kernel to use.
            Currently available: `chunk`, `fused_recurrent`, and `fused_chunk`.
            Default: `chunk`.
        hidden_size (int, Optional):
            The hidden size of the input. Default: 1024.
        expand_k (float, Optional):
            The expansion ratio for the key dim. Default: 0.5.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 1.0.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_kv_heads (int, Optional):
            The number of key/value heads, used for MQA. Default: None.
        feature_map (str, Optional):
            Feature map function applied to queries/keys. Default: None.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `False`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        use_output_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        gate_fn (str, Optional):
            The activation function for the output gate. Default: `swish`.
        elementwise_affine (bool, Optional):
            If `True`, applies elementwise affine to LayerNorm with learnable parameters. Default: `True`.
        norm_eps (float, Optional):
            The epsilon value for the layernorm/rmsnorm layer. Default: 1e-5.
        gate_logit_normalizer (int, Optional):
            The normalizer for the gate logits, appied after `logsigmoid`. Default: 16.
        gate_low_rank_dim (int, Optional):
            The low rank dim for the gate projection. Default: 16.
        clamp_min (float, Optional):
            The minimum value for the gate logits. Default: None.
        fuse_norm (bool, Optional):
            Whether to fuse the norm and the output gate for better memory footprint. Default: `True`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
    """

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
        num_memories: int = 8,
        topk: int = 2,
        capacity: float = 1.0,
        shared_mem: bool = False,
        single_kv_proj: bool = False,
    ) -> MomGatedLinearAttention:
        super().__init__()
        self.num_memories = num_memories
        self.topk = topk
        self.capacity = capacity
        self.shared_mem = shared_mem
        self.single_kv_proj = single_kv_proj

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.feature_map_fn = ACT2FN[feature_map] if feature_map is not None else None

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        if self.single_kv_proj:
            self.shared_k = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
            self.shared_v = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        else:
            self.k_proj =  nn.ModuleList([nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False) for _ in range(self.num_memories)])
            self.v_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False) for _ in range(self.num_memories)])
            if self.shared_mem:
                self.shared_k = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
                self.shared_v = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        self.gate = nn.Linear(self.hidden_size, self.num_memories, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim_per_group, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim_per_group, conv_size, activation='silu')

        if self.single_kv_proj:
            self.shared_gk = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                        nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True))
        else:
            self.gk_proj =  nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                                        nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True))
                                                        for _ in range(self.num_memories)])
            if self.shared_mem:
                self.shared_gk = nn.Sequential(nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
                                        nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True))
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(hidden_size=self.head_v_dim, elementwise_affine=elementwise_affine, eps=norm_eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.gate_logit_normalizer = gate_logit_normalizer

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

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        # 🔍 topk gating
        router_logits = self.gate(hidden_states)  # (bsz, q_len, num_memories)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # (bsz, q_len, top_k_attn)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros((routing_weights.shape[0], routing_weights.shape[1], self.num_memories), dtype=routing_weights.dtype, device=routing_weights.device).scatter(-1, selected_memories, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        if self.use_output_gate:
            g = self.g_proj(hidden_states)

        shared_hidden_states = hidden_states
        hidden_states, indices, sorted_indices, max_len, mask = transform(hidden_states, routing_mask, self.num_memories, selected_memories, self.capacity)

        q = self.q_proj(hidden_states)
        if self.single_kv_proj:
            k = self.shared_k(hidden_states)
            v = self.shared_v(hidden_states)
            gk = self.shared_gk(hidden_states)
        else:
            k = torch.stack([k_expert(hidden_states[i]) for i, k_expert in enumerate(self.k_proj)], dim=0)
            v = torch.stack([v_expert(hidden_states[i]) for i, v_expert in enumerate(self.v_proj)], dim=0)
            gk = torch.stack([gk_expert(hidden_states[i]) for i, gk_expert in enumerate(self.gk_proj)], dim=0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            q, k, v = map(lambda x: rearrange(x, 'e b t d -> (e b) t d'), (q, k, v))
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
            q, k, v = map(lambda x: rearrange(x, '(e b) t d -> e b t d', b=batch_size), (q, k, v))

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))
        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])
        q = rearrange(q, 'e b t (h d) -> e b t h d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v, gk = (repeat(x, 'e b t (h d) -> e b t (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v, gk))
        else:
            k, v, gk = (rearrange(x, 'e b t (h d) -> e b t h d', h=self.num_kv_heads) for x in (k, v, gk))
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else [None for _ in range(self.num_memories + self.shared_mem)]
        if mode == 'fused_recurrent':
            o_list = [None for _ in range(self.num_memories)]
            for e in range(self.num_memories):
                o_e, state_e = fused_recurrent_gla(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    gk=gk[e],
                    initial_state=recurrent_state[e],
                    output_final_state=use_cache,
                    head_first=False
                )
                o_e = o_e[:,-max_len:,:,:]
                o_list[e] = o_e
                # only activated memory updates
                for token in range(state_e.shape[0]):
                    if q[e, token].any() and recurrent_state[e] is not None:
                        recurrent_state[e][token] = state_e[token]
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)

        elif mode == 'fused_chunk':
            o_list = [None for _ in range(self.num_memories)]
            for e in range(self.num_memories):
                o_e, state_e = fused_chunk_gla(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    g=gk[e],
                    initial_state=recurrent_state[e],
                    output_final_state=use_cache,
                    head_first=False
                )
                o_e = o_e[:,-max_len:,:,:]
                o_list[e] = o_e
                recurrent_state[e] = state_e
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)

        elif mode == 'chunk':
            o_list = [None for _ in range(self.num_memories)]
            for e in range(self.num_memories):
                o_e, state_e = chunk_gla(
                    q=q[e],
                    k=k[e],
                    v=v[e],
                    g=gk[e],
                    initial_state=recurrent_state[e],
                    output_final_state=use_cache,
                    head_first=False
                )
                o_e = o_e[:,-max_len:,:,:]
                o_list[e] = o_e
                recurrent_state[e] = state_e
            o_list = torch.stack(o_list, dim=0)
            o = reconstruct(o_list, indices=indices, sorted_indices=sorted_indices, batch_size=q.shape[1], seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)

        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        if self.shared_mem:
            shared_o = self.shared_o(shared_hidden_states, attention_mask, recurrent_state, use_cache)
            o += shared_o

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[2]
            )

        if self.use_output_gate:
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', h=self.num_heads)
                o = self.g_norm_swish_gate(o, g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values, router_logits

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size


    def shared_o(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        recurrent_state = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> torch.Tensor:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        if self.use_short_conv:
            raise NotImplementedError()
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            seq_idx=kwargs.get('seq_idx', None)
            q, conv_state_q = self.q_conv1d(x=self.q_proj(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache,seq_idx=seq_idx)
            k, conv_state_k = self.k_conv1d(x=self.shared_k(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache,seq_idx=seq_idx)
            v, conv_state_v = self.v_conv1d(x=self.shared_v(hidden_states),
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache,seq_idx=seq_idx)
        else:
            q = self.q_proj(hidden_states)
            k = self.shared_k(hidden_states)
            v = self.shared_v(hidden_states)
        gk = self.shared_gk(hidden_states)

        if self.feature_map_fn is not None:
            q, k = map(self.feature_map_fn, (q, k))
        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask[:, -v.shape[-2]:, None])
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)
        if self.num_kv_groups > 1:
            k, v, gk = (repeat(x, 'b t (h d) -> b t (h g) d', h=self.num_kv_heads, g=self.num_kv_groups) for x in (k, v, gk))
        else:
            k, v, gk = (rearrange(x, 'b t (h d) -> b t h d', h=self.num_kv_heads) for x in (k, v, gk))
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer

        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        offsets = kwargs.get('offsets', None)
        if mode == 'fused_recurrent':
            o, recurrent_state[-1] = fused_recurrent_gla(
                q=q,
                k=k,
                v=v,
                gk=gk,
                initial_state=recurrent_state[-1],
                output_final_state=use_cache,
                cu_seqlens=offsets,
                head_first=False
            )
        elif mode == 'fused_chunk':
            o, recurrent_state[-1] = fused_chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state[-1],
                output_final_state=use_cache,
                head_first=False
            )
        elif mode == 'chunk':
            o, recurrent_state[-1] = chunk_gla(
                q=q,
                k=k,
                v=v,
                g=gk,
                initial_state=recurrent_state[-1],
                output_final_state=use_cache,
                cu_seqlens=offsets,
                head_first=False
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        return o