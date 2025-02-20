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
        num_memories: int = 8,
        topk: int = 2,
        capacity: float = 1.0,
        shared_mem: bool = False,
        **kwargs
    ):
        super().__init__()
        self.num_memories = num_memories
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
        self.k_proj =  nn.ModuleList([nn.Linear(self.hidden_size, self.key_dim_per_group, bias=False) for _ in range(self.num_memories + self.shared_mem)])
        self.v_proj = nn.ModuleList([nn.Linear(self.hidden_size, self.value_dim_per_group, bias=False) for _ in range(self.num_memories + self.shared_mem)])
        self.gate = nn.Linear(self.hidden_size, self.num_memories, bias=False)
 

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
        router_logits = self.gate(x)  # (bsz, q_len, num_memories)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # (bsz, q_len, top_k_attn)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        if self.shared_mem:
            selected_memories = torch.cat((torch.full((selected_memories.shape[0], selected_memories.shape[1], 1), self.num_memories, device=selected_memories.device, dtype=selected_memories.dtype), selected_memories), dim=2)
            routing_weights = torch.cat((torch.full((routing_weights.shape[0], routing_weights.shape[1], 1), 1.0, device=routing_weights.device, dtype=routing_weights.dtype), routing_weights), dim=2)
        routing_weights = routing_weights.to(x.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros((routing_weights.shape[0], routing_weights.shape[1], self.num_memories + self.shared_mem), dtype=routing_weights.dtype, device=routing_weights.device).scatter(-1, selected_memories, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        batch_size, seq_len = x.shape[0], x.shape[1]
        x, indices, sorted_indices, max_len, mask = transform(x, routing_mask, self.num_memories + self.shared_mem, selected_memories, self.capacity)

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
            o_list = [None for _ in range(self.num_memories + self.shared_mem)]
            for e in range(self.num_memories + self.shared_mem):
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
            o_list = [None for _ in range(self.num_memories + self.shared_mem)]
            for e in range(self.num_memories + self.shared_mem):
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
            o_list = [None for _ in range(self.num_memories + self.shared_mem)]
            for e in range(self.num_memories + self.shared_mem):
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
