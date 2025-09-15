# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mom.models.mom_linear_attn.configuration_mom_linear_attn import \
    MomLinearAttentionConfig
from mom.models.mom_linear_attn.modeling_mom_linear_attn import (
    MomLinearAttentionForCausalLM, MomLinearAttentionModel)

AutoConfig.register(MomLinearAttentionConfig.model_type, MomLinearAttentionConfig)
AutoModel.register(MomLinearAttentionConfig, MomLinearAttentionModel)
AutoModelForCausalLM.register(MomLinearAttentionConfig, MomLinearAttentionForCausalLM)

__all__ = ['MomLinearAttentionConfig', 'MomLinearAttentionForCausalLM', 'MomLinearAttentionModel']
