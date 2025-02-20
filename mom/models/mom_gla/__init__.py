# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mom.models.mom_gla.configuration_mom_gla import MomGLAConfig
from mom.models.mom_gla.modeling_mom_gla import MomGLAForCausalLM, MomGLAModel

AutoConfig.register(MomGLAConfig.model_type, MomGLAConfig)
AutoModel.register(MomGLAConfig, MomGLAModel)
AutoModelForCausalLM.register(MomGLAConfig, MomGLAForCausalLM)


__all__ = ['MomGLAConfig', 'MomGLAForCausalLM', 'MomGLAModel']
