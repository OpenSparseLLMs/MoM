# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mom.models.mom_gsa.configuration_mom_gsa import MomGSAConfig
from mom.models.mom_gsa.modeling_mom_gsa import MomGSAForCausalLM, MomGSAModel

AutoConfig.register(MomGSAConfig.model_type, MomGSAConfig)
AutoModel.register(MomGSAConfig, MomGSAModel)
AutoModelForCausalLM.register(MomGSAConfig, MomGSAForCausalLM)


__all__ = ['MomGSAConfig', 'MomGSAForCausalLM', 'MomGSAModel']
