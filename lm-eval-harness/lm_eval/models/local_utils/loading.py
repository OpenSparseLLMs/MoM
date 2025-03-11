import torch
from torch import nn
from typing import Union
import os
import hydra
import sys
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM
from .jrt_utils import import_object, load_config

### This code loads models that we trained ###
def load_model(
    run_id: str,
    device: Union[int, str] = None,
    config: any=None,
) -> nn.Module:
    """
    Load a model from a wandb run ID.
    Parameters:
        run_id (str): A full wandb run id like "hazy-research/attention/159o6asi"
    """

    # 1: Get configuration from wandb
    config = load_config(run_id)
    path = config["callbacks"]["model_checkpoint"]["dirpath"]

    if config["model"].get("_instantiate_config_", True):
          
        # SE (01/29): models were trained on flash_attn==2.3.6
        # a newer version sets this parameter to 128 by default, so to make it 
        # compatible while still allowing for upgrades of flash attention, 
        # we set it to 256 here
        if config["model"]["_target_"] == "flash_attn.models.gpt.GPTLMHeadModel":
            config["model"]["config"]["mlp_multiple_of"] = 128

        model_config = hydra.utils.instantiate(
            config["model"]["config"], _recursive_=False, _convert_="object"
        )
        cls = import_object(config["model"]["_target_"])
        model = cls(model_config).to(device=device)
    else: 
        # SE: need this alternate form for models that accept kwargs, not a config object
        model_config = config["model"].pop("config")
        model = hydra.utils.instantiate(config["model"], **model_config, _recursive_=False)

    path = path.replace(
        "/var/cr05_data/sim_data/checkpoints/",         # old machine
        '/home/simarora/based-checkpoints/checkpoints/'
    )

    try:
        assert os.path.exists(path), print(f"Path {path} does not exist")
        ckpt = torch.load(os.path.join(path, "last.ckpt"), map_location=torch.device(device))
    except:
        paths = os.listdir(path)
        paths = [p for p in paths if ".ckpt" in p]
        print(f'Loading model from {paths[0]}')
        ckpt = torch.load(os.path.join(path, paths[0]), map_location=torch.device(device))

    # 3: Load model
    # load the state dict, but remove the "model." prefix and all other keys from the
    # the PyTorch Lightning module that are not in the actual model
    model.load_state_dict({
        k[len("model."):]: v 
        for k, v in ckpt["state_dict"].items() 
        if k.startswith("model.")
    })

    model = model.to(device=device)
    return model


def load_hf_model(model_name: str, device:str = 'cuda') -> nn.Module:
    if "mamba" in model_name:
        
        # SA: can't pass in device here https://github.com/pytorch/pytorch/issues/10622
        model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=torch.float16)
    
    else:
        if "Mixtral" in model_name:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, use_flash_attention_2=True, 
                # load_in_8bit=True,
                torch_dtype=torch.bfloat16, device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token="your token here")
            model.to(device)
    

    try: model.device = device
    except: pass
    model.eval()
    return model


def load_tokenizer(model_name: str, is_hf: bool=False) -> nn.Module:
    if not is_hf:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = 2048
    else:
        if "mamba" in model_name or "mpt" in model_name:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

        