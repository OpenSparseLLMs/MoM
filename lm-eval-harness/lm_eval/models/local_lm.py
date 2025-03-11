from .local_utils.loading import load_model, load_tokenizer, load_hf_model
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("lm_eval_model")
class LMWrapper(HFLM):
    def __init__(
            self, 
            checkpoint_name: str, 
            max_length: int = 2048,
            device: str = "cuda",
            **kwargs
        ) -> None:

        is_hf=not checkpoint_name.startswith("hazy-research")
        tokenizer = load_tokenizer(checkpoint_name, is_hf=is_hf)
        if is_hf:
            model = load_hf_model(checkpoint_name)
        else:
            model = load_model(checkpoint_name, device=device)
        # model.device = device
        # import torch
        # model.to(device, dtype=torch.float32)
        model.to(device)

        super().__init__(
            pretrained=model,
            backend="causal",
            max_length=max_length,
            tokenizer=tokenizer,
            device=device,
            **kwargs,
        )

