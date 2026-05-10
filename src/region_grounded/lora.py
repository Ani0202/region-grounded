"""LoRA setup for SigLIP fine-tuning via PEFT."""
from __future__ import annotations

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor

from .config import LoRAConfig as LoRAConfigDC


def wrap_siglip_with_lora(model_name: str, cfg: LoRAConfigDC, dtype=torch.bfloat16):
    """Return (peft_model, processor). Only LoRA params + logit_scale/logit_bias remain trainable."""
    base = AutoModel.from_pretrained(model_name, torch_dtype=dtype)
    lora_cfg = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        bias="none",
        target_modules=list(cfg.target_modules),
        # Apply across both vision and text towers — PEFT will resolve module names.
    )
    peft_model = get_peft_model(base, lora_cfg)
    # Unfreeze the SigLIP temperature/bias so the global loss can re-balance after LoRA edits
    for name, p in peft_model.named_parameters():
        if name.endswith("logit_scale") or name.endswith("logit_bias"):
            p.requires_grad_(True)
    proc = AutoProcessor.from_pretrained(model_name)
    return peft_model, proc


def trainable_param_count(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
