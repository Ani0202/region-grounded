"""Guardrail eval — zero-shot top-1 on ImageNet-1K and CIFAR-100.

We want parity with the off-the-shelf SigLIP baseline; a large drop here
means region fine-tuning collapsed global alignment.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.no_grad()
def _text_embeddings(model, processor, class_prompts: list[str], device) -> torch.Tensor:
    inputs = processor(text=class_prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
    out = model.text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    # SigLIP exposes text projection through the full model's text_projection (or head).
    txt = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
    return F.normalize(txt, dim=-1)


@torch.no_grad()
def zero_shot_top1(
    model,
    processor,
    dataset,                            # yields (PIL.Image, int label)
    class_names: list[str],
    template: Callable[[str], str] = lambda n: f"a photo of a {n}.",
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict:
    device = device or next(model.parameters()).device
    model.eval()
    prompts = [template(n) for n in class_names]
    text_feats = _text_embeddings(model, processor, prompts, device)

    correct = 0
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, collate_fn=lambda b: list(zip(*b)))
    for imgs, labels in tqdm(loader, desc="zero-shot"):
        inp = processor(images=list(imgs), return_tensors="pt").to(device)
        img_feats = model.vision_model(pixel_values=inp["pixel_values"]).pooler_output
        img_feats = F.normalize(img_feats, dim=-1)
        logits = img_feats @ text_feats.t()
        pred = logits.argmax(dim=-1).cpu().tolist()
        correct += sum(int(p == l) for p, l in zip(pred, labels))
        total += len(labels)
    return {"top1": correct / max(total, 1), "n": total}


# Convenience builders -------------------------------------------------------

def cifar100(root: str = "data/cifar100"):
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root=root, train=False, download=True)
    return ds, ds.classes


def imagenet(root: str):
    """Expects ImageNet-1K validation laid out as ImageFolder. Provide the WordNet classnames separately."""
    from torchvision.datasets import ImageFolder
    return ImageFolder(root=root)
