"""Compositionality benchmarks — ARO, SugarCrepe, Winoground.

All three reduce to the same primitive: given an image plus a positive
caption and one or more confounded negative captions, the model is
"correct" if it scores the positive strictly above every negative. We
provide a unified scoring loop and small adapters for each benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


@dataclass
class ContrastItem:
    image: Image.Image
    positive: str
    negatives: list[str]


@torch.no_grad()
def _score(model, processor, image: Image.Image, texts: list[str], device) -> torch.Tensor:
    inp = processor(
        images=[image] * len(texts), text=texts,
        padding="max_length", truncation=True, return_tensors="pt",
    ).to(device)
    out = model(**inp)
    img = F.normalize(out.image_embeds, dim=-1)
    txt = F.normalize(out.text_embeds, dim=-1)
    return (img * txt).sum(dim=-1).cpu()


def evaluate(model, processor, items: Iterable[ContrastItem], device: torch.device | None = None) -> dict:
    device = device or next(model.parameters()).device
    model.eval()
    n_correct = 0
    n_total = 0
    items = list(items)
    for it in tqdm(items, desc="compositionality"):
        scores = _score(model, processor, it.image, [it.positive] + it.negatives, device)
        pos = scores[0].item()
        negs = scores[1:].tolist()
        if all(pos > s for s in negs):
            n_correct += 1
        n_total += 1
    return {"accuracy": n_correct / max(n_total, 1), "n": n_total}


# Dataset adapters ----------------------------------------------------------
# Each returns an iterator of ContrastItem so they can share `evaluate(...)`.

def aro_items(split: str = "VG_Relation"):
    """Loads ARO (Attribution, Relation, Order). Requires the `aro` dataset on disk or HF mirror.

    Returns: iterator of ContrastItem with exactly 1 negative each.
    """
    from datasets import load_dataset
    ds = load_dataset("gowitheflow/ARO", split=split)
    for ex in ds:
        yield ContrastItem(image=ex["image"], positive=ex["true_caption"], negatives=[ex["false_caption"]])


def sugarcrepe_items(category: str = "replace_obj"):
    """SugarCrepe: 7 categories (replace/swap/add × obj/att/rel). Returns 1 negative per item."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/SugarCrepe", split=category)
    for ex in ds:
        yield ContrastItem(image=ex["image"], positive=ex["caption"], negatives=[ex["negative_caption"]])


def winoground_items():
    """Winoground: image-text pairs in a 2x2 layout. We unfold each pair so each image gets
    its true caption as positive and the swapped caption as negative.
    """
    from datasets import load_dataset
    ds = load_dataset("facebook/winoground", split="test")
    for ex in ds:
        yield ContrastItem(image=ex["image_0"], positive=ex["caption_0"], negatives=[ex["caption_1"]])
        yield ContrastItem(image=ex["image_1"], positive=ex["caption_1"], negatives=[ex["caption_0"]])
