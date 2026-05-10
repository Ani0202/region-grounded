"""Zero-shot semantic segmentation evaluation (mIoU) on PASCAL VOC / COCO-Stuff.

For each image we compute patch-text similarities for every class prompt,
take argmax per patch, upsample to image resolution, and compare to the
ground-truth mask. mIoU is averaged over present classes per image, then
averaged over the dataset.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


@torch.no_grad()
def _class_text_features(model, processor, class_names: list[str], template: Callable[[str], str], device):
    prompts = [template(n) for n in class_names]
    inp = processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(device)
    out = model.text_model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
    txt = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
    return F.normalize(txt, dim=-1)


def _patch_grid_dim(model) -> int:
    return model.vision_model.embeddings.image_size // model.vision_model.embeddings.patch_size


@torch.no_grad()
def predict_mask(
    model, processor, image: Image.Image, text_feats: torch.Tensor, device,
    use_correlation: bool = True,
) -> np.ndarray:
    """Return a [H, W] int array of class predictions."""
    from .stage1_extract import SCLIPVisionEncoder  # reuse correlation forward

    inp = processor(images=image, return_tensors="pt").to(device)
    if use_correlation:
        # Run the correlation last-block on whatever the model's vision tower is.
        vis = model.vision_model if hasattr(model, "vision_model") else model
        hidden = vis.embeddings(inp["pixel_values"])
        for layer in vis.encoder.layers[:-1]:
            hidden = layer(hidden, attention_mask=None)
        # last block — correlation form
        last = vis.encoder.layers[-1]
        attn = last.self_attn
        residual = hidden
        h = last.layer_norm1(hidden)
        B, N, D = h.shape
        H = attn.num_heads
        Dh = D // H
        q = attn.q_proj(h).view(B, N, H, Dh).transpose(1, 2)
        k = attn.k_proj(h).view(B, N, H, Dh).transpose(1, 2)
        v = attn.v_proj(h).view(B, N, H, Dh).transpose(1, 2)
        scale = Dh ** -0.5
        a = ((q @ q.transpose(-1, -2)) * scale + (k @ k.transpose(-1, -2)) * scale) / 2.0
        out = (a.softmax(-1) @ v).transpose(1, 2).contiguous().view(B, N, D)
        hidden = residual + attn.out_proj(out)
        residual = hidden
        hidden = last.layer_norm2(hidden)
        hidden = residual + last.mlp(hidden)
        patches = vis.post_layernorm(hidden)
    else:
        patches = model.vision_model(pixel_values=inp["pixel_values"]).last_hidden_state

    patches = F.normalize(patches[0], dim=-1)  # [N, D]
    sims = patches @ text_feats.t()              # [N, C]
    g = _patch_grid_dim(model)
    cls_map = sims.argmax(dim=-1).view(g, g).cpu().numpy()
    cls_map = np.array(Image.fromarray(cls_map.astype(np.int32)).resize(image.size, Image.NEAREST))
    return cls_map


def per_image_iou(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore: int = 255) -> dict[int, float]:
    out: dict[int, float] = {}
    valid = target != ignore
    for c in range(num_classes):
        gt = (target == c) & valid
        if not gt.any():
            continue
        pr = (pred == c) & valid
        inter = np.logical_and(gt, pr).sum()
        union = np.logical_or(gt, pr).sum()
        out[c] = float(inter) / float(max(union, 1))
    return out


def mean_iou(
    model, processor, samples, class_names: list[str],
    template: Callable[[str], str] = lambda n: f"a photo of a {n}.",
    device: torch.device | None = None,
    use_correlation: bool = True,
) -> dict:
    """`samples` yields (PIL.Image, np.ndarray label map). Class ids must align with `class_names`."""
    device = device or next(model.parameters()).device
    model.eval()
    text_feats = _class_text_features(model, processor, class_names, template, device)
    per_class: dict[int, list[float]] = {}
    for img, mask in tqdm(samples, desc="segmentation"):
        pred = predict_mask(model, processor, img, text_feats, device, use_correlation=use_correlation)
        ious = per_image_iou(pred, mask, len(class_names))
        for c, v in ious.items():
            per_class.setdefault(c, []).append(v)
    means = {class_names[c]: float(np.mean(vs)) for c, vs in per_class.items()}
    miou = float(np.mean(list(means.values()))) if means else 0.0
    return {"mIoU": miou, "per_class": means, "n_classes_seen": len(means)}
