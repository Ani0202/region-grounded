"""Stage 1 — Spatial Region Extraction.

Re-runs SigLIP's final vision encoder block with SCLIP-style correlation
attention (q·qᵀ + k·kᵀ) to recover a patch grid that clusters along
semantic boundaries instead of the standard q·kᵀ "anti-localized" pattern.
Patches are k-means-clustered; each cluster's tight bounding box on the
original image becomes one region.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from transformers import SiglipVisionModel, AutoProcessor

from .config import Stage1Config
from .utils import auto_device, ensure_dir, get_logger

log = get_logger(__name__)


@dataclass
class Region:
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 in [0, 1]
    patch_indices: list[int]
    crop_size: tuple[int, int]                 # (W, H) of original image


class SCLIPVisionEncoder:
    """Wraps a frozen SigLIP vision tower with an alternate last-block forward."""

    def __init__(self, model_name: str, device: torch.device | None = None, dtype=torch.float32):
        self.model: SiglipVisionModel = SiglipVisionModel.from_pretrained(model_name, torch_dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device or auto_device()
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        # `SiglipVisionModel` IS the vision transformer in HF; older `SiglipModel` nests it under
        # `.vision_model`. Support both so the same encoder works on a full SigLIP checkpoint too.
        v = self.model.vision_model if hasattr(self.model, "vision_model") else self.model
        assert hasattr(v, "embeddings") and hasattr(v, "encoder") and hasattr(v, "post_layernorm")
        self._v = v
        self.patch_size = v.embeddings.patch_size
        self.image_size = v.embeddings.image_size
        self.grid = self.image_size // self.patch_size
        self.dim = v.config.hidden_size

    @torch.no_grad()
    def patch_features(self, pixel_values: torch.Tensor, correlation: bool = True) -> torch.Tensor:
        """Return per-patch features [B, N, D]. N = grid·grid."""
        v = self._v
        hidden = v.embeddings(pixel_values.to(self.device))
        layers = v.encoder.layers
        for layer in layers[:-1]:
            hidden = layer(hidden, attention_mask=None)
        if correlation:
            hidden = self._correlation_last_block(layers[-1], hidden)
        else:
            hidden = layers[-1](hidden, attention_mask=None)
        hidden = v.post_layernorm(hidden)
        return hidden

    def _correlation_last_block(self, layer, hidden: torch.Tensor) -> torch.Tensor:
        attn = layer.self_attn
        residual = hidden
        h = layer.layer_norm1(hidden)
        B, N, D = h.shape
        H = attn.num_heads
        Dh = D // H
        q = attn.q_proj(h).view(B, N, H, Dh).transpose(1, 2)
        k = attn.k_proj(h).view(B, N, H, Dh).transpose(1, 2)
        v_ = attn.v_proj(h).view(B, N, H, Dh).transpose(1, 2)
        scale = 1.0 / math.sqrt(Dh)
        a_qq = (q @ q.transpose(-1, -2)) * scale
        a_kk = (k @ k.transpose(-1, -2)) * scale
        weights = ((a_qq + a_kk) / 2.0).softmax(dim=-1)
        out = (weights @ v_).transpose(1, 2).contiguous().view(B, N, D)
        out = attn.out_proj(out)
        hidden = residual + out
        residual = hidden
        hidden = layer.layer_norm2(hidden)
        hidden = layer.mlp(hidden)
        return residual + hidden

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"]


def _cluster_patches(features: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run k-means on L2-normalized patch features. Returns cluster ids of shape [N]."""
    f = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-9)
    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    return km.fit_predict(f)


def _bbox_from_indices(indices: list[int], grid: int, pad: float) -> tuple[float, float, float, float]:
    rows = np.array([i // grid for i in indices])
    cols = np.array([i % grid for i in indices])
    y0 = rows.min() / grid - pad
    y1 = (rows.max() + 1) / grid + pad
    x0 = cols.min() / grid - pad
    x1 = (cols.max() + 1) / grid + pad
    return (float(np.clip(x0, 0, 1)), float(np.clip(y0, 0, 1)),
            float(np.clip(x1, 0, 1)), float(np.clip(y1, 0, 1)))


def extract_regions(
    image: Image.Image,
    encoder: SCLIPVisionEncoder,
    cfg: Stage1Config,
    seed: int = 0,
) -> list[Region]:
    """Run a single image through the SCLIP-feature pipeline and return its crops' boxes."""
    pixel_values = encoder.preprocess(image)
    feats = encoder.patch_features(pixel_values, correlation=cfg.use_correlation_attention)
    feats = feats[0].float().cpu().numpy()
    labels = _cluster_patches(feats, cfg.num_regions, seed)
    regions: list[Region] = []
    W, H = image.size
    for cid in range(cfg.num_regions):
        idx = [int(i) for i in np.where(labels == cid)[0]]
        if len(idx) < cfg.min_patches_per_region:
            continue
        bbox = _bbox_from_indices(idx, encoder.grid, cfg.bbox_padding)
        regions.append(Region(bbox=bbox, patch_indices=idx, crop_size=(W, H)))
    return regions


def crop_region(image: Image.Image, region: Region) -> Image.Image:
    W, H = image.size
    x0, y0, x1, y1 = region.bbox
    box = (int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H))
    # Guard against degenerate boxes
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return image.copy()
    return image.crop(box)


def extract_and_save(
    images: Iterable[tuple[str, str]],  # (image_path, global_caption)
    encoder: SCLIPVisionEncoder,
    cfg: Stage1Config,
    out_dir: str | Path,
    seed: int = 0,
) -> list[dict]:
    """Run Stage 1 over a list of images, save crops to disk, return metadata records."""
    out = ensure_dir(out_dir)
    records: list[dict] = []
    for img_idx, (path, caption) in enumerate(images):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            log.warning("Skipping %s: %s", path, e)
            continue
        regions = extract_regions(image, encoder, cfg, seed=seed + img_idx)
        regions_meta: list[dict] = []
        for r_idx, r in enumerate(regions):
            crop = crop_region(image, r)
            stem = Path(path).stem
            rel = f"{stem}_r{r_idx}.jpg"
            crop.save(out / rel, quality=92)
            regions_meta.append({"region_path": str(out / rel), "bbox": list(r.bbox)})
        records.append({
            "image_path": path,
            "global_caption": caption,
            "regions": regions_meta,
        })
    return records
