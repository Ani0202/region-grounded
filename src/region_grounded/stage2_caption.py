"""Stage 2 — Auto-caption each crop with Qwen2-VL, then filter by SigLIP similarity.

Caption hallucinations on tightly-cropped patches are the dominant noise
source for this pipeline. The filter step computes cosine similarity
between the crop and its generated caption in SigLIP's embedding space
and drops pairs below `cfg.filter_threshold`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)

from .config import Stage2Config
from .utils import auto_device, get_logger

log = get_logger(__name__)


class Qwen2VLCaptioner:
    def __init__(self, model_name: str, device: torch.device | None = None, dtype=torch.bfloat16):
        self.device = device or auto_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, device_map=str(self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def caption_batch(self, images: list[Image.Image], prompt: str, max_new_tokens: int) -> list[str]:
        messages = [
            [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ]}
            ]
            for _ in images
        ]
        texts = [self.processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        gen = out[:, prompt_len:]
        return [t.strip() for t in self.processor.batch_decode(gen, skip_special_tokens=True)]


class SiglipScorer:
    """Cosine similarity in SigLIP space, for filtering only."""

    def __init__(self, model_name: str, device: torch.device | None = None, dtype=torch.float16):
        self.device = device or auto_device()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()

    @torch.no_grad()
    def similarity(self, images: list[Image.Image], texts: list[str]) -> torch.Tensor:
        proc = self.processor(
            images=images, text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        out = self.model(**proc)
        img = F.normalize(out.image_embeds, dim=-1)
        txt = F.normalize(out.text_embeds, dim=-1)
        return (img * txt).sum(dim=-1).float().cpu()


def caption_and_filter(
    stage1_records: Iterable[dict],
    cfg: Stage2Config,
    captioner: Qwen2VLCaptioner | None = None,
    scorer: SiglipScorer | None = None,
) -> list[dict]:
    """Mutates a list of Stage 1 records, adding `caption` and `score` to every region, and
    drops any region with score < `cfg.filter_threshold`. Returns the cleaned list.
    """
    captioner = captioner or Qwen2VLCaptioner(cfg.vlm_model)
    scorer = scorer or SiglipScorer(cfg.filter_model)
    out: list[dict] = []
    # Flatten regions so we can batch across images
    flat: list[tuple[int, int, str]] = []
    for r_idx, rec in enumerate(stage1_records):
        for c_idx, region in enumerate(rec["regions"]):
            flat.append((r_idx, c_idx, region["region_path"]))

    records = list(stage1_records)
    captions: dict[tuple[int, int], str] = {}
    scores: dict[tuple[int, int], float] = {}

    for i in tqdm(range(0, len(flat), cfg.batch_size), desc="caption+score"):
        chunk = flat[i : i + cfg.batch_size]
        imgs = [Image.open(p).convert("RGB") for _, _, p in chunk]
        caps = captioner.caption_batch(imgs, cfg.caption_prompt, cfg.max_new_tokens)
        sims = scorer.similarity(imgs, caps).tolist()
        for (r_idx, c_idx, _), cap, sim in zip(chunk, caps, sims):
            captions[(r_idx, c_idx)] = cap
            scores[(r_idx, c_idx)] = float(sim)

    for r_idx, rec in enumerate(records):
        kept: list[dict] = []
        for c_idx, region in enumerate(rec["regions"]):
            key = (r_idx, c_idx)
            if key not in scores:
                continue
            sim = scores[key]
            region = {**region, "caption": captions[key], "score": sim}
            if sim >= cfg.filter_threshold:
                kept.append(region)
        if kept:
            out.append({**rec, "regions": kept})
    return out


def save_pairs(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pyarrow as pa
        import pyarrow.parquet as pq
        # Schema: image_path, global_caption, regions (list of struct)
        table = pa.Table.from_pylist(records)
        pq.write_table(table, path)
    else:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    log.info("Wrote %d records to %s", len(records), path)
