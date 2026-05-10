"""Stage 3 — LoRA fine-tuning of SigLIP with L_global + λ · L_region.

The training step takes one batch of global (image, caption) pairs and one
batch of (region, region_caption) pairs sampled from the parquet produced
by Stage 2. Both losses are computed in-batch with SigLIP's pairwise
sigmoid (positives on diagonal); only the region loss uses FILIP-style
token-wise late interaction.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config
from .data import RegionPairDataset, collate_region_batch
from .losses import region_sigmoid_loss, sigmoid_loss
from .lora import trainable_param_count, wrap_siglip_with_lora
from .utils import TrainLogger, auto_device, dtype_from_str, ensure_dir, get_logger, set_seed

log = get_logger(__name__)


def _processor_inputs(processor, images, texts, device):
    proc = processor(
        images=images, text=texts,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    out = {k: v.to(device) for k, v in proc.items()}
    # SigLIP's tokenizer pads to a fixed length and doesn't emit attention_mask. We need one for
    # the FILIP late-interaction text mask, so derive it from the pad token id.
    if "attention_mask" not in out:
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = 1  # SigLIP T5-style sentencepiece default
        out["attention_mask"] = (out["input_ids"] != pad_id).long()
    return out


def _token_features(model, pixel_values: torch.Tensor, input_ids: torch.Tensor, attn_mask: torch.Tensor):
    """Return per-patch and per-word features from the SigLIP encoders.

    We use raw last-hidden states (no projection head applied); both towers
    in standard SigLIP share the same hidden size, so late-interaction dot
    products are well-defined.
    """
    vis = model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
    txt = model.text_model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=False)
    return vis.last_hidden_state, txt.last_hidden_state


def _pool_for_global(model, pixel_values, input_ids, attn_mask):
    out = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attn_mask)
    return out.image_embeds, out.text_embeds


@dataclass
class TrainState:
    step: int = 0
    best_loss: float = math.inf


def _flatten_region_batch(batch_regions, k: int) -> list[dict]:
    """For each image, pick up to k regions uniformly at random; return the flat list.

    `batch_regions` may be a list of lists, or a list of numpy object arrays
    when records were materialized via pyarrow → pandas.
    """
    flat: list[dict] = []
    for regs in batch_regions:
        regs = list(regs)
        if len(regs) == 0:
            continue
        picks = random.sample(regs, min(k, len(regs)))
        flat.extend(picks)
    return flat


def train(cfg: Config):
    device = auto_device()
    set_seed(cfg.data.seed)
    dtype = dtype_from_str(cfg.stage3.mixed_precision) if device.type == "cuda" else torch.float32

    model, processor = wrap_siglip_with_lora(cfg.stage3.base_model, cfg.stage3.lora, dtype=dtype)
    model.to(device)
    tr, total = trainable_param_count(model)
    log.info("Trainable params: %d / %d (%.2f%%)", tr, total, 100 * tr / total)

    ds = RegionPairDataset(cfg.data.pairs_out, regions_per_image=cfg.stage3.regions_per_image)
    loader = DataLoader(
        ds,
        batch_size=cfg.stage3.batch_size_global,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_region_batch,
        drop_last=True,
    )

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.stage3.lr,
        weight_decay=cfg.stage3.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < cfg.stage3.warmup_steps:
            return step / max(1, cfg.stage3.warmup_steps)
        progress = (step - cfg.stage3.warmup_steps) / max(1, cfg.stage3.max_steps - cfg.stage3.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.run_name)
    tb_logger = TrainLogger(
        log_dir=out_dir / "tb",
        tensorboard=cfg.stage3.tensorboard,
        wandb_project=cfg.stage3.wandb_project,
        wandb_run_name=cfg.stage3.wandb_run_name or cfg.run_name,
        wandb_config=cfg.to_dict(),
    )
    state = TrainState()
    pbar = tqdm(total=cfg.stage3.max_steps, desc="train")
    while state.step < cfg.stage3.max_steps:
        for batch in loader:
            if state.step >= cfg.stage3.max_steps:
                break
            global_imgs = [Image.open(p).convert("RGB") for p in batch["image_paths"]]
            global_caps = list(batch["global_captions"])
            g_in = _processor_inputs(processor, global_imgs, global_caps, device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                img_e, txt_e = _pool_for_global(
                    model, g_in["pixel_values"], g_in["input_ids"], g_in["attention_mask"]
                )
                ls = model.get_base_model().logit_scale if hasattr(model, "get_base_model") else model.logit_scale
                lb = model.get_base_model().logit_bias if hasattr(model, "get_base_model") else model.logit_bias
                L_global = sigmoid_loss(img_e, txt_e, ls, lb)

                regions_flat = _flatten_region_batch(batch["regions"], cfg.stage3.regions_per_image)
                L_region = torch.tensor(0.0, device=device)
                if len(regions_flat) >= 2:
                    r_imgs = [Image.open(r["region_path"]).convert("RGB") for r in regions_flat]
                    r_caps = [r["caption"] for r in regions_flat]
                    r_in = _processor_inputs(processor, r_imgs, r_caps, device)
                    patch_tok, word_tok = _token_features(
                        model, r_in["pixel_values"], r_in["input_ids"], r_in["attention_mask"]
                    )
                    L_region = region_sigmoid_loss(
                        patch_tok, word_tok, r_in["attention_mask"], ls, lb,
                    )
                loss = L_global + cfg.stage3.lambda_region * L_region

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optim.step()
            scheduler.step()

            state.step += 1
            pbar.update(1)
            l_region_val = L_region.detach().item() if torch.is_tensor(L_region) else float(L_region)
            metrics = {
                "train/loss": loss.detach().item(),
                "train/L_global": L_global.detach().item(),
                "train/L_region": l_region_val,
                "train/lr": scheduler.get_last_lr()[0],
            }
            tb_logger.log(metrics, state.step)
            if state.step % cfg.stage3.log_every == 0:
                log.info(
                    "step %d  loss=%.4f  L_global=%.4f  L_region=%.4f  lr=%.2e",
                    state.step, metrics["train/loss"], metrics["train/L_global"],
                    l_region_val, metrics["train/lr"],
                )
            if state.step % cfg.stage3.save_every == 0:
                ckpt = out_dir / f"step_{state.step}"
                model.save_pretrained(ckpt)
                log.info("saved LoRA checkpoint to %s", ckpt)
    final = out_dir / "final"
    model.save_pretrained(final)
    log.info("training complete; final LoRA at %s", final)
    tb_logger.close()
    return model, processor
