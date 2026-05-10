from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Stage1Config:
    siglip_model: str = "google/siglip-base-patch16-224"
    image_size: int = 224
    num_regions: int = 5
    min_patches_per_region: int = 6
    bbox_padding: float = 0.05
    use_correlation_attention: bool = True
    cluster_method: str = "kmeans"


@dataclass
class Stage2Config:
    vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"
    caption_prompt: str = "Describe the main subject of this image in one short sentence."
    max_new_tokens: int = 48
    filter_model: str = "google/siglip-base-patch16-224"
    filter_threshold: float = 0.18
    batch_size: int = 8


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2",
    )


@dataclass
class Stage3Config:
    base_model: str = "google/siglip-base-patch16-224"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    lambda_region: float = 0.5
    batch_size_global: int = 64
    regions_per_image: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 5000
    grad_accum: int = 1
    mixed_precision: str = "bf16"
    log_every: int = 25
    save_every: int = 500
    tensorboard: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None


@dataclass
class DataConfig:
    cc3m_root: str = "data/cc3m"
    subset_size: int = 50_000
    regions_out: str = "outputs/regions"
    pairs_out: str = "outputs/pairs.parquet"
    num_workers: int = 4
    seed: int = 1234


@dataclass
class Config:
    run_name: str = "rg_siglip_b16_v1"
    output_dir: str = "outputs"
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _from_dict(d: dict[str, Any]) -> Config:
    cfg = Config()
    base = cfg.to_dict()
    merged = _merge(base, d)
    return Config(
        run_name=merged["run_name"],
        output_dir=merged["output_dir"],
        stage1=Stage1Config(**merged["stage1"]),
        stage2=Stage2Config(**merged["stage2"]),
        stage3=Stage3Config(
            **{k: v for k, v in merged["stage3"].items() if k != "lora"},
            lora=LoRAConfig(**merged["stage3"]["lora"]),
        ),
        data=DataConfig(**merged["data"]),
    )


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _from_dict(raw)
