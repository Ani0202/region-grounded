from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_logger(name: str = "region_grounded") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dtype_from_str(s: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[s]


class TrainLogger:
    """Thin scalar-logging facade over TensorBoard (always-on if available) and wandb (opt-in).

    Both backends silently no-op when their packages aren't installed, so calling code stays
    backend-agnostic.
    """

    def __init__(
        self,
        log_dir: str | Path,
        tensorboard: bool = True,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        wandb_config: dict[str, Any] | None = None,
    ):
        self._tb = None
        self._wb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                ensure_dir(log_dir)
                self._tb = SummaryWriter(log_dir=str(log_dir))
            except Exception as e:
                get_logger().warning("TensorBoard unavailable: %s", e)
        if wandb_project:
            try:
                import wandb
                self._wb = wandb.init(
                    project=wandb_project, name=wandb_run_name, config=wandb_config or {},
                    dir=str(log_dir), reinit=True,
                )
            except Exception as e:
                get_logger().warning("wandb init failed: %s", e)

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, float(v), step)
        if self._wb is not None:
            self._wb.log({**metrics, "step": step})

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wb is not None:
            self._wb.finish()
