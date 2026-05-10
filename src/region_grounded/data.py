from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class GlobalSample:
    image_path: str
    caption: str


@dataclass
class RegionSample:
    image_path: str
    global_caption: str
    region_path: str
    region_caption: str
    bbox: tuple[float, float, float, float]
    score: float


class CC3MIndex(Dataset):
    """Index of CC3M-style global (image, caption) pairs.

    Expects an index file (TSV or JSONL) listing local files. Format:
      - tsv: caption\timage_relpath
      - jsonl: {"caption": ..., "image": ...}
    """

    def __init__(self, root: str | Path, index_file: str | Path, limit: int | None = None):
        self.root = Path(root)
        self.entries: list[GlobalSample] = []
        path = Path(index_file)
        if path.suffix == ".tsv":
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                cap, rel = line.split("\t", 1)
                self.entries.append(GlobalSample(str(self.root / rel), cap))
        else:
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.entries.append(GlobalSample(str(self.root / obj["image"]), obj["caption"]))
        if limit is not None:
            self.entries = self.entries[:limit]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> GlobalSample:
        return self.entries[idx]


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def iter_batches(items: list, batch_size: int) -> Iterator[list]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class RegionPairDataset(Dataset):
    """Loads (image, global_caption, list_of_regions) from a parquet/jsonl produced by Stage 2.

    Each row corresponds to a single global sample with N filtered region children.
    """

    def __init__(self, pairs_path: str | Path, regions_per_image: int = 4):
        import pyarrow.parquet as pq

        self.regions_per_image = regions_per_image
        path = Path(pairs_path)
        if path.suffix == ".parquet":
            table = pq.read_table(path).to_pandas()
            self.rows = table.to_dict(orient="records")
        else:
            self.rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        regions = row["regions"][: self.regions_per_image]
        return {
            "image_path": row["image_path"],
            "global_caption": row["global_caption"],
            "regions": regions,
        }


def collate_region_batch(batch: list[dict]) -> dict:
    """Group a batch of global samples with their regions, padding region count."""
    return {
        "image_paths": [b["image_path"] for b in batch],
        "global_captions": [b["global_caption"] for b in batch],
        "regions": [b["regions"] for b in batch],
    }
