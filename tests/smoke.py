"""End-to-end smoke test for all three stages on 4 synthetic images.

Runs with the real SigLIP-B/16 encoder (~88 M params, fits on CPU/MPS) but
mocks the Qwen2-VL captioner so we don't pull 5 GB of weights every time.
Verifies the pipeline wires together cleanly before burning Colab credits.

Usage:
    python tests/smoke.py        # standalone
    pytest tests/smoke.py        # via pytest
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


# ---- Synthetic data --------------------------------------------------------

def _make_images(root: Path, n: int = 4) -> list[tuple[str, str]]:
    """Make n distinct images with simple colored shapes; return (path, caption) pairs."""
    root.mkdir(parents=True, exist_ok=True)
    palette = [
        ("red square on white", (255, 60, 60)),
        ("blue circle on white", (60, 120, 255)),
        ("green triangle on white", (80, 200, 80)),
        ("yellow ring on white", (240, 220, 60)),
    ]
    out = []
    for i, (caption, color) in enumerate(palette[:n]):
        img = Image.new("RGB", (224, 224), (245, 245, 245))
        d = ImageDraw.Draw(img)
        if "square" in caption:
            d.rectangle([60, 60, 164, 164], fill=color)
        elif "circle" in caption:
            d.ellipse([60, 60, 164, 164], fill=color)
        elif "triangle" in caption:
            d.polygon([(112, 50), (50, 174), (174, 174)], fill=color)
        else:
            d.ellipse([50, 50, 174, 174], outline=color, width=18)
        p = root / f"img_{i}.jpg"
        img.save(p, quality=92)
        out.append((str(p), caption))
    return out


class FakeCaptioner:
    """Stands in for Qwen2VLCaptioner so we don't pull the 5 GB checkpoint in CI."""

    def caption_batch(self, images, prompt, max_new_tokens):
        return ["a small colored shape on a light background"] * len(images)


# ---- Stages ---------------------------------------------------------------

def run_smoke(workdir: Path) -> dict:
    import torch
    from region_grounded.config import Config, Stage1Config, Stage2Config, Stage3Config, LoRAConfig, DataConfig
    from region_grounded.stage1_extract import SCLIPVisionEncoder, extract_and_save
    from region_grounded.stage2_caption import SiglipScorer, caption_and_filter, save_pairs
    from region_grounded.stage3_train import train

    images_dir = workdir / "images"
    regions_dir = workdir / "regions"
    pairs_path = workdir / "pairs.parquet"

    pairs = _make_images(images_dir, n=4)

    # Stage 1
    s1 = Stage1Config(num_regions=3, min_patches_per_region=3, bbox_padding=0.02)
    encoder = SCLIPVisionEncoder(s1.siglip_model, dtype=torch.float32)
    records = extract_and_save(pairs, encoder, s1, out_dir=regions_dir, seed=0)
    assert records and all(r["regions"] for r in records), f"Stage 1 produced no regions: {records}"
    print(f"[stage1] {len(records)} images -> {sum(len(r['regions']) for r in records)} regions")

    # Stage 2 (mocked captioner, real SigLIP scorer)
    s2 = Stage2Config(filter_threshold=-1.0, batch_size=4)   # accept everything
    scorer = SiglipScorer(s2.filter_model)
    filtered = caption_and_filter(records, s2, captioner=FakeCaptioner(), scorer=scorer)
    save_pairs(filtered, pairs_path)
    assert filtered and all(r["regions"] for r in filtered), f"Stage 2 dropped everything: {filtered}"
    print(f"[stage2] kept {sum(len(r['regions']) for r in filtered)} (region, caption) pairs")

    # Stage 3 (tiny run)
    cfg = Config(
        run_name="smoke",
        output_dir=str(workdir / "out"),
        stage1=s1,
        stage2=s2,
        stage3=Stage3Config(
            lora=LoRAConfig(r=4, alpha=8),
            batch_size_global=2,
            regions_per_image=2,
            lr=1e-4,
            warmup_steps=1,
            max_steps=3,
            mixed_precision="fp32",
            log_every=1,
            save_every=3,
            tensorboard=True,
        ),
        data=DataConfig(
            cc3m_root=str(images_dir),
            pairs_out=str(pairs_path),
            num_workers=0,
        ),
    )
    train(cfg)
    final = Path(cfg.output_dir) / cfg.run_name / "final"
    assert final.exists(), f"Stage 3 did not save a final checkpoint at {final}"
    print(f"[stage3] saved checkpoint at {final}")
    return {"records": len(records), "filtered": len(filtered), "ckpt": str(final)}


def test_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        result = run_smoke(Path(tmp))
        assert result["records"] == 4
        assert result["filtered"] >= 1


if __name__ == "__main__":
    out = Path("outputs/smoke")
    if out.exists():
        shutil.rmtree(out)
    print(json.dumps(run_smoke(out), indent=2))
