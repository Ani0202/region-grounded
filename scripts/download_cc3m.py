"""Download a CC3M subset and emit `index.jsonl` for the project's data loader.

Two backends:

  --source hf  (default) — stream `pixparse/cc3m-wds` from HuggingFace Hub.
               Images are already prefetched as webdataset tar shards, so
               this is fast and skips the URL-fetching dance entirely.

  --source tsv — fetch from the official CC3M TSV with `img2dataset`.
               Requires you to have accepted Google's CC3M terms and
               downloaded the TSV file separately.

Outputs:
  <out_dir>/img/<shard>/<id>.jpg     image files
  <out_dir>/index.jsonl              {"image": <relpath>, "caption": <str>}

Usage:
  python scripts/download_cc3m.py --out data/cc3m --subset 50000
  python scripts/download_cc3m.py --source tsv --tsv data/cc3m/Train_GCC.tsv \\
         --out data/cc3m --subset 50000 --processes 8 --threads 64
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


# ---- HuggingFace backend ---------------------------------------------------

def _download_via_hf(out_dir: Path, subset: int, hf_split: str = "train", shard_size: int = 1_000) -> Path:
    """Download tars from `pixparse/cc3m-wds` and extract images + captions.

    We read tar shards directly via `huggingface_hub` + `tarfile` instead of
    routing through `datasets.load_dataset`, because the mirror's actual
    schema (jpg / txt / json / __key__ / __url__) doesn't match the
    streaming feature manifest `datasets` infers — which trips its cast
    step on the very first batch.
    """
    import io
    import tarfile

    from huggingface_hub import hf_hub_download, list_repo_files
    from PIL import Image

    repo = "pixparse/cc3m-wds"
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    print(f"listing tar shards in {repo} …", flush=True)
    files = list_repo_files(repo, repo_type="dataset")
    prefix = f"cc3m-{hf_split}-"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".tar"))
    if not shards:
        raise RuntimeError(f"no tars matching {prefix}*.tar in {repo}")
    print(f"found {len(shards)} shards; fetching up to {subset} samples …", flush=True)

    n = 0
    with index_path.open("w") as out:
        for s_idx, shard_name in enumerate(shards):
            if n >= subset:
                break
            print(f"  shard {s_idx + 1}/{len(shards)}: {shard_name}", flush=True)
            local = hf_hub_download(repo, shard_name, repo_type="dataset")
            # Group tar members by the webdataset stem (key) so jpg/txt/json line up.
            by_stem: dict[str, dict[str, bytes]] = {}
            with tarfile.open(local, mode="r") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    name = member.name
                    dot = name.find(".")
                    if dot < 0:
                        continue
                    stem, ext = name[:dot], name[dot + 1:].lower()
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    by_stem.setdefault(stem, {})[ext] = fh.read()
            for stem, parts in by_stem.items():
                if n >= subset:
                    break
                img_bytes = parts.get("jpg") or parts.get("jpeg") or parts.get("png")
                if not img_bytes:
                    continue
                caption = parts.get("txt", b"").decode("utf-8", errors="replace").strip()
                if not caption and "json" in parts:
                    try:
                        meta = json.loads(parts["json"])
                        caption = meta.get("caption") or ""
                    except Exception:
                        caption = ""
                if not caption:
                    continue
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    print(f"    skip {stem}: {e}", file=sys.stderr)
                    continue
                bucket = f"{n // shard_size:05d}"
                bucket_dir = img_dir / bucket
                bucket_dir.mkdir(exist_ok=True)
                rel = f"img/{bucket}/{stem.replace('/', '_')}.jpg"
                img.save(out_dir / rel, quality=92)
                out.write(json.dumps({"image": rel, "caption": caption}) + "\n")
                n += 1
                if n <= 5 or n % 100 == 0:
                    print(f"    {n} / {subset}", flush=True)
    print(f"wrote {n} entries to {index_path}", flush=True)
    return index_path


# ---- TSV + img2dataset backend --------------------------------------------

def _sample_tsv(tsv: Path, n: int, seed: int) -> Path:
    import random
    rng = random.Random(seed)
    with tsv.open("r", encoding="utf-8") as f:
        sampled: list[tuple[str, str]] = []
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if len(row) < 2:
                continue
            cap, url = row[0], row[1]
            if i < n:
                sampled.append((cap, url))
            else:
                j = rng.randint(0, i)
                if j < n:
                    sampled[j] = (cap, url)
    out = tsv.with_suffix(".sampled.tsv")
    with out.open("w", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["caption", "url"])
        for cap, url in sampled:
            w.writerow([cap, url])
    return out


def _run_img2dataset(input_tsv: Path, out_dir: Path, image_size: int, processes: int, threads: int) -> None:
    cmd = [
        sys.executable, "-m", "img2dataset",
        "--url_list", str(input_tsv),
        "--input_format", "tsv",
        "--url_col", "url",
        "--caption_col", "caption",
        "--output_folder", str(out_dir / "shards"),
        "--output_format", "files",
        "--image_size", str(image_size),
        "--processes_count", str(processes),
        "--thread_count", str(threads),
        "--resize_mode", "keep_ratio",
        "--encode_quality", "90",
        "--save_additional_columns", "['caption']",
    ]
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def _build_index_from_shards(out_dir: Path) -> Path:
    index = out_dir / "index.jsonl"
    shards = out_dir / "shards"
    n = 0
    with index.open("w") as out:
        for shard in sorted(shards.glob("[0-9]" * 5)):
            for jpg in sorted(shard.glob("*.jpg")):
                meta = jpg.with_suffix(".json")
                if not meta.exists():
                    continue
                obj = json.loads(meta.read_text())
                cap = obj.get("caption") or obj.get("txt")
                if not cap:
                    continue
                rel = jpg.relative_to(out_dir).as_posix()
                out.write(json.dumps({"image": rel, "caption": cap}) + "\n")
                n += 1
    print(f"wrote {n} entries to {index}")
    return index


def _download_via_tsv(tsv: Path, out_dir: Path, subset: int, image_size: int,
                       processes: int, threads: int, seed: int, keep_sample: bool) -> Path:
    sampled = _sample_tsv(tsv, subset, seed)
    try:
        _run_img2dataset(sampled, out_dir, image_size, processes, threads)
        return _build_index_from_shards(out_dir)
    finally:
        if not keep_sample and sampled.exists():
            sampled.unlink()


# ---- CLI -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=["hf", "tsv"], default="hf")
    ap.add_argument("--out", required=True, help="output root; will contain img/ and index.jsonl")
    ap.add_argument("--subset", type=int, default=50_000)
    # HF-only
    ap.add_argument("--hf_split", default="train", help="split for pixparse/cc3m-wds")
    # TSV-only
    ap.add_argument("--tsv", help="path to CC3M Train_GCC*.tsv (caption<TAB>url)")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep_sample_tsv", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.source == "hf":
        _download_via_hf(out_dir, args.subset, hf_split=args.hf_split)
    else:
        if not args.tsv:
            ap.error("--tsv is required when --source tsv")
        _download_via_tsv(
            Path(args.tsv), out_dir, args.subset, args.image_size,
            args.processes, args.threads, args.seed, args.keep_sample_tsv,
        )


if __name__ == "__main__":
    main()
