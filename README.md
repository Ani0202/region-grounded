# GRAFT: Grounded Region-Augmented Fine-Tuning

**Siddharth Raj and Aniket Agrawal — TTIC 31280, Spring 2026**

Fine-tunes SigLIP-B/16-384 with explicit region-level supervision (FILIP-style patch–token loss) derived from human-annotated and auto-generated phrase–bounding-box pairs, improving patch-level spatial grounding in a model originally trained only on global image–text contrastive objectives.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Most notebooks require a Hugging Face token for dataset access:

```bash
export HF_TOKEN=hf_...
```

Training notebooks (`graft-training.ipynb`) are designed for **Google Colab with an L4 GPU** (23.7 GB VRAM). Evaluation notebooks (`02d`, `03`, `06`) run locally on CPU.

---

## Notebooks

### Data exploration

| Notebook | Purpose | Where to run |
|---|---|---|
| `flickr-30-analysis.ipynb` | Downloads `nlphuji/flickr30k` from HuggingFace, visualises sample images and captions. Run once to populate the HF cache. | Local |
| `flickr-30k-entities-analysis.ipynb` | Loads Flickr30k Entities XML annotations (phrase–bbox pairs), analyses phrase statistics and bbox distributions. Requires the Entities annotations zip — see cell 2 for the download. | Local |

### Baselines and evaluation

| Notebook | Purpose | Where to run |
|---|---|---|
| `02_b0_benchmark.ipynb` | Zero-shot segmentation eval of frozen SigLIP-B/16-384 on PASCAL VOC 2012 val. Establishes the B0 baseline; results locked and reused across all trained-model comparisons. | Local or Colab |
| `02b_segmentation_viz.ipynb` | Companion to `02_b0_benchmark`. Visualises per-class cosine-similarity heatmaps and compares raw-encoder vs. MaskCLIP-style patch features. | Local |
| `02d_maskclip_eval.ipynb` | Runs the same VOC 2012 val eval on **CLIP ViT-B/16** with three patch-extraction methods: standard, MaskCLIP, and SCLIP. Produces the CLIP reference rows in the results table. | Local |
| `03_siglip_b0_eval.ipynb` | Focused τ-sweep for frozen SigLIP B0 on VOC 2012 val. Run once; record the best τ in `report.md`. | Local or Colab |
| `06_pointing_game_eval.ipynb` | Evaluates frozen SigLIP-B/16-384 on two Flickr30k phrase-grounding metrics: **Pointing Game Accuracy** and **Recall@1 (IoU ≥ 0.5)**. Uses 200 val images (seed=42) and MaskCLIP-style patch extraction. Results are the B0 baselines for the trained-model comparison. | Local (CPU) |

**Running `06_pointing_game_eval.ipynb`:**
1. Ensure `flickr-30-analysis.ipynb` has been run so Flickr30k parquet shards are in the HF cache (`~/.cache/huggingface/hub/datasets--nlphuji--flickr30k/`).
2. Ensure Flickr30k Entities XML annotations are in `notebooks/data/flickr30k_entities/` (downloaded by `flickr-30k-entities-analysis.ipynb` cell 2).
3. Run all cells. Outputs are printed; copy results into `report.md`.

### Auto-supervision

| Notebook | Purpose | Where to run |
|---|---|---|
| `05a_florence2_auto_bbox.ipynb` | Pilots Florence-2 `<DENSE_REGION_CAPTION>` on 5 Flickr30k images. Downloads Florence-2 base locally to `notebooks/florence2_local/` (≈885 MB), patches a flash-attn import, and visualises (bbox, label) outputs. | Local |

**Note:** `notebooks/florence2_local/` is git-ignored because it contains large model weights. The notebook re-downloads it automatically on first run.

### Training

| Notebook | Purpose | Where to run |
|---|---|---|
| `graft-training.ipynb` | Main training notebook. Runs SigLIP-B/16-384 + LoRA (r=4, α=16, targets `q_proj`/`v_proj`) on Flickr30k Entities. Change the config dict in cell 2 to switch between experiments (B1 global-only, M_human with region loss, M_auto with Florence-2 annotations). Logs to W&B. | **Colab L4 GPU** |

**Running `graft-training.ipynb` on Colab:**
1. Upload the notebook (or clone the repo with a PAT — see `notebooks/00_setup.ipynb`).
2. Set runtime to **L4 GPU** (Runtime → Change runtime type).
3. Cell 0 installs dependencies; run it first.
4. Set `HF_TOKEN` and `WANDB_API_KEY` when prompted in cell 1.
5. Edit the `CFG` dict in cell 2, then **Run All**.

---

## Results

Baselines and training results are logged in [`report.md`](report.md).

The intermediate progress report is in [`intermediate_report.tex`](intermediate_report.tex) (compile with `pdflatex intermediate_report.tex`).

---

## Project structure

```
.
├── notebooks/
│   ├── data/flickr30k_entities/   # Entities XML annotations (git-ignored)
│   ├── florence2_local/           # Florence-2 weights (git-ignored, auto-downloaded)
│   ├── 02_b0_benchmark.ipynb
│   ├── 02b_segmentation_viz.ipynb
│   ├── 02d_maskclip_eval.ipynb
│   ├── 03_siglip_b0_eval.ipynb
│   ├── 05a_florence2_auto_bbox.ipynb
│   ├── 06_pointing_game_eval.ipynb
│   ├── flickr-30-analysis.ipynb
│   ├── flickr-30k-entities-analysis.ipynb
│   └── graft-training.ipynb
├── scripts/
│   └── flickr30k_entities/        # Annotation download helpers
├── output.png                     # Florence-2 viz used in LaTeX report
├── report.md                      # Running results log
├── intermediate_report.tex        # Progress report (LaTeX)
├── requirements.txt
└── CV_project_proposal_final.pdf
```
