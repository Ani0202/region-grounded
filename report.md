# Region-Grounded Segmentation — Results

## Baselines

### CLIP ViT-B/16 — zero-shot VOC 2012 val (02d_maskclip_eval.ipynb)

All 1449 val images, τ=−0.1 (tuned on 100 images), single prompt template `"a photo of a {}"`.

| Method    | mIoU  | Published |
|-----------|------:|----------:|
| standard  |  8.46% | —        |
| maskclip  | 19.90% | 22.4%    |
| sclip     | 14.01% | 35.6%    |

Published numbers (MaskCLIP / SCLIP papers) may use 336 px input, prompt ensembling, or PAMR post-processing.

### SigLIP-B/16-384 — frozen B0 (graft-training.ipynb, cell-10)

200 val images, τ=0.0, MaskCLIP-style last-attention bypass, no projection head applied.

| Model | mIoU |
|-------|-----:|
| B0 (frozen SigLIP-B/16-384) | 1.43% |

Low baseline is expected: SigLIP was trained with sigmoid contrastive loss on global (MHAP-pooled) features; individual patch tokens were never directly aligned to text tokens. τ=0.0 also suppresses more patches than the τ=−0.1 used for CLIP. See `03_siglip_b0_eval.ipynb` for the full τ sweep.

### SigLIP-B/16-384 — Flickr30k Pointing Game + Recall@1, frozen B0 (06_pointing_game_eval.ipynb)

200 val images (seed=42), MaskCLIP-style last-attention bypass, EOS token as phrase embedding.
2,124 phrase-bbox pairs evaluated (~10.6 pairs/image on average).

**Pointing Game** (argmax patch centre falls inside GT bbox):

| Model | Val images | Phrase-bbox pairs | Correct hits | Pointing Game Acc. |
|-------|----------:|------------------:|-------------:|-------------------:|
| B0 (frozen SigLIP-B/16-384) | 200 | 2,124 | 313 | **14.74%** |

Worst 5 images (0% accuracy): 289625522.jpg (0/5), 481054596.jpg (0/13), 6371136393.jpg (0/19), 2579268572.jpg (0/7), 4407490214.jpg (0/12).
Best 5 images: 1752454466.jpg (4/4, 100%), 3578841731.jpg (3/4, 75%), 1206506157.jpg (3/4, 75%), 210625425.jpg (3/4, 75%), 4678723492.jpg (4/6, 67%).

**Recall@1 at IoU ≥ 0.5** (tight box around top-k patches vs GT bbox):

| Model | Patch selection | Hits | Total | Recall@1 |
|-------|----------------|-----:|------:|---------:|
| B0 (frozen SigLIP-B/16-384) | top-10 | 145 | 2,124 | **6.83%** |
| B0 (frozen SigLIP-B/16-384) | top-25 | 168 | 2,124 | **7.91%** |
| B0 (frozen SigLIP-B/16-384) | halfmax | 161 | 2,124 | **7.58%** |

These are the B0 baselines for M_human and M_auto comparisons. The gap between pointing game (14.74%) and Recall@1 (~7%) indicates the top-k patches are spatially scattered — the model has weak directional signal but no coherent spatial clustering around the referred entity.

---

## Training Experiments

### Exp-01 — B1 vs M_human, LoRA rank=4, 1 epoch (graft-training.ipynb)

SigLIP-B/16-384 + LoRA (r=4, α=16, targets=[q_proj, v_proj]), 1 epoch on Flickr30k,
lr=2e-4, batch=32, region_batch=16, τ=0.0 at eval, W&B run `fs6f3id0`.

| Model | λ_region | train loss (global) | train loss (region) | VOC mIoU |
|-------|--------:|--------------------:|--------------------:|---------:|
| B0 (frozen) | — | — | — | 1.43% |
| B1 | 0.0 | 0.175 | — | 0.25% |
| M_human | 1.0 | 0.182 | 0.288 | 0.18% |

**Finding:** both trained models regressed below frozen B0. Root cause: the global loss
uses `mean(last_hidden_state)` which pushes all patch features uniformly toward the caption,
collapsing spatial diversity. τ=0.0 post-training threshold is also mismatched to the shifted
similarity distribution. Next: (1) sweep τ for trained models, (2) replace mean-pool with
MHAP pooler_output for global loss so patch features are not directly collapsed.
