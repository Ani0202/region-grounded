# Project TODO: Region-Grounded Fine-Tuning for Late-Interaction VLMs

## Experimental Design Summary

Four models compared on zero-shot semantic segmentation (primary), compositionality (secondary), and global zero-shot (guardrail):

| Model | Data | Loss |
|---|---|---|
| **B0** | none (off-the-shelf SigLIP-B/16) | — |
| **B1** | Flickr30k images + captions | L_global only |
| **M_human** | Flickr30k images + captions + human bboxes | L_global + λ·L_region |
| **M_auto** | Flickr30k images + captions + auto-generated bboxes | L_global + λ·L_region |

**Key comparisons:**
- (M_human − B1): does region loss help with clean supervision? (validates the loss)
- (M_auto − B1): does auto-pipeline produce useful supervision? (validates the pipeline)
- (M_human − M_auto): how much annotation quality matters (the headline contribution)

---

## Part 1: Data Setup & Visualization

**Goal:** Load Flickr30k Entities, understand its structure, validate format before any modeling work.

- [ ] Download Flickr30k Entities (images + annotation XMLs + sentence files)
- [ ] Write a parser for the Entities XML format (phrase ID → bbox; sentence → phrase IDs)
- [ ] Write a `FlickrEntitiesDataset` class returning `(image, [(phrase, bbox), ...])` tuples
- [ ] Visualize 20 random images with bboxes and phrase labels overlaid; save as a grid
- [ ] Compute dataset statistics:
  - Phrases per image (distribution)
  - Bbox area / image area (distribution)
  - Head noun vocabulary size
  - % of phrases with multiple bboxes
  - % of phrases with no bbox
- [ ] Use standard Flickr30k train/val/test splits (don't roll your own)
- [ ] Write dataset stats to a notes file — these inform downstream design choices

**Time:** 1 day. **Output:** working dataloader + visualization figure + stats summary.

---

## Part 2: Benchmark Vanilla SigLIP (B0)

**Goal:** Get the segmentation eval pipeline working end-to-end on a frozen model. Debug the pipeline, not the model.

- [ ] Pick segmentation datasets: PASCAL VOC 2012 (val) + COCO-Stuff (val subset)
- [ ] Load SigLIP-B/16 from Hugging Face; confirm patch-token extraction works
- [ ] Implement patch-token extraction with **both** MaskCLIP-style and SCLIP-style attention modifications
- [ ] Implement the segmentation eval function:
  - Input: model, image, list of class names
  - Encode image → patch features
  - Encode "a photo of a {class}" for each class
  - Per-patch cosine similarity → argmax → label map
  - Bilinear upsample to image resolution
  - Compute mIoU vs ground truth
- [ ] Lock down protocol decisions in a config file:
  - Input resolution (suggest 448, sliding window stride 224)
  - Prompt template (single template, no ensembling)
  - Background handling on VOC (similarity threshold τ_seg, tuned once on val and frozen)
  - No post-processing (no PAMR, no CRF)
- [ ] Run B0 eval on VOC and COCO-Stuff under both extraction methods
- [ ] Sanity-check numbers against published SCLIP/MaskCLIP results
- [ ] Write a `compositionality_eval` stub for ARO + SugarCrepe (defer to Part 6)

**Time:** 2 days. **Output:** B0 numbers, locked eval pipeline, eval config file.

**Gate:** Do not proceed until B0 numbers look sane.

---

## Part 3: Train & Benchmark B1 (Global-Loss Fine-Tune)

**Goal:** Build the training infrastructure. Establish the control baseline.

- [ ] Set up LoRA on SigLIP-B/16 (attention + MLP weights)
- [ ] Build training dataloader yielding `(image, full_caption)` pairs from Flickr30k (5 captions per image — sample one per epoch)
- [ ] Implement standard SigLIP sigmoid global loss
- [ ] Set up training loop: AdamW, cosine LR schedule, mixed precision
- [ ] Pick hyperparameters: LR, batch size, epochs, LoRA rank, LoRA dropout — write in config file
- [ ] Run one short training (1 epoch) and confirm:
  - Loss decreases
  - No NaNs
  - Validation contrastive accuracy improves
  - Segmentation mIoU on held-out slice doesn't catastrophically collapse
- [ ] Run full B1 training
- [ ] Eval B1 on VOC + COCO-Stuff under both extraction methods
- [ ] Save the checkpoint

**Time:** 2–3 days. **Output:** B1 checkpoint, B1 segmentation numbers, debugged LoRA training loop.

**Gate:** B1 trains stably and B1 ≈ B0 or slightly better on segmentation. If much worse, fix overfitting before proceeding.

---

## Part 4: Train & Benchmark M_human (Human BBoxes)

**Goal:** Add the region loss, train with clean supervision, establish the ceiling.

- [ ] Decide phrase-token aggregation: mask stop words (recommended start)
- [ ] Decide multi-bbox handling: each (phrase, bbox) is a separate sample
- [ ] Implement L_region:
  - Given a batch of (image, phrase, bbox) tuples
  - For each item: identify patches whose centers fall inside the bbox
  - For each phrase token: max similarity to patches *within bbox* (positive)
  - For each phrase token: max similarity to patches in *other items' bboxes* (negatives)
  - FILIP-style averaging over phrase tokens
  - Sigmoid loss formulation matching SigLIP's style
- [ ] Build dual dataloader: emits both (image, caption) for L_global and (image, phrase, bbox) for L_region in same batch
- [ ] Combined loss: `L_total = L_global + λ · L_region`
- [ ] **λ sweep on M_human:** train with λ ∈ {0.1, 0.5, 1.0, 2.0}
- [ ] Eval each λ checkpoint; pick best λ on val mIoU
- [ ] Run final M_human with chosen λ, 2–3 seeds if compute allows
- [ ] Eval M_human on VOC + COCO-Stuff under both extraction methods

**Time:** 4–5 days (including λ sweep). **Output:** M_human checkpoint(s), λ value chosen, ceiling numbers.

**Gate:** M_human should beat B1 on segmentation. If not, L_region has a bug or λ is wrong — debug before Part 5.

---

## Part 5: Generate Auto-BBoxes & Train M_auto

**Goal:** Replace human bboxes with auto-generated bboxes; measure the gap.

### 5a: Build the auto-localization pipeline

- [ ] Decide region-first vs phrase-first (**recommend: phrase-first**)
- [ ] If phrase-first:
  - For each Flickr30k phrase, compute SigLIP/SCLIP similarity to all patches in image
  - Threshold to get an activation mask
  - Take the tightest bbox around activated region (with minimum size)
- [ ] Run pipeline on 100 sample images; visualize auto-bboxes alongside human bboxes
- [ ] Compute IoU between auto-bbox and human-bbox per phrase; report mean and distribution
- [ ] If mean IoU < 0.2, iterate on threshold/method before scaling
- [ ] (Optional) Add quality filter: drop (phrase, auto-bbox) pairs where SigLIP similarity between phrase and cropped region is below τ
- [ ] Generate auto-bboxes for full Flickr30k train split
- [ ] Save as parallel annotation file matching Flickr30k's structure

### 5b: Train M_auto

- [ ] Use the *same* training code from Part 4, just swap the bbox source
- [ ] Use the λ chosen from M_human's sweep — **do not re-sweep** (M_auto must use same hyperparameters as M_human)
- [ ] Run M_auto, 2–3 seeds if compute allows
- [ ] Eval on VOC + COCO-Stuff

**Time:** 4–5 days. **Output:** auto-bbox annotation set, IoU-vs-human stats, M_auto checkpoint(s), M_auto numbers.

---

## Part 6: Analysis, Comparison & Writeup

**Goal:** Apply the pre-registered decision rule, write the paper.

- [ ] Fill in headline table: {B0, B1, M_human, M_auto} × {VOC, COCO-Stuff} × {MaskCLIP-extract, SCLIP-extract}
- [ ] Compute seed variance for each cell; mark cells where M_auto − B1 > 2σ
- [ ] Apply pre-registered decision rule:
  - **Strong success:** M_auto − B1 > 2 mIoU averaged across cells, M_auto ≥ B1 in all cells
  - **Weak success:** M_auto − B1 > 2σ in ≥2 cells, no cell worse than B1 by >2σ
  - **Null:** M_auto within seed noise of B1
  - **Failure:** M_auto significantly worse than B1 anywhere
- [ ] Compositionality eval on ARO + SugarCrepe for all four models
- [ ] Global zero-shot guardrail: ImageNet-1K + CIFAR-100 top-1 for all four
- [ ] Qualitative figure: 10 images × 4 models × 3 free-form text queries = open-vocab heatmaps
- [ ] Ablations (only if time):
  - Stop-word masking vs no masking
  - Region-first vs phrase-first auto pipeline
  - λ sensitivity
- [ ] Writeup:
  - Methods section (concrete, with L_region formula)
  - Results table + qualitative figure
  - The (M_human − M_auto) gap analysis — the headline contribution
  - Honest limitations section (small dataset, single backbone, λ tuned on M_human)

**Time:** 3–4 days. **Output:** the actual deliverable.

---

## Compute Budget

Rough A100-hour estimate against 300 Colab credits (~25–30 A100-hours):

| Part | A100-hours |
|---|---|
| Part 2 (B0 eval) | 1–2 |
| Part 3 (B1 training + eval) | 3–4 |
| Part 4 (M_human × 4 λ + 2 seeds at best λ) | 15–20 ⚠️ |
| Part 5a (auto-bbox generation, inference) | 2–3 |
| Part 5b (M_auto, 2 seeds) | 6–8 ⚠️ |
| Part 6 (comp + global evals) | 2 |
| **Total** | **30–40** (over budget) |

**Cuts, in order if needed:**
1. Drop one extraction method from eval (pick SCLIP-style OR MaskCLIP-style)
2. Drop seeds (1 seed per condition, accept the noise)
3. Reduce λ sweep to 2 values instead of 4
4. Fall back to SigLIP-B/32 backbone

Do not cut M_auto seeds before M_human λ sweep — clean supervision needs the cleanest experiment.

---

## Week Mapping (5 weeks)

- **Week 1:** Parts 1 + 2 (data, B0 eval)
- **Week 2:** Part 3 (B1) + start Part 4 (L_region implementation)
- **Week 3:** Part 4 (M_human + λ sweep)
- **Week 4:** Part 5 (auto-pipeline + M_auto)
- **Week 5:** Part 6 (analysis, writeup, ablations if time)

---

## Pre-Registered Commitments (write these down BEFORE running experiments)

1. **Primary metric:** mean mIoU across {VOC, COCO-Stuff} × {MaskCLIP-extract, SCLIP-extract}, averaged over seeds.
2. **Extraction protocol:** locked in Part 2 config file. Same for all four models.
3. **λ selection:** swept on M_human only, applied as-is to M_auto.
4. **Seeds:** target 3 per condition; minimum 2 if compute forces it.
5. **Decision rule:** as stated in Part 6.

---

## Open Decisions to Make Early

- [ ] Region-first vs phrase-first for auto-pipeline (recommend phrase-first)
- [ ] Stop-word masking on phrase tokens in L_region (recommend yes, as default)
- [ ] Multi-bbox handling for phrases like "two dogs" (recommend: separate samples per bbox)
- [ ] Input resolution for segmentation eval (recommend 448, sliding window stride 224)
- [ ] Backbone: SigLIP-B/16 vs SigLIP-B/32 (start with B/16, fall back if compute-starved)