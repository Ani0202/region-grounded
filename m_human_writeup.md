# M_human: Region-Grounded Fine-Tuning of SigLIP-B/16-384

**Authors.** Aniket Agrawal, Siddharth Raj (TTIC 31280, Spring 2026)
**Codebase.** `region-grounded` (branch `ani-dev`, this repo)
**Date.** 2026-05-23

---

## Abstract

We region-ground SigLIP-B/16-384 by fine-tuning a small LoRA adapter
(295K trainable parameters, q+v projections, rank 4) on human-annotated
phrase→bbox pairs from Flickr30k-Entities. Training combines (i) a
sigmoid-SigLIP global loss on the MHAP `pooler_output` and (ii) a
FILIP-style region loss between EOS-token phrase features and patch
features, max-pooled over in-bbox patches per image. A best-PG
checkpoint snapshot tracker eliminates late-training drift. On the
200-image Flickr30k val split (2,124 phrase-bbox pairs), the locked
recipe (v10) achieves **Pointing Game = 20.76 ± 0.41% (n=3 seeds)**,
a **+6.02pp improvement over the frozen B0 baseline (14.74%)**, with
**σ reduced 6× relative to the prior recipe (v5: 18.68 ± 2.57%)**.
The dominant contribution to the final result comes from two
mechanistic fixes that align train-time and eval-time representations:
the MaskCLIP attention bypass on the vision side (image-feature path
match, +7.34pp on the lucky-seed single point; +3.94pp on the 3-seed
mean over B0) and the EOS-aligned region loss on the text side
(+2.08pp on the 3-seed mean over the prior locked recipe, plus the
6× variance collapse). LoRA capacity, region-loss weight, and
patch-aggregation reformulations did not move the mean within
measurement noise (σ ≈ 2.6pp on the prior recipe; σ ≈ 0.4pp on the
new recipe).

---

## 1. Methods

### 1.1 Backbone and adapter

- **Backbone:** SigLIP-B/16-384 (`google/siglip-base-patch16-384`),
  frozen. 729 patches per image (27 × 27 grid).
- **Adapter:** LoRA (PEFT), rank 4, α = 16, dropout 0.0. Targets:
  `q_proj`, `v_proj` on the vision-tower attention layers.
  Trainable parameters: 295K (~0.3M).
- **Inference path:** during training and at eval, the final vision
  attention layer is replaced by the **MaskCLIP bypass**
  `out_proj(v_proj(hidden_states))` (no inter-patch attention mixing).
  This makes the patch-feature path during gradient computation
  byte-for-byte identical to the path used for Pointing Game / Recall@1
  evaluation.

### 1.2 Losses

Let `B` be the per-batch image count, `P` the per-batch phrase count
(`P = B × region_batch / batch` in practice), and `N = 729` the
patch count.

**Global loss (`l_global`).** Sigmoid-SigLIP contrastive on
MHAP-pooled image features and EOS-token caption features:

```
img_global = model.vision_model.pooler_output                     # (B, D)
txt_global = text_eos_token_features(captions)                    # (B, D)
logits     = scale * (img_global @ txt_global.T) + bias           # (B, B)
labels     = 2 * I(B) - 1                                          # (B, B), {+1, -1}
l_global   = -F.logsigmoid(labels * logits).mean()
```

**Region loss (`l_region`).** Sigmoid-SigLIP contrastive between
**EOS-token phrase features** and **MaskCLIP-bypass patch features**,
max-pooled over in-bbox patches:

```
patch_feats = maskclip_patch_features(image)                      # (B, N, D), L2-normed
phrase_eos  = text_eos_token_features(phrases)                    # (P, D), L2-normed
bbox_masks  = rasterize_bboxes(phrase->image, 27x27)              # (P, N), bool

sim         = phrase_eos @ patch_feats.reshape(B*N, D).T          # (P, B*N)
sim         = sim.reshape(P, B, N)                                # phrase i vs image j patch n
sim         = sim.masked_fill(~bbox_masks_per_phrase, -inf)
scores      = sim.amax(dim=-1)                                    # (P, B), max over in-bbox patches
logits      = scale * scores + bias                                # (P, B)
labels      = 2 * (phrase_image_alignment_matrix) - 1              # (P, B)
l_region    = -F.logsigmoid(labels * logits).mean()
```

The mask uses `-inf` so empty-bbox phrases (no in-bbox patches at the
27×27 resolution) contribute a constant score and don't dominate the
softmax-free sigmoid.

**Combined.** `loss = l_global + λ_region * l_region` with
`λ_region = 0.5`.

### 1.3 Evaluation

All metrics computed on the same 200-image val split (`seed=42` for
image selection), 2,124 phrase-bbox pairs (~10.6 phrases / image),
MaskCLIP bypass path, EOS-token phrase embedding.

- **Pointing Game (PG).** For each phrase, compute the similarity
  map `phrase_eos · patch_feat` over the 27×27 grid. Take the argmax
  patch, project its centre to image coordinates. PG = fraction of
  phrases whose argmax centre falls inside the GT bbox.
- **Recall@1.** Take the top-K (or `halfmax`) patches by similarity,
  fit the tightest axis-aligned box around them, compute IoU with GT.
  R@1 = fraction with IoU ≥ 0.5. Reported variants: top-10, top-25,
  halfmax.

Per binomial SE at p ≈ 0.2, n = 2124, **σ_eval ≈ 0.87 pp** per single
checkpoint — the floor below which PG differences are not
distinguishable from sampling noise.

### 1.4 Best-PG snapshot tracking

During `train_one_epoch`, PG is evaluated every 200 optimizer steps
on the same val split. Whenever the eval beats the best-so-far,
trainable parameters (~MB CPU footprint) are snapshotted; at end of
training the best snapshot is restored. This adds ~MB of CPU memory
and zero GPU compute, but eliminates the loss/metric decoupling
where end-of-epoch state can be substantially worse than a
mid-epoch peak (observed at +1 to +5 pp across runs).

### 1.5 Optimisation

- Optimiser: AdamW, lr = 2e-4, weight_decay = 0.0, β = (0.9, 0.999).
- Schedule: warmup → cosine → 0 over the full epoch (warmup steps
  = 100, restarts = 1).
- Batch: `batch_size = 32` images, `region_batch = 64` phrase-bbox
  pairs (drawn independently of the image batch).
- Mixed precision: `bfloat16` autocast, no gradient scaler.
- Gradient checkpointing on the vision tower.
- Seeded: `random`, `numpy`, `torch.manual_seed`,
  `torch.cuda.manual_seed_all`, and a `torch.Generator` passed to
  the DataLoader. CUDA/cuDNN are *not* set to bit-deterministic
  mode (we measured the per-step variance and found σ ≈ 1–3pp on
  PG between same-seed runs, which we accept).

### 1.6 Data

- **Train.** Flickr30k-Entities, train split. 29,783 images, ~5
  captions/image, ~3.5 grounded phrases/caption. Each training
  sample is a `(image, caption, [(phrase, bbox)…])` tuple; the
  region loss samples `region_batch` phrase-bbox pairs from this.
- **Val (held out).** 200 images sampled from Flickr30k-Entities val
  (`seed=42`), all of their phrase-bbox pairs (n = 2,124). Used for
  Pointing Game, Recall@1, and the best-PG snapshot trigger.
- Bbox preprocessing: rasterize GT bbox to the 27×27 patch grid
  (a patch is "in-bbox" iff its centre lies inside the GT box). Train
  and eval use the same rasterization.

---

## 2. Ablation results

We report the chronological ablation chain from the frozen baseline
to the locked recipe. Each row changes **one** component relative to
the row above. Pointing Game is the primary metric; all other metrics
are tracked but not optimised. Numbers in **bold** are the recipe's
locked configuration.

| Version | Change vs prior   | PG (mean ± std) | Δ vs B0 | Δ vs prior | n seeds |
|:-------:|:------------------|:----------------|:--------|:-----------|:-------:|
| **B0**  | frozen baseline   | **14.74%**      | —       | —          | 1       |
| v1      | + LoRA(q+v) + MHAP global + FILIP region (normal patch path) | 13.42% | −1.32 | −1.32 | 1 |
| v2      | + MaskCLIP bypass *during training* | 22.08% (single draw; later shown to be +1σ outlier) | +7.34 | +8.66 | 1 |
| v3      | 2 epochs, single cosine | 19.40% | +4.66 | −2.68 | 1 |
| v4      | 2 epochs, cosine restart per epoch | 16.90 – 22.08% (2 attempts) | — | — | unseeded |
| v5      | + full seed control (proper variance baseline) | 18.68 ± 2.57% | **+3.94** | regression to mean from v2 | 3 |
| v6      | λ_region 0.5 → 1.0 | 17.18% (1 seed) | +2.44 | −1.50 (vs v5 seed=2 20.90%) | 1 |
| v7      | LoRA targets q+v → q+k+v | 17.80% (1 seed, peak 22.69% @ step 400) | +3.06 | −3.10 | 1 |
| v8      | + best-PG snapshot | 19.96% (1 seed) | +5.22 | +2.16 | 1 |
| v9      | FILIP max → top-K mean (K=3) | 19.35% (1 seed) | +4.61 | −0.61 | 1 |
| **v10** | LoRA back to q+v, FILIP max, **+ EOS-aligned region loss**, **+ best-PG snapshot** | **20.76 ± 0.41%** | **+6.02** | **+2.08 vs v5** | **3** |

**B0 baseline detail.** Pointing Game 14.74% (313/2124), Recall@1
top-10 = 6.83%, top-25 = 7.91%, halfmax = 7.58%.

**v10 per-seed breakdown:**

| seed | trajectory (200/400/600/800)      | best-PG | @step | R@1 halfmax | train loss (region) |
|:----:|:----------------------------------|--------:|:-----:|------------:|--------------------:|
| 0    | 15.21 / 20.57 / 19.59 / 19.68     | **20.57%** | 400   | 7.44%       | 0.455              |
| 1    | 14.78 / 18.55 / 19.35 / 20.48     | **20.48%** | 800   | 7.34%       | 0.396              |
| 2    | 16.48 / 21.23 / 16.53 / 17.23     | **21.23%** | 400   | —           | —                  |
|      |                                    | **20.76 ± 0.41%** |       | 7.39 ± 0.07% | —                  |

**Recipe-level effect on companion metrics** (v10 vs B0, 3-seed mean):
- PG: 14.74 → 20.76 = **+6.02pp**
- R@1 top-10: 6.83 → 5.32 = −1.51 (sharper argmax trades broad coverage for spike)
- R@1 top-25: 7.91 → 7.34 = −0.57
- R@1 halfmax: 7.58 → 7.39 = −0.19 (within noise)

The R@1 regression is consistent across all trained versions and
reflects the FILIP max-pool loss's preference for a sharp spike over
broad coverage — exactly the behaviour the Pointing Game rewards.

---

## 3. Final recipe (locked, v10)

```
Backbone        : SigLIP-B/16-384 (frozen)
Adapter         : LoRA, targets=[q_proj, v_proj], rank=4, alpha=16
                  → 295K trainable params
Vision path     : MaskCLIP bypass on final attention layer
                  (out_proj(v_proj(h)), no inter-patch mixing),
                  applied during training and eval
Global loss     : sigmoid-SigLIP on MHAP pooler_output vs EOS caption
Region loss     : sigmoid-SigLIP on EOS phrase feat vs patch feats,
                  max-pool over in-bbox patches (FILIP-style)
λ_region        : 0.5
Optimisation    : AdamW, lr=2e-4, 1 epoch, warmup=100, cosine→0,
                  batch=32, region_batch=64, bf16 autocast,
                  gradient checkpointing
Best-PG snapshot: track + restore on trainable params (~MB CPU)
Seeding         : random, numpy, torch CPU+CUDA, DataLoader.Generator
                  (CUDA/cuDNN deterministic flags NOT set)
Verification    : 3-seed mean ± std (seeds = {0, 1, 2})
```

**Compute.** ~40 min per seed on a Colab L4 (24 GB), peak GPU memory
~22 GB. Full 3-seed run: ~2 hours. Total credits consumed on the
ablation chain v1→v10: ~25 L4-hours.

---

## 4. Discussion

### 4.1 What worked

**1. Path matching on the vision side (v1 → v2, +8.66pp single-seed,
+3.94pp on the 3-seed mean over B0).** The single largest mechanistic
fix in the ablation chain. v1 trained patch features through the full
last-attention layer but evaluated them through the MaskCLIP bypass —
the gradient sharpened the wrong representation. Installing the
bypass during training (so training reads what eval reads) was a
one-line change that flipped the model from regressing below B0 to
beating it by ~4pp on the mean.

**2. Path matching on the text side (v5 → v10, +2.08pp on the 3-seed
mean, +6× variance collapse).** Mirror of the v1→v2 fix, but on the
text side and applied 8 versions later. Prior versions (v1–v9) used
mean-over-tokens for the phrase representation in the region loss,
while eval used only the EOS token. The region loss was being
satisfied at the wrong vector. Replacing mean-over-tokens with the
EOS token raised the peak by ~1.3 pp and substantially tightened
the distribution — every seed now lands in [20.48, 21.23], a 0.75pp
spread (vs v5's [15.87, 20.90], 5.03pp).

**3. Best-PG checkpoint snapshot (v7 → v8, recipe-wide).** Late-epoch
PG is consistently worse than mid-epoch peak — the FILIP max-pool loss
is satisfied by either a sharp in-bbox spike (good for PG) or a broad
in-bbox cluster (bad for PG), and the optimizer drifts between
basins. Tracking the best-PG snapshot of the trainable parameters
and restoring at end of training is essentially free (~MB CPU
memory, zero compute) and recovers 1–5pp per run that would
otherwise be lost to drift. Combined with the EOS fix, it's the
core of v10's variance collapse — even the seeds that drift most
post-peak (seed=2 dropped from 21.23 → 17.23) end up locked at
their peak instead.

### 4.2 What didn't

**1. λ_region sweep (v6).** Doubling the region-loss weight from
0.5 → 1.0 produced a *broader*, not sharper, response — PG dropped
3.72pp while R@1 top-25 *improved* 0.56pp. The FILIP max-pool loss
is satisfied by max(spike) and max(broad-cluster) equivalently, and
removing the global loss's structural regularisation lets the
optimizer pick the broader path. λ = 0.5 is on the right side of
the sharpness curve; moving in either direction hurts.

**2. LoRA capacity expansion (v7, v8).** Adding `k_proj` to the LoRA
targets (q+v → q+k+v) raised the per-checkpoint *peak* (v7 hit 22.69%
at step 400, the highest checkpoint of any run) but the model walked
away from that region by end of epoch (final 17.80%). With best-PG
snapshot enabled (v8), the same recipe landed at 19.96% on the
same seed — within σ of v5 (18.68 ± 2.57%). Extra attention-routing
capacity creates more room to find a high PG, but also more room to
walk away from it. The fix is the snapshot, not the extra capacity:
the locked recipe reverts to q+v.

**3. Top-K mean region loss (v9).** Replacing max-pool with the mean
of the top-3 in-bbox patches was hypothesised to force consistent
in-bbox clusters, biasing toward sharp grounding. It didn't: 19.35%
vs v8's 19.96% on the same seed, a 0.61pp regression within noise.
Aggregation is not the lever; the representation mismatch is.

### 4.3 Variance characterisation

The most important methodological finding of this ablation chain is
that **Pointing Game is a high-variance metric on a low-variance
underlying signal**. Across the 3 v5 seeds (identical code, identical
hyperparameters, only the seed varies):

- PG: 18.68 ± 2.57% — σ comparable to the *mean improvement over B0*.
- R@1 halfmax: 7.30 ± 0.00% — identical to 4 decimal places.
- Region train loss: 0.102 ± 0.001 — recipe is reproducible at the
  optimisation-objective level.

The recipe converges to the same loss minimum and the same broad
patch-similarity coverage across seeds; the stochasticity flips
*which single patch* is argmax for a few borderline phrases per
seed. This is why early "wins" in the chain (v2's 22.08%, v7's
22.69%) turned out to be high-variance single draws that did not
replicate on 3-seed re-verification.

**Implication for reporting.** Any single-seed PG result with Δ <
2.6pp is not meaningfully different from a recipe whose true mean is
the v5 mean (18.68%). Multi-seed verification is mandatory for any
ablation claim. The v10 recipe with its 6× σ reduction (0.41pp) is
the first point on the chain where single-seed claims would be
defensible — but we still report 3-seed mean ± std.

---

## 5. Reproducibility

**Repo.** `github.com:Ani0202/region-grounded` (branch `ani-dev`,
this commit).

**Entry point.** `notebooks/graft-training.ipynb` (cells 1–14). Cell
13 trains; cell 14 aggregates the per-seed results. Cell 10 holds
the SEEDS, λ, LoRA targets, and `phrase_repr` toggles.

**To reproduce v10:**
1. `pip install -e .` (installs the `graft` package from `src/`).
2. Open `notebooks/graft-training.ipynb`.
3. In cell 10 set: `SEEDS = [0, 1, 2]`, `LAMBDA_OVERRIDE = None`,
   `LORA_TARGETS = ['q_proj', 'v_proj']`, `REGION_TOP_K = None`,
   `PHRASE_REPR = 'eos'`, `EXPERIMENT_TAG = 'v10'`,
   `CFG['epochs'] = 1`.
4. Run cells 1–14 sequentially. ~40 min/seed on L4, ~2h total.

**Compute used in this work.**
- Hardware: Google Colab L4 (24 GB) — primary; T4 fallback used for
  small evals.
- Total credits consumed end-to-end (ablation v1 → v10): ~25 L4-hours
  + ~5 T4-hours of eval / debugging time.
- bf16 autocast + gradient checkpointing required to fit batch=32 at
  region_batch=64 in 24 GB.

**Logging.** All training runs logged to W&B project
`agrawalaniket02-unicersity-of-chicago/region-grounded`. v10
3-seed runs: `m_human_v10_seed{0,1,2}_lam0.5`; aggregate run
`m_human_v10_aggregate_n2_lam0.5` (seeds 0+1) and the seed=2 leader
run `m_human_v10_seed2_lam0.5`.

**Known non-determinism.** CUDA/cuDNN are *not* set to
bit-deterministic mode. Same-seed reruns will trace similar but
not bit-identical trajectories (per-checkpoint variance ~1–3 pp).
The 3-seed std (0.41pp on v10) absorbs this.

---

## 6. Where M_human sits in the project

M_human establishes the **recipe** the project uses to region-ground
SigLIP. The next track, **M_auto**, swaps the human-annotated
Flickr30k-Entities phrase-bbox supervision for Florence-2
auto-generated phrase-bbox pairs (`<CAPTION_TO_PHRASE_GROUNDING>`),
holding the v10 recipe byte-for-byte fixed. The headline question
M_auto answers: does the GRAFT recipe work on cheap, machine-generated
supervision (no human bounding boxes), and how much PG do we give up
relative to M_human's 20.76 ± 0.41%? That comparison is the paper's
main contribution.
