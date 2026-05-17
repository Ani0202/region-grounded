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

---

### Exp-02 — M_human v1 (MHAP pooler global + normal-path patch features)

SigLIP-B/16-384 + LoRA (r=4, α=16, targets=[q_proj, v_proj]), 1 epoch on Flickr30k,
λ_region=0.5, lr=2e-4, batch=32, region_batch=64, warmup=100, AMP + gradient
checkpointing, Colab L4. Global loss switched from `mean(last_hidden_state)` to
MHAP `pooler_output` (Exp-01 fix). Primary metric is Pointing Game; VOC kept as
guardrail. W&B run `fj677u5e`.

| Metric             |     B0 |   M_human v1 |       Δ |
|--------------------|-------:|-------------:|--------:|
| Pointing Game      | 14.74% |       13.42% |   -1.32 |
| Recall@1 top-10    |  6.83% |     **7.11%** |   +0.28 |
| Recall@1 top-25    |  7.91% |        7.58% |   -0.33 |
| Recall@1 halfmax   |  7.58% |        7.30% |   -0.28 |
| VOC mIoU (guard)   |  1.43% |    **0.19%** |   -1.24 |

Train: global=0.249, region=0.094 (region loss dropped 0.15 → 0.09 over the epoch — task
is being learned). Mid-training PG every 200 steps: 14.74 → 13.75 → 14.31 → 13.79 → 13.42
(plateau under B0; 3 consecutive evals below baseline, > 1σ from B0 given binomial
SE ≈ 0.77pp at n=2124).

**Finding:** Region loss is being optimised but PG / VOC regressed while R@1 top-10
*improved*. This is the signature of a **train/eval patch-path mismatch**: training
read patch features through the full last-attention layer (`get_patch_feats` →
`out.last_hidden_state` after normal forward), while eval reads them through the
MaskCLIP last-attention bypass (`out_proj(v_proj(hidden_states))`, no inter-patch
mixing). The gradient sharpens patches under one representation; the metrics
measure the other. The +0.28 on R@1 top-10 confirms the model is learning useful
region signal — it's just spread across multiple patches in a way that helps
top-K coverage but moves the *single best* patch (PG) away from the GT bbox
centre.

**Next:** Exp-03 = install the MaskCLIP bypass permanently on the last vision
attention layer for the entire training run, so the gradient flows through the
same representation eval reads. All other hyperparameters held constant
(λ=0.5, lr=2e-4, 1 epoch) for a clean A/B vs v1. If Exp-03 still underperforms
B0, then a λ ∈ {0.1, 1.0} mini-sweep follows.

---

### Exp-03 — M_human v2 (MaskCLIP bypass installed during training)

Same hyperparameters as v1 (λ_region=0.5, lr=2e-4, 1 epoch, batch=32,
region_batch=64, LoRA r=4 on q_proj+v_proj). The only change from v1 is
`install_maskclip_bypass(model_mh)` is called before training so the gradient
flows through the same v_proj-only path that eval reads. W&B run `1yqs3mqy`.

**Primary metric beaten by +7.34 pp.**

| Metric             |     B0 |   M_human v1 |  M_human v2  | Δ vs B0 |
|--------------------|-------:|-------------:|-------------:|--------:|
| Pointing Game      | 14.74% |       13.42% |  **22.08%**  |  +7.34  |
| Recall@1 top-10    |  6.83% |        7.11% |        5.60% |   -1.23 |
| Recall@1 top-25    |  7.91% |        7.58% |        7.34% |   -0.57 |
| Recall@1 halfmax   |  7.58% |        7.30% |        7.30% |   -0.28 |
| VOC mIoU (guard)   |  1.43% |        0.19% |        0.28% |   -1.15 |

Train: global=0.263, region=0.102 (region task learned similarly to v1).
Mid-training PG every 200 steps: 14.74 → 16.20 → 19.59 → 21.19 → 22.18 → 22.08
(monotonic; final eval dipped 0.1 pp from step 800, within noise).
At p≈0.22, n=2124, binomial SE ≈ 0.90 pp; the +7.34 pp move over B0 is ~8.2σ
and the +8.66 pp move v2-over-v1 (same data, same hyperparameters, same seed)
isolates the path fix as the sole cause.

**Finding: the bypass fix changed the *shape* of the patch response, not just
the magnitude.** v1 produced broad, weak activation across the bbox (good R@1
top-10 coverage, bad argmax). v2 produces a sharp spike on a single best patch
inside the bbox (best-in-class argmax, weaker top-K coverage). This is the
behaviour the FILIP-style max-pool region loss actually asks for:
`max(patch · phrase | patch ∈ bbox) > max(patch · phrase | patch ∉ bbox)`
rewards one strong patch, not spread. v1's path mismatch added enough noise
to the gradient that the model defaulted to broad attention; v2's clean signal
lets the model express the sharp solution the loss prefers.

**VOC stays broken** (0.28% vs 1.43%) for two independent reasons:
(1) prompt distribution mismatch — VOC uses class-name templates
("a photo of a dog") while training used natural Flickr phrases
("the brown dog on the left"); (2) a spike-on-one-patch response destroys
dense per-pixel labelling. VOC remains a guardrail; not a target metric.

**Recipe locked for M_auto.** The training pipeline used here — MHAP pooler
global loss + FILIP region loss + MaskCLIP bypass during training, λ=0.5,
LoRA r=4 on q_proj+v_proj, 1 epoch — is the M_human recipe. M_auto will reuse
it byte-for-byte with the only change being the bbox source (Florence-2
`<DENSE_REGION_CAPTION>` instead of Flickr30k Entities).

> **Caveat (added after Exp-05):** the 22.08% is a single trajectory drawn
> from a high-variance distribution (see Exp-04 / Exp-05). Two later attempts
> of this *exact recipe* with different DataLoader shuffle orders produced
> 18.41% and 16.90% on the same val protocol. Exp-06 will quantify the spread
> with a seeded variance study; until then read v2 as "one sample" not "the
> recipe's true mean".

---

### Exp-04 — M_human v3 (2 epochs, single cosine schedule)

Same hyperparameters as v2 (λ=0.5, lr=2e-4, batch=32, region_batch=64, LoRA
r=4 on q_proj+v_proj, MaskCLIP bypass installed at training time). Single
change: trained for 2 epochs with one warmup→cosine→0 schedule stretched
across the full step budget (~1814 steps total, warmup=100).
Hypothesis: v2's region loss was still falling at epoch end (undertrained);
a second epoch should improve PG.

| Checkpoint                |  v2 (e=1) |  v3 (e=2 single) |
|---------------------------|----------:|-----------------:|
| step 200                  |   16.20%  |          17.94%  |
| step 400                  |   20.20%  |          20.39%  |
| step 600                  |   22.13%  |          19.54%  |
| step 800                  |   21.94%  |          19.02%  |
| end of epoch 1 (~step 907)|   22.08%  |          19.40%  |
| step 1107                 |     —     |          18.97%  |
| step 1307                 |     —     |          18.97% (identical 403/2124 hits — model stalled) |

Train: global=0.323, region=0.099. Region loss kept falling through epoch 2
while PG regressed vs v2.

**Finding:** stretching the cosine over 2 epochs kept LR too high too long.
At step 800, v2's LR multiplier was ≈0.04 (deep in the anneal); v3's was ≈0.64
(still mid-explore). v2's monotonic climb to 22.08% required the late-cycle
settle phase that v3 never had. The model was still exploring when its
training budget ran out and never converged into v2's basin. Run was killed
at step 1307 once PG plateaued (step 1107 and step 1307 PG were identical
both in percentage and absolute hit count, confirming convergence to a worse
attractor than v2's).

**Next:** Exp-05 — per-epoch warm restart so every epoch gets its own
warmup→cosine→0 cycle (SGDR-style). Cycle 1 should reproduce v2; cycle 2
starts from a different initialisation than v2 had access to and could find
a deeper minimum.

---

### Exp-05 — M_human v4 (2 epochs, cosine restart per epoch)

`make_optimizer_scheduler(..., restarts=CFG['epochs'])` splits the step
budget into N equal cycles, each its own warmup→cosine→0. With epochs=2,
cycle 1 is mathematically identical to v2's schedule (~907 steps,
warmup=100). Two attempts of this exact configuration produced very
different trajectories:

| Checkpoint                | v2 (e=1) | v4 attempt 1 | v4 attempt 2 |
|---------------------------|---------:|-------------:|-------------:|
| step 200                  |  16.20%  |       14.92% |       16.24% |
| step 400                  |  20.20%  |       20.10% |       21.05% |
| step 600                  |  22.13%  |       20.48% |   **16.90%** (collapsed) |
| step 800                  |  21.94%  |       18.60% |       16.67% |
| end of epoch 1 (~step 907)|  22.08%  |       18.41% |       16.90% |
| step 1107 (cycle 2)       |     —    |       19.16% |       (killed) |
| step 1307 (cycle 2)       |     —    |       18.97% |       — |

In v4 attempt 2, PG climbed to 21.05% at step 400 and then dropped to 16.90%
at step 600 — a 4.15pp single-checkpoint drop, well outside binomial noise
(~6σ given SE≈0.85pp at p≈0.2). The W&B training curves show **no
corresponding regression in `loss_region`** (kept monotonically falling),
`loss_global` (flat near zero past warmup), `lr` (on schedule), or
`logit_scale` (smooth drift). The optimizer moved into a parameter region
that scores better on the FILIP max-pool training objective but worse on
val-set argmax — a **loss/metric decoupling**, not a training-instability
collapse.

**Finding: the v2 single-point result was inside a high-variance
distribution.** Across v2, v4-att1, and v4-att2 — identical code, identical
hyperparameters, different DataLoader shuffle (no seed control) — final
end-of-epoch-1 PG spanned 16.90% – 22.08%, a **5.18pp spread (~6σ on the
binomial SE)**. Variance source: shuffle order alone, since nothing else
differed. The "locked recipe" claim from Exp-03 had v2 as its only data
point; that point now appears to be on the lucky tail of the distribution.

**Next:** Exp-06 — full seed control on Python `random`, NumPy, torch CPU+CUDA,
and `torch.Generator` passed to the DataLoader. Re-run v2's exact recipe across
multiple seeds. Until we have a measured variance, no single-point change
to the recipe (loss weight, LR, λ, etc.) can be evaluated.

---

### Exp-06 — M_human v5 (variance study, 3 seeds)

Recipe identical to v2 / v4-cycle-1 (1 epoch, single cosine, restarts=1,
λ=0.5, lr=2e-4, batch=32, region_batch=64). Change: full seed control —
`set_seed(seed)` covers `random.seed`, `np.random.seed`, `torch.manual_seed`,
`torch.cuda.manual_seed_all`; a `torch.Generator().manual_seed(seed)` is
passed to the DataLoader so shuffle order is reproducible run-to-run.
Seeds: {0, 1, 2}. Colab L4, ~39 min/seed.
W&B runs: `m_human_v5_seed{0,1,2}_lam0.5` + aggregate `m_human_v5_aggregate_n3`.

**Per-seed results** (200 val images, seed=42, MaskCLIP bypass at eval):

| metric          |     B0 |  v2 (1pt) | seed 0 | seed 1 | seed 2 | v5 mean ± std |
|-----------------|-------:|----------:|-------:|-------:|-------:|--------------:|
| Pointing Game   | 14.74% |   22.08%  | 19.26% | 15.87% | 20.90% | **18.68 ± 2.57%** |
| R@1 top-10      |  6.83% |    5.60%  |  5.51% |  5.79% |  5.98% |   5.76 ± 0.24% |
| R@1 top-25      |  7.91% |    7.34%  |  7.25% |  7.11% |  7.02% |   7.12 ± 0.12% |
| R@1 halfmax     |  7.58% |    7.30%  |  7.30% |  7.30% |  7.30% |   **7.30 ± 0.00%** |
| Train loss (region) | — | 0.102     | 0.103  | 0.102  | 0.102  |  0.102 ± 0.001 |
| Train loss (global) | — | 0.263     | 0.404  | 0.342  | 0.368  |  0.371 ± 0.031 |

**PG trajectory per seed (every 200 steps, end at ~907):**

| step | seed 0 | seed 1 | seed 2 |
|-----:|-------:|-------:|-------:|
|    0 | 14.74  | 14.74  | 14.74  |
|  200 | 16.43  | 16.15  | 16.48  |
|  400 | 16.71  | 19.02  | 19.92  |
|  600 | 19.30  | **14.92** (collapse) | 20.62 |
|  800 | 19.16  | 16.10  | 20.95  |
|  end | 19.26  | 15.87  | 20.90  |

**Findings:**

1. **The variance is concentrated in PG (argmax).** R@1 halfmax is identical
   to 4 decimal places across all 3 seeds (7.29755%); region loss converges to
   0.102 ± 0.001. The recipe is reproducible at the level of the optimization
   objective and at the level of broad patch-similarity coverage; the
   stochasticity flips *which single patch* happens to be argmax for a few
   borderline phrases per seed. PG turns out to be a high-variance lens on a
   low-variance underlying signal.

2. **v2's 22.08% was an outlier.** It sits outside [v5 min, v5 max] = [15.87,
   20.90] — a +1 to +2σ draw against a true mean near 18.68%. The +7.34pp
   "win over B0" claim in Exp-03 should be read as +3.94pp (the v5 mean).

3. **Recipe-level effect on grounding metrics:**
    - PG: +3.94pp over B0 (real, ~3.5σ on the SE of the mean across 3 seeds).
    - R@1 top-10: −1.07pp (consistent regression).
    - R@1 top-25: −0.79pp.
    - R@1 halfmax: −0.28pp.
   The model trades broad patch coverage for a sharper argmax, consistent
   across seeds. This is the "sharp spike" behaviour predicted by the
   FILIP max-pool loss (see Exp-03 finding), now confirmed.

4. **Mid-training collapses persist with seed control.** Seed 1's PG dipped
   to 14.92% at step 600 (matching B0) before partially recovering to 15.87%.
   This is the same loss/metric decoupling seen in v4 attempt 2 — not an
   unseeded-randomness artefact, but a property of the optimisation
   landscape. Some seeds find a basin where the FILIP objective is satisfied
   by attention patterns that hurt PG argmax.

5. **Error bar for future comparisons: σ ≈ 2.6pp on PG.** Any recipe change
   needs ≥ ~5pp improvement to be confidently above the current mean, or
   needs N≥5 seeds to tighten the SE of the mean enough for smaller deltas.

**Implication for what we report.** PG is no longer a single-number metric.
Going forward, M_human and M_auto results should be reported as mean ± std
over ≥3 seeds, with R@1 halfmax flagged as the most-reproducible companion
metric (and the most-honest measure of whether the recipe is learning a
stable grounding signal).
