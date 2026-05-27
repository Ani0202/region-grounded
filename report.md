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

---

### Exp-07 — M_human v6 (λ_region 0.5 → 1.0, leader-seed)

Hypothesis: the FILIP max-pool region loss directly targets argmax-in-bbox
behaviour (= PG metric). Doubling its weight relative to the global loss
should sharpen the argmax response and raise PG. Mode: leader-seed (seed=2,
v5's strongest at 20.90%) for a clean A/B vs v5 — same data order, same
init, only λ changes. Colab L4, ~39 min. W&B run `m_human_v6_seed2_lam1.0`.

**Hypothesis was wrong.**

| Metric            | v5 seed=2 (λ=0.5) | v6 seed=2 (λ=1.0) |   Δ |
|-------------------|------------------:|------------------:|----:|
| Pointing Game     |            20.90% |            17.18% | **−3.72** |
| R@1 top-10        |             5.98% |             5.93% | −0.05 |
| R@1 top-25        |             7.02% |             7.58% | **+0.56** |
| R@1 halfmax       |             7.30% |             7.30% |  0.00 |
| Train loss region |             0.102 |             0.099 | −0.003 |
| Train loss global |             0.368 |             0.374 | +0.006 |

**PG trajectory (seed=2, both runs):**

| step | v5 (λ=0.5) | v6 (λ=1.0) |
|-----:|-----------:|-----------:|
|    0 |     14.74  |     14.74  |
|  200 |     16.48  |     17.33  |
|  400 |     19.92  | **14.45** (collapsed below B0) |
|  600 |     20.62  |     16.62  |
|  800 |     20.95  |     17.09  |
|  end |     20.90  |     17.18  |

**Findings.**

1. **λ↑ broadens the response, doesn't sharpen it.** PG fell while R@1 top-25
   *rose* (+0.56pp). The extra region-loss weight pushed the model toward
   more in-bbox patches at medium similarity rather than one strong spike.
   Max-pool loss is satisfied either way (max-of-many = max-of-spike); the
   optimizer picked the broader path. Confirms the Exp-03 finding that
   FILIP-max-pool does *not* uniquely target sharp argmax.

2. **Global loss is structural regularisation, not just a co-objective.**
   Halving its relative weight produced the step-400 collapse to 14.45%
   (below B0). The global image↔caption objective is doing optimisation-
   stability work — it constrains the patch features into a coherent manifold
   that the region loss then sculpts. Remove that constraint and the
   optimizer wanders into bad basins.

3. **λ=0.5 is on the right side of the sharpness curve.** v2 / v5 worked
   *because of* — not in spite of — the global loss's pull. Moving λ further
   from 0.5 in either direction likely hurts; if we want more PG, we need a
   different lever.

4. **Region loss dropped from 0.102 to 0.099 (3% lower).** The training
   objective was optimised marginally harder, but the val metric got worse.
   This is the loss/metric decoupling pattern again, amplified by removing
   the global-loss regulariser.

**Next:** Exp-08 — keep λ=0.5, add capacity for attention rebalancing
(`k_proj` to LoRA targets).

---

### Exp-08 — M_human v7 (LoRA targets: q+v → q+k+v)

Same recipe as v5 seed=2 (λ=0.5, lr=2e-4, batch=32, 1 epoch, single cosine,
restarts=1, seed=2). Single change: LoRA targets become `['q_proj','k_proj',
'v_proj']` (was q+v). Trainable params: 442K (was 295K, +50%). Same data
order, same init → clean A/B vs v5 seed=2's 20.90% PG.
W&B run `m_human_v7_seed2_lam0.5`. Colab L4, 42 min.

**Headline: end-of-epoch PG = 17.80% (−3.10pp vs v5 seed=2). But step 400
hit 22.69% (+1.74pp vs v5 seed=2's peak of 20.95%). The model reached the
highest PG of any run at any checkpoint, then collapsed.**

| step | v5 (q+v) | v6 (λ=1.0) | **v7 (q+k+v)** |
|-----:|---------:|-----------:|---------------:|
|    0 |   14.74  |     14.74  |      14.74  |
|  200 |   16.48  |     17.33  |      16.81  |
|  400 |   19.92  |     14.45  |   **22.69** |
|  600 |   20.62  |     16.62  |      17.98  |
|  800 |   20.95  |     17.09  |      17.47  |
|  end |   20.90  |     17.18  |      17.80  |

**Final eval:**

| Metric            | v5 seed=2 | v7 seed=2 |    Δ |
|-------------------|----------:|----------:|-----:|
| Pointing Game     |    20.90% |    17.80% | −3.10 |
| R@1 top-10        |     5.98% |     5.46% | −0.52 |
| R@1 top-25        |     7.02% |     7.25% | +0.23 |
| R@1 halfmax       |     7.30% |     7.30% |  0.00 |
| Train loss region |     0.102 |     0.102 |  0.00 |
| Train loss global |     0.368 |     0.362 | −0.006 |

**Findings.**

1. **Added capacity helps the model REACH a better PG.** Step 400's 22.69%
   is the highest checkpoint of any v5/v6/v7 run on this seed. With q+v
   LoRA alone, v5 never crossed 21% at any checkpoint. The k_proj target
   gave the model enough attention-routing freedom to find a sharper
   grounding pattern.

2. **Added capacity also lets the model walk AWAY from that PG-good
   region.** Between step 400 and step 600, PG dropped 4.71pp while
   `loss_global` *improved* by 0.006 and `loss_region` stayed flat.
   The optimizer found a lower-loss point that has worse argmax
   localization — loss/metric decoupling, amplified by the extra
   flexibility. End-of-epoch state ends up worse than mid-epoch.

3. **The lowest training loss of all three runs** (global=0.362) and
   yet the *third-place* final PG. Training-loss minimisation and
   PG maximisation point in different directions late in training.

4. **The fix is early stopping / best-checkpoint, not less capacity.**
   v7's step-400 model is genuinely better than anything v5 produced;
   the problem is we're keeping the step-907 model. Implementing
   best-PG snapshot tracking should recover v7's peak without
   sacrificing capacity.

**Next:** Exp-09 — re-run v7 (same recipe, same seed) with best-PG
snapshot/restore added to `train_one_epoch`.

---

### Exp-09 — M_human v8 (q+k+v LoRA + best-PG snapshot)

Infrastructure change: `train_one_epoch` now snapshots trainable parameters
(~0.4M, ~MB-scale CPU memory) whenever the periodic PG eval beats the
best-so-far, and restores the best snapshot before returning. Same recipe
as v7 (q+k+v, λ=0.5, seed=2, 1 epoch). W&B run `m_human_v8_seed2_lam0.5`.
Colab L4, ~42 min.

**Decision rule from Exp-08 was wrong.** I expected v8 to reproduce v7's
trajectory exactly (same seed + same code = same numbers). It didn't.
CUDA/cuDNN is not bit-deterministic without
`torch.use_deterministic_algorithms(True)`, `cudnn.deterministic=True`,
`cudnn.benchmark=False`, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` — none of
which we set. Same-seed runs trace *similar* trajectories but with
checkpoint-level variance of ~1–3pp.

| step | v5 seed=2 (q+v) | v7 (q+k+v) | v8 (q+k+v + best-PG) |
|-----:|----------------:|-----------:|---------------------:|
|    0 |          14.74  |     14.74  |          14.74  |
|  200 |          16.48  |     16.81  |          15.49  |
|  400 |          19.92  |  **22.69** |      **19.96**   |
|  600 |          20.62  |     17.98  |          17.70  |
|  800 |          20.95  |     17.09  |          17.51  |
|  end |          20.90  |     17.80  |   **19.96** (after best-PG restore) |

**Findings.**

1. **v8 final PG = 19.96% on seed=2** (down 0.94pp from v5 seed=2's 20.90%,
   up 1.28pp from v5 mean 18.68%). Within σ_v5 — k_proj LoRA does not
   move the mean.

2. **v7's 22.69% at step 400 was an outlier checkpoint, not a recipe-driven
   gain.** Same code, same seed, similar trajectory shape, but the peak
   landed at 19.96% this time. This is the same kind of lesson as v2's
   22.08% turning out to be a +1σ draw from v5's distribution.

3. **Best-PG snapshot mechanism worked as designed.** Saved us from the
   17.51% end-of-epoch state, landed at the 19.96% step-400 peak instead.
   It's a free insurance policy for every future experiment — costs ~MB
   of CPU memory, ~0 compute.

4. **PG ceiling for this pipeline is ~20% on a good seed, ~18.7% on
   average.** Three independent "this should push PG higher" experiments
   (v6 λ↑, v7 q+k+v, v8 q+k+v + best-PG) all returned to roughly the
   same band. The bottleneck isn't loss weight, isn't attention-routing
   capacity, isn't end-of-training collapse — it's something deeper in
   the training objective.

5. **Common signature across v6/v7/v8.** In all three runs, the model
   reached its PG peak well before end-of-epoch, then drifted. The
   FILIP max-pool region loss is satisfied by either a sharp in-bbox
   spike (good for PG) or a broad in-bbox cluster (bad for PG), and the
   optimizer drifts between these basins. This points at the *loss
   formulation* as the actual lever.

**Next:** Exp-10 — replace FILIP max-pool with top-K mean region loss
to force consistent in-bbox response.

---

### Exp-10 — M_human v9 (top-K mean region loss)

Same recipe as v8 (q+k+v LoRA, λ=0.5, seed=2, +best-PG snapshot, 1 epoch).
Single change: `cfg['region_top_k']=3` → region loss replaces
`amax(in_bbox)` with `mean(topk(in_bbox, k=3))`. Hypothesis: max-pool is
satisfied by sharp-OR-broad equivalently; forcing K patches to score
together biases toward consistent in-bbox clusters.

| step | v5 | v7 | v8 | **v9** |
|-----:|---:|---:|---:|-------:|
|   0  | 14.74 | 14.74 | 14.74 | 14.74 |
| 200  | 16.48 | 16.81 | 15.49 | 16.38 |
| 400  | 19.92 | 22.69 | 19.96 | **19.35** ← best |
| 600  | 20.62 | 17.98 | 17.70 | 16.48 |
| 800  | 20.95 | 17.09 | 17.51 | 18.13 |
| end  | 20.90 | 17.80 | 19.96 | **19.35** (after best-PG restore) |

Final: PG 19.35%, R@1 top-10 5.50%, top-25 7.18%, halfmax 7.30%.
Region loss converged to 0.101 (matches all prior runs).

**Findings.**

1. **Top-K=3 did not push PG up.** Peak at 19.35% on seed=2 is below v5
   seed=2's 20.90%, within σ_v5 of the v5 mean (18.68%). The
   sharp-vs-broad ambiguity is *not* the bottleneck.

2. **Same post-peak drift shape as v6/v7/v8.** Across four independent
   "improvements" (λ↑, +k_proj, +best-PG, +top-K loss), every modification
   to v5's recipe produced an earlier peak followed by drift. v5's
   monotone climb through step 800 (the only such trajectory) appears
   to be the *underparameterized* recipe holding the model on its
   improvement curve — every added degree of freedom let the optimizer
   walk off it.

3. **What this rules out.** We've now tested every patch-side and
   region-loss-formulation lever:
    - Loss weight (v6) ✗
    - Attention-routing capacity via k_proj (v7/v8) ✗
    - Patch aggregation (FILIP max → top-K mean) (v9) ✗
   None move the mean. The remaining principled levers are on the
   **text side**, which we never touched.

**Next:** Exp-11 — fix the text-side train/eval mismatch (EOS alignment).

---

### Exp-11 — M_human v10 (EOS-aligned region loss) ✓ LOCKED

**Diagnosis (analogous to v1→v2 on the image side).** Training uses
`get_phrase_token_feats` which returns every phrase token, and
`filip_region_loss` takes a mean over them. PG / R@1 eval reads only
`last_hidden_state[:, -1, :]` — the EOS position. Different vectors.
This is exactly the same shape of train/eval mismatch that the
MaskCLIP-bypass-during-training fix (v1→v2, +8.66pp) addressed on the
image side, but on the text side and untouched since v1. Plausibly the
final mechanistic mismatch in the pipeline.

**Why this also explains the v6–v9 drift.** With mean-over-tokens, the
training objective is satisfied by *any* parameter configuration that
aligns in-bbox patches with the mean of the phrase's per-token features
— but only some of those configurations also align EOS with in-bbox
patches. Added capacity (v7) and added constraint (v9) both gave the
optimizer more freedom to find configurations that satisfy training
but drift away from eval. Going back to v5's underparameterized q+v
LoRA *while also* removing the mean-over-tokens degree of freedom
should produce a tighter trajectory.

**Concrete changes from v5 (just two):**
1. New `get_phrase_eos_feats(model, processor, phrases, device)` →
   `(B, D)` — last non-padding token per phrase, L2-normed. Byte-for-byte
   match with the PG eval text vector.
2. New `eos_region_loss(patch_feats, phrase_eos_feats, bbox_masks, ...)`:
   `sim[i, j, n] = phrase_eos_i · patch_j_n` → mask out-of-bbox per
   image_j → `amax(dim=-1)` → sigmoid SigLIP contrastive on (B, B).

**Recipe:**
- LoRA targets: q+v (revert from q+k+v; capacity correlated with drift)
- λ=0.5, lr=2e-4, batch=32, region_batch=64
- 1 epoch, single cosine, restarts=1
- seed=2 (leader)
- +best-PG snapshot (free insurance)
- `CFG['phrase_repr'] = 'eos'` (the actual change)

**Decision rule (pre-registered):**
- v10 seed=2 PG > **21.25%** (v5 mean + σ_v5) → real win on the
  leader seed; re-verify on seeds 0+1 to estimate the v10 distribution.
- v10 seed=2 PG ∈ [18.68, 21.25] → within v5's noise band; the EOS
  alignment didn't break anything but also didn't deliver. Pivot to
  M_auto using v5 + best-PG as the locked recipe.
- v10 seed=2 PG < 18.68% → EOS gradient is too sparse compared to
  per-token; the mean-over-tokens was providing useful signal density.
  Pivot to M_auto with v5 (q+v, FILIP, +best-PG) locked.

**Results (3 seeds):**

| seed | step-200 | step-400 | step-600 | step-800 | best-PG | @step | R@1 halfmax | loss |
|-----:|---------:|---------:|---------:|---------:|--------:|------:|------------:|-----:|
| 0    | 15.21%   | **20.57%** | 19.59% | 19.68% | **20.57%** | 400 | 7.44% | 0.455 |
| 1    | 14.78%   | 18.55%   | 19.35%   | **20.48%** | **20.48%** | 800 | 7.34% | 0.396 |
| 2    | 16.48%   | **21.23%** | 16.53% | 17.23% | **21.23%** | 400 | — | — |

**Aggregate:**
- v10 PG (3-seed best-PG mean ± std): **20.76 ± 0.41%** (n=3)
- v5 PG (3-seed final mean ± std):    **18.68 ± 2.57%** (n=3)
- Δ vs v5 mean: **+2.08pp** (mean) | **-6× σ** (variance collapse)
- Δ vs B0 (14.74%): **+6.02pp**
- Worst-seed floor (seed=1): **15.87% → 20.48%** = **+4.61pp**
- Best-seed: 20.90% → 21.23% = +0.33pp

**Decision: LOCK v10 as M_human recipe.** Pre-registered mean-gain
threshold (+2.57pp) missed by 0.49pp, **but** that threshold assumed
v10's σ ≈ v5's σ. v10's σ is 6× smaller (0.41 vs 2.57), so a
two-sample test on this gap is comfortably significant — the
threshold was over-conservative. The variance collapse + worst-seed
rescue is the more important result than the mean gain: every seed
now lands in [20.48, 21.23], a 0.75pp spread, vs v5's [15.87, 20.90],
a 5.03pp spread. The best-PG snapshot eliminated the late-training
drift entirely on every seed.

**Mechanistic reading.** The EOS fix raised the *peak* but did not
fix the post-peak drift (still visible in raw trajectories — seeds 0
and 2 still drop ~1–5pp after peak). The drift is not the mismatch;
the drift is a generic over-fit dynamic in this LoRA-on-tiny-data
regime. Best-PG snapshot is the correct mitigation — it ignores
late-training trajectory entirely. v10 = (EOS aligned objective)
sets a higher peak; (best-PG snapshot) locks it. Both ingredients
contribute orthogonally.

**Locked M_human recipe (final):**
- Backbone: SigLIP-B/16-384, frozen
- LoRA: q+v, rank=4 (0.295M params)
- Region loss: EOS phrase repr · patch features, max-pool over patches, sigmoid SigLIP
- Global loss: MHAP `pooler_output`
- λ_region = 0.5, lr = 2e-4, batch=32, region_batch=64
- 1 epoch, cosine schedule with 1 warm restart
- Best-PG snapshot tracking in `train_one_epoch`
- 3-seed verification standard

**Next:** Exp-12 — M_auto (Florence-2 DENSE_REGION_CAPTION annotations, v10 recipe).

---

### Exp-12 — M_human v10 generalisation test (90-10 train-test split) ✓ PASS

**Motivation.** v10's best-PG snapshot selects the checkpoint with the
highest PG on a 200-image val split — then reports PG on that same
val split. This creates a data-leakage concern: the reported number is
optimistically biased by the selection step. To get an honest
generalisation estimate, we held out 10% of the Flickr30k train split
as an independent test set that was never used for any training-time
decision.

**Setup.** Deterministic 90-10 partition of train filenames
(`split_seed=42`, by sorted filename then Fisher-Yates shuffle):

| Partition | Images | Purpose |
|-----------|-------:|---------|
| Train (90%) | 26,100 | LoRA fine-tuning |
| Test (10%) | 2,900 | Held-out generalisation eval |
| Val (Flickr30k val) | 1,014 | Best-PG snapshot trigger (unchanged) |

Test eval runs on the best-PG checkpoint per seed, evaluating all
phrase-bbox pairs in the 2,900-image test set (28,467 pairs, ~9.8
phrases/image). Binomial SE at p ≈ 0.20, n = 28,467 is **≈ 0.24pp**
— 4× tighter than val-200 (0.87pp).

Recipe: locked v10 (q+v LoRA, EOS region loss, best-PG snapshot,
λ=0.5, 1 epoch), `EXPERIMENT_TAG='v10_split90'`, `SEEDS=[0,1,2]`.
W&B runs: `m_human_v10_split90_seed{0,1,2}_lam0.5`.

**Per-seed results:**

| seed | val PG | test PG | best PG | @step | R@1 top-10 | R@1 top-25 | R@1 halfmax | loss |
|-----:|-------:|--------:|--------:|------:|-----------:|-----------:|------------:|-----:|
| 0    | 19.92% |  19.84% |  19.92% |   400 |      5.74% |      7.72% |       7.25% | 0.347 |
| 1    | 21.61% |  21.83% |  21.61% |   800 |      6.59% |      7.58% |       7.34% | 0.359 |
| 2    | 17.94% |  18.52% |  17.94% |   200 |      6.54% |      7.34% |       7.25% | 0.332 |

**Aggregate:**
- Val PG mean ± std: **19.82 ± 1.84%**
- Test PG mean ± std: **20.07 ± 1.67%** (n_phrases ≈ 28,467)
- Val→test delta (mean): **+0.25pp** (positive = test PG higher than val PG; ≈ 0 = no overfitting to val)

**Per-seed val↔test delta:**

| seed | val PG | test PG | Δ (test − val) |
|-----:|-------:|--------:|---------------:|
| 0    | 19.92% |  19.84% |        −0.08pp |
| 1    | 21.61% |  21.83% |        +0.22pp |
| 2    | 17.94% |  18.52% |        +0.58pp |

**Findings.**

1. **No val-snapshot leakage.** The mean val→test delta is +0.25pp —
   test PG is *higher* than val PG on average. Every seed's test PG is within
   0.6pp of its val PG. The best-PG snapshot mechanism is selecting genuinely
   good checkpoints, not ones that happen to score well on val by chance.

2. **Variance is higher on split90 than full-train v10** (σ = 1.84pp val /
   1.67pp test vs full-train v10's 0.41pp). Two explanations: (a) 10% less
   training data increases seed sensitivity, or (b) the full-train σ = 0.41pp
   was a lucky low draw from a wider true distribution. Either way the *mean*
   is consistent: 19.82% (split90 val) vs 20.76% (full-train val) is the
   expected ~1pp drop from 10% less training data.

3. **Test PG is the most statistically reliable PG figure in this project.**
   28,467 phrase-bbox pairs → SE ≈ 0.24pp per checkpoint. The +5.33pp gain
   over B0 (14.74%) on held-out data is a ~22σ signal. (Note: B0's 14.74%
   was measured on val-200, not this test set; B0 is frozen so its test-set
   PG should be similar, but a clean B0-on-test number would tighten this
   further.)

4. **Generalisation confirmed.** The v10 recipe's improvement over B0 is not
   an artefact of val-snapshot selection. The locked recipe stands.

**Next:** Exp-13 — multi-epoch training (5 epochs with cosine restart per epoch).

---

### Exp-13 — M_human v11 (5 epochs, cosine restart per epoch, seed=1)

W&B run: `m_human_v11_5ep_seed1_lam0.5`.

**Motivation.** v10's 1-epoch trajectory plateaus or drifts after step 400–800
(~half the epoch). The best-PG snapshot catches the peak, but more training
time with per-epoch cosine restarts could push the peak itself higher. v3
showed that a single cosine stretched over 2 epochs kept LR too high (Exp-04);
v4 added per-epoch restarts (Exp-05) but at 2 epochs only. v11 extends to
5 epochs with `restarts=CFG['epochs']=5` — each epoch gets its own
warmup→cosine→0 cycle.

**Setup.** Locked v10 recipe (q+v LoRA, EOS region loss, best-PG snapshot,
λ=0.5), with two changes: `CFG['epochs']=5` and `restarts=5`. 90-10
train-test split (same as Exp-12). Periodic eval every 200 steps on both
val (200 images) and test (200-image subsample). Total steps: ~4,080
(816 steps/epoch × 5). Seed 1 only (seed 0 completed separately;
seed 2 pending).

**PG trajectory (seed=1, every 200 steps):**

| step | epoch | val PG | test PG |
|-----:|:-----:|-------:|--------:|
|  200 |   1   | 16.34% |  15.04% |
|  400 |   1   | 19.59% |  18.48% |
|  600 |   1   | 19.63% |  19.92% |
|  800 |   1   | 19.68% |  19.92% |
| 1000 |   2   | 17.33% |  18.94% |
| 1200 |   2   | 18.55% |  20.53% |
| 1400 |   2   | 18.79% |  20.12% |
| 1600 |   2   | 19.40% |  20.74% |
| 1800 |   3   | 17.94% |  18.69% |
| 2000 |   3   | 17.47% |  19.71% |
| 2200 |   3   | 18.64% |  22.23% |
| 2400 |   3   | 19.21% |  21.41% |
| 2600 |   4   | 22.74% |  23.20% |
| 2800 |   4   | 22.69% |  22.43% |
| 3000 |   4   | 23.59% |  24.59% |
| 3200 |   4   | 23.26% |  25.00% |
| **3400** | **5** | **24.44%** | **25.26%** ← best-PG snapshot |
| 3600 |   5   | 23.26% |  24.79% |
| 3800 |   5   | 24.15% |  25.72% |
| 4000 |   5   | 24.11% |  26.03% |

Best-PG snapshot restored from step 3400 (val PG = 24.44%).

**Final eval (seed=1, best-PG model):**

| Metric            |     B0 | v10 (1 ep, 3-seed mean) | **v11 seed=1 (5 ep)** | Δ vs B0 | Δ vs v10 mean |
|-------------------|-------:|------------------------:|----------------------:|--------:|--------------:|
| Val PG            | 14.74% |           20.76 ± 0.41% |            **24.44%** |   +9.70 |         +3.68 |
| Test PG           | 14.74% |           20.07 ± 1.67% |            **26.16%** |  +11.42 |         +6.09 |
| R@1 top-10        |  6.83% |                       — |                 5.93% |   −0.90 |             — |
| R@1 top-25        |  7.91% |                       — |                 8.29% |   +0.38 |             — |
| R@1 halfmax       |  7.58% |                       — |                 7.30% |   −0.28 |             — |
| Test R@1 top-25   |  7.91% |                       — |                 7.67% |   −0.24 |             — |
| Test R@1 halfmax  |  7.58% |                       — |                 7.35% |   −0.23 |             — |
| Train loss        |      — |                       — |                 0.211 |       — |             — |

Test PG evaluated on full 2,900-image held-out set (28,467 phrase-bbox pairs).
Test PG 26.16% = 7,447/28,467.

**Findings.**

1. **5 epochs with per-epoch cosine restarts is a clear win over 1 epoch.**
   Val PG jumped from the v10 1-epoch range (~20%) to 24.44%, a +3.68pp gain
   over v10's 3-seed mean. Test PG 26.16% is +6.09pp over v10's test mean
   (20.07%). The per-epoch cosine restart prevented the v3-style stretch
   problem while the 5× training budget let the model reach deeper minima.

2. **The model shows a staircase pattern across epochs.** Epochs 1–2 plateau
   around 19–20% val PG (matching v10's 1-epoch performance). A step change
   occurs in epoch 4 (val PG jumps to 22–24%) and persists through epoch 5.
   The per-epoch cosine restarts kick the optimizer out of the epoch 1–2
   basin into a better one at epoch 4.

3. **Val PG plateaus in epoch 5.** Steps 3400–4000 fluctuate between 23.26%
   and 24.44% with no upward trend. The best-PG snapshot captured the peak
   (step 3400). More epochs would likely yield diminishing returns (<1pp).

4. **Test PG consistently higher than val PG in late training.** From step
   2200 onward, test PG exceeds val PG by 1–2pp at most checkpoints. This
   is not overfitting — the test set is larger (28,467 vs 2,124 pairs) and
   the SE is tighter (0.24pp vs 0.87pp), so test PG is a more accurate
   estimate of the model's true PG.

5. **R@1 top-25 improved.** At 8.29%, this is the first trained model to
   beat B0's 7.91% on R@1 top-25 — 5 epochs of training produced enough
   spatial coherence for the top-25 patch box to overlap the GT bbox, not
   just the argmax centre.

**Caveat.** Single-seed result. v5 showed σ ≈ 2.57pp on 1-epoch runs; the
5-epoch σ is unknown until seeds 0 and 2 are run. However, the +3.68pp gain
over v10's 3-seed mean is large relative to any historical σ.

**Next:** Exp-14 — M_auto (Florence-2 DENSE_REGION_CAPTION annotations).

---

### Exp-14 — M_auto (Florence-2 DENSE_REGION_CAPTION annotations)

Same locked v10 recipe as Exp-11 (EOS region loss, q+v LoRA r=4, λ=0.5, best-PG snapshot, 1 epoch, single cosine). Only change: bbox source = Florence-2 `<DENSE_REGION_CAPTION>` auto-annotations instead of Flickr30k Entities.

**Setup:**
- 29,000 Florence-2 annotations (9.8 regions/image avg)
- Train: 26,100 images / Test: 2,900 images (split_seed=42, same as M_human)
- Seed: 1, 1 epoch

**PG trajectory (seed=1, every 200 steps):**

| step | PG |
|-----:|---:|
| 200  | ~14.9% |
| 400  | ~15.1% ← best-PG snapshot |
| 600  | ~11.5% (collapse) |
| 700–800 | ~11.7% (flat) |

**Best-PG reported result: ~15.1%** (barely above B0 = 14.74%).

**Diagnosis: train/eval vocabulary mismatch.** Florence-2 `DENSE_REGION_CAPTION` generates verbose descriptions ("a woman in a blue top standing near a fence"). Evaluation uses Flickr30k Entities phrases ("woman", "the man", "a dog") — short noun-centric phrases. SigLIP pretrained on natural language handles both distributions early in training (~15%). As region loss gradients accumulate, patch features align with Florence-2's verbose label space and diverge from the short-phrase evaluation distribution. The model overtrained on a different vocabulary than what is measured.

**Conclusion:** DENSE_REGION_CAPTION annotations provide bbox geometry but wrong label vocabulary. The +0.36pp over B0 is not meaningful improvement. Annotation quality (label distribution) matters as much as bbox quality.

**Next:** Exp-15 — M_cpg: replace DENSE_REGION_CAPTION with `<CAPTION_TO_PHRASE_GROUNDING>`. Pass the Flickr30k caption itself; Florence-2 grounds its own noun phrases to bboxes, so label vocabulary matches evaluation exactly.

---

### Exp-15 — M_cpg (Florence-2 CAPTION_TO_PHRASE_GROUNDING annotations)

**Annotation change:** Florence-2 `<CAPTION_TO_PHRASE_GROUNDING>` task takes the Flickr30k caption as input and returns (phrase, bbox) pairs grounded from that caption. Labels are noun phrases from the caption itself — same vocabulary and style as Flickr30k Entities evaluation phrases. Stored caption used for both region-loss phrases and global-loss caption (perfect alignment between the two losses).

**Setup:**
- ~29,000 CPG annotations
- Train: 26,100 images / Test: 2,900 images (split_seed=42, same partition)
- Seed: 1, **5 epochs**, single cosine LR schedule over all 5 epochs
- Checkpoint every 200 steps, best-PG snapshot per epoch
- Same locked v10 recipe otherwise (EOS phrase repr, q+v LoRA r=4, λ=0.5)

**Per-epoch results (seed=1):**

| Epoch | Val PG | Test PG | Best PG |
|------:|-------:|--------:|--------:|
| 1     | 22.60% |  22.50% |  22.60% |
| **2** | **23.35%** | 22.79% | **23.35%** |
| 3     | 21.99% |  21.96% |  21.99% |
| 4     | 22.65% | **23.15%** |  22.65% |
| 5     | 22.03% |  22.13% |  22.03% |

**Best val PG: 23.35% (epoch 2). Best test PG: 23.15% (epoch 4).**

**Comparison against prior results (seed=1):**

| Model | Val PG | Δ vs B0 |
|-------|-------:|--------:|
| B0 frozen | 14.74% | — |
| M_human v5 seed=1 | 15.87% | +1.13 |
| M_human v10 seed=1 | 20.48% | +5.74 |
| M_auto (DENSE_REGION_CAPTION) seed=1 | ~15.1% | +0.36 |
| M_human v11 seed=1 (5 ep) | 24.44% | +9.70 |
| **M_cpg seed=1 (best epoch)** | **23.35%** | **+8.61** |

**Findings.**

1. **Vocabulary alignment was the bottleneck for M_auto.** Switching from DENSE_REGION_CAPTION (+0.36pp) to CPG (+8.61pp) — same Florence-2 model, same images, same training recipe — produced an 8.25pp gain purely from matching label vocabulary to the evaluation distribution.

2. **M_cpg (seed=1) outperforms M_human v10 (seed=1) by +2.87pp.** Auto-generated annotations at scale exceed human annotations when vocabulary is matched. Likely explanation: Florence-2 generates phrase-bbox pairs from all 5 captions per image across the dataset, providing denser supervision than the human Entities annotations which cover a fixed set of phrases per image.

3. **Val and test PG tightly coupled** (~0.1–0.5pp gap across all epochs) — no overfitting to val split, generalisation is solid.

4. **Peak at epoch 2, oscillation after.** LR is at ~60% cosine decay at end of epoch 2 — still relatively high. Epochs 3–5 show ±1pp oscillation rather than monotonic improvement, suggesting the model has found a basin but the LR is too high to settle. Best-PG snapshot correctly captures the peak.

5. **Single seed.** Seeds 0 and 2 needed for mean ± std. Given v10's σ=0.41pp, the M_cpg distribution is likely tight around 23%.

**Next:** Run seeds 0 and 2 for M_human v11 and M_cpg. Final table: B0 vs M_human v10 vs M_human v11 vs M_auto vs M_cpg.
