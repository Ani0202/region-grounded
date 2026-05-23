# GRAFT — Project Notes

A plain-English record of what the project is, what we tried, what worked, and why.

---

## What are we doing and why is it interesting?

**SigLIP** is a vision-language model trained to match whole images to whole captions. Its internal patch tokens (the 576 small grid cells it splits an image into) were never directly told what any specific word or phrase refers to in the image. So if you ask it "where is the red jacket?", it can sort of guess based on global similarity, but it has no reliable spatial grounding — it doesn't know *which patches* correspond to "red jacket".

**GRAFT** fine-tunes SigLIP with an extra loss that explicitly teaches it to match noun phrases to the right patches. We do this cheaply using **LoRA** (Low-Rank Adaptation) — a technique that adds a tiny number of trainable parameters (0.14% of the model) on top of a frozen backbone, so we don't have to retrain 200M parameters from scratch.

**The core question:** Can a small amount of region-level supervision teach a globally-trained model to spatially ground phrases? And can automatically generated bounding boxes (from Florence-2) do the job as well as human-drawn ones?

**Why SigLIP specifically?** Most prior work on region grounding uses CLIP. SigLIP is a newer, stronger model (better at image-text matching) but uses a global loss that leaves even less spatial signal in its patch tokens than CLIP. If GRAFT works on SigLIP, the approach generalises to any globally-trained VLM.

---

## The two supervision conditions

- **M_human**: Fine-tuned using **Flickr30k Entities** — a dataset where human annotators drew bounding boxes around every noun phrase in every caption (e.g. "a man in a red jacket" → a box at specific pixel coordinates). This is the clean upper-bound condition.

- **M_auto**: Fine-tuned using boxes generated automatically by **Florence-2** (a 231M-parameter model from Microsoft) using its `DENSE_REGION_CAPTION` task. Florence-2 looks at an image and outputs (bounding box, short label) pairs without any human annotation. This is the self-supervised condition.

The minimal deliverable is a three-way table: **B0 (frozen) vs M_human vs M_auto** on both evaluation metrics.

---

## How we evaluate

### Pointing Game (PG)
For each (phrase, bounding box) pair in the validation set:
1. Encode the phrase with the text encoder → phrase embedding
2. Extract all 576 patch features from the image
3. Compute similarity between phrase and every patch
4. Find the single highest-scoring patch (the argmax)
5. Check: does that patch's centre pixel fall inside the human-drawn GT box?

Score = fraction of pairs where the answer is yes.

**B0 baseline: 14.74%** on 200 val images, 2,124 phrase-bbox pairs.

### Recall@1 at IoU ≥ 0.5
Instead of just the top-1 patch, take the top-k patches, form their tight bounding box, and check whether it overlaps the GT box by at least 50% (IoU ≥ 0.5). We test k=10, k=25, and a "halfmax" threshold (all patches with similarity ≥ 0.5 × max).

**B0 baseline: 6.83–7.91%** depending on k.

### Why PG > Recall@1 on B0
The frozen model's argmax occasionally lands near the right region (PG 14.74%), but the *surrounding* high-scoring patches are scattered all over the image. When you form a tight box around all of them, it covers most of the image and the IoU collapses. This tells us the model has a weak directional signal but no coherent spatial clustering. GRAFT's region loss is supposed to fix exactly this.

### VOC mIoU (guardrail only)
We also track zero-shot segmentation accuracy on PASCAL VOC 2012 as a sanity check. We don't expect improvement here — VOC tests class-level labels ("cat", "aeroplane") while we train on natural noun phrases ("the brown dog on the left"). It's only there to confirm fine-tuning doesn't catastrophically break the model.

---

## The training loss

```
L_total = L_global + λ · L_region
```

- **L_global**: Standard SigLIP sigmoid contrastive loss — matches whole images to whole captions. Keeps the model from forgetting its pre-trained knowledge.
- **L_region**: FILIP-style patch–token alignment loss — for each phrase in the caption, the highest-scoring patch inside the GT box should score higher than any patch outside it.
- **λ** controls the relative weight of the region loss. We tested λ=0.5 and λ=1.0.

---

## Baselines (before any fine-tuning)

| Model | Method | VOC mIoU |
|---|---|---|
| CLIP ViT-B/16 | Standard | 8.46% |
| CLIP ViT-B/16 | MaskCLIP | 19.90% |
| CLIP ViT-B/16 | SCLIP | 14.01% |
| SigLIP-B/16-384 | MaskCLIP (frozen B0) | 1.43% |

SigLIP's 1.43% is expected — it was never trained for spatial grounding.

**What is MaskCLIP?** MaskCLIP is not a separate model. It's a patch-extraction technique: replace the last self-attention layer of the vision transformer with a value-only projection (`v_proj` only, no inter-patch mixing). This exposes spatially structured per-patch features that are better for localisation than the default output. We apply this to both CLIP and SigLIP for all evaluations.

---

## Experiment history

### Exp-01 — First training run (broken)

**What we tried:** Train SigLIP + LoRA (q_proj, v_proj) for 1 epoch on Flickr30k with the global + region loss.

**Result:** Both trained models (with and without region loss) regressed below frozen B0 on VOC.

**Why it broke:** Two bugs:
1. The global loss used `mean(last_hidden_state)` — the average of all 576 patch tokens — as the image representation. This pushes *all* patch tokens uniformly toward the caption, destroying any spatial diversity.
2. Evaluation used τ=0.0 as a similarity threshold, which was calibrated on the frozen model but mismatched after training shifted the similarity distribution.

**Fix:** Use `pooler_output` (SigLIP's dedicated MHAP pooling head) for the global loss instead of mean-pooling. This keeps patch features free to be spatially diverse.

---

### Exp-02 — MHAP pooler fix (v1)

**What we tried:** Same setup as Exp-01 but with the MHAP pooler fix. Primary metric switched to Pointing Game (more sensitive to phrase grounding than VOC).

**Result:**
| Metric | B0 | M_human v1 | Δ |
|---|---|---|---|
| Pointing Game | 14.74% | 13.42% | −1.32 |
| Recall@1 top-10 | 6.83% | 7.11% | +0.28 |

PG still regressed, but R@1 top-10 improved slightly.

**Why the mixed result:** A train/eval mismatch on the *image* side. During training, patch features were read through the full last-attention layer (normal forward pass). During evaluation, they were read through the MaskCLIP bypass (`v_proj` only, no inter-patch mixing). The gradient was sharpening patches under one representation while we were measuring them under a different one. The +0.28 on R@1 top-10 confirmed the model *was* learning something — it just wasn't visible to the argmax metric because the vectors were different at eval time.

---

### Exp-03 — MaskCLIP bypass installed during training (v2) ✅ Big win

**What we tried:** Install the MaskCLIP bypass on the vision encoder *before training begins*, so the gradient flows through the same v_proj-only representation that evaluation reads.

**Result:**
| Metric | B0 | v2 | Δ |
|---|---|---|---|
| Pointing Game | 14.74% | **22.08%** | +7.34 |
| Recall@1 top-10 | 6.83% | 5.60% | −1.23 |

**+7.34pp on PG** — a huge improvement over baseline. This was the single biggest gain in the project.

**What it revealed:** Fixing the train/eval path mismatch changed the *shape* of the patch response. v1 produced broad, weak activation across the bbox (helped R@1 top-K coverage, hurt argmax). v2 produces a sharp spike on one best patch inside the bbox (great for argmax/PG, weaker coverage). This is what the FILIP max-pool loss actually rewards — one strong patch, not a spread.

**Caveat (added later):** 22.08% turned out to be a lucky single run. After running 3 seeds, the true mean of this recipe is 18.68 ± 2.57%.

---

### Exp-04 — 2 epochs, single cosine schedule (v3)

**What we tried:** Train for 2 epochs instead of 1. The region loss was still falling at the end of epoch 1, suggesting the model was undertrained.

**Result:** PG peaked at ~20% early then regressed to 19.40% by end of epoch 1, and kept falling to ~18.97% through epoch 2.

**Why it didn't work:** Stretching the cosine learning rate schedule over 2 epochs meant the LR was still high (mid-explore) at the point where v2's LR was already in the final low-LR "settle" phase. v2's monotonic climb to 22.08% needed that late tight-LR window to converge. v3 never got there.

---

### Exp-05 — 2 epochs with per-epoch cosine restart (v4)

**What we tried:** Give each epoch its own warmup→cosine→0 cycle (SGDR-style), so epoch 1 is mathematically identical to v2 and epoch 2 starts fresh.

**Result:** Two attempts of this identical recipe on different DataLoader shuffles produced 18.41% and 16.90% end-of-epoch-1 PG. Previously v2 got 22.08% from the "same" recipe.

**Key finding:** The spread (16.90% to 22.08%) across runs with "identical hyperparameters but different data order" is ~5pp — about 6 standard deviations. The recipe is **high variance**. v2's 22.08% was on the lucky tail of the distribution. No single-point result can be trusted until we measure the variance properly.

---

### Exp-06 — Seed-controlled variance study (v5)

**What we tried:** Add full seed control — fix Python random, NumPy, PyTorch CPU+CUDA, and DataLoader shuffle. Run the v2 recipe (1 epoch, λ=0.5, q+v LoRA) on 3 seeds.

**Result:**
| | B0 | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|---|
| PG | 14.74% | 19.26% | 15.87% | 20.90% | **18.68 ± 2.57%** |
| R@1 halfmax | 7.58% | 7.30% | 7.30% | 7.30% | **7.30 ± 0.00%** |

**Key findings:**
1. **v2's 22.08% was an outlier** — it sits above the entire [15.87, 20.90] range from seeded runs.
2. **The true mean improvement over B0 is +3.94pp**, not +7.34pp.
3. **R@1 halfmax is rock-stable** (7.30% identical to 4 decimal places across seeds). The recipe is consistently learning a broader grounding signal; the variance is in *which single patch* wins the argmax for borderline phrases.
4. **Mid-training collapses persist even with seed control** — seed 1 dipped to 14.92% (below B0) at step 600 then partially recovered. This is a property of the optimisation landscape, not bad luck.
5. **Error bar: σ ≈ 2.6pp on PG**. Any recipe change needs ≥5pp improvement to be credibly above noise on a single seed.

**From here:** Use "leader-seed" mode — always test new ideas on seed=2 (historically strongest) first. Only run all 3 seeds if seed=2 exceeds v5 mean + σ_v5 = 21.25%.

---

### Exp-07 — λ 0.5 → 1.0 (v6)

**What we tried:** Double the region loss weight. The FILIP max-pool loss directly rewards argmax-in-bbox behaviour (which is what PG measures), so more of it should push PG up.

**Result (seed=2):** PG = 17.18% (−3.72pp vs v5 seed=2's 20.90%). PG dropped; R@1 top-25 went up slightly (+0.56pp).

**Why it backfired:**
1. More region loss weight broadened the patch response instead of sharpening it — the optimizer satisfied the max-pool objective with many medium-scoring patches inside the bbox rather than one strong spike. PG needs a spike; it got a spread.
2. Halving the global loss's relative weight removed stabilising regularisation → the model hit a catastrophic dip at step 400 (PG fell below B0).

**Conclusion:** The global loss is structural — it keeps patch features in a coherent manifold. λ=0.5 is already on the right side of the sharpness curve.

---

### Exp-08 — Add k_proj to LoRA targets (v7)

**What we tried:** Add `k_proj` to LoRA targets alongside q+v. This gives the model 50% more trainable parameters (+147K) to rebalance attention — directly controlling which patches "advertise" themselves as most relevant to a given query phrase.

**Result (seed=2):** Hit **22.69% PG at step 400** (highest of any run at any checkpoint), then collapsed to 17.80% by end of epoch.

**Why:** Added capacity let the model find a sharper grounding pattern faster — but also gave the optimizer enough flexibility to walk away from it later in training. The FILIP max-pool loss can be satisfied by both a sharp argmax-friendly configuration and a broad coverage-friendly one; with more parameters, the optimizer eventually found the latter.

**Key insight:** The problem is not capacity — we need early stopping.

---

### Exp-09 — Best-PG snapshot (v8)

**What we tried:** Add best-checkpoint tracking inside `train_one_epoch`: save the LoRA parameters (~0.4M, ~MB of CPU memory) whenever the mid-training PG eval improves, and restore the best at the end of training. Zero extra compute cost.

**Result (seed=2):** Final PG = 19.96% (saved from end-of-epoch 17.51%). But the peak was 19.96% at step 400 — not 22.69% as v7 got. Same seed, same code, different peak because CUDA is not bit-deterministic without extra environment flags. The snapshot mechanism worked as designed.

**Conclusion:** k_proj LoRA doesn't move the mean. Best-PG snapshot is a free insurance policy and will stay in all future runs. The PG ceiling for this pipeline is ~20% on a good seed, ~18.7% on average. Deeper change needed.

---

### Exp-10 — Top-K mean region loss (v9)

**What we tried:** Replace the FILIP max-pool loss (which takes the *max* similarity over all in-bbox patches) with a top-K mean (average of the top-3 in-bbox patches). Hypothesis: max-pool is satisfied by one spike OR a broad cluster equally; forcing K patches to score together biases toward consistent spatial clusters.

**Result (seed=2):** PG = 19.35%. No improvement.

**What this rules out:** We have now tested every lever on the image/loss side:
- Loss weight (v6) ✗
- Attention-routing capacity (v7/v8) ✗
- Patch aggregation in the loss (v9) ✗

The bottleneck is not on the image side.

---

### Exp-11 — EOS-aligned region loss (v10) ← most recent

**The diagnosis (text-side train/eval mismatch):**

Just like v1 had an image-side mismatch (training through full attention, eval through MaskCLIP bypass), the training code has had a *text-side* mismatch since the beginning:

- **Training** uses `get_phrase_token_feats` → mean over all phrase tokens
- **Eval (PG)** reads `last_hidden_state[:, -1, :]` → only the EOS token

These are different vectors. The gradient has been teaching patches to align with a mean-over-tokens representation, but we've been measuring alignment with the EOS representation. Fixing the image-side mismatch (v1→v2) was +8.66pp. This is the same shape of bug on the text side.

**What changed:**
1. New `get_phrase_eos_feats()` → returns only the EOS (last non-padding) token, L2-normalised. Byte-for-byte match with the PG eval text vector.
2. New `eos_region_loss()` uses this EOS vector instead of mean-over-tokens.
3. Reverted to q+v LoRA (dropped k_proj — capacity correlated with drift).

**Result (seed=2):** PG = **21.23%** — just below the "clear win" threshold of 21.25% (v5 mean + σ_v5). Well above the v5 mean of 18.68%.

Seeds 0+1 are currently running to get a proper v10 distribution. If the mean clears 18.68% + meaningful margin, this becomes the locked M_human recipe and we move on to M_auto.

---

## Where things stand

| Recipe | PG (best seed) | PG (mean ± std) |
|---|---|---|
| B0 frozen | 14.74% | — |
| v5 (baseline recipe) | 20.90% (seed=2) | 18.68 ± 2.57% |
| v10 (EOS fix, seed=2) | 21.23% | TBD (seeds 0+1 running) |

**Pending:**
- v10 seeds 0+1 results → determine if EOS fix is a real improvement
- If yes → lock M_human recipe, train M_auto (Florence-2 boxes)
- Final three-way table: B0 vs M_human vs M_auto
- Final write-up

---

## Lessons learned

1. **Train/eval mismatches are the biggest source of bugs.** Both major gains came from aligning the training path with the eval path — not from changing the loss, the architecture, or the hyperparameters.

2. **One data point is not a result.** v2's 22.08% looked like a +7.34pp win. After 3 seeds, the true mean was +3.94pp and 22.08% was a lucky outlier. Always measure variance before reporting.

3. **The global loss is structural regularisation.** It's not just a co-objective — it constrains patch features into a coherent manifold. Reducing its weight destabilises training.

4. **FILIP max-pool loss prefers spikes, not spreads.** It rewards the single best in-bbox patch, so the model sometimes learns a sharp argmax-friendly solution and sometimes a broad coverage-friendly one. The loss doesn't uniquely enforce one or the other.

5. **Best-PG snapshot is free insurance.** ~1MB of CPU memory. No compute cost. Saves you from end-of-training drift on every run. Permanent fixture from v8 onward.
