"""One-shot edit of graft-training.ipynb to:

1. Insert a new cell with Flickr30k Pointing Game + Recall@1 eval functions.
2. Switch the periodic eval inside train_one_epoch from VOC to Pointing Game.
3. Replace the final eval cell to run PG + Recall@1 + VOC guardrail.
4. Replace the results cell to compare against the locked B0 baselines.
5. (v2) Rewrite feature-helpers cell with MaskCLIP bypass helpers, and install
   the bypass globally during M_human training so the gradient flows through the
   same patch representation the metrics read. Fixes the v1 regression (PG=13.42%
   < B0=14.74%) caused by training on full-attention patches while eval reads
   v_proj-bypass patches.
6. (v3 / Exp-04) Wrap train_one_epoch in a multi-epoch loop and override
   CFG['epochs']=2 inside the M_human cell. v2's trajectory was monotonic-climbing
   with region loss still falling at epoch end (undertrained). Spawns a fresh
   W&B run named `m_human_v3_e2_...`. Results cell now shows B0 vs v2 vs v3.
7. (v4 / Exp-05) v3 regressed to PG≈19% because the cosine schedule was
   stretched over 2 epochs, so the model never got a tight settle window like v2
   had. Fix: `make_optimizer_scheduler(..., restarts=CFG['epochs'])` divides the
   total step budget into N equal cycles, each warmup→cosine→0. Each epoch now
   gets its own settle phase. W&B run: `m_human_v4_e2_restart_...`.
8. (v5 / Exp-06) v4 trajectories collapsed mid-epoch (loss kept falling while
   PG dropped 4pp — loss/metric decoupling). Combined with v2=22%, v4 prev=18%,
   v4 this=17%, "same code, different shuffle" spans ~5pp. v5 fixes Python /
   NumPy / torch CPU+CUDA seeds and passes a torch.Generator to the DataLoader,
   then runs v2's exact recipe (1 epoch, single cosine, restarts=1) across
   seeds {0, 1, 2}. Output: PG = 18.68 ± 2.57% (mean ± std, n=3) — first real
   variance estimate. v2's 22.08 was outside [v5 min, v5 max] = [15.87, 20.90].
9. (v6 / Exp-07) Push PG up by raising λ_region 0.5 → 1.0. FILIP max-pool
   region loss directly targets argmax-in-bbox behaviour (= PG metric);
   reducing the global-loss counterweight should sharpen argmax. Uses
   leader-seed mode (seed=2, v5's strongest at 20.90%) for fast iteration;
   if a recipe change wins by > σ_v5 ≈ 2.6pp on the leader seed we re-verify
   on all 3 seeds. Cell 10 toggles: LAMBDA_OVERRIDE, SEEDS, EXPERIMENT_TAG.
   RESULT: PG=17.18% on seed=2, Δ=-3.72pp vs v5 seed=2. λ↑ broadened patch
   coverage instead of sharpening argmax (R@1 top-25 +0.56pp while PG fell);
   also removed global-loss regularisation → step-400 collapse to 14.45%.
   Conclusion: λ=0.5 is on the right side of the sharpness curve.
10. (v7 / Exp-08) Pivot: keep λ=0.5 (proven), add `k_proj` to LoRA targets.
    Hypothesis: PG ceiling at q+v LoRA is set by attention-routing capacity.
    K controls what each patch "advertises" as its key — directly determines
    which patch the phrase Q attends to most. Adding it gives the model
    50% more LoRA params dedicated to attention rebalancing, exactly the
    operation PG depends on. Cell 10 adds LORA_TARGETS toggle.
    RESULT: v7 seed=2 hit PG=22.69% at step 400 (HIGHEST of any run at any
    checkpoint) then collapsed to 17.80% by end-of-epoch. Same global loss
    or better. Capacity helped the model REACH a better point but also let
    it walk away (loss/metric decoupling amplified by added flexibility).
    Conclusion: k_proj works, we just need early stopping / best-ckpt.
11. (v8 / Exp-09) Add best-PG checkpoint tracking to train_one_epoch:
    snapshot trainable params (~0.4M) whenever periodic PG eval improves;
    restore the best snapshot at end of training. Costs ~MB of CPU memory
    per run, zero training compute. Re-run v7 recipe (q+k+v, λ=0.5, seed=2)
    with this in place — trajectory is deterministic, peak should land at
    step 400 PG=22.69%. If v8 final PG ≈ 22.69% (vs v7's 17.80%), the win
    is real and we re-verify on seeds 0+1 for the full 3-seed mean.
    RESULT: v8 seed=2 peak = 19.96% at step 400 (not 22.69%). v7's
    step-400 spike was a noisy outlier — CUDA/cuDNN is not bit-deterministic
    without extra flags, so same-seed runs trace similar but not identical
    trajectories. Best-PG snapshot mechanism works as designed (saved us
    from 17.51% end-of-epoch). Below v5 seed=2's 20.90%; k_proj LoRA does
    not move the mean on its own. PG ceiling for this pipeline ≈ 20% on a
    good seed; need a more fundamental change.
12. (v9 / Exp-10) Replace FILIP max-pool region loss with top-K mean.
    Both v6 (λ↑ broadens) and v7/v8 (capacity ≠ sharper) point at the same
    bottleneck: max-pool is satisfied by ONE strong in-bbox patch OR MANY
    medium ones. Top-K mean (k=3) forces consistent in-bbox response —
    you cannot satisfy it with a single spike. Same recipe as v8 otherwise
    (q+k+v, λ=0.5, seed=2, +best-PG). Cell 10 toggles REGION_TOP_K.

Idempotent: re-running rewrites the same cells; the inserted cell is detected by a marker
in its source so it isn't duplicated.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / 'notebooks' / 'graft-training.ipynb'

PG_EVAL_MARKER = '# ── 8b. Quick Pointing Game + Recall@1 eval (Flickr30k) ──'

DATASET_MARKER = '# ── 4. Dataset'

DATASET_SRC = '''# ── 4. Dataset ────────────────────────────────────────────────────────────────
# Per-image dataset. Returns (pil, caption, orig_size, [(phrase, bbox), ...]).
#
# 90-10 train-test split on the Flickr30k train images (split_seed=42, fixed):
#   - train_ds : 90% of train images (the LoRA sees these gradients).
#   - test_ds  : 10% of train images held out from training. Never seen by
#                the model during training; not the same set as val_ds.
#                Used ONCE per run, after best-PG restore, to report the
#                real generalisation PG/R@1.
#   - val_ds   : full Flickr30k val split. quick_pointing_game_eval samples
#                n_images=CFG["eval_images"]=200 of these as the best-PG
#                snapshot signal during training and for the val-PG number
#                in the per-seed table.
# Split is deterministic in split_seed and the underlying hf_data row order;
# do NOT change split_seed across experiments or the test set will shift.

import random as _split_random


def make_train_test_split(hf_data, split='train', test_frac=0.10, split_seed=42):
    """Deterministic 90-10 partition of a Flickr30k split by filename.
    Returns (train_filenames_set, test_filenames_set)."""
    filenames = sorted([r['filename'] for r in hf_data if r['split'] == split])
    rng = _split_random.Random(split_seed)
    rng.shuffle(filenames)
    n_test = int(round(len(filenames) * test_frac))
    test_filenames  = set(filenames[:n_test])
    train_filenames = set(filenames[n_test:])
    return train_filenames, test_filenames


class FlickrSigLIPDataset(Dataset):
    """One item per image. Returns (pil, caption, orig_size, phrase_boxes).
    Optionally filters to a subset of filenames via image_ids."""

    def __init__(self, hf_data, ann_dir: Path, sent_dir: Path,
                 split: str = 'train', image_ids=None):
        self.ann_dir  = ann_dir
        self.sent_dir = sent_dir
        rows = [r for r in hf_data if r['split'] == split]
        if image_ids is not None:
            id_set = set(image_ids)
            rows = [r for r in rows if r['filename'] in id_set]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # decode PIL image
        img_field = row['image']
        if isinstance(img_field, PILImage.Image):
            pil = img_field.convert('RGB')
        else:
            pil = PILImage.open(io.BytesIO(img_field['bytes'])).convert('RGB')
        orig_size = pil.size  # (W, H)

        caption = random.choice(row['caption'])

        # phrase-bbox pairs from Entities annotations
        stem = row['filename'].replace('.jpg', '')
        try:
            anns  = get_annotations(str(self.ann_dir  / f'{stem}.xml'))
            sents = get_sentence_data(str(self.sent_dir / f'{stem}.txt'))
        except Exception:
            return pil, caption, orig_size, []

        phrase_boxes = []
        seen = set()
        for sent in sents:
            for phrase in sent['phrases']:
                pid = phrase['phrase_id']
                if pid in seen or pid not in anns['boxes']:
                    continue
                seen.add(pid)
                for box in anns['boxes'][pid]:
                    phrase_boxes.append((phrase['phrase'], box))

        return pil, caption, orig_size, phrase_boxes

    @staticmethod
    def collate_fn(batch):
        pils, captions, orig_sizes, phrase_boxes = zip(*batch)
        return list(pils), list(captions), list(orig_sizes), list(phrase_boxes)


TRAIN_IDS, TEST_IDS = make_train_test_split(
    hf_data, split='train', test_frac=0.10, split_seed=42,
)
train_ds = FlickrSigLIPDataset(hf_data, ANN_DIR, SENT_DIR, split='train', image_ids=TRAIN_IDS)
test_ds  = FlickrSigLIPDataset(hf_data, ANN_DIR, SENT_DIR, split='train', image_ids=TEST_IDS)
val_ds   = FlickrSigLIPDataset(hf_data, ANN_DIR, SENT_DIR, split='val')

print(f'Train (90% of train): {len(train_ds):,} images')
print(f'Test  (10% of train, held out): {len(test_ds):,} images')
print(f'Val   (Flickr30k val): {len(val_ds):,} images')

# Sanity check one sample
pil0, cap0, sz0, pb0 = train_ds[0]
print(f'\\nSample: size={sz0}  caption="{cap0[:60]}..."')
print(f'phrase-box pairs: {len(pb0)}  e.g. {pb0[0] if pb0 else "none"}')
'''


FEATURE_HELPERS_SRC = '''# ── 7. Feature extraction helpers ─────────────────────────────────────────────
import types


def _maskclip_attn_fwd(self, hidden_states, attention_mask=None, **kwargs):
    """MaskCLIP-style bypass for the last attention layer: replace full
    self-attention with out_proj(v_proj(hidden_states)). When installed
    permanently during training, the gradient flows through the same patch
    representation that PG / R@1 / VOC eval reads — fixes the v1 mismatch
    where training used full attention but metrics used v_proj-only.
    """
    return self.out_proj(self.v_proj(hidden_states)), None


def install_maskclip_bypass(model):
    """Swap the last vision attention layer's forward to the MaskCLIP bypass.
    Returns the original forward so it can be restored after training.
    Safe under gradient checkpointing: the bypass stays installed for the
    rematerialised forward during backward, so the autograd graph stays
    consistent."""
    last_attn = model.vision_model.encoder.layers[-1].self_attn
    orig_fwd  = last_attn.forward
    last_attn.forward = types.MethodType(_maskclip_attn_fwd, last_attn)
    return orig_fwd


def restore_attn(model, orig_fwd):
    """Undo install_maskclip_bypass."""
    model.vision_model.encoder.layers[-1].self_attn.forward = orig_fwd


def get_image_feats(model, pixel_values):
    """Global image embedding via MHAP pooler_output, L2-normalised.
    Uses pooler_output so the global loss gradient does not collapse patch tokens.
    """
    out = model.vision_model(pixel_values=pixel_values)
    return F.normalize(out.pooler_output, dim=-1)   # (B, D)


def get_patch_feats(model, pixel_values):
    """Per-patch features after post_layernorm, L2-normalised. Shape (B, N, D).
    Caller is expected to have installed the MaskCLIP bypass on the last
    attention layer (see install_maskclip_bypass) so train and eval read
    the same patch representation."""
    out = model.vision_model(pixel_values=pixel_values)
    return F.normalize(out.last_hidden_state, dim=-1)   # (B, N, D)


def get_text_feats(model, input_ids, attention_mask=None):
    """Global text embedding: EOS token, L2-normalised. Shape (B, D)."""
    out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    return F.normalize(out.last_hidden_state[:, -1, :], dim=-1)   # (B, D)


def get_phrase_token_feats(model, processor, phrases, device):
    """Per-token text features for a list of phrases. Returns list of (T_i, D) tensors."""
    inputs = processor(text=phrases, return_tensors='pt',
                       padding=True, truncation=True).to(device)
    out    = model.text_model(**inputs)
    hs     = out.last_hidden_state   # (B, T, D)
    result = []
    for i, phrase in enumerate(phrases):
        if 'attention_mask' in inputs:
            length = inputs['attention_mask'][i].sum().item()
        else:
            length = hs.shape[1]
        tok_feats = F.normalize(hs[i, :length, :], dim=-1)  # (T, D)
        result.append(tok_feats)
    return result


def get_phrase_eos_feats(model, processor, phrases, device):
    """Per-phrase EOS-position feature (last non-padding token), L2-normalised.
    Returns (B, D). This matches the PG / R@1 eval protocol exactly:
    eval reads `last_hidden_state[:, -1, :]` of the text encoder, and
    EOS-region-loss training reads the same vector. Fixes the v1→v9
    text-side train/eval mismatch (training previously meaned over ALL
    phrase tokens via get_phrase_token_feats; eval used only EOS).
    """
    inputs = processor(text=phrases, return_tensors='pt',
                       padding=True, truncation=True).to(device)
    out    = model.text_model(**inputs)
    hs     = out.last_hidden_state                                # (B, T, D)
    if 'attention_mask' in inputs:
        last_idx  = (inputs['attention_mask'].sum(dim=1) - 1).clamp(min=0)
        batch_idx = torch.arange(hs.shape[0], device=hs.device)
        last_hs   = hs[batch_idx, last_idx]                       # (B, D)
    else:
        last_hs = hs[:, -1, :]
    return F.normalize(last_hs, dim=-1)


print('Feature extraction helpers defined (with MaskCLIP bypass).')
'''

PG_EVAL_SRC = '''# ── 8b. Quick Pointing Game + Recall@1 eval (Flickr30k) ──────────────────────
# Same protocol as 06_pointing_game_eval.ipynb so trained-model numbers are
# directly comparable to B0 = 14.74% (PG) / 6.83% (R@1 top-10) / 7.91% (top-25).
# Both eval functions accept an optional rows=<list> kwarg: when provided,
# eval runs on those exact rows (no sub-sampling). When omitted, samples
# n_images rows from the Flickr30k val split (default behaviour used by
# best-PG snapshot and final val-set reporting). Used for the 90-10 train-test
# split test eval: pass rows=test_ds.rows for the held-out test set.

_pg_val_rows = None

def _get_pg_val_rows():
    global _pg_val_rows
    if _pg_val_rows is None:
        _pg_val_rows = [r for r in hf_data if r['split'] == 'val']
    return _pg_val_rows


def _load_phrase_boxes(row):
    stem = row['filename'].replace('.jpg', '')
    try:
        anns  = get_annotations(str(ANN_DIR  / f'{stem}.xml'))
        sents = get_sentence_data(str(SENT_DIR / f'{stem}.txt'))
    except Exception:
        return []
    pairs, seen = [], set()
    for sent in sents:
        for phrase in sent['phrases']:
            pid = phrase['phrase_id']
            if pid in seen or pid not in anns['boxes']:
                continue
            seen.add(pid)
            for box in anns['boxes'][pid]:
                pairs.append((phrase['phrase'], box))
    return pairs


def _row_to_pil(row):
    img = row['image']
    if isinstance(img, PILImage.Image):
        return img.convert('RGB')
    return PILImage.open(io.BytesIO(img['bytes'])).convert('RGB')


def quick_pointing_game_eval(model, processor, n_images=200, seed=42, device=DEVICE,
                              rows=None):
    """Flickr30k Pointing Game. MaskCLIP-style last-attention bypass.
    When rows is None, samples n_images from the val split (default).
    When rows is a list, evaluates on every row (no sampling).
    Returns (acc_pct, n_correct, n_total)."""
    model.eval()
    if rows is None:
        rows = _get_pg_val_rows()
        rng  = random.Random(seed)
        rows = rng.sample(rows, min(n_images, len(rows)))

    n_side    = CFG['n_side']
    eval_size = CFG['eval_size']
    patch_px  = eval_size / n_side

    last_attn = model.vision_model.encoder.layers[-1].self_attn
    orig_fwd  = last_attn.forward
    last_attn.forward = types.MethodType(_maskclip_attn_fwd, last_attn)

    n_correct = n_total = 0
    try:
        for row in tqdm(rows, desc='PG eval', leave=False):
            phrase_boxes = _load_phrase_boxes(row)
            if not phrase_boxes:
                continue
            pil = _row_to_pil(row)
            W, H = pil.size

            pix = processor(images=pil, return_tensors='pt').pixel_values.to(device)
            with torch.no_grad():
                out = model.vision_model(pixel_values=pix)
            patch_feats = F.normalize(out.last_hidden_state[0], dim=-1)

            for phrase, bbox in phrase_boxes:
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                txt_in = processor(text=[phrase], return_tensors='pt',
                                   padding=True, truncation=True).to(device)
                with torch.no_grad():
                    txt_hs = model.text_model(**txt_in).last_hidden_state
                txt_feat = F.normalize(txt_hs[0, -1, :], dim=-1)

                sim  = patch_feats @ txt_feat
                best = sim.argmax().item()
                pr, pc = divmod(best, n_side)
                cx = (pc + 0.5) * patch_px * W / eval_size
                cy = (pr + 0.5) * patch_px * H / eval_size
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    n_correct += 1
                n_total += 1
    finally:
        last_attn.forward = orig_fwd

    return 100.0 * n_correct / max(n_total, 1), n_correct, n_total


def _patches_to_box(indices, n_side, patch_px, W, H, eval_size):
    rs = np.array([i // n_side for i in indices])
    cs = np.array([i %  n_side for i in indices])
    x1 = float(cs.min()       * patch_px * W / eval_size)
    y1 = float(rs.min()       * patch_px * H / eval_size)
    x2 = float((cs.max() + 1) * patch_px * W / eval_size)
    y2 = float((rs.max() + 1) * patch_px * H / eval_size)
    return x1, y1, x2, y2


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def quick_recall1_eval(model, processor, n_images=200, seed=42,
                        iou_thresh=0.5, device=DEVICE, rows=None):
    """Flickr30k Recall@1 @ IoU >= iou_thresh. When rows is None, samples
    n_images from the val split (default). When rows is a list, evaluates
    on every row (no sampling). Returns dict for top10/top25/halfmax."""
    model.eval()
    if rows is None:
        rows = _get_pg_val_rows()
        rng  = random.Random(seed)
        rows = rng.sample(rows, min(n_images, len(rows)))

    n_side    = CFG['n_side']
    eval_size = CFG['eval_size']
    patch_px  = eval_size / n_side

    last_attn = model.vision_model.encoder.layers[-1].self_attn
    orig_fwd  = last_attn.forward
    last_attn.forward = types.MethodType(_maskclip_attn_fwd, last_attn)

    hits   = {'top10': 0, 'top25': 0, 'halfmax': 0}
    totals = {'top10': 0, 'top25': 0, 'halfmax': 0}

    try:
        for row in tqdm(rows, desc='R@1 eval', leave=False):
            phrase_boxes = _load_phrase_boxes(row)
            if not phrase_boxes:
                continue
            pil = _row_to_pil(row)
            W, H = pil.size

            pix = processor(images=pil, return_tensors='pt').pixel_values.to(device)
            with torch.no_grad():
                out = model.vision_model(pixel_values=pix)
            patch_feats = F.normalize(out.last_hidden_state[0], dim=-1)

            for phrase, gt in phrase_boxes:
                x1, y1, x2, y2 = gt
                if x2 <= x1 or y2 <= y1:
                    continue
                txt_in = processor(text=[phrase], return_tensors='pt',
                                   padding=True, truncation=True).to(device)
                with torch.no_grad():
                    txt_hs = model.text_model(**txt_in).last_hidden_state
                txt_feat = F.normalize(txt_hs[0, -1, :], dim=-1)
                sim = (patch_feats @ txt_feat).cpu().numpy()

                strategies = {
                    'top10':   np.argsort(sim)[-10:][::-1],
                    'top25':   np.argsort(sim)[-25:][::-1],
                    'halfmax': np.where(sim >= 0.5 * sim.max())[0],
                }
                for name, idxs in strategies.items():
                    totals[name] += 1
                    if len(idxs) == 0:
                        continue
                    pred = _patches_to_box(list(idxs), n_side, patch_px, W, H, eval_size)
                    if _iou(pred, gt) >= iou_thresh:
                        hits[name] += 1
    finally:
        last_attn.forward = orig_fwd

    return {k: (100.0 * hits[k] / max(totals[k], 1), hits[k], totals[k]) for k in hits}


print('quick_pointing_game_eval, quick_recall1_eval defined.')
'''

TRAIN_LOOP_SRC = '''# ── 9. Training loop ─────────────────────────────────────────────────────────

def eos_region_loss(patch_feats_n, phrase_eos_feats, bbox_masks,
                    logit_scale, logit_bias, k=None):
    """EOS-aligned region loss (v10 / Exp-11). Each phrase is represented
    by its EOS-position embedding (B, D) — matches the PG / R@1 eval
    protocol byte-for-byte. For each (phrase_i, image_j) pair we compute
    sim = max-over-patches(phrase_i_eos · patch_j_n | n ∈ bbox_j), then a
    sigmoid SigLIP contrastive over the (B, B) score matrix.

    Fixes the train/eval text-side mismatch present since v1: training
    previously meaned similarity over ALL phrase tokens (FILIP fine-grained)
    while eval used only EOS. Direct mirror of the v1→v2 patch-side fix
    that gave +8.66pp PG. If k is set, uses top-k mean instead of pure
    max (analogous to topk_region_loss for the patch dimension).
    """
    B, N, D = patch_feats_n.shape
    device  = patch_feats_n.device
    dtype   = patch_feats_n.dtype

    # phrase_eos_feats: (B, D); patch_feats_n: (B, N, D).
    # sim[i, j, n] = phrase_eos_i · image_j_patch_n
    sim = (phrase_eos_feats @ patch_feats_n.reshape(B * N, D).T).reshape(B, B, N)

    # Mask out-of-bbox patches per image_j; broadcast over phrase_i dim.
    sim = sim.masked_fill(~bbox_masks[None, :, :], float('-inf'))

    if k is not None and k > 1:
        k_eff       = min(k, N)
        top_vals, _ = sim.topk(k_eff, dim=-1)                       # (B, B, k)
        valid       = top_vals.isfinite()
        top_vals    = top_vals.masked_fill(~valid, 0.0)
        n_valid     = valid.sum(dim=-1).clamp(min=1).to(dtype)
        scores      = top_vals.sum(dim=-1) / n_valid                # (B, B)
    else:
        scores = sim.amax(dim=-1)                                   # (B, B)

    # Empty bboxes → 0 score (rather than -inf or nan).
    empty_box = ~bbox_masks.any(dim=-1)                             # (B,)
    scores    = scores.masked_fill(empty_box[None, :], 0.0)

    logits = logit_scale.exp() * scores + logit_bias
    labels = 2 * torch.eye(B, device=device) - 1
    return -F.logsigmoid(labels * logits).mean()


def topk_region_loss(patch_feats_n, phrase_token_feats, bbox_masks,
                     logit_scale, logit_bias, k=3):
    """Top-K mean variant of filip_region_loss. Replaces the in-bbox max-pool
    with the mean of the top-K bbox patch similarities. k=1 is mathematically
    equivalent to filip_region_loss. Used by v9 (Exp-10): max-pool can be
    satisfied by ONE strong in-bbox patch (sharp argmax = good PG) OR by
    MANY medium in-bbox patches (broad = bad PG, ok R@1); both v6 (λ↑) and
    v7/v8 (capacity↑) showed the optimizer drifting between these. Top-K
    mean requires K patches to land in the bbox, biasing toward tighter
    spatial clustering. Same shape contract as filip_region_loss.
    """
    B, N, D = patch_feats_n.shape
    device  = patch_feats_n.device
    dtype   = patch_feats_n.dtype

    T_max = max(t.shape[0] for t in phrase_token_feats)
    tokens_pad  = patch_feats_n.new_zeros(B, T_max, D)
    phrase_mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    for i, t in enumerate(phrase_token_feats):
        tokens_pad[i, :t.shape[0]]  = t
        phrase_mask[i, :t.shape[0]] = True

    bbox_patches = [patch_feats_n[j][bbox_masks[j]] for j in range(B)]
    K_max = max(max(p.shape[0] for p in bbox_patches), 1)
    patches_pad = patch_feats_n.new_zeros(B, K_max, D)
    patch_mask  = torch.zeros(B, K_max, dtype=torch.bool, device=device)
    for j, p in enumerate(bbox_patches):
        if p.shape[0] > 0:
            patches_pad[j, :p.shape[0]] = p
            patch_mask[j, :p.shape[0]]  = True

    sim = (tokens_pad.reshape(B * T_max, D) @
           patches_pad.reshape(B * K_max, D).T
           ).reshape(B, T_max, B, K_max)
    sim = sim.masked_fill(~patch_mask[None, None], float('-inf'))

    # Top-K mean over patches. -inf entries (came from bbox padding) are
    # masked out before averaging; we divide by the actual count of valid
    # contributions, which can be < k for small bboxes.
    k_eff       = min(k, K_max)
    top_vals, _ = sim.topk(k_eff, dim=-1)                         # (B, T_max, B, k_eff)
    valid_mask  = top_vals.isfinite()
    top_vals    = top_vals.masked_fill(~valid_mask, 0.0)
    n_valid     = valid_mask.sum(dim=-1).clamp(min=1).to(dtype)   # (B, T_max, B)
    max_sim     = top_vals.sum(dim=-1) / n_valid                  # (B, T_max, B)

    empty_box = ~patch_mask.any(dim=-1)                           # (B,)
    max_sim   = max_sim.masked_fill(empty_box[None, None, :], 0.0)

    max_sim = max_sim.masked_fill(~phrase_mask[:, :, None], 0.0)
    n_toks  = phrase_mask.sum(dim=1).clamp(min=1).to(dtype)
    scores  = max_sim.sum(dim=1) / n_toks[:, None]

    logits = logit_scale.exp() * scores + logit_bias
    labels = 2 * torch.eye(B, device=device) - 1
    return -F.logsigmoid(labels * logits).mean()


def set_seed(seed):
    """Seed Python / NumPy / torch (CPU + CUDA) so runs are reproducible up to
    cuDNN non-determinism. Pair with a torch.Generator passed to DataLoader to
    fix the shuffle order — that's the dominant variance source in our LoRA
    finetune (5pp PG spread observed across v2, v4 prev, v4 this).
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, processor, train_ds, lambda_region,
                    optimizer, scheduler, device, cfg,
                    run_tag='', step_offset=0, eval_every=200,
                    dataloader_gen=None, keep_best_pg=True,
                    test_rows=None, ckpt_save_dir=None, epochs=1):
    """Train for `epochs` passes over train_ds. Best-PG snapshot tracking
    persists across epochs. When test_rows is provided, periodic eval also
    runs PG on a 200-image test subsample. When ckpt_save_dir is set, saves
    LoRA adapter at every eval point and separately saves the best-PG model.
    """
    model.train()
    loader = DataLoader(
        train_ds,
        batch_size   = cfg['batch_size'],
        shuffle      = True,
        num_workers  = 2,
        collate_fn   = FlickrSigLIPDataset.collate_fn,
        pin_memory   = device == 'cuda',
        generator    = dataloader_gen,
    )

    best_pg    = -1.0
    best_step  = -1
    best_state = None   # dict of trainable param tensors on CPU

    test_eval_rows = None
    if test_rows is not None:
        _trng = random.Random(42)
        test_eval_rows = _trng.sample(test_rows, min(200, len(test_rows)))

    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    total_loss = total_global = total_region = 0.0
    n_steps = 0
    _steps_per_epoch = len(loader)

    def _multi_epoch():
        for _ in range(epochs):
            yield from loader

    pbar = tqdm(_multi_epoch(), total=_steps_per_epoch * epochs,
                desc=f'train {run_tag}')
    for pils, captions, orig_sizes, phrase_boxes_batch in pbar:

        # ── L_global: image ↔ caption ────────────────────────────────────────
        img_inputs = processor(images=pils, return_tensors='pt',
                               padding=True).to(device)
        txt_inputs = processor(text=captions, return_tensors='pt',
                               padding=True, truncation=True).to(device)

        with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            img_feats = get_image_feats(model, img_inputs['pixel_values'])
            txt_feats = get_text_feats(model, txt_inputs['input_ids'])
            l_global  = siglip_global_loss(
                img_feats, txt_feats,
                model.logit_scale, model.logit_bias
            )

        # ── L_region: phrase ↔ bbox patches ──────────────────────────────────
        l_region = torch.tensor(0.0, device=device)

        if lambda_region > 0:
            triples = []
            for img_idx, pb_list in enumerate(phrase_boxes_batch):
                for phrase, bbox in pb_list:
                    triples.append((img_idx, phrase, bbox))

            if len(triples) >= 2:
                random.shuffle(triples)
                triples = triples[:cfg['region_batch']]

                reg_img_idxs = [t[0] for t in triples]
                reg_phrases  = [t[1] for t in triples]
                reg_bboxes   = [t[2] for t in triples]
                reg_orig_sz  = [orig_sizes[i] for i in reg_img_idxs]
                reg_pils     = [pils[i] for i in reg_img_idxs]
                reg_pix      = processor(images=reg_pils, return_tensors='pt',
                                         padding=True).pixel_values.to(device)

                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    patch_feats_n = get_patch_feats(model, reg_pix)
                    bbox_masks = torch.stack([
                        bbox_patch_mask(bbox, sz, cfg['n_side'], cfg['eval_size'])
                        for bbox, sz in zip(reg_bboxes, reg_orig_sz)
                    ]).to(device)
                    # Two-axis dispatch:
                    #   cfg['phrase_repr'] in {None, 'tokens'} → FILIP per-token mean
                    #   cfg['phrase_repr'] == 'eos'            → EOS-aligned (v10)
                    #   cfg['region_top_k']: None / 1 → max-pool; int >= 2 → top-K mean
                    _phrase_repr = cfg.get('phrase_repr', 'tokens')
                    _top_k       = cfg.get('region_top_k', None)
                    if _phrase_repr == 'eos':
                        phrase_eos = get_phrase_eos_feats(
                            model, processor, reg_phrases, device
                        )
                        l_region = eos_region_loss(
                            patch_feats_n, phrase_eos, bbox_masks,
                            model.logit_scale, model.logit_bias, k=_top_k,
                        )
                    else:
                        phrase_tok_feats = get_phrase_token_feats(
                            model, processor, reg_phrases, device
                        )
                        if _top_k is not None and _top_k > 1:
                            l_region = topk_region_loss(
                                patch_feats_n, phrase_tok_feats, bbox_masks,
                                model.logit_scale, model.logit_bias, k=_top_k,
                            )
                        else:
                            l_region = filip_region_loss(
                                patch_feats_n, phrase_tok_feats, bbox_masks,
                                model.logit_scale, model.logit_bias,
                            )

        loss = l_global + lambda_region * l_region

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss   += loss.item()
        total_global += l_global.item()
        total_region += l_region.item() if isinstance(l_region, torch.Tensor) else 0.0
        n_steps      += 1
        global_step   = step_offset + n_steps

        wandb.log({
            f'{run_tag}/loss':         loss.item(),
            f'{run_tag}/loss_global':  l_global.item(),
            f'{run_tag}/loss_region':  l_region.item() if isinstance(l_region, torch.Tensor) else 0.0,
            f'{run_tag}/lr':           scheduler.get_last_lr()[0],
            f'{run_tag}/logit_scale':  model.logit_scale.item(),
            f'{run_tag}/logit_bias':   model.logit_bias.item(),
        }, step=global_step)

        pbar.set_postfix({
            'ep':     f'{n_steps // _steps_per_epoch + 1}/{epochs}',
            'loss':   f'{total_loss/n_steps:.3f}',
            'global': f'{total_global/n_steps:.3f}',
            'region': f'{total_region/n_steps:.3f}',
        })

        # ── Periodic eval (val + test PG) + checkpoint save ───────────────────
        if eval_every > 0 and n_steps % eval_every == 0:
            pg_acc, pg_correct, pg_total = quick_pointing_game_eval(
                model, processor,
                n_images = cfg['eval_images'],
                seed     = 42,
                device   = device,
            )
            wandb.log({f'{run_tag}/pointing_game': pg_acc}, step=global_step)
            pbar.write(f'  step {global_step:4d}  val PG = {pg_acc:.2f}%  '
                       f'({pg_correct}/{pg_total})  [B0=14.74%]')

            if test_eval_rows is not None:
                t_pg, t_c, t_t = quick_pointing_game_eval(
                    model, processor, seed=42, device=device,
                    rows=test_eval_rows,
                )
                wandb.log({f'{run_tag}/test_pointing_game': t_pg}, step=global_step)
                pbar.write(f'           test PG = {t_pg:.2f}%  ({t_c}/{t_t})')

            if keep_best_pg and pg_acc > best_pg:
                best_pg    = pg_acc
                best_step  = global_step
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.named_parameters()
                              if v.requires_grad}
                pbar.write(f'    new best PG = {pg_acc:.2f}% (snapshot kept)')
                if ckpt_save_dir is not None:
                    model.save_pretrained(f'{ckpt_save_dir}/best_pg')

            if ckpt_save_dir is not None:
                model.save_pretrained(f'{ckpt_save_dir}/step_{global_step}')

            model.train()

    # Restore best-PG snapshot so subsequent evals see the model's best state.
    if keep_best_pg and best_state is not None:
        for k, v in model.named_parameters():
            if k in best_state:
                v.data.copy_(best_state[k].to(v.device))
        print(f'  Restored best-PG snapshot: step {best_step}, PG = {best_pg:.2f}%')

    stats = {
        'loss':     total_loss   / n_steps,
        'global':   total_global / n_steps,
        'region':   total_region / n_steps,
        'best_pg':  best_pg,
        'best_step': best_step,
    }
    return stats, n_steps


def make_optimizer_scheduler(model, n_steps, cfg, restarts=1):
    """AdamW + LambdaLR. With restarts=1 (default), a single warmup→cosine→0
    schedule covers all n_steps (used by v2, v3). With restarts>1, the schedule
    is divided into `restarts` equal cycles, each its own warmup→cosine→0 —
    every epoch gets a fresh peak LR and a tight late-cycle settle window.
    This is the v3 → v4 fix (Exp-05): v3 stretched the cosine over 2 epochs
    and the LR never decayed enough to let the model settle into v2's basin.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    steps_per_cycle = max(1, n_steps // max(1, restarts))
    warmup          = cfg['warmup_steps']
    def lr_lambda(step):
        cycle_step = step % steps_per_cycle
        if cycle_step < warmup:
            return cycle_step / max(1, warmup)
        progress = (cycle_step - warmup) / max(1, steps_per_cycle - warmup)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


print('Training loop defined.')
'''

TRAIN_MH_SRC = '''# ── 10. Train M_human ────────────────────────────────────────────────────────
# Mode toggles. Same cell runs v5 variance study or v6+ leader-seed sweeps.
# Recap of priors (see report.md for full write-ups):
#   v5 (variance,    Exp-06): n=3 seeds, q+v,   FILIP/tokens → 18.68 ± 2.57% (leader seed=2: 20.90%)
#   v6 (λ=1.0,       Exp-07): seed=2,  q+v,   FILIP/tokens → 17.18% (λ↑ broadens, no help)
#   v7 (q+k+v,       Exp-08): seed=2,  q+k+v, FILIP/tokens → 17.80% (noisy step-400 spike 22.69 was outlier)
#   v8 (+ best-PG,   Exp-09): seed=2,  q+k+v, FILIP/tokens + best-PG → 19.96% (still no gain)
#   v9 (top-K=3,     Exp-10): seed=2,  q+k+v, top3/tokens + best-PG → 19.35% (no gain)
# Pattern across v6/v7/v8/v9: peak at step 400, post-peak drift. v5's q+v +
# FILIP was the only recipe to climb monotonically.
#
# v10 (Exp-11): direct mirror of the v1→v2 patch-side alignment fix (+8.66pp)
# on the TEXT side. Training previously meaned similarity over ALL phrase
# tokens (FILIP fine-grained); eval used ONLY the EOS token. v10 trains on
# the same EOS vector eval reads, removing the mismatch. Reverts to q+v LoRA
# (added capacity in v7/v8/v9 correlated with drift); keeps best-PG snapshot
# as free insurance.

SEEDS           = [0, 1, 2]          # 3-seed verification
LAMBDA_OVERRIDE = None               # keep λ=0.5 (proven)
LORA_TARGETS    = ['q_proj', 'v_proj']   # locked v10: q+v
REGION_TOP_K    = None               # locked v10: pure max-pool over patches
PHRASE_REPR     = 'eos'              # locked v10: EOS phrase repr (matches eval)
EXPERIMENT_TAG  = 'v11_5ep'          # v10 recipe, 5 epochs, periodic test eval + Drive checkpoints
CFG['epochs']   = 5                  # 5 epochs with cosine restart per epoch

if REGION_TOP_K is not None:
    CFG['region_top_k'] = REGION_TOP_K
CFG['phrase_repr'] = PHRASE_REPR

if LAMBDA_OVERRIDE is not None:
    CFG['lambda_region'] = LAMBDA_OVERRIDE
print(f'Running {EXPERIMENT_TAG} | seeds={SEEDS} | λ_region={CFG["lambda_region"]}'
      f' | LoRA targets={LORA_TARGETS if LORA_TARGETS else "default(q+v)"}'
      f' | region_top_k={CFG.get("region_top_k", "max-pool (FILIP)")}'
      f' | phrase_repr={CFG.get("phrase_repr", "tokens (FILIP)")}')

# Build a per-experiment LoraConfig if LORA_TARGETS is set; otherwise reuse
# cell-7's lora_cfg. Reads other hyperparameters from the existing config so
# rank / alpha / dropout / bias / task_type stay consistent.
if LORA_TARGETS is not None:
    from peft import LoraConfig as _LoraConfig
    _lora_cfg_active = _LoraConfig(
        r              = lora_cfg.r,
        lora_alpha     = lora_cfg.lora_alpha,
        target_modules = LORA_TARGETS,
        lora_dropout   = lora_cfg.lora_dropout,
        bias           = lora_cfg.bias,
        task_type      = lora_cfg.task_type,
    )
else:
    _lora_cfg_active = lora_cfg

# Free any leftover model from a previous run before the loop.
import gc
for _var in ['model_mh', 'opt_mh', 'sch_mh']:
    if _var in globals():
        del globals()[_var]
gc.collect()
torch.cuda.empty_cache()
print(f'GPU free at start: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB')

import os as _os
DRIVE_CKPT_DIR = '/content/drive/MyDrive/graft_checkpoints'
_os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)

v5_results = []

for seed in SEEDS:
    print(f'\\n{"="*72}\\n  M_human v5 — seed={seed}\\n{"="*72}')
    set_seed(seed)

    # Fresh W&B run per seed.
    if wandb.run is not None:
        wandb.finish()
    mh_run = wandb.init(
        project = 'region-grounded',
        name    = f'm_human_{EXPERIMENT_TAG}_seed{seed}_lam{CFG["lambda_region"]}',
        config  = {**CFG, 'variant': f'm_human_{EXPERIMENT_TAG}', 'seed': seed,
                   'lr_schedule': 'cosine_single',
                   'lora_targets': list(_lora_cfg_active.target_modules),
                   'lora_rank':    _lora_cfg_active.r,
                   'lora_alpha':   _lora_cfg_active.lora_alpha,
                   'phrase_repr':  PHRASE_REPR or 'tokens',
                   'region_loss':  ('eos_' if PHRASE_REPR == 'eos' else 'filip_') +
                                   (f'top{REGION_TOP_K}' if REGION_TOP_K and REGION_TOP_K > 1 else 'max')},
        tags    = ['siglip', 'flickr30k', 'lora', 'm_human', EXPERIMENT_TAG,
                   f'seed{seed}', f'lam{CFG["lambda_region"]}'],
        reinit  = True,
    )
    print(f'  W&B: {mh_run.url}')

    # Fresh model + LoRA per seed.
    if 'model_mh' in globals():
        del model_mh
        gc.collect()
        torch.cuda.empty_cache()
    base_model_mh = SiglipModel.from_pretrained(CFG['model_id'],
                                                 token=os.environ['HF_TOKEN'])
    model_mh = get_peft_model(base_model_mh, _lora_cfg_active).to(DEVICE)
    for n, p in model_mh.named_parameters():
        if 'logit_scale' in n or 'logit_bias' in n:
            p.requires_grad_(True)
    model_mh.enable_input_require_grads()
    model_mh.gradient_checkpointing_enable()

    _n_trainable = sum(p.numel() for p in model_mh.parameters() if p.requires_grad)
    print(f'  Trainable params: {_n_trainable/1e6:.3f}M | LoRA targets: '
          f'{list(_lora_cfg_active.target_modules)} | rank: {_lora_cfg_active.r}')

    steps_per_epoch = len(train_ds) // CFG['batch_size']
    n_steps_mh     = steps_per_epoch * CFG['epochs']
    opt_mh, sch_mh = make_optimizer_scheduler(
        model_mh, n_steps_mh, CFG, restarts=CFG['epochs'],
    )

    # MaskCLIP bypass — train and eval read the same patch representation.
    orig_attn_fwd_mh = install_maskclip_bypass(model_mh)

    # Step-0 sanity (deterministic — just B0 reload with bypass).
    pg0, pg0_c, pg0_t = quick_pointing_game_eval(
        model_mh, processor, n_images=CFG['eval_images'], seed=42, device=DEVICE,
    )
    print(f'  Seed {seed} step 0 PG: {pg0:.2f}%  ({pg0_c}/{pg0_t})')
    wandb.log({'mhuman/pointing_game': pg0}, step=0)

    # Deterministic DataLoader shuffle: torch.Generator seeded explicitly.
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)

    _seed_ckpt_dir = f'{DRIVE_CKPT_DIR}/{EXPERIMENT_TAG}_seed{seed}_lam{CFG["lambda_region"]}'
    _os.makedirs(_seed_ckpt_dir, exist_ok=True)

    stats_ep, ep_steps = train_one_epoch(
        model_mh, processor, train_ds,
        lambda_region = CFG['lambda_region'],
        optimizer     = opt_mh,
        scheduler     = sch_mh,
        device        = DEVICE,
        cfg           = CFG,
        run_tag       = 'mhuman',
        step_offset   = 0,
        eval_every    = 200,
        dataloader_gen= dl_gen,
        test_rows     = test_ds.rows,
        ckpt_save_dir = _seed_ckpt_dir,
        epochs        = CFG['epochs'],
    )

    # Final PG + R@1 on the val split (same protocol as B0, n_images=200).
    pg_acc, pg_c, pg_t = quick_pointing_game_eval(
        model_mh, processor, n_images=CFG['eval_images'], seed=42, device=DEVICE,
    )
    r1 = quick_recall1_eval(
        model_mh, processor, n_images=CFG['eval_images'], seed=42, device=DEVICE,
    )

    # Held-out test eval: full 10% of train (~2,978 images), never seen by
    # training and not the same set as the best-PG snapshot signal. Runs on
    # the best-PG-restored model state.
    test_pg, test_pg_c, test_pg_t = quick_pointing_game_eval(
        model_mh, processor, seed=42, device=DEVICE, rows=test_ds.rows,
    )
    test_r1 = quick_recall1_eval(
        model_mh, processor, seed=42, device=DEVICE, rows=test_ds.rows,
    )
    print(f'  Seed {seed} TEST PG: {test_pg:.2f}%  ({test_pg_c}/{test_pg_t})')

    model_mh.save_pretrained(
        f'{CFG["ckpt_dir"]}/mhuman_{EXPERIMENT_TAG}_seed{seed}_lam{CFG["lambda_region"]}'
    )
    restore_attn(model_mh, orig_attn_fwd_mh)

    v5_results.append({
        'seed':            seed,
        'pg_acc':          pg_acc,
        'recall1_top10':   r1['top10'][0],
        'recall1_top25':   r1['top25'][0],
        'recall1_halfmax': r1['halfmax'][0],
        'test_pg':         test_pg,
        'test_recall1_top10':   test_r1['top10'][0],
        'test_recall1_top25':   test_r1['top25'][0],
        'test_recall1_halfmax': test_r1['halfmax'][0],
        'test_n_phrases':       test_pg_t,
        'loss':            stats_ep['loss'],
        'global':          stats_ep['global'],
        'region':          stats_ep['region'],
        'best_pg':         stats_ep.get('best_pg',   pg_acc),
        'best_step':       stats_ep.get('best_step', -1),
    })

    wandb.log({
        'eval/mhuman_pointing_game'      : pg_acc,
        'eval/mhuman_recall1_top10'      : r1['top10'][0],
        'eval/mhuman_recall1_top25'      : r1['top25'][0],
        'eval/mhuman_recall1_halfmax'    : r1['halfmax'][0],
        'eval/mhuman_test_pointing_game' : test_pg,
        'eval/mhuman_test_recall1_top10' : test_r1['top10'][0],
        'eval/mhuman_test_recall1_top25' : test_r1['top25'][0],
        'eval/mhuman_test_recall1_halfmax': test_r1['halfmax'][0],
        'mhuman/final_loss'              : stats_ep['loss'],
        'mhuman/final_loss_global'       : stats_ep['global'],
        'mhuman/final_loss_region'       : stats_ep['region'],
    }, step=ep_steps)
    wandb.summary.update({
        'seed'                : seed,
        'pointing_game'       : pg_acc,
        'recall1_top10'       : r1['top10'][0],
        'recall1_top25'       : r1['top25'][0],
        'recall1_halfmax'     : r1['halfmax'][0],
        'test_pointing_game'  : test_pg,
        'test_recall1_top10'  : test_r1['top10'][0],
        'test_recall1_top25'  : test_r1['top25'][0],
        'test_recall1_halfmax': test_r1['halfmax'][0],
        'test_n_phrases'      : test_pg_t,
    })
    wandb.finish()

    print(f'  Seed {seed} final val PG: {pg_acc:.2f}%  ({pg_c}/{pg_t})')

# Cross-seed summary (printed here AND aggregated more fully in cell 11).
pg_vals = np.array([r['pg_acc'] for r in v5_results])
print(f'\\n{"="*72}\\n  {EXPERIMENT_TAG} summary ({len(v5_results)} seed(s), λ={CFG["lambda_region"]})\\n{"="*72}')
for r in v5_results:
    print(f'  seed {r["seed"]}: PG = {r["pg_acc"]:.2f}%')
if len(pg_vals) >= 2:
    print(f'  mean ± std : {pg_vals.mean():.2f} ± {pg_vals.std(ddof=1):.2f}%')
    print(f'  range      : [{pg_vals.min():.2f}, {pg_vals.max():.2f}]')
else:
    print(f'  single seed (no std). vs v5 seed{v5_results[0]["seed"]} baseline 20.90%: '
          f'Δ = {v5_results[0]["pg_acc"] - 20.90:+.2f}pp')
'''

RESULTS_SRC = '''# ── 11. Results — per-seed table + reference deltas ───────────────────────────
# Reads v5_results (per-seed dicts) populated by cell 10. Works for any
# EXPERIMENT_TAG ('v5' variance study with n=3, 'v6' leader-seed with n=1,
# or future sweeps). Single-seed runs are compared against v5's leader
# (seed=2 @ λ=0.5, PG=20.90%); multi-seed runs print mean ± std and compare
# against v5's full distribution.
B0 = dict(
    pointing_game   = 14.74,
    recall1_top10   =  6.83,
    recall1_top25   =  7.91,
    recall1_halfmax =  7.58,
    voc_miou        =  1.43,
)
MH_V5 = dict(   # locked v5 baseline (Exp-06, λ=0.5, n=3 seeds)
    pg_mean         = 18.68,
    pg_std          =  2.57,
    pg_seed2_leader = 20.90,
    recall1_top10   =  5.76,
    recall1_top25   =  7.12,
    recall1_halfmax =  7.30,
)

pg_vals      = np.array([r['pg_acc']          for r in v5_results])
r10_vals     = np.array([r['recall1_top10']   for r in v5_results])
r25_vals     = np.array([r['recall1_top25']   for r in v5_results])
rhm_vals     = np.array([r['recall1_halfmax'] for r in v5_results])
has_test     = all('test_pg' in r for r in v5_results)
test_pg_vals = np.array([r.get('test_pg', np.nan) for r in v5_results]) if has_test else None
test_rhm_vals= np.array([r.get('test_recall1_halfmax', np.nan) for r in v5_results]) if has_test else None

print('=' * 110)
print(f'  M_human {EXPERIMENT_TAG}  —  {len(v5_results)} seed(s), {CFG["epochs"]} epoch(s), λ={CFG["lambda_region"]}')
print('=' * 110)
print(f'  {"seed":>5} | {"val PG":>8} | {"test PG":>8} | {"best PG":>8} | {"@step":>6} | '
      f'{"R@1 top-10":>12} | {"R@1 top-25":>12} | {"R@1 halfmax":>13} | {"loss":>7}')
print('-' * 110)
for r in v5_results:
    bp = r.get('best_pg', r['pg_acc'])
    bs = r.get('best_step', -1)
    tp = r.get('test_pg', float('nan'))
    print(f'  {r["seed"]:>5} | {r["pg_acc"]:7.2f}% | {tp:7.2f}% | {bp:7.2f}% | {bs:>6d} | '
          f'{r["recall1_top10"]:11.2f}% | {r["recall1_top25"]:11.2f}% | '
          f'{r["recall1_halfmax"]:12.2f}% | {r["loss"]:7.3f}')
print('-' * 110)

pg_mean = float(pg_vals.mean())
test_pg_mean = float(test_pg_vals.mean()) if has_test else float('nan')
if len(pg_vals) >= 2:
    pg_std = float(pg_vals.std(ddof=1))
    test_pg_std = float(test_pg_vals.std(ddof=1)) if has_test else float('nan')
    def _agg(vals):
        return f'{vals.mean():.2f} ± {vals.std(ddof=1):.2f}'
    print(f'  val mean± | {_agg(pg_vals):>7}% | {_agg(r10_vals):>11}% | '
          f'{_agg(r25_vals):>11}% | {_agg(rhm_vals):>12}% |')
    print(f'  val range | [{pg_vals.min():.2f}, {pg_vals.max():.2f}]  '
          f'(spread = {pg_vals.max() - pg_vals.min():.2f}pp)')
    if has_test:
        print(f'  test mean±: PG = {_agg(test_pg_vals):>7}%  '
              f'(n_phrases ≈ {int(v5_results[0]["test_n_phrases"]):,})')
        print(f'  test range: [{test_pg_vals.min():.2f}, {test_pg_vals.max():.2f}]  '
              f'(spread = {test_pg_vals.max() - test_pg_vals.min():.2f}pp)')
        print(f'  val→test delta (mean): {test_pg_mean - pg_mean:+.2f}pp  '
              f'(positive = test PG higher than val PG; ~0 = no overfitting to val)')
else:
    pg_std = float('nan')
    test_pg_std = float('nan')
    print(f'  (single seed — no std)')
print('=' * 96)

print(f'\\nReference comparisons:')
print(f'  B0 (frozen)            : {B0["pointing_game"]:.2f}%')
print(f'  v5 mean ± std (λ=0.5)  : {MH_V5["pg_mean"]:.2f} ± {MH_V5["pg_std"]:.2f}%')
print(f'  v5 seed=2 leader (λ=0.5): {MH_V5["pg_seed2_leader"]:.2f}%')
print(f'  this run PG            : {pg_mean:.2f}%' +
      (f' ± {pg_std:.2f}' if len(pg_vals) >= 2 else ''))
print(f'  Δ vs B0                : {pg_mean - B0["pointing_game"]:+.2f}pp')
print(f'  Δ vs v5 mean           : {pg_mean - MH_V5["pg_mean"]:+.2f}pp')
print(f'  Δ vs v5 seed=2 leader  : {pg_mean - MH_V5["pg_seed2_leader"]:+.2f}pp')
# Significance call: leader-seed run is "real win" only if it beats v5 mean by > σ_v5.
if len(pg_vals) == 1:
    real_win = pg_mean > MH_V5['pg_mean'] + MH_V5['pg_std']
    print(f'\\n>> Single-seed gain > σ_v5 (noise floor)? '
          f'{"YES — re-verify on all 3 seeds" if real_win else "NO (within noise)"}')

# Aggregate W&B run capturing this experiment's headline numbers.
if wandb.run is not None:
    wandb.finish()
agg_run = wandb.init(
    project = 'region-grounded',
    name    = f'm_human_{EXPERIMENT_TAG}_aggregate_n{len(v5_results)}_lam{CFG["lambda_region"]}',
    config  = {**CFG, 'variant': f'm_human_{EXPERIMENT_TAG}_aggregate', 'seeds': SEEDS},
    tags    = ['siglip', 'flickr30k', 'lora', 'm_human', EXPERIMENT_TAG, 'aggregate'],
    reinit  = True,
)
_summary = {
    'experiment_tag'    : EXPERIMENT_TAG,
    'lambda_region'     : CFG['lambda_region'],
    'seeds_n'           : len(v5_results),
    'pg_mean'           : pg_mean,
    'pg_std'            : pg_std,
    'pg_min'            : float(pg_vals.min()),
    'pg_max'            : float(pg_vals.max()),
    'b0_pointing_game'  : B0['pointing_game'],
    'v5_pg_mean'        : MH_V5['pg_mean'],
    'v5_pg_seed2_leader': MH_V5['pg_seed2_leader'],
    'delta_vs_b0'       : pg_mean - B0['pointing_game'],
    'delta_vs_v5_mean'  : pg_mean - MH_V5['pg_mean'],
    'delta_vs_v5_leader': pg_mean - MH_V5['pg_seed2_leader'],
}
if has_test:
    _summary.update({
        'test_pg_mean'       : test_pg_mean,
        'test_pg_std'        : test_pg_std,
        'test_pg_min'        : float(test_pg_vals.min()),
        'test_pg_max'        : float(test_pg_vals.max()),
        'test_delta_vs_b0'   : test_pg_mean - B0['pointing_game'],
        'test_minus_val_pg'  : test_pg_mean - pg_mean,
    })
wandb.summary.update(_summary)
wandb.finish()
print('Aggregate W&B run finished.')
'''


TRAIN_B1_MARKER = '# ── 12. Train B1 (control: λ_region=0, no region loss) ──'


def code_cell(src: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': src.splitlines(keepends=True),
    }


def first_line(cell) -> str:
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    return src.split('\n', 1)[0]


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb['cells']

    # ── 0. Rewrite feature-helpers cell with MaskCLIP bypass helpers ───────────
    fh_idx = next(i for i, c in enumerate(cells)
                  if first_line(c).startswith('# ── 7. Feature extraction helpers'))
    cells[fh_idx]['source'] = FEATURE_HELPERS_SRC.splitlines(keepends=True)
    cells[fh_idx]['outputs'] = []
    cells[fh_idx]['execution_count'] = None
    print(f'Rewrote feature-helpers cell at index {fh_idx}')

    # ── 0b. Rewrite dataset cell (adds 90-10 train-test split + image_ids filter)
    ds_idx = next(i for i, c in enumerate(cells)
                  if first_line(c).startswith(DATASET_MARKER))
    cells[ds_idx]['source'] = DATASET_SRC.splitlines(keepends=True)
    cells[ds_idx]['outputs'] = []
    cells[ds_idx]['execution_count'] = None
    print(f'Rewrote dataset cell at index {ds_idx}')

    # ── 1. Insert PG eval cell after the VOC eval cell, if not already present ──
    voc_idx = next(i for i, c in enumerate(cells)
                   if first_line(c).startswith('# ── 8. Quick VOC eval'))
    pg_exists = any(PG_EVAL_MARKER in ''.join(c['source']) for c in cells)
    if not pg_exists:
        cells.insert(voc_idx + 1, code_cell(PG_EVAL_SRC))
        print(f'Inserted PG eval cell at index {voc_idx + 1}')
    else:
        # rewrite in place to keep idempotent
        pg_idx = next(i for i, c in enumerate(cells)
                      if PG_EVAL_MARKER in ''.join(c['source']))
        cells[pg_idx]['source'] = PG_EVAL_SRC.splitlines(keepends=True)
        cells[pg_idx]['outputs'] = []
        cells[pg_idx]['execution_count'] = None
        print(f'Rewrote existing PG eval cell at index {pg_idx}')

    # ── 2. Replace training loop cell ──────────────────────────────────────────
    loop_idx = next(i for i, c in enumerate(cells)
                    if first_line(c).startswith('# ── 9. Training loop'))
    cells[loop_idx]['source'] = TRAIN_LOOP_SRC.splitlines(keepends=True)
    cells[loop_idx]['outputs'] = []
    cells[loop_idx]['execution_count'] = None
    print(f'Rewrote training loop cell at index {loop_idx}')

    # ── 3. Replace train M_human cell ──────────────────────────────────────────
    train_idx = next(i for i, c in enumerate(cells)
                     if first_line(c).startswith('# ── 10. Train M_human'))
    cells[train_idx]['source'] = TRAIN_MH_SRC.splitlines(keepends=True)
    cells[train_idx]['outputs'] = []
    cells[train_idx]['execution_count'] = None
    print(f'Rewrote train M_human cell at index {train_idx}')

    # ── 4. Replace results cell ────────────────────────────────────────────────
    res_idx = next(i for i, c in enumerate(cells)
                   if first_line(c).startswith('# ── 11. Results'))
    cells[res_idx]['source'] = RESULTS_SRC.splitlines(keepends=True)
    cells[res_idx]['outputs'] = []
    cells[res_idx]['execution_count'] = None
    print(f'Rewrote results cell at index {res_idx}')

    # ── 5. Remove any pre-existing B1 control cell (deprecated; see report.md) ──
    b1_indices = [i for i, c in enumerate(cells)
                  if TRAIN_B1_MARKER in ''.join(c['source'])]
    for bi in reversed(b1_indices):
        cells.pop(bi)
        print(f'Removed B1 cell at index {bi}')

    NB_PATH.write_text(json.dumps(nb, indent=1) + '\n')
    print(f'Wrote {NB_PATH}')


if __name__ == '__main__':
    main()
