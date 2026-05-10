"""Losses for region-grounded SigLIP fine-tuning.

- `sigmoid_loss`: SigLIP's pairwise sigmoid loss (Zhai et al. 2023).
- `filip_late_interaction`: token-wise max-similarity, mean over text tokens
  (matches the proposal's L_late formulation).
- `region_sigmoid_loss`: applies SigLIP's sigmoid loss to a B×B matrix of
  late-interaction similarities — used for region pairs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_loss(
    image_embeds: torch.Tensor,    # [B, D]
    text_embeds: torch.Tensor,     # [B, D]
    logit_scale: torch.Tensor,     # scalar
    logit_bias: torch.Tensor,      # scalar
) -> torch.Tensor:
    img = F.normalize(image_embeds, dim=-1)
    txt = F.normalize(text_embeds, dim=-1)
    logits = logit_scale.exp() * img @ txt.t() + logit_bias
    B = logits.size(0)
    labels = 2 * torch.eye(B, device=logits.device) - 1  # +1 diag, -1 off
    return -F.logsigmoid(labels * logits).mean()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    m = mask.float()
    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)


def filip_late_interaction(
    patch_tokens: torch.Tensor,    # [B_img, N, D] — already L2-normalized
    word_tokens: torch.Tensor,     # [B_txt, M, D] — already L2-normalized
    text_mask: torch.Tensor,       # [B_txt, M] {0,1}
    symmetric: bool = False,
) -> torch.Tensor:
    """Return similarity matrix [B_img, B_txt].

    s_t→v(i, j) = mean_m max_k <v_k^i, w_m^j>   (proposal formulation)
    If symmetric, average with s_v→t(i, j) = mean_k max_m <v_k^i, w_m^j>.
    """
    # [B_img, 1, N, D] · [1, B_txt, M, D] -> [B_img, B_txt, N, M]
    sim = torch.einsum("ind,jmd->ijnm", patch_tokens, word_tokens)
    # text-side: max over patches, then masked mean over words
    max_over_patches = sim.max(dim=2).values            # [B_img, B_txt, M]
    mask = text_mask.unsqueeze(0).expand(sim.size(0), -1, -1).float()
    s_t2v = (max_over_patches * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
    if not symmetric:
        return s_t2v
    s_v2t = sim.max(dim=3).values.mean(dim=2)            # [B_img, B_txt]
    return (s_t2v + s_v2t) / 2.0


def region_sigmoid_loss(
    patch_tokens: torch.Tensor,    # [B, N, D]
    word_tokens: torch.Tensor,     # [B, M, D]
    text_mask: torch.Tensor,       # [B, M]
    logit_scale: torch.Tensor,
    logit_bias: torch.Tensor,
    symmetric: bool = False,
) -> torch.Tensor:
    v = F.normalize(patch_tokens, dim=-1)
    w = F.normalize(word_tokens, dim=-1)
    sims = filip_late_interaction(v, w, text_mask, symmetric=symmetric)
    logits = logit_scale.exp() * sims + logit_bias
    B = logits.size(0)
    labels = 2 * torch.eye(B, device=logits.device) - 1
    return -F.logsigmoid(labels * logits).mean()
