"""Segmentation losses and hard-pixel mining helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multiclass_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Multiclass Dice loss over valid pixels."""
    valid = labels != ignore_index
    safe_labels = labels.clone()
    safe_labels[~valid] = 0

    probs = torch.softmax(logits, dim=1)
    target = F.one_hot(safe_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_4d = valid.unsqueeze(1).float()
    probs = probs * valid_4d
    target = target * valid_4d

    dims = (0, 2, 3)
    inter = (probs * target).sum(dims)
    denom = probs.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Multiclass focal loss."""
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    valid = labels != ignore_index
    pt = torch.exp(-ce)
    loss = ((1.0 - pt) ** gamma) * ce
    return loss[valid].mean() if valid.any() else logits.new_tensor(0.0)


def topk_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    score_map: torch.Tensor | None = None,
    ratio: float = 0.05,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Cross-entropy on top-k valid pixels.

    If score_map is None, per-pixel CE is used for loss-top-k mining.
    Otherwise, score_map should be [B,1,H,W] or [B,H,W], and the top-k pixels
    are selected according to the score.
    """
    ce = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")
    valid = labels != ignore_index
    if not valid.any():
        return logits.new_tensor(0.0)

    if score_map is None:
        scores = ce.detach()
    else:
        if score_map.dim() == 4:
            score_map = score_map.squeeze(1)
        scores = score_map.detach()

    valid_scores = scores[valid]
    valid_losses = ce[valid]
    k = max(1, int(round(ratio * valid_scores.numel())))
    k = min(k, valid_scores.numel())
    _, idx = torch.topk(valid_scores, k=k, largest=True)
    return valid_losses[idx].mean()


def entropy_uncertainty(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1, keepdim=True)
    return entropy / torch.log(torch.tensor(float(logits.shape[1]), device=logits.device))
