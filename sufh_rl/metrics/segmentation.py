"""Segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    mask = target != ignore_index
    pred = pred[mask].astype(np.int64)
    target = target[mask].astype(np.int64)
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    hist = np.bincount(
        num_classes * target[valid] + pred[valid],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def compute_miou_macc_oa(hist: np.ndarray) -> dict:
    diag = np.diag(hist).astype(np.float64)
    gt = hist.sum(axis=1).astype(np.float64)
    pred = hist.sum(axis=0).astype(np.float64)
    union = gt + pred - diag
    iou = np.divide(diag, union, out=np.zeros_like(diag), where=union > 0)
    acc = np.divide(diag, gt, out=np.zeros_like(diag), where=gt > 0)
    oa = diag.sum() / max(hist.sum(), 1)
    return {
        "mIoU": float(np.nanmean(iou)),
        "mAcc": float(np.nanmean(acc)),
        "OA": float(oa),
        "IoU": iou.tolist(),
        "Acc": acc.tolist(),
    }


def update_hist_from_logits(
    hist: np.ndarray,
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    for p, t in zip(pred, target):
        hist += confusion_matrix(p, t, num_classes, ignore_index)
    return hist
