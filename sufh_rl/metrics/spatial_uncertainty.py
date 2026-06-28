"""Spatial uncertainty metrics used in SUF-HRL.

BFUR, DSCG, and MSAD are adapted from the spatially aware uncertainty
evaluation idea and reformulated for remote sensing semantic segmentation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .boundary_miou import boundary_band, semantic_boundary

EPS = 1e-8


def bfur(uncertainty: np.ndarray, label: np.ndarray, width: int = 5, ignore_index: int = 255) -> float:
    band = boundary_band(label, width=width, ignore_index=ignore_index)
    valid = label != ignore_index
    nonband = valid & (~band)
    if band.sum() == 0 or nonband.sum() == 0:
        return 0.0
    mu_b = float(uncertainty[band].mean())
    mu_nb = float(uncertainty[nonband].mean())
    return mu_b / (mu_b + mu_nb + EPS)


def dscg(
    uncertainty: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray,
    bins=(0, 3, 5, 7, 1e9),
    ignore_index: int = 255,
) -> float:
    valid = label != ignore_index
    boundary = semantic_boundary(label, ignore_index=ignore_index)
    dist = distance_transform_edt(~boundary)
    error = (pred != label) & valid
    total = 0.0
    weight_sum = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = valid & (dist >= lo) & (dist < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        w = n / max(int(valid.sum()), 1)
        total += w * abs(float(uncertainty[mask].mean()) - float(error[mask].mean()))
        weight_sum += w
    return total / max(weight_sum, EPS)


def msad(
    uncertainty: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray,
    sigmas=(1.0, 3.0, 5.0),
    weights=(0.5, 0.3, 0.2),
    ignore_index: int = 255,
) -> float:
    valid = label != ignore_index
    error = ((pred != label) & valid).astype(np.float32)
    uncertainty = uncertainty.astype(np.float32) * valid.astype(np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    denom = max(float(valid.sum()), 1.0)
    score = 0.0
    for sigma, w in zip(sigmas, weights):
        u = gaussian_filter(uncertainty, sigma=float(sigma))
        e = gaussian_filter(error, sigma=float(sigma))
        score += float(w) * float(np.abs(u - e)[valid].sum() / denom)
    return score


def spatial_uncertainty_metrics(
    uncertainty: np.ndarray,
    pred: np.ndarray,
    label: np.ndarray,
    ignore_index: int = 255,
) -> dict:
    return {
        "BFUR": bfur(uncertainty, label, ignore_index=ignore_index),
        "DSCG": dscg(uncertainty, pred, label, ignore_index=ignore_index),
        "MSAD": msad(uncertainty, pred, label, ignore_index=ignore_index),
    }
