"""Boundary-region mIoU for semantic segmentation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .segmentation import confusion_matrix, compute_miou_macc_oa


def semantic_boundary(label: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Return a semantic boundary mask for an integer label map."""
    valid = label != ignore_index
    boundary = np.zeros(label.shape, dtype=bool)

    diff_lr = valid[:, :-1] & valid[:, 1:] & (label[:, :-1] != label[:, 1:])
    boundary[:, :-1] |= diff_lr
    boundary[:, 1:] |= diff_lr

    diff_ud = valid[:-1, :] & valid[1:, :] & (label[:-1, :] != label[1:, :])
    boundary[:-1, :] |= diff_ud
    boundary[1:, :] |= diff_ud
    return boundary & valid


def boundary_band(label: np.ndarray, width: int = 5, ignore_index: int = 255) -> np.ndarray:
    """Dilated semantic boundary band."""
    b = semantic_boundary(label, ignore_index=ignore_index)
    if width <= 0:
        return b
    structure = np.ones((3, 3), dtype=bool)
    band = b.copy()
    for _ in range(width):
        band = binary_dilation(band, structure=structure)
    return band & (label != ignore_index)


def boundary_miou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    width: int = 5,
    ignore_index: int = 255,
) -> float:
    band = boundary_band(target, width=width, ignore_index=ignore_index)
    target_band = target.copy()
    target_band[~band] = ignore_index
    hist = confusion_matrix(pred, target_band, num_classes, ignore_index=ignore_index)
    return compute_miou_macc_oa(hist)["mIoU"]
