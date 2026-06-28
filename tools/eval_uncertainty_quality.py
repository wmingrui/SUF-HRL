#!/usr/bin/env python3
"""Evaluate uncertainty quality: AUROC, AUPR, BFUR, DSCG, and MSAD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from sufh_rl.datasets import DATASET_REGISTRY
from sufh_rl.losses import entropy_uncertainty
from sufh_rl.metrics.spatial_uncertainty import spatial_uncertainty_metrics
from sufh_rl.models import build_model
from sufh_rl.utils import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--source", choices=["learned", "msp", "entropy"], default="learned")
    p.add_argument("--save-json", default=None)
    return p.parse_args()


def build_dataset(cfg, split):
    root = Path(cfg["dataset"]["root"])
    cls = DATASET_REGISTRY[cfg["dataset"]["name"].lower()]
    return cls(
        image_dir=str(root / cfg["dataset"].get("image_dir", "processed_multiclass/images")),
        label_dir=str(root / cfg["dataset"].get("label_dir", "processed_multiclass/labels")),
        split_file=str(root / "splits" / f"{split}.txt"),
        crop_size=cfg["dataset"].get("crop_size", 512),
        mode="val",
        normalize=True,
        ignore_index=cfg["dataset"].get("ignore_index", 255),
    )


def safe_auc(y_true, score):
    y_true = np.asarray(y_true).astype(np.uint8)
    score = np.asarray(score).astype(np.float64)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y_true, score)), float(average_precision_score(y_true, score))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)

    num_classes = int(cfg["dataset"]["num_classes"])
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))
    method = cfg.get("method", "suf_hrl")
    if args.source == "learned":
        method = "suf_hrl"
    model = build_model(
        method=method,
        num_classes=num_classes,
        backbone_name=cfg["model"].get("backbone", "nvidia/mit-b2"),
        fuse_dim=int(cfg["model"].get("fuse_dim", 256)),
        residual_scale=float(cfg["model"].get("residual_scale", 0.15)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = build_dataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["training"].get("num_workers", 4))

    all_errors, all_scores = [], []
    spatial_rows = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            label = batch["label"].numpy()[0]
            out = model(images)
            logits = out["seg_logits"]
            pred = logits.argmax(dim=1).cpu().numpy()[0]
            if args.source == "learned" and "unc_map" in out:
                unc = out["unc_map"].squeeze().cpu().numpy()
            elif args.source == "entropy":
                unc = entropy_uncertainty(logits).squeeze().cpu().numpy()
            else:
                prob = torch.softmax(logits, dim=1)
                unc = (1.0 - prob.max(dim=1).values).squeeze().cpu().numpy()

            valid = label != ignore_index
            error = (pred != label) & valid
            all_errors.append(error[valid].ravel())
            all_scores.append(unc[valid].ravel())
            spatial_rows.append(spatial_uncertainty_metrics(unc, pred, label, ignore_index=ignore_index))

    y_true = np.concatenate(all_errors)
    score = np.concatenate(all_scores)
    auroc, aupr = safe_auc(y_true, score)
    result = {"AUROC": auroc, "AUPR": aupr}
    for key in ["BFUR", "DSCG", "MSAD"]:
        result[key] = float(np.nanmean([r[key] for r in spatial_rows]))
    print(json.dumps(result, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
