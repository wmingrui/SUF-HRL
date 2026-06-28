#!/usr/bin/env python3
"""Evaluate mIoU, mAcc, and OA from a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sufh_rl.datasets import DATASET_REGISTRY
from sufh_rl.metrics import compute_miou_macc_oa, update_hist_from_logits
from sufh_rl.models import build_model
from sufh_rl.utils import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)

    num_classes = int(cfg["dataset"]["num_classes"])
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))
    model = build_model(
        method=cfg.get("method", "suf_hrl"),
        num_classes=num_classes,
        backbone_name=cfg["model"].get("backbone", "nvidia/mit-b2"),
        fuse_dim=int(cfg["model"].get("fuse_dim", 256)),
        residual_scale=float(cfg["model"].get("residual_scale", 0.15)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = build_dataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=cfg["training"].get("val_batch_size", 4), shuffle=False, num_workers=cfg["training"].get("num_workers", 4))
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            out = model(images)
            hist = update_hist_from_logits(hist, out["seg_logits"], labels, num_classes, ignore_index)
    metrics = compute_miou_macc_oa(hist)
    print(json.dumps(metrics, indent=2))
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
