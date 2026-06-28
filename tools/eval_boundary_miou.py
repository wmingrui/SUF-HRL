#!/usr/bin/env python3
"""Evaluate boundary mIoU at multiple boundary widths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sufh_rl.datasets import DATASET_REGISTRY
from sufh_rl.metrics.boundary_miou import boundary_miou
from sufh_rl.models import build_model
from sufh_rl.utils import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--widths", nargs="+", type=int, default=[3, 5, 7])
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
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg["training"].get("num_workers", 4))
    accum = {w: [] for w in args.widths}
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            out = model(images)
            pred = out["seg_logits"].argmax(dim=1).cpu().numpy()[0]
            label = labels.numpy()[0]
            for w in args.widths:
                accum[w].append(boundary_miou(pred, label, num_classes, width=w, ignore_index=ignore_index))
    result = {f"B@{w}": float(np.mean(vals)) for w, vals in accum.items()}
    result["Avg_B_mIoU"] = float(np.mean(list(result.values())))
    print(json.dumps(result, indent=2))
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
