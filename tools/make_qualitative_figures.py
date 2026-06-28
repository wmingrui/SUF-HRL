#!/usr/bin/env python3
"""Create qualitative visualization panels for SUF-HRL predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from sufh_rl.datasets import DATASET_REGISTRY
from sufh_rl.losses import entropy_uncertainty
from sufh_rl.models import build_model
from sufh_rl.utils import load_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out-dir", default="qualitative")
    p.add_argument("--num-samples", type=int, default=6)
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


def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    img = img * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    num_classes = int(cfg["dataset"]["num_classes"])
    model = build_model(
        method="suf_hrl",
        num_classes=num_classes,
        backbone_name=cfg["model"].get("backbone", "nvidia/mit-b2"),
        fuse_dim=int(cfg["model"].get("fuse_dim", 256)),
        residual_scale=float(cfg["model"].get("residual_scale", 0.15)),
        pretrained=False,
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    ds = build_dataset(cfg, args.split)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    for idx, batch in enumerate(loader):
        if idx >= args.num_samples:
            break
        image = batch["image"].to(device)
        label = batch["label"].numpy()[0]
        with torch.no_grad():
            out = model(image)
            logits = out["seg_logits"]
            pred = logits.argmax(dim=1).cpu().numpy()[0]
            msp = (1.0 - torch.softmax(logits, dim=1).max(dim=1).values).cpu().numpy()[0]
            ent = entropy_uncertainty(logits).cpu().numpy()[0, 0]
            learned = out.get("unc_map", torch.zeros_like(logits[:, :1])).cpu().numpy()[0, 0]
        img = denorm(batch["image"].numpy()[0])
        panels = [("Image", img), ("GT", label), ("Pred", pred), ("MSP", msp), ("Entropy", ent), ("Learned", learned)]
        fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3))
        for ax, (title, arr) in zip(axes, panels):
            ax.imshow(arr, cmap=None if title == "Image" else "viridis")
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.axis("off")
        fig.tight_layout()
        name = batch.get("id", batch.get("name", [f"sample_{idx:04d}"]))[0]
        fig.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
