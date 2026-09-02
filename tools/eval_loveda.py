#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

# Ensure that public CLI entry points always use this repository checkout,
# even when another SUF-HRL version is installed in the Python environment.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sufh_rl.datasets import LoveDAMulticlassDataset
from sufh_rl.metrics import compute_miou_macc_oa, update_hist_from_logits
from sufh_rl.models import build_model
from sufh_rl.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate released SUF-HRL checkpoint using the "
            "official LoveDA full-image validation protocol."
        )
    )

    p.add_argument("--config", default="configs/loveda.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-json", default=None)
    p.add_argument("--device", default="cuda")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    dcfg = cfg["dataset"]
    root = Path(dcfg["root"])

    # IMPORTANT:
    # crop_size=None reproduces the paper's full-image validation.
    ds = LoveDAMulticlassDataset(
        image_dir=str(root / dcfg["image_dir"]),
        label_dir=str(root / dcfg["label_dir"]),
        split_file=str(root / "splits" / f"{args.split}.txt"),
        crop_size=None,
        mode="val",
        normalize=True,
        ignore_index=dcfg.get("ignore_index", 255),
    )

    print(f"[DATASET] LoveDA full-image {args.split}: {len(ds)} images")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        method="suf_hrl",
        num_classes=7,
        backbone_name=cfg["model"].get("backbone", "nvidia/mit-b2"),
        fuse_dim=cfg["model"].get("fuse_dim", 256),
        residual_scale=cfg["model"].get("residual_scale", 0.15),
        delta_hidden_dims=(128, 64),
    ).to(device)

    state = torch.load(args.checkpoint, map_location="cpu")

    model.load_state_dict(state, strict=True)
    print("[CKPT] strict=True: PASS")

    model.eval()

    hist = np.zeros((7, 7), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            out = model(images)

            hist = update_hist_from_logits(
                hist,
                out["seg_logits"],
                labels,
                7,
                255,
            )

    metrics = compute_miou_macc_oa(hist)

    metrics["protocol"] = "LoveDA 7-class full-image validation"

    print(json.dumps(metrics, indent=2))

    if args.save_json:
        p = Path(args.save_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
