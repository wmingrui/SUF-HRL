#!/usr/bin/env python3
"""Train SegFormer baselines or SUF-HRL from a YAML config.

The script is intentionally compact and readable. It reproduces the main
training logic used in the paper: CE+Dice segmentation supervision, optional
spatial uncertainty losses, and optional top-k hard-region supervision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sufh_rl.datasets import DATASET_REGISTRY
from sufh_rl.losses import (
    BoundaryConcentrationLossMapMultiClass,
    LocalAlignmentLossMapMultiClass,
    entropy_uncertainty,
    focal_loss,
    multiclass_dice_loss,
    topk_cross_entropy,
)
from sufh_rl.metrics import compute_miou_macc_oa, update_hist_from_logits
from sufh_rl.models import build_model
from sufh_rl.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--method",
        default=None,
        choices=["baseline", "suf_hrl", "loss_topk", "msp_topk", "entropy_topk", "focal", "ohem"],
        help="Override cfg['method'].",
    )
    return parser.parse_args()


def build_dataset(cfg: dict, split: str):
    dataset_name = cfg["dataset"]["name"].lower()
    dataset_cls = DATASET_REGISTRY[dataset_name]
    root = Path(cfg["dataset"]["root"])
    crop_size = cfg["dataset"].get("crop_size", 512)
    return dataset_cls(
        image_dir=str(root / cfg["dataset"].get("image_dir", "processed_multiclass/images")),
        label_dir=str(root / cfg["dataset"].get("label_dir", "processed_multiclass/labels")),
        split_file=str(root / "splits" / f"{split}.txt"),
        crop_size=crop_size,
        mode="train" if split == "train" else "val",
        normalize=True,
        ignore_index=cfg["dataset"].get("ignore_index", 255),
    )


def validate(model, loader, num_classes: int, ignore_index: int, device: torch.device) -> dict:
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            out = model(images)
            hist = update_hist_from_logits(hist, out["seg_logits"], labels, num_classes, ignore_index)
    return compute_miou_macc_oa(hist)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.method is not None:
        cfg["method"] = args.method
    method = cfg.get("method", "suf_hrl")

    set_seed(int(cfg.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = build_dataset(cfg, "train")
    val_set = build_dataset(cfg, "val")
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["training"].get("batch_size", 16)),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["training"].get("val_batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True,
    )

    num_classes = int(cfg["dataset"]["num_classes"])
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))
    model = build_model(
        method=method,
        num_classes=num_classes,
        backbone_name=cfg["model"].get("backbone", "nvidia/mit-b2"),
        fuse_dim=int(cfg["model"].get("fuse_dim", 256)),
        residual_scale=float(cfg["model"].get("residual_scale", 0.15)),
        pretrained=bool(cfg["model"].get("pretrained", True)),
    ).to(device)

    init_ckpt = cfg.get("init_checkpoint")
    if init_ckpt:
        ckpt = torch.load(init_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        msg = model.load_state_dict(state, strict=False)
        print(f"Loaded init checkpoint with msg: {msg}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 6e-5)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    local_loss_fn = LocalAlignmentLossMapMultiClass(
        sigmas=tuple(cfg["loss"].get("local_sigmas", [1.0, 3.0, 5.0])),
        weights=tuple(cfg["loss"].get("local_weights", [0.5, 0.3, 0.2])),
        ignore_index=ignore_index,
    )
    boundary_loss_fn = BoundaryConcentrationLossMapMultiClass(
        band_width=int(cfg["loss"].get("boundary_width", 3)),
        margin=float(cfg["loss"].get("boundary_margin", 0.18)),
        ignore_index=ignore_index,
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    epochs = int(cfg["training"].get("epochs", 100))
    warmup_epochs = int(cfg["training"].get("topk_warmup_epochs", 5))
    topk_ratio = float(cfg["loss"].get("topk_ratio", 0.05))
    lambda_dice = float(cfg["loss"].get("lambda_dice", 1.0))
    lambda_local = float(cfg["loss"].get("lambda_local", 0.10))
    lambda_boundary = float(cfg["loss"].get("lambda_boundary", 0.05))
    lambda_hard = float(cfg["loss"].get("lambda_hard", 0.40))

    best_miou = -1.0
    history = []
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["training"].get("amp", True)))

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
        running = []
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(cfg["training"].get("amp", True))):
                out = model(images)
                logits = out["seg_logits"]
                ce = F.cross_entropy(logits, labels, ignore_index=ignore_index)
                dice = multiclass_dice_loss(logits, labels, num_classes, ignore_index)
                loss = ce + lambda_dice * dice

                if method == "focal":
                    loss = focal_loss(logits, labels, gamma=float(cfg["loss"].get("focal_gamma", 2.0)), ignore_index=ignore_index) + lambda_dice * dice

                if method in {"loss_topk", "ohem"} and epoch >= warmup_epochs:
                    loss = loss + lambda_hard * topk_cross_entropy(logits, labels, None, topk_ratio, ignore_index)

                if method == "msp_topk" and epoch >= warmup_epochs:
                    probs = torch.softmax(logits, dim=1)
                    score = 1.0 - probs.max(dim=1, keepdim=True).values
                    loss = loss + lambda_hard * topk_cross_entropy(logits, labels, score, topk_ratio, ignore_index)

                if method == "entropy_topk" and epoch >= warmup_epochs:
                    score = entropy_uncertainty(logits)
                    loss = loss + lambda_hard * topk_cross_entropy(logits, labels, score, topk_ratio, ignore_index)

                if method in {"suf_hrl", "suf-hrl", "residual", "uncertainty_topk"}:
                    unc_map = out["unc_map"]
                    local_loss, _ = local_loss_fn(logits, unc_map, labels)
                    boundary_loss, _ = boundary_loss_fn(unc_map, labels)
                    loss = loss + lambda_local * local_loss + lambda_boundary * boundary_loss
                    if epoch >= warmup_epochs:
                        loss = loss + lambda_hard * topk_cross_entropy(logits, labels, unc_map, topk_ratio, ignore_index)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(running):.4f}")

        metrics = validate(model, val_loader, num_classes, ignore_index, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(running)), **metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "metrics": metrics, "epoch": epoch + 1},
                out_dir / "checkpoints" / "best.pth",
            )
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
