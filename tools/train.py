#!/usr/bin/env python3
"""Train SegFormer baselines or SUF-HRL from a YAML config.

The public training loop follows the manuscript protocol:
- 512x512 training crops
- total batch size 16 for the standard single-process configuration
- AdamW
- base learning rate 6e-5
- weight decay 0.01
- linear warm-up for 1500 iterations
- polynomial learning-rate decay
- 80,000 total training iterations

SUF-HRL adds local uncertainty alignment, boundary concentration,
and uncertainty-guided top-k hard-region supervision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure that public CLI entry points always use this repository checkout.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML config.",
    )

    parser.add_argument(
        "--method",
        default=None,
        choices=[
            "baseline",
            "suf_hrl",
            "loss_topk",
            "msp_topk",
            "entropy_topk",
            "focal",
            "ohem",
        ],
        help="Override cfg['method'].",
    )

    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Optional checkpoint used only to initialize model weights. "
            "Optimizer/scheduler states are not restored."
        ),
    )
    return parser.parse_args()


def load_initial_weights(model, checkpoint_path):
    """Load model weights from a clean or legacy checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {checkpoint_path}"
        )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    # Current public checkpoints are clean state_dict files.
    # These two branches also allow older internal checkpoint formats.
    if (
        isinstance(state, dict)
        and "model_state_dict" in state
        and isinstance(state["model_state_dict"], dict)
    ):
        state = state["model_state_dict"]

    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    model.load_state_dict(
        state,
        strict=True,
    )

    return checkpoint_path


def build_dataset(cfg: dict, split: str):
    dataset_name = cfg["dataset"]["name"].lower()
    dataset_cls = DATASET_REGISTRY[dataset_name]

    root = Path(cfg["dataset"]["root"])
    crop_size = int(cfg["dataset"].get("crop_size", 512))

    return dataset_cls(
        image_dir=str(
            root
            / cfg["dataset"].get(
                "image_dir",
                "processed_multiclass/images",
            )
        ),
        label_dir=str(
            root
            / cfg["dataset"].get(
                "label_dir",
                "processed_multiclass/labels",
            )
        ),
        split_file=str(root / "splits" / f"{split}.txt"),
        crop_size=crop_size,
        mode="train" if split == "train" else "val",
        normalize=True,
        ignore_index=int(
            cfg["dataset"].get("ignore_index", 255)
        ),
    )


def validate(
    model,
    loader,
    num_classes: int,
    ignore_index: int,
    device: torch.device,
) -> dict:
    model.eval()

    hist = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )
            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            out = model(images)

            hist = update_hist_from_logits(
                hist,
                out["seg_logits"],
                labels,
                num_classes,
                ignore_index,
            )

    return compute_miou_macc_oa(hist)


def compute_poly_lr(
    base_lr: float,
    current_iter: int,
    max_iters: int,
    warmup_iters: int,
    warmup_start_factor: float = 1e-6,
    power: float = 1.0,
) -> float:
    """Linear warm-up followed by polynomial LR decay."""

    if max_iters <= 0:
        raise ValueError("max_iters must be positive.")

    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")

    if current_iter < warmup_iters and warmup_iters > 0:
        alpha = current_iter / float(warmup_iters)

        factor = (
            warmup_start_factor
            + (1.0 - warmup_start_factor) * alpha
        )

        return float(base_lr * factor)

    decay_iters = max(max_iters - warmup_iters, 1)

    progress = (
        current_iter - warmup_iters
    ) / float(decay_iters)

    progress = min(max(progress, 0.0), 1.0)

    factor = (1.0 - progress) ** power

    return float(base_lr * factor)


def set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_metadata(
    path: Path,
    *,
    iteration: int,
    metrics: dict,
    cfg: dict,
) -> None:
    payload = {
        "iteration": int(iteration),
        "metrics": metrics,
        "config": cfg,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.method is not None:
        cfg["method"] = args.method

    method = cfg.get("method", "suf_hrl")

    set_seed(int(cfg.get("seed", 0)))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_set = build_dataset(cfg, "train")
    val_set = build_dataset(cfg, "val")

    train_loader = DataLoader(
        train_set,
        batch_size=int(
            cfg["training"].get("batch_size", 16)
        ),
        shuffle=True,
        num_workers=int(
            cfg["training"].get("num_workers", 4)
        ),
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=int(
            cfg["training"].get("val_batch_size", 4)
        ),
        shuffle=False,
        num_workers=int(
            cfg["training"].get("num_workers", 4)
        ),
        pin_memory=True,
    )

    num_classes = int(
        cfg["dataset"]["num_classes"]
    )

    ignore_index = int(
        cfg["dataset"].get("ignore_index", 255)
    )

    model = build_model(
        method=method,
        num_classes=num_classes,
        backbone_name=cfg["model"].get(
            "backbone",
            "nvidia/mit-b2",
        ),
        fuse_dim=int(
            cfg["model"].get("fuse_dim", 256)
        ),
        residual_scale=float(
            cfg["model"].get(
                "residual_scale",
                0.15,
            )
        ),
        delta_hidden_dims=tuple(
            cfg["model"].get(
                "delta_hidden_dims",
                [],
            )
        ),
    ).to(device)

    base_lr = float(
        cfg["training"].get("lr", 6e-5)
    )

    weight_decay = float(
        cfg["training"].get(
            "weight_decay",
            0.01,
        )
    )

    if args.init_checkpoint is not None:
        init_path = load_initial_weights(
            model,
            args.init_checkpoint,
        )

        print(
            f"[INIT CKPT] loaded with strict=True: "
            f"{init_path}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )

    local_loss_fn = LocalAlignmentLossMapMultiClass(
        sigmas=tuple(
            cfg["loss"].get(
                "local_sigmas",
                [1.0, 3.0, 5.0],
            )
        ),
        weights=tuple(
            cfg["loss"].get(
                "local_weights",
                [0.5, 0.3, 0.2],
            )
        ),
        ignore_index=ignore_index,
    )

    boundary_loss_fn = BoundaryConcentrationLossMapMultiClass(
        band_width=int(
            cfg["loss"].get(
                "boundary_width",
                3,
            )
        ),
        margin=float(
            cfg["loss"].get(
                "boundary_margin",
                0.18,
            )
        ),
        ignore_index=ignore_index,
    )

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_cfg = cfg["training"]

    max_iters = int(
        training_cfg.get(
            "max_iters",
            80000,
        )
    )

    warmup_iters = int(
        training_cfg.get(
            "warmup_iters",
            1500,
        )
    )

    topk_start_iter = int(
        training_cfg.get(
            "topk_start_iter",
            warmup_iters,
        )
    )

    warmup_start_factor = float(
        training_cfg.get(
            "warmup_start_factor",
            1e-6,
        )
    )

    poly_power = float(
        training_cfg.get(
            "poly_power",
            1.0,
        )
    )

    val_interval = int(
        training_cfg.get(
            "val_interval",
            1000,
        )
    )

    log_interval = int(
        training_cfg.get(
            "log_interval",
            50,
        )
    )

    topk_ratio = float(
        cfg["loss"].get(
            "topk_ratio",
            0.05,
        )
    )

    lambda_dice = float(
        cfg["loss"].get(
            "lambda_dice",
            1.0,
        )
    )

    lambda_local = float(
        cfg["loss"].get(
            "lambda_local",
            0.10,
        )
    )

    lambda_boundary = float(
        cfg["loss"].get(
            "lambda_boundary",
            0.05,
        )
    )

    lambda_hard = float(
        cfg["loss"].get(
            "lambda_hard",
            0.40,
        )
    )

    amp_enabled = bool(
        training_cfg.get("amp", True)
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled
    )

    best_miou = -1.0
    history = []

    global_iter = 0

    pbar = tqdm(
        total=max_iters,
        desc="training",
        dynamic_ncols=True,
    )

    while global_iter < max_iters:

        model.train()

        for batch in train_loader:

            if global_iter >= max_iters:
                break

            current_lr = compute_poly_lr(
                base_lr=base_lr,
                current_iter=global_iter,
                max_iters=max_iters,
                warmup_iters=warmup_iters,
                warmup_start_factor=warmup_start_factor,
                power=poly_power,
            )

            set_optimizer_lr(
                optimizer,
                current_lr,
            )

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast(
                enabled=amp_enabled
            ):

                out = model(images)

                logits = out["seg_logits"]

                ce = F.cross_entropy(
                    logits,
                    labels,
                    ignore_index=ignore_index,
                )

                dice = multiclass_dice_loss(
                    logits,
                    labels,
                    num_classes,
                    ignore_index,
                )

                loss = (
                    ce
                    + lambda_dice * dice
                )

                if method == "focal":
                    loss = (
                        focal_loss(
                            logits,
                            labels,
                            gamma=float(
                                cfg["loss"].get(
                                    "focal_gamma",
                                    2.0,
                                )
                            ),
                            ignore_index=ignore_index,
                        )
                        + lambda_dice * dice
                    )

                if (
                    method
                    in {"loss_topk", "ohem"}
                    and global_iter
                    >= topk_start_iter
                ):
                    loss = (
                        loss
                        + lambda_hard
                        * topk_cross_entropy(
                            logits,
                            labels,
                            None,
                            topk_ratio,
                            ignore_index,
                        )
                    )

                if (
                    method == "msp_topk"
                    and global_iter
                    >= topk_start_iter
                ):
                    probs = torch.softmax(
                        logits,
                        dim=1,
                    )

                    score = (
                        1.0
                        - probs.max(
                            dim=1,
                            keepdim=True,
                        ).values
                    )

                    loss = (
                        loss
                        + lambda_hard
                        * topk_cross_entropy(
                            logits,
                            labels,
                            score,
                            topk_ratio,
                            ignore_index,
                        )
                    )

                if (
                    method == "entropy_topk"
                    and global_iter
                    >= topk_start_iter
                ):
                    score = entropy_uncertainty(
                        logits
                    )

                    loss = (
                        loss
                        + lambda_hard
                        * topk_cross_entropy(
                            logits,
                            labels,
                            score,
                            topk_ratio,
                            ignore_index,
                        )
                    )

                if method in {
                    "suf_hrl",
                    "suf-hrl",
                    "residual",
                    "uncertainty_topk",
                }:

                    unc_map = out["unc_map"]

                    local_loss, _ = (
                        local_loss_fn(
                            logits,
                            unc_map,
                            labels,
                        )
                    )

                    boundary_loss, _ = (
                        boundary_loss_fn(
                            unc_map,
                            labels,
                        )
                    )

                    loss = (
                        loss
                        + lambda_local
                        * local_loss
                        + lambda_boundary
                        * boundary_loss
                    )

                    if (
                        global_iter
                        >= topk_start_iter
                    ):
                        loss = (
                            loss
                            + lambda_hard
                            * topk_cross_entropy(
                                logits,
                                labels,
                                unc_map,
                                topk_ratio,
                                ignore_index,
                            )
                        )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_iter += 1
            pbar.update(1)

            if (
                global_iter % log_interval == 0
                or global_iter == 1
            ):
                pbar.set_postfix(
                    loss=f"{float(loss.detach()):.4f}",
                    lr=f"{current_lr:.3e}",
                )

            should_validate = (
                global_iter % val_interval == 0
                or global_iter == max_iters
            )

            if should_validate:

                metrics = validate(
                    model,
                    val_loader,
                    num_classes,
                    ignore_index,
                    device,
                )

                row = {
                    "iteration": global_iter,
                    "train_loss": float(
                        loss.detach().cpu()
                    ),
                    "lr": current_lr,
                    **metrics,
                }

                history.append(row)

                print()
                print(
                    json.dumps(
                        row,
                        indent=2,
                    )
                )

                if metrics["mIoU"] > best_miou:

                    best_miou = metrics["mIoU"]

                    # Save a clean state_dict so that the public
                    # evaluation scripts can load it with strict=True.
                    torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pth")

                    save_metadata(
                        checkpoint_dir
                        / "best_meta.json",
                        iteration=global_iter,
                        metrics=metrics,
                        cfg=cfg,
                    )

                model.train()

                with (
                    out_dir
                    / "history.json"
                ).open(
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        history,
                        f,
                        indent=2,
                    )

    pbar.close()

    # Save final weights in the same clean format.
    torch.save(
        model.state_dict(),
        checkpoint_dir / "last.pth",
    )

    print(
        f"[DONE] training finished at "
        f"{global_iter} iterations"
    )

    print(
        f"[DONE] best mIoU = "
        f"{best_miou:.6f}"
    )


if __name__ == "__main__":
    main()
