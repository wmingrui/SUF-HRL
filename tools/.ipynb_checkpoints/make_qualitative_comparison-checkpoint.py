#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate qualitative semantic-segmentation comparisons.

Official release usage:
    Image | GT | SUF-HRL | Error

Optional comparison usage:
    Image | GT | Baseline | Loss-topk | SUF-HRL | Error diff.

The official SUF-HRL checkpoint is the only required model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------
# Always import this repository checkout.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from sufh_rl.datasets import (  # noqa: E402
    LoveDAMulticlassDataset,
    PotsdamMulticlassDataset,
    VaihingenMulticlassDataset,
)
from sufh_rl.models import build_model  # noqa: E402
from sufh_rl.utils import load_config  # noqa: E402


DATASET_SPECS = {
    "potsdam": {
        "dataset_cls": PotsdamMulticlassDataset,
        "num_classes": 6,
        "delta_hidden_dims": (),
        "palette": [
            [255, 255, 255],  # impervious surface
            [0, 0, 255],      # building
            [0, 255, 255],    # low vegetation
            [0, 255, 0],      # tree
            [255, 255, 0],    # car
            [255, 0, 0],      # clutter/background
        ],
    },
    "vaihingen": {
        "dataset_cls": VaihingenMulticlassDataset,
        "num_classes": 6,
        "delta_hidden_dims": (128, 128),
        "palette": [
            [255, 255, 255],
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ],
    },
    "loveda": {
        "dataset_cls": LoveDAMulticlassDataset,
        "num_classes": 7,
        "delta_hidden_dims": (128, 64),
        "palette": [
            [0, 0, 0],        # background
            [255, 0, 0],      # building
            [255, 255, 255],  # road
            [0, 0, 255],      # water
            [255, 255, 0],    # barren
            [0, 255, 0],      # forest
            [0, 255, 255],    # agriculture
        ],
    },
}


IMAGENET_MEAN = np.asarray(
    [0.485, 0.456, 0.406], dtype=np.float32
)
IMAGENET_STD = np.asarray(
    [0.229, 0.224, 0.225], dtype=np.float32
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SUF-HRL qualitative comparison figures."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["potsdam", "vaihingen", "loveda"],
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Default: configs/<dataset>.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Released SUF-HRL checkpoint.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default=None,
        help="Optional baseline checkpoint.",
    )
    parser.add_argument(
        "--loss-topk-checkpoint",
        default=None,
        help="Optional loss-topk checkpoint.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-cases", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="outputs/qualitative",
    )
    parser.add_argument("--cell-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")

    return parser.parse_args()


def extract_logits(out):
    """Extract segmentation logits from supported public model outputs."""
    if torch.is_tensor(out):
        return out

    if isinstance(out, dict):
        if "seg_logits" in out:
            return out["seg_logits"]
        if "logits" in out:
            return out["logits"]

    if hasattr(out, "logits"):
        return out.logits

    if isinstance(out, (tuple, list)) and out:
        if torch.is_tensor(out[0]):
            return out[0]

    raise RuntimeError(
        f"Cannot extract segmentation logits from {type(out)}"
    )


def make_error_map(pred, gt, ignore_index=255):
    """
    White = correct
    Red   = incorrect
    Gray  = ignored
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)

    rgb = np.full((*gt.shape, 3), 255, dtype=np.uint8)

    ignore = gt == ignore_index
    wrong = (pred != gt) & (~ignore)

    rgb[wrong] = [255, 0, 0]
    rgb[ignore] = [220, 220, 220]

    return rgb


def make_error_overlay(image_rgb, pred, gt, ignore_index=255):
    """Overlay segmentation errors in red on the original RGB image."""
    out = image_rgb.copy()

    valid = gt != ignore_index
    wrong = (pred != gt) & valid

    out[wrong] = (
        0.30 * out[wrong].astype(np.float32)
        + 0.70 * np.asarray([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)

    return out


def make_error_difference_overlay(
    image_rgb,
    baseline_pred,
    ours_pred,
    gt,
    ignore_index=255,
):
    """
    Green  = Baseline wrong, SUF-HRL correct
    Red    = Baseline correct, SUF-HRL wrong
    Yellow = Both wrong but predictions differ
    """
    valid = gt != ignore_index

    base_wrong = (baseline_pred != gt) & valid
    ours_wrong = (ours_pred != gt) & valid

    fixed = base_wrong & (~ours_wrong)
    new_error = (~base_wrong) & ours_wrong
    changed_wrong = (
        base_wrong
        & ours_wrong
        & (baseline_pred != ours_pred)
    )

    out = image_rgb.copy().astype(np.float32)

    green = np.asarray([0, 220, 0], dtype=np.float32)
    red = np.asarray([255, 0, 0], dtype=np.float32)
    yellow = np.asarray([255, 220, 0], dtype=np.float32)

    out[fixed] = 0.25 * out[fixed] + 0.75 * green
    out[new_error] = 0.25 * out[new_error] + 0.75 * red
    out[changed_wrong] = (
        0.25 * out[changed_wrong] + 0.75 * yellow
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def colorize_mask(mask, palette, ignore_index=255):
    mask = np.asarray(mask)
    palette = np.asarray(palette, dtype=np.uint8)

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)

    valid = (
        (mask >= 0)
        & (mask < len(palette))
        & (mask != ignore_index)
    )

    rgb[valid] = palette[mask[valid]]
    rgb[~valid] = [0, 0, 0]

    return rgb


def tensor_to_rgb(image):
    """Undo ImageNet normalization used by public datasets."""
    arr = (
        image.detach()
        .cpu()
        .float()
        .permute(1, 2, 0)
        .numpy()
    )

    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)

    return np.round(arr * 255.0).astype(np.uint8)


def unwrap_checkpoint(obj):
    """
    Official release checkpoints are pure state_dicts.
    Wrapper support is kept for optional historical comparison models.
    """
    if not isinstance(obj, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]

    return obj


def build_public_model(dataset, cfg, method="suf_hrl"):
    spec = DATASET_SPECS[dataset]
    mcfg = cfg["model"]

    return build_model(
        method=method,
        num_classes=spec["num_classes"],
        backbone_name=mcfg.get(
            "backbone", "nvidia/mit-b2"
        ),
        fuse_dim=mcfg.get("fuse_dim", 256),
        residual_scale=mcfg.get(
            "residual_scale", 0.15
        ),
        delta_hidden_dims=tuple(
            spec["delta_hidden_dims"]
        ),
    )


def load_model(
    dataset,
    cfg,
    checkpoint,
    device,
    role="SUF-HRL",
    method="suf_hrl",
):
    model = build_public_model(
        dataset=dataset,
        cfg=cfg,
        method=method,
    )

    state = torch.load(
        checkpoint,
        map_location="cpu",
    )
    state = unwrap_checkpoint(state)

    model.load_state_dict(state, strict=True)

    print(f"[CKPT] {role} strict=True: PASS")
    print(f"       {checkpoint}")

    model = model.to(device)
    model.eval()

    return model


def build_dataset(dataset, cfg, split):
    spec = DATASET_SPECS[dataset]
    dcfg = cfg["dataset"]

    root = Path(dcfg["root"])

    # Match the released evaluator:
    # LoveDA uses original-resolution validation images.
    if dataset == "loveda":
        crop_size = None
    else:
        crop_size = dcfg.get("crop_size", 512)

    dataset_cls = spec["dataset_cls"]

    ds = dataset_cls(
        image_dir=str(root / dcfg["image_dir"]),
        label_dir=str(root / dcfg["label_dir"]),
        split_file=str(
            root / "splits" / f"{split}.txt"
        ),
        crop_size=crop_size,
        mode="val",
        normalize=True,
        ignore_index=dcfg.get(
            "ignore_index", 255
        ),
    )

    return ds


@torch.no_grad()
def predict(model, image, output_hw, device):
    x = image.unsqueeze(0).to(device)

    out = model(x)
    logits = extract_logits(out)

    if tuple(logits.shape[-2:]) != tuple(output_hw):
        logits = F.interpolate(
            logits,
            size=output_hw,
            mode="bilinear",
            align_corners=False,
        )

    pred = (
        logits.argmax(dim=1)[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    return pred


def get_font(size=22, bold=True):
    candidates = []

    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)

    return ImageFont.load_default()


def resize_cell(arr, size, nearest=False):
    im = Image.fromarray(arr.astype(np.uint8))

    resample = (
        Image.Resampling.NEAREST
        if nearest
        else Image.Resampling.BILINEAR
    )

    return im.resize(
        (size, size),
        resample=resample,
    )


def make_grid(rows, titles, output_path, cell_size=256):
    gap = 6
    bottom_title_h = 48
    outer = 6

    nrows = len(rows)
    ncols = len(titles)

    width = (
        2 * outer
        + ncols * cell_size
        + (ncols - 1) * gap
    )
    height = (
        2 * outer
        + nrows * cell_size
        + max(0, nrows - 1) * gap
        + bottom_title_h
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    for r, cells in enumerate(rows):
        y = outer + r * (cell_size + gap)

        for c, cell in enumerate(cells):
            x = outer + c * (cell_size + gap)
            canvas.paste(cell, (x, y))

    draw = ImageDraw.Draw(canvas)
    font = get_font(size=22, bold=True)

    title_y = (
        outer
        + nrows * cell_size
        + max(0, nrows - 1) * gap
        + 8
    )

    for c, title in enumerate(titles):
        x0 = outer + c * (cell_size + gap)

        try:
            box = draw.textbbox(
                (0, 0),
                title,
                font=font,
            )
            tw = box[2] - box[0]
        except Exception:
            tw = len(title) * 12

        tx = x0 + (cell_size - tw) / 2

        draw.text(
            (tx, title_y),
            title,
            fill=(0, 0, 0),
            font=font,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(output_path)

    print(f"[SAVE] {output_path}")


def main():
    args = parse_args()

    config_path = (
        args.config
        if args.config is not None
        else f"configs/{args.dataset}.yaml"
    )

    cfg = load_config(config_path)

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    spec = DATASET_SPECS[args.dataset]

    print(f"[DATASET] {args.dataset}")
    print(f"[CONFIG] {config_path}")
    print(f"[DEVICE] {device}")

    ds = build_dataset(
        dataset=args.dataset,
        cfg=cfg,
        split=args.split,
    )

    print(
        f"[SPLIT] {args.split}: "
        f"{len(ds)} samples"
    )

    ours_model = load_model(
        dataset=args.dataset,
        cfg=cfg,
        checkpoint=args.checkpoint,
        device=device,
        role="SUF-HRL",
        method="suf_hrl",
    )

    baseline_model = None
    loss_topk_model = None

    if args.baseline_checkpoint:
        baseline_model = load_model(
            dataset=args.dataset,
            cfg=cfg,
            checkpoint=args.baseline_checkpoint,
            device=device,
            role="Baseline",
            method="baseline",
        )

    if args.loss_topk_checkpoint:
        # Loss-topk checkpoints from the residual/SUF family
        # normally share the released SUF-HRL architecture.
        loss_topk_model = load_model(
            dataset=args.dataset,
            cfg=cfg,
            checkpoint=args.loss_topk_checkpoint,
            device=device,
            role="Loss-topk",
            method="suf_hrl",
        )

    start = max(0, int(args.start_index))
    stop = min(
        len(ds),
        start + max(1, int(args.num_cases)),
    )

    if start >= len(ds):
        raise IndexError(
            f"start-index={start} but dataset "
            f"contains only {len(ds)} samples."
        )

    rows = []
    metadata = []

    for idx in range(start, stop):
        sample = ds[idx]

        image = sample["image"]
        label = sample["label"]

        rgb = tensor_to_rgb(image)

        gt = (
            label.detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )

        h, w = gt.shape

        ours_pred = predict(
            ours_model,
            image,
            (h, w),
            device,
        )

        baseline_pred = None
        loss_topk_pred = None

        if baseline_model is not None:
            baseline_pred = predict(
                baseline_model,
                image,
                (h, w),
                device,
            )

        if loss_topk_model is not None:
            loss_topk_pred = predict(
                loss_topk_model,
                image,
                (h, w),
                device,
            )

        cells = [
            resize_cell(
                rgb,
                args.cell_size,
                nearest=False,
            ),
            resize_cell(
                colorize_mask(
                    gt,
                    spec["palette"],
                ),
                args.cell_size,
                nearest=True,
            ),
        ]

        titles = [
            "Image",
            "GT",
        ]

        if baseline_pred is not None:
            cells.append(
                resize_cell(
                    colorize_mask(
                        baseline_pred,
                        spec["palette"],
                    ),
                    args.cell_size,
                    nearest=True,
                )
            )
            titles.append("Baseline")

        if loss_topk_pred is not None:
            cells.append(
                resize_cell(
                    colorize_mask(
                        loss_topk_pred,
                        spec["palette"],
                    ),
                    args.cell_size,
                    nearest=True,
                )
            )
            titles.append("Loss-topk")

        cells.append(
            resize_cell(
                colorize_mask(
                    ours_pred,
                    spec["palette"],
                ),
                args.cell_size,
                nearest=True,
            )
        )
        titles.append("SUF-HRL")

        if baseline_pred is not None:
            diff = make_error_difference_overlay(
                rgb,
                baseline_pred,
                ours_pred,
                gt,
            )
            error_title = "Error diff."
        else:
            diff = make_error_overlay(
                rgb,
                ours_pred,
                gt,
            )
            error_title = "Error"

        cells.append(
            resize_cell(
                diff,
                args.cell_size,
                nearest=False,
            )
        )
        titles.append(error_title)

        rows.append(cells)

        valid = gt != cfg["dataset"].get(
            "ignore_index", 255
        )

        if valid.any():
            ours_error = float(
                (ours_pred[valid] != gt[valid]).mean()
            )
        else:
            ours_error = None

        metadata.append({
            "dataset_index": idx,
            "ours_error_rate": ours_error,
            "height": int(h),
            "width": int(w),
        })

        print(
            f"[CASE] index={idx} "
            f"size={h}x{w} "
            f"error={ours_error}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_png = (
        output_dir
        / f"qualitative_{args.dataset}_{args.split}.png"
    )

    make_grid(
        rows=rows,
        titles=titles,
        output_path=output_png,
        cell_size=args.cell_size,
    )

    metadata_path = (
        output_dir
        / f"qualitative_{args.dataset}_{args.split}.json"
    )

    metadata_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "checkpoint": str(
                    args.checkpoint
                ),
                "baseline_checkpoint":
                    args.baseline_checkpoint,
                "loss_topk_checkpoint":
                    args.loss_topk_checkpoint,
                "cases": metadata,
                "output_png": str(output_png),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[SAVE] {metadata_path}")
    print("[DONE] qualitative comparison generated.")


if __name__ == "__main__":
    main()
