#!/usr/bin/env python3

import argparse
import time
import yaml

import torch
import torch.nn.functional as F

from sufh_rl.models import build_model
from sufh_rl.losses import (
    multiclass_dice_loss,
    LocalAlignmentLossMapMultiClass,
    BoundaryConcentrationLossMapMultiClass,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def ohem_loss(logits, labels, prob_thresh=0.7, min_pixels=256, ignore_index=255):
    per_ce = F.cross_entropy(
        logits, labels,
        ignore_index=ignore_index,
        reduction="none",
    )

    valid = labels != ignore_index

    with torch.no_grad():
        prob = torch.softmax(logits, dim=1)
        safe = labels.clone()
        safe[~valid] = 0
        p_gt = prob.gather(1, safe.unsqueeze(1)).squeeze(1)
        hard = valid & (p_gt < prob_thresh)

    hard_vals = per_ce[hard]

    if hard_vals.numel() >= min_pixels:
        return hard_vals.mean()

    vals = per_ce[valid]
    if vals.numel() == 0:
        return logits.sum() * 0.0

    k = min(max(min_pixels, hard_vals.numel()), vals.numel())
    return torch.topk(vals, k=k, largest=True).values.mean()


def extract_logits(out):
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        if "seg_logits" in out:
            return out["seg_logits"]
        if "logits" in out:
            return out["logits"]
    if hasattr(out, "logits"):
        return out.logits
    raise RuntimeError(type(out))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_forward(model, x, warmup, runs):
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        sync()

        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        sync()

    return (time.perf_counter() - t0) * 1000.0 / runs


def benchmark_train(
    variant,
    model,
    x,
    y,
    cfg,
    local_fn,
    boundary_fn,
    warmup,
    runs,
):
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-5,
        weight_decay=0.01,
    )

    lambda_dice = float(cfg["loss"].get("lambda_dice", 1.0))
    lambda_local = float(cfg["loss"].get("lambda_local", 0.1))
    lambda_bound = float(cfg["loss"].get("lambda_bound", 0.05))
    lambda_hard = float(cfg["loss"].get("lambda_hard", 0.4))
    topk_ratio = float(cfg["loss"].get("topk_ratio", 0.05))
    ignore_index = int(cfg["dataset"].get("ignore_index", 255))
    num_classes = int(cfg["dataset"]["num_classes"])

    def one_step():
        optimizer.zero_grad(set_to_none=True)

        out = model(x)
        logits = extract_logits(out)

        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=y.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        ce = F.cross_entropy(
            logits,
            y,
            ignore_index=ignore_index,
        )
        dice = multiclass_dice_loss(
            logits,
            y,
            num_classes,
            ignore_index,
        )

        loss = ce + lambda_dice * dice

        if variant == "ohem":
            loss = loss + lambda_hard * ohem_loss(
                logits,
                y,
                ignore_index=ignore_index,
            )

        elif variant == "suf_hrl":
            unc = out["unc_map"]

            local, _ = local_fn(
                logits,
                unc,
                y,
            )
            bound, _ = boundary_fn(
                unc,
                y,
            )

            valid = y != ignore_index
            per_ce = F.cross_entropy(
                logits,
                y,
                ignore_index=ignore_index,
                reduction="none",
            )

            score = unc.detach()
            if score.ndim == 4:
                score = score[:, 0]

            if score.shape[-2:] != y.shape[-2:]:
                score = F.interpolate(
                    score.unsqueeze(1),
                    size=y.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]

            vals = per_ce[valid]
            scores = score[valid]

            k = max(256, int(vals.numel() * topk_ratio))
            k = min(k, vals.numel())

            idx = torch.topk(
                scores,
                k=k,
                largest=True,
            ).indices

            hard = vals[idx].mean()

            loss = (
                loss
                + lambda_local * local
                + lambda_bound * bound
                + lambda_hard * hard
            )

        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()

    sync()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()

    for _ in range(runs):
        one_step()

    sync()

    elapsed = (time.perf_counter() - t0) * 1000.0 / runs
    mem = torch.cuda.max_memory_allocated() / 1024**3

    return elapsed, mem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/potsdam.yaml")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--runs", type=int, default=30)
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    num_classes = int(cfg["dataset"]["num_classes"])
    h = w = 512

    x = torch.randn(
        args.batch_size, 3, h, w,
        device=device,
    )

    y = torch.randint(
        0,
        num_classes,
        (args.batch_size, h, w),
        device=device,
    )

    local_fn = LocalAlignmentLossMapMultiClass(
        sigmas=tuple(cfg["loss"].get(
            "local_sigmas", [1.0, 3.0, 5.0]
        )),
        weights=tuple(cfg["loss"].get(
            "local_weights", [0.5, 0.3, 0.2]
        )),
        ignore_index=int(
            cfg["dataset"].get("ignore_index", 255)
        ),
    )

    boundary_fn = BoundaryConcentrationLossMapMultiClass(
        band_width=int(
            cfg["loss"].get("boundary_width", 3)
        ),
        margin=float(
            cfg["loss"].get("boundary_margin", 0.18)
        ),
        ignore_index=int(
            cfg["dataset"].get("ignore_index", 255)
        ),
    )

    variants = [
        ("baseline", "baseline"),
        ("ohem", "baseline"),
        ("suf_hrl", "suf_hrl"),
    ]

    print(
        "method,params_M,train_ms_iter,"
        "peak_mem_GB,infer_ms_batch"
    )

    for name, method in variants:
        torch.cuda.empty_cache()

        model = build_model(
            method=method,
            num_classes=num_classes,
            backbone_name=cfg["model"].get(
                "backbone", "nvidia/mit-b2"
            ),
            fuse_dim=int(
                cfg["model"].get("fuse_dim", 256)
            ),
            residual_scale=float(
                cfg["model"].get(
                    "residual_scale", 0.15
                )
            ),
            delta_hidden_dims=tuple(
                cfg["model"].get(
                    "delta_hidden_dims", []
                )
            ),
        ).to(device)

        params = count_params(model)

        infer_ms = benchmark_forward(
            model,
            x,
            args.warmup,
            args.runs,
        )

        train_ms, mem = benchmark_train(
            name,
            model,
            x,
            y,
            cfg,
            local_fn,
            boundary_fn,
            args.warmup,
            args.runs,
        )

        print(
            f"{name},{params:.3f},"
            f"{train_ms:.2f},{mem:.3f},"
            f"{infer_ms:.2f}"
        )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
