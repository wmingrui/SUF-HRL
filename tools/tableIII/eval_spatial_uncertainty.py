
import argparse
import csv
import importlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter


EPS = 1e-8


def trapz_compat(y, x):
    """Compatible trapezoidal integration for different NumPy versions."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))

    if len(y) < 2:
        return 0.0

    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["potsdam", "vaihingen"],
        help="Dataset name. Current version supports Potsdam and Vaihingen.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline", "residual", "loss_topk", "uncertainty_topk"],
        help="Model type used to build the network.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])

    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save spatial uncertainty evaluation outputs.",
    )

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--ignore_index", type=int, default=255)

    parser.add_argument(
        "--eval_class_ids",
        type=int,
        nargs="*",
        default=None,
        help="Class ids used for evaluation. For Vaihingen usually use 0 1 2 3 4.",
    )

    parser.add_argument(
        "--boundary_radius",
        type=int,
        default=5,
        help="Boundary dilation radius used for BFUR.",
    )

    parser.add_argument(
        "--distance_bins",
        type=float,
        nargs="+",
        default=[0, 3, 5, 7, 15, 1e9],
        help="Distance bin edges for DSCG.",
    )

    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 4.0],
        help="Gaussian smoothing scales for MSAD.",
    )

    parser.add_argument(
        "--unc_norm",
        type=str,
        default="clip",
        choices=["clip", "minmax", "none"],
        help=(
            "How to normalize uncertainty maps. "
            "clip keeps values in [0,1]; minmax normalizes each map; none uses raw values."
        ),
    )

    parser.add_argument(
        "--trad_bins",
        type=int,
        default=1000,
        help="Number of bins for binned Error AUROC, Error AUPR, and UCE.",
    )

    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def add_project_paths():
    sys.path.insert(0, "/root/autodl-tmp/scripts_new")
    sys.path.insert(0, "/root/autodl-tmp/script_Vaihingen")
    sys.path.insert(0, "/root/autodl-tmp")


def filter_kwargs_for_class(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            valid[k] = v
    return valid


def build_dataset(args):
    if args.dataset == "potsdam":
        data_root = Path("/root/autodl-tmp/data/potsdam")
        module_name = "new_02_potsdam_multiclass_dataset"
        class_name = "PotsdamMulticlassDataset"
    else:
        data_root = Path("/root/autodl-tmp/data/vaihingen")
        module_name = "new_3_vaihingen_multiclass_dataset"
        class_name = "VaihingenMulticlassDataset"

    image_dir = data_root / "processed_multiclass/images"
    label_dir = data_root / "processed_multiclass/labels"
    split_file = data_root / f"splits/{args.split}.txt"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    module = importlib.import_module(module_name)
    dataset_cls = getattr(module, class_name)

    common_kwargs = {
        "image_dir": image_dir,
        "label_dir": label_dir,
        "split_file": split_file,
        "root_dir": data_root,
        "data_root": data_root,
        "root": data_root,
        "split": args.split,
        "crop_size": args.crop_size,
        "ignore_index": args.ignore_index,
    }

    try:
        kwargs = filter_kwargs_for_class(dataset_cls, common_kwargs)
        dataset = dataset_cls(**kwargs)
    except Exception as e1:
        print("[WARN] keyword init failed:", repr(e1))
        print("[INFO] trying positional init: image_dir, label_dir, split_file, crop_size")
        try:
            dataset = dataset_cls(image_dir, label_dir, split_file, args.crop_size)
        except Exception as e2:
            print("[WARN] positional init with crop_size failed:", repr(e2))
            print("[INFO] trying positional init: image_dir, label_dir, split_file")
            dataset = dataset_cls(image_dir, label_dir, split_file)

    return dataset


def build_model(args):
    backbone_name = "nvidia/mit-b2"
    fuse_dim = 256
    residual_scale = 0.15

    if args.dataset == "potsdam":
        if args.method == "baseline":
            module_name = "new_23_train_segformer_multiclass_baseline_b2_512"
            class_name = "SegFormerBaselineMulticlass"
        else:
            module_name = "new_04_segformer_residual_uncertainty_multiclass"
            class_name = "SegFormerResidualUncertaintyMulticlass"
    else:
        if args.method == "baseline":
            module_name = "new_4_train_segformer_vaihingen_baseline_b2_512"
            class_name = "SegFormerBaselineMulticlass"
        elif args.method == "residual":
            module_name = "new_5_train_segformer_residual_uncertainty_vaihingen_b2_512"
            class_name = "SegFormerResidualUncertaintyMulticlass"
        elif args.method == "loss_topk":
            module_name = "new_6_train_segformer_residual_uncertainty_loss_topk_vaihingen_b2_512"
            class_name = "SegFormerResidualUncertaintyMulticlass"
        else:
            module_name = "new_7_train_segformer_residual_uncertainty_topk_vaihingen_b2_512"
            class_name = "SegFormerResidualUncertaintyMulticlass"

    module = importlib.import_module(module_name)
    model_cls = getattr(module, class_name)

    candidate_kwargs = {
        "num_classes": args.num_classes,
        "backbone_name": backbone_name,
        "fuse_dim": fuse_dim,
        "residual_scale": residual_scale,
    }

    kwargs = filter_kwargs_for_class(model_cls, candidate_kwargs)

    print("[INFO] build model:", module_name, class_name)
    print("[INFO] model kwargs:", kwargs)

    model = model_cls(**kwargs)
    return model


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in [
            "model_state_dict",
            "state_dict",
            "model",
            "net",
            "network",
            "ema_state_dict",
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]

        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt

    return ckpt


def clean_state_dict_keys(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "model."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_sd[nk] = v
    return new_sd


def load_checkpoint(model, ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("[INFO] loading checkpoint:", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = extract_state_dict(ckpt)
    state_dict = clean_state_dict_keys(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        print("[INFO] checkpoint loaded with strict=True")
    except Exception as e:
        print("[WARN] strict=True load failed:", repr(e))
        msg = model.load_state_dict(state_dict, strict=False)
        print("[WARN] checkpoint loaded with strict=False")
        print("[WARN] missing keys:", len(msg.missing_keys))
        print("[WARN] unexpected keys:", len(msg.unexpected_keys))
        if len(msg.missing_keys) > 0:
            print("[WARN] first missing keys:", msg.missing_keys[:10])
        if len(msg.unexpected_keys) > 0:
            print("[WARN] first unexpected keys:", msg.unexpected_keys[:10])

    return model


def get_batch_image_label(batch):
    if isinstance(batch, dict):
        image = None
        label = None

        for k in ["image", "img", "images"]:
            if k in batch:
                image = batch[k]
                break

        for k in ["label", "mask", "gt", "labels", "masks"]:
            if k in batch:
                label = batch[k]
                break

        if image is None or label is None:
            raise KeyError(f"Cannot find image/label in batch keys: {list(batch.keys())}")

        return image, label

    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def extract_logits(outputs):
    if isinstance(outputs, dict):
        for k in ["seg_logits", "logits", "out", "pred", "prediction"]:
            if k in outputs:
                return outputs[k]
        raise KeyError(f"Cannot find logits in output keys: {list(outputs.keys())}")

    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, (list, tuple)):
        return outputs[0]

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def extract_learned_uncertainty(outputs):
    """
    Try to extract learned uncertainty from different output formats.
    Return None if the model does not expose uncertainty.
    """
    if isinstance(outputs, dict):
        candidate_keys = [
            "uncertainty",
            "unc",
            "unc_map",
            "uncertainty_map",
            "learned_uncertainty",
            "learned_unc",
            "final_uncertainty",
            "u",
            "U",
        ]
        for k in candidate_keys:
            if k in outputs:
                return outputs[k]
        return None

    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
        if torch.is_tensor(outputs[1]):
            return outputs[1]

    return None


def entropy_from_probs(probs, eps=1e-8):
    c = probs.shape[1]
    entropy = -(probs * torch.log(probs + eps)).sum(dim=1)
    entropy = entropy / np.log(c)
    return entropy

def make_valid_mask(label, ignore_index=255, eval_class_ids=None):
    valid = label != ignore_index

    if eval_class_ids is not None and len(eval_class_ids) > 0:
        eval_ids = np.array(eval_class_ids, dtype=np.int64)
        valid &= np.isin(label, eval_ids)

    return valid


def make_semantic_boundary_from_valid(label, valid):
    h, w = label.shape
    boundary = np.zeros((h, w), dtype=bool)

    diff_h = (label[:, 1:] != label[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundary[:, 1:] |= diff_h
    boundary[:, :-1] |= diff_h

    diff_v = (label[1:, :] != label[:-1, :]) & valid[1:, :] & valid[:-1, :]
    boundary[1:, :] |= diff_v
    boundary[:-1, :] |= diff_v

    boundary &= valid
    return boundary


def boundary_band_from_valid(label, valid, radius=5):
    boundary = make_semantic_boundary_from_valid(label, valid)

    if radius <= 0:
        band = boundary
    else:
        structure = np.ones((3, 3), dtype=bool)
        band = boundary.copy()
        for _ in range(radius):
            band = binary_dilation(band, structure=structure)

    band &= valid
    return band

def prepare_uncertainty(u, mode="clip"):
    u = u.astype(np.float32)

    if mode == "none":
        return u

    if mode == "clip":
        return np.clip(u, 0.0, 1.0).astype(np.float32)

    if mode == "minmax":
        u_min = np.nanmin(u)
        u_max = np.nanmax(u)
        if u_max - u_min < EPS:
            return np.zeros_like(u, dtype=np.float32)
        return ((u - u_min) / (u_max - u_min + EPS)).astype(np.float32)

    raise ValueError(f"Unknown uncertainty normalization mode: {mode}")

def init_traditional_uncertainty_accumulator(num_bins=1000):
    return {
        "num_bins": int(num_bins),
        "pos_hist": np.zeros(num_bins, dtype=np.float64),
        "neg_hist": np.zeros(num_bins, dtype=np.float64),
        "bin_count": np.zeros(num_bins, dtype=np.float64),
        "bin_unc_sum": np.zeros(num_bins, dtype=np.float64),
        "bin_err_sum": np.zeros(num_bins, dtype=np.float64),
        "num_valid_pixels": 0,
        "num_error_pixels": 0,
    }


def update_traditional_uncertainty_accumulator(
    acc,
    uncertainty,
    pred,
    label,
    ignore_index=255,
    eval_class_ids=None,
    unc_norm="clip",
):
    """
    Streaming update for Error AUROC, Error AUPR, and UCE.

    Positive class:
        error pixel, i.e., pred != label.

    Score:
        uncertainty.
    """
    valid = make_valid_mask(
        label=label,
        ignore_index=ignore_index,
        eval_class_ids=eval_class_ids,
    )

    if valid.sum() == 0:
        return acc

    u = prepare_uncertainty(uncertainty, mode=unc_norm)
    u_valid = u[valid].astype(np.float64)

    err_valid = (pred[valid] != label[valid]).astype(np.float64)

    num_bins = acc["num_bins"]

    bin_ids = np.floor(u_valid * num_bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    pos = err_valid == 1
    neg = ~pos

    if pos.any():
        acc["pos_hist"] += np.bincount(
            bin_ids[pos],
            minlength=num_bins,
        ).astype(np.float64)

    if neg.any():
        acc["neg_hist"] += np.bincount(
            bin_ids[neg],
            minlength=num_bins,
        ).astype(np.float64)

    acc["bin_count"] += np.bincount(
        bin_ids,
        minlength=num_bins,
    ).astype(np.float64)

    acc["bin_unc_sum"] += np.bincount(
        bin_ids,
        weights=u_valid,
        minlength=num_bins,
    ).astype(np.float64)

    acc["bin_err_sum"] += np.bincount(
        bin_ids,
        weights=err_valid,
        minlength=num_bins,
    ).astype(np.float64)

    acc["num_valid_pixels"] += int(valid.sum())
    acc["num_error_pixels"] += int(err_valid.sum())

    return acc


def compute_auc_from_hist(pos_hist, neg_hist):
    """
    Approximate AUROC from score histograms.
    Bins are ordered from low uncertainty to high uncertainty.
    """
    pos_hist = pos_hist.astype(np.float64)
    neg_hist = neg_hist.astype(np.float64)

    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    if total_pos <= 0 or total_neg <= 0:
        return None

    pos_desc = pos_hist[::-1]
    neg_desc = neg_hist[::-1]

    tps = np.cumsum(pos_desc)
    fps = np.cumsum(neg_desc)

    tpr = tps / (total_pos + EPS)
    fpr = fps / (total_neg + EPS)

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auroc = trapz_compat(tpr, fpr)
    return float(auroc)


def compute_aupr_from_hist(pos_hist, neg_hist):
    """
    Approximate AUPR / average precision from score histograms.
    Positive class is error pixel.
    """
    pos_hist = pos_hist.astype(np.float64)
    neg_hist = neg_hist.astype(np.float64)

    total_pos = pos_hist.sum()

    if total_pos <= 0:
        return None

    pos_desc = pos_hist[::-1]
    neg_desc = neg_hist[::-1]

    tp = np.cumsum(pos_desc)
    fp = np.cumsum(neg_desc)

    precision = tp / np.maximum(tp + fp, EPS)
    recall = tp / (total_pos + EPS)

    recall_prev = np.concatenate([[0.0], recall[:-1]])
    delta_recall = recall - recall_prev

    aupr = np.sum(precision * delta_recall)
    return float(aupr)


def compute_uce_from_bins(bin_count, bin_unc_sum, bin_err_sum):
    """
    UCE: Uncertainty Calibration Error.

    For each uncertainty bin:
        |mean uncertainty - empirical error rate|
    weighted by bin frequency.
    """
    total = bin_count.sum()

    if total <= 0:
        return None

    valid_bins = bin_count > 0

    mean_unc = np.zeros_like(bin_unc_sum, dtype=np.float64)
    mean_err = np.zeros_like(bin_err_sum, dtype=np.float64)

    mean_unc[valid_bins] = bin_unc_sum[valid_bins] / bin_count[valid_bins]
    mean_err[valid_bins] = bin_err_sum[valid_bins] / bin_count[valid_bins]

    weights = bin_count / (total + EPS)

    uce = np.sum(
        weights[valid_bins] * np.abs(mean_unc[valid_bins] - mean_err[valid_bins])
    )

    return float(uce)


def finalize_traditional_uncertainty_accumulator(acc):
    auroc = compute_auc_from_hist(acc["pos_hist"], acc["neg_hist"])
    aupr = compute_aupr_from_hist(acc["pos_hist"], acc["neg_hist"])
    uce = compute_uce_from_bins(
        acc["bin_count"],
        acc["bin_unc_sum"],
        acc["bin_err_sum"],
    )

    total_valid = int(acc["num_valid_pixels"])
    total_error = int(acc["num_error_pixels"])
    error_rate = float(total_error / max(total_valid, 1))

    return {
        "Error_AUROC": auroc,
        "Error_AUPR": aupr,
        "UCE": uce,
        "num_valid_pixels": total_valid,
        "num_error_pixels": total_error,
        "error_rate": error_rate,
    }

def compute_bfur(uncertainty, label, valid, radius=5, unc_norm="clip"):
    """
    BFUR: Boundary-Focused Uncertainty Ratio.

    BFUR = mean uncertainty inside boundary band /
           (mean uncertainty inside boundary band + mean uncertainty outside boundary band)

    Higher is better.
    """
    u = prepare_uncertainty(uncertainty, mode=unc_norm)

    band = boundary_band_from_valid(label, valid, radius=radius)
    non_band = valid & (~band)

    if band.sum() == 0 or non_band.sum() == 0:
        return np.nan

    mu_b = float(u[band].mean())
    mu_nb = float(u[non_band].mean())

    return float(mu_b / (mu_b + mu_nb + EPS))


def compute_dscg(
    uncertainty,
    pred,
    label,
    valid,
    band_edges=(0, 3, 5, 7, 15, 1e9),
    unc_norm="clip",
):
    """
    DSCG: Distance-Stratified Calibration Gap.

    For each distance band from semantic boundary:
        gap_k = |mean uncertainty - mean binary error|

    Lower is better.
    """
    u = prepare_uncertainty(uncertainty, mode=unc_norm)

    error = (pred != label).astype(np.float32)
    error[~valid] = 0.0

    boundary = make_semantic_boundary_from_valid(label, valid)
    if boundary.sum() == 0:
        return np.nan

    dist = distance_transform_edt(~boundary)
    dist[~valid] = np.inf

    gaps = []
    weights = []

    for k in range(len(band_edges) - 1):
        lo = band_edges[k]
        hi = band_edges[k + 1]

        if k == 0:
            mask = (dist >= lo) & (dist <= hi) & valid
        else:
            mask = (dist > lo) & (dist <= hi) & valid

        if mask.sum() == 0:
            continue

        mu_u = float(u[mask].mean())
        mu_e = float(error[mask].mean())
        gap = abs(mu_u - mu_e)

        mean_d = float(dist[mask].mean())
        weight = 1.0 / (mean_d + 1.0)

        gaps.append(gap)
        weights.append(weight)

    if len(gaps) == 0:
        return np.nan

    gaps = np.asarray(gaps, dtype=np.float32) / 2
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + EPS)

    return float(np.sum(weights * gaps))


def compute_msad(
    uncertainty,
    pred,
    label,
    valid,
    sigmas=(1.0, 2.0, 4.0),
    unc_norm="clip",
):
    """
    MSAD: Multi-Scale Spatial Alignment Deviation.

    Compare Gaussian-smoothed uncertainty map and binary error map
    at multiple spatial scales.

    Lower is better.
    """
    u = prepare_uncertainty(uncertainty, mode=unc_norm)

    error = (pred != label).astype(np.float32)

    u = u.copy()
    error = error.copy()

    u[~valid] = 0.0
    error[~valid] = 0.0

    values = []

    for sigma in sigmas:
        u_s = gaussian_filter(u, sigma=sigma)
        e_s = gaussian_filter(error, sigma=sigma)

        gap = np.abs(u_s - e_s)
        values.append(float(gap[valid].mean()))

    return float(np.mean(values))


def compute_spatial_metrics_for_one_map(
    uncertainty,
    pred,
    label,
    ignore_index=255,
    eval_class_ids=None,
    boundary_radius=5,
    band_edges=(0, 3, 5, 7, 15, 1e9),
    sigmas=(1.0, 2.0, 4.0),
    unc_norm="clip",
):
    valid = make_valid_mask(
        label=label,
        ignore_index=ignore_index,
        eval_class_ids=eval_class_ids,
    )

    if valid.sum() == 0:
        return {
            "BFUR": np.nan,
            "DSCG": np.nan,
            "MSAD": np.nan,
        }

    return {
        "BFUR": compute_bfur(
            uncertainty=uncertainty,
            label=label,
            valid=valid,
            radius=boundary_radius,
            unc_norm=unc_norm,
        ),
        "DSCG": compute_dscg(
            uncertainty=uncertainty,
            pred=pred,
            label=label,
            valid=valid,
            band_edges=band_edges,
            unc_norm=unc_norm,
        ),
        "MSAD": compute_msad(
            uncertainty=uncertainty,
            pred=pred,
            label=label,
            valid=valid,
            sigmas=sigmas,
            unc_norm=unc_norm,
        ),
    }

def summarize_spatial_records(records):
    summary = {}

    for source in ["MSP", "Entropy", "Learned"]:
        source_records = [r for r in records if r["source"] == source]
        if len(source_records) == 0:
            continue

        summary[source] = {}

        for metric in ["BFUR", "DSCG", "MSAD"]:
            arr = np.array(
                [r[metric] for r in source_records if not np.isnan(r[metric])],
                dtype=np.float32,
            )

            summary[source][metric] = {
                "mean": float(arr.mean()) if len(arr) > 0 else None,
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "num_images": int(len(arr)),
            }

    return summary


def format_value(x):
    if x is None:
        return "--"
    return f"{x:.4f}"


def format_mean_std(metric_dict):
    mean = metric_dict.get("mean", None)
    std = metric_dict.get("std", None)

    if mean is None:
        return "--"

    if std is None:
        return f"{mean:.4f}"

    return f"{mean:.4f} ± {std:.4f}"


def save_per_image_csv(records, csv_path):
    if len(records) == 0:
        return

    fieldnames = [
        "dataset",
        "method",
        "split",
        "image_index",
        "source",
        "BFUR",
        "DSCG",
        "MSAD",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow({
                "dataset": r.get("dataset", ""),
                "method": r.get("method", ""),
                "split": r.get("split", ""),
                "image_index": r.get("image_index", ""),
                "source": r.get("source", ""),
                "BFUR": r.get("BFUR", ""),
                "DSCG": r.get("DSCG", ""),
                "MSAD": r.get("MSAD", ""),
            })


def save_markdown_table(spatial_summary, traditional_summary, md_path):
    lines = []
    lines.append(
        "| Source | Error AUROC ↑ | Error AUPR ↑ | UCE ↓ | BFUR ↑ | DSCG ↓ | MSAD ↓ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for source in ["MSP", "Entropy", "Learned"]:
        if source not in spatial_summary:
            continue

        trad = traditional_summary.get(source, {})
        spatial = spatial_summary[source]

        row = [
            source,
            format_value(trad.get("Error_AUROC")),
            format_value(trad.get("Error_AUPR")),
            format_value(trad.get("UCE")),
            format_mean_std(spatial["BFUR"]),
            format_mean_std(spatial["DSCG"]),
            format_mean_std(spatial["MSAD"]),
        ]

        lines.append("| " + " | ".join(row) + " |")

    text = "\n".join(lines)
    md_path.write_text(text, encoding="utf-8")
    return text

def main():
    args = parse_args()
    add_project_paths()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda not available, fallback to cpu")
        args.device = "cpu"

    device = torch.device(args.device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_json = save_dir / "spatial_uncertainty_summary.json"
    out_csv = save_dir / "spatial_uncertainty_per_image.csv"
    out_md = save_dir / "spatial_uncertainty_table.md"

    print("=" * 100)
    print("Spatial + conventional uncertainty evaluation from checkpoint")
    print("dataset        :", args.dataset)
    print("method         :", args.method)
    print("split          :", args.split)
    print("checkpoint     :", args.checkpoint)
    print("save_dir       :", save_dir)
    print("out_json       :", out_json)
    print("out_csv        :", out_csv)
    print("out_md         :", out_md)
    print("device         :", device)
    print("eval classes   :", args.eval_class_ids)
    print("boundary radius:", args.boundary_radius)
    print("distance bins  :", args.distance_bins)
    print("sigmas         :", args.sigmas)
    print("unc norm       :", args.unc_norm)
    print("trad bins      :", args.trad_bins)
    print("=" * 100)

    dataset = build_dataset(args)
    print("[INFO] dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args)
    model = load_checkpoint(model, args.checkpoint, device)
    model.to(device)
    model.eval()

    spatial_records = []

    traditional_acc = {
        "MSP": init_traditional_uncertainty_accumulator(args.trad_bins),
        "Entropy": init_traditional_uncertainty_accumulator(args.trad_bins),
        "Learned": init_traditional_uncertainty_accumulator(args.trad_bins),
    }

    learned_available_count = 0
    learned_missing_count = 0
    first_output_printed = False
    global_image_index = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="uncertainty_eval"):
            images, labels = get_batch_image_label(batch)

            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            if labels.ndim == 4 and labels.shape[1] == 1:
                labels = labels[:, 0]

            outputs = model(images)

            if not first_output_printed:
                print("[DEBUG] output type:", type(outputs))
                if isinstance(outputs, dict):
                    print("[DEBUG] output keys:", list(outputs.keys()))
                elif isinstance(outputs, (list, tuple)):
                    print("[DEBUG] output tuple/list length:", len(outputs))
                    for i, x in enumerate(outputs):
                        if torch.is_tensor(x):
                            print(f"[DEBUG] outputs[{i}] shape:", tuple(x.shape))
                        else:
                            print(f"[DEBUG] outputs[{i}] type:", type(x))
                elif torch.is_tensor(outputs):
                    print("[DEBUG] output tensor shape:", tuple(outputs.shape))
                first_output_printed = True

            logits = extract_logits(outputs)
            learned_unc = extract_learned_uncertainty(outputs)

            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            msp_unc = 1.0 - torch.max(probs, dim=1).values
            entropy_unc = entropy_from_probs(probs)

            if learned_unc is not None:
                if learned_unc.ndim == 4 and learned_unc.shape[1] == 1:
                    learned_unc = learned_unc[:, 0]
                elif learned_unc.ndim == 3:
                    pass
                elif learned_unc.ndim == 4 and learned_unc.shape[1] != 1:
                    print(
                        "[WARN] learned_unc has multi-channel shape, ignored:",
                        tuple(learned_unc.shape),
                    )
                    learned_unc = None
                else:
                    print(
                        "[WARN] learned_unc has unsupported shape, ignored:",
                        tuple(learned_unc.shape),
                    )
                    learned_unc = None

            if learned_unc is not None:
                if learned_unc.shape[-2:] != labels.shape[-2:]:
                    learned_unc = F.interpolate(
                        learned_unc.unsqueeze(1),
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]

            preds_np = preds.detach().cpu().numpy().astype(np.int64)
            labels_np = labels.detach().cpu().numpy().astype(np.int64)
            msp_np = msp_unc.detach().cpu().numpy().astype(np.float32)
            entropy_np = entropy_unc.detach().cpu().numpy().astype(np.float32)

            if learned_unc is not None:
                learned_np = learned_unc.detach().cpu().numpy().astype(np.float32)
                learned_available_count += labels_np.shape[0]
            else:
                learned_np = None
                learned_missing_count += labels_np.shape[0]

            for b in range(labels_np.shape[0]):
                gt = labels_np[b]
                pred = preds_np[b]

                # MSP
                metric_msp = compute_spatial_metrics_for_one_map(
                    uncertainty=msp_np[b],
                    pred=pred,
                    label=gt,
                    ignore_index=args.ignore_index,
                    eval_class_ids=args.eval_class_ids,
                    boundary_radius=args.boundary_radius,
                    band_edges=args.distance_bins,
                    sigmas=args.sigmas,
                    unc_norm=args.unc_norm,
                )
                spatial_records.append({
                    "dataset": args.dataset,
                    "method": args.method,
                    "split": args.split,
                    "image_index": global_image_index,
                    "source": "MSP",
                    **metric_msp,
                })

                traditional_acc["MSP"] = update_traditional_uncertainty_accumulator(
                    traditional_acc["MSP"],
                    uncertainty=msp_np[b],
                    pred=pred,
                    label=gt,
                    ignore_index=args.ignore_index,
                    eval_class_ids=args.eval_class_ids,
                    unc_norm=args.unc_norm,
                )

                # Entropy
                metric_entropy = compute_spatial_metrics_for_one_map(
                    uncertainty=entropy_np[b],
                    pred=pred,
                    label=gt,
                    ignore_index=args.ignore_index,
                    eval_class_ids=args.eval_class_ids,
                    boundary_radius=args.boundary_radius,
                    band_edges=args.distance_bins,
                    sigmas=args.sigmas,
                    unc_norm=args.unc_norm,
                )
                spatial_records.append({
                    "dataset": args.dataset,
                    "method": args.method,
                    "split": args.split,
                    "image_index": global_image_index,
                    "source": "Entropy",
                    **metric_entropy,
                })

                traditional_acc["Entropy"] = update_traditional_uncertainty_accumulator(
                    traditional_acc["Entropy"],
                    uncertainty=entropy_np[b],
                    pred=pred,
                    label=gt,
                    ignore_index=args.ignore_index,
                    eval_class_ids=args.eval_class_ids,
                    unc_norm=args.unc_norm,
                )

                # Learned
                if learned_np is not None:
                    metric_learned = compute_spatial_metrics_for_one_map(
                        uncertainty=learned_np[b],
                        pred=pred,
                        label=gt,
                        ignore_index=args.ignore_index,
                        eval_class_ids=args.eval_class_ids,
                        boundary_radius=args.boundary_radius,
                        band_edges=args.distance_bins,
                        sigmas=args.sigmas,
                        unc_norm=args.unc_norm,
                    )
                    spatial_records.append({
                        "dataset": args.dataset,
                        "method": args.method,
                        "split": args.split,
                        "image_index": global_image_index,
                        "source": "Learned",
                        **metric_learned,
                    })

                    traditional_acc["Learned"] = update_traditional_uncertainty_accumulator(
                        traditional_acc["Learned"],
                        uncertainty=learned_np[b],
                        pred=pred,
                        label=gt,
                        ignore_index=args.ignore_index,
                        eval_class_ids=args.eval_class_ids,
                        unc_norm=args.unc_norm,
                    )

                global_image_index += 1

    spatial_summary = summarize_spatial_records(spatial_records)

    traditional_summary = {
        source: finalize_traditional_uncertainty_accumulator(acc)
        for source, acc in traditional_acc.items()
    }

    # Remove Learned traditional metrics if learned uncertainty is unavailable.
    if learned_available_count == 0 and "Learned" in traditional_summary:
        traditional_summary.pop("Learned", None)

    results = {
        "dataset": args.dataset,
        "method": args.method,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "save_dir": str(save_dir),
        "num_samples": len(dataset),
        "num_classes": args.num_classes,
        "ignore_index": args.ignore_index,
        "eval_class_ids_arg": args.eval_class_ids,
        "boundary_radius": args.boundary_radius,
        "distance_bins": args.distance_bins,
        "sigmas": args.sigmas,
        "unc_norm": args.unc_norm,
        "trad_bins": args.trad_bins,
        "learned_available_count": learned_available_count,
        "learned_missing_count": learned_missing_count,
        "spatial_metrics": spatial_summary,
        "traditional_metrics": traditional_summary,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_per_image_csv(spatial_records, out_csv)
    md_text = save_markdown_table(spatial_summary, traditional_summary, out_md)

    print("\nSaved:")
    print("JSON:", out_json)
    print("CSV :", out_csv)
    print("MD  :", out_md)

    print("\nMarkdown table:")
    print(md_text)

    print("\nConventional uncertainty metrics:")
    for source, mm in traditional_summary.items():
        print(f"[{source}]")
        for metric in ["Error_AUROC", "Error_AUPR", "UCE"]:
            value = mm.get(metric, None)
            if value is None:
                print(f"  {metric}: None")
            else:
                print(f"  {metric}: {value:.4f}")
        print(f"  error_rate: {mm.get('error_rate', 0.0):.4f}")

    print("\nSpatial uncertainty metrics:")
    for source, mm in spatial_summary.items():
        print(f"[{source}]")
        for metric in ["BFUR", "DSCG", "MSAD"]:
            if metric not in mm:
                continue
            mean = mm[metric]["mean"]
            std = mm[metric]["std"]
            n = mm[metric]["num_images"]
            if mean is None:
                print(f"  {metric}: None")
            else:
                print(f"  {metric}: {mean:.4f} ± {std:.4f}  n={n}")

    if learned_available_count == 0:
        print("\n[WARNING] Learned uncertainty was not found in model outputs.")
        print("[WARNING] Please check the debug output keys above.")
        print("[WARNING] You may need to modify model.forward() to return uncertainty.")


if __name__ == "__main__":
    main()
