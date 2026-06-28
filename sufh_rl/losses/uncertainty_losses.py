import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_gaussian_kernel(kernel_size: int, sigma: float, device, dtype):
    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()

    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    x: [B, C, H, W]
    """
    if sigma <= 0:
        return x

    kernel_size = max(3, int(2 * round(3 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = _make_gaussian_kernel(
        kernel_size=kernel_size,
        sigma=sigma,
        device=x.device,
        dtype=x.dtype,
    )
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    with torch.backends.cudnn.flags(enabled=False):
        return F.conv2d(
            x,
            kernel,
            padding=kernel_size // 2,
            groups=x.shape[1],
        )


def make_multiclass_boundary(labels: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    从多类标签构造 semantic boundary

    labels: [B, H, W], int64
    return: boundary [B, 1, H, W], bool

    规则：
    - 如果相邻像素类别不同，则两侧都视为 boundary
    - ignore 区域不参与 boundary 构造
    """
    assert labels.dim() == 3, f"labels should be [B,H,W], got {labels.shape}"

    b, h, w = labels.shape
    valid = labels != ignore_index

    boundary = torch.zeros((b, h, w), dtype=torch.bool, device=labels.device)

    # 左右相邻比较
    diff_lr = (
        valid[:, :, :-1] &
        valid[:, :, 1:] &
        (labels[:, :, :-1] != labels[:, :, 1:])
    )
    boundary[:, :, :-1] |= diff_lr
    boundary[:, :, 1:] |= diff_lr

    # 上下相邻比较
    diff_ud = (
        valid[:, :-1, :] &
        valid[:, 1:, :] &
        (labels[:, :-1, :] != labels[:, 1:, :])
    )
    boundary[:, :-1, :] |= diff_ud
    boundary[:, 1:, :] |= diff_ud

    return boundary.unsqueeze(1)  # [B,1,H,W]


class LocalAlignmentLossMapMultiClass(nn.Module):
    """
    直接对 uncertainty map 做多类 local alignment

    核心：
        prob = softmax(seg_logits)
        p_y  = GT 类别对应概率
        pseudo_error = 1 - p_y

    然后做多尺度局部对齐：
        loss = sum_w mean(|G_sigma(U) - G_sigma(pseudo_error)|)

    注意：
    - 对 p_y 做 detach，避免 uncertainty loss 直接反推 segmentation head
    - unc_map 已经在 [0,1]，不再对它做 sigmoid
    """

    def __init__(
        self,
        sigmas=(1.0, 3.0, 5.0),
        weights=(0.5, 0.3, 0.2),
        ignore_index: int = 255,
    ):
        super().__init__()
        assert len(sigmas) == len(weights), "sigmas and weights must have same length"

        self.sigmas = sigmas
        self.ignore_index = ignore_index

        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        self.register_buffer("weights", w)

    def forward(self, seg_logits: torch.Tensor, unc_map: torch.Tensor, labels: torch.Tensor):
        """
        seg_logits: [B,C,H,W]
        unc_map:    [B,1,H,W], already in [0,1]
        labels:     [B,H,W], int64
        """
        probs = torch.softmax(seg_logits, dim=1)  # [B,C,H,W]

        valid = labels != self.ignore_index                     # [B,H,W]
        valid_4d = valid.unsqueeze(1).float()                  # [B,1,H,W]

        safe_labels = labels.clone()
        safe_labels[~valid] = 0

        p_y = probs.gather(1, safe_labels.unsqueeze(1))        # [B,1,H,W]
        pseudo_error = (1.0 - p_y.detach()) * valid_4d         # [B,1,H,W]

        total = seg_logits.new_tensor(0.0)
        per_scale = {}

        denom = torch.clamp(valid_4d.sum(), min=1.0)

        for sigma, w in zip(self.sigmas, self.weights):
            unc_s = gaussian_blur_2d(unc_map, float(sigma))
            err_s = gaussian_blur_2d(pseudo_error, float(sigma))

            diff = torch.abs(unc_s - err_s) * valid_4d
            loss_s = diff.sum() / denom

            total = total + w * loss_s
            per_scale[f"sigma_{sigma}"] = float(loss_s.item())

        stats = {
            "valid_ratio": float(valid.float().mean().item()),
            "pseudo_error_mean": float((pseudo_error.sum() / denom).item()),
            "unc_mean": float(((unc_map * valid_4d).sum() / denom).item()),
        }

        return total, {**per_scale, **stats}


class BoundaryConcentrationLossMapMultiClass(nn.Module):
    """
    直接对 uncertainty map 做多类 boundary concentration loss

    多类版本边界：
    - 只要相邻像素类别不同，就算 semantic boundary
    - 再做 dilation 得到 boundary band

    loss:
        relu(margin - (mean_band - mean_nonband))

    目标：
    - band 内 uncertainty 高
    - non-band 内 uncertainty 低
    """

    def __init__(
        self,
        band_width: int = 3,
        margin: float = 0.18,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.band_width = band_width
        self.margin = margin
        self.ignore_index = ignore_index

    def _make_boundary_band(self, labels: torch.Tensor):
        """
        labels: [B,H,W]
        return:
            boundary: [B,1,H,W], bool
            band:     [B,1,H,W], bool
            valid:    [B,1,H,W], bool
        """
        boundary = make_multiclass_boundary(labels, ignore_index=self.ignore_index)  # [B,1,H,W]
        valid = (labels != self.ignore_index).unsqueeze(1)                           # [B,1,H,W]

        band = boundary.float()
        for _ in range(self.band_width):
            band = F.max_pool2d(band, kernel_size=3, stride=1, padding=1)

        band = band > 0.5
        band = band & valid

        return boundary, band, valid

    def forward(self, unc_map: torch.Tensor, labels: torch.Tensor):
        """
        unc_map: [B,1,H,W], already in [0,1]
        labels:  [B,H,W], int64
        """
        _, band, valid = self._make_boundary_band(labels)

        nonband = valid & (~band)

        band_vals = unc_map[band]
        nonband_vals = unc_map[nonband]

        if band_vals.numel() == 0 or nonband_vals.numel() == 0:
            zero = unc_map.new_tensor(0.0)
            stats = {
                "band_mean": 0.0,
                "nonband_mean": 0.0,
                "gap": 0.0,
                "band_ratio": 0.0,
                "valid_ratio": float(valid.float().mean().item()),
            }
            return zero, stats

        band_mean = band_vals.mean()
        nonband_mean = nonband_vals.mean()
        gap = band_mean - nonband_mean

        loss = F.relu(self.margin - gap)

        stats = {
            "band_mean": float(band_mean.item()),
            "nonband_mean": float(nonband_mean.item()),
            "gap": float(gap.item()),
            "band_ratio": float(band.float().sum().item() / torch.clamp(valid.float().sum(), min=1.0).item()),
            "valid_ratio": float(valid.float().mean().item()),
        }
        return loss, stats


if __name__ == "__main__":
    # 简单自检
    b, c, h, w = 2, 6, 256, 256

    seg_logits = torch.randn(b, c, h, w)
    unc_map = torch.rand(b, 1, h, w)
    labels = torch.randint(low=0, high=6, size=(b, h, w), dtype=torch.long)

    local_loss_fn = LocalAlignmentLossMapMultiClass(
        sigmas=(1.0, 3.0, 5.0),
        weights=(0.5, 0.3, 0.2),
        ignore_index=255,
    )

    boundary_loss_fn = BoundaryConcentrationLossMapMultiClass(
        band_width=3,
        margin=0.18,
        ignore_index=255,
    )

    local_loss, local_stats = local_loss_fn(seg_logits, unc_map, labels)
    boundary_loss, boundary_stats = boundary_loss_fn(unc_map, labels)

    print("local_loss:", local_loss.item())
    print("local_stats:", local_stats)
    print("boundary_loss:", boundary_loss.item())
    print("boundary_stats:", boundary_stats)