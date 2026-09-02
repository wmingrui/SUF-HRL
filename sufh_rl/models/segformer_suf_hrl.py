"""SegFormer models used by SUF-HRL.

This file contains a plain SegFormer segmentation baseline and the residual
spatial uncertainty model proposed in the paper. The implementation uses the
HuggingFace SegFormer encoder and a lightweight multi-scale decoder.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig


class SegFormerBaseline(nn.Module):
    """SegFormer encoder with a lightweight multiclass decoder."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "nvidia/mit-b2",
        fuse_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if pretrained:
            try:
                self.backbone = SegformerModel.from_pretrained(
                    backbone_name,
                    local_files_only=True,
                )
            except Exception:
                print(
                    f"[INFO] Cannot load pretrained backbone {backbone_name}. "
                    "Initialize SegFormer from config."
                )
                config = SegformerConfig(
                    depths=[3, 4, 6, 3],
                    hidden_sizes=[32, 64, 160, 256],
                    num_attention_heads=[1, 2, 5, 8],
                    sr_ratios=[8, 4, 2, 1],
                    mlp_ratios=[4, 4, 4, 4],
                    num_channels=3,
                    decoder_hidden_size=256,
                )
                self.backbone = SegformerModel(config)
        else:
            config = SegformerConfig(
                depths=[3, 4, 6, 3],
                hidden_sizes=[32, 64, 160, 256],
                num_attention_heads=[1, 2, 5, 8],
                sr_ratios=[8, 4, 2, 1],
                mlp_ratios=[4, 4, 4, 4],
                num_channels=3,
                decoder_hidden_size=256,
            )
            self.backbone = SegformerModel(config)
        hidden_sizes = list(self.backbone.config.hidden_sizes)

        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(h, fuse_dim, kernel_size=1) for h in hidden_sizes]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_dim * len(hidden_sizes), fuse_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(fuse_dim, num_classes, kernel_size=1)

    def _fuse_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        target_h, target_w = hidden_states[0].shape[-2:]

        feats = []
        for feat, proj in zip(hidden_states, self.proj_layers):
            feat = proj(feat)
            if feat.shape[-2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            feats.append(feat)
        return self.fuse(torch.cat(feats, dim=1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_h, input_w = x.shape[-2:]
        fused = self._fuse_features(x)
        seg_logits = self.seg_head(fused)
        seg_logits = F.interpolate(
            seg_logits,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )
        return {"seg_logits": seg_logits}


class SegFormerSUFHRL(SegFormerBaseline):
    """SegFormer with residual spatial uncertainty learning.

    The model first derives an MSP uncertainty prior from the segmentation
    probabilities and then learns a bounded residual correction from decoder
    features. The final uncertainty map is used for spatial uncertainty losses
    and top-k hard-region supervision during training.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "nvidia/mit-b2",
        fuse_dim: int = 256,
        residual_scale: float = 0.15,
        pretrained: bool = True,
        delta_hidden_dims: Sequence[int] = (),
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            backbone_name=backbone_name,
            fuse_dim=fuse_dim,
            pretrained=pretrained,
        )
        self.residual_scale = residual_scale

        delta_hidden_dims = tuple(int(x) for x in delta_hidden_dims)

        if len(delta_hidden_dims) == 0:
            # Potsdam / original lightweight head
            self.delta_head = nn.Conv2d(
                fuse_dim,
                1,
                kernel_size=1,
            )
        else:
            # Deeper residual uncertainty head used by the released
            # Vaihingen and LoveDA checkpoints.
            layers = []
            in_channels = fuse_dim

            for hidden_channels in delta_hidden_dims:
                layers.extend([
                    nn.Conv2d(
                        in_channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                ])
                in_channels = hidden_channels

            layers.append(
                nn.Conv2d(
                    in_channels,
                    1,
                    kernel_size=1,
                )
            )

            self.delta_head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_h, input_w = x.shape[-2:]
        fused = self._fuse_features(x)

        seg_logits = self.seg_head(fused)
        raw_delta = self.delta_head(fused)
        seg_logits = F.interpolate(
            seg_logits,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )
        raw_delta = F.interpolate(
            raw_delta,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )

        probs = torch.softmax(seg_logits, dim=1)
        max_prob, pred_label = torch.max(probs, dim=1)
        msp_uncertainty = (1.0 - max_prob).unsqueeze(1)
        delta_map = self.residual_scale * torch.tanh(raw_delta)
        unc_map = torch.clamp(msp_uncertainty + delta_map, min=0.0, max=1.0)

        return {
            "seg_logits": seg_logits,
            "unc_map": unc_map,
            "msp_uncertainty": msp_uncertainty,
            "delta_map": delta_map,
            "raw_delta": raw_delta,
            "max_prob": max_prob.unsqueeze(1),
            "pred_label": pred_label,
        }


def build_model(
    method: str,
    num_classes: int,
    backbone_name: str = "nvidia/mit-b2",
    fuse_dim: int = 256,
    residual_scale: float = 0.15,
    pretrained: bool = True,
    delta_hidden_dims: Sequence[int] = (),
) -> nn.Module:
    """Factory for the baseline and SUF-HRL models."""
    method = method.lower()
    if method in {"baseline", "loss_topk", "msp_topk", "entropy_topk", "focal", "ohem"}:
        return SegFormerBaseline(
            num_classes=num_classes,
            backbone_name=backbone_name,
            fuse_dim=fuse_dim,
            pretrained=pretrained,
        )
    if method in {"suf_hrl", "suf-hrl", "residual", "uncertainty_topk"}:
        return SegFormerSUFHRL(
            num_classes=num_classes,
            backbone_name=backbone_name,
            fuse_dim=fuse_dim,
            residual_scale=residual_scale,
            pretrained=pretrained,
            delta_hidden_dims=delta_hidden_dims,
        )
    raise ValueError(f"Unknown method: {method}")
