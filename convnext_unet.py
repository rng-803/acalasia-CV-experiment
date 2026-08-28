"""ConvNeXt-Tiny encoder with a compact U-Net/FPN-style decoder.

The model returns raw semantic-segmentation logits with the same spatial size
as its input. It intentionally supports only ConvNeXt-Tiny for the first
controlled comparison in the shared patient-wise experiment runner.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


ENCODER_STAGE_CHANNELS = (96, 192, 384, 768)
ENCODER_STAGE_INDICES = (1, 3, 5, 7)


def _group_count(channels: int, maximum: int = 32) -> int:
    """Choose the largest valid GroupNorm group count up to ``maximum``."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _normalization(name: str, channels: int) -> nn.Module:
    if name == "groupnorm":
        return nn.GroupNorm(_group_count(channels), channels)
    if name == "batchnorm":
        return nn.BatchNorm2d(channels)
    raise ValueError(f"Unsupported decoder normalization: {name!r}")


class ConvNormActivation(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, norm: str, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            _normalization(norm, out_channels),
            nn.GELU(),
        )


class FusionBlock(nn.Module):
    """Upsample a deep feature, fuse a projected skip, and refine it."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, norm: str, dropout: float):
        super().__init__()
        self.skip_projection = ConvNormActivation(skip_channels, out_channels, norm, kernel_size=1)
        layers: list[nn.Module] = [
            ConvNormActivation(in_channels + out_channels, out_channels, norm),
            ConvNormActivation(out_channels, out_channels, norm),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.refine = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.skip_projection(skip)
        return self.refine(torch.cat([x, skip], dim=1))


class ConvNeXtTinyUNet(nn.Module):
    """ImageNet-pretrained ConvNeXt-Tiny encoder and three-stage decoder."""

    encoder_stage_channels = ENCODER_STAGE_CHANNELS
    encoder_stage_indices = ENCODER_STAGE_INDICES

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        decoder_channels: Sequence[int] = (256, 128, 64),
        decoder_norm: str = "groupnorm",
        dropout: float = 0.0,
    ):
        super().__init__()
        decoder_channels = tuple(int(value) for value in decoder_channels)
        if len(decoder_channels) != 3 or any(value <= 0 for value in decoder_channels):
            raise ValueError("decoder_channels must contain three positive integers")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.encoder = backbone.features

        c3, c2, c1 = decoder_channels
        self.deep_projection = ConvNormActivation(ENCODER_STAGE_CHANNELS[3], c3, decoder_norm, kernel_size=1)
        self.fuse_stage3 = FusionBlock(c3, ENCODER_STAGE_CHANNELS[2], c3, decoder_norm, dropout)
        self.fuse_stage2 = FusionBlock(c3, ENCODER_STAGE_CHANNELS[1], c2, decoder_norm, dropout)
        self.fuse_stage1 = FusionBlock(c2, ENCODER_STAGE_CHANNELS[0], c1, decoder_norm, dropout)
        self.classifier = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        stages: list[torch.Tensor] = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in ENCODER_STAGE_INDICES:
                stages.append(x)
        channels = tuple(stage.shape[1] for stage in stages)
        if len(stages) != 4 or channels != ENCODER_STAGE_CHANNELS:
            raise RuntimeError(
                "Unexpected torchvision ConvNeXt-Tiny feature layout: "
                f"expected channels {ENCODER_STAGE_CHANNELS}, got {channels}"
            )
        return tuple(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        stage1, stage2, stage3, stage4 = self.forward_features(x)
        decoded = self.deep_projection(stage4)
        decoded = self.fuse_stage3(decoded, stage3)
        decoded = self.fuse_stage2(decoded, stage2)
        decoded = self.fuse_stage1(decoded, stage1)
        logits = self.classifier(decoded)
        if logits.shape[-2:] != input_size:
            logits = nn.functional.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits
