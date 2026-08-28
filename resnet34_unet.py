"""ResNet34 encoder + U-Net decoder for the shared experiment runner.

The encoder follows torchvision's ResNet34 feature hierarchy and the decoder
restores the input resolution with skip connections. The module returns raw
segmentation logits shaped [batch, classes, height, width].
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet34UNet(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet34 encoder."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True, decoder_channels: int = 64):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        encoder = resnet34(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layer1 = encoder.layer1  # 64 channels, 1/4 resolution
        self.layer2 = encoder.layer2  # 128 channels, 1/8 resolution
        self.layer3 = encoder.layer3  # 256 channels, 1/16 resolution
        self.layer4 = encoder.layer4  # 512 channels, 1/32 resolution

        c = decoder_channels
        self.dec4 = DecoderBlock(512, 256, c)
        self.dec3 = DecoderBlock(c, 128, c)
        self.dec2 = DecoderBlock(c, 64, c)
        self.dec1 = DecoderBlock(c, 64, c)
        self.classifier = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        stem = self.stem(x)              # 1/2
        x1 = self.layer1(self.pool(stem))  # 1/4
        x2 = self.layer2(x1)             # 1/8
        x3 = self.layer3(x2)             # 1/16
        x4 = self.layer4(x3)             # 1/32

        x = self.dec4(x4, x3)
        x = self.dec3(x, x2)
        x = self.dec2(x, x1)
        x = self.dec1(x, stem)
        x = self.classifier(x)
        if x.shape[-2:] != input_size:
            x = nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
