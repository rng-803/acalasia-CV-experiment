"""Self-contained copy of the project U-Net used by the experiment runner."""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__(); self.conv = ConvBlock(in_ch, out_ch); self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.conv(x); return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__(); self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2); self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            diff_y = skip.shape[-2] - x.shape[-2]; diff_x = skip.shape[-1] - x.shape[-1]
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base_filters: int = 32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_filters); self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4); self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)
        self.dec1 = DecoderBlock(base_filters * 16, base_filters * 8, base_filters * 8); self.dec2 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4)
        self.dec3 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2); self.dec4 = DecoderBlock(base_filters * 2, base_filters, base_filters)
        self.classifier = nn.Conv2d(base_filters, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1, p1 = self.enc1(x); c2, p2 = self.enc2(p1); c3, p3 = self.enc3(p2); c4, p4 = self.enc4(p3)
        x = self.bottleneck(p4); x = self.dec1(x, c4); x = self.dec2(x, c3); x = self.dec3(x, c2); x = self.dec4(x, c1)
        return self.classifier(x)
