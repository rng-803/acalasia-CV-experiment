#!/usr/bin/env python3
"""Download and cache pretrained torchvision backbones used by experiments."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["convnext_tiny", "resnet34"], required=True)
    args = parser.parse_args()

    if args.model == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        convnext_tiny(weights=weights)
    else:
        from torchvision.models import ResNet34_Weights, resnet34
        weights = ResNet34_Weights.DEFAULT
        resnet34(weights=weights)
    print(f"Cached torchvision weights for {args.model}: {weights.url}")


if __name__ == "__main__":
    main()
