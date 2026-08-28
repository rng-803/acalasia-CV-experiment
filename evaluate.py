#!/usr/bin/env python3
"""Evaluate one saved refit checkpoint on the locked patient-level test set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import discover, split
from run_experiment import build_model, load_config, make_dataset, resolve_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs.json")
    p.add_argument("--architecture", required=True, choices=["unet", "resnet34_unet", "segformer_b0", "segformer_b1", "segformer_b2"])
    p.add_argument("--checkpoint", required=True); p.add_argument("--dataset-root"); p.add_argument("--output", default="test_metrics.json"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(); config = load_config(args.config); root = resolve_dataset(config, args.dataset_root); _, test = split(discover(root), config["test_patient"])
    if args.dry_run:
        print(json.dumps({"status": "ok", "mode": "dry-run", "test_patient": config["test_patient"], "test_images": len(test), "checkpoint": str(Path(args.checkpoint).resolve()), "note": "No checkpoint was loaded and no evaluation was run."}, indent=2)); return
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_options = dict(config["architectures"][args.architecture])
    # The checkpoint already contains the encoder weights. Avoid a second
    # torchvision download when evaluating a ResNet34-U-Net checkpoint.
    if args.architecture == "resnet34_unet":
        model_options["pretrained"] = False
    model = build_model(args.architecture, model_options).to(device)
    saved = torch.load(args.checkpoint, map_location=device); model.load_state_dict(saved.get("model", saved), strict=True); model.eval()
    loader = DataLoader(make_dataset(test, int(config.get("image_size", 256)), False), batch_size=int(config.get("batch_size", 4)), shuffle=False)
    cm = torch.zeros((3, 3), dtype=torch.int64)
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device)).logits if args.architecture.startswith("segformer_") else model(images.to(device)); logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(1).cpu(); valid = (masks >= 0) & (masks < 3); cm += torch.bincount((3 * masks[valid] + pred[valid]), minlength=9).reshape(3, 3)
    cm_np = cm.numpy().astype(float); tp = np.diag(cm_np); den = cm_np.sum(1) + cm_np.sum(0); dice = [None if den[i] == 0 else float(2 * tp[i] / den[i]) for i in range(3)]
    result = {"architecture": args.architecture, "test_patient": config["test_patient"], "test_images": len(test), "dice_background": dice[0], "dice_complete_myotomy": dice[1], "dice_incomplete_myotomy": dice[2], "foreground_macro_dice": float(np.mean([x for x in dice[1:] if x is not None]))}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
