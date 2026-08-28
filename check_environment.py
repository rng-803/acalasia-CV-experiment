#!/usr/bin/env python3
"""Report whether the selected experiment runtime is installed and usable."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--architecture", choices=["unet", "resnet34_unet", "convnext_tiny_unet", "segformer_b0", "segformer_b1", "segformer_b2", "sam2"], default="unet"); p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_small.pt")
    args = p.parse_args(); required = ["numpy", "PIL", "cv2", "torch"]
    if args.architecture in {"resnet34_unet", "convnext_tiny_unet"}: required.append("torchvision")
    if args.architecture.startswith("segformer_"): required.append("transformers")
    if args.architecture == "sam2": required.extend(["sam2", "hydra"])
    modules = {name: importlib.util.find_spec(name) is not None for name in required}
    result = {"python": sys.executable, "python_version": platform.python_version(), "architecture": args.architecture, "modules": modules}
    if modules.get("torch"):
        import torch
        result["torch_version"] = torch.__version__; result["cuda_available"] = torch.cuda.is_available(); result["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if args.architecture == "sam2": result["checkpoint_present"] = Path(args.checkpoint).expanduser().is_file()
    result["status"] = "ok" if all(modules.values()) and (args.architecture != "sam2" or result["checkpoint_present"]) else "incomplete"
    print(json.dumps(result, indent=2)); raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__": main()
