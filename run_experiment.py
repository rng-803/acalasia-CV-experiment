#!/usr/bin/env python3
"""Plan or execute patient-wise cross-validation experiments.

Execution is opt-in: without ``--execute`` this script only validates configuration.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from data import Sample, balanced_weights, discover, patient_counts, patient_cv_folds, split

ROOT = Path(__file__).resolve().parents[1]


def require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for --execute but is not installed in this Python environment. "
            "Create/activate the project environment and run: "
            "python3 -m pip install -r experiments/requirements-experiments.txt"
        ) from exc


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    required = {"dataset_root", "test_patient", "architectures"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    return config


def resolve_dataset(config: dict, override: str | None) -> Path:
    raw = override or config["dataset_root"]
    path = Path(raw).expanduser()
    if not path.is_absolute():
        cwd_candidate = (Path.cwd() / path).resolve()
        script_candidate = (Path(__file__).resolve().parent / path).resolve()
        path = cwd_candidate if cwd_candidate.is_dir() else script_candidate
    return path


def validate_architecture(name: str, options: dict, local_files_only: bool) -> str:
    if name == "unet":
        return "RESNET/teste_vid_01/model_unet.py"
    if name.startswith("segformer_"):
        model_name = options.get("model_name", "nvidia/mit-" + name.rsplit("_", 1)[1])
        try:
            import transformers  # noqa: F401
        except ImportError:
            return f"{model_name} (unavailable: install requirements.txt)"
        return model_name + (" (local cache)" if local_files_only else "")
    if name == "sam2":
        if not options.get("enabled", False):
            return "disabled (set architectures.sam2.enabled=true only after defining prompts/checkpoint)"
        try:
            import sam2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("SAM2 is opt-in and requires the optional sam2 package") from exc
        if not options.get("checkpoint"):
            raise ValueError("SAM2 requires architectures.sam2.checkpoint")
        return "optional SAM2 prompt-based adapter"
    raise ValueError(f"Unsupported architecture: {name}")


def dry_run(config: dict, architecture: str, dataset_root: Path, local_files_only: bool) -> None:
    samples = discover(dataset_root)
    train, test = split(samples, config["test_patient"])
    options = config["architectures"].get(architecture)
    if options is None:
        raise ValueError(f"No configuration for architecture {architecture!r}")
    model = validate_architecture(architecture, options, local_files_only)
    folds = patient_cv_folds(train, config.get("seed", 42))
    print(json.dumps({
        "status": "ok",
        "mode": "dry-run",
        "architecture": architecture,
        "model": model,
        "dataset_root": str(dataset_root),
        "patient_counts": patient_counts(samples),
        "train_patients": sorted({s.patient for s in train}),
        "test_patient": config["test_patient"],
        "train_images": len(train),
        "test_images": len(test),
        "folds": [{"fold": f["fold"], "train_patients": f["train_patients"], "val_patients": f["val_patients"], "train_images": len(f["train_samples"]), "val_images": len(f["val_samples"])} for f in folds],
        "balance": "patient-uniform weighted sampling (weight=1/patient_image_count)",
        "epochs": config.get("epochs", 10),
        "batch_size": config.get("batch_size", 4),
        "execute_hint": "Add --execute to start training; this command did not initialize a model or optimizer."
    }, indent=2))


def make_dataset(samples: list[Sample], image_size: int, augment: bool):
    from PIL import Image, ImageEnhance
    import torch
    from torch.utils.data import Dataset

    class DatasetImpl(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            image = Image.open(samples[i].image).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
            mask = np.asarray(Image.open(samples[i].mask).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST), dtype=np.int64)
            if augment and random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT); mask = np.fliplr(mask).copy()
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = (image - np.asarray([0.485, 0.456, 0.406])) / np.asarray([0.229, 0.224, 0.225])
            return torch.from_numpy(image.transpose(2, 0, 1)).float(), torch.from_numpy(mask).long()
    return DatasetImpl()


def build_model(architecture: str, options: dict, local_files_only: bool = False):
    if architecture == "unet":
        try:
            from unet_model import UNet
        except ImportError:
            sys.path.insert(0, str(ROOT / "RESNET" / "teste_vid_01"))
            from model_unet import UNet
        return UNet(3, 3, base_filters=int(options.get("base_filters", 32)))
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(options.get("model_name", f"nvidia/mit-{architecture[-2:]}"), num_labels=3, ignore_mismatched_sizes=True, local_files_only=local_files_only)


def execute(config: dict, architecture: str, dataset_root: Path, output_dir: Path, local_files_only: bool) -> None:
    require_torch()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    seed = int(config.get("seed", 42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    samples = discover(dataset_root); train, _ = split(samples, config["test_patient"])
    options = config["architectures"][architecture]
    validate_architecture(architecture, options, local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold in patient_cv_folds(train, seed):
        fold_train = [Sample(**s) for s in fold["train_samples"]]; fold_val = [Sample(**s) for s in fold["val_samples"]]
        model = build_model(architecture, options, local_files_only).to(device)
        train_ds = make_dataset(fold_train, int(config.get("image_size", 256)), True)
        val_ds = make_dataset(fold_val, int(config.get("image_size", 256)), False)
        sampler = WeightedRandomSampler(torch.as_tensor(balanced_weights(fold_train), dtype=torch.double), len(fold_train), replacement=True)
        loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 4)), sampler=sampler, num_workers=int(config.get("num_workers", 0)))
        val_loader = DataLoader(val_ds, batch_size=int(config.get("batch_size", 4)), shuffle=False, num_workers=int(config.get("num_workers", 0)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
        for epoch in range(int(config.get("epochs", 10))):
            model.train()
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device); optimizer.zero_grad(set_to_none=True)
                logits = model(images).logits if architecture.startswith("segformer_") else model(images)
                if logits.shape[-2:] != masks.shape[-2:]: logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, masks); loss.backward(); optimizer.step()
        model.eval(); confusion = torch.zeros((3, 3), dtype=torch.int64)
        with torch.no_grad():
            for images, masks in val_loader:
                logits = model(images.to(device)).logits if architecture.startswith("segformer_") else model(images.to(device))
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                pred = logits.argmax(1).cpu(); target = masks
                valid = (target >= 0) & (target < 3)
                encoded = 3 * target[valid] + pred[valid]
                confusion += torch.bincount(encoded, minlength=9).reshape(3, 3)
        tp = torch.diag(confusion).numpy().astype(float); actual = confusion.sum(1).numpy(); predicted = confusion.sum(0).numpy()
        dice = [None if actual[i] + predicted[i] == 0 else float(2 * tp[i] / (actual[i] + predicted[i])) for i in range(3)]
        iou = [None if actual[i] + predicted[i] - tp[i] == 0 else float(tp[i] / (actual[i] + predicted[i] - tp[i])) for i in range(3)]
        metrics = {"fold": fold["fold"], "val_patient": fold["val_patients"][0], "dice_background": dice[0], "dice_complete_myotomy": dice[1], "dice_incomplete_myotomy": dice[2], "foreground_macro_dice": float(np.mean([x for x in dice[1:] if x is not None])) if any(x is not None for x in dice[1:]) else None, "iou_background": iou[0], "iou_complete_myotomy": iou[1], "iou_incomplete_myotomy": iou[2]}
        (output_dir / f"fold_{fold['fold']}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        torch.save({"model": model.state_dict(), "architecture": architecture, "fold": fold["fold"], "config": config, "metrics": metrics}, output_dir / f"fold_{fold['fold']}.pt")
    print(f"Completed execution for {architecture}; outputs are in {output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/configs.json"); p.add_argument("--architecture", required=True, choices=["unet", "segformer_b0", "segformer_b1", "segformer_b2", "sam2"])
    p.add_argument("--dataset-root"); p.add_argument("--output-dir", default="runs/experiments"); p.add_argument("--local-files-only", action="store_true"); p.add_argument("--execute", action="store_true"); p.add_argument("--dry-run", action="store_true", help="Explicit no-training mode (the default)")
    args = p.parse_args(); config = load_config(args.config); root = resolve_dataset(config, args.dataset_root)
    if not args.execute or args.dry_run: dry_run(config, args.architecture, root, args.local_files_only)
    else: execute(config, args.architecture, root, Path(args.output_dir), args.local_files_only)


if __name__ == "__main__": main()
