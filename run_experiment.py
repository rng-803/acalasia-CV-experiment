#!/usr/bin/env python3
"""Plan or execute patient-wise cross-validation experiments.

Execution is opt-in: without ``--execute`` this script only validates configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from data import Sample, balanced_weights, discover, patient_counts, patient_cv_folds, split

ROOT = Path(__file__).resolve().parents[1]
AGGREGATE_METRIC_NAMES = (
    "dice_background",
    "dice_complete_myotomy",
    "dice_incomplete_myotomy",
    "foreground_macro_dice",
    "iou_background",
    "iou_complete_myotomy",
    "iou_incomplete_myotomy",
    "foreground_macro_iou",
)


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
    if name == "resnet34_unet":
        try:
            import torchvision  # noqa: F401
        except ImportError:
            return "torchvision ResNet34 (unavailable: install requirements-experiments.txt)"
        return "torchvision ResNet34 encoder + project U-Net decoder"
    if name == "convnext_tiny_unet":
        try:
            import torchvision  # noqa: F401
        except ImportError:
            return "torchvision ConvNeXt-Tiny (unavailable: install requirements-experiments.txt)"
        if options.get("variant", "tiny") != "tiny":
            raise ValueError("convnext_tiny_unet currently supports only variant='tiny'")
        return "torchvision ConvNeXt-Tiny encoder + project U-Net/FPN decoder"
    if name.startswith("segformer_"):
        model_name = options.get("model_name", "nvidia/mit-" + name.rsplit("_", 1)[1])
        try:
            import transformers  # noqa: F401
        except ImportError:
            return f"{model_name} (unavailable: install requirements.txt)"
        if local_files_only:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(model_name, local_files_only=True)
            except Exception as exc:
                raise SystemExit(
                    f"{model_name} is not present in the local Hugging Face cache. "
                    f"Run 'python download_hf_model.py --model {model_name}' with network access, "
                    "or omit --local-files-only for this run."
                ) from exc
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
    if architecture == "resnet34_unet":
        from resnet34_unet import ResNet34UNet
        return ResNet34UNet(
            num_classes=3,
            pretrained=bool(options.get("pretrained", True)),
            decoder_channels=int(options.get("decoder_channels", 64)),
        )
    if architecture == "convnext_tiny_unet":
        from convnext_unet import ConvNeXtTinyUNet
        if options.get("variant", "tiny") != "tiny":
            raise ValueError("convnext_tiny_unet currently supports only variant='tiny'")
        return ConvNeXtTinyUNet(
            num_classes=3,
            pretrained=bool(options.get("pretrained", True)),
            decoder_channels=options.get("decoder_channels", [256, 128, 64]),
            decoder_norm=str(options.get("decoder_norm", "groupnorm")),
            dropout=float(options.get("dropout", 0.0)),
        )
    from transformers import SegformerForSemanticSegmentation
    return SegformerForSemanticSegmentation.from_pretrained(options.get("model_name", f"nvidia/mit-{architecture[-2:]}"), num_labels=3, ignore_mismatched_sizes=True, local_files_only=local_files_only)


def metric_values(confusion):
    import torch
    tp = torch.diag(confusion).numpy().astype(float); actual = confusion.sum(1).numpy(); predicted = confusion.sum(0).numpy()
    dice = [None if actual[i] + predicted[i] == 0 else float(2 * tp[i] / (actual[i] + predicted[i])) for i in range(3)]
    iou = [None if actual[i] + predicted[i] - tp[i] == 0 else float(tp[i] / (actual[i] + predicted[i] - tp[i])) for i in range(3)]
    foreground_dice = [x for x in dice[1:] if x is not None]; foreground_iou = [x for x in iou[1:] if x is not None]
    return {"dice_background": dice[0], "dice_complete_myotomy": dice[1], "dice_incomplete_myotomy": dice[2], "foreground_macro_dice": float(np.mean(foreground_dice)) if foreground_dice else None, "iou_background": iou[0], "iou_complete_myotomy": iou[1], "iou_incomplete_myotomy": iou[2], "foreground_macro_iou": float(np.mean(foreground_iou)) if foreground_iou else None}


def evaluate_model(model, loader, device, architecture):
    import torch
    import torch.nn.functional as F
    model.eval(); confusion = torch.zeros((3, 3), dtype=torch.int64)
    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device)).logits if architecture.startswith("segformer_") else model(images.to(device))
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = logits.argmax(1).cpu(); valid = (masks >= 0) & (masks < 3); encoded = 3 * masks[valid] + pred[valid]
            confusion += torch.bincount(encoded, minlength=9).reshape(3, 3)
    return metric_values(confusion)


def save_overlays(model, samples, image_size, device, architecture, output_dir, count=5):
    import cv2
    import torch
    import torch.nn.functional as F
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval(); dataset = make_dataset(samples[:count], image_size, False)
    for index, sample in enumerate(samples[:count]):
        original = cv2.cvtColor(cv2.imread(sample.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        ground_truth = cv2.resize(cv2.imread(sample.mask, cv2.IMREAD_GRAYSCALE), (image_size, image_size), interpolation=cv2.INTER_NEAREST)
        with torch.no_grad():
            logits = model(dataset[index][0][None].to(device)).logits if architecture.startswith("segformer_") else model(dataset[index][0][None].to(device))
            prediction = F.interpolate(logits, size=(image_size, image_size), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
        palette = np.asarray([[0, 0, 0], [102, 255, 102], [51, 221, 255]], dtype=np.uint8)
        gt_rgb = palette[np.clip(ground_truth, 0, 2)]; pred_rgb = palette[np.clip(prediction, 0, 2)]
        overlay = cv2.addWeighted(original, 0.55, pred_rgb, 0.45, 0)
        panel = np.concatenate([original, gt_rgb, pred_rgb, overlay], axis=1)
        stem = Path(sample.image).stem.replace("/", "_")
        cv2.imwrite(str(output_dir / f"{index:02d}_{sample.patient}_{stem}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def aggregate_metric_rows(fold_rows: list[dict]) -> list[dict]:
    rows = []
    for metric in AGGREGATE_METRIC_NAMES:
        values = [row[metric] for row in fold_rows if row.get(metric) is not None]
        rows.append({"metric": metric, "folds": len(values), "mean": float(np.mean(values)) if values else None, "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None})
    return rows


def execute(config: dict, architecture: str, dataset_root: Path, output_dir: Path, local_files_only: bool, run_name: str | None = None, overlay_count: int = 5) -> None:
    require_torch()
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    seed = int(config.get("seed", 42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    samples = discover(dataset_root); train, _ = split(samples, config["test_patient"])
    options = config["architectures"][architecture]
    validate_architecture(architecture, options, local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting {architecture} on {device}; {len(train)} training images; test patient={config['test_patient']}", flush=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = run_name or f"{timestamp}_{architecture}"
    run_dir = output_dir / safe_name
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}. Choose another --run-name.")
    run_dir.mkdir(parents=True, exist_ok=False)
    folds = patient_cv_folds(train, seed)
    run_config = {"run_name": safe_name, "created_at_utc": timestamp, "architecture": architecture, "architecture_options": options, "dataset_root": str(dataset_root), "patient_counts": patient_counts(samples), "train_patients": sorted({s.patient for s in train}), "test_patient": config["test_patient"], "seed": seed, "image_size": int(config.get("image_size", 256)), "batch_size": int(config.get("batch_size", 4)), "num_workers": int(config.get("num_workers", 0)), "epochs": int(config.get("epochs", 10)), "learning_rate": float(config.get("learning_rate", 1e-4)), "device": str(device), "local_files_only": local_files_only, "git_revision": git_revision(), "folds": [{"fold": f["fold"], "train_patients": f["train_patients"], "val_patients": f["val_patients"]} for f in folds]}
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    history_rows = []; fold_rows = []
    print(f"Run directory: {run_dir}", flush=True)
    for fold in folds:
        fold_train = [Sample(**s) for s in fold["train_samples"]]; fold_val = [Sample(**s) for s in fold["val_samples"]]
        model = build_model(architecture, options, local_files_only).to(device)
        train_ds = make_dataset(fold_train, int(config.get("image_size", 256)), True)
        val_ds = make_dataset(fold_val, int(config.get("image_size", 256)), False)
        sampler = WeightedRandomSampler(torch.as_tensor(balanced_weights(fold_train), dtype=torch.double), len(fold_train), replacement=True)
        loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 4)), sampler=sampler, num_workers=int(config.get("num_workers", 0)))
        val_loader = DataLoader(val_ds, batch_size=int(config.get("batch_size", 4)), shuffle=False, num_workers=int(config.get("num_workers", 0)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
        fold_dir = run_dir / f"fold_{fold['fold']:02d}"
        fold_dir.mkdir()
        print(f"Fold {fold['fold'] + 1}/{len(folds)}: train={len(fold_train)} images from {fold['train_patients']}; validation={len(fold_val)} images from {fold['val_patients']}", flush=True)
        best_score = -1.0; best_epoch = 0; best_state = None
        for epoch in range(int(config.get("epochs", 10))):
            model.train()
            loss_sum = 0.0
            for images, masks in loader:
                images, masks = images.to(device), masks.to(device); optimizer.zero_grad(set_to_none=True)
                logits = model(images).logits if architecture.startswith("segformer_") else model(images)
                if logits.shape[-2:] != masks.shape[-2:]: logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, masks); loss.backward(); optimizer.step(); loss_sum += loss.item()
            metrics = evaluate_model(model, val_loader, device, architecture)
            history_rows.append({"fold": fold["fold"], "val_patient": fold["val_patients"][0], "epoch": epoch + 1, "train_loss": loss_sum / max(1, len(loader)), **metrics})
            score = metrics["foreground_macro_dice"] if metrics["foreground_macro_dice"] is not None else -1.0
            print(f"  epoch {epoch + 1}/{int(config.get('epochs', 10))}: train_loss={loss_sum / max(1, len(loader)):.4f} val_fg_dice={score:.4f}", flush=True)
            if score > best_score:
                best_score = score; best_epoch = epoch + 1; best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is not None: model.load_state_dict(best_state)
        final_metrics = evaluate_model(model, val_loader, device, architecture)
        fold_result = {"fold": fold["fold"], "val_patient": fold["val_patients"][0], "best_epoch": best_epoch, **final_metrics}; fold_rows.append(fold_result)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_result, indent=2) + "\n")
        torch.save({"model": model.state_dict(), "architecture": architecture, "fold": fold["fold"], "config": run_config, "metrics": fold_result}, fold_dir / "best_model.pt")
        save_overlays(model, fold_val, int(config.get("image_size", 256)), device, architecture, fold_dir / "overlays", overlay_count)
        write_csv(fold_dir / "history.csv", [row for row in history_rows if row["fold"] == fold["fold"]])
    write_csv(run_dir / "history.csv", history_rows); write_csv(run_dir / "fold_metrics.csv", fold_rows)
    aggregate_rows = aggregate_metric_rows(fold_rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    summary = {"run_name": safe_name, "architecture": architecture, "test_patient": config["test_patient"], "fold_count": len(fold_rows), "aggregate_metrics": aggregate_rows}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed execution for {architecture}; outputs are in {run_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="experiments/configs.json"); p.add_argument("--architecture", required=True, choices=["unet", "resnet34_unet", "convnext_tiny_unet", "segformer_b0", "segformer_b1", "segformer_b2", "sam2"])
    p.add_argument("--dataset-root"); p.add_argument("--output-dir", default="runs/experiments"); p.add_argument("--run-name", help="Unique child directory name; defaults to UTC timestamp plus architecture"); p.add_argument("--overlay-count", type=int, default=5); p.add_argument("--local-files-only", action="store_true"); p.add_argument("--execute", action="store_true"); p.add_argument("--dry-run", action="store_true", help="Explicit no-training mode (the default)")
    args = p.parse_args(); config = load_config(args.config); root = resolve_dataset(config, args.dataset_root)
    if not args.execute or args.dry_run: dry_run(config, args.architecture, root, args.local_files_only)
    else: execute(config, args.architecture, root, Path(args.output_dir), args.local_files_only, args.run_name, args.overlay_count)


if __name__ == "__main__": main()
