#!/usr/bin/env python3
"""Fine-tune SAM2.1 Hiera-S with mask-derived positive point prompts.

The default mode is a dry-run. Add ``--execute`` only in a CUDA cloud
environment after installing SAM2 and downloading the checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import zlib
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from data import Sample, balanced_weights, discover, patient_counts, patient_cv_folds, split
from run_experiment import aggregate_metric_rows, git_revision, metric_values, write_csv
from sam2_points import compose_semantic_prediction, sample_class_prompts


SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"


def load_sample(sample: Sample, max_side: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(sample.image, cv2.IMREAD_COLOR)
    mask = cv2.imread(sample.mask, cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or mask is None:
        raise RuntimeError(f"Could not read {sample.image} or {sample.mask}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale != 1.0:
        size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return image, mask


def has_foreground(sample: Sample) -> bool:
    mask = cv2.imread(sample.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Could not read {sample.mask}")
    return bool(np.any((mask == 1) | (mask == 2)))


def device_context(device: str):
    import torch
    if device == "cuda" and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def predictor_forward(predictor, image: np.ndarray, points: np.ndarray, labels: np.ndarray):
    """Run the trainable prompt encoder + mask decoder, matching SAM2 internals."""
    import torch
    predictor.set_image(image)
    _, unnorm_coords, point_labels, _ = predictor._prep_prompts(points[:, None, :], labels[:, None], box=None, mask_logits=None, normalize_coords=True)
    sparse, dense = predictor.model.sam_prompt_encoder(points=(unnorm_coords, point_labels), boxes=None, masks=None)
    high_res = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
    low_res, scores, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=True, repeat_image=unnorm_coords.shape[0] > 1,
        high_res_features=high_res,
    )
    batch_indices = torch.arange(low_res.shape[0], device=low_res.device)
    best_indices = scores.argmax(dim=1)
    selected_low_res = low_res[batch_indices, best_indices][:, None]
    selected_scores = scores[batch_indices, best_indices]
    masks = predictor._transforms.postprocess_masks(selected_low_res, predictor._orig_hw[-1])
    return masks[:, 0], selected_scores


def loss_and_iou(pred_logits, scores, target):
    import torch
    import torch.nn.functional as F
    pred = torch.sigmoid(pred_logits)
    if target.shape != pred.shape:
        target = target.expand_as(pred)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    hard = (pred > 0.5).float(); intersection = (target * hard).flatten(1).sum(1)
    union = (target + hard - target * hard).flatten(1).sum(1).clamp_min(1)
    iou = intersection / union
    return bce + 0.05 * torch.abs(scores - iou).mean(), iou.mean()


def sample_seed(base_seed: int, sample: Sample) -> int:
    return int(base_seed + zlib.crc32(sample.image.encode("utf-8")))


def predict_semantic_mask(predictor, image: np.ndarray, source_mask: np.ndarray, seed: int, device: str, threshold: float) -> tuple[np.ndarray, float | None]:
    """Predict an indexed mask from one oracle positive prompt per present class."""
    import torch
    if not np.any((source_mask == 1) | (source_mask == 2)):
        return np.zeros_like(source_mask, dtype=np.uint8), None
    targets, points, point_labels, class_ids = sample_class_prompts(source_mask, seed=seed)
    target_t = torch.from_numpy(targets).to(device)
    with device_context(device):
        logits, scores = predictor_forward(predictor, image, points, point_labels)
        _, prompt_iou = loss_and_iou(logits, scores, target_t)
    probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()
    prediction = compose_semantic_prediction(probabilities, class_ids, threshold)
    return prediction, float(prompt_iou.detach())


def evaluate_sam2(predictor, samples: list[Sample], max_side: int, device: str, seed: int, threshold: float) -> tuple[dict, float]:
    import torch
    confusion = torch.zeros((3, 3), dtype=torch.int64)
    prompt_ious = []
    predictor.model.eval()
    with torch.no_grad():
        for sample in samples:
            image, source_mask = load_sample(sample, max_side)
            prediction, prompt_iou = predict_semantic_mask(predictor, image, source_mask, sample_seed(seed, sample), device, threshold)
            target = torch.from_numpy(source_mask.astype(np.int64))
            predicted = torch.from_numpy(prediction.astype(np.int64))
            valid = (target >= 0) & (target < 3)
            encoded = 3 * target[valid] + predicted[valid]
            confusion += torch.bincount(encoded, minlength=9).reshape(3, 3)
            if prompt_iou is not None:
                prompt_ious.append(prompt_iou)
    return metric_values(confusion), float(np.mean(prompt_ious)) if prompt_ious else 0.0


def save_sam2_overlays(predictor, samples: list[Sample], max_side: int, device: str, seed: int, threshold: float, output_dir: Path, count: int) -> None:
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = np.asarray([[0, 0, 0], [102, 255, 102], [51, 221, 255]], dtype=np.uint8)
    predictor.model.eval()
    with torch.no_grad():
        for index, sample in enumerate(samples[:count]):
            image, ground_truth = load_sample(sample, max_side)
            prediction, _ = predict_semantic_mask(predictor, image, ground_truth, sample_seed(seed, sample), device, threshold)
            gt_rgb = palette[np.clip(ground_truth, 0, 2)]
            pred_rgb = palette[np.clip(prediction, 0, 2)]
            overlay = cv2.addWeighted(image, 0.55, pred_rgb, 0.45, 0)
            panel = np.concatenate([image, gt_rgb, pred_rgb, overlay], axis=1)
            stem = Path(sample.image).stem.replace("/", "_")
            cv2.imwrite(str(output_dir / f"{index:02d}_{sample.patient}_{stem}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def dry_run(dataset_root: Path, test_patient: str, checkpoint: Path, folds: int, max_side: int) -> None:
    samples = discover(dataset_root); train, test = split(samples, test_patient); cv = patient_cv_folds(train, 42)
    print(json.dumps({"status": "ok", "mode": "dry-run", "architecture": "sam2", "variant": "sam2.1_hiera_small", "model_config": SAM2_MODEL_CONFIG, "checkpoint": str(checkpoint), "checkpoint_url": SAM2_CHECKPOINT_URL, "checkpoint_present": checkpoint.is_file(), "dataset_root": str(dataset_root), "patient_counts": patient_counts(samples), "train_images": len(train), "test_patient": test_patient, "test_images": len(test), "cv_folds": [{"fold": f["fold"], "train_patients": f["train_patients"], "val_patients": f["val_patients"]} for f in cv[:folds]], "prompt_strategy": "one 5x5-eroded oracle positive point and one binary target per present foreground class", "semantic_composition": "threshold each class-conditioned probability map and resolve overlaps by highest probability", "metrics": ["Dice per class", "IoU per class", "foreground macro Dice", "foreground macro IoU", "prompt IoU"], "trainable_parts": ["sam_prompt_encoder", "sam_mask_decoder"], "frozen_parts": ["image_encoder"], "max_side": max_side, "note": "No SAM2 import, checkpoint load, optimizer, or training was started."}, indent=2))


def execute(args, dataset_root: Path, checkpoint: Path) -> None:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 small training requires CUDA; run this command in the CUDA cloud environment.")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    samples = discover(dataset_root); train, _ = split(samples, args.test_patient); folds = patient_cv_folds(train, args.seed)[:args.folds]
    device = "cuda"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = args.run_name or f"{timestamp}_sam2"
    run_dir = Path(args.output_dir) / safe_name
    if run_dir.exists():
        raise SystemExit(f"Run directory already exists: {run_dir}. Choose another --run-name.")
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {
        "run_name": safe_name,
        "created_at_utc": timestamp,
        "architecture": "sam2",
        "variant": "sam2.1_hiera_small",
        "model_config": args.model_config,
        "initial_checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "patient_counts": patient_counts(samples),
        "train_patients": sorted({sample.patient for sample in train}),
        "test_patient": args.test_patient,
        "seed": args.seed,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_side": args.max_side,
        "threshold": args.threshold,
        "device": device,
        "prompt_strategy": "one oracle positive point sampled inside each present ground-truth class",
        "semantic_composition": "class-conditioned probabilities thresholded and overlaps resolved by highest probability",
        "comparison_warning": "SAM2 metrics are prompt-conditioned using annotation-derived points; other architectures are unprompted semantic segmenters.",
        "git_revision": git_revision(),
        "folds": [{"fold": fold["fold"], "train_patients": fold["train_patients"], "val_patients": fold["val_patients"]} for fold in folds],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    history_rows = []
    fold_rows = []
    print(f"Run directory: {run_dir}", flush=True)
    for fold in folds:
        fold_train = [Sample(**s) for s in fold["train_samples"]]; fold_val = [Sample(**s) for s in fold["val_samples"]]
        promptable_train = [sample for sample in fold_train if has_foreground(sample)]
        if not promptable_train:
            raise RuntimeError(f"Fold {fold['fold']} has no foreground masks available for positive-point training.")
        fold_sampling_weights = balanced_weights(promptable_train).tolist()
        fold_dir = run_dir / f"fold_{fold['fold']:02d}"
        fold_dir.mkdir()
        predictor = SAM2ImagePredictor(build_sam2(args.model_config, str(checkpoint), device=device))
        for parameter in predictor.model.image_encoder.parameters(): parameter.requires_grad = False
        predictor.model.image_encoder.eval(); predictor.model.sam_prompt_encoder.train(True); predictor.model.sam_mask_decoder.train(True)
        params = [p for p in predictor.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        optimizer.zero_grad(set_to_none=True)
        best_score = float("-inf"); best_step = 0; loss_window = []; prompt_iou_window = []
        print(f"Fold {fold['fold'] + 1}/{len(folds)}: train={len(fold_train)} images from {fold['train_patients']}; validation={len(fold_val)} images from {fold['val_patients']}", flush=True)
        for step in range(1, args.steps + 1):
            sample = random.choices(promptable_train, weights=fold_sampling_weights, k=1)[0]; image, source_mask = load_sample(sample, args.max_side)
            targets, points, point_labels, _ = sample_class_prompts(source_mask, seed=args.seed + step)
            target_t = torch.from_numpy(targets).to(device)
            with device_context(device):
                logits, scores = predictor_forward(predictor, image, points, point_labels); loss, score = loss_and_iou(logits, scores, target_t)
            scaler.scale(loss / args.accumulation_steps).backward()
            loss_window.append(float(loss.detach())); prompt_iou_window.append(float(score.detach()))
            if step % args.accumulation_steps == 0 or step == args.steps:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"fold={fold['fold']} step={step}/{args.steps} train_loss={float(loss.detach()):.4f} prompt_iou={float(score.detach()):.4f}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics, val_prompt_iou = evaluate_sam2(predictor, fold_val, args.max_side, device, args.seed, args.threshold)
                row = {"fold": fold["fold"], "val_patient": fold["val_patients"][0], "step": step, "train_loss": float(np.mean(loss_window)), "train_prompt_iou": float(np.mean(prompt_iou_window)), "val_prompt_iou": val_prompt_iou, **metrics}
                history_rows.append(row); loss_window = []; prompt_iou_window = []
                score_value = metrics["foreground_macro_dice"] if metrics["foreground_macro_dice"] is not None else -1.0
                iou_value = metrics["foreground_macro_iou"] if metrics["foreground_macro_iou"] is not None else -1.0
                if score_value > best_score:
                    best_score = score_value; best_step = step
                    torch.save({"model": predictor.model.state_dict(), "architecture": "sam2", "fold": fold["fold"], "config": run_config, "metrics": row}, fold_dir / "best_model.pt")
                print(f"fold={fold['fold']} step={step} val_fg_dice={score_value:.4f} val_fg_iou={iou_value:.4f} prompt_iou={val_prompt_iou:.4f}", flush=True)
                predictor.model.train(True); predictor.model.image_encoder.eval()
        best_checkpoint = torch.load(fold_dir / "best_model.pt", map_location=device, weights_only=False)
        predictor.model.load_state_dict(best_checkpoint["model"], strict=True)
        final_metrics, final_prompt_iou = evaluate_sam2(predictor, fold_val, args.max_side, device, args.seed, args.threshold)
        fold_result = {"fold": fold["fold"], "val_patient": fold["val_patients"][0], "best_step": best_step, "val_prompt_iou": final_prompt_iou, **final_metrics}
        fold_rows.append(fold_result)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_result, indent=2) + "\n")
        best_checkpoint["metrics"] = fold_result
        torch.save(best_checkpoint, fold_dir / "best_model.pt")
        save_sam2_overlays(predictor, fold_val, args.max_side, device, args.seed, args.threshold, fold_dir / "overlays", args.overlay_count)
        write_csv(fold_dir / "history.csv", [row for row in history_rows if row["fold"] == fold["fold"]])
    write_csv(run_dir / "history.csv", history_rows)
    write_csv(run_dir / "fold_metrics.csv", fold_rows)
    aggregate_rows = aggregate_metric_rows(fold_rows)
    write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    summary = {"run_name": safe_name, "architecture": "sam2", "variant": "sam2.1_hiera_small", "test_patient": args.test_patient, "fold_count": len(fold_rows), "prompt_conditioned": True, "aggregate_metrics": aggregate_rows}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed execution for sam2; outputs are in {run_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="../dataset/processed"); p.add_argument("--test-patient", default="p5"); p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_small.pt"); p.add_argument("--model-config", default=SAM2_MODEL_CONFIG); p.add_argument("--output-dir", default="runs/experiments"); p.add_argument("--run-name"); p.add_argument("--folds", type=int, default=4); p.add_argument("--steps", type=int, default=600); p.add_argument("--eval-every", type=int, default=100); p.add_argument("--accumulation-steps", type=int, default=8); p.add_argument("--learning-rate", type=float, default=5e-5); p.add_argument("--max-side", type=int, default=1024); p.add_argument("--threshold", type=float, default=0.5); p.add_argument("--overlay-count", type=int, default=5); p.add_argument("--seed", type=int, default=42); p.add_argument("--execute", action="store_true"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(); root = Path(args.dataset_root).expanduser()
    if args.folds <= 0 or args.steps <= 0 or args.eval_every <= 0 or args.accumulation_steps <= 0:
        p.error("--folds, --steps, --eval-every, and --accumulation-steps must be positive")
    if not 0.0 < args.threshold < 1.0:
        p.error("--threshold must be between 0 and 1")
    if args.overlay_count < 0:
        p.error("--overlay-count must be non-negative")
    if not root.is_absolute():
        cwd_candidate = (Path.cwd() / root).resolve()
        root = cwd_candidate if cwd_candidate.is_dir() else (Path(__file__).resolve().parent / root).resolve()
    checkpoint = Path(args.checkpoint).expanduser()
    checkpoint = checkpoint if checkpoint.is_absolute() else (Path.cwd() / checkpoint).resolve()
    if args.execute and not args.dry_run: execute(args, root, checkpoint)
    else: dry_run(root, args.test_patient, checkpoint, args.folds, args.max_side)


if __name__ == "__main__": main()
