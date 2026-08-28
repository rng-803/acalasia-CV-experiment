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
import sys
from pathlib import Path

import cv2
import numpy as np

from data import Sample, discover, patient_counts, patient_cv_folds, split
from sam2_points import sample_positive_points


SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_small.pt"


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
    masks = predictor._transforms.postprocess_masks(low_res, predictor._orig_hw[-1])
    return masks[:, 0], scores[:, 0]


def loss_and_iou(pred_logits, scores, target):
    import torch
    pred = torch.sigmoid(pred_logits)
    target = target.expand_as(pred)
    bce = -(target * torch.log(pred + 1e-6) + (1 - target) * torch.log(1 - pred + 1e-6)).mean()
    hard = (pred > 0.5).float(); intersection = (target * hard).flatten(1).sum(1)
    union = (target + hard - target * hard).flatten(1).sum(1).clamp_min(1)
    iou = intersection / union
    return bce + 0.05 * torch.abs(scores - iou).mean(), iou.mean()


def dry_run(dataset_root: Path, test_patient: str, checkpoint: Path, folds: int, max_side: int) -> None:
    samples = discover(dataset_root); train, test = split(samples, test_patient); cv = patient_cv_folds(train, 42)
    print(json.dumps({"status": "ok", "mode": "dry-run", "architecture": "sam2.1_hiera_small", "model_config": SAM2_MODEL_CONFIG, "checkpoint": str(checkpoint), "checkpoint_url": SAM2_CHECKPOINT_URL, "checkpoint_present": checkpoint.is_file(), "dataset_root": str(dataset_root), "patient_counts": patient_counts(samples), "train_images": len(train), "test_patient": test_patient, "test_images": len(test), "cv_folds": [{"fold": f["fold"], "train_patients": f["train_patients"], "val_patients": f["val_patients"]} for f in cv[:folds]], "prompt_strategy": "binary mask=(mask>0), 5x5 erosion, one random positive point per present non-background label", "trainable_parts": ["sam_prompt_encoder", "sam_mask_decoder"], "frozen_parts": ["image_encoder"], "max_side": max_side, "note": "No SAM2 import, checkpoint load, optimizer, or training was started."}, indent=2))


def execute(args, dataset_root: Path, checkpoint: Path) -> None:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():
        raise RuntimeError("SAM2 small training requires CUDA; run this command in the CUDA cloud environment.")
    samples = discover(dataset_root); train, _ = split(samples, args.test_patient); folds = patient_cv_folds(train, args.seed)[:args.folds]
    device = "cuda"; out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for fold in folds:
        fold_train = [Sample(**s) for s in fold["train_samples"]]; fold_val = [Sample(**s) for s in fold["val_samples"]]
        predictor = SAM2ImagePredictor(build_sam2(args.model_config, str(checkpoint), device=device))
        for parameter in predictor.model.image_encoder.parameters(): parameter.requires_grad = False
        predictor.model.image_encoder.eval(); predictor.model.sam_prompt_encoder.train(True); predictor.model.sam_mask_decoder.train(True)
        params = [p for p in predictor.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=True); best = -1.0
        for step in range(1, args.steps + 1):
            sample = random.choice(fold_train); image, source_mask = load_sample(sample, args.max_side)
            target, points, point_labels = sample_positive_points(source_mask, seed=args.seed + step)
            target_t = torch.from_numpy(target[None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with device_context(device):
                logits, scores = predictor_forward(predictor, image, points, point_labels); loss, score = loss_and_iou(logits, scores, target_t)
            scaler.scale(loss / args.accumulation_steps).backward()
            if step % args.accumulation_steps == 0:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(optimizer); scaler.update()
            if step % args.eval_every == 0:
                predictor.model.eval(); vals = []
                with torch.no_grad():
                    for val_sample in fold_val:
                        val_image, val_mask = load_sample(val_sample, args.max_side); val_target, val_points, val_labels = sample_positive_points(val_mask, seed=args.seed + val_sample.image.__hash__() % 100000)
                        with device_context(device):
                            val_logits, val_scores = predictor_forward(predictor, val_image, val_points, val_labels); _, val_iou = loss_and_iou(val_logits, val_scores, torch.from_numpy(val_target[None]).to(device)); vals.append(float(val_iou))
                predictor.model.train(True); predictor.model.image_encoder.eval(); mean_iou = float(np.mean(vals))
                if mean_iou > best:
                    best = mean_iou; torch.save({"model": predictor.model.state_dict(), "fold": fold["fold"], "best_val_iou": best, "config": vars(args)}, out / f"fold_{fold['fold']}_best.pt")
                print(f"fold={fold['fold']} step={step} val_iou={mean_iou:.4f}", flush=True)
        all_metrics.append({"fold": fold["fold"], "val_patient": fold["val_patients"][0], "best_val_iou": best})
    (out / "summary.json").write_text(json.dumps({"architecture": "sam2.1_hiera_small", "metrics": all_metrics, "config": vars(args)}, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="../dataset/processed"); p.add_argument("--test-patient", default="p5"); p.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_small.pt"); p.add_argument("--model-config", default=SAM2_MODEL_CONFIG); p.add_argument("--output-dir", default="runs/sam2_small"); p.add_argument("--folds", type=int, default=4); p.add_argument("--steps", type=int, default=600); p.add_argument("--eval-every", type=int, default=100); p.add_argument("--accumulation-steps", type=int, default=8); p.add_argument("--learning-rate", type=float, default=5e-5); p.add_argument("--max-side", type=int, default=1024); p.add_argument("--seed", type=int, default=42); p.add_argument("--execute", action="store_true"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(); root = Path(args.dataset_root).expanduser()
    if not root.is_absolute():
        cwd_candidate = (Path.cwd() / root).resolve()
        root = cwd_candidate if cwd_candidate.is_dir() else (Path(__file__).resolve().parent / root).resolve()
    checkpoint = Path(args.checkpoint).expanduser()
    checkpoint = checkpoint if checkpoint.is_absolute() else (Path.cwd() / checkpoint).resolve()
    if args.execute and not args.dry_run: execute(args, root, checkpoint)
    else: dry_run(root, args.test_patient, checkpoint, args.folds, args.max_side)


if __name__ == "__main__": main()
