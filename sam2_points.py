"""Mask-derived point prompts following the LearnOpenCV SAM2 recipe."""
from __future__ import annotations

import random

import cv2
import numpy as np


def sample_positive_points(mask: np.ndarray, seed: int | None = None, erosion_kernel: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return binary target, XY positive points, and point labels.

    One point is sampled for every non-background label present in the source
    mask, from an eroded union of all foreground labels. This mirrors the
    strategy described in the referenced LearnOpenCV article and avoids
    placing prompts on annotation boundaries.
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D indexed mask, got {mask.shape}")
    labels = np.unique(mask)
    foreground_labels = [int(label) for label in labels if int(label) != 0]
    binary = (mask > 0).astype(np.uint8)
    if not foreground_labels:
        raise ValueError("Mask contains no foreground label")
    kernel = np.ones((erosion_kernel, erosion_kernel), dtype=np.uint8)
    interior = cv2.erode(binary, kernel, iterations=1)
    coords = np.argwhere(interior > 0)
    if len(coords) == 0:
        coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        raise ValueError("Foreground disappeared while sampling prompt points")
    rng = random.Random(seed)
    points = np.asarray([[int(yx[1]), int(yx[0])] for _ in foreground_labels for yx in [coords[rng.randrange(len(coords))]]], dtype=np.float32)
    point_labels = np.ones((len(points),), dtype=np.int32)
    return binary.astype(np.float32), points, point_labels


def sample_class_prompts(mask: np.ndarray, seed: int | None = None, erosion_kernel: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one positive point and one binary target for each foreground class."""
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D indexed mask, got {mask.shape}")
    class_ids = np.asarray([int(label) for label in np.unique(mask) if int(label) != 0], dtype=np.int64)
    if len(class_ids) == 0:
        raise ValueError("Mask contains no foreground label")
    rng = random.Random(seed)
    kernel = np.ones((erosion_kernel, erosion_kernel), dtype=np.uint8)
    targets = []
    points = []
    for class_id in class_ids:
        target = (mask == class_id).astype(np.uint8)
        interior = cv2.erode(target, kernel, iterations=1)
        coords = np.argwhere(interior > 0)
        if len(coords) == 0:
            coords = np.argwhere(target > 0)
        if len(coords) == 0:
            raise ValueError(f"Class {class_id} disappeared while sampling prompt points")
        y, x = coords[rng.randrange(len(coords))]
        targets.append(target.astype(np.float32))
        points.append([int(x), int(y)])
    return (
        np.stack(targets),
        np.asarray(points, dtype=np.float32),
        np.ones((len(points),), dtype=np.int32),
        class_ids,
    )


def compose_semantic_prediction(class_probabilities: np.ndarray, class_ids: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Compose class-conditioned probability maps into one indexed mask."""
    if class_probabilities.ndim != 3:
        raise ValueError(f"Expected probabilities shaped [classes,H,W], got {class_probabilities.shape}")
    if len(class_probabilities) != len(class_ids):
        raise ValueError("class_probabilities and class_ids must have the same length")
    prediction = np.zeros(class_probabilities.shape[1:], dtype=np.uint8)
    best = np.full(class_probabilities.shape[1:], float(threshold), dtype=np.float32)
    for probability, class_id in zip(class_probabilities, class_ids):
        update = probability > best
        prediction[update] = int(class_id)
        best[update] = probability[update]
    return prediction
