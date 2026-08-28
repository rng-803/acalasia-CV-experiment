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
