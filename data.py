from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Sample:
    patient: str
    image: str
    mask: str


def discover(dataset_root: str | Path) -> list[Sample]:
    root = Path(dataset_root).expanduser().resolve()
    samples: list[Sample] = []
    for patient_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("p")):
        patient = patient_dir.name
        image_dirs = [patient_dir / f"{patient}_images", patient_dir / "images"]
        mask_dirs = [patient_dir / f"{patient}_masks", patient_dir / "masks"]
        image_dir = next((p for p in image_dirs if p.is_dir()), None)
        mask_dir = next((p for p in mask_dirs if p.is_dir()), None)
        if image_dir is None or mask_dir is None:
            continue
        images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
        masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"}
        for stem in sorted(images.keys() & masks.keys()):
            samples.append(Sample(patient, str(images[stem]), str(masks[stem])))
    if not samples:
        raise FileNotFoundError(f"No paired patient data found under {root}")
    return samples


def patient_counts(samples: Iterable[Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.patient] = counts.get(sample.patient, 0) + 1
    return dict(sorted(counts.items()))


def split(samples: list[Sample], test_patient: str) -> tuple[list[Sample], list[Sample]]:
    train = [s for s in samples if s.patient != test_patient]
    test = [s for s in samples if s.patient == test_patient]
    if not test:
        raise ValueError(f"Unknown or empty test patient: {test_patient}")
    if not train:
        raise ValueError("At least one training patient is required")
    return train, test


def patient_cv_folds(samples: list[Sample], seed: int = 42) -> list[dict]:
    """Leave-one-patient-out folds. The held-out test patient never enters here."""
    patients = sorted({s.patient for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patients)
    folds = []
    for index, val_patient in enumerate(patients):
        train_patients = [p for p in patients if p != val_patient]
        folds.append({
            "fold": index,
            "train_patients": sorted(train_patients),
            "val_patients": [val_patient],
            "train_samples": [asdict(s) for s in samples if s.patient in train_patients],
            "val_samples": [asdict(s) for s in samples if s.patient == val_patient],
        })
    return folds


def write_manifest(samples: list[Sample], output: str | Path, test_patient: str, seed: int) -> dict:
    train, test = split(samples, test_patient)
    manifest = {
        "dataset_root": str(Path(samples[0].image).parents[2]),
        "seed": seed,
        "test_patient": test_patient,
        "patient_counts": patient_counts(samples),
        "train_patients": sorted({s.patient for s in train}),
        "test_samples": [asdict(s) for s in test],
        "cv_folds": patient_cv_folds(train, seed),
    }
    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def balanced_weights(samples: list[Sample]) -> np.ndarray:
    counts = patient_counts(samples)
    return np.asarray([1.0 / counts[s.patient] for s in samples], dtype=np.float64)
