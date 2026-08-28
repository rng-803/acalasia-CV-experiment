#!/usr/bin/env python3
"""Validate data discovery and experiment split without importing training frameworks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import discover, patient_counts, split, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="../dataset/processed")
    parser.add_argument("--test-patient", default="p5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default="experiment_manifest.json", help="Manifest output path; parent directories are created when possible")
    args = parser.parse_args()
    samples = discover(args.dataset_root)
    train, test = split(samples, args.test_patient)
    manifest = write_manifest(samples, args.manifest, args.test_patient, args.seed)
    print(json.dumps({
        "status": "ok",
        "mode": "dry-run",
        "total_samples": len(samples),
        "patient_counts": patient_counts(samples),
        "train_samples": len(train),
        "test_samples": len(test),
        "test_patient": args.test_patient,
        "cv_folds": [{"fold": f["fold"], "train_patients": f["train_patients"], "val_patients": f["val_patients"]} for f in manifest["cv_folds"]],
        "manifest": str(Path(args.manifest).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
