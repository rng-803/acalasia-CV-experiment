#!/usr/bin/env python3
"""Build a compact comparison table from completed experiment directories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--runs-root", default="runs/experiments"); p.add_argument("--output", default="runs/experiment_comparison.csv"); args = p.parse_args()
    rows = []
    for summary_path in sorted(Path(args.runs_root).glob("*/summary.json")):
        summary = json.loads(summary_path.read_text()); values = {row["metric"]: row for row in summary.get("aggregate_metrics", [])}
        rows.append({"run_name": summary.get("run_name", summary_path.parent.name), "architecture": summary.get("architecture"), "test_patient": summary.get("test_patient"), "foreground_macro_dice_mean": values.get("foreground_macro_dice", {}).get("mean"), "foreground_macro_dice_std": values.get("foreground_macro_dice", {}).get("std"), "foreground_macro_iou_mean": values.get("foreground_macro_iou", {}).get("mean"), "foreground_macro_iou_std": values.get("foreground_macro_iou", {}).get("std"), "run_dir": str(summary_path.parent)})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_name", "architecture", "test_patient", "foreground_macro_dice_mean", "foreground_macro_dice_std", "foreground_macro_iou_mean", "foreground_macro_iou_std", "run_dir"]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} runs to {output}")


if __name__ == "__main__": main()
