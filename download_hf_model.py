#!/usr/bin/env python3
"""Cache a Hugging Face model locally for offline experiment execution."""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--model", default="nvidia/mit-b0"); p.add_argument("--revision", default=None)
    args = p.parse_args()
    path = snapshot_download(repo_id=args.model, revision=args.revision)
    print(f"Cached {args.model} at {path}")


if __name__ == "__main__": main()
