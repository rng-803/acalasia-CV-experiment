#!/usr/bin/env bash
set -euo pipefail

# Run this in the CUDA cloud environment, not on the local development host.
CHECKPOINT_DIR="${1:-checkpoints}"
mkdir -p "$CHECKPOINT_DIR"
CHECKPOINT_PATH="$CHECKPOINT_DIR/sam2.1_hiera_small.pt"
PARTIAL_PATH="$CHECKPOINT_PATH.partial"
trap 'rm -f "$PARTIAL_PATH"' EXIT
curl -fL --retry 3 \
  -o "$PARTIAL_PATH" \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
mv "$PARTIAL_PATH" "$CHECKPOINT_PATH"
echo "Downloaded $CHECKPOINT_PATH"
