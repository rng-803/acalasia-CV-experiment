#!/usr/bin/env bash
set -euo pipefail

# Run this in the CUDA cloud environment, not on the local development host.
CHECKPOINT_DIR="${1:-checkpoints}"
mkdir -p "$CHECKPOINT_DIR"
curl -fL --retry 3 \
  -o "$CHECKPOINT_DIR/sam2.1_hiera_small.pt" \
  "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_small.pt"
echo "Downloaded $CHECKPOINT_DIR/sam2.1_hiera_small.pt"
