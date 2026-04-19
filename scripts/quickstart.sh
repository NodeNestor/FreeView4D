#!/bin/bash
# FreeView4D quickstart — runs the full pipeline on MoSca's bundled
# DAVIS breakdance-flare clip and produces the demo videos.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="deps/MoSca/demo/breakdance-flare/images"
if [ ! -d "$SRC" ]; then
    echo "Error: $SRC not found. Run setup/install.sh first."
    exit 1
fi

echo "Running FreeView4D on breakdance-flare (71 source frames, subsample to 16)..."
bash scripts/run_pipeline.sh \
    --input "$SRC" \
    --output "output/breakdance_demo" \
    --click_x 450 --click_y 260 \
    --n_frames 16 \
    --target_size 448

echo
echo "Quickstart finished. Videos:"
ls -la output/breakdance_demo/render/*.mp4
