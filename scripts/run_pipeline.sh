#!/bin/bash
# FreeView4D end-to-end pipeline:
#   video (or frames dir) → SAM2 → inpaint → WorldMirror × 2 → extract dynamic → render 4D
#
# Usage:
#   bash scripts/run_pipeline.sh \
#       --input  path/to/frames_dir  (directory of numeric-named jpgs)  OR  --video path.mp4
#       --output output/scene_name
#       --click_x 480 --click_y 260
#       [--n_frames 16]
#       [--target_size 448]
#       [--gpu 0]

set -e

# Defaults
N_FRAMES=16
TARGET_SIZE=448
GPU=0
CLICK_X=""
CLICK_Y=""
INPUT=""
VIDEO=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) INPUT="$2"; shift 2;;
        --video) VIDEO="$2"; shift 2;;
        --output) OUTPUT="$2"; shift 2;;
        --click_x) CLICK_X="$2"; shift 2;;
        --click_y) CLICK_Y="$2"; shift 2;;
        --n_frames) N_FRAMES="$2"; shift 2;;
        --target_size) TARGET_SIZE="$2"; shift 2;;
        --gpu) GPU="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$OUTPUT" || -z "$CLICK_X" || -z "$CLICK_Y" ]]; then
    echo "Usage: $0 --output DIR --click_x X --click_y Y [--input FRAMES_DIR | --video VIDEO.mp4] [--n_frames N] [--target_size S]"
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate freeview4d

mkdir -p "$OUTPUT"

# --- Step 1: if given a video, extract frames; else copy + renumber frames ---
FRAMES_DIR="$OUTPUT/frames"
rm -rf "$FRAMES_DIR" && mkdir -p "$FRAMES_DIR"
if [[ -n "$VIDEO" ]]; then
    echo "[1/6] Extracting $N_FRAMES frames from $VIDEO..."
    TOTAL=$(python - <<EOF
import imageio.v2 as iio
print(iio.get_reader("$VIDEO").count_frames())
EOF
)
    STRIDE=$(( TOTAL / N_FRAMES ))
    [[ $STRIDE -lt 1 ]] && STRIDE=1
    python - <<EOF
import imageio.v2 as iio
from PIL import Image
r = iio.get_reader("$VIDEO")
total = r.count_frames()
stride = max(1, total // $N_FRAMES)
out = 0
for i in range(0, total, stride):
    if out >= $N_FRAMES: break
    frame = r.get_data(i)
    Image.fromarray(frame).save(f"$FRAMES_DIR/{out:05d}.jpg", quality=95)
    out += 1
print(f"extracted {out} frames (stride={stride})")
EOF
elif [[ -n "$INPUT" ]]; then
    echo "[1/6] Staging $N_FRAMES frames from $INPUT..."
    src_files=($(ls "$INPUT" | sort))
    total=${#src_files[@]}
    stride=$(( total / N_FRAMES ))
    [[ $stride -lt 1 ]] && stride=1
    i=0
    for ((k=0; k<total && i<N_FRAMES; k+=stride)); do
        cp "$INPUT/${src_files[$k]}" "$(printf "$FRAMES_DIR/%05d.jpg" $i)"
        i=$((i+1))
    done
    echo "staged $i frames (stride=$stride)"
else
    echo "Error: must pass either --video or --input"
    exit 1
fi

# --- Step 2: SAM2 video predictor ---
echo
echo "[2/6] SAM2 video predictor (single-click propagation)..."
cd "$ROOT/deps/sam2"
python -m freeview4d.sam2_video \
    --frames_dir "$FRAMES_DIR" \
    --output_dir "$OUTPUT/sam2" \
    --click_x "$CLICK_X" --click_y "$CLICK_Y"
cd "$ROOT"

# --- Step 3: cv2 inpaint person out of each frame ---
echo
echo "[3/6] cv2 inpaint person out of each frame..."
python -m freeview4d.inpaint_frames \
    --frames_dir "$FRAMES_DIR" \
    --masks_dir "$OUTPUT/sam2/masks" \
    --output_dir "$OUTPUT/frames_inpainted" \
    --dilate 21 --radius 9

# --- Step 4a: WorldMirror on CLEAN frames -> static 3DGS ---
echo
echo "[4a/6] WorldMirror on inpainted frames (static 3DGS)..."
cd "$ROOT/deps/HY-World-2.0"
python -m hyworld2.worldrecon.pipeline \
    --input_path "$OUTPUT/frames_inpainted" \
    --output_path "$OUTPUT/wm_static" \
    --enable_bf16 \
    --target_size "$TARGET_SIZE" \
    --video_max_frames "$N_FRAMES" 2>&1 | tail -15
STATIC_RUN="$(ls -td "$OUTPUT/wm_static/frames_inpainted/"*/ | head -1)"

# --- Step 4b: WorldMirror on ORIGINAL frames -> depth+cam for dynamic ---
echo
echo "[4b/6] WorldMirror on original frames (dynamic depth+cam)..."
python -m hyworld2.worldrecon.pipeline \
    --input_path "$FRAMES_DIR" \
    --output_path "$OUTPUT/wm_dynamic" \
    --enable_bf16 \
    --target_size "$TARGET_SIZE" \
    --video_max_frames "$N_FRAMES" 2>&1 | tail -15
DYN_RUN="$(ls -td "$OUTPUT/wm_dynamic/frames/"*/ | head -1)"
cd "$ROOT"

# --- Step 5: extract per-frame dynamic points ---
echo
echo "[5/6] Extract per-frame dynamic point clouds..."
python -m freeview4d.extract_dynamic \
    --wm_run "$DYN_RUN" \
    --sam2_dir "$OUTPUT/sam2" \
    --frames_dir "$FRAMES_DIR" \
    --output_dir "$OUTPUT/dynamic"

# --- Step 6: 4D composite render ---
echo
echo "[6/6] Render 4D composite..."
python -m freeview4d.render_4d \
    --wm_run "$STATIC_RUN" \
    --static_ply "${STATIC_RUN}gaussians.ply" \
    --dyn_dir "$OUTPUT/dynamic" \
    --output_dir "$OUTPUT/render" \
    --width 448 --height 252 \
    --n_per_seg 4 \
    --dyn_scale 0.008

echo
echo "======================================"
echo "  Done. Outputs in $OUTPUT/render/"
echo "    time_only.mp4     — fixed camera, time sweeps"
echo "    spacetime.mp4     — camera orbits + time advances"
echo "    static_only.mp4   — clean static reference orbit"
echo "======================================"
