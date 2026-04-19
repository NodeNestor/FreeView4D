"""SAM2 video predictor: propagate a single-click person mask through all staged frames.

Expects `frames_dir` to contain zero-padded numeric JPGs (e.g., 00000.jpg, 00001.jpg, ...)
as SAM2's video loader requires numeric filenames.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


DEFAULT_CKPT = "deps/sam2/checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--cfg", default=DEFAULT_CFG)
    ap.add_argument("--click_x", type=int, required=True)
    ap.add_argument("--click_y", type=int, required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)

    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device="cuda")
    print("[sam2] model loaded")

    frame_names = sorted(f for f in os.listdir(args.frames_dir) if f.endswith((".jpg", ".png")))
    print(f"[sam2] {len(frame_names)} frames in {args.frames_dir}")

    state = predictor.init_state(video_path=args.frames_dir)
    predictor.reset_state(state)

    pts = np.array([[args.click_x, args.click_y]], dtype=np.float32)
    lbls = np.array([1], dtype=np.int32)
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1, points=pts, labels=lbls
    )

    masks_per_frame = {}
    for fi, _, logits in predictor.propagate_in_video(state):
        masks_per_frame[fi] = (logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8)

    for fi, mask in masks_per_frame.items():
        fname = frame_names[fi]
        img = np.array(Image.open(Path(args.frames_dir) / fname).convert("RGB"))
        H, W, _ = img.shape
        if mask.shape != (H, W):
            mask = np.array(
                Image.fromarray(mask * 255).resize((W, H), Image.NEAREST)
            ) > 0
            mask = mask.astype(np.uint8)
        Image.fromarray(mask * 255).save(out / "masks" / fname.replace(".jpg", ".png"))

        overlay = img.copy()
        mb = mask.astype(bool)
        overlay[mb] = (0.5 * overlay[mb] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(out / "overlays" / fname.replace(".jpg", ".png"))

    with open(out / "meta.json", "w") as f:
        json.dump({"frames": frame_names, "click": [args.click_x, args.click_y]}, f, indent=2)

    print(f"[sam2] saved {len(masks_per_frame)} masks to {out}")
    print(f"[vram] peak={torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
