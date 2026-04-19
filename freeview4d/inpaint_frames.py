"""cv2.inpaint the person out of each frame using SAM2 masks (dilated).

Multi-view WorldMirror fusion over these cleaned frames recovers the true background
because in different frames the person occupies different pixels.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dilate", type=int, default=21)
    ap.add_argument("--radius", type=int, default=9)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(args.frames_dir).glob("*.jpg"))
    masks = sorted(Path(args.masks_dir).glob("*.png"))
    assert len(frames) == len(masks), f"count mismatch: {len(frames)} frames vs {len(masks)} masks"

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate, args.dilate))
    for i, (fp, mp) in enumerate(zip(frames, masks)):
        img = cv2.imread(str(fp))
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_dil = cv2.dilate(mask, kernel, iterations=1)
        inpainted = cv2.inpaint(img, mask_dil, args.radius, cv2.INPAINT_TELEA)
        cv2.imwrite(str(out / f"{i:05d}.jpg"), inpainted)
        print(f"  {i:05d}  mask={int((mask > 0).mean() * 100)}%  dilated={int((mask_dil > 0).mean() * 100)}%")

    print(f"[inpaint] wrote {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
