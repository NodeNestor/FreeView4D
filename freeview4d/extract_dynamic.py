"""Extract per-frame dynamic point cloud (moving object) by unprojecting
pixels under the SAM2 mask using WorldMirror's per-frame depth + camera.

Dependencies:
    - WorldMirror run directory with `depth/`, `camera_params.json`
    - SAM2 masks directory
    - Original frames directory (for color sampling)

Outputs per-frame .npz files {points, colors} in world coordinates
(using WorldMirror's camera-to-world extrinsics convention) plus a
combined all-frames PLY for inspection.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

# Find HY-World repo to import the geometry utility
HY_WORLD_PATH = os.environ.get(
    "HY_WORLD_PATH",
    str(Path(__file__).resolve().parents[1] / "deps" / "HY-World-2.0"),
)
sys.path.insert(0, HY_WORLD_PATH)
from hyworld2.worldrecon.hyworldmirror.models.utils.geometry import (  # noqa: E402
    depth_to_world_coords_points,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_run", required=True, help="WorldMirror run dir (has depth/, camera_params.json)")
    ap.add_argument("--sam2_dir", required=True, help="SAM2 dir (has masks/)")
    ap.add_argument("--frames_dir", required=True, help="Original frames for RGB sampling")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    depth_dir = Path(args.wm_run) / "depth"
    mask_dir = Path(args.sam2_dir) / "masks"
    frames_dir = Path(args.frames_dir)

    cams = json.load(open(Path(args.wm_run) / "camera_params.json"))
    extrinsics = np.stack([np.array(e["matrix"]) for e in cams["extrinsics"]]).astype(np.float32)
    intrinsics = np.stack([np.array(k["matrix"]) for k in cams["intrinsics"]]).astype(np.float32)

    depth_files = sorted(depth_dir.glob("depth_*.npy"))
    mask_files = sorted(mask_dir.glob("*.png"))
    frame_files = sorted(frames_dir.glob("*.jpg"))
    n = len(depth_files)
    assert n == len(mask_files) == len(frame_files) == extrinsics.shape[0], (
        f"count mismatch: depths={n} masks={len(mask_files)} "
        f"frames={len(frame_files)} cams={extrinsics.shape[0]}"
    )
    print(f"[extract] n={n} frames")

    all_pts, all_cols, counts = [], [], []
    for t in range(n):
        depth = np.load(depth_files[t]).astype(np.float32)
        H, W = depth.shape

        mask = np.array(Image.open(mask_files[t])).astype(bool)
        if mask.shape != (H, W):
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
            ).astype(bool)

        rgb = np.array(Image.open(frame_files[t]).convert("RGB"))
        if rgb.shape[:2] != (H, W):
            rgb = np.array(Image.fromarray(rgb).resize((W, H), Image.BILINEAR))
        rgb = rgb.astype(np.float32) / 255.0

        dt = torch.from_numpy(depth)[None]
        et = torch.from_numpy(extrinsics[t])[None]
        kt = torch.from_numpy(intrinsics[t])[None]
        world, _, valid = depth_to_world_coords_points(dt, et, kt)
        world = world[0].numpy()
        valid = valid[0].numpy()

        sel = mask & valid
        pts, cols = world[sel], rgb[sel]
        print(f"  t={t:02d} depth={depth.shape} mask_px={mask.sum()} kept={sel.sum()}")

        np.savez(
            out / f"dyn_{t:02d}.npz",
            points=pts.astype(np.float32),
            colors=cols.astype(np.float32),
        )
        all_pts.append(pts)
        all_cols.append(cols)
        counts.append(int(sel.sum()))

    combo_pts = np.concatenate(all_pts, axis=0)
    combo_cols = np.concatenate(all_cols, axis=0)
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.empty(len(combo_pts), dtype=dtype)
    verts["x"], verts["y"], verts["z"] = combo_pts[:, 0], combo_pts[:, 1], combo_pts[:, 2]
    verts["red"], verts["green"], verts["blue"] = (combo_cols * 255).astype(np.uint8).T
    PlyData([PlyElement.describe(verts, "vertex")]).write(str(out / "dyn_all_frames.ply"))

    with open(out / "summary.json", "w") as f:
        json.dump(
            {"n_frames": n, "counts": counts, "total_points": int(combo_pts.shape[0])},
            f, indent=2,
        )
    print(f"[extract] {combo_pts.shape[0]} total dynamic points, saved per-frame + combined ply")


if __name__ == "__main__":
    main()
