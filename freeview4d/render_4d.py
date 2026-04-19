"""4D compositing renderer.

Loads a static 3DGS PLY (Inria format) + per-frame dynamic point clouds,
then renders:
    A. time_only.mp4    — fixed camera at input pose 0, time advances 0..T-1
    B. spacetime.mp4    — smooth orbit through all input poses, time advances
    C. static_only.mp4  — orbit with no dynamic content (clean static reference)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization
from scipy.spatial.transform import Rotation as R, Slerp

SH_C0 = 0.28209479177387814


def load_inria_ply(path):
    pd = PlyData.read(str(path))["vertex"]
    means = np.stack([pd["x"], pd["y"], pd["z"]], axis=1).astype(np.float32)
    scales = np.exp(
        np.stack([pd["scale_0"], pd["scale_1"], pd["scale_2"]], axis=1)
    ).astype(np.float32)
    quats = np.stack([pd["rot_0"], pd["rot_1"], pd["rot_2"], pd["rot_3"]], axis=1).astype(np.float32)
    opac = 1.0 / (1.0 + np.exp(-pd["opacity"].astype(np.float32)))
    rgb = 0.5 + SH_C0 * np.stack([pd["f_dc_0"], pd["f_dc_1"], pd["f_dc_2"]], axis=1).astype(np.float32)
    rgb = np.clip(rgb, 0.0, 1.0)
    return means, quats, scales, opac, rgb


def points_to_gaussians(pts, cols, scale=0.008, opacity=0.95):
    N = pts.shape[0]
    means = pts.astype(np.float32)
    scales = np.full((N, 3), scale, dtype=np.float32)
    quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (N, 1))
    opac = np.full((N,), opacity, dtype=np.float32)
    rgb = cols.astype(np.float32)
    return means, quats, scales, opac, rgb


def invert_cam_to_world(E_c2w):
    R_m = E_c2w[:, :3, :3]
    t = E_c2w[:, :3, 3]
    Rt = np.transpose(R_m, (0, 2, 1))
    t_inv = -np.einsum("bij,bj->bi", Rt, t)
    out = np.tile(np.eye(4, dtype=np.float32), (E_c2w.shape[0], 1, 1))
    out[:, :3, :3] = Rt
    out[:, :3, 3] = t_inv
    return out


def interp_c2w(E_list, n_per_seg):
    out = [E_list[0]]
    for i in range(len(E_list) - 1):
        T0, T1 = E_list[i], E_list[i + 1]
        R0, R1 = R.from_matrix(T0[:3, :3]), R.from_matrix(T1[:3, :3])
        slerp = Slerp([0, 1], R.concatenate([R0, R1]))
        for j in range(1, n_per_seg + 1):
            t = j / n_per_seg
            Rt = slerp([t])[0].as_matrix()
            tt = (1 - t) * T0[:3, 3] + t * T1[:3, 3]
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rt
            T[:3, 3] = tt
            out.append(T)
    return out


def render_composite(static, dyn_t, viewmat_w2c, K, W, H, device="cuda"):
    sm, sq, ss, so, sc = static
    dm, dq, ds, do, dc = dyn_t
    means = torch.from_numpy(np.concatenate([sm, dm], axis=0)).to(device)
    quats = torch.from_numpy(np.concatenate([sq, dq], axis=0)).to(device)
    scales = torch.from_numpy(np.concatenate([ss, ds], axis=0)).to(device)
    opac = torch.from_numpy(np.concatenate([so, do], axis=0)).to(device)
    rgb = torch.from_numpy(np.concatenate([sc, dc], axis=0)).to(device)
    vm = torch.from_numpy(viewmat_w2c[None]).to(device)
    Ks = torch.from_numpy(K[None]).to(device)
    img, _, _ = rasterization(
        means, quats, scales, opac, rgb, vm, Ks, W, H, render_mode="RGB"
    )
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_run", required=True)
    ap.add_argument("--dyn_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--static_ply", default="")
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=252)
    ap.add_argument("--n_per_seg", type=int, default=4)
    ap.add_argument("--dyn_scale", type=float, default=0.008)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    static_ply = args.static_ply or str(Path(args.wm_run) / "gaussians.ply")
    print(f"[load] static {static_ply}")
    static = load_inria_ply(static_ply)
    print(f"  {static[0].shape[0]:,} static gaussians")

    dyn_npz = sorted(Path(args.dyn_dir).glob("dyn_[0-9][0-9].npz"))
    dyn_gauss = []
    for f in dyn_npz:
        d = np.load(f)
        dyn_gauss.append(points_to_gaussians(d["points"], d["colors"], scale=args.dyn_scale))
    T = len(dyn_gauss)
    print(f"  {T} dynamic snapshots, {sum(g[0].shape[0] for g in dyn_gauss):,} person points")

    cams = json.load(open(Path(args.wm_run) / "camera_params.json"))
    extr_c2w = np.stack([np.array(e["matrix"]) for e in cams["extrinsics"]]).astype(np.float32)
    intr = np.stack([np.array(k["matrix"]) for k in cams["intrinsics"]]).astype(np.float32)
    K0 = intr[0]

    # Mode A: fixed camera, time sweep
    print("[render] mode A: fixed camera, time sweep")
    vm_A = invert_cam_to_world(extr_c2w[0:1])[0]
    frames_A = []
    for t in range(T):
        img = render_composite(static, dyn_gauss[t], vm_A, K0, args.width, args.height)
        frames_A.append(img)
        Image.fromarray(img).save(out / f"timeA_{t:02d}.png")
    with imageio.get_writer(out / "time_only.mp4", fps=6, codec="libx264", macro_block_size=1) as w:
        for f in frames_A:
            w.append_data(f)
    print(f"  -> {out / 'time_only.mp4'}")

    # Mode B: space + time sweep
    print("[render] mode B: space + time sweep")
    c2w_path = interp_c2w([extr_c2w[i] for i in range(T)], args.n_per_seg)
    N_path = len(c2w_path)
    vm_path = invert_cam_to_world(np.stack(c2w_path))
    frames_B = []
    for i in range(N_path):
        t_float = i * (T - 1) / max(1, N_path - 1)
        t_idx = int(np.clip(round(t_float), 0, T - 1))
        img = render_composite(static, dyn_gauss[t_idx], vm_path[i], K0, args.width, args.height)
        frames_B.append(img)
        Image.fromarray(img).save(out / f"spacetime_{i:03d}.png")
    with imageio.get_writer(out / "spacetime.mp4", fps=12, codec="libx264", macro_block_size=1) as w:
        for f in frames_B:
            w.append_data(f)
    print(f"  -> {out / 'spacetime.mp4'} ({N_path} frames)")

    # Mode C: static only
    print("[render] mode C: static only")
    empty_dyn = (
        np.zeros((0, 3), np.float32), np.zeros((0, 4), np.float32),
        np.zeros((0, 3), np.float32), np.zeros((0,), np.float32),
        np.zeros((0, 3), np.float32),
    )
    with imageio.get_writer(out / "static_only.mp4", fps=12, codec="libx264", macro_block_size=1) as w:
        for i in range(N_path):
            w.append_data(
                render_composite(static, empty_dyn, vm_path[i], K0, args.width, args.height)
            )
    print(f"[done] all outputs in {out}")


if __name__ == "__main__":
    main()
