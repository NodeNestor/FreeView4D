# Architecture

## Design goal

Take a monocular casually-captured video of a mostly-static scene with one (or a few) moving object(s) and produce a **4D representation** navigable in **both space and time** — all with **no per-scene optimization**.

## Why static/dynamic decomposition

Most video content is mostly-static. Running full 4D optimization (deformable Gaussians, motion-basis fields, 4D primitives) over the whole scene is wasteful — the walls, floor, and buildings don't move. Only a small fraction of pixels (the moving object) actually needs time-dependent modeling.

Splitting the scene into `static + dynamic(t)` lets us:

- Use **a feed-forward multi-view-to-3DGS prior** (WorldMirror 2.0) for the static part → static 3DGS in 1–10 seconds, no per-scene training
- Handle the dynamic part with a **much smaller** per-frame representation (point cloud or local 3DGS) → cheap to compute, cheap to composite
- Composite at render time as one rasterization pass using **gsplat**

This is the same decomposition used (with different implementations) in DynaSplat, DAS3R, DreamScene4D, DeGauss, and the broader dynamic-scene literature — see the survey linked from the README.

## Pipeline

```
input video / frames directory
        │
        ▼  subsample to N frames, renumber as zero-padded
frames[T]
        │
        ├── [SAM2 video predictor]
        │     One point click on the moving object in frame 0.
        │     SAM2 propagates the mask temporally to frames 1..T-1.
        │     → masks[T]
        │
        ├── [cv2 inpaint]
        │     For each frame, dilate the mask and run cv2.INPAINT_TELEA
        │     to erase the moving object with a plausible background guess.
        │     → clean_frames[T]
        │
        ├── [WorldMirror 2.0 on clean_frames]
        │     Feed-forward multi-view reconstruction over the N frames.
        │     Because the person was in a different position in each frame,
        │     the true background is visible in SOME frame for most pixels,
        │     and WorldMirror's multi-view fusion recovers it.
        │     → static_3DGS (N × 50K–100K gaussians typical)
        │       + per-frame depth[T] + per-frame cam[T]  (unused from this pass)
        │
        ├── [WorldMirror 2.0 on original frames]
        │     Same model, same N frames, but WITH the moving object visible.
        │     We use this pass ONLY for the per-frame depth+cam that lets us
        │     unproject the person into world coordinates.
        │     → depth_orig[T] + cam_orig[T]  (static gaussians discarded)
        │
        ├── [Unproject(depth_orig ∩ mask) → world coords]
        │     For each frame t, take the pixels under mask[t], read their
        │     depth from depth_orig[t], and unproject to world coords using
        │     cam_orig[t] (WorldMirror uses camera-to-world extrinsics).
        │     → dyn_points[t] = (N_t × 3) world-space points + colors
        │
        └── [gsplat composite renderer]
              For each render sample (view V, time t):
                  frame_pixels = rasterize(static_3DGS + dyn(t), V)
              Produces:
                  A. fixed-camera time sweep (the cleanest demo of the time axis)
                  B. smooth camera orbit that advances time simultaneously
                  C. static-only orbit (reference, no dynamic content)
```

## Why we run WorldMirror twice

We need two different depth maps per frame:

1. **Background depth** (from inpainted frames) → builds the static 3DGS
2. **Foreground (object) depth** (from original frames) → locates the object in 3D

Running WorldMirror twice is the cleanest way to get both without interfering. Both passes use the same N frames and produce compatible coordinate systems because:
- Both normalize camera 0 to the world origin
- The non-person image content is near-identical between passes, so camera estimation is near-identical

For shaving one WorldMirror pass, an alternative is:
- Run WorldMirror once on the original frames
- Project every Gaussian to every frame's image plane
- Drop Gaussians whose projected pixel lies inside `mask[t]` at a depth close to the person's depth at that pixel
- Keep Gaussians that are behind the person (true background observed in other frames)

This would give one clean static 3DGS in a single pass but requires a more careful visibility test. The two-pass version is ~30 seconds total on a 4060 so we picked simplicity.

## Why the dynamic is a point cloud, not a 3DGS

A per-frame dynamic representation can be:

1. **Raw point cloud** (what this repo does) — rendered as uniform-size colored Gaussians at render time. Fast, lightweight, no fitting needed.
2. **Fitted 3DGS per frame** — run a short gsplat optimization per frame against that frame's RGB. Higher quality, but meaningful compute cost × T frames.
3. **Canonical 3DGS + deformation field** — fit one 3DGS of the object once, learn a motion field that warps it through time. This is Shape-of-Motion territory and needs substantial per-scene optimization.

We pick (1) because it keeps the pipeline feed-forward. It's a reasonable tradeoff for a strong visual demo and useful as initialization for (2) or (3).

## Where the black holes come from

If the moving object occludes the same chunk of background in *every* input frame, WorldMirror has never "seen" that background region. cv2 inpaint can only guess at texture based on surrounding pixels — for large occluded regions the guess is poor and multi-view fusion doesn't converge on a believable static geometry there.

Solutions in order of cost:

- **Use more frames** spread across a wider time range → more chances to observe any given pixel
- **Use a smarter inpainter** — Stable Diffusion / LaMa / VACE video diffusion → coherent fill that multi-view consensus agrees on
- **Observation: this is exactly the "unseen (view, time) coverage hole"** that full 4DGS methods like FreeView4D's original iterative-inpaint loop address. A follow-up iteration of this repo could wire VACE in here.

## Comparison to other 4DGS approaches

| Method | Dynamic representation | Per-scene optimization | Feed-forward prior |
|---|---|---|---|
| **FreeView4D (this repo)** | per-frame point cloud | ❌ none | ✅ WorldMirror + SAM2 |
| **Shape of Motion** | motion-basis × canonical 3DGS | ✅ hours | partial (depth priors, tracks) |
| **MoSca** | motion-scaffold deformation | ✅ hours | partial |
| **Deformable 3DGS** | canonical + deformation MLP | ✅ hours | ❌ |
| **4D-GS (HexPlane)** | canonical + spatio-temporal plane decomposition | ✅ hours | ❌ |
| **SpaceTime Gaussians** | native 4D Gaussian primitives | ✅ hours | ❌ |

The feed-forward prior is what makes this repo interesting: you get a usable 4D demo in under a minute on a consumer GPU, which was a days-long per-scene optimization in the original 4DGS literature. The output is **not as temporally smooth** as Shape of Motion, but it's dramatically cheaper to produce and serves as a strong initializer for any of the per-scene optimization methods above.

## Coordinate conventions

- **Extrinsics in `camera_params.json`** are 4×4 **camera-to-world** matrices (WorldMirror's convention — see the docstring in `hyworldmirror/models/utils/geometry.py::depth_to_world_coords_points`)
- **gsplat's `viewmats`** are **world-to-camera** matrices. Our renderer inverts c2w → w2c before passing to gsplat
- **Intrinsics** are 3×3, in pixels, for the image resolution that WorldMirror downsampled to (`target_size`)
- **Depth maps** are saved at the same downsampled resolution as `.npy` float32 arrays; values are in the same world-space units as the Gaussian positions (WorldMirror self-normalizes per-scene)

## Hardware budget (4060 8 GB, typical)

| Step | Time | Peak VRAM |
|---|---|---|
| SAM2 video propagation (16 frames) | 4 s | 0.9 GB |
| cv2 inpaint (16 frames) | <1 s | 0 GB |
| WorldMirror clean pass (16 frames × 448) | 16 s | 5 GB |
| WorldMirror dynamic pass (16 frames × 448) | 16 s | 5 GB |
| Dynamic unprojection (16 frames) | <1 s | 0 GB |
| gsplat composite render (76 frames) | 15 s | 4 GB |
| **Total** | **~50 s** | **5 GB peak** |

Scaling to 24 frames requires `target_size=352` on the 4060 (activation memory scales quadratically with total token count in the multi-view attention).
