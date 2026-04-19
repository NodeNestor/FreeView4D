# Running FreeView4D on your own video

## Good inputs

The pipeline works best on clips that look like:

- **5–15 seconds long** — a dozen or so time steps capture most interesting motion
- **One primary moving object** (person, animal, vehicle) against a largely static background
- **Some camera motion is fine** — drone orbit, slow pan, hand-held walking around. But not wild shaky-cam.
- **Object is always fully visible** — the tracker needs to see it in every frame
- **Reasonable lighting** — drastic exposure changes between frames will hurt WorldMirror's multi-view consistency

## Running on an mp4

```bash
bash scripts/run_pipeline.sh \
    --video path/to/your_clip.mp4 \
    --output output/your_scene \
    --click_x 480 --click_y 300 \
    --n_frames 16 --target_size 448
```

- `--click_x --click_y`: one pixel coordinate inside the moving object **as it appears in the first frame** of your clip. Open the first extracted frame (after running once), note the pixel, and pass those values.
- `--n_frames`: 16 is the sweet spot on an 8 GB GPU at `target_size=448`. You can push to 24 if you drop `target_size` to 352.
- `--target_size`: 448 is the WorldMirror default; lower = less VRAM, 384 works on an 8 GB card for 24 frames.

## Running on a pre-extracted frames directory

If you already have frames (numeric-named JPGs work best: `00000.jpg`, `00001.jpg`, ...):

```bash
bash scripts/run_pipeline.sh \
    --input path/to/frames_dir \
    --output output/your_scene \
    --click_x 480 --click_y 300 \
    --n_frames 16
```

The pipeline will subsample them evenly.

## Iterating on the click point

If SAM2 grabs the wrong thing from your click, re-run with a different `--click_x --click_y` to pick a more distinctive pixel inside the moving object (e.g., not on a thin limb edge). SAM2 is fast (~4 s), and the pipeline caches nothing, so you'll need to rerun from step 1.

To check whether SAM2 got the right object, inspect `$OUTPUT/sam2/overlays/00000.png` after the first run — it shows the red-tinted mask over frame 0.

## Multi-object (not yet — manual workaround)

The current pipeline assumes one moving object per click. For two moving objects:

1. Run the pipeline once per object with its own `--click_x/y` and a unique `--output` dir
2. Merge the per-object `dyn_*.npz` files and render with `freeview4d/render_4d.py` pointed at the union

Proper multi-object support (one click per object, one SAM2 session with multiple `obj_id`s) is a ~20-line patch in `freeview4d/sam2_video.py` — PRs welcome.

## Troubleshooting

- **OOM during WorldMirror** — drop `--target_size` to 384 or 320, drop `--n_frames` to 12
- **Everything's black / empty render** — SAM2 didn't pick up the object from your click. Inspect the overlays in `$OUTPUT/sam2/overlays/`.
- **Person looks ghostly / shifted** — the dynamic unprojection depends on WorldMirror's depth being accurate at the object. For very thin limbs or very small objects, lower `--target_size` actually hurts.
- **"CUDA error: no kernel image"** — your GPU is newer than what PyTorch 2.4.0 cu124 supports (Blackwell / `sm_120+`). Either use a 4090/A100-and-older card or upgrade PyTorch (breaks the pinned gsplat wheel; needs a newer gsplat).

## Weird corner cases

- **Moving camera + static scene, no moving object**: the pipeline still runs but SAM2 will track whatever you clicked. If you want pure static reconstruction, use WorldMirror directly (bypass this pipeline — see `deps/HY-World-2.0/README.md`).
- **Everything moves, nothing static** (e.g., a car interior shot going around a bend): the "static" 3DGS will smear badly because the premise breaks. This repo is not the right tool.
