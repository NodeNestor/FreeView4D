# Example: DAVIS breakdance-flare

The quickstart demo uses the DAVIS 2017 **breakdance-flare** clip, bundled with MoSca's demo data at `deps/MoSca/demo/breakdance-flare/images/` after running `setup/install.sh`.

- **Source**: [DAVIS Challenge 2017](https://davischallenge.org/)
- **Content**: 71 frames, 854×480, one person performing a breakdance flare move against a stone wall and cobblestone courtyard background
- **License**: CC BY-NC 4.0 (attribution, non-commercial)

To run the demo:

```bash
bash scripts/quickstart.sh
```

This subsamples 16 of the 71 frames, clicks SAM2 at pixel (450, 260) to track the dancer, and produces the `time_only`, `spacetime`, and `static_only` videos in `output/breakdance_demo/render/`.

## Expected runtime

- 4060 8 GB: ~50 seconds total
- 4090 / A100: ~20 seconds total

## Why this clip is good for the demo

- Dramatic motion (arms/legs swinging through space) — visually clear that the dynamic decomposition is working
- Mostly-static background (stone wall + cobblestones + flowers) — the static 3DGS is easy to see
- Single subject — one SAM2 click tracks everything
- Short enough to fit in an 8 GB VRAM budget at full `target_size=448`

## Citation

If you use this clip in a publication, cite DAVIS:

```bibtex
@article{pont2017_davis,
  title={The 2017 DAVIS Challenge on Video Object Segmentation},
  author={Pont-Tuset, Jordi and Perazzi, Federico and Caelles, Sergi and Arbel\'aez, Pablo and Sorkine-Hornung, Alexander and {Van Gool}, Luc},
  journal={arXiv:1704.00675},
  year={2017}
}
```
