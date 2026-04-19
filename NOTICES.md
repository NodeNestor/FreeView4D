# Third-party notices

FreeView4D orchestrates several independently-licensed projects. This document summarizes who owns what and which license governs each piece. **Nothing below replaces the upstream license texts — always consult the original project.**

## Our wrapper code — Apache License 2.0

Everything under:
- `freeview4d/`
- `scripts/`
- `setup/` (except the patch files, which document changes to HY-World code)
- `configs/`
- `docs/`
- `examples/`

is copyright © 2026 the FreeView4D contributors and released under the Apache License 2.0 (see `LICENSE`).

## Upstream dependencies (cloned at install time into `deps/`)

| Project | Upstream | License | Notes |
|---|---|---|---|
| Tencent HY-World 2.0 / WorldMirror 2.0 | [Tencent-Hunyuan/HY-World-2.0](https://github.com/Tencent-Hunyuan/HY-World-2.0) | [Tencent HY-World 2.0 Community License](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt) | **Territory-restricted** — see below |
| Meta SAM 2.1 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | Apache 2.0 | — |
| MoSca | [JiahuiLei/MoSca](https://github.com/JiahuiLei/MoSca) | MIT | Demo data only (breakdance-flare, duck) |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | Apache 2.0 | Installed as a pip wheel |
| DAVIS dataset | [davischallenge.org](https://davischallenge.org/) | CC BY-NC 4.0 | The breakdance-flare clip originates here. Attributed in-place. |

## HY-World 2.0 Territory restriction

Tencent's Community License grants use **only within a defined Territory** that **explicitly excludes the European Union, the United Kingdom, and South Korea** (Section 1(l), 5(c)). If you reside or operate in one of those jurisdictions, Tencent has not granted you a license to use the model under this agreement.

Section 5(b) additionally prohibits using HY-World 2.0 outputs to train any other AI model. Section 3(d) requires that any distribution includes the notice:

> Tencent HY-WORLD 2.0 is licensed under the Tencent HY-WORLD 2.0 Community License Agreement, Copyright © 2026 Tencent. All Rights Reserved. The trademark rights of "Tencent HY" are owned by Tencent or its affiliate.

This notice is included in compliance with Section 3(d).

Products built with HY-World 2.0 should be marked "Powered by Tencent HY" (Section 3(c)).

## Our modifications to HY-World 2.0 code

Per Section 3(b), modifications must be prominently marked. Our modifications to HY-World 2.0 live in `setup/patches/` as standalone `.patch` files and are applied by `setup/install.sh` *after* cloning the upstream repo. The patches are:

- `hyworld_attention.patch` — makes the `flash_attn` import truly optional in `hyworldmirror/models/layers/attention.py` by falling back to `torch.nn.functional.scaled_dot_product_attention` when neither flash-attn v2 nor v3 is installed. No model logic changes; the behavior difference is that bf16/fp16 attention goes through SDPA instead of flash-attn kernels (numerically equivalent, somewhat slower).

The modified files in the upstream clone carry comments stating that they were changed, per the community license terms.

## Acceptable Use Policy (HY-World 2.0)

Per Section 5(a), users of this repository agree to abide by Tencent's Acceptable Use Policy, which prohibits using HY-World 2.0 for: violent extremism, targeting minors, disseminating malware, impersonation, high-stakes automated decisions in safety/rights domains, military applications, etc. See the full list in the upstream license file.

## Swapping to a fully permissive backend

If you need a stack usable in all jurisdictions and without the downstream-training restriction, replace `freeview4d/worldmirror_runner.py` with a driver for:

- [VGGT](https://github.com/facebookresearch/vggt) (Apache 2.0) — feed-forward multi-view → 3D, similar interface to WorldMirror
- [DUSt3R / MASt3R](https://github.com/naver/mast3r) (research license — check terms)

The rest of the pipeline (SAM2, cv2 inpaint, unprojection, gsplat compositing) is unaffected.
