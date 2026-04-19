#!/bin/bash
# FreeView4D one-shot installer.
#   1. Clones HY-World 2.0, SAM2, MoSca (demo data) into deps/
#   2. Applies a patch to HY-World that makes flash-attn truly optional
#   3. Creates a conda env `freeview4d` with PyTorch 2.4.0 + cu124
#   4. Installs gsplat, sam2, hyworld2 requirements, and our own code
#   5. Downloads the SAM2.1 tiny checkpoint
#
# Requires: conda, git, wget, a CUDA 12.4+ driver, Linux or WSL Ubuntu.
# WorldMirror 2.0 weights auto-download on first run via huggingface_hub.

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# 1. Clone external deps
# ---------------------------------------------------------------------------
mkdir -p deps
cd deps

if [ ! -d HY-World-2.0 ]; then
    echo "[1/5] Cloning HY-World 2.0..."
    git clone --depth 1 https://github.com/Tencent-Hunyuan/HY-World-2.0
fi
if [ ! -d sam2 ]; then
    echo "[1/5] Cloning SAM 2..."
    git clone --depth 1 https://github.com/facebookresearch/sam2
fi
if [ ! -d MoSca ]; then
    echo "[1/5] Cloning MoSca (demo data only)..."
    git clone --depth 1 https://github.com/JiahuiLei/MoSca
fi
cd "$ROOT"

# ---------------------------------------------------------------------------
# 2. Patch HY-World to make flash-attn import optional
# ---------------------------------------------------------------------------
echo
echo "[2/5] Patching HY-World attention.py..."
PATCH="$ROOT/setup/patches/hyworld_attention.patch"
TARGET="deps/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/layers/attention.py"
if grep -q "_HAS_FLASH" "$TARGET"; then
    echo "  already patched, skipping"
else
    ( cd deps/HY-World-2.0 && patch -p1 < "$PATCH" )
    echo "  patched."
fi

# ---------------------------------------------------------------------------
# 3. Create conda env
# ---------------------------------------------------------------------------
echo
echo "[3/5] Creating conda env 'freeview4d'..."

# Make conda scripts work in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Accept Anaconda ToS for default channels (conda 26+ requires this)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true

if ! conda env list | grep -q "^freeview4d "; then
    conda create -n freeview4d python=3.10 -y
fi
conda activate freeview4d
python --version

# ---------------------------------------------------------------------------
# 4. Install Python deps
# ---------------------------------------------------------------------------
echo
echo "[4/5] Installing PyTorch 2.4.0 + cu124..."
pip install --quiet torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

echo "[4/5] Installing HY-World 2.0 requirements (gsplat, pycolmap, open3d, uniception, ...)..."
pip install --quiet -r deps/HY-World-2.0/requirements.txt

echo "[4/5] Installing SAM 2 (editable, --no-deps to avoid bumping torch)..."
pip install -e deps/sam2 --no-deps --quiet
pip install --quiet hydra-core iopath

echo "[4/5] Installing FreeView4D and its extras..."
pip install -e . --quiet

python - <<'PY'
import torch
import gsplat  # noqa: F401
import sam2    # noqa: F401
print(f"  torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
PY

# ---------------------------------------------------------------------------
# 5. Download SAM2 tiny checkpoint
# ---------------------------------------------------------------------------
echo
echo "[5/5] Downloading SAM 2.1 tiny checkpoint..."
mkdir -p deps/sam2/checkpoints
if [ ! -f deps/sam2/checkpoints/sam2.1_hiera_tiny.pt ]; then
    wget -q -O deps/sam2/checkpoints/sam2.1_hiera_tiny.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
fi
ls -lh deps/sam2/checkpoints/sam2.1_hiera_tiny.pt

echo
echo "========================================================================"
echo "  FreeView4D installed. Try the demo:"
echo "      conda activate freeview4d"
echo "      bash scripts/quickstart.sh"
echo "========================================================================"
echo
echo "  NOTE: HY-World 2.0 is under the Tencent Community License which"
echo "        EXCLUDES the EU, UK, and South Korea from its Territory."
echo "        See NOTICES.md before distributing or using commercially."
echo "========================================================================"
