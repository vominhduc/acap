#!/bin/bash
set -euo pipefail

# --- Container-internal config (override via env vars for your setup) ---
ACAP_PROJECT_DIR="${ACAP_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ACAP_SCRATCH="${ACAP_SCRATCH:-/tmp}"
ACAP_PACKAGES="${ACAP_PACKAGES:-${ACAP_SCRATCH}/acap_packages}"
ACAP_DETECTRON2="${ACAP_DETECTRON2:-${ACAP_SCRATCH}/detectron2}"
ACAP_UV_ARCHIVE="${ACAP_UV_ARCHIVE:-/root/.cache/uv/archive-v0/P91JaJeY1jLXWzwf62GhB/lib/python3.10/site-packages}"
ACAP_PRECOMPUTED_DIR="${ACAP_PRECOMPUTED_DIR:-${ACAP_SCRATCH}/acap_data}"

PROJECT_DIR="${ACAP_PROJECT_DIR}"
EXTRA_PACKAGES="${ACAP_PACKAGES}"

export PYTHONPATH="${ACAP_DETECTRON2}:${EXTRA_PACKAGES}:${ACAP_UV_ARCHIVE}:${PROJECT_DIR}"
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
# HF_TOKEN is passed through from the SLURM submit env (see train_slurm.sh).
export HF_TOKEN=${HF_TOKEN:-}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=0
export PATH="/root/.local/bin:${PATH}"
export UV_LINK_MODE=copy

echo "=============================================="
echo "A-CAP Training (Precomputed Pipeline)"
echo "Python: $(python3 --version)"
echo "Date: $(date)"
echo "=============================================="

echo "=== Step 1: Install CUDA-compatible PyTorch ==="
python3 -c "import torch; print(f'torch {torch.__version__}'); assert torch.cuda.is_available()" 2>&1 || {
    echo "Installing CUDA 12.x PyTorch..."
    uv pip install --target ${EXTRA_PACKAGES} --python /usr/bin/python3 \
        torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>&1
}
python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python3 -c "import torchvision; print(f'torchvision {torchvision.__version__}')"

echo "=== Step 2: Install Missing Packages ==="
python3 -c "from pycocoevalcap.bleu.bleu import Bleu; print('pycocoevalcap OK')" 2>&1 || {
    uv pip install --target ${EXTRA_PACKAGES} --python /usr/bin/python3 --no-deps pycocoevalcap pycocotools 2>&1
}
python3 -c "from pycocoevalcap.bleu.bleu import Bleu; print('pycocoevalcap OK')" 2>&1
python3 -c "from pycocotools.coco import COCO; print('pycocotools OK')" 2>&1
python3 -c "import requests; print(f'requests {requests.__version__}')" 2>&1 || {
    uv pip install --target ${EXTRA_PACKAGES} --python /usr/bin/python3 requests urllib3 2>&1
}
python3 -c "import requests; print(f'requests {requests.__version__}')" 2>&1

echo "=== Step 3: Verify Precomputed Data ==="
cd ${PROJECT_DIR}

# Precomputed pkls live under ACAP_PRECOMPUTED_DIR. Use a large-file-friendly
# scratch path; data/vist/preprocessed/ can suffer filesystem corruption on the
# ~15GB train pkl on some shared filesystems. config.vist.precomputed_dir must
# match (set ACAP_PRECOMPUTED_DIR there too).
PRECOMP_DIR="${ACAP_PRECOMPUTED_DIR}"

# Retry for up to a few minutes before declaring failure — a bind-mount
# inside the container can briefly report ENOENT on a specific file right
# after start.
verify_precomputed() {
    local split="$1"
    local tries=0
    while [ $tries -lt 30 ]; do
        if [ -f "${PRECOMP_DIR}/${split}_preprocessed.pkl" ]; then
            return 0
        fi
        echo "  waiting for ${split}_preprocessed.pkl in ${PRECOMP_DIR} ... [try $tries]"
        sleep 10
        tries=$((tries + 1))
    done
    return 1
}

for split in train val test; do
    if ! verify_precomputed "${split}"; then
        echo "ERROR: Precomputed data not found: ${PRECOMP_DIR}/${split}_preprocessed.pkl"
        echo "  Run preprocessing first: sbatch scripts/preprocess_slurm.sh"
        exit 1
    fi
done

for split in train val test; do
    SIZE=$(python3 -c "import pickle; print(len(pickle.load(open('${PRECOMP_DIR}/${split}_preprocessed.pkl', 'rb'))))")
    echo "  ${split}: ${SIZE} preprocessed samples"
done

echo "=== Step 4: Verify Imports ==="
python3 -c "from config import Config; print('config OK')"
python3 -c "from models.acap import ACap; print('acap OK')"
python3 -c "from data.precomputed_dataset import build_precomputed_dataloaders; print('dataset OK')"
python3 -c "from metrics import MetricEvaluator; print('metrics OK')"

echo "=== Step 4b: Verify VinVL decoder loads ==="
python3 -c "
from config import Config
from models.vinvl_wrapper import VinVLWrapper
cfg = Config()
w = VinVLWrapper(model_name=cfg.model.vinvl_model_name)
print('vinvl OK ->', type(w.backbone).__name__, '| from', cfg.model.vinvl_model_name)
" \
    || { echo "FATAL: VinVL decoder failed to load. If the VinVL decoder dir (ACAP_VINVL_MODEL) is missing, run scripts/setup_vinvl.sh first (see README). Aborting before the long run."; exit 1; }

echo "=== Step 5: Training ==="
python3 train.py \
    --data-root data/vist \
    --batch-size 16 \
    --lr 3e-5 \
    --epochs 10 \
    --device cuda \
    --output-dir checkpoints

echo "=============================================="
echo "Training Complete: $(date)"
echo "=============================================="
