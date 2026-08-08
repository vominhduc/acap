#!/bin/bash
# Full A-CAP evaluation over the test split using the precomputed pipeline.
# Runs inside the enroot container (invoked by eval_full_slurm.sh).
set -euo pipefail

# --- Container-internal config (override via env vars for your setup) ---
ACAP_PROJECT_DIR="${ACAP_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ACAP_SCRATCH="${ACAP_SCRATCH:-/tmp}"
ACAP_PACKAGES="${ACAP_PACKAGES:-${ACAP_SCRATCH}/acap_packages}"
ACAP_DETECTRON2="${ACAP_DETECTRON2:-${ACAP_SCRATCH}/detectron2}"
ACAP_UV_ARCHIVE="${ACAP_UV_ARCHIVE:-/root/.cache/uv/archive-v0/P91JaJeY1jLXWzwf62GhB/lib/python3.10/site-packages}"

PROJECT_DIR="${ACAP_PROJECT_DIR}"
EXTRA_PACKAGES="${ACAP_PACKAGES}"

export PYTHONPATH="${ACAP_DETECTRON2}:${EXTRA_PACKAGES}:${ACAP_UV_ARCHIVE}:${PROJECT_DIR}"
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export HF_TOKEN=${HF_TOKEN:-}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=0
export PATH="/root/.local/bin:${PATH}"

# Default to the best checkpoint; allow override via CHECKPOINT env var.
CHECKPOINT="${CHECKPOINT:-checkpoints/acap_vist/checkpoint_best.pt}"
SPLIT="${SPLIT:-test}"
OUT="${OUT:-eval_results_${SPLIT}.json}"

cd "${PROJECT_DIR}"

echo "=============================================="
echo "A-CAP Full Evaluation"
echo "Checkpoint: ${CHECKPOINT}"
echo "Split: ${SPLIT}"
echo "Date: $(date)"
echo "=============================================="

# Fail fast if the decoder can't load (same guard as training).
python3 -c "
from config import Config
from models.vinvl_wrapper import VinVLWrapper
cfg = Config()
w = VinVLWrapper(model_name=cfg.model.vinvl_model_name)
print('vinvl OK ->', type(w.backbone).__name__)
" || { echo "FATAL: VinVL decoder failed to load. Aborting."; exit 1; }

python3 eval.py \
    --checkpoint "${CHECKPOINT}" \
    --data-root data/vist \
    --batch-size 16 \
    --device cuda \
    --split "${SPLIT}" \
    --output "${OUT}"

echo "=============================================="
echo "Eval complete: $(date)"
echo "=============================================="
