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
export CUDA_VISIBLE_DEVICES=0
export PATH="/root/.local/bin:${PATH}"

echo "=============================================="
echo "A-CAP Preprocessing: Concepts + Features + KG"
echo "Date: $(date)"
echo "=============================================="

cd ${PROJECT_DIR}

python3 -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"

echo "=== Running Preprocessing ==="
# Write precomputed pkls to ACAP_PRECOMPUTED_DIR (a large-file-friendly scratch
# path). data/vist/preprocessed/ can suffer filesystem corruption on the ~15GB
# train pkl on some shared filesystems. config.vist.precomputed_dir must match
# (set ACAP_PRECOMPUTED_DIR there too).
PRECOMP_DIR="${ACAP_PRECOMPUTED_DIR}"
mkdir -p ${PRECOMP_DIR}
python3 preprocess_concepts.py \
    --data-root data/vist \
    --output-dir ${PRECOMP_DIR} \
    --device cuda \
    --splits ${SPLITS:-train val test}

echo "=============================================="
echo "Preprocessing Complete: $(date)"
echo "=============================================="
