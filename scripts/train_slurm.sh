#!/bin/bash
#SBATCH --job-name=acap_train
#SBATCH --partition=${ACAP_PARTITION}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
# Exclude nodes that can't reach the OST holding val data (set via
# ACAP_EXCLUDE_NODES, e.g. "srdgx00129"). Empty by default.
#SBATCH --exclude=${ACAP_EXCLUDE_NODES}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# No wall-clock limit on partitions that allow it; override ACAP_LOG_DIR if
# you want logs elsewhere.
#SBATCH --output=${ACAP_LOG_DIR}/train_%j.log
#SBATCH --error=${ACAP_LOG_DIR}/train_%j.err

set -euo pipefail

# --- Cluster-specific config (override these env vars for your cluster) ---
ACAP_PROJECT_DIR="${ACAP_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ACAP_DATA_MOUNT="${ACAP_DATA_MOUNT:-${ACAP_PROJECT_DIR}}"   # host path bind-mounted into the container
ACAP_HF_CACHE="${ACAP_HF_CACHE:-$HOME/.cache/huggingface}"  # host HF cache to mount
ACAP_SCRATCH="${ACAP_SCRATCH:-/tmp}"                        # enroot runtime/cache/data scratch
ACAP_CONTAINER="${ACAP_CONTAINER:-acap}"
ACAP_SLURM_CONF="${ACAP_SLURM_CONF:-}"
ACAP_SLURM_BIN="${ACAP_SLURM_BIN:-}"
export SLURM_CONF="${ACAP_SLURM_CONF:-${SLURM_CONF:-}}"
export PATH="${ACAP_SLURM_BIN:+${ACAP_SLURM_BIN}/}${PATH:+:${PATH}}"
export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-${ACAP_SCRATCH}/enroot_runtime}"
export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-${ACAP_SCRATCH}/enroot_cache}"
export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-${ACAP_SCRATCH}/enroot_data}"

mkdir -p "${ACAP_PROJECT_DIR}/logs"

echo "=============================================="
echo "A-CAP Training Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
echo "=============================================="

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Enroot Container ==="
enroot list

echo "=== Starting Training in Container ==="
# The frozen VinVL decoder is loaded from a local safetensors dir
# (config.model.vinvl_model_name -> ACAP_VINVL_MODEL), so no HF_TOKEN is
# required for it. HF_TOKEN only helps bert/roberta rate limits.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "NOTE: HF_TOKEN unset (optional; only affects HF Hub rate limits for bert/roberta)."
fi
enroot start \
    --root \
    --rw \
    --mount "${ACAP_DATA_MOUNT}:${ACAP_DATA_MOUNT}:none:bind" \
    --mount "${ACAP_HF_CACHE}:/root/.cache/huggingface:none:bind" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_TOKEN=${HF_TOKEN:-} \
    --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} \
    ${ACAP_CONTAINER} \
    bash "${ACAP_PROJECT_DIR}/scripts/train_in_container.sh"

echo "=============================================="
echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "=============================================="
