#!/bin/bash
#SBATCH --job-name=acap_preprocess
#SBATCH --partition=002-partition-default
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# No time limit — ConceptNet/WordNet 2-hop queries make this much slower
# than the original (empty-cache) preprocessing.
#SBATCH --output=/lustre/users/vmduc/Projects/acap/logs/preprocess_%j.log
#SBATCH --error=/lustre/users/vmduc/Projects/acap/logs/preprocess_%j.err

set -euo pipefail

# --- Cluster-specific config (override these env vars for your cluster) ---
ACAP_PROJECT_DIR="${ACAP_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ACAP_DATA_MOUNT="${ACAP_DATA_MOUNT:-${ACAP_PROJECT_DIR}}"
ACAP_HF_CACHE="${ACAP_HF_CACHE:-$HOME/.cache/huggingface}"
ACAP_SCRATCH="${ACAP_SCRATCH:-/tmp}"
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
echo "A-CAP Preprocessing Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
echo "=============================================="

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Starting Preprocessing in Container ==="
enroot start \
    --root \
    --rw \
    --mount "${ACAP_DATA_MOUNT}:${ACAP_DATA_MOUNT}:none:bind" \
    --mount "${ACAP_HF_CACHE}:/root/.cache/huggingface:none:bind" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env SPLITS="${SPLITS:-train val test}" \
    --env ACAP_PRECOMPUTED_DIR="${ACAP_PRECOMPUTED_DIR:-${ACAP_SCRATCH}/acap_data}" \
    ${ACAP_CONTAINER} \
    bash "${ACAP_PROJECT_DIR}/scripts/preprocess_in_container.sh"

echo "=============================================="
echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "=============================================="
