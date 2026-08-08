#!/bin/bash
#SBATCH --job-name=acap_eval_full
#SBATCH --partition=${ACAP_PARTITION}
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Full-test-set generation is slow; leave the cap off on partitions that allow
# it (override below if your cluster requires one).
#SBATCH --output=${ACAP_LOG_DIR}/eval_full_%j.log
#SBATCH --error=${ACAP_LOG_DIR}/eval_full_%j.err

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
echo "A-CAP Full Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
echo "=============================================="

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== Starting Eval in Container ==="
# CHECKPOINT / SPLIT / OUT can be set in the submitting environment; defaults
# are checkpoint_best.pt / test / eval_results_test.json (see run_eval.sh).
enroot start \
    --root \
    --rw \
    --mount "${ACAP_DATA_MOUNT}:${ACAP_DATA_MOUNT}:none:bind" \
    --mount "${ACAP_HF_CACHE}:/root/.cache/huggingface:none:bind" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env HF_TOKEN=${HF_TOKEN:-} \
    --env HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} \
    --env CHECKPOINT=${CHECKPOINT:-checkpoints/acap_vist/checkpoint_best.pt} \
    --env SPLIT=${SPLIT:-test} \
    --env OUT=${OUT:-eval_results_test.json} \
    ${ACAP_CONTAINER} \
    bash "${ACAP_PROJECT_DIR}/scripts/run_eval.sh"

echo "=============================================="
echo "Job ${SLURM_JOB_ID} finished: $(date)"
echo "=============================================="
