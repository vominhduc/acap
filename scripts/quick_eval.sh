#!/bin/bash
set -euo pipefail

# --- Container-internal config (override via env vars for your setup) ---
ACAP_PROJECT_DIR="${ACAP_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ACAP_SCRATCH="${ACAP_SCRATCH:-/tmp}"
ACAP_PACKAGES="${ACAP_PACKAGES:-${ACAP_SCRATCH}/acap_packages}"
ACAP_UV_ARCHIVE="${ACAP_UV_ARCHIVE:-/root/.cache/uv/archive-v0/P91JaJeY1jLXWzwf62GhB/lib/python3.10/site-packages}"

PROJECT_DIR="${ACAP_PROJECT_DIR}"
EXTRA_PACKAGES="${ACAP_PACKAGES}"

export PYTHONPATH="${EXTRA_PACKAGES}:${ACAP_UV_ARCHIVE}:${PROJECT_DIR}"
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export HF_TOKEN=${HF_TOKEN:-}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export CUDA_VISIBLE_DEVICES=0
export PATH="/root/.local/bin:${PATH}"

echo "=============================================="
echo "A-CAP Quick Eval Test"
echo "Date: $(date)"
echo "=============================================="

cd ${PROJECT_DIR}

python3 -c "
import json
import logging
import torch
from data.vist_dataset import VISTDataset, vist_collate_fn
from torch.utils.data import DataLoader, Subset
from models.acap import ACap
from metrics import MetricEvaluator
from config import Config, ModelConfig, TrainConfig, VISTConfig
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device: {device}')

config = Config()

# Load model
logger.info('Loading model...')
model = ACap(
    num_input_images=config.vist.num_input_images,
    num_detected_per_image=config.model.num_detected_per_image,
    num_forecasted=config.model.num_forecasted,
    num_rois_per_image=config.model.num_rois_per_image,
    word_seq_length=config.model.word_seq_length,
    embed_dim=config.model.embed_dim,
    hidden_dim=config.model.hidden_dim,
    num_gat_layers=config.model.num_gat_layers,
    num_gat_heads=config.model.num_gat_heads,
    dropout=config.model.dropout,
    bert_model_name=config.model.bert_model_name,
    vinvl_model_name=config.model.vinvl_model_name,
    device=str(device),
    use_gnn=config.model.use_gnn,
    use_context=config.model.use_context,
    roi_feature_dim=config.model.roi_feature_dim,
).to(device)

checkpoint_path = 'checkpoints/acap_vist/checkpoint_epoch_4.pt'
logger.info(f'Loading checkpoint: {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']
state_dict = {k: v for k, v in state_dict.items() if not k.startswith('vinvl._embeddings')}
model.load_state_dict(state_dict, strict=False)
model.eval()
logger.info(f'Checkpoint loaded. global_step={checkpoint.get(\"global_step\")}, best_score={checkpoint.get(\"best_score\")}')

# Load small test subset
logger.info('Loading test dataset...')
dataset = VISTDataset(
    ann_file=config.vist.test_ann_file,
    image_root=config.vist.image_root,
    num_input_images=config.vist.num_input_images,
    target_sentence_index=config.vist.target_sentence_index,
    is_test=True,
)
logger.info(f'Full test set: {len(dataset)} samples')

# Use only 20 samples for quick test
test_indices = list(range(min(20, len(dataset))))
subset = Subset(dataset, test_indices)
loader = DataLoader(subset, batch_size=4, shuffle=False, num_workers=2, collate_fn=vist_collate_fn)

# Generate captions
logger.info('Generating captions...')
all_generated = []
all_targets = []
all_oracle_images = []

with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        logger.info(f'  Batch {batch_idx+1}/{len(loader)}...')
        input_images = batch['input_images'].to(device)
        target_captions = batch['target_caption']
        oracle_pil = batch.get('oracle_image_pil', [])

        generated = model.generate_caption(input_images)
        all_generated.extend(generated)
        all_targets.extend(target_captions)
        all_oracle_images.extend(oracle_pil)

        for i, (gen, tgt) in enumerate(zip(generated, target_captions)):
            logger.info(f'  Sample {batch_idx*4+i}:')
            logger.info(f'    Generated: {gen}')
            logger.info(f'    Target:    {tgt}')

# Compute metrics
logger.info('Computing metrics...')
metric_eval = MetricEvaluator(device=str(device))
metrics = metric_eval.compute_all_metrics(all_generated, all_targets, all_oracle_images)

logger.info(f'')
logger.info(f'========== RESULTS (20 samples) ==========')
for k, v in metrics.items():
    logger.info(f'  {k}: {v:.4f}')
logger.info(f'==========================================')
"
