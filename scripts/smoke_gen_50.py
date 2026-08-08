import os
import torch, pickle
from config import Config
from models.acap import ACap
from data.precomputed_dataset import precomputed_collate_fn

cfg = Config()
m = ACap(
    num_input_images=cfg.vist.num_input_images,
    num_detected_per_image=cfg.model.num_detected_per_image,
    num_forecasted=cfg.model.num_forecasted,
    num_rois_per_image=cfg.model.num_rois_per_image,
    word_seq_length=cfg.model.word_seq_length,
    embed_dim=cfg.model.embed_dim,
    hidden_dim=cfg.model.hidden_dim,
    num_gat_layers=cfg.model.num_gat_layers,
    num_gat_heads=cfg.model.num_gat_heads,
    dropout=cfg.model.dropout,
    bert_model_name=cfg.model.bert_model_name,
    vinvl_model_name=cfg.model.vinvl_model_name,
    device="cuda",
    use_gnn=cfg.model.use_gnn,
    use_context=cfg.model.use_context,
    roi_feature_dim=cfg.model.roi_feature_dim,
    freeze_vinvl=cfg.model.freeze_vinvl,
    mlm_mask_prob=cfg.model.mlm_mask_prob,
).to("cuda")
ck = torch.load("checkpoints/acap_vist/checkpoint_best.pt", map_location="cuda", weights_only=False)
m.load_state_dict(ck["model_state_dict"], strict=False)
m.eval()
print(f"Loaded checkpoint (epoch 1, val ppl ~19.7)")

samples = pickle.load(open(os.environ.get("ACAP_PRECOMPUTED_DIR","data/vist/preprocessed") + "/test_preprocessed.pkl", "rb"))[:4]
batch = precomputed_collate_fn(samples)
with torch.no_grad():
    caps = m.generate_caption(precomputed=batch, max_length=35)
for c, t in zip(caps, samples):
    print(f"GEN: {repr(c)[:120]}")
    print(f"TGT: {repr(t['target_caption'])[:120]}")
    print()
