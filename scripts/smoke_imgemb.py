import torch
from config import Config
from models.acap import ACap

cfg = Config()
print(f"freeze_vinvl={cfg.model.freeze_vinvl} mlm_mask_prob={cfg.model.mlm_mask_prob}")

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

total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Model: {total:,} total, {trainable:,} trainable")

# Check img_embedding loaded
print(f"img_embedding weight shape: {tuple(m.vinvl.img_embedding.weight.shape)}")
print(f"img_embedding requires_grad: {m.vinvl.img_embedding.weight.requires_grad}")
print(f"feat_proj weight shape: {tuple(m.vinvl.feat_proj.weight.shape)}")
print(f"feat_proj requires_grad: {m.vinvl.feat_proj.weight.requires_grad}")

# Forward smoke test with random data
bs, n_nodes, n_roi = 2, 40, 100
node_embs = [torch.randn(n_nodes, 1536, device="cuda") for _ in range(bs)]
edge_indices = [torch.randint(0, n_nodes, (2, 10), device="cuda") for _ in range(bs)]
roi = torch.randn(bs, n_roi, 1024, device="cuda")
precomputed = {
    "node_embeddings": node_embs,
    "edge_index": edge_indices,
    "roi_features": roi,
}
caps = ["a man playing baseball .", "two children at the park ."]
out = m(precomputed=precomputed, target_captions=caps)
print(f"logits: {tuple(out['logits'].shape)}, labels: {tuple(out['labels'].shape)}")

# Generation smoke test
gen = m.generate_caption(precomputed=precomputed, max_length=35)
print(f"Generated: {gen[:2]}")
print("SMOKE OK")
