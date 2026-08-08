import os
import torch, re, sys
sys.path.insert(0, os.environ.get("ACAP_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ckpt = torch.load("/root/.cache/huggingface/hub/models--michelecafagna26--vinvl_vg_x152c4/snapshots/6259db93d704633db4c572ea4ea54e2d2fbb7f7e/vinvl_vg_x152c4.pth", map_location="cpu", weights_only=False)
sd = ckpt["model"] if "model" in ckpt else ckpt
new_sd = {}
for k, v in sd.items():
    nk = k.replace("module.", "")
    if not (nk.startswith("backbone.") or nk.startswith("rpn.") or nk.startswith("roi_heads.")):
        continue
    nk = nk.replace("backbone.body.layer1", "backbone.body.res2")
    nk = nk.replace("backbone.body.layer2", "backbone.body.res3")
    nk = nk.replace("backbone.body.layer3", "backbone.body.res4")
    nk = nk.replace("stem.bn1", "stem.norm")
    nk = nk.replace("downsample.0.", "shortcut.")
    nk = nk.replace("downsample.1.", "shortcut.norm.")
    nk = re.sub(r"\.bn1\.", ".norm1.", nk)
    nk = re.sub(r"\.bn2\.", ".norm2.", nk)
    nk = re.sub(r"\.bn3\.", ".norm3.", nk)
    nk = nk.replace("rpn.head.cls_logits", "proposal_generator.rpn_head.objectness_logits")
    nk = nk.replace("rpn.head.bbox_pred", "proposal_generator.rpn_head.anchor_deltas")
    nk = nk.replace("rpn.head.conv", "proposal_generator.rpn_head.conv")
    nk = nk.replace("rpn.anchor_generator", "proposal_generator.anchor_generator")
    nk = nk.replace("roi_heads.box.feature_extractor.head.layer4", "roi_heads.res5")
    nk = nk.replace("roi_heads.box.predictor.bbox_pred", "roi_heads.box.predictor.bbox_delta")
    new_sd[nk] = v

# Check ROI keys
roi_keys = [k for k in new_sd if "roi_heads.res5" in k]
print("Remapped ROI res5 keys (first 10):")
for k in roi_keys[:10]:
    print(f"  {k}: {tuple(new_sd[k].shape)}")

# Check against model keys
from models.vinvl_feature_extractor import VinVLFeatureExtractor
ext = VinVLFeatureExtractor(device="cpu")
model_sd = ext.model.state_dict()
model_roi = [k for k in model_sd if "roi_heads.res5" in k]
print("\nModel ROI res5 keys (first 10):")
for k in model_roi[:10]:
    print(f"  {k}: {tuple(model_sd[k].shape)}")

# Check which remapped keys match
matched = [k for k in new_sd if k in model_sd and model_sd[k].shape == new_sd[k].shape]
roi_matched = [k for k in matched if "roi_heads.res5" in k]
print(f"\nROI res5 matched: {len(roi_matched)}")
roi_unmatched_model = [k for k in model_roi if k not in new_sd or model_sd[k].shape != new_sd.get(k, torch.tensor([])).shape]
print(f"ROI res5 unmatched in model: {len(roi_unmatched_model)}")
if roi_unmatched_model:
    for k in roi_unmatched_model[:5]:
        print(f"  MISSING: {k} model_shape={tuple(model_sd[k].shape)} ckpt_shape={tuple(new_sd.get(k, torch.tensor([])).shape) if k in new_sd else 'NOT IN CKPT'}")
