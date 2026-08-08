import os
import sys
sys.path.insert(0, os.environ.get("ACAP_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.vinvl_feature_extractor import VinVLFeatureExtractor
ext = VinVLFeatureExtractor(device="cpu")
model_keys = list(ext.model.state_dict().keys())
rpn_keys = [k for k in model_keys if "rpn" in k.lower() or "proposal" in k.lower()][:10]
roi_keys = [k for k in model_keys if "roi_heads" in k][:15]
print("RPN keys:", rpn_keys)
print("ROI keys:", roi_keys)
