import os
import torch, json, pickle
from PIL import Image
from models.vinvl_feature_extractor import VinVLFeatureExtractor

ext = VinVLFeatureExtractor(device="cuda")

# Load a real VIST image
samples = pickle.load(open(os.environ.get("ACAP_PRECOMPUTED_DIR","data/vist/preprocessed") + "/train_preprocessed.pkl", "rb"))
# The samples have roi_features but no concepts. Let's test with a real image.
# Load from the raw VIST data
import sys
sys.path.insert(0, os.environ.get("ACAP_PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from preprocess_concepts import load_vist_samples
from config import Config
cfg = Config()
raw_samples = load_vist_samples(cfg.vist.train_ann_file, cfg.vist.image_root, 4)
print(f"Loaded {len(raw_samples)} raw samples")
if raw_samples:
    s = raw_samples[0]
    print(f"Sample 0 input_images: {s['input_images']}")
    # Load actual image
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    img = transform(Image.open(f"{cfg.vist.image_root}/{s['input_images'][0]}").convert("RGB")).cuda()
    print(f"Image shape: {tuple(img.shape)}")
    # Run detection
    feats, labels = ext._extract_single(img)
    print(f"ROI features: {tuple(feats.shape)}")
    print(f"Concept labels: {labels}")

    # Also test directly with model inference
    import numpy as np, cv2
    img_np = img.cpu().numpy()
    img_bgr = img_np[::-1].copy() * 255.0
    img_input = torch.from_numpy(img_bgr).cuda()
    inputs = [{"image": img_input, "height": 224, "width": 224}]
    with torch.no_grad():
        outputs = ext.model(inputs)
    instances = outputs[0]["instances"]
    print(f"Detected {len(instances)} instances")
    if len(instances) > 0:
        print(f"  scores: {instances.scores[:5].tolist()}")
        print(f"  classes: {instances.pred_classes[:5].tolist()}")
        print(f"  class names: {[ext.vg_classes[c.item()] for c in instances.pred_classes[:5]]}")
