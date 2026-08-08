"""VinVL X152C4 feature extractor using detectron2.

Replaces torchvision FasterRCNN with the real VinVL detector that produces
2054-dim ROI features (2048-dim visual + 6-dim spatial) over 1594 VG classes.
This is the matched feature space the frozen VinVL decoder was pretrained on.
"""
import json
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms


_VINVL_BACKBONE_REGISTERED = False


def _ensure_backbone_registered(BACKBONE_REGISTRY, build_resnet_backbone, Backbone):
    """Register the VinVLBackbone wrapper once, idempotently."""
    global _VINVL_BACKBONE_REGISTERED
    if _VINVL_BACKBONE_REGISTERED:
        return

    class VinVLBackbone(Backbone):
        def __init__(self, cfg, input_shape):
            super().__init__()
            self.body = build_resnet_backbone(cfg, input_shape)
            out_shape = self.body.output_shape()
            self._out_features = cfg.MODEL.RESNETS.OUT_FEATURES
            self._out_feature_channels = {k: out_shape[k].channels for k in self._out_features}
            self._size_divisibility = max(out_shape[k].stride for k in self._out_features)

        @property
        def size_divisibility(self):
            return self._size_divisibility

        def forward(self, x):
            features = self.body(x)
            return {k: features[k] for k in self._out_features}

        def output_shape(self):
            from detectron2.layers import ShapeSpec
            return {k: ShapeSpec(channels=self._out_feature_channels[k], stride=16)
                    for k in self._out_features}

    BACKBONE_REGISTRY.register(VinVLBackbone)
    _VINVL_BACKBONE_REGISTERED = True


class VinVLFeatureExtractor(nn.Module):
    """Extracts 2054-dim ROI features using the VinVL X152C4 detector.

    Features = concat(visual_features (2048), spatial_features (6)) per ROI,
    matching the paper's img_feature_dim=2054 that feeds VinVL's img_embedding.
    """

    def __init__(
        self,
        num_rois_per_image: int = 25,
        top_k_concepts: int = 10,
        score_threshold: float = 0.2,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_rois = num_rois_per_image
        self.top_k_concepts = top_k_concepts
        self.score_threshold = score_threshold
        self.device = device
        self.feature_dim = 2054

        # Paths to downloaded VinVL weights
        hf_cache = os.environ.get("HF_HOME", "/root/.cache/huggingface")
        model_dir = os.path.join(
            hf_cache, "hub",
            "models--michelecafagna26--vinvl_vg_x152c4",
            "snapshots",
        )
        # Find the snapshot dir
        snapshot_dir = None
        if os.path.isdir(model_dir):
            for d in os.listdir(model_dir):
                candidate = os.path.join(model_dir, d)
                if os.path.isdir(candidate):
                    snapshot_dir = candidate
                    break
        if snapshot_dir is None:
            raise FileNotFoundError(f"VinVL snapshot not found under {model_dir}")

        self.weights_path = os.path.join(snapshot_dir, "vinvl_vg_x152c4.pth")
        self.dict_path = os.path.join(snapshot_dir, "VG-SGG-dicts-vgoi6-clipped.json")

        # Load VG class names
        with open(self.dict_path) as f:
            vg_dicts = json.load(f)
        self.vg_classes = list(vg_dicts["idx_to_label"].values())  # 1594 classes, idx 0-1593

        # Build detectron2 model
        self._build_model()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _build_model(self):
        """Build the VinVL GeneralizedRCNN via detectron2 config."""
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model, BACKBONE_REGISTRY
        from detectron2.modeling.backbone import build_resnet_backbone, Backbone
        from detectron2.modeling.backbone.resnet import ResNet
        import torch.nn as nn

        # Register the VinVL backbone wrapper once. The VinVL checkpoint wraps
        # ResNet under `backbone.body.*` (from scene_graph_benchmark); standard
        # detectron2 puts it under `backbone.*` directly. This wrapper adds the
        # `body` submodule. Defined at module scope (via _ensure_backbone_regis-
        # tered) so re-building the detector for a second split doesn't
        # double-register and raise "already registered".
        _ensure_backbone_registered(BACKBONE_REGISTRY, build_resnet_backbone, Backbone)

        cfg = get_cfg()

        cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
        cfg.MODEL.BACKBONE.NAME = "VinVLBackbone"
        cfg.MODEL.RESNETS.DEPTH = 152
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res4"]
        cfg.MODEL.RESNETS.NUM_GROUPS = 32
        cfg.MODEL.RESNETS.WIDTH_PER_GROUP = 8
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = True
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]  # Caffe2 BGR
        cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]
        cfg.MODEL.INPUT_FORMAT = "BGR"

        # RPN
        cfg.MODEL.RPN.IN_FEATURES = ["res4"]
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

        # ROI Heads (C4 variant — uses layer4 as box head)
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.vg_classes)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5

        # Box head: C4 (ResNet layer4, output 2048)
        cfg.MODEL.ROI_BOX_HEAD.NAME = "Res5Head"
        cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 0
        cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
        cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
        cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

        # Freeze
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.DEVICE = self.device

        self.model = build_model(cfg)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Load weights — remap checkpoint keys to detectron2 naming
        ckpt = torch.load(self.weights_path, map_location=self.device, weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else (ckpt.get("state_dict", ckpt))
        new_sd = {}
        import re
        for k, v in sd.items():
            nk = k.replace("module.", "")
            # Only keep backbone, rpn, roi_heads (skip attribute/SGG)
            if not (nk.startswith("backbone.") or nk.startswith("rpn.") or nk.startswith("roi_heads.")):
                continue
            # ROI heads FIRST: feature_extractor.head.layer4 -> res5
            # (do this before generic bn remap, since ROI uses different norm naming)
            nk = nk.replace("roi_heads.box.feature_extractor.head.layer4", "roi_heads.res5")
            # ROI res5 BN: detectron2 Res5Head uses conv1.norm (not norm1).
            # Checkpoint has bn1/bn2/bn3 -> remap to conv1.norm/conv2.norm/conv3.norm
            nk = re.sub(r"(roi_heads\.res5\.\d+\.)bn1\.", r"\1conv1.norm.", nk)
            nk = re.sub(r"(roi_heads\.res5\.\d+\.)bn2\.", r"\1conv2.norm.", nk)
            nk = re.sub(r"(roi_heads\.res5\.\d+\.)bn3\.", r"\1conv3.norm.", nk)
            nk = nk.replace("roi_heads.box.predictor.cls_score", "roi_heads.box_predictor.cls_score")
            nk = nk.replace("roi_heads.box.predictor.bbox_pred", "roi_heads.box_predictor.bbox_pred")
            # VinVL's classifier puts BACKGROUND at index 0 (VinVL row 0 = bg,
            # rows 1..1594 = the 1594 VG classes). But detectron2's
            # fast_rcnn_inference assumes background is LAST and drops it via
            # `probs[:, :-1]`. Without reordering, detectron2 keeps VinVL's bg
            # (row 0) as a "foreground" class and drops a real VG class (row
            # 1594) -> every detection reports as class 0. Move the bg row to
            # the end so detectron2's bg-last assumption holds. After this,
            # detectron2 class 0 = VinVL class 1 = vg_classes[0].
            if nk.endswith("box_predictor.cls_score.weight") or nk.endswith("box_predictor.cls_score.bias"):
                v = torch.cat([v[1:], v[0:1]], dim=0)
            # VinVL bbox_pred includes a background row (1595*4=6380); detectron2's
            # excludes it (1594*4=6376). Drop the first 4 (background) rows so the
            # shape matches. bg is class index 0 in VinVL, so rows 0:4 are bg
            # deltas; rows 4: are VG classes 1..1594 which align with detectron2
            # classes 0..1593 after the cls_score reorder above.
            if nk.endswith("box_predictor.bbox_pred.weight") or nk.endswith("box_predictor.bbox_pred.bias"):
                if v.size(0) == 6380:
                    v = v[4:]
            # RPN remapping
            nk = nk.replace("rpn.head.cls_logits", "proposal_generator.rpn_head.objectness_logits")
            nk = nk.replace("rpn.head.bbox_pred", "proposal_generator.rpn_head.anchor_deltas")
            nk = nk.replace("rpn.head.conv", "proposal_generator.rpn_head.conv")
            nk = nk.replace("rpn.anchor_generator", "proposal_generator.anchor_generator")
            # Backbone remapping
            nk = nk.replace("backbone.body.layer1", "backbone.body.res2")
            nk = nk.replace("backbone.body.layer2", "backbone.body.res3")
            nk = nk.replace("backbone.body.layer3", "backbone.body.res4")
            nk = nk.replace("downsample.0.", "shortcut.")
            nk = nk.replace("downsample.1.", "shortcut.norm.")
            # detectron2's ResNet wraps each block conv with a NESTED norm module:
            # model has res2.0.conv1.norm / conv2.norm / conv3.norm (norm lives
            # inside the Conv2d wrapper), and stem.conv1.norm. The scene_graph_
            # benchmark checkpoint stores them as sibling bn1/bn2/bn3 and
            # stem.bn1. Remap bn1->conv1.norm, bn2->conv2.norm, bn3->conv3.norm
            # so the norm lands inside the conv wrapper where detectron2 expects it.
            nk = nk.replace("stem.bn1.", "stem.conv1.norm.")
            nk = re.sub(r"\.bn1\.", ".conv1.norm.", nk)
            nk = re.sub(r"\.bn2\.", ".conv2.norm.", nk)
            nk = re.sub(r"\.bn3\.", ".conv3.norm.", nk)
            new_sd[nk] = v

        # Filter out keys with shape mismatches
        model_sd = self.model.state_dict()
        filtered_sd = {}
        for k, v in new_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered_sd[k] = v
            elif k not in model_sd:
                # try without the key
                pass
        missing, unexpected = self.model.load_state_dict(filtered_sd, strict=False)
        print(f"VinVL detector loaded: {len(missing)} missing, {len(unexpected)} unexpected (filtered {len(new_sd) - len(filtered_sd)} mismatched)")
        if missing:
            # Check if any critical (non-norm) keys are missing
            critical = [k for k in missing if "norm" not in k and "num_batches" not in k]
            if critical:
                print(f"  CRITICAL missing: {critical[:10]}")

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def extract_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[List[str]]]]:
        """Extract 2054-dim ROI features + detected concept labels.

        Args:
            images: (B, num_images, C, H, W) tensor of input images
        Returns:
            roi_features: (B, num_images * num_rois, 2054)
            all_concepts: List[List[List[str]]] — per-sample, per-image concept lists
        """
        from detectron2.structures import Boxes, Instances
        from detectron2.data.transforms import TransformList

        batch_size, num_images = images.size(0), images.size(1)
        all_roi_features = []
        all_concepts = []

        for b in range(batch_size):
            sample_rois = []
            sample_concepts = []
            for k in range(num_images):
                feats, labels = self._extract_single(images[b, k])
                sample_rois.append(feats)
                sample_concepts.append(labels)

            cat_rois = torch.cat(sample_rois, dim=0)
            all_roi_features.append(cat_rois)
            all_concepts.append(sample_concepts)

        roi_features = torch.stack(all_roi_features)
        return roi_features, all_concepts

    @torch.no_grad()
    def _extract_single(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extract ROI features for a single image.

        Returns (num_rois, 2054) features and concept label list.
        """
        from detectron2.structures import Boxes, Instances
        import detectron2.data.transforms as T

        # VinVL expects BGR, Caffe2-style preprocessing
        # img is (C, H, W) RGB, 0-1 range. Convert to BGR int range.
        img_np = img.cpu().numpy()  # (C, H, W) RGB, 0-1
        # Convert RGB to BGR and scale to 0-255 (detectron2 handles mean subtraction)
        img_bgr = img_np[::-1].copy()  # BGR, (3, H, W)
        img_scaled = img_bgr * 255.0  # (3, H, W)

        h, w = img_scaled.shape[1], img_scaled.shape[2]
        max_size = 1000
        scale = min(max_size / max(h, w), 1.0)
        new_h, new_w = int(h * scale), int(w * scale)

        import cv2
        img_resized = np.zeros((3, new_h, new_w), dtype=np.float32)
        for c in range(3):
            img_resized[c] = cv2.resize(img_scaled[c], (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # detectron2 expects (C, H, W) tensor on device; it does pixel_mean/std internally
        img_input = torch.from_numpy(img_resized).to(self.device)  # (3, H, W)

        # Run detection + feature extraction in ONE pass through the model's
        # own components, so the backbone sees the SAME normalized image the
        # detector used (preprocess_image applies pixel_mean/std). Previously
        # the backbone was re-run on the raw (un-normalized) image with a
        # buggy permute(2,0,1) that turned (3,H,W) into (W,3,H) -> 224-channel
        # crash, so all ROI features came back zero.
        inputs = [{"image": img_input, "height": new_h, "width": new_w}]
        images = self.model.preprocess_image(inputs)
        features = self.model.backbone(images.tensor)
        proposals, _ = self.model.proposal_generator(images, features, None)
        inst_list, _ = self.model.roi_heads(images, features, proposals, None)
        instances = inst_list[0]

        if len(instances) == 0:
            return (
                torch.zeros(self.num_rois, self.feature_dim, device=self.device),
                [],
            )

        # Get boxes, scores, classes
        boxes = instances.pred_boxes.tensor  # (N, 4)
        scores = instances.scores  # (N,)
        classes = instances.pred_classes  # (N,)

        # Re-pool box features for the FINAL predicted boxes through the same
        # backbone features. This detectron2 build uses Res5ROIHeads, whose
        # submodules are `pooler` (ROIPooler) + `res5` (the C4 box head). res5
        # outputs (N, 2048, 7, 7); global-avg-pool over the spatial dims to get
        # the 2048-dim box features the paper feeds (concat 2048+6=2054).
        rh = self.model.roi_heads
        feat_list = [features[f] for f in rh.in_features]
        roi_pooled = rh.pooler(feat_list, [instances.pred_boxes])
        res5_out = rh.res5(roi_pooled)  # (N, 2048, h, w)
        box_features = res5_out.mean(dim=[2, 3])  # (N, 2048)

        # Spatial features: normalized box coords (cx, cy, w, h, area, ratio) = 6
        scaled_boxes = boxes.clone()
        scaled_boxes[:, 0] /= new_w
        scaled_boxes[:, 2] /= new_w
        scaled_boxes[:, 1] /= new_h
        scaled_boxes[:, 3] /= new_h
        cx = (scaled_boxes[:, 0] + scaled_boxes[:, 2]) / 2
        cy = (scaled_boxes[:, 1] + scaled_boxes[:, 3]) / 2
        bw = (scaled_boxes[:, 2] - scaled_boxes[:, 0]).clamp(min=0)
        bh = (scaled_boxes[:, 3] - scaled_boxes[:, 1]).clamp(min=0)
        area = (bw * bh).clamp(min=1e-6)
        ratio = (bw / bh).clamp(min=1e-6)
        spatial = torch.stack([cx, cy, bw, bh, area, ratio], dim=-1)  # (N, 6)

        # Concat: 2048 + 6 = 2054
        roi_feats = torch.cat([box_features, spatial], dim=-1)  # (N, 2054)

        # Select top-N by score
        n_select = min(self.num_rois, len(scores))
        top_scores, top_indices = torch.topk(scores, n_select)

        selected_features = roi_feats[top_indices]
        if selected_features.size(0) < self.num_rois:
            pad = torch.zeros(
                self.num_rois - selected_features.size(0),
                self.feature_dim,
                device=self.device,
            )
            selected_features = torch.cat([selected_features, pad], dim=0)

        # Concept labels from VG classes. After the cls_score reorder in
        # _build_model (bg moved to the end), detectron2's predicted class
        # indices 0..1593 correspond to VinVG classes 1..1594, which are exactly
        # self.vg_classes[0..1593] (vg_classes = list(idx_to_label.values()),
        # keyed "1".."1594"). Background (now the last cls_score row) is dropped
        # by detectron2 inference, so it never appears in pred_classes.
        labels = []
        seen = set()
        for idx in classes[top_indices]:
            c = idx.item()
            if c >= len(self.vg_classes):
                continue
            cls_name = self.vg_classes[c]
            if cls_name not in seen:
                labels.append(cls_name)
                seen.add(cls_name)
            if len(labels) >= self.top_k_concepts:
                break

        return selected_features, labels

    @torch.no_grad()
    def detect_concepts(self, images: torch.Tensor) -> List[List[List[str]]]:
        _, concepts = self.extract_features(images)
        return concepts
