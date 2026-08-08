from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class FasterRCNNFeatureExtractor(nn.Module):
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
        self.feature_dim = 1024

        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.coco_labels = COCO_INSTANCE_CATEGORY_NAMES
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def extract_features(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[List[str]]]]:
        batch_size, num_images = images.size(0), images.size(1)
        all_roi_features = []
        all_concepts = []

        for b in range(batch_size):
            sample_rois = []
            sample_concepts = []
            for k in range(num_images):
                img = images[b, k]
                feats, labels = self._extract_single(img)
                sample_rois.append(feats)
                sample_concepts.append(labels)

            cat_rois = torch.cat(sample_rois, dim=0)
            all_roi_features.append(cat_rois)
            all_concepts.append(sample_concepts)

        roi_features = torch.stack(all_roi_features)
        return roi_features, all_concepts

    def _extract_single(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        images = [img.to(self.device)]

        transformed, _ = self.model.transform(images, None)
        features = self.model.backbone(transformed.tensors)
        proposals, _ = self.model.rpn(transformed, features)

        if len(proposals[0]) == 0:
            return (
                torch.zeros(self.num_rois, self.feature_dim, device=self.device),
                [],
            )

        box_features = self.model.roi_heads.box_roi_pool(
            features, proposals, transformed.image_sizes
        )
        box_features = self.model.roi_heads.box_head(box_features)

        class_logits, _ = self.model.roi_heads.box_predictor(box_features)
        scores = torch.softmax(class_logits, dim=-1)

        max_scores, max_classes = scores[:, 1:].max(dim=-1)

        keep = max_scores >= self.score_threshold
        box_features = box_features[keep]
        max_scores = max_scores[keep]
        max_classes = max_classes[keep] + 1

        if box_features.size(0) == 0:
            return (
                torch.zeros(self.num_rois, self.feature_dim, device=self.device),
                [],
            )

        n_select = min(self.num_rois, box_features.size(0))
        top_scores, top_indices = torch.topk(max_scores, n_select)

        selected_features = box_features[top_indices]

        labels = []
        seen = set()
        for idx in max_classes[top_indices]:
            label = self.coco_labels[idx.item()]
            if label not in ("N/A", "__background__") and label not in seen:
                labels.append(label)
                seen.add(label)
            if len(labels) >= self.top_k_concepts:
                break

        if selected_features.size(0) < self.num_rois:
            pad = torch.zeros(
                self.num_rois - selected_features.size(0),
                self.feature_dim,
                device=self.device,
            )
            selected_features = torch.cat([selected_features, pad], dim=0)

        return selected_features, labels

    @torch.no_grad()
    def detect_concepts(self, images: torch.Tensor) -> List[List[List[str]]]:
        _, concepts = self.extract_features(images)
        return concepts
