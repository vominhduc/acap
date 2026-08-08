from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .feature_extractor import FasterRCNNFeatureExtractor


IMAGENET_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ConceptDetector(nn.Module):
    def __init__(
        self,
        top_k: int = 10,
        device: str = "cuda",
    ):
        super().__init__()
        self.top_k = top_k
        self.device = device

        self.extractor = FasterRCNNFeatureExtractor(
            num_rois_per_image=25,
            top_k_concepts=top_k,
            device=device,
        )

    @torch.no_grad()
    def detect_from_paths(self, image_paths: List[str]) -> List[List[str]]:
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = IMAGENET_DEFAULT_TRANSFORM(img)
            images.append(img)

        batch = torch.stack(images).to(self.device)
        batch = batch.unsqueeze(0)
        return self.extractor.detect_concepts(batch)[0]

    @torch.no_grad()
    def detect_from_tensors(self, image_tensors: torch.Tensor) -> List[List[str]]:
        batch = image_tensors.to(self.device)
        if batch.dim() == 3:
            batch = batch.unsqueeze(0).unsqueeze(0)
        elif batch.dim() == 4:
            batch = batch.unsqueeze(0)
        return self.extractor.detect_concepts(batch)[0]
