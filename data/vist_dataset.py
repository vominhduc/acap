import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class VISTDataset(Dataset):
    def __init__(
        self,
        ann_file: str,
        image_root: str,
        num_input_images: int = 4,
        target_sentence_index: int = -1,
        transform: Optional[transforms.Compose] = None,
        is_test: bool = False,
    ):
        self.image_root = Path(image_root)
        self.num_input_images = num_input_images
        self.target_sentence_index = target_sentence_index
        self.is_test = is_test

        with open(ann_file) as f:
            self.annotations = json.load(f)

        self.samples = self._build_samples()

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _is_valid_image(self, filename: str) -> bool:
        path = self.image_root / filename
        if not path.exists():
            return False
        try:
            if path.stat().st_size < 500:
                return False
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def _build_samples(self) -> List[Dict]:
        samples = []
        for item in self.annotations:
            if isinstance(item, dict) and "images" in item and "sentences" in item:
                images = item["images"]
                sentences = item["sentences"]

                if len(images) < self.num_input_images + 1:
                    continue
                if len(sentences) < self.num_input_images + 1:
                    continue

                input_images = images[:self.num_input_images]
                oracle_image = images[self.num_input_images]

                all_images = input_images + [oracle_image]
                if any(not self._is_valid_image(img) for img in all_images):
                    continue

                if self.target_sentence_index == -1:
                    target_caption = sentences[-1]
                else:
                    target_caption = sentences[self.target_sentence_index]

                input_captions = sentences[:self.num_input_images]

                samples.append({
                    "input_images": input_images,
                    "oracle_image": oracle_image,
                    "target_caption": target_caption,
                    "input_captions": input_captions,
                    "image_dir": item.get("image_dir", ""),
                    "album_id": item.get("album_id", ""),
                    "story_id": item.get("story_id", ""),
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: str) -> torch.Tensor:
        full_path = self.image_root / image_path
        image = Image.open(full_path).convert("RGB")
        return self.transform(image)

    def _load_image_pil(self, image_path: str) -> Image.Image:
        full_path = self.image_root / image_path
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        input_images = torch.stack([
            self._load_image(img) for img in sample["input_images"]
        ])
        target_caption = sample["target_caption"]

        result = {
            "input_images": input_images,
            "target_caption": target_caption,
            "input_captions": sample["input_captions"],
            "oracle_image_path": sample["oracle_image"],
            "album_id": sample["album_id"],
            "story_id": sample["story_id"],
        }

        oracle_image = self._load_image(sample["oracle_image"])
        result["oracle_image"] = oracle_image

        result["oracle_image_pil"] = self._load_image_pil(sample["oracle_image"])

        return result


def vist_collate_fn(batch: List[Dict]) -> Dict:
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def build_vist_dataloaders(
    config,
    train_transform=None,
    val_transform=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = VISTDataset(
        ann_file=config.vist.train_ann_file,
        image_root=config.vist.image_root,
        num_input_images=config.vist.num_input_images,
        target_sentence_index=config.vist.target_sentence_index,
        transform=train_transform,
    )
    val_dataset = VISTDataset(
        ann_file=config.vist.val_ann_file,
        image_root=config.vist.image_root,
        num_input_images=config.vist.num_input_images,
        target_sentence_index=config.vist.target_sentence_index,
        transform=val_transform,
    )
    test_dataset = VISTDataset(
        ann_file=config.vist.test_ann_file,
        image_root=config.vist.image_root,
        num_input_images=config.vist.num_input_images,
        target_sentence_index=config.vist.target_sentence_index,
        transform=val_transform,
        is_test=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=vist_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=vist_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=vist_collate_fn,
    )

    return train_loader, val_loader, test_loader
