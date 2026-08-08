import io
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


def _robust_pickle_load(path: str, attempts: int = 360, delay: float = 5.0):
    """Open a (possibly large) precomputed pkl on Lustre with retries + cat fallback.

    On this cluster, stat/open of a specific large pkl intermittently returns
    ENOENT even though `ls` lists it and `cat` can read it (Lustre metadata/OST
    inconsistency that cycles in/out over minutes). So: retry the normal open()
    for up to ~30 min (attempts*delay) to ride out a bad window; if that whole
    window fails, fall back to reading the bytes via `cat` (which uses a
    different kernel read path and sometimes works when open() doesn't).
    """
    last_err = None
    for i in range(attempts):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, OSError) as e:
            last_err = e
            print(f"  [robust_load] {os.path.basename(path)} open() failed "
                  f"(attempt {i + 1}/{attempts}): {e!r}; retrying in {delay}s")
            time.sleep(delay)

    # Direct open() never succeeded. Fall back to `cat`, which reads via a path
    # that bypasses the failing stat/open in this Lustre state.
    print(f"  [robust_load] {os.path.basename(path)}: open() failed after "
          f"{attempts} attempts; falling back to `cat` pipe")
    try:
        import subprocess
        proc = subprocess.run(["cat", path], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise FileNotFoundError(
                f"cat fallback also failed (rc={proc.returncode}): "
                f"{proc.stderr.decode(errors='replace')[:200]}"
            )
        return pickle.load(io.BytesIO(proc.stdout))
    except FileNotFoundError:
        raise
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load {path} via open() (after {attempts} retries: "
            f"{last_err!r}) nor cat fallback ({e!r})"
        )


class PrecomputedVISTDataset(Dataset):
    def __init__(
        self,
        precomputed_file: str,
        image_root: str,
        is_test: bool = False,
    ):
        self.image_root = Path(image_root)
        self.is_test = is_test

        self.samples = _robust_pickle_load(precomputed_file)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image_pil(self, image_path: str) -> Image.Image:
        full_path = self.image_root / image_path
        return Image.open(full_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        result = {
            "concepts": sample["concepts"],
            "node_texts": sample["node_texts"],
            "node_embeddings": sample["node_embeddings"],
            "edge_index": sample["edge_index"],
            "roi_features": sample["roi_features"],
            "context_feature": sample["context_feature"],
            "target_caption": sample["target_caption"],
            "input_captions": sample["input_captions"],
            "oracle_image_path": sample["oracle_image"],
            "story_id": sample["story_id"],
        }

        result["oracle_image_pil"] = self._load_image_pil(sample["oracle_image"])

        return result


def precomputed_collate_fn(batch: List[Dict]) -> Dict:
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                result[key] = torch.stack(values)
            except RuntimeError:
                result[key] = values
        else:
            result[key] = values
    return result


def build_precomputed_dataloaders(
    config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    precomputed_dir = Path(getattr(config.vist, "precomputed_dir", "data/vist/preprocessed"))

    def _pkl(split: str) -> str:
        ost3 = precomputed_dir / f"{split}_preprocessed_ost3.pkl"
        return str(ost3 if ost3.exists() else precomputed_dir / f"{split}_preprocessed.pkl")

    train_dataset = PrecomputedVISTDataset(
        precomputed_file=_pkl("train"),
        image_root=config.vist.image_root,
    )
    val_dataset = PrecomputedVISTDataset(
        precomputed_file=_pkl("val"),
        image_root=config.vist.image_root,
    )
    test_dataset = PrecomputedVISTDataset(
        precomputed_file=_pkl("test"),
        image_root=config.vist.image_root,
        is_test=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=precomputed_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=precomputed_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=precomputed_collate_fn,
    )

    return train_loader, val_loader, test_loader
