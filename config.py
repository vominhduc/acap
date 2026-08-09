from dataclasses import dataclass, field
import os
from typing import Optional


def _env(name: str, default: str) -> str:
    """Look up a path-like config value from the env, falling back to default."""
    return os.environ.get(name, default)


@dataclass
class ModelConfig:
    k: int = 4
    num_detected_per_image: int = 10
    num_forecasted: int = 60
    num_rois_per_image: int = 25
    word_seq_length: int = 35
    embed_dim: int = 768
    hidden_dim: int = 1536
    num_gat_layers: int = 2
    num_gat_heads: int = 8
    dropout: float = 0.1
    conceptnet_max_retries: int = 3
    conceptnet_timeout: int = 10
    relevance_top_k: int = 60
    # VinVL is NOT on the HF Hub (microsoft/vinvl-base 401s — it doesn't exist
    # there). We use the community reupload michelecafagna26/vinvl-base-image-
    # captioning: a standard BertForMaskedLM (bidirectional, MLM, BERT WordPiece
    # vocab) — exactly the frozen decoder A-CAP needs.
    #
    # The reupload ships only pytorch_model.bin (pickle), but transformers 5.x +
    # torch 2.5 refuses pickle loads (CVE-2025-32434). So we point at a local dir
    # whose weights were pre-converted to model.safetensors (see
    # scripts/setup_vinvl.sh). A-CAP injects concepts/ROI via its own
    # concatenation, so VinVL's internal img_embedding (2054->768) is unused —
    # correct, and matches the paper's "change the input of VinVL" design.
    vinvl_model_name: str = _env("ACAP_VINVL_MODEL", "vinvl_base")
    bert_model_name: str = "bert-base-uncased"
    roberta_model_name: str = "roberta-base"
    roi_feature_dim: int = 2054
    device: str = "cuda"
    use_gnn: bool = True
    use_context: bool = True
    # Paper freezes the decoder. With the real VinVL X152C4 detector producing
    # 2054-dim features in the decoder's native space, the frozen decoder can
    # apply its pretrained visual grounding — the paper-faithful setup.
    freeze_vinvl: bool = True
    # Paper-faithful: the paper feeds L×<mask> as pseudo-words at inference and
    # trains with cross-entropy against the ground-truth caption. Training with
    # ALL-mask (prob=1.0) forces the frozen decoder (a pretrained image
    # captioner) to generate from concepts+ROI alone — exactly its pretraining
    # task — with no text shortcut. With 15% masking the decoder can predict
    # masked tokens from the 85% visible caption text and never learns to use
    # vision, collapsing to mode tokens at all-mask inference.
    mlm_mask_prob: float = 1.0


@dataclass
class TrainConfig:
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "checkpoints"
    max_grad_norm: float = 1.0


@dataclass
class VISTConfig:
    data_root: str = "data/vist"
    train_ann_file: str = "data/vist/train.json"
    val_ann_file: str = "data/vist/val.json"
    test_ann_file: str = "data/vist/test.json"
    image_root: str = "data/vist/images"
    num_input_images: int = 4
    target_sentence_index: int = -1
    # Precomputed (features+concepts+KG) pkls live here. Override with the
    # ACAP_PRECOMPUTED_DIR env var. Use a large-file-friendly scratch path
    # (the default data/vist/preprocessed/ can suffer filesystem corruption on
    # the large ~15GB train pkl on some shared filesystems).
    precomputed_dir: str = _env("ACAP_PRECOMPUTED_DIR", "data/vist/preprocessed")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    vist: VISTConfig = field(default_factory=VISTConfig)
    seed: int = 42
    experiment_name: str = "acap_vist"
