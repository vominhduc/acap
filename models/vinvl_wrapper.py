import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForMaskedLM,
    BertTokenizer,
    AutoModelForMaskedLM,
)


class VinVLWrapper(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/vinvl-base",
        embed_dim: int = 768,
        freeze: bool = True,
        roi_feature_dim: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.model_name = model_name

        # VinVL is a gated model. A bare `except` here previously swallowed the
        # load failure (e.g. 401 Unauthorized when HF_TOKEN is unset) and silently
        # fell back to plain bert-base-uncased — a vision-less model that makes
        # the whole A-CAP objective degenerate. Fail loudly instead so a broken
        # run can never look like a successful one.
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load VinVL decoder '{model_name}'. A-CAP requires "
                f"the frozen VinVL vision-language model; silently falling back "
                f"to plain BERT makes the objective degenerate. "
                f"Either (a) set HF_TOKEN to a token with access to "
                f"'microsoft/vinvl-base', or (b) pre-download VinVL into HF_HOME "
                f"and set HF_HUB_OFFLINE=1. Underlying error: {exc!r}"
            ) from exc

        # The VinVL/Oscar vocabulary is the BERT WordPiece vocabulary.
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Restore VinVL's native visual-injection layer (img_embedding: 2054->768),
        # which was dropped when loading as a plain BertForMaskedLM. This is how the
        # decoder was pretrained to receive image/object features — as "word"
        # tokens of token_type 1. Using it (vs raw concatenation) lets the frozen
        # decoder apply its pretrained visual grounding.
        img_cfg_dim = getattr(self.config, "img_feature_dim", 2054)
        self.img_embedding = nn.Linear(img_cfg_dim, self.embed_dim)
        self._load_img_embedding()

        # Learn a projection from our 1024-dim torchvision ROI features into
        # the 2054-dim space img_embedding was pretrained on. Trainable even
        # when the backbone is frozen — this is the adapter that bridges the
        # feature-space gap.
        self.feat_proj = nn.Linear(roi_feature_dim, img_cfg_dim)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # img_embedding is pretrained too — freeze with the backbone
            for param in self.img_embedding.parameters():
                param.requires_grad = False

        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id

    def _load_img_embedding(self):
        """Load bert.img_embedding.* from the safetensors checkpoint if present."""
        from safetensors.torch import load_file
        path = os.path.join(self.model_name, "model.safetensors") if os.path.isdir(self.model_name) else None
        if path and os.path.exists(path):
            sd = load_file(path)
            w = sd.get("bert.img_embedding.weight")
            b = sd.get("bert.img_embedding.bias")
            if w is not None and b is not None:
                with torch.no_grad():
                    self.img_embedding.weight.copy_(w)
                    self.img_embedding.bias.copy_(b)
                print(f"[vinvl] loaded pretrained img_embedding {tuple(w.shape)}")
                return
        print("[vinvl] WARNING: img_embedding not in checkpoint; using random init")

    @property
    def embeddings(self):
        if hasattr(self.backbone, "bert"):
            return self.backbone.bert.embeddings
        elif hasattr(self.backbone, "roberta"):
            return self.backbone.roberta.embeddings
        else:
            return self.backbone.get_input_embeddings()

    def forward(
        self,
        word_embeddings: Optional[torch.Tensor] = None,
        concept_embeddings: Optional[torch.Tensor] = None,
        roi_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        encoder = self._get_encoder()

        if input_ids is not None:
            outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            return sequence_output, pooled_output

        # Build the input sequence VinVL-style: word tokens (type 0) + visual
        # tokens (type 1). ROI features go through the pretrained img_embedding
        # (2054->768) — the decoder's native visual pathway — so the frozen
        # decoder can apply its pretrained visual grounding. A learned feat_proj
        # maps our 1024-dim torchvision features into the 2054-dim space
        # img_embedding expects. Concepts (text-based, 768-dim from GNN) stay
        # as type-0 embeddings like words.
        img_cfg_dim = getattr(self.config, "img_feature_dim", 2054)
        segments = []  # list of (embeddings, token_type_id) per segment
        type_ids_list = []

        if word_embeddings is not None:
            segments.append(word_embeddings)
            type_ids_list.append(torch.zeros(
                word_embeddings.size(0), word_embeddings.size(1),
                dtype=torch.long, device=word_embeddings.device
            ))

        def _sep_like(ref, ttype):
            n = 1
            sep = torch.zeros(ref.size(0), n, self.embed_dim, device=ref.device)
            return sep, torch.full((ref.size(0), n), ttype,
                                   dtype=torch.long, device=ref.device)

        if concept_embeddings is not None:
            if segments:
                sep, stype = _sep_like(concept_embeddings, 0)
                segments.append(sep); type_ids_list.append(stype)
            segments.append(concept_embeddings)
            type_ids_list.append(torch.zeros(
                concept_embeddings.size(0), concept_embeddings.size(1),
                dtype=torch.long, device=concept_embeddings.device
            ))

        if roi_features is not None:
            # Project 1024->2054 then through pretrained img_embedding ->768.
            roi_2054 = self.feat_proj(roi_features)
            roi_emb = self.img_embedding(roi_2054)  # (B, N_roi, 768)
            if segments:
                sep, stype = _sep_like(roi_emb, 1)
                segments.append(sep); type_ids_list.append(stype)
            segments.append(roi_emb)
            type_ids_list.append(torch.ones(
                roi_emb.size(0), roi_emb.size(1),
                dtype=torch.long, device=roi_emb.device
            ))

        if not segments:
            raise ValueError("No inputs provided")

        embeddings = torch.cat(segments, dim=1)
        token_type_ids = torch.cat(type_ids_list, dim=1)

        attention_mask = torch.ones(
            embeddings.size(0), embeddings.size(1),
            device=embeddings.device
        )

        outputs = encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        return sequence_output, pooled_output

    def lm_head(self, sequence_output: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "cls"):
            return self.backbone.cls(sequence_output)
        elif hasattr(self.backbone, "lm_head"):
            return self.backbone.lm_head(sequence_output)
        else:
            return self.backbone.get_output_embeddings()(sequence_output)

    def _get_encoder(self):
        if hasattr(self.backbone, "bert"):
            return self.backbone.bert
        elif hasattr(self.backbone, "roberta"):
            return self.backbone.roberta
        else:
            return self.backbone

    @property
    def device(self):
        return next(self.backbone.parameters()).device
