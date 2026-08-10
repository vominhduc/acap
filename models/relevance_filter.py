from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class RelevanceFilter(nn.Module):
    """Scores forecasted concepts by relevance to the visual context.

    Two scoring modes:
    - "cosine" (default): cosine similarity between the concept's RoBERTa
      embedding and the context feature (unsupervised, no random head needed).
      This replaces the original randomly-initialized Linear head, which scored
      concepts with random weights and selected an essentially random subset.
    - "head": original approach with a trainable Linear head (kept for
      backwards compatibility; the head is randomly initialized and never
      trained in the current pipeline, so scores are meaningless).
    """

    def __init__(self, model_name: str = "roberta-base", device: str = "cuda",
                 context_dim: int = 768, mode: str = "cosine"):
        super().__init__()
        self.device = device
        self.context_dim = context_dim
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        if mode == "head":
            self.head = nn.Linear(self.encoder.config.hidden_size + context_dim, 2)
            self.head = self.head.to(device)
        self.encoder.eval()

    @torch.no_grad()
    def score_concepts(
        self,
        concepts: List[str],
        context_feature: torch.Tensor,
        device: str = "cuda",
    ) -> torch.Tensor:
        if not concepts:
            return torch.tensor([])

        # Encode all concepts in a single batch (much faster than per-concept)
        texts = [c.replace("_", " ") for c in concepts]
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=32
        ).to(device)
        outputs = self.encoder(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]  # (N, 768)

        context = context_feature.unsqueeze(0) if context_feature.dim() == 1 else context_feature
        context = context.to(device)

        if self.mode == "cosine":
            # Cosine similarity between concept embedding and context feature.
            # Unsupervised but meaningful: concepts semantically related to the
            # visual context get higher scores. The context feature is the mean
            # of projected ROI features (768-dim), and the concept embeddings
            # are RoBERTa CLS tokens (768-dim) — both in a similar semantic
            # space due to pretraining.
            concept_norm = F.normalize(pooled, dim=-1)  # (N, 768)
            ctx_norm = F.normalize(context, dim=-1)     # (1, 768)
            scores = (concept_norm @ ctx_norm.T).squeeze(-1)  # (N,)
            # Shift to [0, 1] range for compatibility with downstream code
            scores = (scores + 1) / 2
        else:
            # Original: random Linear head (kept for backwards compat)
            combined = torch.cat([pooled, context.expand(pooled.size(0), -1)], dim=-1)
            logits = self.head(combined)
            prob = torch.softmax(logits, dim=-1)
            scores = prob[:, 1]

        return scores.cpu()

    def forward(self, concept: str, context: torch.Tensor) -> float:
        return self.score_concepts([concept], context, self.device)[0].item()
