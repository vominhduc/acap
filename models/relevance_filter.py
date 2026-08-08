from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RelevanceFilter(nn.Module):
    def __init__(self, model_name: str = "roberta-base", device: str = "cuda", context_dim: int = 768):
        super().__init__()
        self.device = device
        self.context_dim = context_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.head = nn.Linear(self.encoder.config.hidden_size + context_dim, 2)
        self.head = self.head.to(device)
        self.encoder.eval()
        self.head.eval()

    @torch.no_grad()
    def score_concepts(
        self,
        concepts: List[str],
        context_feature: torch.Tensor,
        device: str = "cuda",
    ) -> torch.Tensor:
        if not concepts:
            return torch.tensor([])

        scores = []
        for concept in concepts:
            text = f"The image shows {concept.replace('_', ' ')}"
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=32
            ).to(device)

            outputs = self.encoder(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]

            context = context_feature.unsqueeze(0) if context_feature.dim() == 1 else context_feature
            combined = torch.cat([pooled, context.to(device)], dim=-1)

            logits = self.head(combined)
            prob = torch.softmax(logits, dim=-1)
            relevance_prob = prob[0, 1].item()
            scores.append(relevance_prob)

        return torch.tensor(scores)

    def forward(self, concept: str, context: torch.Tensor) -> float:
        return self.score_concepts([concept], context, self.device)[0].item()
