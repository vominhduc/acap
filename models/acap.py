from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .concept_detector import ConceptDetector
from .concept_net_client import ConceptNetClient
from .knowledge_graph import (
    KnowledgeGraphConstructor,
    KnowledgeGraph,
    build_temporal_edges,
)
from .relevance_filter import RelevanceFilter
from .gat import ConceptGNN
from .vinvl_wrapper import VinVLWrapper
from .feature_extractor import FasterRCNNFeatureExtractor


class ACap(nn.Module):
    def __init__(
        self,
        num_input_images: int = 4,
        num_detected_per_image: int = 10,
        num_forecasted: int = 60,
        num_rois_per_image: int = 25,
        word_seq_length: int = 35,
        embed_dim: int = 768,
        hidden_dim: int = 1536,
        num_gat_layers: int = 2,
        num_gat_heads: int = 8,
        dropout: float = 0.1,
        bert_model_name: str = "bert-base-uncased",
        vinvl_model_name: str = "microsoft/vinvl-base",
        device: str = "cuda",
        use_gnn: bool = True,
        use_context: bool = True,
        roi_feature_dim: int = 1024,
        freeze_vinvl: bool = False,
        mlm_mask_prob: float = 1.0,
    ):
        super().__init__()
        self.num_input_images = num_input_images
        self.num_detected_per_image = num_detected_per_image
        self.num_forecasted = num_forecasted
        self.num_rois_per_image = num_rois_per_image
        self.word_seq_length = word_seq_length
        self.embed_dim = embed_dim
        self.device = device
        self.use_gnn = use_gnn
        self.use_context = use_context

        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.bert_model.eval()
        for p in self.bert_model.parameters():
            p.requires_grad = False
        self.bert_model.to(device)

        # Use the real VinVL X152C4 detector (2054-dim, 1594 VG classes) — the
        # matched feature space the frozen VinVL decoder was pretrained on.
        from .vinvl_feature_extractor import VinVLFeatureExtractor
        self.feature_extractor = VinVLFeatureExtractor(
            num_rois_per_image=num_rois_per_image,
            top_k_concepts=num_detected_per_image,
            device=device,
        )

        self.conceptnet_client = ConceptNetClient()
        self.relevance_filter = RelevanceFilter(device=device)

        self.knowledge_graph_constructor = KnowledgeGraphConstructor(
            conceptnet_client=self.conceptnet_client,
            relevance_filter=self.relevance_filter,
            num_forecasted=num_forecasted,
        )

        gnn_input_dim = embed_dim * 2 if use_context else embed_dim
        gnn_use_projection = use_context

        self.gnn = ConceptGNN(
            input_dim=gnn_input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            num_layers=num_gat_layers,
            num_heads=num_gat_heads,
            dropout=dropout,
            use_projection=gnn_use_projection,
        ).to(device)

        self.vinvl = VinVLWrapper(
            model_name=vinvl_model_name,
            embed_dim=embed_dim,
            freeze=freeze_vinvl,
            roi_feature_dim=roi_feature_dim,
        ).to(device)
        self.mlm_mask_prob = mlm_mask_prob

        self.roi_proj = nn.Linear(roi_feature_dim, embed_dim).to(device)

    def extract_features_and_concepts(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[List[str]]]]:
        return self.feature_extractor.extract_features(images)

    def build_context_feature(self, roi_features: torch.Tensor) -> torch.Tensor:
        projected = self.roi_proj(roi_features)
        return projected.mean(dim=1)

    def get_concept_embeddings(
        self,
        graphs: List[KnowledgeGraph],
        context_features: torch.Tensor,
    ) -> List[torch.Tensor]:
        batch_embeddings = []

        for i, graph in enumerate(graphs):
            node_embs = []
            for concept in graph.nodes:
                inputs = self.bert_tokenizer(
                    concept.replace("_", " "),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=16,
                ).to(self.device)
                with torch.no_grad():
                    output = self.bert_model(**inputs)
                    emb = output.last_hidden_state[:, 0, :]
                node_embs.append(emb)

            if not node_embs:
                node_embs = [torch.zeros(1, self.embed_dim, device=self.device)]

            node_embeddings = torch.cat(node_embs, dim=0)

            if self.use_context:
                context = context_features[i].unsqueeze(0).expand(
                    node_embeddings.size(0), -1
                )
                augmented = torch.cat([node_embeddings, context], dim=-1)
            else:
                augmented = node_embeddings

            batch_embeddings.append(augmented)

        return batch_embeddings

    def build_graphs(
        self,
        concepts: List[List[List[str]]],
        context_features: torch.Tensor,
    ) -> List[KnowledgeGraph]:
        graphs = []
        for i in range(len(concepts)):
            graph = self.knowledge_graph_constructor.build(
                concepts[i], context_features[i], self.device
            )
            build_temporal_edges(graph, concepts[i])
            graphs.append(graph)
        return graphs

    def _pad_and_stack(
        self, embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        max_nodes = max(emb.size(0) for emb in embeddings)
        padded = []
        for emb in embeddings:
            if emb.size(0) < max_nodes:
                pad = torch.zeros(
                    max_nodes - emb.size(0), emb.size(-1),
                    device=self.device
                )
                emb = torch.cat([emb, pad], dim=0)
            padded.append(emb)
        return torch.stack(padded)

    def _get_edge_indices(
        self, graphs: List[KnowledgeGraph]
    ) -> List[torch.Tensor]:
        edge_indices = []
        for graph in graphs:
            edge_indices.append(graph.to_torch_edge_index().to(self.device))
        return edge_indices

    def forward(
        self,
        input_images: Optional[torch.Tensor] = None,
        target_captions: Optional[List[str]] = None,
        precomputed: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        if precomputed is not None:
            return self._forward_precomputed(precomputed, target_captions)

        input_images = input_images.to(self.device)
        batch_size = input_images.size(0)

        roi_features, concepts = self.extract_features_and_concepts(input_images)
        context_features = self.build_context_feature(roi_features)
        graphs = self.build_graphs(concepts, context_features)
        node_embs = self.get_concept_embeddings(graphs, context_features)
        edge_indices = self._get_edge_indices(graphs)

        enriched_concepts = []
        for idx in range(batch_size):
            emb = node_embs[idx]
            edge_idx = edge_indices[idx]
            if self.use_gnn:
                emb_batch = emb.unsqueeze(0)
                enriched = self.gnn(emb_batch, edge_idx)
            else:
                enriched = emb
            if enriched.dim() == 1:
                enriched = enriched.unsqueeze(0)
            enriched_concepts.append(enriched)

        concept_embeds = self._pad_and_stack(enriched_concepts)
        # Pass RAW 1024-dim roi_features — the VinVLWrapper projects them through
        # feat_proj(1024->2054) + pretrained img_embedding(2054->768), using the
        # decoder's native visual pathway instead of raw concatenation.
        return self._decode(concept_embeds, roi_features, batch_size, target_captions)

    def _forward_precomputed(
        self, precomputed: Dict, target_captions: Optional[List[str]]
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(precomputed["node_embeddings"])

        node_embs_list = [
            emb.to(self.device) for emb in precomputed["node_embeddings"]
        ]
        edge_indices = [
            ei.to(self.device) for ei in precomputed["edge_index"]
        ]
        roi_features = precomputed["roi_features"].to(self.device)

        enriched_concepts = []
        for idx in range(batch_size):
            emb = node_embs_list[idx]
            edge_idx = edge_indices[idx]
            if self.use_gnn:
                emb_batch = emb.unsqueeze(0)
                enriched = self.gnn(emb_batch, edge_idx)
            else:
                enriched = emb
            if enriched.dim() == 1:
                enriched = enriched.unsqueeze(0)
            enriched_concepts.append(enriched)

        concept_embeds = self._pad_and_stack(enriched_concepts)
        return self._decode(concept_embeds, roi_features, batch_size, target_captions)

    def _mlm_mask(
        self, input_ids: torch.Tensor, prob: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mask caption tokens for MLM training.

        Returns (masked_input_ids, labels). ``labels`` keeps the true token only
        at the selected positions and is -100 (ignored by CE) elsewhere; special
        tokens (CLS/SEP/PAD) are never selected.

        prob=0.15 (paper): BERT-style 80/10/10 — select 15%, of which 80% -> __,
        10% -> random, 10% -> keep. The model predicts masked words from the
        concept/ROI context plus surrounding visible text.

        prob>=1.0: mask ALL non-special tokens -> ọc. No text shortcut remains;
        the model must predict the full caption from concepts+ROI only, which
        matches the all-mask inference condition exactly. Used when the frozen
        decoder's features don't match (torchvision vs VinVL), forcing visual
        grounding so the (now-trainable) decoder learns to read our features.
        """
        if prob is None:
            prob = self.mlm_mask_prob
        labels = input_ids.clone()
        special = (
            (input_ids == self.vinvl.cls_token_id)
            | (input_ids == self.vinvl.sep_token_id)
            | (input_ids == self.vinvl.pad_token_id)
        )

        if prob >= 1.0:
            # All non-special tokens -> ọc, all are targets. No text leak.
            selected = ~special
            labels[~selected] = -100
            masked = input_ids.clone()
            masked[selected] = self.vinvl.mask_token_id
            return masked, labels

        selected = (torch.rand_like(input_ids, dtype=torch.float) < prob) & ~special
        labels[~selected] = -100

        masked = input_ids.clone()
        # 80% of selected -> ọc; 10% -> random token; 10% -> keep original.
        to_mask = selected & (torch.rand_like(input_ids, dtype=torch.float) < 0.8)
        to_rand = (
            selected & ~to_mask
            & (torch.rand_like(input_ids, dtype=torch.float) < 0.5)
        )
        masked[to_mask] = self.vinvl.mask_token_id
        vocab = self.vinvl.tokenizer.vocab_size
        masked[to_rand] = torch.randint(
            0, vocab, (int(to_rand.sum()),), device=input_ids.device
        )
        return masked, labels

    def _decode(
        self,
        concept_embeds: torch.Tensor,
        roi_features: torch.Tensor,
        batch_size: int,
        target_captions: Optional[List[str]],
    ) -> Dict[str, torch.Tensor]:
        tokenized = (
            self.vinvl.tokenizer(
                target_captions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.word_seq_length,
            ).to(self.device)
            if target_captions is not None
            else None
        )

        if target_captions is not None:
            input_ids = tokenized["input_ids"]
            masked_ids, labels = self._mlm_mask(input_ids)
            with torch.no_grad():
                word_embeds = self.vinvl.embeddings(input_ids=masked_ids)
        else:
            mask_ids = torch.full(
                (batch_size, self.word_seq_length),
                self.vinvl.mask_token_id,
                device=self.device,
            )
            word_embeds = self.vinvl.embeddings(input_ids=mask_ids)

        sequence_output, _ = self.vinvl(
            word_embeddings=word_embeds,
            concept_embeddings=concept_embeds,
            roi_features=roi_features,
        )

        word_output = sequence_output[:, :self.word_seq_length, :]
        logits = self.vinvl.lm_head(word_output)

        output = {"logits": logits}
        if target_captions is not None:
            output["labels"] = labels

        return output

    @torch.no_grad()
    def generate_caption(
        self,
        input_images: Optional[torch.Tensor] = None,
        max_length: int = 35,
        precomputed: Optional[Dict] = None,
    ) -> List[str]:
        self.eval()

        if precomputed is not None:
            return self._generate_precomputed(precomputed, max_length)

        input_images = input_images.to(self.device)
        batch_size = input_images.size(0)

        roi_features, concepts = self.extract_features_and_concepts(input_images)
        context_features = self.build_context_feature(roi_features)
        graphs = self.build_graphs(concepts, context_features)
        node_embs = self.get_concept_embeddings(graphs, context_features)
        edge_indices = self._get_edge_indices(graphs)

        enriched_concepts = []
        for idx in range(batch_size):
            emb = node_embs[idx]
            edge_idx = edge_indices[idx]
            if self.use_gnn:
                emb_batch = emb.unsqueeze(0)
                enriched = self.gnn(emb_batch, edge_idx)
            else:
                enriched = emb
            if enriched.dim() == 1:
                enriched = enriched.unsqueeze(0)
            enriched_concepts.append(enriched)

        concept_embeds = self._pad_and_stack(enriched_concepts)
        return self._generate(concept_embeds, roi_features, batch_size, max_length)

    def _generate_precomputed(
        self, precomputed: Dict, max_length: int
    ) -> List[str]:
        batch_size = len(precomputed["node_embeddings"])

        node_embs_list = [
            emb.to(self.device) for emb in precomputed["node_embeddings"]
        ]
        edge_indices = [
            ei.to(self.device) for ei in precomputed["edge_index"]
        ]
        roi_features = precomputed["roi_features"].to(self.device)

        enriched_concepts = []
        for idx in range(batch_size):
            emb = node_embs_list[idx]
            edge_idx = edge_indices[idx]
            if self.use_gnn:
                emb_batch = emb.unsqueeze(0)
                enriched = self.gnn(emb_batch, edge_idx)
            else:
                enriched = emb
            if enriched.dim() == 1:
                enriched = enriched.unsqueeze(0)
            enriched_concepts.append(enriched)

        concept_embeds = self._pad_and_stack(enriched_concepts)
        return self._generate(concept_embeds, roi_features, batch_size, max_length)

    def _generate(
        self,
        concept_embeds: torch.Tensor,
        roi_features: torch.Tensor,
        batch_size: int,
        max_length: int,
        num_iter: int = 10,
    ) -> List[str]:
        # Iterative Mask-Predict decoding (Ghazvininejad et al., 2019).
        # The decoder is a bidirectional MLM, not autoregressive. Single-pass
        # all-mask argmax is crude (every position independent, no refinement).
        # Instead: predict all, keep the most confident, re-mask the rest, repeat.
        # Over num_iter iterations the caption is progressively refined, giving
        # the model more context each iteration (like the partial masking in
        # training with mlm_mask_prob < 1.0).
        L = self.word_seq_length
        mask_id = self.vinvl.mask_token_id

        tokens = torch.full((batch_size, L), mask_id, dtype=torch.long,
                            device=self.device)

        for t in range(num_iter):
            word_embeds = self.vinvl.embeddings(input_ids=tokens)
            seq_out, _ = self.vinvl(
                word_embeddings=word_embeds,
                concept_embeddings=concept_embeds,
                roi_features=roi_features,
            )
            logits = self.vinvl.lm_head(seq_out[:, :L, :])
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)

            n_unmask = int(L * (t + 1) / num_iter)

            if t < num_iter - 1:
                for b in range(batch_size):
                    conf = confidence[b].clone()
                    already = tokens[b] != mask_id
                    conf[already] = float('inf')
                    n_keep_masked = L - n_unmask
                    if n_keep_masked > 0:
                        _, low_conf_idx = conf.sort()
                        keep_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
                        keep_mask[low_conf_idx[:n_keep_masked]] = True
                        fill = ~keep_mask
                        tokens[b, fill] = predicted[b, fill]
                    else:
                        tokens[b] = predicted[b]
            else:
                tokens = predicted

        captions = []
        for ids in tokens:
            caption = self.vinvl.tokenizer.decode(ids, skip_special_tokens=True)
            captions.append(caption)

        return captions
