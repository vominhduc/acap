from typing import Dict, List, Optional, Set, Tuple

import torch


class KnowledgeGraph:
    def __init__(self):
        self.nodes: List[str] = []
        self.node_to_idx: Dict[str, int] = {}
        self.edge_index: List[Tuple[int, int]] = []
        self.detected_mask: List[bool] = []
        self.forecasted_mask: List[bool] = []
        self._concept_neighbors: Dict[str, Set[str]] = {}

    def add_node(self, concept: str, is_detected: bool) -> int:
        if concept not in self.node_to_idx:
            idx = len(self.nodes)
            self.nodes.append(concept)
            self.node_to_idx[concept] = idx
            self.detected_mask.append(is_detected)
            self.forecasted_mask.append(not is_detected)
            return idx
        return self.node_to_idx[concept]

    def add_edge(self, src: str, dst: str):
        if src in self.node_to_idx and dst in self.node_to_idx:
            self.edge_index.append((self.node_to_idx[src], self.node_to_idx[dst]))

    def set_neighbors(self, concept: str, neighbors: Set[str]):
        self._concept_neighbors[concept] = neighbors

    def has_edge_between(self, concept_a: str, concept_b: str) -> bool:
        neighbors = self._concept_neighbors.get(concept_a, set())
        return concept_b in neighbors

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edge_index)

    def to_torch_edge_index(self) -> torch.Tensor:
        if not self.edge_index:
            return torch.zeros((2, 0), dtype=torch.long)
        return torch.tensor(self.edge_index, dtype=torch.long).t()

    def get_node_embeddings(
        self, bert_model, tokenizer, device: str
    ) -> torch.Tensor:
        embeddings = []
        for concept in self.nodes:
            inputs = tokenizer(
                concept, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                output = bert_model(**inputs)
                emb = output.last_hidden_state[:, 0, :]
            embeddings.append(emb)
        return torch.cat(embeddings, dim=0)


class KnowledgeGraphConstructor:
    def __init__(
        self,
        conceptnet_client: "ConceptNetClient",
        relevance_filter: "RelevanceFilter",
        num_forecasted: int = 60,
    ):
        self.conceptnet = conceptnet_client
        self.relevance_filter = relevance_filter
        self.num_forecasted = num_forecasted

    def build(
        self,
        detected_concepts: List[List[str]],
        context_feature: torch.Tensor,
        device: str = "cuda",
    ) -> KnowledgeGraph:
        graph = KnowledgeGraph()

        all_detected: Set[str] = set()
        for concepts in detected_concepts:
            for c in concepts:
                cleaned = c.lower().replace(" ", "_")
                all_detected.add(cleaned)
                graph.add_node(cleaned, is_detected=True)

        raw_forecasted: Set[str] = set()
        concept_neighbors: Dict[str, Set[str]] = {}

        for concept in all_detected:
            one_hop, two_hop = self.conceptnet.query_with_neighbors(concept)

            neighbors = set()
            for neighbor, _ in one_hop:
                neighbors.add(neighbor)
            concept_neighbors[concept] = neighbors

            for neighbor, _ in one_hop:
                if neighbor not in all_detected:
                    raw_forecasted.add(neighbor)
                if neighbor not in concept_neighbors:
                    second_hop_neighbors = set()
                    for n2, _ in two_hop:
                        second_hop_neighbors.add(n2)
                    concept_neighbors[neighbor] = second_hop_neighbors

            for neighbor2, _ in two_hop:
                if neighbor2 not in all_detected:
                    raw_forecasted.add(neighbor2)

        raw_forecasted_list = list(raw_forecasted)

        if raw_forecasted_list:
            scores = self.relevance_filter.score_concepts(
                raw_forecasted_list, context_feature, device
            )
            scored = list(zip(raw_forecasted_list, scores))
            scored.sort(key=lambda x: -x[1])
            top_forecasted = scored[: self.num_forecasted]
        else:
            top_forecasted = []

        for concept, _ in top_forecasted:
            graph.add_node(concept, is_detected=False)

        for concept in graph.nodes:
            neighbors = concept_neighbors.get(concept, set())
            graph.set_neighbors(concept, neighbors)

        self._build_edges(graph)

        return graph

    def _build_edges(self, graph: KnowledgeGraph):
        nodes = graph.nodes
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                if graph.has_edge_between(src, dst) or graph.has_edge_between(dst, src):
                    graph.add_edge(src, dst)
                    graph.add_edge(dst, src)


def build_temporal_edges(graph: KnowledgeGraph, detected_concepts: List[List[str]]):
    num_inputs = len(detected_concepts)
    for t in range(num_inputs - 1):
        current = set(c.lower().replace(" ", "_") for c in detected_concepts[t])
        next_set = set(c.lower().replace(" ", "_") for c in detected_concepts[t + 1])
        for c in current:
            for n in next_set:
                if c in graph.node_to_idx and n in graph.node_to_idx:
                    graph.add_edge(c, n)
                    graph.add_edge(n, c)
