import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests


CONCEPTNET_API_URL = "http://api.conceptnet.io"

RELATION_WHITELIST = {
    "IsA", "HasA", "PartOf", "UsedFor", "CapableOf",
    "AtLocation", "RelatedTo", "SymbolOf", "DefinedAs",
    "Entails", "MannerOf", "LocatedNear", "HasProperty",
    "MotivatedByGoal", "Causes", "HasSubevent", "MadeOf",
}

DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "conceptnet_cache"
)


class ConceptNetClient:
    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 10,
        language: str = "en",
        cache_dir: Optional[str] = None,
    ):
        self.max_retries = max_retries
        self.timeout = timeout
        self.language = language
        self.cache_dir = Path(cache_dir) if cache_dir else Path(DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._bulk_loaded = False

    def _bulk_load(self):
        """Load ALL cache files into memory at once. Avoids per-concept file
        I/O on Lustre (which is 17s/batch otherwise). Called once on first use."""
        if self._bulk_loaded:
            return
        import glob
        for path in glob.glob(str(self.cache_dir / "*.json")):
            key = Path(path).stem
            try:
                with open(path) as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    self._mem_cache[key] = [tuple(item) for item in data[:50]]
            except Exception:
                pass
        self._bulk_loaded = True
        print(f"[ConceptNet] bulk-loaded {len(self._mem_cache)} concepts into memory")

    def query_concept(self, concept: str) -> List[Tuple[str, str]]:
        self._bulk_load()
        concept_key = concept.lower().replace(" ", "_")

        if concept_key in self._mem_cache:
            return self._mem_cache[concept_key]

        disk_path = self.cache_dir / f"{concept_key}.json"
        if disk_path.exists():
            with open(disk_path) as f:
                result = [tuple(item) for item in json.load(f)]
            # Cap 1-hop neighbors to keep graph construction tractable.
            # With 200+ neighbors per concept, 2-hop queries explode O(N*M).
            result = result[:50]
            self._mem_cache[concept_key] = result
            return result

        # API unreachable — return empty (cache was pre-populated offline)
        self._mem_cache[concept_key] = []
        return []

    def _parse_edges(
        self, data: Dict, source_concept: str
    ) -> List[Tuple[str, str]]:
        related = []
        for edge in data.get("edges", []):
            rel_label = edge.get("rel", {}).get("label", "")
            if rel_label not in RELATION_WHITELIST:
                continue

            start = edge.get("start", {})
            end = edge.get("end", {})

            start_label = self._get_concept_label(start)
            end_label = self._get_concept_label(end)

            if start_label is None or end_label is None:
                continue

            if source_concept in start_label.lower():
                related.append((end_label, rel_label))
            elif source_concept in end_label.lower():
                related.append((start_label, rel_label))

        return related

    def _get_concept_label(self, node: Dict) -> Optional[str]:
        label = node.get("label", "")
        lang = node.get("language", "")
        if lang == self.language or lang == "":
            return label.lower().strip()
        return None

    def get_2hop_neighbors(self, concept: str) -> Dict[str, Set[str]]:
        direct = self.query_concept(concept)
        result: Dict[str, Set[str]] = {"direct": set(), "indirect": set()}

        for neighbor, _ in direct:
            result["direct"].add(neighbor)
            second_hop = self.query_concept(neighbor)
            for neighbor2, _ in second_hop:
                result["indirect"].add(neighbor2)

        return result

    def query_with_neighbors(
        self, concept: str
    ) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        direct = self.query_concept(concept)
        one_hop: List[Tuple[str, str]] = list(direct)
        two_hop: List[Tuple[str, str]] = []

        # Limit 2-hop queries to the top-N most relevant 1-hop neighbors
        # (sorted by relation strength) to avoid O(N*M) cache lookups when
        # 1-hop returns hundreds of neighbors. The paper only keeps top-60
        # forecasted concepts total, so querying all 1-hop neighbors for 2-hop
        # is wasteful.
        MAX_2HOP_QUERIES = 20
        for neighbor, rel in one_hop[:MAX_2HOP_QUERIES]:
            second_hop = self.query_concept(neighbor)
            for neighbor2, rel2 in second_hop:
                two_hop.append((neighbor2, rel2))

        return one_hop, two_hop
