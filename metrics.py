import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MetricEvaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._clip_model = None
        self._clip_processor = None

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor

            # transformers 5.x refuses torch.load on torch <2.6 (CVE check), so
            # force safetensors loading — the CLIP repo ships model.safetensors.
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", use_safetensors=True
            ).to(self.device)
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_model.eval()
            for p in self._clip_model.parameters():
                p.requires_grad = False

    def compute_caption_metrics(
        self,
        generated: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        results: Dict[str, float] = {}

        try:
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.cider.cider import Cider

            gts = {i: [r] for i, r in enumerate(references)}
            res = {i: [g] for i, g in enumerate(generated)}

            bleu_scorer = Bleu(4)
            bleu_scores, _ = bleu_scorer.compute_score(gts, res)
            results["BLEU-1"] = bleu_scores[0] * 100
            results["BLEU-4"] = bleu_scores[3] * 100

            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(gts, res)
            results["CIDEr"] = cider_score * 100

            try:
                from pycocoevalcap.spice.spice import Spice

                spice_scorer = Spice()
                spice_score, _ = spice_scorer.compute_score(gts, res)
                results["SPICE"] = float(spice_score["All"]["f"]) * 100
            except Exception as e:
                logger.warning(f"SPICE computation failed (requires Java): {e}")
                results["SPICE"] = 0.0

        except ImportError:
            logger.warning(
                "pycocoevalcap not installed, falling back to simple BLEU"
            )
            results.update(self._simple_bleu(generated, references))

        return results

    def _simple_bleu(
        self, generated: List[str], references: List[str]
    ) -> Dict[str, float]:
        from collections import Counter
        import math

        def bleu_n(candidate: str, reference: str, n: int) -> float:
            cand_tokens = candidate.lower().split()
            ref_tokens = reference.lower().split()
            if len(cand_tokens) < n or len(ref_tokens) < n:
                return 0.0
            cand_ngrams = Counter(
                tuple(cand_tokens[i : i + n])
                for i in range(len(cand_tokens) - n + 1)
            )
            ref_ngrams = Counter(
                tuple(ref_tokens[i : i + n])
                for i in range(len(ref_tokens) - n + 1)
            )
            matches = sum(
                min(count, ref_ngrams.get(ngram, 0))
                for ngram, count in cand_ngrams.items()
            )
            total = sum(cand_ngrams.values())
            if total == 0:
                return 0.0
            precision = matches / total
            bp = min(
                1.0, math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
            )
            return bp * precision

        b1_scores = [bleu_n(g, r, 1) for g, r in zip(generated, references)]
        b4_scores = [bleu_n(g, r, 4) for g, r in zip(generated, references)]

        return {
            "BLEU-1": sum(b1_scores) / len(b1_scores) * 100,
            "BLEU-4": sum(b4_scores) / len(b4_scores) * 100,
            "CIDEr": 0.0,
            "SPICE": 0.0,
        }

    def compute_clip_score(
        self,
        generated: List[str],
        oracle_images: List[Image.Image],
    ) -> float:
        self._load_clip()

        scores = []
        for caption, image in zip(generated, oracle_images):
            inputs = self._clip_processor(
                text=[caption], images=image, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                sim = outputs.logits_per_image.item()
            scores.append(max(sim, 0.0))

        w = 2.5
        avg_score = np.mean(scores) if scores else 0.0
        return w * avg_score

    def compute_ref_clip_score(
        self,
        generated: List[str],
        references: List[str],
        oracle_images: List[Image.Image],
    ) -> float:
        self._load_clip()

        scores = []
        for caption, reference, image in zip(generated, references, oracle_images):
            inputs = self._clip_processor(
                text=[reference], images=image, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                ref_sim = outputs.logits_per_image.item()

            inputs = self._clip_processor(
                text=[caption], images=image, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                cap_sim = outputs.logits_per_image.item()

            harmonic = (
                2 * cap_sim * ref_sim / max(cap_sim + ref_sim, 1e-8)
                if (cap_sim + ref_sim) > 0
                else 0.0
            )
            scores.append(max(harmonic, 0.0))

        w = 2.5
        avg_score = np.mean(scores) if scores else 0.0
        return w * avg_score

    def compute_retrieval(
        self,
        generated: List[str],
        oracle_images: List[Image.Image],
        k_values: Tuple[int, ...] = (1, 5, 10),
    ) -> Dict[str, float]:
        self._load_clip()

        text_inputs = self._clip_processor(
            text=generated, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            text_features = self._clip_model.get_text_features(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features_list = []
        for image in oracle_images:
            inputs = self._clip_processor(
                images=image, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                feat = self._clip_model.get_image_features(
                    pixel_values=inputs["pixel_values"]
                )
                feat = feat / feat.norm(dim=-1, keepdim=True)
            image_features_list.append(feat.squeeze(0))

        image_features = torch.stack(image_features_list)

        sim_matrix = text_features @ image_features.T

        n = len(generated)
        ranks = []
        for i in range(n):
            scores = sim_matrix[i]
            sorted_indices = torch.argsort(scores, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
            ranks.append(rank + 1)

        results = {}
        for k in k_values:
            hit = sum(1 for r in ranks if r <= k)
            results[f"R@{k}"] = hit / n * 100

        return results

    def compute_all_metrics(
        self,
        generated: List[str],
        references: List[str],
        oracle_images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, float]:
        metrics = self.compute_caption_metrics(generated, references)

        if oracle_images is not None and len(oracle_images) == len(generated):
            try:
                metrics["CLIPScore"] = self.compute_clip_score(
                    generated, oracle_images
                )
            except Exception as e:
                logger.warning(f"CLIPScore failed: {e}")
                metrics["CLIPScore"] = 0.0

            try:
                metrics["RefCLIPScore"] = self.compute_ref_clip_score(
                    generated, references, oracle_images
                )
            except Exception as e:
                logger.warning(f"RefCLIPScore failed: {e}")
                metrics["RefCLIPScore"] = 0.0

            try:
                retrieval = self.compute_retrieval(generated, oracle_images)
                metrics.update(retrieval)
            except Exception as e:
                logger.warning(f"Retrieval failed: {e}")
                for k in (1, 5, 10):
                    metrics[f"R@{k}"] = 0.0
        else:
            metrics["CLIPScore"] = 0.0
            metrics["RefCLIPScore"] = 0.0
            for k in (1, 5, 10):
                metrics[f"R@{k}"] = 0.0

        return metrics
