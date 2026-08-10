import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from config import Config
from models.vinvl_feature_extractor import VinVLFeatureExtractor
from models.concept_net_client import ConceptNetClient
from models.relevance_filter import RelevanceFilter
from models.knowledge_graph import KnowledgeGraphConstructor, build_temporal_edges
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_vist_samples(ann_file: str, image_root: str, num_input_images: int = 4):
    with open(ann_file) as f:
        annotations = json.load(f)

    samples = []
    image_root = Path(image_root)
    for item in annotations:
        if not isinstance(item, dict) or "images" not in item or "sentences" not in item:
            continue
        images = item["images"]
        sentences = item["sentences"]
        if len(images) < num_input_images + 1 or len(sentences) < num_input_images + 1:
            continue

        input_images = images[:num_input_images]
        oracle_image = images[num_input_images]
        all_imgs = input_images + [oracle_image]

        if any(not (image_root / img).exists() for img in all_imgs):
            continue

        if any(not _is_valid_image(image_root / img) for img in all_imgs):
            continue

        target_caption = sentences[-1]
        input_captions = sentences[:num_input_images]

        samples.append({
            "input_images": input_images,
            "oracle_image": oracle_image,
            "target_caption": target_caption,
            "input_captions": input_captions,
            "story_id": item.get("story_id", ""),
        })
    return samples


def preprocess_split(
    config: Config,
    split: str,
    output_dir: str,
):
    ann_file = getattr(config.vist, f"{split}_ann_file")
    image_root = Path(config.vist.image_root)
    num_input = config.vist.num_input_images

    logger.info(f"Loading {split} annotations from {ann_file}")
    samples = load_vist_samples(ann_file, image_root, num_input)
    logger.info(f"Found {len(samples)} valid samples")

    device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    logger.info("Initializing Faster-RCNN feature extractor...")
    extractor = VinVLFeatureExtractor(
        num_rois_per_image=config.model.num_rois_per_image,
        top_k_concepts=config.model.num_detected_per_image,
        device=str(device),
    )

    logger.info("Initializing ConceptNet client...")
    conceptnet_client = ConceptNetClient()

    logger.info("Initializing BERT for context...")
    bert_tokenizer = AutoTokenizer.from_pretrained(config.model.bert_model_name)
    bert_model = AutoModel.from_pretrained(config.model.bert_model_name)
    bert_model.eval()
    bert_model.to(device)
    for p in bert_model.parameters():
        p.requires_grad = False

    logger.info("Initializing RoBERTa relevance filter...")
    relevance_filter = RelevanceFilter(device=str(device))

    logger.info("Initializing knowledge graph constructor...")
    kg_constructor = KnowledgeGraphConstructor(
        conceptnet_client=conceptnet_client,
        relevance_filter=relevance_filter,
        num_forecasted=config.model.num_forecasted,
    )

    roi_proj = torch.nn.Linear(config.model.roi_feature_dim, config.model.embed_dim).to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    batch_size = 8

    for start_idx in tqdm(range(0, len(samples), batch_size), desc=f"Preprocessing {split}"):
        batch_samples = samples[start_idx:start_idx + batch_size]

        try:
            image_tensors = []
            for sample in batch_samples:
                imgs = torch.stack([
                    transform(Image.open(image_root / img).convert("RGB"))
                    for img in sample["input_images"]
                ])
                image_tensors.append(imgs)
            image_batch = torch.stack(image_tensors).to(device)
        except Exception as e:
            logger.warning(f"Skipping batch {start_idx}: {e}")
            continue

        with torch.no_grad():
            roi_features, all_concepts = extractor.extract_features(image_batch)
            roi_projected = roi_proj(roi_features)
            context_features = roi_projected.mean(dim=1)

        for i, sample in enumerate(batch_samples):
            concepts_i = all_concepts[i]
            context_feat = context_features[i]
            roi_feat = roi_features[i]

            # Enhance context with input caption text embeddings. The visual
            # context alone (mean of ROI features) doesn't capture the story's
            # narrative direction. The 4 input captions contain the narrative
            # arc ("family goes to carnival", "rides and games") that should
            # guide which forecasted concepts are relevant. We BERT-encode the
            # concatenated input captions and combine with the visual context.
            input_captions = sample.get("input_captions", [])
            if input_captions:
                caption_text = " ".join(input_captions)
                cap_inputs = bert_tokenizer(
                    caption_text, return_tensors="pt", padding=True,
                    truncation=True, max_length=128
                ).to(device)
                with torch.no_grad():
                    cap_output = bert_model(**cap_inputs)
                    cap_emb = cap_output.last_hidden_state[:, 0, :]  # (1, 768)
                # Combine: 50% visual + 50% text context
                context_feat = 0.5 * context_feat + 0.5 * cap_emb.squeeze(0)

            graph = kg_constructor.build(concepts_i, context_feat, str(device))
            build_temporal_edges(graph, concepts_i)

            node_texts = graph.nodes
            edge_index = graph.to_torch_edge_index()

            # Batch ALL concepts into a single BERT forward (vs 100 individual
            # forwards). This is the preprocessing bottleneck with ~100 nodes.
            if node_texts:
                batch_inputs = bert_tokenizer(
                    [c.replace("_", " ") for c in node_texts],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=16,
                ).to(device)
                with torch.no_grad():
                    output = bert_model(**batch_inputs)
                    node_embeddings = output.last_hidden_state[:, 0, :].cpu()
            else:
                node_embeddings = torch.zeros(1, config.model.embed_dim)

            if config.model.use_context:
                context_expanded = context_feat.unsqueeze(0).expand(node_embeddings.size(0), -1).cpu()
                augmented = torch.cat([node_embeddings, context_expanded], dim=-1)
            else:
                augmented = node_embeddings

            results.append({
                "story_id": sample["story_id"],
                "concepts": concepts_i,
                "node_texts": node_texts,
                "node_embeddings": augmented,
                "edge_index": edge_index,
                "roi_features": roi_feat.cpu(),
                "context_feature": context_feat.cpu(),
                "target_caption": sample["target_caption"],
                "input_captions": sample["input_captions"],
                "oracle_image": sample["oracle_image"],
                "input_images": sample["input_images"],
            })

        if (start_idx // batch_size + 1) % 50 == 0:
            logger.info(f"Processed {start_idx + len(batch_samples)}/{len(samples)} samples")

    output_file = output_dir / f"{split}_preprocessed.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved {len(results)} preprocessed samples to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-compute concepts and features for A-CAP")
    parser.add_argument("--data-root", type=str, default="data/vist")
    parser.add_argument("--output-dir", type=str, default="data/vist/preprocessed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num-forecasted", type=int, default=60)
    parser.add_argument("--no-gnn", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    args = parser.parse_args()

    config = Config(
        model=__import__("config").ModelConfig(
            device=args.device,
            num_forecasted=args.num_forecasted,
            use_gnn=not args.no_gnn,
            use_context=not args.no_context,
        ),
        vist=__import__("config").VISTConfig(data_root=args.data_root),
    )

    for split in args.splits:
        logger.info(f"\n{'='*60}\nProcessing {split}\n{'='*60}")
        preprocess_split(config, split, args.output_dir)


if __name__ == "__main__":
    main()
