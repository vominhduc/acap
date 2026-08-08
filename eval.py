import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from config import Config, ModelConfig, TrainConfig, VISTConfig
from data.vist_dataset import VISTDataset, vist_collate_fn
from models.acap import ACap
from metrics import MetricEvaluator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: Config, checkpoint_path: str):
        self.device = torch.device(
            config.model.device if torch.cuda.is_available() else "cpu"
        )

        self.model = ACap(
            num_input_images=config.vist.num_input_images,
            num_detected_per_image=config.model.num_detected_per_image,
            num_forecasted=config.model.num_forecasted,
            num_rois_per_image=config.model.num_rois_per_image,
            word_seq_length=config.model.word_seq_length,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.hidden_dim,
            num_gat_layers=config.model.num_gat_layers,
            num_gat_heads=config.model.num_gat_heads,
            dropout=config.model.dropout,
            bert_model_name=config.model.bert_model_name,
            vinvl_model_name=config.model.vinvl_model_name,
            device=str(self.device),
            use_gnn=config.model.use_gnn,
            use_context=config.model.use_context,
            roi_feature_dim=config.model.roi_feature_dim,
            freeze_vinvl=config.model.freeze_vinvl,
            mlm_mask_prob=config.model.mlm_mask_prob,
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("vinvl._embeddings")}
        self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.model.eval()

        self.metric_evaluator = MetricEvaluator(device=str(self.device))

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict:
        self.model.eval()
        all_generated = []
        all_targets = []
        all_oracle_images = []

        for batch in tqdm(data_loader, desc="Evaluating"):
            target_captions = batch["target_caption"]

            generated = self.model.generate_caption(precomputed=batch)
            all_generated.extend(generated)
            all_targets.extend(target_captions)

            if "oracle_image_pil" in batch:
                all_oracle_images.extend(batch["oracle_image_pil"])

        results = {
            "generated": all_generated,
            "targets": all_targets,
            "oracle_images": all_oracle_images,
        }
        return results

    def compute_metrics(self, results: Dict) -> Dict[str, float]:
        generated = results["generated"]
        targets = results["targets"]
        oracle_images = results.get("oracle_images", [])

        oracle_pil = None
        if oracle_images and isinstance(oracle_images[0], Image.Image):
            oracle_pil = oracle_images

        metrics = self.metric_evaluator.compute_all_metrics(
            generated, targets, oracle_pil
        )
        metrics["count"] = len(generated)
        return metrics

    def save_results(self, results: Dict, output_path: str):
        save_data = {
            "generated": results["generated"],
            "targets": results["targets"],
        }
        with open(output_path, "w") as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate A-CAP model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/vist")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="eval_results.json")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--no-gnn", action="store_true", help="Ablation: disable GNN")
    parser.add_argument("--no-context", action="store_true", help="Ablation: disable context")
    args = parser.parse_args()

    config = Config(
        model=ModelConfig(
            device=args.device,
            use_gnn=not args.no_gnn,
            use_context=not args.no_context,
        ),
        train=TrainConfig(batch_size=args.batch_size),
        vist=VISTConfig(data_root=args.data_root),
    )

    from data.precomputed_dataset import PrecomputedVISTDataset, precomputed_collate_fn

    precomputed_dir = getattr(config.vist, "precomputed_dir", "data/vist/preprocessed")
    dataset = PrecomputedVISTDataset(
        precomputed_file=f"{precomputed_dir}/{args.split}_preprocessed.pkl",
        image_root=config.vist.image_root,
        is_test=(args.split == "test"),
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=precomputed_collate_fn,
    )

    evaluator = Evaluator(config, args.checkpoint)
    results = evaluator.evaluate(data_loader)

    metrics = evaluator.compute_metrics(results)
    logger.info(f"Metrics:\n{json.dumps(metrics, indent=2)}")

    results["metrics"] = metrics
    evaluator.save_results(results, args.output)

    return metrics


if __name__ == "__main__":
    main()
