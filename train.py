import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from config import Config, ModelConfig, TrainConfig, VISTConfig
from data import VISTDataset, build_vist_dataloaders
from data.precomputed_dataset import build_precomputed_dataloaders
from models.acap import ACap


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            config.model.device if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

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

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Model: {total_params:,} total params, "
            f"{trainable_params:,} trainable params"
        )

        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.output_dir = Path(config.train.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # score = -val_loss (always negative), so init to -inf so the first
        # epoch's val always becomes the first best and checkpoint_best.pt is
        # actually written. Was 0.0, which (score is negative) never beat, so
        # checkpoint_best.pt was never saved.
        self.best_score = float("-inf")
        self.global_step = 0
        self.scheduler = None

    def setup_scheduler(self, num_training_steps: int):
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.train.warmup_steps,
            num_training_steps=num_training_steps,
        )
        logger.info(
            f"Scheduler: warmup={self.config.train.warmup_steps}, "
            f"total={num_training_steps}"
        )

    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        start_time = time.time()

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch}/{self.config.train.num_epochs}"
        )
        for batch_idx, batch in enumerate(progress):
            target_captions = batch["target_caption"]

            outputs = self.model(precomputed=batch, target_captions=target_captions)
            logits = outputs["logits"]
            labels = outputs["labels"]

            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.config.train.max_grad_norm,
            )
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            total_loss += loss.item()

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            if self.global_step % self.config.train.log_interval == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {self.global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            if self.global_step % self.config.train.eval_interval == 0:
                val_metrics = self.evaluate(self.val_loader)
                logger.info(f"Validation metrics: {val_metrics}")
                self.save_checkpoint(f"step_{self.global_step}")

                if val_metrics.get("score", 0) > self.best_score:
                    self.best_score = val_metrics["score"]
                    self.save_checkpoint("best")
                    logger.info(f"New best score: {self.best_score:.4f}")

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = len(data_loader)

        for batch in tqdm(data_loader, desc="Evaluating"):
            target_captions = batch["target_caption"]

            outputs = self.model(precomputed=batch, target_captions=target_captions)
            logits = outputs["logits"]
            labels = outputs["labels"]

            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            total_loss += loss.item()

        avg_loss = total_loss / max(num_batches, 1)

        return {
            "loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
            "score": -avg_loss,
        }

    def save_checkpoint(self, tag: str):
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "config": self.config,
            "global_step": self.global_step,
            "best_score": self.best_score,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_score = checkpoint.get("best_score", 0.0)
        logger.info(f"Checkpoint loaded: {path}")

    def train(self):
        logger.info("Building precomputed dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = build_precomputed_dataloaders(
            self.config
        )
        logger.info(
            f"Train: {len(self.train_loader.dataset)} samples, "
            f"Val: {len(self.val_loader.dataset)} samples, "
            f"Test: {len(self.test_loader.dataset)} samples"
        )

        num_training_steps = (
            len(self.train_loader) * self.config.train.num_epochs
        )
        self.setup_scheduler(num_training_steps)

        logger.info("Starting training...")
        for epoch in range(1, self.config.train.num_epochs + 1):
            train_metrics = self.train_epoch(self.train_loader, epoch)
            logger.info(f"Epoch {epoch} train metrics: {train_metrics}")

            val_metrics = self.evaluate(self.val_loader)
            logger.info(f"Epoch {epoch} val metrics: {val_metrics}")

            if val_metrics.get("score", 0) > self.best_score:
                self.best_score = val_metrics["score"]
                self.save_checkpoint("best")
                logger.info(f"New best score: {self.best_score:.4f}")

            self.save_checkpoint(f"epoch_{epoch}")

        logger.info("Training completed!")
        logger.info(f"Best score: {self.best_score}")

        test_metrics = self.evaluate(self.test_loader)
        logger.info(f"Test metrics: {test_metrics}")

        return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train A-CAP model")
    parser.add_argument("--data-root", type=str, default="data/vist")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--no-gnn", action="store_true", help="Ablation: disable GNN")
    parser.add_argument(
        "--no-context", action="store_true", help="Ablation: disable context"
    )
    args = parser.parse_args()

    experiment_name = "acap_vist"
    if args.no_gnn:
        experiment_name += "_no_gnn"
    if args.no_context:
        experiment_name += "_no_context"

    config = Config(
        model=ModelConfig(
            device=args.device,
            use_gnn=not args.no_gnn,
            use_context=not args.no_context,
        ),
        train=TrainConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            output_dir=args.output_dir,
        ),
        vist=VISTConfig(data_root=args.data_root),
        experiment_name=experiment_name,
    )

    trainer = Trainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
