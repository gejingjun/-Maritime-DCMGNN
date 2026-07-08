"""
Maritime-STPCN Training System (v2)

Aligned with paper Section 4.10:
- Adam optimizer (beta1=0.9, beta2=0.999)
- Warmup-cosine annealing (Eq.36): warmup 5 epochs from 1e-4 to 5e-3
- Gradient clipping max_norm=1.0
- NaN detection + checkpoint restoration
- Early stopping patience=20 on val NDCG@10
- Fixed seed 123 for reproducibility
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import optim

from .config import STPCNConfig
from .data import MaritimeDataset, BPRBatchSampler, generate_synthetic_maritime_data
from .model import MaritimeSTPCN
from .losses import STPCNLoss
from .evaluator import STPCNEvaluator
from .visualization import plot_training_curves, plot_comparison_table


class WarmupCosineScheduler:
    """
    LR scheduler (Eq.36): linear warmup then cosine annealing.
    warmup: lr = lr_start + (lr_0 - lr_start) * (step / warmup_steps)
    cosine: lr = lr_min + 0.5 * (lr_0 - lr_min) * (1 + cos(pi * progress))
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        lr_0: float = 5e-3,
        warmup_epochs: int = 5,
        warmup_start_lr: float = 1e-4,
        min_lr: float = 1e-5,
        total_epochs: int = 200,
    ):
        self.optimizer = optimizer
        self.lr_0 = lr_0
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        """Update learning rate for current epoch."""
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup (Eq.36 warmup phase)
            lr = self.warmup_start_lr + (self.lr_0 - self.warmup_start_lr) * (
                self.current_epoch / self.warmup_epochs
            )
        else:
            # Cosine annealing (Eq.36)
            t_warmup = self.warmup_epochs
            progress = (self.current_epoch - t_warmup) / max(
                self.total_epochs - t_warmup, 1
            )
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.lr_0 - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


@dataclass
class TrainingHistory:
    """Track all training metrics."""
    epochs: list = field(default_factory=list)
    train_losses: list = field(default_factory=list)
    val_metrics: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    best_val_metric: float = 0.0
    best_epoch: int = 0


class STPCNTrainer:
    """
    Complete training loop for Maritime-STPCN.

    Features per paper Section 4.10-4.11:
    - Adam optimizer, warmup-cosine LR
    - Gradient clipping (max_norm=1.0)
    - NaN detection with checkpoint restoration
    - Early stopping (patience=20 on val NDCG@10)
    - Rcl warmup: linearly ramped from 0 to lambda_rcl over first 5 epochs
    - Seed 123 fixed for reproducibility
    """

    def __init__(
        self,
        config: STPCNConfig,
        device: str = "auto",
    ):
        self.config = config
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set random seed (paper Section 4.11: seed=123)
        seed = config.train.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Generate data
        print("Generating synthetic maritime dataset...")
        data_dict = generate_synthetic_maritime_data(
            num_vessels=config.data.num_vessels,
            num_zones=config.data.num_zones,
            seed=seed,
        )
        self.dataset = MaritimeDataset(data_dict, device=str(self.device))

        # Create model
        self.model = MaritimeSTPCN(
            num_vessels=config.data.num_vessels,
            num_zones=config.data.num_zones,
            behaviors=config.data.behaviors,
            target_behavior=config.data.target_behavior,
            relation_order=config.data.relation_order,
            behavior_layers=config.data.behavior_layers,
            embedding_dim=config.model.embedding_dim,
            contrastive_dim=config.model.contrastive_dim,
            max_propagation_depth=config.model.max_propagation_depth,
            num_stp_layers=config.model.num_layers,
            global_bias_init=data_dict["global_bias"],
            per_zone_bias_init=data_dict.get("per_zone_bias_init"),
            dropout=config.model.dropout,
        ).to(self.device)

        # Loss function
        self.loss_fn = STPCNLoss(
            lambda_rcl=config.train.lambda_rcl,
            lambda_chain=config.train.lambda_chain,
            lambda_entropy=config.train.lambda_entropy,
            lambda_l2=config.train.lambda_l2,
            num_behaviors=len(config.data.behaviors),
            contrastive_dim=config.model.contrastive_dim,
        )

        # Evaluator
        self.evaluator = STPCNEvaluator(config)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            lr_0=config.train.learning_rate,
            warmup_epochs=config.train.warmup_epochs,
            warmup_start_lr=config.train.warmup_start_lr,
            min_lr=config.train.min_lr,
            total_epochs=config.train.max_epochs,
        )

        # BPR sampler
        self.sampler = BPRBatchSampler(self.dataset, num_negatives=1)

        # History
        self.history = TrainingHistory()
        self.patience_counter = 0
        self.best_state_dict = None

        # Rcl warmup tracking
        self._rcl_current = 0.0  # Start at 0, ramp to lambda_rcl

        print(f"\n{'='*60}")
        print(f"Maritime-STPCN Training Setup")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Vessels: {self.dataset.num_vessels}, Zones: {self.dataset.num_zones}")
        print(f"Behaviors: {self.dataset.num_behaviors}")
        print(f"Target: {self.dataset.target_behavior}")
        print(f"{'='*60}\n")

    def train(self) -> TrainingHistory:
        """Full training loop."""
        max_epochs = self.config.train.max_epochs
        eval_every = 5  # Evaluate every 5 epochs
        patience = self.config.train.early_stopping_patience
        batch_size = self.config.train.batch_size

        for epoch in range(1, max_epochs + 1):
            epoch_start = time.time()
            self.model.train()

            # ── Rcl warmup (linearly ramp from 0 to lambda_rcl over 5 epochs) ──
            if epoch <= self.config.train.warmup_epochs:
                self._rcl_current = self.config.train.lambda_rcl * (
                    epoch / self.config.train.warmup_epochs
                )
            else:
                self._rcl_current = self.config.train.lambda_rcl
            self.loss_fn.lambda_rcl = self._rcl_current

            # ── Gumbel-Softmax tau annealing ──
            self.model.anneal_gumbel_tau(epoch, max_epochs)

            # ── Training batches ──
            epoch_losses = []
            epoch_loss_dict = {}
            n_batches = max(1, len(self.sampler.pos_pairs) // batch_size)

            for batch_idx in range(n_batches):
                vessel_ids, pos_zones, neg_zones = self.sampler.sample_batch(batch_size)
                vessel_ids = vessel_ids.to(self.device)
                pos_zones = pos_zones.to(self.device)
                neg_zones = neg_zones.to(self.device)

                # Forward
                outputs = self.model(self.dataset)

                # Loss
                loss, loss_dict = self.loss_fn.compute_total_loss(
                    self.model, outputs,
                    vessel_ids, pos_zones, neg_zones,
                    self.dataset.num_vessels,
                )

                # NaN detection
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  [WARNING] NaN/Inf at epoch {epoch}, batch {batch_idx}")
                    if self.best_state_dict is not None:
                        self.model.load_state_dict(self.best_state_dict)
                        print("  Restored best checkpoint")
                    continue

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.grad_clip_norm
                )

                self.optimizer.step()
                epoch_losses.append(loss.item())
                epoch_loss_dict = loss_dict

            # ── LR update ──
            current_lr = self.scheduler.step()

            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            epoch_time = time.time() - epoch_start

            # ── Evaluation ──
            val_metrics = {}
            if epoch % eval_every == 0 or epoch == max_epochs:
                val_metrics = self.evaluator.evaluate(self.model, self.dataset, str(self.device))

                r10 = val_metrics.get("Recall@10", 0)
                n10 = val_metrics.get("NDCG@10", 0)
                f10 = val_metrics.get("F1@10", 0)

                print(
                    f"Epoch {epoch:>4}/{max_epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"R@10: {r10:.4f} | N@10: {n10:.4f} | "
                    f"F1@10: {f10:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # Best model check
                if r10 > self.history.best_val_metric:
                    self.history.best_val_metric = r10
                    self.history.best_epoch = epoch
                    self.best_state_dict = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                    self.patience_counter = 0
                    print(f"  >>> New best R@10: {r10:.4f}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch}")
                    break
            else:
                print(
                    f"Epoch {epoch:>4}/{max_epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            self.history.epochs.append(epoch)
            self.history.train_losses.append(avg_loss)
            self.history.val_metrics.append(val_metrics)
            self.history.learning_rates.append(current_lr)

        # Restore best model
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            print(f"\nRestored best model from epoch {self.history.best_epoch}")

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best R@10: {self.history.best_val_metric:.4f} at epoch {self.history.best_epoch}")
        print(f"{'='*60}")

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
        }, path)

    def save_results(self, save_dir: str = "results"):
        """Save training history, evaluation results, and plots."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save training history
        history_dict = {
            "epochs": self.history.epochs,
            "train_losses": self.history.train_losses,
            "val_metrics": self.history.val_metrics,
            "learning_rates": self.history.learning_rates,
            "best_val_metric": self.history.best_val_metric,
            "best_epoch": self.history.best_epoch,
        }
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history_dict, f, indent=2)

        # Final evaluation with bootstrap CI
        final_metrics = self.evaluator.evaluate(self.model, self.dataset, str(self.device))
        with open(save_dir / "final_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=2)

        # Generate plots
        plot_files = plot_training_curves(history_dict, str(save_dir))
        print(f"Results saved to {save_dir}")
        print(f"Plots: {plot_files}")

        return save_dir


import random
