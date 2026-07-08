"""
Maritime-STPCN: Standalone Evaluation Script

Evaluate a saved model checkpoint on test data with bootstrap CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from maritime_stpcn.config import STPCNConfig
from maritime_stpcn.data import generate_synthetic_maritime_data, MaritimeDataset
from maritime_stpcn.model import MaritimeSTPCN
from maritime_stpcn.evaluator import STPCNEvaluator


def evaluate_checkpoint(checkpoint_path: str, config_path: str = "configs/default.yaml",
                        device: str = "auto"):
    """Evaluate a saved checkpoint."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config
    config = STPCNConfig.from_yaml(config_path)

    # Generate data
    data_dict = generate_synthetic_maritime_data(
        num_vessels=config.data.num_vessels,
        num_zones=config.data.num_zones,
        seed=config.train.seed,
    )
    dataset = MaritimeDataset(data_dict, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_from_checkpoint = checkpoint.get("config", config.to_dict())

    # Create model and load weights
    model = MaritimeSTPCN(
        num_vessels=config.data.num_vessels,
        num_zones=config.data.num_zones,
        behaviors=config.data.behaviors,
        target_behavior=config.data.target_behavior,
        relation_order=config.data.relation_order,
        behavior_layers=config.data.behavior_layers,
        embedding_dim=config.model.embedding_dim,
        contrastive_dim=config.model.contrastive_dim,
        global_bias_init=data_dict["global_bias"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint loaded from {checkpoint_path}")

    # Evaluate
    evaluator = STPCNEvaluator(config)
    metrics = evaluator.evaluate(model, dataset, device)

    print("\nEvaluation Results:")
    for k in [5, 10, 20, 40]:
        print(f"  Recall@{k}:  {metrics.get(f'Recall@{k}', 0):.4f}")
        print(f"  NDCG@{k}:    {metrics.get(f'NDCG@{k}', 0):.4f}")
        print(f"  F1@{k}:      {metrics.get(f'F1@{k}', 0):.4f}")
        if f"Recall@{k}_CI_low" in metrics:
            print(f"  Recall@{k} CI: [{metrics[f'Recall@{k}_CI_low']:.4f}, {metrics[f'Recall@{k}_CI_high']:.4f}]")

    # Save results
    results_dir = Path("results/evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Maritime-STPCN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.config, args.device)
