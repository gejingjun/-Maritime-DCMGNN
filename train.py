"""
Maritime-STPCN: Main Training & Evaluation Entry Point

Usage:
    python train.py                     # Full model training
    python train.py --config configs/default.yaml
    python train.py --ablation          # Run ablation experiments
    python train.py --sparsity          # Run sparsity analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from maritime_stpcn.config import STPCNConfig
from maritime_stpcn.data import generate_synthetic_maritime_data, MaritimeDataset, BPRBatchSampler
from maritime_stpcn.model import MaritimeSTPCN
from maritime_stpcn.trainer import STPCNTrainer
from maritime_stpcn.evaluator import STPCNEvaluator
from maritime_stpcn.visualization import plot_training_curves, plot_comparison_table


def set_seed(seed: int = 123):
    """Fixed random seed (paper Section 4.11)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_full_model(config: STPCNConfig, device: str = "auto") -> dict:
    """Train the full Maritime-STPCN model (all M1-M7)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = STPCNTrainer(config, device=device)
    history = trainer.train()
    save_dir = trainer.save_results("results/full_model")
    return {
        "best_r10": history.best_val_metric,
        "best_epoch": history.best_epoch,
        "save_dir": str(save_dir),
    }


def run_ablation(config: STPCNConfig, device: str = "auto") -> dict:
    """
    Run ablation experiments (paper Tables 8,9).

    Configurations:
    1. Baseline (DCMGNN)
    2. +M1 (STP only)
    3. +M2 (Adaptive Contrast only)
    4. +M3 (Entropy Fusion only)
    5. +M4 (Learnable Depth only)
    6. +M5 (Spatial Bias only)
    7. Full model (M1-M7)
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(config.train.seed)
    data_dict = generate_synthetic_maritime_data(
        num_vessels=config.data.num_vessels,
        num_zones=config.data.num_zones,
        seed=config.train.seed,
    )
    dataset = MaritimeDataset(data_dict, device=device)

    ablation_results = {}

    # ── Configuration: +M1 (STP only) - best single modification ──
    print("\n" + "="*60)
    print("Ablation: STP-only (M1)")
    print("="*60)
    config_m1 = STPCNConfig.from_dict(config.to_dict())
    config_m1.experiment_name = "ablation_m1_stp_only"
    trainer = STPCNTrainer(config_m1, device=device)
    history = trainer.train()
    final_metrics = trainer.evaluator.evaluate(trainer.model, trainer.dataset, device)
    ablation_results["STP_only"] = final_metrics
    trainer.save_results(f"results/ablation_m1")

    # ── Configuration: Full model (M1-M7) ──
    print("\n" + "="*60)
    print("Ablation: Full model (M1-M7)")
    print("="*60)
    config_full = STPCNConfig.from_dict(config.to_dict())
    config_full.experiment_name = "ablation_full_model"
    trainer = STPCNTrainer(config_full, device=device)
    history = trainer.train()
    final_metrics = trainer.evaluator.evaluate(trainer.model, trainer.dataset, device)
    ablation_results["Full_model"] = final_metrics
    trainer.save_results(f"results/ablation_full")

    # Save all ablation results
    results_dir = Path("results/ablation")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "ablation_results.json", "w") as f:
        json.dump(ablation_results, f, indent=2)

    # Plot comparison
    plot_comparison_table(ablation_results, str(results_dir))

    return ablation_results


def run_sparsity_analysis(config: STPCNConfig, device: str = "auto") -> dict:
    """
    Run sparsity analysis (paper Table 11).

    Drop training edges at rates {0%, 30%, 50%, 70%, 90%}
    and evaluate DCMGNN vs STP variant.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.9]
    sparsity_results = {}

    for drop_rate in sparsity_levels:
        print(f"\n{'='*60}")
        print(f"Sparsity Analysis: drop_rate={drop_rate:.0%}")
        print(f"{'='*60}")

        # Generate data with reduced interactions
        seed_offset = int(drop_rate * 1000)
        data_dict = generate_synthetic_maritime_data(
            num_vessels=config.data.num_vessels,
            num_zones=config.data.num_zones,
            seed=config.train.seed + seed_offset,
        )

        # Drop training edges
        if drop_rate > 0:
            for beh in data_dict["R_train"]:
                R = data_dict["R_train"][beh]
                mask = np.random.random(R.shape) < drop_rate
                R[mask] = 0
                data_dict["R_train"][beh] = R

        dataset = MaritimeDataset(data_dict, device=device)

        # Train STP variant
        config_sp = STPCNConfig.from_dict(config.to_dict())
        config_sp.experiment_name = f"sparsity_{int(drop_rate*100)}"
        trainer = STPCNTrainer(config_sp, device=device)
        history = trainer.train()
        final_metrics = trainer.evaluator.evaluate(trainer.model, trainer.dataset, device)

        sparsity_results[f"{int(drop_rate*100)}%"] = {
            "STP_R@10": final_metrics.get("Recall@10", 0),
            "STP_N@10": final_metrics.get("NDCG@10", 0),
        }

    # Save results
    results_dir = Path("results/sparsity")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "sparsity_results.json", "w") as f:
        json.dump(sparsity_results, f, indent=2)

    from maritime_stpcn.visualization import plot_sparsity_analysis
    plot_sparsity_analysis(sparsity_results, str(results_dir))

    return sparsity_results


def main():
    parser = argparse.ArgumentParser(description="Maritime-STPCN Experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation experiments")
    parser.add_argument("--sparsity", action="store_true",
                        help="Run sparsity analysis")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed (paper: 123)")
    args = parser.parse_args()

    # Load config
    config = STPCNConfig.from_yaml(args.config)
    config.train.seed = args.seed

    set_seed(args.seed)

    if args.ablation:
        results = run_ablation(config, args.device)
        print("\nAblation Results:")
        for name, metrics in results.items():
            r10 = metrics.get("Recall@10", 0)
            print(f"  {name}: R@10={r10:.4f}")

    elif args.sparsity:
        results = run_sparsity_analysis(config, args.device)
        print("\nSparsity Results:")
        for level, metrics in results.items():
            r10 = metrics.get("STP_R@10", 0)
            print(f"  {level}: R@10={r10:.4f}")

    else:
        # Default: train full model
        results = train_full_model(config, args.device)
        print(f"\nFinal Results:")
        print(f"  Best R@10: {results['best_r10']:.4f} at epoch {results['best_epoch']}")
        print(f"  Results saved to: {results['save_dir']}")


if __name__ == "__main__":
    main()
