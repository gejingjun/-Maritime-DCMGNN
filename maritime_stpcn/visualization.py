"""
Maritime-STPCN Visualization Module (v2)

Provides visualization for:
- Training/validation loss curves
- Validation NDCG@10 over epochs
- Prediction result comparison
- Channel weight distribution (M3 fusion)
- Sparsity analysis results
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def plot_training_curves(
    history: dict,
    save_dir: str | Path = "results",
) -> list[str]:
    """
    Plot training loss and validation metrics curves.

    Generates:
    - training_loss.png: Training loss over epochs
    - val_ndcg10.png: Validation NDCG@10 over epochs
    - combined_curves.png: Combined view
    """
    if not _HAS_MPL:
        print("matplotlib not installed, skipping visualization")
        return []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    epochs = history.get("epochs", [])
    train_losses = history.get("train_losses", [])
    val_metrics_list = history.get("val_metrics", [])
    learning_rates = history.get("learning_rates", [])

    # ── Training Loss Curve ──
    if train_losses:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_yscale('log')
        ax.set_title('Maritime-STPCN Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = save_dir / "training_loss.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_files.append(str(path))

    # ── Validation NDCG@10 ──
    val_ndcg = [m.get("NDCG@10", 0) for m in val_metrics_list if m]
    val_recall = [m.get("Recall@10", 0) for m in val_metrics_list if m]
    eval_epochs = [e for e, m in zip(epochs, val_metrics_list) if m]

    if val_ndcg:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax1.plot(eval_epochs, val_ndcg, 'r-', linewidth=2, label='NDCG@10')
        ax1.plot(eval_epochs, val_recall, 'g-', linewidth=2, label='Recall@10')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Metric Value')
        ax1.set_title('Validation Metrics Over Epochs')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        path = save_dir / "val_metrics.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_files.append(str(path))

    # ── Combined curves (Figure 2 style) ──
    if train_losses and val_ndcg:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss (log scale)')
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(eval_epochs, val_ndcg, 'r-', linewidth=2, label='Val NDCG@10')
        ax2.set_ylabel('Validation NDCG@10')
        ax2.tick_params(axis='y', labelcolor='r')

        ax1.set_title('Training Dynamics (Figure 2 style)')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        path = save_dir / "training_dynamics.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_files.append(str(path))

    # ── Learning Rate Schedule ──
    if learning_rates:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(epochs, learning_rates, 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Warmup-Cosine LR Schedule (Eq.36)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = save_dir / "lr_schedule.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_files.append(str(path))

    return saved_files


def plot_comparison_table(
    results: dict[str, dict[str, float]],
    save_dir: str | Path = "results",
) -> str:
    """
    Plot comparison table of different model configurations.

    Paper Table 7 style comparison.
    """
    if not _HAS_MPL:
        return ""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    methods = list(results.keys())
    metrics = ["Recall@10", "NDCG@10", "F1@10"]

    data = np.array([[results[m].get(metric, 0) for metric in metrics] for m in methods])

    fig, ax = plt.subplots(figsize=(10, max(3, len(methods) * 0.5)))
    ax.axis('off')

    # Build table
    col_labels = ["Method"] + metrics
    cell_text = [[m] + [f"{results[m].get(metric, 0):.4f}" for metric in metrics] for m in methods]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(col_labels))))

    # Highlight best values
    for metric_idx in range(len(metrics)):
        values = [results[m].get(metrics[metric_idx], 0) for m in methods]
        best_idx = np.argmax(values)
        table[best_idx + 1, metric_idx + 1].set_facecolor('#90EE90')

    ax.set_title('Performance Comparison (Table 7 style)', fontsize=14, pad=20)
    fig.tight_layout()
    path = save_dir / "comparison_table.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return str(path)


def plot_sparsity_analysis(
    sparsity_results: dict,
    save_dir: str | Path = "results",
) -> str:
    """
    Plot sparsity analysis (paper Table 11).
    """
    if not _HAS_MPL:
        return ""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    sparsity_levels = list(sparsity_results.keys())
    dcmgnn_r10 = [sparsity_results[s].get("DCMGNN_R@10", 0) for s in sparsity_levels]
    stp_r10 = [sparsity_results[s].get("STP_R@10", 0) for s in sparsity_levels]
    relative_gain = [sparsity_results[s].get("relative_gain", 0) for s in sparsity_levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(sparsity_levels, dcmgnn_r10, 'b-o', linewidth=2, label='DCMGNN')
    ax1.plot(sparsity_levels, stp_r10, 'r-o', linewidth=2, label='STP Variant')
    ax1.set_xlabel('Data Sparsity (%)')
    ax1.set_ylabel('Recall@10')
    ax1.set_title('Recall@10 under Increasing Sparsity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(sparsity_levels, relative_gain, color='green', alpha=0.7)
    ax2.set_xlabel('Data Sparsity (%)')
    ax2.set_ylabel('Relative Gain (%)')
    ax2.set_title('STP Relative Gain over DCMGNN')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = save_dir / "sparsity_analysis.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_ablation_results(
    ablation_results: dict,
    save_dir: str | Path = "results",
) -> str:
    """Plot ablation study results (paper Tables 8,9)."""
    if not _HAS_MPL:
        return ""

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    configs = list(ablation_results.keys())
    r10_values = [ablation_results[c].get("R@10", 0) for c in configs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(configs)), r10_values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Recall@10')
    ax.set_title('Ablation Study Results')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, r10_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    path = save_dir / "ablation_results.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
