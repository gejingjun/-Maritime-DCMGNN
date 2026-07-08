"""
Utility functions for Maritime-STPCN
"""

from __future__ import annotations

import random
import numpy as np
import torch


def set_seed(seed: int = 123):
    """Set random seed for reproducibility (paper Section 4.11)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics(metrics: dict) -> str:
    """Format evaluation metrics for display."""
    lines = []
    for k in [5, 10, 20, 40]:
        r = metrics.get(f"Recall@{k}", 0)
        n = metrics.get(f"NDCG@{k}", 0)
        f1 = metrics.get(f"F1@{k}", 0)
        lines.append(f"R@{k}={r:.4f}  N@{k}={n:.4f}  F1@{k}={f1:.4f}")
        if f"Recall@{k}_CI_low" in metrics:
            low = metrics[f"Recall@{k}_CI_low"]
            high = metrics[f"Recall@{k}_CI_high"]
            lines.append(f"  CI: [{low:.4f}, {high:.4f}]")
    return "\n".join(lines)
