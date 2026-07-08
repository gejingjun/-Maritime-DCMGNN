"""
Maritime-STPCN Evaluator (v2)

Evaluation metrics (paper Section 5.2):
- Recall@K, NDCG@K, F1@K for K in {5, 10, 20, 40}
- M7: Bootstrap 95% confidence intervals (B=100 resamples)
"""

from __future__ import annotations

import numpy as np
import torch

from .config import STPCNConfig


def recall_at_k(predicted: list[int], actual: set[int], k: int) -> float:
    """Recall@K = |R(k) intersect T(v)| / |T(v)|"""
    if len(actual) == 0:
        return 0.0
    return len(set(predicted[:k]) & actual) / len(actual)


def ndcg_at_k(predicted: list[int], actual: set[int], k: int) -> float:
    """NDCG@K = DCG@K / IDCG@K"""
    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / max(idcg, 1e-12)


def f1_at_k(predicted: list[int], actual: set[int], k: int) -> float:
    """F1@K = harmonic mean of precision and recall at K"""
    if len(actual) == 0 or k == 0:
        return 0.0
    precision = len(set(predicted[:k]) & actual) / k
    recall = len(set(predicted[:k]) & actual) / len(actual)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class STPCNEvaluator:
    """Evaluator with bootstrap confidence intervals (M7)."""

    def __init__(self, config: STPCNConfig | None = None):
        self.k_values = (5, 10, 20, 40)
        self.bootstrap_samples = 100
        self.confidence_level = 0.95

        if config is not None:
            self.k_values = config.eval.k_values
            self.bootstrap_samples = config.eval.bootstrap_samples
            self.confidence_level = config.eval.confidence_level

    def evaluate(
        self, model, dataset, device: str = "cpu"
    ) -> dict[str, float]:
        """
        Evaluate model on test set.

        For each vessel, rank all zones by predicted score and compute
        Recall@K, NDCG@K, F1@K.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(dataset)

        # Ground truth: test set target behavior interactions
        R_test_target = dataset.R_test[dataset.target_behavior]

        all_metrics = {"Recall@5": [], "Recall@10": [], "Recall@20": [], "Recall@40": [],
                       "NDCG@5": [], "NDCG@10": [], "NDCG@20": [], "NDCG@40": [],
                       "F1@5": [], "F1@10": [], "F1@20": [], "F1@40": []}

        fused_v = outputs["fused_v"]
        fused_z = outputs["fused_z"]

        for v in range(dataset.num_vessels):
            # Actual zones for this vessel in test set
            actual_zones = set()
            for z in range(dataset.num_zones):
                if R_test_target[v, z] > 0:
                    actual_zones.add(z)

            if len(actual_zones) == 0:
                continue

            # Predict scores for all zones
            vessel_ids = torch.tensor([v], dtype=torch.long, device=device)
            zone_ids = torch.arange(dataset.num_zones, dtype=torch.long, device=device)
            scores = model.predict(outputs, vessel_ids.expand(dataset.num_zones), zone_ids)

            # Rank zones by predicted score (descending)
            ranked_zones = torch.argsort(scores, descending=True).cpu().tolist()

            for k in self.k_values:
                r = recall_at_k(ranked_zones, actual_zones, k)
                n = ndcg_at_k(ranked_zones, actual_zones, k)
                f = f1_at_k(ranked_zones, actual_zones, k)

                all_metrics[f"Recall@{k}"].append(r)
                all_metrics[f"NDCG@{k}"].append(n)
                all_metrics[f"F1@{k}"].append(f)

        # Average metrics
        results = {}
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                results[metric_name] = np.mean(values)
            else:
                results[metric_name] = 0.0

        # M7: Bootstrap confidence intervals
        ci_results = {}
        for metric_name, values in all_metrics.items():
            if len(values) > 0:
                ci = bootstrap_ci(values, self.bootstrap_samples, self.confidence_level)
                ci_results[f"{metric_name}_CI_low"] = ci[0]
                ci_results[f"{metric_name}_CI_high"] = ci[1]

        results.update(ci_results)
        return results


def bootstrap_ci(
    values: list[float],
    num_samples: int = 100,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    M7: Bootstrap 95% confidence intervals (Eq.33).

    CI_95% = [percentile_2.5, percentile_97.5] from B=100 resamples.
    """
    if len(values) == 0:
        return (0.0, 0.0)

    bootstrap_means = []
    for _ in range(num_samples):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    low = np.percentile(bootstrap_means, 100 * alpha / 2)
    high = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    return (float(low), float(high))
