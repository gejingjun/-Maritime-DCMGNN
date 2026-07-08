"""
Maritime-STPCN Configuration Module (v2)

Aligned with paper Table 4 hyperparameters and Section 4.10 optimization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


@dataclass
class ModelConfig:
    """Model architecture configuration (paper Table 4)."""
    embedding_dim: int = 64           # d
    contrastive_dim: int = 32         # d/2, contrastive projection output
    max_propagation_depth: int = 4    # L_max
    num_stp_patterns: int = 2         # dual graph (geo + beh)
    num_layers: int = 3
    dropout: float = 0.1


@dataclass
class DataConfig:
    """Data pipeline configuration (paper Table 5)."""
    num_vessels: int = 200
    num_zones: int = 100
    behaviors: tuple = (
        "transit", "loitering", "gear_deployment",
        "rendezvous", "at_sea_transshipment", "illegal_fishing",
    )
    target_behavior: str = "illegal_fishing"
    relation_order: tuple = (
        "transit", "loitering", "gear_deployment",
        "rendezvous", "at_sea_transshipment", "illegal_fishing",
    )
    behavior_layers: tuple = (2, 2, 2, 1, 1, 1)
    split_type: str = "temporal"
    train_ratio: float = 0.75
    val_ratio: float = 0.125
    test_ratio: float = 0.125
    geo_threshold_km: float = 50.0    # theta_geo (Eq.6)
    beh_similarity_threshold: float = 0.7  # theta_beh (Eq.9)


@dataclass
class TrainConfig:
    """Training configuration (paper Table 4 + Section 4.10)."""
    batch_size: int = 128
    learning_rate: float = 5.0e-3     # lr_0
    weight_decay: float = 1.0e-4      # lambda_l2
    max_epochs: int = 200
    warmup_epochs: int = 5
    warmup_start_lr: float = 1.0e-4   # warmup start
    min_lr: float = 1.0e-5            # lr_min
    early_stopping_patience: int = 20
    grad_clip_norm: float = 1.0
    # Loss weights (paper Eq.34)
    lambda_rcl: float = 3.0
    lambda_chain: float = 0.5
    lambda_entropy: float = 0.01
    lambda_l2: float = 1.0e-4
    seed: int = 123                    # paper Section 4.11
    negative_sample_range_km: float = 200.0
    # Gumbel-Softmax annealing
    gumbel_tau_init: float = 5.0       # gs initial
    gumbel_tau_final: float = 0.1      # gs final


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    k_values: tuple = (5, 10, 20, 40)
    bootstrap_samples: int = 100       # B=100 (M7)
    confidence_level: float = 0.95     # 95% CI


@dataclass
class STPCNConfig:
    """Top-level configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str = "maritime_stpcn_full"
    output_dir: str = "results"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "STPCNConfig":
        if not _HAS_YAML:
            raise ImportError("pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "STPCNConfig":
        config = cls()
        for section_key in ("model", "data", "train", "eval"):
            if section_key in data:
                sub_cfg = getattr(config, section_key)
                for k, v in data[section_key].items():
                    if k in ("behaviors", "relation_order", "behavior_layers", "k_values"):
                        setattr(sub_cfg, k, tuple(v))
                    elif hasattr(sub_cfg, k):
                        setattr(sub_cfg, k, v)
        for k in ("experiment_name", "output_dir"):
            if k in data:
                setattr(config, k, data[k])
        return config

    def to_dict(self) -> dict:
        return asdict(self)
