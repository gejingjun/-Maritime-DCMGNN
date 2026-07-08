"""
Maritime-STPCN Package
"""

from .config import STPCNConfig, ModelConfig, DataConfig, TrainConfig, EvalConfig
from .data import MaritimeDataset, BPRBatchSampler, generate_synthetic_maritime_data
from .model import (
    MaritimeSTPCN,
    STPEncoder,
    MultiRelationEncoder,
    CascadeEncoder,
    ChainEncoder,
    ContrastiveHead,
    FusionGate,
    GumbelDepthSelector,
    SpatialBiasCorrection,
)
from .losses import STPCNLoss
from .trainer import STPCNTrainer, WarmupCosineScheduler, TrainingHistory
from .evaluator import STPCNEvaluator, recall_at_k, ndcg_at_k, f1_at_k, bootstrap_ci
from .visualization import (
    plot_training_curves,
    plot_comparison_table,
    plot_sparsity_analysis,
    plot_ablation_results,
)
