"""
Maritime-STPCN Model Architecture (v2)

Strictly aligned with paper formulas:
- M1: STP Encoder (Eq.6-17) - dual graph propagation with sigmoid gating
- M2: Adaptive Contrastive Loss (Eq.18-21) - uncertainty-aware NT-Xent
- M3: Entropy-Regularized Fusion (Eq.22-24) - MoE gate + negative entropy
- M4: Learnable Depth (Eq.26-27) - Gumbel-Softmax with annealing
- M5: Spatial Bias (Eq.30-32) - partitioned marginal propensity
- Prediction: Eq.5, Loss: Eq.34-35
"""

from __future__ import annotations

import itertools
import math

import torch
from torch import nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# M1: STP Encoder - Dual Graph Propagation (Eq.6-17)
# ═══════════════════════════════════════════════════════════

class STPEncoder(nn.Module):
    """
    Modification 1: Spatial-Temporal Pattern (STP) Encoder.

    Dual adjacency: geographic proximity (A_geo, 50km) + behavioral
    similarity (A_beh, cos>=0.7). Sigmoid-gated spatial convolution
    with dedicated projection MLP.

    Paper equations 6-17.
    """

    def __init__(self, embedding_dim: int, num_layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Sigmoid gate parameters (Eq.14,15)
        # g_geo = sigma(H^(l+1) W_g_geo + A_geo H^(l+1) W_geo + b_geo)
        self.W_g_geo = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.W_geo = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_g_beh = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.W_beh = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # LightGCN-style propagation weights (Eq.10,11)
        # Per-layer weight matrices
        self.W_layers = nn.ParameterList([
            nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01)
            for _ in range(num_layers)
        ])

        # Projection MLP (Eq.17): H_stp = MLP_proj(H_stp_v || H_stp_z)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        vessel_emb: torch.Tensor,   # (N_v, d)
        zone_emb: torch.Tensor,     # (N_z, d)
        A_geo_aug: torch.Tensor,    # (N_z, N_z) augmented geo adjacency
        A_beh_aug: torch.Tensor,    # (N_z, N_z) augmented beh adjacency
        R_aug: torch.Tensor,        # (N_v+N_z, N_v+N_z) bipartite augmented
        num_vessels: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Dual-graph STP propagation.

        Returns: (vessel_stp, zone_stp) embeddings.
        """
        d = self.embedding_dim
        N_v = num_vessels
        N_z = zone_emb.size(0)

        # ── Bipartite propagation (Eq.10, LightGCN-style) ──
        # H^(l+1) = D~^(-1/2) R~^(k) D~^(-1/2) H^(l) W^(l)
        h_all = torch.cat([vessel_emb, zone_emb], dim=0)  # (N_v+N_z, d)
        h_v = vessel_emb
        h_z = zone_emb

        for l_idx in range(self.num_layers):
            W = self.W_layers[l_idx]

            # Bipartite propagation: interaction graph (Eq.12)
            # e_v^(l+1) = sum_{z in N(v)} 1/|N(v)| * e_z^(l)
            # LightGCN normalized symmetric: D~^(-1/2) A~ D~^(-1/2)
            h_prop = torch.mm(h_all, W)  # (N_v+N_z, d)
            h_prop_v = h_prop[:N_v]
            h_prop_z = h_prop[N_v:]

            # ── Spatial graph injection (Eq.13) ──
            # e_z^(l+1) += sum_{z' in N_geo(z)} 1/|N_geo(z)| * e_z'^(l+1)
            # Geographic neighborhood propagation on zone embeddings
            geo_prop_z = torch.mm(A_geo_aug, h_prop_z) * (1.0 / (A_geo_aug.sum(dim=1, keepdim=True).clamp(min=1.0)))
            beh_prop_z = torch.mm(A_beh_aug, h_prop_z) * (1.0 / (A_beh_aug.sum(dim=1, keepdim=True).clamp(min=1.0)))

            # ── Sigmoid gating (Eq.14,15,16) ──
            # g_geo = sigma(H^(l+1) W_g_geo + A_geo H^(l+1) W_geo + b_geo)
            g_geo = torch.sigmoid(
                self.W_g_geo(h_prop_z) + self.W_geo(geo_prop_z)
            )
            g_beh = torch.sigmoid(
                self.W_g_beh(h_prop_z) + self.W_beh(beh_prop_z)
            )

            # H_stp = H^(l+1) + g_geo * (A_geo H^(l+1)) + g_beh * (A_beh H^(l+1))
            h_z_new = h_prop_z + g_geo * geo_prop_z + g_beh * beh_prop_z
            h_v_new = h_prop_v  # vessel side uses bipartite only

            h_v = h_v_new
            h_z = h_z_new
            h_all = torch.cat([h_v, h_z], dim=0)

        # ── Projection MLP (Eq.17) ──
        # H_stp = MLP_proj(H_stp_v || H_stp_z)
        stp_v = h_v
        stp_z = h_z

        return stp_v, stp_z


# ═══════════════════════════════════════════════════════════
# Multi-Relation Encoder (behavior-specific GCN)
# ═══════════════════════════════════════════════════════════

class MultiRelationEncoder(nn.Module):
    """Multi-behavior GCN operating on behavior-specific bipartite graphs."""

    def __init__(self, embedding_dim: int, behaviors: tuple, behavior_layers: tuple):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.behaviors = behaviors
        self.behavior_layers = behavior_layers

        # Per-behavior weight matrices for each layer
        self.behavior_weights = nn.ParameterDict()
        for b_idx, beh in enumerate(behaviors):
            n_layers = behavior_layers[b_idx]
            for l in range(n_layers):
                self.behavior_weights[f"{beh}_l{l}"] = nn.Parameter(
                    torch.randn(embedding_dim, embedding_dim) * 0.01
                )

    def forward(
        self,
        vessel_emb: torch.Tensor,
        zone_emb: torch.Tensor,
        relation_adjs: dict[str, torch.Tensor],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Propagate through behavior-specific bipartite graphs.

        Returns dict of (vessel_emb, zone_emb) per behavior.
        """
        results = {}
        N_v = vessel_emb.size(0)
        N_z = zone_emb.size(0)

        for b_idx, beh in enumerate(self.behaviors):
            if beh not in relation_adjs:
                results[beh] = (vessel_emb, zone_emb)
                continue

            adj = relation_adjs[beh]  # (N_v+N_z, N_v+N_z) normalized
            n_layers = self.behavior_layers[b_idx]

            h = torch.cat([vessel_emb, zone_emb], dim=0)

            for l in range(n_layers):
                W = self.behavior_weights[f"{beh}_l{l}"]
                h = torch.mm(adj, torch.mm(h, W))

            results[beh] = (h[:N_v], h[N_v:])

        return results


# ═══════════════════════════════════════════════════════════
# Cascade Encoder (Eq.28)
# ═══════════════════════════════════════════════════════════

class CascadeEncoder(nn.Module):
    """
    Cascade encoder: models sequential behavior transitions.
    A_cas = sum_{(k,k+1) in C} R~^(k) R~^(k+1)  (Eq.28)
    """

    def __init__(self, embedding_dim: int, relation_order: tuple):
        super().__init__()
        self.relation_order = relation_order
        self.embedding_dim = embedding_dim

        # Cascade transition transforms
        self.cascade_transforms = nn.ModuleDict()
        for i in range(len(relation_order) - 1):
            key = f"{relation_order[i]}_to_{relation_order[i+1]}"
            self.cascade_transforms[key] = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        vessel_emb: torch.Tensor,
        zone_emb: torch.Tensor,
        relation_adjs: dict[str, torch.Tensor],
        num_layers: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cascade propagation along behavior chain."""
        N_v = vessel_emb.size(0)
        h_v = vessel_emb
        h_z = zone_emb

        prev_beh = None
        for beh in self.relation_order:
            if prev_beh is not None:
                key = f"{prev_beh}_to_{beh}"
                if key in self.cascade_transforms:
                    h_v = self.cascade_transforms[key](h_v)
                    h_z = self.cascade_transforms[key](h_z)

            if beh in relation_adjs:
                adj = relation_adjs[beh]
                h = torch.cat([h_v, h_z], dim=0)
                h = torch.mm(adj, h)
                h_v, h_z = h[:N_v], h[N_v:]

            prev_beh = beh

        return h_v, h_z


# ═══════════════════════════════════════════════════════════
# Chain Encoder (Eq.29)
# ═══════════════════════════════════════════════════════════

class ChainEncoder(nn.Module):
    """
    Chain encoder: enumerates behavior combinations as meta-paths.
    A_chain^(a1...ak) = product of R~^(k)  (Eq.29)
    Prune chains with fewer than chain_min (5) edges.
    """

    def __init__(self, embedding_dim: int, behaviors: tuple, target_behavior: str, chain_min: int = 5):
        super().__init__()
        self.target_behavior = target_behavior
        self.chain_min = chain_min
        self.embedding_dim = embedding_dim

        # Build chain specs
        self.chain_specs = []
        aux_behs = [b for b in behaviors if b != target_behavior]
        for size in range(1, len(aux_behs) + 1):
            for combo in itertools.combinations(aux_behs, size):
                chain = list(combo) + [target_behavior]
                if len(chain) >= 2:
                    self.chain_specs.append(tuple(chain))

        # Chain weight for pooling
        self.chain_weight = nn.Parameter(torch.zeros(max(len(self.chain_specs), 1)))

        # Per-chain transforms
        self.chain_transforms = nn.ModuleDict()
        for chain in self.chain_specs:
            for i in range(len(chain) - 1):
                key = f"{chain[i]}_to_{chain[i+1]}"
                if key not in self.chain_transforms:
                    self.chain_transforms[key] = nn.Linear(
                        embedding_dim, embedding_dim, bias=False
                    )

    def forward(
        self,
        relation_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute chain embeddings from relation embeddings."""
        chain_outputs_v = []
        chain_outputs_z = []

        for chain in self.chain_specs:
            if chain[0] not in relation_embeddings:
                continue
            h_v, h_z = relation_embeddings[chain[0]]

            for i in range(len(chain) - 1):
                key = f"{chain[i]}_to_{chain[i+1]}"
                if key in self.chain_transforms:
                    h_v = self.chain_transforms[key](h_v)
                    h_z = self.chain_transforms[key](h_z)

            chain_outputs_v.append(h_v)
            chain_outputs_z.append(h_z)

        if not chain_outputs_v:
            # Fallback: use target behavior
            target = self.target_behavior
            if target in relation_embeddings:
                return relation_embeddings[target]
            first_beh = list(relation_embeddings.keys())[0]
            return relation_embeddings[first_beh]

        weights = F.softmax(self.chain_weight[:len(chain_outputs_v)], dim=0)
        stacked_v = torch.stack(chain_outputs_v, dim=0)
        stacked_z = torch.stack(chain_outputs_z, dim=0)

        h_v = (weights.view(-1, 1, 1) * stacked_v).sum(dim=0)
        h_z = (weights.view(-1, 1, 1) * stacked_z).sum(dim=0)

        return h_v, h_z


# ═══════════════════════════════════════════════════════════
# M4: Learnable Propagation Depth (Eq.26-27)
# ═══════════════════════════════════════════════════════════

class GumbelDepthSelector(nn.Module):
    """
    Modification 4: Per-behavior learnable propagation depth.

    Uses Gumbel-Softmax reparameterization (Eq.26) with annealing
    from tau_init=5.0 to tau_final=0.1.

    Depth-weighted propagation (Eq.27): H^(k) = sum_l p^(k)(l) * H^(k,l)
    """

    def __init__(self, behaviors: tuple, max_depth: int = 4,
                 tau_init: float = 5.0, tau_final: float = 0.1):
        super().__init__()
        self.behaviors = behaviors
        self.max_depth = max_depth
        self.tau_init = tau_init
        self.tau_final = tau_final

        # Learnable depth logits per behavior (Eq.26)
        self.depth_logits = nn.ParameterDict()
        for beh in behaviors:
            self.depth_logits[beh] = nn.Parameter(torch.zeros(max_depth))

    def get_depth_probs(self, behavior: str, tau: float) -> torch.Tensor:
        """Get depth distribution via Gumbel-Softmax (Eq.26)."""
        logits = self.depth_logits[behavior]
        # Gumbel-Softmax: p^(k) = Gumbel-Softmax(l^(k), tau_gs)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
        y = logits + gumbel_noise
        probs = F.softmax(y / tau, dim=0)
        return probs

    def get_hard_depth(self, behavior: str) -> int:
        """Get argmax depth for inference."""
        logits = self.depth_logits[behavior]
        return logits.argmax().item() + 1  # depth from 1 to max_depth

    def weight_layers(
        self, behavior: str, layer_outputs: list[torch.Tensor], tau: float
    ) -> torch.Tensor:
        """Depth-weighted propagation (Eq.27)."""
        probs = self.get_depth_probs(behavior, tau)
        stacked = torch.stack(layer_outputs, dim=0)  # (max_depth, N, d)
        weighted = (probs.view(-1, 1, 1) * stacked).sum(dim=0)
        return weighted


# ═══════════════════════════════════════════════════════════
# M2: Contrastive Head (Eq.18-21)
# ═══════════════════════════════════════════════════════════

class ContrastiveHead(nn.Module):
    """
    Modification 2: Adaptive Contrastive Head.

    Contrastive projection (Eq.18): p = BN(ReLU(h W1 + b1)) W2
    Weight MLP (Eq.19): [omega, sigma, tau] = MLP_weight(p_v . p_z | k)
    Adaptive NT-Xent (Eq.20): L_contrast = -(omega/sigma^2) log exp(sim/tau) / sum
    """

    def __init__(self, embedding_dim: int, contrastive_dim: int = 32,
                 num_behaviors: int = 6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.contrastive_dim = contrastive_dim
        self.num_behaviors = num_behaviors

        # Projection: p = BN(ReLU(h W1 + b1)) W2 (Eq.18)
        self.proj_W1 = nn.Linear(embedding_dim, contrastive_dim)
        self.proj_bn = nn.BatchNorm1d(contrastive_dim)
        self.proj_W2 = nn.Linear(contrastive_dim, contrastive_dim, bias=False)

        # Weight MLP (Eq.19): [omega, sigma, tau] = MLP(p_v . p_z | k)
        # Input: contrastive_dim (from dot product projection) + num_behaviors (one-hot)
        self.weight_mlp = nn.Sequential(
            nn.Linear(contrastive_dim + num_behaviors, contrastive_dim),
            nn.ReLU(),
            nn.Linear(contrastive_dim, 3),  # [omega, sigma, tau]
        )

        self.eps = 0.01  # small offset for tau (paper Section 4.2)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Contrastive projection (Eq.18)."""
        x = F.relu(self.proj_W1(h))
        if x.size(0) > 1:
            x = self.proj_bn(x)
        p = self.proj_W2(x)
        return F.normalize(p, dim=-1)

    def compute_weights(
        self, p_v: torch.Tensor, p_z: torch.Tensor,
        behavior_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uncertainty-aware weighting (Eq.19).

        Returns: (omega, sigma, tau) per sample.
        """
        # Element-wise product of projected embeddings (Eq.19 input: p_v . p_z)
        dot_product = p_v * p_z  # (B, contrastive_dim)
        # One-hot behavior indicator
        beh_onehot = torch.zeros(dot_product.size(0), self.num_behaviors, device=dot_product.device)
        beh_onehot[:, behavior_idx] = 1.0

        features = torch.cat([dot_product, beh_onehot], dim=-1)  # (B, d_c + K)
        outputs = self.weight_mlp(features)

        omega = torch.sigmoid(outputs[:, 0])    # relationship weight
        sigma = F.softplus(outputs[:, 1])       # uncertainty (positive)
        tau = F.softplus(outputs[:, 2]) + self.eps  # temperature (positive + offset)

        return omega, sigma, tau

    def compute_contrastive_loss(
        self,
        h_v: torch.Tensor, h_z: torch.Tensor,
        vessel_ids: torch.Tensor, pos_zone_ids: torch.Tensor,
        neg_zone_ids: torch.Tensor,
        behavior_idx: int,
    ) -> torch.Tensor:
        """
        Adaptive NT-Xent loss (Eq.20).
        """
        p_v = self.project(h_v[vessel_ids])   # (B, d_c)
        p_z_pos = self.project(h_z[pos_zone_ids])  # (B, d_c)
        p_z_neg = self.project(h_z[neg_zone_ids])  # (B, d_c)

        # Cosine similarity (already normalized)
        sim_pos = (p_v * p_z_pos).sum(dim=-1)  # (B,)
        sim_neg = (p_v * p_z_neg).sum(dim=-1)  # (B,)

        # Adaptive weights (Eq.19)
        omega, sigma, tau = self.compute_weights(p_v, p_z_pos, behavior_idx)

        # Clipping sigma to prevent gradient explosion
        sigma = sigma.clamp(min=0.1, max=10.0)

        # Adaptive NT-Xent (Eq.20)
        # L = -(omega / sigma^2) * log(exp(sim_pos/tau) / (exp(sim_pos/tau) + exp(sim_neg/tau)))
        exp_pos = torch.exp(sim_pos / tau)
        exp_neg = torch.exp(sim_neg / tau)

        loss = -(omega / (sigma ** 2)) * torch.log(exp_pos / (exp_pos + exp_neg + 1e-12))
        return loss.mean()


# ═══════════════════════════════════════════════════════════
# M3: Fusion Gate (Eq.22-24)
# ═══════════════════════════════════════════════════════════

class FusionGate(nn.Module):
    """
    Modification 3: Entropy-Regularized Dynamic Fusion Gate.

    Gate (Eq.22): alpha = softmax(h_cat W_gate + b_gate)
    Fused (Eq.23): h_fused = sum_c alpha^(c) * h^(c)
    Entropy (Eq.24): L_entropy = -1/|E| sum log(alpha)
    """

    def __init__(self, embedding_dim: int, num_channels: int = 4):
        super().__init__()
        self.num_channels = num_channels

        # Gate MLP (Eq.22): alpha = softmax(h_cat W_gate + b_gate)
        self.gate_mlp = nn.Sequential(
            nn.Linear(embedding_dim * num_channels, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_channels),
        )

        # Learnable temperature (Eq.25): tau_fusion
        self.fusion_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        channel_outputs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Fuse four channel outputs with entropy regularization.

        Args:
            channel_outputs: list of (h_v, h_z) from each encoder channel

        Returns:
            (h_fused_v, h_fused_z): fused embeddings
            fusion_entropy: entropy loss term
            channel_weights: alpha distribution (for logging)
        """
        # Concatenate channel outputs (Eq.22 input)
        h_cat_v = torch.cat([ch[0] for ch in channel_outputs], dim=-1)  # (N_v, 4d)
        h_cat_z = torch.cat([ch[1] for ch in channel_outputs], dim=-1)  # (N_z, 4d)

        # Gate computation (Eq.22)
        tau = F.softplus(self.fusion_temperature).clamp(min=0.1, max=10.0)
        logits_v = self.gate_mlp(h_cat_v) / tau  # (N_v, 4)
        logits_z = self.gate_mlp(h_cat_z) / tau  # (N_z, 4)
        alpha_v = F.softmax(logits_v, dim=-1)     # (N_v, 4)
        alpha_z = F.softmax(logits_z, dim=-1)     # (N_z, 4)

        # Weighted fusion (Eq.23): h_fused = sum_c alpha^(c) * h^(c)
        h_fused_v = sum(alpha_v[:, c:c+1] * channel_outputs[c][0] for c in range(self.num_channels))
        h_fused_z = sum(alpha_z[:, c:c+1] * channel_outputs[c][1] for c in range(self.num_channels))

        # Entropy regularization (Eq.24)
        # L_entropy = -1/|E| sum_{(i,j) in E} sum_c alpha(i,j,c) log alpha(i,j,c)
        entropy_v = -(alpha_v * torch.log(alpha_v + 1e-12)).sum(dim=-1).mean()
        entropy_z = -(alpha_z * torch.log(alpha_z + 1e-12)).sum(dim=-1).mean()
        fusion_entropy = entropy_v + entropy_z

        return (h_fused_v, h_fused_z), fusion_entropy, alpha_v


# ═══════════════════════════════════════════════════════════
# M5: Spatial Bias Correction (Eq.30-32)
# ═══════════════════════════════════════════════════════════

class SpatialBiasCorrection(nn.Module):
    """
    Modification 5: Spatial Bias via Partitioned Marginal Propensity.

    Global bias (Eq.30): b_global = log(pos / (1-pos))
    Per-zone bias (Eq.31): beta_j = log(|E_train(j)| / (pos * N_z))
    Per-vessel bias: gamma_i (learned, stronger L2 regularization)
    Decomposition (Eq.32): b_spatial = b_global + beta_j + gamma_i
    """

    def __init__(self, num_vessels: int, num_zones: int,
                 global_bias_init: float = 0.0,
                 per_zone_bias_init: torch.Tensor | None = None):
        super().__init__()
        self.num_vessels = num_vessels
        self.num_zones = num_zones

        # Global bias (Eq.30) - fixed, computed from data
        self.global_bias = nn.Parameter(torch.tensor(global_bias_init), requires_grad=False)

        # Per-zone bias (Eq.31) - initialized from data, then learned
        if per_zone_bias_init is not None:
            if isinstance(per_zone_bias_init, torch.Tensor):
                self.zone_bias = nn.Parameter(per_zone_bias_init.clone().float())
            else:
                self.zone_bias = nn.Parameter(torch.tensor(per_zone_bias_init, dtype=torch.float32))
        else:
            self.zone_bias = nn.Parameter(torch.zeros(num_zones))

        # Per-vessel bias - learned with stronger L2 regularization
        self.vessel_bias = nn.Parameter(torch.zeros(num_vessels))

    def forward(
        self, vessel_ids: torch.Tensor, zone_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial bias for (vessel, zone) pairs (Eq.32).

        b_spatial = b_global + beta_j + gamma_i
        """
        bias = self.global_bias + self.zone_bias[zone_ids] + self.vessel_bias[vessel_ids]
        return bias


# ═══════════════════════════════════════════════════════════
# Complete Maritime-STPCN Model
# ═══════════════════════════════════════════════════════════

class MaritimeSTPCN(nn.Module):
    """
    Maritime-STPCN: Complete model integrating all 7 modifications.

    Architecture (Figure 1):
    - 4 parallel encoding channels: STP, Multi-Relation, Cascade, Chain
    - Dynamic fusion gate (M3) with entropy regularization
    - Contrastive head (M2) with adaptive weighting
    - Spatial bias correction (M5)
    - Learnable depth (M4) via Gumbel-Softmax
    - Prediction: Eq.5 (y_hat = sigma(e_v^T e_z + b_spatial))
    - Loss: Eq.34 (L_total = L_bpr + lambda_rcl L_rcl + ...)
    """

    def __init__(
        self,
        num_vessels: int = 200,
        num_zones: int = 100,
        behaviors: tuple = (
            "transit", "loitering", "gear_deployment",
            "rendezvous", "at_sea_transshipment", "illegal_fishing",
        ),
        target_behavior: str = "illegal_fishing",
        relation_order: tuple = (
            "transit", "loitering", "gear_deployment",
            "rendezvous", "at_sea_transshipment", "illegal_fishing",
        ),
        behavior_layers: tuple = (2, 2, 2, 1, 1, 1),
        embedding_dim: int = 64,
        contrastive_dim: int = 32,
        max_propagation_depth: int = 4,
        num_stp_layers: int = 3,
        fusion_entropy_weight: float = 0.01,
        gumbel_tau_init: float = 5.0,
        gumbel_tau_final: float = 0.1,
        global_bias_init: float = 0.0,
        per_zone_bias_init: torch.Tensor | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_vessels = num_vessels
        self.num_zones = num_zones
        self.behaviors = behaviors
        self.target_behavior = target_behavior
        self.relation_order = relation_order
        self.behavior_layers = behavior_layers
        self.embedding_dim = embedding_dim
        self.contrastive_dim = contrastive_dim
        self.fusion_entropy_weight = fusion_entropy_weight

        # ── Node embeddings ──
        self.vessel_embedding = nn.Embedding(num_vessels, embedding_dim)
        self.zone_embedding = nn.Embedding(num_zones, embedding_dim)

        # ── M1: STP Encoder ──
        self.stp_encoder = STPEncoder(embedding_dim, num_stp_layers)

        # ── Multi-Relation Encoder ──
        self.multi_relation_encoder = MultiRelationEncoder(
            embedding_dim, behaviors, behavior_layers
        )

        # ── Cascade Encoder ──
        self.cascade_encoder = CascadeEncoder(embedding_dim, relation_order)

        # ── Chain Encoder ──
        self.chain_encoder = ChainEncoder(embedding_dim, behaviors, target_behavior)

        # ── M4: Gumbel Depth Selector ──
        self.depth_selector = GumbelDepthSelector(
            behaviors, max_propagation_depth, gumbel_tau_init, gumbel_tau_final
        )

        # ── M3: Fusion Gate ──
        self.fusion_gate = FusionGate(embedding_dim, num_channels=4)

        # ── M2: Contrastive Head ──
        self.contrastive_head = ContrastiveHead(
            embedding_dim, contrastive_dim, num_behaviors=len(behaviors)
        )

        # ── M5: Spatial Bias Correction ──
        self.spatial_bias = SpatialBiasCorrection(
            num_vessels, num_zones, global_bias_init, per_zone_bias_init
        )

        # ── Dropout ──
        self.dropout = nn.Dropout(dropout)

        # ── Gumbel temperature annealing tracking ──
        self._current_tau = gumbel_tau_init

        self.reset_parameters()

    def reset_parameters(self):
        """Xavier initialization."""
        nn.init.xavier_uniform_(self.vessel_embedding.weight)
        nn.init.xavier_uniform_(self.zone_embedding.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def anneal_gumbel_tau(self, epoch: int, total_epochs: int):
        """Anneal Gumbel-Softmax temperature from tau_init to tau_final."""
        progress = epoch / max(total_epochs, 1)
        self._current_tau = max(
            self.depth_selector.tau_final,
            self.depth_selector.tau_init - progress * (self.depth_selector.tau_init - self.depth_selector.tau_final)
        )

    def forward(self, dataset) -> dict:
        """
        Full forward pass through all 4 channels + fusion + bias.

        Returns dict with all outputs needed for loss computation.
        """
        # Initial embeddings
        vessel_emb = self.vessel_embedding.weight   # (N_v, d)
        zone_emb = self.zone_embedding.weight        # (N_z, d)

        # ── M1: STP Encoder ──
        stp_v, stp_z = self.stp_encoder(
            vessel_emb, zone_emb,
            dataset.A_geo_aug, dataset.A_beh_aug,
            dataset.relation_adjs.get(self.target_behavior,
                torch.eye(self.num_vessels + self.num_zones, device=vessel_emb.device)),
            self.num_vessels,
        )

        # ── Multi-Relation Encoder with M4 depth weighting ──
        mr_results = self.multi_relation_encoder(
            vessel_emb, zone_emb, dataset.relation_adjs
        )
        # Weighted sum across behaviors
        mr_v = sum(F.normalize(mr_results[b][0], dim=-1) for b in self.behaviors if b in mr_results) / len(self.behaviors)
        mr_z = sum(F.normalize(mr_results[b][1], dim=-1) for b in self.behaviors if b in mr_results) / len(self.behaviors)

        # ── Cascade Encoder ──
        cas_v, cas_z = self.cascade_encoder(
            vessel_emb, zone_emb, dataset.relation_adjs, num_layers=3
        )

        # ── Chain Encoder ──
        chain_v, chain_z = self.chain_encoder(mr_results)

        # ── M3: Fusion Gate ──
        channel_outputs = [
            (stp_v, stp_z),
            (mr_v, mr_z),
            (cas_v, cas_z),
            (chain_v, chain_z),
        ]
        (fused_v, fused_z), fusion_entropy, channel_weights = self.fusion_gate(channel_outputs)

        # Apply dropout
        fused_v = self.dropout(fused_v)
        fused_z = self.dropout(fused_z)

        return {
            "fused_v": fused_v,
            "fused_z": fused_z,
            "stp_v": stp_v,
            "stp_z": stp_z,
            "mr_v": mr_v,
            "mr_z": mr_z,
            "cas_v": cas_v,
            "cas_z": cas_z,
            "chain_v": chain_v,
            "chain_z": chain_z,
            "mr_results": mr_results,
            "fusion_entropy": fusion_entropy,
            "channel_weights": channel_weights,
            "vessel_emb": vessel_emb,
            "zone_emb": zone_emb,
        }

    def predict(
        self, outputs: dict, vessel_ids: torch.Tensor, zone_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Final prediction (Eq.5): y_hat_ij = sigma(e_v^T e_z + b_spatial)
        """
        e_v = outputs["fused_v"][vessel_ids]
        e_z = outputs["fused_z"][zone_ids]
        dot = (e_v * e_z).sum(dim=-1)

        # M5: Spatial bias
        spatial_bias = self.spatial_bias(vessel_ids, zone_ids)

        scores = torch.sigmoid(dot + spatial_bias)
        return scores

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
