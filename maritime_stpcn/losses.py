"""
Maritime-STPCN Loss Functions (v2)

Paper Eq.34: L_total = L_bpr + lambda_rcl L_rcl + lambda_chain L_chain
                        + lambda_entropy L_entropy + lambda_2 ||Theta||^2

- L_bpr (Eq.35): BPR loss for target behavior
- L_rcl (Eq.21): Adaptive relation contrastive loss (M2)
- L_chain: Chain contrastive loss (analogous to L_rcl)
- L_entropy (Eq.24): Fusion entropy regularization (M3)
- L_2: Weight decay
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class STPCNLoss:
    """
    Composite training loss for Maritime-STPCN.

    All loss weights follow paper Table 4:
    lambda_rcl=3.0, lambda_chain=0.5, lambda_entropy=0.01, lambda_l2=1e-4
    """

    def __init__(
        self,
        lambda_rcl: float = 3.0,
        lambda_chain: float = 0.5,
        lambda_entropy: float = 0.01,
        lambda_l2: float = 1e-4,
        num_behaviors: int = 6,
        contrastive_dim: int = 32,
    ):
        self.lambda_rcl = lambda_rcl
        self.lambda_chain = lambda_chain
        self.lambda_entropy = lambda_entropy
        self.lambda_l2 = lambda_l2
        self.num_behaviors = num_behaviors
        self.contrastive_dim = contrastive_dim

    def compute_bpr_loss(
        self,
        model_outputs: dict,
        vessel_ids: torch.Tensor,
        pos_zone_ids: torch.Tensor,
        neg_zone_ids: torch.Tensor,
        num_vessels: int,
    ) -> torch.Tensor:
        """
        BPR Loss (Eq.35): L_bpr = -1/|D| sum log sigma(y+ - y-)
        """
        fused_v = model_outputs["fused_v"]
        fused_z = model_outputs["fused_z"]

        e_v = fused_v[vessel_ids]
        e_z_pos = fused_z[pos_zone_ids]
        e_z_neg = fused_z[neg_zone_ids]

        # Prediction scores (Eq.5 without spatial bias for BPR)
        y_pos = (e_v * e_z_pos).sum(dim=-1)
        y_neg = (e_v * e_z_neg).sum(dim=-1)

        # BPR loss (Eq.35)
        bpr_loss = -F.logsigmoid(y_pos - y_neg).mean()
        return bpr_loss

    def compute_contrastive_loss(
        self,
        model: torch.nn.Module,
        model_outputs: dict,
        vessel_ids: torch.Tensor,
        pos_zone_ids: torch.Tensor,
        neg_zone_ids: torch.Tensor,
        num_vessels: int,
    ) -> torch.Tensor:
        """
        Adaptive Relation Contrastive Loss (Eq.21).
        Sum over auxiliary behaviors of adaptive NT-Xent (Eq.20).
        """
        total_loss = torch.tensor(0.0, device=vessel_ids.device)

        # Compute for each auxiliary behavior
        for beh_idx, beh_name in enumerate(model.behaviors):
            if beh_name == model.target_behavior:
                continue  # Only auxiliary behaviors

            # Use contrastive head (M2)
            beh_loss = model.contrastive_head.compute_contrastive_loss(
                model_outputs["fused_v"],
                model_outputs["fused_z"],
                vessel_ids, pos_zone_ids, neg_zone_ids,
                beh_idx,
            )
            total_loss = total_loss + beh_loss

        return total_loss

    def compute_chain_contrastive_loss(
        self,
        model: torch.nn.Module,
        model_outputs: dict,
        vessel_ids: torch.Tensor,
        pos_zone_ids: torch.Tensor,
        neg_zone_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Chain contrastive loss (analogous to L_rcl over chain representations).
        """
        chain_v = model_outputs["chain_v"]
        chain_z = model_outputs["chain_z"]

        # Simple contrastive on chain embeddings
        p_v = F.normalize(chain_v[vessel_ids], dim=-1)
        p_z_pos = F.normalize(chain_z[pos_zone_ids], dim=-1)
        p_z_neg = F.normalize(chain_z[neg_zone_ids], dim=-1)

        sim_pos = (p_v * p_z_pos).sum(dim=-1)
        sim_neg = (p_v * p_z_neg).sum(dim=-1)

        # NT-Xent style
        tau = 0.1
        loss = -torch.log(
            torch.exp(sim_pos / tau) /
            (torch.exp(sim_pos / tau) + torch.exp(sim_neg / tau) + 1e-12)
        ).mean()

        return loss

    def compute_total_loss(
        self,
        model: torch.nn.Module,
        model_outputs: dict,
        vessel_ids: torch.Tensor,
        pos_zone_ids: torch.Tensor,
        neg_zone_ids: torch.Tensor,
        num_vessels: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Total loss (Eq.34):
        L_total = L_bpr + lambda_rcl * L_rcl + lambda_chain * L_chain
                  + lambda_entropy * L_entropy + lambda_2 * ||Theta||^2
        """
        device = vessel_ids.device

        # BPR loss (Eq.35)
        bpr_loss = self.compute_bpr_loss(
            model_outputs, vessel_ids, pos_zone_ids, neg_zone_ids, num_vessels
        )

        # Contrastive loss (Eq.21)
        rcl_loss = self.compute_contrastive_loss(
            model, model_outputs, vessel_ids, pos_zone_ids, neg_zone_ids, num_vessels
        )

        # Chain contrastive loss
        chain_loss = self.compute_chain_contrastive_loss(
            model, model_outputs, vessel_ids, pos_zone_ids, neg_zone_ids
        )

        # Fusion entropy (Eq.24) - computed during forward pass
        entropy_loss = model_outputs["fusion_entropy"]

        # L2 regularization
        l2_reg = torch.tensor(0.0, device=device)
        for param in model.parameters():
            l2_reg = l2_reg + torch.norm(param, p=2)

        # Total loss (Eq.34)
        total = (
            bpr_loss
            + self.lambda_rcl * rcl_loss
            + self.lambda_chain * chain_loss
            + self.lambda_entropy * entropy_loss
            + self.lambda_l2 * l2_reg
        )

        loss_dict = {
            "bpr": bpr_loss.item(),
            "rcl": rcl_loss.item(),
            "chain": chain_loss.item(),
            "entropy": entropy_loss.item(),
            "l2": l2_reg.item(),
            "total": total.item(),
        }

        return total, loss_dict
