"""
Maritime-STPCN Synthetic Data Generator & Loader

Generates AIS-inspired maritime dataset per paper Table 5:
- 200 vessels, 100 fishing zones, 6 behavior types
- ~15,748 total interactions, density ~0.787%
- Target behavior (illegal fishing) rate ~18.3%
- M6: Temporal split 75/12.5/12.5% chronological

M6 (Temporal Split): strict chronological partitioning.
All interactions in val occur after latest interaction in train;
all in test after latest in val.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ──────────────── Synthetic Data Generation ────────────────

def generate_synthetic_maritime_data(
    num_vessels: int = 200,
    num_zones: int = 100,
    seed: int = 123,
    save_dir: str | None = None,
) -> dict:
    """
    Generate synthetic maritime AIS-inspired dataset.

    Returns dict with all tensors and metadata needed for Maritime-STPCN.
    """
    rng = np.random.RandomState(seed)

    # ── Zone coordinates (spread over ~400x400 km area) ──
    zone_lon = rng.uniform(110.0, 114.0, num_zones)   # longitude
    zone_lat = rng.uniform(18.0, 22.0, num_zones)      # latitude

    # ── Zone behavior profiles: b_z = (sog, cog_change, duration, distance) ──
    # Eq.8: zone behavior vector aggregated from vessel kinematic profiles
    zone_sog = rng.uniform(2.0, 12.0, num_zones)       # avg speed (knots)
    zone_cog_change = rng.uniform(5.0, 45.0, num_zones) # avg course change (deg)
    zone_duration = rng.uniform(1.0, 8.0, num_zones)    # avg duration (hours)
    zone_distance = rng.uniform(5.0, 30.0, num_zones)   # avg distance (km)
    zone_behavior_vectors = np.stack(
        [zone_sog, zone_cog_change, zone_duration, zone_distance], axis=1
    )  # (num_zones, 4)

    # ── Geographic proximity graph (Eq.6,7) ──
    # A_geo[z,z'] = I[dist(z,z') <= theta_geo], theta=50km
    def haversine_km(lon1, lat1, lon2, lat2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))

    geo_threshold_km = 50.0
    A_geo = np.zeros((num_zones, num_zones), dtype=np.float32)
    for i in range(num_zones):
        for j in range(num_zones):
            d = haversine_km(zone_lon[i], zone_lat[i], zone_lon[j], zone_lat[j])
            if d <= geo_threshold_km:
                A_geo[i, j] = 1.0
    # Add self-loops (A~ = A + I)
    A_geo_aug = A_geo + np.eye(num_zones, dtype=np.float32)

    # ── Behavioral similarity graph (Eq.9) ──
    # A_beh[z,z'] = I[cos(b_z, b_z') >= theta_beh], theta=0.7
    beh_threshold = 0.7
    A_beh = np.zeros((num_zones, num_zones), dtype=np.float32)
    for i in range(num_zones):
        for j in range(num_zones):
            b_i = zone_behavior_vectors[i]
            b_j = zone_behavior_vectors[j]
            cos_sim = np.dot(b_i, b_j) / (np.linalg.norm(b_i) * np.linalg.norm(b_j) + 1e-12)
            if cos_sim >= beh_threshold:
                A_beh[i, j] = 1.0
    A_beh_aug = A_beh + np.eye(num_zones, dtype=np.float32)

    # ── Behavior types ──
    behaviors = [
        "transit", "loitering", "gear_deployment",
        "rendezvous", "at_sea_transshipment", "illegal_fishing",
    ]
    target_behavior = "illegal_fishing"
    relation_order = tuple(behaviors)

    # ── Generate vessel-zone interactions ──
    # Target: ~15,748 total interactions, density ~0.787%
    # Each behavior has different density; target behavior rate ~18.3%
    total_density = 0.00787  # ~0.787%
    target_rate = 0.183

    # Assign per-behavior interaction densities
    behavior_densities = {
        "transit":               0.35,   # most common
        "loitering":             0.22,
        "gear_deployment":       0.15,
        "rendezvous":            0.08,
        "at_sea_transshipment":  0.03,
        "illegal_fishing":       target_rate,  # target behavior
    }

    # Generate timestamps for temporal split
    num_time_steps = 1000
    time_range = np.arange(num_time_steps, dtype=np.float64)

    # Generate interactions for each behavior
    R_behaviors = {}
    all_interactions = []  # (vessel, zone, behavior_idx, timestamp)

    for beh_idx, beh_name in enumerate(behaviors):
        density = behavior_densities[beh_name]
        # Number of interactions for this behavior
        n_interactions = int(num_vessels * num_zones * density * total_density / target_rate)
        n_interactions = max(n_interactions, 100)

        # Generate interactions with geographic constraints
        interactions = []
        for _ in range(n_interactions):
            v = rng.randint(0, num_vessels)
            # Vessel can only interact with zones within operational range (~200km)
            nearby_zones = []
            for z in range(num_zones):
                d = haversine_km(
                    zone_lon[z], zone_lat[z],
                    zone_lon[z], zone_lat[z]  # simplified: vessel near some zone
                )
                nearby_zones.append(z)

            # Sample zone (geographically constrained)
            if beh_name == "transit":
                # Transit: more uniform across zones
                z = rng.randint(0, num_zones)
            elif beh_name == "illegal_fishing":
                # Illegal fishing: concentrated in productive fishing grounds
                productive_zones = np.argsort(zone_duration)[-30:]
                z = productive_zones[rng.randint(0, len(productive_zones))]
            else:
                # Other behaviors: moderately localized
                z = rng.randint(0, num_zones)

            # Timestamp (chronological)
            t = rng.uniform(0, num_time_steps)

            interactions.append((v, z, beh_idx, t))

        all_interactions.extend(interactions)

    # ── M6: Temporal Split ──
    # Sort by timestamp, then split 75/12.5/12.5%
    all_interactions.sort(key=lambda x: x[3])
    n_total = len(all_interactions)
    n_train = int(n_total * 0.75)
    n_val = int(n_total * 0.125)
    n_test = n_total - n_train - n_val

    train_interactions = all_interactions[:n_train]
    val_interactions = all_interactions[n_train:n_train + n_val]
    test_interactions = all_interactions[n_train + n_val:]

    # ── Build interaction matrices R^(k) ──
    # R^(k) in {0,1}^{N_v x N_z}, binary
    num_behaviors = len(behaviors)

    def build_interaction_matrix(interactions_list):
        """Build per-behavior interaction matrices."""
        R = {}
        for beh_idx, beh_name in enumerate(behaviors):
            mat = np.zeros((num_vessels, num_zones), dtype=np.float32)
            for v, z, b, t in interactions_list:
                if b == beh_idx:
                    mat[v, z] = 1.0
            R[beh_name] = mat
        return R

    R_train = build_interaction_matrix(train_interactions)
    R_val = build_interaction_matrix(val_interactions)
    R_test = build_interaction_matrix(test_interactions)

    # ── Build augmented bipartite adjacency matrices ──
    # For GCN propagation: R~^(k) = R^(k) + I (augmented)
    # We construct the full (N_v+N_z) x (N_v+N_z) bipartite adjacency
    def build_bipartite_adj(R_dict, num_v, num_z):
        """Build augmented bipartite adjacency for LightGCN propagation."""
        adjs = {}
        for beh_name, R_mat in R_dict.items():
            # Bipartite adjacency: upper-left=0, upper-right=R, lower-left=R^T, lower-right=0
            N = num_v + num_z
            adj = np.zeros((N, N), dtype=np.float32)
            adj[:num_v, num_v:] = R_mat
            adj[num_v:, :num_v] = R_mat.T
            # Add self-loops
            adj += np.eye(N, dtype=np.float32)
            # Normalize: D~^(-1/2) A~ D~^(-1/2)
            degree = adj.sum(axis=1)
            d_inv_sqrt = np.power(degree, -0.5)
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            D_inv_sqrt = np.diag(d_inv_sqrt)
            adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
            adjs[beh_name] = adj_norm
        return adjs

    relation_adjs_train = build_bipartite_adj(R_train, num_vessels, num_zones)

    # ── Build cascade adjacency (Eq.28) ──
    # A_cas = sum_{(k,k+1) in C} R~^(k) R~^(k+1)
    def build_cascade_adj(R_dict, relation_order, num_v, num_z):
        """Build cascade adjacency matrices."""
        cascade_adjs = {}
        for i in range(len(relation_order) - 1):
            beh_src = relation_order[i]
            beh_dst = relation_order[i + 1]
            if beh_src in R_dict and beh_dst in R_dict:
                R_src = R_dict[beh_src][:num_v, num_v:]  # (N_v, N_z)
                R_dst = R_dict[beh_dst][:num_v, num_v:]  # (N_v, N_z)
                # Cascade: R_src * R_dst^T for vessel-vessel, R_src^T * R_dst for zone-zone
                cas_name = f"{beh_src}_to_{beh_dst}"
                cascade_adjs[cas_name] = (beh_src, beh_dst)
        return cascade_adjs

    cascade_specs = build_cascade_adj(R_train, relation_order, num_vessels, num_zones)

    # ── Build chain adjacency (Eq.29) ──
    # A_chain^(a1...ak) = product of R~^(k) along chain
    # Prune chains with fewer than chain_min (5) edges
    chain_min_edges = 5
    chain_specs = []
    aux_behaviors = [b for b in behaviors if b != target_behavior]
    # Enumerate combinations of auxiliary behaviors + target
    from itertools import combinations
    for size in range(1, len(aux_behaviors) + 1):
        for combo in combinations(range(len(aux_behaviors)), size):
            chain = tuple([aux_behaviors[c] for c in combo]) + (target_behavior,)
            if len(chain) >= 2:  # at least one transition
                chain_specs.append(chain)

    # ── Compute global bias (Eq.30) ──
    # b_global = log(pos / (1 - pos)), pos = |E_train| / (N_v * N_z)
    total_train_edges = sum(R_train[b].sum() for b in behaviors)
    pos_rate = total_train_edges / (num_vessels * num_zones * num_behaviors)
    pos_rate = max(min(pos_rate, 0.999), 0.001)
    global_bias = math.log(pos_rate / (1 - pos_rate))

    # ── Compute per-zone bias (Eq.31) ──
    # beta_j^(0) = log(|E_train(j)| / (pos * N_z))
    zone_edge_counts = np.zeros(num_zones, dtype=np.float32)
    for beh_name in behaviors:
        zone_edge_counts += R_train[beh_name].sum(axis=0)  # sum over vessels
    per_zone_bias_init = np.log(zone_edge_counts / (pos_rate * num_zones + 1e-12))

    # ── Zone pattern features for STP encoder ──
    zone_pattern_features = zone_behavior_vectors.astype(np.float32)

    data = {
        "num_vessels": num_vessels,
        "num_zones": num_zones,
        "num_behaviors": num_behaviors,
        "behaviors": behaviors,
        "target_behavior": target_behavior,
        "relation_order": relation_order,
        "behavior_layers": [2, 2, 2, 1, 1, 1],
        # Zone coordinates
        "zone_lon": zone_lon,
        "zone_lat": zone_lat,
        # Zone behavior vectors (Eq.8)
        "zone_behavior_vectors": zone_behavior_vectors,
        # Spatial adjacency matrices
        "A_geo_aug": A_geo_aug,
        "A_beh_aug": A_beh_aug,
        # Interaction matrices
        "R_train": R_train,
        "R_val": R_val,
        "R_test": R_test,
        # Augmented bipartite adjacency
        "relation_adjs_train": relation_adjs_train,
        # Cascade and chain specs
        "cascade_specs": cascade_specs,
        "chain_specs": chain_specs,
        # Bias terms
        "global_bias": global_bias,
        "per_zone_bias_init": per_zone_bias_init,
        # Zone pattern features
        "zone_pattern_features": zone_pattern_features,
        # All interactions with timestamps
        "train_interactions": train_interactions,
        "val_interactions": val_interactions,
        "test_interactions": test_interactions,
        # Statistics
        "total_interactions": n_total,
        "interaction_density": total_density,
    }

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path / "maritime_data.npz",
            **{k: v for k, v in data.items() if isinstance(v, (np.ndarray, float, int))}
        )
        # Save metadata
        meta = {
            "behaviors": behaviors,
            "target_behavior": target_behavior,
            "relation_order": list(relation_order),
            "behavior_layers": [2, 2, 2, 1, 1, 1],
            "num_vessels": num_vessels,
            "num_zones": num_zones,
            "num_behaviors": num_behaviors,
            "global_bias": global_bias,
            "total_interactions": n_total,
            "interaction_density": total_density,
        }
        with open(save_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Data saved to {save_path}")

    return data


# ──────────────── Dataset Class ────────────────

class MaritimeDataset:
    """Maritime-STPCN dataset with all adjacency matrices and metadata."""

    def __init__(self, data_dict: dict, device: str = "cpu"):
        self.num_vessels = data_dict["num_vessels"]
        self.num_zones = data_dict["num_zones"]
        self.num_behaviors = data_dict["num_behaviors"]
        self.behaviors = data_dict["behaviors"]
        self.target_behavior = data_dict["target_behavior"]
        self.relation_order = data_dict["relation_order"]
        self.behavior_layers = data_dict["behavior_layers"]
        self.global_bias = data_dict["global_bias"]

        self.device = torch.device(device)

        # Convert numpy arrays to torch tensors
        self.zone_behavior_vectors = torch.tensor(
            data_dict["zone_behavior_vectors"], dtype=torch.float32, device=self.device
        )
        self.A_geo_aug = torch.tensor(
            data_dict["A_geo_aug"], dtype=torch.float32, device=self.device
        )
        self.A_beh_aug = torch.tensor(
            data_dict["A_beh_aug"], dtype=torch.float32, device=self.device
        )

        # Per-behavior interaction matrices
        self.R_train = {}
        self.R_val = {}
        self.R_test = {}
        for beh in self.behaviors:
            self.R_train[beh] = torch.tensor(
                data_dict["R_train"][beh], dtype=torch.float32, device=self.device
            )
            self.R_val[beh] = torch.tensor(
                data_dict["R_val"][beh], dtype=torch.float32, device=self.device
            )
            self.R_test[beh] = torch.tensor(
                data_dict["R_test"][beh], dtype=torch.float32, device=self.device
            )

        # Augmented bipartite adjacency matrices
        self.relation_adjs = {}
        for beh, adj_norm in data_dict["relation_adjs_train"].items():
            self.relation_adjs[beh] = torch.tensor(
                adj_norm, dtype=torch.float32, device=self.device
            )

        # Per-zone bias initialization
        self.per_zone_bias_init = torch.tensor(
            data_dict["per_zone_bias_init"], dtype=torch.float32, device=self.device
        )

        # Zone pattern features
        self.zone_pattern_features = torch.tensor(
            data_dict["zone_pattern_features"], dtype=torch.float32, device=self.device
        )

        # STP adjacency list (for M1)
        self.stp_adjs = [self.A_geo_aug, self.A_beh_aug]

    def to(self, device: str) -> "MaritimeDataset":
        """Move all tensors to device."""
        self.device = torch.device(device)
        self.zone_behavior_vectors = self.zone_behavior_vectors.to(self.device)
        self.A_geo_aug = self.A_geo_aug.to(self.device)
        self.A_beh_aug = self.A_beh_aug.to(self.device)
        for beh in self.behaviors:
            self.R_train[beh] = self.R_train[beh].to(self.device)
            self.R_val[beh] = self.R_val[beh].to(self.device)
            self.R_test[beh] = self.R_test[beh].to(self.device)
            if beh in self.relation_adjs:
                self.relation_adjs[beh] = self.relation_adjs[beh].to(self.device)
        self.per_zone_bias_init = self.per_zone_bias_init.to(self.device)
        self.zone_pattern_features = self.zone_pattern_features.to(self.device)
        self.stp_adjs = [self.A_geo_aug, self.A_beh_aug]
        return self


# ──────────────── BPR Sampling & DataLoader ────────────────

class BPRBatchSampler:
    """
    BPR sampling for vessel-zone pairs.
    Negative sampling restricted to geographic neighborhood (200km).
    """

    def __init__(self, dataset: MaritimeDataset, num_negatives: int = 1):
        self.dataset = dataset
        self.num_negatives = num_negatives

        # Build positive pairs for target behavior from train
        R_target = self.dataset.R_train[self.dataset.target_behavior]
        self.pos_pairs = []
        for v in range(self.dataset.num_vessels):
            for z in range(self.dataset.num_zones):
                if R_target[v, z] > 0:
                    self.pos_pairs.append((v, z))

        # Build vessel zone neighborhood for negative sampling
        # All zones within 200km operational range
        self.vessel_neg_candidates = {}
        for v in range(self.dataset.num_vessels):
            # Simplified: all zones (in real implementation, would filter by distance)
            self.vessel_neg_candidates[v] = list(range(self.dataset.num_zones))

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a BPR batch: (vessel, positive_zone, negative_zone)."""
        n = min(batch_size, len(self.pos_pairs))
        indices = random.sample(range(len(self.pos_pairs)), n)

        vessels = []
        pos_zones = []
        neg_zones = []

        for idx in indices:
            v, z_pos = self.pos_pairs[idx]
            # Negative: sample from zones that vessel did NOT interact with (target behavior)
            R_target = self.dataset.R_train[self.dataset.target_behavior]
            neg_candidates = [z for z in self.vessel_neg_candidates[v] if R_target[v, z] == 0]
            if not neg_candidates:
                neg_candidates = list(range(self.dataset.num_zones))
            z_neg = random.choice(neg_candidates)

            vessels.append(v)
            pos_zones.append(z_pos)
            neg_zones.append(z_neg)

        return (
            torch.tensor(vessels, dtype=torch.long),
            torch.tensor(pos_zones, dtype=torch.long),
            torch.tensor(neg_zones, dtype=torch.long),
        )


import json
