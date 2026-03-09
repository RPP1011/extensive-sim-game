#!/usr/bin/env python3
"""Pre-train entity encoder V3 on next-state prediction.

Decomposed approach: separate prediction heads for each dynamic feature group,
trained with symlog + beta-NLL. Static features (armor, ranges, base stats)
are skipped entirely.

Feature groups predicted:
  - hp:       [0, 1, 2]  — hp_pct, shield_pct, resource_pct
  - position: [5, 6]     — x, y
  - cooldown: [17, 20, 23] — ability/heal/CC cd_remaining_pct
  - state:    [24, 25, 26, 27] — is_casting, cast_progress, is_stunned, has_shield
  - exists:   [29]        — alive/dead

Skipped (static/derived): armor(3,4), distance(7), terrain(8-11),
  combat stats(12-14), ability/heal/CC base stats(15-16,18-19,21-22),
  cumulative(28).

Short-horizon training: start at delta 1-3 ticks for easy predictions,
expand to full range only after solidly beating baseline.

Usage:
    uv run --with numpy --with torch training/pretrain_nextstate.py \
        generated/nextstate_combined.npz \
        -o generated/entity_encoder_nextstate.pt \
        --max-steps 50000
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from grokfast import GrokfastEMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTITY_DIM = 30       # raw features per entity in npz
ENTITY_INPUT_DIM = 32  # +2 relative position features (dx_from_self, dy_from_self)
THREAT_DIM = 8
POSITION_DIM = 8
ABILITY_EMB_DIM = 32  # frozen ability transformer [CLS] embedding dimension
MAX_ABILITIES = 9     # max abilities per entity
NUM_TYPES = 6  # self=0, enemy=1, ally=2, threat=3, position=4, ability=5

# Dynamic feature groups: name → list of feature indices
# exists uses BCE (Bernoulli), all others use beta-NLL (Gaussian)
DYNAMIC_GROUPS = {
    "hp": [0, 1],             # hp_pct, shield_pct (resource_pct removed — different dynamics)
    "pos": [5, 6],            # absolute x, y (predicted as movement delta)
    "cd": [17, 20, 23],
    "state": [24, 25, 26, 27],
    "exists": [29],           # BCE loss, not Gaussian
}
BCE_GROUPS = {"exists"}       # groups using sigmoid + BCE instead of beta-NLL
# All dynamic indices (13 total)
DYNAMIC_INDICES = []
for indices in DYNAMIC_GROUPS.values():
    DYNAMIC_INDICES.extend(indices)
DYNAMIC_INDICES = sorted(DYNAMIC_INDICES)
N_DYNAMIC = len(DYNAMIC_INDICES)  # 13


# ---------------------------------------------------------------------------
# Symlog transform (DreamerV3)
# ---------------------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (1 + x.abs()).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    return x.sign() * (x.abs().exp() - 1)


# ---------------------------------------------------------------------------
# Beta-NLL loss (Seitzer et al. 2022)
# ---------------------------------------------------------------------------


def beta_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    variance = log_var.exp()
    weight = variance.detach() ** beta
    nll = 0.5 * log_var + 0.5 * (target - mean) ** 2 / variance
    return (weight * nll).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class PredictionHead(nn.Module):
    """Small MLP predicting mean + log_var for a Gaussian feature group."""

    def __init__(self, d_in: int, n_features: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_features * 2),  # mean + log_var
        )
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean, log_var = out.split(self.n_features, dim=-1)
        return mean, log_var.clamp(-10.0, 10.0)


class BinaryHead(nn.Module):
    """Small MLP predicting logits for a Bernoulli feature group (BCE loss)."""

    def __init__(self, d_in: int, n_features: int, d_hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_features),  # raw logits
        )
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits, not sigmoid


class EntityEncoderDecomposed(nn.Module):
    """EntityEncoderV3 + per-group decomposed prediction heads.

    The encoder portion matches model.py::EntityEncoderV3 exactly,
    so weights transfer directly via `encoder.*` prefix.

    Each dynamic feature group gets its own small prediction head with
    independent beta-NLL loss and learned task weight.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 4,
                 use_abilities: bool = False):
        super().__init__()
        self.d_model = d_model
        self.use_abilities = use_abilities

        # Entity/threat/position projections (match EntityEncoderV3)
        self.encoder = nn.Module()
        self.encoder.entity_proj = nn.Linear(ENTITY_INPUT_DIM, d_model)
        self.encoder.threat_proj = nn.Linear(THREAT_DIM, d_model)
        self.encoder.position_proj = nn.Linear(POSITION_DIM, d_model)
        if use_abilities:
            self.encoder.ability_proj = nn.Linear(ABILITY_EMB_DIM, d_model)
        self.encoder.type_emb = nn.Embedding(NUM_TYPES, d_model)
        self.encoder.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder.out_norm = nn.LayerNorm(d_model)

        # Per-group prediction heads (input: d_model + 1 for delta)
        d_in = d_model + 1
        d_hidden = d_model * 2
        self.heads = nn.ModuleDict()
        for name, indices in DYNAMIC_GROUPS.items():
            if name in BCE_GROUPS:
                self.heads[name] = BinaryHead(d_in, len(indices), d_hidden)
            else:
                self.heads[name] = PredictionHead(d_in, len(indices), d_hidden)

        # Kendall-style per-group task weights
        n_groups = len(DYNAMIC_GROUPS)
        self.log_task_vars = nn.Parameter(torch.zeros(n_groups))

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.007)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
        ability_features: torch.Tensor | None = None,
        ability_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode entities + threats + positions [+ abilities].

        Returns tokens (B, S, d_model) where S = n_entities + n_threats [+ n_positions] [+ n_abilities].
        """
        B = entity_features.shape[0]
        device = entity_features.device

        ent_tokens = self.encoder.entity_proj(entity_features)
        ent_type_embs = self.encoder.type_emb(entity_type_ids)
        ent_tokens = ent_tokens + ent_type_embs
        ent_tokens = self.encoder.input_norm(ent_tokens)

        thr_tokens = self.encoder.threat_proj(threat_features)
        thr_type = torch.full(
            (B, threat_features.shape[1]), 3, dtype=torch.long, device=device,
        )
        thr_tokens = thr_tokens + self.encoder.type_emb(thr_type)
        thr_tokens = self.encoder.input_norm(thr_tokens)

        tokens = torch.cat([ent_tokens, thr_tokens], dim=1)
        full_mask = torch.cat([entity_mask, threat_mask], dim=1)

        if position_features is not None and position_mask is not None:
            pos_tokens = self.encoder.position_proj(position_features)
            pos_type = torch.full(
                (B, position_features.shape[1]), 4, dtype=torch.long, device=device,
            )
            pos_tokens = pos_tokens + self.encoder.type_emb(pos_type)
            pos_tokens = self.encoder.input_norm(pos_tokens)
            tokens = torch.cat([tokens, pos_tokens], dim=1)
            full_mask = torch.cat([full_mask, position_mask], dim=1)

        if self.use_abilities and ability_features is not None and ability_mask is not None:
            abl_tokens = self.encoder.ability_proj(ability_features)
            abl_type = torch.full(
                (B, ability_features.shape[1]), 5, dtype=torch.long, device=device,
            )
            abl_tokens = abl_tokens + self.encoder.type_emb(abl_type)
            abl_tokens = self.encoder.input_norm(abl_tokens)
            tokens = torch.cat([tokens, abl_tokens], dim=1)
            full_mask = torch.cat([full_mask, ability_mask], dim=1)

        tokens = self.encoder.encoder(tokens, src_key_padding_mask=full_mask)
        tokens = self.encoder.out_norm(tokens)

        return tokens

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        delta_normalized: torch.Tensor,
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
        ability_features: torch.Tensor | None = None,
        ability_mask: torch.Tensor | None = None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Predict per-group (mean, log_var) for each dynamic feature group.

        Returns dict: group_name → (mean, log_var), each (B, E, n_features).
        Mean is residual in symlog space: symlog(current) + predicted_delta.
        """
        tokens = self.encode(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask, position_features, position_mask,
            ability_features, ability_mask,
        )

        n_entities = entity_features.shape[1]
        entity_tokens = tokens[:, :n_entities]  # (B, E, d_model)

        # Broadcast delta
        delta_feat = delta_normalized.unsqueeze(-1).unsqueeze(-1).expand(
            -1, n_entities, 1,
        )
        conditioned = torch.cat([entity_tokens, delta_feat], dim=-1)  # (B, E, d_model+1)

        results = {}
        for name, indices in DYNAMIC_GROUPS.items():
            if name in BCE_GROUPS:
                # Binary head: returns logits directly
                logits = self.heads[name](conditioned)
                results[name] = logits  # (B, E, n_features)
            else:
                mean_delta, log_var = self.heads[name](conditioned)
                # Residual in symlog space
                current_symlog = symlog(entity_features[:, :, indices])
                mean = current_symlog + mean_delta
                results[name] = (mean, log_var)

        return results


# ---------------------------------------------------------------------------
# Dataset (unchanged from previous version)
# ---------------------------------------------------------------------------


class NextStateDataset:
    """Loads pre-processed .npz next-state dataset."""

    def __init__(self, path: Path, max_samples: int = 0):
        if path.suffix != ".npz":
            npz_path = path.with_suffix(".npz")
            if npz_path.exists():
                path = npz_path
            else:
                raise ValueError(f"Expected .npz file, got {path}")

        data = np.load(path)
        self.ent_feat = data["ent_feat"]
        self.ent_types = data["ent_types"]
        self.ent_mask = data["ent_mask"]
        self.ent_unit_ids = data["ent_unit_ids"]
        self.thr_feat = data["thr_feat"]
        self.thr_mask = data["thr_mask"]
        self.pos_feat = data["pos_feat"]
        self.pos_mask = data["pos_mask"]
        self.ticks = data["ticks"]
        self.scenario_ids = data["scenario_ids"]

        # Ability LUT (optional — may not exist in older datasets)
        if "abl_lut" in data:
            self.abl_lut = data["abl_lut"]          # (n_entries, MAX_ABILITIES, D_MODEL)
            self.abl_lut_counts = data["abl_lut_counts"]  # (n_entries,)
            self.abl_ent_idx = data["abl_ent_idx"]   # (n_samples, MAX_ENTS)
            self.has_abilities = True
            n_with = (self.abl_ent_idx >= 0).any(axis=1).sum()
            print(f"  Ability LUT: {len(self.abl_lut)} entries, "
                  f"{n_with}/{len(self.ticks)} samples have ability data")
        else:
            self.has_abilities = False

        if max_samples > 0 and max_samples < len(self.ticks):
            idx = np.random.choice(len(self.ticks), max_samples, replace=False)
            idx.sort()
            self._slice(idx)

        self._build_scenario_index()

        n = len(self.ticks)
        n_scenarios = len(self._scenario_groups)
        print(f"Loaded {n} snapshots from {n_scenarios} scenarios ({path})")
        group_sizes = [len(v) for v in self._scenario_groups.values()]
        print(f"  Snapshots per scenario: min={min(group_sizes)}, "
              f"max={max(group_sizes)}, median={sorted(group_sizes)[len(group_sizes)//2]}")

    def _slice(self, idx: np.ndarray):
        self.ent_feat = self.ent_feat[idx]
        self.ent_types = self.ent_types[idx]
        self.ent_mask = self.ent_mask[idx]
        self.ent_unit_ids = self.ent_unit_ids[idx]
        self.thr_feat = self.thr_feat[idx]
        self.thr_mask = self.thr_mask[idx]
        self.pos_feat = self.pos_feat[idx]
        self.pos_mask = self.pos_mask[idx]
        self.ticks = self.ticks[idx]
        self.scenario_ids = self.scenario_ids[idx]
        if self.has_abilities:
            self.abl_ent_idx = self.abl_ent_idx[idx]

    def _build_scenario_index(self):
        self._scenario_groups: dict[int, np.ndarray] = {}
        for sc_id in np.unique(self.scenario_ids):
            mask = self.scenario_ids == sc_id
            indices = np.where(mask)[0]
            order = np.argsort(self.ticks[indices])
            self._scenario_groups[int(sc_id)] = indices[order]

        self._flat_index: list[tuple[int, int]] = []
        for sc_id, indices in self._scenario_groups.items():
            for pos in range(len(indices)):
                self._flat_index.append((sc_id, pos))

    def __len__(self) -> int:
        return len(self._flat_index)

    def split(self, val_frac: float = 0.15) -> tuple["NextStateDataset", "NextStateDataset"]:
        scenario_ids_list = list(self._scenario_groups.keys())

        # Stratified split: ensure val set has ability-bearing scenarios if available
        if self.has_abilities:
            abl_scenarios = set()
            no_abl_scenarios = set()
            for sc_id, group in self._scenario_groups.items():
                if (self.abl_ent_idx[group] >= 0).any():
                    abl_scenarios.add(sc_id)
                else:
                    no_abl_scenarios.add(sc_id)
            abl_list = list(abl_scenarios)
            no_abl_list = list(no_abl_scenarios)
            random.shuffle(abl_list)
            random.shuffle(no_abl_list)
            n_val = max(1, int(len(scenario_ids_list) * val_frac))
            # Take at least 1 from each stratum if possible
            n_val_abl = max(1, int(n_val * len(abl_list) / len(scenario_ids_list))) if abl_list else 0
            n_val_no = n_val - n_val_abl
            val_set = set(abl_list[:n_val_abl]) | set(no_abl_list[:n_val_no])
            print(f"Stratified val split: {n_val_abl} with abilities, {n_val_no} without")
        else:
            random.shuffle(scenario_ids_list)
            n_val = max(1, int(len(scenario_ids_list) * val_frac))
            val_set = set(scenario_ids_list[:n_val])

        val_idx = np.concatenate([self._scenario_groups[s] for s in val_set])
        train_ids = [s for s in scenario_ids_list if s not in val_set]
        train_idx = np.concatenate([self._scenario_groups[s] for s in train_ids])

        def _make_subset(parent, idx):
            ds = NextStateDataset.__new__(NextStateDataset)
            ds.ent_feat = parent.ent_feat
            ds.ent_types = parent.ent_types
            ds.ent_mask = parent.ent_mask
            ds.ent_unit_ids = parent.ent_unit_ids
            ds.thr_feat = parent.thr_feat
            ds.thr_mask = parent.thr_mask
            ds.pos_feat = parent.pos_feat
            ds.pos_mask = parent.pos_mask
            ds.ticks = parent.ticks
            ds.scenario_ids = parent.scenario_ids
            ds.has_abilities = parent.has_abilities
            if parent.has_abilities:
                ds.abl_lut = parent.abl_lut
                ds.abl_lut_counts = parent.abl_lut_counts
                ds.abl_ent_idx = parent.abl_ent_idx
            ds._scenario_groups = {}
            for sc_id in np.unique(parent.scenario_ids[idx]):
                sc_idx = idx[parent.scenario_ids[idx] == sc_id]
                order = np.argsort(parent.ticks[sc_idx])
                ds._scenario_groups[int(sc_id)] = sc_idx[order]
            ds._flat_index = []
            for sc_id, indices in ds._scenario_groups.items():
                for pos in range(len(indices)):
                    ds._flat_index.append((sc_id, pos))
            return ds

        train_ds = _make_subset(self, train_idx)
        val_ds = _make_subset(self, val_idx)

        print(f"Split: {len(train_ds)} train ({len(train_ds._scenario_groups)} scenarios), "
              f"{len(val_ds)} val ({len(val_ds._scenario_groups)} scenarios)")
        return train_ds, val_ds

    def precompute_pairs(self, min_delta: int, max_delta: int):
        """Precompute all valid (now, future) pairs and target features.

        Call once before training. After this, sample_batch just indexes
        into flat precomputed arrays — no per-step unit_id matching.
        """
        MAX_ENTS = self.ent_feat.shape[1]
        now_list = []
        target_list = []
        delta_list = []

        for sc_id, group in self._scenario_groups.items():
            for pos in range(len(group)):
                idx_now = group[pos]
                t_now = self.ticks[idx_now]

                for delta in range(min_delta, max_delta + 1):
                    t_target = t_now + delta
                    # Find first snapshot >= t_target
                    future_idx = None
                    for fp in range(pos + 1, len(group)):
                        if self.ticks[group[fp]] >= t_target:
                            future_idx = group[fp]
                            break
                    if future_idx is None:
                        continue

                    # Build aligned target via unit_id matching
                    target_feat = np.zeros((MAX_ENTS, ENTITY_DIM), dtype=np.float32)
                    future_feats = {}
                    for j in range(MAX_ENTS):
                        uid = self.ent_unit_ids[future_idx, j]
                        if uid < 0:
                            break
                        future_feats[uid] = self.ent_feat[future_idx, j]
                    for j in range(MAX_ENTS):
                        uid = self.ent_unit_ids[idx_now, j]
                        if uid < 0:
                            break
                        if uid in future_feats:
                            target_feat[j] = future_feats[uid]

                    actual_delta = int(self.ticks[future_idx] - t_now)
                    now_list.append(idx_now)
                    target_list.append(target_feat)
                    delta_list.append(actual_delta / max_delta)

        self._pre_now = np.array(now_list)
        self._pre_targets = np.array(target_list)
        self._pre_deltas = np.array(delta_list, dtype=np.float32)

        n = len(self._pre_now)
        print(f"Precomputed {n} pairs ({n / len(self._flat_index):.1f}x snapshots)")

    @staticmethod
    def _augment_ent(raw: np.ndarray) -> np.ndarray:
        """Add relative position features (dx, dy from self) per batch."""
        self_x = raw[:, 0:1, 5:6]
        self_y = raw[:, 0:1, 6:7]
        return np.concatenate([
            raw,
            raw[:, :, 5:6] - self_x,
            raw[:, :, 6:7] - self_y,
        ], axis=-1)

    def _expand_abilities(self, now_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand LUT indices to ability features/mask for a batch.

        Returns:
          abl_feat: (B, total_slots, D_MODEL) — flattened abilities across all entities
          abl_mask: (B, total_slots) — True = padding
        """
        B = len(now_indices)
        MAX_ENTS = self.ent_feat.shape[1]
        # Each entity can have up to MAX_ABILITIES, total = MAX_ENTS * MAX_ABILITIES
        total_slots = MAX_ENTS * MAX_ABILITIES

        abl_feat = np.zeros((B, total_slots, self.abl_lut.shape[2]), dtype=np.float32)
        abl_mask = np.ones((B, total_slots), dtype=np.bool_)

        ent_idx_batch = self.abl_ent_idx[now_indices]  # (B, MAX_ENTS)

        for b in range(B):
            slot = 0
            for e in range(MAX_ENTS):
                lut_idx = ent_idx_batch[b, e]
                if lut_idx >= 0:
                    na = int(self.abl_lut_counts[lut_idx])
                    for a in range(na):
                        if slot < total_slots:
                            abl_feat[b, slot] = self.abl_lut[lut_idx, a]
                            abl_mask[b, slot] = False
                            slot += 1
        return abl_feat, abl_mask

    def _expand_abilities_fast(self, now_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized ability expansion — no Python loops over batch."""
        B = len(now_indices)
        MAX_ENTS = self.ent_feat.shape[1]
        total_slots = MAX_ENTS * MAX_ABILITIES
        d = self.abl_lut.shape[2]

        abl_feat = np.zeros((B, total_slots, d), dtype=np.float32)
        abl_mask = np.ones((B, total_slots), dtype=np.bool_)

        ent_idx = self.abl_ent_idx[now_indices]  # (B, MAX_ENTS)

        # For each entity slot, expand its abilities into consecutive ability slots
        for e in range(MAX_ENTS):
            lut_indices = ent_idx[:, e]  # (B,)
            has_ability = lut_indices >= 0  # (B,)
            if not has_ability.any():
                continue

            valid_lut = lut_indices[has_ability]
            counts = self.abl_lut_counts[valid_lut]  # (n_valid,)
            embs = self.abl_lut[valid_lut]  # (n_valid, MAX_ABILITIES, d)

            for a in range(MAX_ABILITIES):
                slot = e * MAX_ABILITIES + a
                if slot >= total_slots:
                    break
                # Entities with this many abilities
                has_a = counts > a
                if not has_a.any():
                    break
                # Map back to batch indices
                batch_idx = np.where(has_ability)[0][has_a]
                abl_feat[batch_idx, slot] = embs[has_a, a]
                abl_mask[batch_idx, slot] = False

        return abl_feat, abl_mask

    def sample_batch(
        self, batch_size: int, min_delta: int = 1, max_delta: int = 5,
    ) -> dict[str, torch.Tensor] | None:
        idx = np.random.randint(0, len(self._pre_now), size=batch_size)
        now_indices = self._pre_now[idx]

        result = {
            "entity_features": torch.tensor(
                self._augment_ent(self.ent_feat[now_indices]),
                dtype=torch.float, device=DEVICE),
            "entity_type_ids": torch.tensor(
                self.ent_types[now_indices], dtype=torch.long, device=DEVICE),
            "threat_features": torch.tensor(
                self.thr_feat[now_indices], dtype=torch.float, device=DEVICE),
            "entity_mask": torch.tensor(
                self.ent_mask[now_indices], device=DEVICE),
            "threat_mask": torch.tensor(
                self.thr_mask[now_indices], device=DEVICE),
            "position_features": torch.tensor(
                self.pos_feat[now_indices], dtype=torch.float, device=DEVICE),
            "position_mask": torch.tensor(
                self.pos_mask[now_indices], device=DEVICE),
            "target_features": torch.tensor(
                self._pre_targets[idx], dtype=torch.float, device=DEVICE),
            "delta_normalized": torch.tensor(
                self._pre_deltas[idx], device=DEVICE),
        }

        if self.has_abilities:
            abl_feat, abl_mask = self._expand_abilities_fast(now_indices)
            result["ability_features"] = torch.tensor(
                abl_feat, dtype=torch.float, device=DEVICE)
            result["ability_mask"] = torch.tensor(abl_mask, device=DEVICE)

        return result

    def iter_batches(
        self, batch_size: int, min_delta: int = 1, max_delta: int = 5,
    ):
        """Iterate all precomputed pairs in shuffled order."""
        indices = np.random.permutation(len(self._pre_now))
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            if len(idx) < 2:
                continue
            now_indices = self._pre_now[idx]
            result = {
                "entity_features": torch.tensor(
                    self._augment_ent(self.ent_feat[now_indices]),
                    dtype=torch.float, device=DEVICE),
                "entity_type_ids": torch.tensor(
                    self.ent_types[now_indices], dtype=torch.long, device=DEVICE),
                "threat_features": torch.tensor(
                    self.thr_feat[now_indices], dtype=torch.float, device=DEVICE),
                "entity_mask": torch.tensor(
                    self.ent_mask[now_indices], device=DEVICE),
                "threat_mask": torch.tensor(
                    self.thr_mask[now_indices], device=DEVICE),
                "position_features": torch.tensor(
                    self.pos_feat[now_indices], dtype=torch.float, device=DEVICE),
                "position_mask": torch.tensor(
                    self.pos_mask[now_indices], device=DEVICE),
                "target_features": torch.tensor(
                    self._pre_targets[idx], dtype=torch.float, device=DEVICE),
                "delta_normalized": torch.tensor(
                    self._pre_deltas[idx], device=DEVICE),
            }
            if self.has_abilities:
                abl_feat, abl_mask = self._expand_abilities_fast(now_indices)
                result["ability_features"] = torch.tensor(
                    abl_feat, dtype=torch.float, device=DEVICE)
                result["ability_mask"] = torch.tensor(abl_mask, device=DEVICE)
            yield result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = NextStateDataset(Path(args.dataset), max_samples=args.max_samples)
    train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    print("Precomputing pairs...")
    train_ds.precompute_pairs(args.min_delta, args.max_delta)
    val_ds.precompute_pairs(args.min_delta, args.max_delta)

    use_abilities = args.use_abilities and train_ds.has_abilities
    if args.use_abilities and not train_ds.has_abilities:
        print("WARNING: --use-abilities requested but dataset has no ability data")

    model = EntityEncoderDecomposed(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        use_abilities=use_abilities,
    ).to(DEVICE)

    # Warm-start encoder from Phase 1 checkpoint if provided
    if args.warm_start:
        state = torch.load(args.warm_start, map_location=DEVICE, weights_only=True)
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            # Only load encoder weights (not old state_head)
            if k.startswith("encoder.") and k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        print(f"Warm-started encoder from {args.warm_start} ({loaded} params loaded)")

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_heads = sum(p.numel() for p in model.heads.parameters())
    print(f"Model: {n_params:,} total, encoder: {n_encoder:,}, heads: {n_heads:,}")
    for name, head in model.heads.items():
        hp = sum(p.numel() for p in head.parameters())
        print(f"  {name}: {hp} params ({len(DYNAMIC_GROUPS[name])} features)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10,
    )

    batch_size = args.batch_size
    gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    log_path = Path(args.output).with_suffix(".csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)

    group_names = list(DYNAMIC_GROUPS.keys())
    log_writer.writerow([
        "step", "train_loss",
        *[f"L_{g}" for g in group_names],
        *[f"tw_{g}" for g in group_names],
        *[f"val_mae_{g}" for g in group_names],
        *[f"base_mae_{g}" for g in group_names],
        "val_mae_all", "baseline_mae_all",
        "weight_norm", "grad_norm", "lr", "elapsed_s",
    ])

    best_metric = float("inf")
    start_time = time.time()
    model.train()

    # Fixed delta range — no curriculum, train at target range directly
    def get_delta_range(step: int) -> tuple[int, int]:
        return args.min_delta, args.max_delta

    beta = args.beta_nll

    print(f"\nDecomposed next-state prediction: max_steps={args.max_steps}")
    print(f"Dynamic features: {N_DYNAMIC} dims in {len(DYNAMIC_GROUPS)} groups")
    for name, indices in DYNAMIC_GROUPS.items():
        print(f"  {name}: indices {indices}")
    print(f"Delta range: [{args.min_delta}, {args.max_delta}] ticks (fixed)")
    print(f"Beta-NLL beta={beta}, Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Ability tokens: {'enabled' if use_abilities else 'disabled'}")
    print(f"Device: {DEVICE}\n")

    for step in range(1, args.max_steps + 1):
        cur_min, cur_max = get_delta_range(step)
        batch = train_ds.sample_batch(
            batch_size, min_delta=cur_min, max_delta=cur_max,
        )
        if batch is None:
            print("WARNING: Failed to sample batch, skipping step")
            continue

        predictions = model(
            batch["entity_features"], batch["entity_type_ids"],
            batch["threat_features"], batch["entity_mask"], batch["threat_mask"],
            batch["delta_normalized"],
            batch["position_features"], batch["position_mask"],
            batch.get("ability_features"), batch.get("ability_mask"),
        )

        valid = ~batch["entity_mask"]  # (B, E)

        # Per-group losses
        group_losses = {}
        # Get exists target for hp masking (only train hp where target is alive)
        target_exists = batch["target_features"][:, :, 29]  # (B, E)
        for i, (name, indices) in enumerate(DYNAMIC_GROUPS.items()):
            if name in BCE_GROUPS:
                # BCE loss for binary features
                logits = predictions[name]
                target_bin = batch["target_features"][:, :, indices]
                valid_exp = valid.unsqueeze(-1).expand_as(logits)
                group_losses[name] = F.binary_cross_entropy_with_logits(
                    logits[valid_exp], target_bin[valid_exp],
                )
            else:
                mean, log_var = predictions[name]
                target = symlog(batch["target_features"][:, :, indices])

                # For hp group: mask out entities that die (target exists=0)
                # — predicting hp of dead entities is meaningless noise
                if name == "hp":
                    alive_mask = valid & (target_exists > 0.5)  # (B, E)
                    valid_exp = alive_mask.unsqueeze(-1).expand_as(mean)
                else:
                    valid_exp = valid.unsqueeze(-1).expand_as(mean)

                if valid_exp.any():
                    group_losses[name] = beta_nll_loss(
                        mean[valid_exp], log_var[valid_exp], target[valid_exp], beta=beta,
                    )
                else:
                    group_losses[name] = torch.tensor(0.0, device=mean.device)

        # Fixed per-group loss weights (cd too noisy, state low signal)
        GROUP_LOSS_WEIGHTS = {"hp": 0.0, "pos": 1.0, "cd": 0.0, "state": 0.1, "exists": 1.0}
        loss = sum(
            GROUP_LOSS_WEIGHTS[name] * group_losses[name]
            for name in group_names
        )

        optimizer.zero_grad()
        loss.backward()
        gf.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step <= 10:
            warmup_sched.step()

        if step % args.eval_every == 0:
            model.eval()

            # Accumulate per-group error sums (vectorized)
            group_err_sum: dict[str, float] = {g: 0.0 for g in group_names}
            group_base_sum: dict[str, float] = {g: 0.0 for g in group_names}
            group_count: dict[str, int] = {g: 0 for g in group_names}

            for vb in val_ds.iter_batches(
                batch_size, min_delta=args.min_delta, max_delta=args.max_delta,
            ):
                with torch.no_grad():
                    vpreds = model(
                        vb["entity_features"], vb["entity_type_ids"],
                        vb["threat_features"], vb["entity_mask"], vb["threat_mask"],
                        vb["delta_normalized"],
                        vb["position_features"], vb["position_mask"],
                        vb.get("ability_features"), vb.get("ability_mask"),
                    )

                vvalid = ~vb["entity_mask"]  # (B, E)
                vtarget_exists = vb["target_features"][:, :, 29]

                for name, indices in DYNAMIC_GROUPS.items():
                    vtarget_orig = vb["target_features"][:, :, indices]
                    vcurrent_orig = vb["entity_features"][:, :, indices]

                    if name in BCE_GROUPS:
                        vpred_orig = torch.sigmoid(vpreds[name])
                    else:
                        vmean, _ = vpreds[name]
                        vpred_orig = symexp(vmean)

                    # Build mask: valid entities, plus alive-only for hp
                    mask = vvalid  # (B, E)
                    if name == "hp":
                        mask = mask & (vtarget_exists > 0.5)

                    mask_exp = mask.unsqueeze(-1).expand_as(vpred_orig)  # (B, E, F)
                    n = mask_exp.sum().item()
                    if n > 0:
                        group_err_sum[name] += (vpred_orig - vtarget_orig).abs()[mask_exp].sum().item()
                        group_base_sum[name] += (vcurrent_orig - vtarget_orig).abs()[mask_exp].sum().item()
                        group_count[name] += int(n)

            if group_count["exists"] == 0:
                model.train()
                continue

            # Compute per-group MAE
            val_maes = {}
            base_maes = {}
            for name in group_names:
                if group_count[name] > 0:
                    val_maes[name] = group_err_sum[name] / group_count[name]
                    base_maes[name] = group_base_sum[name] / group_count[name]
                else:
                    val_maes[name] = 0.0
                    base_maes[name] = 0.0

            # Overall MAE as mean of per-group MAEs (groups may have different sample counts)
            val_mae_all = sum(val_maes[g] for g in group_names) / len(group_names)
            base_mae_all = sum(base_maes[g] for g in group_names) / len(group_names)

            # Task weights (fixed)
            tw = GROUP_LOSS_WEIGHTS

            weight_norm = sum(
                p.data.norm().item() ** 2 for p in model.parameters()
            ) ** 0.5

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            log_writer.writerow([
                step, f"{loss.item():.6f}",
                *[f"{group_losses[g].item():.6f}" for g in group_names],
                *[f"{tw[g]:.4f}" for g in group_names],
                *[f"{val_maes[g]:.4f}" for g in group_names],
                *[f"{base_maes[g]:.4f}" for g in group_names],
                f"{val_mae_all:.4f}", f"{base_mae_all:.4f}",
                f"{weight_norm:.4f}", f"{grad_norm:.4f}", f"{lr:.6f}", f"{elapsed:.1f}",
            ])
            log_file.flush()

            marker = ""
            if val_mae_all < best_metric:
                best_metric = val_mae_all
                torch.save(model.state_dict(), args.output)
                marker = " *"

            # Per-group improvement summary
            parts = []
            for name in group_names:
                imp = (1 - val_maes[name] / base_maes[name]) * 100 if base_maes[name] > 0 else 0
                parts.append(f"{name} {val_maes[name]:.4f}({imp:+.0f}%)")

            overall_imp = (1 - val_mae_all / base_mae_all) * 100 if base_mae_all > 0 else 0
            print(
                f"step {step:>7d} | "
                f"loss {loss.item():.4f} | "
                f"mae {val_mae_all:.4f} (base {base_mae_all:.4f}, {overall_imp:+.1f}%) | "
                f"{' '.join(parts)} | "
                f"delta [{cur_min},{cur_max}]"
                f"{marker}"
            )

            model.train()

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed:.0f}s")
    print(f"Best val MAE (dynamic features): {best_metric:.4f}")
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {log_path}")


def main():
    p = argparse.ArgumentParser(
        description="Pre-train entity encoder with decomposed next-state prediction"
    )
    p.add_argument("dataset", help=".npz next-state dataset")
    p.add_argument("-o", "--output", default="generated/entity_encoder_nextstate.pt")

    p.add_argument("--max-samples", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--min-delta", type=int, default=1,
                    help="Min prediction horizon in ticks (default: 1)")
    p.add_argument("--max-delta", type=int, default=10,
                    help="Max prediction horizon in ticks (default: 10)")

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)

    p.add_argument("--beta-nll", type=float, default=0.5)

    p.add_argument("--warm-start", type=str, default=None,
                    help="Path to checkpoint to warm-start encoder from")

    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)

    p.add_argument("--use-abilities", action="store_true",
                    help="Include frozen ability transformer embeddings as type=5 context tokens")

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
