#!/usr/bin/env python3
"""Pre-train entity encoder V3 on next-state prediction.

Instead of predicting fight outcome (long-horizon), predict per-entity state
changes over a short horizon (5-40 ticks). This teaches the encoder to
understand tactical dynamics: who takes damage, who gets healed, who dies.

Key techniques (from world model / multi-task learning literature):
- **Symlog transform**: compress feature magnitudes so all 30 dims contribute
  equally to loss (DreamerV3, Hafner 2023)
- **Gaussian output head + beta-NLL**: predict mean + log-variance per feature,
  naturally downweighting stochastic/noisy features while modeling uncertainty
  (Seitzer et al. 2022, beta=0.5)
- **Residual prediction**: predict delta from current state, stable features
  get free baseline

Usage:
    uv run --with numpy --with torch training/pretrain_nextstate.py \
        generated/nextstate_combined.npz \
        -o generated/entity_encoder_nextstate.pt \
        --max-steps 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from grokfast import GrokfastEMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTITY_DIM = 30
THREAT_DIM = 8
POSITION_DIM = 8
NUM_TYPES = 5  # self=0, enemy=1, ally=2, threat=3, position=4


# ---------------------------------------------------------------------------
# Symlog transform (DreamerV3)
# ---------------------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) * log(1 + |x|). Compresses large magnitudes, preserves small."""
    return x.sign() * (1 + x.abs()).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
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
    """Gaussian NLL with beta-weighting to prevent variance inflation.

    L = (variance^beta) * [0.5 * log_var + 0.5 * (target - mean)^2 / variance]

    beta=0.5 balances mean accuracy and calibrated variance.
    """
    variance = log_var.exp()
    # Detach variance^beta so it acts as a weighting, not a gradient path
    weight = variance.detach() ** beta
    nll = 0.5 * log_var + 0.5 * (target - mean) ** 2 / variance
    return (weight * nll).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EntityEncoderNextState(nn.Module):
    """EntityEncoderV3 + per-entity next-state prediction heads.

    The encoder portion matches model.py::EntityEncoderV3 exactly,
    so weights transfer directly via `encoder.*` prefix.

    Output head predicts 30 means + 30 log-variances (Gaussian),
    trained with beta-NLL loss for calibrated uncertainty.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model

        # Entity/threat/position projections (match EntityEncoderV3)
        self.encoder = nn.Module()
        self.encoder.entity_proj = nn.Linear(ENTITY_DIM, d_model)
        self.encoder.threat_proj = nn.Linear(THREAT_DIM, d_model)
        self.encoder.position_proj = nn.Linear(POSITION_DIM, d_model)
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

        # Per-entity next-state prediction head
        # Predicts 30 mean deltas + 30 log-variances (Gaussian output)
        self.state_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model * 2),  # +1 for delta encoding
            nn.GELU(),
            nn.Linear(d_model * 2, ENTITY_DIM * 2),  # mean + log_var
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode entities + threats + positions.

        Returns (tokens, full_mask) where tokens[:, :n_entities] are entity tokens.
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

        tokens = self.encoder.encoder(tokens, src_key_padding_mask=full_mask)
        tokens = self.encoder.out_norm(tokens)

        return tokens, full_mask

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict per-entity state at T+delta as Gaussian (mean, log_var).

        Parameters
        ----------
        delta_normalized : (batch,) — delta / max_delta, in [0, 1]

        Returns
        -------
        mean : (batch, max_entities, 30) — predicted symlog(entity features) at T+delta
        log_var : (batch, max_entities, 30) — per-feature log-variance (uncertainty)
        """
        tokens, full_mask = self.encode(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask, position_features, position_mask,
        )

        n_entities = entity_features.shape[1]
        entity_tokens = tokens[:, :n_entities]  # (B, E, d_model)

        # Broadcast delta to each entity token
        delta_feat = delta_normalized.unsqueeze(-1).unsqueeze(-1).expand(
            -1, n_entities, 1
        )  # (B, E, 1)
        conditioned = torch.cat([entity_tokens, delta_feat], dim=-1)  # (B, E, d_model+1)

        out = self.state_head(conditioned)  # (B, E, 60)
        mean_delta, log_var = out.split(ENTITY_DIM, dim=-1)  # each (B, E, 30)

        # Clamp log_var to prevent numerical issues
        log_var = log_var.clamp(-10.0, 10.0)

        # Residual in symlog space: predict delta from current symlog state
        mean = symlog(entity_features) + mean_delta

        return mean, log_var


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class NextStateDataset:
    """Loads pre-processed .npz next-state dataset.

    Npz format:
        ent_feat: (N, MAX_ENTS, 30)
        ent_types: (N, MAX_ENTS)
        ent_mask: (N, MAX_ENTS) — True = padding
        ent_unit_ids: (N, MAX_ENTS) — unit ID per slot, -1 = padding
        thr_feat: (N, MAX_THR, 8)
        thr_mask: (N, MAX_THR)
        pos_feat: (N, MAX_POS, 8)
        pos_mask: (N, MAX_POS)
        uhp_ids: (N, MAX_UNITS) — unit IDs for HP lookup, -1 = padding
        uhp_vals: (N, MAX_UNITS) — HP ratios
        ticks: (N,) — tick number
        scenario_ids: (N,) — integer scenario ID
    """

    def __init__(self, path: Path, max_samples: int = 0):
        if path.suffix != ".npz":
            npz_path = path.with_suffix(".npz")
            if npz_path.exists():
                path = npz_path
            else:
                raise ValueError(f"Expected .npz file, got {path}. "
                                 "Convert JSONL first.")

        data = np.load(path)
        self.ent_feat = data["ent_feat"]
        self.ent_types = data["ent_types"]
        self.ent_mask = data["ent_mask"]
        self.ent_unit_ids = data["ent_unit_ids"]
        self.thr_feat = data["thr_feat"]
        self.thr_mask = data["thr_mask"]
        self.pos_feat = data["pos_feat"]
        self.pos_mask = data["pos_mask"]
        self.uhp_ids = data["uhp_ids"]
        self.uhp_vals = data["uhp_vals"]
        self.ticks = data["ticks"]
        self.scenario_ids = data["scenario_ids"]

        if max_samples > 0 and max_samples < len(self.ticks):
            idx = np.random.choice(len(self.ticks), max_samples, replace=False)
            idx.sort()
            self._slice(idx)

        # Build per-scenario sorted indices for pairing
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
        self.uhp_ids = self.uhp_ids[idx]
        self.uhp_vals = self.uhp_vals[idx]
        self.ticks = self.ticks[idx]
        self.scenario_ids = self.scenario_ids[idx]

    def _build_scenario_index(self):
        """Group indices by scenario, sorted by tick within each group."""
        self._scenario_groups: dict[int, np.ndarray] = {}
        for sc_id in np.unique(self.scenario_ids):
            mask = self.scenario_ids == sc_id
            indices = np.where(mask)[0]
            # Sort by tick
            order = np.argsort(self.ticks[indices])
            self._scenario_groups[int(sc_id)] = indices[order]

        # Flat index: (scenario_id, position_in_group)
        self._flat_index: list[tuple[int, int]] = []
        for sc_id, indices in self._scenario_groups.items():
            for pos in range(len(indices)):
                self._flat_index.append((sc_id, pos))

    def __len__(self) -> int:
        return len(self._flat_index)

    def split(self, val_frac: float = 0.15) -> tuple["NextStateDataset", "NextStateDataset"]:
        """Split by scenario to prevent data leakage."""
        scenario_ids_list = list(self._scenario_groups.keys())
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
            ds.uhp_ids = parent.uhp_ids
            ds.uhp_vals = parent.uhp_vals
            ds.ticks = parent.ticks
            ds.scenario_ids = parent.scenario_ids
            # Rebuild scenario groups for this subset
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

    def _find_future(self, sc_id: int, pos: int, delta: int) -> int | None:
        """Find index of snapshot at tick T+delta in the same scenario."""
        group = self._scenario_groups[sc_id]
        idx_now = group[pos]
        t_now = self.ticks[idx_now]
        t_target = t_now + delta

        # Linear scan forward (group is sorted by tick)
        for future_pos in range(pos + 1, len(group)):
            future_idx = group[future_pos]
            if self.ticks[future_idx] >= t_target:
                return future_idx
        return None

    def _build_targets(self, idx_now: int, idx_future: int, max_delta: int, delta: int):
        """Build per-entity full feature targets by matching unit IDs."""
        MAX_ENTS = self.ent_feat.shape[1]
        target_feat = np.zeros((MAX_ENTS, ENTITY_DIM), dtype=np.float32)

        # Build future feature lookup: unit_id → 30-dim features
        future_feats = {}
        for j in range(MAX_ENTS):
            uid = self.ent_unit_ids[idx_future, j]
            if uid < 0:
                break
            future_feats[uid] = self.ent_feat[idx_future, j]

        # Match each entity's unit_id to its future features
        for j in range(MAX_ENTS):
            uid = self.ent_unit_ids[idx_now, j]
            if uid < 0:
                break
            if uid in future_feats:
                target_feat[j] = future_feats[uid]
            # else: unit died or left — target stays zeros (hp=0, exists=0)

        return target_feat, delta / max_delta

    def sample_batch(
        self, batch_size: int, min_delta: int = 5, max_delta: int = 40,
    ) -> dict[str, torch.Tensor] | None:
        pairs = []
        attempts = 0
        while len(pairs) < batch_size and attempts < batch_size * 3:
            attempts += 1
            sc_id, pos = random.choice(self._flat_index)
            delta = random.randint(min_delta, max_delta)
            future_idx = self._find_future(sc_id, pos, delta)
            if future_idx is not None:
                now_idx = self._scenario_groups[sc_id][pos]
                actual_delta = int(self.ticks[future_idx] - self.ticks[now_idx])
                pairs.append((now_idx, future_idx, actual_delta))

        if len(pairs) < 2:
            return None

        return self._collate(pairs, max_delta)

    def _collate(self, pairs: list[tuple[int, int, int]], max_delta: int):
        B = len(pairs)
        MAX_ENTS = self.ent_feat.shape[1]

        now_indices = np.array([p[0] for p in pairs])

        target_feat = np.zeros((B, MAX_ENTS, ENTITY_DIM), dtype=np.float32)
        deltas = np.zeros(B, dtype=np.float32)

        for i, (now_idx, future_idx, delta) in enumerate(pairs):
            feat, d_norm = self._build_targets(now_idx, future_idx, max_delta, delta)
            target_feat[i] = feat
            deltas[i] = d_norm

        return {
            "entity_features": torch.tensor(
                self.ent_feat[now_indices], dtype=torch.float, device=DEVICE),
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
            "target_features": torch.tensor(target_feat, dtype=torch.float, device=DEVICE),
            "delta_normalized": torch.tensor(deltas, device=DEVICE),
        }

    def iter_batches(
        self, batch_size: int, min_delta: int = 5, max_delta: int = 40,
    ):
        indices = list(range(len(self._flat_index)))
        random.shuffle(indices)

        pairs = []
        for idx in indices:
            sc_id, pos = self._flat_index[idx]
            delta = random.randint(min_delta, max_delta)
            future_idx = self._find_future(sc_id, pos, delta)
            if future_idx is None:
                continue
            now_idx = self._scenario_groups[sc_id][pos]
            actual_delta = int(self.ticks[future_idx] - self.ticks[now_idx])
            pairs.append((now_idx, future_idx, actual_delta))

            if len(pairs) >= batch_size:
                yield self._collate(pairs, max_delta)
                pairs = []

        if pairs:
            yield self._collate(pairs, max_delta)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = NextStateDataset(Path(args.dataset), max_samples=args.max_samples)
    train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    model = EntityEncoderNextState(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    print(f"Model: {n_params:,} total, encoder: {n_encoder:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10,
    )

    batch_size = args.batch_size
    gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    log_path = Path(args.output).with_suffix(".csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    # Feature names for per-feature MAE reporting
    FEATURE_GROUPS = {
        "hp": [0], "shield": [1], "resource": [2],
        "pos": [5, 6], "dist": [7],
        "combat": [12, 13, 14],
        "ability_cd": [17], "heal_cd": [20], "cc_cd": [23],
        "state": [24, 25, 26, 27],
        "exists": [29],
    }

    log_writer.writerow([
        "step", "train_loss", "val_mse", "val_mae",
        "val_hp_mae", "val_pos_mae", "val_cd_mae", "val_exists_mae",
        "baseline_mae", "weight_norm", "grad_norm", "lr", "elapsed_s",
    ])

    best_metric = float("inf")  # full state MAE, lower is better
    start_time = time.time()
    model.train()

    # Sliding window: start with short deltas, expand to full range over warmup period
    delta_warmup_steps = args.max_steps // 4  # first 25% of training
    def get_delta_range(step: int) -> tuple[int, int]:
        if step >= delta_warmup_steps:
            return args.min_delta, args.max_delta
        progress = step / delta_warmup_steps
        current_max = args.min_delta + int((args.max_delta - args.min_delta) * progress)
        return args.min_delta, max(current_max, args.min_delta + 1)

    beta = args.beta_nll

    print(f"\nStarting next-state prediction pre-training: max_steps={args.max_steps}")
    print(f"Predicting full 30-dim entity state at T+delta (symlog + beta-NLL)")
    print(f"Delta range: [{args.min_delta}, {args.max_delta}] ticks "
          f"(sliding warmup over {delta_warmup_steps} steps)")
    print(f"Beta-NLL beta={beta}")
    print(f"Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Grokfast EMA: alpha={args.grokfast_alpha}, lamb={args.grokfast_lamb}")
    print(f"Device: {DEVICE}\n")

    for step in range(1, args.max_steps + 1):
        cur_min, cur_max = get_delta_range(step)
        batch = train_ds.sample_batch(
            batch_size, min_delta=cur_min, max_delta=cur_max,
        )
        if batch is None:
            print("WARNING: Failed to sample batch, skipping step")
            continue

        mean, log_var = model(
            batch["entity_features"], batch["entity_type_ids"],
            batch["threat_features"], batch["entity_mask"], batch["threat_mask"],
            batch["delta_normalized"],
            batch["position_features"], batch["position_mask"],
        )

        # Symlog targets
        target_symlog = symlog(batch["target_features"])

        # Mask: only compute loss on real entities (not padding)
        valid = ~batch["entity_mask"]  # (B, E) True = real entity
        valid_3d = valid.unsqueeze(-1).expand_as(mean)  # (B, E, 30)

        loss = beta_nll_loss(
            mean[valid_3d], log_var[valid_3d], target_symlog[valid_3d], beta=beta,
        )

        optimizer.zero_grad()
        loss.backward()
        gf.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step <= 10:
            warmup.step()

        if step % args.eval_every == 0:
            model.eval()

            all_errors = []  # (N_valid, 30)
            all_baseline_errors = []  # naive: predict current state = future state
            all_mean_var = []  # track learned variance

            for vb in val_ds.iter_batches(
                batch_size, min_delta=args.min_delta, max_delta=args.max_delta,
            ):
                with torch.no_grad():
                    vmean, vlog_var = model(
                        vb["entity_features"], vb["entity_type_ids"],
                        vb["threat_features"], vb["entity_mask"], vb["threat_mask"],
                        vb["delta_normalized"],
                        vb["position_features"], vb["position_mask"],
                    )
                    # Convert predictions back from symlog to original space
                    vpred = symexp(vmean)
                    vtarget_symlog = symlog(vb["target_features"])

                valid = ~vb["entity_mask"]  # (B, E)
                # Per-entity errors in original space for interpretable MAE
                for i in range(valid.shape[0]):
                    for j in range(valid.shape[1]):
                        if valid[i, j]:
                            err = (vpred[i, j] - vb["target_features"][i, j]).abs()
                            all_errors.append(err)
                            baseline_err = (vb["entity_features"][i, j] - vb["target_features"][i, j]).abs()
                            all_baseline_errors.append(baseline_err)
                            all_mean_var.append(vlog_var[i, j].exp())

            if not all_errors:
                model.train()
                continue

            errors = torch.stack(all_errors)  # (N, 30)
            baseline = torch.stack(all_baseline_errors)  # (N, 30)
            mean_var = torch.stack(all_mean_var).mean(dim=0)  # (30,) avg variance per feature

            val_mae = errors.mean().item()
            baseline_mae = baseline.mean().item()

            # Per-group MAE
            hp_mae = errors[:, FEATURE_GROUPS["hp"]].mean().item()
            pos_mae = errors[:, FEATURE_GROUPS["pos"]].mean().item()
            cd_indices = FEATURE_GROUPS["ability_cd"] + FEATURE_GROUPS["heal_cd"] + FEATURE_GROUPS["cc_cd"]
            cd_mae = errors[:, cd_indices].mean().item()
            exists_mae = errors[:, FEATURE_GROUPS["exists"]].mean().item()

            # Learned variance summary: show which features the model thinks are noisy
            hp_var = mean_var[FEATURE_GROUPS["hp"]].mean().item()
            pos_var = mean_var[FEATURE_GROUPS["pos"]].mean().item()
            cd_var = mean_var[cd_indices].mean().item()

            weight_norm = sum(
                p.data.norm().item() ** 2 for p in model.parameters()
            ) ** 0.5

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            log_writer.writerow([
                step, f"{loss.item():.6f}", f"{loss.item():.6f}",
                f"{val_mae:.4f}", f"{hp_mae:.4f}", f"{pos_mae:.4f}",
                f"{cd_mae:.4f}", f"{exists_mae:.4f}",
                f"{baseline_mae:.4f}", f"{weight_norm:.4f}",
                f"{grad_norm:.4f}", f"{lr:.6f}", f"{elapsed:.1f}",
            ])
            log_file.flush()

            marker = ""
            if val_mae < best_metric:
                best_metric = val_mae
                torch.save(model.state_dict(), args.output)
                marker = " *"

            improvement = (1 - val_mae / baseline_mae) * 100 if baseline_mae > 0 else 0
            print(
                f"step {step:>7d} | "
                f"loss {loss.item():.4f} | "
                f"val_mae {val_mae:.4f} (base {baseline_mae:.4f}, {improvement:+.1f}%) | "
                f"hp {hp_mae:.4f} pos {pos_mae:.4f} cd {cd_mae:.4f} | "
                f"var hp={hp_var:.3f} pos={pos_var:.3f} cd={cd_var:.3f} | "
                f"delta [{cur_min},{cur_max}] | "
                f"w {weight_norm:.1f}"
                f"{marker}"
            )

            model.train()

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed:.0f}s")
    print(f"Best val state MAE: {best_metric:.4f}")
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {log_path}")


def main():
    p = argparse.ArgumentParser(
        description="Pre-train entity encoder V3 on next-state prediction"
    )
    p.add_argument("dataset", help=".npz next-state dataset (or .jsonl, auto-detects .npz)")
    p.add_argument("-o", "--output", default="generated/entity_encoder_nextstate.pt")

    p.add_argument("--max-samples", type=int, default=0,
                    help="Max samples to load (0 = all)")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--min-delta", type=int, default=5,
                    help="Min prediction horizon in ticks")
    p.add_argument("--max-delta", type=int, default=40,
                    help="Max prediction horizon in ticks")

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)

    p.add_argument("--beta-nll", type=float, default=0.5,
                    help="Beta for beta-NLL loss (0.5 recommended)")

    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
