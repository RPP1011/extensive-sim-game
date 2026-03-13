#!/usr/bin/env python3
"""IMPALA training loop for V4 dual-head actor-critic.

Iterates:
1. Generate episodes using Rust sim actors (current policy weights)
2. Train on collected episodes with V-trace off-policy correction
3. Export updated weights for next iteration
4. Evaluate periodically

Usage:
    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/impala_learner.py \
        --scenarios scenarios/hvh \
        --checkpoint generated/actor_critic_v4_full_unfrozen.pt \
        --output-dir generated/impala \
        --embedding-registry generated/ability_embedding_registry.json \
        --external-cls-dim 128 \
        --threads 32 --episodes-per-scenario 2 \
        --iters 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (
    AbilityActorCriticV4,
    NUM_MOVE_DIRS,
    NUM_COMBAT_TYPES,
    MAX_ABILITIES,
    POSITION_DIM,
)
from tokenizer import AbilityTokenizer
from grokfast import GrokfastEMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENTITY_DIM = 30
THREAT_DIM = 8

SHM_NAME = "impala_inf"
SHM_PATH_PREFIX = "/dev/shm/"


# ---------------------------------------------------------------------------
# V-trace
# ---------------------------------------------------------------------------


def vtrace_targets(
    log_rhos: np.ndarray,
    discounts: np.ndarray,
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_value: float = 0.0,
    clip_rho: float = 1.0,
    clip_c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute V-trace targets and advantages for a single trajectory.

    Args:
        log_rhos: log importance ratios log(π/μ), shape (T,)
        discounts: per-step discount factor γ, shape (T,)
        rewards: per-step rewards, shape (T,)
        values: V(s_t) from value head, shape (T,)
        bootstrap_value: V(s_T) for non-terminal episodes
        clip_rho: clipping threshold for ρ̄
        clip_c: clipping threshold for c̄

    Returns:
        vs: V-trace value targets, shape (T,)
        advantages: policy gradient advantages, shape (T,)
    """
    T = len(log_rhos)
    rhos = np.exp(np.clip(log_rhos, -20, 20))
    rho_bar = np.minimum(rhos, clip_rho)
    c_bar = np.minimum(rhos, clip_c)

    values_tp1 = np.append(values[1:], bootstrap_value)
    deltas = rho_bar * (rewards + discounts * values_tp1 - values)

    # Backward accumulation
    vs_minus_v = np.zeros(T)
    acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = deltas[t] + discounts[t] * c_bar[t] * acc
        vs_minus_v[t] = acc

    vs = vs_minus_v + values
    # Policy gradient advantage: ρ̄_t * (r_t + γ v_{t+1} - V(s_t))
    vs_tp1 = np.append(vs[1:], bootstrap_value)
    advantages = rho_bar * (rewards + discounts * vs_tp1 - values)

    return vs, advantages


# ---------------------------------------------------------------------------
# Episode loading and trajectory extraction
# ---------------------------------------------------------------------------


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def extract_trajectories(episodes: list[dict]) -> list[dict]:
    """Extract per-unit trajectories from episodes.

    Each trajectory is a sequence of steps from one unit within one episode,
    ordered by tick. V-trace operates on these sequential trajectories.
    """
    trajectories = []
    for ep_idx, ep in enumerate(episodes):
        unit_steps: dict[int, list[dict]] = defaultdict(list)
        for step in ep["steps"]:
            if step.get("move_dir") is None:
                continue
            unit_steps[step["unit_id"]].append(step)

        for uid, steps in unit_steps.items():
            steps.sort(key=lambda s: s["tick"])
            trajectories.append({
                "steps": steps,
                "unit_id": uid,
                "episode_reward": ep["reward"],
                "outcome": ep.get("outcome", ""),
                "episode_idx": ep_idx,
                "unit_abilities": ep.get("unit_abilities", {}),
                "unit_ability_names": ep.get("unit_ability_names", {}),
            })
    return trajectories


# ---------------------------------------------------------------------------
# Pre-tensorized data — convert all steps to GPU tensors once
# ---------------------------------------------------------------------------


class PreTensorizedData:
    """All episode steps pre-converted to GPU tensors for fast batched access."""

    def __init__(
        self,
        all_steps: list[dict],
        embedding_registry: dict | None,
        cls_dim: int,
    ):
        N = len(all_steps)
        self.N = N

        # Find max dimensions across all steps
        max_ents = max(len(s["entities"]) for s in all_steps)
        max_threats = max((len(s.get("threats", [])) for s in all_steps), default=1)
        max_threats = max(max_threats, 1)
        max_positions = max((len(s.get("positions", [])) for s in all_steps), default=1)
        max_positions = max(max_positions, 1)

        # Pre-allocate numpy arrays (much faster than per-element torch.tensor)
        ent_feat_np = np.zeros((N, max_ents, ENTITY_DIM), dtype=np.float32)
        ent_types_np = np.zeros((N, max_ents), dtype=np.int64)
        ent_mask_np = np.ones((N, max_ents), dtype=np.bool_)
        thr_feat_np = np.zeros((N, max_threats, THREAT_DIM), dtype=np.float32)
        thr_mask_np = np.ones((N, max_threats), dtype=np.bool_)
        pos_feat_np = np.zeros((N, max_positions, POSITION_DIM), dtype=np.float32)
        pos_mask_np = np.ones((N, max_positions), dtype=np.bool_)

        move_dirs_np = np.zeros(N, dtype=np.int64)
        combat_types_np = np.zeros(N, dtype=np.int64)
        target_indices_np = np.zeros(N, dtype=np.int64)
        combat_masks_np = np.zeros((N, NUM_COMBAT_TYPES), dtype=np.bool_)
        behav_lp_np = np.zeros(N, dtype=np.float32)

        # Ability CLS: pre-allocate numpy arrays
        ability_cls_np = np.zeros((MAX_ABILITIES, N, cls_dim), dtype=np.float32) if embedding_registry else None
        ability_cls_has = [False] * MAX_ABILITIES
        embs = embedding_registry["embeddings"] if embedding_registry else {}
        # Pre-convert embedding tensors to numpy once
        embs_np = {}
        if embedding_registry:
            for k, v in embs.items():
                if isinstance(v, torch.Tensor):
                    embs_np[k] = v.cpu().numpy()
                else:
                    embs_np[k] = np.array(v, dtype=np.float32)

        for i, s in enumerate(all_steps):
            # Entities — direct numpy array assignment
            ents = s["entities"]
            n_e = len(ents)
            ent_feat_np[i, :n_e] = ents
            ent_types_np[i, :n_e] = s["entity_types"]
            ent_mask_np[i, :n_e] = False

            # Threats
            threats = s.get("threats")
            if threats:
                n_t = len(threats)
                thr_feat_np[i, :n_t] = threats
                thr_mask_np[i, :n_t] = False

            # Positions
            positions = s.get("positions")
            if positions:
                n_p = len(positions)
                pos_feat_np[i, :n_p] = positions
                pos_mask_np[i, :n_p] = False

            # Actions
            move_dirs_np[i] = s["move_dir"]
            combat_types_np[i] = s["combat_type"]
            target_indices_np[i] = min(s.get("target_idx", 0), max_ents - 1)

            # Combat masks
            mask = s.get("mask", [])
            combat_masks_np[i, 0] = any(mask[j] for j in range(min(3, len(mask))))
            combat_masks_np[i, 1] = True
            n_mask = len(mask)
            for ab_idx in range(min(MAX_ABILITIES, n_mask - 3)):
                combat_masks_np[i, 2 + ab_idx] = mask[3 + ab_idx]

            # Ability CLS embeddings
            if embedding_registry:
                names = s.get("_ability_names", [])
                for ab_idx in range(min(MAX_ABILITIES, len(names))):
                    if names[ab_idx]:
                        lookup = names[ab_idx].replace(" ", "_")
                        if lookup in embs_np:
                            ability_cls_np[ab_idx, i] = embs_np[lookup]
                            ability_cls_has[ab_idx] = True

            # Behavior log probs (for KL penalty)
            behav_lp_np[i] = ((s.get("lp_move") or 0.0)
                              + (s.get("lp_combat") or 0.0)
                              + (s.get("lp_pointer") or 0.0))

        # Single bulk transfer: numpy → torch → GPU
        ent_feat = torch.from_numpy(ent_feat_np)
        ent_types = torch.from_numpy(ent_types_np)
        ent_mask = torch.from_numpy(ent_mask_np)
        thr_feat = torch.from_numpy(thr_feat_np)
        thr_mask = torch.from_numpy(thr_mask_np)
        pos_feat = torch.from_numpy(pos_feat_np)
        pos_mask = torch.from_numpy(pos_mask_np)
        move_dirs = torch.from_numpy(move_dirs_np)
        combat_types = torch.from_numpy(combat_types_np)
        target_indices = torch.from_numpy(target_indices_np)
        combat_masks = torch.from_numpy(combat_masks_np)
        behav_lps = torch.from_numpy(behav_lp_np)

        # Keep on CPU (pinned memory for fast async transfer), move per-batch to GPU
        self.ent_feat = ent_feat.pin_memory()
        self.ent_types = ent_types.pin_memory()
        self.ent_mask = ent_mask.pin_memory()
        self.thr_feat = thr_feat.pin_memory()
        self.thr_mask = thr_mask.pin_memory()
        self.pos_feat = pos_feat.pin_memory()
        self.pos_mask = pos_mask.pin_memory()
        self.move_dirs = move_dirs.pin_memory()
        self.combat_types = combat_types.pin_memory()
        self.target_indices = target_indices.pin_memory()
        self.combat_masks = combat_masks.pin_memory()
        self.behav_lps = behav_lps.pin_memory()
        self.max_ents = max_ents

        # Ability CLS: list of (N, cls_dim) or None per slot — kept on CPU
        self.ability_cls: list[torch.Tensor | None] = [None] * MAX_ABILITIES
        if ability_cls_np is not None:
            for ab_idx in range(MAX_ABILITIES):
                if ability_cls_has[ab_idx]:
                    self.ability_cls[ab_idx] = torch.from_numpy(ability_cls_np[ab_idx]).pin_memory()

        print(f"    Pre-tensorized {N} steps: {max_ents} ents, {max_threats} threats, "
              f"{max_positions} positions, CPU mem ~{self.ent_feat.nelement()*4/1e6:.0f}MB", flush=True)

    def get_batch(self, idx: torch.Tensor | np.ndarray) -> tuple[dict, list, torch.Tensor]:
        """Get pre-tensorized batch by index. Returns (state_dict, ability_cls, combat_masks).
        Data lives on CPU (pinned); each batch is transferred to GPU on demand."""
        if isinstance(idx, torch.Tensor):
            idx_cpu = idx.cpu()
        elif isinstance(idx, np.ndarray):
            idx_cpu = torch.from_numpy(idx).long()
        else:
            idx_cpu = idx
        state = {
            "entity_features": self.ent_feat[idx_cpu].to(DEVICE, non_blocking=True),
            "entity_type_ids": self.ent_types[idx_cpu].to(DEVICE, non_blocking=True),
            "threat_features": self.thr_feat[idx_cpu].to(DEVICE, non_blocking=True),
            "entity_mask": self.ent_mask[idx_cpu].to(DEVICE, non_blocking=True),
            "threat_mask": self.thr_mask[idx_cpu].to(DEVICE, non_blocking=True),
            "position_features": self.pos_feat[idx_cpu].to(DEVICE, non_blocking=True),
            "position_mask": self.pos_mask[idx_cpu].to(DEVICE, non_blocking=True),
        }
        ab_cls = [t[idx_cpu].to(DEVICE, non_blocking=True) if t is not None else None for t in self.ability_cls]
        return state, ab_cls, self.combat_masks[idx_cpu].to(DEVICE, non_blocking=True)


# Legacy collation functions for backward compat
def collate_states(steps: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-length game states into padded tensors."""
    B = len(steps)
    max_ents = max(len(s["entities"]) for s in steps)
    max_threats = max(
        (len(s.get("threats", [])) for s in steps), default=1)
    max_threats = max(max_threats, 1)
    max_positions = max(
        (len(s.get("positions", [])) for s in steps), default=1)
    max_positions = max(max_positions, 1)

    ent_feat = torch.zeros(B, max_ents, ENTITY_DIM, device=DEVICE)
    ent_types = torch.zeros(B, max_ents, dtype=torch.long, device=DEVICE)
    ent_mask = torch.ones(B, max_ents, dtype=torch.bool, device=DEVICE)
    thr_feat = torch.zeros(B, max_threats, THREAT_DIM, device=DEVICE)
    thr_mask = torch.ones(B, max_threats, dtype=torch.bool, device=DEVICE)
    pos_feat = torch.zeros(B, max_positions, POSITION_DIM, device=DEVICE)
    pos_mask = torch.ones(B, max_positions, dtype=torch.bool, device=DEVICE)

    for i, s in enumerate(steps):
        n_e = len(s["entities"])
        ent_feat[i, :n_e] = torch.tensor(s["entities"], dtype=torch.float)
        ent_types[i, :n_e] = torch.tensor(s["entity_types"], dtype=torch.long)
        ent_mask[i, :n_e] = False
        threats = s.get("threats", [])
        if threats:
            thr_feat[i, :len(threats)] = torch.tensor(threats, dtype=torch.float)
            thr_mask[i, :len(threats)] = False
        positions = s.get("positions", [])
        if positions:
            pos_feat[i, :len(positions)] = torch.tensor(positions, dtype=torch.float)
            pos_mask[i, :len(positions)] = False

    return {
        "entity_features": ent_feat,
        "entity_type_ids": ent_types,
        "threat_features": thr_feat,
        "entity_mask": ent_mask,
        "threat_mask": thr_mask,
        "position_features": pos_feat,
        "position_mask": pos_mask,
    }


def build_ability_cls(
    steps: list[dict],
    embedding_registry: dict | None,
    cls_dim: int,
) -> list[torch.Tensor | None]:
    """Build per-ability CLS batch from embedding registry."""
    B = len(steps)
    ability_cls: list[torch.Tensor | None] = [None] * MAX_ABILITIES

    if embedding_registry is None:
        return ability_cls

    embs = embedding_registry["embeddings"]
    for ab_idx in range(MAX_ABILITIES):
        vecs = []
        has_any = False
        for s in steps:
            uid_str = str(s["unit_id"])
            names = s.get("_ability_names", [])
            if ab_idx < len(names) and names[ab_idx]:
                lookup = names[ab_idx].replace(" ", "_")
                if lookup in embs:
                    vecs.append(embs[lookup])
                    has_any = True
                else:
                    vecs.append(torch.zeros(cls_dim, device=DEVICE))
            else:
                vecs.append(torch.zeros(cls_dim, device=DEVICE))
        if has_any:
            ability_cls[ab_idx] = torch.stack(vecs)

    return ability_cls


def build_combat_masks(steps: list[dict]) -> torch.Tensor:
    """Build combat type masks [B, 10]."""
    B = len(steps)
    masks = torch.zeros(B, NUM_COMBAT_TYPES, dtype=torch.bool, device=DEVICE)
    for i, s in enumerate(steps):
        mask = s.get("mask", [])
        masks[i, 0] = any(mask[j] for j in range(min(3, len(mask))))
        masks[i, 1] = True
        for ab_idx in range(MAX_ABILITIES):
            if 3 + ab_idx < len(mask):
                masks[i, 2 + ab_idx] = mask[3 + ab_idx]
    return masks


# ---------------------------------------------------------------------------
# Compute current policy log probs and values
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_policy_values(
    model: AbilityActorCriticV4,
    ptd: PreTensorizedData,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass on all steps to get current log probs + values.

    Returns:
        lp_move, lp_combat, lp_pointer: (N,) arrays of log probs
        values: (N,) array of state values
        entropies: (N,) array of total entropy
    """
    N = ptd.N
    lp_move = np.zeros(N)
    lp_combat = np.zeros(N)
    lp_pointer = np.zeros(N)
    values = np.zeros(N)
    entropies = np.zeros(N)

    LOG_PROB_MIN = -10.0

    model.eval()
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = np.arange(start, end)
        B = end - start

        state, ability_cls, combat_masks = ptd.get_batch(idx)

        output, value = model(
            state["entity_features"], state["entity_type_ids"],
            state["threat_features"], state["entity_mask"], state["threat_mask"],
            ability_cls,
            state["position_features"], state["position_mask"],
        )

        # Move log probs + entropy
        move_logp = F.log_softmax(output["move_logits"], dim=-1)
        move_ent = -(move_logp.exp() * move_logp).sum(-1)

        # Combat log probs (masked) + entropy
        combat_logits = output["combat_logits"].masked_fill(~combat_masks, -1e9)
        combat_logp = F.log_softmax(combat_logits, dim=-1)
        combat_ent = -(combat_logp.exp() * combat_logp).sum(-1)

        move_dirs_t = ptd.move_dirs[start:end].to(DEVICE, non_blocking=True)
        combat_types_t = ptd.combat_types[start:end].to(DEVICE, non_blocking=True)
        target_indices_t = ptd.target_indices[start:end].to(DEVICE, non_blocking=True).clamp(max=output["attack_ptr"].shape[1] - 1)

        batch_lp_move = move_logp.gather(1, move_dirs_t.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
        batch_lp_combat = combat_logp.gather(1, combat_types_t.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
        lp_move[start:end] = batch_lp_move.cpu().numpy()
        lp_combat[start:end] = batch_lp_combat.cpu().numpy()
        values[start:end] = value.squeeze(-1).cpu().numpy()

        # Pointer log probs — vectorized by combat type
        batch_lp_ptr = torch.zeros(B, device=DEVICE)
        batch_ptr_ent = torch.zeros(B, device=DEVICE)

        attack_mask = (combat_types_t == 0)
        if attack_mask.any():
            atk_logp = F.log_softmax(output["attack_ptr"], dim=-1)
            sel_lp = atk_logp.gather(1, target_indices_t.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
            batch_lp_ptr += torch.where(attack_mask, sel_lp, torch.zeros_like(sel_lp))
            batch_ptr_ent += torch.where(attack_mask, -(atk_logp.exp() * atk_logp).sum(-1), torch.zeros_like(sel_lp))

        ab_ptrs = output.get("ability_ptrs", [])
        for ab_idx, ab_ptr in enumerate(ab_ptrs):
            if ab_ptr is None:
                continue
            ab_mask = (combat_types_t == ab_idx + 2)
            if not ab_mask.any():
                continue
            ab_logp = F.log_softmax(ab_ptr, dim=-1)
            ti_clamped = target_indices_t.clamp(max=ab_logp.shape[1] - 1)
            sel_lp = ab_logp.gather(1, ti_clamped.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
            batch_lp_ptr += torch.where(ab_mask, sel_lp, torch.zeros_like(sel_lp))
            batch_ptr_ent += torch.where(ab_mask, -(ab_logp.exp() * ab_logp).sum(-1), torch.zeros_like(sel_lp))

        lp_pointer[start:end] = batch_lp_ptr.cpu().numpy()
        entropies[start:end] = (move_ent + combat_ent + batch_ptr_ent).cpu().numpy()

    return lp_move, lp_combat, lp_pointer, values, entropies


# ---------------------------------------------------------------------------
# Training step with V-trace
# ---------------------------------------------------------------------------


def flatten_trajectories(trajectories: list[dict]) -> tuple[list[dict], list[tuple[int, int]]]:
    """Flatten trajectories into a list of steps with ability names annotated."""
    all_steps = []
    step_traj_map = []
    for ti, traj in enumerate(trajectories):
        ab_names = traj.get("unit_ability_names", {})
        uid_str = str(traj["unit_id"])
        names = ab_names.get(uid_str, [])
        for si, step in enumerate(traj["steps"]):
            step["_ability_names"] = names
            all_steps.append(step)
            step_traj_map.append((ti, si))
    return all_steps, step_traj_map


def train_on_trajectories(
    model: AbilityActorCriticV4,
    optimizer: torch.optim.Optimizer,
    grokfast: GrokfastEMA | None,
    trajectories: list[dict],
    all_steps: list[dict],
    step_traj_map: list[tuple[int, int]],
    ptd: PreTensorizedData | None,
    gamma: float = 0.99,
    step_interval: int = 3,
    batch_size: int = 256,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    clip_rho: float = 1.0,
    clip_c: float = 1.0,
    max_grad_norm: float = 1.0,
    reward_scale: float = 10.0,
    ppo_clip: float = 0.2,
    kl_coeff: float = 2.0,
    max_train_steps: int = 0,
    target_entropy: float = 1.0,
    no_vtrace: bool = False,
    advantage_clip: float = 0.0,
    freeze_policy: bool = False,
) -> dict:
    """One training pass on collected trajectories using V-trace + PPO clipping."""

    if not all_steps or ptd is None:
        return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "n_steps": 0}

    N = len(all_steps)

    # 2. Forward pass (no grad) → current log probs + values
    t_fwd = time.time()
    curr_lp_move, curr_lp_combat, curr_lp_pointer, vals, ents = \
        compute_policy_values(model, ptd, batch_size)
    t_fwd = time.time() - t_fwd
    print(f"    Timing: forward={t_fwd:.1f}s", flush=True)

    # 3. Compute V-trace targets per action component
    # Instead of multiplying importance ratios (which compounds noise),
    # run V-trace 3 times with per-component ratios and average advantages.
    gamma_eff = gamma ** step_interval

    # Pre-extract behavior log probs into arrays
    behav_lp_move_arr = np.array([s.get("lp_move") or 0.0 for s in all_steps])
    behav_lp_combat_arr = np.array([s.get("lp_combat") or 0.0 for s in all_steps])
    behav_lp_pointer_arr = np.array([s.get("lp_pointer") or 0.0 for s in all_steps])

    # Build traj_indices lookup
    traj_start_indices = []
    offset = 0
    for ti, traj in enumerate(trajectories):
        T = len(traj["steps"])
        traj_start_indices.append((offset, T))
        offset += T

    # Compute advantages: V-trace (default) or simple A2C (--no-vtrace)
    all_advantages = np.zeros(N)
    all_vtrace_targets = np.zeros(N)

    for ti, traj in enumerate(trajectories):
        start_idx, T = traj_start_indices[ti]
        traj_indices = list(range(start_idx, start_idx + T))

        rewards = np.zeros(T)
        traj_vals = np.zeros(T)

        for si in range(T):
            idx = traj_indices[si]
            step = traj["steps"][si]
            rewards[si] = step.get("step_reward", 0.0) * reward_scale
            traj_vals[si] = vals[idx]

        is_terminal = traj["outcome"] in ("Victory", "Defeat")
        bootstrap = 0.0 if is_terminal else traj_vals[-1]
        discounts = np.full(T, gamma_eff)
        if is_terminal:
            discounts[-1] = 0.0

        if no_vtrace:
            # Simple A2C: Monte Carlo returns - values
            returns = np.zeros(T)
            R = bootstrap
            for t in range(T - 1, -1, -1):
                R = rewards[t] + discounts[t] * R
                returns[t] = R
            adv = returns - traj_vals
            vs = returns
        else:
            # V-trace with importance ratios
            log_rhos = np.zeros(T)
            for si in range(T):
                idx = traj_indices[si]
                lr_move = curr_lp_move[idx] - behav_lp_move_arr[idx]
                lr_combat = curr_lp_combat[idx] - behav_lp_combat_arr[idx]
                lr_pointer = curr_lp_pointer[idx] - behav_lp_pointer_arr[idx]
                log_rhos[si] = lr_move + lr_combat + lr_pointer

            vs, adv = vtrace_targets(
                log_rhos, discounts, rewards, traj_vals, bootstrap,
                clip_rho=clip_rho, clip_c=clip_c,
            )

        for si in range(T):
            idx = traj_indices[si]
            all_advantages[idx] = adv[si]
            all_vtrace_targets[idx] = vs[si]

    # Diagnostics
    lr_move_all = curr_lp_move - behav_lp_move_arr
    lr_combat_all = curr_lp_combat - behav_lp_combat_arr
    lr_ptr_all = curr_lp_pointer - behav_lp_pointer_arr
    lr_mc = lr_move_all + lr_combat_all
    print(f"    log_rho(m+c): mean={lr_mc.mean():.2f} std={lr_mc.std():.2f} | "
          f"ptr(excluded): mean={lr_ptr_all.mean():.2f} std={lr_ptr_all.std():.2f} | "
          f"values min={vals.min():.3f} max={vals.max():.3f}", flush=True)

    # Global advantage normalization
    adv_std = all_advantages.std()
    if adv_std > 1e-8:
        all_advantages = (all_advantages - all_advantages.mean()) / adv_std

    # Advantage clipping (E6)
    if advantage_clip > 0:
        all_advantages = np.clip(all_advantages, -advantage_clip, advantage_clip)

    adv_tensor = torch.tensor(all_advantages, dtype=torch.float32, device=DEVICE)

    vtarget_tensor = torch.tensor(all_vtrace_targets, dtype=torch.float32, device=DEVICE)
    print(f"    V-trace targets: mean={all_vtrace_targets.mean():.4f} std={all_vtrace_targets.std():.4f}", flush=True)

    # 4. Training pass with gradients
    model.train()
    indices = np.random.permutation(N)
    if max_train_steps > 0:
        max_samples = max_train_steps * batch_size
        if max_samples < N:
            indices = indices[:max_samples]
            N_eff = max_samples
        else:
            N_eff = N
    else:
        N_eff = N

    LOG_PROB_MIN = -10.0  # clamp log probs to prevent -inf from masked pointers

    metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "kl_div": 0.0, "n_steps": 0,
    }

    for start in range(0, N_eff, batch_size):
        end = min(start + batch_size, N_eff)
        idx = indices[start:end]
        B = len(idx)

        state, ability_cls, combat_masks = ptd.get_batch(idx)

        output, value = model(
            state["entity_features"], state["entity_type_ids"],
            state["threat_features"], state["entity_mask"], state["threat_mask"],
            ability_cls,
            state["position_features"], state["position_mask"],
        )

        # Compute log probs for taken actions
        move_logp = F.log_softmax(output["move_logits"], dim=-1)
        combat_logits_masked = output["combat_logits"].masked_fill(~combat_masks, -1e9)
        combat_logp = F.log_softmax(combat_logits_masked, dim=-1)

        batch_lp = torch.zeros(B, device=DEVICE)
        batch_ent = torch.zeros(B, device=DEVICE)

        # Move component (CPU → GPU)
        move_dirs = ptd.move_dirs[idx].to(DEVICE, non_blocking=True)
        batch_lp += move_logp.gather(1, move_dirs.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
        batch_ent += -(move_logp.exp() * move_logp).sum(-1)

        # Combat component (CPU → GPU)
        combat_types = ptd.combat_types[idx].to(DEVICE, non_blocking=True)
        batch_lp += combat_logp.gather(1, combat_types.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
        batch_ent += -(combat_logp.exp() * combat_logp).sum(-1)

        # Pointer component — vectorized by combat type (CPU → GPU)
        target_indices = ptd.target_indices[idx].to(DEVICE, non_blocking=True).clamp(max=output["attack_ptr"].shape[1] - 1)

        # Attack pointers (ct == 0)
        attack_mask = (combat_types == 0)
        if attack_mask.any():
            atk_logp = F.log_softmax(output["attack_ptr"], dim=-1)
            sel_lp = atk_logp.gather(1, target_indices.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
            batch_lp += torch.where(attack_mask, sel_lp, torch.zeros_like(sel_lp))
            atk_ent = -(atk_logp.exp() * atk_logp).sum(-1)
            batch_ent += torch.where(attack_mask, atk_ent, torch.zeros_like(atk_ent))

        # Ability pointers (ct >= 2)
        ab_ptrs = output.get("ability_ptrs", [])
        for ab_idx, ab_ptr in enumerate(ab_ptrs):
            if ab_ptr is None:
                continue
            ab_mask = (combat_types == ab_idx + 2)
            if not ab_mask.any():
                continue
            ab_logp = F.log_softmax(ab_ptr, dim=-1)
            ti_clamped = target_indices.clamp(max=ab_logp.shape[1] - 1)
            sel_lp = ab_logp.gather(1, ti_clamped.unsqueeze(1)).squeeze(1).clamp(min=LOG_PROB_MIN)
            batch_lp += torch.where(ab_mask, sel_lp, torch.zeros_like(sel_lp))
            ab_ent = -(ab_logp.exp() * ab_logp).sum(-1)
            batch_ent += torch.where(ab_mask, ab_ent, torch.zeros_like(ab_ent))

        # V-trace losses with PPO-style clipping
        batch_adv = adv_tensor[idx]
        batch_vtarget = vtarget_tensor[idx]

        # V-trace policy gradient
        behav_lps_batch = ptd.behav_lps[idx].to(DEVICE, non_blocking=True)
        policy_loss = -(batch_lp * batch_adv.detach()).mean()

        value_loss = 0.5 * (value.squeeze(-1) - batch_vtarget.detach()).pow(2).mean()
        entropy_loss = -entropy_coeff * batch_ent.mean()

        # Optional KL penalty
        kl_div = (behav_lps_batch - batch_lp).abs().mean()
        kl_penalty = kl_coeff * kl_div

        if freeze_policy:
            # Value-head warmup: only value loss contributes gradients
            loss = value_coeff * value_loss
        else:
            loss = policy_loss + value_coeff * value_loss + entropy_loss + kl_penalty

        optimizer.zero_grad()
        loss.backward()
        if grokfast is not None:
            grokfast.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        metrics["policy_loss"] += policy_loss.item() * B
        metrics["value_loss"] += value_loss.item() * B
        metrics["entropy"] += batch_ent.mean().item() * B
        metrics["kl_div"] += kl_div.item() * B
        metrics["n_steps"] += B

    ns = max(metrics["n_steps"], 1)
    metrics["policy_loss"] /= ns
    metrics["value_loss"] /= ns
    metrics["entropy"] /= ns
    metrics["kl_div"] /= ns
    metrics["grad_steps"] = (N_eff + batch_size - 1) // batch_size
    metrics["mean_advantage"] = float(all_advantages.mean())
    metrics["std_advantage"] = float(all_advantages.std())
    metrics["mean_reward"] = float(np.mean([
        s.get("step_reward", 0.0) for s in all_steps]))

    return metrics


# ---------------------------------------------------------------------------
# GPU server management (shared memory)
# ---------------------------------------------------------------------------

# SHM header offsets — must match gpu_inference_server.py and gpu_client.rs
OFF_RELOAD_PATH = 0x80
RELOAD_PATH_LEN = 256
OFF_RELOAD_REQ = 0x180
OFF_RELOAD_ACK = 0x184


def start_gpu_server(
    checkpoint_path: str,
    shm_name: str = SHM_NAME,
    max_batch_size: int = 1024,
    d_model: int = 32,
    d_ff: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    entity_encoder_layers: int = 4,
    external_cls_dim: int = 0,
    temperature: float = 1.0,
) -> subprocess.Popen:
    """Start GPU inference server (shared memory) as subprocess."""
    cmd = [
        sys.executable, "training/gpu_inference_server.py",
        "--weights", checkpoint_path,
        "--shm-name", shm_name,
        "--max-batch-size", str(max_batch_size),
        "--d-model", str(d_model),
        "--d-ff", str(d_ff),
        "--n-layers", str(n_layers),
        "--n-heads", str(n_heads),
        "--entity-encoder-layers", str(entity_encoder_layers),
        "--temperature", str(temperature),
    ]
    if external_cls_dim > 0:
        cmd.extend(["--external-cls-dim", str(external_cls_dim)])

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    # Wait for server to be ready
    ready = False
    for line in proc.stdout:
        line = line.rstrip()
        print(f"  [gpu] {line}", flush=True)
        if "Ready" in line:
            ready = True
            break

    if not ready:
        raise RuntimeError("GPU server failed to start")

    # Drain remaining stdout in background
    def drain():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  [gpu] {line}", flush=True)

    t = threading.Thread(target=drain, daemon=True)
    t.start()

    return proc


def reload_gpu_server(checkpoint_path: str, shm_name: str = SHM_NAME):
    """Request weight reload via shared memory flags."""
    import mmap as mmap_mod

    abs_path = str(Path(checkpoint_path).resolve())
    shm_path = f"{SHM_PATH_PREFIX}{shm_name}"

    fd = os.open(shm_path, os.O_RDWR)
    mm = mmap_mod.mmap(fd, 0)
    os.close(fd)

    # Write reload path (null-terminated)
    path_bytes = abs_path.encode('utf-8')[:RELOAD_PATH_LEN - 1] + b'\x00'
    mm[OFF_RELOAD_PATH:OFF_RELOAD_PATH + len(path_bytes)] = path_bytes

    # Clear ack, then set request
    struct.pack_into('<I', mm, OFF_RELOAD_ACK, 0)
    struct.pack_into('<I', mm, OFF_RELOAD_REQ, 1)

    # Poll for ack
    for _ in range(100_000):
        ack = struct.unpack_from('<I', mm, OFF_RELOAD_ACK)[0]
        if ack != 0:
            mm.close()
            return
        time.sleep(50e-6)

    mm.close()
    print("  WARNING: reload timed out", flush=True)


# ---------------------------------------------------------------------------
# Episode generation (Rust subprocess)
# ---------------------------------------------------------------------------


def generate_episodes(
    scenario_dirs: list[str] | str,
    weights_path: str,
    output_path: str,
    episodes_per_scenario: int,
    threads: int,
    temperature: float,
    step_interval: int,
    embedding_registry: str | None = None,
    enemy_weights: str | None = None,
    enemy_registry: str | None = None,
    gpu_shm: str | None = None,
    sims_per_thread: int = 1,
) -> tuple[float, str]:
    """Call Rust to generate episodes. Returns (elapsed_seconds, stderr_output)."""
    if isinstance(scenario_dirs, str):
        scenario_dirs = [scenario_dirs]
    cmd = [
        "cargo", "run", "--release", "--bin", "xtask", "--",
        "scenario", "oracle", "transformer-rl", "generate",
        *scenario_dirs,
        "--episodes", str(episodes_per_scenario),
        "-j", str(threads),
        "--temperature", str(temperature),
        "--step-interval", str(step_interval),
        "-o", output_path,
    ]
    if sims_per_thread > 1:
        cmd.extend(["--sims-per-thread", str(sims_per_thread)])
    if gpu_shm:
        cmd.extend(["--gpu-shm", gpu_shm])
    else:
        cmd.extend(["--weights", weights_path])
    if embedding_registry:
        cmd.extend(["--embedding-registry", embedding_registry])
    if enemy_weights:
        cmd.extend(["--enemy-weights", enemy_weights])
    if enemy_registry:
        cmd.extend(["--enemy-registry", enemy_registry])

    t0 = time.time()
    # Stream stderr live so user sees progress, capture it for parsing
    stderr_lines = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, bufsize=1,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    for line in proc.stderr:
        line = line.rstrip()
        # Filter noisy per-scenario manifest loads and compiler warnings
        if (not line.strip()
                or "Loaded ability manifest" in line
                or line.startswith("warning:")
                or line.lstrip().startswith("-->")
                or line.lstrip().startswith("|")
                or line.lstrip().startswith("=")
                or (line.strip() and line.strip()[0].isdigit() and "|" in line)):
            stderr_lines.append(line)
            continue
        print(f"    [gen] {line}", flush=True)
        stderr_lines.append(line)
    proc.wait()
    elapsed = time.time() - t0

    stderr_text = "\n".join(stderr_lines)
    if proc.returncode != 0:
        print(f"  ERROR: generate failed (rc={proc.returncode})")
        return elapsed, stderr_text

    return elapsed, stderr_text


def export_weights(
    checkpoint_path: str,
    output_path: str,
    d_model: int = 32,
    d_ff: int = 64,
    n_layers: int = 4,
    n_heads: int = 4,
    entity_encoder_layers: int = 4,
    external_cls_dim: int = 0,
) -> None:
    """Export PyTorch checkpoint to JSON for Rust inference."""
    cmd = [
        sys.executable, "training/export_actor_critic_v4.py",
        checkpoint_path,
        "-o", output_path,
        "--d-model", str(d_model),
        "--d-ff", str(d_ff),
        "--n-layers", str(n_layers),
        "--n-heads", str(n_heads),
        "--entity-encoder-layers", str(entity_encoder_layers),
    ]
    if external_cls_dim > 0:
        cmd.extend(["--external-cls-dim", str(external_cls_dim)])

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if result.returncode != 0:
        print(f"  Export failed: {result.stderr[-300:]}", flush=True)
    else:
        print(f"  {result.stdout.strip()}", flush=True)


# ---------------------------------------------------------------------------
# Main IMPALA loop
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="IMPALA V4 training loop")
    p.add_argument("--scenarios", required=True, nargs="+",
                   help="Scenario directory(ies) (e.g. scenarios/hvh dataset/scenarios)")
    p.add_argument("--checkpoint", required=True,
                   help="Initial V4 checkpoint (.pt)")
    p.add_argument("--output-dir", default="generated/impala",
                   help="Output directory for checkpoints and logs")

    # Model architecture
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--external-cls-dim", type=int, default=0)

    # Episode generation
    p.add_argument("--embedding-registry",
                   help="Pre-computed CLS embedding registry JSON")
    p.add_argument("--episodes-per-scenario", type=int, default=2)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--step-interval", type=int, default=3)
    p.add_argument("--sims-per-thread", type=int, default=64,
                   help="Concurrent sims per thread for GPU pipelining")

    # Training
    p.add_argument("--iters", type=int, default=100,
                   help="Number of generate→train iterations")
    p.add_argument("--train-epochs", type=int, default=1,
                   help="Training epochs per iteration on collected data")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--value-coeff", type=float, default=2.0)
    p.add_argument("--entropy-coeff", type=float, default=0.1,
                   help="Weight for target entropy penalty")
    p.add_argument("--clip-rho", type=float, default=1.0)
    p.add_argument("--clip-c", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--reward-scale", type=float, default=3.0,
                   help="Scale dense step rewards (default 3x)")
    p.add_argument("--ppo-clip", type=float, default=0.2,
                   help="PPO clipping epsilon for policy ratio")
    p.add_argument("--kl-coeff", type=float, default=2.0,
                   help="KL penalty coefficient (higher = more conservative)")
    p.add_argument("--max-train-steps", type=int, default=512,
                   help="Max gradient steps per epoch (0 = unlimited)")
    p.add_argument("--target-entropy", type=float, default=1.0,
                   help="Target entropy for regularization (penalize deviation)")
    p.add_argument("--lr-decay", choices=["none", "cosine"], default="none",
                   help="LR schedule: none (constant) or cosine decay to --lr-min")
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="Minimum LR for cosine decay schedule")
    p.add_argument("--advantage-clip", type=float, default=0.0,
                   help="Clip advantages to [-C, C] after normalization (0 = disabled)")
    p.add_argument("--no-vtrace", action="store_true",
                   help="Disable V-trace; use simple A2C advantages (returns - values)")
    p.add_argument("--polyak-tau", type=float, default=0.0,
                   help="EMA smoothing for weights used in generation (0 = disabled)")

    # Freeze/unfreeze
    p.add_argument("--freeze-transformer", action="store_true")
    p.add_argument("--freeze-policy", action="store_true",
                   help="Freeze policy heads; only train value head (for warmup)")
    p.add_argument("--warmup-iters", type=int, default=0,
                   help="Number of iterations to keep policy frozen (value-head warmup)")
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--no-grokfast", action="store_true")
    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)

    # GPU inference (shared memory)
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU inference server for episode generation (shared memory)")
    p.add_argument("--shm-name", default=SHM_NAME,
                   help="Shared memory name (creates /dev/shm/<name>)")

    # Self-play (league training)
    p.add_argument("--enemy-weights", help="Enemy policy weights JSON")
    p.add_argument("--enemy-registry", help="Enemy embedding registry JSON")

    # Eval
    p.add_argument("--eval-scenarios",
                   help="Separate eval scenario dir (default: same as --scenarios)")
    p.add_argument("--eval-every", type=int, default=5,
                   help="Evaluate every N iterations")
    p.add_argument("--eval-episodes", type=int, default=1)

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AbilityTokenizer()

    # Load embedding registry
    embedding_registry = None
    if args.embedding_registry:
        data = json.load(open(args.embedding_registry))
        embs = {}
        for name, vec in data["embeddings"].items():
            embs[name] = torch.tensor(vec, dtype=torch.float32, device=DEVICE)
        embedding_registry = {"embeddings": embs, "d_model": data["d_model"]}
        print(f"Loaded embedding registry: {len(embs)} abilities, d={data['d_model']}")

    cls_dim = embedding_registry["d_model"] if embedding_registry else args.d_model

    # Build model
    model = AbilityActorCriticV4(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=args.entity_encoder_layers,
        external_cls_dim=args.external_cls_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(DEVICE)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    model_sd = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd)
    print(f"Loaded {loaded} params from {args.checkpoint}")

    # Freeze/unfreeze
    if args.freeze_transformer:
        for n, param in model.named_parameters():
            if n.startswith("transformer."):
                param.requires_grad = False
    if not args.unfreeze_encoder:
        for n, param in model.named_parameters():
            if n.startswith("entity_encoder."):
                param.requires_grad = False

    trainable = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    total_params = sum(pp.numel() for pp in model.parameters())
    print(f"Model: {total_params:,} params, {trainable:,} trainable")
    print(f"Device: {DEVICE}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [pp for pp in model.parameters() if pp.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )

    # LR scheduler (E3)
    lr_scheduler = None
    if args.lr_decay == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iters, eta_min=args.lr_min,
        )

    gf = None
    if not args.no_grokfast:
        gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    # EMA model for episode generation (E8)
    ema_model = None
    if args.polyak_tau > 0:
        import copy
        ema_model = copy.deepcopy(model)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad = False

    # CSV log
    log_path = out_dir / "training.csv"
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "iter", "gen_time", "n_episodes", "win_rate",
        "n_steps", "policy_loss", "value_loss", "entropy",
        "mean_reward", "elapsed_s",
    ])

    weights_path = str(out_dir / "current_weights.json")
    episodes_path = str(out_dir / "episodes.jsonl")
    eval_scenarios = args.eval_scenarios or args.scenarios[0]

    # GPU server setup (shared memory)
    gpu_server_proc = None
    gpu_shm_path = None
    if args.gpu:
        ckpt_path = str(out_dir / "current.pt")
        torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
        print(f"\nStarting GPU inference server (shm={args.shm_name})...")
        gpu_server_proc = start_gpu_server(
            ckpt_path,
            shm_name=args.shm_name,
            d_model=args.d_model, d_ff=args.d_ff,
            n_layers=args.n_layers, n_heads=args.n_heads,
            entity_encoder_layers=args.entity_encoder_layers,
            external_cls_dim=args.external_cls_dim,
            temperature=args.temperature,
        )
        gpu_shm_path = f"{SHM_PATH_PREFIX}{args.shm_name}"
        print(f"GPU server ready at {gpu_shm_path}")

    t_start = time.time()
    best_win_rate = -1.0

    print(f"\nStarting IMPALA loop: {args.iters} iterations")
    print(f"  Scenarios: {' '.join(args.scenarios)}")
    print(f"  Episodes/scenario: {args.episodes_per_scenario}")
    print(f"  Threads: {args.threads}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Train epochs/iter: {args.train_epochs}")
    print(f"  GPU inference: {gpu_shm_path or 'no (CPU)'}")
    if args.no_vtrace:
        print(f"  V-trace: DISABLED (A2C mode)")
    if args.advantage_clip > 0:
        print(f"  Advantage clip: [-{args.advantage_clip}, {args.advantage_clip}]")
    if args.lr_decay != "none":
        print(f"  LR schedule: {args.lr_decay} ({args.lr} → {args.lr_min})")
    if args.warmup_iters > 0:
        print(f"  Value-head warmup: {args.warmup_iters} iters (policy frozen)")
    if args.polyak_tau > 0:
        print(f"  Polyak EMA: tau={args.polyak_tau}")
    print()

    for iteration in range(1, args.iters + 1):
        iter_t0 = time.time()

        # 1. Export current weights (use EMA if available)
        ckpt_path = str(out_dir / "current.pt")
        export_sd = ema_model.state_dict() if ema_model is not None else model.state_dict()
        torch.save({"model_state_dict": export_sd}, ckpt_path)

        if gpu_shm_path:
            # Hot-reload weights on GPU server via shared memory
            if iteration > 1:
                reload_gpu_server(ckpt_path, shm_name=args.shm_name)
        else:
            # CPU mode: export to JSON for Rust inference
            export_weights(
                ckpt_path, weights_path,
                d_model=args.d_model, d_ff=args.d_ff,
                n_layers=args.n_layers, n_heads=args.n_heads,
                entity_encoder_layers=args.entity_encoder_layers,
                external_cls_dim=args.external_cls_dim,
            )

        # 2. Generate episodes
        gen_time, gen_stderr = generate_episodes(
            scenario_dirs=args.scenarios,
            weights_path=weights_path,
            output_path=episodes_path,
            episodes_per_scenario=args.episodes_per_scenario,
            threads=args.threads,
            temperature=args.temperature,
            step_interval=args.step_interval,
            embedding_registry=args.embedding_registry,
            enemy_weights=args.enemy_weights,
            enemy_registry=args.enemy_registry,
            gpu_shm=gpu_shm_path,
            sims_per_thread=args.sims_per_thread,
        )

        # Parse win rate from stderr
        win_rate = 0.0
        n_episodes = 0
        for line in gen_stderr.split("\n"):
            if "Win rate:" in line:
                try:
                    win_rate = float(line.split("Win rate:")[1].strip().rstrip("%")) / 100
                except (ValueError, IndexError):
                    pass
            if "Episodes:" in line:
                try:
                    n_episodes = int(line.split("Episodes:")[1].split()[0])
                except (ValueError, IndexError):
                    pass

        # 3. Load episodes
        episodes = load_episodes(Path(episodes_path))
        if not episodes:
            print(f"  Iter {iteration}: no episodes generated, skipping")
            continue

        trajectories = extract_trajectories(episodes)
        total_steps = sum(len(t["steps"]) for t in trajectories)

        print(f"  Iter {iteration}: {n_episodes} eps, {len(trajectories)} trajs, "
              f"{total_steps} steps, win={win_rate:.1%}, gen={gen_time:.1f}s", flush=True)

        # 4. Train on collected data
        # Pre-tensorize once, reuse across epochs
        all_steps, step_traj_map = flatten_trajectories(trajectories)
        if all_steps:
            t_tens = time.time()
            ptd = PreTensorizedData(all_steps, embedding_registry, cls_dim)
            t_tens = time.time() - t_tens
            print(f"    Tensorized in {t_tens:.1f}s", flush=True)
        else:
            ptd = None

        # Determine if policy is frozen this iteration (E5: value-head warmup)
        policy_frozen = args.freeze_policy or (args.warmup_iters > 0 and iteration <= args.warmup_iters)

        for epoch in range(1, args.train_epochs + 1):
            m = train_on_trajectories(
                model, optimizer, gf, trajectories,
                all_steps, step_traj_map, ptd,
                gamma=args.gamma,
                step_interval=args.step_interval,
                batch_size=args.batch_size,
                value_coeff=args.value_coeff,
                entropy_coeff=args.entropy_coeff,
                clip_rho=args.clip_rho,
                clip_c=args.clip_c,
                max_grad_norm=args.max_grad_norm,
                reward_scale=args.reward_scale,
                ppo_clip=args.ppo_clip,
                kl_coeff=args.kl_coeff,
                max_train_steps=args.max_train_steps,
                target_entropy=args.target_entropy,
                no_vtrace=args.no_vtrace,
                advantage_clip=args.advantage_clip,
                freeze_policy=policy_frozen,
            )
            print(f"    Epoch {epoch}/{args.train_epochs} ({m['grad_steps']} steps): "
                  f"pg={m['policy_loss']:.4f} vl={m['value_loss']:.4f} "
                  f"ent={m['entropy']:.3f} kl={m['kl_div']:.3f} "
                  f"rew={m['mean_reward']:.4f}", flush=True)

        # LR scheduler step (E3)
        if lr_scheduler is not None:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    LR: {current_lr:.2e}", flush=True)

        # EMA update (E8)
        if ema_model is not None:
            tau = args.polyak_tau
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.mul_(tau).add_(p, alpha=1 - tau)

        elapsed = time.time() - t_start

        # Log
        csv_writer.writerow([
            iteration, f"{gen_time:.1f}", n_episodes, f"{win_rate:.3f}",
            total_steps, f"{m['policy_loss']:.4f}", f"{m['value_loss']:.4f}",
            f"{m['entropy']:.3f}", f"{m['mean_reward']:.4f}", f"{elapsed:.0f}",
        ])
        csv_file.flush()

        # Save checkpoint
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            torch.save(
                {"model_state_dict": model.state_dict(), "iteration": iteration},
                str(out_dir / "best.pt"),
            )
            print(f"    New best: {win_rate:.1%}")

        if iteration % 10 == 0:
            torch.save(
                {"model_state_dict": model.state_dict(), "iteration": iteration},
                str(out_dir / f"iter_{iteration}.pt"),
            )

        # 5. Evaluation (on separate scenarios if provided)
        if args.eval_scenarios and iteration % args.eval_every == 0:
            eval_weights_path = str(out_dir / "eval_weights.json")
            eval_ckpt_path = str(out_dir / "eval.pt")
            torch.save({"model_state_dict": model.state_dict()}, eval_ckpt_path)
            export_weights(
                eval_ckpt_path, eval_weights_path,
                d_model=args.d_model, d_ff=args.d_ff,
                n_layers=args.n_layers, n_heads=args.n_heads,
                entity_encoder_layers=args.entity_encoder_layers,
                external_cls_dim=args.external_cls_dim,
            )
            eval_out = str(out_dir / "eval_episodes.jsonl")
            eval_time, eval_stderr = generate_episodes(
                scenario_dirs=eval_scenarios,
                weights_path=eval_weights_path,
                output_path=eval_out,
                episodes_per_scenario=args.eval_episodes,
                threads=args.threads,
                temperature=0.1,  # near-greedy for eval
                step_interval=args.step_interval,
                embedding_registry=args.embedding_registry,
            )
            for line in eval_stderr.split("\n"):
                if "Win rate:" in line or "Episodes:" in line:
                    print(f"    EVAL: {line.strip()}")

        iter_elapsed = time.time() - iter_t0
        print(f"    Iter time: {iter_elapsed:.1f}s (total: {elapsed:.0f}s)")
        print()

    csv_file.close()
    if gpu_server_proc:
        gpu_server_proc.terminate()
        gpu_server_proc.wait(timeout=5)
    print(f"Training complete. Best win rate: {best_win_rate:.1%}")
    print(f"Checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
