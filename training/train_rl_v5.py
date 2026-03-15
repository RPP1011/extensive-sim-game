#!/usr/bin/env python3
"""Stage 1 RL fine-tuning for V5 actor-critic via PPO-Clip.

Loads pretrained components (encoder, latent interface, CfC, BC heads) and
runs on-policy PPO updates from episode JSONL produced by the GPU inference
server + xtask episode generation.

Graduated unfreezing via --stage:
  1a: Only move head + combat head + value head trainable
  1b: + latent interface
  1c: (reserved for spatial cross-attention, skipped)
  1d: + encoder last layer
  1e: Full model at reduced LR

Usage:
    uv run --with numpy --with torch python training/train_rl_v5.py \
        generated/v5_episodes.jsonl \
        --encoder-ckpt generated/entity_encoder_v5_full.pt \
        --cfc-ckpt generated/cfc_temporal_v5.pt \
        --latent-ckpt generated/latent_interface_v5.pt \
        --bc-ckpt generated/actor_critic_v5_pointer_bc.pt \
        --stage 1a \
        -o generated/actor_critic_v5_rl.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (
    AbilityActorCriticV5,
    EntityEncoderV5,
    LatentInterface,
    CfCCell,
    CombatPointerHeadV5,
    NUM_MOVE_DIRS,
    NUM_COMBAT_TYPES,
    MAX_ABILITIES,
    V5_DEFAULT_D,
    V5_DEFAULT_HEADS,
    V5_DEFAULT_LATENTS,
    CFC_H_DIM,
)
from tokenizer import AbilityTokenizer
from grokfast import GrokfastEMA

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTITY_DIM = 34
THREAT_DIM = 10
POSITION_DIM = 8
AGG_DIM = 16


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def apply_reward_shaping(episodes: list[dict], scale: float = 0.1) -> None:
    """Add dense step rewards from HP differentials. Modifies episodes in-place."""
    for ep in episodes:
        steps = ep["steps"]
        prev_advantage = None
        for step in steps:
            entities = step["entities"]
            entity_types = step["entity_types"]
            ally_hp = sum(
                ent[0] for ent, et in zip(entities, entity_types) if et < 2
            )
            enemy_hp = sum(
                ent[0] for ent, et in zip(entities, entity_types) if et == 2
            )
            advantage = ally_hp - enemy_hp
            if prev_advantage is not None:
                step.setdefault("step_reward", 0.0)
                step["step_reward"] += scale * (advantage - prev_advantage)
            prev_advantage = advantage


def flatten_steps(episodes: list[dict]) -> list[dict]:
    steps = []
    for ep in episodes:
        for step in ep["steps"]:
            step["_episode_reward"] = ep["reward"]
            steps.append(step)
    return steps


# ---------------------------------------------------------------------------
# Collation: build padded tensors from variable-length episode steps
# ---------------------------------------------------------------------------


def collate_v5_states(steps: list[dict], indices) -> dict[str, torch.Tensor]:
    """Collate variable-length V5 game states into padded tensors."""
    batch = [steps[i] for i in indices]
    B = len(batch)

    max_ents = max(len(s["entities"]) for s in batch)
    max_threats = max(
        (len(s["threats"]) for s in batch if s.get("threats")),
        default=1,
    )
    max_threats = max(max_threats, 1)
    max_positions = max(
        (len(s.get("positions", [])) for s in batch),
        default=1,
    )
    max_positions = max(max_positions, 1)

    ent_feat = torch.zeros(B, max_ents, ENTITY_DIM, device=DEVICE)
    ent_types = torch.zeros(B, max_ents, dtype=torch.long, device=DEVICE)
    ent_mask = torch.ones(B, max_ents, dtype=torch.bool, device=DEVICE)

    thr_feat = torch.zeros(B, max_threats, THREAT_DIM, device=DEVICE)
    thr_mask = torch.ones(B, max_threats, dtype=torch.bool, device=DEVICE)

    pos_feat = torch.zeros(B, max_positions, POSITION_DIM, device=DEVICE)
    pos_mask = torch.ones(B, max_positions, dtype=torch.bool, device=DEVICE)

    # Aggregate features -- fixed 16-dim
    agg_feat = torch.zeros(B, AGG_DIM, device=DEVICE)

    for i, s in enumerate(batch):
        n_e = len(s["entities"])
        ent_feat[i, :n_e] = torch.tensor(s["entities"], dtype=torch.float)
        ent_types[i, :n_e] = torch.tensor(s["entity_types"], dtype=torch.long)
        ent_mask[i, :n_e] = False

        threats = s.get("threats", [])
        n_t = len(threats)
        if n_t > 0:
            thr_feat[i, :n_t] = torch.tensor(threats, dtype=torch.float)
            thr_mask[i, :n_t] = False

        positions = s.get("positions", [])
        n_p = len(positions)
        if n_p > 0:
            pos_feat[i, :n_p] = torch.tensor(positions, dtype=torch.float)
            pos_mask[i, :n_p] = False

        agg = s.get("aggregate_features")
        if agg is not None:
            agg_feat[i] = torch.tensor(agg, dtype=torch.float)

    return {
        "entity_features": ent_feat,
        "entity_type_ids": ent_types,
        "threat_features": thr_feat,
        "entity_mask": ent_mask,
        "threat_mask": thr_mask,
        "position_features": pos_feat,
        "position_mask": pos_mask,
        "aggregate_features": agg_feat,
    }


# ---------------------------------------------------------------------------
# Ability CLS cache
# ---------------------------------------------------------------------------


def build_ability_cls_cache(
    model: AbilityActorCriticV5,
    episodes: list[dict],
    embedding_registry: dict | None = None,
) -> tuple[dict[int, list[list[int]]], dict[tuple[int, int], torch.Tensor]]:
    """Pre-compute frozen CLS embeddings for all unit abilities.

    Uses embedding registry when available, falls back to transformer forward.
    """
    unit_ability_tokens: dict[int, list[list[int]]] = {}
    unit_ability_names: dict[int, list[str]] = {}
    for ep in episodes:
        for uid_str, tokens_list in ep.get("unit_abilities", {}).items():
            uid = int(uid_str)
            if uid not in unit_ability_tokens:
                unit_ability_tokens[uid] = tokens_list
        for uid_str, names_list in ep.get("unit_ability_names", {}).items():
            uid = int(uid_str)
            if uid not in unit_ability_names:
                unit_ability_names[uid] = names_list

    cls_cache: dict[tuple[int, int], torch.Tensor] = {}
    reg_hits = 0

    if embedding_registry is not None:
        reg_embs = embedding_registry["embeddings"]
        for uid, names in unit_ability_names.items():
            for aidx, name in enumerate(names):
                key = name.replace(" ", "_")
                if key in reg_embs:
                    cls_cache[(uid, aidx)] = reg_embs[key]
                    reg_hits += 1

    for uid, tokens_list in unit_ability_tokens.items():
        for aidx, tokens in enumerate(tokens_list):
            if (uid, aidx) not in cls_cache:
                ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
                amask = (ids != 0).float()
                with torch.no_grad():
                    cls_emb = model.transformer.cls_embedding(ids, amask)
                cls_cache[(uid, aidx)] = cls_emb.squeeze(0)

    if embedding_registry is not None:
        print(f"  CLS cache: {reg_hits} from registry, "
              f"{len(cls_cache) - reg_hits} from transformer")
    else:
        print(f"  CLS cache: {len(cls_cache)} abilities from transformer")

    return unit_ability_tokens, cls_cache


def build_ability_cls_batch(
    steps: list[dict],
    idx,
    unit_ability_tokens: dict[int, list[list[int]]],
    cls_cache: dict[tuple[int, int], torch.Tensor],
    d_model: int,
) -> list[torch.Tensor | None]:
    """Build batched ability CLS tensors for a mini-batch."""
    B = len(idx)
    per_step_cls = []
    for ii in idx:
        step = steps[ii]
        uid = step["unit_id"]
        slot_cls = []
        if uid in unit_ability_tokens:
            for aidx in range(MAX_ABILITIES):
                key = (uid, aidx)
                slot_cls.append(cls_cache.get(key))
        else:
            slot_cls = [None] * MAX_ABILITIES
        per_step_cls.append(slot_cls)

    ability_cls_batch: list[torch.Tensor | None] = [None] * MAX_ABILITIES
    for aidx in range(MAX_ABILITIES):
        valid = []
        valid_indices = []
        for bi, step_cls in enumerate(per_step_cls):
            if aidx < len(step_cls) and step_cls[aidx] is not None:
                valid.append(step_cls[aidx])
                valid_indices.append(bi)

        if valid:
            stacked = torch.stack(valid)
            full = torch.zeros(B, d_model, device=DEVICE)
            for vi, bi in enumerate(valid_indices):
                full[bi] = stacked[vi]
            ability_cls_batch[aidx] = full

    return ability_cls_batch


def load_embedding_registry(path: str) -> dict:
    """Load pre-computed CLS embeddings from registry JSON."""
    data = json.load(open(path))
    embs = {}
    for name, vec in data["embeddings"].items():
        embs[name] = torch.tensor(vec, dtype=torch.float32, device=DEVICE)
    d_model = data["d_model"]
    print(f"Loaded embedding registry: {len(embs)} abilities, d={d_model}, "
          f"hash={data['model_hash']}")
    return {"embeddings": embs, "d_model": d_model}


# ---------------------------------------------------------------------------
# V5 action log-prob computation
# ---------------------------------------------------------------------------


def compute_v5_log_prob(
    move_logits: torch.Tensor,       # [B, 9]
    combat_output: dict,             # from CombatPointerHeadV5
    move_dirs: torch.Tensor,         # [B]
    combat_types: torch.Tensor,      # [B]
    target_indices: torch.Tensor,    # [B]
    combat_type_masks: torch.Tensor, # [B, NUM_COMBAT_TYPES]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log P(move) + log P(combat_type) + log P(target | combat_type).

    Returns (log_probs [B], entropy [B]).
    """
    B = move_dirs.shape[0]

    # Move log probs
    move_lp_all = F.log_softmax(move_logits, dim=-1)  # [B, 9]
    move_lp = move_lp_all.gather(1, move_dirs.unsqueeze(1)).squeeze(1)  # [B]
    move_probs = F.softmax(move_logits, dim=-1)
    move_entropy = -(move_probs * move_lp_all).sum(-1)  # [B]

    # Combat type log probs
    combat_logits = combat_output["combat_logits"]  # [B, NUM_COMBAT_TYPES]
    combat_logits = combat_logits.masked_fill(~combat_type_masks, -1e9)
    combat_lp_all = F.log_softmax(combat_logits, dim=-1)
    combat_lp = combat_lp_all.gather(1, combat_types.unsqueeze(1)).squeeze(1)
    combat_probs = F.softmax(combat_logits, dim=-1)
    combat_entropy = -(combat_probs * combat_lp_all).sum(-1)

    # Target pointer log probs (conditioned on combat type)
    target_lp = torch.zeros(B, device=DEVICE)
    target_entropy = torch.zeros(B, device=DEVICE)

    def _target_lp_for_ptr(ptr_logits, sel, target_indices):
        ptr_mask = ptr_logits > -1e8
        ptr_logits_masked = ptr_logits.masked_fill(~ptr_mask, -1e9)
        ptr_lp = F.log_softmax(ptr_logits_masked, dim=-1)
        N = ptr_logits.shape[1]
        sel_targets = target_indices[sel].clamp(0, N - 1)
        sel_lp = ptr_lp[sel].gather(1, sel_targets.unsqueeze(1)).squeeze(1)
        sel_lp = sel_lp.clamp(min=-10.0)
        ptr_probs = F.softmax(ptr_logits_masked[sel], dim=-1)
        sel_ent = -(ptr_probs * ptr_lp[sel]).sum(-1)
        return sel_lp, sel_ent

    # Attack (type=0) uses attack_ptr
    sel_atk = combat_types == 0
    if sel_atk.any():
        atk_lp, atk_ent = _target_lp_for_ptr(
            combat_output["attack_ptr"], sel_atk, target_indices)
        target_lp[sel_atk] = atk_lp
        target_entropy[sel_atk] = atk_ent

    # Hold (type=1): no pointer needed, target_lp stays 0

    # Ability pointers (types 2..9)
    for ab_idx in range(MAX_ABILITIES):
        ct_val = 2 + ab_idx
        sel = combat_types == ct_val
        if not sel.any():
            continue
        ab_ptrs = combat_output["ability_ptrs"]
        if ab_idx < len(ab_ptrs) and ab_ptrs[ab_idx] is not None:
            ab_lp, ab_ent = _target_lp_for_ptr(ab_ptrs[ab_idx], sel, target_indices)
            target_lp[sel] = ab_lp
            target_entropy[sel] = ab_ent

    composite_lp = move_lp + combat_lp + target_lp
    composite_entropy = move_entropy + combat_entropy + target_entropy

    return composite_lp, composite_entropy


# ---------------------------------------------------------------------------
# Build combat type masks
# ---------------------------------------------------------------------------


def build_combat_type_masks(steps: list[dict], idx) -> torch.Tensor:
    """Build [B, NUM_COMBAT_TYPES] masks from step data."""
    B = len(idx)
    masks = torch.zeros(B, NUM_COMBAT_TYPES, dtype=torch.bool, device=DEVICE)
    for bi, ii in enumerate(idx):
        s = steps[ii]
        mask_data = s.get("combat_mask") or s.get("mask")
        if mask_data is not None:
            # attack always valid if enemies exist
            has_enemies = any(
                et == 1 for et in s.get("entity_types", [])
            )
            masks[bi, 0] = has_enemies  # attack
            masks[bi, 1] = True  # hold
            for ab_idx in range(MAX_ABILITIES):
                key = 2 + ab_idx
                if key < len(mask_data):
                    masks[bi, key] = bool(mask_data[key])
                elif 3 + ab_idx < len(mask_data):
                    # V3 format: ability masks start at index 3
                    masks[bi, key] = bool(mask_data[3 + ab_idx])
        else:
            # Fallback: attack + hold always valid
            masks[bi, 0] = True
            masks[bi, 1] = True
    return masks


# ---------------------------------------------------------------------------
# Value function wrapper for GAE
# ---------------------------------------------------------------------------


class ValueFnWrapper:
    """Wraps the full model to produce value estimates for GAE."""

    def __init__(self, model: AbilityActorCriticV5):
        self.model = model

    @torch.no_grad()
    def __call__(self, steps: list[dict], indices) -> torch.Tensor:
        state = collate_v5_states(steps, indices)
        # Encode state through the full pipeline (no CfC for offline GAE)
        enc = self.model.encode_state(
            state["entity_features"], state["entity_type_ids"],
            state["threat_features"], state["entity_mask"], state["threat_mask"],
            [None] * MAX_ABILITIES,  # no ability CLS needed for value
            state["position_features"], state["position_mask"],
            state["aggregate_features"],
        )
        # Value from pooled latent representation
        pooled = enc["pooled"]
        value = self.model.value_head(pooled).squeeze(-1)  # [B]
        return value


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def compute_gae(
    episodes: list[dict],
    model: AbilityActorCriticV5,
    steps: list[dict],
    gamma: float = 0.99,
    lam: float = 0.95,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns for all steps.

    Returns (advantages, returns) as numpy arrays.
    """
    value_fn = ValueFnWrapper(model)

    all_advantages = []
    all_returns = []

    step_offset = 0
    for ep in episodes:
        ep_steps = ep["steps"]
        n = len(ep_steps)
        if n == 0:
            continue

        # Compute values for all steps in this episode
        idx = list(range(step_offset, step_offset + n))
        values_list = []
        for chunk_start in range(0, n, batch_size):
            chunk_end = min(chunk_start + n, n)
            chunk_idx = idx[chunk_start:chunk_end]
            chunk_values = value_fn(steps, chunk_idx).cpu().numpy()
            values_list.append(chunk_values)
        values = np.concatenate(values_list)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            step_reward = ep_steps[t].get("step_reward", 0.0)
            if t == n - 1:
                step_reward += ep["reward"]
                next_value = 0.0
            else:
                next_value = values[t + 1]
            delta = step_reward + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        returns = advantages + values
        all_advantages.extend(advantages)
        all_returns.extend(returns)
        step_offset += n

    return (
        np.array(all_advantages, dtype=np.float32),
        np.array(all_returns, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Graduated unfreezing
# ---------------------------------------------------------------------------


def apply_freeze_policy(model: AbilityActorCriticV5, stage: str) -> None:
    """Freeze/unfreeze model components based on training stage.

    Stages:
      1a: Only move head + combat head + value head trainable
      1b: + latent interface
      1c: (reserved, same as 1b)
      1d: + encoder last layer
      1e: Full model (all unfrozen)

    The ability transformer is ALWAYS frozen (CLS embeddings are pre-computed).
    """
    # Start by freezing everything
    for p in model.parameters():
        p.requires_grad = False

    # Always unfreeze decision heads
    trainable_modules = []

    # Move head
    for p in model.move_head.parameters():
        p.requires_grad = True
    trainable_modules.append("move_head")

    # Combat head
    for p in model.combat_head.parameters():
        p.requires_grad = True
    trainable_modules.append("combat_head")

    # Value head
    if hasattr(model, "value_head"):
        for p in model.value_head.parameters():
            p.requires_grad = True
        trainable_modules.append("value_head")

    if stage in ("1b", "1c", "1d", "1e"):
        for p in model.latent_interface.parameters():
            p.requires_grad = True
        trainable_modules.append("latent_interface")

    if stage in ("1c", "1d", "1e"):
        # Cross-attention (1c is reserved for spatial cross-attn, we treat it same as 1d)
        for p in model.cross_attn.parameters():
            p.requires_grad = True
        trainable_modules.append("cross_attn")

    if stage in ("1d", "1e"):
        # Unfreeze last encoder layer only
        encoder_layers = model.entity_encoder.encoder.layers
        last_layer = encoder_layers[-1]
        for p in last_layer.parameters():
            p.requires_grad = True
        # Also unfreeze encoder output norm
        for p in model.entity_encoder.out_norm.parameters():
            p.requires_grad = True
        trainable_modules.append("encoder_last_layer")

    if stage == "1e":
        # Full model (except transformer)
        for name, p in model.named_parameters():
            if not name.startswith("transformer."):
                p.requires_grad = True
        trainable_modules = ["all (except transformer)"]

    # CfC temporal cell: unfrozen from 1b onwards
    if stage in ("1b", "1c", "1d", "1e"):
        for p in model.temporal_cell.parameters():
            p.requires_grad = True
        if "all (except transformer)" not in trainable_modules:
            trainable_modules.append("temporal_cell")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Stage {stage}: {n_trainable:,}/{n_total:,} params trainable")
    print(f"  Unfrozen: {', '.join(trainable_modules)}")


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    model: AbilityActorCriticV5,
    optimizer: torch.optim.Optimizer,
    grokfast: GrokfastEMA | None,
    steps: list[dict],
    advantages: np.ndarray,
    returns: np.ndarray,
    unit_ability_tokens: dict[int, list[list[int]]],
    cls_cache: dict[tuple[int, int], torch.Tensor],
    clip_eps: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    max_grad_norm: float = 0.5,
) -> dict:
    """Run PPO-Clip update on collected episode data."""
    n = len(steps)
    if n == 0:
        return {}

    # Extract actions and old log probs from steps
    move_dirs = torch.tensor(
        [s["move_dir"] for s in steps], dtype=torch.long, device=DEVICE)
    combat_types = torch.tensor(
        [s["combat_type"] for s in steps], dtype=torch.long, device=DEVICE)
    target_indices = torch.tensor(
        [s["target_idx"] for s in steps], dtype=torch.long, device=DEVICE)

    # Old log probs: sum of move + combat + pointer from episode generation
    old_lp_move = torch.tensor(
        [s.get("lp_move", 0.0) for s in steps], dtype=torch.float, device=DEVICE)
    old_lp_combat = torch.tensor(
        [s.get("lp_combat", 0.0) for s in steps], dtype=torch.float, device=DEVICE)
    old_lp_pointer = torch.tensor(
        [s.get("lp_pointer", 0.0) for s in steps], dtype=torch.float, device=DEVICE)
    old_log_probs = old_lp_move + old_lp_combat + old_lp_pointer

    adv_tensor = torch.tensor(advantages, dtype=torch.float, device=DEVICE)
    ret_tensor = torch.tensor(returns, dtype=torch.float, device=DEVICE)

    # Normalize advantages
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    model.train()
    d_model = model.d_model

    metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0,
    }

    for epoch in range(ppo_epochs):
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]

            state = collate_v5_states(steps, idx)

            batch_move_dirs = move_dirs[idx]
            batch_combat_types = combat_types[idx]
            batch_target_indices = target_indices[idx]
            batch_old_lp = old_log_probs[idx]
            batch_adv = adv_tensor[idx]
            batch_ret = ret_tensor[idx]

            combat_type_masks = build_combat_type_masks(steps, idx)

            ability_cls_batch = build_ability_cls_batch(
                steps, idx, unit_ability_tokens, cls_cache, d_model)

            # Forward: encode_state -> temporal_cell -> decide
            enc = model.encode_state(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
                ability_cls_batch,
                state["position_features"], state["position_mask"],
                state["aggregate_features"],
            )

            # No CfC for offline PPO (no sequential h_prev)
            pooled = enc["pooled"]

            output = model.decide(
                pooled, enc["tokens"], enc["full_mask"],
                enc["ability_cross_embs"], enc["full_type_ids"],
                aggregate_features=state["aggregate_features"],
            )

            move_logits = output["move_logits"]

            # Value prediction
            value = model.value_head(pooled).squeeze(-1)

            # Compute new log probs
            action_log_probs, entropy_per_sample = compute_v5_log_prob(
                move_logits, output, batch_move_dirs,
                batch_combat_types, batch_target_indices,
                combat_type_masks,
            )
            entropy = entropy_per_sample.mean()

            # PPO-Clip objective
            ratio = (action_log_probs - batch_old_lp).exp()
            surr1 = ratio * batch_adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value, batch_ret)
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            loss.backward()
            if grokfast is not None:
                grokfast.step()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (batch_old_lp - action_log_probs).mean().item()
                clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy"] += entropy.item()
            metrics["approx_kl"] += approx_kl
            metrics["clip_frac"] += clip_frac
            metrics["n_updates"] += 1

    nu = max(metrics["n_updates"], 1)
    for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"]:
        metrics[k] /= nu

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def load_component_checkpoints(
    model: AbilityActorCriticV5,
    encoder_ckpt: str | None,
    cfc_ckpt: str | None,
    latent_ckpt: str | None,
    bc_ckpt: str | None,
) -> None:
    """Load pretrained component checkpoints into the full model."""

    if encoder_ckpt:
        print(f"Loading encoder: {encoder_ckpt}")
        ckpt = torch.load(encoder_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # Pretraining wraps encoder as "encoder.*"
        enc_sd = {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "entity_encoder.", 1)
                enc_sd[new_key] = v
        if enc_sd:
            missing, unexpected = model.load_state_dict(enc_sd, strict=False)
            loaded = len(enc_sd) - len(unexpected)
            print(f"  Loaded {loaded} encoder params")
        else:
            # Try direct entity_encoder.* keys
            direct_sd = {k: v for k, v in sd.items() if k.startswith("entity_encoder.")}
            if direct_sd:
                missing, unexpected = model.load_state_dict(direct_sd, strict=False)
                print(f"  Loaded {len(direct_sd) - len(unexpected)} encoder params (direct)")

    if cfc_ckpt:
        print(f"Loading CfC: {cfc_ckpt}")
        ckpt = torch.load(cfc_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cfc_sd = {}
        for k, v in sd.items():
            if k.startswith("temporal_cell."):
                cfc_sd[k] = v
            elif not k.startswith(("encoder.", "entity_encoder.", "latent_", "move_", "combat_", "value_")):
                cfc_sd[f"temporal_cell.{k}"] = v
        if cfc_sd:
            missing, unexpected = model.load_state_dict(cfc_sd, strict=False)
            loaded = len(cfc_sd) - len(unexpected)
            print(f"  Loaded {loaded} CfC params")

    if latent_ckpt:
        print(f"Loading latent interface: {latent_ckpt}")
        ckpt = torch.load(latent_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        lat_sd = {}
        for k, v in sd.items():
            if k.startswith("latent_interface."):
                lat_sd[k] = v
            elif not k.startswith(("encoder.", "entity_encoder.", "temporal_", "move_", "combat_", "value_")):
                lat_sd[f"latent_interface.{k}"] = v
        if lat_sd:
            missing, unexpected = model.load_state_dict(lat_sd, strict=False)
            loaded = len(lat_sd) - len(unexpected)
            print(f"  Loaded {loaded} latent interface params")

    if bc_ckpt:
        print(f"Loading BC heads: {bc_ckpt}")
        ckpt = torch.load(bc_ckpt, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        head_sd = {}
        head_prefixes = ("move_head.", "combat_head.", "cross_attn.")
        for k, v in sd.items():
            if any(k.startswith(pfx) for pfx in head_prefixes):
                head_sd[k] = v
        if head_sd:
            missing, unexpected = model.load_state_dict(head_sd, strict=False)
            loaded = len(head_sd) - len(unexpected)
            print(f"  Loaded {loaded} BC head params")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Stage 1: PPO-Clip RL for V5 actor-critic")
    p.add_argument("episodes", nargs="+", help="JSONL episode file(s)")
    p.add_argument("-o", "--output", default="generated/actor_critic_v5_rl.pt")
    p.add_argument("--log", default="generated/actor_critic_v5_rl.csv")

    # Model
    p.add_argument("--d-model", type=int, default=V5_DEFAULT_D)
    p.add_argument("--n-heads", type=int, default=V5_DEFAULT_HEADS)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--n-latents", type=int, default=V5_DEFAULT_LATENTS)
    p.add_argument("--h-dim", type=int, default=CFC_H_DIM)
    p.add_argument("--vocab-size", type=int, default=0,
                   help="Override tokenizer vocab size (0 = auto)")

    # Checkpoints
    p.add_argument("--checkpoint", help="Full model checkpoint (overrides component ckpts)")
    p.add_argument("--encoder-ckpt", help="Pretrained entity encoder (.pt)")
    p.add_argument("--cfc-ckpt", help="Pretrained CfC temporal cell (.pt)")
    p.add_argument("--latent-ckpt", help="Pretrained latent interface (.pt)")
    p.add_argument("--bc-ckpt", help="BC-warmed action heads (.pt)")
    p.add_argument("--embedding-registry", help="Pre-computed CLS embedding registry JSON")

    # Graduated unfreezing
    p.add_argument("--stage", default="1a", choices=["1a", "1b", "1c", "1d", "1e"],
                   help="Unfreezing stage (1a=heads only, 1e=full)")

    # PPO hyperparameters
    p.add_argument("--iterations", type=int, default=10,
                   help="Number of PPO iterations (re-loads data each iteration)")
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-reduced", type=float, default=3e-5,
                   help="LR for stage 1e (full model)")
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Reward shaping
    p.add_argument("--reward-shaping", type=float, default=0.1,
                   help="Dense reward scale from HP differential (0 = disabled)")

    # Grokfast
    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)
    p.add_argument("--no-grokfast", action="store_true")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1,
                   help="Save checkpoint every N iterations")

    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    print(f"Stage: {args.stage}")

    # Tokenizer (for ability transformer CLS computation)
    tok = AbilityTokenizer()
    vocab_size = args.vocab_size if args.vocab_size > 0 else tok.vocab_size

    # Build model
    model = AbilityActorCriticV5(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ff=args.d_model * 2,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        entity_encoder_layers=args.entity_encoder_layers,
        h_dim=args.h_dim,
        n_latents=args.n_latents,
    )

    # Add value head if not present
    if not hasattr(model, "value_head"):
        model.value_head = nn.Sequential(
            nn.Linear(args.d_model, args.d_model), nn.GELU(),
            nn.Linear(args.d_model, 1),
        )
    model = model.to(DEVICE)

    # Load checkpoints
    if args.checkpoint:
        print(f"Loading full checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"  Loaded: {len(sd) - len(unexpected)} params, "
              f"missing: {len(missing)}, unexpected: {len(unexpected)}")
    else:
        load_component_checkpoints(
            model, args.encoder_ckpt, args.cfc_ckpt,
            args.latent_ckpt, args.bc_ckpt,
        )

    # Apply freeze policy based on stage
    apply_freeze_policy(model, args.stage)

    # Optimizer: use reduced LR for stage 1e
    lr = args.lr_reduced if args.stage == "1e" else args.lr

    # Separate param groups: value head gets full LR, actor params may differ
    value_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "value_head" in name:
            value_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if value_params:
        param_groups.append({"params": value_params, "lr": lr, "label": "value"})
    if other_params:
        param_groups.append({"params": other_params, "lr": lr, "label": "actor"})

    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.98), weight_decay=args.weight_decay,
    )
    print(f"Optimizer: AdamW lr={lr:.2e}, weight_decay={args.weight_decay}")

    # Grokfast
    grokfast = None
    if not args.no_grokfast:
        grokfast = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    # Embedding registry
    embedding_registry = None
    if args.embedding_registry:
        embedding_registry = load_embedding_registry(args.embedding_registry)

    # CSV logging
    log_path = Path(args.log)
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "iteration", "mean_reward", "win_rate",
        "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_frac", "elapsed_s",
    ])

    best_mean_reward = -float("inf")

    # Main training loop: iterate over episode data
    for iteration in range(1, args.iterations + 1):
        iter_t0 = time.time()

        # Load episodes (supports multiple JSONL files for data aggregation)
        all_episodes = []
        for ep_path in args.episodes:
            eps = load_episodes(Path(ep_path))
            all_episodes.extend(eps)
        print(f"\n=== Iteration {iteration}/{args.iterations} ===")
        print(f"Loaded {len(all_episodes)} episodes from {len(args.episodes)} file(s)")

        # Apply reward shaping
        if args.reward_shaping > 0:
            apply_reward_shaping(all_episodes, scale=args.reward_shaping)

        # Flatten steps
        all_steps = flatten_steps(all_episodes)
        print(f"  {len(all_steps)} steps total")

        # Episode statistics
        wins = sum(1 for e in all_episodes if e.get("outcome") == "Victory")
        mean_reward = np.mean([e["reward"] for e in all_episodes])
        win_rate = wins / max(len(all_episodes), 1)
        print(f"  Win rate: {wins}/{len(all_episodes)} ({win_rate*100:.1f}%)")
        print(f"  Mean reward: {mean_reward:.4f}")

        # Build ability CLS cache (once per iteration, since abilities are fixed)
        if iteration == 1:
            unit_ability_tokens, cls_cache = build_ability_cls_cache(
                model, all_episodes, embedding_registry)

        # Compute GAE
        print("Computing GAE advantages...")
        advantages, returns = compute_gae(
            all_episodes, model, all_steps,
            gamma=args.gamma, lam=args.gae_lambda,
            batch_size=args.batch_size * 2,
        )
        print(f"  Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")

        # PPO update
        print(f"PPO update ({args.ppo_epochs} epochs, batch={args.batch_size})...")
        metrics = ppo_update(
            model, optimizer, grokfast,
            all_steps, advantages, returns,
            unit_ability_tokens, cls_cache,
            clip_eps=args.clip_eps,
            value_coeff=args.value_coeff,
            entropy_coeff=args.entropy_coeff,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
        )

        elapsed = time.time() - iter_t0

        print(f"  Policy loss:   {metrics['policy_loss']:.4f}")
        print(f"  Value loss:    {metrics['value_loss']:.4f}")
        print(f"  Entropy:       {metrics['entropy']:.4f}")
        print(f"  Approx KL:     {metrics['approx_kl']:.4f}")
        print(f"  Clip fraction: {metrics['clip_frac']:.3f}")
        print(f"  Elapsed:       {elapsed:.1f}s")

        # Log to CSV
        writer.writerow([
            iteration, f"{mean_reward:.4f}", f"{win_rate:.3f}",
            f"{metrics['policy_loss']:.4f}", f"{metrics['value_loss']:.4f}",
            f"{metrics['entropy']:.4f}", f"{metrics['approx_kl']:.4f}",
            f"{metrics['clip_frac']:.3f}", f"{elapsed:.1f}",
        ])
        log_file.flush()

        # Save best by mean reward
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            torch.save({
                "model_state_dict": model.state_dict(),
                "iteration": iteration,
                "mean_reward": mean_reward,
                "win_rate": win_rate,
                "stage": args.stage,
                "args": vars(args),
            }, args.output.replace(".pt", "_best.pt"))
            print(f"  New best reward: {mean_reward:.4f} (saved *_best.pt)")

        # Periodic checkpoint
        if iteration % args.save_every == 0:
            ckpt_path = args.output.replace(".pt", f"_iter{iteration}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "iteration": iteration,
                "mean_reward": mean_reward,
                "win_rate": win_rate,
                "stage": args.stage,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    log_file.close()

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "iteration": args.iterations,
        "stage": args.stage,
        "args": vars(args),
    }, args.output)
    print(f"\nFinal model saved to {args.output}")
    print(f"Best mean reward: {best_mean_reward:.4f}")
    print(f"Training log: {args.log}")


if __name__ == "__main__":
    main()
