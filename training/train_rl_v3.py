#!/usr/bin/env python3
"""PPO training for V3 actor-critic with pointer-based action space.

Uses hierarchical action: action_type (11) + target_pointer over entity tokens.
Episodes must contain `positions`, `action_type`, `target_idx` fields in each step.

Usage:
    uv run --with numpy --with torch training/train_rl_v3.py \
        generated/rl_episodes_v3.jsonl \
        --entity-encoder generated/entity_encoder_pretrained_v4.pt \
        -o generated/actor_critic_v3.pt \
        --ppo-epochs 4
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
    AbilityActorCriticV3,
    NUM_ACTION_TYPES,
    MAX_ABILITIES,
    POSITION_DIM,
)
from tokenizer import AbilityTokenizer
from grokfast import GrokfastEMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENTITY_DIM = 30
THREAT_DIM = 8


# ---------------------------------------------------------------------------
# Data loading + collation
# ---------------------------------------------------------------------------


def load_embedding_registry(path: str, device: str = DEVICE) -> dict:
    """Load pre-computed CLS embeddings from registry JSON."""
    data = json.load(open(path))
    embs = {}
    for name, vec in data["embeddings"].items():
        embs[name] = torch.tensor(vec, dtype=torch.float32, device=device)
    d_model = data["d_model"]
    print(f"Loaded embedding registry: {len(embs)} abilities, d={d_model}, "
          f"hash={data['model_hash']}")
    return {"embeddings": embs, "d_model": d_model}


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def apply_reward_shaping(episodes: list[dict], scale: float = 0.1) -> None:
    """Compute dense step rewards from HP differentials.

    For each step, step_reward = scale * delta(ally_hp_sum - enemy_hp_sum).
    Modifies episodes in-place.
    """
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
                step["step_reward"] = scale * (advantage - prev_advantage)
            prev_advantage = advantage


def flatten_steps(episodes: list[dict], wins_only: bool = False) -> list[dict]:
    steps = []
    for ep in episodes:
        if wins_only and ep.get("outcome") != "Victory":
            continue
        for step in ep["steps"]:
            step["_episode_reward"] = ep["reward"]
            steps.append(step)
    return steps


def collate_v3_states(steps: list[dict], indices) -> dict[str, torch.Tensor]:
    """Collate variable-length v3 game states into padded tensors.

    Extends v2 collation with position tokens.
    """
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

    # Total sequence length per sample for pointer logit padding
    max_seq = max_ents + max_threats + max_positions

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

    return {
        "entity_features": ent_feat,
        "entity_type_ids": ent_types,
        "threat_features": thr_feat,
        "entity_mask": ent_mask,
        "threat_mask": thr_mask,
        "position_features": pos_feat,
        "position_mask": pos_mask,
        "max_seq": max_seq,
    }


# ---------------------------------------------------------------------------
# Hierarchical action log prob
# ---------------------------------------------------------------------------


def compute_hierarchical_log_prob(
    pointer_output: dict,
    action_types: torch.Tensor,  # [B]
    target_indices: torch.Tensor,  # [B]
    type_masks: torch.Tensor,  # [B, 11]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute log P(action) = log P(type) + log P(target | type).

    Returns (log_probs [B], entropy [B]).
    """
    B = action_types.shape[0]

    # Action type log probs
    type_logits = pointer_output["type_logits"]  # [B, 11]
    type_logits = type_logits.masked_fill(~type_masks, -1e9)
    type_log_probs = F.log_softmax(type_logits, dim=-1)  # [B, 11]
    type_lp = type_log_probs.gather(1, action_types.unsqueeze(1)).squeeze(1)  # [B]

    # Type entropy
    type_probs = F.softmax(type_logits, dim=-1)
    type_entropy = -(type_probs * type_log_probs).sum(-1)  # [B]

    # Per-sample target log prob based on action type
    target_lp = torch.zeros(B, device=type_logits.device)
    target_entropy = torch.zeros(B, device=type_logits.device)

    def _target_lp_for_ptr(ptr_logits, sel, target_indices):
        """Compute target log probs, clamping out-of-range indices."""
        ptr_mask = ptr_logits > -1e8  # valid tokens
        ptr_logits_masked = ptr_logits.masked_fill(~ptr_mask, -1e9)
        ptr_lp = F.log_softmax(ptr_logits_masked, dim=-1)
        N = ptr_logits.shape[1]
        sel_targets_raw = target_indices[sel]
        sel_targets = sel_targets_raw.clamp(0, N - 1)
        sel_lp = ptr_lp[sel].gather(1, sel_targets.unsqueeze(1)).squeeze(1)
        # Clamp log_prob floor to -10 to prevent KL explosion from OOB targets
        sel_lp = sel_lp.clamp(min=-10.0)
        ptr_probs = F.softmax(ptr_logits_masked[sel], dim=-1)
        sel_ent = -(ptr_probs * ptr_lp[sel]).sum(-1)
        return sel_lp, sel_ent

    for at_val, ptr_key in [
        (0, "attack_ptr"),
        (1, "move_ptr"),
    ]:
        sel = action_types == at_val
        if not sel.any():
            continue
        ptr_logits = pointer_output[ptr_key]  # [B, N]
        sel_lp, sel_ent = _target_lp_for_ptr(ptr_logits, sel, target_indices)
        target_lp[sel] = sel_lp
        target_entropy[sel] = sel_ent

    # Ability pointers (3..10)
    for ab_idx in range(MAX_ABILITIES):
        at_val = 3 + ab_idx
        sel = action_types == at_val
        if not sel.any():
            continue
        ab_ptrs = pointer_output["ability_ptrs"]  # list of [B, N] or None
        if ab_idx < len(ab_ptrs) and ab_ptrs[ab_idx] is not None:
            ptr_logits = ab_ptrs[ab_idx]  # [B, N]
            sel_lp, sel_ent = _target_lp_for_ptr(ptr_logits, sel, target_indices)
            target_lp[sel] = sel_lp
            target_entropy[sel] = sel_ent

    # Hold (type=2): no pointer, target_lp stays 0
    composite_lp = type_lp + target_lp
    composite_entropy = type_entropy + target_entropy

    return composite_lp, composite_entropy


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def compute_gae(
    episodes: list[dict],
    value_fn,
    gamma: float = 0.99,
    lam: float = 0.95,
    chunk_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute GAE with V3 value function."""
    all_advantages = []
    all_returns = []
    all_values = []

    for ep in episodes:
        steps = ep["steps"]
        if not steps:
            continue

        idx = list(range(len(steps)))
        state = collate_v3_states(steps, idx)

        with torch.no_grad():
            values = value_fn(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
                state["position_features"], state["position_mask"],
            ).squeeze(-1).cpu().numpy()

        n = len(steps)
        advantages = np.zeros(n, dtype=np.float32)

        if chunk_size > 0:
            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                gae = 0.0
                for t in reversed(range(chunk_start, chunk_end)):
                    reward = steps[t]["step_reward"]
                    if t == n - 1:
                        reward += ep["reward"]
                        next_value = 0.0
                    elif t == chunk_end - 1:
                        next_value = values[t + 1]
                    else:
                        next_value = values[t + 1]
                    delta = reward + gamma * next_value - values[t]
                    gae = delta + gamma * lam * gae
                    advantages[t] = gae
        else:
            gae = 0.0
            for t in reversed(range(n)):
                reward = steps[t]["step_reward"]
                if t == n - 1:
                    reward += ep["reward"]
                    next_value = 0.0
                else:
                    next_value = values[t + 1]
                delta = reward + gamma * next_value - values[t]
                gae = delta + gamma * lam * gae
                advantages[t] = gae

        returns = advantages + values
        all_advantages.extend(advantages)
        all_returns.extend(returns)
        all_values.extend(values)

    return (
        np.array(all_advantages, dtype=np.float32),
        np.array(all_returns, dtype=np.float32),
        np.array(all_values, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Ability CLS cache
# ---------------------------------------------------------------------------


def build_ability_cls_cache(
    model: AbilityActorCriticV3,
    episodes: list[dict],
    embedding_registry: dict | None = None,
) -> tuple[dict[int, list[list[int]]], dict[tuple[int, int], torch.Tensor]]:
    """Pre-compute frozen CLS embeddings for all unit abilities.

    If embedding_registry is provided, looks up CLS by ability name first,
    falling back to transformer forward pass for unknown abilities.
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

    # Try registry lookup first
    if embedding_registry is not None:
        reg_embs = embedding_registry["embeddings"]
        for uid, names in unit_ability_names.items():
            for aidx, name in enumerate(names):
                key = name.replace(" ", "_")
                if key in reg_embs:
                    cls_cache[(uid, aidx)] = reg_embs[key]
                    reg_hits += 1

    # Fall back to transformer for abilities not in registry
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


# ---------------------------------------------------------------------------
# Build action type masks for a batch
# ---------------------------------------------------------------------------


def build_type_masks(steps: list[dict], idx) -> torch.Tensor:
    """Build [B, 11] action type masks from v1 action masks."""
    B = len(idx)
    type_masks = torch.zeros(B, NUM_ACTION_TYPES, dtype=torch.bool, device=DEVICE)
    for bi, ii in enumerate(idx):
        s = steps[ii]
        mask = s["mask"]  # 14-element v1 mask
        has_enemies = mask[0]  # attack nearest valid → enemies exist
        type_masks[bi, 0] = has_enemies  # attack
        type_masks[bi, 1] = True  # move
        type_masks[bi, 2] = True  # hold
        for ab_idx in range(MAX_ABILITIES):
            if 3 + ab_idx < len(mask):
                type_masks[bi, 3 + ab_idx] = mask[3 + ab_idx]
    return type_masks


# ---------------------------------------------------------------------------
# Critic warmup
# ---------------------------------------------------------------------------


def warmup_critic(
    model: AbilityActorCriticV3,
    steps: list[dict],
    returns: np.ndarray,
    episodes: list[dict],
    unit_ability_tokens: dict[int, list[list[int]]],
    cls_cache: dict[tuple[int, int], torch.Tensor],
    lr: float = 3e-4,
    epochs: int = 5,
    batch_size: int = 256,
    cls_dim: int = 0,
) -> float:
    """Pre-train the value head on GAE returns before PPO."""
    critic_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and n.startswith("value_head")]
    if not critic_params:
        return 0.0

    opt = torch.optim.AdamW(critic_params, lr=lr, weight_decay=0.01)
    ret_tensor = torch.tensor(returns, dtype=torch.float, device=DEVICE)
    n = len(steps)

    total_loss = 0.0
    n_updates = 0

    for epoch in range(epochs):
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            B = len(idx)

            state = collate_v3_states(steps, idx)
            batch_ret = ret_tensor[idx]

            ability_cls_batch = build_ability_cls_batch(
                steps, idx, unit_ability_tokens, cls_cache, cls_dim or model.d_model)

            _, values = model(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
                ability_cls_batch,
                state["position_features"], state["position_mask"],
            )
            values = values.squeeze(-1)

            loss = F.mse_loss(values, batch_ret)
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_updates += 1

    return total_loss / max(n_updates, 1)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


def ppo_update(
    model: AbilityActorCriticV3,
    optimizer: torch.optim.Optimizer,
    grokfast: GrokfastEMA | None,
    steps: list[dict],
    advantages: np.ndarray,
    returns: np.ndarray,
    episodes: list[dict],
    tokenizer: AbilityTokenizer,
    unit_ability_tokens: dict[int, list[list[int]]],
    cls_cache: dict[tuple[int, int], torch.Tensor],
    clip_eps: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.01,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    max_grad_norm: float = 0.5,
    cls_dim: int = 0,
) -> dict:
    n = len(steps)
    if n == 0:
        return {}

    action_types = torch.tensor(
        [s["action_type"] for s in steps], dtype=torch.long, device=DEVICE)
    target_indices = torch.tensor(
        [s["target_idx"] for s in steps], dtype=torch.long, device=DEVICE)
    old_log_probs = torch.tensor(
        [s["log_prob"] for s in steps], dtype=torch.float, device=DEVICE)
    adv_tensor = torch.tensor(advantages, dtype=torch.float, device=DEVICE)
    ret_tensor = torch.tensor(returns, dtype=torch.float, device=DEVICE)

    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    model.train()

    metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "approx_kl": 0.0, "clip_frac": 0.0, "n_updates": 0,
    }

    for epoch in range(ppo_epochs):
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            B = len(idx)

            state = collate_v3_states(steps, idx)

            batch_action_types = action_types[idx]
            batch_target_indices = target_indices[idx]
            batch_old_lp = old_log_probs[idx]
            batch_adv = adv_tensor[idx]
            batch_ret = ret_tensor[idx]

            type_masks = build_type_masks(steps, idx)

            ability_cls_batch = build_ability_cls_batch(
                steps, idx, unit_ability_tokens, cls_cache, cls_dim or model.d_model)

            # Forward pass
            pointer_output, values = model(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
                ability_cls_batch,
                state["position_features"], state["position_mask"],
            )
            values = values.squeeze(-1)

            # Hierarchical log prob
            action_log_probs, entropy_per_sample = compute_hierarchical_log_prob(
                pointer_output, batch_action_types, batch_target_indices, type_masks,
            )
            entropy = entropy_per_sample.mean()

            ratio = (action_log_probs - batch_old_lp).exp()
            surr1 = ratio * batch_adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_ret)
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
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="PPO training for V3 pointer actor-critic")
    p.add_argument("episodes", help="JSONL episode file with v3 fields (positions, action_type, target_idx)")
    p.add_argument("--pretrained", help="Pretrained checkpoint (.pt)")
    p.add_argument("--entity-encoder", help="Pretrained entity encoder (.pt)")
    p.add_argument("-o", "--output", default="generated/actor_critic_v3.pt")
    p.add_argument("--log", default="generated/actor_critic_v3.csv")

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)

    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    p.add_argument("--critic-warmup-epochs", type=int, default=0)

    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)
    p.add_argument("--no-grokfast", action="store_true")

    p.add_argument("--actor-lr-ratio", type=float, default=1.0)

    p.add_argument("--freeze-transformer", action="store_true", default=True)
    p.add_argument("--unfreeze-transformer", action="store_true")
    p.add_argument("--unfreeze-encoder", action="store_true")

    p.add_argument("--bc-epochs", type=int, default=0,
                   help="Behavioral cloning epochs (supervised on oracle actions)")
    p.add_argument("--bc-wins-only", action="store_true",
                   help="Only train BC on winning episodes")
    p.add_argument("--reinforce-epochs", type=int, default=0,
                   help="REINFORCE epochs: reward-weighted policy gradient (no value function)")
    p.add_argument("--reward-shaping", type=float, default=0.0,
                   help="Dense reward scale from HP differential (0 = disabled)")
    p.add_argument("--embedding-registry", help="Pre-computed CLS embedding registry JSON")
    p.add_argument("--external-cls-dim", type=int, default=0,
                   help="External CLS dimension (e.g. 128 for behavioral embeddings)")

    args = p.parse_args()

    tok = AbilityTokenizer()
    print(f"Device: {DEVICE}")

    model = AbilityActorCriticV3(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=args.entity_encoder_layers,
        external_cls_dim=args.external_cls_dim,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(DEVICE)

    # Load pretrained weights (partial)
    if args.pretrained:
        print(f"Loading pretrained: {args.pretrained}")
        state = torch.load(args.pretrained, map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded: {len(state) - len(unexpected)} params, "
              f"missing: {len(missing)}, unexpected: {len(unexpected)}")

    # Load pretrained entity encoder
    has_entity_encoder = args.pretrained and any(
        k.startswith("entity_encoder.") for k in state.keys()
    ) if args.pretrained else False

    if args.entity_encoder and not has_entity_encoder:
        print(f"Loading entity encoder: {args.entity_encoder}")
        state = torch.load(args.entity_encoder, map_location=DEVICE, weights_only=True)
        enc_state = {}
        for k, v in state.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "entity_encoder.", 1)
                enc_state[new_key] = v
        missing, unexpected = model.load_state_dict(enc_state, strict=False)
        loaded = len(enc_state) - len(unexpected)
        print(f"  Loaded {loaded} entity encoder params")
    elif has_entity_encoder:
        print(f"Using entity encoder from pretrained checkpoint")

    # Freeze transformer
    freeze_transformer = args.freeze_transformer and not args.unfreeze_transformer
    if freeze_transformer:
        for param in model.transformer.parameters():
            param.requires_grad = False
        n_frozen = sum(p.numel() for p in model.transformer.parameters())
        print(f"Froze ability transformer ({n_frozen:,} params)")

    # Freeze entity encoder
    freeze_encoder = (has_entity_encoder or args.entity_encoder) and not args.unfreeze_encoder
    if freeze_encoder:
        for param in model.entity_encoder.parameters():
            param.requires_grad = False
        n_frozen_enc = sum(p.numel() for p in model.entity_encoder.parameters())
        print(f"Froze entity encoder ({n_frozen_enc:,} params)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total:,} total, {n_trainable:,} trainable")

    # Optimizer with separate actor/critic LRs
    critic_names = {"value_head"}
    actor_params = []
    critic_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(cn) for cn in critic_names):
            critic_params.append(param)
        else:
            actor_params.append(param)

    actor_lr = args.lr * args.actor_lr_ratio
    param_groups = [
        {"params": critic_params, "lr": args.lr},
        {"params": actor_params, "lr": actor_lr},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    if args.actor_lr_ratio != 1.0:
        print(f"Actor LR: {actor_lr:.2e} ({args.actor_lr_ratio:.0%} of critic LR {args.lr:.2e})")

    grokfast = None
    if not args.no_grokfast:
        grokfast = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    # Load embedding registry
    embedding_registry = None
    if args.embedding_registry:
        embedding_registry = load_embedding_registry(args.embedding_registry)

    # Load episodes
    print(f"\nLoading episodes from {args.episodes}...")
    episodes = load_episodes(Path(args.episodes))

    # Apply dense reward shaping
    if args.reward_shaping > 0:
        apply_reward_shaping(episodes, scale=args.reward_shaping)
        print(f"  Applied HP-differential reward shaping (scale={args.reward_shaping})")

    use_wins_only = args.bc_wins_only and args.bc_epochs > 0
    all_steps = flatten_steps(episodes, wins_only=use_wins_only)
    if use_wins_only:
        print(f"  {len(episodes)} episodes, {len(all_steps)} steps (wins only)")
    else:
        print(f"  {len(episodes)} episodes, {len(all_steps)} steps")

    # Verify v3 fields
    has_v3 = all(
        s.get("action_type") is not None and s.get("target_idx") is not None
        for s in all_steps[:10]
    )
    if not has_v3:
        print("ERROR: Episodes missing v3 fields (action_type, target_idx)")
        print("Re-generate episodes with V3 policy (actor_critic_v3 weights)")
        sys.exit(1)

    wins = sum(1 for e in episodes if e["outcome"] == "Victory")
    print(f"  Win rate in data: {wins}/{len(episodes)} ({wins/max(len(episodes),1)*100:.1f}%)")

    # Build ability CLS cache
    unit_ability_tokens, cls_cache = build_ability_cls_cache(
        model, episodes, embedding_registry)

    # Behavioral cloning: supervised on oracle (action_type, target_idx)
    if args.bc_epochs > 0:
        print(f"\nBC training ({args.bc_epochs} epochs)...")
        bc_t0 = time.time()

        all_action_types_bc = torch.tensor(
            [s["action_type"] for s in all_steps], dtype=torch.long, device=DEVICE)
        all_target_indices_bc = torch.tensor(
            [s["target_idx"] for s in all_steps], dtype=torch.long, device=DEVICE)

        cls_dim_bc = (embedding_registry["d_model"]
                      if embedding_registry is not None else model.d_model)
        n_bc = len(all_steps)
        model.train()

        for bc_ep in range(args.bc_epochs):
            bc_indices = np.random.permutation(n_bc)
            epoch_loss = 0.0
            epoch_type_acc = 0.0
            epoch_target_acc = 0.0
            n_batches_bc = 0

            for start in range(0, n_bc, args.batch_size):
                end = min(start + args.batch_size, n_bc)
                idx = bc_indices[start:end]
                B_bc = len(idx)

                state = collate_v3_states(all_steps, idx)
                batch_action_types = all_action_types_bc[idx]
                batch_target_indices = all_target_indices_bc[idx]

                type_masks = build_type_masks(all_steps, idx)

                ability_cls_batch = build_ability_cls_batch(
                    all_steps, idx, unit_ability_tokens, cls_cache, cls_dim_bc)

                pointer_output, _ = model(
                    state["entity_features"], state["entity_type_ids"],
                    state["threat_features"], state["entity_mask"], state["threat_mask"],
                    ability_cls_batch,
                    state["position_features"], state["position_mask"],
                )

                # Action type CE loss
                type_logits = pointer_output["type_logits"]
                type_logits = type_logits.masked_fill(~type_masks, -1e9)
                type_loss = F.cross_entropy(type_logits, batch_action_types)

                # Target pointer CE loss (per action type)
                target_loss = torch.tensor(0.0, device=DEVICE)
                n_target = 0
                for at_val, ptr_key in [(0, "attack_ptr"), (1, "move_ptr")]:
                    sel = batch_action_types == at_val
                    if not sel.any():
                        continue
                    ptr_logits = pointer_output[ptr_key]
                    ptr_mask = ptr_logits > -1e8
                    ptr_logits = ptr_logits.masked_fill(~ptr_mask, -1e9)
                    target_loss = target_loss + F.cross_entropy(
                        ptr_logits[sel], batch_target_indices[sel])
                    n_target += 1

                for ab_idx in range(MAX_ABILITIES):
                    at_val = 3 + ab_idx
                    sel = batch_action_types == at_val
                    if not sel.any():
                        continue
                    ab_ptrs = pointer_output["ability_ptrs"]
                    if ab_idx < len(ab_ptrs) and ab_ptrs[ab_idx] is not None:
                        ptr_logits = ab_ptrs[ab_idx]
                        ptr_mask = ptr_logits > -1e8
                        ptr_logits = ptr_logits.masked_fill(~ptr_mask, -1e9)
                        target_loss = target_loss + F.cross_entropy(
                            ptr_logits[sel], batch_target_indices[sel])
                        n_target += 1

                if n_target > 0:
                    target_loss = target_loss / n_target

                loss = type_loss + target_loss

                optimizer.zero_grad()
                loss.backward()
                if grokfast is not None:
                    grokfast.step()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                # Accuracy tracking
                with torch.no_grad():
                    type_pred = type_logits.argmax(-1)
                    type_acc = (type_pred == batch_action_types).float().mean().item()
                    epoch_type_acc += type_acc

                epoch_loss += loss.item()
                n_batches_bc += 1

            avg_loss = epoch_loss / max(n_batches_bc, 1)
            avg_type_acc = epoch_type_acc / max(n_batches_bc, 1) * 100
            print(f"  BC epoch {bc_ep+1}/{args.bc_epochs}: "
                  f"loss={avg_loss:.4f}, type_acc={avg_type_acc:.1f}%")

        bc_elapsed = time.time() - bc_t0
        print(f"  BC done in {bc_elapsed:.1f}s")

    # REINFORCE: reward-weighted policy gradient (no value function needed)
    if args.reinforce_epochs > 0:
        print(f"\nREINFORCE training ({args.reinforce_epochs} epochs)...")
        rf_t0 = time.time()

        all_action_types = torch.tensor(
            [s["action_type"] for s in all_steps], dtype=torch.long, device=DEVICE)
        all_target_indices = torch.tensor(
            [s["target_idx"] for s in all_steps], dtype=torch.long, device=DEVICE)
        # Compute discounted returns per episode (γ-weighted future rewards)
        gamma_rf = args.gamma
        all_returns = []
        ep_start = 0
        for ep in episodes:
            n_steps_ep = len(ep["steps"])
            ep_end = ep_start + n_steps_ep
            # Terminal reward from episode outcome
            terminal_reward = ep["reward"]
            # Compute discounted returns backwards through episode steps
            returns = [0.0] * n_steps_ep
            running_return = terminal_reward
            for t in reversed(range(n_steps_ep)):
                step_r = all_steps[ep_start + t].get("step_reward", 0.0)
                running_return = step_r + gamma_rf * running_return
                returns[t] = running_return
            all_returns.extend(returns)
            ep_start = ep_end
        all_advantages_rf = torch.tensor(all_returns, dtype=torch.float32, device=DEVICE)
        # Normalize advantages (zero mean, unit variance) for stable gradients
        adv_mean = all_advantages_rf.mean()
        adv_std = all_advantages_rf.std().clamp(min=1e-8)
        all_advantages_rf = (all_advantages_rf - adv_mean) / adv_std
        print(f"  Advantages: mean={adv_mean:.4f}, std={adv_std:.4f}")

        cls_dim_rf = (embedding_registry["d_model"]
                      if embedding_registry is not None else model.d_model)
        n_rf = len(all_steps)
        model.train()

        for rf_ep in range(args.reinforce_epochs):
            rf_indices = np.random.permutation(n_rf)
            epoch_pg_loss = 0.0
            epoch_ent = 0.0
            n_batches_rf = 0

            for start in range(0, n_rf, args.batch_size):
                end = min(start + args.batch_size, n_rf)
                idx = rf_indices[start:end]
                B_rf = len(idx)

                state = collate_v3_states(all_steps, idx)
                batch_action_types = all_action_types[idx]
                batch_target_indices = all_target_indices[idx]
                batch_adv = all_advantages_rf[idx]

                type_masks = build_type_masks(all_steps, idx)

                ability_cls_batch = build_ability_cls_batch(
                    all_steps, idx, unit_ability_tokens, cls_cache, cls_dim_rf)

                pointer_output, _ = model(
                    state["entity_features"], state["entity_type_ids"],
                    state["threat_features"], state["entity_mask"], state["threat_mask"],
                    ability_cls_batch,
                    state["position_features"], state["position_mask"],
                )

                action_log_probs, entropy_per_sample = compute_hierarchical_log_prob(
                    pointer_output, batch_action_types, batch_target_indices, type_masks,
                )
                entropy = entropy_per_sample.mean()

                pg_loss = -(batch_adv * action_log_probs).mean()
                loss = pg_loss - args.entropy_coeff * entropy

                optimizer.zero_grad()
                loss.backward()
                if grokfast is not None:
                    grokfast.step()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                epoch_pg_loss += pg_loss.item()
                epoch_ent += entropy.item()
                n_batches_rf += 1

            avg_pg = epoch_pg_loss / max(n_batches_rf, 1)
            avg_ent = epoch_ent / max(n_batches_rf, 1)
            print(f"  REINFORCE epoch {rf_ep+1}/{args.reinforce_epochs}: "
                  f"pg_loss={avg_pg:.4f}, entropy={avg_ent:.4f}")

        rf_elapsed = time.time() - rf_t0
        print(f"  REINFORCE done in {rf_elapsed:.1f}s")

    advantages, returns = None, None
    if args.ppo_epochs > 0:
        # Critic warmup
        if args.critic_warmup_epochs > 0:
            print(f"\nWarming up critic ({args.critic_warmup_epochs} epochs)...")
            _, warmup_returns, _ = compute_gae(
                episodes, model.forward_value,
                gamma=args.gamma, lam=args.gae_lambda,
                chunk_size=args.chunk_size,
            )
            cls_dim_ppo = (embedding_registry["d_model"]
                           if embedding_registry is not None else model.d_model)
            warmup_loss = warmup_critic(
                model, all_steps, warmup_returns, episodes,
                unit_ability_tokens, cls_cache,
                lr=args.lr, epochs=args.critic_warmup_epochs,
                batch_size=args.batch_size,
                cls_dim=cls_dim_ppo,
            )
            print(f"  Critic warmup loss: {warmup_loss:.4f}")

        # GAE
        chunk_msg = f", chunk_size={args.chunk_size}" if args.chunk_size > 0 else " (full episode)"
        print(f"Computing GAE advantages{chunk_msg}...")
        advantages, returns, _ = compute_gae(
            episodes, model.forward_value,
            gamma=args.gamma, lam=args.gae_lambda,
            chunk_size=args.chunk_size,
        )
        print(f"  Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")

    # CSV logging
    log_path = Path(args.log)
    log_file = open(log_path, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow([
        "epoch", "policy_loss", "value_loss", "entropy",
        "approx_kl", "clip_frac", "elapsed_s",
    ])

    if args.ppo_epochs > 0:
        t0 = time.time()
        print(f"\nStarting PPO training ({args.ppo_epochs} epochs)...")

        cls_dim_ppo = (embedding_registry["d_model"]
                       if embedding_registry is not None else model.d_model)
        metrics = ppo_update(
            model, optimizer, grokfast,
            all_steps, advantages, returns, episodes, tok,
            unit_ability_tokens, cls_cache,
            clip_eps=args.clip_eps,
            value_coeff=args.value_coeff,
            entropy_coeff=args.entropy_coeff,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            max_grad_norm=args.max_grad_norm,
            cls_dim=cls_dim_ppo,
        )

        elapsed = time.time() - t0
        print(f"\nPPO update complete in {elapsed:.1f}s")
        print(f"  Policy loss:  {metrics['policy_loss']:.4f}")
        print(f"  Value loss:   {metrics['value_loss']:.4f}")
        print(f"  Entropy:      {metrics['entropy']:.4f}")
        print(f"  Approx KL:    {metrics['approx_kl']:.4f}")
        print(f"  Clip fraction: {metrics['clip_frac']:.3f}")

        writer.writerow([
            1, metrics["policy_loss"], metrics["value_loss"],
            metrics["entropy"], metrics["approx_kl"], metrics["clip_frac"],
            f"{elapsed:.1f}",
        ])
    log_file.close()

    torch.save(model.state_dict(), args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
