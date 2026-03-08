#!/usr/bin/env python3
"""PPO training for actor-critic with V2 entity encoder.

Uses variable-length entities + threats from V2 game state format.
Episodes must contain `entities`, `entity_types`, `threats` fields in each step.

Usage:
    uv run --with numpy --with torch training/train_rl_v2.py \
        generated/rl_episodes_v2.jsonl \
        --entity-encoder generated/entity_encoder_pretrained_v4.pt \
        -o generated/actor_critic_v2.pt \
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
    AbilityActorCriticV2,
    NUM_ACTIONS,
    MAX_ABILITIES,
)
from tokenizer import AbilityTokenizer
from grokfast import GrokfastEMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENTITY_DIM = 30
THREAT_DIM = 8


# ---------------------------------------------------------------------------
# Data loading + collation
# ---------------------------------------------------------------------------


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def flatten_steps(episodes: list[dict]) -> list[dict]:
    steps = []
    for ep in episodes:
        for step in ep["steps"]:
            step["_episode_reward"] = ep["reward"]
            steps.append(step)
    return steps


def collate_v2_states(steps: list[dict], indices) -> dict[str, torch.Tensor]:
    """Collate variable-length v2 game states into padded tensors."""
    batch = [steps[i] for i in indices]
    B = len(batch)

    max_ents = max(len(s["entities"]) for s in batch)
    max_threats = max(
        (len(s["threats"]) for s in batch if s.get("threats")),
        default=1,
    )
    max_threats = max(max_threats, 1)  # at least 1 for shape

    ent_feat = torch.zeros(B, max_ents, ENTITY_DIM, device=DEVICE)
    ent_types = torch.zeros(B, max_ents, dtype=torch.long, device=DEVICE)
    ent_mask = torch.ones(B, max_ents, dtype=torch.bool, device=DEVICE)

    thr_feat = torch.zeros(B, max_threats, THREAT_DIM, device=DEVICE)
    thr_mask = torch.ones(B, max_threats, dtype=torch.bool, device=DEVICE)

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

    return {
        "entity_features": ent_feat,
        "entity_type_ids": ent_types,
        "threat_features": thr_feat,
        "entity_mask": ent_mask,
        "threat_mask": thr_mask,
    }


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
    """Compute GAE with V2 value function.

    If chunk_size > 0, splits each episode into chunks and bootstraps with
    V(s) at chunk boundaries. This gives much denser learning signal —
    instead of waiting for a terminal +1/-1 reward 400 steps away, each
    chunk uses the critic's value estimate as pseudo-terminal reward.

    With gamma=0.99 and lam=0.95, the effective horizon per step is
    0.9405^N. At chunk_size=20 steps (100 ticks), the earliest step still
    "sees" 0.9405^20 ≈ 0.29 of the chunk-end signal.
    """
    all_advantages = []
    all_returns = []
    all_values = []

    for ep in episodes:
        steps = ep["steps"]
        if not steps:
            continue

        # Collate all steps in this episode
        idx = list(range(len(steps)))
        state = collate_v2_states(steps, idx)

        with torch.no_grad():
            values = value_fn(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
            ).squeeze(-1).cpu().numpy()

        n = len(steps)
        advantages = np.zeros(n, dtype=np.float32)

        if chunk_size > 0:
            # Chunked GAE: reset GAE at chunk boundaries, bootstrap with V(s)
            for chunk_start in range(0, n, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n)
                gae = 0.0

                for t in reversed(range(chunk_start, chunk_end)):
                    reward = steps[t]["step_reward"]

                    if t == n - 1:
                        # True episode end: add terminal reward
                        reward += ep["reward"]
                        next_value = 0.0
                    elif t == chunk_end - 1:
                        # Chunk boundary: bootstrap with V(next_step)
                        next_value = values[t + 1]
                    else:
                        next_value = values[t + 1]

                    delta = reward + gamma * next_value - values[t]
                    gae = delta + gamma * lam * gae
                    advantages[t] = gae
        else:
            # Original: full-episode GAE
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
# PPO update
# ---------------------------------------------------------------------------


def build_ability_cls_cache(
    model: AbilityActorCriticV2,
    episodes: list[dict],
) -> tuple[dict[int, list[list[int]]], dict[tuple[int, int], torch.Tensor]]:
    """Pre-compute frozen CLS embeddings for all unit abilities."""
    unit_ability_tokens: dict[int, list[list[int]]] = {}
    for ep in episodes:
        for uid_str, tokens_list in ep.get("unit_abilities", {}).items():
            uid = int(uid_str)
            if uid not in unit_ability_tokens:
                unit_ability_tokens[uid] = tokens_list

    cls_cache: dict[tuple[int, int], torch.Tensor] = {}
    for uid, tokens_list in unit_ability_tokens.items():
        for aidx, tokens in enumerate(tokens_list):
            ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
            amask = (ids != 0).float()
            with torch.no_grad():
                cls_emb = model.transformer.cls_embedding(ids, amask)
            cls_cache[(uid, aidx)] = cls_emb.squeeze(0)

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
# Critic warmup
# ---------------------------------------------------------------------------


def warmup_critic(
    model: AbilityActorCriticV2,
    steps: list[dict],
    returns: np.ndarray,
    episodes: list[dict],
    unit_ability_tokens: dict[int, list[list[int]]],
    cls_cache: dict[tuple[int, int], torch.Tensor],
    lr: float = 3e-4,
    epochs: int = 5,
    batch_size: int = 256,
) -> float:
    """Pre-train the value head on GAE returns before PPO, so the critic
    starts with reasonable estimates instead of random noise."""
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

            state = collate_v2_states(steps, idx)
            batch_ret = ret_tensor[idx]

            with torch.no_grad():
                # We only need the entity encoding, not the full forward pass
                pass

            # Full forward to get values (but only backprop through value_head)
            ability_cls_batch = build_ability_cls_batch(
                steps, idx, unit_ability_tokens, cls_cache, model.d_model)

            _, values = model(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
                ability_cls_batch,
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
    model: AbilityActorCriticV2,
    ref_model: AbilityActorCriticV2 | None,
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
    kl_coeff: float = 0.0,
    bc_coeff: float = 0.0,
    ppo_epochs: int = 4,
    batch_size: int = 256,
    max_grad_norm: float = 0.5,
) -> dict:
    n = len(steps)
    if n == 0:
        return {}

    actions = torch.tensor([s["action"] for s in steps], dtype=torch.long, device=DEVICE)
    old_log_probs = torch.tensor([s["log_prob"] for s in steps], dtype=torch.float, device=DEVICE)
    masks = torch.tensor([s["mask"] for s in steps], dtype=torch.bool, device=DEVICE)
    adv_tensor = torch.tensor(advantages, dtype=torch.float, device=DEVICE)
    ret_tensor = torch.tensor(returns, dtype=torch.float, device=DEVICE)

    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    model.train()

    # Pre-compute reference logits for KL/BC if needed
    ref_cls_cache: dict[tuple[int, int], torch.Tensor] | None = None
    if ref_model is not None and (kl_coeff > 0 or bc_coeff > 0):
        _, ref_cls_cache = build_ability_cls_cache(ref_model, episodes)

    metrics = {
        "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
        "approx_kl": 0.0, "clip_frac": 0.0, "kl_ref": 0.0,
        "bc_loss": 0.0, "n_updates": 0,
    }

    for epoch in range(ppo_epochs):
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            B = len(idx)

            # Collate v2 game states
            state = collate_v2_states(steps, idx)

            batch_actions = actions[idx]
            batch_old_lp = old_log_probs[idx]
            batch_masks = masks[idx]
            batch_adv = adv_tensor[idx]
            batch_ret = ret_tensor[idx]

            ability_cls_batch = build_ability_cls_batch(
                steps, idx, unit_ability_tokens, cls_cache, model.d_model)

            # Forward pass
            logits, values = model(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
                ability_cls_batch,
            )
            values = values.squeeze(-1)

            logits = logits.masked_fill(~batch_masks, -1e9)
            log_probs = F.log_softmax(logits, dim=-1)
            action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(-1).mean()

            ratio = (action_log_probs - batch_old_lp).exp()
            surr1 = ratio * batch_adv
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, batch_ret)
            loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            # KL penalty against frozen reference policy
            kl_ref_val = 0.0
            bc_loss_val = 0.0
            if ref_model is not None and (kl_coeff > 0 or bc_coeff > 0):
                with torch.no_grad():
                    ref_ability_cls = build_ability_cls_batch(
                        steps, idx, unit_ability_tokens, ref_cls_cache, ref_model.d_model)
                    ref_logits, _ = ref_model(
                        state["entity_features"], state["entity_type_ids"],
                        state["threat_features"], state["entity_mask"], state["threat_mask"],
                        ref_ability_cls,
                    )
                    ref_logits = ref_logits.masked_fill(~batch_masks, -1e9)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_probs = F.softmax(ref_logits, dim=-1)

                if kl_coeff > 0:
                    # KL(π_ref || π_current) — penalize diverging from reference
                    kl_div = F.kl_div(log_probs, ref_probs, reduction="batchmean")
                    loss = loss + kl_coeff * kl_div
                    kl_ref_val = kl_div.item()

                if bc_coeff > 0:
                    # BC loss: cross-entropy with reference policy's actions
                    # Use soft targets (reference distribution)
                    bc_loss = -(ref_probs * log_probs).sum(-1).mean()
                    loss = loss + bc_coeff * bc_loss
                    bc_loss_val = bc_loss.item()

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
            metrics["kl_ref"] += kl_ref_val
            metrics["bc_loss"] += bc_loss_val
            metrics["n_updates"] += 1

    nu = max(metrics["n_updates"], 1)
    for k in ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "kl_ref", "bc_loss"]:
        metrics[k] /= nu

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="PPO training for V2 actor-critic")
    p.add_argument("episodes", help="JSONL episode file with v2 game state fields")
    p.add_argument("--pretrained", help="Phase 2 decision checkpoint (.pt)")
    p.add_argument("--entity-encoder", help="Pretrained V2 entity encoder (.pt)")
    p.add_argument("-o", "--output", default="generated/actor_critic_v2.pt")
    p.add_argument("--log", default="generated/actor_critic_v2.csv")

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)

    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--chunk-size", type=int, default=20,
                    help="GAE chunk size in steps (0=full episode). "
                         "With step_interval=5, chunk_size=20 = 100 ticks.")
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-coeff", type=float, default=0.5)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    p.add_argument("--kl-coeff", type=float, default=0.0,
                    help="KL penalty coefficient against frozen reference policy")
    p.add_argument("--bc-coeff", type=float, default=0.0,
                    help="Behavioral cloning coefficient (distill from reference)")
    p.add_argument("--critic-warmup-epochs", type=int, default=0,
                    help="Pre-train value head on returns before PPO (0=disabled)")

    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)
    p.add_argument("--no-grokfast", action="store_true")

    p.add_argument("--actor-lr-ratio", type=float, default=1.0,
                    help="Actor LR = lr * actor_lr_ratio (e.g. 0.05 = 5%% of critic LR)")

    p.add_argument("--freeze-transformer", action="store_true", default=True)
    p.add_argument("--unfreeze-transformer", action="store_true")
    p.add_argument("--unfreeze-encoder", action="store_true",
                    help="Unfreeze entity encoder (default: frozen when --entity-encoder provided)")

    args = p.parse_args()

    tok = AbilityTokenizer()
    print(f"Device: {DEVICE}")

    model = AbilityActorCriticV2(
        vocab_size=tok.vocab_size,
        entity_encoder_layers=args.entity_encoder_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(DEVICE)

    # Load pretrained ability transformer weights (partial)
    if args.pretrained:
        print(f"Loading pretrained decision model: {args.pretrained}")
        state = torch.load(args.pretrained, map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded: {len(state) - len(unexpected)} params, missing: {len(missing)}, unexpected: {len(unexpected)}")

    # Load pretrained V2 entity encoder — but skip if pretrained checkpoint
    # already contains entity_encoder weights (they may have diverged during
    # fine-tuning, and overwriting would regress the model).
    has_entity_encoder = args.pretrained and any(
        k.startswith("entity_encoder.") for k in state.keys()
    ) if args.pretrained else False

    if args.entity_encoder and not has_entity_encoder:
        print(f"Loading V2 entity encoder: {args.entity_encoder}")
        state = torch.load(args.entity_encoder, map_location=DEVICE, weights_only=True)
        enc_state = {}
        for k, v in state.items():
            # Map from EntityEncoderV2Pretraining → AbilityActorCriticV2
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "entity_encoder.", 1)
                enc_state[new_key] = v
        missing, unexpected = model.load_state_dict(enc_state, strict=False)
        loaded = len(enc_state) - len(unexpected)
        print(f"  Loaded {loaded} entity encoder params")
    elif has_entity_encoder:
        print(f"Using entity encoder from pretrained checkpoint (skipping standalone {args.entity_encoder})")

    # Freeze transformer
    freeze_transformer = args.freeze_transformer and not args.unfreeze_transformer
    if freeze_transformer:
        for param in model.transformer.parameters():
            param.requires_grad = False
        n_frozen = sum(p.numel() for p in model.transformer.parameters())
        print(f"Froze ability transformer ({n_frozen:,} params)")

    # Freeze entity encoder — when pretrained or standalone encoder provided
    freeze_encoder = has_entity_encoder or (args.entity_encoder and not args.unfreeze_encoder)
    if freeze_encoder:
        for param in model.entity_encoder.parameters():
            param.requires_grad = False
        n_frozen_enc = sum(p.numel() for p in model.entity_encoder.parameters())
        print(f"Froze entity encoder ({n_frozen_enc:,} params)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total:,} total, {n_trainable:,} trainable")

    # Split parameters into actor vs critic groups with different LRs
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

    # Create frozen reference model for KL/BC regularization
    ref_model = None
    if args.kl_coeff > 0 or args.bc_coeff > 0:
        import copy
        ref_model = AbilityActorCriticV2(
            vocab_size=tok.vocab_size,
            entity_encoder_layers=args.entity_encoder_layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ).to(DEVICE)
        # Copy current (warmstart) weights as the reference
        ref_model.load_state_dict(model.state_dict())
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"Created frozen reference model for KL({args.kl_coeff}) + BC({args.bc_coeff})")

    grokfast = None
    if not args.no_grokfast:
        grokfast = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    # Load episodes
    print(f"\nLoading episodes from {args.episodes}...")
    episodes = load_episodes(Path(args.episodes))
    all_steps = flatten_steps(episodes)
    print(f"  {len(episodes)} episodes, {len(all_steps)} steps")

    # Verify v2 fields exist
    has_v2 = all(s.get("entities") is not None for s in all_steps[:10])
    if not has_v2:
        print("ERROR: Episodes missing v2 fields (entities/entity_types/threats)")
        print("Re-generate episodes with updated transformer-rl generate")
        sys.exit(1)

    wins = sum(1 for e in episodes if e["outcome"] == "Victory")
    print(f"  Win rate in data: {wins}/{len(episodes)} ({wins/max(len(episodes),1)*100:.1f}%)")

    # Build ability CLS cache
    unit_ability_tokens, cls_cache = build_ability_cls_cache(model, episodes)

    # Critic warmup: pre-train value head before PPO
    if args.critic_warmup_epochs > 0:
        print(f"\nWarming up critic ({args.critic_warmup_epochs} epochs)...")
        # First compute returns with current (random) critic for targets
        _, warmup_returns, _ = compute_gae(
            episodes, model.forward_value,
            gamma=args.gamma, lam=args.gae_lambda,
            chunk_size=args.chunk_size,
        )
        warmup_loss = warmup_critic(
            model, all_steps, warmup_returns, episodes,
            unit_ability_tokens, cls_cache,
            lr=args.lr, epochs=args.critic_warmup_epochs,
            batch_size=args.batch_size,
        )
        print(f"  Critic warmup loss: {warmup_loss:.4f}")

    # GAE (recompute with warmed-up critic if applicable)
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
        "approx_kl", "clip_frac", "kl_ref", "bc_loss", "elapsed_s",
    ])

    t0 = time.time()
    print(f"\nStarting PPO training ({args.ppo_epochs} epochs)...")

    metrics = ppo_update(
        model, ref_model, optimizer, grokfast,
        all_steps, advantages, returns, episodes, tok,
        unit_ability_tokens, cls_cache,
        clip_eps=args.clip_eps,
        value_coeff=args.value_coeff,
        entropy_coeff=args.entropy_coeff,
        kl_coeff=args.kl_coeff,
        bc_coeff=args.bc_coeff,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        max_grad_norm=args.max_grad_norm,
    )

    elapsed = time.time() - t0
    print(f"\nPPO update complete in {elapsed:.1f}s")
    print(f"  Policy loss:  {metrics['policy_loss']:.4f}")
    print(f"  Value loss:   {metrics['value_loss']:.4f}")
    print(f"  Entropy:      {metrics['entropy']:.4f}")
    print(f"  Approx KL:    {metrics['approx_kl']:.4f}")
    print(f"  Clip fraction: {metrics['clip_frac']:.3f}")
    if args.kl_coeff > 0:
        print(f"  KL vs ref:    {metrics['kl_ref']:.4f}")
    if args.bc_coeff > 0:
        print(f"  BC loss:      {metrics['bc_loss']:.4f}")

    writer.writerow([
        1, metrics["policy_loss"], metrics["value_loss"],
        metrics["entropy"], metrics["approx_kl"], metrics["clip_frac"],
        metrics["kl_ref"], metrics["bc_loss"],
        f"{elapsed:.1f}",
    ])
    log_file.close()

    torch.save(model.state_dict(), args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
