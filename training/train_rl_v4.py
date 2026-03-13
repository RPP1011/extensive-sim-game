#!/usr/bin/env python3
"""V4 dual-head training: directional movement (9-way) + combat pointer.

Two independent heads trained simultaneously:
  - Move head: 9-way classification (8 cardinal + stay), class-weighted
  - Combat head: type classification (attack/hold/ab0..7) + pointer for targeting

Usage:
    uv run --with numpy --with torch training/train_rl_v4.py \
        generated/rl_v4_oracle.jsonl \
        --pretrained generated/actor_critic_v3_ph.pt \
        --embedding-registry generated/ability_embedding_registry.json \
        --external-cls-dim 128 \
        -o generated/actor_critic_v4.pt
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_embedding_registry(path: str, device: str = DEVICE) -> dict:
    data = json.load(open(path))
    embs = {}
    for name, vec in data["embeddings"].items():
        embs[name] = torch.tensor(vec, dtype=torch.float32, device=device)
    d_model = data["d_model"]
    print(f"Loaded embedding registry: {len(embs)} abilities, d={d_model}")
    return {"embeddings": embs, "d_model": d_model}


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


class NpzDataset:
    """Pre-tensorized dataset from convert_v4_npz.py. Loads in <1s."""

    def __init__(self, path: str, device: str = DEVICE):
        t0 = time.time()
        data = np.load(path, allow_pickle=False)
        self.entities = torch.tensor(data["entities"], dtype=torch.float32, device=device)
        self.entity_types = torch.tensor(data["entity_types"], dtype=torch.long, device=device)
        self.entity_counts = data["entity_counts"]  # keep as numpy for indexing
        self.mask = torch.tensor(data["mask"], dtype=torch.bool, device=device)
        self.move_dir = torch.tensor(data["move_dir"], dtype=torch.long, device=device)
        self.combat_type = torch.tensor(data["combat_type"], dtype=torch.long, device=device)
        self.target_idx = torch.tensor(data["target_idx"], dtype=torch.long, device=device)
        self.unit_id = data["unit_id"]  # numpy, for CLS lookup
        self.unit_abilities = json.loads(str(data["unit_abilities_json"][0]))
        self.n = len(self.move_dir)
        elapsed = time.time() - t0
        mem_mb = (self.entities.nelement() * 4 + self.entity_types.nelement() * 8 +
                  self.mask.nelement() + self.move_dir.nelement() * 8 +
                  self.combat_type.nelement() * 8 + self.target_idx.nelement() * 8) / 1e6
        print(f"  Loaded {self.n:,} steps from npz in {elapsed:.1f}s ({mem_mb:.0f} MB on {device})")

    def __len__(self):
        return self.n


def flatten_steps(
    episodes: list[dict],
    require_v4: bool = True,
    smart_sample: bool = False,
    hold_keep_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """Flatten episodes to steps, optionally filtering to V4-labelled steps.

    If smart_sample=True, keeps all "interesting" steps (movement, attack,
    ability, action transitions) and only hold_keep_ratio of boring hold+stay
    steps.  This dramatically reduces dataset size while preserving signal.
    """
    rng = np.random.default_rng(seed)
    steps = []
    for ep in episodes:
        prev_action: dict[int, tuple[int, int]] = {}
        for step in ep["steps"]:
            if require_v4 and step.get("move_dir") is None:
                continue
            step["_episode_reward"] = ep["reward"]
            step["_outcome"] = ep.get("outcome", "")

            if smart_sample:
                md = step["move_dir"]
                ct = step["combat_type"]
                uid = step["unit_id"]
                cur = (md, ct)
                is_transition = uid in prev_action and prev_action[uid] != cur
                prev_action[uid] = cur

                is_interesting = (
                    md != 8          # actual movement
                    or ct == 0       # attack
                    or ct >= 2       # ability
                    or is_transition # action changed
                )
                if not is_interesting:
                    # Boring hold+stay — subsample
                    if rng.random() > hold_keep_ratio:
                        continue

            steps.append(step)
    return steps


def collate_v4_states(steps: list[dict], indices) -> dict[str, torch.Tensor]:
    """Collate variable-length game states into padded tensors."""
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
    }


def build_ability_cls_batch(
    steps: list[dict],
    indices,
    unit_ability_tokens: dict[int, list[list[int]]],
    cls_cache: dict[tuple[int, int], torch.Tensor],
    cls_dim: int,
) -> list[torch.Tensor | None]:
    """Build per-ability CLS embedding batch for cross-attention."""
    batch = [steps[i] for i in indices]
    B = len(batch)
    ability_cls: list[torch.Tensor | None] = [None] * MAX_ABILITIES

    for ab_idx in range(MAX_ABILITIES):
        vecs = []
        has_any = False
        for s in batch:
            uid = s["unit_id"]
            tokens = unit_ability_tokens.get(uid, [])
            if ab_idx < len(tokens) and tokens[ab_idx]:
                key = (uid, ab_idx)
                if key in cls_cache:
                    vecs.append(cls_cache[key])
                    has_any = True
                else:
                    vecs.append(torch.zeros(cls_dim, device=DEVICE))
            else:
                vecs.append(torch.zeros(cls_dim, device=DEVICE))
        if has_any:
            ability_cls[ab_idx] = torch.stack(vecs)

    return ability_cls


def build_combat_masks(steps: list[dict], indices) -> torch.Tensor:
    """Build combat type masks [B, 10]: attack(0), hold(1), ab0..7(2..9)."""
    batch = [steps[i] for i in indices]
    B = len(batch)
    masks = torch.zeros(B, NUM_COMBAT_TYPES, dtype=torch.bool, device=DEVICE)

    for bi, s in enumerate(batch):
        mask = s["mask"]
        # attack: any of flat mask[0:3] (attack nearest/weakest/focus)
        masks[bi, 0] = mask[0] or mask[1] or mask[2]
        # hold always valid
        masks[bi, 1] = True
        # abilities: mask[3..10]
        for ab_idx in range(MAX_ABILITIES):
            if 3 + ab_idx < len(mask):
                masks[bi, 2 + ab_idx] = mask[3 + ab_idx]

    return masks


# ---------------------------------------------------------------------------
# BC training
# ---------------------------------------------------------------------------


def bc_epoch(
    model: AbilityActorCriticV4,
    optimizer: torch.optim.Optimizer,
    grokfast: GrokfastEMA | None,
    steps: list[dict],
    unit_ability_tokens: dict[int, list[list[int]]],
    cls_cache: dict[tuple[int, int], torch.Tensor],
    cls_dim: int,
    batch_size: int = 512,
    move_weight: float = 5.0,
    max_grad_norm: float = 0.5,
) -> dict:
    """One epoch of behavioral cloning with dual-head loss."""
    n = len(steps)
    indices = np.random.permutation(n)

    # Compute class weights for movement (inverse frequency)
    move_labels = [s["move_dir"] for s in steps]
    move_counts = [0] * NUM_MOVE_DIRS
    for md in move_labels:
        move_counts[md] += 1
    total = sum(move_counts)
    # Weight = total / (n_classes * count), capped
    move_class_weights = torch.ones(NUM_MOVE_DIRS, device=DEVICE)
    for i in range(NUM_MOVE_DIRS):
        if move_counts[i] > 0:
            w = total / (NUM_MOVE_DIRS * move_counts[i])
            move_class_weights[i] = min(w, move_weight)

    metrics = {
        "move_loss": 0.0, "combat_loss": 0.0, "ptr_loss": 0.0,
        "move_acc": 0.0, "combat_acc": 0.0, "ptr_acc": 0.0,
        "n_updates": 0, "n_ptr_samples": 0,
    }

    model.train()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        B = len(idx)

        state = collate_v4_states(steps, idx)
        ability_cls_batch = build_ability_cls_batch(
            steps, idx, unit_ability_tokens, cls_cache, cls_dim)

        output, _ = model(
            state["entity_features"], state["entity_type_ids"],
            state["threat_features"], state["entity_mask"], state["threat_mask"],
            ability_cls_batch,
            state["position_features"], state["position_mask"],
        )

        # Move direction loss (class-weighted CE)
        move_targets = torch.tensor(
            [steps[i]["move_dir"] for i in idx], dtype=torch.long, device=DEVICE)
        move_loss = F.cross_entropy(
            output["move_logits"], move_targets, weight=move_class_weights)

        # Combat type loss
        combat_targets = torch.tensor(
            [steps[i]["combat_type"] for i in idx], dtype=torch.long, device=DEVICE)
        combat_masks = build_combat_masks(steps, idx)
        combat_logits = output["combat_logits"].masked_fill(~combat_masks, -1e9)
        combat_loss = F.cross_entropy(combat_logits, combat_targets)

        # Pointer loss: only for attack and ability actions (not hold)
        ptr_loss = torch.tensor(0.0, device=DEVICE)
        n_ptr = 0

        # Attack pointer
        atk_sel = combat_targets == 0
        if atk_sel.any():
            atk_targets = torch.tensor(
                [steps[i].get("target_idx", 0) for i in idx], dtype=torch.long, device=DEVICE)
            atk_ptr = output["attack_ptr"]  # [B, N]
            atk_tgt = atk_targets[atk_sel].clamp(0, atk_ptr.shape[1] - 1)
            atk_lp = F.log_softmax(atk_ptr[atk_sel], dim=-1)
            atk_nll = (-atk_lp.gather(1, atk_tgt.unsqueeze(1)).squeeze(1)).clamp(max=10.0)
            ptr_loss = ptr_loss + atk_nll.mean()
            n_ptr += atk_sel.sum().item()

            # Pointer accuracy for attack
            with torch.no_grad():
                pred_ptr = atk_ptr[atk_sel].argmax(dim=-1)
                metrics["ptr_acc"] += (pred_ptr == atk_tgt).float().sum().item()
                metrics["n_ptr_samples"] += atk_sel.sum().item()

        # Ability pointers
        for ab_idx in range(MAX_ABILITIES):
            ab_sel = combat_targets == (2 + ab_idx)
            if not ab_sel.any():
                continue
            ab_ptrs = output.get("ability_ptrs", [])
            if ab_idx < len(ab_ptrs) and ab_ptrs[ab_idx] is not None:
                ab_targets = torch.tensor(
                    [steps[i].get("target_idx", 0) for i in idx],
                    dtype=torch.long, device=DEVICE)
                ab_ptr = ab_ptrs[ab_idx]
                ab_tgt = ab_targets[ab_sel].clamp(0, ab_ptr.shape[1] - 1)
                ab_lp = F.log_softmax(ab_ptr[ab_sel], dim=-1)
                ab_nll = (-ab_lp.gather(1, ab_tgt.unsqueeze(1)).squeeze(1)).clamp(max=10.0)
                ptr_loss = ptr_loss + ab_nll.mean()
                n_ptr += ab_sel.sum().item()

                with torch.no_grad():
                    pred_ptr = ab_ptr[ab_sel].argmax(dim=-1)
                    metrics["ptr_acc"] += (pred_ptr == ab_tgt).float().sum().item()
                    metrics["n_ptr_samples"] += ab_sel.sum().item()

        if n_ptr > 0:
            ptr_loss = ptr_loss / max(1, sum(1 for ab_idx in range(MAX_ABILITIES)
                                             if (combat_targets == 2 + ab_idx).any()) + (1 if atk_sel.any() else 0))

        loss = move_loss + combat_loss + ptr_loss

        optimizer.zero_grad()
        loss.backward()
        if grokfast is not None:
            grokfast.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            move_pred = output["move_logits"].argmax(dim=-1)
            combat_pred = combat_logits.argmax(dim=-1)
            metrics["move_acc"] += (move_pred == move_targets).float().sum().item()
            metrics["combat_acc"] += (combat_pred == combat_targets).float().sum().item()

        metrics["move_loss"] += move_loss.item() * B
        metrics["combat_loss"] += combat_loss.item() * B
        metrics["ptr_loss"] += ptr_loss.item() * B
        metrics["n_updates"] += B

    nu = max(metrics["n_updates"], 1)
    metrics["move_loss"] /= nu
    metrics["combat_loss"] /= nu
    metrics["ptr_loss"] /= nu
    metrics["move_acc"] = 100 * metrics["move_acc"] / nu
    metrics["combat_acc"] = 100 * metrics["combat_acc"] / nu
    if metrics["n_ptr_samples"] > 0:
        metrics["ptr_acc"] = 100 * metrics["ptr_acc"] / metrics["n_ptr_samples"]

    return metrics


def bc_epoch_npz(
    model: AbilityActorCriticV4,
    optimizer: torch.optim.Optimizer,
    grokfast: GrokfastEMA | None,
    ds: NpzDataset,
    cls_cache: dict[tuple[int, int], torch.Tensor],
    cls_dim: int,
    batch_size: int = 512,
    move_weight: float = 5.0,
    max_grad_norm: float = 0.5,
) -> dict:
    """BC epoch on pre-tensorized npz data. No per-step JSON overhead."""
    n = len(ds)
    indices = np.random.permutation(n)

    # Class weights for movement
    move_counts = torch.bincount(ds.move_dir, minlength=NUM_MOVE_DIRS).float()
    total = move_counts.sum()
    move_class_weights = torch.ones(NUM_MOVE_DIRS, device=DEVICE)
    for i in range(NUM_MOVE_DIRS):
        if move_counts[i] > 0:
            w = total / (NUM_MOVE_DIRS * move_counts[i])
            move_class_weights[i] = min(w.item(), move_weight)

    metrics = {
        "move_loss": 0.0, "combat_loss": 0.0, "ptr_loss": 0.0,
        "move_acc": 0.0, "combat_acc": 0.0, "ptr_acc": 0.0,
        "n_updates": 0, "n_ptr_samples": 0,
    }

    # Empty tensors for unused threats/positions
    empty_threats = torch.zeros(1, 1, THREAT_DIM, device=DEVICE)
    empty_threat_mask = torch.ones(1, 1, dtype=torch.bool, device=DEVICE)
    empty_positions = torch.zeros(1, 1, POSITION_DIM, device=DEVICE)
    empty_position_mask = torch.ones(1, 1, dtype=torch.bool, device=DEVICE)

    model.train()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = indices[start:end]
        B = len(idx)

        # Slice pre-tensorized data (already on GPU)
        ent_feat = ds.entities[idx]
        ent_types = ds.entity_types[idx]
        # Build entity mask from counts
        max_e = ent_feat.shape[1]
        counts = ds.entity_counts[idx]
        ent_mask = torch.ones(B, max_e, dtype=torch.bool, device=DEVICE)
        for i, c in enumerate(counts):
            ent_mask[i, :c] = False

        # No threats/positions in V4 data (empty)
        thr_feat = empty_threats.expand(B, -1, -1)
        thr_mask = empty_threat_mask.expand(B, -1)
        pos_feat = empty_positions.expand(B, -1, -1)
        pos_mask = empty_position_mask.expand(B, -1)

        # Build ability CLS batch from cache
        ability_cls: list[torch.Tensor | None] = [None] * MAX_ABILITIES
        for ab_idx in range(MAX_ABILITIES):
            vecs = []
            has_any = False
            for i in idx:
                uid = int(ds.unit_id[i])
                key = (uid, ab_idx)
                if key in cls_cache:
                    vecs.append(cls_cache[key])
                    has_any = True
                else:
                    vecs.append(torch.zeros(cls_dim, device=DEVICE))
            if has_any:
                ability_cls[ab_idx] = torch.stack(vecs)

        output, _ = model(
            ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
            ability_cls, pos_feat, pos_mask,
        )

        move_targets = ds.move_dir[idx]
        move_loss = F.cross_entropy(
            output["move_logits"], move_targets, weight=move_class_weights)

        combat_targets = ds.combat_type[idx]
        # Build combat masks
        combat_masks = torch.zeros(B, NUM_COMBAT_TYPES, dtype=torch.bool, device=DEVICE)
        batch_mask = ds.mask[idx]
        combat_masks[:, 0] = batch_mask[:, 0] | batch_mask[:, 1] | batch_mask[:, 2]
        combat_masks[:, 1] = True
        for ab_i in range(MAX_ABILITIES):
            if 3 + ab_i < batch_mask.shape[1]:
                combat_masks[:, 2 + ab_i] = batch_mask[:, 3 + ab_i]

        combat_logits = output["combat_logits"].masked_fill(~combat_masks, -1e9)
        combat_loss = F.cross_entropy(combat_logits, combat_targets)

        # Pointer loss
        ptr_loss = torch.tensor(0.0, device=DEVICE)
        n_ptr_heads = 0

        atk_sel = combat_targets == 0
        if atk_sel.any():
            atk_targets_b = ds.target_idx[idx]
            atk_ptr = output["attack_ptr"]
            atk_tgt = atk_targets_b[atk_sel].clamp(0, atk_ptr.shape[1] - 1)
            atk_lp = F.log_softmax(atk_ptr[atk_sel], dim=-1)
            atk_nll = (-atk_lp.gather(1, atk_tgt.unsqueeze(1)).squeeze(1)).clamp(max=10.0)
            ptr_loss = ptr_loss + atk_nll.mean()
            n_ptr_heads += 1
            with torch.no_grad():
                pred_ptr = atk_ptr[atk_sel].argmax(dim=-1)
                metrics["ptr_acc"] += (pred_ptr == atk_tgt).float().sum().item()
                metrics["n_ptr_samples"] += atk_sel.sum().item()

        for ab_idx in range(MAX_ABILITIES):
            ab_sel = combat_targets == (2 + ab_idx)
            if not ab_sel.any():
                continue
            ab_ptrs = output.get("ability_ptrs", [])
            if ab_idx < len(ab_ptrs) and ab_ptrs[ab_idx] is not None:
                ab_targets_b = ds.target_idx[idx]
                ab_ptr = ab_ptrs[ab_idx]
                ab_tgt = ab_targets_b[ab_sel].clamp(0, ab_ptr.shape[1] - 1)
                ab_lp = F.log_softmax(ab_ptr[ab_sel], dim=-1)
                ab_nll = (-ab_lp.gather(1, ab_tgt.unsqueeze(1)).squeeze(1)).clamp(max=10.0)
                ptr_loss = ptr_loss + ab_nll.mean()
                n_ptr_heads += 1
                with torch.no_grad():
                    pred_ptr = ab_ptr[ab_sel].argmax(dim=-1)
                    metrics["ptr_acc"] += (pred_ptr == ab_tgt).float().sum().item()
                    metrics["n_ptr_samples"] += ab_sel.sum().item()

        if n_ptr_heads > 0:
            ptr_loss = ptr_loss / n_ptr_heads

        loss = move_loss + combat_loss + ptr_loss

        optimizer.zero_grad()
        loss.backward()
        if grokfast is not None:
            grokfast.step()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            move_pred = output["move_logits"].argmax(dim=-1)
            combat_pred = combat_logits.argmax(dim=-1)
            metrics["move_acc"] += (move_pred == move_targets).float().sum().item()
            metrics["combat_acc"] += (combat_pred == combat_targets).float().sum().item()

        metrics["move_loss"] += move_loss.item() * B
        metrics["combat_loss"] += combat_loss.item() * B
        metrics["ptr_loss"] += ptr_loss.item() * B
        metrics["n_updates"] += B

    nu = max(metrics["n_updates"], 1)
    metrics["move_loss"] /= nu
    metrics["combat_loss"] /= nu
    metrics["ptr_loss"] /= nu
    metrics["move_acc"] = 100 * metrics["move_acc"] / nu
    metrics["combat_acc"] = 100 * metrics["combat_acc"] / nu
    if metrics["n_ptr_samples"] > 0:
        metrics["ptr_acc"] = 100 * metrics["ptr_acc"] / metrics["n_ptr_samples"]

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="V4 dual-head BC training")
    p.add_argument("episodes", help="JSONL episode file with move_dir + combat_type fields")
    p.add_argument("--pretrained", help="Pretrained V3 checkpoint (.pt) to warm-start from")
    p.add_argument("--entity-encoder", help="Pretrained entity encoder (.pt)")
    p.add_argument("-o", "--output", default="generated/actor_critic_v4.pt")
    p.add_argument("--log", help="CSV log file")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--move-weight", type=float, default=5.0,
                   help="Max class weight for movement directions (inverse freq)")
    p.add_argument("--freeze-transformer", action="store_true")
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--no-grokfast", action="store_true")
    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)
    p.add_argument("--embedding-registry", help="Pre-computed CLS embedding registry JSON")
    p.add_argument("--external-cls-dim", type=int, default=0)
    p.add_argument("--wins-only", action="store_true")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Subsample to at most N steps (0 = all)")
    p.add_argument("--smart-sample", action="store_true",
                   help="Keep all interesting steps (move/attack/ability/transitions), "
                        "subsample boring hold+stay to 10%%")
    p.add_argument("--hold-keep-ratio", type=float, default=0.1,
                   help="Fraction of hold+stay steps to keep with --smart-sample")
    args = p.parse_args()

    tok = AbilityTokenizer()

    use_npz = args.episodes.endswith(".npz")

    if use_npz:
        print(f"Loading npz from {args.episodes}...")
        ds = NpzDataset(args.episodes)

        # Distribution summary
        n_moving = int((ds.move_dir != 8).sum())
        n_attack = int((ds.combat_type == 0).sum())
        n_hold = int((ds.combat_type == 1).sum())
        n_ability = int((ds.combat_type >= 2).sum())
        N = len(ds)
        print(f"  Movement: {n_moving}/{N} ({100*n_moving/N:.1f}%) actual, "
              f"{N-n_moving}/{N} ({100*(N-n_moving)/N:.1f}%) stay")
        print(f"  Combat: attack={n_attack} hold={n_hold} abilities={n_ability}")

        # Build CLS cache from registry + unit abilities stored in npz
        embedding_registry = None
        if args.embedding_registry:
            embedding_registry = load_embedding_registry(args.embedding_registry)

        cls_cache: dict[tuple[int, int], torch.Tensor] = {}
        if embedding_registry:
            for uid_str, ab_names in ds.unit_abilities.items():
                uid = int(uid_str)
                for ab_idx, name in enumerate(ab_names):
                    key = (uid, ab_idx)
                    if key in cls_cache:
                        continue
                    lookup = name.replace(" ", "_")
                    if lookup in embedding_registry["embeddings"]:
                        cls_cache[key] = embedding_registry["embeddings"][lookup]
        print(f"  CLS cache: {len(cls_cache)} entries")
        cls_dim = embedding_registry["d_model"] if embedding_registry else args.d_model
        steps = None
        episodes = None
        unit_ability_tokens = None
    else:
        print(f"Loading episodes from {args.episodes}...")
        t0 = time.time()
        episodes = load_episodes(Path(args.episodes))
        print(f"  {len(episodes)} episodes in {time.time()-t0:.1f}s")

        if args.wins_only:
            episodes = [ep for ep in episodes if ep.get("outcome") == "Victory"]
            print(f"  Filtered to {len(episodes)} wins")

        steps = flatten_steps(
            episodes, require_v4=True,
            smart_sample=args.smart_sample,
            hold_keep_ratio=args.hold_keep_ratio,
        )
        if args.max_steps > 0 and len(steps) > args.max_steps:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(steps), size=args.max_steps, replace=False)
            steps = [steps[i] for i in sorted(idx)]
            print(f"  Subsampled to {len(steps)} steps")
        print(f"  {len(steps)} steps with V4 data")
        if len(steps) == 0:
            print("No V4 steps found! Make sure episodes were generated with --policy combined")
            return

        from collections import Counter
        md_dist = Counter(s["move_dir"] for s in steps)
        ct_dist = Counter(s["combat_type"] for s in steps)
        n_moving = sum(v for k, v in md_dist.items() if k != 8)
        print(f"  Movement: {n_moving}/{len(steps)} ({100*n_moving/len(steps):.1f}%) actual, "
              f"{md_dist[8]}/{len(steps)} ({100*md_dist[8]/len(steps):.1f}%) stay")
        print(f"  Combat: attack={ct_dist[0]} hold={ct_dist[1]} abilities={sum(ct_dist[k] for k in range(2,10))}")

        embedding_registry = None
        if args.embedding_registry:
            embedding_registry = load_embedding_registry(args.embedding_registry)

        unit_ability_tokens: dict[int, list[list[int]]] = {}
        cls_cache: dict[tuple[int, int], torch.Tensor] = {}
        for ep in episodes:
            for uid_str, token_lists in ep.get("unit_abilities", {}).items():
                uid = int(uid_str)
                unit_ability_tokens[uid] = token_lists

            if embedding_registry:
                for uid_str, ab_names in ep.get("unit_ability_names", {}).items():
                    uid = int(uid_str)
                    for ab_idx, name in enumerate(ab_names):
                        key = (uid, ab_idx)
                        if key in cls_cache:
                            continue
                        lookup = name.replace(" ", "_")
                        if lookup in embedding_registry["embeddings"]:
                            cls_cache[key] = embedding_registry["embeddings"][lookup]

        print(f"  CLS cache: {len(cls_cache)} entries")
        cls_dim = embedding_registry["d_model"] if embedding_registry else args.d_model
        ds = None

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

    # Load pretrained weights (V3 compatible — shares entity encoder, transformer, cross_attn)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location=DEVICE, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        # Load matching keys (V3→V4 compatible for shared components)
        model_sd = model.state_dict()
        loaded = 0
        skipped = []
        for k, v in sd.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k] = v
                loaded += 1
            else:
                skipped.append(k)
        model.load_state_dict(model_sd)
        print(f"  Loaded {loaded} params from {args.pretrained}")
        if skipped:
            print(f"  Skipped: {skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    if args.entity_encoder:
        ee_ckpt = torch.load(args.entity_encoder, map_location=DEVICE, weights_only=False)
        ee_sd = ee_ckpt.get("model_state_dict", ee_ckpt)
        mapped = {}
        for k, v in ee_sd.items():
            mapped_k = f"entity_encoder.{k}"
            if mapped_k in model.state_dict() and model.state_dict()[mapped_k].shape == v.shape:
                mapped[mapped_k] = v
        if mapped:
            model.load_state_dict(mapped, strict=False)
            print(f"  Loaded {len(mapped)} entity encoder params")

    # Freeze/unfreeze
    if args.freeze_transformer:
        for n, p in model.named_parameters():
            if n.startswith("transformer."):
                p.requires_grad = False

    freeze_encoder = not args.unfreeze_encoder
    if freeze_encoder:
        for n, p in model.named_parameters():
            if n.startswith("entity_encoder."):
                p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {total_params:,} params, {trainable:,} trainable")

    # Optimizer + Grokfast
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98),
    )

    gf = None
    if not args.no_grokfast:
        gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    # CSV log
    log_path = args.log or args.output.replace(".pt", ".csv")
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "move_loss", "combat_loss", "ptr_loss",
        "move_acc", "combat_acc", "ptr_acc", "elapsed_s",
    ])

    # BC training
    t_start = time.time()
    for epoch in range(1, args.bc_epochs + 1):
        if ds is not None:
            m = bc_epoch_npz(
                model, optimizer, gf, ds,
                cls_cache, cls_dim,
                batch_size=args.batch_size,
                move_weight=args.move_weight,
                max_grad_norm=args.max_grad_norm,
            )
        else:
            m = bc_epoch(
                model, optimizer, gf, steps,
                unit_ability_tokens, cls_cache, cls_dim,
                batch_size=args.batch_size,
                move_weight=args.move_weight,
                max_grad_norm=args.max_grad_norm,
            )
        elapsed = time.time() - t_start
        print(f"  Epoch {epoch}/{args.bc_epochs}: "
              f"move_loss={m['move_loss']:.4f} combat_loss={m['combat_loss']:.4f} "
              f"ptr_loss={m['ptr_loss']:.4f} | "
              f"move_acc={m['move_acc']:.1f}% combat_acc={m['combat_acc']:.1f}% "
              f"ptr_acc={m['ptr_acc']:.1f}% | {elapsed:.0f}s")
        csv_writer.writerow([
            epoch, m["move_loss"], m["combat_loss"], m["ptr_loss"],
            m["move_acc"], m["combat_acc"], m["ptr_acc"], f"{elapsed:.1f}",
        ])
        csv_file.flush()

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": args,
    }, args.output)
    print(f"Saved to {args.output}")
    csv_file.close()


if __name__ == "__main__":
    main()
