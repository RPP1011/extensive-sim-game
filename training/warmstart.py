#!/usr/bin/env python3
"""Supervised warmstart: distill combined eval+student policy into transformer actor-critic.

Trains via cross-entropy on expert action labels (log_prob=0.0 marks expert data).
Uses the same model architecture as PPO so the checkpoint can be used directly.

Usage:
    uv run --with numpy --with torch training/warmstart.py \
        generated/distill_combined_hvh.jsonl \
        --entity-encoder generated/entity_encoder_pretrained_v4.pt \
        -o generated/actor_critic_v2_warmstart.pt \
        --epochs 20
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
from model import AbilityActorCriticV2, NUM_ACTIONS, MAX_ABILITIES
from tokenizer import AbilityTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENTITY_DIM = 30
THREAT_DIM = 8


def load_steps(path: Path) -> tuple[list[dict], list[dict]]:
    episodes = []
    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            episodes.append(ep)
            for s in ep["steps"]:
                steps.append(s)
    return episodes, steps


def collate_v2_states(steps: list[dict], indices) -> dict[str, torch.Tensor]:
    batch = [steps[i] for i in indices]
    B = len(batch)

    max_ents = max(len(s["entities"]) for s in batch)
    max_threats = max(
        (len(s["threats"]) for s in batch if s.get("threats")),
        default=1,
    )
    max_threats = max(max_threats, 1)

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


def main():
    p = argparse.ArgumentParser(description="Supervised warmstart for actor-critic")
    p.add_argument("episodes", help="JSONL expert episodes (from --policy combined)")
    p.add_argument("--pretrained", help="Pretrained transformer checkpoint (.pt)")
    p.add_argument("--entity-encoder", help="Pretrained V2 entity encoder (.pt)")
    p.add_argument("-o", "--output", default="generated/actor_critic_v2_warmstart.pt")
    p.add_argument("--log", default="generated/warmstart.csv")

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--entity-encoder-layers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-split", type=float, default=0.1)

    p.add_argument("--unfreeze-transformer", action="store_true",
                    help="Finetune ability transformer (default: frozen)")
    p.add_argument("--unfreeze-encoder", action="store_true",
                    help="Finetune entity encoder (default: frozen)")

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

    # Load pretrained ability transformer
    if args.pretrained:
        print(f"Loading pretrained: {args.pretrained}")
        state = torch.load(args.pretrained, map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded: {len(state) - len(unexpected)} params, "
              f"missing: {len(missing)}, unexpected: {len(unexpected)}")

    # Load pretrained V2 entity encoder
    if args.entity_encoder:
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

    # Freeze pretrained components
    if not args.unfreeze_transformer:
        for param in model.transformer.parameters():
            param.requires_grad = False
        n_frozen = sum(p.numel() for p in model.transformer.parameters())
        print(f"Froze ability transformer ({n_frozen:,} params)")

    if args.entity_encoder and not args.unfreeze_encoder:
        for param in model.entity_encoder.parameters():
            param.requires_grad = False
        n_frozen_enc = sum(p.numel() for p in model.entity_encoder.parameters())
        print(f"Froze entity encoder ({n_frozen_enc:,} params)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_total:,} total, {n_trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    # Load data
    print(f"\nLoading episodes from {args.episodes}...")
    episodes, all_steps = load_steps(Path(args.episodes))
    print(f"  {len(episodes)} episodes, {len(all_steps)} steps")

    wins = sum(1 for e in episodes if e["outcome"] == "Victory")
    print(f"  Win rate in data: {wins}/{len(episodes)} ({wins/max(len(episodes),1)*100:.1f}%)")

    # Cache ability CLS embeddings
    unit_ability_tokens: dict[int, list[list[int]]] = {}
    for ep in episodes:
        for uid_str, tokens_list in ep.get("unit_abilities", {}).items():
            uid = int(uid_str)
            if uid not in unit_ability_tokens:
                unit_ability_tokens[uid] = tokens_list

    cls_cache: dict[tuple[int, int], torch.Tensor] = {}
    model.eval()
    with torch.no_grad():
        for uid, tokens_list in unit_ability_tokens.items():
            for aidx, tokens in enumerate(tokens_list):
                ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
                amask = (ids != 0).float()
                cls_emb = model.transformer.cls_embedding(ids, amask)
                cls_cache[(uid, aidx)] = cls_emb.squeeze(0)

    # Train/val split
    n = len(all_steps)
    indices = np.random.permutation(n)
    n_val = max(int(n * args.val_split), 1)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

    actions = torch.tensor([s["action"] for s in all_steps], dtype=torch.long, device=DEVICE)
    masks = torch.tensor([s["mask"] for s in all_steps], dtype=torch.bool, device=DEVICE)

    # Action distribution in training data
    from collections import Counter
    action_counts = Counter(s["action"] for s in all_steps)
    labels = {0:'AtkNear',1:'AtkWeak',2:'AtkFocus',11:'MvTo',12:'MvAway',13:'Hold'}
    for i in range(3,11): labels[i] = f'Abl{i-3}'
    print("\n  Action distribution:")
    for a in sorted(action_counts.keys()):
        print(f"    {labels.get(a,a):8s}: {action_counts[a]:6d} ({100*action_counts[a]/n:.1f}%)")

    def build_ability_cls_batch(idx_batch):
        B = len(idx_batch)
        ability_cls_batch: list[torch.Tensor | None] = [None] * MAX_ABILITIES
        per_step_cls = []
        for ii in idx_batch:
            step = all_steps[ii]
            uid = step["unit_id"]
            slot_cls = []
            for aidx in range(MAX_ABILITIES):
                slot_cls.append(cls_cache.get((uid, aidx)))
            per_step_cls.append(slot_cls)

        for aidx in range(MAX_ABILITIES):
            valid = []
            valid_indices = []
            for bi, step_cls in enumerate(per_step_cls):
                if aidx < len(step_cls) and step_cls[aidx] is not None:
                    valid.append(step_cls[aidx])
                    valid_indices.append(bi)
            if valid:
                stacked = torch.stack(valid)
                full = torch.zeros(B, model.d_model, device=DEVICE)
                for vi, bi in enumerate(valid_indices):
                    full[bi] = stacked[vi]
                ability_cls_batch[aidx] = full
        return ability_cls_batch

    # CSV logging
    log_file = open(args.log, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "elapsed_s"])

    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(len(train_idx))
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for start in range(0, len(perm), args.batch_size):
            end = min(start + args.batch_size, len(perm))
            idx = train_idx[perm[start:end]]
            B = len(idx)

            state = collate_v2_states(all_steps, idx)
            batch_actions = actions[idx]
            batch_masks = masks[idx]

            ability_cls_batch = build_ability_cls_batch(idx)

            logits, _ = model(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"], state["threat_mask"],
                ability_cls_batch,
            )
            logits = logits.masked_fill(~batch_masks, -1e9)

            loss = F.cross_entropy(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch_actions).sum().item()
            total_loss += loss.item() * B
            total_samples += B

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                end = min(start + args.batch_size, len(val_idx))
                idx = val_idx[start:end]
                B = len(idx)

                state = collate_v2_states(all_steps, idx)
                batch_actions = actions[idx]
                batch_masks = masks[idx]

                ability_cls_batch = build_ability_cls_batch(idx)

                logits, _ = model(
                    state["entity_features"], state["entity_type_ids"],
                    state["threat_features"], state["entity_mask"], state["threat_mask"],
                    ability_cls_batch,
                )
                logits = logits.masked_fill(~batch_masks, -1e9)

                loss = F.cross_entropy(logits, batch_actions)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_actions).sum().item()
                val_loss += loss.item() * B

        val_loss /= max(len(val_idx), 1)
        val_acc = val_correct / max(len(val_idx), 1)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} ({elapsed:.1f}s)")

        writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
                         f"{val_loss:.4f}", f"{val_acc:.4f}", f"{elapsed:.1f}"])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output)

    log_file.close()
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
