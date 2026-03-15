#!/usr/bin/env python3
"""Behavioral cloning for full V5 actor-critic with pointer heads.

Trains the complete AbilityActorCriticV5 including CombatPointerHeadV5
(attack_query, pointer_key, ability_queries) via supervised learning.

Targets: move_dir (9-class CE) + combat_type (10-class CE) + target_idx (pointer CE)

This produces a checkpoint where ALL weights are trained — no random pointer
components that cause the 14% → 20% gap in gameplay evaluation.

Usage:
    uv run --with numpy --with torch python training/train_bc_v5.py \
        generated/v5_full_dataset.npz \
        --encoder-ckpt generated/entity_encoder_v5_full.pt \
        -o generated/actor_critic_v5_bc_full.pt \
        --max-steps 30000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (
    AbilityActorCriticV5, NUM_MOVE_DIRS, NUM_COMBAT_TYPES, MAX_ABILITIES,
    V5_DEFAULT_D, V5_DEFAULT_HEADS,
)
from tokenizer import AbilityTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser(description="V5 behavioral cloning (full model with pointers)")
    p.add_argument("data", help="npz file from convert_v5_npz.py")
    p.add_argument("-o", "--output", default="generated/actor_critic_v5_bc_full.pt")
    p.add_argument("--encoder-ckpt", help="Pretrained encoder checkpoint")
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--unfreeze-step", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-unfrozen", type=float, default=5e-5)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--use-teacher", action="store_true",
                   help="Train on teacher (GOAP) labels instead of model's own actions (DAgger)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.data}...")
    d = np.load(args.data)
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]

    ent_feat = torch.from_numpy(d["ent_feat"]).to(DEVICE)
    ent_types = torch.from_numpy(d["ent_types"]).long().to(DEVICE)
    ent_mask = torch.from_numpy(d["ent_mask"]).bool().to(DEVICE)
    thr_feat = torch.from_numpy(d["thr_feat"]).to(DEVICE)
    thr_mask = torch.from_numpy(d["thr_mask"]).bool().to(DEVICE)
    pos_feat = torch.from_numpy(d["pos_feat"]).to(DEVICE)
    pos_mask = torch.from_numpy(d["pos_mask"]).bool().to(DEVICE)
    agg_feat = torch.from_numpy(d["agg_feat"]).to(DEVICE)
    move_dir = torch.from_numpy(d["move_dir"]).long().to(DEVICE)
    combat_type = torch.from_numpy(d["combat_type"]).long().to(DEVICE)
    target_idx = torch.from_numpy(d["target_idx"]).long().to(DEVICE)

    # DAgger: use teacher labels if available and requested
    if args.use_teacher and "teacher_move" in d:
        print("  Using TEACHER labels (DAgger mode)")
        move_dir = torch.from_numpy(d["teacher_move"]).long().to(DEVICE)
        combat_type = torch.from_numpy(d["teacher_combat"]).long().to(DEVICE)
        target_idx = torch.from_numpy(d["teacher_target"]).long().to(DEVICE)

    N = ent_feat.shape[0]
    print(f"  {N} samples, train={len(train_idx)}, val={len(val_idx)}")

    # Full V5 model with pointer heads
    tok = AbilityTokenizer()
    model = AbilityActorCriticV5(
        vocab_size=tok.vocab_size, d_model=V5_DEFAULT_D,
        n_heads=V5_DEFAULT_HEADS, n_layers=4,
        entity_encoder_layers=4, external_cls_dim=128,
    ).to(DEVICE)

    if args.encoder_ckpt:
        print(f"Loading pretrained encoder from {args.encoder_ckpt}...")
        ckpt = torch.load(args.encoder_ckpt, map_location=DEVICE, weights_only=False)
        encoder_sd = {k[len("encoder."):]: v for k, v in ckpt["model_state_dict"].items()
                      if k.startswith("encoder.")}
        model.entity_encoder.load_state_dict(encoder_sd, strict=True)
        print(f"  Loaded encoder ({len(encoder_sd)} params)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    # Combat type weights (inverse frequency)
    combat_counts = torch.zeros(NUM_COMBAT_TYPES)
    for i in train_idx:
        combat_counts[combat_type[i]] += 1
    combat_weights = torch.zeros(NUM_COMBAT_TYPES, device=DEVICE)
    for c in range(NUM_COMBAT_TYPES):
        combat_weights[c] = min(len(train_idx) / (NUM_COMBAT_TYPES * max(combat_counts[c], 1)), 20.0)
    print(f"Combat weights: {[f'{w:.1f}' for w in combat_weights.tolist()]}")

    # Freeze encoder + transformer initially
    encoder_frozen = args.unfreeze_step > 0
    if encoder_frozen:
        for p in model.entity_encoder.parameters():
            p.requires_grad = False
        for p in model.transformer.parameters():
            p.requires_grad = False
        print(f"Encoder+transformer frozen until step {args.unfreeze_step}")

    optimizer = torch.optim.AdamW(
        filter(lambda pp: pp.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.1, betas=(0.9, 0.98),
    )

    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    train_perm = np.random.permutation(train_idx)
    train_ptr = 0

    print(f"\nTraining for {args.max_steps} steps, batch={args.batch_size}")

    while step < args.max_steps:
        if encoder_frozen and step >= args.unfreeze_step:
            for p in model.entity_encoder.parameters():
                p.requires_grad = True
            encoder_frozen = False
            optimizer = torch.optim.AdamW(
                filter(lambda pp: pp.requires_grad, model.parameters()),
                lr=args.lr_unfrozen, weight_decay=0.1, betas=(0.9, 0.98),
            )
            print(f"  === Encoder unfrozen at step {step}, lr={args.lr_unfrozen} ===")

        model.train()

        if train_ptr + args.batch_size > len(train_perm):
            train_perm = np.random.permutation(train_idx)
            train_ptr = 0
        idx = train_perm[train_ptr:train_ptr + args.batch_size]
        train_ptr += args.batch_size

        # Forward through full model including CfC (matches GPU inference path)
        output, _ = model(
            ent_feat[idx], ent_types[idx], thr_feat[idx],
            ent_mask[idx], thr_mask[idx],
            [None] * MAX_ABILITIES,
            pos_feat[idx], pos_mask[idx],
            aggregate_features=agg_feat[idx],
        )

        # Move loss
        loss_move = F.cross_entropy(output["move_logits"], move_dir[idx])

        # Combat type loss (weighted)
        loss_combat = F.cross_entropy(output["combat_logits"], combat_type[idx], weight=combat_weights)

        # Pointer loss for attack actions
        loss_ptr = torch.tensor(0.0, device=DEVICE)
        attack_mask = combat_type[idx] == 0
        if attack_mask.any():
            atk_ptr = output["attack_ptr"]
            valid = atk_ptr > -1e8
            atk_logits = atk_ptr.masked_fill(~valid, -1e9)
            n_tok = atk_logits.shape[1]
            tgt = target_idx[idx].clamp(0, n_tok - 1)
            loss_ptr = F.cross_entropy(atk_logits[attack_mask], tgt[attack_mask])

        loss = loss_move + loss_combat + 0.5 * loss_ptr

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                vm_ok = vc_ok = vp_ok = vp_n = vn = 0
                vl_sum = 0.0
                for vs in range(0, len(val_idx), args.batch_size):
                    vi = val_idx[vs:vs + args.batch_size]
                    vo, _ = model(
                        ent_feat[vi], ent_types[vi], thr_feat[vi],
                        ent_mask[vi], thr_mask[vi], [None] * MAX_ABILITIES,
                        pos_feat[vi], pos_mask[vi],
                        aggregate_features=agg_feat[vi],
                    )
                    vl_sum += (F.cross_entropy(vo["move_logits"], move_dir[vi]) +
                               F.cross_entropy(vo["combat_logits"], combat_type[vi])).item() * len(vi)
                    vm_ok += (vo["move_logits"].argmax(-1) == move_dir[vi]).sum().item()
                    vc_ok += (vo["combat_logits"].argmax(-1) == combat_type[vi]).sum().item()
                    vatk = combat_type[vi] == 0
                    if vatk.any():
                        vptr = vo["attack_ptr"].masked_fill(vo["attack_ptr"] < -1e8, -1e9)
                        vt = target_idx[vi].clamp(0, vptr.shape[1] - 1)
                        vp_ok += (vptr[vatk].argmax(-1) == vt[vatk]).sum().item()
                        vp_n += vatk.sum().item()
                    vn += len(vi)

            vl = vl_sum / max(vn, 1)
            ma = 100 * vm_ok / max(vn, 1)
            ca = 100 * vc_ok / max(vn, 1)
            pa = 100 * vp_ok / max(vp_n, 1)
            tag = " [frozen]" if encoder_frozen else ""
            print(f"  step {step:6d} | val={vl:.3f} move={ma:.1f}% combat={ca:.1f}% ptr={pa:.1f}%{tag} | {time.time()-t0:.0f}s")

            if vl < best_val_loss:
                best_val_loss = vl
                torch.save({"model_state_dict": model.state_dict(), "step": step,
                             "val_loss": vl, "move_acc": ma, "combat_acc": ca, "ptr_acc": pa,
                             "args": vars(args)}, args.output)

    print(f"\nBest val_loss: {best_val_loss:.3f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
