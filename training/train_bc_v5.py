#!/usr/bin/env python3
"""Behavioral cloning for V5 actor-critic.

Trains AbilityActorCriticV5 to imitate the tactical AI's decisions via
supervised learning on move_dir + combat_type + target_idx labels.

Two phases:
  1. Frozen encoder: only train decision heads (move, combat, pointer)
  2. Unfrozen encoder: fine-tune everything at lower LR

Usage:
    uv run --with numpy --with torch python training/train_bc_v5.py \
        generated/v5_stage0a.npz \
        --encoder-ckpt generated/entity_encoder_v5_pretrained.pt \
        -o generated/actor_critic_v5_bc.pt \
        --max-steps 20000
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

sys.path.insert(0, str(Path(__file__).parent))
from model import (
    AbilityActorCriticV5, EntityEncoderV5, LatentInterface,
    NUM_MOVE_DIRS, NUM_COMBAT_TYPES, MAX_ABILITIES,
    V5_DEFAULT_D, V5_DEFAULT_HEADS, V5_DEFAULT_LATENTS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser(description="V5 behavioral cloning")
    p.add_argument("data", help="npz file from convert_v5_npz.py")
    p.add_argument("-o", "--output", default="generated/actor_critic_v5_bc.pt")
    p.add_argument("--encoder-ckpt", help="Pretrained encoder checkpoint")
    p.add_argument("--d-model", type=int, default=V5_DEFAULT_D)
    p.add_argument("--n-heads", type=int, default=V5_DEFAULT_HEADS)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--unfreeze-step", type=int, default=10000,
                    help="Step at which to unfreeze encoder (0=never freeze)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-unfrozen", type=float, default=5e-5)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
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
    move_vec = torch.from_numpy(d["move_vec"]).to(DEVICE) if "move_vec" in d else None
    combat_type = torch.from_numpy(d["combat_type"]).long().to(DEVICE)
    target_idx = torch.from_numpy(d["target_idx"]).long().to(DEVICE)

    N = ent_feat.shape[0]
    print(f"  {N} samples, train={len(train_idx)}, val={len(val_idx)}")

    # Model — we only need the entity encoder + decision heads for BC
    # (no transformer encoder or CLS embeddings needed, those are for ability selection)
    from pretrain_encoder_v5 import V5EncoderPretraining

    # Build a lightweight BC model: encoder + move head + combat type head
    model = V5BCModel(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(DEVICE)

    # Load pretrained encoder weights
    if args.encoder_ckpt:
        print(f"Loading pretrained encoder from {args.encoder_ckpt}...")
        ckpt = torch.load(args.encoder_ckpt, map_location=DEVICE, weights_only=False)
        # The pretraining model wraps EntityEncoderV5 as self.encoder,
        # so keys are "encoder.entity_proj.weight", "encoder.encoder.layers.0...", etc.
        prefix = "encoder."
        encoder_sd = {k[len(prefix):]: v
                      for k, v in ckpt["model_state_dict"].items()
                      if k.startswith(prefix)}
        model.encoder.load_state_dict(encoder_sd, strict=True)
        print(f"  Loaded encoder weights (step={ckpt.get('step', '?')})")

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    print(f"Model: {n_params:,} params ({n_encoder:,} encoder, {n_params - n_encoder:,} heads)")

    # Compute class weights for combat type (inverse frequency, capped)
    combat_counts = torch.zeros(NUM_COMBAT_TYPES)
    for i in train_idx:
        combat_counts[combat_type[i]] += 1
    combat_weights = torch.zeros(NUM_COMBAT_TYPES, device=DEVICE)
    for c in range(NUM_COMBAT_TYPES):
        if combat_counts[c] > 0:
            combat_weights[c] = min(len(train_idx) / (NUM_COMBAT_TYPES * combat_counts[c]), 20.0)
        else:
            combat_weights[c] = 1.0
    print(f"Combat class weights: {[f'{w:.1f}' for w in combat_weights.tolist()]}")

    # Same for move dirs
    move_counts = torch.zeros(NUM_MOVE_DIRS)
    for i in train_idx:
        move_counts[move_dir[i]] += 1
    move_weights = torch.zeros(NUM_MOVE_DIRS, device=DEVICE)
    for d in range(NUM_MOVE_DIRS):
        if move_counts[d] > 0:
            move_weights[d] = min(len(train_idx) / (NUM_MOVE_DIRS * move_counts[d]), 10.0)
        else:
            move_weights[d] = 1.0
    print(f"Move class weights: {[f'{w:.1f}' for w in move_weights.tolist()]}")

    # Freeze encoder initially
    encoder_frozen = args.unfreeze_step > 0
    if encoder_frozen:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print(f"Encoder frozen until step {args.unfreeze_step}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.1, betas=(0.9, 0.98),
    )

    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    train_perm = np.random.permutation(train_idx)
    train_ptr = 0

    print(f"\nTraining for {args.max_steps} steps, batch={args.batch_size}")

    while step < args.max_steps:
        # Unfreeze encoder at specified step
        if encoder_frozen and step >= args.unfreeze_step:
            for p in model.encoder.parameters():
                p.requires_grad = True
            encoder_frozen = False
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr_unfrozen,
                weight_decay=0.1, betas=(0.9, 0.98),
            )
            print(f"  === Encoder unfrozen at step {step}, lr={args.lr_unfrozen} ===")

        model.train()

        if train_ptr + args.batch_size > len(train_perm):
            train_perm = np.random.permutation(train_idx)
            train_ptr = 0
        idx = train_perm[train_ptr:train_ptr + args.batch_size]
        train_ptr += args.batch_size

        move_logits, combat_logits, move_cont = model(
            ent_feat[idx], ent_types[idx], thr_feat[idx],
            ent_mask[idx], thr_mask[idx],
            pos_feat[idx], pos_mask[idx], agg_feat[idx],
        )

        loss_move = F.cross_entropy(move_logits, move_dir[idx], weight=move_weights)
        loss_combat = F.cross_entropy(combat_logits, combat_type[idx], weight=combat_weights)
        loss = loss_move + loss_combat
        if move_vec is not None:
            loss_cont = F.mse_loss(move_cont, move_vec[idx])
            loss = loss + 0.5 * loss_cont

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_move_correct = 0
                val_combat_correct = 0
                val_loss_sum = 0.0
                val_n = 0
                val_cont_sum = 0.0
                for vs in range(0, len(val_idx), args.batch_size):
                    vidx = val_idx[vs:vs + args.batch_size]
                    vm, vc, vmc = model(
                        ent_feat[vidx], ent_types[vidx], thr_feat[vidx],
                        ent_mask[vidx], thr_mask[vidx],
                        pos_feat[vidx], pos_mask[vidx], agg_feat[vidx],
                    )
                    vl = F.cross_entropy(vm, move_dir[vidx]) + F.cross_entropy(vc, combat_type[vidx])
                    val_loss_sum += vl.item() * len(vidx)
                    if move_vec is not None:
                        val_cont_sum += F.mse_loss(vmc, move_vec[vidx]).item() * len(vidx)
                    val_move_correct += (vm.argmax(-1) == move_dir[vidx]).sum().item()
                    val_combat_correct += (vc.argmax(-1) == combat_type[vidx]).sum().item()
                    val_n += len(vidx)

            val_loss = val_loss_sum / max(val_n, 1)
            val_cont = val_cont_sum / max(val_n, 1)
            move_acc = 100 * val_move_correct / max(val_n, 1)
            combat_acc = 100 * val_combat_correct / max(val_n, 1)
            frozen_tag = " [frozen]" if encoder_frozen else ""
            elapsed = time.time() - t0
            print(f"  step {step:6d} | loss={loss.item():.3f} val={val_loss:.3f} cont={val_cont:.4f} move={move_acc:.1f}% combat={combat_acc:.1f}%{frozen_tag} | {elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "move_acc": move_acc,
                    "combat_acc": combat_acc,
                    "args": vars(args),
                }, args.output)

    print(f"\nBest val_loss: {best_val_loss:.3f}")
    print(f"Saved to {args.output}")


# Fixed direction unit vectors matching Rust move_dir_offset()
DIR_VECTORS = torch.tensor([
    [ 0.000,  1.000],  # 0: N
    [ 0.707,  0.707],  # 1: NE
    [ 1.000,  0.000],  # 2: E
    [ 0.707, -0.707],  # 3: SE
    [ 0.000, -1.000],  # 4: S
    [-0.707, -0.707],  # 5: SW
    [-1.000,  0.000],  # 6: W
    [-0.707,  0.707],  # 7: NW
    [ 0.000,  0.000],  # 8: stay
], dtype=torch.float32)


class V5BCModel(nn.Module):
    """Lightweight BC model: V5 encoder + move/combat heads.

    Movement: 9-dir classification head + continuous projection MLP.
    The 9-dir logits are softmaxed and used as weights over fixed direction
    vectors, then an MLP refines into (dx, dy, speed). This gives continuous
    movement while keeping the discrete head for RL compatibility.
    """

    def __init__(self, d_model=128, n_heads=8, n_layers=4):
        super().__init__()
        self.encoder = EntityEncoderV5(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.d_model = d_model

        self.move_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, NUM_MOVE_DIRS),
        )
        # Continuous movement: takes pooled state + softmax-weighted direction → (dx, dy, speed)
        # Input: d_model (pooled) + 2 (weighted direction vector) = d_model + 2
        self.move_continuous = nn.Sequential(
            nn.Linear(d_model + 2, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 3),
        )
        self.combat_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, NUM_COMBAT_TYPES),
        )

        self.register_buffer("dir_vectors", DIR_VECTORS)

    def forward(self, ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                pos_feat, pos_mask, agg_feat):
        tokens, full_mask = self.encoder(
            ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
            pos_feat, pos_mask, agg_feat,
        )
        mask_exp = (~full_mask).unsqueeze(-1).float()
        pooled = (tokens * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)

        move_logits = self.move_head(pooled)
        combat_logits = self.combat_head(pooled)
        move_cont = self.continuous_move(pooled, move_logits)

        return move_logits, combat_logits, move_cont

    def continuous_move(self, pooled, move_logits):
        """Convert discrete move logits to continuous (dx, dy, speed).

        Returns (B, 3) where [:, :2] is the direction vector and [:, 2] is speed [0, 1].
        """
        # Soft-weight the fixed direction vectors
        weights = F.softmax(move_logits, dim=-1)  # (B, 9)
        weighted_dir = weights @ self.dir_vectors  # (B, 2)

        # MLP refines direction and adds speed
        combined = torch.cat([pooled, weighted_dir], dim=-1)  # (B, d+2)
        raw = self.move_continuous(combined)  # (B, 3)

        # dx, dy as tanh (bounded [-1,1]), speed as sigmoid (bounded [0,1])
        dx = torch.tanh(raw[:, 0])
        dy = torch.tanh(raw[:, 1])
        speed = torch.sigmoid(raw[:, 2])

        return torch.stack([dx, dy, speed], dim=-1)


if __name__ == "__main__":
    main()
