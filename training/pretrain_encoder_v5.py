#!/usr/bin/env python3
"""Stage 0a: Pre-train V5 entity encoder on game state reading.

Loads pre-converted npz data (from convert_v5_npz.py) and trains
EntityEncoderV5 (d=128, 34-dim entities, 10-dim threats, 8-dim positions,
16-dim aggregate) on two dense per-tick targets:

  - hp_advantage: mean(hero HP%) - mean(enemy HP%), range [-1, 1]
  - survival_ratio: heroes_alive / total_alive, range [0, 1]

Both targets are directly computable from entity features at every tick,
giving unique dense supervision without needing episode-level labels.

The encoder portion can be extracted after pre-training and used as the
frozen backbone for the full V5 actor-critic.

Usage:
    uv run --with numpy --with torch python training/convert_v5_npz.py \
        generated/v5_stage0a_random.jsonl generated/v5_stage0a_combined.jsonl \
        -o generated/v5_stage0a.npz

    uv run --with numpy --with torch python training/pretrain_encoder_v5.py \
        generated/v5_stage0a.npz \
        -o generated/entity_encoder_v5_pretrained.pt \
        --max-steps 50000
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
from model import EntityEncoderV5, AGG_FEATURE_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model: encoder + prediction heads
# ---------------------------------------------------------------------------

class V5EncoderPretraining(nn.Module):
    """V5 entity encoder with multi-task prediction heads.

    Two dense per-tick targets:
      - hp_advantage: mean(hero HP%) - mean(enemy HP%), range [-1, 1]
      - survival_ratio: heroes_alive / total_alive, range [0, 1]

    Both are directly computable from entity features at every tick,
    giving unique targets without episode-level labels.
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.encoder = EntityEncoderV5(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.d_model = d_model

        self.hp_advantage_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )
        self.survival_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1),
        )

    def forward(self, entity_features, entity_type_ids, zone_features,
                entity_mask, zone_mask,
                aggregate_features=None):
        tokens, full_mask = self.encoder(
            entity_features, entity_type_ids, zone_features,
            entity_mask, zone_mask,
            aggregate_features,
        )
        mask_expanded = (~full_mask).unsqueeze(-1).float()
        pooled = (tokens * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)

        hp_adv = self.hp_advantage_head(pooled).squeeze(-1)
        surv = self.survival_head(pooled).squeeze(-1)
        return hp_adv, surv


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Stage 0a: pre-train V5 entity encoder")
    p.add_argument("data", help="npz file from convert_v5_npz.py")
    p.add_argument("-o", "--output", default="generated/entity_encoder_v5_pretrained.pt")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load npz — all data into GPU memory
    print(f"Loading {args.data}...")
    d = np.load(args.data)

    train_idx = d["train_idx"]
    val_idx = d["val_idx"]

    # Move all data to GPU as tensors
    ent_feat = torch.from_numpy(d["ent_feat"]).to(DEVICE)
    ent_types = torch.from_numpy(d["ent_types"]).long().to(DEVICE)
    ent_mask = torch.from_numpy(d["ent_mask"]).bool().to(DEVICE)
    # Prefer zone_feat/zone_mask; fall back to thr_feat/thr_mask for old npz files
    if "zone_feat" in d:
        zone_feat = torch.from_numpy(d["zone_feat"]).to(DEVICE)
        zone_mask = torch.from_numpy(d["zone_mask"]).bool().to(DEVICE)
    else:
        print("  WARNING: zone_feat not found in npz, falling back to thr_feat/thr_mask")
        zone_feat = torch.from_numpy(d["thr_feat"]).to(DEVICE)
        zone_mask = torch.from_numpy(d["thr_mask"]).bool().to(DEVICE)
    agg_feat = torch.from_numpy(d["agg_feat"]).to(DEVICE)
    hp_adv = torch.from_numpy(d["hp_adv"]).to(DEVICE)
    surv = torch.from_numpy(d["surv"]).to(DEVICE)

    N = ent_feat.shape[0]
    print(f"  {N} samples, train={len(train_idx)}, val={len(val_idx)}")
    print(f"  Entity shape: {ent_feat.shape}, Zone shape: {zone_feat.shape}")
    print(f"  hp_adv: mean={hp_adv[train_idx].mean():.3f} std={hp_adv[train_idx].std():.3f} unique~{len(torch.unique(torch.round(hp_adv[train_idx] * 1000)))}")
    print(f"  surv:   mean={surv[train_idx].mean():.3f} std={surv[train_idx].std():.3f} unique~{len(torch.unique(torch.round(surv[train_idx] * 1000)))}")

    # Model
    model = V5EncoderPretraining(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params, d={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1.0, betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    train_perm = np.random.permutation(train_idx)
    train_ptr = 0

    print(f"\nTraining for {args.max_steps} steps, batch={args.batch_size}, lr={args.lr}")

    while step < args.max_steps:
        model.train()

        # Get batch indices
        if train_ptr + args.batch_size > len(train_perm):
            train_perm = np.random.permutation(train_idx)
            train_ptr = 0
        idx = train_perm[train_ptr:train_ptr + args.batch_size]
        train_ptr += args.batch_size

        hp_pred, surv_pred = model(
            ent_feat[idx], ent_types[idx], zone_feat[idx],
            ent_mask[idx], zone_mask[idx], agg_feat[idx],
        )
        loss_hp = F.mse_loss(hp_pred, hp_adv[idx])
        loss_surv = F.mse_loss(surv_pred, surv[idx])
        loss = 0.7 * loss_hp + 0.3 * loss_surv

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Eval on full val set in chunks
                val_hp_sum = 0.0
                val_surv_sum = 0.0
                val_n = 0
                for vstart in range(0, len(val_idx), args.batch_size):
                    vidx = val_idx[vstart:vstart + args.batch_size]
                    vhp, vsurv = model(
                        ent_feat[vidx], ent_types[vidx], zone_feat[vidx],
                        ent_mask[vidx], zone_mask[vidx], agg_feat[vidx],
                    )
                    val_hp_sum += F.mse_loss(vhp, hp_adv[vidx]).item() * len(vidx)
                    val_surv_sum += F.mse_loss(vsurv, surv[vidx]).item() * len(vidx)
                    val_n += len(vidx)

            val_hp = val_hp_sum / max(val_n, 1)
            val_surv = val_surv_sum / max(val_n, 1)
            val_loss = 0.7 * val_hp + 0.3 * val_surv
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {step:6d} | train={loss.item():.4f} val_hp={val_hp:.4f} val_surv={val_surv:.4f} | lr={lr:.2e} | {elapsed:.0f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "val_hp_mse": val_hp,
                    "val_surv_mse": val_surv,
                    "args": vars(args),
                }, args.output)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
