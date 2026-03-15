#!/usr/bin/env python3
"""Stage 0b: Pre-train spatial cross-attention on cover/choke prediction.

Trains a SpatialCrossAttention layer on top of a pretrained (frozen)
EntityEncoderV5 to learn cross-entity spatial relationships.

Task: leave-one-out prediction of per-entity spatial features.
For each entity i, predict its spatial features (visible_corner_count,
avg_passage_width, min_passage_width, avg_corner_distance) from ALL OTHER
entities' encoded representations. This forces the spatial cross-attention
layer to learn genuine cross-entity spatial reasoning rather than copying
inputs.

Targets come from entity feature indices 30-33 (the 4 spatial summary
features appended in V5's 34-dim entity representation).

Usage:
    uv run --with numpy --with torch python training/pretrain_spatial_v5.py \
        generated/v5_stage0a.npz \
        --encoder-ckpt generated/entity_encoder_v5_pretrained.pt \
        -o generated/spatial_crossattn_v5.pt \
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

sys.path.insert(0, str(Path(__file__).parent))
from models.encoder_v5 import EntityEncoderV5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Spatial feature indices within the 34-dim entity feature vector
SPATIAL_START = 30
SPATIAL_END = 34
SPATIAL_DIM = SPATIAL_END - SPATIAL_START  # 4


class SpatialCrossAttention(nn.Module):
    """Cross-attention layer that learns spatial relationships between entities.

    For each entity, attends to all OTHER entities to predict its spatial
    context (cover, choke, corner distances).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        leave_out_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, d) -- encoded entity tokens
            mask: (B, N) -- True where padded
            leave_out_mask: (B, N, N) -- True to block self-attention (diagonal)

        Returns:
            out: (B, N, d) -- spatially enriched per-entity representations
        """
        q = self.norm_q(tokens)
        kv = self.norm_kv(tokens)
        # leave_out_mask: (B, N, N), need (B*n_heads, N, N)
        # Use attn_mask parameter which expects (N, N) or (B*H, N, N)
        n_heads = self.cross_attn.num_heads
        B, N, _ = tokens.shape
        attn_mask = leave_out_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)
        attn_mask = attn_mask.reshape(B * n_heads, N, N)

        attn_out, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=mask,
            attn_mask=attn_mask,
        )
        x = tokens + attn_out
        x = x + self.ff(self.ff_norm(x))
        return x


class SpatialPretraining(nn.Module):
    """Frozen encoder + spatial cross-attention + per-entity spatial prediction."""

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.encoder = EntityEncoderV5(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.spatial_attn = SpatialCrossAttention(d_model=d_model, n_heads=n_heads)
        self.spatial_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, SPATIAL_DIM),
        )

    def forward(
        self,
        entity_features, entity_type_ids, threat_features,
        entity_mask, threat_mask,
        position_features=None, position_mask=None,
        aggregate_features=None,
    ):
        """
        Returns:
            spatial_pred: (B, E, 4) -- predicted spatial features per entity
        """
        B = entity_features.shape[0]
        E = entity_features.shape[1]

        tokens, full_mask = self.encoder(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask,
            position_features, position_mask,
            aggregate_features,
        )

        N = tokens.shape[1]

        # Build leave-one-out mask: block diagonal (entity can't attend to itself)
        # Only for the entity positions (first E tokens); threats/positions can attend freely
        leave_out_mask = torch.zeros(B, N, N, dtype=torch.bool, device=tokens.device)
        diag = torch.eye(E, dtype=torch.bool, device=tokens.device)
        leave_out_mask[:, :E, :E] = diag.unsqueeze(0)
        # Convert bool mask to float: True -> -inf, False -> 0
        leave_out_float = torch.zeros_like(leave_out_mask, dtype=tokens.dtype)
        leave_out_float.masked_fill_(leave_out_mask, float("-inf"))

        enriched = self.spatial_attn(tokens, full_mask, leave_out_float)

        # Predict spatial features only for entity tokens (first E)
        entity_enriched = enriched[:, :E, :]  # (B, E, d)
        spatial_pred = self.spatial_head(entity_enriched)  # (B, E, 4)

        return spatial_pred


def main():
    p = argparse.ArgumentParser(description="Stage 0b: pre-train spatial cross-attention")
    p.add_argument("data", help="npz file from convert_v5_npz.py")
    p.add_argument("-o", "--output", default="generated/spatial_crossattn_v5.pt")
    p.add_argument("--encoder-ckpt", required=True, help="Pretrained encoder checkpoint from Stage 0a")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
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

    N = ent_feat.shape[0]
    E = ent_feat.shape[1]
    print(f"  {N} samples, train={len(train_idx)}, val={len(val_idx)}")
    print(f"  Entity shape: {ent_feat.shape}")

    # Extract spatial targets: entity features indices 30-33
    spatial_targets = ent_feat[:, :, SPATIAL_START:SPATIAL_END].clone()  # (N, E, 4)

    # Statistics on spatial targets
    valid = ~ent_mask  # (N, E) True where entity exists
    flat_targets = spatial_targets[valid]  # (valid_count, 4)
    print(f"  Spatial targets: shape={spatial_targets.shape}")
    for i, name in enumerate(["visible_corners", "avg_passage_w", "min_passage_w", "avg_corner_dist"]):
        vals = flat_targets[:, i]
        print(f"    {name}: mean={vals.mean():.3f} std={vals.std():.3f}")

    # Build model
    model = SpatialPretraining(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(DEVICE)

    # Load pretrained encoder (frozen)
    print(f"Loading pretrained encoder from {args.encoder_ckpt}...")
    ckpt = torch.load(args.encoder_ckpt, map_location=DEVICE, weights_only=False)
    prefix = "encoder."
    encoder_sd = {k[len(prefix):]: v
                  for k, v in ckpt["model_state_dict"].items()
                  if k.startswith(prefix)}
    model.encoder.load_state_dict(encoder_sd, strict=True)
    print(f"  Loaded encoder weights (step={ckpt.get('step', '?')})")

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    print(f"\nModel: {n_params:,} params total, {n_trainable:,} trainable, {n_encoder:,} encoder (frozen)")

    # Only optimize trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1.0, betas=(0.9, 0.98),
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

        # Get batch
        if train_ptr + args.batch_size > len(train_perm):
            train_perm = np.random.permutation(train_idx)
            train_ptr = 0
        idx = train_perm[train_ptr:train_ptr + args.batch_size]
        train_ptr += args.batch_size

        spatial_pred = model(
            ent_feat[idx], ent_types[idx], thr_feat[idx],
            ent_mask[idx], thr_mask[idx],
            pos_feat[idx], pos_mask[idx], agg_feat[idx],
        )

        # Masked MSE: only compute loss where entities exist
        target = spatial_targets[idx]  # (B, E, 4)
        valid_mask = ~ent_mask[idx]    # (B, E) True where entity exists
        valid_mask_exp = valid_mask.unsqueeze(-1).expand_as(target)  # (B, E, 4)

        if valid_mask.any():
            loss = F.mse_loss(spatial_pred[valid_mask_exp], target[valid_mask_exp])
        else:
            loss = torch.tensor(0.0, device=DEVICE)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_n = 0
                per_feat_mse = torch.zeros(SPATIAL_DIM, device=DEVICE)
                per_feat_n = 0

                for vstart in range(0, len(val_idx), args.batch_size):
                    vidx = val_idx[vstart:vstart + args.batch_size]
                    vpred = model(
                        ent_feat[vidx], ent_types[vidx], thr_feat[vidx],
                        ent_mask[vidx], thr_mask[vidx],
                        pos_feat[vidx], pos_mask[vidx], agg_feat[vidx],
                    )
                    vtarget = spatial_targets[vidx]
                    vmask = ~ent_mask[vidx]
                    vmask_exp = vmask.unsqueeze(-1).expand_as(vtarget)

                    if vmask.any():
                        vl = F.mse_loss(vpred[vmask_exp], vtarget[vmask_exp]).item()
                        val_loss_sum += vl * vmask.sum().item()
                        val_n += vmask.sum().item()

                        # Per-feature MSE
                        diff_sq = (vpred - vtarget) ** 2  # (B, E, 4)
                        for f in range(SPATIAL_DIM):
                            per_feat_mse[f] += (diff_sq[:, :, f] * vmask.float()).sum()
                        per_feat_n += vmask.sum().item()

                val_loss = val_loss_sum / max(val_n, 1)
                per_feat = per_feat_mse / max(per_feat_n, 1)
                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]

                feat_str = " ".join([f"{v:.4f}" for v in per_feat.tolist()])
                print(f"  step {step:6d} | train={loss.item():.4f} val={val_loss:.4f} | per_feat=[{feat_str}] | lr={lr:.2e} | {elapsed:.0f}s")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "spatial_attn_state_dict": model.spatial_attn.state_dict(),
                        "spatial_head_state_dict": model.spatial_head.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "per_feat_mse": per_feat.tolist(),
                        "args": vars(args),
                    }, args.output)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
