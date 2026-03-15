#!/usr/bin/env python3
"""Stage 0d: Latent interface warmup — train Read/Write/Process pipeline.

Loads pretrained encoder (frozen) and CfC (frozen), then trains the
LatentInterface to compress spatial tokens into latent representations
and predict fight outcome through the full pipeline:

    encoder (frozen) -> LatentInterface (trainable) -> CfC (frozen) -> outcome

This is a short warmup (10-15 min) to get the latent interface to
compress 22 tokens -> 12 latents before RL begins. The latent interface
starts with zero-initialized write projection, so at init it's an
identity pass-through; this warmup teaches it to do useful compression.

Includes tail-dropping regularization: randomly drop trailing latents
during training to enforce importance ordering.

Usage:
    uv run --with numpy --with torch python training/pretrain_latent_v5.py \
        generated/v5_stage0a.npz \
        --encoder-ckpt generated/entity_encoder_v5_pretrained.pt \
        --cfc-ckpt generated/cfc_temporal_v5.pt \
        -o generated/latent_interface_v5.pt \
        --max-steps 10000
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
from models.latent_interface import LatentInterface
from models.cfc_cell import CfCCell, CFC_H_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LatentWarmup(nn.Module):
    """Full pipeline: encoder (frozen) -> latent interface (trainable) -> CfC (frozen) -> outcome.

    Also includes a value head on the latent-pooled representation to
    pre-warm the value function for RL.
    """

    def __init__(self, d_model: int = 128, h_dim: int = CFC_H_DIM,
                 n_heads: int = 8, n_layers: int = 4,
                 n_latents: int = 12, n_latent_blocks: int = 2):
        super().__init__()
        self.encoder = EntityEncoderV5(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.latent_interface = LatentInterface(
            d_model=d_model, n_latents=n_latents, n_heads=n_heads,
            n_latent_blocks=n_latent_blocks,
        )
        self.cfc = CfCCell(d_model, h_dim)
        self.d_model = d_model

        # Outcome prediction (hp_advantage)
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Survival prediction
        self.survival_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Value head (pre-warm for RL)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
                pos_feat, pos_mask, agg_feat,
                h_prev=None, n_latents_override=None):
        """
        Returns:
            hp_pred: (B, 1) -- predicted hp_advantage
            surv_pred: (B, 1) -- predicted survival_ratio
            value_pred: (B, 1) -- value estimate
            h_new: (B, h_dim) -- CfC hidden state
        """
        # Encoder (frozen)
        tokens, full_mask = self.encoder(
            ent_feat, ent_types, thr_feat, ent_mask, thr_mask,
            pos_feat, pos_mask, agg_feat,
        )

        # Latent interface (trainable)
        tokens, pooled = self.latent_interface(tokens, full_mask, n_latents_override)

        # CfC (frozen) -- single tick
        proj, h_new = self.cfc(pooled, h_prev)

        # Prediction heads (trainable)
        hp_pred = self.outcome_head(proj)
        surv_pred = self.survival_head(proj)
        value_pred = self.value_head(proj)

        return hp_pred, surv_pred, value_pred, h_new


def main():
    p = argparse.ArgumentParser(description="Stage 0d: latent interface warmup")
    p.add_argument("data", help="npz file from convert_v5_npz.py")
    p.add_argument("-o", "--output", default="generated/latent_interface_v5.pt")
    p.add_argument("--encoder-ckpt", required=True, help="Pretrained encoder from Stage 0a")
    p.add_argument("--cfc-ckpt", help="Pretrained CfC from Stage 0c (optional)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--h-dim", type=int, default=CFC_H_DIM)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-latents", type=int, default=12)
    p.add_argument("--n-latent-blocks", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--tail-drop-prob", type=float, default=0.3,
                    help="Probability of tail-dropping latents during training")
    p.add_argument("--tail-drop-min", type=int, default=4,
                    help="Minimum number of latents to keep when tail-dropping")
    p.add_argument("--eval-every", type=int, default=500)
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
    hp_adv = torch.from_numpy(d["hp_adv"]).to(DEVICE)
    surv = torch.from_numpy(d["surv"]).to(DEVICE)

    N = ent_feat.shape[0]
    print(f"  {N} samples, train={len(train_idx)}, val={len(val_idx)}")

    # Build model
    model = LatentWarmup(
        d_model=args.d_model, h_dim=args.h_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        n_latents=args.n_latents, n_latent_blocks=args.n_latent_blocks,
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
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Load pretrained CfC (frozen) if available
    if args.cfc_ckpt:
        print(f"Loading pretrained CfC from {args.cfc_ckpt}...")
        cfc_ckpt = torch.load(args.cfc_ckpt, map_location=DEVICE, weights_only=False)
        if "cfc_state_dict" in cfc_ckpt:
            model.cfc.load_state_dict(cfc_ckpt["cfc_state_dict"], strict=True)
            print(f"  Loaded CfC weights (step={cfc_ckpt.get('step', '?')})")
        else:
            # Try loading from full model state dict
            cfc_prefix = "cfc."
            cfc_sd = {k[len(cfc_prefix):]: v
                      for k, v in cfc_ckpt["model_state_dict"].items()
                      if k.startswith(cfc_prefix)}
            model.cfc.load_state_dict(cfc_sd, strict=True)
            print(f"  Loaded CfC weights from model state dict")
        for param in model.cfc.parameters():
            param.requires_grad = False
    else:
        print("  No CfC checkpoint provided -- CfC will be trained from scratch")
        # Don't freeze CfC if no pretrained weights

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_latent = sum(p.numel() for p in model.latent_interface.parameters())
    print(f"\nModel: {n_params:,} params total, {n_trainable:,} trainable, {n_latent:,} latent interface")

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
    print(f"Tail dropping: prob={args.tail_drop_prob}, min_latents={args.tail_drop_min}")

    while step < args.max_steps:
        model.train()

        if train_ptr + args.batch_size > len(train_perm):
            train_perm = np.random.permutation(train_idx)
            train_ptr = 0
        idx = train_perm[train_ptr:train_ptr + args.batch_size]
        train_ptr += args.batch_size

        # Tail dropping: randomly reduce number of latents
        n_latents_override = None
        if np.random.random() < args.tail_drop_prob:
            n_latents_override = np.random.randint(
                args.tail_drop_min, args.n_latents + 1
            )

        hp_pred, surv_pred, value_pred, _ = model(
            ent_feat[idx], ent_types[idx], thr_feat[idx],
            ent_mask[idx], thr_mask[idx],
            pos_feat[idx], pos_mask[idx], agg_feat[idx],
            n_latents_override=n_latents_override,
        )

        loss_hp = F.mse_loss(hp_pred.squeeze(-1), hp_adv[idx])
        loss_surv = F.mse_loss(surv_pred.squeeze(-1), surv[idx])
        # Value target: use hp_advantage as a proxy for value
        loss_value = F.mse_loss(value_pred.squeeze(-1), hp_adv[idx])

        loss = 0.4 * loss_hp + 0.3 * loss_surv + 0.3 * loss_value

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_hp_sum = 0.0
                val_surv_sum = 0.0
                val_value_sum = 0.0
                val_n = 0

                # Eval with full latents (no tail dropping)
                for vstart in range(0, len(val_idx), args.batch_size):
                    vidx = val_idx[vstart:vstart + args.batch_size]
                    vhp, vsurv, vval, _ = model(
                        ent_feat[vidx], ent_types[vidx], thr_feat[vidx],
                        ent_mask[vidx], thr_mask[vidx],
                        pos_feat[vidx], pos_mask[vidx], agg_feat[vidx],
                    )
                    val_hp_sum += F.mse_loss(vhp.squeeze(-1), hp_adv[vidx]).item() * len(vidx)
                    val_surv_sum += F.mse_loss(vsurv.squeeze(-1), surv[vidx]).item() * len(vidx)
                    val_value_sum += F.mse_loss(vval.squeeze(-1), hp_adv[vidx]).item() * len(vidx)
                    val_n += len(vidx)

                val_hp = val_hp_sum / max(val_n, 1)
                val_surv = val_surv_sum / max(val_n, 1)
                val_value = val_value_sum / max(val_n, 1)
                val_loss = 0.4 * val_hp + 0.3 * val_surv + 0.3 * val_value

                # Also eval with reduced latents to check degradation
                val_reduced_sum = 0.0
                val_reduced_n = 0
                n_reduced = args.tail_drop_min
                for vstart in range(0, len(val_idx), args.batch_size):
                    vidx = val_idx[vstart:vstart + args.batch_size]
                    vhp_r, _, _, _ = model(
                        ent_feat[vidx], ent_types[vidx], thr_feat[vidx],
                        ent_mask[vidx], thr_mask[vidx],
                        pos_feat[vidx], pos_mask[vidx], agg_feat[vidx],
                        n_latents_override=n_reduced,
                    )
                    val_reduced_sum += F.mse_loss(vhp_r.squeeze(-1), hp_adv[vidx]).item() * len(vidx)
                    val_reduced_n += len(vidx)
                val_reduced = val_reduced_sum / max(val_reduced_n, 1)

                elapsed = time.time() - t0
                lr = optimizer.param_groups[0]["lr"]
                td_tag = f" [td={n_latents_override}]" if n_latents_override else ""

                print(f"  step {step:6d} | train={loss.item():.4f}{td_tag} "
                      f"| val_hp={val_hp:.4f} val_surv={val_surv:.4f} val_value={val_value:.4f} "
                      f"| K={args.n_latents}:{val_hp:.4f} K={n_reduced}:{val_reduced:.4f} "
                      f"| lr={lr:.2e} | {elapsed:.0f}s")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "latent_interface_state_dict": model.latent_interface.state_dict(),
                        "outcome_head_state_dict": model.outcome_head.state_dict(),
                        "survival_head_state_dict": model.survival_head.state_dict(),
                        "value_head_state_dict": model.value_head.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "val_hp_mse": val_hp,
                        "val_surv_mse": val_surv,
                        "val_value_mse": val_value,
                        "args": vars(args),
                    }, args.output)

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
