#!/usr/bin/env python3
"""Train ELIT-DiT for room grid generation.

Usage:
    uv run --with torch --with sentence-transformers python training/roomgen/train.py \
        --data generated/rooms_captioned.jsonl \
        --epochs 200 \
        --batch-size 64

    # Without captions (text encoder uses empty strings):
    uv run --with torch python training/roomgen/train.py \
        --data generated/rooms.jsonl \
        --text-encoder simple
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roomgen.dataset import RoomGridDataset, collate_room_grids
from roomgen.elit_dit import ELITDiT
from roomgen.flow_matching import rectified_flow_forward, compute_loss
from roomgen.text_encoder import build_text_encoder


def main():
    parser = argparse.ArgumentParser(description="Train ELIT-DiT room generator")
    parser.add_argument("--data", default="generated/rooms_captioned.jsonl")
    parser.add_argument("--output", default="generated/elit_dit_weights.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-beta", type=float, default=0.9999)
    parser.add_argument("--cfg-dropout", type=float, default=0.15)
    parser.add_argument("--j-min", type=int, default=2)
    parser.add_argument("--j-max", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-head-blocks", type=int, default=2)
    parser.add_argument("--n-core-blocks", type=int, default=8)
    parser.add_argument("--n-tail-blocks", type=int, default=2)
    parser.add_argument("--text-encoder", default="minilm", choices=["minilm", "simple"])
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # --- Dataset ---
    print(f"Loading dataset from {args.data}...")
    dataset = RoomGridDataset(args.data)
    print(f"  {len(dataset)} rooms loaded")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_room_grids, num_workers=4, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_room_grids, num_workers=2, pin_memory=True,
    )

    # --- Text Encoder ---
    print(f"Building text encoder: {args.text_encoder}")
    text_enc = build_text_encoder(args.text_encoder, device=str(device))
    text_enc = text_enc.to(device)
    d_text = text_enc.dim

    # --- Model ---
    model = ELITDiT(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        d_text=d_text,
        n_head_blocks=args.n_head_blocks,
        n_core_blocks=args.n_core_blocks,
        n_tail_blocks=args.n_tail_blocks,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # EMA model
    ema_model = ELITDiT(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_model * 4,
        d_text=d_text,
        n_head_blocks=args.n_head_blocks,
        n_core_blocks=args.n_core_blocks,
        n_tail_blocks=args.n_tail_blocks,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.requires_grad_(False)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.98), weight_decay=1.0,
    )

    # Cosine schedule with warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        total_steps = args.epochs * len(train_loader)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    global_step = 0

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"  Resumed from epoch {start_epoch}, step {global_step}")

    # --- Training loop ---
    print(f"\nStarting training: {args.epochs} epochs, {len(train_loader)} batches/epoch")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_obs_loss = 0.0
        epoch_h_loss = 0.0
        epoch_e_loss = 0.0
        epoch_dim_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            obs_type = batch["obs_type"].to(device)
            height = batch["height"].to(device)
            elevation = batch["elevation"].to(device)
            mask = batch["mask"].to(device)
            room_type = batch["room_type"].to(device)
            width = batch["width"].to(device)
            depth = batch["depth"].to(device)
            captions = batch["captions"]

            B = obs_type.shape[0]

            # CFG dropout: replace text with empty string
            if args.cfg_dropout > 0:
                drop_mask = torch.rand(B) < args.cfg_dropout
                captions = [("" if drop_mask[i] else c) for i, c in enumerate(captions)]

            # Encode text
            with torch.no_grad():
                text_emb = text_enc(captions)  # (B, d_text)
                if text_emb.device != device:
                    text_emb = text_emb.to(device)

            # Multi-budget: sample random J per batch
            j_budget = torch.randint(args.j_min, args.j_max + 1, (1,)).item()

            # Rectified flow forward
            flow = rectified_flow_forward(obs_type, height, elevation)

            # Model forward
            pred = model(
                flow["noisy_obs"], flow["noisy_height"], flow["noisy_elevation"],
                flow["t"], text_emb, width, depth,
                mask=mask, j_budget=j_budget,
            )

            # Velocity loss
            losses = compute_loss(pred, flow, mask)

            # Dimension prediction loss
            pred_dims = model.predict_dims(text_emb, room_type)
            gt_dims = torch.stack([width.float(), depth.float()], dim=-1)
            dim_loss = F.mse_loss(pred_dims, gt_dims)

            total_loss = losses["total"] + 0.1 * dim_loss

            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            # EMA update
            with torch.no_grad():
                for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
                    p_ema.mul_(args.ema_beta).add_(p_model, alpha=1 - args.ema_beta)

            epoch_loss += losses["total"].item()
            epoch_obs_loss += losses["obs"].item()
            epoch_h_loss += losses["height"].item()
            epoch_e_loss += losses["elevation"].item()
            epoch_dim_loss += dim_loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  step={global_step} loss={losses['total'].item():.4f} "
                    f"obs={losses['obs'].item():.4f} h={losses['height'].item():.4f} "
                    f"e={losses['elevation'].item():.4f} dim={dim_loss.item():.4f} "
                    f"J={j_budget} lr={lr:.2e}"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
        print(
            f"Epoch {epoch}: loss={avg_loss:.4f} "
            f"obs={epoch_obs_loss/max(1,n_batches):.4f} "
            f"h={epoch_h_loss/max(1,n_batches):.4f} "
            f"e={epoch_e_loss/max(1,n_batches):.4f} "
            f"dim={epoch_dim_loss/max(1,n_batches):.4f} "
            f"({elapsed:.0f}s)"
        )

        # --- Validation ---
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    obs_type = batch["obs_type"].to(device)
                    height = batch["height"].to(device)
                    elevation = batch["elevation"].to(device)
                    mask = batch["mask"].to(device)
                    width = batch["width"].to(device)
                    depth = batch["depth"].to(device)
                    captions = batch["captions"]

                    text_emb = text_enc(captions)
                    if text_emb.device != device:
                        text_emb = text_emb.to(device)

                    flow = rectified_flow_forward(obs_type, height, elevation)
                    pred = ema_model(
                        flow["noisy_obs"], flow["noisy_height"], flow["noisy_elevation"],
                        flow["t"], text_emb, width, depth, mask=mask,
                    )
                    losses = compute_loss(pred, flow, mask)
                    val_loss += losses["total"].item()
                    val_n += 1

            val_avg = val_loss / max(1, val_n)
            print(f"  Val loss: {val_avg:.4f}")

            # Save checkpoint
            ckpt_path = args.output.replace(".pt", f"_epoch{epoch}.pt")
            torch.save({
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_avg,
                "args": vars(args),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                torch.save({
                    "model": ema_model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_loss": val_avg,
                }, args.output)
                print(f"  Best model saved: {args.output}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
