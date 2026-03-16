#!/usr/bin/env python3
"""Stage 0c: Pre-train CfC temporal cell on fight outcome prediction.

Trains the CfC cell to accumulate temporal context from contiguous tick
windows and predict fight outcome from the final hidden state.

Requires SEQUENTIAL data -- contiguous tick windows from the same episode.
The npz must contain `ep_idx` (per-sample episode index) so that we can
reconstruct contiguous windows.

Architecture:
    EntityEncoderV5 (frozen from Stage 0a)
    -> pool per-tick -> CfCCell (trainable) -> outcome_head

Compares single-tick vs sequence-based accuracy to validate that the CfC
cell is learning useful temporal patterns.

Usage:
    uv run --with numpy --with torch python training/pretrain_temporal_v5.py \
        generated/v5_stage0a.npz \
        --encoder-ckpt generated/entity_encoder_v5_pretrained.pt \
        -o generated/cfc_temporal_v5.pt \
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
from models.cfc_cell import CfCCell, CFC_H_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalPretraining(nn.Module):
    """Frozen encoder + CfC cell + fight outcome head.

    Processes contiguous tick windows through the encoder (frozen),
    pools each tick, feeds through CfC, and predicts fight outcome
    from the final hidden state.
    """

    def __init__(self, d_model: int = 128, h_dim: int = CFC_H_DIM,
                 n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.encoder = EntityEncoderV5(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        self.cfc = CfCCell(d_model, h_dim)
        self.outcome_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # Single-tick baseline head (bypasses CfC, for comparison)
        self.single_tick_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.d_model = d_model

    def encode_tick(self, ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
                    agg_feat):
        """Encode a single tick and return pooled representation."""
        tokens, full_mask = self.encoder(
            ent_feat, ent_types, zone_feat, ent_mask, zone_mask,
            agg_feat,
        )
        mask_exp = (~full_mask).unsqueeze(-1).float()
        pooled = (tokens * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        return pooled  # (B, d_model)

    def forward_sequence(self, pooled_sequence: list[torch.Tensor],
                         h_init: torch.Tensor | None = None):
        """Run CfC over a sequence of pooled representations.

        Args:
            pooled_sequence: list of (B, d_model) tensors, one per tick
            h_init: (B, h_dim) or None

        Returns:
            outcome_pred: (B, 1) -- predicted fight outcome from final state
            single_pred: (B, 1) -- single-tick prediction from last tick only
            h_final: (B, h_dim)
        """
        h = h_init
        for pooled in pooled_sequence:
            proj, h = self.cfc(pooled, h)

        outcome_pred = self.outcome_head(proj)
        single_pred = self.single_tick_head(pooled_sequence[-1])

        return outcome_pred, single_pred, h


def build_windows(ep_idx: np.ndarray, hp_adv: np.ndarray,
                  min_window: int = 10, max_window: int = 20,
                  rng: np.random.RandomState | None = None):
    """Build contiguous tick windows from episode-indexed samples.

    Returns:
        windows: list of (start_idx, length, target) tuples
            start_idx: index into the flat sample array
            length: window length (min_window to max_window)
            target: hp_advantage at the LAST tick of the window
    """
    if rng is None:
        rng = np.random.RandomState(42)

    # Group samples by episode
    episodes = {}
    for i, ep in enumerate(ep_idx):
        ep = int(ep)
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append(i)

    # Sort each episode's samples by index (assumes temporal ordering within episode)
    for ep in episodes:
        episodes[ep].sort()

    windows = []
    for ep, indices in episodes.items():
        if len(indices) < min_window:
            continue
        # Slide over the episode with random window sizes
        pos = 0
        while pos + min_window <= len(indices):
            wlen = rng.randint(min_window, min(max_window, len(indices) - pos) + 1)
            window_indices = indices[pos:pos + wlen]
            # Check contiguity: indices should be consecutive
            if window_indices[-1] - window_indices[0] == wlen - 1:
                target = hp_adv[window_indices[-1]]
                windows.append((window_indices[0], wlen, float(target)))
            pos += wlen // 2  # 50% overlap

    return windows


def main():
    p = argparse.ArgumentParser(description="Stage 0c: pre-train CfC temporal cell")
    p.add_argument("data", help="npz file from convert_v5_npz.py")
    p.add_argument("-o", "--output", default="generated/cfc_temporal_v5.pt")
    p.add_argument("--encoder-ckpt", required=True, help="Pretrained encoder checkpoint from Stage 0a")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--h-dim", type=int, default=CFC_H_DIM)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--min-window", type=int, default=10)
    p.add_argument("--max-window", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=64,
                    help="Number of windows per batch (each window is 10-20 ticks)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # Load data
    print(f"Loading {args.data}...")
    d = np.load(args.data)

    train_idx = d["train_idx"]
    val_idx = d["val_idx"]

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

    N = ent_feat.shape[0]
    print(f"  {N} samples, train={len(train_idx)}, val={len(val_idx)}")

    # Check for ep_idx
    if "ep_idx" not in d:
        print("ERROR: npz does not contain 'ep_idx'. Cannot build contiguous windows.")
        print("Re-run convert_v5_npz.py to include episode indices.")
        sys.exit(1)

    ep_idx_np = d["ep_idx"]

    # Build windows for train and val
    # Determine which episodes are train vs val
    train_set = set(train_idx.tolist())
    val_set = set(val_idx.tolist())

    print("Building contiguous windows...")
    all_windows = build_windows(
        ep_idx_np, d["hp_adv"],
        min_window=args.min_window, max_window=args.max_window, rng=rng,
    )

    # Split windows: a window is "train" if its first sample is in train_idx
    train_windows = []
    val_windows = []
    for start, length, target in all_windows:
        if start in train_set:
            train_windows.append((start, length, target))
        elif start in val_set:
            val_windows.append((start, length, target))

    print(f"  Train windows: {len(train_windows)}, Val windows: {len(val_windows)}")

    if len(train_windows) == 0:
        print("ERROR: No training windows found. Check that episodes have >= min_window contiguous ticks.")
        sys.exit(1)

    # Show window length distribution
    wlens = [w[1] for w in train_windows]
    print(f"  Window lengths: min={min(wlens)} max={max(wlens)} mean={np.mean(wlens):.1f}")

    # Build model
    model = TemporalPretraining(
        d_model=args.d_model, h_dim=args.h_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
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
    print(f"\nModel: {n_params:,} params total, {n_trainable:,} trainable")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1.0, betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    train_perm = np.random.permutation(len(train_windows))
    train_ptr = 0

    print(f"\nTraining for {args.max_steps} steps, batch={args.batch_size}, lr={args.lr}")

    while step < args.max_steps:
        model.train()

        # Get batch of windows
        if train_ptr + args.batch_size > len(train_perm):
            train_perm = np.random.permutation(len(train_windows))
            train_ptr = 0

        batch_windows = [train_windows[train_perm[i]]
                         for i in range(train_ptr, train_ptr + args.batch_size)]
        train_ptr += args.batch_size

        # Encode all ticks in batch (with frozen encoder, no grad)
        # Group by window, pad to max length
        max_len = max(w[1] for w in batch_windows)
        B = len(batch_windows)

        # Pre-encode all ticks
        with torch.no_grad():
            pooled_all = []
            for t_offset in range(max_len):
                # Gather indices for this tick position across all windows
                indices = []
                for start, length, _ in batch_windows:
                    if t_offset < length:
                        indices.append(start + t_offset)
                    else:
                        # Pad with last valid tick
                        indices.append(start + length - 1)
                idx = torch.tensor(indices, dtype=torch.long, device=DEVICE)

                pooled = model.encode_tick(
                    ent_feat[idx], ent_types[idx], zone_feat[idx],
                    ent_mask[idx], zone_mask[idx], agg_feat[idx],
                )
                pooled_all.append(pooled)

        # Run CfC (trainable) over the sequence
        targets = torch.tensor([w[2] for w in batch_windows], device=DEVICE)

        # Need grad for CfC parameters
        # Re-enable grad for pooled (detached from encoder graph)
        pooled_sequence = [p.detach() for p in pooled_all]

        outcome_pred, single_pred, _ = model.forward_sequence(pooled_sequence)

        loss_seq = F.mse_loss(outcome_pred.squeeze(-1), targets)
        loss_single = F.mse_loss(single_pred.squeeze(-1), targets)
        # Main loss is sequence-based; single-tick is auxiliary
        loss = loss_seq + 0.3 * loss_single

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_seq_sum = 0.0
                val_single_sum = 0.0
                val_n = 0

                for vstart in range(0, len(val_windows), args.batch_size):
                    vbatch = val_windows[vstart:vstart + args.batch_size]
                    if not vbatch:
                        continue

                    vmax_len = max(w[1] for w in vbatch)
                    vB = len(vbatch)

                    vpooled_all = []
                    for t_offset in range(vmax_len):
                        indices = []
                        for start, length, _ in vbatch:
                            if t_offset < length:
                                indices.append(start + t_offset)
                            else:
                                indices.append(start + length - 1)
                        vidx = torch.tensor(indices, dtype=torch.long, device=DEVICE)

                        vpooled = model.encode_tick(
                            ent_feat[vidx], ent_types[vidx], zone_feat[vidx],
                            ent_mask[vidx], zone_mask[vidx], agg_feat[vidx],
                        )
                        vpooled_all.append(vpooled)

                    vtargets = torch.tensor([w[2] for w in vbatch], device=DEVICE)
                    vpred_seq, vpred_single, _ = model.forward_sequence(vpooled_all)

                    val_seq_sum += F.mse_loss(vpred_seq.squeeze(-1), vtargets).item() * vB
                    val_single_sum += F.mse_loss(vpred_single.squeeze(-1), vtargets).item() * vB
                    val_n += vB

            val_seq = val_seq_sum / max(val_n, 1)
            val_single = val_single_sum / max(val_n, 1)
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            improvement = (val_single - val_seq) / max(val_single, 1e-6) * 100

            print(f"  step {step:6d} | train_seq={loss_seq.item():.4f} train_single={loss_single.item():.4f} "
                  f"| val_seq={val_seq:.4f} val_single={val_single:.4f} (CfC {improvement:+.1f}%) "
                  f"| lr={lr:.2e} | {elapsed:.0f}s")

            if val_seq < best_val_loss:
                best_val_loss = val_seq
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "cfc_state_dict": model.cfc.state_dict(),
                    "outcome_head_state_dict": model.outcome_head.state_dict(),
                    "step": step,
                    "val_seq_mse": val_seq,
                    "val_single_mse": val_single,
                    "args": vars(args),
                }, args.output)

    print(f"\nBest val_seq_mse: {best_val_loss:.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
