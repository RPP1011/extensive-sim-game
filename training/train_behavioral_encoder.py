#!/usr/bin/env python3
"""Train behavioral ability encoder: learn embeddings that predict sim outcomes.

Each ability gets a learned embedding vector. An MLP context encoder processes
the 4-dim condition (hp_pct, distance, n_targets, armor) and combines it with
the ability embedding to predict the 119-dim outcome vector.

Architecture:
    ability_embedding[ability_id]  (embed_dim)
    context_encoder(condition)     (embed_dim)
    predictor(ability_emb + context_emb) → outcome (119-dim)

Loss: MSE on outcome vector, with optional per-group weighting.

Usage:
    uv run --with numpy --with torch training/train_behavioral_encoder.py \
        dataset/ability_profiles.npz \
        -o generated/behavioral_encoder.pt \
        --embed-dim 32 --max-steps 50000
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITION_DIM = 4   # hp_pct, distance, n_targets, armor
OUTCOME_DIM = 119   # 4 targets × 23 + caster × 23 + 4 aggregates
PER_TARGET_DIM = 23


class BehavioralEncoder(nn.Module):
    """Ability embedding + condition encoder → outcome predictor."""

    def __init__(self, n_abilities: int, embed_dim: int = 32,
                 hidden_dim: int = 128, n_hidden: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.ability_emb = nn.Embedding(n_abilities, embed_dim)

        # Context encoder: condition → embed_dim
        self.context_enc = nn.Sequential(
            nn.Linear(CONDITION_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Outcome predictor: (ability_emb + context_emb) → 119
        layers = []
        in_dim = embed_dim * 2  # concat ability + context
        for _ in range(n_hidden):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, OUTCOME_DIM))
        self.predictor = nn.Sequential(*layers)

    def get_embedding(self, ability_ids: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized ability embeddings."""
        emb = self.ability_emb(ability_ids)
        return F.normalize(emb, dim=-1)

    def forward(self, ability_ids: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Predict outcome from ability ID and condition."""
        ability_emb = self.get_embedding(ability_ids)
        context_emb = self.context_enc(conditions)
        combined = torch.cat([ability_emb, context_emb], dim=-1)
        return self.predictor(combined)


def load_dataset(path: Path):
    """Load npz and split by ability (80/20)."""
    data = np.load(path, allow_pickle=True)
    ability_ids = data["ability_id"].astype(np.int64)
    conditions = data["condition"].astype(np.float32)
    outcomes = data["outcome"].astype(np.float32)

    # Decode ability names
    ability_names_bytes = data["ability_names"].tobytes()
    ability_names = ability_names_bytes.decode("utf-8").split("\n")

    n_abilities = len(ability_names)

    # Split by sample (not by ability) — every ability needs its embedding trained.
    # Stratified: for each ability, 80% of its conditions go to train, 20% to val.
    n = len(ability_ids)
    rng = np.random.RandomState(42)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)

    for aid in np.unique(ability_ids):
        indices = np.where(ability_ids == aid)[0]
        rng.shuffle(indices)
        split = max(1, int(len(indices) * 0.8))
        train_mask[indices[:split]] = True
        val_mask[indices[split:]] = True

    print(f"Loaded {n} samples, {n_abilities} abilities")
    print(f"Train: {train_mask.sum()} samples, Val: {val_mask.sum()} samples")

    return {
        "ability_ids": ability_ids,
        "conditions": conditions,
        "outcomes": outcomes,
        "ability_names": ability_names,
        "n_abilities": n_abilities,
        "train_mask": train_mask,
        "val_mask": val_mask,
    }


def sample_batch(data: dict, mask: np.ndarray, batch_size: int, device: torch.device):
    """Sample a batch from masked subset."""
    indices = np.where(mask)[0]
    chosen = np.random.choice(indices, size=min(batch_size, len(indices)), replace=False)
    return {
        "ability_ids": torch.from_numpy(data["ability_ids"][chosen]).to(device),
        "conditions": torch.from_numpy(data["conditions"][chosen]).to(device),
        "outcomes": torch.from_numpy(data["outcomes"][chosen]).to(device),
    }


def compute_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """MSE loss with per-group breakdown."""
    mse = F.mse_loss(pred, target)

    # Per-group MSE for monitoring
    with torch.no_grad():
        # Targets 0-3: dims 0..92
        target_mse = F.mse_loss(pred[:, :92], target[:, :92])
        # Caster: dims 92..115
        caster_mse = F.mse_loss(pred[:, 92:115], target[:, 92:115])
        # Aggregates: dims 115..119
        agg_mse = F.mse_loss(pred[:, 115:119], target[:, 115:119])

    return mse, {"target": target_mse.item(), "caster": caster_mse.item(), "agg": agg_mse.item()}


def evaluate(model: BehavioralEncoder, data: dict, device: torch.device,
             batch_size: int = 4096) -> dict:
    """Evaluate on validation set."""
    model.eval()
    val_indices = np.where(data["val_mask"])[0]
    total_loss = 0.0
    total_samples = 0

    # Also track per-ability mean prediction quality
    with torch.no_grad():
        for start in range(0, len(val_indices), batch_size):
            idx = val_indices[start:start + batch_size]
            batch = {
                "ability_ids": torch.from_numpy(data["ability_ids"][idx]).to(device),
                "conditions": torch.from_numpy(data["conditions"][idx]).to(device),
                "outcomes": torch.from_numpy(data["outcomes"][idx]).to(device),
            }
            pred = model(batch["ability_ids"], batch["conditions"])
            loss = F.mse_loss(pred, batch["outcomes"], reduction="sum")
            total_loss += loss.item()
            total_samples += len(idx)

    model.train()
    return {"val_mse": total_loss / total_samples / OUTCOME_DIM}


def nearest_neighbors(model: BehavioralEncoder, data: dict, device: torch.device, k: int = 5):
    """Find k nearest neighbors for each ability by cosine similarity of embeddings."""
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(data["n_abilities"], device=device)
        embs = model.get_embedding(all_ids)  # (N, embed_dim)
        sim = embs @ embs.T  # cosine similarity (embeddings are L2-normalized)
        # Zero out self-similarity
        sim.fill_diagonal_(-1.0)
        topk_vals, topk_ids = sim.topk(k, dim=1)

    names = data["ability_names"]
    results = []
    for i in range(len(names)):
        neighbors = [(names[topk_ids[i, j].item()], topk_vals[i, j].item()) for j in range(k)]
        results.append((names[i], neighbors))
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser(description="Train behavioral ability encoder")
    parser.add_argument("data", type=Path, help="Path to ability_profiles.npz")
    parser.add_argument("-o", "--output", type=Path, default=Path("generated/behavioral_encoder.pt"))
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-hidden", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--log", type=Path, default=None, help="CSV log path")
    parser.add_argument("--export-embeddings", type=Path, default=None,
                        help="Export embeddings JSON after training")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    data = load_dataset(args.data)

    model = BehavioralEncoder(
        n_abilities=data["n_abilities"],
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params (embed_dim={args.embed_dim}, hidden={args.hidden_dim})")

    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from {args.resume}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay, betas=(0.9, 0.98))

    # Preload all data to GPU
    all_train_ids = torch.from_numpy(data["ability_ids"][data["train_mask"]]).to(DEVICE)
    all_train_cond = torch.from_numpy(data["conditions"][data["train_mask"]]).to(DEVICE)
    all_train_out = torch.from_numpy(data["outcomes"][data["train_mask"]]).to(DEVICE)
    n_train = len(all_train_ids)
    print(f"Training data on {DEVICE}: {n_train} samples")

    log_file = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, "w", newline="")
        writer = csv.writer(log_file)
        writer.writerow(["step", "train_mse", "val_mse", "target_mse", "caster_mse", "agg_mse"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for step in range(1, args.max_steps + 1):
        # Sample batch from GPU tensors
        idx = torch.randint(0, n_train, (args.batch_size,), device=DEVICE)
        ability_ids = all_train_ids[idx]
        conditions = all_train_cond[idx]
        outcomes = all_train_out[idx]

        pred = model(ability_ids, conditions)
        loss, groups = compute_loss(pred, outcomes)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == 1:
            val_metrics = evaluate(model, data, DEVICE)
            elapsed = time.time() - t0
            print(f"[{step:6d}] train_mse={loss.item():.6f}  val_mse={val_metrics['val_mse']:.6f}  "
                  f"target={groups['target']:.4f} caster={groups['caster']:.4f} agg={groups['agg']:.4f}  "
                  f"({elapsed:.0f}s)")

            if log_file:
                writer.writerow([step, f"{loss.item():.6f}", f"{val_metrics['val_mse']:.6f}",
                                 f"{groups['target']:.6f}", f"{groups['caster']:.6f}",
                                 f"{groups['agg']:.6f}"])
                log_file.flush()

            if val_metrics["val_mse"] < best_val:
                best_val = val_metrics["val_mse"]
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "n_abilities": data["n_abilities"],
                        "embed_dim": args.embed_dim,
                        "hidden_dim": args.hidden_dim,
                        "n_hidden": args.n_hidden,
                    },
                    "step": step,
                    "val_mse": best_val,
                    "ability_names": data["ability_names"],
                }, args.output)
                print(f"  → saved best (val_mse={best_val:.6f})")

    # Final eval
    val_metrics = evaluate(model, data, DEVICE)
    print(f"\nFinal val_mse={val_metrics['val_mse']:.6f} (best={best_val:.6f})")

    # Nearest neighbor analysis
    print("\n=== Nearest Neighbors (sample) ===")
    nn_results = nearest_neighbors(model, data, DEVICE, k=5)

    # Show a diverse sample: pick abilities with non-zero outcomes
    nonzero_abilities = set()
    for i, name in enumerate(data["ability_names"]):
        mask = data["ability_ids"] == i
        if np.any(data["outcomes"][mask] != 0):
            nonzero_abilities.add(i)

    shown = 0
    for i, (name, neighbors) in enumerate(nn_results):
        if i not in nonzero_abilities:
            continue
        neighbor_str = ", ".join(f"{n}({s:.3f})" for n, s in neighbors)
        print(f"  {name}: {neighbor_str}")
        shown += 1
        if shown >= 20:
            break

    # Export embeddings if requested
    export_path = args.export_embeddings or Path("generated/behavioral_embeddings.json")
    export_embeddings(model, data, DEVICE, export_path)

    if log_file:
        log_file.close()

    print(f"\nDone. Model: {args.output}, Embeddings: {export_path}")


def export_embeddings(model: BehavioralEncoder, data: dict, device: torch.device, path: Path):
    """Export ability embeddings to JSON."""
    model.eval()
    with torch.no_grad():
        all_ids = torch.arange(data["n_abilities"], device=device)
        embs = model.get_embedding(all_ids).cpu().numpy()

    embeddings = {}
    for i, name in enumerate(data["ability_names"]):
        embeddings[name] = embs[i].tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "embed_dim": model.embed_dim,
            "n_abilities": data["n_abilities"],
            "embeddings": embeddings,
        }, f)

    print(f"Exported {len(embeddings)} embeddings to {path}")


if __name__ == "__main__":
    main()
