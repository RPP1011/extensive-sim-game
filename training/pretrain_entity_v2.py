#!/usr/bin/env python3
"""Pre-train entity encoder V2 on fight outcome prediction.

V2 changes from V1:
  - Variable-length entities (no 3+3 cap) — all living units included
  - Threat tokens (projectiles, zones, enemy casts) as separate 8-dim features
  - 4 type embeddings: self=0, enemy=1, ally=2, threat=3
  - Dual projections: 30-dim entities, 8-dim threats → shared d_model
  - Deeper model: 4-6 layers (default 4)
  - Padded to max-in-batch for efficient batching

Dataset format (JSONL):
    {
        "entities": [[0.5, ...], ...],      // N × 30
        "entity_types": [0, 1, 1, 2, ...],  // N
        "threats": [[0.1, ...], ...],        // M × 8
        "hero_wins": 1.0,
        "hero_hp_remaining": 0.45,
        "fight_progress": 0.3
    }

Usage:
    uv run --with numpy --with torch training/pretrain_entity_v2.py \
        generated/outcome_dataset_v2.jsonl \
        -o generated/entity_encoder_pretrained_v4.pt \
        --max-steps 300000
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

sys.path.insert(0, str(Path(__file__).parent))
from grokfast import GrokfastEMA
from model import EntityEncoderV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model: entity encoder + outcome heads
# ---------------------------------------------------------------------------

class EntityEncoderV2Pretraining(nn.Module):
    """V2 entity encoder with outcome prediction heads."""

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.encoder = EntityEncoderV2(
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        )
        self.d_model = d_model

        # Outcome prediction heads
        self.win_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.hp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Reduced init scale (3x smaller) per Kumar et al. (2310.06110)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.007)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict fight outcome.

        Returns (win_logit, hp_pred), both (B, 1).
        """
        tokens, full_mask = self.encoder(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask,
        )

        # Mean pool over existing tokens
        exist_mask = (~full_mask).float().unsqueeze(-1)  # (B, S, 1)
        pooled = (tokens * exist_mask).sum(dim=1) / exist_mask.sum(dim=1).clamp(min=1)

        return self.win_head(pooled), self.hp_head(pooled)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OutcomeDatasetV2:
    def __init__(self, path: Path):
        self.samples: list[dict] = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {path}")
        wins = sum(1 for s in self.samples if s["hero_wins"] > 0.5)
        losses = len(self.samples) - wins
        print(f"  {wins} win snapshots, {losses} loss snapshots ({wins/(wins+losses)*100:.1f}% win)")

        # Stats on entity/threat counts
        n_ents = [len(s["entities"]) for s in self.samples]
        n_threats = [len(s["threats"]) for s in self.samples]
        print(f"  entities: min={min(n_ents)}, max={max(n_ents)}, mean={np.mean(n_ents):.1f}")
        print(f"  threats: min={min(n_threats)}, max={max(n_threats)}, mean={np.mean(n_threats):.1f}")

    def __len__(self) -> int:
        return len(self.samples)

    def split(self, val_frac: float = 0.15) -> tuple["OutcomeDatasetV2", "OutcomeDatasetV2"]:
        n_val = max(1, int(len(self.samples) * val_frac))
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        val_ds = OutcomeDatasetV2.__new__(OutcomeDatasetV2)
        val_ds.samples = [self.samples[i] for i in indices[:n_val]]

        train_ds = OutcomeDatasetV2.__new__(OutcomeDatasetV2)
        train_ds.samples = [self.samples[i] for i in indices[n_val:]]

        print(f"Split: {len(train_ds)} train, {len(val_ds)} val")
        return train_ds, val_ds

    def _collate(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Pad variable-length entities and threats to max-in-batch."""
        max_ents = max(len(s["entities"]) for s in batch)
        max_threats = max(len(s["threats"]) for s in batch) if any(s["threats"] for s in batch) else 1
        B = len(batch)

        ent_dim = 30
        threat_dim = 8

        entity_features = torch.zeros(B, max_ents, ent_dim, device=DEVICE)
        entity_type_ids = torch.zeros(B, max_ents, dtype=torch.long, device=DEVICE)
        entity_mask = torch.ones(B, max_ents, dtype=torch.bool, device=DEVICE)  # True = padding

        threat_features = torch.zeros(B, max_threats, threat_dim, device=DEVICE)
        threat_mask = torch.ones(B, max_threats, dtype=torch.bool, device=DEVICE)

        for i, s in enumerate(batch):
            n_e = len(s["entities"])
            entity_features[i, :n_e] = torch.tensor(s["entities"], dtype=torch.float)
            entity_type_ids[i, :n_e] = torch.tensor(s["entity_types"], dtype=torch.long)
            entity_mask[i, :n_e] = False

            n_t = len(s["threats"])
            if n_t > 0:
                threat_features[i, :n_t] = torch.tensor(s["threats"], dtype=torch.float)
                threat_mask[i, :n_t] = False

        return {
            "entity_features": entity_features,
            "entity_type_ids": entity_type_ids,
            "threat_features": threat_features,
            "entity_mask": entity_mask,
            "threat_mask": threat_mask,
            "hero_wins": torch.tensor([s["hero_wins"] for s in batch], dtype=torch.float, device=DEVICE),
            "hero_hp": torch.tensor([s["hero_hp_remaining"] for s in batch], dtype=torch.float, device=DEVICE),
        }

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = random.choices(range(len(self.samples)), k=batch_size)
        batch = [self.samples[i] for i in indices]
        return self._collate(batch)

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            if batch_idx:
                batch = [self.samples[j] for j in batch_idx]
                yield self._collate(batch)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = OutcomeDatasetV2(Path(args.dataset))
    train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    model = EntityEncoderV2Pretraining(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10,
    )

    batch_size = min(args.batch_size, len(train_ds) // 2) if len(train_ds) > 4 else len(train_ds)

    gf = GrokfastEMA(model, alpha=args.grokfast_alpha, lamb=args.grokfast_lamb)

    log_path = Path(args.output).with_suffix(".csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "step", "train_loss", "win_loss", "hp_loss",
        "val_win_acc", "val_hp_mae", "weight_norm", "grad_norm", "max_eigenvalue", "lr", "elapsed_s",
    ])

    best_metric = 0.0
    start_time = time.time()
    model.train()

    print(f"\nStarting entity encoder V2 pre-training: max_steps={args.max_steps}")
    print(f"d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print(f"Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Grokfast EMA: alpha={args.grokfast_alpha}, lamb={args.grokfast_lamb}")
    print(f"Device: {DEVICE}\n")

    for step in range(1, args.max_steps + 1):
        batch = train_ds.sample_batch(batch_size)
        win_logit, hp_pred = model(
            batch["entity_features"], batch["entity_type_ids"],
            batch["threat_features"], batch["entity_mask"], batch["threat_mask"],
        )

        win_loss = F.binary_cross_entropy_with_logits(
            win_logit.squeeze(-1), batch["hero_wins"],
        )
        hp_loss = F.mse_loss(hp_pred.squeeze(-1), batch["hero_hp"])
        loss = win_loss + hp_loss

        optimizer.zero_grad()
        loss.backward()
        gf.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step <= 10:
            warmup.step()

        if step % args.eval_every == 0:
            model.eval()

            total, correct, hp_errors = 0, 0, []
            for vb in val_ds.iter_batches(batch_size, shuffle=False):
                with torch.no_grad():
                    vw, vh = model(
                        vb["entity_features"], vb["entity_type_ids"],
                        vb["threat_features"], vb["entity_mask"], vb["threat_mask"],
                    )
                pred_win = (vw.squeeze(-1) > 0).float()
                correct += (pred_win == vb["hero_wins"]).sum().item()
                hp_errors.append((vh.squeeze(-1) - vb["hero_hp"]).abs())
                total += len(vb["hero_wins"])

            val_win_acc = correct / max(total, 1)
            val_hp_mae = torch.cat(hp_errors).mean().item()
            weight_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

            max_eig = 0.0
            for p in model.parameters():
                if p.ndim == 2 and p.shape[0] >= 4 and p.shape[1] >= 4:
                    try:
                        s = torch.linalg.svdvals(p.data)
                        max_eig = max(max_eig, s[0].item())
                    except Exception:
                        pass

            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]

            log_writer.writerow([
                step, f"{loss.item():.6f}", f"{win_loss.item():.6f}", f"{hp_loss.item():.6f}",
                f"{val_win_acc:.4f}", f"{val_hp_mae:.4f}",
                f"{weight_norm:.4f}", f"{grad_norm:.4f}", f"{max_eig:.4f}", f"{lr:.6f}",
                f"{elapsed:.1f}",
            ])
            log_file.flush()

            marker = ""
            if val_win_acc > best_metric:
                best_metric = val_win_acc
                torch.save(model.state_dict(), args.output)
                marker = " *"

            print(
                f"step {step:>7d} | "
                f"loss {loss.item():.4f} (win={win_loss.item():.4f} hp={hp_loss.item():.4f}) | "
                f"val_acc {val_win_acc:.4f} | hp_mae {val_hp_mae:.4f} | "
                f"w_norm {weight_norm:.1f} | eig {max_eig:.2f}"
                f"{marker}"
            )

            model.train()

    log_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete. {step} steps in {elapsed:.0f}s")
    print(f"Best val win accuracy: {best_metric:.4f}")
    print(f"Model saved to {args.output}")
    print(f"Metrics saved to {log_path}")


def main():
    p = argparse.ArgumentParser(description="Pre-train entity encoder V2 on outcome prediction")
    p.add_argument("dataset", help="JSONL dataset with entities + threats + hero_wins")
    p.add_argument("-o", "--output", default="generated/entity_encoder_pretrained_v4.pt")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)

    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
