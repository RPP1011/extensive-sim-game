#!/usr/bin/env python3
"""Pre-train entity encoder V3 on fight outcome prediction.

Extends V2 with position tokens (type=4). Learns representations over
entities + threats + positions via shared self-attention.

Dataset format (JSONL, from `xtask scenario oracle outcome-dataset --v2`):
    {
        "entities": [[...], ...],     // variable-length, 30-dim each
        "entity_types": [0, 1, 1, 2], // type IDs
        "threats": [[...], ...],      // variable-length, 8-dim each
        "positions": [[...], ...],    // variable-length, 8-dim each
        "hero_wins": 1.0,
        "hero_hp_remaining": 0.45,
        "fight_progress": 0.3
    }

Usage:
    uv run --with numpy --with torch training/pretrain_entity_v3.py \
        generated/outcome_dataset_v2.jsonl \
        -o generated/entity_encoder_pretrained_v5.pt \
        --max-steps 50000
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENTITY_DIM = 30
THREAT_DIM = 8
POSITION_DIM = 8
NUM_TYPES = 5  # self=0, enemy=1, ally=2, threat=3, position=4


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EntityEncoderV3Pretraining(nn.Module):
    """EntityEncoderV3 + outcome prediction heads.

    The encoder portion matches model.py::EntityEncoderV3 exactly,
    so weights transfer directly.
    """

    def __init__(self, d_model: int = 32, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model

        # Entity/threat/position projections (match EntityEncoderV3)
        self.encoder = nn.Module()
        self.encoder.entity_proj = nn.Linear(ENTITY_DIM, d_model)
        self.encoder.threat_proj = nn.Linear(THREAT_DIM, d_model)
        self.encoder.position_proj = nn.Linear(POSITION_DIM, d_model)
        self.encoder.type_emb = nn.Embedding(NUM_TYPES, d_model)
        self.encoder.input_norm = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.encoder.out_norm = nn.LayerNorm(d_model)

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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.007)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode entities + threats + positions.

        Returns (tokens, full_mask, pooled).
        """
        B = entity_features.shape[0]
        device = entity_features.device

        # Project entities
        ent_tokens = self.encoder.entity_proj(entity_features)
        ent_type_embs = self.encoder.type_emb(entity_type_ids)
        ent_tokens = ent_tokens + ent_type_embs
        ent_tokens = self.encoder.input_norm(ent_tokens)

        # Project threats (type=3)
        thr_tokens = self.encoder.threat_proj(threat_features)
        thr_type = torch.full(
            (B, threat_features.shape[1]), 3, dtype=torch.long, device=device,
        )
        thr_tokens = thr_tokens + self.encoder.type_emb(thr_type)
        thr_tokens = self.encoder.input_norm(thr_tokens)

        # Concatenate entity + threat sequences
        tokens = torch.cat([ent_tokens, thr_tokens], dim=1)
        full_mask = torch.cat([entity_mask, threat_mask], dim=1)

        # Position tokens (type=4)
        if position_features is not None and position_mask is not None:
            pos_tokens = self.encoder.position_proj(position_features)
            pos_type = torch.full(
                (B, position_features.shape[1]), 4, dtype=torch.long, device=device,
            )
            pos_tokens = pos_tokens + self.encoder.type_emb(pos_type)
            pos_tokens = self.encoder.input_norm(pos_tokens)
            tokens = torch.cat([tokens, pos_tokens], dim=1)
            full_mask = torch.cat([full_mask, position_mask], dim=1)

        # Self-attention
        tokens = self.encoder.encoder(tokens, src_key_padding_mask=full_mask)
        tokens = self.encoder.out_norm(tokens)

        # Pool over non-padding tokens
        exist = (~full_mask).float().unsqueeze(-1)
        pooled = (tokens * exist).sum(dim=1) / exist.sum(dim=1).clamp(min=1)

        return tokens, full_mask, pooled

    def forward(
        self,
        entity_features: torch.Tensor,
        entity_type_ids: torch.Tensor,
        threat_features: torch.Tensor,
        entity_mask: torch.Tensor,
        threat_mask: torch.Tensor,
        position_features: torch.Tensor | None = None,
        position_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict fight outcome. Returns (win_logit, hp_pred)."""
        _, _, pooled = self.encode(
            entity_features, entity_type_ids, threat_features,
            entity_mask, threat_mask, position_features, position_mask,
        )
        return self.win_head(pooled), self.hp_head(pooled)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OutcomeDatasetV3:
    """Loads pre-processed .npz dataset (compact numpy arrays).

    Use convert_jsonl_to_npz() to create from JSONL first.
    Falls back to JSONL loading if .npz not found.
    """

    def __init__(self, path: Path):
        if path.suffix == ".npz":
            self._load_npz(path)
        elif path.with_suffix(".npz").exists():
            print(f"Found .npz version, loading {path.with_suffix('.npz')}")
            self._load_npz(path.with_suffix(".npz"))
        else:
            self._load_jsonl(path)

    def _load_npz(self, path: Path):
        data = np.load(path)
        self.ent_feat = data["ent_feat"]
        self.ent_types = data["ent_types"]
        self.ent_mask = data["ent_mask"]
        self.thr_feat = data["thr_feat"]
        self.thr_mask = data["thr_mask"]
        self.pos_feat = data["pos_feat"]
        self.pos_mask = data["pos_mask"]
        self.hero_wins = data["hero_wins"]
        self.hero_hp = data["hero_hp"]
        self.scenario_ids = data["scenario_ids"]
        self._indices = np.arange(len(self.hero_wins))

        n = len(self)
        wins = int((self.hero_wins > 0.5).sum())
        losses = n - wins
        n_scenarios = len(np.unique(self.scenario_ids))
        has_pos = int((~self.pos_mask[:, 0]).sum())
        print(f"Loaded {n} samples from {path}")
        print(f"  {wins} win, {losses} loss ({wins/n*100:.1f}% win)")
        print(f"  {n_scenarios} unique scenarios")
        print(f"  {has_pos}/{n} samples have position tokens ({has_pos/n*100:.1f}%)")

    def _load_jsonl(self, path: Path):
        """Fallback: convert JSONL to npz on the fly."""
        print(f"Loading JSONL {path} (consider pre-converting to .npz)...")
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))

        MAX_ENTS, MAX_THR, MAX_POS = 16, 8, 8
        n = len(samples)
        self.ent_feat = np.zeros((n, MAX_ENTS, ENTITY_DIM), dtype=np.float32)
        self.ent_types = np.zeros((n, MAX_ENTS), dtype=np.int8)
        self.ent_mask = np.ones((n, MAX_ENTS), dtype=np.bool_)
        self.thr_feat = np.zeros((n, MAX_THR, THREAT_DIM), dtype=np.float32)
        self.thr_mask = np.ones((n, MAX_THR), dtype=np.bool_)
        self.pos_feat = np.zeros((n, MAX_POS, POSITION_DIM), dtype=np.float32)
        self.pos_mask = np.ones((n, MAX_POS), dtype=np.bool_)
        self.hero_wins = np.zeros(n, dtype=np.float32)
        self.hero_hp = np.zeros(n, dtype=np.float32)

        scenario_map: dict[str, int] = {}
        self.scenario_ids = np.zeros(n, dtype=np.int32)

        for i, s in enumerate(samples):
            ents = s["entities"]
            types = s["entity_types"]
            ne = min(len(ents), MAX_ENTS)
            for j in range(ne):
                self.ent_feat[i, j] = ents[j]
                self.ent_types[i, j] = types[j]
                self.ent_mask[i, j] = False

            threats = s.get("threats", [])
            nt = min(len(threats), MAX_THR)
            for j in range(nt):
                self.thr_feat[i, j] = threats[j]
                self.thr_mask[i, j] = False

            positions = s.get("positions", [])
            np_ = min(len(positions), MAX_POS)
            for j in range(np_):
                self.pos_feat[i, j] = positions[j]
                self.pos_mask[i, j] = False

            self.hero_wins[i] = s["hero_wins"]
            self.hero_hp[i] = s["hero_hp_remaining"]

            sc = s.get("scenario", "unknown")
            if sc not in scenario_map:
                scenario_map[sc] = len(scenario_map)
            self.scenario_ids[i] = scenario_map[sc]

        self._indices = np.arange(n)
        wins = int((self.hero_wins > 0.5).sum())
        print(f"Loaded {n} samples, {wins} wins, {len(scenario_map)} scenarios")

    def __len__(self) -> int:
        return len(self._indices)

    def split(self, val_frac: float = 0.15) -> tuple["OutcomeDatasetV3", "OutcomeDatasetV3"]:
        """Split by scenario to prevent data leakage."""
        unique_scenarios = np.unique(self.scenario_ids[self._indices])
        random.shuffle(unique_scenarios)
        n_val = max(1, int(len(unique_scenarios) * val_frac))
        val_set = set(unique_scenarios[:n_val].tolist())

        val_mask = np.array([self.scenario_ids[i] in val_set for i in self._indices])
        train_idx = self._indices[~val_mask]
        val_idx = self._indices[val_mask]

        train_ds = OutcomeDatasetV3.__new__(OutcomeDatasetV3)
        val_ds = OutcomeDatasetV3.__new__(OutcomeDatasetV3)
        # Share underlying arrays, just use different index sets
        for ds, idx in [(train_ds, train_idx), (val_ds, val_idx)]:
            ds.ent_feat = self.ent_feat
            ds.ent_types = self.ent_types
            ds.ent_mask = self.ent_mask
            ds.thr_feat = self.thr_feat
            ds.thr_mask = self.thr_mask
            ds.pos_feat = self.pos_feat
            ds.pos_mask = self.pos_mask
            ds.hero_wins = self.hero_wins
            ds.hero_hp = self.hero_hp
            ds.scenario_ids = self.scenario_ids
            ds._indices = idx

        n_val_sc = len(val_set)
        n_train_sc = len(unique_scenarios) - n_val_sc
        print(f"Split: {len(train_ds)} train ({n_train_sc} scenarios), "
              f"{len(val_ds)} val ({n_val_sc} scenarios)")
        return train_ds, val_ds

    def collate(self, indices: list[int]) -> dict[str, torch.Tensor]:
        real_idx = self._indices[indices]
        return {
            "entity_features": torch.tensor(
                self.ent_feat[real_idx], dtype=torch.float, device=DEVICE),
            "entity_type_ids": torch.tensor(
                self.ent_types[real_idx], dtype=torch.long, device=DEVICE),
            "threat_features": torch.tensor(
                self.thr_feat[real_idx], dtype=torch.float, device=DEVICE),
            "entity_mask": torch.tensor(
                self.ent_mask[real_idx], device=DEVICE),
            "threat_mask": torch.tensor(
                self.thr_mask[real_idx], device=DEVICE),
            "position_features": torch.tensor(
                self.pos_feat[real_idx], dtype=torch.float, device=DEVICE),
            "position_mask": torch.tensor(
                self.pos_mask[real_idx], device=DEVICE),
            "hero_wins": torch.tensor(
                self.hero_wins[real_idx], dtype=torch.float, device=DEVICE),
            "hero_hp": torch.tensor(
                self.hero_hp[real_idx], dtype=torch.float, device=DEVICE),
        }

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        indices = random.choices(range(len(self)), k=batch_size)
        return self.collate(indices)

    def iter_batches(self, batch_size: int, shuffle: bool = True):
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            if batch_idx:
                yield self.collate(batch_idx)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = OutcomeDatasetV3(Path(args.dataset))
    train_ds, val_ds = dataset.split(val_frac=args.val_frac)

    model = EntityEncoderV3Pretraining(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(DEVICE)

    # Optionally load V2 pretrained weights (entity_proj, threat_proj, type_emb transfer)
    if args.init_from:
        print(f"Initializing from V2 encoder: {args.init_from}")
        state = torch.load(args.init_from, map_location=DEVICE, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded {len(state) - len(unexpected)} params, "
              f"missing: {len(missing)}, unexpected: {len(unexpected)}")

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    print(f"Model: {n_params:,} total, encoder: {n_encoder:,}")

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
        "val_win_acc", "val_hp_mae", "weight_norm", "grad_norm",
        "max_eigenvalue", "lr", "elapsed_s",
    ])

    best_metric = 0.0
    start_time = time.time()
    model.train()

    print(f"\nStarting entity encoder V3 pre-training: max_steps={args.max_steps}")
    print(f"Weight decay={args.weight_decay}, lr={args.lr}")
    print(f"Grokfast EMA: alpha={args.grokfast_alpha}, lamb={args.grokfast_lamb}")
    print(f"Device: {DEVICE}\n")

    for step in range(1, args.max_steps + 1):
        batch = train_ds.sample_batch(batch_size)
        win_logit, hp_pred = model(
            batch["entity_features"], batch["entity_type_ids"],
            batch["threat_features"], batch["entity_mask"], batch["threat_mask"],
            batch["position_features"], batch["position_mask"],
        )

        win_loss = F.binary_cross_entropy_with_logits(
            win_logit.squeeze(-1), batch["hero_wins"]
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
                        vb["position_features"], vb["position_mask"],
                    )
                pred_win = (vw.squeeze(-1) > 0).float()
                correct += (pred_win == vb["hero_wins"]).sum().item()
                hp_errors.append((vh.squeeze(-1) - vb["hero_hp"]).abs())
                total += len(vb["hero_wins"])

            val_win_acc = correct / max(total, 1)
            val_hp_mae = torch.cat(hp_errors).mean().item()
            weight_norm = sum(
                p.data.norm().item() ** 2 for p in model.parameters()
            ) ** 0.5

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
                step, f"{loss.item():.6f}", f"{win_loss.item():.6f}",
                f"{hp_loss.item():.6f}", f"{val_win_acc:.4f}",
                f"{val_hp_mae:.4f}", f"{weight_norm:.4f}",
                f"{grad_norm:.4f}", f"{max_eig:.4f}", f"{lr:.6f}",
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
    p = argparse.ArgumentParser(
        description="Pre-train entity encoder V3 (with position tokens) on outcome prediction"
    )
    p.add_argument("dataset", help="JSONL v2 dataset with entities/threats/positions")
    p.add_argument("-o", "--output", default="generated/entity_encoder_pretrained_v5.pt")

    p.add_argument("--init-from", help="V2 entity encoder checkpoint to initialize from")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=50_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)

    p.add_argument("--grokfast-alpha", type=float, default=0.98)
    p.add_argument("--grokfast-lamb", type=float, default=2.0)

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
