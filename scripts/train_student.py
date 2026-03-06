#!/usr/bin/env python3
"""Train a student model on oracle-generated dataset.

Usage:
    python scripts/train_student.py generated/oracle_dataset.jsonl --output generated/student_model.json
    python scripts/train_student.py data.jsonl -o model.json --arch attention --layers 128 64
    python scripts/train_student.py data.jsonl -o model.json --curriculum

Architecture options:
  mlp:       input_dim -> h1 -> h2 [-> ...] -> 10  (standard MLP)
  attention: Split features into semantic blocks, apply self-attention, then MLP head
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Feature block layout for attention model (must match dataset.rs)
# ---------------------------------------------------------------------------
# These are the semantic groups of the 115 features:
FEATURE_BLOCKS = [
    ("self_state", 0, 10),        # 10 features: HP, shield, speed, etc.
    ("ability_0", 10, 15),        # 5 per ability slot
    ("ability_1", 15, 20),
    ("ability_2", 20, 25),
    ("ability_3", 25, 30),
    ("ability_4", 30, 35),
    ("ability_5", 35, 40),
    ("ability_6", 40, 45),
    ("ability_7", 45, 50),
    ("self_status", 50, 59),      # 9 status effect flags
    ("enemy_0", 59, 67),          # 8 per enemy
    ("enemy_1", 67, 75),
    ("enemy_2", 75, 83),
    ("weakest_ally", 83, 86),     # 3 features
    ("team_context", 86, 98),     # 12 features
    ("squad_coord", 98, 108),     # 10 features
    ("environment", 108, 113),    # 5 features
    ("game_phase", 113, 115),     # 2 features
]


class StudentMLP(nn.Module):
    def __init__(self, input_dim=115, hidden_sizes=(128, 64), output_dim=10, dropout=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.arch_type = "mlp"

    def forward(self, x):
        return self.net(x)

    def export_weights(self) -> dict:
        linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        exported = {"layers": []}
        for layer in linear_layers:
            exported["layers"].append({
                "w": layer.weight.cpu().T.tolist(),
                "b": layer.bias.cpu().tolist(),
            })
        sizes = [linear_layers[0].in_features] + [l.out_features for l in linear_layers]
        exported["architecture"] = sizes
        exported["type"] = "mlp"
        return exported


class StudentAttention(nn.Module):
    """Attention-based model that treats feature blocks as tokens.

    Each semantic block (self state, each ability, each enemy, etc.) is projected
    to a common embedding dimension, then self-attention lets the model learn
    cross-entity relationships (e.g., "this ability is good against that enemy").
    """

    def __init__(self, embed_dim=64, n_heads=4, n_attn_layers=2,
                 hidden_sizes=(128, 64), output_dim=10, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.blocks = FEATURE_BLOCKS

        # Per-block linear projections to embed_dim
        self.block_projections = nn.ModuleList([
            nn.Linear(end - start, embed_dim) for _, start, end in self.blocks
        ])

        # Learnable block-type embeddings (so attention knows what kind of token this is)
        self.block_type_embed = nn.Embedding(len(self.blocks), embed_dim)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_attn_layers)

        # MLP head on pooled output
        head_input = embed_dim  # mean-pool over tokens
        layers = []
        prev = head_input
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.head = nn.Sequential(*layers)
        self.arch_type = "attention"

    def forward(self, x):
        batch_size = x.shape[0]
        n_blocks = len(self.blocks)

        # Project each feature block to embed_dim
        tokens = []
        for i, (_, start, end) in enumerate(self.blocks):
            block_features = x[:, start:end]
            projected = self.block_projections[i](block_features)  # [B, embed_dim]
            type_embed = self.block_type_embed(
                torch.tensor(i, device=x.device).expand(batch_size)
            )
            tokens.append(projected + type_embed)

        # Stack into sequence: [B, n_blocks, embed_dim]
        token_seq = torch.stack(tokens, dim=1)

        # Self-attention
        attended = self.transformer(token_seq)  # [B, n_blocks, embed_dim]

        # Mean pool over tokens
        pooled = attended.mean(dim=1)  # [B, embed_dim]

        return self.head(pooled)

    def export_weights(self) -> dict:
        """Export the full model as a state dict for Rust inference.

        For attention models, we export the raw state dict since the architecture
        is more complex than simple weight matrices.
        """
        state = {}
        for k, v in self.state_dict().items():
            state[k] = v.cpu().tolist()

        exported = {
            "type": "attention",
            "embed_dim": self.embed_dim,
            "n_blocks": len(self.blocks),
            "blocks": [(name, start, end) for name, start, end in self.blocks],
            "state_dict": state,
        }

        # Also export as flat MLP for backwards compatibility:
        # Run a "compilation" pass — freeze attention and export equivalent MLP
        # (not possible in general, so we just export the attention format)
        return exported


ACTION_NAMES_10 = [
    "AttackNearest", "AttackWeakest", "DamageAbility", "HealAbility",
    "CcAbility", "DefenseAbility", "UtilityAbility", "MoveToward", "MoveAway", "Hold",
]

ACTION_NAMES_5 = [
    "AttackNearest", "AttackWeakest", "MoveToward", "MoveAway", "Hold",
]


def load_dataset(path: str):
    features = []
    labels = []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            features.append(sample["features"])
            labels.append(sample["label"])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


def train(features, labels, epochs=100, lr=1e-3, batch_size=256, val_split=0.15,
          hidden_sizes=(64, 32), dropout=0.0, weight_decay=0.0, arch="mlp",
          embed_dim=64, n_heads=4, n_attn_layers=2, curriculum=False,
          num_classes=None):
    # Auto-detect number of classes from labels
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    output_dim = num_classes

    n = len(features)
    perm = np.random.permutation(n)
    val_n = int(n * val_split)
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    X_train = torch.from_numpy(features[train_idx]).to(DEVICE)
    y_train = torch.from_numpy(labels[train_idx]).to(DEVICE)
    X_val = torch.from_numpy(features[val_idx]).to(DEVICE)
    y_val = torch.from_numpy(labels[val_idx]).to(DEVICE)

    # Class weights for imbalanced data
    class_counts = np.bincount(labels, minlength=output_dim).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    weight_tensor = torch.from_numpy(class_weights).to(DEVICE)

    input_dim = features.shape[1]
    if arch == "attention":
        model = StudentAttention(
            embed_dim=embed_dim, n_heads=n_heads, n_attn_layers=n_attn_layers,
            hidden_sizes=hidden_sizes, output_dim=output_dim, dropout=dropout
        ).to(DEVICE)
    else:
        model = StudentMLP(
            input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim, dropout=dropout
        ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warmup + cosine decay
    warmup_epochs = min(10, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Curriculum learning: start with easy samples, gradually include harder ones
    if curriculum:
        # Use oracle score as difficulty proxy — low-score samples are harder
        train_scores = np.array([
            json.loads(line)["score"]
            for i, line in enumerate(open(args.dataset))
            if i in set(train_idx)
        ], dtype=np.float32) if hasattr(args, 'dataset') else None
        # Fallback: use class frequency as difficulty (rare classes = harder)
        if train_scores is None:
            train_labels_np = labels[train_idx]
            train_scores = np.array([class_counts[l] for l in train_labels_np], dtype=np.float32)
            train_scores = train_scores.max() - train_scores  # invert: rare = high score = hard

    best_val_acc = 0.0
    best_state = None
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {arch}")
    if arch == "attention":
        print(f"  embed_dim={embed_dim}, heads={n_heads}, attn_layers={n_attn_layers}")
    arch_str = f"{arch}: " + "->".join(str(s) for s in [input_dim] + list(hidden_sizes) + [output_dim])
    print(f"Model: {arch_str}  ({param_count:,} params)")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_idx)}  Val: {val_n}  Dropout: {dropout}  WD: {weight_decay}")
    if curriculum:
        print("Curriculum learning: enabled")
    print(f"Classes: {class_counts.astype(int).tolist()}")
    print()

    for epoch in range(epochs):
        model.train()

        # Curriculum: gradually increase the fraction of training data used
        if curriculum and epoch < epochs // 2:
            frac = 0.3 + 0.7 * (epoch / (epochs // 2))  # 30% -> 100% over first half
            n_use = int(len(train_idx) * frac)
            # Sort by difficulty (easiest first), take top n_use
            if train_scores is not None:
                easy_idx = np.argsort(train_scores)[:n_use]
                curr_X = X_train[easy_idx]
                curr_y = y_train[easy_idx]
            else:
                curr_X = X_train[:n_use]
                curr_y = y_train[:n_use]
        else:
            curr_X = X_train
            curr_y = y_train

        train_ds = TensorDataset(curr_X, curr_y)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(xb)
        scheduler.step()

        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = val_logits.argmax(1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            extra = f"  data={len(curr_X)}" if curriculum and epoch < epochs // 2 else ""
            print(
                f"Epoch {epoch+1:3d}/{epochs}  "
                f"loss={total_loss/total:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                f"best={best_val_acc:.3f}{extra}"
            )

    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)

    # Per-class accuracy
    model.eval()
    with torch.no_grad():
        all_features = torch.from_numpy(features).to(DEVICE)
        # Process in chunks to avoid OOM on large datasets
        all_preds = []
        for i in range(0, len(features), 8192):
            chunk = all_features[i:i+8192]
            preds = model(chunk).argmax(1).cpu()
            all_preds.append(preds)
        all_preds = torch.cat(all_preds).numpy()

    action_names = ACTION_NAMES_5 if output_dim == 5 else ACTION_NAMES_10
    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print("\nPer-class accuracy:")
    for c in range(output_dim):
        mask = labels == c
        if mask.sum() > 0:
            name = action_names[c] if c < len(action_names) else f"Class_{c}"
            acc = (all_preds[mask] == c).mean()
            print(f"  {name:<18} {acc:.3f}  (n={mask.sum()})")

    return model


def main():
    global args
    parser = argparse.ArgumentParser(description="Train student model on oracle dataset")
    parser.add_argument("dataset", help="Path to JSONL dataset")
    parser.add_argument("--output", "-o", default="generated/student_model.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layers", type=int, nargs="+", default=[128, 64],
                        help="Hidden layer sizes for MLP head")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arch", choices=["mlp", "attention"], default="mlp")
    parser.add_argument("--embed-dim", type=int, default=64, help="Attention embedding dim")
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--n-attn-layers", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Loading dataset from {args.dataset}...")
    features, labels = load_dataset(args.dataset)
    print(f"Loaded {len(features)} samples, {features.shape[1]} features, {len(np.unique(labels))} classes")

    model = train(features, labels, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                  hidden_sizes=tuple(args.layers), dropout=args.dropout,
                  weight_decay=args.weight_decay, arch=args.arch,
                  embed_dim=args.embed_dim, n_heads=args.n_heads,
                  n_attn_layers=args.n_attn_layers, curriculum=args.curriculum)

    # Export
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    weights = model.export_weights()
    with open(args.output, "w") as f:
        json.dump(weights, f)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
