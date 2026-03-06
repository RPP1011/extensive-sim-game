#!/usr/bin/env python3
"""Train per-category ability evaluators on oracle-generated dataset.

Usage:
    python scripts/train_ability_eval.py generated/ability_eval_dataset.jsonl
    python scripts/train_ability_eval.py data.jsonl --output generated/ability_eval_weights.json
    python scripts/train_ability_eval.py data.jsonl --categories damage_unit cc_unit heal_unit

Each category gets a tiny MLP that outputs:
  - urgency (regression, sigmoid output)
  - target_idx (classification among top-3 candidates, for unit-targeted categories)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Categories and their output structure
# Unit-targeted categories output urgency + 3 target logits
# Simple categories output urgency only
UNIT_TARGET_CATEGORIES = {"damage_unit", "cc_unit", "heal_unit"}
AOE_CATEGORIES = {"damage_aoe", "obstacle"}
SIMPLE_CATEGORIES = {"heal_aoe", "defense", "utility", "summon"}


class AbilityEvalMLP(nn.Module):
    """Tiny MLP for a single ability category."""

    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

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
        return exported


def load_dataset(path: str):
    """Load JSONL dataset, split by category."""
    by_category = defaultdict(lambda: {"features": [], "urgency": [], "target_idx": []})

    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            cat = sample["category"]
            by_category[cat]["features"].append(sample["features"])
            by_category[cat]["urgency"].append(sample["urgency"])
            by_category[cat]["target_idx"].append(sample.get("target_idx", 0))

    result = {}
    for cat, data in by_category.items():
        result[cat] = {
            "features": np.array(data["features"], dtype=np.float32),
            "urgency": np.array(data["urgency"], dtype=np.float32),
            "target_idx": np.array(data["target_idx"], dtype=np.int64),
        }
    return result


def train_category(cat_name, features, urgency, target_idx,
                   epochs=200, lr=1e-3, batch_size=128, val_split=0.15,
                   hidden_sizes=None):
    """Train a single category evaluator."""
    n = len(features)
    if n < 10:
        print(f"  Skipping {cat_name}: only {n} samples")
        return None

    input_dim = features.shape[1]

    # Determine output dim and architecture
    if cat_name in UNIT_TARGET_CATEGORIES:
        output_dim = 4  # 1 urgency + 3 target logits
        if hidden_sizes is None:
            hidden_sizes = (32, 16)
    elif cat_name in AOE_CATEGORIES:
        output_dim = 1  # just urgency (position is from candidate list)
        if hidden_sizes is None:
            hidden_sizes = (32, 16)
    else:
        output_dim = 1  # urgency only
        if hidden_sizes is None:
            hidden_sizes = (16, 8)

    # Split
    perm = np.random.permutation(n)
    val_n = max(1, int(n * val_split))
    val_idx, train_idx = perm[:val_n], perm[val_n:]

    X_train = torch.from_numpy(features[train_idx]).to(DEVICE)
    X_val = torch.from_numpy(features[val_idx]).to(DEVICE)
    u_train = torch.from_numpy(urgency[train_idx]).to(DEVICE)
    u_val = torch.from_numpy(urgency[val_idx]).to(DEVICE)
    t_train = torch.from_numpy(target_idx[train_idx]).to(DEVICE)
    t_val = torch.from_numpy(target_idx[val_idx]).to(DEVICE)

    model = AbilityEvalMLP(input_dim, hidden_sizes, output_dim).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine decay
    warmup_epochs = min(10, epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    urgency_loss_fn = nn.MSELoss()
    target_loss_fn = nn.CrossEntropyLoss()
    target_weight = 0.5  # lambda for target loss

    param_count = sum(p.numel() for p in model.parameters())
    arch_str = "->".join(str(s) for s in [input_dim] + list(hidden_sizes) + [output_dim])
    print(f"  {cat_name}: {arch_str} ({param_count:,} params), {len(train_idx)} train, {val_n} val")

    best_val_loss = float("inf")
    best_state = None
    has_targets = cat_name in UNIT_TARGET_CATEGORIES

    for epoch in range(epochs):
        model.train()
        train_ds = TensorDataset(X_train, u_train, t_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        total = 0
        for xb, ub, tb in train_dl:
            optimizer.zero_grad()
            out = model(xb)

            # Urgency loss: sigmoid of first output vs target urgency
            pred_urgency = torch.sigmoid(out[:, 0])
            loss = urgency_loss_fn(pred_urgency, ub)

            # Target loss (for unit-targeted categories)
            if has_targets and out.shape[1] > 1:
                target_logits = out[:, 1:]
                loss = loss + target_weight * target_loss_fn(target_logits, tb)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            total += len(xb)

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_pred_urgency = torch.sigmoid(val_out[:, 0])
            val_loss = urgency_loss_fn(val_pred_urgency, u_val).item()

            if has_targets and val_out.shape[1] > 1:
                val_target_logits = val_out[:, 1:]
                val_loss += target_weight * target_loss_fn(val_target_logits, t_val).item()
                val_target_acc = (val_target_logits.argmax(1) == t_val).float().mean().item()
            else:
                val_target_acc = 0.0

            # Urgency MAE
            val_urgency_mae = (val_pred_urgency - u_val).abs().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0 or epoch == 0:
            target_str = f"  target_acc={val_target_acc:.3f}" if has_targets else ""
            print(
                f"    Epoch {epoch+1:3d}/{epochs}  "
                f"loss={total_loss/total:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"urgency_mae={val_urgency_mae:.3f}{target_str}"
            )

    if best_state:
        model.load_state_dict(best_state)

    # Final stats
    model.eval()
    with torch.no_grad():
        all_X = torch.from_numpy(features).to(DEVICE)
        all_out = model(all_X)
        all_pred = torch.sigmoid(all_out[:, 0]).cpu().numpy()

    print(f"    Best val loss: {best_val_loss:.4f}")
    print(f"    Urgency range: pred [{all_pred.min():.3f}, {all_pred.max():.3f}]  "
          f"true [{urgency.min():.3f}, {urgency.max():.3f}]")

    # Show urgency distribution
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        above_pred = (all_pred >= threshold).mean()
        above_true = (urgency >= threshold).mean()
        print(f"    urgency >= {threshold:.1f}: pred {above_pred:.1%}  true {above_true:.1%}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train ability evaluators")
    parser.add_argument("dataset", help="Path to JSONL dataset")
    parser.add_argument("--output", "-o", default="generated/ability_eval_weights.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Only train these categories (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Loading dataset from {args.dataset}...")
    by_category = load_dataset(args.dataset)
    print(f"Categories found: {list(by_category.keys())}")
    for cat, data in sorted(by_category.items()):
        print(f"  {cat}: {len(data['features'])} samples, "
              f"{data['features'].shape[1]} features, "
              f"avg_urgency={data['urgency'].mean():.3f}")
    print(f"Device: {DEVICE}\n")

    all_weights = {}

    for cat_name, data in sorted(by_category.items()):
        if args.categories and cat_name not in args.categories:
            continue

        print(f"\nTraining {cat_name}...")
        model = train_category(
            cat_name,
            data["features"],
            data["urgency"],
            data["target_idx"],
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )

        if model is not None:
            all_weights[cat_name] = model.export_weights()

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_weights, f)
    print(f"\nWeights saved to {args.output}")
    print(f"Categories trained: {list(all_weights.keys())}")


if __name__ == "__main__":
    main()
