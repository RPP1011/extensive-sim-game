#!/usr/bin/env python3
"""REINFORCE training on self-play episodes.

Usage:
    # Generate episodes (Rust):
    cargo run --release --bin xtask -- scenario oracle self-play generate scenarios/ -j 8 --episodes 20

    # Train:
    python scripts/train_selfplay.py generated/self_play_episodes.jsonl -o generated/selfplay_policy.json

    # Generate more episodes with trained policy, then retrain (iterate):
    cargo run --release --bin xtask -- scenario oracle self-play generate scenarios/ \
        --policy generated/selfplay_policy.json --episodes 20 -o generated/sp_ep_iter2.jsonl
    python scripts/train_selfplay.py generated/sp_ep_iter2.jsonl -o generated/selfplay_policy.json \
        --resume generated/selfplay_policy.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def export_weights(self) -> dict:
        """Export in the format expected by Rust PolicyWeights."""
        layers = []
        for m in self.net:
            if isinstance(m, nn.Linear):
                layers.append({
                    "w": m.weight.cpu().T.tolist(),  # [in, out]
                    "b": m.bias.cpu().tolist(),
                })
        return {"layers": layers}


def load_episodes(path: str):
    episodes = []
    with open(path) as f:
        for line in f:
            ep = json.loads(line)
            episodes.append(ep)
    return episodes


def episodes_to_tensors(episodes, gamma=0.99):
    """Convert episodes to training tensors.

    Uses per-step rewards with discounted future returns.
    step_reward gives per-tick HP-delta signal; episode reward adds terminal bonus.
    """
    all_features = []
    all_actions = []
    all_masks = []
    all_returns = []
    all_old_log_probs = []

    for ep in episodes:
        if not ep["steps"]:
            continue

        n = len(ep["steps"])
        terminal_bonus = ep["reward"]  # +1 win, -1 loss, shaped on timeout

        # Compute discounted returns from per-step rewards + terminal bonus
        returns = [0.0] * n
        G = terminal_bonus  # terminal reward
        for i in range(n - 1, -1, -1):
            step_r = ep["steps"][i].get("step_reward", 0.0)
            G = step_r + gamma * G
            returns[i] = G

        for i, step in enumerate(ep["steps"]):
            all_features.append(step["features"])
            all_actions.append(step["action"])
            all_masks.append(step["mask"])
            all_returns.append(returns[i])
            all_old_log_probs.append(step.get("log_prob", 0.0))

    if not all_features:
        return None, None, None, None

    features = torch.tensor(all_features, dtype=torch.float32)
    # Normalize by max absolute value per feature
    feat_scale = features.abs().amax(dim=0).clamp(min=1e-6)
    features = features / feat_scale
    actions = torch.tensor(all_actions, dtype=torch.long)
    masks = torch.tensor(all_masks, dtype=torch.bool)
    returns = torch.tensor(all_returns, dtype=torch.float32)
    old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32)

    # Normalize returns (baseline subtraction)
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return features, actions, masks, returns, old_log_probs


def train(episodes, input_dim, output_dim, hidden_dim=128, epochs=50,
          lr=3e-4, gamma=0.99, l1_lambda=1e-5, entropy_coeff=0.01,
          dropout=0.0, resume_path=None, clip_eps=0.2):
    """REINFORCE training with entropy regularization and gradient clipping."""

    model = Policy(input_dim, hidden_dim, output_dim, dropout=dropout).to(DEVICE)

    if resume_path:
        data = json.load(open(resume_path))
        linear_modules = [(name, m) for name, m in model.net.named_modules() if isinstance(m, nn.Linear)]
        for i, (name, m) in enumerate(linear_modules):
            if i < len(data["layers"]):
                w = torch.tensor(data["layers"][i]["w"], dtype=torch.float32).T
                b = torch.tensor(data["layers"][i]["b"], dtype=torch.float32)
                m.weight.data = w.to(DEVICE)
                m.bias.data = b.to(DEVICE)
        print(f"Resumed from {resume_path}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    result = episodes_to_tensors(episodes, gamma)
    features, actions, masks, returns, old_log_probs = result
    if features is None:
        print("No steps in episodes!")
        return model

    features = features.to(DEVICE)
    actions = actions.to(DEVICE)
    masks = masks.to(DEVICE)
    returns = returns.to(DEVICE)

    n_samples = len(features)
    print(f"Training on {n_samples} steps from {len(episodes)} episodes")
    print(f"Model: {input_dim} -> {hidden_dim} -> {output_dim}  ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Device: {DEVICE}")

    wins_in_data = sum(1 for e in episodes if e["reward"] > 0)
    losses_in_data = sum(1 for e in episodes if e["reward"] < 0)
    print(f"Episodes: {len(episodes)} ({wins_in_data}W / {losses_in_data}L)")
    print(f"Win rate in data: {wins_in_data / max(len(episodes), 1) * 100:.1f}%")
    print(f"entropy_coeff={entropy_coeff}, l1={l1_lambda}")
    print()

    batch_size = min(4096, n_samples)
    best_entropy = 0.0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=DEVICE)

        total_loss = 0.0
        total_pg_loss = 0.0
        total_entropy = 0.0
        total_l1 = 0.0
        batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            feat_batch = features[idx]
            act_batch = actions[idx]
            mask_batch = masks[idx]
            ret_batch = returns[idx]

            logits = model(feat_batch)
            logits = logits.masked_fill(~mask_batch, float('-inf'))

            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            # REINFORCE policy gradient
            selected_log_probs = log_probs.gather(1, act_batch.unsqueeze(1)).squeeze(1)
            pg_loss = -(selected_log_probs * ret_batch).mean()

            # Entropy bonus
            safe_log_probs = log_probs.masked_fill(~mask_batch, 0.0)
            entropy = -(probs * safe_log_probs).sum(dim=-1).mean()

            # L1 regularization
            l1_loss = sum(p.abs().sum() for p in model.parameters()) * l1_lambda

            loss = pg_loss - entropy_coeff * entropy + l1_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_pg_loss += pg_loss.item()
            total_entropy += entropy.item()
            total_l1 += l1_loss.item()
            batches += 1

        avg_entropy = total_entropy / batches
        # Early stop if entropy collapses
        if epoch > 10 and avg_entropy < 0.1:
            print(f"  Early stop: entropy collapsed ({avg_entropy:.3f})")
            break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Compute action distribution
            model.eval()
            with torch.no_grad():
                all_logits = model(features)
                all_logits = all_logits.masked_fill(~masks, float('-inf'))
                all_probs = torch.softmax(all_logits, dim=-1)
                avg_probs = all_probs.mean(dim=0).cpu().numpy()

            action_names = [
                "AtkNear", "AtkWeak", "AtkFocus",
                "Abi0", "Abi1", "Abi2", "Abi3", "Abi4", "Abi5", "Abi6", "Abi7",
                "MoveTo", "MoveAwy", "Hold",
            ]
            dist_str = " ".join(f"{action_names[i]}={avg_probs[i]:.2f}" for i in range(len(action_names)) if avg_probs[i] > 0.01)

            print(
                f"Epoch {epoch+1:3d}/{epochs}  "
                f"loss={total_loss/batches:.4f}  "
                f"pg={total_pg_loss/batches:.4f}  "
                f"ent={total_entropy/batches:.3f}  "
                f"l1={total_l1/batches:.5f}"
            )
            print(f"  Actions: {dist_str}")

    # Sparsity report
    model.eval()
    first_layer = None
    for m in model.net:
        if isinstance(m, nn.Linear):
            first_layer = m
            break

    if first_layer is not None:
        w = first_layer.weight.data.abs().cpu()
        feature_importance = w.sum(dim=0).numpy()  # sum of absolute weights per input
        top_k = min(20, len(feature_importance))
        top_indices = np.argsort(feature_importance)[::-1][:top_k]
        print(f"\nTop {top_k} features by L1 weight magnitude:")
        for i, idx in enumerate(top_indices):
            print(f"  [{idx:3d}] importance={feature_importance[idx]:.4f}")

        # Count near-zero features
        threshold = feature_importance.max() * 0.01
        dead = (feature_importance < threshold).sum()
        print(f"\nDead features (<1% of max): {dead}/{len(feature_importance)}")

    return model


def _get_linear_key(model, idx):
    """Helper to get the key name of the idx-th linear layer."""
    count = 0
    for name, m in model.net.named_modules():
        if isinstance(m, nn.Linear):
            if count == idx:
                return name
            count += 1
    return "0"


def main():
    parser = argparse.ArgumentParser(description="REINFORCE training on self-play episodes")
    parser.add_argument("episodes", help="Path to episodes JSONL")
    parser.add_argument("--output", "-o", default="generated/selfplay_policy.json")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--l1", type=float, default=1e-5, help="L1 regularization")
    parser.add_argument("--entropy", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--resume", type=str, default=None, help="Resume from policy JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading episodes from {args.episodes}...")
    episodes = load_episodes(args.episodes)
    print(f"Loaded {len(episodes)} episodes")

    if not episodes or not episodes[0]["steps"]:
        print("No steps found!")
        return

    input_dim = len(episodes[0]["steps"][0]["features"])
    output_dim = len(episodes[0]["steps"][0]["mask"])
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    model = train(
        episodes, input_dim, output_dim,
        hidden_dim=args.hidden, epochs=args.epochs,
        lr=args.lr, gamma=args.gamma,
        l1_lambda=args.l1, entropy_coeff=args.entropy,
        dropout=args.dropout, resume_path=args.resume,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    weights = model.export_weights()

    # Save input_scale: per-feature max absolute value for normalization
    all_feats = []
    for ep in episodes:
        for s in ep["steps"]:
            all_feats.append(s["features"])
    feat_arr = np.array(all_feats)
    input_scale = np.abs(feat_arr).max(axis=0).clip(min=1e-6).tolist()
    weights["input_scale"] = input_scale

    with open(args.output, "w") as f:
        json.dump(weights, f)
    print(f"\nPolicy saved to {args.output}")


if __name__ == "__main__":
    main()
