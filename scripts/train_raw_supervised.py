#!/usr/bin/env python3
"""
Supervised training on oracle-labeled raw dataset (311 features, 14 actions).
Cross-entropy loss + L1 regularization for automatic feature selection.

Usage:
  python scripts/train_raw_supervised.py generated/raw_dataset_combined.jsonl \
    --epochs 200 --l1 0.001 --lr 0.001 --output generated/raw_policy.json
"""

import argparse
import json
import sys
import random
import numpy as np

def load_dataset(path):
    features, labels, masks = [], [], []
    with open(path) as f:
        for line in f:
            sample = json.loads(line)
            features.append(sample["features"])
            labels.append(sample["label"])
            masks.append(sample["mask"])
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(masks, dtype=np.float32)

def softmax(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def forward(x, layers):
    """Forward pass through MLP. Returns (logits, activations_list)."""
    acts = [x]
    h = x
    for i, (w, b) in enumerate(layers):
        h = h @ w + b
        if i < len(layers) - 1:
            h = relu(h)
        acts.append(h)
    return h, acts

def cross_entropy_loss(logits, labels, masks):
    """Masked cross-entropy loss."""
    # Apply mask: set invalid actions to -1e9
    masked_logits = np.where(masks > 0.5, logits, -1e9)
    # Stable softmax
    probs = softmax(masked_logits)
    n = len(labels)
    log_probs = np.log(probs[np.arange(n), labels] + 1e-10)
    return -log_probs.mean()

def l1_penalty(layers):
    """Sum of absolute values of all weights (not biases)."""
    total = 0.0
    for w, b in layers:
        total += np.abs(w).sum()
    return total

def count_dead_features(layers):
    """Count input features where all outgoing weights are near zero."""
    w0 = layers[0][0]  # (input_dim, hidden_dim)
    max_abs = np.abs(w0).max(axis=1)
    return int((max_abs < 1e-4).sum())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to JSONL dataset")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l1", type=float, default=0.001, help="L1 regularization strength")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping")
    parser.add_argument("--output", default="generated/raw_policy.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.dataset}...")
    features, labels, masks = load_dataset(args.dataset)
    n_samples, n_features = features.shape
    n_actions = masks.shape[1]
    print(f"Samples: {n_samples}, Features: {n_features}, Actions: {n_actions}")

    # Input scaling (max-abs per feature)
    feat_scale = np.abs(features).max(axis=0)
    feat_scale = np.maximum(feat_scale, 1e-6)
    features = features / feat_scale

    # Train/val split
    indices = list(range(n_samples))
    random.shuffle(indices)
    val_n = int(n_samples * args.val_split)
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    X_train, y_train, m_train = features[train_idx], labels[train_idx], masks[train_idx]
    X_val, y_val, m_val = features[val_idx], labels[val_idx], masks[val_idx]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    # Label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    label_names = [
        "AttackNearest", "AttackWeakest", "AttackFocus",
        "Ability0", "Ability1", "Ability2", "Ability3",
        "Ability4", "Ability5", "Ability6", "Ability7",
        "MoveToward", "MoveAway", "Hold",
    ]
    for u, c in zip(unique, counts):
        pct = c / n_samples * 100
        name = label_names[u] if u < len(label_names) else f"Action{u}"
        print(f"  {name:<18} {c:>6} ({pct:.1f}%)")

    # Class weights (inverse frequency, capped)
    class_weights = np.ones(n_actions, dtype=np.float32)
    for u, c in zip(unique, counts):
        class_weights[u] = n_samples / (n_actions * c)
    class_weights = np.minimum(class_weights, 10.0)  # cap to avoid extreme weights
    print(f"Class weights: {class_weights.round(2).tolist()}")

    # Initialize weights (He init)
    hidden = args.hidden
    scale1 = np.sqrt(2.0 / n_features)
    scale2 = np.sqrt(2.0 / hidden)

    w1 = np.random.randn(n_features, hidden).astype(np.float32) * scale1
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = np.random.randn(hidden, n_actions).astype(np.float32) * scale2
    b2 = np.zeros(n_actions, dtype=np.float32)

    layers = [(w1, b1), (w2, b2)]

    # Adam optimizer state
    adam_m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in layers]
    adam_v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in layers]
    adam_t = 0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    best_val_loss = float("inf")
    best_layers = None
    patience = 20
    patience_counter = 0

    for epoch in range(args.epochs):
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        X_shuf, y_shuf, m_shuf = X_train[perm], y_train[perm], m_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_shuf), args.batch_size):
            xb = X_shuf[i:i+args.batch_size]
            yb = y_shuf[i:i+args.batch_size]
            mb = m_shuf[i:i+args.batch_size]
            bs = len(xb)

            # Forward
            h1 = xb @ layers[0][0] + layers[0][1]  # (bs, hidden)
            h1_relu = relu(h1)
            logits = h1_relu @ layers[1][0] + layers[1][1]  # (bs, n_actions)

            # Masked softmax
            masked_logits = np.where(mb > 0.5, logits, -1e9)
            probs = softmax(masked_logits)

            # Weighted cross-entropy gradient
            sample_weights = class_weights[yb]
            grad_logits = probs.copy()
            grad_logits[np.arange(bs), yb] -= 1.0
            grad_logits *= (sample_weights / bs)[:, None]
            # Zero out invalid action gradients
            grad_logits = np.where(mb > 0.5, grad_logits, 0.0)

            # Backprop through layer 2
            dw2 = h1_relu.T @ grad_logits
            db2 = grad_logits.sum(axis=0)

            # Backprop through ReLU + layer 1
            dh1 = grad_logits @ layers[1][0].T
            dh1 = dh1 * (h1 > 0).astype(np.float32)
            dw1 = xb.T @ dh1
            db1 = dh1.sum(axis=0)

            # L1 gradient (subgradient on weights only)
            l1_dw1 = args.l1 * np.sign(layers[0][0])
            l1_dw2 = args.l1 * np.sign(layers[1][0])

            grads = [(dw1, db1), (dw2, db2)]

            # Adam update (CE gradient only, L1 applied as proximal step)
            adam_t += 1
            for li in range(len(layers)):
                for pi in range(2):  # w, b
                    g = grads[li][pi]
                    adam_m[li] = list(adam_m[li])
                    adam_v[li] = list(adam_v[li])
                    adam_m[li][pi] = beta1 * adam_m[li][pi] + (1 - beta1) * g
                    adam_v[li][pi] = beta2 * adam_v[li][pi] + (1 - beta2) * g**2
                    m_hat = adam_m[li][pi] / (1 - beta1**adam_t)
                    v_hat = adam_v[li][pi] / (1 - beta2**adam_t)

                    layers[li] = list(layers[li])
                    layers[li][pi] = layers[li][pi] - args.lr * m_hat / (np.sqrt(v_hat) + eps)
                    layers[li] = tuple(layers[li])
                    adam_m[li] = tuple(adam_m[li])
                    adam_v[li] = tuple(adam_v[li])

            # Proximal L1: soft-threshold weights (not biases)
            if args.l1 > 0:
                lam = args.l1 * args.lr
                for li in range(len(layers)):
                    w, b = layers[li]
                    w = np.sign(w) * np.maximum(np.abs(w) - lam, 0)
                    layers[li] = (w, b)

            # Track loss
            loss = -np.log(probs[np.arange(bs), yb] + 1e-10)
            epoch_loss += (loss * sample_weights).sum()
            n_batches += bs

        train_loss = epoch_loss / max(n_batches, 1)

        # Validation loss + accuracy
        val_logits, _ = forward(X_val, layers)
        val_masked = np.where(m_val > 0.5, val_logits, -1e9)
        val_probs = softmax(val_masked)
        val_ce = -np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-10).mean()
        val_l1 = args.l1 * l1_penalty(layers)
        val_loss = val_ce  # early stop on CE only, not L1
        val_preds = val_masked.argmax(axis=1)
        val_acc = (val_preds == y_val).mean() * 100

        dead = count_dead_features(layers)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>3}/{args.epochs}  train_loss={train_loss:.4f}  "
                  f"val_ce={val_ce:.4f}  val_l1={val_l1:.4f}  val_acc={val_acc:.1f}%  "
                  f"dead_features={dead}/{n_features}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_layers = [(w.copy(), b.copy()) for w, b in layers]
            patience_counter = 0
        else:
            patience_counter += 1
            if not args.no_early_stop and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if args.no_early_stop:
        # Use final weights when early stopping is disabled
        pass
    elif best_layers:
        layers = best_layers

    # Final stats
    dead = count_dead_features(layers)
    print(f"\nFinal dead features: {dead}/{n_features}")

    # Show which features survived
    w0 = layers[0][0]
    max_abs = np.abs(w0).max(axis=1)
    alive_idx = np.where(max_abs >= 1e-4)[0]
    print(f"Alive features ({len(alive_idx)}): {alive_idx.tolist()}")

    # Final validation accuracy
    val_logits, _ = forward(X_val, layers)
    val_masked = np.where(m_val > 0.5, val_logits, -1e9)
    val_preds = val_masked.argmax(axis=1)
    val_acc = (val_preds == y_val).mean() * 100
    print(f"Final val accuracy: {val_acc:.1f}%")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for c in range(n_actions):
        idx = y_val == c
        if idx.sum() == 0:
            continue
        acc = (val_preds[idx] == c).mean() * 100
        name = label_names[c] if c < len(label_names) else f"Action{c}"
        print(f"  {name:<18} {acc:5.1f}%  (n={idx.sum()})")

    # Export policy JSON (same format as PolicyWeights in Rust)
    policy = {
        "layers": [],
        "input_scale": feat_scale.tolist(),
    }
    for w, b in layers:
        policy["layers"].append({
            "w": w.tolist(),
            "b": b.tolist(),
        })

    with open(args.output, "w") as f:
        json.dump(policy, f)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
