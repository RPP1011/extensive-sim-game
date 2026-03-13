#!/usr/bin/env python3
"""Probe frozen CLS embeddings for targeting semantics.

Usage:
    uv run --with numpy --with torch --with scikit-learn training/probe_cls.py \
        --data generated/operator_dataset_curriculum.npz
"""

import argparse
import numpy as np
from pathlib import Path


def encode_with_autoencoder(props: np.ndarray, weights_path: str) -> np.ndarray:
    """Run the 80→64→32 ability autoencoder on raw property vectors."""
    import json

    w = json.load(open(weights_path))
    enc = w["encoder"]
    w1 = np.array(enc["w1"])  # (80, 64)
    b1 = np.array(enc["b1"])  # (64,)
    w2 = np.array(enc["w2"])  # (64, 32)
    b2 = np.array(enc["b2"])  # (32,)

    h = np.maximum(0, props @ w1 + b1)  # ReLU
    z = h @ w2 + b2  # (N, 32)
    # L2 normalize
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / (norms + 1e-8)
    return z


def extract_labels(props: np.ndarray) -> dict[str, np.ndarray]:
    """Extract classification labels from 80-dim ability property vectors."""
    labels = {}
    # Targeting type: argmax of props[0:8]
    labels["targeting_type"] = props[:, 0:8].argmax(axis=1)
    # AI hint: argmax of props[32:38]
    labels["ai_hint"] = props[:, 32:38].argmax(axis=1)
    # Delivery method: argmax of props[14:21]
    labels["delivery"] = props[:, 14:21].argmax(axis=1)
    # Range bucket: quantize props[8] (range/10)
    r = props[:, 8]
    labels["range_bucket"] = np.digitize(r, bins=[0.2, 0.4, 0.6])  # 0,1,2,3
    # Is AoE: props[74] > 0
    labels["is_aoe"] = (props[:, 74] > 0).astype(int)
    # Has damage: props[41] > 0
    labels["has_damage"] = (props[:, 41] > 0).astype(int)
    # Has heal: props[45] > 0
    labels["has_heal"] = (props[:, 45] > 0).astype(int)
    # Has hard CC: any of props[48:63] > 0
    labels["has_hard_cc"] = (props[:, 48:63].max(axis=1) > 0).astype(int)
    return labels


def scenario_split(scenario_ids: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    """Split by scenario_id to avoid data leakage."""
    rng = np.random.RandomState(seed)
    unique = np.unique(scenario_ids)
    rng.shuffle(unique)
    n_test = max(1, int(len(unique) * test_frac))
    test_ids = set(unique[:n_test].tolist())
    train_mask = np.array([s not in test_ids for s in scenario_ids])
    val_mask = ~train_mask
    return train_mask, val_mask


def run_linear_probes(cls_emb, labels, train_mask, val_mask):
    """Test 1: Linear probing on frozen CLS embeddings."""
    from sklearn.linear_model import LogisticRegression

    thresholds = {
        "targeting_type": 0.90,
        "ai_hint": 0.85,
        "delivery": 0.80,
        "range_bucket": 0.70,
        "is_aoe": 0.80,
        "has_damage": 0.90,
        "has_heal": 0.90,
        "has_hard_cc": 0.85,
    }

    print("\n=== Test 1: Linear Probes on Frozen CLS ===")
    print(f"{'Probe':<18} {'Train':>6} {'Val':>6} {'Thresh':>6} {'Pass':>5} {'Classes':>8}")
    print("-" * 58)

    results = {}
    for name, y in labels.items():
        n_classes = len(np.unique(y))
        if n_classes < 2:
            print(f"{name:<18} {'SKIP':>6} — only 1 class")
            continue
        X_tr, y_tr = cls_emb[train_mask], y[train_mask]
        X_va, y_va = cls_emb[val_mask], y[val_mask]
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr, y_tr)
        train_acc = clf.score(X_tr, y_tr)
        val_acc = clf.score(X_va, y_va)
        thresh = thresholds[name]
        passed = val_acc >= thresh
        results[name] = {"train": train_acc, "val": val_acc, "pass": passed}
        mark = "YES" if passed else "NO"
        print(f"{name:<18} {train_acc:>5.1%} {val_acc:>5.1%} {thresh:>5.0%} {mark:>5} {n_classes:>8}")

    n_pass = sum(1 for r in results.values() if r["pass"])
    print(f"\nPassed: {n_pass}/{len(results)}")
    return results


def run_knn_consistency(cls_emb, labels):
    """Test 2: Nearest-neighbor consistency."""
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n=== Test 2: kNN Consistency (k=5, cosine) ===")

    # Compute cosine similarity matrix (can be large — subsample if needed)
    n = len(cls_emb)
    if n > 10000:
        idx = np.random.RandomState(42).choice(n, 10000, replace=False)
        cls_sub = cls_emb[idx]
        labels_sub = {k: v[idx] for k, v in labels.items()}
    else:
        cls_sub = cls_emb
        labels_sub = labels

    sim = cosine_similarity(cls_sub)
    np.fill_diagonal(sim, -1)  # exclude self
    top5 = np.argsort(sim, axis=1)[:, -5:]  # top 5 most similar

    print(f"{'Property':<18} {'kNN%':>6} {'Random%':>8}")
    print("-" * 35)
    for name in ["targeting_type", "ai_hint"]:
        y = labels_sub[name]
        # kNN consistency: fraction of neighbors with same label
        neighbor_labels = y[top5]  # (n, 5)
        matches = (neighbor_labels == y[:, None]).mean()
        # Random baseline: most-frequent class probability
        _, counts = np.unique(y, return_counts=True)
        random_base = (counts / counts.sum()).max()
        print(f"{name:<18} {matches:>5.1%} {random_base:>7.1%}")


def run_reconstruction_probe(cls_emb, props, train_mask, val_mask):
    """Test 3: Linear reconstruction CLS → Props."""
    from sklearn.linear_model import Ridge

    print("\n=== Test 3: Reconstruction Probe (CLS→Props, linear R²) ===")

    reg = Ridge(alpha=1.0)
    reg.fit(cls_emb[train_mask], props[train_mask])
    pred = reg.predict(cls_emb[val_mask])
    target = props[val_mask]

    # Per-dimension R²
    ss_res = ((target - pred) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)

    # Report by group
    groups = {
        "targeting [0:8]": slice(0, 8),
        "range [8]": slice(8, 9),
        "delivery [14:21]": slice(14, 21),
        "ai_hint [32:38]": slice(32, 38),
        "AoE [70:75]": slice(70, 75),
        "damage [41]": slice(41, 42),
        "heal [45]": slice(45, 46),
        "overall [0:80]": slice(0, 80),
    }
    print(f"{'Group':<22} {'Mean R²':>8} {'Min R²':>8}")
    print("-" * 40)
    for gname, sl in groups.items():
        r2_group = r2[sl]
        print(f"{gname:<22} {r2_group.mean():>7.3f} {r2_group.min():>7.3f}")


def run_tsne(cls_emb, labels, out_path: Path):
    """Test 4: t-SNE visualization."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n=== Test 4: t-SNE — skipped (matplotlib not installed) ===")
        return

    print("\n=== Test 4: t-SNE Visualization ===")

    # Subsample for speed
    n = len(cls_emb)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        cls_sub = cls_emb[idx]
        labels_sub = {k: v[idx] for k, v in labels.items()}
    else:
        cls_sub = cls_emb
        labels_sub = labels

    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(cls_sub)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    color_maps = {
        "targeting_type": {
            "names": ["enemy", "ally", "self", "aoe", "ground", "dir", "vec", "global"],
        },
        "ai_hint": {
            "names": ["damage", "heal", "cc", "defense", "utility", "other"],
        },
        "range_bucket": {
            "names": ["melee", "short", "mid", "long"],
        },
    }

    for ax, (name, info) in zip(axes, color_maps.items()):
        y = labels_sub[name]
        for c in np.unique(y):
            mask = y == c
            label = info["names"][c] if c < len(info["names"]) else f"cls_{c}"
            ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.5, label=label)
        ax.set_title(name)
        ax.legend(markerscale=3, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved t-SNE plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Probe CLS embeddings for targeting info")
    parser.add_argument("--data", required=True, help="Path to operator dataset npz")
    parser.add_argument("--tsne-out", default="generated/cls_tsne.png", help="t-SNE output path")
    parser.add_argument("--no-tsne", action="store_true", help="Skip t-SNE")
    parser.add_argument(
        "--encoder", default="generated/ability_encoder.json",
        help="Path to ability autoencoder weights for comparison",
    )
    args = parser.parse_args()

    data = np.load(args.data)
    cls_emb = data["ability_cls"]  # (N, 32)
    props = data["ability_props"]  # (N, 80)
    scenario_ids = data["scenario_ids"].ravel()  # (N,)

    print(f"Loaded {len(cls_emb)} samples, {len(np.unique(scenario_ids))} scenarios")

    labels = extract_labels(props)
    train_mask, val_mask = scenario_split(scenario_ids)
    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}")

    # --- Transformer CLS ---
    print("\n" + "=" * 60)
    print("  TRANSFORMER CLS (Phase 2 v2, d=32)")
    print("=" * 60)

    probe_results_cls = run_linear_probes(cls_emb, labels, train_mask, val_mask)
    run_knn_consistency(cls_emb, labels)
    run_reconstruction_probe(cls_emb, props, train_mask, val_mask)

    if not args.no_tsne:
        run_tsne(cls_emb, labels, Path(args.tsne_out))

    # --- Autoencoder ---
    encoder_path = Path(args.encoder)
    if encoder_path.exists():
        ae_emb = encode_with_autoencoder(props, str(encoder_path))
        print("\n" + "=" * 60)
        print("  AUTOENCODER (contrastive + recon, 80→32)")
        print("=" * 60)

        probe_results_ae = run_linear_probes(ae_emb, labels, train_mask, val_mask)
        run_knn_consistency(ae_emb, labels)
        run_reconstruction_probe(ae_emb, props, train_mask, val_mask)

        if not args.no_tsne:
            run_tsne(ae_emb, labels, Path(str(args.tsne_out).replace(".png", "_ae.png")))
    else:
        print(f"\nAutoencoder weights not found at {encoder_path}, skipping comparison.")
        probe_results_ae = None

    # --- Comparison summary ---
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Probe':<18} {'CLS Val':>8} {'AE Val':>8} {'Thresh':>8}")
    print("-" * 44)
    for name in probe_results_cls:
        cls_val = probe_results_cls[name]["val"]
        ae_val = probe_results_ae[name]["val"] if probe_results_ae and name in probe_results_ae else float("nan")
        thresholds = {"targeting_type": 0.90, "ai_hint": 0.85, "delivery": 0.80,
                      "range_bucket": 0.70, "is_aoe": 0.80, "has_damage": 0.90,
                      "has_heal": 0.90, "has_hard_cc": 0.85}
        thresh = thresholds[name]
        print(f"{name:<18} {cls_val:>7.1%} {ae_val:>7.1%} {thresh:>7.0%}")

    # Decision
    print("\n--- Decision ---")
    n_pass_cls = sum(1 for r in probe_results_cls.values() if r["pass"])
    n_pass_ae = sum(1 for r in (probe_results_ae or {}).values() if r["pass"])
    n_total = len(probe_results_cls)
    print(f"Transformer CLS: {n_pass_cls}/{n_total} pass")
    if probe_results_ae is not None:
        print(f"Autoencoder:     {n_pass_ae}/{n_total} pass")
        if n_pass_ae > n_pass_cls:
            print("→ Autoencoder preserves more targeting info — consider using AE embeddings for operator")
        elif n_pass_ae == n_pass_cls:
            print("→ Both similar — targeting info bottleneck may be in d=32 dimensionality")
        else:
            print("→ Transformer CLS is better despite lower probe scores — unexpected")


if __name__ == "__main__":
    main()
