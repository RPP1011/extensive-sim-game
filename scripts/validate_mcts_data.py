#!/usr/bin/env python3
"""Validate MCTS bootstrap data: strategy diversity, balance, and token statistics.

Usage:
    uv run --with numpy scripts/validate_mcts_data.py <path_to_mcts_export.jsonl>
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_samples(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        print("Run mcts-bootstrap first to generate export data.")
        sys.exit(1)
    samples = []
    with open(p) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {i + 1}: {e}")
    if not samples:
        print(f"No valid samples found in {path}")
        sys.exit(1)
    return samples


def entropy(counts: dict) -> float:
    """Shannon entropy in bits from a dict of counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def kl_divergence(p_counts: dict, q_counts: dict) -> float:
    """KL(P || Q) with Laplace smoothing."""
    all_keys = set(p_counts) | set(q_counts)
    if not all_keys:
        return 0.0
    # Laplace smoothing
    alpha = 1.0
    p_total = sum(p_counts.values()) + alpha * len(all_keys)
    q_total = sum(q_counts.values()) + alpha * len(all_keys)
    kl = 0.0
    for k in all_keys:
        p_prob = (p_counts.get(k, 0) + alpha) / p_total
        q_prob = (q_counts.get(k, 0) + alpha) / q_total
        kl += p_prob * math.log2(p_prob / q_prob)
    return kl


def report_per_starting_choice(samples: list[dict]):
    print("=" * 72)
    print("PER STARTING CHOICE")
    print("=" * 72)

    by_choice = defaultdict(list)
    for s in samples:
        by_choice[s.get("starting_choice_name", "<unknown>")].append(s)

    tick_buckets = [0, 500, 1000, 2000]

    for choice in sorted(by_choice):
        group = by_choice[choice]
        print(f"\n--- {choice} ---")
        print(f"  Samples: {len(group)}")

        # Win rate
        outcomes = [s.get("campaign_outcome") for s in group]
        resolved = [o for o in outcomes if o is not None]
        victories = sum(1 for o in resolved if o == "Victory")
        if resolved:
            print(f"  Win rate: {victories}/{len(resolved)} ({100 * victories / len(resolved):.1f}%)")
        else:
            print("  Win rate: no resolved outcomes")

        # Mean value estimate
        values = [s["value_estimate"] for s in group if "value_estimate" in s]
        if values:
            print(f"  Mean value estimate: {np.mean(values):.4f} (std {np.std(values):.4f})")

        # Best action at tick 0
        tick0 = [s for s in group if s.get("tick", -1) == 0]
        if tick0:
            ba0 = Counter(s.get("best_action", "?") for s in tick0)
            most_common = ba0.most_common(1)[0]
            print(f"  Best action at tick 0: {most_common[0]} ({most_common[1]}/{len(tick0)})")

        # Best actions at later ticks
        for t in tick_buckets[1:]:
            at_tick = [s for s in group if s.get("tick", -1) == t]
            if at_tick:
                ba = Counter(s.get("best_action", "?") for s in at_tick)
                top3 = ba.most_common(3)
                top_str = ", ".join(f"{a} ({c})" for a, c in top3)
                print(f"  Best actions at tick {t}: {top_str}")
            else:
                print(f"  Best actions at tick {t}: (no samples)")

        # Campaign duration (max tick per seed)
        seeds = defaultdict(int)
        for s in group:
            seed = s.get("seed")
            if seed is not None:
                seeds[seed] = max(seeds[seed], s.get("tick", 0))
        if seeds:
            durations = list(seeds.values())
            print(f"  Mean campaign duration: {np.mean(durations):.0f} ticks "
                  f"(min {min(durations)}, max {max(durations)})")


def report_per_world_template(samples: list[dict]):
    print("\n" + "=" * 72)
    print("PER WORLD TEMPLATE")
    print("=" * 72)

    by_template = defaultdict(list)
    for s in samples:
        by_template[s.get("world_template_name", "<unknown>")].append(s)

    for template in sorted(by_template):
        group = by_template[template]
        print(f"\n--- {template} ---")
        print(f"  Samples: {len(group)}")

        outcomes = [s.get("campaign_outcome") for s in group]
        resolved = [o for o in outcomes if o is not None]
        victories = sum(1 for o in resolved if o == "Victory")
        if resolved:
            print(f"  Win rate: {victories}/{len(resolved)} ({100 * victories / len(resolved):.1f}%)")
        else:
            print("  Win rate: no resolved outcomes")

        values = [s["value_estimate"] for s in group if "value_estimate" in s]
        if values:
            print(f"  Mean value estimate: {np.mean(values):.4f} (std {np.std(values):.4f})")


def report_strategy_diversity(samples: list[dict]):
    print("\n" + "=" * 72)
    print("STRATEGY DIVERSITY")
    print("=" * 72)

    # Group by tick
    by_tick = defaultdict(list)
    for s in samples:
        by_tick[s.get("tick", -1)].append(s)

    # Action entropy at each tick
    print("\nAction entropy by tick (higher = more diverse):")
    dominant_warnings = []
    tick_action_counts = {}

    for tick in sorted(by_tick):
        action_counts = Counter()
        for s in by_tick[tick]:
            dist = s.get("action_distribution", [])
            for action_name, visit_count in dist:
                action_counts[action_name] += visit_count
        tick_action_counts[tick] = action_counts
        h = entropy(action_counts)
        total = sum(action_counts.values())
        print(f"  Tick {tick:>5d}: H = {h:.3f} bits ({len(action_counts)} actions, {len(by_tick[tick])} samples)")

        # Check for dominant actions
        if total > 0:
            for action, count in action_counts.most_common():
                frac = count / total
                if frac > 0.80:
                    dominant_warnings.append((tick, action, frac))

    if dominant_warnings:
        print("\n  *** BALANCE WARNINGS: dominant actions (>80% share) ***")
        for tick, action, frac in dominant_warnings:
            print(f"    Tick {tick}: {action} = {100 * frac:.1f}%")
    else:
        print("\n  No dominant actions detected (all <80% share).")

    # Pairwise KL divergence between starting choices
    print("\nPairwise KL divergence between starting choices:")
    by_choice = defaultdict(lambda: Counter())
    for s in samples:
        choice = s.get("starting_choice_name", "<unknown>")
        dist = s.get("action_distribution", [])
        for action_name, visit_count in dist:
            by_choice[choice][action_name] += visit_count

    choices = sorted(by_choice)
    if len(choices) < 2:
        print("  (Need at least 2 starting choices for pairwise comparison)")
    else:
        # Print header
        max_name = max(len(c) for c in choices)
        header = " " * (max_name + 2) + "  ".join(f"{c[:8]:>8s}" for c in choices)
        print(f"  {header}")
        for i, ci in enumerate(choices):
            row = f"  {ci:<{max_name}s}"
            for j, cj in enumerate(choices):
                if i == j:
                    row += f"{'---':>10s}"
                else:
                    kl = kl_divergence(by_choice[ci], by_choice[cj])
                    row += f"{kl:>10.4f}"
            print(row)

        # Summary
        kl_values = []
        for i, ci in enumerate(choices):
            for j, cj in enumerate(choices):
                if i < j:
                    kl_fwd = kl_divergence(by_choice[ci], by_choice[cj])
                    kl_rev = kl_divergence(by_choice[cj], by_choice[ci])
                    kl_values.append((ci, cj, (kl_fwd + kl_rev) / 2))
        kl_values.sort(key=lambda x: -x[2])
        print("\n  Most divergent pairs (symmetric KL):")
        for ci, cj, kl_sym in kl_values[:5]:
            print(f"    {ci} vs {cj}: {kl_sym:.4f} bits")
        if kl_values:
            mean_kl = np.mean([x[2] for x in kl_values])
            print(f"  Mean pairwise symmetric KL: {mean_kl:.4f} bits")


def report_token_statistics(samples: list[dict]):
    print("\n" + "=" * 72)
    print("TOKEN STATISTICS")
    print("=" * 72)

    # Gather token counts and feature vectors
    token_counts = []
    all_features = []
    type_counter = Counter()

    for s in samples:
        tokens = s.get("tokens", [])
        token_counts.append(len(tokens))
        for tok in tokens:
            type_counter[tok.get("type_id", -1)] += 1
            feats = tok.get("features")
            if feats is not None:
                all_features.append(feats)

    token_counts = np.array(token_counts)
    print(f"\nToken count per state:")
    print(f"  Mean: {np.mean(token_counts):.1f}")
    print(f"  Std:  {np.std(token_counts):.1f}")
    print(f"  Min:  {np.min(token_counts)}")
    print(f"  Max:  {np.max(token_counts)}")
    pcts = np.percentile(token_counts, [25, 50, 75])
    print(f"  Quartiles: {pcts[0]:.0f} / {pcts[1]:.0f} / {pcts[2]:.0f}")

    print(f"\nToken type distribution:")
    for tid, cnt in type_counter.most_common():
        print(f"  type_id {tid}: {cnt} tokens")

    if not all_features:
        print("\n  No feature vectors found.")
        return

    # Pad to uniform length if needed
    max_dim = max(len(f) for f in all_features)
    min_dim = min(len(f) for f in all_features)
    if max_dim != min_dim:
        print(f"\n  Warning: feature dimensions vary ({min_dim} to {max_dim}), padding to {max_dim}")
        padded = []
        for f in all_features:
            if len(f) < max_dim:
                f = f + [0.0] * (max_dim - len(f))
            padded.append(f)
        features = np.array(padded, dtype=np.float32)
    else:
        features = np.array(all_features, dtype=np.float32)

    print(f"\nFeature dimensions: {features.shape[1]}")
    print(f"Total tokens analyzed: {features.shape[0]}")

    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)

    print(f"\nPer-dimension mean: min={np.min(means):.4f}, max={np.max(means):.4f}")
    print(f"Per-dimension std:  min={np.min(stds):.6f}, max={np.max(stds):.4f}")

    # Dead features (zero variance)
    dead = np.where(stds < 1e-8)[0]
    if len(dead) > 0:
        print(f"\n  *** DEAD FEATURES (zero variance): {len(dead)} dimensions ***")
        if len(dead) <= 20:
            print(f"    Indices: {dead.tolist()}")
            print(f"    Constant values: {means[dead].tolist()}")
        else:
            print(f"    First 20 indices: {dead[:20].tolist()}")
            print(f"    (showing {len(dead)} total)")
    else:
        print("\n  No dead features detected.")

    # Top variance dimensions
    top_var = np.argsort(-stds)[:10]
    print(f"\n  Top 10 highest-variance dimensions:")
    for idx in top_var:
        print(f"    dim {idx:>3d}: mean={means[idx]:>8.4f}  std={stds[idx]:>8.4f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_mcts_data.py <path_to_mcts_export.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    samples = load_samples(path)
    print(f"Loaded {len(samples)} samples from {path}")

    # Quick summary
    seeds = set(s.get("seed") for s in samples)
    choices = set(s.get("starting_choice_name") for s in samples)
    templates = set(s.get("world_template_name") for s in samples)
    print(f"Unique seeds: {len(seeds)}")
    print(f"Starting choices: {sorted(choices)}")
    print(f"World templates: {sorted(templates)}")

    report_per_starting_choice(samples)
    report_per_world_template(samples)
    report_strategy_diversity(samples)
    report_token_statistics(samples)

    print("\n" + "=" * 72)
    print("VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
