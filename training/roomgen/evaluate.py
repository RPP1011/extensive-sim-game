#!/usr/bin/env python3
"""Evaluate ML-generated rooms against proc-gen baseline.

Computes:
  - Metric distributions (blocked%, cover density, chokepoints, etc.)
  - Connectivity pass rate
  - Dimension distribution analysis
  - Pairwise diversity metrics
  - Multi-budget quality curve (if model weights provided)

Usage:
    # Compare two JSONL files
    uv run --with torch python training/roomgen/evaluate.py \
        --procgen generated/rooms.jsonl \
        --ml generated/ml_rooms.jsonl

    # Evaluate single file
    uv run --with torch python training/roomgen/evaluate.py \
        --procgen generated/rooms.jsonl
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_records(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def check_connectivity(record: dict) -> bool:
    """BFS connectivity check on the obstacle_type grid."""
    obs = record["grid"]["obstacle_type"]
    w, d = record["width"], record["depth"]

    start = None
    goal = None
    for r in range(1, d - 1):
        if obs[r][w // 6] == 0:
            start = (r, w // 6)
            break
    for r in range(1, d - 1):
        if obs[r][w - w // 6] == 0:
            goal = (r, w - w // 6)
            break

    if start is None or goal is None:
        return False

    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        r, c = queue.pop(0)
        if (r, c) == goal:
            return True
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < d and 0 <= nc < w and (nr, nc) not in visited and obs[nr][nc] == 0:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def hamming_distance(a: dict, b: dict) -> float:
    """Grid-level Hamming distance normalized by area. Only comparable for same dimensions."""
    wa, da = a["width"], a["depth"]
    wb, db = b["width"], b["depth"]
    if wa != wb or da != db:
        return 1.0  # incomparable

    obs_a = a["grid"]["obstacle_type"]
    obs_b = b["grid"]["obstacle_type"]
    total = 0
    differ = 0
    for r in range(1, da - 1):
        for c in range(1, wa - 1):
            total += 1
            # Compare walkable vs not walkable
            if (obs_a[r][c] == 0) != (obs_b[r][c] == 0):
                differ += 1
    return differ / max(total, 1)


def evaluate_dataset(records: list[dict], label: str) -> dict:
    """Compute evaluation metrics for a dataset."""
    by_type = defaultdict(list)
    for r in records:
        by_type[r["room_type"]].append(r)

    results = {"label": label, "total_rooms": len(records), "by_type": {}}

    # Global metrics
    all_metrics = {
        "blocked_pct": [],
        "chokepoint_count": [],
        "cover_density": [],
        "elevation_zones": [],
        "flanking_routes": [],
        "spawn_quality_diff": [],
        "mean_wall_proximity": [],
        "aspect_ratio": [],
    }
    widths = []
    depths = []
    connectivity_pass = 0

    for rec in records:
        m = rec.get("metrics", {})
        for key in all_metrics:
            if key in m:
                all_metrics[key].append(m[key])
        widths.append(rec["width"])
        depths.append(rec["depth"])
        if check_connectivity(rec):
            connectivity_pass += 1

    results["connectivity_rate"] = connectivity_pass / max(len(records), 1)
    results["metrics"] = {k: compute_stats(v) for k, v in all_metrics.items()}
    results["dimensions"] = {
        "width": compute_stats([float(w) for w in widths]),
        "depth": compute_stats([float(d) for d in depths]),
    }

    # Blocked percentage in-range rate (2-35%)
    blocked_ok = sum(1 for b in all_metrics["blocked_pct"] if 0.02 <= b <= 0.35)
    results["blocked_in_range_rate"] = blocked_ok / max(len(all_metrics["blocked_pct"]), 1)

    # Per-type breakdown
    for rt, type_records in by_type.items():
        type_metrics = defaultdict(list)
        for rec in type_records:
            m = rec.get("metrics", {})
            for key in all_metrics:
                if key in m:
                    type_metrics[key].append(m[key])

        results["by_type"][rt] = {
            "count": len(type_records),
            "metrics": {k: compute_stats(v) for k, v in type_metrics.items()},
            "dim_width": compute_stats([float(r["width"]) for r in type_records]),
            "dim_depth": compute_stats([float(r["depth"]) for r in type_records]),
        }

    # Diversity: sample pairwise Hamming distances
    n_pairs = min(500, len(records) * (len(records) - 1) // 2)
    if len(records) > 1 and n_pairs > 0:
        rng = np.random.default_rng(42)
        distances = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(records), 2, replace=False)
            distances.append(hamming_distance(records[i], records[j]))
        results["diversity"] = compute_stats(distances)
    else:
        results["diversity"] = compute_stats([])

    return results


def print_comparison(procgen_results: dict, ml_results: dict = None):
    """Print formatted comparison table."""

    def fmt(stats: dict) -> str:
        return f"{stats['mean']:.3f} ± {stats['std']:.3f} [{stats['min']:.3f}, {stats['max']:.3f}]"

    print("=" * 80)
    print("ROOM GENERATION EVALUATION")
    print("=" * 80)

    datasets = [("Proc-Gen", procgen_results)]
    if ml_results:
        datasets.append(("ML-Gen", ml_results))

    for label, results in datasets:
        print(f"\n--- {label} ({results['total_rooms']} rooms) ---")
        print(f"  Connectivity rate:    {results['connectivity_rate']:.1%}")
        print(f"  Blocked in-range:     {results['blocked_in_range_rate']:.1%}")

        print(f"\n  Metric distributions:")
        for metric, stats in results["metrics"].items():
            print(f"    {metric:25s} {fmt(stats)}")

        print(f"\n  Dimensions:")
        print(f"    Width:  {fmt(results['dimensions']['width'])}")
        print(f"    Depth:  {fmt(results['dimensions']['depth'])}")

        print(f"\n  Diversity (Hamming): {fmt(results['diversity'])}")

        print(f"\n  Per-type breakdown:")
        for rt in sorted(results["by_type"].keys()):
            info = results["by_type"][rt]
            w_stats = info["dim_width"]
            d_stats = info["dim_depth"]
            blocked = info["metrics"].get("blocked_pct", {})
            print(
                f"    {rt:12s} n={info['count']:5d}  "
                f"dims={w_stats['mean']:.0f}×{d_stats['mean']:.0f}  "
                f"blocked={blocked.get('mean', 0):.2f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate room generation quality")
    parser.add_argument("--procgen", required=True, help="Proc-gen JSONL")
    parser.add_argument("--ml", default=None, help="ML-generated JSONL")
    parser.add_argument("--output", default=None, help="Save results JSON")
    args = parser.parse_args()

    print(f"Loading proc-gen data: {args.procgen}")
    procgen_records = load_records(args.procgen)
    procgen_results = evaluate_dataset(procgen_records, "procgen")

    ml_results = None
    if args.ml:
        print(f"Loading ML data: {args.ml}")
        ml_records = load_records(args.ml)
        ml_results = evaluate_dataset(ml_records, "ml")

    print_comparison(procgen_results, ml_results)

    if args.output:
        output = {"procgen": procgen_results}
        if ml_results:
            output["ml"] = ml_results
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
