#!/usr/bin/env python3
"""Evaluate entity encoder predictions against actual fight outcomes.

Loads an outcome dataset (JSONL with game_state + hero_wins + hero_hp_remaining
+ fight_progress) and runs the pretrained entity encoder to compare predicted
vs actual outcomes.

Reports:
  - Overall accuracy and calibration
  - Accuracy bucketed by fight progress (early/mid/late)
  - Prediction confidence distribution
  - Per-scenario breakdown

Usage:
    uv run --with numpy --with torch training/eval/entity_encoder_eval.py \
        generated/entity_encoder_pretrained_v3.pt \
        --dataset generated/outcome_dataset_v3_combined.jsonl \
        --held-out scenarios/combos_26c4  # generate fresh held-out data
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import numpy as np

# Import the pretraining model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pretrain_entity import EntityEncoderPretraining

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dataset(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def generate_held_out(scenario_path: str, sample_interval: int = 10) -> list[dict]:
    """Generate held-out outcome data by running xtask."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        out_path = f.name

    cmd = [
        "cargo", "run", "--release", "--bin", "xtask", "--",
        "scenario", "oracle", "outcome-dataset",
        scenario_path,
        "--output", out_path,
        "--sample-interval", str(sample_interval),
    ]
    print(f"Generating held-out data from {scenario_path}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating data: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    # Print the generation summary (last few lines of stderr)
    for line in result.stderr.strip().split("\n")[-3:]:
        print(f"  {line}")

    samples = load_dataset(Path(out_path))
    Path(out_path).unlink(missing_ok=True)
    return samples


def evaluate(
    model: EntityEncoderPretraining,
    samples: list[dict],
    progress_buckets: int = 5,
):
    """Run evaluation and print results."""
    game_states = torch.tensor(
        [s["game_state"] for s in samples], dtype=torch.float32, device=DEVICE
    )
    true_wins = np.array([s["hero_wins"] for s in samples])
    true_hp = np.array([s["hero_hp_remaining"] for s in samples])
    progress = np.array([s["fight_progress"] for s in samples])
    scenarios = [s["scenario"] for s in samples]

    model.eval()
    with torch.no_grad():
        win_logits, hp_preds = model(game_states)
        win_probs = torch.sigmoid(win_logits).squeeze(-1).cpu().numpy()
        hp_preds = hp_preds.squeeze(-1).cpu().numpy()

    pred_wins = (win_probs > 0.5).astype(float)

    # --- Overall metrics ---
    accuracy = (pred_wins == true_wins).mean()
    hp_mae = np.abs(hp_preds - true_hp).mean()

    print(f"\n{'='*60}")
    print(f"Entity Encoder Evaluation — {len(samples)} samples")
    print(f"{'='*60}")
    print(f"  Win prediction accuracy:  {accuracy*100:.1f}%")
    print(f"  HP remaining MAE:         {hp_mae:.4f}")
    print(f"  Mean predicted win prob:   {win_probs.mean():.3f}")
    print(f"  Actual win rate:           {true_wins.mean()*100:.1f}%")

    # --- Calibration ---
    print(f"\n--- Calibration (predicted prob vs actual win rate) ---")
    cal_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for i in range(len(cal_bins) - 1):
        lo, hi = cal_bins[i], cal_bins[i + 1]
        mask = (win_probs >= lo) & (win_probs < hi)
        if mask.sum() == 0:
            continue
        actual = true_wins[mask].mean()
        count = mask.sum()
        print(f"  pred [{lo:.1f}-{hi:.1f}):  actual={actual*100:5.1f}%  n={count:5d}")

    # --- Accuracy by fight progress ---
    print(f"\n--- Accuracy by fight progress ---")
    bucket_edges = np.linspace(0, 1, progress_buckets + 1)
    for i in range(progress_buckets):
        lo, hi = bucket_edges[i], bucket_edges[i + 1]
        mask = (progress >= lo) & (progress < hi + (1e-6 if i == progress_buckets - 1 else 0))
        if mask.sum() == 0:
            continue
        acc = (pred_wins[mask] == true_wins[mask]).mean()
        mae = np.abs(hp_preds[mask] - true_hp[mask]).mean()
        conf = np.abs(win_probs[mask] - 0.5).mean() + 0.5  # avg confidence
        print(
            f"  progress [{lo:.1f}-{hi:.1f}]:  acc={acc*100:5.1f}%  hp_mae={mae:.4f}"
            f"  confidence={conf:.3f}  n={mask.sum():5d}"
        )

    # --- Per-scenario breakdown ---
    unique_scenarios = sorted(set(scenarios))
    if len(unique_scenarios) > 1:
        print(f"\n--- Per-scenario accuracy (worst 10) ---")
        scenario_stats = []
        for sc in unique_scenarios:
            mask = np.array([s == sc for s in scenarios])
            acc = (pred_wins[mask] == true_wins[mask]).mean()
            n = mask.sum()
            actual_outcome = "win" if true_wins[mask][0] > 0.5 else "loss"
            scenario_stats.append((sc, acc, n, actual_outcome))

        scenario_stats.sort(key=lambda x: x[1])
        for sc, acc, n, outcome in scenario_stats[:10]:
            print(f"  {sc:<45s}  acc={acc*100:5.1f}%  n={n:4d}  ({outcome})")

        # Aggregate: per-scenario win prediction (tick-0 only)
        print(f"\n--- Scenario-level win prediction (majority vote) ---")
        correct = 0
        total = 0
        for sc in unique_scenarios:
            mask = np.array([s == sc for s in scenarios])
            majority_pred = (pred_wins[mask].mean() > 0.5)
            actual = true_wins[mask][0] > 0.5
            if majority_pred == actual:
                correct += 1
            total += 1
        print(f"  {correct}/{total} scenarios correctly predicted ({correct/total*100:.1f}%)")


def main():
    p = argparse.ArgumentParser(description="Evaluate entity encoder predictions")
    p.add_argument("checkpoint", help="Pretrained entity encoder checkpoint (.pt)")
    p.add_argument("--dataset", help="Existing JSONL outcome dataset to evaluate on")
    p.add_argument(
        "--held-out",
        help="Path to scenario dir/file to generate fresh held-out data from",
    )
    p.add_argument("--sample-interval", type=int, default=10)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    args = p.parse_args()

    if not args.dataset and not args.held_out:
        p.error("Provide --dataset or --held-out (or both)")

    # Load model
    model = EntityEncoderPretraining(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(DEVICE)
    state = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded model from {args.checkpoint}")

    # Evaluate on training data (to check for overfitting vs generalization)
    if args.dataset:
        print(f"\n{'#'*60}")
        print(f"# Training data evaluation")
        print(f"{'#'*60}")
        samples = load_dataset(Path(args.dataset))
        evaluate(model, samples)

    # Evaluate on held-out data
    if args.held_out:
        print(f"\n{'#'*60}")
        print(f"# Held-out evaluation ({args.held_out})")
        print(f"{'#'*60}")
        samples = generate_held_out(args.held_out, args.sample_interval)
        if samples:
            evaluate(model, samples)
        else:
            print("  No held-out samples generated!")


if __name__ == "__main__":
    main()
