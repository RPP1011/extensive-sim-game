#!/usr/bin/env python3
"""Analyze VAE training dataset quality and distributions.

Usage:
    uv run --with pandas python3 training/analyze_vae_dataset.py
    uv run --with pandas python3 training/analyze_vae_dataset.py --contexts generated/vae_contexts.jsonl
    uv run --with pandas python3 training/analyze_vae_dataset.py --dataset generated/vae_dataset.jsonl
"""

import argparse
import json
import sys

import pandas as pd


def analyze_contexts(path: str):
    """Analyze sweep output (trigger contexts)."""
    print(f"\n{'='*60}")
    print(f"CONTEXT ANALYSIS: {path}")
    print(f"{'='*60}")

    df = pd.read_json(path, lines=True)
    print(f"\nTotal contexts: {len(df):,}")

    # Content type distribution
    print(f"\n--- Content Type Distribution ---")
    print(df.content_type.value_counts().to_string())

    # Trigger distribution
    print(f"\n--- Top 20 Triggers ---")
    print(df.trigger.value_counts().head(20).to_string())

    # Archetype distribution
    if "archetype" in df.columns:
        print(f"\n--- Archetype Distribution ---")
        print(df[df.archetype != ""].archetype.value_counts().to_string())

    # Level distribution
    if "level" in df.columns:
        levels = df[df.level > 0].level
        if len(levels) > 0:
            print(f"\n--- Level Distribution ---")
            print(f"  Mean: {levels.mean():.1f}, Median: {levels.median():.0f}, "
                  f"Min: {levels.min()}, Max: {levels.max()}")

    # Input vector analysis
    if "input" in df.columns:
        inputs = pd.DataFrame(df["input"].tolist())
        print(f"\n--- Input Vector ({inputs.shape[1]} dims) ---")

        # Dead dimensions (std < 0.001)
        stds = inputs.std()
        dead = stds[stds < 0.001]
        if len(dead) > 0:
            print(f"  WARNING: {len(dead)} dead dimensions (std < 0.001):")
            print(f"    Indices: {list(dead.index)}")
        else:
            print(f"  No dead dimensions (all stds > 0.001)")

        # Feature coverage
        means = inputs.mean()
        all_zero = (inputs == 0).all()
        always_zero = all_zero[all_zero].index.tolist()
        if always_zero:
            print(f"  Always-zero dims: {always_zero}")

        print(f"  Mean range: [{means.min():.4f}, {means.max():.4f}]")
        print(f"  Std range: [{stds.min():.4f}, {stds.max():.4f}]")

        # Feature group means (per the 124-dim layout)
        groups = {
            "Identity (0-38)": range(0, 39),
            "Quest History (39-56)": range(39, 57),
            "Guild State (57-68)": range(57, 69),
            "World State (69-105)": range(69, 106),
            "Trigger Context (106-123)": range(106, 124),
        }
        print(f"\n  Feature Group Means:")
        for name, dims in groups.items():
            valid_dims = [d for d in dims if d < inputs.shape[1]]
            if valid_dims:
                group_mean = inputs[valid_dims].mean().mean()
                group_std = inputs[valid_dims].std().mean()
                group_dead = sum(1 for d in valid_dims if stds.get(d, 0) < 0.001)
                print(f"    {name}: mean={group_mean:.3f}, std={group_std:.3f}, dead={group_dead}")

    # Dedup analysis
    if "context_text" in df.columns:
        unique_contexts = df.context_text.nunique()
        print(f"\n--- Dedup Potential ---")
        print(f"  Total contexts: {len(df):,}")
        print(f"  Unique prompts: {unique_contexts:,}")
        print(f"  Dedup ratio: {unique_contexts/len(df)*100:.1f}% unique")

    # Cross-tab: trigger type × content type
    print(f"\n--- Trigger × Content Type ---")
    df["trigger_type"] = df.trigger.str.extract(r'^([a-z_]+?)_?\d*$')[0].fillna(df.trigger)
    ct = pd.crosstab(df.trigger_type, df.content_type)
    print(ct.to_string())


def analyze_dataset(path: str):
    """Analyze final dataset (with slots)."""
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS: {path}")
    print(f"{'='*60}")

    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("  No records found!")
        return

    df = pd.DataFrame(records)
    print(f"\nTotal records: {len(df):,}")

    # Valid vs invalid
    valid = df[df.valid == True]
    invalid = df[df.valid == False]
    print(f"Valid: {len(valid):,} ({len(valid)/len(df)*100:.0f}%)")
    print(f"Invalid: {len(invalid):,} ({len(invalid)/len(df)*100:.0f}%)")

    # Per-type validity
    print(f"\n--- Validity by Content Type ---")
    for ct in df.content_type.unique():
        sub = df[df.content_type == ct]
        v = sub[sub.valid == True]
        print(f"  {ct}: {len(v)}/{len(sub)} valid ({len(v)/len(sub)*100:.0f}%)")

    # Score distribution (valid only)
    if len(valid) > 0 and "score" in valid.columns:
        print(f"\n--- Score Distribution (valid only) ---")
        print(valid.groupby("content_type").score.describe().to_string())

    # Slot vector analysis
    if len(valid) > 0 and "slots" in valid.columns:
        print(f"\n--- Slot Vector Dimensions ---")
        for ct in valid.content_type.unique():
            sub = valid[valid.content_type == ct]
            slot_lens = sub.slots.apply(len)
            print(f"  {ct}: {slot_lens.iloc[0]} dims ({len(sub)} samples)")

        # Check for collapsed slots (all same value)
        for ct in valid.content_type.unique():
            sub = valid[valid.content_type == ct]
            if len(sub) < 5:
                continue
            slots_df = pd.DataFrame(sub.slots.tolist())
            slot_stds = slots_df.std()
            collapsed = slot_stds[slot_stds < 0.001]
            print(f"  {ct}: {len(collapsed)}/{len(slot_stds)} collapsed slots")

    # Trigger distribution
    print(f"\n--- Top Triggers (valid) ---")
    if len(valid) > 0:
        print(valid.trigger.value_counts().head(15).to_string())

    # Sample outputs
    if len(valid) > 0 and "raw_dsl" in valid.columns:
        print(f"\n--- Sample Outputs ---")
        for ct in valid.content_type.unique():
            sub = valid[valid.content_type == ct].head(2)
            for _, row in sub.iterrows():
                print(f"\n[{ct}] trigger={row.trigger} score={row.get('score', '?')}")
                dsl = row.raw_dsl
                if len(dsl) > 300:
                    dsl = dsl[:300] + "..."
                print(dsl)


def analyze_llm_store(path: str):
    """Analyze LLM content store."""
    print(f"\n{'='*60}")
    print(f"LLM STORE ANALYSIS: {path}")
    print(f"{'='*60}")

    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        print("  No records found!")
        return

    df = pd.DataFrame(records)
    print(f"\nTotal LLM calls: {len(df):,}")

    cache_hits = df[df.cache_hit == True]
    valid = df[df.selected.notna()]
    print(f"Cache hits: {len(cache_hits):,} ({len(cache_hits)/len(df)*100:.0f}%)")
    print(f"Valid generations: {len(valid):,} ({len(valid)/len(df)*100:.0f}%)")

    if "time_s" in df.columns:
        non_cache = df[df.cache_hit == False]
        if len(non_cache) > 0:
            print(f"\n--- Generation Time (non-cache) ---")
            print(f"  Mean: {non_cache.time_s.mean():.1f}s")
            print(f"  Median: {non_cache.time_s.median():.1f}s")
            print(f"  Total: {non_cache.time_s.sum():.0f}s ({non_cache.time_s.sum()/60:.1f}min)")

    if "gen_type" in df.columns:
        print(f"\n--- By Generation Type ---")
        for gt in df.gen_type.unique():
            sub = df[df.gen_type == gt]
            v = sub[sub.selected.notna()]
            print(f"  {gt}: {len(v)}/{len(sub)} valid ({len(v)/len(sub)*100:.0f}%)")

    if "score" in df.columns:
        print(f"\n--- Score Distribution ---")
        print(df[df.selected.notna()].groupby("gen_type").score.describe().to_string())


def main():
    parser = argparse.ArgumentParser(description="Analyze VAE dataset quality")
    parser.add_argument("--contexts", default="generated/vae_contexts.jsonl",
                        help="Path to contexts JSONL (from sweep)")
    parser.add_argument("--dataset", default="generated/vae_dataset.jsonl",
                        help="Path to final dataset JSONL")
    parser.add_argument("--store", default="generated/llm_content_store.jsonl",
                        help="Path to LLM content store JSONL")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all available files")
    args = parser.parse_args()

    import os

    analyzed = False

    if args.all or os.path.exists(args.contexts):
        if os.path.exists(args.contexts):
            analyze_contexts(args.contexts)
            analyzed = True

    if args.all or os.path.exists(args.store):
        if os.path.exists(args.store):
            analyze_llm_store(args.store)
            analyzed = True

    if args.all or os.path.exists(args.dataset):
        if os.path.exists(args.dataset):
            analyze_dataset(args.dataset)
            analyzed = True

    if not analyzed:
        print("No data files found. Run the pipeline first:")
        print("  cargo run --bin xtask -- vae-dataset --campaigns 10 --sweep-only")
        sys.exit(1)


if __name__ == "__main__":
    main()
