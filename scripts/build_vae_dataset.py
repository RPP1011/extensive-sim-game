#!/usr/bin/env python3
"""Combine sweep contexts + Gemini content store into final VAE training dataset.

Reads:
  - Sweep contexts (from vae-dataset --sweep-only)
  - Gemini content store (from gemini_generate.py)
Writes:
  - Final dataset JSONL with (input_vector, raw_dsl, content_type, metadata)

The Rust slot extraction step can be run separately afterwards.

Usage:
    uv run --with pandas python3 scripts/build_vae_dataset.py \
        --contexts generated/vae_v6/vae_contexts.jsonl \
        --store generated/gemini_content_store.jsonl \
        --output generated/vae_dataset_final.jsonl
"""

import argparse
import json
import os
import sys


def coarse_key(ctx):
    level_bucket = (ctx.get("level", 0) // 5) * 5
    return f"{ctx.get('archetype', '')}_{level_bucket}_{ctx.get('trigger', '')}_{ctx.get('content_type', '')}"


def main():
    parser = argparse.ArgumentParser(description="Build final VAE dataset")
    parser.add_argument("--contexts", default="generated/vae_v6/vae_contexts.jsonl")
    parser.add_argument("--store", default="generated/gemini_content_store.jsonl")
    parser.add_argument("--output", "-o", default="generated/vae_dataset_final.jsonl")
    parser.add_argument("--valid-only", action="store_true", help="Only include valid parses")
    args = parser.parse_args()

    # Load content store: coarse_key → best content
    print(f"Loading content store from {args.store}...", file=sys.stderr)
    content_map = {}  # coarse_key → (selected_text, score)
    store_total = 0
    store_valid = 0
    with open(args.store) as f:
        for line in f:
            try:
                r = json.loads(line)
                store_total += 1
                key = r.get("coarse_key", "")
                if r.get("selected") and r.get("valid", False):
                    store_valid += 1
                    existing = content_map.get(key)
                    if existing is None or r.get("score", 0) > existing[1]:
                        content_map[key] = (r["selected"], r.get("score", 0))
            except json.JSONDecodeError:
                continue

    print(f"  {store_total} records, {store_valid} valid, {len(content_map)} unique keys", file=sys.stderr)

    # Load contexts and match to content
    print(f"Loading contexts from {args.contexts}...", file=sys.stderr)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    matched = 0
    unmatched = 0
    total_written = 0
    by_type = {}

    with open(args.contexts) as fin, open(args.output, "w") as fout:
        for line in fin:
            try:
                ctx = json.loads(line)
            except json.JSONDecodeError:
                continue

            ct = ctx.get("content_type", "")
            if ct not in ("ability", "class"):
                continue

            key = coarse_key(ctx)
            content_entry = content_map.get(key)

            if content_entry is None:
                unmatched += 1
                if args.valid_only:
                    continue
                # Write with empty content for completeness
                record = {
                    "input": ctx.get("input", []),
                    "content_type": ct,
                    "raw_dsl": "",
                    "valid": False,
                    "score": 0,
                    "trigger": ctx.get("trigger", ""),
                    "archetype": ctx.get("archetype", ""),
                    "level": ctx.get("level", 0),
                    "seed": ctx.get("seed", 0),
                    "tick": ctx.get("tick", 0),
                    "coarse_key": key,
                }
            else:
                matched += 1
                dsl_text, score = content_entry
                record = {
                    "input": ctx.get("input", []),
                    "content_type": ct,
                    "raw_dsl": dsl_text,
                    "valid": True,
                    "score": score,
                    "trigger": ctx.get("trigger", ""),
                    "archetype": ctx.get("archetype", ""),
                    "level": ctx.get("level", 0),
                    "seed": ctx.get("seed", 0),
                    "tick": ctx.get("tick", 0),
                    "coarse_key": key,
                }

            fout.write(json.dumps(record) + "\n")
            total_written += 1

            entry = by_type.setdefault(ct, [0, 0])
            entry[0] += 1
            if content_entry:
                entry[1] += 1

    print(f"\n=== Dataset Built ===", file=sys.stderr)
    print(f"Total written: {total_written:,}", file=sys.stderr)
    print(f"Matched to content: {matched:,}", file=sys.stderr)
    print(f"Unmatched (no LLM output): {unmatched:,}", file=sys.stderr)
    for ct, (total, valid) in sorted(by_type.items()):
        pct = valid / total * 100 if total > 0 else 0
        print(f"  {ct}: {valid:,}/{total:,} with content ({pct:.0f}%)", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    # Quick quality stats
    if matched > 0:
        print(f"\nDataset ready for training:", file=sys.stderr)
        print(f"  Input dim: 124", file=sys.stderr)
        print(f"  Matched records: {matched:,}", file=sys.stderr)
        print(f"  Run Rust slot extraction next:", file=sys.stderr)
        print(f"    cargo run --release --bin xtask -- vae-dataset --extract-only", file=sys.stderr)


if __name__ == "__main__":
    main()
