#!/usr/bin/env python3
"""Convert V4 JSONL episodes to npz for fast training.

Flattens episodes to steps, applies smart sampling, pads entities to fixed
max, and stores everything as dense numpy arrays. Loads in <1s vs ~35s for JSON.

Usage:
    uv run --with numpy training/convert_v4_npz.py \
        generated/rl_v4_phase3_combined.jsonl \
        -o generated/rl_v4_phase3.npz \
        --smart-sample
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


ENTITY_DIM = 30
THREAT_DIM = 8
MAX_MASK = 14  # 3 base + up to 8 abilities + padding


def main():
    p = argparse.ArgumentParser(description="Convert V4 JSONL → npz")
    p.add_argument("input", help="JSONL episode file")
    p.add_argument("-o", "--output", required=True, help="Output .npz file")
    p.add_argument("--smart-sample", action="store_true")
    p.add_argument("--hold-keep-ratio", type=float, default=0.1)
    p.add_argument("--wins-only", action="store_true")
    p.add_argument("--max-steps", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(42)

    print(f"Loading {args.input}...")
    t0 = time.time()

    # First pass: collect steps and determine max entities
    all_steps: list[dict] = []
    # unit_id → list of ability names (for CLS lookup)
    unit_abilities: dict[int, list[str]] = {}

    with open(args.input) as f:
        for line in f:
            ep = json.loads(line)
            if args.wins_only and ep.get("outcome") != "Victory":
                continue

            # Collect ability names
            for uid_str, names in ep.get("unit_ability_names", {}).items():
                unit_abilities[int(uid_str)] = names

            prev_action: dict[int, tuple[int, int]] = {}
            for step in ep["steps"]:
                if step.get("move_dir") is None:
                    continue

                if args.smart_sample:
                    md = step["move_dir"]
                    ct = step["combat_type"]
                    uid = step["unit_id"]
                    cur = (md, ct)
                    is_transition = uid in prev_action and prev_action[uid] != cur
                    prev_action[uid] = cur
                    is_interesting = (
                        md != 8 or ct == 0 or ct >= 2 or is_transition
                    )
                    if not is_interesting and rng.random() > args.hold_keep_ratio:
                        continue

                all_steps.append(step)

    print(f"  {len(all_steps)} steps in {time.time()-t0:.1f}s")

    if args.max_steps > 0 and len(all_steps) > args.max_steps:
        idx = rng.choice(len(all_steps), size=args.max_steps, replace=False)
        all_steps = [all_steps[i] for i in sorted(idx)]
        print(f"  Subsampled to {len(all_steps)} steps")

    N = len(all_steps)
    if N == 0:
        print("No steps found!")
        return

    # Determine max entities across all steps
    max_ents = max(len(s["entities"]) for s in all_steps)
    print(f"  Max entities: {max_ents}")

    # Allocate arrays
    entities = np.zeros((N, max_ents, ENTITY_DIM), dtype=np.float32)
    entity_types = np.zeros((N, max_ents), dtype=np.int8)
    entity_counts = np.zeros(N, dtype=np.int8)
    mask = np.zeros((N, MAX_MASK), dtype=np.bool_)
    move_dir = np.zeros(N, dtype=np.int8)
    combat_type = np.zeros(N, dtype=np.int8)
    target_idx = np.zeros(N, dtype=np.int16)
    unit_id = np.zeros(N, dtype=np.int32)

    for i, s in enumerate(all_steps):
        n_e = len(s["entities"])
        entities[i, :n_e] = s["entities"]
        entity_types[i, :n_e] = s["entity_types"]
        entity_counts[i] = n_e
        m = s["mask"]
        mask[i, :len(m)] = m
        move_dir[i] = s["move_dir"]
        combat_type[i] = s["combat_type"]
        target_idx[i] = s.get("target_idx", 0)
        unit_id[i] = s["unit_id"]

    # Save unit abilities as JSON string (small, ~few KB)
    ua_json = json.dumps({str(k): v for k, v in unit_abilities.items()})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        entities=entities,
        entity_types=entity_types,
        entity_counts=entity_counts,
        mask=mask,
        move_dir=move_dir,
        combat_type=combat_type,
        target_idx=target_idx,
        unit_id=unit_id,
        unit_abilities_json=np.array([ua_json]),
    )

    size_mb = out.stat().st_size / 1e6
    print(f"  Saved {N:,} steps to {out} ({size_mb:.1f} MB)")

    # Distribution summary
    n_moving = np.sum(move_dir != 8)
    n_attack = np.sum(combat_type == 0)
    n_hold = np.sum(combat_type == 1)
    n_ability = np.sum(combat_type >= 2)
    print(f"  Movement: {n_moving}/{N} ({100*n_moving/N:.1f}%) actual, "
          f"{N-n_moving}/{N} ({100*(N-n_moving)/N:.1f}%) stay")
    print(f"  Combat: attack={n_attack} hold={n_hold} abilities={n_ability}")


if __name__ == "__main__":
    main()
