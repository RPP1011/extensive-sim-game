#!/usr/bin/env python3
"""Filter episode data for high-signal combat pointer training (Stage 0e).

Identifies steps where target selection was tactically meaningful:
  1. Kill-securing: enemy target has <25% HP with multiple enemies alive
  2. CC targeting: ability used on a specific target (not just nearest)
  3. Heal targeting: heal used on most-damaged ally (not self)
  4. Focus switch: target changed from previous step
  5. Multi-target available: >1 valid target existed (choice was non-trivial)

Outputs a filtered npz with only high-signal steps + a "signal_type" label
for each step indicating what kind of tactical decision was made.

Usage:
    uv run --with numpy python training/filter_high_signal.py \
        generated/v5_data.jsonl \
        -o generated/v5_pointer_bc.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Signal type labels
SIGNAL_KILL_SECURE = 1    # targeting low-HP enemy for kill
SIGNAL_CC_TARGET = 2      # using CC on specific high-value target
SIGNAL_HEAL_TRIAGE = 3    # healing most-in-danger ally
SIGNAL_FOCUS_SWITCH = 4   # switching target (not same as last step)
SIGNAL_MULTI_TARGET = 5   # multiple valid targets, chose one


def classify_step_signal(
    step: dict,
    prev_step: dict | None,
    entity_types: list[int],
    entities: list[list[float]],
) -> list[int]:
    """Classify what tactical signals are present in this step.
    Returns list of signal types (can have multiple)."""
    signals = []

    combat_type = step.get("combat_type", 1)
    target_idx = step.get("target_idx", 0)

    # Count living enemies and allies
    enemy_indices = []
    ally_indices = []
    for ei, etype in enumerate(entity_types):
        if ei >= len(entities):
            break
        e = entities[ei]
        if len(e) <= 29 or e[29] < 0.5:
            continue
        if etype == 1:
            enemy_indices.append(ei)
        elif etype in (0, 2):
            ally_indices.append(ei)

    # Need at least 1 enemy for combat signals
    if not enemy_indices:
        return signals

    # 1. Kill-securing: targeting enemy with <25% HP
    if combat_type == 0 and target_idx < len(entities):  # attack
        target_ent = entities[target_idx] if target_idx < len(entities) else None
        if target_ent and len(target_ent) > 29 and target_ent[29] > 0.5:
            if target_ent[0] < 0.25 and len(enemy_indices) > 1:
                signals.append(SIGNAL_KILL_SECURE)

    # 2. CC targeting: ability with CC effect on a specific target
    if combat_type >= 2:  # ability usage
        signals.append(SIGNAL_CC_TARGET)  # any ability on a target is signal

    # 3. Heal triage: heal ability targeting an ally
    if combat_type >= 2:
        # Check if this looks like a heal (target is an ally)
        if target_idx < len(entity_types) and entity_types[target_idx] in (0, 2):
            target_ent = entities[target_idx] if target_idx < len(entities) else None
            if target_ent and len(target_ent) > 0 and target_ent[0] < 0.5:
                signals.append(SIGNAL_HEAL_TRIAGE)

    # 4. Focus switch: target changed from previous step
    if prev_step is not None:
        prev_target = prev_step.get("target_idx", -1)
        if target_idx != prev_target and combat_type == 0:
            signals.append(SIGNAL_FOCUS_SWITCH)

    # 5. Multi-target: multiple valid targets available
    if len(enemy_indices) >= 2 and combat_type == 0:
        signals.append(SIGNAL_MULTI_TARGET)

    return signals


def main():
    p = argparse.ArgumentParser(description="Filter high-signal steps for pointer BC")
    p.add_argument("data", nargs="+", help="Episode JSONL files")
    p.add_argument("-o", "--output", default="generated/v5_pointer_bc.npz")
    p.add_argument("--min-signals", type=int, default=1,
                    help="Minimum signal types per step to include")
    p.add_argument("--val-split", type=float, default=0.1)
    args = p.parse_args()

    MAX_ENTITIES = 20
    MAX_THREATS = 6
    MAX_POSITIONS = 8
    ENTITY_DIM = 34
    THREAT_DIM = 10
    POSITION_DIM = 8
    AGG_DIM = 16

    episodes = []
    for path in args.data:
        print(f"Loading {path}...")
        with open(path) as f:
            for line in f:
                episodes.append(json.loads(line))
    print(f"  {len(episodes)} episodes loaded")

    # Extract high-signal steps
    ent_list = []
    ent_type_list = []
    ent_mask_list = []
    thr_list = []
    thr_mask_list = []
    pos_list = []
    pos_mask_list = []
    agg_list = []
    combat_type_list = []
    target_idx_list = []
    move_dir_list = []
    signal_type_list = []  # bitmask of signal types
    ep_idx_list = []

    total_steps = 0
    signal_counts = {
        SIGNAL_KILL_SECURE: 0,
        SIGNAL_CC_TARGET: 0,
        SIGNAL_HEAL_TRIAGE: 0,
        SIGNAL_FOCUS_SWITCH: 0,
        SIGNAL_MULTI_TARGET: 0,
    }

    for epi, ep in enumerate(episodes):
        steps = ep["steps"]
        prev_step = None
        for step in steps:
            total_steps += 1
            entities = step.get("entities")
            if not entities:
                prev_step = step
                continue

            entity_types = step.get("entity_types", [])
            signals = classify_step_signal(step, prev_step, entity_types, entities)
            prev_step = step

            if len(signals) < args.min_signals:
                continue

            # Encode signal types as bitmask
            signal_mask = 0
            for s in signals:
                signal_mask |= (1 << s)
                signal_counts[s] += 1

            # Pad entities
            ent = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
            et = np.zeros(MAX_ENTITIES, dtype=np.int32)
            em = np.ones(MAX_ENTITIES, dtype=np.bool_)
            for j in range(min(len(entities), MAX_ENTITIES)):
                e = entities[j]
                ent[j, :min(len(e), ENTITY_DIM)] = e[:ENTITY_DIM]
                em[j] = False
            for j in range(min(len(entity_types), MAX_ENTITIES)):
                et[j] = entity_types[j]

            threats = step.get("threats", [])
            thr = np.zeros((MAX_THREATS, THREAT_DIM), dtype=np.float32)
            tm = np.ones(MAX_THREATS, dtype=np.bool_)
            for j in range(min(len(threats), MAX_THREATS)):
                t = threats[j]
                thr[j, :min(len(t), THREAT_DIM)] = t[:THREAT_DIM]
                tm[j] = False

            positions = step.get("positions", [])
            pos = np.zeros((MAX_POSITIONS, POSITION_DIM), dtype=np.float32)
            pm = np.ones(MAX_POSITIONS, dtype=np.bool_)
            for j in range(min(len(positions), MAX_POSITIONS)):
                pp = positions[j]
                pos[j, :min(len(pp), POSITION_DIM)] = pp[:POSITION_DIM]
                pm[j] = False

            agg_raw = step.get("aggregate_features", [0.0] * AGG_DIM)
            agg = np.zeros(AGG_DIM, dtype=np.float32)
            if agg_raw:
                agg[:min(len(agg_raw), AGG_DIM)] = agg_raw[:AGG_DIM]

            ent_list.append(ent)
            ent_type_list.append(et)
            ent_mask_list.append(em)
            thr_list.append(thr)
            thr_mask_list.append(tm)
            pos_list.append(pos)
            pos_mask_list.append(pm)
            agg_list.append(agg)
            combat_type_list.append(step.get("combat_type", 1))
            target_idx_list.append(step.get("target_idx", 0))
            move_dir_list.append(step.get("move_dir", 8))
            signal_type_list.append(signal_mask)
            ep_idx_list.append(epi)

    N = len(ent_list)
    print(f"\n  Total steps scanned: {total_steps}")
    print(f"  High-signal steps: {N} ({100*N/max(total_steps,1):.1f}%)")
    print(f"\n  Signal breakdown:")
    signal_names = {
        SIGNAL_KILL_SECURE: "kill_secure",
        SIGNAL_CC_TARGET: "cc_target",
        SIGNAL_HEAL_TRIAGE: "heal_triage",
        SIGNAL_FOCUS_SWITCH: "focus_switch",
        SIGNAL_MULTI_TARGET: "multi_target",
    }
    for sig, name in signal_names.items():
        print(f"    {name:15s}: {signal_counts[sig]:6d} ({100*signal_counts[sig]/max(N,1):.1f}%)")

    if N == 0:
        print("No high-signal steps found!")
        return

    # Stack arrays
    ent_feat = np.stack(ent_list)
    ent_types = np.stack(ent_type_list)
    ent_mask = np.stack(ent_mask_list)
    thr_feat = np.stack(thr_list)
    thr_mask = np.stack(thr_mask_list)
    pos_feat = np.stack(pos_list)
    pos_mask = np.stack(pos_mask_list)
    agg_feat = np.stack(agg_list)
    combat_type = np.array(combat_type_list, dtype=np.int32)
    target_idx = np.array(target_idx_list, dtype=np.int32)
    move_dir = np.array(move_dir_list, dtype=np.int32)
    signal_type = np.array(signal_type_list, dtype=np.int32)
    ep_idx = np.array(ep_idx_list, dtype=np.int32)

    # Train/val split by episode
    n_episodes = len(episodes)
    perm = np.random.RandomState(42).permutation(n_episodes)
    n_val_ep = max(1, int(n_episodes * args.val_split))
    val_ep_set = set(perm[:n_val_ep].tolist())

    val_mask_arr = np.array([ep_idx[i] in val_ep_set for i in range(N)])
    train_idx = np.where(~val_mask_arr)[0]
    val_idx = np.where(val_mask_arr)[0]

    print(f"\n  Train: {len(train_idx)} samples")
    print(f"  Val:   {len(val_idx)} samples")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ent_feat=ent_feat, ent_types=ent_types, ent_mask=ent_mask,
        thr_feat=thr_feat, thr_mask=thr_mask,
        pos_feat=pos_feat, pos_mask=pos_mask,
        agg_feat=agg_feat,
        combat_type=combat_type, target_idx=target_idx,
        move_dir=move_dir, signal_type=signal_type,
        train_idx=train_idx, val_idx=val_idx,
    )
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\nSaved {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
