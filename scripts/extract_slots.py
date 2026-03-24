#!/usr/bin/env python3
"""Extract slot vectors from VAE dataset DSL text via Rust parser.

Reads vae_dataset_final.jsonl, shells out to a Rust extractor binary for
each record, writes (input, slots, content_type) tuples to NPZ for training.

Usage:
    # First build the extractor
    cargo build --release --bin xtask

    # Then run extraction
    uv run --with numpy python3 scripts/extract_slots.py \
        --input generated/vae_dataset_final.jsonl \
        --output generated/vae_training_data.npz
"""

import argparse
import json
import subprocess
import sys
import os
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="generated/vae_dataset_final.jsonl")
    parser.add_argument("--output", "-o", default="generated/vae_training_data.npz")
    parser.add_argument("--rust-bin", default="target/release/xtask")
    args = parser.parse_args()

    if not os.path.exists(args.rust_bin):
        print(f"Rust binary not found at {args.rust_bin}, building...", file=sys.stderr)
        subprocess.run(["cargo", "build", "--release", "--bin", "xtask"], check=True)

    print(f"Loading {args.input}...", file=sys.stderr)
    records = []
    with open(args.input) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(records):,} records", file=sys.stderr)

    # Write DSL texts to temp file for batch processing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        tmp_path = tmp.name
        for r in records:
            json.dump({"content_type": r["content_type"], "raw_dsl": r["raw_dsl"]}, tmp)
            tmp.write("\n")

    print(f"Running Rust slot extractor...", file=sys.stderr)
    result = subprocess.run(
        [args.rust_bin, "vae-extract-slots", "--input", tmp_path],
        capture_output=True, text=True
    )

    os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"Extractor failed: {result.stderr}", file=sys.stderr)
        # Fall back to Python-only extraction
        print("Falling back to regex-based Python extraction...", file=sys.stderr)
        return extract_python_fallback(records, args.output)

    # Parse slot vectors from stdout
    slot_records = []
    for line in result.stdout.strip().split("\n"):
        if line:
            slot_records.append(json.loads(line))

    if len(slot_records) != len(records):
        print(f"WARNING: {len(slot_records)} slots vs {len(records)} records", file=sys.stderr)

    build_npz(records, slot_records, args.output)


def extract_python_fallback(records, output_path):
    """Simple regex-based slot extraction when Rust isn't available."""
    import re

    ABILITY_SLOT_DIM = 142
    CLASS_SLOT_DIM = 75

    slot_records = []
    for r in records:
        dsl = r["raw_dsl"]
        ct = r["content_type"]

        if ct == "ability":
            slots = extract_ability_slots_py(dsl)
        elif ct == "class":
            slots = extract_class_slots_py(dsl)
        else:
            slots = []

        slot_records.append({"slots": slots, "valid": len(slots) > 0})

    build_npz(records, slot_records, output_path)


def extract_ability_slots_py(dsl):
    """Python regex-based ability slot extraction."""
    import re
    slots = [0.0] * 142

    # output_type: active=0, passive=1
    if "type: active" in dsl:
        slots[0] = 1.0
    elif "type: passive" in dsl:
        slots[1] = 1.0

    # targeting (8 dims at offset 3)
    # Default to TargetEnemy for active, skip for passive
    if slots[0] == 1.0:
        slots[3] = 1.0  # TargetEnemy

    # range, cooldown, cast, cost (offset 11)
    cd_m = re.search(r'cooldown:\s*(\d+)s', dsl)
    if cd_m:
        slots[12] = int(cd_m.group(1)) / 30.0

    # hint (offset 15)
    tag_m = re.search(r'tag:\s*(\w+)', dsl)
    if tag_m:
        tag = tag_m.group(1)
        hint_map = {"melee": 0, "defense": 2, "arcane": 0, "healing": 4,
                    "stealth": 3, "ranged": 0, "nature": 3, "assassination": 0}
        hi = hint_map.get(tag, 3)
        slots[15 + hi] = 1.0

    # delivery (offset 20): default instant
    slots[20] = 1.0

    # effect type (offset 42, per-effect at 25-dim blocks)
    effect_m = re.search(r'effect:\s*(.+)', dsl)
    if effect_m:
        effect_line = effect_m.group(1).lower()
        if "damage" in effect_line: slots[42] = 1.0
        elif "heal" in effect_line: slots[43] = 1.0
        elif "shield" in effect_line: slots[44] = 1.0
        elif "stun" in effect_line: slots[45] = 1.0
        elif "buff" in effect_line: slots[51] = 1.0
        elif "debuff" in effect_line: slots[52] = 1.0
        elif "stealth" in effect_line: slots[54] = 1.0

        # Extract primary param
        num_m = re.search(r'(\d+)', effect_line)
        if num_m:
            slots[59] = int(num_m.group(1)) / 155.0

        # Duration
        dur_m = re.search(r'(\d+)s', effect_line)
        if dur_m:
            slots[60] = int(dur_m.group(1)) * 1000 / 10000.0

    # trigger (for passives)
    if slots[1] == 1.0:
        trig_m = re.search(r'trigger:\s*(\w+)', dsl)
        if trig_m:
            # Encode trigger presence
            slots[66] = 1.0  # has condition flag

    return slots


def extract_class_slots_py(dsl):
    """Python regex-based class slot extraction."""
    import re
    slots = [0.0] * 75

    # stat_growth (5 dims)
    growth_m = re.search(r'stat_growth:\s*(.+?)per level', dsl)
    if growth_m:
        line = growth_m.group(1)
        for stat_i, stat in enumerate(["attack", "defense", "speed", "max_hp", "ability_power"]):
            m = re.search(rf'\+(\d+)\s*{stat}', line)
            if m:
                slots[stat_i] = int(m.group(1)) / 5.0

    # tags multi-hot (16 dims at offset 5)
    tags_m = re.search(r'tags:\s*(.+)', dsl)
    if tags_m:
        tag_map = {"ranged": 0, "nature": 1, "stealth": 2, "tracking": 3,
                   "survival": 4, "melee": 5, "defense": 6, "leadership": 7,
                   "arcane": 8, "elemental": 9, "healing": 10, "divine": 11,
                   "assassination": 12, "agility": 13, "deception": 14, "sabotage": 15}
        for tag in tags_m.group(1).split(","):
            tag = tag.strip()
            if tag in tag_map:
                slots[5 + tag_map[tag]] = 1.0

    # scaling source (11 dims at offset 21)
    scale_m = re.search(r'scaling\s+(\w+)', dsl)
    if scale_m:
        source_map = {"party_alive_count": 0, "faction_allied_count": 2,
                      "crisis_active": 4, "threat_level": 10, "fame": 5}
        si = source_map.get(scale_m.group(1), 0)
        slots[21 + si] = 1.0

    # abilities count (offset 56-65, cap 5)
    levels = re.findall(r'level\s+(\d+):', dsl)
    for i, lv in enumerate(levels[:5]):
        slots[56 + i * 2] = int(lv) / 40.0
        slots[56 + i * 2 + 1] = 1.0

    # requirements (offset 66-73)
    req_m = re.search(r'requirements:\s*(.+)', dsl)
    if req_m:
        req_line = req_m.group(1)
        level_m = re.search(r'level\s+(\d+)', req_line)
        if level_m:
            slots[66] = 0.0  # type index
            slots[67] = int(level_m.group(1)) / 20.0
        fame_m = re.search(r'fame\s+(\d+)', req_line)
        if fame_m:
            slots[68] = 1.0 / 7.0
            slots[69] = int(fame_m.group(1)) / 2000.0

    # consolidates_at (offset 74)
    cons_m = re.search(r'consolidates_at:\s*(\d+)', dsl)
    if cons_m:
        slots[74] = int(cons_m.group(1)) / 20.0

    return slots


def build_npz(records, slot_records, output_path):
    """Build NPZ from records + slot vectors."""
    inputs = []
    slots_ability = []
    slots_class = []
    content_types = []

    ability_count = 0
    class_count = 0
    skipped = 0

    for i, r in enumerate(records):
        sr = slot_records[i] if i < len(slot_records) else {"slots": [], "valid": False}
        slot_vec = sr.get("slots", [])

        if not slot_vec or not sr.get("valid", True):
            skipped += 1
            continue

        inp = r.get("input", [])
        if len(inp) != 124:
            skipped += 1
            continue

        ct = r["content_type"]
        if ct == "ability" and len(slot_vec) == 142:
            inputs.append(inp)
            slots_ability.append(slot_vec)
            slots_class.append([0.0] * 75)
            content_types.append(0)
            ability_count += 1
        elif ct == "class" and len(slot_vec) == 75:
            inputs.append(inp)
            slots_ability.append([0.0] * 142)
            slots_class.append(slot_vec)
            content_types.append(1)
            class_count += 1
        else:
            skipped += 1

    inputs = np.array(inputs, dtype=np.float32)
    slots_ability = np.array(slots_ability, dtype=np.float32)
    slots_class = np.array(slots_class, dtype=np.float32)
    content_types = np.array(content_types, dtype=np.int32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=inputs,
        slots_ability=slots_ability,
        slots_class=slots_class,
        content_types=content_types,
    )

    print(f"\n=== NPZ Written ===", file=sys.stderr)
    print(f"  Abilities: {ability_count:,}", file=sys.stderr)
    print(f"  Classes: {class_count:,}", file=sys.stderr)
    print(f"  Skipped: {skipped:,}", file=sys.stderr)
    print(f"  Input shape: {inputs.shape}", file=sys.stderr)
    print(f"  Ability slots shape: {slots_ability.shape}", file=sys.stderr)
    print(f"  Class slots shape: {slots_class.shape}", file=sys.stderr)
    print(f"  Output: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
