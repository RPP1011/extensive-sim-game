#!/usr/bin/env python3
"""Create holdout structure set for Phase 3 generalization validation.

Generates ability DSL strings using grammar rule combinations that are
deliberately withheld from the training corpus. Each combination uses features
that appear individually in training but are never combined together — testing
whether the model has learned compositional rules vs memorized examples.

Must be run BEFORE any training. Outputs:
  - holdout_structures.jsonl: the held-out ability DSL strings
  - holdout_hashes.txt: SHA256 hashes for exclusion filtering

Usage:
    uv run training/data/create_holdout.py \
        -o training/data/holdout_structures.jsonl \
        --hashes training/data/holdout_hashes.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

random.seed(42)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Holdout combination templates
# ---------------------------------------------------------------------------
# Each function produces a list of valid DSL strings exercising a specific
# rare combination. Individual features (tether, ring, channel, etc.) all
# appear in training — only the *combination* is held out.


def _tether_ring(n: int) -> list[str]:
    """Tether delivery + ring area — both exist separately, never combined."""
    results = []
    for i in range(n):
        dmg = random.randint(20, 60)
        inner = random.randint(1, 3)
        outer = inner + random.randint(2, 5)
        max_r = random.randint(3, 7)
        cd = random.randint(8, 20)
        results.append(f"""ability HoldoutTR{i} {{
    target: enemy, range: {max_r}.0
    cooldown: {cd}s, cast: 0ms
    hint: damage

    deliver tether {{ max_range: {max_r}.0 }} {{
        on_complete {{
    damage {dmg} in ring({inner}.0, {outer}.0)
        }}
    }}
}}""")
    return results


def _channel_or_condition(n: int) -> list[str]:
    """Channel delivery + or compound condition."""
    results = []
    conds = [
        ("target_hp_below(50%)", "caster_hp_below(30%)"),
        ("target_is_stunned", "target_is_rooted"),
        ("target_hp_above(80%)", "caster_hp_above(60%)"),
        ("target_is_slowed", "target_is_silenced"),
    ]
    for i in range(n):
        dmg = random.randint(15, 50)
        dur = random.randint(2, 5)
        tick = random.randint(300, 800)
        cd = random.randint(10, 25)
        c1, c2 = random.choice(conds)
        results.append(f"""ability HoldoutCO{i} {{
    target: enemy, range: {random.randint(3, 7)}.0
    cooldown: {cd}s, cast: {random.randint(500, 1500)}ms
    hint: damage

    deliver channel {{ duration: {dur}s, tick: {tick}ms }} {{
        on_hit {{
    damage {dmg} when or({c1}, {c2})
        }}
    }}
}}""")
    return results


def _recast_zone(n: int) -> list[str]:
    """Recast + zone delivery — both common alone, rarely combined."""
    results = []
    for i in range(n):
        dmg = random.randint(15, 40)
        dur = random.randint(3, 8)
        tick = random.randint(500, 1500)
        cd = random.randint(8, 18)
        results.append(f"""ability HoldoutRZ{i} {{
    target: ground, range: {random.randint(3, 6)}.0
    cooldown: {cd}s, cast: {random.randint(200, 800)}ms
    hint: damage
    recast: 2
    recast_window: {random.randint(3, 6)}s

    deliver zone {{ duration: {dur}s, tick: {tick}ms }} {{
        on_hit {{
    damage {dmg}
        }}
    }}
}}""")
    return results


def _trap_cone(n: int) -> list[str]:
    """Trap delivery + cone area."""
    results = []
    for i in range(n):
        dmg = random.randint(25, 65)
        trap_dur = random.randint(5, 15)
        radius = random.randint(1, 3)
        cone_r = random.randint(3, 6)
        angle = random.choice([60, 90, 120])
        cd = random.randint(10, 25)
        results.append(f"""ability HoldoutTC{i} {{
    target: ground, range: {random.randint(3, 6)}.0
    cooldown: {cd}s, cast: 0ms
    hint: crowd_control

    deliver trap {{ duration: {trap_dur}s, trigger_radius: {radius}.0 }} {{
        on_hit {{
    damage {dmg} in cone({cone_r}.0, {angle}.0)
    stun {random.randint(1, 2)}s
        }}
    }}
}}""")
    return results


def _chain_scaling(n: int) -> list[str]:
    """Chain delivery + % scaling — both common, rarely combined."""
    results = []
    stats = ["target_max_hp", "target_missing_hp", "caster_max_hp",
             "caster_attack_damage"]
    for i in range(n):
        dmg = random.randint(15, 45)
        bounces = random.randint(2, 4)
        b_range = random.randint(3, 6)
        pct = random.randint(5, 25)
        stat = random.choice(stats)
        cd = random.randint(6, 15)
        results.append(f"""ability HoldoutCS{i} {{
    target: enemy, range: {random.randint(4, 7)}.0
    cooldown: {cd}s, cast: {random.randint(200, 600)}ms
    hint: damage

    deliver chain {{ bounces: {bounces}, range: {b_range}.0 }} {{
        on_hit {{
    damage {dmg} + {pct}% {stat}
        }}
    }}
}}""")
    return results


def _heal_spread_charges(n: int) -> list[str]:
    """Heal + spread area + charges — each common, combination rare."""
    results = []
    for i in range(n):
        heal_amt = random.randint(20, 50)
        radius = random.randint(2, 5)
        max_t = random.randint(2, 4)
        charges = random.randint(2, 3)
        recharge = random.randint(6, 12)
        cd = random.randint(4, 10)
        results.append(f"""ability HoldoutHS{i} {{
    target: ally, range: {random.randint(3, 6)}.0
    cooldown: {cd}s, cast: {random.randint(200, 800)}ms
    hint: heal
    charges: {charges}
    recharge: {recharge}s

    heal {heal_amt} in spread({radius}.0, {max_t})
}}""")
    return results


def _shield_line_conditional(n: int) -> list[str]:
    """Shield + line area + conditional — structural combo test."""
    results = []
    for i in range(n):
        shield_amt = random.randint(25, 55)
        shield_dur = random.randint(2, 5)
        length = random.randint(3, 7)
        width = random.randint(1, 2)
        cd = random.randint(8, 18)
        threshold = random.randint(20, 50)
        results.append(f"""ability HoldoutSL{i} {{
    target: direction, range: {random.randint(3, 6)}.0
    cooldown: {cd}s, cast: {random.randint(0, 500)}ms
    hint: defense

    shield {shield_amt} for {shield_dur}s in line({length}.0, {width}.0) when caster_hp_below({threshold}%)
}}""")
    return results


def _projectile_multi_cc(n: int) -> list[str]:
    """Projectile with multiple CC types in on_hit — tests nesting composition."""
    results = []
    for i in range(n):
        dmg = random.randint(20, 50)
        speed = random.randint(6, 14)
        cc1_dur = random.randint(1, 2)
        cc2_dur = random.randint(1, 3)
        cd = random.randint(10, 20)
        cc_pairs = [("stun", "slow 0.4 for"), ("root", "silence"),
                     ("fear", "slow 0.3 for"), ("silence", "root")]
        cc1, cc2 = random.choice(cc_pairs)
        results.append(f"""ability HoldoutPM{i} {{
    target: enemy, range: {random.randint(4, 8)}.0
    cooldown: {cd}s, cast: {random.randint(300, 800)}ms
    hint: crowd_control

    deliver projectile {{ speed: {speed}.0 }} {{
        on_hit {{
    damage {dmg}
    {cc1} {cc1_dur}s
    {cc2} {cc2_dur}s
        }}
    }}
}}""")
    return results


# All holdout generators
HOLDOUT_GENERATORS = [
    ("tether_ring", _tether_ring),
    ("channel_or_cond", _channel_or_condition),
    ("recast_zone", _recast_zone),
    ("trap_cone", _trap_cone),
    ("chain_scaling", _chain_scaling),
    ("heal_spread_charges", _heal_spread_charges),
    ("shield_line_cond", _shield_line_conditional),
    ("projectile_multi_cc", _projectile_multi_cc),
]


def main():
    p = argparse.ArgumentParser(description="Create holdout structure set")
    p.add_argument("-o", "--output", default="training/data/holdout_structures.jsonl")
    p.add_argument("--hashes", default="training/data/holdout_hashes.txt")
    p.add_argument("--per-combo", type=int, default=60,
                   help="Abilities per combination type")
    args = p.parse_args()

    holdout = []
    hashes = []

    for name, gen_fn in HOLDOUT_GENERATORS:
        abilities = gen_fn(args.per_combo)
        print(f"  {name}: {len(abilities)} abilities")
        for dsl in abilities:
            h = _hash(dsl)
            holdout.append({"combo": name, "dsl": dsl, "hash": h})
            hashes.append(h)

    # Write JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in holdout:
            f.write(json.dumps(item) + "\n")

    # Write hashes
    hash_path = Path(args.hashes)
    hash_path.write_text("\n".join(hashes) + "\n")

    print(f"\nCreated {len(holdout)} holdout abilities ({len(HOLDOUT_GENERATORS)} combos)")
    print(f"  Structures: {out_path}")
    print(f"  Hashes:     {hash_path}")


if __name__ == "__main__":
    main()
