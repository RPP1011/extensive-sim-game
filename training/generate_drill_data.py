#!/usr/bin/env python3
"""Generate navigation training data from drill scenarios.

For each drill, computes the optimal movement direction at each step
(move toward target/away from threat) and records it as training data.

This doesn't run the sim — it reads the scenario config and computes
what direction the hero SHOULD move based on the drill objective.

Usage:
    uv run --with numpy python training/generate_drill_data.py \
        dataset/scenarios/drills/phase1/ \
        -o generated/v5_drill_nav.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import numpy as np


DIR_VECS = [
    (0.000, 1.000),   # 0: N
    (0.707, 0.707),   # 1: NE
    (1.000, 0.000),   # 2: E
    (0.707, -0.707),  # 3: SE
    (0.000, -1.000),  # 4: S
    (-0.707, -0.707), # 5: SW
    (-1.000, 0.000),  # 6: W
    (-0.707, 0.707),  # 7: NW
]


def angle_to_dir(dx: float, dy: float) -> int:
    """Convert a direction vector to the nearest of 8 cardinal directions."""
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return 8  # stay
    angle = math.atan2(dy, dx)
    # Normalize to [0, 2π)
    if angle < 0:
        angle += 2 * math.pi
    # Map to 8 dirs: E=0rad → dir 2, N=π/2 → dir 0, etc.
    # Our dirs: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    # atan2 gives: E=0, N=π/2, W=π, S=-π/2
    # Convert: shifted = (π/2 - angle) mod 2π, then divide by π/4
    shifted = (math.pi / 2 - angle) % (2 * math.pi)
    idx = int((shifted + math.pi / 8) / (math.pi / 4)) % 8
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="Drill scenario directories or files")
    p.add_argument("-o", "--output", default="generated/v5_drill_nav.jsonl")
    args = p.parse_args()

    # Collect all toml files
    toml_files = []
    for path in args.paths:
        path = Path(path)
        if path.is_file():
            toml_files.append(path)
        elif path.is_dir():
            toml_files.extend(sorted(path.rglob("*.toml")))

    print(f"Found {len(toml_files)} drill scenarios")

    episodes = []
    for tf in toml_files:
        with open(tf, "rb") as f:
            cfg = tomllib.load(f)

        scenario = cfg.get("scenario", {})
        drill_type = scenario.get("drill_type", "")
        target_pos = scenario.get("target_position")
        hero_positions = scenario.get("hero_positions", [])
        objective = scenario.get("objective", {})
        obj_type = objective.get("objective_type", "")
        obj_pos = objective.get("position")

        # Determine the target to move toward
        target = None
        if target_pos:
            target = target_pos
        elif obj_pos:
            target = obj_pos

        if not target or not hero_positions:
            continue

        # Simulate simple navigation: hero moves toward target
        hero_pos = list(hero_positions[0]) if hero_positions else [10.0, 10.0]
        target = list(target)
        move_speed = 3.2 * 0.1  # move_speed_per_sec * dt (100ms ticks)
        max_ticks = scenario.get("max_ticks", 200)

        steps = []
        for tick in range(0, max_ticks, 3):  # step_interval=3
            dx = target[0] - hero_pos[0]
            dy = target[1] - hero_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 1.0:
                # Reached target
                move_dir = 8  # stay
            else:
                move_dir = angle_to_dir(dx, dy)

            # Create a minimal step with entity features
            # Self entity at hero_pos, no enemies for reach drills
            self_feats = [0.0] * 34
            self_feats[0] = 1.0   # hp%
            self_feats[5] = hero_pos[0] / 20.0  # pos_x
            self_feats[6] = hero_pos[1] / 20.0  # pos_y
            self_feats[27] = 3.2 / 5.0  # move_speed
            self_feats[29] = 1.0  # exists

            steps.append({
                "tick": tick,
                "unit_id": 1,
                "game_state": [0.0] * 210,
                "action": 0,
                "log_prob": 0.0,
                "mask": [True] * 14,
                "step_reward": 0.0,
                "entities": [self_feats],
                "entity_types": [0],
                "threats": [],
                "positions": [],
                "move_dir": move_dir,
                "combat_type": 1,  # hold (no enemies)
                "target_idx": 0,
                "teacher_move_dir": move_dir,
                "teacher_combat_type": 1,
                "teacher_target_idx": 0,
                "aggregate_features": [0.0] * 16,
            })

            # Simulate movement
            if dist >= 1.0 and move_dir < 8:
                vx, vy = DIR_VECS[move_dir]
                hero_pos[0] += vx * move_speed
                hero_pos[1] += vy * move_speed

            if dist < 1.0:
                break

        if steps:
            outcome = "Victory" if math.sqrt(
                (hero_pos[0] - target[0])**2 + (hero_pos[1] - target[1])**2
            ) < 1.5 else "Timeout"
            episodes.append({
                "scenario": scenario.get("name", tf.stem),
                "outcome": outcome,
                "reward": 0.5 if outcome == "Victory" else 0.0,
                "ticks": steps[-1]["tick"] if steps else 0,
                "unit_abilities": {},
                "unit_ability_names": {},
                "steps": steps,
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    wins = sum(1 for ep in episodes if ep["outcome"] == "Victory")
    total_steps = sum(len(ep["steps"]) for ep in episodes)
    print(f"Generated {len(episodes)} episodes, {total_steps} steps")
    print(f"  Reach success: {wins}/{len(episodes)} ({100*wins/max(len(episodes),1):.0f}%)")
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
