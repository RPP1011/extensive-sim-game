#!/usr/bin/env python3
"""Convert V5 episode JSONL to npz for fast training data loading.

Extracts per-step samples with dense targets:
  - hp_advantage: mean(hero HP%) - mean(enemy HP%)
  - survival_ratio: heroes_alive / total_alive

Usage:
    uv run --with numpy python training/convert_v5_npz.py \
        generated/v5_stage0a_random.jsonl \
        generated/v5_stage0a_combined.jsonl \
        -o generated/v5_stage0a.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MAX_ENTITIES = 20
MAX_THREATS = 6
MAX_POSITIONS = 8
ENTITY_DIM = 34
THREAT_DIM = 10
POSITION_DIM = 8
AGG_DIM = 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", nargs="+", help="Episode JSONL files")
    p.add_argument("-o", "--output", default="generated/v5_stage0a.npz")
    p.add_argument("--val-split", type=float, default=0.1)
    args = p.parse_args()

    # Load all episodes
    episodes = []
    for path in args.data:
        print(f"Loading {path}...")
        with open(path) as f:
            for line in f:
                episodes.append(json.loads(line))
    print(f"  {len(episodes)} episodes total")

    # Extract samples
    ent_list = []
    ent_type_list = []
    ent_mask_list = []
    thr_list = []
    thr_mask_list = []
    pos_list = []
    pos_mask_list = []
    agg_list = []
    hp_adv_list = []
    surv_list = []
    move_dir_list = []
    move_vec_list = []  # continuous (dx, dy, speed) ground truth
    combat_type_list = []
    target_idx_list = []
    target_move_pos_list = []  # continuous movement target (x, y)
    teacher_move_list = []   # DAgger teacher labels
    teacher_combat_list = []
    teacher_target_list = []
    has_teacher = False
    ep_idx_list = []  # track which episode each sample came from

    # Direction vectors matching Rust move_dir_offset()
    _DIR_VECS = [
        ( 0.000,  1.000),  # 0: N
        ( 0.707,  0.707),  # 1: NE
        ( 1.000,  0.000),  # 2: E
        ( 0.707, -0.707),  # 3: SE
        ( 0.000, -1.000),  # 4: S
        (-0.707, -0.707),  # 5: SW
        (-1.000,  0.000),  # 6: W
        (-0.707,  0.707),  # 7: NW
        ( 0.000,  0.000),  # 8: stay
    ]

    for epi, ep in enumerate(episodes):
        # Compute FINAL state HP advantage and survival from last step
        final_step = ep["steps"][-1] if ep["steps"] else None
        if not final_step or not final_step.get("entities"):
            continue

        final_types = final_step.get("entity_types", [])
        final_ents = final_step["entities"]
        final_hero_hp = 0.0
        final_hero_alive = 0
        final_enemy_hp = 0.0
        final_enemy_alive = 0
        for ei, etype in enumerate(final_types):
            if ei >= len(final_ents):
                break
            e = final_ents[ei]
            if len(e) <= 29 or e[29] < 0.5:
                continue
            if etype in (0, 2):
                final_hero_hp += e[0]
                final_hero_alive += 1
            elif etype == 1:
                final_enemy_hp += e[0]
                final_enemy_alive += 1

        if final_hero_alive + final_enemy_alive == 0:
            continue

        # Targets: predict the FINAL state from any mid-fight snapshot
        # This requires the model to extrapolate, not just read current HP
        final_hp_advantage = (final_hero_hp / max(final_hero_alive, 1)
                              - final_enemy_hp / max(final_enemy_alive, 1))
        final_survival = final_hero_alive / (final_hero_alive + final_enemy_alive)

        for step in ep["steps"]:
            entities = step.get("entities")
            if not entities:
                continue

            entity_types = step.get("entity_types", [0] * len(entities))

            # Verify at least some units exist
            any_alive = False
            for ei, etype in enumerate(entity_types):
                if ei < len(entities) and len(entities[ei]) > 29 and entities[ei][29] > 0.5:
                    any_alive = True
                    break
            if not any_alive:
                continue

            hp_advantage = final_hp_advantage
            survival_ratio = final_survival

            # Pad entities
            ent = np.zeros((MAX_ENTITIES, ENTITY_DIM), dtype=np.float32)
            et = np.zeros(MAX_ENTITIES, dtype=np.int32)
            em = np.ones(MAX_ENTITIES, dtype=np.bool_)  # True = padded
            for j in range(min(len(entities), MAX_ENTITIES)):
                e = entities[j]
                ent[j, :min(len(e), ENTITY_DIM)] = e[:ENTITY_DIM]
                em[j] = False
            for j in range(min(len(entity_types), MAX_ENTITIES)):
                et[j] = entity_types[j]

            # Pad threats
            threats = step.get("threats", [])
            thr = np.zeros((MAX_THREATS, THREAT_DIM), dtype=np.float32)
            tm = np.ones(MAX_THREATS, dtype=np.bool_)
            for j in range(min(len(threats), MAX_THREATS)):
                t = threats[j]
                thr[j, :min(len(t), THREAT_DIM)] = t[:THREAT_DIM]
                tm[j] = False

            # Pad positions
            positions = step.get("positions", [])
            pos = np.zeros((MAX_POSITIONS, POSITION_DIM), dtype=np.float32)
            pm = np.ones(MAX_POSITIONS, dtype=np.bool_)
            for j in range(min(len(positions), MAX_POSITIONS)):
                pp = positions[j]
                pos[j, :min(len(pp), POSITION_DIM)] = pp[:POSITION_DIM]
                pm[j] = False

            # Aggregate
            agg_raw = step.get("aggregate_features", [0.0] * AGG_DIM)
            agg = np.zeros(AGG_DIM, dtype=np.float32)
            if agg_raw:
                agg[:min(len(agg_raw), AGG_DIM)] = agg_raw[:AGG_DIM]

            # Action labels (for BC training)
            move_dir = step.get("move_dir", 8)  # 8 = stay
            combat_type = step.get("combat_type", 1)  # 1 = hold
            target_idx = step.get("target_idx", 0)

            # Continuous movement vector from discrete dir
            dx, dy = _DIR_VECS[min(move_dir, 8)]
            speed = 0.0 if move_dir >= 8 else 1.0
            move_vec = (dx, dy, speed)

            # Target movement position (if recorded)
            tmp = step.get("target_move_pos")
            if tmp is not None:
                target_move_pos_list.append(tmp)
            else:
                # Fallback: compute from move_dir + unit position
                self_pos_x = entities[0][5] * 20.0 if entities else 0.0
                self_pos_y = entities[0][6] * 20.0 if entities else 0.0
                if move_dir < 8:
                    dx_d, dy_d = _DIR_VECS[move_dir]
                    target_move_pos_list.append([self_pos_x + dx_d * 3.2 * 0.3, self_pos_y + dy_d * 3.2 * 0.3])
                else:
                    target_move_pos_list.append([self_pos_x, self_pos_y])

            # DAgger teacher labels (if present)
            t_move = step.get("teacher_move_dir", -1)
            t_combat = step.get("teacher_combat_type", -1)
            t_target = step.get("teacher_target_idx", -1)
            if t_move >= 0:
                has_teacher = True

            ent_list.append(ent)
            ent_type_list.append(et)
            ent_mask_list.append(em)
            thr_list.append(thr)
            thr_mask_list.append(tm)
            pos_list.append(pos)
            pos_mask_list.append(pm)
            agg_list.append(agg)
            hp_adv_list.append(hp_advantage)
            surv_list.append(survival_ratio)
            move_dir_list.append(move_dir)
            move_vec_list.append(move_vec)
            combat_type_list.append(combat_type)
            target_idx_list.append(target_idx)
            teacher_move_list.append(t_move if t_move >= 0 else move_dir)
            teacher_combat_list.append(t_combat if t_combat >= 0 else combat_type)
            teacher_target_list.append(t_target if t_target >= 0 else target_idx)
            ep_idx_list.append(epi)

    N = len(ent_list)
    print(f"  {N} samples extracted")

    # Stack into arrays
    ent_feat = np.stack(ent_list)        # (N, 20, 34)
    ent_types = np.stack(ent_type_list)  # (N, 20)
    ent_mask = np.stack(ent_mask_list)   # (N, 20)
    thr_feat = np.stack(thr_list)        # (N, 6, 10)
    thr_mask = np.stack(thr_mask_list)   # (N, 6)
    pos_feat = np.stack(pos_list)        # (N, 8, 8)
    pos_mask = np.stack(pos_mask_list)   # (N, 8)
    agg_feat = np.stack(agg_list)        # (N, 16)
    hp_adv = np.array(hp_adv_list, dtype=np.float32)      # (N,)
    surv = np.array(surv_list, dtype=np.float32)          # (N,)
    move_dir = np.array(move_dir_list, dtype=np.int32)        # (N,)
    move_vec = np.array(move_vec_list, dtype=np.float32)      # (N, 3) — dx, dy, speed
    combat_type = np.array(combat_type_list, dtype=np.int32)  # (N,)
    target_idx = np.array(target_idx_list, dtype=np.int32)    # (N,)
    target_move_pos = np.array(target_move_pos_list, dtype=np.float32)  # (N, 2)
    teacher_move = np.array(teacher_move_list, dtype=np.int32)      # (N,)
    teacher_combat = np.array(teacher_combat_list, dtype=np.int32)  # (N,)
    teacher_target = np.array(teacher_target_list, dtype=np.int32)  # (N,)
    ep_idx = np.array(ep_idx_list, dtype=np.int32)        # (N,)

    # Split by episode index (not by sample) to prevent leakage
    n_episodes = len(episodes)
    perm = np.random.RandomState(42).permutation(n_episodes)
    n_val_ep = max(1, int(n_episodes * args.val_split))
    val_ep_set = set(perm[:n_val_ep].tolist())

    val_mask_arr = np.array([ep_idx[i] in val_ep_set for i in range(N)])
    train_idx = np.where(~val_mask_arr)[0]
    val_idx = np.where(val_mask_arr)[0]

    print(f"  Train: {len(train_idx)} samples ({len(perm) - n_val_ep} episodes)")
    print(f"  Val:   {len(val_idx)} samples ({n_val_ep} episodes)")
    print(f"  hp_advantage: mean={hp_adv.mean():.3f} std={hp_adv.std():.3f} unique={len(np.unique(np.round(hp_adv, 4)))}")
    print(f"  survival:     mean={surv.mean():.3f} std={surv.std():.3f} unique={len(np.unique(np.round(surv, 4)))}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ent_feat=ent_feat, ent_types=ent_types, ent_mask=ent_mask,
        thr_feat=thr_feat, thr_mask=thr_mask,
        pos_feat=pos_feat, pos_mask=pos_mask,
        agg_feat=agg_feat,
        hp_adv=hp_adv, surv=surv,
        move_dir=move_dir, move_vec=move_vec, combat_type=combat_type, target_idx=target_idx,
        target_move_pos=target_move_pos,
        teacher_move=teacher_move, teacher_combat=teacher_combat, teacher_target=teacher_target,
        ep_idx=ep_idx, train_idx=train_idx, val_idx=val_idx,
    )
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Saved {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
