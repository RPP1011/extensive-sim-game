# Pointer-Based Action Space + Terrain-Aware Features

## Problem Statement

The current 14-action discrete space has fundamental limitations:
1. **No target selection** — attack always picks nearest/weakest/focus via heuristics
2. **No zone avoidance** — `move_away` only moves from nearest enemy, not from threats
3. **No positional movement** — cannot move to cover, elevated ground, or specific positions
4. **No terrain awareness** — cover_bonus and elevation are always 0 in training (headless sim)
5. **Ground-target abilities** always aim at enemy centroid, no strategic placement

## Design Overview

Replace the flat action space with a **hierarchical pointer architecture**:

```
Action = (action_type, target_pointer)
```

- **action_type**: what to do (attack, ability_0..7, move, hold)
- **target_pointer**: scaled dot-product attention over entity/position tokens to select a target

The entity encoder sequence is extended with **position tokens** (type=4) representing areas of interest the agent can move to or target with ground abilities.

## Architecture Changes

### 1. Extended Entity Sequence

Current token types:
| Type | ID | Features | Count |
|------|-----|----------|-------|
| Self | 0 | 30-dim | 1 |
| Enemy | 1 | 30-dim | variable |
| Ally | 2 | 30-dim | variable |
| Threat | 3 | 8-dim | variable |

New:
| Type | ID | Features | Count |
|------|-----|----------|-------|
| Self | 0 | 30-dim | 1 |
| Enemy | 1 | 30-dim | variable |
| Ally | 2 | 30-dim | variable |
| Threat | 3 | 8-dim | variable |
| **Position** | **4** | **8-dim** | **up to 8** |

### 2. Position Token Features (8-dim)

Only intrinsic spatial properties — no enemy-relative features (attention learns relationships):

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | dx from self | /20 |
| 1 | dy from self | /20 |
| 2 | path distance from self | /30 (divergence from euclidean reveals walls) |
| 3 | elevation | /5 |
| 4 | chokepoint_score | /3 (blocked cardinal neighbors) |
| 5 | wall_proximity | /5 (min raycast distance to nearest wall) |
| 6 | n_hostile_zones | /3 |
| 7 | n_friendly_zones | /3 |
No `exists` feature — the attention mask handles padding. No enemy-relative features
(cover, LOS, distance-to-enemies) — self-attention between position tokens and
entity tokens learns those relationships from dx/dy already present on both.

### 3. Position Token Generation (Rust, each tick)

**Implemented in** `src/ai/core/ability_eval/game_state.rs::extract_position_tokens()`.

Samples 8 directions × 3 distances (0.5x, 1.0x, 1.5x move range), filters to walkable,
deduplicates within 2.0, sorts by tactical score (elevation + chokepoint), takes top 8.
Returns empty Vec when no GridNav is present (graceful degradation).

### 4. Pointer Action Head

**Implemented in** `training/model.py` as `PointerHead` and `AbilityActorCriticV3`.

- `PointerHead`: action_type_head (11 logits), shared pointer_key projection,
  per-type query projections (attack_query, move_query, ability_queries[0..7])
- `AbilityActorCriticV3`: EntityEncoderV3 + CrossAttentionBlock + PointerHead + value_head
- 109K params total (up from 97K with V2)

Target masks (built into PointerHead):
- Attack: only enemy tokens (type=1)
- Move: everything except self (type=0)
- Ability: all non-padding (external per-ability targeting masks applied in training)
- Hold: no pointer needed

Key insight: **move + threat pointer = zone avoidance**. Selecting a threat token as
move target → "move away from this threat." Solves zone avoidance structurally.

## Implementation Status

### Done
- [x] Position token extraction in Rust (`game_state.rs::extract_position_tokens`)
- [x] GameStateV2 extended with `positions` field
- [x] OutcomeSampleV2 extended with `positions` field
- [x] EntityEncoderV3 with position projection (type=4)
- [x] PointerHead with action type + pointer distributions
- [x] AbilityActorCriticV3 model class (109K params)
- [x] Forward pass verified (Python: masking correct)
- [x] Rust inference: FlatEntityEncoderV3 + ActorCriticWeightsV3 in `weights.rs`
- [x] Rust pointer head: FlatPointerHead with PointerOutput
- [x] Pointer action-to-intent in `actions.rs` (TokenInfo + pointer_action_to_intent)
- [x] Export script (`training/export_actor_critic_v3.py`)
- [x] Round-trip test: Python → JSON → Rust (masking verified)

### Remaining
- [x] GridNav injection in episode generation (`transformer_rl.rs`)
- [x] Episode generation with V3 policy (`Policy::ActorCriticV3`)
- [x] PPO training script (`train_rl_v3.py`) with hierarchical log probs
- [ ] Entity encoder V5 pretraining with position tokens
- [ ] BC warmstart from oracle data
