# Ability Priority System — Design

## Problem

The current student model treats ability usage as one of 10 action classes competing in a single softmax. This has fundamental issues:

1. **No urgency modeling**: A CC on a casting healer is worth dropping everything for. The model can't express "use this ability RIGHT NOW" vs "maybe use it if nothing better."

2. **Post-decision targeting**: The model picks "use damage ability" then a separate heuristic picks the target. The ability choice and target choice should be coupled — a ground-targeted AoE on 3 clustered enemies is great, the same AoE on 1 isolated enemy is a waste.

3. **Forced generalization**: A heal's decision logic ("ally HP low, no HoT active") has nothing in common with a CC's decision logic ("enemy healer casting, no immunity"). Forcing them through shared weights hurts both.

4. **No positional targeting**: Ground-targeted and direction abilities need WHERE to place them, not just WHO to hit. The current system can't express "drop AoE between these 3 enemies."

## Architecture: Interrupt-Driven Ability Priority

```
Each tick, for each hero unit:

1. ABILITY INTERRUPT CHECK (runs first)
   For each ability that is ready (off cooldown, affordable):
       urgency, target = ability_evaluator(game_state, ability_def)

   If max(urgency) > URGENCY_THRESHOLD:
       USE that ability on that target — interrupt any other plan

2. BASIC COMBAT (fallback)
   If no ability triggered:
       action = combat_model(game_state) → attack/move/hold
```

The key insight: abilities are **interrupts**, not options. When a perfect CC window opens, the unit shouldn't finish its auto-attack animation then maybe consider CCing — it should fire immediately.

## Ability Evaluator

One small model per ability CATEGORY (not per individual ability — categories share enough structure):

### Categories by targeting type

| Category | Targeting | Output | Examples |
|----------|-----------|--------|----------|
| **damage_unit** | TargetEnemy | urgency + enemy_id | Single-target nukes, executes |
| **damage_aoe** | SelfAoe, GroundTarget | urgency + position | AoE damage, ground slams |
| **cc_unit** | TargetEnemy | urgency + enemy_id | Stuns, roots, silences |
| **heal_unit** | TargetAlly | urgency + ally_id | Single-target heals |
| **heal_aoe** | SelfAoe, SelfCast | urgency | AoE heals, self-heals |
| **defense** | SelfCast, TargetAlly | urgency [+ ally_id] | Shields, damage reduction |
| **utility** | Various | urgency [+ target] | Dashes, buffs, summons |

### Evaluator inputs (per category)

**damage_unit evaluator** (~20 features):
- Self: HP%, resource, position
- Ability: damage estimate, range, cooldown_ms, cast_time
- Per candidate target (top 3 enemies): HP%, dist, is_focus, effective_HP, DPS, is_casting, is_healer
- Context: can_kill_target (damage > HP), overkill (ally already attacking), numeric_advantage

**damage_aoe evaluator** (~15 features):
- Self: HP%, resource, position
- Ability: damage, range, radius
- Spatial: enemies_in_radius at each candidate position, ally count in radius
- Candidate positions: enemy centroid, densest cluster center, focus target position

**cc_unit evaluator** (~18 features):
- Self: position
- Ability: CC duration, range, cast_time
- Per candidate target: DPS, is_healer, is_casting, is_cc_immune, current_cc_remaining, HP% (don't CC a dying unit), is_focus_target
- Context: team_pressure (high pressure = CC more urgent), ally_HP_critical

**heal_unit evaluator** (~12 features):
- Self: position, is_in_danger
- Ability: heal_amount, range, cast_time, has_HoT
- Per candidate ally: HP%, dist, incoming_damage, has_HoT_active, is_tank (prioritize frontline)
- Context: team_avg_HP, enemies_alive

**heal_aoe / defense / utility** — similar structure, simpler

### Evaluator outputs

Each evaluator outputs:
- `urgency: f32` in [0, 1] — how critical is using this ability RIGHT NOW
- `target: AbilityTarget` — Unit(id), Position(x,y), or None (self-cast)

**Urgency semantics:**
- 0.0 = don't use (bad timing, no good target, ability wouldn't help)
- 0.3 = nice to have (minor damage, topped-up ally)
- 0.6 = clearly good (low-HP enemy in range, hurt ally needs heal)
- 0.9 = CRITICAL (can kill enemy healer, ally about to die, perfect CC window)
- 1.0 = guaranteed value (execute on 1-HP enemy, interrupt on key cast)

**Threshold:** URGENCY_THRESHOLD = 0.4 (below this, just auto-attack)

## Oracle Changes for Training Data

### New oracle mode: per-ability evaluation

Instead of "what's the best action overall?", the oracle answers per-ability:

```
For each ability that's ready:
    score_with_ability = rollout(state, unit, USE_ABILITY(best_target))
    score_without      = rollout(state, unit, BEST_NON_ABILITY_ACTION)

    ability_value = score_with_ability - score_without
    urgency = sigmoid(ability_value / SCALE)
```

This directly measures "how much better is using this ability vs not using it?" — the definition of urgency.

For targeting, the oracle already tries each valid target and picks the best. We record which target the oracle chose.

### Training sample format

```json
{
  "ability_category": "cc_unit",
  "features": [...],           // category-specific features
  "urgency": 0.87,             // regression target
  "target_id": 5,              // classification target (which enemy/ally)
  "target_pos": [3.2, 1.5],   // for ground-targeted
  "ability_hint": "crowd_control",
  "scenario": "balanced_classic_4v4"
}
```

### Training

Two losses per evaluator:
- **Urgency**: MSE loss on urgency score (regression)
- **Targeting**: Cross-entropy on target selection (classification among valid targets)

Combined: `loss = urgency_loss + lambda * target_loss`

## Candidate Position Generation (for ground-targeted)

For GroundTarget/Direction abilities, we need to generate candidate positions:

1. **Enemy centroid**: Mean position of all enemies in range
2. **Dense cluster**: Position that maximizes enemies within ability radius (greedy)
3. **Per-enemy positions**: Each enemy's exact position (single-target ground abilities)
4. **Predicted positions**: Where enemies will be in cast_time_ms (position + velocity)
5. **Cut-off positions**: Between enemies and their retreat path

For the oracle, rollout each candidate position and score by total damage dealt.

For the evaluator, output is a position regression (x, y) rather than classification.

## Integration with Existing System

### Phase 1: Train evaluators alongside current model
- Keep the 10-class combat model for attack/move/hold decisions
- Add ability evaluators as an interrupt layer on top
- If any evaluator fires (urgency > threshold), use its decision
- Otherwise fall through to combat model

### Phase 2: Simplify combat model
- Remove ability classes (3-6) from the combat model — evaluators handle those
- Combat model becomes: AttackNearest, AttackWeakest, MoveToward, MoveAway, Hold (5 classes)
- Much simpler classification problem, should get higher accuracy

### Runtime architecture

```
tick:
  for each hero unit:
    // Phase 1: Ability interrupts
    best_urgency = 0
    best_action = None
    for each ready ability:
      category = categorize(ability)
      features = extract_category_features(state, unit, ability)
      urgency, target = evaluator[category].predict(features)
      if urgency > best_urgency:
        best_urgency = urgency
        best_action = UseAbility(ability_index, target)

    if best_urgency > URGENCY_THRESHOLD:
      emit(best_action)
    else:
      // Phase 2: Basic combat
      features = extract_combat_features(state, unit)
      action_class = combat_model.predict(features)
      emit(action_class_to_intent(action_class))
```

### Rust inference cost
- Each evaluator: ~5K params, <20 features → trivial forward pass
- 8 ability slots × 1 evaluator each = 8 tiny forward passes per unit per tick
- Combat model: 1 forward pass (smaller than current, 5 classes instead of 10)
- Total: faster than current single 71K-param model

## Files to create/modify

| File | Action | Purpose |
|------|--------|---------|
| `src/ai/core/ability_eval.rs` | NEW | Evaluator inference + feature extraction |
| `src/ai/core/oracle.rs` | EDIT | Add per-ability scoring mode |
| `src/ai/core/dataset.rs` | EDIT | Add ability-specific sample generation |
| `scripts/train_ability_eval.py` | NEW | Train per-category evaluators |
| `src/bin/xtask/oracle_cmd.rs` | EDIT | CLI for ability eval dataset/training |

## Model sizes

| Evaluator | Input features | Architecture | Params |
|-----------|---------------|--------------|--------|
| damage_unit | ~20 | 20→32→16→2 | ~1.2K |
| damage_aoe | ~15 | 15→32→16→3 | ~1.0K |
| cc_unit | ~18 | 18→32→16→2 | ~1.1K |
| heal_unit | ~12 | 12→24→12→2 | ~600 |
| heal_aoe | ~8 | 8→16→8→1 | ~300 |
| defense | ~10 | 10→16→8→2 | ~400 |
| utility | ~12 | 12→24→12→2 | ~600 |
| **combat (fallback)** | ~50 | 50→128→64→5 | ~13K |
| **Total** | | | **~18K** |

18K total params vs 71K for the current monolithic model — smaller, faster, and more capable because each sub-model is specialized.
