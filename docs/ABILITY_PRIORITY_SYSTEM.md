# Ability Priority System

How the AI decides which ability to use and when. Covers the interrupt-driven evaluation system, squad-level ability scoring, and the integration between them.

---

## Table of Contents

1. [Overview](#overview)
2. [Interrupt-Driven Ability Evaluation](#interrupt-driven-ability-evaluation)
3. [Ability Categories](#ability-categories)
4. [Feature Extraction](#feature-extraction)
5. [Neural Network Evaluation](#neural-network-evaluation)
6. [Post-Prediction Modifiers](#post-prediction-modifiers)
7. [Squad Combat Ability Scoring](#squad-combat-ability-scoring)
8. [Integration with Squad AI](#integration-with-squad-ai)
9. [Oracle Training Pipeline](#oracle-training-pipeline)
10. [Key Files](#key-files)

---

## Overview

The ability priority system uses a two-layer architecture:

```
Each tick, for each hero unit:

1. ABILITY INTERRUPT CHECK (runs first)
   For each ability that is ready (off cooldown, affordable):
       urgency, target = ability_evaluator(game_state, ability_def)

   If max(urgency) > URGENCY_THRESHOLD:
       USE that ability on that target — interrupt any other plan

2. SQUAD COMBAT SCORING (fallback)
   If no ability triggered above threshold:
       Score abilities via threat-reduction heuristics
       Fall through to force-based movement/attack
```

The key insight: abilities are **interrupts**, not options. When a perfect CC window opens, the unit should fire immediately rather than finishing its current auto-attack.

---

## Interrupt-Driven Ability Evaluation

**File:** `src/ai/core/ability_eval/eval.rs`

The primary entry point is `evaluate_abilities()`:

```rust
pub fn evaluate_abilities(
    state: &SimState,
    squad_ai: &SquadAiState,
    unit_id: u32,
    weights: &AbilityEvalWeights,
) -> Option<(IntentAction, f32)>
```

For each ready ability on the unit:
1. Classify the ability into a category
2. Extract category-specific features
3. Run the neural network for that category
4. Apply post-prediction modifiers (heal saturation, cleanup boost)
5. Return the highest-urgency ability above `URGENCY_THRESHOLD`

An enhanced version `evaluate_abilities_with_encoder()` accepts an optional `AbilityEncoder` for enriched feature extraction using ability embeddings.

### Urgency Semantics

Urgency is a float in `[0, 1]`:

| Score | Meaning |
|-------|---------|
| 0.0 | Don't use (bad timing, no good target) |
| 0.3 | Nice to have (minor damage, topped-up ally) |
| 0.6 | Clearly good (low-HP enemy in range, hurt ally needs heal) |
| 0.9 | Critical (can kill enemy healer, ally about to die, perfect CC window) |
| 1.0 | Guaranteed value (execute on 1-HP enemy, interrupt on key cast) |

**Threshold:** `URGENCY_THRESHOLD = 0.4` — below this, the unit falls through to normal combat behavior.

---

## Ability Categories

**File:** `src/ai/core/ability_eval/categories.rs`

9 categories classify abilities by function:

```rust
pub enum AbilityCategory {
    DamageUnit,   // Single-target damage (TargetEnemy + damage hint)
    DamageAoe,    // AoE damage (SelfAoe/GroundTarget + damage hint)
    CcUnit,       // Crowd control (TargetEnemy + crowd_control hint)
    HealUnit,     // Single-target heal (TargetAlly + heal hint)
    HealAoe,      // AoE/self heal (SelfCast/SelfAoe + heal hint)
    Defense,      // Shields, damage reduction (defense hint)
    Utility,      // Dashes, buffs (utility hint)
    Summon,       // Creates ally units (summon effects)
    Obstacle,     // Terrain/wall creation (obstacle effects)
}
```

Classification logic in `from_ability_full()`:
1. Check effects for `Summon` or `Obstacle` — these override `ai_hint`
2. Also check delivery `on_hit`/`on_arrival` for summon/obstacle effects
3. Fall back to `ai_hint` + `targeting` for category selection

### Category → Evaluator Inputs

| Category | Targeting | Output | Examples |
|----------|-----------|--------|----------|
| **damage_unit** | TargetEnemy | urgency + enemy_id | Single-target nukes, executes |
| **damage_aoe** | SelfAoe, GroundTarget | urgency + position | AoE damage, ground slams |
| **cc_unit** | TargetEnemy | urgency + enemy_id | Stuns, roots, silences |
| **heal_unit** | TargetAlly | urgency + ally_id | Single-target heals |
| **heal_aoe** | SelfAoe, SelfCast | urgency | AoE heals, self-heals |
| **defense** | SelfCast, TargetAlly | urgency [+ ally_id] | Shields, damage reduction |
| **utility** | Various | urgency [+ target] | Dashes, buffs, summons |
| **summon** | Various | urgency + position | Directed/autonomous summons |
| **obstacle** | GroundTarget | urgency + position | Walls, terrain |

---

## Feature Extraction

**Files:** `src/ai/core/ability_eval/features.rs`, `features_aoe.rs`

Category-specific feature extractors produce ~20 features each. Helper functions:

- `unit_dps(unit)` — `attack_damage / (attack_cooldown_ms / 1000)`
- `is_healer(unit)` — checks for heal abilities or `heal_amount > 0`
- `terrain_features(state, pos)` — `[cover_bonus, elevation, hostile_zones, friendly_zones]`
- `team_healing_context(state, team)` — healing saturation analysis for dampening heal urgency

### Per-Category Feature Summary

**damage_unit** (~20 features):
- Self: HP%, resource, position
- Ability: damage estimate, range, cooldown_ms, cast_time
- Per candidate target (top 3 enemies): HP%, dist, is_focus, effective_HP, DPS, is_casting, is_healer
- Context: can_kill_target, overkill (ally already attacking), numeric_advantage

**damage_aoe** (~15 features):
- Self: HP%, resource, position
- Ability: damage, range, radius
- Spatial: enemies_in_radius at candidate positions, ally count in radius
- Candidate positions: enemy centroid, densest cluster center, focus target position

**cc_unit** (~18 features):
- Self: position
- Ability: CC duration, range, cast_time
- Per candidate target: DPS, is_healer, is_casting, is_cc_immune, current_cc_remaining, HP%, is_focus_target
- Context: team_pressure, ally_HP_critical

**heal_unit** (~12 features):
- Self: position, is_in_danger
- Ability: heal_amount, range, cast_time, has_HoT
- Per candidate ally: HP%, dist, incoming_damage, has_HoT_active, is_tank
- Context: team_avg_HP, enemies_alive

**heal_aoe / defense / utility / summon / obstacle** — similar structure, simpler

---

## Neural Network Evaluation

**File:** `src/ai/core/ability_eval/weights.rs`

Each category has its own tiny neural network:

```rust
pub struct EvalWeights {
    layers: Vec<(Vec<Vec<f32>>, Vec<f32>)>, // weight matrices + bias vectors
}
```

Forward pass: ReLU activation for hidden layers, linear output. `sigmoid(output[0])` produces the urgency score.

```rust
pub struct AbilityEvalWeights {
    evaluators: HashMap<AbilityCategory, EvalWeights>,
}
```

Weights are loaded from JSON at startup. Multiple weight versions exist:
- `generated/ability_eval_weights.json` (main)
- `ability_eval_weights_v2.json`, `v3_embedded.json`, `v4_healctx.json` (variants)

### Model Sizes

| Evaluator | Input features | Architecture | Params |
|-----------|---------------|--------------|--------|
| damage_unit | ~20 | 20→32→16→2 | ~1.2K |
| damage_aoe | ~15 | 15→32→16→3 | ~1.0K |
| cc_unit | ~18 | 18→32→16→2 | ~1.1K |
| heal_unit | ~12 | 12→24→12→2 | ~600 |
| heal_aoe | ~8 | 8→16→8→1 | ~300 |
| defense | ~10 | 10→16→8→2 | ~400 |
| utility | ~12 | 12→24→12→2 | ~600 |
| **Total** | | | **~5.2K** |

Each forward pass is trivial. 8 ability slots × 1 evaluator each = 8 tiny forward passes per unit per tick.

---

## Post-Prediction Modifiers

After neural network prediction, three modifiers adjust urgency:

### Heal Saturation

`apply_heal_saturation()` — dampens heal urgency when the team is healthy and has multiple healers. Prevents overhealing in compositions with redundant healing.

### Cleanup Boost

`apply_cleanup_boost()` — boosts offensive ability urgency when the team has 2:1+ numeric advantage in late game (tick > 2000). Encourages aggressive play when winning.

### Cleanup Suppress

`apply_cleanup_suppress()` — zeros out healing and defense ability urgency in ultra-late game (tick > 5000). Prevents stalling when the fight should be decided.

---

## Squad Combat Ability Scoring

**File:** `src/ai/squad/combat/abilities.rs`

When neural network evaluators aren't loaded (or as a fallback), the squad AI uses heuristic scoring:

```rust
pub fn evaluate_hero_ability(
    state: &SimState, unit_id: u32, target_id: u32,
    mode: &FormationMode, ctx: &TickContext,
) -> Option<IntentAction>
```

### Threat-Based Scoring

1. **Threat Calculation:**
   - `target_dps = target.attack_damage / (target.attack_cooldown_ms / 1000)`
   - Conditional bonus damage from abilities with met conditions (stun, slow)
   - Whether the unit has CC ready

2. **Per-Ability Effect Analysis:**
   - Detects: AoE, shields, buffs, debuffs, dashes
   - Counts AoE targets hit
   - Evaluates conditional effects

3. **Score = threat reduction + damage + bonuses:**
   ```
   total_damage = (unconditional + met_conditional) × aoe_targets
   cc_threat_reduction = cc_duration × target_dps + setup_value
   kill_bonus = if would_kill { target_dps × 10.0 } else { 0.0 }
   ```

### Scoring by AI Hint

| Hint | Base Formula | Notes |
|------|-------------|-------|
| `"crowd_control"` | `cc_threat_reduction + total_damage` | Skipped if target already CC'd or in Retreat mode |
| `"damage"` / `"opener"` | `total_damage + kill_bonus + opener_bonus + cc_threat_reduction` | **Deferral logic**: defers conditional damage abilities if CC is ready (use Garrote before Backstab) |
| `"defense"` | HP%-scaled: 10.0 if HP < 35%, 7.0 if HP < 60%, 1.5 otherwise | Multiplied by nearby allies if AoE shield/buff |
| `"heal"` | 8.0 if HP < 40%, 4.0 if HP < 60%, 0.0 otherwise | |
| `"utility"` | 3.0 + buffs/debuffs/dashes + threat reduction | |

### Zone Reaction Bonus

Checks for compatible existing zones from the same caster. Fire-Frost, Fire-Lightning, Frost-Lightning combos get a +25.0 score bonus.

### Cooldown Penalty

Long cooldown abilities (>12s) with low score (<8.0) are penalized by ×0.7 to avoid wasting big cooldowns.

---

## Integration with Squad AI

**File:** `src/ai/squad/intents.rs`

The ability evaluation integrates into the per-unit decision loop:

```
Per unit per tick:
1. Leash check — enforce anchor position max distance
2. Ability interrupt — if eval_weights loaded, run evaluate_abilities()
   → If urgency > threshold, emit that ability as the intent
3. Force calculation:
   a. compute_raw_forces() — situation analysis
   b. weighted_forces() — apply personality modifiers (7 traits × 9 forces)
   c. dominant_force() — pick primary tactic
4. Dominant force handling:
   → Heal, Protect, Attack, Retreat, etc.
   → Within Attack/Focus, call evaluate_hero_ability() for heuristic scoring
```

### Personality Influence on Forces

The personality system (7 traits) drives 9 tactical forces via a weight matrix:

```
          Attack Heal Retreat Control Focus Protect Pursue Regroup Position
Aggression [ 0.8   0.0  -0.2    0.0    0.2    0.0    0.6   -0.1    0.0]
Compassion [ 0.0   0.9   0.1    0.0    0.0    0.8    0.0    0.3    0.0]
Caution   [-0.2   0.2   0.8    0.1    0.0    0.0   -0.2    0.5    0.4]
Discipline[ 0.1   0.0   0.0    0.0    0.9    0.2    0.0    0.5    0.2]
Cunning   [ 0.2   0.0   0.0    0.8    0.2    0.0    0.3    0.0    0.5]
Tenacity  [ 0.3   0.0  -0.3    0.0    0.4    0.0    0.8   -0.1    0.0]
Patience  [-0.2   0.1   0.0    0.3    0.0    0.1   -0.1    0.2    0.3]
```

Heal wins ties in force selection.

### Target Selection

**File:** `src/ai/squad/combat/targeting.rs`

```rust
fn target_score(state, unit, personality, target_id, focus, ctx) -> f32 {
    hp_factor = (target.max_hp - target.hp) × 0.2
    focus_bonus = if focus == target_id { personality.discipline × 10.0 } else { 0.0 }
    dist_bias = -dist × (0.5 + (1.0 - personality.aggression) × 1.5)
    combo_bonus = if target.control_remaining_ms > 0 { 15.0 } else { 0.0 }

    hp_factor + focus_bonus + dist_bias + combo_bonus
}
```

Sticky target lock prevents retargeting for N ticks (default 4).

---

## Oracle Training Pipeline

### Per-Ability Oracle Scoring

**File:** `src/ai/core/ability_eval/oracle_scoring.rs`

Instead of "what's the best action overall?", the oracle answers per-ability:

```
For each ability that's ready:
    score_with_ability = rollout(state, unit, USE_ABILITY(best_target))
    score_without      = rollout(state, unit, BEST_NON_ABILITY_ACTION)

    ability_value = score_with_ability - score_without
    urgency = sigmoid(ability_value / SCALE)
```

### Training Data Format

**File:** `src/ai/core/ability_eval/dataset.rs`

```rust
pub struct AbilityEvalSample {
    ability_category: AbilityCategory,
    features: Vec<f32>,
    urgency: f32,
    target_id: Option<u32>,
    target_pos: Option<(f32, f32)>,
}
```

### Training

Two losses per evaluator:
- **Urgency**: MSE loss on urgency score (regression)
- **Targeting**: Cross-entropy on target selection (classification among valid targets)

Combined: `loss = urgency_loss + lambda × target_loss`

### Candidate Position Generation (for ground-targeted)

For GroundTarget/Direction abilities:
1. **Enemy centroid**: Mean position of all enemies in range
2. **Dense cluster**: Position that maximizes enemies within ability radius
3. **Per-enemy positions**: Each enemy's exact position
4. **Predicted positions**: Where enemies will be in `cast_time_ms`
5. **Cut-off positions**: Between enemies and their retreat path

---

## Key Files

| File | Purpose |
|------|---------|
| `src/ai/core/ability_eval/eval.rs` | Runtime ability evaluation (interrupt check) |
| `src/ai/core/ability_eval/weights.rs` | Neural network weight storage and forward pass |
| `src/ai/core/ability_eval/categories.rs` | `AbilityCategory` enum and classification |
| `src/ai/core/ability_eval/features.rs` | Per-category feature extraction |
| `src/ai/core/ability_eval/features_aoe.rs` | AoE-specific feature extraction |
| `src/ai/core/ability_eval/oracle_scoring.rs` | Oracle-based urgency scoring |
| `src/ai/core/ability_eval/dataset.rs` | Training data generation and I/O |
| `src/ai/squad/combat/abilities.rs` | Heuristic ability scoring (fallback) |
| `src/ai/squad/combat/targeting.rs` | Target selection with personality influence |
| `src/ai/squad/forces.rs` | Force calculation and personality weight matrix |
| `src/ai/squad/intents.rs` | Per-unit intent generation (integrates both systems) |
| `src/ai/squad/state.rs` | Squad state, blackboard, formation mode |
| `generated/ability_eval_weights.json` | Trained neural network weights |
