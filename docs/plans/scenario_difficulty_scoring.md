# Scenario Difficulty Scoring via Entity Encoder v2

## Overview

Use the pretrained entity encoder v2 (99.1% win accuracy, HP MAE 0.037) to score
scenario difficulty without running simulations. The encoder takes a 210-dimensional
game state (7 entities x 30 features) and predicts win probability. By constructing a
synthetic tick-0 game state from hero templates and enemy specifications, we can
estimate difficulty for any scenario configuration in microseconds rather than the
milliseconds required by full simulation.

## 1. Constructing Initial Game State from Scenario Config

### The 210-dim Feature Vector

The entity encoder expects 7 entity slots, each with 30 features, laid out as:
`[self(30), enemy0(30), enemy1(30), enemy2(30), ally0(30), ally1(30), ally2(30)]`

At tick 0, many dynamic features have known default values:

| Feature Index | Name | Tick-0 Source |
|---|---|---|
| 0 | hp_pct | 1.0 (full HP) |
| 1 | shield_pct | 0.0 (no shields yet) |
| 2 | resource_pct | 1.0 (full resource) |
| 3 | armor / 200 | From hero template |
| 4 | magic_resist / 200 | From hero template |
| 5-6 | position_x/y / 20 | From spawn layout |
| 7 | distance / 10 | Computed from spawn positions |
| 8-11 | cover, elevation, zones | 0.0 (no terrain at tick 0) |
| 12 | auto_dps / 30 | `attack_damage / (attack_cooldown_ms / 1000)` |
| 13 | attack_range / 10 | From template |
| 14 | attack_cd_remaining_pct | 0.0 (ready at start) |
| 15-17 | ability damage/range/cd | From template strongest ability, 0.0 cd |
| 18-20 | heal amount/range/cd | From template heal ability, 0.0 cd |
| 21-23 | control range/duration/cd | From template CC ability, 0.0 cd |
| 24-27 | casting/cc/movespeed | 0.0 / 0.0 / from template |
| 28 | total_damage_done / 1000 | 0.0 |
| 29 | exists | 1.0 if slot filled, 0.0 if padding |

### Implementation: `initial_game_state_from_scenario`

A new function that takes a `ScenarioCfg` and produces 210-dim `Vec<f32>` per hero:

1. Resolve hero templates via `resolve_hero_templates()`
2. For HvH: resolve enemy hero templates. For PvE: use `default_enemy_wave()`
3. Convert templates to `UnitState` with spawn positions matching runner layout
4. Apply `hp_multiplier` from `ScenarioCfg`
5. Construct minimal `SimState` (tick=0, no zones, no projectiles)
6. Call `extract_game_state()` from each hero's perspective
7. Return the set of 210-dim vectors

### Handling Asymmetric Perspectives

Compute win probability from each hero's perspective and average them.
This naturally handles different threat assessments (melee vs ranged heroes
see different distances).

## 2. Win Probability as Difficulty Score

### Difficulty Scale

| Win Probability | Label | Interpretation |
|---|---|---|
| > 0.85 | Trivial | Heroes massively outstat enemies |
| 0.70 - 0.85 | Easy | Clear hero advantage |
| 0.50 - 0.70 | Moderate | Roughly balanced, slight hero edge |
| 0.30 - 0.50 | Hard | Enemy advantage, requires good play |
| < 0.30 | Very Hard | Near-unwinnable with default AI |

A **difficulty score** = `1.0 - win_probability` (0-1, higher = harder).

The secondary HP prediction adds texture: two scenarios might both have 0.6 win
probability, but one leaves heroes at 80% HP (easy win when they win) while another
at 20% (pyrrhic victory).

### Combined Difficulty Metric

```
difficulty = (1.0 - win_prob) * 0.7 + (1.0 - hp_pred) * 0.3
```

Weights binary outcome more heavily but factors in expected attrition.

## 3. Applications

### 3.1 Filtering Generated Scenarios for Training Balance

~3,300 generated scenarios span an unknown difficulty distribution. Scoring allows:
- **Balanced training sets**: Sample equal numbers from each difficulty band
- **Degenerate removal**: Discard trivial (>0.95) or impossible (<0.05) scenarios
- **Stratified oracle rollouts**: Run expensive oracle evaluation only on moderate scenarios

### 3.2 Difficulty Curves for Campaigns

- Sort scenarios by difficulty score for natural difficulty ramp
- Identify difficulty gaps (large jumps between adjacent scenarios)
- Auto-suggest scenario parameters to fill gaps

### 3.3 Identifying Degenerate Compositions

- **Auto-win**: Win probability > 0.95 regardless of enemy = overpowered synergy
- **Auto-lose**: Win probability < 0.05 = unviable composition
- **Sensitivity analysis**: Score against many enemy configs; high variance = brittle

### 3.4 Hero Balance Diagnostics

Compute average difficulty contribution per hero:
- Score all scenarios, measure how adding/removing each hero shifts win probability
- Heroes that consistently raise win probability beyond their fair share may be overtuned

## 4. Integration with Scenario Generation Pipeline

### 4.1 Score During Generation

Modify `src/scenario/gen/strategies.rs` to optionally score each `ScenarioFile`:
1. Add a `DifficultyScorer` struct holding entity encoder weights + prediction heads
2. After building each `ScenarioFile`, call `scorer.score(&scenario_file)`
3. Attach score as metadata

### 4.2 Filter and Rank

Add post-generation filtering:
- `--min-difficulty 0.2 --max-difficulty 0.8` to filter to a balanced band
- `--sort-by-difficulty` to output in difficulty order
- `--stratify N` to output N scenarios evenly sampled across difficulty bands

### 4.3 Score Existing Scenarios: `xtask scenario score`

New CLI command:

```
xtask scenario score <path> --encoder generated/entity_encoder_scorer_v2.json
```

Options:
- `--format table|csv|json` (default: table)
- `--output <file>` (default: stdout)
- `--sort` (sort by difficulty descending)
- `--filter-min <f32>` / `--filter-max <f32>` (win probability bounds)

## 5. What Needs to Be Built

### 5.1 Standalone Entity Encoder with Outcome Heads (Rust)

New file: `src/ai/core/difficulty.rs` (shared with fight difficulty estimation plan)

Contains `DifficultyScorer` struct with:
- `FlatEntityEncoder` (reuse from shared `nn.rs`)
- `win_head`: two `FlatLinear` layers (32 -> 32 GELU -> 1)
- `hp_head`: two `FlatLinear` layers (32 -> 32 GELU -> 1 sigmoid)
- Mean-pool over existing entity tokens

### 5.2 Extended Entity Encoder Export Script

Modify `training/export_entity_encoder.py` with `--include-heads` flag
to export `win_head` and `hp_head` (~8 KB additional).

### 5.3 Game State Construction from Templates

New function in `src/ai/core/ability_eval/game_state.rs`:

```rust
pub fn initial_game_state_from_scenario(cfg: &ScenarioCfg) -> Vec<Vec<f32>>
```

### 5.4 CLI Command: `xtask scenario score`

Wire template loading + game state construction + scorer into a CLI subcommand.

### 5.5 Generation Pipeline Integration

Add optional `scorer: Option<DifficultyScorer>` to `GenConfig`. Score during
generation, include difficulty in coverage report, optionally reject scenarios
outside configured difficulty band.

## 6. Limitations

### 6.1 Trained on Mid-Fight States, Not Initial States

The entity encoder was trained on snapshots sampled throughout fights. At tick 0,
many features are at default values. The model has seen tick-0 states but the
majority of training samples are mid-fight.

**Mitigation**: Validate initial-state predictions against actual simulation outcomes
on ~3,300 generated scenarios. If accuracy drops, consider fine-tuning with tick-0
samples weighted higher.

### 6.2 Static Analysis -- No Dynamics

The score captures team composition strength but not:
- AI quality (default vs student+ability-eval)
- Positional play (terrain, room layout, pathing)
- Ability synergies (CC chains, AoE stacking)
- Cooldown dynamics (all ready at tick 0)

### 6.3 Entity Slot Limits

At most 4 heroes (1 self + 3 allies) and 3 enemies. Larger teams need truncation.

### 6.4 Win Probability is Not Difficulty

Win probability measures expected outcome under the default AI used during training.
The combined system at 92.9% wins many scenarios the model predicts as "hard".
The score is relative to the training AI, not absolute.

## 7. Implementation Order

1. **Export with heads** -- Modify export script, generate weights. (30 min)
2. **Shared nn.rs** -- Extract types from weights.rs. (1 hr, shared with difficulty estimation)
3. **DifficultyScorer in Rust** -- JSON loading, mean pooling, predict. (1 hr, shared)
4. **Initial game state builder** -- Template loading + spawn positions. (1.5 hr)
5. **CLI command** -- `xtask scenario score` wiring. (1 hr)
6. **Validation** -- Score 28 hand-crafted scenarios, compare vs actual outcomes. (1 hr)
7. **Generation integration** -- Optional scoring in GenConfig. (1 hr)

Total estimated effort: ~7 hours (with ~2.5 hrs shared with fight difficulty estimation).
