# Fight Difficulty Estimation via Entity Encoder v2

## Overview

The entity encoder v2 (`generated/entity_encoder_pretrained_v2.pt`) is pretrained
on fight outcome prediction: given a 210-dim game state snapshot (7 entities x 30
features), it predicts **win probability** and **remaining HP%** with high accuracy
(99.1% win accuracy, HP MAE 0.037). Currently, the export script
(`training/export_entity_encoder.py`) drops the `win_head` and `hp_head`, exporting
only the encoder trunk for cross-attention use in the ability transformer.

This plan describes how to export and use the prediction heads standalone as a
**fight difficulty estimator** that calibrates AI aggressiveness, formation choice,
and ability usage thresholds at fight start and during combat.

---

## 1. What the Encoder Predicts

The `EntityEncoderPretraining` model (defined in `training/pretrain_entity.py`)
has three stages:

1. **Entity encoder trunk** -- projects 7x30 entity features through a linear
   layer + type embeddings + 2-layer self-attention + output norm, producing
   7 entity tokens of dimension `d_model=32`.

2. **win_head** -- mean-pools the entity tokens, then passes through
   `Linear(32,32) -> GELU -> Linear(32,1)`. Output is a raw logit;
   `sigmoid(logit) > 0.5` means hero team predicted to win.

3. **hp_head** -- same pooling, then `Linear(32,32) -> GELU -> Linear(32,1)
   -> Sigmoid`. Output is predicted hero team average HP% at fight end (0-1).

### Difficulty Signal

From these two outputs we derive a single **difficulty score**:

```
win_prob = sigmoid(win_head_output)
predicted_hp = hp_head_output          // already in [0, 1]

// Difficulty: 0.0 = trivial stomp, 1.0 = near-certain loss
difficulty = 1.0 - win_prob
// Tightness: how close the fight is expected to be (even if winning)
tightness = 1.0 - predicted_hp
```

| win_prob | predicted_hp | Interpretation | difficulty | tightness |
|----------|-------------|----------------|------------|-----------|
| 0.95     | 0.80        | Easy win       | 0.05       | 0.20      |
| 0.75     | 0.40        | Favored but costly | 0.25   | 0.60      |
| 0.50     | 0.20        | Coin flip      | 0.50       | 0.80      |
| 0.20     | 0.05        | Likely loss    | 0.80       | 0.95      |

Both signals are useful: `difficulty` drives macro strategy (fight or flee),
while `tightness` drives micro decisions (conserve cooldowns or go all-in).

---

## 2. How Difficulty Maps to AI Behavior

### 2a. Personality Drift at Fight Start

When a fight begins (or on first tick of combat), run the encoder once and
apply a **difficulty-driven personality drift** to all hero units via
`SquadAiState::apply_drift`. This is additive to the existing per-unit
personality, clamped by the existing `DRIFT_CAP = 0.15`.

| Difficulty Range | Personality Effect |
|-----------------|-------------------|
| 0.0 - 0.3 (easy) | +aggression, -caution, -patience (go aggressive, finish fast) |
| 0.3 - 0.6 (medium) | No drift (baseline personality is calibrated for this) |
| 0.6 - 0.8 (hard) | +caution, +discipline, +patience (play safe, conserve resources) |
| 0.8 - 1.0 (desperate) | +aggression, +tenacity, -patience (high risk plays are the only chance) |

The desperate range flips back to aggressive because conservative play in a
losing matchup just delays the inevitable -- the AI should gamble on burst
windows and all-in plays.

Implementation: a new `DriftTrigger::DifficultyAssessment(f32)` variant, or
a dedicated method on `SquadAiState` that takes the difficulty score and applies
graduated drift to all team units.

### 2b. Formation Mode Override

The blackboard's `FormationMode` (Hold / Advance / Retreat) in
`src/ai/squad/state.rs` currently uses HP averages and count
ratios. The difficulty estimator provides a stronger pre-fight signal:

- **Easy (difficulty < 0.3)**: Force `FormationMode::Advance` on first
  evaluation. The team should close distance and press the advantage.
- **Medium (0.3 - 0.7)**: No override, use the existing HP-based heuristic.
- **Hard (difficulty > 0.7)**: Start in `FormationMode::Hold`. Do not advance
  until an enemy is killed or team HP advantage shifts.

This integrates in `evaluate_blackboard()` by checking a difficulty field on the
`SquadAiState` struct.

### 2c. Ability Urgency Threshold Scaling

The ability evaluator fires when urgency exceeds `URGENCY_THRESHOLD = 0.4`
(defined in `src/ai/core/ability_eval/categories.rs`). Difficulty should
modulate this:

```
effective_threshold = base_threshold + difficulty_adjustment

// Easy fights: lower threshold (use abilities freely, they're not needed for survival)
//   difficulty 0.0 -> adjustment = -0.05
// Hard fights: raise threshold (conserve powerful abilities for critical moments)
//   difficulty 0.7 -> adjustment = +0.08
// Desperate fights: lower threshold again (use everything, nothing to save for)
//   difficulty 0.9 -> adjustment = -0.10
```

This is a V-shaped curve: both easy and desperate fights lower the threshold,
while medium-hard fights raise it. The rationale: in hard (but winnable) fights,
wasting a big cooldown on a low-value target can cost the fight. In desperate
fights, every ability must fire ASAP.

### 2d. Cleanup Boost Calibration

The existing `apply_cleanup_boost` and `apply_cleanup_suppress` in
`src/ai/core/ability_eval/eval.rs` use hard-coded tick thresholds (2000 and 5000)
and advantage ratios. The difficulty estimator can make these adaptive:

- If the encoder predicted an easy fight and the fight is dragging past the
  predicted duration, activate cleanup behavior earlier.
- If the encoder predicted a hard fight, delay cleanup suppression since the
  team may still need defensive abilities.

---

## 3. Integration Points in the AI Pipeline

### Where it runs

The difficulty estimation should run at **two points**:

1. **Fight initialization** (tick 0 or first combat contact): Full assessment,
   stores `difficulty` and `tightness` on `SquadAiState`. Drives personality
   drift and initial formation.

2. **Periodic reassessment** (every N ticks, e.g., every 50 ticks / 5 seconds):
   Re-run the encoder on current state to detect snowballing. Update stored
   difficulty. Do NOT re-apply personality drift (that was one-shot); instead
   update only the formation mode and ability threshold.

### Struct changes

```rust
// In src/ai/squad/state.rs, add to SquadAiState:
pub struct DifficultyAssessment {
    pub win_prob: f32,        // sigmoid(win_head output), 0-1
    pub predicted_hp: f32,    // hp_head output, 0-1
    pub difficulty: f32,      // 1.0 - win_prob
    pub tightness: f32,       // 1.0 - predicted_hp
    pub assessed_at_tick: u64,
}
```

### Call sites

1. `SquadAiState::new()` or `new_inferred()` -- after construction, if
   `difficulty_estimator` is loaded, run initial assessment.

2. `evaluate_blackboards_if_needed()` in `src/ai/squad/state.rs` -- add
   difficulty reassessment on a slower cadence (every 50 ticks vs every 5 for
   blackboards).

3. `generate_intents_with_terrain()` in `src/ai/squad/intents.rs` -- read
   difficulty to adjust ability threshold before the ability eval interrupt block.

4. `compute_raw_forces()` in `src/ai/squad/forces.rs` -- optionally scale
   retreat/attack base forces by difficulty.

---

## 4. What Needs to Be Built

### 4a. Export Script: Include Prediction Heads

Modify `training/export_entity_encoder.py` to optionally export `win_head`
and `hp_head` alongside the encoder trunk. Add a `--with-heads` flag:

```python
if args.with_heads:
    export["win_head"] = {
        "linear1": export_linear(sd, "win_head.0"),
        "linear2": export_linear(sd, "win_head.2"),
    }
    export["hp_head"] = {
        "linear1": export_linear(sd, "hp_head.0"),
        "linear2": export_linear(sd, "hp_head.2"),
    }
```

Output file: `generated/entity_encoder_weights_v2_with_heads.json`.
Estimated additional size: ~8 KB (two small `32->32->1` MLPs).

### 4b. Rust Inference: DifficultyEstimator Struct

New file: `src/ai/core/difficulty.rs`

```rust
pub struct DifficultyEstimator {
    encoder: FlatEntityEncoder,  // reuse from ability_transformer/weights.rs
    win_l1: FlatLinear,          // 32 -> 32
    win_l2: FlatLinear,          // 32 -> 1
    hp_l1: FlatLinear,           // 32 -> 32
    hp_l2: FlatLinear,           // 32 -> 1
}

impl DifficultyEstimator {
    pub fn from_json(json_str: &str) -> Result<Self, String>;

    /// Run inference on 210-dim game state.
    /// Returns (win_probability, predicted_hp_remaining).
    pub fn predict(&self, game_state: &[f32]) -> (f32, f32);

    /// Convenience: returns DifficultyAssessment struct.
    pub fn assess(&self, game_state: &[f32]) -> DifficultyAssessment;
}
```

The `FlatEntityEncoder` type already exists in
`src/ai/core/ability_transformer/weights.rs` but is private. Extract it into a
shared `src/ai/core/nn.rs` module that both `ability_transformer/weights.rs`
and `difficulty.rs` import.

### 4c. Mean Pooling

The entity encoder trunk produces 7 entity tokens. The prediction heads in
Python use masked mean pooling (average over entities where `exists > 0.5`).

```rust
fn mean_pool(tokens: &[f32], mask: &[bool], d_model: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; d_model];
    let mut count = 0.0f32;
    for (ent, &exists) in mask.iter().enumerate() {
        if exists {
            for i in 0..d_model {
                pooled[i] += tokens[ent * d_model + i];
            }
            count += 1.0;
        }
    }
    if count > 0.0 {
        for v in pooled.iter_mut() { *v /= count; }
    }
    pooled
}
```

### 4d. CLI Integration

Add `--difficulty-estimator <path>` flag to the scenario runner CLI, similar
to the existing `--ability-eval` and `--ability-encoder` flags.

---

## 5. Edge Cases

### 5a. Mirror Matches
When both teams have similar compositions, the encoder predicts ~50% win
probability (difficulty ~0.5, medium). No personality drift, baseline formation.

### 5b. Snowballing
A fight that starts 50/50 but swings after an early kill needs mid-fight
reassessment. The periodic reassessment (every 50 ticks) handles this:
dead unit's entity slot has `exists = 0.0`, encoder sees 3v2 and updates sharply.

### 5c. Reassessment Frequency
Every 50 ticks (500ms) is sufficient. Kills happen on the order of seconds.
Personality drift is NOT re-applied on reassessment (one-shot at fight start).

### 5d. Asymmetric Team Sizes
The encoder handles 1v1 through 4v3 via the `exists` mask. Extreme asymmetries
(1v3) may be out of training distribution if the dataset is dominated by 3v3.

### 5e. Encoder Reuse with Ability Transformer
The entity encoder trunk is identical between the difficulty estimator and the
ability transformer's cross-attention path. If both are loaded, the entity
encoding can be computed once and shared via the existing `EncodedEntities` struct.

### 5f. Calibration Drift
The encoder was trained on outcomes from a specific AI configuration. If the
AI improves, the encoder's difficulty estimates may become miscalibrated.
Periodic retraining on fresh outcome data is necessary.

---

## 6. Implementation Order

1. **Export with heads** -- Modify `export_entity_encoder.py`, generate weights. (30 min)
2. **Shared nn.rs** -- Extract `FlatLinear`, `FlatLayerNorm`, `TransformerLayer`, `FlatEntityEncoder`. (1 hr)
3. **DifficultyEstimator** -- New `difficulty.rs` with JSON loading, mean pooling, predict/assess. (1 hr)
4. **SquadAiState integration** -- Add fields, initial assessment, periodic reassessment. (1 hr)
5. **Behavior hooks** -- Personality drift, formation override, threshold scaling. (1 hr)
6. **CLI flag** -- `--difficulty-estimator` in xtask. (30 min)
7. **Benchmarking** -- Run 28-scenario benchmark with/without. Tune drift magnitudes. (2 hrs)

Total estimated effort: ~7 hours.
