# Threat Assessment via Entity Encoder Attention Weights

## Overview

The entity encoder v2 is a 2-layer, 4-head self-attention model (d=32) pretrained
on fight outcome prediction (99.1% win accuracy). It processes 7 entity tokens
`[self, enemy0, enemy1, enemy2, ally0, ally1, ally2]`, each with 30 features
covering vitals, position, combat stats, ability readiness, healing, CC capability,
and current state.

During self-attention, each entity token attends to every other token. The resulting
attention weights encode learned relationships: which entities are relevant to which,
as judged by a model trained to predict fight outcomes. These weights are currently
computed during inference in `FlatEntityEncoder::forward()` but are discarded after
the weighted-sum step. This plan describes how to extract, interpret, aggregate, and
integrate these attention weights as a threat/protection priority signal.

## 1. Extracting Attention Weights from the Entity Encoder

### Current Code Path

The entity encoder's self-attention lives in `TransformerLayer::multi_head_attention()`
in `src/ai/core/ability_transformer/weights.rs`. The method computes softmax attention
scores per head per query position, uses them for the weighted sum of values, and then
drops them. The calling chain is:

```
FlatEntityEncoder::forward()
  -> TransformerLayer::forward()
    -> multi_head_attention()
```

`FlatEntityEncoder::forward()` returns `(Vec<f32>, Vec<bool>)` -- entity tokens and mask.
`EncodedEntities` stores these for reuse across abilities.

### Proposed Changes

**a) New return type: `EntityAttentionWeights`**

```rust
pub struct EntityAttentionWeights {
    /// Per-layer, per-head attention: [n_layers][n_heads][seq_len][seq_len]
    /// attn[layer][head][query][key] = how much query attends to key
    pub weights: Vec<Vec<Vec<Vec<f32>>>>,
}
```

This is 2 layers * 4 heads * 7 * 7 = 392 floats (1.5 KB). Negligible memory cost.

**b) `multi_head_attention_with_weights()`**

Add a variant that returns both the output and the per-head attention matrices.
The existing method remains unchanged for the ability transformer's main path.

**c) `FlatEntityEncoder::forward_with_attention()`**

New method parallel to `forward()` that calls the attention-returning variant
and collects weights from both layers.

**d) Extended `EncodedEntities`**

Add an optional `attention: Option<EntityAttentionWeights>` field. The standard
`encode_entities()` path leaves this as `None`. A new
`encode_entities_with_attention()` method populates it.

### Files to Modify

- `src/ai/core/ability_transformer/weights.rs` -- all structural changes above

## 2. Interpreting Attention Patterns

Entity order is fixed: `[self(0), enemy0(1), enemy1(2), enemy2(3), ally0(4), ally1(5), ally2(6)]`.

### Self-to-Enemy Attention = Threat

`attn[layer][head][0][1..=3]` -- the "self" token attending to enemy tokens. High
attention means the model considers that enemy highly relevant for predicting the
fight outcome from this unit's perspective. This is a learned threat signal incorporating
all 30 features: DPS, ability damage, CC capability, range, proximity, HP, casting state.

### Self-to-Ally Attention = Protection Priority

`attn[layer][head][0][4..=6]` -- how much the self token attends to allies. High
attention on a low-HP ally signals that ally's survival is critical to the fight outcome.

### Enemy-to-Enemy Attention = Focus Fire Synergy

`attn[layer][head][1][2]` etc. -- enemies attending to each other reveals synergy
patterns. Less directly actionable but useful for advanced squad coordination.

### Ally-to-Enemy Attention = Coordination

`attn[layer][head][4..=6][1..=3]` -- allies attending to enemies shows which enemies
the model expects allies to engage. Could inform focus fire decisions.

## 3. Aggregating Attention into Threat Scores

### Per-Entity Threat Score

For each enemy slot `e in {1, 2, 3}`:

```
threat_raw[e] = sum over layers L, heads H of:
    head_weight[L][H] * attn[L][H][0][e]
```

**Head weighting strategy:** Start with uniform weights (1/8 across 2 layers x 4 heads).
Later, learn head weights via a small regression on fight outcomes or oracle decisions.

**Layer aggregation:** Layer 2 (deeper) typically captures higher-order interactions.
Weight layer 2 at 0.6, layer 1 at 0.4.

**Normalization:** Softmax attention already sums to 1 across keys for each query.
Aggregate with a weighted mean, then normalize across enemies:

```
threat[e] = threat_raw[e] / sum(threat_raw[1..=3])
```

This gives a probability distribution over enemies representing relative threat.

### Per-Ally Protection Score

Same aggregation for ally slots `a in {4, 5, 6}`:

```
protection_priority[a] = sum over L, H of:
    head_weight[L][H] * attn[L][H][0][a]
```

Normalized across allies similarly.

### Proposed Struct

```rust
pub struct ThreatAssessment {
    /// Threat score per enemy slot [0..3], sums to ~1.0
    pub enemy_threat: [f32; 3],
    /// Protection priority per ally slot [0..3], sums to ~1.0
    pub ally_priority: [f32; 3],
    /// Entity IDs corresponding to enemy slots
    pub enemy_ids: [u32; 3],
    /// Entity IDs corresponding to ally slots
    pub ally_ids: [u32; 3],
}
```

## 4. Integration Points

### 4a. Targeting (highest impact)

**Current:** `choose_target()` in `src/ai/squad/combat/targeting.rs` uses a heuristic
score: HP damage dealt, focus bonus (discipline-weighted), distance bias, and combo
bonus (CC'd targets). The blackboard `focus_target` is set to the lowest-HP enemy.

**With threat scores:** Replace or augment the `focus_target` selection in
`evaluate_blackboard()` (`src/ai/squad/state.rs`). Instead of always focusing the
lowest-HP enemy, use attention-derived threat to pick the most threatening enemy.

Additionally, modify `target_score()` in targeting.rs to incorporate threat:

```rust
let threat_bonus = threat_assessment.enemy_threat[enemy_slot_index] * 15.0;
// replaces or supplements the existing focus_bonus
```

### 4b. Ability Evaluation Target Selection

**Current:** `evaluate_abilities_with_encoder()` in `src/ai/core/ability_eval/eval.rs`
extracts per-category features and uses micro-model predictions for target selection.

**With threat scores:** For DamageUnit and CcUnit categories, bias the target logits
by threat score before argmax:

```rust
for (i, &tid) in target_ids.iter().enumerate() {
    if let Some(slot_idx) = threat.enemy_slot_for_id(tid) {
        output[1 + i] += threat.enemy_threat[slot_idx] * threat_bias_strength;
    }
}
```

### 4c. Positioning / Force Computation

**Current:** `compute_raw_forces()` in `src/ai/squad/forces.rs` computes
protect/retreat forces using HP thresholds and distance heuristics.

**With threat scores:** Modulate the retreat force by how much attention enemies
pay to the self token (reverse: how threatening are enemies to me):

```rust
let incoming_threat = sum(attn[L][H][enemy_e][0] for all enemies e, layers L, heads H);
```

### 4d. Heal Target Selection

**Current:** HealUnit category picks the best heal target from extracted features.

**With threat scores:** Weight heal targets by `ally_priority` -- an ally the model
considers critical to fight outcome should be healed preferentially, even if not the
lowest-HP ally.

## 5. Implementation Plan

### Phase 1: Attention Extraction (Rust only)

1. Add `EntityAttentionWeights` struct to `weights.rs`
2. Add `multi_head_attention_with_weights()` to `TransformerLayer`
3. Add `forward_with_attention()` to `FlatEntityEncoder`
4. Add `encode_entities_with_attention()` to `AbilityTransformerWeights`
5. Extend `EncodedEntities` with `attention: Option<EntityAttentionWeights>`
6. Unit test: verify attention weights sum to 1.0 per query, match forward pass

### Phase 2: Threat Scoring

1. Add `ThreatAssessment` struct and `compute_threat()` in new `src/ai/core/threat.rs`
2. Wire entity ID mapping: `extract_game_state()` already sorts enemies by distance
   and allies by HP -- expose the sorted ID lists alongside the feature vector
3. Aggregation: uniform head weights initially, configurable via JSON
4. Cache `ThreatAssessment` per hero per tick (reuse entity encoding from ability eval)

### Phase 3: Integration

1. **Blackboard focus target:** Replace lowest-HP heuristic with highest-threat enemy
2. **Target score:** Add threat bonus term in `target_score()` in targeting.rs
3. **Ability eval bias:** Add optional threat-aware target biasing
4. **Force modulation:** Scale protect/retreat forces by attention-derived signals
5. Each integration point individually toggleable for A/B testing

### Phase 4: Validation

See next section.

## 6. Validation Plan

### 6a. Attention Visualization (offline)

Export attention matrices for a corpus of game states. Visualize as 7x7 heatmaps
to verify patterns match intuition:
- Self should attend strongly to nearby high-DPS enemies
- Self should attend strongly to low-HP allies when healer
- Attention should shift as the fight progresses

### 6b. Attention-Threat vs Heuristic Correlation

For each game state in the outcome dataset, compute both:
- Heuristic threat: current `target_score()` output
- Attention-derived threat: from entity encoder

Measure rank correlation (Spearman). If they agree >80%, the attention signal is
redundant. Disagreements are where the value lies.

### 6c. Oracle Agreement

Run oracle rollouts with attention-derived targeting vs current targeting. Measure:
- Win rate across the 28-scenario benchmark
- Timeout rate (attention targeting should reduce stalemates)
- Decision agreement with oracle-optimal actions

### 6d. A/B Benchmark

1. Baseline: current system (92.9% win rate)
2. Threat-targeting: replace focus_target with attention-derived threat
3. Threat-targeting + ability bias: add target logit biasing
4. Full integration: all four integration points

Target: maintain or exceed 92.9% win rate while reducing timeouts.

### 6e. Computational Cost

The entity encoder already runs once per hero per tick. Extracting attention adds:
- 392 extra floats stored (no extra compute -- scores already computed, just currently discarded)
- Aggregation: ~50 multiplies + additions per hero per tick
- Total overhead: <1% of current tick budget

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Attention weights may not be interpretable as threat | Validate with heatmap visualization before integrating |
| 4-head model may not have specialized heads | Start with uniform head weights; learn head weights if needed |
| Entity ordering dependence (enemies sorted by distance) | Track entity IDs through the pipeline |
| Overfitting to pretrain distribution | Entity encoder is frozen; threat scoring is read-only |
| Regression in win rate | All integration points independently toggleable; A/B test each one |
