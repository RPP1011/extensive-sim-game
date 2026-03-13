# Ability Latent Operator: Architecture & Training Plan

## Overview

The goal is a two-component system:

1. **State Encoder** — maps raw sim state (entity features + threats + positions + ability slots) into a latent representation suitable for downstream RL and operator queries
2. **Ability Operator** — a learned neural function that transforms the encoded state according to a specific ability cast, predicting the latent state after the ability resolves

The core framing: abilities are **latent space operators**. Rather than predicting raw feature deltas directly, we encode state -> apply the operator in latent space -> decode back to interpretable feature deltas. This gives the system two desirable properties: the encoder learns a representation shaped by ability effects (useful for RL), and the operator gives the actor a differentiable single-ability simulator (useful for planning).

```
sim_state_t --encode--> z_t
                          |
                ability --> operator --> z_t+w
                          |
                          +--decode--> delta_state (hp, cc, pos, exists)
```

---

## Part 1: State Encoder

### Input Representation

The encoder processes a heterogeneous token sequence. All tokens are projected to `d_model=32` and processed jointly by a transformer encoder.

#### Token Types

| Type ID | Slots | Raw Dim | Description |
|---------|-------|---------|-------------|
| 0 | 1 | 23 | Self entity |
| 1 | 3 | 23 | Enemy entities (sorted by distance) |
| 2 | 3 | 23 | Ally entities (sorted by HP%) |
| 3 | 8 | 8 | Threat tokens |
| 4 | 8 | 8 | Position tokens |
| 5-7 | N*8 | 34 | Ability slot tokens (new) |

**Total sequence length:** 23 fixed tokens + up to 56 ability tokens (7 entities * 8 abilities) = 79 max, ~51 typical

#### Ability Slot Tokens (New)

Each ability slot for each entity becomes a token in the sequence:

```
ability_token[i] = [
    ability_transformer_cls,   # 32-dim: frozen, from DSL pre-training
    is_ready,                  # 0/1: dynamic runtime state
    cooldown_fraction          # [0,1]: dynamic runtime state
]  # 34-dim, projected -> d_model via new Linear(34, 32)
```

Type embedding distinguishes owner role: `self_ability (5)`, `enemy_ability (6)`, `ally_ability (7)`.

The ability transformer [CLS] embeddings are **frozen** -- pre-trained via MLM on 75K DSL sequences with auxiliary hint classification. They encode targeting type, delivery method, damage/heal/CC profiles, area shapes, mechanic flags, and timing -- everything the entity encoder needs to understand what each unit is capable of.

#### Entity Feature Changes

Remove the 9 collapsed ability scalars from entity features (indices 15-23: ability_damage, ability_range, ability_cd, heal_amount, heal_range, heal_cd, cc_duration, cc_range, cc_cd). These are now fully superseded by per-ability tokens. The entity feature vector shrinks from 32 to 23 dims.

**Updated entity features (23 dims):**

| Indices | Group | Features | Predicted? |
|---------|-------|----------|------------|
| 0-2 | vitals | hp_pct, shield_pct, resource_pct | Yes |
| 3-4 | static | armor/200, magic_resist/200 | No |
| 5-6 | position | x/20, y/20 | Yes |
| 7-11 | terrain | dist_to_enemy, cover, elevation, hostile_zones, friendly_zones | No |
| 12-14 | combat | auto_dps/30, attack_range/10, attack_cd_pct | No |
| 15-18 | state | is_casting, cast_progress, cc_remaining/2000, move_speed/5 | Low weight |
| 19 | meta | total_damage_done/1000 | No |
| 20 | exists | alive 0/1 | Yes (BCE) |
| 21-22 | relative | dx_from_self, dy_from_self | Derived |

#### Input Projections

```python
projections = {
    'entity':   Linear(23, d_model),   # updated dim
    'threat':   Linear(8,  d_model),
    'position': Linear(8,  d_model),
    'ability':  Linear(34, d_model),   # new
}
type_embedding = Embedding(8, d_model) # 5 existing + 3 ability owner roles
```

### Encoder Architecture

```
Input tokens (variable length, padded)
    | per-type linear projection -> d_model=32
    | + type embedding
    | LayerNorm
    | TransformerEncoder
         n_layers=4, n_heads=4, d_ff=64
         pre-norm, GELU, no dropout
         grokking init std=0.007
    | LayerNorm
    | extract entity tokens [0:7]     -> (B, 7, 32)
Output: 7 entity tokens in latent space
```

The transformer attends across all token types simultaneously -- entity tokens will learn to attend to their own ability tokens for readiness/cooldown context, and to threat/position tokens for spatial context.

### Pre-training Objective

The encoder is pre-trained jointly with the operator (see Part 2). There is no separate next-state prediction objective. The operator training loss backpropagates through the encoder, shaping the latent space to support ability effect queries.

**Ablation note:** if joint training causes the encoder to collapse to a representation that only serves ability effect prediction (verified by monitoring whether entity token cosine similarities become degenerate), fall back to a two-stage approach: pre-train encoder on a lightweight auxiliary objective (e.g. masked entity feature prediction), then freeze and train operator.

---

## Part 2: Ability Operator

### Framing

The operator is a learned function:

```
z_after = operator(z_before, ability_cls, caster_slot, duration_norm)
```

where `z_before` is the 7x32 latent state from the encoder, `ability_cls` is the frozen 32-dim transformer embedding of the cast ability, and `duration_norm` is the window length normalized by max window.

### Operator Architecture

The ability is injected as an additional token appended to the entity sequence:

```python
# Construct operator input sequence
ability_token = LayerNorm(
    ability_cls
    + caster_slot_embedding    # Embedding(7, 32): which entity slot is casting
    + duration_embedding       # sinusoidal encoding of duration_norm
)

operator_input = concat([z_before, ability_token.unsqueeze(1)], dim=1)
# shape: (B, 8, 32) -- 7 entity tokens + 1 ability token

z_after = TransformerEncoder(operator_input)[:, :7, :]
# extract entity tokens, discard ability token
# shape: (B, 7, 32)
```

Using the ability as a sequence token rather than FiLM/hypernetwork conditioning is deliberate: the transformer learns asymmetric attention patterns -- entities inside an AoE radius attend strongly to the ability token while out-of-range entities learn to ignore it. FiLM applies the same modulation to all entities uniformly, which is incorrect for targeted and area abilities.

**Operator transformer hyperparams:**

| Param | Value | Note |
|-------|-------|------|
| n_layers | 2-4 | Start 2, scale if needed |
| n_heads | 4 | Match entity encoder |
| d_ff | 64 | Match entity encoder |
| d_model | 32 | Match entity encoder |
| init std | 0.007 | Grokking setup |
| dropout | 0.0 | Grokking setup |

### Decoder Heads

Decode `z_after` entity tokens to feature-group deltas. Predict only what abilities can affect; loss is zero-weighted for effect types the ability cannot produce (derived from the 80-dim property vector).

```python
# Per-entity, applied to operator output tokens
hp_head     = MLP(32 -> 64 -> 3)   # delta_hp_pct, delta_shield_pct, delta_resource_pct
                                    # beta-NLL loss (mean + log_var)
cc_head     = MLP(32 -> 64 -> 2)   # cc_remaining_norm, is_stunned
                                    # beta-NLL + BCE respectively
pos_head    = MLP(32 -> 64 -> 2)   # delta_x, delta_y (knockback, dash, pull)
                                    # beta-NLL loss
exists_head = MLP(32 -> 32 -> 1)   # death probability
                                    # BCE loss, sigmoid output
```

**Loss weight masking by ability type:**

```python
def loss_mask_for_ability(ability_props_80dim):
    return {
        'hp':     ability_props_80dim[41] > 0  # has damage
                  or ability_props_80dim[45] > 0,  # has heal
        'cc':     ability_props_80dim[48:63].any(),  # has any CC
        'pos':    ability_props_80dim[63],  # has dash/mobility
        'exists': ability_props_80dim[41] > 0  # only damage can kill
    }
```

This prevents the cc_head from receiving gradient noise from a pure-damage ability like Fireball, and the pos_head from receiving gradient noise from a stationary zone like Blizzard.

### Window Size

```python
def compute_window(ability_def, grace_factor=0.2, max_window_ms=6000):
    effect_duration = max(
        ability_def.cast_time_ms,
        ability_def.delivery.duration_ms if ability_def.delivery else 0,
        max(e.duration_ms for e in collect_all_effects(ability_def)),
        dot_duration(ability_def)
    )
    window = min(effect_duration * (1 + grace_factor), max_window_ms)
    return max(window, 500)  # minimum 500ms even for instant abilities
```

Pass `window / max_window_ms` as `duration_norm` scalar to the operator. The operator learns to time-weight effects: Blizzard's damage spreads across the window, Fireball's is front-loaded.

### Handling Duration Ability Types

| Type | Window anchor | Special handling |
|------|--------------|------------------|
| Instant (Fireball) | cast_time | Short window, effect at t=0 |
| Projectile (delayed) | cast_time + travel_time | Effect mid-window |
| Channel (tick damage) | channel_duration | Effect distributed across window |
| Zone (Blizzard) | zone_duration | Effect depends on entity movement -- high variance |
| DoT/HoT | dot_duration | Effect distributed, deterministic rate |
| Trap | arm_time + zone_duration | Conditional on entity entry -- see below |
| Conditional zone | zone_duration | Flag `is_conditional_zone=1` in ability token |

**Conditional zones and combos:**

Conditional zones (trigger on entity entry) have high target variance because effect depends on AI movement decisions during the window. Flag them explicitly and expect the model to learn a probabilistic expectation: "given current entity positions, expected fraction will enter the zone." This is useful for planning even if imprecise.

Zone-tag combos (overlapping zones producing a third effect) are **excluded from Phase 1 training**. They require a two-ability operator and separate labeling logic. Add in Phase 2 once single-ability operator is solid.

---

## Part 3: Training Data Pipeline

### Dataset Construction

```python
def build_operator_dataset(sim_logs):
    samples = []

    for scenario in sim_logs:
        for tick, event in scenario.cast_events:
            ability = event.ability_def
            caster_slot = event.caster_slot

            # Skip combo zone events for Phase 1
            if is_combo_zone_event(event):
                continue

            window_ms = compute_window(ability)
            window_ticks = ms_to_ticks(window_ms)
            target_tick = tick + window_ticks

            if target_tick >= len(scenario.snapshots):
                continue

            state_before = scenario.snapshots[tick - 1]  # pre-cast
            state_after  = scenario.snapshots[target_tick]

            target_delta = compute_target_delta(
                state_before, state_after, caster_slot
            )

            samples.append({
                'state':          state_before,
                'ability_cls':    ability_transformer.encode(ability),  # frozen
                'ability_props':  extract_80dim_props(ability),
                'caster_slot':    caster_slot,
                'duration_norm':  window_ms / MAX_WINDOW_MS,
                'target_delta':   target_delta,
                'is_cond_zone':   is_conditional_zone(ability),
            })

    return samples
```

### Stratified Sampling

Sample proportionally to target variance, not uniform by ability type. High-variance effects (conditional zones, long-duration AoE) need more examples to average out concurrent-unit noise.

```python
# Measure per-ability-type target variance at dataset build time
variance_by_type = compute_target_variance(samples, group_by='ai_hint')

# Sample weight = sqrt(variance) -- soft upweighting, not hard oversampling
sample_weights = [sqrt(variance_by_type[s['ability'].ai_hint]) for s in samples]
```

### Dataset Size Expectations

Based on ~416 scenarios with heroes having 8 abilities each on ~5s cooldowns in ~30s fights:

| Ability type | Estimated cast events |
|-------------|----------------------|
| Damage (instant) | ~8,000-12,000 |
| Damage (zone/channel) | ~2,000-4,000 |
| Heal | ~1,000-3,000 |
| CC | ~2,000-4,000 |
| Utility/mobility | ~1,000-2,000 |
| Conditional zone | ~500-1,500 |
| **Total** | **~15,000-25,000** |

This is a small dataset. Full-dataset epochs are fine. Augment by:
- Random entity slot permutation (preserve caster identity, shuffle others)
- Small Gaussian noise on entity features (conditioning augmentation, GameNGen-style)
- Mirror positions (flip x-axis)

---

## Part 4: Training Setup

### Phase 1: Encoder + Operator Joint Training

Train encoder and operator end-to-end together. The ability transformer [CLS] embeddings remain **frozen throughout**.

```python
optimizer = AdamW(
    list(encoder.parameters()) + list(operator.parameters()),
    lr=1e-3, betas=(0.9, 0.98), weight_decay=1.0
)
# Grokfast EMA gradient filter
grokfast = GrokfastEMA(alpha=0.98, lamb=2.0)

scheduler = LinearWarmup(optimizer, warmup_steps=10, factor_range=(0.1, 1.0))
grad_clip = 1.0
```

### Loss

```python
def compute_loss(pred, target, ability_props):
    mask = loss_mask_for_ability(ability_props)
    loss = 0.0

    if mask['hp']:
        loss += beta_nll(pred.hp_mean, pred.hp_logvar, target.hp, beta=0.5)

    if mask['cc']:
        loss += beta_nll(pred.cc_mean, pred.cc_logvar, target.cc_remaining)
        loss += F.binary_cross_entropy_with_logits(pred.is_stunned, target.is_stunned)

    if mask['pos']:
        loss += beta_nll(pred.pos_mean, pred.pos_logvar, target.pos_delta)

    if mask['exists']:
        loss += F.binary_cross_entropy_with_logits(pred.exists_logit, target.exists)

    return loss
```

### Evaluation Metrics

Report against "predict no effect" baseline (predict zero delta for all features, exists=1 for all):

| Metric | Formula | Target |
|--------|---------|--------|
| hp_mae_improvement | (baseline_mae - model_mae) / baseline_mae | >25% |
| exists_bce_improvement | (baseline_bce - model_bce) / baseline_bce | >30% |
| cc_fidelity | % of CC events where model predicts cc_remaining > 0 | >70% |
| pos_mae_improvement | for mobility abilities only | >20% |
| per_type_breakdown | above metrics split by ai_hint | diagnostic |

### Phase 2: Operator Fine-tuning (Frozen Encoder)

Once joint training converges, freeze the encoder and fine-tune only the operator. This prevents RL transfer degradation from further encoder drift. Reduce learning rate 10x.

---

## Part 5: Autoresearch Integration

Structure for autonomous hyperparameter and architecture search following Karpathy's autoresearch pattern.

### File Structure

```
autoresearch/
  prepare.py          <- fixed: data loading, eval, baselines (agent never touches)
  train_encoder.py    <- Stage A: agent modifies
  train_operator.py   <- Stage B: agent modifies
  program_encoder.md  <- human writes search instructions for encoder
  program_operator.md <- human writes search instructions for operator
```

### program_encoder.md (template)

```markdown
# Encoder Research Program

## Objective
Maximize: hp_mae_improvement_pct + 2 * exists_bce_improvement_pct
Baseline = 0 (predict no change). Higher is better.
Training budget: [N] minutes. Report val score at end.

## Architecture search space
- n_layers: [2, 4, 6]
- d_ff: [64, 128, 256]
- n_heads: [4] (keep fixed -- d_model=32 constrains this)
- type_embedding_dim: [16, 32] (may differ from d_model, add projection)

## Loss search space
- hp group: beta value in [0.3, 0.5, 0.7]
- exists: BCE only, do not change
- state group weight: [0.0, 0.05, 0.1]
- symlog: try with and without on position features

## Optimizer search space
- lr: [5e-4, 1e-3, 2e-3]
- weight_decay: [0.1, 1.0, 2.0]
- grokfast lamb: [1.0, 2.0, 4.0]

## Known constraints (do not violate)
- cd group loss weight = 0.0 (phase transition noise poisons encoder)
- ability transformer weights frozen at all times
- exists prediction head uses BCE + sigmoid, not Gaussian
- resource_pct predicted separately from hp/shield (different dynamics)

## Known failure modes to watch for
- If hp_mae gets worse while train loss improves: cd leakage, check loss mask
- If exists_bce does not improve at all: check BCE vs entity mask alignment
- If val diverges from train after step 1000: curriculum expanding too fast,
  try slower delta range expansion
```

### Time Budget

5-minute budget is likely too short for meaningful convergence on this dataset. Use **15-minute budget** and run ~30 experiments overnight. Subsample to 20% of scenarios for the autoresearch phase -- relative ordering of approaches is stable on a subsample. Run top-3 configurations at full scale manually.

---

## Part 6: RL Integration

### Interface at Actor-Critic Time

With encoder frozen, the operator provides a differentiable single-ability simulator:

```python
def score_ability(state, ability, caster_slot):
    z = encoder(state)                                   # frozen
    z_after = operator(z, ability_cls, caster_slot,
                       duration_norm)                    # frozen or fine-tuned
    delta = decoder(z_after)

    # Actor uses delta to rank ability options
    # e.g. expected damage to nearest enemy
    return delta.hp[nearest_enemy_slot].mean * -1        # negative = damage

# Critic uses full latent state as before
value = critic(z.mean(dim=1))                            # pooled entity tokens
```

### What the Operator Adds vs Current Architecture

| Capability | Current (cross-attn) | With operator |
|-----------|---------------------|---------------|
| Ability scoring | Learned dot product | Explicit effect simulation |
| Interpretability | Black box logit | Predicted hp/cc/pos delta |
| Planning horizon | 1 step | Composable (chain operators) |
| CC awareness | Implicit | Explicit cc_remaining prediction |
| Position reasoning | None | Explicit delta_pos for mobility abilities |

The operator does not replace the cross-attention block in the actor -- it augments it. The actor still cross-attends ability [CLS] with entity tokens for urgency/target classification. The operator adds a second scoring signal: predicted effect magnitude and direction per entity.

---

## Open Questions

1. **Joint vs staged training:** if the encoder collapses under joint training, fall back to pre-training encoder on masked entity feature prediction first, then train operator with frozen encoder.

2. **Passive ability handling:** passives with `OnHpBelow` or `OnDamageTaken` triggers fire reactively during ability windows. They will appear as unexplained state changes in operator targets. Consider adding a `has_active_passive` flag per entity token as a conditioning signal.

3. **Morph abilities:** `morph_into` replaces the ability def after cast. The ability token at T+window is a different ability than at T. Flag `has_morph=1` in ability token and treat morph events as separate training samples (pre-morph and post-morph).

4. **Multi-ability operator (Phase 2):** zone-tag combos require a two-ability operator. Architecture extension: `[entity_0..6, ability_token_A, ability_token_B]`. The transformer can learn interaction between the two ability tokens. Requires explicit combo event logging.

5. **Autoregressive rollout stability:** for multi-step planning, chaining operators accumulates error. Apply conditioning augmentation (Gaussian noise on entity features at each step, GameNGen-style) during operator training to learn robustness to imperfect input states.

---

## Implementation Order

### Step 1: Dataset Generation (Rust)
- Add cast event logging to sim replay (tick, caster_slot, ability_def_id)
- Compute window per ability cast
- Export (state_before, state_after, ability_id, caster_slot, window_ms) tuples
- Convert to npz with ability [CLS] LUT

### Step 2: Encoder + Operator Model (Python)
- StateEncoder class (23-dim entities, 34-dim ability tokens, transformer)
- AbilityOperator class (latent tokens + ability token, transformer)
- Decoder heads (hp, cc, pos, exists)
- Loss with ability-type masking

### Step 3: Training Script
- Joint encoder+operator training
- bf16 autocast, GrokfastEMA, AdamW
- Eval against no-effect baseline
- Per-ability-type breakdown reporting

### Step 4: Validation
- Beat no-effect baseline on hp, exists, cc
- Verify attention patterns (entities attend to relevant ability token)
- Compare against current ability_eval micro-models

### Step 5: RL Integration
- Freeze encoder, export weights
- Add operator scoring signal to actor-critic
- Fine-tune actor with operator augmentation
