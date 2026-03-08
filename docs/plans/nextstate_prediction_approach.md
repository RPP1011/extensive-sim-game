# Next-State Prediction: Current Approach & Research Context

## Problem Statement

We're pre-training an entity encoder for a tactical combat game AI. The encoder must learn a useful representation of game state (unit positions, HP, cooldowns, combat stats) that transfers to downstream actor-critic RL.

**Previous approach (outcome prediction)** — predict fight winner from a single snapshot — failed. The encoder achieved 97.8% training accuracy but the learned representations were "worthless" for RL because:
- Fight outcome is too distant; many different trajectories lead to the same outcome
- The encoder learned shallow correlations (HP advantage → win) rather than tactical dynamics
- The critic (value function) trained on these representations couldn't attribute value to specific actions

**Current approach (next-state prediction)** — predict the full 30-dimensional entity feature vector at time T+Δ given the state at time T, where Δ is a short horizon (5–40 ticks). This forces the encoder to understand *how* the game state evolves: who takes damage, who heals, who dies, how positions change.

## Architecture

### Input Representation

Each game state snapshot contains:

| Token Type | Dim | Description |
|------------|-----|-------------|
| Entity (type 0–2) | 30 | Self/enemy/ally unit features |
| Threat (type 3) | 8 | Incoming projectiles, hostile zones, enemy casts |
| Position (type 4) | 8 | Points of interest (cover, chokepoints, elevated ground) |

**Entity features (30 dimensions):**
- Vitals (0–4): hp_pct, shield_pct, resource_pct, armor/200, magic_resist/200
- Position/terrain (5–11): x/20, y/20, dist_to_nearest_enemy/10, elevation, chokepoint_proximity, hostile_zones/3, friendly_zones/3
- Combat stats (12–14): auto_dps/30, attack_range/10, in_range_of_target (0/1)
- Ability stats (15–17): strongest_damage/500, ability_range/10, ability_cd_remaining_pct
- Heal stats (18–20): strongest_heal/500, heal_range/10, heal_cd_remaining_pct
- CC stats (21–23): control_duration/2000, control_cd_remaining_pct, (reserved)
- Current state (24–27): is_casting, cast_progress, is_stunned, has_shield
- Cumulative (28): damage_dealt_pct (cumulative damage / max_hp)
- Exists flag (29): 1.0 if alive, 0.0 if dead/absent

**Threat features (8 dimensions):** damage_ratio, dx, dy, distance, time_to_impact, is_aoe, aoe_radius, is_cc

**Position features (8 dimensions):** dx, dy, pathfinding_distance, elevation, chokepoint_proximity, wall_proximity, hostile_zones, friendly_zones

### Encoder

Standard transformer encoder (EntityEncoderV3):
- Input projections: separate linear layers for entities (30→d), threats (8→d), positions (8→d)
- Type embedding: learned embedding per token type, added to projected features
- LayerNorm on input tokens
- `nn.TransformerEncoder` with pre-norm, GELU, no dropout
- Output LayerNorm
- **Current hyperparams:** d_model=32, n_heads=4, n_layers=4, d_ff=64
- **Total params:** 40,158 (36,032 encoder + 4,126 prediction head)

### Prediction Head

```
state_head = Sequential(
    Linear(d_model + 1, d_model * 2),  # +1 for normalized delta
    GELU(),
    Linear(d_model * 2, 30),           # predict per-entity delta
)
```

**Residual/delta prediction:** The head predicts the *change* from current state, not absolute future state:
```python
state_delta = state_head(concat(entity_tokens, delta_normalized))
predicted_future = current_entity_features + state_delta
```
This gives a free baseline on stable features (armor, range, etc.) — the model only needs to learn what changes.

### Delta Conditioning

The prediction horizon Δ (in ticks) is normalized to [0, 1] by dividing by max_delta, then concatenated to each entity token before the prediction head. This tells the model *how far ahead* to predict.

## Training Setup

### Dataset

- **773K snapshots** from 416 scenarios (hand-crafted + generated)
- Sampled every tick during simulation (dense temporal coverage)
- Stored as `.npz` (numpy compressed arrays) — 9MB vs 2GB+ JSONL
- **Entity alignment via unit IDs:** Each entity slot tracks its unit_id. When pairing snapshots at T and T+Δ, entities are matched by unit_id to build targets (handles re-sorting by distance/HP).
- **Scenario-level train/val split** (354 train / 62 val scenarios) to prevent data leakage from correlated neighboring ticks

### Pairing Strategy

At each training step:
1. Sample a random snapshot (scenario, tick)
2. Sample a random Δ from current delta range
3. Find the snapshot at tick T+Δ in the same scenario
4. Match entities by unit_id to build 30-dim target vectors
5. Dead/absent units get target = zeros (hp=0, exists=0)

### Sliding Window Delta Schedule

To bias training toward easier (shorter horizon) predictions early:
- **First 25% of training:** delta range expands linearly from [5, 6] to [5, 40]
- **Remaining 75%:** full range [5, 40] ticks

Intuition: short-term predictions have simpler dynamics (fewer ability casts, less movement), letting the model learn basic patterns before tackling longer horizons.

### Optimization

- **AdamW:** lr=1e-3, β=(0.9, 0.98), weight_decay=1.0
- **Grokfast EMA:** alpha=0.98, lamb=2.0 (gradient filter that amplifies slow-varying components, accelerating generalization)
- **Small initialization:** std=0.007 (3x smaller than default)
- **No dropout, no early stopping** (consistent with grokking literature)
- **Batch size:** 256
- **Warmup:** 10-step linear LR warmup (0.1→1.0)
- **Gradient clipping:** max_norm=1.0

### Loss

MSE on all 30 features of real (non-padding) entities:
```python
valid = ~entity_mask  # True = real entity
loss = F.mse_loss(predicted[valid], target[valid])
```

### Evaluation

Compared against **"predict no change" baseline** (assume future = current):
- Overall MAE (model vs baseline)
- Per-feature-group MAE: hp, position, cooldowns, exists
- Reported as improvement % over baseline

## Key Design Decisions & Open Questions

### What's Working
1. **Residual prediction** — previous absolute prediction was -125% vs baseline; residual brought it to ~-11% (still training)
2. **Dense sampling** (every tick) with random Δ pairing gives maximum flexibility
3. **npz format** — 100x faster loading, 100x less memory than JSONL
4. **Streaming dataset generation** — callback-based writing avoids OOM on large scenarios

### Known Limitations
1. **All features weighted equally** — HP change matters much more than armor (which rarely changes). Loss is dominated by stable features.
2. **Single-step prediction** — no autoregressive rollout, no multi-step consistency
3. **No temporal encoding** — the model doesn't know absolute game time (early vs late fight)
4. **Delta as scalar** — single normalized value appended to each token. Could be more expressive.
5. **Entity-local predictions** — each entity predicted independently from its token. No explicit pairwise interaction in the prediction head (though the transformer encoder does attend across entities).
6. **No stochasticity modeling** — predicts mean state, doesn't capture variance/uncertainty
7. **Small model** — 40K params, d_model=32. May lack capacity for complex dynamics.

### Analogies to Other Domains

This problem has structural similarities to:
- **Video prediction:** Predicting future frames from current frames (but our "frames" are structured entity sets, not pixels)
- **Physics simulation:** Predicting particle/object states forward in time (GNNs, interaction networks)
- **Diffusion models for planning:** World models that predict future states for model-based RL
- **Point cloud prediction:** Variable-size unordered sets with per-element features evolving over time

### What We Want from Research

We're looking for techniques from image generation, video prediction, physics simulation, or world model research that could improve:
1. **Feature weighting** — How to focus learning on features that actually change
2. **Multi-scale temporal prediction** — Better ways to condition on prediction horizon
3. **Uncertainty/distributional prediction** — Model aleatoric uncertainty in outcomes
4. **Consistency over rollouts** — Autoregressive stability if we want to chain predictions
5. **Set prediction** — Better architectures for predicting evolution of variable-size entity sets
6. **Representation quality** — Pre-training objectives that produce better downstream RL representations

## Downstream Use

The encoder will be frozen and used as the backbone for an actor-critic RL agent:
- **Actor:** Cross-attention between ability tokens (from a separate DSL transformer) and entity tokens → action selection
- **Critic:** Pooled entity representation → scalar value estimate
- Entity encoder weights transfer via `encoder.*` prefix

Previous best RL result: 96.4% win rate on 28 attrition scenarios (PPO iteration 1, using V2 encoder pretrained on outcome prediction). The goal is to improve this with a dynamics-aware encoder.
