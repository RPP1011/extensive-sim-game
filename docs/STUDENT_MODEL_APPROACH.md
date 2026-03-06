# Student Model: Tactical Combat AI via Oracle Distillation

## Problem

We have a real-time tactical combat game (Bevy/Rust) where teams of heroes fight enemies. The default AI uses hand-tuned heuristic scoring (personality weights, squad blackboard, utility functions). We want a learned model that makes better per-unit tactical decisions.

## Approach: Rollout Oracle + Supervised Distillation

### 1. Oracle (Teacher)

For each unit at each game tick, we enumerate candidate actions and score them via short simulation rollouts:

- **Candidate enumeration**: For a given unit, generate ~15-30 candidate actions:
  - Attack top-3 nearest enemies + squad focus target (range-gated)
  - Cast damage/CC/heal abilities on valid targets (range + resource gated)
  - Use hero abilities (8 slots, cooldown + resource gated)
  - MoveTo: toward nearest enemy, away from nearest enemy, toward weakest ally (for healers)
  - Hold (baseline)

- **Rollout scoring**: For each candidate:
  1. Clone the full game state
  2. Override the unit's action to the candidate on tick 0
  3. Run 10 ticks of simulation (all other units use the default AI)
  4. Score = (enemy HP lost - ally HP lost) + kill bonus + CC value

- **Optimizations**:
  - Rayon parallel rollouts across candidates
  - Early exit: if a candidate is >50 score behind the best after tick 20, abort
  - Range gating: skip attacks on targets > 3x attack range
  - Resource gating: skip abilities the unit can't afford

- **Label selection**: The highest-scoring candidate becomes the training label. The oracle also uses this action to drive the game forward (oracle-played episodes, not default-AI episodes).

### 2. Feature Extraction (115 features)

Per-unit features extracted from game state + squad AI state:

| Block | Count | Features |
|-------|-------|----------|
| Self state | 10 | HP%, shield%, speed, range, damage, attack CD fraction, cast state, cast remaining, resource fraction, channeling |
| Per-ability (8 slots) | 40 | CD fraction, hint encoding (damage/heal/CC/defense/utility), range, is_AoE, affordable+ready |
| Self status effects | 9 | DoT, HoT, silenced, rooted, stealthed, reflect, lifesteal, blind, shield_buff |
| Top-3 enemies | 24 | distance, HP%, DPS, is_healer, is_CC'd, is_focus_target, has_reflect, has_stealth |
| Weakest ally | 3 | HP%, distance, is_healer |
| Team context | 12 | ally/enemy counts, avg HP, role composition (tank/healer/ranged fractions), team spread, centroid distance, healer positioning, enemies in range, threats |
| Squad coordination | 10 | formation mode, on_focus_target, allies_on_same_target, allies_on_focus, self role, nearest ally dist, frontline, enemy healer fraction, nearest enemy DPS, is_CC'd |
| Environment | 5 | cover, elevation, elev advantage, hostile/friendly zones |
| Game phase | 2 | tick progress, numeric advantage |

All features normalized to roughly [0, 1] range.

### 3. Action Classes (10)

Actions are abstracted into 10 classes to decouple from specific target IDs:

0. AttackNearest
1. AttackWeakest
2. UseDamageAbility
3. UseHealAbility
4. UseCcAbility
5. UseDefenseAbility
6. UseUtilityAbility
7. MoveToward (toward nearest enemy)
8. MoveAway (away from nearest enemy)
9. Hold

### 4. Dataset Generation

- **Scenarios**: 779 TOML scenario files (751 training + 28 attrition) across:
  - Team sizes: 2v3 through 6v8
  - Hero compositions: balanced, dual-healer, all-ranged, boss fights, random
  - Difficulty 1-5, HP multipliers 1x-5x
  - Room types: Entry, Pressure, Recovery, Climax
  - 3 seed variants per composition for trajectory diversity

- **Oracle-played episodes**: Each scenario runs with the oracle making hero decisions. At each tick, for each living hero not casting/CC'd, the oracle runs rollouts, picks the best action, and we record (features, label). This means the dataset reflects oracle-quality play, not default-AI play.

- **Dataset size**: ~485K samples (382K training scenarios + 26K attrition × 4 oversampling)

- **Class distribution** (combined):
  - AttackNearest: 33%, DamageAbility: 42%, CcAbility: 11%
  - MoveAway: 3%, MoveToward: 2%, HealAbility: 3.7%, UtilityAbility: 3.8%
  - Hold: 0.2%, AttackWeakest: 0.5%, DefenseAbility: 1%

### 5. Training

- **Architecture**: MLP with variable depth. Currently testing:
  - 115→256→128→10 (2 hidden, ~34K params)
  - 115→256→128→64→10 (3 hidden, ~50K params)
  - 115→256→128→64→32→10 (4 hidden, ~52K params)

- **Loss**: Cross-entropy with inverse-frequency class weights
- **Optimizer**: AdamW with weight decay (1e-4)
- **Schedule**: Linear warmup (10 epochs) + cosine annealing
- **Regularization**: Dropout (0.05-0.15), weight decay
- **Training**: 300 epochs, batch size 512, lr 5e-4

### 6. Inference at Runtime

Exported model weights (JSON) are loaded in Rust. Forward pass is a simple sequence of matrix multiply + ReLU (no framework dependency). At each tick, for each hero unit:
1. Extract 115 features from game state
2. Forward pass through MLP → 10 logits
3. Argmax → action class
4. Map action class back to concrete game action (e.g., AttackNearest → Attack{target_id: nearest_enemy_id})

### 7. Evaluation

Models are evaluated on two scenario sets:
- **Attrition** (28 scenarios): 3 warriors + 1 specialist vs 6 enemies, difficulty 2, 5x HP. Hard endurance fights.
- **Training** (553 unique scenarios): Diverse compositions and conditions.

### Results

| Model | Features | Data | Params | Val Acc | Attrition (28) | Training (553) |
|-------|----------|------|--------|---------|----------------|----------------|
| Default AI | — | — | — | — | 7 (25%) | — |
| v2 old | 32 | 22K | 4.5K | 49.3% | 20 (71.4%) | 98/191 (51%) |
| v2 balanced | 48 | 485K | 15K | 77.0% | 25 (89.3%) | 307 (55.5%) |
| v2 115-feat (training...) | 115 | 485K | ~50K | TBD | TBD | TBD |

## Known Limitations / Open Questions

1. **Oracle horizon**: 10-tick rollouts (~1 second of game time). The oracle can't plan multi-step strategies (e.g., "kite for 3 seconds then re-engage"). Deeper rollouts (100 ticks) were tested but the single-tick action override gets drowned out by 99 ticks of default-AI behavior, causing Hold to dominate labels at 90%+.

2. **Action class granularity**: 10 classes may be too coarse. "AttackNearest" doesn't distinguish between "attack the nearest warrior" vs "attack the nearest healer who happens to be nearest." The model learns WHEN to attack nearest but not target prioritization within that class.

3. **No sequence modeling**: Each decision is independent — no memory of what the unit did last tick. An RNN/transformer could learn temporal patterns like ability combos or kiting cycles.

4. **Oracle quality**: The oracle uses default AI for all non-oracle units during rollouts. If the default AI makes bad plays (e.g., ignoring healers), the oracle's scoring is distorted because it assumes teammates will play at default-AI level.

5. **Single-agent oracle**: Each unit is scored independently. The oracle doesn't consider joint actions (e.g., "if unit A focuses healer AND unit B CCs tank, the combined outcome is better"). This limits coordinated play.

6. **Feature engineering vs learned representations**: The 115 hand-crafted features encode our assumptions about what matters. A learned representation (e.g., attention over all units) could discover patterns we didn't anticipate.

7. **Oversampling bias**: Attrition scenarios are oversampled 4× to improve performance on hard fights. This might bias the model toward attrition-style play in non-attrition scenarios.

## Potential Improvements

- **Monte Carlo Tree Search** instead of 1-step rollouts for deeper planning
- **Multi-agent scoring** (enumerate joint actions for 2+ units simultaneously)
- **Temporal modeling** (feed last N actions/states, use RNN or small transformer)
- **Curriculum learning** (train on easy scenarios first, progressively harder)
- **Self-play** (use the student model for rollout partners instead of default AI)
- **Attention-based architecture** over unit embeddings instead of fixed feature vector

---

## Ability Embedding Autoencoder

### Motivation

The self-play policy (`src/ai/core/self_play.rs`) originally encoded each ability as just 4 features: `[ready, cd_fraction, range_norm, is_aoe]`. This discards almost all semantic information — a 200-damage nuke and a minor heal with similar range/cooldown look identical to the policy. When new abilities are added (e.g., from LoL champion imports), the policy has no way to understand what they do.

### Architecture

An autoencoder trained via **supervised contrastive learning + reconstruction**:

```
Encoder:  80 (ability properties) → 64 (hidden, ReLU) → 32 (L2-normalized embedding)
Decoder:  32 (embedding) → 64 (hidden, ReLU) → 80 (reconstructed properties)
Total: 14,576 parameters (7,264 encoder + 7,312 decoder)
```

The 80-dimensional input property vector is extracted from `AbilityDef` TOML fields:

| Feature Group | Dims | Contents |
|---------------|------|----------|
| Targeting | 8 | One-hot targeting mode (TargetEnemy, TargetAlly, SelfCast, etc.) |
| Core stats | 6 | Cooldown, range, cast time, resource cost, channel duration, delivery method |
| Delivery | 13 | One-hot delivery (Instant, Projectile, Channel, Zone, Tether, Trap, Chain) + delivery params |
| Mechanics | 5 | Number of effects, has conditions, has triggers, has sub-delivery, max bounces |
| AI hints | 6 | Damage hint, heal hint, CC hint, defense hint, utility hint, summon hint |
| Damage type | 3 | Physical, magical, true damage fractions |
| Damage | 4 | Total damage, max single-hit, DoT amount, execute threshold |
| Healing | 3 | Total healing, HoT amount, shield amount |
| Hard CC | 7 | Stun, root, silence, fear, knockback, pull, taunt durations |
| Soft CC | 4 | Slow amount, slow duration, blind, disarm |
| Other CC | 4 | Duel, swap, transform, vulnerability |
| Mobility | 3 | Dash distance, has dash, movement speed buff |
| Buffs | 4 | Buff count, debuff count, lifesteal, reflect |
| Area | 5 | AoE radius, zone duration, is persistent, tether range, chain targets |
| Special | 5 | Summon, stealth, dispel, rewind, terrain interaction |

Property extraction walks all `ConditionalEffects` including delivery sub-effects (`on_hit`, `on_arrival`, `on_complete`).

### Training

**Loss function**: Joint supervised contrastive + reconstruction MSE:
```
loss = loss_supcon + recon_weight × loss_recon
```

- **SupCon component**: 70% category-only labels (9 classes: DamageUnit, DamageAoe, CcUnit, HealUnit, HealAoe, Defense, Utility, Summon, Obstacle) + 30% category×targeting combined labels (72 classes). Temperature τ=0.1.
- **Reconstruction component**: MSE between input properties and decoder output.

The SupCon loss clusters abilities by function; the reconstruction loss preserves quantitative detail (damage values, CC durations, etc.) that pure contrastive learning would discard.

**Pipeline**:
```bash
# 1. Export training data from all TOML templates
cargo run --release --bin xtask -- scenario oracle ability-encoder-export \
    --output generated/ability_encoder_data.json

# 2. Train autoencoder
uv run --with numpy --with torch scripts/train_ability_encoder.py \
    generated/ability_encoder_data.json -o generated/ability_encoder.json

# 3. Use in self-play (encoder frozen within policy)
cargo run --release --bin xtask -- scenario oracle self-play generate scenarios/ \
    --ability-encoder generated/ability_encoder.json --episodes 20
```

**Results** (856 abilities from 193 heroes):
- kNN@1: 99.4%, kNN@3: 99.9%, kNN@5: 99.9%
- Reconstruction MSE: 0.035
- All 9 categories achieve 100% kNN@5 accuracy

### Integration with Self-Play

The encoder is frozen and embedded in the self-play feature extraction. Each ability slot expands from 4 features to 34 features (32-dim embedding + ready + cd_fraction):

| Encoding | Per-ability | 8 slots | Total features |
|----------|------------|---------|----------------|
| Legacy | 4 | 32 | 311 |
| Encoded | 34 | 272 | **919** |

Key functions in `src/ai/core/self_play.rs`:
- `extract_features_encoded(state, unit_id, encoder)` — full encoder-aware feature extraction
- `FEATURE_DIM_ENCODED = 919` — new feature dimension

The encoder weights are loaded once at startup via `AbilityEncoder::load()` from JSON, then called per-ability during feature extraction. No gradient computation at inference time.

### Decoder Uses

The decoder enables:
- **Reconstruction quality checking**: Verify the embedding preserves ability semantics
- **Embedding interpolation**: Blend two ability embeddings and decode to inspect intermediate properties
- **Debugging**: Decode an embedding back to properties to understand what the policy "sees"

### Key Files

| File | Purpose |
|------|---------|
| `src/ai/core/ability_encoding.rs` | Property extraction, encoder/decoder, training data export |
| `scripts/train_ability_encoder.py` | Python training script (SupCon + reconstruction) |
| `generated/ability_encoder.json` | Trained encoder + decoder weights |
| `generated/ability_encoder_data.json` | Exported training data (856 abilities × 80 features) |
| `generated/ability_encoder_embeddings.json` | Pre-computed embeddings for visualization |
