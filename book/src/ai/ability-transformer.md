# Ability Transformer

The ability transformer uses a **grokking-based cross-attention architecture** to
select optimal abilities. Unlike the urgency-based evaluator, the transformer
considers the full context of all entities and all available abilities
simultaneously.

## Module: `src/ai/core/ability_transformer/`

```
ability_transformer/
├── mod.rs          # Public API, inference
├── model.rs        # Transformer architecture
├── features.rs     # Token generation
├── weights.rs      # Weight loading
├── training.rs     # Training loop
├── dataset.rs      # Training data
└── eval.rs         # Evaluation
```

## Architecture

The model uses cross-attention between two token sequences:

1. **Entity tokens** — one per unit on the battlefield, encoding position, HP,
   team, role, current action, and status effects

2. **Ability tokens** — one per available ability on the acting unit, encoding
   ability type, cooldown state, range, effects, and tags

```
Entity tokens:  [hero1] [hero2] [enemy1] [enemy2] [enemy3]
                    ↕ cross-attention ↕
Ability tokens: [Q] [W] [E] [R] [basic_attack]
                    ↓
                [decision logits]
```

The cross-attention lets the model reason about which ability is best given
the current battlefield state. For example, if enemies are clustered, it
should favor AoE abilities; if an ally is dying, it should favor heals.

## Grokking

The model uses **grokking** — a phenomenon where neural networks suddenly
generalize long after memorizing training data. The training procedure:

1. Train on a dataset of (game state, optimal ability choice) pairs
2. Continue training well past memorization (high training accuracy)
3. Eventually, the model "groks" the underlying patterns and generalizes

Grokking-specific techniques:
- Weight decay to encourage generalization
- Grokfast (gradient filtering) to accelerate the grokking transition
- Long training schedules with patience

## Frozen Inference

At runtime, the transformer runs in **frozen** mode — weights are loaded once
and never updated during gameplay. This keeps inference fast and deterministic.

```rust
pub fn select_ability(
    state: &SimState,
    unit_idx: usize,
    weights: &TransformerWeights,
) -> Option<(usize, AbilityTarget)>
```

Returns the selected ability index and target, or `None` if the transformer
thinks holding is optimal.

## Integration

The transformer is optional — it is only active when weights are loaded. When
absent, the simpler ability evaluator + squad AI handles ability selection.

```rust
if let Some(weights) = &transformer_weights {
    if let Some(action) = select_ability(state, unit_idx, weights) {
        intents[unit_idx].action = IntentAction::UseAbility {
            ability_index: action.0,
            target: action.1,
        };
    }
}
```
