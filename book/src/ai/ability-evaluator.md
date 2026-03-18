# Ability Evaluator (Neural)

The ability evaluator is a neural network that estimates **urgency** — how
important it is for a unit to use a specific ability right now. It acts as an
interrupt layer in the AI pipeline, overriding the base intent when urgency
exceeds a threshold (0.4).

## Module: `src/ai/core/ability_eval/`

```
ability_eval/
├── mod.rs          # Public API
├── features.rs     # Feature extraction from SimState
├── model.rs        # Neural network architecture
├── inference.rs    # Urgency computation
├── training.rs     # Training data collection
└── eval.rs         # Evaluation metrics
```

## How It Works

1. **Feature Extraction** — for each unit, extract a feature vector from the
   current `SimState`:
   - Unit's HP, position, cooldowns
   - Nearby ally/enemy positions and states
   - Available abilities and their properties
   - Current casting states in the area

2. **Urgency Scoring** — the neural network produces a score for each ability:
   - `urgency = model(features)` → `[0.0, 1.0]` per ability

3. **Threshold Check** — if `max(urgency) > 0.4`, the evaluator overrides the
   current intent with a `UseAbility` intent for the highest-urgency ability

## Example Scenarios

The evaluator learns to recognize situations like:

- **Enemy casting a big spell** → urgency spike for interrupt/stun abilities
- **Ally about to die** → urgency spike for healing/shielding abilities
- **Enemies clustered** → urgency spike for AoE damage abilities
- **About to be focused** → urgency spike for defensive abilities (dash, shield)

## Training

The evaluator is trained on labeled data from oracle games and self-play:

- Positive examples: situations where using an ability led to a good outcome
- Negative examples: situations where holding the ability was correct
- The model learns a mapping from game state features to ability urgency scores

## Integration

The evaluator plugs into the intent pipeline after squad AI / GOAP:

```rust
if let Some(urgent_intent) = ability_evaluator.evaluate(state, unit_id) {
    // Override the base intent
    intents[unit_idx] = urgent_intent;
}
```

It only fires when it detects a high-urgency situation, so most ticks it
passes through without modification.
