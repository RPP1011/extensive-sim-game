# Action-Conditional World Model

**Status:** Deferred — revisit if unconditional entity encoder doesn't yield sufficient performance.

## Idea

Turn the entity encoder from unconditional next-state prediction (S → S') into an action-conditional world model: (S, a) → S'. This lets the agent simulate "what happens if I take action X?" and compare futures.

## Why It Matters

- Model-based planning: evaluate multiple actions by predicting their consequences
- Better value estimation: critic imagines futures instead of estimating from one state
- MuZero-style rollouts: pick action whose predicted future scores best
- Action effects compound at longer deltas — synergizes with curriculum training

## Design

### Action Conditioning

- Condition on the **self entity's action only** (hero's chosen action)
- Enemy actions implicitly averaged over likely behavior (default AI in training data)
- Action space: 11 types (attack_nearest, attack_weakest, attack_focus, move_toward, move_away, hold, ability_0..7) — matches actor-critic action space

### Architecture Change

Minimal — add action embedding before prediction heads:

```
current:  entity_token (d_model) + delta (1) → prediction heads
proposed: entity_token (d_model) + delta (1) + action_emb (d_model) → prediction heads
```

Action embedding: `nn.Embedding(11, d_model)` or one-hot + linear.

### Dataset Changes

- `generate_nextstate_dataset_streaming` in `game_state.rs` must emit per-entity action labels
- Self-play system already tracks actions — extract from `IntentAction` at each tick
- Store as `action_ids: (n_samples, MAX_ENTS)` int8 in npz

### Training

- Same curriculum approach (delta 1 → 3 → 5 → 10)
- At delta=1, action effects are tiny (one tick of damage) — signal is weak
- At delta=5-10, action consequences compound — this is where conditioning pays off
- Could train both conditional and unconditional (action=-1 = unconditional) for flexibility

## Prerequisites

- Rust side: emit `IntentAction` per entity per tick in nextstate dataset
- Python side: action embedding in `EntityEncoderDecomposed`, new dataset field
- Regenerate nextstate dataset with action labels
