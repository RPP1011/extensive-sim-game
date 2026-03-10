# Integrating Next-State Encoder into Self-Play Actor-Critic

## Overview

The entity encoder pretrained on next-state prediction learns entity dynamics — how HP, positions, and alive/dead states evolve. The actor-critic uses the same encoder architecture (EntityEncoderV3) but trained from scratch via PPO. We can use the pretrained encoder as initialization to give the actor-critic a head start on understanding game dynamics.

## Architecture Compatibility

Both use identical structure:

| Component | Pretrained Encoder | Actor-Critic V3 |
|-----------|-------------------|-----------------|
| d_model | 32 | 32 |
| n_heads | 4 | 4 |
| n_layers | 4 | 4 |
| d_ff | 64 | 64 |
| entity_proj | Linear(32→32) | Linear(32→32) |
| threat_proj | Linear(8→32) | Linear(8→32) |
| position_proj | Linear(8→32) | Linear(8→32) |
| type_emb | Embedding(6, 32) | Embedding(5, 32) |

**One difference:** The pretrained encoder has 6 type embeddings (includes type=5 for ability tokens). The actor-critic V3 has 5 (entities, enemies, allies, threats, positions). The extra embedding is harmless — just ignore type=5 weights when loading.

## Integration Approach: Warm-Start + Differential LR

1. **Load pretrained encoder weights** into the actor-critic's entity encoder
2. **Fine-tune everything** with PPO, but use a lower learning rate for the encoder (e.g., 0.1× the head LR)
3. **No architecture changes** needed — same projections, same transformer layers

### Weight Mapping

Pretrained (`EntityEncoderDecomposed`) → Actor-Critic (`AbilityActorCriticV3`):

```
encoder.entity_proj.*   → entity_encoder.entity_proj.*
encoder.threat_proj.*   → entity_encoder.threat_proj.*
encoder.position_proj.* → entity_encoder.position_proj.*
encoder.type_emb.*      → entity_encoder.type_emb.* (first 5 rows)
encoder.input_norm.*    → entity_encoder.input_norm.*
encoder.encoder.*       → entity_encoder.encoder.* (transformer layers)
encoder.out_norm.*      → entity_encoder.out_norm.*
```

Prediction heads (`heads.*`, `log_task_vars`) are discarded — they're next-state specific.

### Ability projection

The pretrained encoder optionally has `encoder.ability_proj` (Linear(32→32)) for type=5 ability tokens. The actor-critic handles abilities differently — via cross-attention with the ability transformer's [CLS] embeddings. So `ability_proj` weights are discarded.

## Implementation

### Step 1: Add warm-start to PPO training script

In `training/train_rl_v3.py` (or whichever is current):

```python
parser.add_argument("--pretrained-encoder", type=str, default=None,
                    help="Path to pretrained entity encoder .pt (next-state prediction)")

# After model creation:
if args.pretrained_encoder:
    state = torch.load(args.pretrained_encoder, map_location=device, weights_only=True)
    model_state = model.state_dict()
    loaded = 0
    for k, v in state.items():
        if not k.startswith("encoder."):
            continue
        # Map pretrained encoder keys to actor-critic keys
        ac_key = k.replace("encoder.", "entity_encoder.", 1)
        if ac_key in model_state and model_state[ac_key].shape == v.shape:
            model_state[ac_key] = v
            loaded += 1
    model.load_state_dict(model_state)
    print(f"Loaded pretrained encoder: {loaded} params")
```

### Step 2: Differential learning rate

```python
encoder_params = list(model.entity_encoder.parameters())
encoder_ids = {id(p) for p in encoder_params}
head_params = [p for p in model.parameters() if id(p) not in encoder_ids]

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": args.lr * 0.1},
    {"params": head_params, "lr": args.lr},
], weight_decay=args.weight_decay)
```

### Step 3: No Rust changes needed

The exported JSON format is the same regardless of how the encoder was initialized. The Rust inference code (`ActorCriticWeightsV3`) works identically.

## Expected Benefits

- **Faster PPO convergence**: Encoder starts with useful features instead of random
- **Better value estimation**: Critic's pooled state already encodes entity dynamics
- **Position understanding**: Pretrained encoder learned pos prediction (+40-50% vs baseline)
- **Death prediction**: Encoder learned exists prediction (+60-70% vs baseline)

## What to Measure

1. **Win rate curve**: Compare PPO iterations with vs without pretrained encoder
2. **Convergence speed**: How many PPO iterations to reach 90%+ win rate
3. **Value loss**: Should start lower with pretrained encoder (better state representations)
4. **Per-group analysis**: Does the pretrained encoder help more in scenarios with complex movement or ability usage?

## Curriculum Checkpoint Selection

Use the final curriculum stage checkpoint (delta=[1,10]) for maximum horizon coverage. The encoder from this stage has seen prediction tasks from 1-10 ticks ahead, giving it the broadest understanding of dynamics.

## Alternative: Auxiliary Loss (if warm-start alone isn't enough)

Add next-state prediction as a secondary training objective during PPO:

```python
# During PPO step, also predict next-state for current observation
nextstate_pred = model.predict_nextstate(entities, threats, positions, delta=1)
nextstate_loss = compute_nextstate_loss(nextstate_pred, actual_next_state)
total_loss = ppo_loss + 0.1 * nextstate_loss
```

This prevents catastrophic forgetting of the encoder's dynamics knowledge during fine-tuning. Requires storing next-state targets in episode data (already available — consecutive RlSteps contain the future state).
