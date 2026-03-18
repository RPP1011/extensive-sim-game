# Self-Play & RL

The self-play system trains AI agents by having them play against each other.
Both teams use the same policy network, generating training data from their
own experience.

## Module: `src/ai/core/self_play/`

```
self_play/
├── mod.rs        # Public API
├── trainer.rs    # Training loop
└── policy.rs     # Policy network integration
```

## Self-Play Loop

```
┌─────────────────────────────────────────┐
│                                         │
│  ┌──────────┐      ┌──────────┐        │
│  │ Policy A │      │ Policy B │        │
│  │ (hero)   │      │ (enemy)  │        │
│  └────┬─────┘      └────┬─────┘        │
│       │ intents          │ intents      │
│       └──────────┬───────┘              │
│                  ▼                      │
│         step(state, intents)            │
│                  │                      │
│         (state, events)                 │
│                  │                      │
│          ┌──────▼───────┐               │
│          │  Experience   │              │
│          │  Buffer       │──────────▶ Training
│          └──────────────┘               │
│                                         │
└─────────────────────────────────────────┘
```

Both Policy A and Policy B are copies of the same network (or slightly different
versions for league training). The experience from both sides is collected
into a buffer and used for gradient updates.

## Algorithms

### REINFORCE with Baseline
The simplest policy gradient method:
- Compute returns (discounted future rewards) for each step
- Subtract a baseline (value estimate) to reduce variance
- Update policy proportional to `advantage * log_prob(action)`

### PPO (Proximal Policy Optimization)
More stable training:
- Clip the policy ratio to prevent large updates
- Multiple gradient steps per batch of experience
- Value function loss combined with policy loss

### Gaussian Policy
For continuous action spaces (positioning):
- Policy outputs mean and log-variance for each action dimension
- Actions are sampled from the Gaussian
- Training uses the reparameterization trick

### Pointer Action Space
For discrete targeting:
- The policy scores each possible target using an attention mechanism
- Softmax over scores gives a probability distribution
- This naturally handles variable numbers of targets

## Reward Design

Reward signals extracted from `SimEvent`:

| Signal | Reward |
|--------|--------|
| Deal damage | +small per HP |
| Kill enemy | +large bonus |
| Heal ally | +medium per HP healed |
| Team wins | +large terminal bonus |
| Team loses | -large terminal penalty |
| Ally dies | -medium penalty |
| Unit takes avoidable damage | -small penalty |

Rewards are shaped to encourage good play even in losing games.

## Running Self-Play

```bash
# Rust-side (generate episodes)
cargo run --bin xtask -- scenario oracle transformer-rl generate scenarios/

# Python-side (train on episodes)
uv run --with numpy --with torch python training/train_rl_v5.py \
    --episodes dataset/episodes/ \
    --algorithm ppo \
    --lr 3e-4 \
    --epochs 100
```
