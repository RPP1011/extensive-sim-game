# ML Training Pipeline

The machine learning pipeline teaches AI agents to play through a combination
of behavior cloning, self-play reinforcement learning, and distributed training.
The pipeline spans two languages: **Rust** (simulation, feature extraction,
inference) and **Python** (model training, experiment management).

## Architecture

```
                    ┌──────────────────┐
                    │  Rust Simulation  │
                    │  (deterministic)  │
                    └────────┬─────────┘
                             │ episodes, features
                    ┌────────▼─────────┐
                    │  Dataset (.npz)   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼───┐  ┌──────▼──────┐  ┌───▼──────────┐
    │  Behavior   │  │  Self-Play  │  │   IMPALA      │
    │  Cloning    │  │  RL (PPO)   │  │  Distributed  │
    └─────────┬───┘  └──────┬──────┘  └───┬──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │ trained weights
                    ┌────────▼─────────┐
                    │  Export (.npz)    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Rust Inference   │
                    │  (Burn / ndarray) │
                    └──────────────────┘
```

## Training Approaches

### 1. Behavior Cloning (BC)
Learn from expert demonstrations (oracle games):
- Record oracle policy decisions during simulation
- Train a supervised model to predict oracle actions
- Fast to train, gives a reasonable starting policy

### 2. Self-Play RL
Improve beyond the oracle through experience:
- REINFORCE with baseline for policy gradients
- PPO for stable training with clipped objectives
- Gaussian policy for continuous action spaces
- Pointer action space for discrete target selection

### 3. IMPALA Distributed Training
Scale up with distributed actors:
- Multiple simulation instances generate experience in parallel
- Central learner processes batches from all actors
- V-trace importance correction for off-policy data
- Achieved 22.5% → 26.2% win rate improvement in experiments

## Model Versions

The project has gone through multiple model architecture versions:

| Version | Architecture | Status |
|---------|-------------|--------|
| V1-V4 | Various (deprecated) | Removed |
| V5 | Actor-critic + entity encoder | Current (Python) |
| V6 | Burn-based actor-critic | In progress (Rust) |

V5 is the current production architecture, trained in Python and exported
to NumPy arrays for Rust inference. V6 is the Burn migration, enabling
end-to-end Rust training.

## Key Directories

```
training/               # Python training code
├── models/             # Model architectures
├── roomgen/            # Room generation ML
├── eval/               # Evaluation scripts
└── data/               # Data utilities

src/ai/core/
├── ability_eval/       # Urgency evaluator
├── ability_transformer/ # Grokking transformer
├── burn_model/         # Burn models (V6)
└── self_play/          # RL training loop (Rust)
```
