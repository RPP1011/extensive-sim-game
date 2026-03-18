# Burn Model Integration

[Burn](https://burn.dev) is a Rust-native ML framework that enables end-to-end
training and inference without leaving Rust. The V6 migration brings model
training into the Rust codebase, replacing the Python → NumPy → Rust export
pipeline.

## Module: `src/ai/core/burn_model/`

This module is behind feature flags:
- `burn-gpu` — GPU acceleration via the tch (LibTorch) backend
- `burn-cpu` — CPU computation via the ndarray backend

```
burn_model/
├── mod.rs              # Module root, public API
├── actor_critic.rs     # Actor-critic network
├── entity_encoder.rs   # Entity state encoder
├── combat_head.rs      # Combat action prediction head
├── training.rs         # Training loop
├── episode.rs          # Episode recording
├── dataset.rs          # Training data loading
├── export.rs           # Weight export
└── ...
```

## Why Burn?

| Concern | Python (PyTorch) | Rust (Burn) |
|---------|-----------------|-------------|
| Training speed | Fast (CUDA) | Comparable (tch backend) |
| Inference speed | Requires ONNX/TorchScript | Native, zero-copy |
| Deployment | Python runtime needed | Single binary |
| Determinism | Hard to guarantee | Inherits sim's determinism |
| Integration | IPC / file exchange | Direct function calls |

The key advantage is **integration**: Burn models can call simulation functions
directly, access `SimState` without serialization, and maintain the determinism
contract.

## Actor-Critic Architecture

```
            SimState
               │
        Entity Encoder
               │
         Entity Embeddings
          ╱          ╲
    Actor Head     Critic Head
       │                │
  Action logits    Value estimate
```

- **Actor** — produces probability distribution over actions
- **Critic** — estimates the value of the current state
- Both share the entity encoder (parameter sharing)

## Training with Burn

```bash
# GPU training
cargo run --features burn-gpu --bin xtask -- train-v6 \
    --data dataset/episodes/ \
    --epochs 50 \
    --lr 3e-4

# CPU training (slower but no GPU needed)
cargo run --features burn-cpu --bin xtask -- train-v6 \
    --data dataset/episodes/ \
    --epochs 50
```

## Episode Recording

The V6 system records episodes during simulation for offline training:

```rust
pub struct Episode {
    pub states: Vec<CompressedState>,
    pub actions: Vec<ActionRecord>,
    pub rewards: Vec<f32>,
    pub outcome: Outcome,
}
```

Episodes are serialized to disk and loaded in batches for training.

## Weight Interop

Weights can be transferred between Python (V5) and Rust (V6):

```bash
# Export V5 PyTorch weights to NumPy
uv run --with numpy --with torch python training/export_actor_critic_v5.py

# Convert to Burn format
cargo run --features burn-cpu --bin xtask -- convert-weights \
    --input weights/v5.npz \
    --output weights/v6.burn
```
