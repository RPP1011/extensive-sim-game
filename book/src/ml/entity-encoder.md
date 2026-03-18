# Entity Encoder

The entity encoder transforms raw `UnitState` data into fixed-size embedding
vectors suitable for neural network consumption. It is the first stage in all
ML models — every downstream component (actor-critic, transformer, evaluator)
consumes entity embeddings.

## Purpose

Raw `UnitState` has variable-length fields (abilities, status effects) and
mixed types (positions, HP, booleans). The entity encoder:

1. Normalizes all fields to `[0, 1]` or `[-1, 1]` ranges
2. Handles variable-length sequences (abilities, statuses) via padding/pooling
3. Produces a fixed-size embedding vector per entity

## Feature Groups

| Group | Fields | Encoding |
|-------|--------|----------|
| Vitals | HP, max_hp, shield_hp | Ratio (hp/max_hp) |
| Position | x, y, elevation | Normalized to arena bounds |
| Combat | attack_damage, range, cooldowns | Log-scaled |
| Defenses | armor, magic_resist, cover | Sigmoid-normalized |
| Abilities | per-slot: type, cooldown, tags | Pooled embedding |
| Status | active effects, durations | Binary + duration |
| Team | hero/enemy, role | One-hot |
| Casting | is_casting, cast_type, progress | Categorical + ratio |

## Pretraining

The entity encoder can be pretrained separately before being used in the full
actor-critic model:

```bash
uv run --with numpy --with torch python training/pretrain_encoder_v5.py \
    --data dataset/episodes/ \
    --epochs 50 \
    --output weights/encoder_v5.pt
```

Pretraining tasks:
- **Next-state prediction** — predict entity state at t+1 given state at t
- **Masked entity prediction** — reconstruct a masked entity from context
- **Outcome prediction** — predict win/loss from initial state

## V5 Architecture (Python)

```python
class EntityEncoder(nn.Module):
    def __init__(self, d_model=128):
        self.vitals_proj = nn.Linear(4, 32)
        self.position_proj = nn.Linear(3, 32)
        self.combat_proj = nn.Linear(8, 32)
        self.ability_pool = nn.Sequential(
            nn.Linear(ability_dim, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.combiner = nn.Linear(32*3 + 64 + 16, d_model)
```

## V6 Architecture (Burn)

The V6 migration reimplements the entity encoder in Burn for end-to-end
Rust training:

```
src/ai/core/burn_model/
├── entity_encoder.rs  # Burn implementation
└── ...
```

The Burn version matches the Python architecture weight-for-weight, enabling
seamless weight transfer.
