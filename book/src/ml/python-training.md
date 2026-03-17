# Python Training Scripts

The `training/` directory contains PyTorch-based training scripts for the V5
model architecture. These scripts are run with `uv` for dependency management.

## Directory Layout

```
training/
├── models/                    # Model architectures
│   ├── actor_critic_v5.py     # V5 actor-critic
│   ├── encoder_v5.py          # Entity encoder
│   ├── combat_head.py         # Combat action head
│   ├── cfc_cell.py            # Closed-form continuous RNN
│   └── latent_interface.py    # Latent space interface
│
├── Main training scripts
│   ├── train_rl_v5.py         # RL training (PPO/REINFORCE)
│   ├── train_bc_v5.py         # Behavior cloning
│   ├── curriculum_v5.py       # Curriculum learning
│   ├── impala_learner_v5.py   # IMPALA distributed learning
│   └── gpu_inference_server.py # GPU inference server
│
├── Pretraining
│   ├── pretrain_encoder_v5.py  # Entity encoder pretraining
│   ├── pretrain_latent_v5.py   # Latent space pretraining
│   ├── pretrain_spatial_v5.py  # Spatial representation
│   └── pretrain_temporal_v5.py # Temporal representation
│
├── Utilities
│   ├── export_actor_critic_v5.py  # Model export to NumPy
│   ├── convert_v5_npz.py         # Data format conversion
│   ├── probe_v5_data.py          # Data analysis
│   └── grokfast.py               # Grokking acceleration
│
├── eval/                      # Evaluation
│   ├── entity_encoder_eval.py
│   └── oracle_agreement.py
│
├── roomgen/                   # Room generation ML
│   ├── elite_dit.py           # Diffusion transformer
│   ├── flow_matching.py       # Flow matching training
│   ├── train.py               # Training script
│   ├── sample.py              # Generation
│   └── ...
│
└── data/
    └── create_holdout.py      # Train/test split
```

## Running Training

All Python scripts are run via `uv` (no virtualenv needed):

```bash
# RL training
uv run --with numpy --with torch python training/train_rl_v5.py \
    --episodes dataset/episodes/ \
    --algorithm ppo \
    --lr 3e-4 \
    --batch-size 256 \
    --epochs 100

# Behavior cloning
uv run --with numpy --with torch python training/train_bc_v5.py \
    --data dataset/oracle.jsonl \
    --epochs 50

# Entity encoder pretraining
uv run --with numpy --with torch python training/pretrain_encoder_v5.py \
    --data dataset/episodes/ \
    --epochs 30

# Curriculum learning
uv run --with numpy --with torch python training/curriculum_v5.py \
    --stages easy,medium,hard \
    --epochs-per-stage 20

# IMPALA distributed training
uv run --with numpy --with torch python training/impala_learner_v5.py \
    --actors 8 \
    --learner-lr 1e-4
```

## Data Pipeline

Training data flows from Rust simulation to Python training:

1. **Episode generation** — `cargo run --bin xtask -- scenario oracle dataset`
   runs scenarios and records episodes as `.npz` files

2. **Dataset creation** — `create_holdout.py` splits data into train/validation/test

3. **Training** — Python scripts load `.npz` files and train models

4. **Export** — `export_actor_critic_v5.py` saves weights as NumPy arrays

5. **Inference** — Rust loads `.npz` weights via `ndarray-npy` for runtime use

## Key Design Decisions

### NumPy as Exchange Format
Weights are exchanged between Python and Rust as `.npz` (NumPy archive) files.
This avoids ONNX complexity and keeps the format simple and inspectable.

### uv for Dependency Management
Using `uv run --with` avoids the need to maintain a virtualenv or requirements
file. Dependencies are resolved on-the-fly.

### Curriculum Learning
Training progresses through difficulty stages:
1. **Easy** — 2v2 with simple enemies
2. **Medium** — 4v4 with diverse enemies
3. **Hard** — 4v4 with intelligent enemies and complex abilities

This prevents the model from being overwhelmed by hard scenarios early in training.
