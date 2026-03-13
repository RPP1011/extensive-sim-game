#!/bin/bash
# V4 Dual-Head Curriculum Training
# Phase 1: Tier1 (movement + attack only, no abilities)
# Phase 2: Tier2 (add single ability per hero)
# Phase 3: Full HvH (complex kits, all abilities)
set -e

REGISTRY="generated/ability_embedding_registry.json"
PRETRAINED="generated/actor_critic_v3_ph.pt"
CLS_DIM=128

echo "=== Phase 1: Tier1 (movement + attack targeting) ==="
uv run --with numpy --with torch -- python3 training/train_rl_v4.py \
  generated/rl_v4_phase1_t1.jsonl \
  --pretrained "$PRETRAINED" \
  --embedding-registry "$REGISTRY" \
  --external-cls-dim $CLS_DIM \
  --bc-epochs 20 \
  --batch-size 512 \
  --lr 3e-4 \
  --move-weight 3.0 \
  --freeze-transformer \
  --smart-sample \
  -o generated/actor_critic_v4_p1.pt

echo ""
echo "=== Phase 2: Tier1 + Tier2 (accumulated, prevents forgetting) ==="
uv run --with numpy --with torch -- python3 training/train_rl_v4.py \
  generated/rl_v4_phase2_combined.jsonl \
  --pretrained generated/actor_critic_v4_p1.pt \
  --embedding-registry "$REGISTRY" \
  --external-cls-dim $CLS_DIM \
  --bc-epochs 15 \
  --batch-size 512 \
  --lr 1e-4 \
  --move-weight 3.0 \
  --freeze-transformer \
  --smart-sample \
  -o generated/actor_critic_v4_p2.pt

echo ""
echo "=== Phase 3: Tier1 + Tier2 + Full HvH (accumulated) ==="
uv run --with numpy --with torch -- python3 training/train_rl_v4.py \
  generated/rl_v4_phase3_combined.jsonl \
  --pretrained generated/actor_critic_v4_p2.pt \
  --embedding-registry "$REGISTRY" \
  --external-cls-dim $CLS_DIM \
  --bc-epochs 10 \
  --batch-size 512 \
  --lr 5e-5 \
  --move-weight 5.0 \
  --freeze-transformer \
  --smart-sample \
  -o generated/actor_critic_v4_p3.pt

echo ""
echo "=== Done ==="
echo "Checkpoints: actor_critic_v4_p1.pt (move), _p2.pt (+ability), _p3.pt (full)"
