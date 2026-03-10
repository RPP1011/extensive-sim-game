#!/bin/bash
# Curriculum training for entity encoder at d_model=128
# Phase 1: delta=[1,1] → Phase 2: [1,3] → Phase 3: [1,5] → Phase 4: [1,10]
# Each phase: 500K steps, warm-starting from previous

set -euo pipefail

COMMON="uv run --with numpy --with torch training/pretrain_nextstate.py \
  generated/nextstate_hvh.npz \
  --max-steps 500000 --eval-every 10000 \
  --max-samples 500000 --use-abilities --seed 42 \
  --batch-size 1024 --max-val-pairs 100000"

echo "=== Phase 1: delta=[1,1] ==="
$COMMON -o generated/entity_encoder_d128_d1.pt --min-delta 1 --max-delta 1

echo ""
echo "=== Phase 2: delta=[1,3] ==="
$COMMON -o generated/entity_encoder_d128_d1_3.pt --min-delta 1 --max-delta 3 \
  --warm-start generated/entity_encoder_d128_d1.pt

echo ""
echo "=== Phase 3: delta=[1,5] ==="
$COMMON -o generated/entity_encoder_d128_d1_5.pt --min-delta 1 --max-delta 5 \
  --warm-start generated/entity_encoder_d128_d1_3.pt

echo ""
echo "=== Phase 4: delta=[1,10] ==="
$COMMON -o generated/entity_encoder_d128_d1_10.pt --min-delta 1 --max-delta 10 \
  --warm-start generated/entity_encoder_d128_d1_5.pt

echo ""
echo "=== Curriculum complete ==="
