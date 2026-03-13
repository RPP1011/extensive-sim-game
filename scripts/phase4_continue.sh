#!/bin/bash
# Continue Phase 4 training indefinitely, warm-starting from previous checkpoint
set -euo pipefail

ITER=1
while true; do
    echo ""
    echo "=== Phase 4 continuation, iteration $ITER ==="
    uv run --with numpy --with torch training/pretrain_nextstate.py \
        generated/nextstate_hvh.npz \
        --max-steps 500000 --eval-every 10000 \
        --max-samples 300000 --use-abilities --seed $((42 + ITER)) \
        --batch-size 4096 --max-val-pairs 100000 \
        --lr 5e-4 \
        -o generated/entity_encoder_d128_d1_10.pt \
        --min-delta 1 --max-delta 10 \
        --warm-start generated/entity_encoder_d128_d1_10.pt
    ITER=$((ITER + 1))
done
