#!/usr/bin/env bash
# PPO iteration loop for v2 actor-critic.
# Usage: bash scripts/ppo_loop_v2.sh [start_iter] [num_iters]
#
# Requires:
#   - Built xtask binary (cargo build --release --bin xtask)
#   - Pretrained entity encoder at generated/entity_encoder_pretrained_v4.pt
#   - Pretrained ability transformer at generated/ability_transformer_pretrained_v3.pt

set -euo pipefail

START_ITER=${1:-1}
NUM_ITERS=${2:-10}
END_ITER=$((START_ITER + NUM_ITERS - 1))

# Use HvH scenarios for training, attrition for eval
TRAIN_SCENARIOS="scenarios/hvh/"
EVAL_SCENARIOS="scenarios/"
EPISODES_PER_SCENARIO=3
TEMPERATURE=1.0
STEP_INTERVAL=5
THREADS=0

ENTITY_ENCODER="generated/entity_encoder_pretrained_v4.pt"

# Weights file for Rust inference
WEIGHTS_JSON="generated/actor_critic_weights_v2_enc.json"
# Checkpoint for Python training
CHECKPOINT="generated/actor_critic_v2_enc.pt"

D_MODEL=64
D_FF=128
N_LAYERS=4
N_HEADS=4
EE_LAYERS=4

PPO_EPOCHS=4
BATCH_SIZE=256
LR=3e-4
ENTROPY_COEFF=0.05

echo "=== PPO v2 loop: iterations ${START_ITER}..${END_ITER} ==="
echo "  Train scenarios: ${TRAIN_SCENARIOS}"
echo "  Eval scenarios:  ${EVAL_SCENARIOS}"
echo "  Episodes/scenario: ${EPISODES_PER_SCENARIO}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Entropy coeff: ${ENTROPY_COEFF}"
echo ""

for ITER in $(seq "$START_ITER" "$END_ITER"); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ITERATION ${ITER}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EPISODES_FILE="generated/rl_episodes_v2_hvh_iter${ITER}.jsonl"

    # Step 1: Generate episodes on HvH scenarios
    echo "[iter${ITER}] Generating episodes on HvH scenarios..."
    cargo run --release --bin xtask -- scenario oracle transformer-rl generate \
        "${TRAIN_SCENARIOS}" \
        --weights "${WEIGHTS_JSON}" \
        --output "${EPISODES_FILE}" \
        --episodes "${EPISODES_PER_SCENARIO}" \
        --temperature "${TEMPERATURE}" \
        --step-interval "${STEP_INTERVAL}" \
        -j "${THREADS}"

    # Step 2: PPO training with higher entropy to prevent collapse
    echo "[iter${ITER}] PPO training..."
    uv run --with numpy --with torch training/train_rl_v2.py \
        "${EPISODES_FILE}" \
        --pretrained "${CHECKPOINT}" \
        --entity-encoder "${ENTITY_ENCODER}" \
        -o "${CHECKPOINT}" \
        --log "generated/actor_critic_v2_hvh_iter${ITER}.csv" \
        --d-model "${D_MODEL}" \
        --d-ff "${D_FF}" \
        --n-layers "${N_LAYERS}" \
        --n-heads "${N_HEADS}" \
        --entity-encoder-layers "${EE_LAYERS}" \
        --ppo-epochs "${PPO_EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --entropy-coeff "${ENTROPY_COEFF}"

    # Step 3: Export to JSON for Rust
    echo "[iter${ITER}] Exporting weights..."
    uv run --with numpy --with torch training/export_actor_critic_v2.py \
        "${CHECKPOINT}" \
        -o "${WEIGHTS_JSON}" \
        --d-model "${D_MODEL}" \
        --d-ff "${D_FF}" \
        --n-layers "${N_LAYERS}" \
        --n-heads "${N_HEADS}" \
        --entity-encoder-layers "${EE_LAYERS}"

    # Step 4: Quick eval on attrition scenarios
    echo "[iter${ITER}] Evaluating on attrition scenarios..."
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        "${EVAL_SCENARIOS}" \
        --weights "${WEIGHTS_JSON}"

    echo "[iter${ITER}] Done."
    echo ""
done

echo "=== PPO v2 loop complete (${NUM_ITERS} iterations) ==="
