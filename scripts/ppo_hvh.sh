#!/bin/bash
# Iterative PPO for HvH (hero-vs-hero) performance
# Focus: beat the default AI in head-to-head
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_ITER=${MAX_ITER:-20}
EPISODES=${EPISODES:-10}
TEMP=${TEMP:-0.8}
MAX_TICKS=${MAX_TICKS:-1000}

REGISTRY="generated/ability_embedding_registry.json"
WEIGHTS_JSON="generated/actor_critic_weights_hvh.json"
WEIGHTS_PT="generated/actor_critic_hvh.pt"

# Initialize from BC model if no existing weights
if [ ! -f "$WEIGHTS_PT" ]; then
    echo "Initializing from BC model..."
    cp generated/actor_critic_hvh_bc.pt "$WEIGHTS_PT"
    cp generated/actor_critic_weights_hvh_bc.json "$WEIGHTS_JSON"
fi

ITER=1
while [ -f "generated/rl_hvh_iter${ITER}.jsonl" ]; do
    ITER=$((ITER + 1))
done
echo "Starting from iteration $ITER"

while [ "$ITER" -le "$MAX_ITER" ]; do
    echo ""
    echo "=== HvH PPO Iteration $ITER / $MAX_ITER ==="

    EP_FILE="generated/rl_hvh_iter${ITER}.jsonl"
    LOG_FILE="generated/actor_critic_hvh_iter${ITER}.csv"

    # Generate on-policy HvH episodes
    echo "--- Generate episodes ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl generate \
        scenarios/hvh/ \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        -o "$EP_FILE" \
        --episodes "$EPISODES" \
        --temperature "$TEMP" \
        --step-interval 3 \
        --max-ticks "$MAX_TICKS" \
        -j 0

    N_EP=$(wc -l < "$EP_FILE")
    N_WIN=$(grep -c '"Victory"' "$EP_FILE" || true)
    echo "  Episodes: $N_EP, Wins: $N_WIN ($(python3 -c "print(f'{$N_WIN/$N_EP*100:.1f}%')"))"

    # PPO training (on-policy with recomputed log probs)
    echo "--- PPO training ---"
    uv run --with numpy --with torch training/train_rl.py \
        "$EP_FILE" \
        --pretrained "$WEIGHTS_PT" \
        --embedding-registry "$REGISTRY" \
        -o "$WEIGHTS_PT" \
        --log "$LOG_FILE" \
        --ppo-epochs 2 \
        --batch-size 512 \
        --lr 3e-5 \
        --clip-eps 0.1 \
        --entropy-coeff 0.02 \
        --recompute-log-probs

    # Export
    echo "--- Export ---"
    uv run --with numpy --with torch training/export_actor_critic.py \
        "$WEIGHTS_PT" \
        -o "$WEIGHTS_JSON" \
        --external-cls-dim 128

    # Eval on HvH
    echo "--- Eval (HvH) ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        scenarios/hvh/ \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        --max-ticks 2000

    ITER=$((ITER + 1))
done
