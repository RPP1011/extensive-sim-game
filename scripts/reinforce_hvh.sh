#!/bin/bash
# Iterative REINFORCE for HvH
# Best recipe: BC on expert wins → REINFORCE 1 epoch (lr=5e-7) with unfrozen transformer
# Peak result: 54.4% HvH win rate from single REINFORCE step on expert data
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_ITER=${MAX_ITER:-10}
EPISODES=${EPISODES:-20}
TEMP=${TEMP:-0.8}
MAX_TICKS=${MAX_TICKS:-1000}
LR=${LR:-5e-7}
ENTROPY=${ENTROPY:-0.05}

REGISTRY="generated/ability_embedding_registry.json"
WEIGHTS_JSON="generated/actor_critic_weights_hvh.json"
WEIGHTS_PT="generated/actor_critic_hvh.pt"

# Initialize from best model if available, else BC model
if [ ! -f "$WEIGHTS_PT" ]; then
    if [ -f "generated/actor_critic_hvh_best.pt" ]; then
        echo "Initializing from best model..."
        cp generated/actor_critic_hvh_best.pt "$WEIGHTS_PT"
        cp generated/actor_critic_weights_hvh_best.json "$WEIGHTS_JSON"
    elif [ -f "generated/actor_critic_hvh_bc.pt" ]; then
        echo "Initializing from BC model..."
        cp generated/actor_critic_hvh_bc.pt "$WEIGHTS_PT"
        cp generated/actor_critic_weights_hvh_bc.json "$WEIGHTS_JSON"
    else
        echo "No model found. Run BC training first."
        exit 1
    fi
fi

ITER=1
while [ -f "generated/rl_reinforce_iter${ITER}.jsonl" ]; do
    ITER=$((ITER + 1))
done
echo "Starting from iteration $ITER"

while [ "$ITER" -le "$MAX_ITER" ]; do
    echo ""
    echo "================================================================"
    echo "=== REINFORCE Iteration $ITER / $MAX_ITER ==="
    echo "================================================================"

    EP_FILE="generated/rl_reinforce_iter${ITER}.jsonl"

    # 1. Generate on-policy episodes
    echo "--- Step 1: Generate episodes (temp=${TEMP}) ---"
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

    # 2. REINFORCE training (1 epoch, conservative lr, unfrozen transformer)
    echo "--- Step 2: REINFORCE (lr=${LR}) ---"
    uv run --with numpy --with torch training/train_rl.py \
        "$EP_FILE" \
        --pretrained "$WEIGHTS_PT" \
        --embedding-registry "$REGISTRY" \
        -o "$WEIGHTS_PT" \
        --log "generated/actor_critic_reinforce_iter${ITER}.csv" \
        --reinforce-epochs 1 \
        --ppo-epochs 0 \
        --batch-size 512 \
        --lr "$LR" \
        --entropy-coeff "$ENTROPY" \
        --unfreeze-transformer

    # 3. Export
    echo "--- Step 3: Export ---"
    uv run --with numpy --with torch training/export_actor_critic.py \
        "$WEIGHTS_PT" \
        -o "$WEIGHTS_JSON" \
        --external-cls-dim 128

    # 4. Eval on HvH
    echo "--- Step 4: Eval (HvH greedy) ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        scenarios/hvh/ \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        --max-ticks 2000

    echo "=== REINFORCE iteration $ITER complete ==="
    ITER=$((ITER + 1))
done
