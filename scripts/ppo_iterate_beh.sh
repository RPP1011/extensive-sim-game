#!/bin/bash
# Iterative PPO with behavioral embeddings
# Loop: generate episodes → train PPO → export JSON → eval on attrition
set -euo pipefail

cd "$(dirname "$0")/.."

# --- Config ---
MAX_ITER=${MAX_ITER:-20}
EPISODES_PER_SCENARIO=${EPISODES:-10}
TEMPERATURE=${TEMP:-1.0}
THREADS=${THREADS:-0}
STEP_INTERVAL=${STEP_INTERVAL:-3}
MAX_TICKS=${MAX_TICKS:-200}

# Paths
REGISTRY="generated/ability_embedding_registry.json"
WEIGHTS_JSON="${WEIGHTS_JSON:-generated/actor_critic_weights_fresh.json}"
WEIGHTS_PT="${WEIGHTS_PT:-generated/actor_critic_fresh.pt}"
ENTITY_ENC="generated/entity_encoder_pretrained_v3.pt"
PRETRAINED_DECISION="generated/ability_transformer_decision_v2.pt"
HVH_SCENARIOS="scenarios/hvh/"
ATTRITION_SCENARIOS="scenarios/"
# Train on both HvH and attrition for better generalization
TRAIN_SCENARIOS=("scenarios/hvh/" "scenarios/")

# Check prerequisites
for f in "$REGISTRY" "$WEIGHTS_JSON" "$WEIGHTS_PT"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

# Determine starting iteration from existing files
ITER=1
while [ -f "generated/rl_episodes_beh_iter${ITER}.jsonl" ]; do
    ITER=$((ITER + 1))
done
echo "Starting from iteration $ITER (found existing files up to iter $((ITER-1)))"

while [ "$ITER" -le "$MAX_ITER" ]; do
    echo ""
    echo "================================================================"
    echo "=== PPO Iteration $ITER / $MAX_ITER ==="
    echo "================================================================"

    EP_FILE="generated/rl_episodes_beh_iter${ITER}.jsonl"
    LOG_FILE="generated/actor_critic_beh_iter${ITER}.csv"

    # 1. Generate episodes from both HvH and attrition scenarios
    echo ""
    echo "--- Step 1: Generate episodes (${EPISODES_PER_SCENARIO}/scenario, temp=${TEMPERATURE}) ---"
    # Clear episode file
    > "$EP_FILE"
    for SCENARIO_DIR in "${TRAIN_SCENARIOS[@]}"; do
        echo "  Generating from $SCENARIO_DIR..."
        TEMP_EP="/tmp/ppo_beh_temp_${ITER}.jsonl"
        cargo run --release --bin xtask -- scenario oracle transformer-rl generate \
            "$SCENARIO_DIR" \
            --weights "$WEIGHTS_JSON" \
            --embedding-registry "$REGISTRY" \
            -o "$TEMP_EP" \
            --episodes "$EPISODES_PER_SCENARIO" \
            --temperature "$TEMPERATURE" \
            --step-interval "$STEP_INTERVAL" \
            --max-ticks "$MAX_TICKS" \
            -j "$THREADS"
        cat "$TEMP_EP" >> "$EP_FILE"
        rm -f "$TEMP_EP"
    done

    # Quick stats
    N_EP=$(wc -l < "$EP_FILE")
    N_WIN=$(grep -c '"Victory"' "$EP_FILE" || true)
    echo "  Total episodes: $N_EP, Wins: $N_WIN ($(echo "scale=1; $N_WIN * 100 / $N_EP" | bc)%)"

    # 2. Train PPO on episodes (on-policy: recompute log probs from model)
    echo ""
    echo "--- Step 2: PPO training ---"
    uv run --with numpy --with torch training/train_rl.py \
        "$EP_FILE" \
        --pretrained "$WEIGHTS_PT" \
        --embedding-registry "$REGISTRY" \
        -o "$WEIGHTS_PT" \
        --log "$LOG_FILE" \
        --ppo-epochs 4 \
        --batch-size 256 \
        --lr 3e-4 \
        --entropy-coeff 0.01 \
        --recompute-log-probs

    # 3. Export to JSON for Rust inference
    echo ""
    echo "--- Step 3: Export weights ---"
    uv run --with numpy --with torch training/export_actor_critic.py \
        "$WEIGHTS_PT" \
        -o "$WEIGHTS_JSON" \
        --external-cls-dim 128

    # 4. Evaluate on attrition scenarios
    echo ""
    echo "--- Step 4: Evaluate on attrition scenarios ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        "$ATTRITION_SCENARIOS" \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        --max-ticks 2000

    echo ""
    echo "=== Iteration $ITER complete ==="
    ITER=$((ITER + 1))
done

echo ""
echo "Done! Completed $MAX_ITER PPO iterations."
