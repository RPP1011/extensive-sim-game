#!/bin/bash
# DAgger-style iterative BC for HvH
# Loop: generate with current policy → filter wins → BC retrain → eval
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_ITER=${MAX_ITER:-20}
EPISODES=${EPISODES:-20}
TEMP=${TEMP:-0.8}
MAX_TICKS=${MAX_TICKS:-1000}
BC_EPOCHS=${BC_EPOCHS:-15}

REGISTRY="generated/ability_embedding_registry.json"
WEIGHTS_JSON="generated/actor_critic_weights_hvh.json"
WEIGHTS_PT="generated/actor_critic_hvh.pt"

# Initialize from BC model
if [ ! -f "$WEIGHTS_PT" ]; then
    echo "Initializing from BC model..."
    cp generated/actor_critic_hvh_bc.pt "$WEIGHTS_PT"
    cp generated/actor_critic_weights_hvh_bc.json "$WEIGHTS_JSON"
fi

# Accumulate winning episodes across iterations for richer training data
CUMULATIVE_WINS="generated/rl_dagger_cumulative_wins.jsonl"
if [ ! -f "$CUMULATIVE_WINS" ]; then
    # Start with original BC training data
    cp generated/rl_bootstrap_hvh_wins.jsonl "$CUMULATIVE_WINS"
fi

ITER=1
while [ -f "generated/rl_dagger_iter${ITER}.jsonl" ]; do
    ITER=$((ITER + 1))
done
echo "Starting DAgger from iteration $ITER"

while [ "$ITER" -le "$MAX_ITER" ]; do
    echo ""
    echo "================================================================"
    echo "=== DAgger Iteration $ITER / $MAX_ITER ==="
    echo "================================================================"

    EP_FILE="generated/rl_dagger_iter${ITER}.jsonl"

    # 1. Generate episodes with current policy
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
    echo "  Episodes: $N_EP, Wins: $N_WIN"

    # 2. Add winning episodes to cumulative dataset
    python3 -c "
import json
new_wins = 0
with open('$EP_FILE') as f:
    with open('$CUMULATIVE_WINS', 'a') as out:
        for line in f:
            ep = json.loads(line)
            if ep['outcome'] == 'Victory':
                out.write(line)
                new_wins += 1
total = sum(1 for _ in open('$CUMULATIVE_WINS'))
print(f'  Added {new_wins} wins, cumulative total: {total} episodes')
"

    # 3. BC retrain on cumulative winning episodes
    echo "--- Step 3: BC retrain (${BC_EPOCHS} epochs) ---"
    uv run --with numpy --with torch training/train_rl.py \
        "$CUMULATIVE_WINS" \
        --pretrained "$WEIGHTS_PT" \
        --embedding-registry "$REGISTRY" \
        -o "$WEIGHTS_PT" \
        --log "generated/actor_critic_dagger_iter${ITER}.csv" \
        --bc-epochs "$BC_EPOCHS" \
        --ppo-epochs 0 \
        --batch-size 512 \
        --lr 3e-4

    # 4. Export
    echo "--- Step 4: Export ---"
    uv run --with numpy --with torch training/export_actor_critic.py \
        "$WEIGHTS_PT" \
        -o "$WEIGHTS_JSON" \
        --external-cls-dim 128

    # 5. Eval on HvH
    echo "--- Step 5: Eval (HvH greedy) ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        scenarios/hvh/ \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        --max-ticks 2000

    echo "=== DAgger iteration $ITER complete ==="
    ITER=$((ITER + 1))
done
