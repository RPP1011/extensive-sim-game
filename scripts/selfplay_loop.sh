#!/usr/bin/env bash
# Self-play training loop: generate → train → evaluate → repeat.
# Usage: ./scripts/selfplay_loop.sh [stage] [iterations]
#   stage: move, kill, 2v2, 4v4 (default: kill)
#   iterations: number of iterations (default: 10)

set -euo pipefail

STAGE="${1:-kill}"
ITERS="${2:-10}"
EPISODES=2000
PYTHON="${PYTHON:-$HOME/Projects/lfm-agent/.venv/bin/python}"
OUTDIR="generated/selfplay"
SCENARIO_PATH="scenarios/attrition/"

mkdir -p "$OUTDIR"

POLICY=""
POLICY_FLAG=""

echo "=== Self-play loop: stage=$STAGE, iterations=$ITERS ==="

for i in $(seq 1 "$ITERS"); do
    echo ""
    echo "--- Iteration $i/$ITERS ---"

    EP_FILE="$OUTDIR/${STAGE}_ep_iter${i}.jsonl"
    NEW_POLICY="$OUTDIR/${STAGE}_policy_iter${i}.json"

    # Generate episodes
    echo "[gen] Generating episodes..."
    cargo run --release --bin xtask -- scenario oracle self-play generate \
        --stage "$STAGE" \
        "$SCENARIO_PATH" \
        --episodes "$EPISODES" \
        --temperature 0.8 \
        $POLICY_FLAG \
        --output "$EP_FILE" \
        2>&1 | grep -E "(Generated|Win rate)"

    # Train
    echo "[train] Training..."
    $PYTHON scripts/train_selfplay.py "$EP_FILE" \
        --output "$NEW_POLICY" \
        --epochs 50 \
        --lr 3e-4 \
        --l1 1e-5 \
        --entropy 0.02 \
        --hidden 64 \
        ${POLICY:+--resume "$POLICY"} \
        2>&1 | grep -E "(Epoch\s+(1|25|50)/|Dead|saved)"

    # Evaluate (greedy)
    echo "[eval] Evaluating..."
    cargo run --release --bin xtask -- scenario oracle self-play generate \
        --stage "$STAGE" \
        "$SCENARIO_PATH" \
        --episodes 200 \
        --temperature 0.3 \
        --policy "$NEW_POLICY" \
        --output /tmp/selfplay_eval.jsonl \
        2>&1 | grep -E "(Generated|Win rate)"

    POLICY="$NEW_POLICY"
    POLICY_FLAG="--policy $POLICY"
done

echo ""
echo "=== Done. Final policy: $POLICY ==="
