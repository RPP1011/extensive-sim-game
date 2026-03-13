#!/bin/bash
# IMPALA V4 Curriculum Training
# Phase 1: tier1 (108 scenarios, 10 iters)
# Phase 2: tier1+2 (148 scenarios, 10 iters)
# Phase 3: tier1-4 (272 scenarios, 10 iters)
# Phase 4: all (474 scenarios, 10 iters)
set -euo pipefail

OUT_DIR="generated/impala_curriculum"
CHECKPOINT="generated/actor_critic_v4_full_unfrozen.pt"
REGISTRY="generated/ability_embedding_registry.json"
LOG="/tmp/impala_curriculum.log"

COMMON_FLAGS=(
    --embedding-registry "$REGISTRY"
    --external-cls-dim 128
    --threads 64 --sims-per-thread 64
    --episodes-per-scenario 5
    --gpu --temperature 1.0 --batch-size 1024
    --train-epochs 1 --lr 1e-4
    --value-coeff 0.5 --entropy-coeff 0.01
    --reward-scale 1.0 --kl-coeff 0.0 --max-train-steps 0
    --eval-scenarios scenarios/hvh/ --eval-every 5
)

mkdir -p "$OUT_DIR"

run_phase() {
    local phase=$1
    local scenarios=$2
    local iters=$3
    local ckpt=$4

    echo "=== Phase $phase: $scenarios ($iters iters) ===" | tee -a "$LOG"
    echo "  Checkpoint: $ckpt" | tee -a "$LOG"

    PYTHONUNBUFFERED=1 uv run --with numpy --with torch training/impala_learner.py \
        --scenarios "$scenarios" \
        --checkpoint "$ckpt" \
        --output-dir "$OUT_DIR/phase${phase}" \
        --iters "$iters" \
        "${COMMON_FLAGS[@]}" \
        2>&1 | tee -a "$LOG"

    echo "  Phase $phase complete" | tee -a "$LOG"
}

echo "Starting IMPALA curriculum training $(date)" | tee "$LOG"

# Phase 1: tier1 only
run_phase 1 "dataset/scenarios/curriculum/phase1" 10 "$CHECKPOINT"

# Phase 2: tier1+2, from phase1 best
run_phase 2 "dataset/scenarios/curriculum/phase2" 10 "$OUT_DIR/phase1/best.pt"

# Phase 3: tier1-4, from phase2 best
run_phase 3 "dataset/scenarios/curriculum/phase3" 10 "$OUT_DIR/phase2/best.pt"

# Phase 4: all, from phase3 best
run_phase 4 "dataset/scenarios/curriculum/phase4" 10 "$OUT_DIR/phase3/best.pt"

echo "Curriculum complete $(date)" | tee -a "$LOG"
echo "Final checkpoint: $OUT_DIR/phase4/best.pt" | tee -a "$LOG"
