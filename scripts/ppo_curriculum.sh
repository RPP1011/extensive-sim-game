#!/usr/bin/env bash
# Curriculum PPO: start with easy scenarios, expand as model improves.
# Usage: bash scripts/ppo_curriculum.sh [stage] [start_iter] [num_iters]
#
# Stages:
#   0 (basic)    — Target dummies, punching bags, kiting, ability usage, focus fire
#   1 (zones)    — Zone avoidance, max range positioning
#   2 (easy)     — Easy Warmup, Bench 30, Climax Boss, Steamroll
#   3 (medium)   — + Boss Rush, Basic 4v4, Skirmish, Duel, HvH mirror
#   4 (hard)     — All attrition scenarios
#
# Starts from warmstart checkpoint.

set -euo pipefail

STAGE=${1:-1}
START_ITER=${2:-1}
NUM_ITERS=${3:-10}
END_ITER=$((START_ITER + NUM_ITERS - 1))

ENTITY_ENCODER="generated/entity_encoder_pretrained_v4.pt"
WEIGHTS_JSON="generated/actor_critic_v2_curriculum.json"
CHECKPOINT="generated/actor_critic_v2_curriculum.pt"

# Initialize from warmstart if no curriculum checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Initializing from warmstart checkpoint..."
    cp generated/actor_critic_v2_warmstart_heavy.pt "$CHECKPOINT"
    uv run --with numpy --with torch training/export_actor_critic_v2.py \
        "$CHECKPOINT" -o "$WEIGHTS_JSON" \
        --d-model 64 --d-ff 128 --n-layers 4 --n-heads 4 --entity-encoder-layers 4
fi

# Select scenarios based on stage
SCENARIO_DIR=$(mktemp -d)
if [ "$STAGE" -eq 0 ]; then
    # Basic combat: attack, navigate, use abilities, dodge zones, focus fire
    SCENARIOS=(
        "scenarios/tutorials/t01_kill_dummy_1v1.toml"
        "scenarios/tutorials/t02_kill_3_dummies.toml"
        "scenarios/tutorials/t03_kill_tough_dummy.toml"
        "scenarios/tutorials/t04_punching_bag.toml"
        "scenarios/tutorials/t06_team_kill_dummies.toml"
        "scenarios/tutorials/t07_ability_burst.toml"
        "scenarios/tutorials/t09_focus_fire.toml"
        "scenarios/tutorials/t10_first_real_fight.toml"
        "scenarios/tutorials/t11_dodge_zones.toml"
    )
    EPISODES=20
    TEMPERATURE=0.6
elif [ "$STAGE" -eq 1 ]; then
    # Advanced micro: kiting, sustain, zone positioning
    SCENARIOS=(
        "scenarios/tutorials/t05_kite_brute.toml"
        "scenarios/tutorials/t08_heal_under_pressure.toml"
        "scenarios/tutorials/t12_dodge_zones_hard.toml"
        "scenarios/tutorials/t13_max_range.toml"
    )
    EPISODES=20
    TEMPERATURE=0.6
elif [ "$STAGE" -eq 2 ]; then
    # Easy combat scenarios
    SCENARIOS=(
        "scenarios/easy_warmup_4v3.toml"
        "scenarios/bench_30units.toml"
        "scenarios/climax_boss.toml"
        "scenarios/steamroll_6v3.toml"
    )
    EPISODES=10
    TEMPERATURE=0.8
elif [ "$STAGE" -eq 3 ]; then
    # Medium combat scenarios
    SCENARIOS=(
        "scenarios/easy_warmup_4v3.toml"
        "scenarios/bench_30units.toml"
        "scenarios/climax_boss.toml"
        "scenarios/steamroll_6v3.toml"
        "scenarios/boss_rush_6v2.toml"
        "scenarios/basic_4v4.toml"
        "scenarios/skirmish_3v3.toml"
        "scenarios/duel_1v1.toml"
        "scenarios/hvh_mirror.toml"
    )
    EPISODES=5
    TEMPERATURE=0.8
else
    # All scenarios
    SCENARIOS=(scenarios/*.toml)
    EPISODES=3
    TEMPERATURE=1.0
fi

# Symlink selected scenarios into temp dir
for f in "${SCENARIOS[@]}"; do
    ln -sf "$(realpath "$f")" "${SCENARIO_DIR}/$(basename "$f")"
done

D_MODEL=64
D_FF=128
N_LAYERS=4
N_HEADS=4
EE_LAYERS=4
PPO_EPOCHS=3
BATCH_SIZE=256
LR=3e-5
ENTROPY_COEFF=0.02
CLIP_EPS=0.1
ACTOR_LR_RATIO=0.05
KL_COEFF=0.5
BC_COEFF=0.1
CRITIC_WARMUP=5
CRITIC_ONLY_ITERS=${4:-3}

echo "=== Curriculum PPO: stage=${STAGE} iters=${START_ITER}..${END_ITER} ==="
echo "  Scenarios: ${#SCENARIOS[@]}"
echo "  Episodes/scenario: ${EPISODES}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Critic-only warmup: ${CRITIC_ONLY_ITERS} iters"
echo ""

# ── Phase 1: Critic-only warmup (actor frozen) ──────────────────────────
# Generate episodes with frozen warmstart policy, train only the value head.
# This calibrates the critic so advantages aren't noise when we start PPO.
for CITER in $(seq 1 "$CRITIC_ONLY_ITERS"); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  CRITIC WARMUP ${CITER}/${CRITIC_ONLY_ITERS} (stage ${STAGE})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EPISODES_FILE="generated/curriculum_s${STAGE}_critic${CITER}.jsonl"

    echo "[critic${CITER}] Generating episodes (actor frozen)..."
    cargo run --release --bin xtask -- scenario oracle transformer-rl generate \
        "${SCENARIO_DIR}" \
        --weights "${WEIGHTS_JSON}" \
        --output "${EPISODES_FILE}" \
        --episodes "${EPISODES}" \
        --temperature "${TEMPERATURE}" \
        --step-interval 3 \
        -j 0

    echo "[critic${CITER}] Training critic only (actor-lr-ratio=0)..."
    uv run --with numpy --with torch training/train_rl_v2.py \
        "${EPISODES_FILE}" \
        --pretrained "${CHECKPOINT}" \
        --entity-encoder "${ENTITY_ENCODER}" \
        -o "${CHECKPOINT}" \
        --log "generated/curriculum_s${STAGE}_critic${CITER}.csv" \
        --d-model "${D_MODEL}" \
        --d-ff "${D_FF}" \
        --n-layers "${N_LAYERS}" \
        --n-heads "${N_HEADS}" \
        --entity-encoder-layers "${EE_LAYERS}" \
        --ppo-epochs "${PPO_EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --entropy-coeff "${ENTROPY_COEFF}" \
        --clip-eps "${CLIP_EPS}" \
        --actor-lr-ratio 0.0 \
        --critic-warmup-epochs "${CRITIC_WARMUP}"

    echo "[critic${CITER}] Exporting weights..."
    uv run --with numpy --with torch training/export_actor_critic_v2.py \
        "${CHECKPOINT}" -o "${WEIGHTS_JSON}" \
        --d-model "${D_MODEL}" --d-ff "${D_FF}" \
        --n-layers "${N_LAYERS}" --n-heads "${N_HEADS}" \
        --entity-encoder-layers "${EE_LAYERS}"

    echo "[critic${CITER}] Evaluating (should match warmstart)..."
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        scenarios/tutorials/ --weights "${WEIGHTS_JSON}"

    echo "[critic${CITER}] Done."
    echo ""
done

# ── Phase 2: PPO with calibrated critic ─────────────────────────────────
for ITER in $(seq "$START_ITER" "$END_ITER"); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  PPO ITERATION ${ITER} (stage ${STAGE})"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    EPISODES_FILE="generated/curriculum_s${STAGE}_iter${ITER}.jsonl"

    # Generate episodes
    echo "[iter${ITER}] Generating episodes..."
    cargo run --release --bin xtask -- scenario oracle transformer-rl generate \
        "${SCENARIO_DIR}" \
        --weights "${WEIGHTS_JSON}" \
        --output "${EPISODES_FILE}" \
        --episodes "${EPISODES}" \
        --temperature "${TEMPERATURE}" \
        --step-interval 3 \
        -j 0

    # PPO training with calibrated critic
    echo "[iter${ITER}] PPO training..."
    uv run --with numpy --with torch training/train_rl_v2.py \
        "${EPISODES_FILE}" \
        --pretrained "${CHECKPOINT}" \
        --entity-encoder "${ENTITY_ENCODER}" \
        -o "${CHECKPOINT}" \
        --log "generated/curriculum_s${STAGE}_iter${ITER}.csv" \
        --d-model "${D_MODEL}" \
        --d-ff "${D_FF}" \
        --n-layers "${N_LAYERS}" \
        --n-heads "${N_HEADS}" \
        --entity-encoder-layers "${EE_LAYERS}" \
        --ppo-epochs "${PPO_EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --entropy-coeff "${ENTROPY_COEFF}" \
        --clip-eps "${CLIP_EPS}" \
        --actor-lr-ratio "${ACTOR_LR_RATIO}" \
        --kl-coeff "${KL_COEFF}" \
        --bc-coeff "${BC_COEFF}" \
        --critic-warmup-epochs "${CRITIC_WARMUP}"

    # Export
    echo "[iter${ITER}] Exporting weights..."
    uv run --with numpy --with torch training/export_actor_critic_v2.py \
        "${CHECKPOINT}" -o "${WEIGHTS_JSON}" \
        --d-model "${D_MODEL}" --d-ff "${D_FF}" \
        --n-layers "${N_LAYERS}" --n-heads "${N_HEADS}" \
        --entity-encoder-layers "${EE_LAYERS}"

    # Eval on tutorial + main scenarios
    echo "[iter${ITER}] Evaluating..."
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        scenarios/tutorials/ --weights "${WEIGHTS_JSON}"
    if [ "$STAGE" -ge 2 ]; then
        cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
            scenarios/ --weights "${WEIGHTS_JSON}"
    fi

    echo "[iter${ITER}] Done."
    echo ""
done

# Cleanup
rm -rf "${SCENARIO_DIR}"

echo "=== Curriculum PPO complete (stage ${STAGE}, ${NUM_ITERS} iterations) ==="
