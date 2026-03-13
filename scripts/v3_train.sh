#!/bin/bash
# V3 pointer action space training pipeline
# Step 1: Generate oracle BC data (default AI plays, V3 format recorded)
# Step 2: BC warmstart on oracle data
# Step 3: Iterative REINFORCE with dense rewards
set -euo pipefail

cd "$(dirname "$0")/.."

MAX_ITER=${MAX_ITER:-5}
EPISODES=${EPISODES:-20}
TEMP=${TEMP:-0.8}
MAX_TICKS=${MAX_TICKS:-1000}
LR=${LR:-5e-7}
BC_LR=${BC_LR:-3e-4}
BC_EPOCHS=${BC_EPOCHS:-30}
ENTROPY=${ENTROPY:-0.05}
REWARD_SHAPING=${REWARD_SHAPING:-0.1}
STEP_INTERVAL=${STEP_INTERVAL:-3}

REGISTRY="generated/ability_embedding_registry.json"
WEIGHTS_PT="generated/actor_critic_v3.pt"
WEIGHTS_JSON="generated/actor_critic_weights_v3.json"
BC_DATA="generated/rl_v3_bc.jsonl"

# Step 0: Initialize V3 model if needed
if [ ! -f "$WEIGHTS_PT" ]; then
    echo "=== Initializing V3 model from pretrained components ==="
    uv run --with numpy --with torch python3 -c "
import torch, sys
sys.path.insert(0, 'training')
from model import AbilityActorCriticV3
from tokenizer import AbilityTokenizer

tok = AbilityTokenizer()
model = AbilityActorCriticV3(
    vocab_size=tok.vocab_size,
    entity_encoder_layers=4,
    external_cls_dim=128,
    d_model=32, d_ff=64, n_layers=4, n_heads=4,
)

# Load pretrained d=32 ability transformer (MLM Phase 1, v2)
pretrained = torch.load('generated/ability_transformer_pretrained_v2.pt',
                        map_location='cpu', weights_only=True)
transformer_keys = {k: v for k, v in pretrained.items()
                    if k.startswith('transformer.')}
missing, unexpected = model.load_state_dict(transformer_keys, strict=False)
print(f'Loaded {len(transformer_keys) - len(unexpected)} transformer params')
print(f'  Missing (expected): {len(missing)} (entity encoder, cross-attn, pointer head, value head, cls_proj)')

torch.save(model.state_dict(), '$WEIGHTS_PT')
print(f'Saved initial V3 model to $WEIGHTS_PT')
"
    uv run --with numpy --with torch training/export_actor_critic_v3.py \
        "$WEIGHTS_PT" -o "$WEIGHTS_JSON" --external-cls-dim 128
    echo "Initial V3 model created."
fi

# Step 1: Generate oracle BC data (default AI plays, V3 fields recorded)
if [ ! -f "$BC_DATA" ]; then
    echo ""
    echo "================================================================"
    echo "=== Step 1: Generate oracle BC data ==="
    echo "================================================================"
    # Use --policy combined so default AI drives decisions, V3 format recorded
    cargo run --release --bin xtask -- scenario oracle transformer-rl generate \
        scenarios/hvh/ \
        --policy combined \
        -o "$BC_DATA" \
        --episodes "$EPISODES" \
        --temperature 1.0 \
        --step-interval "$STEP_INTERVAL" \
        --max-ticks "$MAX_TICKS" \
        -j 0

    N_EP=$(wc -l < "$BC_DATA")
    N_WIN=$(grep -c '"Victory"' "$BC_DATA" || true)
    echo "  BC data: $N_EP episodes, $N_WIN wins ($(python3 -c "print(f'{$N_WIN/$N_EP*100:.1f}%')"))"
fi

# Step 2: BC warmstart
if [ ! -f "generated/actor_critic_v3_bc.pt" ]; then
    echo ""
    echo "================================================================"
    echo "=== Step 2: BC warmstart ($BC_EPOCHS epochs) ==="
    echo "================================================================"
    uv run --with numpy --with torch training/train_rl_v3.py \
        "$BC_DATA" \
        --pretrained "$WEIGHTS_PT" \
        --embedding-registry "$REGISTRY" \
        --external-cls-dim 128 \
        -o "generated/actor_critic_v3_bc.pt" \
        --log "generated/actor_critic_v3_bc.csv" \
        --bc-epochs "$BC_EPOCHS" \
        --ppo-epochs 0 \
        --batch-size 512 \
        --lr "$BC_LR" \
        --entropy-coeff 0.01 \
        --unfreeze-transformer

    cp "generated/actor_critic_v3_bc.pt" "$WEIGHTS_PT"

    uv run --with numpy --with torch training/export_actor_critic_v3.py \
        "$WEIGHTS_PT" -o "$WEIGHTS_JSON" --external-cls-dim 128

    echo "--- BC Eval ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        scenarios/hvh/ \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        --max-ticks 2000
fi

# Step 3: Iterative REINFORCE with dense rewards
ITER=1
while [ -f "generated/rl_v3_rf_iter${ITER}.jsonl" ]; do
    ITER=$((ITER + 1))
done
echo ""
echo "Starting REINFORCE from iteration $ITER"

while [ "$ITER" -le "$MAX_ITER" ]; do
    echo ""
    echo "================================================================"
    echo "=== REINFORCE Iteration $ITER / $MAX_ITER ==="
    echo "================================================================"

    EP_FILE="generated/rl_v3_rf_iter${ITER}.jsonl"

    # Generate on-policy episodes
    echo "--- Generate episodes (temp=${TEMP}) ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl generate \
        scenarios/hvh/ \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        -o "$EP_FILE" \
        --episodes "$EPISODES" \
        --temperature "$TEMP" \
        --step-interval "$STEP_INTERVAL" \
        --max-ticks "$MAX_TICKS" \
        -j 0

    N_EP=$(wc -l < "$EP_FILE")
    N_WIN=$(grep -c '"Victory"' "$EP_FILE" || true)
    echo "  Episodes: $N_EP, Wins: $N_WIN ($(python3 -c "print(f'{$N_WIN/$N_EP*100:.1f}%')"))"

    # REINFORCE training
    echo "--- REINFORCE (lr=${LR}, reward_shaping=${REWARD_SHAPING}) ---"
    uv run --with numpy --with torch training/train_rl_v3.py \
        "$EP_FILE" \
        --pretrained "$WEIGHTS_PT" \
        --embedding-registry "$REGISTRY" \
        --external-cls-dim 128 \
        -o "$WEIGHTS_PT" \
        --log "generated/actor_critic_v3_rf_iter${ITER}.csv" \
        --reinforce-epochs 1 \
        --ppo-epochs 0 \
        --batch-size 512 \
        --lr "$LR" \
        --entropy-coeff "$ENTROPY" \
        --reward-shaping "$REWARD_SHAPING" \
        --unfreeze-transformer

    # Export
    uv run --with numpy --with torch training/export_actor_critic_v3.py \
        "$WEIGHTS_PT" -o "$WEIGHTS_JSON" --external-cls-dim 128

    # Eval
    echo "--- Eval (HvH greedy) ---"
    cargo run --release --bin xtask -- scenario oracle transformer-rl eval \
        scenarios/hvh/ \
        --weights "$WEIGHTS_JSON" \
        --embedding-registry "$REGISTRY" \
        --max-ticks 2000

    echo "=== REINFORCE iteration $ITER complete ==="
    ITER=$((ITER + 1))
done
