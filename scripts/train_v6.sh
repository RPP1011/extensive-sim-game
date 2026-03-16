#!/usr/bin/env bash
# Train V6 actor-critic using Burn IMPALA (in-process GPU, no Python/SHM)
#
# Usage:
#   ./scripts/train_v6.sh                          # default settings
#   ./scripts/train_v6.sh --iters 200 --lr 3e-4    # override any CLI arg
#   ./scripts/train_v6.sh --checkpoint generated/impala_v6/v6_iter0050.bin  # resume
#
# Environment:
#   RTX 4090, CUDA 12.0, libtorch from uv PyTorch cache

set -euo pipefail
cd "$(dirname "$0")/.."

# --- Find CUDA-enabled PyTorch in uv cache ---
TORCH_SITE="$(find ~/.cache/uv/archive-v0 -path "*/torch/lib/libtorch_cuda.so" 2>/dev/null | head -1 | xargs dirname | xargs dirname | xargs dirname)"
if [ -z "$TORCH_SITE" ]; then
    echo "ERROR: No CUDA-enabled PyTorch found in uv cache."
    echo "Run: uv pip install torch"
    exit 1
fi
echo "Using PyTorch from: $TORCH_SITE/torch"

# --- Environment for torch-sys / libtorch ---
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export PYTHONPATH="$TORCH_SITE"
export LD_LIBRARY_PATH="$TORCH_SITE/torch/lib:${LD_LIBRARY_PATH:-}"
export CXX=/usr/bin/g++-12
export LIBRARY_PATH="/usr/lib/gcc/x86_64-linux-gnu/12:${LIBRARY_PATH:-}"

# Ensure c++ is available (torch-sys cc-rs looks for it)
mkdir -p ~/bin
ln -sf /usr/bin/g++-12 ~/bin/c++
export PATH="$HOME/bin:$PATH"

# --- Default training parameters (override via env or CLI args) ---
SCENARIOS="${SCENARIOS:-dataset/scenarios/hvh}"
OUTPUT_DIR="${OUTPUT_DIR:-generated/impala_v6}"

mkdir -p "$OUTPUT_DIR"

echo "=== V6 IMPALA Training (Burn, in-process GPU) ==="
echo "  PyTorch:    $TORCH_SITE/torch"
echo "  Scenarios:  $SCENARIOS"
echo "  Output:     $OUTPUT_DIR"
echo ""

# --- Build ---
echo "Building with burn-gpu (release)..."
cargo build --release --features burn-gpu --bin xtask 2>&1 | tail -3

# --- Run ---
exec cargo run --release --features burn-gpu --bin xtask -- \
    scenario oracle transformer-rl impala-train \
    "$SCENARIOS" \
    --output-dir "$OUTPUT_DIR" \
    --iters "${ITERS:-100}" \
    --episodes "${EPISODES:-2}" \
    --threads "${THREADS:-32}" \
    --sims-per-thread "${SIMS:-64}" \
    --batch-size "${BATCH:-512}" \
    --train-steps "${TRAIN_STEPS:-50}" \
    --lr "${LR:-5e-4}" \
    --temperature "${TEMP:-1.0}" \
    --step-interval "${STEP_INTERVAL:-3}" \
    --entropy-coef "${ENTROPY:-0.01}" \
    --value-coef "${VALUE_COEF:-0.5}" \
    "$@"
