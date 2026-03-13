#!/usr/bin/env bash
# Run all operator training experiments (E0-E10) and collect results.
# Each experiment runs for 5K steps (~2 min each), total ~25 min.
#
# Usage: bash scripts/run_operator_experiments.sh [DATA_PATH]

set -euo pipefail

DATA="${1:-generated/operator_dataset_hvh.npz}"
LOGDIR="generated/operator_experiments"
mkdir -p "$LOGDIR"

EXPERIMENTS="e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e10"

echo "═══════════════════════════════════════════════════════════════"
echo "Ability Latent Operator — Training Experiments"
echo "Data: $DATA"
echo "═══════════════════════════════════════════════════════════════"
echo ""

for exp in $EXPERIMENTS; do
    echo "── Running $exp ──────────────────────────────────────────────"
    uv run --with numpy --with torch training/train_operator.py \
        --data "$DATA" \
        --max-steps 5000 --eval-every 5000 --batch-size 1024 \
        --compile --experiment "$exp" \
        --output "generated/operator_${exp}.pt" \
        2>&1 | tee "$LOGDIR/${exp}.log"
    echo ""
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Results Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
printf "%-5s | %-40s | %8s | %8s | %10s | %9s\n" "Exp" "Description" "HP" "Pos" "Exists" "Val Loss"
printf "%-5s-+-%-40s-+-%8s-+-%8s-+-%-10s-+-%-9s\n" "-----" "----------------------------------------" "--------" "--------" "----------" "---------"

for exp in $EXPERIMENTS; do
    if grep -q "^RESULT" "$LOGDIR/${exp}.log" 2>/dev/null; then
        line=$(grep "^RESULT" "$LOGDIR/${exp}.log" | tail -1)
        hp=$(echo "$line" | sed 's/.*hp=\([^ ]*\)%.*/\1/')
        pos=$(echo "$line" | sed 's/.*pos=\([^ ]*\)%.*/\1/')
        exists=$(echo "$line" | sed 's/.*exists=\([^ ]*\)%.*/\1/')
        val_loss=$(echo "$line" | sed 's/.*val_loss=\([^ ]*\)/\1/')
        desc="${exp}"
        printf "%-5s | %-40s | %7s%% | %7s%% | %9s%% | %9s\n" \
            "$exp" "$(grep "^Experiment:" "$LOGDIR/${exp}.log" | head -1 | sed 's/Experiment: [^ ]* — //')" \
            "$hp" "$pos" "$exists" "$val_loss"
    else
        printf "%-5s | %-40s | %8s | %8s | %10s | %9s\n" "$exp" "FAILED" "-" "-" "-" "-"
    fi
done

echo ""
echo "Logs in: $LOGDIR/"
