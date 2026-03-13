#!/usr/bin/env bash
# Run Round 2 operator training experiments (E11-E21) and collect results.
# All Huber-based, building on E8 (Huber delta=0.1) as the Round 1 winner.
# E11-E20: 5K steps (~2 min each), E21: 20K steps (~7 min). Total ~27 min.
#
# Usage: bash scripts/run_operator_round2.sh [DATA_PATH]

set -euo pipefail

DATA="${1:-generated/operator_dataset_hvh.npz}"
LOGDIR="generated/operator_experiments"
mkdir -p "$LOGDIR"

EXPERIMENTS="e11 e12 e13 e14 e15 e16 e17 e18 e19 e20 e21"

echo "═══════════════════════════════════════════════════════════════"
echo "Ability Latent Operator — Round 2 Experiments (Huber-based)"
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
echo "Round 2 Results Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
printf "%-5s | %-40s | %8s | %8s | %10s | %9s\n" "Exp" "Description" "HP" "Pos" "Exists" "Val Loss"
printf "%-5s-+-%-40s-+-%8s-+-%8s-+-%-10s-+-%-9s\n" "-----" "----------------------------------------" "--------" "--------" "----------" "---------"

# Include E8 baseline if available
for exp in e8 $EXPERIMENTS; do
    if grep -q "^RESULT" "$LOGDIR/${exp}.log" 2>/dev/null; then
        line=$(grep "^RESULT" "$LOGDIR/${exp}.log" | tail -1)
        hp=$(echo "$line" | sed 's/.*hp=\([^ ]*\)%.*/\1/')
        pos=$(echo "$line" | sed 's/.*pos=\([^ ]*\)%.*/\1/')
        exists=$(echo "$line" | sed 's/.*exists=\([^ ]*\)%.*/\1/')
        val_loss=$(echo "$line" | sed 's/.*val_loss=\([^ ]*\)/\1/')
        printf "%-5s | %-40s | %7s%% | %7s%% | %9s%% | %9s\n" \
            "$exp" "$(grep "^Experiment:" "$LOGDIR/${exp}.log" | head -1 | sed 's/Experiment: [^ ]* — //')" \
            "$hp" "$pos" "$exists" "$val_loss"
    else
        printf "%-5s | %-40s | %8s | %8s | %10s | %9s\n" "$exp" "(not available)" "-" "-" "-" "-"
    fi
done

echo ""
echo "Logs in: $LOGDIR/"
