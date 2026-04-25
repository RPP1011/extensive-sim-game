#!/usr/bin/env bash
# session-end-engine-review.sh — Stop hook.
# If engine files were touched in the working tree, run dispatch-critics.

set -e

cd "$(git rev-parse --show-toplevel)"

session_diff=$(git diff --name-only)
if [[ -z "$session_diff" ]]; then
    session_diff=$(git diff HEAD~1..HEAD --name-only 2>/dev/null || echo "")
fi

if ! echo "$session_diff" | grep -qE '^(crates/engine|crates/engine_rules|crates/engine_data|crates/engine_gpu)/|^assets/sim/'; then
    exit 0
fi

echo "" >&2
echo "=== Session-end engine review ===" >&2
echo "Engine files touched. Running critics..." >&2

bash .claude/scripts/dispatch-critics.sh --target "WORKING-TREE" --all || true

# Stop hook itself exits 0 — the .githooks/pre-commit (Task 5) is the
# real block. This script just ensures verdict files are fresh.
exit 0
