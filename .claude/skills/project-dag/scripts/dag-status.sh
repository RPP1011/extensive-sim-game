#!/usr/bin/env bash
# Print a human summary of DAG state.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"

if [[ ! -f "$STATE" ]]; then
    echo "state.json not found. Run dag-bootstrap.sh first." >&2
    exit 1
fi

echo "=== DAG status ==="
jq -r '
    "Roadmap items: \(.roadmap | length)",
    "Plans: \(.plans | length) (done: \(.plans | map(select(.status == "done")) | length))",
    "Tasks: \(.tasks | length)",
    "  done:        \(.tasks | map(select(.status == "done")) | length)",
    "  in_progress: \(.tasks | map(select(.status == "in_progress")) | length)",
    "  pending:     \(.tasks | map(select(.status == "pending")) | length)",
    "  blocked:     \(.tasks | map(select(.status == "blocked")) | length)",
    "By owner_class:",
    (.tasks | group_by(.owner_class) | map("  \(.[0].owner_class): \(length)") | .[])
' "$STATE"

echo
echo "=== Pending implementer tasks (next 10 actionable) ==="
jq -r '
    .tasks
    | map(select(.status == "pending" and .owner_class == "implementer"))
    | .[0:10]
    | map("  \(.id): \(.title)")
    | .[]
' "$STATE"

echo
echo "=== Run state ==="
jq -r '.run | "  iterations: \(.iterations_completed)\n  closed: \(.tasks_closed)\n  blocked: \(.tasks_blocked)\n  phase: \(.current_phase)"' "$STATE"
