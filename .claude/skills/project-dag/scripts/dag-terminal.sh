#!/usr/bin/env bash
# Detect terminal condition: DAG_COMPLETE, HUMAN_BLOCKED, or NONE.
# Outputs the terminal kind on stdout; nothing if run should continue.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Any pending implementer task with all deps satisfied?
READY=$(jq -r '
    . as $root
    | .tasks
    | map(select(.status == "pending" and .owner_class == "implementer"))
    | map(. as $t |
        ($t.deps | map(. as $d |
            ($root.tasks | map(select(.id == $d)) | first | .status)
        ) | all(. == "done" or . == "deferred" or . == "skipped" or . == null))
    )
    | map(select(. == true))
    | length
' "$STATE")

if [[ "$READY" -gt 0 ]]; then
    # Run can continue
    exit 0
fi

# No actionable implementer tasks. Determine terminal kind.
ANY_HUMAN=$(jq -r '
    .tasks
    | map(select(.status == "pending" and (.owner_class == "plan-writer" or .owner_class == "spec-needed")))
    | length
' "$STATE")

ANY_PENDING=$(jq -r '.tasks | map(select(.status != "done" and .status != "skipped" and .status != "deferred")) | length' "$STATE")

if [[ "$ANY_PENDING" -eq 0 ]]; then
    KIND="DAG_COMPLETE"
elif [[ "$ANY_HUMAN" -gt 0 ]]; then
    KIND="HUMAN_BLOCKED"
else
    KIND="HUMAN_BLOCKED"  # blocked tasks awaiting unblock
fi

echo "{\"ts\":\"$NOW\",\"event\":\"terminal\",\"reason\":\"$KIND\"}" >> "$LOG"
echo "$KIND"
