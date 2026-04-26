#!/usr/bin/env bash
# Mark a task blocked. Args: <task_id> <reason>
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: dag-mark-blocked.sh <task_id> <reason>" >&2
    exit 2
fi

TASK_ID="$1"
REASON="$2"

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

TMP="$(mktemp)"
jq --arg id "$TASK_ID" --arg reason "$REASON" --arg now "$NOW" '
    .tasks |= map(
        if .id == $id then
            .status = "blocked"
            | .blocked_reason = $reason
            | .retry_count = (.retry_count + 1)
        else . end
    )
    | .run.tasks_blocked += 1
    | .run.last_iteration_at = $now
' "$STATE" > "$TMP"
mv "$TMP" "$STATE"

REASON_JSON=$(jq -Rn --arg r "$REASON" '$r')
echo "{\"ts\":\"$NOW\",\"event\":\"task_blocked\",\"task_id\":\"$TASK_ID\",\"reason\":$REASON_JSON}" >> "$LOG"
echo "Marked $TASK_ID blocked: $REASON"
