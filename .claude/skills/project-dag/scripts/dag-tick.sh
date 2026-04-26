#!/usr/bin/env bash
# Pick the next eligible implementer task and mark it in_progress.
# Output the task as JSON for the calling Claude session to dispatch.
# If no eligible task: invoke dag-terminal.sh and exit 0.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Find first eligible implementer task (deps all done/deferred/skipped or empty).
TASK=$(jq -r '
    . as $root
    | .tasks
    | map(select(.status == "pending" and .owner_class == "implementer"))
    | map(. as $t |
        if ($t.deps | length) == 0 then $t
        else
            (($t.deps | map(. as $d |
                ($root.tasks | map(select(.id == $d)) | first | .status)
            ) | all(. == "done" or . == "deferred" or . == "skipped" or . == null))
            as $ready
            | if $ready then $t else empty end)
        end
    )
    | first // empty
' "$STATE")

if [[ -z "$TASK" || "$TASK" == "null" ]]; then
    # No eligible work — emit terminal
    KIND=$(bash "$REPO_ROOT/.claude/skills/project-dag/scripts/dag-terminal.sh")
    echo "TERMINAL: $KIND"
    exit 0
fi

TASK_ID=$(echo "$TASK" | jq -r '.id')

# Mark in_progress
TMP="$(mktemp)"
jq --arg id "$TASK_ID" --arg now "$NOW" '
    .tasks |= map(
        if .id == $id then .status = "in_progress" | .started_at = $now
        else . end
    )
    | .run.iterations_completed += 1
    | .run.last_iteration_at = $now
    | .run.current_phase = "dispatching"
' "$STATE" > "$TMP"
mv "$TMP" "$STATE"

echo "{\"ts\":\"$NOW\",\"event\":\"task_started\",\"task_id\":\"$TASK_ID\"}" >> "$LOG"

# Output task JSON for Claude session to consume
echo "$TASK"
