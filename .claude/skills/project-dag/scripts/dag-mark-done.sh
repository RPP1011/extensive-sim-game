#!/usr/bin/env bash
# Mark a task done. Args: <task_id> <commit_sha> [critic_verdicts_json]
#
# Each Task is an h3 header in a plan file (`### Task N: ...`); its body
# contains - [ ] **Step N: ...** checkboxes. Marking a Task done flips ALL
# step checkboxes in that Task's body (between this Task's checkbox_line
# and the next Task's checkbox_line, or EOF for the last Task).
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: dag-mark-done.sh <task_id> <commit_sha> [critic_verdicts_json]" >&2
    exit 2
fi

TASK_ID="$1"
COMMIT="$2"
VERDICTS="${3-}"
if [[ -z "$VERDICTS" ]]; then
    VERDICTS='{}'
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
STATE="$REPO_ROOT/docs/superpowers/dag/state.json"
LOG="$REPO_ROOT/docs/superpowers/dag/run.jsonl"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Update state.json
TMP="$(mktemp)"
jq --arg id "$TASK_ID" --arg commit "$COMMIT" --arg now "$NOW" --argjson verdicts "$VERDICTS" '
    .tasks |= map(
        if .id == $id then
            .status = "done"
            | .completed_commit = $commit
            | .completed_at = $now
            | .critic_verdicts = $verdicts
            | .steps_done = .steps_total
        else . end
    )
    | .run.tasks_closed += 1
    | .run.last_iteration_at = $now
' "$STATE" > "$TMP"
mv "$TMP" "$STATE"

# Flip step checkboxes within this Task's body.
PLAN_ID=$(jq -r --arg id "$TASK_ID" '.tasks[] | select(.id == $id) | (.plan // empty)' "$STATE")
LINE=$(jq -r --arg id "$TASK_ID" '.tasks[] | select(.id == $id) | (.checkbox_line // empty)' "$STATE")
if [[ -n "$PLAN_ID" && -n "$LINE" && "$LINE" != "null" ]]; then
    PLAN_FILE=$(jq -r --arg p "$PLAN_ID" '.plans[] | select(.id == $p) | .file' "$STATE")
    if [[ -f "$REPO_ROOT/$PLAN_FILE" ]]; then
        # Find the checkbox_line of the NEXT task in the same plan (if any),
        # to bound the sed range.
        NEXT_LINE=$(jq -r --arg id "$TASK_ID" --arg p "$PLAN_ID" '
            .tasks
            | map(select(.plan == $p and .checkbox_line != null))
            | sort_by(.checkbox_line)
            | . as $sorted
            | (map(.id) | index($id)) as $i
            | if $i == null or ($i + 1) >= ($sorted | length) then "$"
              else (($sorted[$i + 1].checkbox_line) - 1 | tostring)
              end
        ' "$STATE")
        # Use BSD/GNU compatible sed: flip "- [ ]" to "- [x]" in the range.
        sed -i "${LINE},${NEXT_LINE}{s/- \[ \]/- [x]/}" "$REPO_ROOT/$PLAN_FILE"
    fi
fi

# Append to run.jsonl
echo "{\"ts\":\"$NOW\",\"event\":\"task_done\",\"task_id\":\"$TASK_ID\",\"commit\":\"$COMMIT\",\"critic_verdicts\":$VERDICTS}" >> "$LOG"
echo "Marked $TASK_ID done (commit $COMMIT)"
