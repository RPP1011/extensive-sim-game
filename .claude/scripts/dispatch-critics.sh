#!/usr/bin/env bash
# dispatch-critics.sh — orchestrator for biased-against critic agents.
#
# Reads .claude/skills/critic-<name>/SKILL.md, spawns `claude --print`
# per applicable critic, collects per-critic output to
# .claude/critic-output-<name>.txt, aggregates verdicts.
#
# Usage:
#   dispatch-critics.sh [--target <git-ref|WORKING-TREE>]
#                       [--critic <name>]...
#                       [--all]
#                       [--model sonnet|haiku|opus]

set -e

TARGET="${TARGET:-HEAD~1..HEAD}"
MODEL="${MODEL:-sonnet}"
declare -a CRITICS
SELECT_BY_HEURISTIC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --model)  MODEL="$2";  shift 2 ;;
        --critic) CRITICS+=("$2"); shift 2 ;;
        --all)    SELECT_BY_HEURISTIC=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#CRITICS[@]} -eq 0 ]]; then
    SELECT_BY_HEURISTIC=1
fi

cd "$(git rev-parse --show-toplevel)"

# §6 heuristic — populate CRITICS from diff if --all
if [[ -n "$SELECT_BY_HEURISTIC" ]]; then
    if [[ "$TARGET" == "WORKING-TREE" ]]; then
        diff_paths=$(git diff --name-only ; git diff --cached --name-only)
    else
        diff_paths=$(git diff --name-only "$TARGET")
    fi
    if echo "$diff_paths" | grep -qE '^crates/engine'; then
        CRITICS+=("compiler-first")
    fi
    if echo "$diff_paths" | grep -qE '^(crates/engine/src/state|crates/engine_data/src/(events|enums|scoring))'; then
        CRITICS+=("schema-bump")
    fi
    if echo "$diff_paths" | grep -qE '^assets/sim/'; then
        CRITICS+=("schema-bump" "cross-backend-parity" "reduction-determinism")
    fi
    if echo "$diff_paths" | grep -q '^crates/engine/build.rs$'; then
        if [[ "$TARGET" == "WORKING-TREE" ]]; then
            build_diff=$(git diff -- crates/engine/build.rs ; git diff --cached -- crates/engine/build.rs)
        else
            build_diff=$(git diff "$TARGET" -- crates/engine/build.rs)
        fi
        if echo "$build_diff" | grep -qE 'ALLOWED_(DIRS|TOP_LEVEL)'; then
            CRITICS+=("allowlist-gate")
        fi
    fi
    if echo "$diff_paths" | grep -qE '^crates/engine_rules/'; then
        CRITICS+=("compiler-first" "cross-backend-parity" "reduction-determinism")
    fi
    if echo "$diff_paths" | grep -qE '^crates/engine/src/(step\.rs|cascade/)'; then
        CRITICS+=("no-runtime-panic")
    fi
    if echo "$diff_paths" | grep -qE '^crates/engine_gpu/'; then
        CRITICS+=("cross-backend-parity" "reduction-determinism")
    fi
    # De-duplicate
    if [[ ${#CRITICS[@]} -gt 0 ]]; then
        IFS=$'\n' read -r -d '' -a CRITICS < <(printf '%s\n' "${CRITICS[@]}" | sort -u && printf '\0')
    fi
fi

if [[ ${#CRITICS[@]} -eq 0 ]]; then
    echo "No critics applicable to target $TARGET; nothing to do." >&2
    exit 0
fi

echo "Dispatching ${#CRITICS[@]} critic(s) on $TARGET: ${CRITICS[*]}" >&2

# Clean up old per-critic output files (so stale FAILs don't carry forward).
mkdir -p .claude
rm -f .claude/critic-output-*.txt

# Spawn each critic in parallel
declare -a PIDS
for c in "${CRITICS[@]}"; do
    skill_file=".claude/skills/critic-${c}/SKILL.md"
    if [[ ! -f "$skill_file" ]]; then
        echo "ERROR: $skill_file not found" >&2
        exit 1
    fi
    out_file=".claude/critic-output-${c}.txt"
    {
        skill_body=$(cat "$skill_file")
        prompt="${skill_body}

## Target

git-ref: ${TARGET}

Run the required tools (rg, ast-grep, git diff, etc. as cited in the prompt above).
Follow the rigid output format. If TOOLS RUN is empty, return FAIL."
        echo "$prompt" | claude --print --model "$MODEL" > "$out_file" 2>&1 || true
    } &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done

echo "" >&2
echo "=== Critic verdicts (target: $TARGET) ===" >&2
fails=0
for c in "${CRITICS[@]}"; do
    out_file=".claude/critic-output-${c}.txt"
    verdict=$(grep -m1 '^VERDICT:' "$out_file" 2>/dev/null | head -1 || echo "VERDICT: ??? (parse error)")
    echo "  critic-${c}: ${verdict#VERDICT: }" >&2
    if echo "$verdict" | grep -q "FAIL"; then
        fails=$((fails + 1))
    fi
done

echo "" >&2
if [[ "$fails" -gt 0 ]]; then
    echo "=== ${fails} FAIL, $((${#CRITICS[@]} - fails)) PASS ===" >&2
    echo "Review .claude/critic-output-*.txt for full EVIDENCE + REASONING." >&2
    exit 2
else
    echo "=== ${#CRITICS[@]} PASS ===" >&2
    exit 0
fi
