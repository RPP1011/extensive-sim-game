#!/usr/bin/env bash
# pre-tool-engine-edit.sh — fast static checks on Edit/Write/MultiEdit.
# Reads tool input JSON from stdin (Claude Code convention).
# exit 0 = allow; exit 2 = block.

set -e
input_json=$(cat)
file_path=$(echo "$input_json" | jq -r '.tool_input.file_path // empty')
new_content=$(echo "$input_json" | jq -r '.tool_input.new_string // .tool_input.content // empty')

# (a) Editing files in engine_rules/src/ (other than lib.rs) → block.
if [[ "$file_path" =~ ^crates/engine_rules/src/ ]] && [[ "$(basename "$file_path")" != "lib.rs" ]]; then
    echo "BLOCK: $file_path is in engine_rules/ (compiler-emitted)." >&2
    echo "Edit the DSL source (assets/sim/) and rerun cargo run --bin xtask -- compile-dsl." >&2
    exit 2
fi

# (b) Editing engine/build.rs ALLOWED_* requires gate flag.
if [[ "$file_path" == "crates/engine/build.rs" ]]; then
    if echo "$new_content" | grep -qE 'ALLOWED_(DIRS|TOP_LEVEL)'; then
        if [[ ! -f .claude/allowlist-gate-approved ]]; then
            echo "BLOCK: editing engine/build.rs allowlist requires gate approval." >&2
            echo "Run: bash .claude/scripts/dispatch-critics.sh --critic compiler-first --critic allowlist-gate" >&2
            echo "Both must PASS, then 'touch .claude/allowlist-gate-approved' before retry." >&2
            exit 2
        fi
    fi
fi

# (c) impl CascadeHandler outside engine_rules/ → block.
if echo "$new_content" | grep -qE '^impl[[:space:]]+CascadeHandler\b'; then
    if [[ ! "$file_path" =~ ^crates/engine_rules/src/ ]]; then
        echo "BLOCK: impl CascadeHandler outside crates/engine_rules/. Violates P1." >&2
        echo "Express the rule in assets/sim/physics.sim and let the emitter generate the impl." >&2
        exit 2
    fi
fi

exit 0
