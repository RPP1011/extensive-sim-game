# Dispatch-Critics Hooks Design Spec

> **Status:** Design (2026-04-25). Spec D-amendment. Wraps the 6
> critic skills (Spec D, already landed) into auto-dispatched
> hook-driven enforcement on engine-directory touches.

## §1 Goals

1. **Auto-dispatch critics** when engine files are touched. No manual paste-in for the common case; the `Stop` hook fires on session end and surfaces verdicts before the user commits.
2. **Fast pre-tool blocks** on obvious P1 violations (file path patterns + simple syntactic patterns). Synchronous, ~50ms, blocks at write-time. Catches the dumb cases without 60s of critic dispatch latency.
3. **Allowlist gate flag** — `engine/build.rs` ALLOWED_* edits are blocked unless `.claude/allowlist-gate-approved` exists. Forces the gate to run per-developer, per-edit.
4. **Single orchestrator** — `dispatch-critics.sh` is the one place that knows how to spawn critic agents. Hooks call it, users call it, future Spec C DAG-skill calls it. No duplication.
5. **Headless `claude` CLI** for critic spawning. Reuses Claude Code's auth/tool surface; each critic gets a fresh session via `claude --print` automatically.

## §2 Non-goals

- Synchronous blocking on full critic dispatch (latency unacceptable for routine edits).
- Anthropic SDK direct calls (adds dependency drift; reimplements auth).
- Replacing the existing 6 critic SKILL.md files (they're consumed verbatim).
- Critic scoring / merging logic beyond simple verdict aggregation. The user reads each critic's output.
- CI integration (different concern; the orchestrator is local-dev-time).
- Network-dependent flows (everything runs against local files + local critic outputs).

## §3 Architecture

```
.claude/
  scripts/
    dispatch-critics.sh           — orchestrator (called by hooks + manually)
    pre-tool-engine-edit.sh       — fast static checks on Edit/Write/MultiEdit
    session-end-engine-review.sh  — calls dispatch-critics on session end
  skills/
    critic-compiler-first/SKILL.md       (existing; from Spec D plan)
    critic-schema-bump/SKILL.md          (existing)
    critic-cross-backend-parity/SKILL.md (existing)
    critic-no-runtime-panic/SKILL.md     (existing)
    critic-reduction-determinism/SKILL.md (existing)
    critic-allowlist-gate/SKILL.md       (existing)
    dispatch-critics/SKILL.md            (NEW; thin doc + invocation guide)
  settings.json                    — adds 2 hook entries (PreToolUse, Stop)
  allowlist-gate-approved          — single-use flag file (created by user, deleted post-edit)
  critic-output-<name>.txt         — per-critic stdout, transient

src/bin/xtask/  (or crates/xtask/ if Spec B has landed)
  dispatch_critics_cmd.rs          — `xtask dispatch-critics` Rust wrapper
```

Three entry points hit the same orchestrator:
- **`PreToolUse` hook** → fast checks, no critic spawn.
- **`Stop` hook** → calls `dispatch-critics.sh` async if engine was touched.
- **Manual** — `bash .claude/scripts/dispatch-critics.sh` or `xtask dispatch-critics` for explicit pre-commit review.

## §4 Hook scripts

### §4.1 `pre-tool-engine-edit.sh`

**Trigger:** `PreToolUse` on `Edit|Write|MultiEdit`.

**Mechanism:** reads JSON from stdin (Claude Code passes tool input). Parses with `jq`. Three categories of synchronous block (~50ms total):

```bash
#!/usr/bin/env bash
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
```

Patterns are deliberately conservative (path-shape, not full-grammar parse). Anything more sophisticated is the critic's job.

### §4.2 `session-end-engine-review.sh`

**Trigger:** `Stop` hook (Claude session ends).

**Mechanism:** check if engine files were touched in the session's diff; if so, run `dispatch-critics.sh` on those changes; print verdict summary. ~30-90s; doesn't block (Stop fires after the session is wrapping up).

```bash
#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"

# Determine the session's diff. Best-effort: prefer working-tree changes.
session_diff=$(git diff --name-only)
if [[ -z "$session_diff" ]]; then
    # No uncommitted changes — check the most recent commit on this branch.
    session_diff=$(git diff HEAD~1..HEAD --name-only 2>/dev/null || echo "")
fi

# Did this session touch engine files?
if ! echo "$session_diff" | grep -qE '^(crates/engine|crates/engine_rules|crates/engine_data|crates/engine_gpu)/|^assets/sim/'; then
    exit 0  # nothing engine-relevant; skip
fi

echo "" >&2
echo "=== Session-end engine review ===" >&2
echo "Engine files touched. Running critics..." >&2

# Use working-tree state as the target. Critics evaluate the changes.
bash .claude/scripts/dispatch-critics.sh --target "WORKING-TREE" --all

# Always exit 0 — Stop hook is advisory, doesn't gate anything.
# The user sees the verdicts as session-end output and can revert before commit.
exit 0
```

### §4.3 `settings.json` additions

```json
{
  "hooks": {
    "SessionStart": [
      { "type": "command", "command": "cat docs/constitution.md" }
    ],
    "PreToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          { "type": "command", "command": ".claude/scripts/pre-tool-engine-edit.sh" }
        ]
      }
    ],
    "Stop": [
      {
        "type": "command",
        "command": ".claude/scripts/session-end-engine-review.sh"
      }
    ]
  }
}
```

The existing `SessionStart` hook (constitution auto-load) is preserved.

## §5 `dispatch-critics.sh` orchestrator

**Usage:**
```
dispatch-critics.sh [--target <git-ref|WORKING-TREE>]
                    [--critic <name>...]
                    [--all]
                    [--model sonnet|haiku|opus]
```

**Behavior:**

```bash
#!/usr/bin/env bash
set -e

TARGET="${TARGET:-HEAD~1..HEAD}"
MODEL="${MODEL:-sonnet}"
declare -a CRITICS

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        --model)  MODEL="$2";  shift 2 ;;
        --critic) CRITICS+=("$2"); shift 2 ;;
        --all)    SELECT_BY_HEURISTIC=1; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# If no critics specified, default to --all
if [[ ${#CRITICS[@]} -eq 0 ]]; then
    SELECT_BY_HEURISTIC=1
fi

# §6 heuristic — populate CRITICS from diff if --all
if [[ -n "${SELECT_BY_HEURISTIC:-}" ]]; then
    if [[ "$TARGET" == "WORKING-TREE" ]]; then
        diff_paths=$(git diff --name-only)
    else
        diff_paths=$(git diff --name-only "$TARGET")
    fi
    # Always include compiler-first if engine touched
    if echo "$diff_paths" | grep -qE '^crates/engine'; then
        CRITICS+=("compiler-first")
    fi
    # Layout/state changes → schema-bump
    if echo "$diff_paths" | grep -qE '^(crates/engine/src/state|crates/engine_data/src/(events|enums|scoring))'; then
        CRITICS+=("schema-bump")
    fi
    # DSL changes → schema-bump + cross-backend + reduction
    if echo "$diff_paths" | grep -qE '^assets/sim/'; then
        CRITICS+=("schema-bump" "cross-backend-parity" "reduction-determinism")
    fi
    # engine/build.rs ALLOWED edits → allowlist-gate
    if echo "$diff_paths" | grep -q '^crates/engine/build.rs$' \
       && git diff "$TARGET" -- crates/engine/build.rs 2>/dev/null | grep -qE 'ALLOWED_(DIRS|TOP_LEVEL)'; then
        CRITICS+=("allowlist-gate")
    fi
    # engine_rules/ changes → compiler-first + cross-backend + reduction
    if echo "$diff_paths" | grep -qE '^crates/engine_rules/'; then
        CRITICS+=("compiler-first" "cross-backend-parity" "reduction-determinism")
    fi
    # step.rs / cascade → no-runtime-panic
    if echo "$diff_paths" | grep -qE '^crates/engine/src/(step\.rs|cascade/)'; then
        CRITICS+=("no-runtime-panic")
    fi
    # engine_gpu → cross-backend + reduction
    if echo "$diff_paths" | grep -qE '^crates/engine_gpu/'; then
        CRITICS+=("cross-backend-parity" "reduction-determinism")
    fi
    # De-duplicate
    CRITICS=($(printf '%s\n' "${CRITICS[@]}" | sort -u))
fi

if [[ ${#CRITICS[@]} -eq 0 ]]; then
    echo "No critics applicable to target $TARGET; nothing to do." >&2
    exit 0
fi

echo "Dispatching ${#CRITICS[@]} critic(s) on $TARGET: ${CRITICS[*]}" >&2

# Spawn each critic in parallel
mkdir -p .claude
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
        echo "$prompt" | claude --print --model "$MODEL" > "$out_file" 2>&1
    } &
    PIDS+=($!)
done

# Wait for all
for pid in "${PIDS[@]}"; do wait "$pid"; done

# Aggregate
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
    echo "Review .claude/critic-output-*.txt for details + EVIDENCE." >&2
    exit 2
else
    echo "=== ${#CRITICS[@]} PASS ===" >&2
    exit 0
fi
```

Per-critic output goes to `.claude/critic-output-<name>.txt` (gitignored — transient, regenerated per run). The user reads these for the full VERDICT/EVIDENCE/REASONING/TOOLS-RUN block.

## §6 Critic selection heuristic

Codified in §5's bash; summarized as a table:

| Diff path | Critics |
|---|---|
| `crates/engine/**` (any) | always: `+compiler-first` |
| `crates/engine/src/state/**` | `+schema-bump` |
| `crates/engine/src/step.rs`, `crates/engine/src/cascade/**` | `+no-runtime-panic` |
| `crates/engine/build.rs` ALLOWED_* edits | `+allowlist-gate` |
| `crates/engine_data/src/{events,enums,scoring}/**` | `+schema-bump` |
| `crates/engine_rules/**` | `+compiler-first`, `+cross-backend-parity`, `+reduction-determinism` |
| `crates/engine_gpu/**` | `+cross-backend-parity`, `+reduction-determinism` |
| `assets/sim/**` (DSL changes) | `+schema-bump`, `+cross-backend-parity`, `+reduction-determinism` |

De-duplicated on union. Worst case (all paths touched) → all 6 critics.

User overrides: `--critic <name>` to force a single critic regardless of diff; can be repeated.

## §7 Allowlist gate flag

**File:** `.claude/allowlist-gate-approved` — empty file, presence is the signal. Single-use per allowlist edit.

**Workflow:**

1. User wants to add `voxel/` to `engine/build.rs` ALLOWED_DIRS.
2. User edits `crates/engine/build.rs` via `Edit` tool → `pre-tool-engine-edit.sh` matches the ALLOWED_* pattern, no flag → exit 2 with stderr instructing the user.
3. User runs:
   ```
   bash .claude/scripts/dispatch-critics.sh --critic compiler-first --critic allowlist-gate --target WORKING-TREE
   ```
   Both critics run on the proposed change. Per Spec D, both must return PASS.
4. If both PASS: `touch .claude/allowlist-gate-approved`. (`xtask dispatch-critics` could automate this on success — see §8.)
5. User re-attempts the Edit → pre-tool hook sees the flag → allows.
6. After commit: a `Stop` hook (or a separate `PostToolUse Bash` hook on `git commit`) deletes the flag. Single-use; next allowlist edit needs to re-run the gate.

The flag is in `.claude/` which is gitignored (with explicit exceptions for `settings.json` + `skills/`). The flag intentionally never commits.

## §8 Manual invocation surfaces

### §8.1 `dispatch-critics` skill

Project-local skill at `.claude/skills/dispatch-critics/SKILL.md`. Description: "Use when reviewing engine changes before committing — manual gate for the allowlist edit workflow, or to spot-check a specific principle."

Body documents:
- The bash script invocation pattern.
- The 6 critic names + when each applies.
- The expected output format.
- The allowlist-gate workflow.

The skill is documentation; the work happens in the bash script.

### §8.2 xtask command

Add `cargo run --bin xtask -- dispatch-critics [args]` (or `crates/xtask/` post-Spec-B). The xtask command is a thin Rust wrapper:

```rust
// roughly
fn dispatch_critics(args: Vec<String>) -> Result<()> {
    let status = std::process::Command::new("bash")
        .arg(".claude/scripts/dispatch-critics.sh")
        .args(args)
        .status()?;
    std::process::exit(status.code().unwrap_or(1));
}
```

Convenience for users who think in `cargo run --bin xtask`.

## §9 Decision log

- **D1.** Headless `claude --print` for critic dispatch (vs Anthropic SDK or manual paste). Reuses Claude Code auth + tool surface.
- **D2.** Three entry points (PreToolUse, Stop, manual) all hit the same orchestrator script. No logic duplication.
- **D3.** Pre-tool hook does only fast static checks. Anything requiring agent reasoning lives in the orchestrator + critics.
- **D4.** Stop hook is advisory, not blocking. Stop fires post-edit; user sees verdict and chooses to revert/fix before commit. Avoids false-positive blast radius from bad heuristics.
- **D5.** Allowlist gate flag is presence-only, single-use, gitignored. Forces re-gating per edit and per developer.
- **D6.** Critic selection heuristic codified in `dispatch-critics.sh` (not in the skills themselves). Single source of truth for "which critics run when."
- **D7.** Per-critic output to `.claude/critic-output-<name>.txt`. Transient (gitignored), regenerated per run.

## §10 Out of scope

- Critic verdict merging beyond simple per-critic verdict aggregation. The user reads each critic's full output for context.
- Network-dependent flows (CI integration, PR-comment automation).
- Caching critic output (regenerated per dispatch; small enough to not matter).
- Auto-fix suggestions from critics (Spec D explicitly out-of-scope).
- Pre-commit Git hook (separate concern; would duplicate the Claude-side `Stop` hook).
- Spec B integrations beyond the allowlist gate (e.g., a hook for `crates/engine/src/state/**` schema-bump warning) — defer until evidence the schema-bump test isn't enough.
