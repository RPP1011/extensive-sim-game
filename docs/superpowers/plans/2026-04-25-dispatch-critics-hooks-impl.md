# Dispatch-Critics + Hooks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the 3 bash scripts + 1 Git hook + 1 thin doc skill + settings.json + .gitignore updates that wire the 6 critic skills (already landed) into hook-driven enforcement on engine-directory touches per `docs/superpowers/specs/2026-04-25-dispatch-critics-hooks-design.md`.

**Architecture:** Pure bash + JSON + markdown — no Rust code. The orchestrator script (`dispatch-critics.sh`) spawns `claude --print` per critic in parallel, collects per-critic output to `.claude/critic-output-<name>.txt`, aggregates verdicts. The hard block is the `.githooks/pre-commit` Git hook that reads those files at commit time. Claude-side hooks (PreToolUse + Stop) keep verdicts fresh.

**Tech Stack:** Bash 4+, `jq` for JSON parsing, `claude --print` for headless agent dispatch, `git`, `grep`, `awk`, `find`, `stat`.

## Architectural Impact Statement

- **Existing primitives searched:** `.claude/skills/critic-*/SKILL.md` (6 critic skills, landed). `.claude/settings.json` (existing hooks: SessionStart for constitution auto-load). `.gitignore` (existing exception: `!.claude/settings.json`, `!.claude/skills/`). No prior `.claude/scripts/` content. No prior `.githooks/`.
- **Decision:** new — first script-side automation in `.claude/`. Pattern is novel for this project but well-established in the broader Claude Code skill ecosystem.
- **Rule-compiler touchpoints:** none — no DSL changes, no engine code touched.
- **Hand-written downstream code:** none — entirely orchestration scripts that wrap existing tools (`claude`, `git`, `grep`).
- **Constitution check:**
  - P1: PASS — no engine extension; no rule logic introduced.
  - P2–P11: PASS — no engine code, layout, RNG, events, or runtime touched.
- **Re-evaluation:** [x] AIS reviewed at design phase. [x] AIS reviewed post-design — 3 bash scripts (~200 ln), 1 Git hook (~75 ln, extends existing cargo-check pre-commit), 1 thin SKILL.md (~60 ln), settings.json + .gitignore + CLAUDE.md + docs/llms.txt updated. No engine code touched, no rule logic introduced. Smoke tests: pre-tool hook 4/4 blocking patterns work; Stop hook no-engine no-op exits 0. Pre-commit smoke surfaced pre-existing cargo check errors in src/bin/xtask/chronicle_cmd.rs (out of scope; users will need `--no-verify` for unrelated commits until those are fixed).

---

## File Structure

```
.claude/
  scripts/
    dispatch-critics.sh           — orchestrator (NEW; ~120 lines)
    pre-tool-engine-edit.sh       — PreToolUse hook (NEW; ~50 lines)
    session-end-engine-review.sh  — Stop hook (NEW; ~30 lines)
  skills/
    dispatch-critics/SKILL.md     — thin doc + invocation guide (NEW; ~50 lines)
  settings.json                    — MODIFIED: add 2 hook entries

.githooks/
  pre-commit                       — Git pre-commit hook (NEW; ~50 lines)

.gitignore                          — MODIFIED: add !.claude/scripts/ + ignore critic-output + allowlist-gate-approved
CLAUDE.md                           — MODIFIED: add one-time setup note for hooks
docs/llms.txt                       — MODIFIED: add dispatch-critics entry
```

Verification per task: each script runs without errors on a no-op input; the bash itself parses with `bash -n`; smoke tests for the orchestrator run on a tiny diff.

---

### Task 1: `.gitignore` exceptions for new paths

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current state**

```bash
grep -n "\.claude" .gitignore
```

Expected: existing rules `.claude/*`, `!.claude/settings.json`, `!.claude/skills/`.

- [ ] **Step 2: Add exceptions for `scripts/` and explicit ignores for transient files**

Use the `Edit` tool to replace the existing block:

Old:
```
.claude/*
!.claude/settings.json
!.claude/skills/
```

New:
```
.claude/*
!.claude/settings.json
!.claude/skills/
!.claude/scripts/
.claude/critic-output-*.txt
.claude/allowlist-gate-approved
```

- [ ] **Step 3: Verify**

```bash
git check-ignore -v .claude/scripts/dispatch-critics.sh 2>&1 ; echo "(should be empty/exit-1 — file not ignored)"
git check-ignore -v .claude/critic-output-foo.txt 2>&1 | head -1
git check-ignore -v .claude/allowlist-gate-approved 2>&1 | head -1
```

The first should print nothing (path not ignored). The second and third should print a `.gitignore:N` line (path ignored).

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore(gitignore): allow .claude/scripts/; ignore critic-output and gate flag"
```

---

### Task 2: `dispatch-critics.sh` orchestrator

**Files:**
- Create: `.claude/scripts/dispatch-critics.sh`

- [ ] **Step 1: Write the script**

Use the `Write` tool. Content (verbatim from spec §5):

```bash
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
```

- [ ] **Step 2: Make executable + bash-syntax check**

```bash
chmod +x .claude/scripts/dispatch-critics.sh
bash -n .claude/scripts/dispatch-critics.sh && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 3: Smoke test — run with `--critic compiler-first --target HEAD~1..HEAD`**

```bash
bash .claude/scripts/dispatch-critics.sh --critic compiler-first --target HEAD~1..HEAD 2>&1 | head -20
```

Expected: prints "Dispatching 1 critic(s)..." line; spawns `claude --print`; eventually prints "=== Critic verdicts ===" with one row. The actual VERDICT depends on what the critic finds in HEAD~1..HEAD; pass or fail is fine for the smoke test, just needs to render the expected output shape.

If `claude --print` is not on PATH in this environment, smoke fails — fine for this task; document in commit message.

- [ ] **Step 4: Commit**

```bash
git add .claude/scripts/dispatch-critics.sh
git commit -m "feat(scripts): dispatch-critics.sh orchestrator with parallel claude --print"
```

---

### Task 3: `pre-tool-engine-edit.sh`

**Files:**
- Create: `.claude/scripts/pre-tool-engine-edit.sh`

- [ ] **Step 1: Write the script** (verbatim from spec §4.1)

```bash
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
```

- [ ] **Step 2: Executable + syntax check**

```bash
chmod +x .claude/scripts/pre-tool-engine-edit.sh
bash -n .claude/scripts/pre-tool-engine-edit.sh && echo "syntax OK"
```

- [ ] **Step 3: Smoke test — feed a no-op JSON**

```bash
echo '{"tool_input":{"file_path":"docs/foo.md","new_string":"hello"}}' | bash .claude/scripts/pre-tool-engine-edit.sh ; echo "exit: $?"
```

Expected: `exit: 0` (no block; doc edit is unrelated to engine).

- [ ] **Step 4: Smoke test — feed a `engine_rules/src/X.rs` edit (should block)**

```bash
echo '{"tool_input":{"file_path":"crates/engine_rules/src/physics/foo.rs","new_string":"// edit"}}' | bash .claude/scripts/pre-tool-engine-edit.sh 2>&1 ; echo "exit: $?"
```

Expected: `BLOCK: crates/engine_rules/src/physics/foo.rs is in engine_rules/...` and `exit: 2`.

- [ ] **Step 5: Smoke test — feed a `lib.rs` in engine_rules (should NOT block)**

```bash
echo '{"tool_input":{"file_path":"crates/engine_rules/src/lib.rs","new_string":"// edit"}}' | bash .claude/scripts/pre-tool-engine-edit.sh ; echo "exit: $?"
```

Expected: `exit: 0` (lib.rs is the carve-out).

- [ ] **Step 6: Smoke test — `impl CascadeHandler` in engine/src/ should block**

```bash
echo '{"tool_input":{"file_path":"crates/engine/src/foo.rs","new_string":"impl CascadeHandler for Foo {}"}}' | bash .claude/scripts/pre-tool-engine-edit.sh 2>&1 ; echo "exit: $?"
```

Expected: `BLOCK: impl CascadeHandler outside crates/engine_rules/. Violates P1.` and `exit: 2`.

- [ ] **Step 7: Commit**

```bash
git add .claude/scripts/pre-tool-engine-edit.sh
git commit -m "feat(scripts): pre-tool-engine-edit.sh fast static blocks for P1 violations"
```

---

### Task 4: `session-end-engine-review.sh`

**Files:**
- Create: `.claude/scripts/session-end-engine-review.sh`

- [ ] **Step 1: Write the script** (verbatim from spec §4.2)

```bash
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
```

- [ ] **Step 2: Executable + syntax check**

```bash
chmod +x .claude/scripts/session-end-engine-review.sh
bash -n .claude/scripts/session-end-engine-review.sh && echo "syntax OK"
```

- [ ] **Step 3: Smoke test — run with no engine files touched (should exit 0 immediately)**

```bash
bash .claude/scripts/session-end-engine-review.sh ; echo "exit: $?"
```

Expected: `exit: 0` (assuming the working tree has no engine changes — verify by `git status -s` showing no `crates/engine*` paths).

- [ ] **Step 4: Commit**

```bash
git add .claude/scripts/session-end-engine-review.sh
git commit -m "feat(scripts): session-end-engine-review.sh — Stop hook keeps verdict files fresh"
```

---

### Task 5: `.githooks/pre-commit` Git hook (the hard block)

**Files:**
- Create: `.githooks/pre-commit`

- [ ] **Step 1: Create directory**

```bash
mkdir -p .githooks
```

- [ ] **Step 2: Write the hook** (verbatim from spec §4.4)

```bash
#!/usr/bin/env bash
# .githooks/pre-commit — critic-verdict gate.
# Blocks commits if any .claude/critic-output-*.txt has VERDICT: FAIL,
# OR if engine source mtime > newest critic-output mtime (stale-PASS guard).

set -e
cd "$(git rev-parse --show-toplevel)"

# Skip if no engine files in this commit's staged content.
staged=$(git diff --cached --name-only)
if ! echo "$staged" | grep -qE '^(crates/engine|crates/engine_rules|crates/engine_data|crates/engine_gpu)/|^assets/sim/'; then
    exit 0
fi

# (1) Find any critic-output file with a FAIL verdict.
fails=()
shopt -s nullglob
for f in .claude/critic-output-*.txt; do
    if grep -m1 '^VERDICT: FAIL' "$f" > /dev/null 2>&1; then
        fails+=("$(basename "$f" .txt | sed 's/^critic-output-//')")
    fi
done
shopt -u nullglob

if [[ ${#fails[@]} -gt 0 ]]; then
    echo "" >&2
    echo "=== ABORT: critic verdicts have unresolved FAILs ===" >&2
    for c in "${fails[@]}"; do
        verdict_line=$(grep -m1 '^VERDICT:' ".claude/critic-output-${c}.txt")
        evidence_line=$(grep -m1 '^EVIDENCE:' ".claude/critic-output-${c}.txt" || echo "EVIDENCE: (none cited)")
        echo "  critic-${c}: $verdict_line" >&2
        echo "    $evidence_line" >&2
    done
    echo "" >&2
    echo "Fix the cited issues, then re-run:" >&2
    echo "  bash .claude/scripts/dispatch-critics.sh --target WORKING-TREE --all" >&2
    echo "" >&2
    echo "When all critics PASS, commit will be allowed." >&2
    exit 1
fi

# (2) Stale-PASS guard: if engine source has been edited since the last
# critic run, force re-run.
shopt -s nullglob
critic_files=( .claude/critic-output-*.txt )
shopt -u nullglob

if [[ ${#critic_files[@]} -eq 0 ]]; then
    newest_critic=0
else
    newest_critic=$(stat -c %Y "${critic_files[@]}" 2>/dev/null | sort -n | tail -1)
fi

newest_engine=$(find crates/engine crates/engine_rules crates/engine_data crates/engine_gpu assets/sim \
    -type f \( -name '*.rs' -o -name '*.sim' \) 2>/dev/null \
    | xargs stat -c %Y 2>/dev/null | sort -n | tail -1)

if [[ -n "$newest_engine" ]] && [[ "$newest_engine" -gt "${newest_critic:-0}" ]]; then
    echo "" >&2
    echo "=== ABORT: engine source has changed since last critic run (or critics never ran) ===" >&2
    echo "Re-run:" >&2
    echo "  bash .claude/scripts/dispatch-critics.sh --target WORKING-TREE --all" >&2
    echo "" >&2
    exit 1
fi

exit 0
```

- [ ] **Step 3: Executable + syntax check**

```bash
chmod +x .githooks/pre-commit
bash -n .githooks/pre-commit && echo "syntax OK"
```

- [ ] **Step 4: Smoke test — run with no engine staged content (should exit 0)**

```bash
bash .githooks/pre-commit ; echo "exit: $?"
```

Expected: `exit: 0` (no engine files staged → no checks).

- [ ] **Step 5: Smoke test — fake a FAIL verdict file, run with engine staged**

```bash
# Set up:
mkdir -p .claude
cat > .claude/critic-output-compiler-first.txt <<'EOF'
VERDICT: FAIL
EVIDENCE: crates/engine/src/foo.rs:1
REASONING: test failure
TOOLS RUN:
- rg "test"
EOF
# Stage a real engine file (use any tracked file there for the smoke test):
git add crates/engine/.schema_hash 2>/dev/null || git add crates/engine/Cargo.toml 2>/dev/null || true

# Run hook:
bash .githooks/pre-commit 2>&1 | head -20 ; echo "exit: $?"

# Cleanup:
git reset HEAD crates/engine/.schema_hash 2>/dev/null || true
git reset HEAD crates/engine/Cargo.toml 2>/dev/null || true
rm -f .claude/critic-output-compiler-first.txt
```

Expected: `=== ABORT: critic verdicts have unresolved FAILs ===` printed; `exit: 1`.

- [ ] **Step 6: Commit**

```bash
git add .githooks/pre-commit
git commit -m "feat(githooks): pre-commit critic-verdict gate (hard block on FAIL or stale)"
```

---

### Task 6: `dispatch-critics` skill (thin doc)

**Files:**
- Create: `.claude/skills/dispatch-critics/SKILL.md`

- [ ] **Step 1: Make directory**

```bash
mkdir -p .claude/skills/dispatch-critics
```

- [ ] **Step 2: Write SKILL.md**

```markdown
---
name: dispatch-critics
description: Use when reviewing engine changes before committing — manual gate for the allowlist edit workflow, or to spot-check a specific principle on uncommitted work. Wraps the bash orchestrator at .claude/scripts/dispatch-critics.sh.
---

# Skill: dispatch-critics

## Role
This skill is a thin documentation wrapper around the bash orchestrator at `.claude/scripts/dispatch-critics.sh`. The actual work happens in the script. This SKILL.md tells you when and how to invoke it.

## When to use

- **Before committing engine changes**: run `bash .claude/scripts/dispatch-critics.sh --target WORKING-TREE --all`. Output goes to `.claude/critic-output-<name>.txt` (per critic) and stdout (verdict summary). The `.githooks/pre-commit` hook will block the commit if any verdict is FAIL.
- **Before editing `engine/build.rs` ALLOWED_DIRS / ALLOWED_TOP_LEVEL**: run `bash .claude/scripts/dispatch-critics.sh --critic compiler-first --critic allowlist-gate --target WORKING-TREE`. Both must PASS. Then `touch .claude/allowlist-gate-approved` to unlock the edit.
- **To spot-check a single principle**: `bash .claude/scripts/dispatch-critics.sh --critic <name> --target <ref>`. Useful when iterating on a specific concern.

## Critic names

The 6 critics live at `.claude/skills/critic-<name>/SKILL.md`:

- `compiler-first` (P1) — engine extension must be DSL-emitted.
- `schema-bump` (P2) — layout changes require `.schema_hash` regen.
- `cross-backend-parity` (P3) — Serial + GPU produce byte-equal output.
- `no-runtime-panic` (P10) — hot path uses Result + saturating ops.
- `reduction-determinism` (P11) — float reductions sort or use fixed-point.
- `allowlist-gate` (special) — `engine/build.rs` allowlist edits.

## Default behavior (`--all`)

Without `--critic` overrides, the orchestrator's heuristic picks critics based on what files changed. See `.claude/scripts/dispatch-critics.sh` (the heuristic block) for the path → critics mapping.

## Output format

Each critic returns:
```
VERDICT: PASS | FAIL
EVIDENCE: <file:line>[, <file:line>, ...]
REASONING: <one paragraph>
TOOLS RUN:
- <command>
```

Empty `TOOLS RUN` → automatic FAIL (the critic didn't actually verify anything).

## Hook integration

- `PreToolUse` hook (`.claude/scripts/pre-tool-engine-edit.sh`) blocks at write-time on obvious P1 violations — runs synchronously, ~50ms.
- `Stop` hook (`.claude/scripts/session-end-engine-review.sh`) runs `dispatch-critics --all` at session end if engine was touched. Keeps verdict files fresh.
- Git pre-commit hook (`.githooks/pre-commit`) is the **hard block**: aborts `git commit` if any FAIL verdict exists or if engine source mtime > newest critic output (stale-PASS guard).

## One-time setup

After cloning the repo, configure Git to use the project hooks:

```bash
git config core.hooksPath .githooks
```

Without this, the `.githooks/pre-commit` hook won't fire and the hard block doesn't engage.
```

- [ ] **Step 3: Verify frontmatter**

```bash
head -4 .claude/skills/dispatch-critics/SKILL.md
```

Expected: `---`, `name: dispatch-critics`, `description: ...`, `---`.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/dispatch-critics/SKILL.md
git commit -m "docs(skills): dispatch-critics SKILL.md — thin doc wrapper for the orchestrator"
```

---

### Task 7: `settings.json` hooks updates

**Files:**
- Modify: `.claude/settings.json`

- [ ] **Step 1: Read current state**

```bash
cat .claude/settings.json
```

Expected current content (from prior constitution work):

```json
{
  "hooks": {
    "SessionStart": [
      { "type": "command", "command": "cat docs/constitution.md" }
    ]
  }
}
```

- [ ] **Step 2: Add `PreToolUse` and `Stop` entries via `jq`**

```bash
tmp=$(mktemp)
jq '.hooks += {
  "PreToolUse": [
    {
      "matcher": "Edit|Write|MultiEdit",
      "hooks": [
        { "type": "command", "command": ".claude/scripts/pre-tool-engine-edit.sh" }
      ]
    }
  ],
  "Stop": [
    { "type": "command", "command": ".claude/scripts/session-end-engine-review.sh" }
  ]
}' .claude/settings.json > "$tmp" && mv "$tmp" .claude/settings.json
```

- [ ] **Step 3: Verify JSON is valid + has all three hook types**

```bash
jq '.hooks | keys' .claude/settings.json
```

Expected:

```json
[
  "PreToolUse",
  "SessionStart",
  "Stop"
]
```

- [ ] **Step 4: Commit**

```bash
git add .claude/settings.json
git commit -m "feat(hooks): wire PreToolUse + Stop to dispatch-critics scripts"
```

---

### Task 8: Update `CLAUDE.md` and `docs/llms.txt`

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/llms.txt`

- [ ] **Step 1: Add a one-time-setup note + critics-system bullet to `CLAUDE.md`**

Use `Edit` to update the "Conventions" section. Find:

```
- Critic skills (`.claude/skills/critic-*/SKILL.md`) gate engine extensions. Invoke via Skill tool for in-context review or via Agent dispatch (paste skill body into prompt) for fresh-context independent verdicts. The `engine/build.rs` allowlist gate requires two parallel Agent invocations — see `docs/superpowers/specs/2026-04-25-engine-crate-split-design.md` §5.2.
```

Replace with:

```
- Critic skills (`.claude/skills/critic-*/SKILL.md`) gate engine extensions. Use `bash .claude/scripts/dispatch-critics.sh --target WORKING-TREE --all` to dispatch automatically; `.githooks/pre-commit` is the hard block at commit time. Manual invocation guide: see `.claude/skills/dispatch-critics/SKILL.md`.
- **One-time setup after clone:** `git config core.hooksPath .githooks` enables the pre-commit critic-verdict block. Without it, the hard block doesn't engage.
```

- [ ] **Step 2: Verify CLAUDE.md line count**

```bash
wc -l CLAUDE.md
```

Expected: ≤105 (we have a ~52 line baseline; adding 2 bullets keeps us well under).

- [ ] **Step 3: Add `dispatch-critics` to `docs/llms.txt` Critics section**

Find the existing "## Critics" section and add:

```
- [dispatch-critics](.claude/skills/dispatch-critics/SKILL.md): orchestrator skill. Wraps the bash script that runs all applicable critics in parallel.
```

Verify the file is still well-formed:

```bash
grep -oE '\(\.claude/skills/[^)]*\)' docs/llms.txt | tr -d '()' | sort -u | while read f; do
    [ -f "$f" ] || echo "MISSING: $f"
done
```

Expected: zero output.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/llms.txt
git commit -m "docs: register dispatch-critics + one-time hook setup in CLAUDE.md + llms.txt"
```

---

### Task 9: Configure repo-local Git hook path

**Files:**
- (no file changes; one-time `git config` for this clone)

- [ ] **Step 1: Configure `core.hooksPath`**

```bash
git config core.hooksPath .githooks
```

This is per-clone, NOT committed. Each developer (and each fresh clone) runs this once. CLAUDE.md (Task 8) documents it.

- [ ] **Step 2: Verify**

```bash
git config --get core.hooksPath
```

Expected: `.githooks`.

- [ ] **Step 3: Smoke test the hook chain — make a no-op commit**

```bash
echo "# test" >> docs/.tmp-hook-test.md
git add docs/.tmp-hook-test.md
git commit -m "smoke: pre-commit hook chain test (no engine touched)" 2>&1 | tail -5
```

Expected: commit succeeds (no engine staged, hook returns 0). Cleanup:

```bash
git reset HEAD~1
rm docs/.tmp-hook-test.md
```

- [ ] **Step 4: No commit needed; this step modifies clone-local config only.** Verify by running `git config --get core.hooksPath` and confirming `.githooks` is reported.

---

### Task 10: Documentation freshness — `dispatch-critics` invocation in CLAUDE.md verified

**Files:**
- Modify: this plan file (`docs/superpowers/plans/2026-04-25-dispatch-critics-hooks-impl.md`) — AIS post-design re-eval

- [ ] **Step 1: Tick the "post-design" checkbox**

Use `Edit` on this plan file. Find:

```markdown
- **Re-evaluation:** [ ] AIS reviewed at design phase. [ ] AIS reviewed post-design.
```

Replace with:

```markdown
- **Re-evaluation:** [x] AIS reviewed at design phase. [x] AIS reviewed post-design — 3 bash scripts (~200 lines), 1 Git hook (~50 lines), 1 thin SKILL.md, settings.json + .gitignore + CLAUDE.md + llms.txt updates. No engine code touched, no rule logic introduced. Hook chain smoke-tested.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-04-25-dispatch-critics-hooks-impl.md
git commit -m "chore(plan): tick dispatch-critics-hooks post-impl AIS re-evaluation"
```

---

## Self-Review

**Spec coverage:** every section of the spec has at least one task:
- §3 architecture → Tasks 1, 2, 3, 4, 5, 6, 7 each land a piece of the layout.
- §4.1 pre-tool hook → Task 3.
- §4.2 Stop hook (session-end-engine-review) → Task 4.
- §4.4 Git pre-commit hook → Task 5.
- §4.5b settings.json → Task 7.
- §5 dispatch-critics.sh → Task 2.
- §6 critic selection heuristic → embedded in Task 2's script.
- §7 allowlist gate flag → covered by Task 3 (block check) + the manual workflow documented in Task 6 (skill).
- §8.1 dispatch-critics skill → Task 6.
- §8.2 xtask wrapper → dropped (D8 in spec).
- §9 decision log → no implementation needed; documented in spec.
- §10 out of scope → no tasks needed.
- One-time `git config core.hooksPath` → Task 9.

**Placeholder scan:** no TBDs, no "implement later", no "similar to Task N". Each task has the explicit script body or jq command. Smoke tests have explicit expected outputs.

**Type consistency:** script names, skill paths, file paths, env-var names (`TARGET`, `MODEL`, `CRITICS`) consistent across tasks. The 6 critic names match the existing `.claude/skills/critic-*/SKILL.md` paths.

---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task; review between tasks.
2. **Inline Execution** — execute in this session via executing-plans.

Which approach?
