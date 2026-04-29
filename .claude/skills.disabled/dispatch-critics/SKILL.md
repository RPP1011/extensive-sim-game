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
