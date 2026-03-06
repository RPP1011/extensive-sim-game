# Guardrails (Signs)

> Lessons learned from failures. Read before acting.

## Core Signs

### Sign: Read Before Writing
- **Trigger**: Before modifying any file
- **Instruction**: Read the file first
- **Added after**: Core principle

### Sign: Test Before Commit
- **Trigger**: Before committing changes
- **Instruction**: Run required tests and verify outputs
- **Added after**: Core principle

---

## Learned Signs

### Sign: Isolate Story Changes From Known Full-Suite Failures
- **Trigger**: When `cargo test` fails in tests not touched by the current story.
- **Instruction**: Verify story-targeted tests first, then run full suite and explicitly record unchanged failing tests and their names in `.ralph/errors.log` before commit.
- **Added after**: US-002 iteration 1 - repeated failures in unrelated game_core/non-headless smoke tests.

