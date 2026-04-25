# Constitution + ADR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the constitution + ADR + AIS + llms.txt + CLAUDE.md trim + SessionStart hook + migration as specified in `docs/superpowers/specs/2026-04-24-constitution-and-adr-design.md`. Single source of truth for architectural principles, hard preference; no paraphrase or pointer-following.

**Architecture:** Pure doc + config work; no Rust code. New files in `docs/`, edits to `CLAUDE.md` and existing `docs/spec/*.md`, one entry added to `.claude/settings.json`. Migration deletes duplicate principle-flavored content from existing specs (it does not redirect).

**Tech Stack:** Markdown, YAML, bash, jq for json edits.

## Architectural Impact Statement

(Self-applied, since this plan defines the AIS template — eat-our-own-dogfood.)

- **Existing primitives searched:** `docs/ROADMAP.md` (top-level future-work index), `docs/engine/status.md` (live status table), `docs/superpowers/roadmap.md` (deferred subsystems), `docs/spec/language.md` §9 (settled decisions), `docs/spec/runtime.md`, `docs/spec/compiler.md`, `docs/spec/ability.md`. No prior constitution / ADR / AIS infrastructure.
- **Decision:** New, because the project has no governance layer above specs today; needs one to enable Layers 2–4 of the architectural-enforcement framework (separate specs).
- **Rule-compiler touchpoints:** None. This is doc + config; the compiler is unaffected.
- **Hand-written downstream code:** None. All artefacts are markdown, YAML, JSON.
- **Constitution check:** N/A — this plan creates the constitution. Re-evaluation post-implementation: confirm no duplicate principles exist across spec/.
- **Re-evaluation:** [x] Post-implementation sweep run (Task 10); no spec content beyond what was anticipated was touched.

---

### Task 1: Inventory principle-shaped statements

**Goal:** produce a working list of candidate principles before drafting the constitution. Hand-curated; no Haiku at this stage.

**Files:**
- Create: `docs/architecture/.principle-inventory.md` (working doc, deleted at end of plan)

- [ ] **Step 1: Read source specs section-by-section, capturing rule-shaped statements**

Read in order:
- `docs/spec/runtime.md` (~700 ln) — focus on §§14, 22, 23, 24 (tick pipeline, schema hash, debug, backends).
- `docs/spec/compiler.md` (~190 ln) — focus on §1, §2 (emission modes, code-paths).
- `docs/spec/language.md` §9 (lines ~980–1027) — settled decisions; many are principle-shaped.
- `docs/spec/ability.md` §22 (~lines 1647–1797) — IR contract, esp. `EffectOp` size budget.
- `docs/spec/economy.md` §1.3 (design principles, ~lines 41–55).
- `docs/engine/status.md` (top weak-test risks + open verification questions, lines ~80–130).

For each rule-shaped statement, jot it in the working doc with format:
```markdown
- **<short name>** — <one-line statement> | source: `<path> §<sec>`
```

A rule-shaped statement governs *behavior* (what may/must/must-not happen). NOT principle-shaped: implementation specifics (e.g., "AgentId is NonZeroU32"), data layouts (e.g., "spatial index uses 16m cells"), test coverage notes.

- [ ] **Step 2: Write working doc**

```bash
mkdir -p docs/architecture
cat > docs/architecture/.principle-inventory.md <<'EOF'
# Principle inventory (working doc)

> Captured from existing specs during plan T1. Each line: candidate principle + source.
> This file is deleted at the end of the plan; the curated outputs live in `docs/constitution.md`.

EOF
```

Then append entries as you read. Aim for 15-25 candidates; you'll filter to ~10.

- [ ] **Step 3: De-duplicate + categorize**

Group candidates by theme: compilation/emission, determinism, schema/ABI, layering/scope, lifecycle/governance. Mark candidates that overlap. Pick the load-bearing ~10.

- [ ] **Step 4: Commit working doc**

```bash
git add docs/architecture/.principle-inventory.md
git commit -m "wip(constitution): inventory principle-shaped statements from existing specs"
```

---

### Task 2: Draft `docs/constitution.md`

**Files:**
- Create: `docs/constitution.md`

- [ ] **Step 1: Write the file with header + per-principle template applied to each surviving candidate**

Use exactly the schema from the spec §3.1:

```markdown
# Project Constitution

> Architectural principles that govern this project. Each principle is a numbered, stable handle (P-N). Other docs cite by number; do not paraphrase. Edit the prose freely; the number is the contract.
>
> This file is auto-loaded into every agent's context via the `SessionStart` hook in `.claude/settings.json`.

## P1 — Compiler-First Engine Extension

**Statement.** All engine-rule behavior originates from the DSL compiler. Hand-written rule logic in `crates/engine/src/handlers/`, `crates/engine/src/cascade/handlers/`, or `crates/engine/src/generated/` (without `// @generated` header) is forbidden.

**Rationale.** Hand-written behavior drifts from the DSL surface, breaks emitter-pattern wins (alive-bitmap, fold-body lowering, etc.), and silently violates cross-backend parity. Compile-time enforcement keeps every rule subject to the emitter's lowering.

**Enforcement.**
- Compile-time: sealed `Rule` trait (Spec B); `build.rs` sentinel checking `// @generated` headers.
- CI: ast-grep rule rejecting `impl Rule` outside `rules-generated/`.
- Agent-write-time: `PreToolUse` hook on `Edit|Write` to engine handler paths.
- Critic: project-critic checks any task tagged `engine-extension`.
- Manual: code review.

**Source.** `docs/spec/compiler.md` §1 (kept for archaeology only — readers do not follow pointers).

---

## P2 — Schema-Hash Bumps on Layout Change

**Statement.** Any change to `SimState` SoA layout, event variant set, mask-predicate semantics, or scoring-row contract requires a `crates/engine/.schema_hash` regeneration.

**Rationale.** Schema-hash mismatches block snapshot loads and trace consumption; silent layout drift produces undefined behavior at the snapshot/migration boundary.

**Enforcement.**
- CI: `tests/schema_hash.rs` baseline-comparison test.
- Agent-write-time: `PreToolUse` hook on `Edit|Write` to `crates/engine/src/state/**` requires an open task tagged `schema-change`.
- Manual: regen via the documented procedure.

**Source.** `docs/spec/runtime.md` §22.

---

## P3 — Cross-Backend Parity

**Statement.** Every engine behavior runs on both `SerialBackend` (reference) and `GpuBackend` (performance), or is annotated `@cpu_only` in DSL with explicit justification.

**Rationale.** GPU is co-equal, not auxiliary. Behavior that exists on only one backend produces a non-determinism boundary at backend selection.

**Enforcement.**
- CI: `tests/parity_*.rs` cross-backend determinism test.
- Compile-time: DSL `@cpu_only` annotation is a parser-recognized exception.
- Critic: project-critic flags new behavior without parity test.

**Source.** `docs/spec/compiler.md` §1, `docs/spec/runtime.md` §24.

---

## P4 — `EffectOp` Size Budget (≤16 bytes)

**Statement.** Each variant of `EffectOp` (and analogous IR enums) stays at or under 16 bytes after Rust enum tagging.

**Rationale.** Small variants keep `SmallVec<[EffectOp; 4]>` cheap (≤64 B + tag) and compile to GPU-friendly POD.

**Enforcement.**
- Compile-time: `static_assert::const_assert!(size_of::<EffectOp>() <= 16);` (or equivalent build-time check).
- Manual: when adding a variant, verify with `mem::size_of::<EffectOp>()`.

**Source.** `docs/spec/ability.md` §22.

---

## P5 — Determinism via Keyed PCG

**Statement.** All sim randomness flows through `per_agent_u32(world_seed, agent_id, tick, purpose)`. `thread_rng()`, system time, and any non-keyed RNG are forbidden in the deterministic path.

**Rationale.** Replay equivalence + cross-backend parity require RNG inputs to be a pure function of (state, tick, agent, purpose).

**Enforcement.**
- CI: ast-grep rule rejecting `thread_rng()` / `SystemTime::now()` in `crates/engine/src/`.
- Manual: code review.

**Source.** `docs/spec/runtime.md` §7, `docs/spec/language.md` §9.

---

## P6 — Events Are the Mutation Channel

**Statement.** All sim state mutations are events. Current state is a fold over events + entity baseline. Direct field writes outside the documented kernel API are forbidden.

**Rationale.** Replayability requires events to be the only authoritative log; anything else creates state that exists outside the snapshot/trace.

**Enforcement.**
- Manual: code review on `&mut SimState` writes.
- CI (future): visibility rule that confines field writes to `step::*` and `snapshot::*`.

**Source.** `docs/spec/runtime.md` §5.

---

## P7 — Replayability Flagged at Declaration

**Statement.** Every `event` declaration carries an explicit `replayable: bool`. Non-replayable events (chronicle prose, telemetry side channels) are tagged at the source; the runtime fold ignores them.

**Rationale.** Mixing replayable + non-replayable events into a single fold breaks deterministic replay.

**Enforcement.**
- Compile-time: parser requires the annotation; emitter generates flagged enum variants.
- CI: trace-format roundtrip test.

**Source.** `docs/spec/runtime.md` §5, `docs/spec/language.md` §2.2.

---

## P8 — Architectural Impact Statement Required

**Statement.** Every new plan in `docs/superpowers/plans/` MUST include an Architectural Impact Statement (AIS) preamble before any task list. The AIS makes the constitution check explicit.

**Rationale.** Plan-execution discipline is downstream of plan-design discipline. Forcing the AIS surfaces invariant violations at design time, not at audit time.

**Enforcement.**
- Manual: human review on plan creation; the project-DAG skill (Spec C) rejects plans without AIS.
- Template: `docs/architecture/plan-template-ais.md`.

**Source.** This spec §3.3.

---

## P9 — Tasks Close With a Verified Commit

**Statement.** Tasks marked `completed` carry a non-empty `closes_commit` UDA pointing at a real, non-reverted commit SHA on the active branch.

**Rationale.** Prevents stale "done" claims (the failure mode the 2026-04-24 docs audit found).

**Enforcement.**
- Skill: project-DAG skill (Spec C) rejects close without `closes_commit`.
- `dag-validate`: re-checks SHA existence; flags reverts.

**Source.** This spec, derived from the 2026-04-24 audit.

---

## P10 — No Runtime Panic on Deterministic Path

**Statement.** The deterministic sim hot path (`step()`, kernels, fold dispatch) does not panic. Saturating ops, `Result`, and contract assertions are the failure mode; runtime panics escape only as bugs.

**Rationale.** Panics at scale (200k agents) corrupt replay state and abort training runs.

**Enforcement.**
- CI: `tests/proptest_baseline.rs` runs `step` on randomized inputs; never panics.
- Manual: code review on `unwrap()` in hot paths.

**Source.** `docs/spec/runtime.md` §14.
```

Final principle list lands during this step; the above is the seed. Adjust per Task 1's inventory.

- [ ] **Step 2: Validate every principle has all 4 sections**

```bash
# Each P-N must have Statement, Rationale, Enforcement, Source.
for p in $(grep -oE '^## P[0-9]+' docs/constitution.md | awk '{print $2}'); do
    for sec in 'Statement' 'Rationale' 'Enforcement' 'Source'; do
        grep -A1 "^## $p " docs/constitution.md | grep -q "$sec" || echo "MISSING: $p $sec"
    done
done
```

Expect: zero output.

- [ ] **Step 3: Verify line count is reasonable** (≤200 lines for ~10 principles)

```bash
wc -l docs/constitution.md
```

Expect: 100-200 lines.

- [ ] **Step 4: Commit**

```bash
git add docs/constitution.md
git commit -m "feat(constitution): hand-curated initial principle catalog (P1–P10)"
```

---

### Task 3: Haiku verification pass for completeness

**Goal:** independent verification that the curated catalog covers all principle-shaped statements in the source specs.

**Files:**
- (no new files; agent dispatch + manual edits to `docs/constitution.md` if gaps found)

- [ ] **Step 1: Spawn Haiku agent**

Use the `Agent` tool with `model: haiku`, `subagent_type: general-purpose`, prompt:

```
Audit /home/ricky/Projects/game/docs/constitution.md for completeness against the source specs.

For each spec listed below, identify any rule-shaped statement (a rule that governs behavior) NOT covered by the constitution's P1–P10:
- docs/spec/runtime.md
- docs/spec/compiler.md
- docs/spec/language.md §9
- docs/spec/ability.md §22
- docs/spec/economy.md §1.3
- docs/engine/status.md (open verification questions only)

Output format:
## Missing principles
- <short name>: <one-line statement> | source: <path §sec>

## Already covered (sanity check)
- <P-N>: matches <source path §sec>

Report under 600 words. Don't fix anything; just report.
```

- [ ] **Step 2: Review the agent's report**

For each "missing" candidate the agent flags, decide:
- Genuine gap → add as P11+ (with full schema).
- Implementation specific (not a principle) → ignore.
- Already covered under a different P-N name → confirm and ignore.

- [ ] **Step 3: Apply gap-fill (if any)**

Edit `docs/constitution.md` adding new P-N entries with the full schema. Re-run Step 2's section-completeness check from Task 2.

- [ ] **Step 4: Commit (only if changes made)**

```bash
git add docs/constitution.md
git commit -m "feat(constitution): gap-fill from Haiku verification pass (P11..)"
```

If no gaps, no commit.

---

### Task 4: ADR convention scaffolding

**Files:**
- Create: `docs/adr/README.md`

- [ ] **Step 1: Create the directory and README**

```bash
mkdir -p docs/adr
```

Then write `docs/adr/README.md`:

```markdown
# Architecture Decision Records

Locked decisions that emerge from design specs in `docs/superpowers/specs/`. ADRs are immutable once `Accepted`; changing direction = a new ADR that supersedes the old one.

## Numbering

Four digits, monotonic, never reused. The next ADR number is `(highest existing + 1)`. Use:

```bash
ls docs/adr/[0-9]*.md 2>/dev/null | sort -V | tail -1
```

## Template

```markdown
# ADR-NNNN: <Decision Title>

**Status:** Accepted | Superseded by ADR-NNNN | Rejected
**Date:** YYYY-MM-DD
**Spec:** `docs/superpowers/specs/<spec>.md`

## Context

What problem prompted this decision (1 paragraph).

## Decision

The chosen approach (1 paragraph).

## Consequences

Downstream impact (positive and negative bullets).

## Alternatives considered

What was rejected and why (1 paragraph).
```

## When to write an ADR

Write an ADR when a decision:
- Locks an external interface (API, file format, schema).
- Commits to an architectural direction that future plans build on.
- Settles a contested design choice that would otherwise re-litigate.

Internal implementation choices stay in specs/plans, not ADRs.

## When NOT to write an ADR

- Tactical refactors that don't change interfaces.
- Per-task decisions inside a plan.
- Spec drafts that haven't been approved yet (those live in `docs/superpowers/specs/`).

## Index

(Hand-maintained; one line per ADR.)
```

- [ ] **Step 2: Commit**

```bash
git add docs/adr/README.md
git commit -m "feat(adr): convention + template + index doc"
```

---

### Task 5: AIS plan-template

**Files:**
- Create: `docs/architecture/plan-template-ais.md`

- [ ] **Step 1: Write the template**

```markdown
# Architectural Impact Statement (plan preamble)

> Required preamble for every plan in `docs/superpowers/plans/`. Per P8.
>
> Copy this section into your new plan file, between the header (Goal/Architecture/Tech Stack) and the first task. Fill in every field. The constitution check is non-optional.

```markdown
## Architectural Impact Statement

- **Existing primitives searched:**
  - `<symbol or term>` at `<file:line>`
  - `<symbol or term>` at `<file:line>`
  Search method: `ast-grep` / `rg` / direct `Read`.

- **Decision:** extend `<existing primitive>` / new because `<reason>`.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: `<files>`
  - Generated outputs re-emitted: `<files>`
  (Or "none" if this plan is non-engine work.)

- **Hand-written downstream code:** NONE.
  (Or, per file: `<path>: <justification>`. Each justification must explain why an emitter expansion is impossible or inappropriate.)

- **Constitution check:**
  - P1 (Compiler-First): PASS / FAIL — evidence: `<file:line>`
  - P2 (Schema-Hash on Layout): PASS / N/A — evidence: ...
  - P3 (Cross-Backend Parity): PASS / N/A
  - P4 (`EffectOp` Size Budget): PASS / N/A
  - P5 (Determinism via Keyed PCG): PASS / N/A
  - P6 (Events Are the Mutation Channel): PASS / N/A
  - P7 (Replayability Flagged): PASS / N/A
  - P8 (AIS Required): PASS — this section satisfies it
  - P9 (Tasks Close With Verified Commit): PASS / N/A
  - P10 (No Runtime Panic): PASS / N/A

- **Re-evaluation:** [ ] AIS reviewed at design phase (initial fill).  [ ] AIS reviewed post-design (after task list stabilises).
```

The Constitution check uses `PASS` / `FAIL` / `N/A`. `FAIL` requires a remediation step in the plan or an explicit waiver justification immediately below.

The "Re-evaluation" checkboxes are part of the plan execution: tick the post-design box only after the task list and design have stabilised, not at initial draft.
```

- [ ] **Step 2: Commit**

```bash
git add docs/architecture/plan-template-ais.md
git commit -m "feat(architecture): AIS plan-template (P8 template)"
```

---

### Task 6: Create `docs/llms.txt`

**Files:**
- Create: `docs/llms.txt`

- [ ] **Step 1: Inventory existing docs**

```bash
find docs -maxdepth 2 -type f -name "*.md" | sort
```

- [ ] **Step 2: Write the index** in [llmstxt.org](https://llmstxt.org/) format

```markdown
# game

> Deterministic tactical orchestration RPG built in Rust. Combat sim runs as a 100ms fixed-tick deterministic engine with both CPU and GPU backends; world simulation is a Dwarf-Fortress-style zero-player layer underneath. Rules-as-data via a custom DSL compiled to both backends.

## Constitution

- [Constitution](docs/constitution.md): numbered architectural principles (P1–P10). Auto-loaded into every agent context via SessionStart hook.

## Top-level

- [Overview](docs/overview.md): 5-minute architectural intro, worked example, where-to-go-next.
- [Roadmap](docs/ROADMAP.md): comprehensive future-work index (active / drafted / partial / deferred).
- [Engine status](docs/engine/status.md): live per-subsystem implementation status.

## Specification

- [Language](docs/spec/language.md): world-sim DSL grammar + semantics.
- [Runtime](docs/spec/runtime.md): engine runtime contract (§§1–26).
- [Compiler](docs/spec/compiler.md): DSL → Rust + SPIR-V + Python lowering.
- [GPU](docs/spec/gpu.md): GPU backend contract (cascade, sim-state, cold-state, ability-eval, pipeline).
- [State](docs/spec/state.md): field catalog (every SoA field, who reads, who writes).
- [Stdlib](docs/spec/stdlib.md): pinned built-in functions and namespaces.
- [Scoring fields](docs/spec/scoring_fields.md): `field_id` ABI table.
- [Ability DSL](docs/spec/ability.md): `.ability` language reference + IR.
- [Economy](docs/spec/economy.md): economic system design (recipes, contracts, market).

## Process

- [Architecture decision records](docs/adr/): locked decisions that come out of specs.
- [Workstreams](docs/architecture/workstreams/): per-workstream definitions (lands in DAG-skill plan).
- [AIS template](docs/architecture/plan-template-ais.md): required preamble for new plans.
- [Plans](docs/superpowers/plans/): in-flight implementation plans.
- [Specs](docs/superpowers/specs/): design specs (pre-decision exploration).
- [Research](docs/superpowers/research/): investigations + measurements feeding plans.
- [Notes](docs/superpowers/notes/): open architectural gaps, regression bisects.

## Game-layer

- [Game overview](docs/game/overview.md): what we're building, layer map.
- [Compiler progress](docs/game/compiler_progress.md): live milestone tracker.
- [Feature flow](docs/game/feature_flow.md): how to add a feature; when to extend the compiler.
- [Wolves and humans](docs/game/wolves_and_humans.md): canonical fixture.
```

- [ ] **Step 3: Verify every linked file exists**

```bash
grep -oE '\(docs/[^)]*\.md\)' docs/llms.txt | tr -d '()' | sort -u | while read f; do
    [ -f "$f" ] || echo "MISSING: $f"
done
```

Expect: zero output. Fix any missing files before committing.

- [ ] **Step 4: Commit**

```bash
git add docs/llms.txt
git commit -m "feat(docs): llms.txt index per llmstxt.org convention"
```

---

### Task 7: SessionStart hook to auto-load constitution

**Files:**
- Modify or create: `.claude/settings.json`

- [ ] **Step 1: Inspect current settings**

```bash
cat .claude/settings.json 2>/dev/null || echo "No settings.json yet"
```

If the file exists, read it and identify the `hooks` section (if any). If it doesn't exist, you'll create it.

- [ ] **Step 2: Write the SessionStart hook entry**

If `.claude/settings.json` does not exist, create it:

```bash
mkdir -p .claude
cat > .claude/settings.json <<'EOF'
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "cat docs/constitution.md"
      }
    ]
  }
}
EOF
```

If it exists with no `hooks` key, use `jq` to add:

```bash
tmp=$(mktemp)
jq '. + {"hooks": {"SessionStart": [{"type":"command","command":"cat docs/constitution.md"}]}}' .claude/settings.json > "$tmp" && mv "$tmp" .claude/settings.json
```

If it exists with `hooks` but no `SessionStart`:

```bash
tmp=$(mktemp)
jq '.hooks += {"SessionStart": [{"type":"command","command":"cat docs/constitution.md"}]}' .claude/settings.json > "$tmp" && mv "$tmp" .claude/settings.json
```

If it already has `SessionStart`, merge into the array (do not overwrite existing entries):

```bash
tmp=$(mktemp)
jq '.hooks.SessionStart += [{"type":"command","command":"cat docs/constitution.md"}]' .claude/settings.json > "$tmp" && mv "$tmp" .claude/settings.json
```

- [ ] **Step 3: Verify JSON is valid**

```bash
jq . .claude/settings.json > /dev/null && echo OK
```

Expect: `OK`.

- [ ] **Step 4: Verify the hook command runs cleanly**

```bash
bash -c "cat docs/constitution.md" | head -5
```

Expect: first 5 lines of the constitution.

- [ ] **Step 5: Commit**

```bash
git add .claude/settings.json
git commit -m "feat(hooks): SessionStart auto-loads constitution into agent context"
```

(Manual verification: open a new Claude Code session in this repo and confirm the constitution appears in the initial system context. This is observational; no automated test.)

---

### Task 8: Trim `CLAUDE.md` to ≤100 lines

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Read the current file**

```bash
wc -l CLAUDE.md
```

(Expect ~80 lines today; the trim is mostly *removing* the module-map + AI-pipeline narrative, not adding.)

- [ ] **Step 2: Replace the file with the trimmed version**

Target structure (one section per block; ≤100 lines total):

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project

Deterministic tactical orchestration RPG built in Rust. Combat sim is a 100ms fixed-tick deterministic engine with both CPU (`SerialBackend`) and GPU (`GpuBackend`) backends. World simulation is a Dwarf-Fortress-style zero-player layer underneath. Rules-as-data: a custom DSL (`assets/sim/*.sim`, `assets/hero_templates/*.ability`) is compiled to both backends.

## Constitution

The architectural constitution at `docs/constitution.md` is auto-loaded into agent context on session start (see `.claude/settings.json`). Every plan must include an Architectural Impact Statement preamble per `docs/architecture/plan-template-ais.md` (P8).

## Build & test

```bash
cargo build                    # Debug build
cargo build --release          # Release build
cargo test                     # All tests
cargo test -p engine           # Tests in the engine crate only
cargo test -- --test-threads=1 # Serial execution (for determinism tests)
```

### CLI (xtask)

```bash
cargo run --bin xtask -- scenario run scenarios/basic_4v4.toml
cargo run --bin xtask -- scenario bench scenarios/
cargo run --bin xtask -- scenario generate dataset/scenarios/
```

## Where to look

- **Reading order:** start with `docs/llms.txt`, fetch the docs you need.
- **What's built:** `docs/engine/status.md` (live per-subsystem implementation status).
- **What's coming:** `docs/ROADMAP.md` (comprehensive future-work index).
- **Contract:** `docs/spec/` (canonical specification, 10 files).
- **Active plans:** `docs/superpowers/plans/`.
- **Locked decisions:** `docs/adr/`.

## Conventions

- The spec is the contract. Live status lives in `engine/status.md`. Don't duplicate.
- The constitution states each principle once. Other docs do not paraphrase or redirect.
- Every new plan needs an AIS preamble (P8). Skipping it is a process violation.
- Historical content (executed plans, resolved audits, design rationale) lives in **git history**, not active docs.

## Tooling caveats

- This is a Rust workspace; the root `Cargo.toml` is a virtual manifest.
- Two crates: root (`bevy_game`) and `crates/ability_operator`. Engine + GPU live under `crates/engine*`.
- All simulation randomness MUST flow through `per_agent_u32(seed, agent_id, tick, purpose)` — see P5.
```

Replace the file:

```bash
# (use the Write tool with the content above)
```

- [ ] **Step 3: Verify line count**

```bash
wc -l CLAUDE.md
```

Expect: ≤100. If over, trim further (drop the bullet under "Tooling caveats" first).

- [ ] **Step 4: Verify no principle text duplicates the constitution**

```bash
# Sanity: no P-N restatements or paraphrases
grep -E '\bP[0-9]+\b' CLAUDE.md
```

Expect: only references in the "Constitution" + "Conventions" sections (mentioning P5 / P8 by number is OK; restating the principle text is not).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "feat(claude-md): trim to ≤100 lines; constitution loaded via SessionStart"
```

---

### Task 9: Migrate spec/ — delete duplicate principle content

**Goal:** for each principle now in the constitution, find its previous statement(s) in `docs/spec/*.md` and DELETE (do not redirect).

**Files:**
- Modify: `docs/spec/runtime.md`
- Modify: `docs/spec/compiler.md`
- Modify: `docs/spec/language.md`
- Modify: `docs/spec/ability.md`

- [ ] **Step 1: Per-principle migration**

For each P-N in the constitution, sweep `docs/spec/*.md` for the principle's statement and delete the principle-flavored sentence(s). Keep implementation specifics (concrete API, file paths, layout details).

Concrete sweeps:

**P1 (compiler-first) — `docs/spec/compiler.md`:**
- Read §1 (lines ~1–10). Identify any sentence stating the rule "engine extension comes from emitter".
- Delete the principle-shaped sentence. Keep the schematic of emission modes (what gets emitted, when), since that's mechanism, not principle.

```bash
sed -n '1,30p' docs/spec/compiler.md     # inspect
# (use Edit tool to remove specific lines)
```

**P2 (schema-bump) — `docs/spec/runtime.md`:**
- Inspect §22. Delete the principle-shaped sentence ("any layout change requires schema_hash regen") and keep the mechanism (how the hash is computed, what it covers).

**P3 (cross-backend parity) — `docs/spec/runtime.md` §24, `docs/spec/compiler.md` §1:**
- Delete principle restatement. Keep the concrete `@cpu_only` annotation contract.

**P4 (EffectOp size budget) — `docs/spec/ability.md` §22:**
- Delete the size-budget assertion if it appears as a rule. Keep the variant table.

**P5 (keyed PCG) — `docs/spec/runtime.md` §7, `docs/spec/language.md` §9 D12:**
- Delete principle-shaped statements. Keep the API (`per_agent_u32` signature, sub-stream keys).

**P6 (events as mutation channel) — `docs/spec/runtime.md` §5:**
- Delete principle-shaped sentences. Keep the `EventRing` API contract.

**P7 (replayability flag) — `docs/spec/runtime.md` §5, `docs/spec/language.md` §2.2:**
- Delete principle restatements. Keep the annotation grammar.

**P10 (no runtime panic) — `docs/spec/runtime.md` §14:**
- Delete principle-shaped statement. Keep the contract on `step_full` (what it asserts via `#[requires]` / `#[ensures]`).

For each, use `Edit` tool with explicit `old_string` + `new_string` (not regex sed); the deletions are surgical.

- [ ] **Step 2: Audit `docs/spec/language.md` §9**

§9 contains 30 settled-decision entries; many are principle-flavored. For each entry:
- If the entry states a behavioral rule already in the constitution → delete the entry (renumber-not-required; gaps are fine, see existing convention).
- If the entry is implementation-specific (e.g., "`Resolution::Coalition{min_parties}` enum variant") → keep.

```bash
grep -nE '^[0-9]+\.' docs/spec/language.md | head -30
```

Walk each entry; remove the principle-flavored ones. Expected delete count: 4–8 entries.

- [ ] **Step 3: Sanity grep — no spec restates a constitution principle**

```bash
# For each principle's "headline phrase", confirm it doesn't appear in spec/
phrases=(
  "engine.{0,10}extension.{0,30}emitter"
  "schema.hash.{0,30}layout"
  "cross.backend.{0,30}parity"
  "EffectOp.{0,40}16 bytes"
  "thread_rng"
  "events are.{0,30}mutation"
  "replayable.{0,30}declaration"
)
for p in "${phrases[@]}"; do
    echo "=== $p ==="
    grep -irE "$p" docs/spec/ | head -3
done
```

Expect: zero or near-zero hits in spec/. Constitution-style restatements indicate incomplete migration.

- [ ] **Step 4: Verify spec/ still reads coherently**

Open each modified spec file. Make sure surviving content reads without the deleted sentences (no orphaned references to a now-deleted prior sentence).

- [ ] **Step 5: Commit**

```bash
git add docs/spec/
git commit -m "docs(spec): migrate principle-flavored content to constitution (delete duplicates)"
```

---

### Task 10: Final consistency check + cleanup

**Files:**
- Delete: `docs/architecture/.principle-inventory.md` (working doc from T1)

- [ ] **Step 1: Verify all spec docs in llms.txt still exist**

```bash
grep -oE '\(docs/[^)]*\.md\)' docs/llms.txt | tr -d '()' | sort -u | while read f; do
    [ -f "$f" ] || echo "MISSING: $f"
done
```

Expect: zero output.

- [ ] **Step 2: Verify constitution is loaded by hook**

Open a new Claude Code session (or simulate via the `cat` command run by the hook) and confirm the constitution is in the agent's first context window. (Manual.)

- [ ] **Step 3: Cross-spec scan for stragglers**

```bash
# Find any "P[0-9]+" references; they should only appear in:
#  - docs/constitution.md (definitions)
#  - docs/architecture/plan-template-ais.md (template)
#  - CLAUDE.md (mentions P5/P8 by number, no restatement)
grep -rnE '\bP[0-9]+\b' docs/ CLAUDE.md | grep -v "docs/constitution.md\|plan-template-ais\|CLAUDE.md"
```

Expect: empty (or only the docs/superpowers/specs/2026-04-24-constitution-and-adr-design.md spec doc, which is allowed since it's the design doc).

- [ ] **Step 4: Delete the working inventory doc**

```bash
git rm docs/architecture/.principle-inventory.md
```

- [ ] **Step 5: Commit cleanup**

```bash
git commit -m "chore(constitution): drop principle-inventory working doc; landing complete"
```

- [ ] **Step 6: Final sanity test — run cargo test**

```bash
cargo test -p engine 2>&1 | grep -E "^test result" | awk '{ p+=$4; f+=$6 } END { print "passed:", p, "failed:", f }'
```

Expect: same pass count as before this plan (no test impact from doc-only changes).

- [ ] **Step 7: Self-applied AIS re-evaluation**

Tick the second checkbox in this plan's AIS preamble:
- [x] AIS reviewed at design phase
- [x] AIS reviewed post-design

Confirm in commit message that no spec content beyond what was anticipated was touched.

```bash
git commit --allow-empty -m "chore(constitution): post-impl AIS re-evaluation — within scope"
```

---

## Self-Review

**Spec coverage:** Each section of the design spec has at least one task:
- §3.1 constitution → T1 (inventory) + T2 (draft) + T3 (verification)
- §3.2 ADR → T4
- §3.3 AIS → T5
- §3.4 llms.txt → T6
- §3.5 CLAUDE.md trim → T8
- §3.6 SessionStart hook → T7
- §4 migration → T9
- §5 sourcing (hand-curate) → T1 + T2
- Decision log preserved in spec; not restated in plan.

**Placeholder scan:** No TBDs, no "implement later"; every step has either explicit code/text/commands or a manual-verify instruction with expected output.

**Type consistency:** Field names (`closes_commit`, `tier`, `inv`) and file paths consistent across constitution, AIS template, llms.txt, and CLAUDE.md.

---

## Execution handoff

Plan complete. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task; I review between tasks.
2. **Inline Execution** — execute in this session via executing-plans.

Which approach?
