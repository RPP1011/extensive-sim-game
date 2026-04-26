# Constitution + ADR Convention — Design Spec

> **Status:** Design (2026-04-24). Foundational layer for the
> architectural-enforcement framework. Other specs (engine crate
> split, project-DAG skill, project-critic skill) reference principles
> defined here by P-number.

## §1 Goals

1. **Single source of truth** for architectural principles. The constitution states every load-bearing rule once. Other docs do not paraphrase, summarize, or restate principles — they may cite P-numbers when describing consequences.
2. **Stable handle** via P-number. Principle prose can be edited without invalidating cross-references.
3. **ADR-NNNN** record convention for locked decisions that come out of design specs.
4. **Architectural Impact Statement (AIS)** required preamble in every new plan, forcing an explicit Constitution check before code is written.
5. **`llms.txt`** index so agents look up docs by name, not by hunting through the tree.
6. **Trim root `CLAUDE.md`** to ≤100 lines. Long-form content moves to `docs/llms.txt` and the spec tree.

## §2 Non-goals

- Defining the principles' content. The principle catalog is the *implementation* of this spec; the spec only fixes the schema and conventions.
- Retroactively numbering existing `docs/superpowers/specs/*.md` files as ADRs. Forward-only.
- Replacing the spec/ tree. ADRs record decisions; specs explore designs. Both stay.
- Enforcement mechanisms beyond hooks/CI. Compile-time enforcement is Spec B (engine crate split).
- Multi-author governance ("approval" workflows). Single-developer use.

## §3 Components

### §3.1 `docs/constitution.md`

Top-level location, ~120 lines. Format:

```markdown
# Project Constitution

> Architectural principles that govern this project. Each principle
> is a numbered, stable handle (P-N). Other docs cite by number; do
> not paraphrase. Edit the prose freely; the number is the contract.

## P1 — <Short name>

**Statement.** Single sentence stating the rule.

**Rationale.** 1–3 sentences: why this rule exists, what failure mode it
prevents.

**Enforcement.**
- Compile-time: <e.g., sealed trait, build.rs sentinel> (or "none")
- CI: <e.g., ast-grep rule path, fitness function> (or "none")
- Agent-write-time: <e.g., PreToolUse hook script> (or "none")
- Critic: <e.g., project-critic invocation> (or "none")
- Manual: <human review trigger> (or "none")

**Source.** `<spec path>` (the original design doc that motivated the
principle, kept for archaeology only — readers do not follow pointers).

---

## P2 — ...
```

Initial population: ~10 principles, hand-curated. Likely candidates (final list lands during implementation):
- Compiler-first engine extension
- Schema-bump on layout change
- Cross-backend parity (Serial + GPU; or explicit `@cpu_only`)
- `EffectOp` size budget (≤16 bytes per variant)
- Determinism via `per_agent_u32(seed, agent, tick, purpose)`
- No `thread_rng()` / no panics on the deterministic path
- All sim-state mutations are events; current state is a fold
- Replayable events flagged at declaration; chronicle/non-replayable side channels are explicit
- Constitution check (AIS) required before any new plan executes
- Tasks close only with a verified commit SHA

### §3.2 ADR convention

`docs/adr/NNNN-<slug>.md`. Schema:

```markdown
# ADR-0001: <Decision Title>

**Status:** Accepted | Superseded by ADR-NNNN | Rejected
**Date:** YYYY-MM-DD
**Spec:** `docs/superpowers/specs/<spec>.md` (the design doc that
preceded this; ADR records the locked outcome)

## Context

What problem prompted this decision (1 paragraph).

## Decision

The chosen approach (1 paragraph).

## Consequences

Downstream impact (positive and negative bullets).

## Alternatives considered

What was rejected and why (1 paragraph).
```

ADRs are immutable once `Accepted`. Changing direction = new ADR that
supersedes the old one (mark old as `Superseded by ADR-NNNN`).

Numbering: 4 digits, monotonic, never reused.

ADRs are used for decisions that lock an external interface or
architectural commitment. Internal implementation choices stay in
specs/plans.

### §3.3 Architectural Impact Statement (AIS)

Required preamble for every plan in `docs/superpowers/plans/`:

```markdown
## Architectural Impact Statement

- **Existing primitives searched:** <symbol or term> at <file:line>,
  <symbol> at <file:line>. Search method: ast-grep / rg / read.
- **Decision:** extend `<existing primitive>` / new primitive because
  <reason>.
- **Rule-compiler touchpoints:** DSL inputs edited: <files>. Generated
  outputs re-emitted: <files>.
- **Hand-written downstream code:** NONE | <justification per file>.
- **Constitution check:**
  - P1: PASS (evidence: <file:line> shows generated entry-point)
  - P2: N/A (no layout change)
  - ...
- **Re-evaluation:** This AIS was reviewed at <design phase> and again
  at <post-implementation>. (Second review occurs after design lands;
  AIS is updated if scope shifted.)
```

This section comes BEFORE any task list in the plan. Plans without an AIS are rejected by `superpowers:writing-plans` skill output check (or by a hook on plan-file writes if we add one).

The AIS is project-specific; the upstream `superpowers:writing-plans` skill emits the standard plan structure, and a project-local addendum (loaded by Claude when working in this repo) requires the AIS section be present.

### §3.4 `docs/llms.txt`

Top-level index, agent-consumed. Format ([llmstxt.org](https://llmstxt.org/)
convention):

```
# Project Name
> 1-line project description.

## Docs
- [Constitution](docs/constitution.md): numbered architectural principles.
- [Roadmap](docs/ROADMAP.md): comprehensive future-work index.
- [Engine status](docs/engine/status.md): live per-subsystem implementation status.
- [Overview](docs/overview.md): 5-minute architectural intro.

## Spec
- [DSL](docs/spec/dsl.md): world-sim DSL grammar + semantics, stdlib, compiler architecture, and scoring field-id mapping.
- [Engine](docs/spec/engine.md): engine runtime contract (§§1–16, GPU annexes §§9–12).
- [State](docs/spec/state.md): field catalog.
- [Ability DSL](docs/spec/ability.md): .ability language + IR.
- [Economy](docs/spec/economy.md): economic system design.

## Process
- [Workstreams](docs/architecture/workstreams/): per-workstream definitions.
- [Plans](docs/superpowers/plans/): in-flight implementation plans.
- [ADRs](docs/adr/): locked decisions.
```

Maintained by hand; small enough to keep current.

### §3.5 Root `CLAUDE.md` trim

Target ≤100 lines. Keeps:
- Project name + 1-line description (5 lines)
- Primary build / test commands (15 lines)
- One sentence: "The constitution at `docs/constitution.md` defines the
  architectural principles; it is auto-loaded into agent context on
  session start (see §3.6)." (1 line)
- llms.txt location + reading-paths block (15 lines)
- Conventions (e.g., "use spec/ for canonical contract; don't duplicate";
  "every plan needs an AIS preamble — see §3.3") (20 lines)

Removes (moved to llms.txt + spec/):
- Detailed module map (engine src layout, ai/* layout)
- AI decision pipeline narrative
- ML model details
- Hero template details
- Test helper details
- Per-principle summary lines (would duplicate constitution; D1 forbids).

CLAUDE.md does NOT enumerate principles. Agents see the constitution via the SessionStart hook (§3.6), not via a CLAUDE.md preamble.

### §3.6 SessionStart hook to auto-load constitution

Add a `SessionStart` entry to `.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      { "type": "command",
        "command": "cat docs/constitution.md" }
    ]
  }
}
```

Output of a `SessionStart` hook is injected into the agent's context as additional system content. This guarantees the constitution is in every agent's working context without a CLAUDE.md duplicate and without relying on the agent to follow a pointer.

If this becomes too token-heavy, replace with a leaner emission (e.g., principle headlines only, with full text fetched on demand via Read). The full content is the default; trim only if measured cost is a problem.

## §4 Migration: existing duplicate content

Single-source-of-truth has a one-time cleanup cost. Once the constitution lands, sweep these locations for duplicate principle-flavored content and **delete** (not redirect):

- `docs/spec/engine.md` §8, §6, §15 — schema-hash rules, determinism contract, backend selection. Move principle content to constitution; delete from spec or keep only as implementation detail (e.g., concrete API for a principle).
- `docs/spec/dsl.md` §9 (compiler architecture) — collapse to spec-of-emission-mechanics; the principle moves to constitution.
- `docs/spec/dsl.md` §13 (settled decisions) — many entries are de facto principles. Audit each; principles → constitution; implementation specifics → stay in spec.
- `docs/spec/ability.md` §22.2 (`EffectOp` size budget) — principle to constitution; the variant table stays.
- `docs/engine/status.md` "Top weak-test risks" + "Open verification questions" — these are fine; they're audit findings, not principles.

Migration is included in the implementation plan, not deferred.

## §5 Sourcing

Hand-curated extraction from existing specs (no Haiku scan for the principle catalog). Process:

1. Read `docs/spec/engine.md`, `docs/spec/dsl.md` §13, `ability.md`, `economy.md` §1.3 cover-to-cover.
2. Identify principle-shaped statements (rules that govern behavior, not implementation specifics).
3. Draft each as P-N with statement + rationale + enforcement + source.
4. Cross-check no duplicates across principles.
5. After draft, run a Haiku verification pass: "Given the constitution and these specs, is anything principle-shaped missing?" — gap-fill from its findings.

## §6 File layout

```
docs/
  constitution.md                         (new)
  llms.txt                                (new)
  adr/
    0001-<first-decision>.md              (created on first decision lock)
    ...
  superpowers/specs/                      (existing)
    2026-04-24-constitution-and-adr-design.md  (this file)
    ...
  superpowers/plans/                      (existing)
    <future-plans-include-AIS-preamble>.md
    ...
```

`CLAUDE.md` at repo root, trimmed.

## §7 Decision log

Decisions locked during brainstorming (2026-04-24):

- **D1.** Single source of truth. Constitution is the only place principles live; other docs do not restate or paraphrase. Pointers will not be followed in practice, so we don't write them.
- **D2.** P-N number is the stable handle; prose is mutable.
- **D3.** ADRs are forward-only (existing specs are not retroactively renumbered).
- **D4.** AIS is required preamble; project-local extension to `superpowers:writing-plans` skill output.
- **D5.** Initial principle catalog is hand-curated, ~10 principles. Haiku verification pass after draft.
- **D6.** Migration (deletion of duplicate principle-flavored content from existing specs) is part of the implementation plan, not deferred.
- **D7.** Constitution lands in `docs/` root, not under `spec/`. Reason: `spec/` is contract-of-the-system; constitution is governance-of-the-spec.
- **D8.** Constitution is auto-loaded into every agent context via `SessionStart` hook. CLAUDE.md does not enumerate principles (would duplicate; D1 forbids).

## §8 Out of scope (explicit)

- Multi-author approval workflows on principle changes. Single developer.
- Versioning the constitution itself (e.g., "v2 of P3"). Edit prose in place; if a principle materially changes, write an ADR that supersedes prior usage.
- Tooling that auto-extracts principles from spec text. Hand-curated.
- Layer 1 (engine crate split / sealed trait / build.rs sentinel) — separate spec (Spec B).
- Layer 2 (DAG skill, hooks) — separate spec (Spec C, drafted).
- Layer 4 (project-critic skill) — separate spec (Spec D, future).
