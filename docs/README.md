# Docs

Start at **[`overview.md`](overview.md)** for a 5-minute project intro.
Then dive into the spec, status, or roadmap depending on what you need.

## Layout

```
docs/
  README.md      — this file
  overview.md    — 5-minute project intro (architecture, tick, worked example)
  ROADMAP.md     — comprehensive index of future work (active / drafted / deferred)

  spec/          — canonical specification (the contract)
    README.md    — spec index, reading order, cross-ref convention
    language.md  — world-sim DSL grammar + semantics
    state.md     — field catalog (every SoA field, who reads, who writes)
    stdlib.md    — pinned built-ins
    scoring_fields.md — `field_id` ABI table
    runtime.md   — engine runtime contract §§1–26
    compiler.md  — DSL → Rust + SPIR-V + Python lowering
    gpu.md       — GPU backend contract (resident cascade, sim-state, cold-state, ability eval, pipeline)
    ability.md   — `.ability` DSL — ability definitions + IR
    economy.md   — economic system (recipes, contracts, labor, market)

  engine/
    status.md    — live per-subsystem implementation status (✅/⚠️/❌)

  game/          — player-facing layer (overview, feature flow, fixtures)
  superpowers/   — process artefacts: plans + brainstorms + research + notes
                   (skill output target — leave directory shape alone)
```

## Reading paths

**New contributor (15 min):**
`overview.md` → `engine/status.md` → `spec/README.md` → `ROADMAP.md`

**Engineer adding a feature:**
`game/feature_flow.md` → `spec/<relevant>.md` → `engine/status.md` for current scaffolding

**Reviewer / planner:**
`ROADMAP.md` → `engine/status.md` → relevant `superpowers/plans/<active>.md`

## Conventions

- The **spec** locks contract. Live status lives in `engine/status.md`.
- The **roadmap** lists future work; in-flight plans live in `superpowers/plans/`.
- Historical content (executed plans, resolved audits, design rationale) lives in **git history**, not in active docs.
