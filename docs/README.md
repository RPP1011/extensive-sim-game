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
    engine.md    — engine runtime + GPU backend contract (absorbed the old
                   runtime.md and gpu.md, 2026-04-26 consolidation)
    dsl.md       — world-sim DSL grammar, semantics, stdlib, compiler
                   architecture, scoring field-id mapping (absorbed the old
                   language.md, stdlib.md, compiler.md, scoring_fields.md)
    state.md     — field catalog (every SoA field, who reads, who writes)
    ability.md   — `.ability` DSL — ability definitions + IR
    ability_dsl_unified.md — unified ability-DSL reference
    ability_dsl_test_sims.md — ability-DSL test-fixture catalog
    belief-primitive.md — Theory-of-Mind belief primitive spec
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
