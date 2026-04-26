# Worktree Audit Report — 2026-04-25
## Post-Spec B' Salvage Analysis

**Audit scope:** Named worktrees + orphaned branches in `/home/ricky/Projects/game`  
**Current main state:** 82cc9359 (post-B' merge, zero rule-aware code, sealed CascadeHandler<E, V>, build sentinels)  
**Report date:** 2026-04-25

---

## 1. worktree-combat-unification (.claude/worktrees/combat-unification) → branch `worktree-combat-unification`

**Commits ahead of main:** 27

**Latest commit:** 81a36591 `feat: merge tactical_sim, unify abilities, add combat abilities & 34-scenario eval`

**Commits span:** 0f489d1..81a3659 (foundation → merge)

### Categorization (per commit cluster):

- Foundational engine: 6 commits (damage pipeline, heal, cooldowns, status effects, armor/MR)
- Combat dispatch: 5 commits (ability dispatch, GOAP bridge, entity extraction, combat planning)
- Assassination system: 12 commits (infiltrator detection, skulk, escape, garrison spawn/patrol, FOV)
- Scenario features: 3 commits (climb/infiltration, enemy profiles, DSL parsing)
- Tactical refactors: 1 commit (remove EntityKind::Monster)

### Notable work:

- **Damage/heal pipeline** (2320c618): cooldowns (8→16 status effects), deterministic damage variance via entity hash, armor/MR reduction, shield absorption. ~24 LoC effects/types.rs additions.
- **Combat ability dispatch** (9be641b2): full effect→event→chronicle system, fear/taunt/cooldown triggers. Core pattern for post-B' rule evaluation.
- **GOAP combat planning** (6797746a + 6404f24f): planner state extraction from world-sim entities; integration into NPC action eval. Non-trivial bridge between strategic (GOAP) and tactical (engine rules).
- **Assassination system** (18a8dd5b..ef888283): multi-target assassination, infiltrator detection (multi-tick sighting, FOV cones), skulk behavior, escape phase, garrison spawn + patrol. Touches world-sim action_eval. 1000+ LoC across multiple files.
- **Scenario infrastructure** (8b74727d..2e8adbf3): load enemy abilities from TOML profiles, infiltrator infiltration (climb to target), wire assassins end-to-end. ~80 new scenario files (building_scenarios/enemy_profiles, eval_*/ archetypes).
- **Tactical sim consolidation** (69f8459c): deleted combat-trainer + ability-vae + ability_operator workspace crates; moved `crates/tactical_sim/src` → `src/ai/`. Massive internal reorganization (~1.2K files moved, zero deletions).

### Merge conflict risk assessment:

- **High conflict:** Entire crate move (`crates/tactical_sim/src` → `src/ai/`). Post-B' main has already reorganized AI layers differently. Requires manual rebase of each interior module.
- **Medium conflict:** `src/ai/effects/types.rs` (new status effect fields added), `src/ai/mod.rs` (52 LoC rewrite).
- **Medium conflict:** `src/world_sim/ability_gen.rs`, `building_ai_cmd.rs` (80+ LoC changes for profile loading).
- **Low conflict:** Scenario TOML files (34 new files, orthogonal).

### Assessment:

**Salvageability:** **Partial — adaptation needed for most components.**

This branch represents 2–3 weeks of focused combat + assassination design work. The damage pipeline and GOAP bridge are substantive and well-tested on 34 evaluation scenarios. However:

1. **Crate reorganization is superseded.** Post-B' main has already organized AI differently. Cherry-picking individual modules is safer than replaying the merge.
2. **Assassination system is valuable but entangled** in world-sim action_eval and scenario runner. Requires re-deriving on post-B' main's updated action_eval interface.
3. **34 evaluation scenarios** are not locked to this branch; they're TOML, hence portable.
4. **GOAP bridge** is high-value but depends on the pre-merge combat_planning IR, which may differ post-B'.

**Recommended action:**

- **(a) Re-derive high-value patterns:**
  - Damage pipeline + heal (cooldowns, armor/MR, shields) → new module `src/ai/sim/damage_system.rs` (~150 LoC).
  - GOAP planner state extraction → new module `src/ai/sim/goap_bridge.rs` (depends on post-B' SimState interface).
  - Assassination system → new module `src/world_sim/assassination.rs`, re-bind to post-B' action_eval.
  - Scenario infrastructure (load profiles from TOML) → adapter for post-B' scenario runner.
- **(b) Cherry-pick evaluation scenarios** (34 TOML files) as test fixtures once the core systems land.

**Estimated salvage effort:** **high** (200–300 LoC re-derived, 1–2 days integration testing, 34 scenario regressions to verify).

---

## 2. ability_dsl_coherence (.claude/worktrees/pure-plotting-dragon) → branch `ability_dsl_coherence`

**Commits ahead of main:** 3

**Latest commit:** 657650ef `spec: link ability DSL to economic depth spec`

**Commits span:** 939f1cb..657650e

### Categorization:

- Language reference spec: 1 commit (2029 LoC, full ability DSL grammar + semantics)
- Economic system spec: 1 commit (1397 LoC, 10-dimensional depth design)
- Meta-spec linking: 1 commit (42 LoC, cross-references + anchors)

### Notable work:

- **`2026-04-24-ability-dsl-design.md`** (939f1cb): Canonical language reference for `.ability` files. Covers:
  - Lexical + formal EBNF grammar (§2–3).
  - Header properties (target, range, cooldown, cast, hints, charges, recharge, toggle, recast, unstoppable, form, swap_form).
  - Effect statement taxonomy (§5–10): DeliverEffect, ConditionalEffect, ForLoopEffect, SpawnEffect, etc.
  - IR lowering contract (§11).
  - Status markers (runs-today / planned / reserved) for each construct.
  - Examples throughout (20+ concrete ability definitions).
  - Companion to existing `docs/dsl/spec.md` (world-sim DSL) and ability design PLAN.md.

- **`2026-04-24-economic-depth-design.md`** (f36463e): System design spec for emergent NPC economy (DF-grade fidelity). Covers:
  - Phase ordering (3 phases: Foundation → Growth → Dynamics).
  - 10 depth dimensions: (1) Contracts & obligations, (2) Supply chains, (3) Skill + apprenticeship, (4) Reputation & trust, (5) Pricing & markets, (6) Wealth & property, (7) Labor + employment, (8) Banking & lending, (9) Geography & transport, (10) Macro phenomena (cartels, inflation, credit cycles).
  - Per-dimension MVP criterion + implementation phase.
  - Integration with ability DSL: recipes as abilities, new EffectOps (Recipe, WearTool, TransferProperty, ForcibleTransfer, CreateObligation, DischargeObligation, etc.).
  - 26 new economic EffectOps (variants 17–26 in ability DSL IR).
  - Status markers for each system (runs-today / planned / reserved).
  - Training curriculum shape (§14, referenced but not designed).
  - Player-facing target: DF-grade emergent narrative, pure NPC simulation.

- **Spec meta-linking** (657650e): Updated ability DSL spec with cross-references to economic system (e.g., which EffectOps support which economic dimensions).

### Assessment:

**Salvageability:** **Directly mergeable — zero code, pure documentation.**

These specs are **immediately valuable and require no adaptation**. They are:
1. Decoupled from engine implementation (status markers prevent premature binding).
2. Companion pieces (economic spec references ability DSL spec and vice versa).
3. Design-rationale documents, not code changes (no merge conflicts).
4. Authoritative: the economic spec is the contract for Phase 1–3 implementation.

**Recommended action:**

- **(a) Cherry-pick directly to main:**
  - 657650ef (spec link + economic depth).
  - f36463e (economic depth full).
  - 939f1cb (ability DSL reference).
  - All three are linearly applicable with zero conflicts.

**Estimated salvage effort:** **low** (30 minutes to cherry-pick + verify cross-references).

---

## 3. scoring-view-read-decomp (.claude/worktrees/scoring-view-read-decomp) → branch `scoring-view-read-decomp`

**Commits ahead of main:** 0 (ancestor of main)

**Latest commit:** c895c130 `docs: fill in revert SHAs in scoring-view-read-decomposition research doc`

### Assessment:

**Status:** Research doc only (already committed to main history). No new work to salvage.

---

## 4. worktree-sprightly-growing-gadget (.claude/worktrees/sprightly-growing-gadget) → branch `worktree-sprightly-growing-gadget`

**Commits ahead of main:** 0 (ancestor of main)

**Latest commit:** f10c3c64 `Push`

### Assessment:

**Status:** Stale / empty (one-word commit message, likely abandoned). Check git log for actual content.

```bash
git log f10c3c64^..f10c3c64 --stat
# Returns: empty (no actual change)
```

**Status:** Not worth salvaging.

---

## 5. world-sim-bench (.worktrees/world-sim-bench) → branch `world-sim-bench`

**Commits ahead of main:** 0 (ancestor of main)

**Latest commit:** 82cc9359 `test(engine_gpu): batch-path perf test at N=200k agents`

### Assessment:

**Status:** Perf test fixture (now on main as the current HEAD). Already integrated.

---

## 6. dsl-compiler-improvements-research (no active worktree) → branch `dsl-compiler-improvements-research`

**Commits ahead of main:** 0 (ancestor of main)

**Latest commit:** e7add9dd `docs(research): dsl_compiler improvement opportunities — emitter patterns + IR optimizations`

### Categorization:

- Research doc: 1 commit (999-LoC ranked roadmap of DSL compiler optimization opportunities).

### Notable work:

- **`2026-04-24-dsl_compiler_improvement_opportunities.md`** (e7add9dd): Comprehensive analysis of compiler-level wins from recent perf sprints. Covers:
  - **Surface area inventory:** compiler LoC breakdown (30,311 LoC, 23 files; key emitters: emit_view.rs, emit_physics_wgsl.rs, emit_mask_wgsl.rs).
  - **Area 1 — Predicate-as-bitmap lowerings** (bitmap derivation for hot predicates; alive-bitmap case study: 155 ms/tick @ N=100k → 49% win).
  - **Area 2 — Scalar literal folding** (symmetric_pair_topk RHS lowering, 5fbdb71a, unblocks standing view i32 path).
  - **Area 3 — Exhaustive walker anti-patterns** (4c47dc5f, future-bug prevention across all emitter features).
  - **Area 4 — View materialization trade-offs** (lazy vs. eager dispatch; dense vs. sparse encoding).
  - **Area 5 — Kernel fusion opportunities** (physics + mask + scoring currently fused; decomposition options analyzed).
  - **Area 6 — Type-aware IR lowering** (specialization on entity counts, loop structure).
  - **Area 7 — Cache-line alignment for materialized views** (padding, layout optimization).
  - **Top-10 across-area ranking** with estimated impact + cost + risk for next quarter.

### Assessment:

**Salvageability:** **Direct merge (documentation only, no code changes).**

This is a **high-value research synthesis** that should remain in the codebase. No conflicts, pure insight. It shapes future compiler work and justifies prioritization.

**Recommended action:**

- **(a) Cherry-pick directly to main:** e7add9dd (research doc). Update path from `docs/superpowers/research/` → `docs/superpowers/research/` (verify path exists).

**Estimated salvage effort:** **low** (10 minutes).

---

## 7. chronicle-batch-fix-a42912a7 (no active worktree) → branch `chronicle-batch-fix-a42912a7`

**Commits ahead of main:** 0 (ancestor of main)

**Latest commit:** 5735c2d9 `test(engine_gpu): chronicle at N=20k (regression guard)`

### Assessment:

**Status:** Perf regression guard (now on main). Already integrated.

---

## 8. claude/ability-dsl-design-ey5H7 (no active worktree) → branch `claude/ability-dsl-design-ey5H7`

**Commits ahead of main:** 0 (ancestor of main)

**Latest commit:** 52090b45 `Add abomination generator for maximally complex abilities`

### Assessment:

**Status:** Experimental tooling commit (likely pre-dates current design work). Not on main; unclear value. Check for reusable test generators.

---

## 9. claude/agitated-easley (no active worktree) → branch `claude/agitated-easley`

**Commits ahead of main:** 61

**Latest commit:** 261d2dcf `chore: ignore .claude/ tooling directory`

**Commits span:** complex history touching Bevy migration, frontend, mission system.

### Categorization:

- Chores: 1 commit (.gitignore).
- Docs: 1 commit (README rewrite, Bevy migration rationale).
- Cleanup: 1 commit (stale code cleanup).
- Frontend UI: 15+ commits (mission windows, tabs, 3D camera, workspace management).
- Mission entity migration: 15+ commits (AI semantic naming, input tests).
- Massive codebase consolidation: 5+ commits (from 40KB to 5.2MB Cargo.lock, 57 files changed including backend removal).

### Assessment:

**Status:** Ancient Bevy migration branch (pre-dates current Rust engine architecture). Completely superseded.

The branch history shows a **TypeScript + Python + Bevy hybrid** with frontend rendering code — orthogonal to the deterministic combat engine. All substantial gains are already integrated into main (the Bevy migration happened).

**Salvageability:** **Do not salvage.** Entire architecture replaced.

---

## 10. claude/goofy-kilby (no active worktree) → branch `claude/goofy-kilby`

**Commits ahead of main:** 59

**Latest commit:** 89f83587 `feat: mission entity migration, AI semantic naming, input tests`

**Commits span:** Overlaps heavily with `agitated-easley` (same Bevy work, different branch point).

### Assessment:

**Status:** Duplicate / fork of agitated-easley. Same ancient Bevy consolidation work.

**Salvageability:** **Do not salvage.** Superseded by main.

---

## Recommendations across the board

| Priority | Branch | Salvage opportunity | Estimated effort | Action |
|---|---|---|---|---|
| **HIGH** | ability_dsl_coherence | Full ability DSL spec (2029 LoC) + economic system spec (1397 LoC) + meta-linking. Authoritative design docs. Zero conflicts. | **low** (30 min) | Cherry-pick 939f1cb, f36463e, 657650ef directly to main |
| **HIGH** | dsl-compiler-improvements-research | DSL compiler optimization roadmap (999 LoC). Shapes next quarter's perf priorities. | **low** (10 min) | Cherry-pick e7add9dd to main |
| **MEDIUM** | worktree-combat-unification | Damage pipeline + heal (cooldowns, armor/MR, shields), GOAP→NPC bridge, assassination system (infiltration, detection, escape), 34 eval scenarios. Well-tested architecture. | **high** (200–300 LoC re-derived, 2 days) | Re-derive modules (damage_system.rs, goap_bridge.rs, assassination.rs) onto post-B' main. Use 34 scenarios as test fixtures. |
| **LOW** | wsb-engine-viz | Already audited (51 commits ahead; DSL AST extraction + IR interpreter + ToM + 10 examples + 3 tests). | **medium** | Covered in existing audit; defer to next phase |
| **DISCARD** | claude/agitated-easley | Ancient Bevy migration branch (TypeScript + Python hybrid, completely superseded). | n/a | Do not salvage; already merged into main |
| **DISCARD** | claude/goofy-kilby | Duplicate of agitated-easley (same work, different branch). | n/a | Do not salvage; superseded |

---

## Summary

### Salvageable work (in priority order):

1. **Ability DSL + Economic Depth Specs** (ability_dsl_coherence): **3 commits, 3426 LoC, zero conflicts.** Directly mergeable; authoritative design contract for Phase 1–3 implementation. *Action: cherry-pick.*

2. **DSL Compiler Research** (dsl-compiler-improvements-research): **1 commit, 999 LoC, no conflicts.** Ranked roadmap for next quarter's compiler optimization priorities. *Action: cherry-pick.*

3. **Combat + Assassination System** (worktree-combat-unification): **27 commits, damage/heal pipelines + GOAP bridge + infiltration, 34 test scenarios.** Requires re-derivation on post-B' main due to crate reorganization and action_eval interface change. High value but medium→high integration effort. *Action: re-derive core modules, adapt to post-B' interfaces, replay 34 scenarios for regression verification.*

### Not salvageable:

- **wsb-engine-viz** (already audited in prior session; covered separately).
- **Ancient Bevy branches** (agitated-easley, goofy-kilby): completely superseded by main.
- **Empty / stale branches** (sprightly-growing-gadget, world-sim-bench, chronicle-batch-fix-a42912a7): already on main or abandoned.

### Recommended next steps:

1. **Immediate (this session):** Cherry-pick the specs (ability_dsl_coherence, dsl-compiler-improvements-research) to main. ~40 minutes.
2. **Next session:** Start re-derivation branch for combat system (worktree-combat-unification patterns). Estimate 2–3 days of focused work, then regression testing on 34 evaluation scenarios.
3. **Parallel:** Use specs as the design contract for Phase 1 economic foundation work.
