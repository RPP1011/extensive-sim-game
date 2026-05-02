# Phase 8: CG Body-Emit Closure → parity_with_cpu Green — Plan Stub

> **Status: STUB.** Phase 6 (Tasks 1-5) shipped the structural IR + schema-layer + Movement-as-rule plumbing but parity stayed RED. The remaining work is the body-emit infrastructure: real WGSL bodies for masks, scoring, and movement that read/write the SoA correctly. Phase 5 Task 5's investigation (commit `5c334381`, no follow-up commit) characterized the seven blockers below; this stub captures them as a cohesive plan.

**Goal:** `parity_with_cpu_n4_t{1,10,100}` GREEN under `--features gpu` on the smoke fixture (or its target-game equivalent). At the end of Phase 8, the CG-emitted SCHEDULE produces real per-agent decisions and applies them, matching CPU output byte-for-byte.

**Architecture:** Three structural cross-cutting fixes + three body-content fixes + one cleanup. The structural fixes (body-emit naming, mask predicate lowering, scoring body extensions) are reusable across any DSL surface — they're load-bearing infrastructure the current emit tree doesn't yet have. The body-content fixes are wolf-sim-specific and may overlap or be replaced if Phase 7 (kin removal + target-game DSL) lands first.

**Tech Stack:** Rust workspace; `dsl_compiler` (lowering + emit), `engine_gpu_rules` (CG output). `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical` regenerates; `cargo test -p engine_gpu --features gpu --test parity_with_cpu` is the gate.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `lower_cg_expr_to_wgsl::CgExpr::Read(AgentField{...})` arm at `crates/dsl_compiler/src/cg/emit/wgsl_body.rs::structural_handle_name` (line ~152): emits `agent_<ref>_<field>` bare identifier per AgentRef variant.
  - `structural_binding_name` for `DataHandle::AgentField` at `crates/dsl_compiler/src/cg/emit/kernel.rs:934`: declares bindings as `agent_<field>` (drops the agent-ref discriminator — all agent-refs alias the same SoA buffer).
  - `lower_scoring_argmax_body` at `crates/dsl_compiler/src/cg/emit/kernel.rs:1769`: produces argmax body without mask gating, without target-bound row argmax.
  - `mask_predicate_per_agent_body` and `mask_predicate_per_pair_body` at `crates/dsl_compiler/src/cg/emit/kernel.rs:1731+`: emit hardcoded `let mask_<id>_value: bool = <wildcard placeholder>` for B1.
  - `well_formed::TypeCheckCtx::new` (no `prog.view_signatures` wired) — leaves every embedded `BuiltinId::ViewCall { view }` as `ViewSignatureUnresolved`.
  - `lower_expr` arm for `BeliefsAccessor` at `crates/dsl_compiler/src/cg/lower/expr.rs` — currently rejects.
  - Reference: legacy `crates/engine_gpu_rules/src/movement.wgsl` body (~40 LOC) — the proof-of-concept for what CG should emit.

  Search method: `rg`, `grep -n`, direct `Read`. Backed by Phase 6 Task 5 investigation (subagent transcript on 2026-05-01).

- **Decision:** Cross-cutting body-emit rewrite, NOT a per-feature extension. The naming drift between `structural_binding_name` (no agent-ref) and `structural_handle_name` (with agent-ref) is the load-bearing structural bug. Fixing it in isolation breaks every other emit site that uses the bare-identifier shape; deferring it forces every downstream fix to work around it. Land the rewrite first; downstream fixes inherit the corrected shape.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: NONE in the structural fixes (Tasks 1-3). Tasks 4-7 may extend `assets/sim/*` or the DSL grammar (`@phase(per_agent)` annotation).
  - Generated outputs re-emitted: every `crates/engine_gpu_rules/src/*.{wgsl,rs}` regen.

- **Hand-written downstream code:** the Route C splice (`crates/xtask/src/compile_dsl_cmd.rs:1671-1812`) retires fully at the end of Phase 8. Until then the splice keeps the parity test runnable while CG bodies mature.

- **Constitution check:**
  - P1 (Compiler-First): PASS — every change in the DSL compiler. Splice retirement is explicit goal.
  - P3 (Cross-Backend Parity): PASS — the entire goal is parity green.
  - P5 (Determinism via Keyed PCG): PASS — RNG paths preserved.
  - P10 (No Runtime Panic): PASS — every dispatch arm has a typed body, no unreachables on the deterministic path.
  - P11 (Reduction Determinism): VERIFY — scoring argmax + apply-actions sweeps must produce bit-equal reductions across CPU/GPU. The argmax tie-breaking rule is the load-bearing constraint here; document it explicitly when Task 5 lands.

- **Runtime gate:** `parity_with_cpu_n4_t{1,10,100}` under `--features gpu`. All three GREEN at end of Phase 8 = closure.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill, 2026-05-01). [ ] AIS reviewed post-design (after task list stabilises).

## Pre-existing context (do NOT re-investigate)

Phase 6 Task 5's investigation produced these definitive findings; treat them as ground truth:

1. CG's `lower_scoring` aborts on `BeliefsAccessor` at scoring.sim:13449. The `?` propagation in `lower_standard_row` skips the whole row, and the function's `for entry in &ir.entries { rows.push(...?) }` loop drops every subsequent row.
2. Even when BeliefsAccessor is patched, `well_formed::TypeCheckCtx::new` doesn't wire `prog.view_signatures`. The unresolved `BuiltinId::ViewCall { view }` produces `ViewSignatureUnresolved` for every scoring row that reads `view::*`.
3. Even with both patches, the CG-emitted body fails naga validation: bindings declared as `agent_hp` (no agent-ref), body emits `agent_self_hp` (with agent-ref). Cross-cutting naming drift.
4. Mask kernel bodies hardcode `false` for the predicate value (B1 short-circuit from `lower_scoring`'s wildcard fix wasn't extended to mask bodies). All four mask kernels emit no real bits.
5. `lower_scoring_argmax_body` walks every row unconditionally — no `if mask_<id>_bitmap[agent_id] { ... }` gate.
6. `lower_scoring_argmax_body` writes `best_target = NO_TARGET` for target-bound rows because `row.target` is always `None`.
7. Movement body is still a structural placeholder; even with real Scoring upstream, no position update fires.

## Tasks (skeleton)

| # | Task | Files | LOC est. | Depends |
|---|---|---|---|---|
| 1 | **Body-emit AgentField rewrite** — rewrite `lower_cg_expr_to_wgsl`'s `Read(AgentField{ref, field})` and `Assign(AgentField{ref, field}, ...)` arms. New form: `agent_<field>[<index_expr>]` indexed access where `<index_expr>` is `agent_id` for Self_, `target_local_id` for Target, etc. Update binding declarations to match. Update slot-aware naming strategy (`crates/dsl_compiler/src/cg/emit/wgsl_body.rs::HandleNamingStrategy`). Touch every kernel-body emit site that reads/writes agent fields (masks, folds, physics, future scoring). | `cg/emit/wgsl_body.rs`, `cg/emit/kernel.rs`, possibly `cg/emit/cross_cutting.rs` (binding_sources) | ~250 | — |
| 2 | **BeliefsAccessor + BeliefsConfidence → 0.0 lowering** — match arm in `lower_expr` returning `Lit(F32 0.0)`. Documented as semantically correct under default features (`#[cfg(feature = "theory-of-mind")]` not enabled). | `cg/lower/expr.rs` | ~10 | — |
| 3 | **`well_formed::TypeCheckCtx` view_signature wiring** — replace `TypeCheckCtx::new(arena)` with `TypeCheckCtx::with_view_signatures(arena, &prog.view_signatures)` so `BuiltinId::ViewCall { view }` resolves through the registered signature instead of `ViewSignatureUnresolved`. | `cg/well_formed.rs` | ~5 | — |
| 4 | **Mask predicate body lowering** — replace the B1 `false` hardcode with real predicate evaluation. The mask predicate is a `CgExprId` carried on `ComputeOpKind::MaskPredicate`; lower it to WGSL via `lower_cg_expr_to_wgsl` (which now produces correct indexed access post-Task 1). Test: each of `mask_Hold`, `mask_MoveToward`, `fused_mask_Flee`, `mask_Attack` emits a non-trivial predicate. | `cg/emit/kernel.rs::mask_predicate_*_body` | ~50 | 1 |
| 5 | **Scoring body mask gating + target argmax** — in `lower_scoring_argmax_body`, wrap each row in `if (mask_<row.mask_id>_bitmap[agent_id] & mask_bit) != 0u { ... }`. For target-bound rows (rows where the action head has a positional binder), iterate per-pair-candidates inside the row body, computing utility per candidate and picking the candidate with max utility as `best_target`. | `cg/emit/kernel.rs::lower_scoring_argmax_body` | ~120 | 1, 4 |
| 6 | **Movement body** — extend `MOVEMENT_BODY` const (or rewrite via `lower_cg_stmt_list_to_wgsl`) to read `scoring_output[agent_id]`, decode (action, target), compute direction delta, write `agent_pos[agent_id] += dir * move_speed`. Use the indexed AgentField shape from Task 1. | `cg/emit/kernel.rs` | ~40 | 1, 5 |
| 7 | **Route C splice retirement** — drop the legacy scoring + movement emit calls from `crates/xtask/src/compile_dsl_cmd.rs::compile_dsl_cmd::*` once Tasks 4-6 produce CG kernels that match CPU semantics. | `xtask/src/compile_dsl_cmd.rs` | -200 | 4, 5, 6 |
| 8 | **Parity gate green** — run `parity_with_cpu_n4_t{1,10,100}` under `--features gpu`. All three pass. If divergence remains, characterize per-tick + per-field, pick the next blocker. | `engine_gpu/tests/parity_with_cpu.rs` | 0 | 7 |

**Net: ~+475 / -200 LOC**, estimated 2-4 weeks of focused work depending on whether Task 1's rewrite surfaces additional structural issues (e.g., per-pair-candidate iteration in mask predicate bodies, view storage indexing for ViewCall reads, etc.).

## Sequencing with Phase 7 (kin removal)

Two reasonable orderings:

**Option (a)**: Phase 7 first, then Phase 8 on the new fixtures.
- Phase 7 replaces the wolf-sim DSL with target-game DSL + new test fixtures.
- Phase 8 closes parity on the new fixtures.
- Avoids "wasted" work on wolf-sim-specific scoring/mask bodies that may be replaced wholesale.

**Option (b)**: Phase 8 on wolf-sim first, then Phase 7.
- Phase 8 produces a green parity gate on the existing fixtures.
- Phase 7 then has a regression catcher to validate against during the kin-removal work.
- Risk: scoring/mask body content for wolf-sim may not match target-game shape; some Phase 8 task work gets discarded.

**Recommendation: Option (a)**. The structural fixes (Tasks 1-3) are reusable across any DSL surface — those land independently of fixture choice. The body-content tasks (4-6) are DSL-specific and benefit from being written once against the final fixture set. Phase 7's design (`FilteredWalk { filter }`) is independent of Phase 8 — they don't conflict.

If schedule pressure favors a green parity gate ASAP (regression-catcher value > rewrite cost), invert to Option (b).

## Out-of-scope

- **Apply-actions chain (HP/state mutations from event handlers)**: separate Phase 9. The smoke fixture doesn't exercise damage/heal/etc. (no attacks fire at d=10 with attack_range=3), so apply-actions isn't required for the smoke parity gate. Track as Phase 9.
- **Per-unit-fold-over-view-storage IR primitive**: needed for richer fixtures where view-storage is non-empty. Phase 6 Task 1's PerUnit-as-multiplication simplification is fine for smoke parity. Track separately.
- **Multi-AgentField buffer aliasing strategy** beyond Movement's single-`Pos` write: needed when ApplyActions writes HP + status + multiple fields. Track with Phase 9.

## References

- Phase 6 plan: `docs/superpowers/plans/2026-05-01-phase-6-cg-decision-lowering.md` (Tasks 1-5 done — IR + schema in place; bodies are placeholders).
- Phase 7 plan: `docs/superpowers/plans/2026-05-01-phase-7-general-spatial-queries.md` (kin removal, predicate-driven `FilteredWalk`).
- Phase 6 Task 5 investigation transcript: in-session 2026-05-01 (subagent characterized all 7 blockers).
- Reference body shape: `crates/engine_gpu_rules/src/movement.wgsl` (legacy splice; ~40 LOC of real position-update logic).
