# Phase 6: CG Decision-Chain Lowering — Plan Stub

> **Status: STUB.** This plan captures the work deferred from `2026-05-01-cg-lowering-gap-closure.md` so `parity_with_cpu --features gpu` turns green for the smoke fixture (and beyond). The detailed task breakdown lives in a future expansion of this stub; today's content is the goal, the architectural shape, and the scope estimate.

**Goal:** Close the CG-side decision/movement chain so the smoke fixture's parity gate (`parity_with_cpu_n4_t{1,10,100}`) turns green without legacy-emitter splicing. Specifically: emit real Scoring + Movement (and downstream apply/event) kernels driven by CG IR rather than the hand-written `emit_scoring_*` / `emit_movement_*` modules.

**Architecture:** Three structural extensions to the CG lowering pipeline:

1. **`PerUnit` modifier lowering.** The DSL writes `score(action) = base + (view::X(self, _) per_unit C) + ...` to mean "add C × (sum of view X's primary slot reads, K-bounded by the view's storage hint)". Today `PerUnit` is rejected at `cg/lower/expr.rs:599`. Lowering needs either a new `CgExpr::PerUnitFold { view, anchor, scale }` variant (per-agent ring walk over view storage) or a desugar that synthesizes the fold inline. The choice between these is driven by what the IR's other consumers (well_formed, schedule fusion) need to see.

2. **`Positional` action-head lowering.** Scoring rules with a positional binder (`Attack(target) = ...`) currently route through `parametric_scoring_head_label` which rejects `IrActionHeadShape::Positional`. The fix follows Task 1's pattern (event-pattern bindings): in the scoring-row lowering, walk the action's pattern bindings, register `target` as a `LocalRef → LocalId` mapping, synthesize a `CgStmt::Let { local: target_local, value: <something>, ty: AgentId }`. The "value" depends on the surrounding dispatch shape — for `PerPair { source: SpatialQuery(_) }` rows the target is the per-pair candidate.

3. **Movement-as-rule design.** Movement is structurally NOT a physics rule (it doesn't observe an event — it's a per-agent post-scoring sweep). Today's `ComputeOpKind` set has no shape for "per-agent dispatch reading scoring_output, no source event." Two options:
   - (a) New `ComputeOpKind::ScoringApply { ... }` variant with `DispatchShape::PerAgent`, reading `ScoringOutput` and writing agent fields.
   - (b) Extend `PhysicsRule` semantics to allow per-agent dispatch over the alive bitmap with no source event.

   (a) is structurally cleaner but requires `cg/op.rs`, `cg/well_formed.rs`, `cg/emit/kernel.rs`, `cg/dispatch.rs` updates. (b) is a smaller surface but conflates two distinct execution models.

## Tasks (skeleton)

| # | Task | Files | LOC est. |
|---|---|---|---|
| 1 | `PerUnit` lowering: design + implement | `cg/lower/expr.rs`, `cg/op.rs`, `cg/emit/wgsl_body.rs` | ~150 |
| 2 | `Positional` action-head lowering | `cg/lower/scoring.rs`, `cg/lower/event_binding.rs` (reuse) | ~80 |
| 3 | Movement-as-rule: pick (a) or (b), implement | `cg/op.rs`, `cg/lower/physics.rs` (or new `cg/lower/scoring_apply.rs`), `cg/emit/kernel.rs`, `cg/dispatch.rs` | ~300 |
| 4 | Apply-actions chain (HP/state mutations from event handlers) | `cg/lower/physics.rs` (statement-level namespace-call lowering, mirror of Task 4) | ~200 |
| 5 | Cleanup: retire Route C splice in `xtask/compile_dsl_cmd.rs` | `xtask/src/compile_dsl_cmd.rs` | -200 |
| 6 | Parity gate green | `engine_gpu/tests/parity_with_cpu.rs` | 0 |

**Total: ~530 LOC net** (730 added + 200 deleted), 4-7 days estimated.

## Architectural Impact Statement (skeleton)

- **Decision:** New IR variants (yes for `PerUnit` + `ScoringApply`; reuse for Positional binders via Task 1's `event_binding` pattern).
- **Rule-compiler touchpoints:** Scoring DSL surface unchanged; lowering pipeline extended.
- **Hand-written downstream code:** Once this lands, `crates/dsl_compiler/src/emit_scoring_*.rs` + `emit_movement_*.rs` retire (Task 5.8 from the parent plan finally lands too).
- **Constitution check:** P1 (Compiler-First), P3 (Cross-Backend Parity) — both PASS by construction. P11 (Reduction Determinism) — verify the `PerUnit` fold maintains deterministic ordering across the K-bounded ring walk.
- **Runtime gate:** `parity_with_cpu_n4_t{1,10,100}` GREEN. This is the closure metric.

## Out-of-scope

- `@phase(event)` physics-rule statement-level namespace-call mutations (`agents.set_hp(t, x)` etc.) — they're a separate diagnostic class (Task 1's namespace-CALL expression lowering closed the read side; the statement-level write side is its own work). Track as Phase 7 if the smoke fixture's parity-green state doesn't need it (it doesn't — d=10 with attack_range=3 means no attacks fire, no damage events, no HP mutations).

## References

- Closing plan: `docs/superpowers/plans/2026-05-01-cg-lowering-gap-closure.md` (Tasks 6, 11, 12 deferred here).
- Route C splice site: `crates/xtask/src/compile_dsl_cmd.rs:1671-1812` (commit `7d566a92`).
- Legacy emitters being retired: `emit_scoring_*.rs`, `emit_movement_*.rs` (~4,246 LOC total in `crates/dsl_compiler/src/`).
- Investigation findings: parent plan's Task 9 inventory section.
