# Phase 6: CG Decision-Chain Lowering — Plan Stub

> **Status: STUB.** This plan captures the work deferred from `2026-05-01-cg-lowering-gap-closure.md` so `parity_with_cpu --features gpu` turns green for the smoke fixture (and beyond). The detailed task breakdown lives in a future expansion of this stub; today's content is the goal, the architectural shape, and the scope estimate.

**Goal:** Close the CG-side decision/movement chain so the smoke fixture's parity gate (`parity_with_cpu_n4_t{1,10,100}`) turns green without legacy-emitter splicing. Specifically: emit real Scoring + Movement (and downstream apply/event) kernels driven by CG IR rather than the hand-written `emit_scoring_*` / `emit_movement_*` modules.

**Architecture:** Three structural extensions to the CG lowering pipeline:

1. **`PerUnit` modifier lowering.** The DSL writes `score(action) = base + (view::X(self, _) per_unit C) + ...` to mean "add C × (sum of view X's primary slot reads, K-bounded by the view's storage hint)". Today `PerUnit` is rejected at `cg/lower/expr.rs:599`. Lowering needs either a new `CgExpr::PerUnitFold { view, anchor, scale }` variant (per-agent ring walk over view storage) or a desugar that synthesizes the fold inline. The choice between these is driven by what the IR's other consumers (well_formed, schedule fusion) need to see.

2. **`Positional` action-head lowering.** Scoring rules with a positional binder (`Attack(target) = ...`) currently route through `parametric_scoring_head_label` which rejects `IrActionHeadShape::Positional`. The fix follows Task 1's pattern (event-pattern bindings): in the scoring-row lowering, walk the action's pattern bindings, register `target` as a `LocalRef → LocalId` mapping, synthesize a `CgStmt::Let { local: target_local, value: <something>, ty: AgentId }`. The "value" depends on the surrounding dispatch shape — for `PerPair { source: SpatialQuery(_) }` rows the target is the per-pair candidate.

3. **Movement-as-rule design — LOCKED to option (b).** Movement is structurally a "per-agent rule that reads decided actions and writes a field." The same pattern fits other future per-agent sweeps: cooldown ticking, stun/slow expiry, need decay, regen, queued-movement-target updates, voxel-local-pos integration. **The IR shape generalizes by extending `PhysicsRule` with an optional `source_event`** (or equivalent dispatch-shape selector that distinguishes PerEvent from PerAgent dispatch).
   - When `source_event = Some(kind)` → existing PerEvent dispatch over the source ring (today's chronicle/damage/heal/etc. handlers).
   - When `source_event = None` → PerAgent dispatch over the alive bitmap (Movement; future per-agent sweeps).

   The DSL surfaces this with a phase annotation (`@phase(per_agent)` or similar) so future rules slot in without IR changes. Cost is minimal — one new variant field + one new phase tag — and avoids the "Movement is special" trap. Rejected option: a dedicated `ComputeOpKind::ScoringApply` variant. It's a narrower shape that only fits Movement; would need a refactor when the second per-agent sweep arrives. The user's question (2026-05-01): "Won't other vec3 fields eventually exist and potentially benefit from similar hyperoptimization?" anchors this decision.

## Tasks (skeleton)

| # | Task | Files | LOC est. |
|---|---|---|---|
| 1 | `PerUnit` lowering: design + implement | `cg/lower/expr.rs`, `cg/op.rs`, `cg/emit/wgsl_body.rs` | ~150 |
| 2 | `Positional` action-head lowering | `cg/lower/scoring.rs`, `cg/lower/event_binding.rs` (reuse) | ~80 |
| 3 | Movement-as-rule via `PhysicsRule` extension (option b — LOCKED). Add optional `source_event` (or PerAgent dispatch shape selector); DSL `@phase(per_agent)` annotation; Movement body lowering reads scoring_output, writes pos. | `cg/op.rs`, `cg/lower/physics.rs`, `cg/emit/kernel.rs`, `cg/dispatch.rs`, `dsl_resolver` (phase tag) | ~250 |
| 4 | Apply-actions chain (HP/state mutations from event handlers) | `cg/lower/physics.rs` (statement-level namespace-call lowering, mirror of Task 4) | ~200 |
| 5 | Cleanup: retire Route C splice in `xtask/compile_dsl_cmd.rs` | `xtask/src/compile_dsl_cmd.rs` | -200 |
| 6 | Parity gate green + Movement emit-quality verification | `engine_gpu/tests/parity_with_cpu.rs`, new emit-shape test | 0+~50 |

**Movement emit-quality acceptance criterion (added 2026-05-01)**: position is structurally privileged — spatial-hash + kin/engagement queries + distance builtins read `agent_pos` 5+ times per tick. The hand-written legacy Movement kernel may carry cache-friendly vec3 load patterns (single 12-byte read vs three scalar reads), branch-free MOVE_TOWARD/FLEE deltas, etc. Option (b) preserves the SoA contract and schedule ordering automatically (via reads/writes), but the emit-body quality is not automatic. Add a test that:
- Loads the legacy `engine_gpu_rules/src/movement.wgsl` (pre-Phase-6 reference).
- Loads the CG-emitted `movement.wgsl` post-Phase-6.
- Compares structural shape: same `bitcast<f32>` count for pos reads, same atomicity (none today), same write pattern. Allows divergence on naming + comments but flags if vec3-load merging regresses.

The user's question (2026-05-01) "we have optimizations built into movement as it is position based ... spatial queries heavily benefit from that and we assume it is always present" anchors this acceptance test: position's privileged role is the contract; emit-quality preserves it.

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
