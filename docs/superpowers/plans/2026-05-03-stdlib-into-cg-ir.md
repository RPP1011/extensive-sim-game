# Stdlib Calls Into CG IR — Unify Lowering Across Emit Contexts

> **Status: DRAFT (2026-05-03).** Phase 7 + Phase 8 landed the spatial-query + event-ring + view-fold infrastructure end-to-end across `predator_prey`, `particle_collision`, `crowd_navigation`. The audit immediately after exposed a duplication shape: each `cg/emit/<context>.rs` and `cg/lower/<context>.rs` separately decides how to translate stdlib calls (`agents.pos`, `query.nearby_agents`, `agents.creature_type`, etc.) into structural IR + WGSL. As a result the same stdlib op works in some contexts (mask) and not others (physics, view, scoring), forcing every fixture to author around the gap.
>
> This plan unifies stdlib lowering by pushing semantics into the CG IR + the namespace registry once, so every emit context walks the same normalized variants.

**Goal:** every `agents.<field>(<expr>)` and `query.<…>(<expr>)` call resolves to a single CG IR shape regardless of the surrounding rule context (mask / fold / physics / scoring / view-body). Emit contexts dispatch on CG variant, not on `(ns, method)` strings. Adding a new stdlib op is **one** registry entry + **one** structural-lowering case, not N (op kinds) × M (emit contexts) match arms.

**Architecture:** Two unifications:

1. **Per-row stdlib calls** (`agents.<field>`, `agents.is_hostile_to`, `agents.creature_type`) — already partially unified via `prog.namespace_registry` + `CgExpr::NamespaceCall`. The remaining gap is **prelude / binding completeness per emit context**: physics + scoring + view emit contexts don't pre-register the `agents.*` family the way `emit_mask.rs` does. Fix: hoist the registration into the namespace-registry seed pass so every kernel composer inherits the same set; emit contexts just look up `wgsl_fn_name` from the registry.

2. **Structural stdlib calls** (`query.nearby_agents`, `query.within_planar`, `query.nearby_items`, plus the standalone `@spatial query` decl path) — these are NOT per-row function calls; they're per-pair iteration shapes. Today the lowering from `IrExpr::NamespaceCall(Spatial, …)` into `ComputeOpKind::SpatialQuery` + `CgStmt::ForEachNeighbor*` lives only in `crates/dsl_compiler/src/cg/lower/mask.rs`. Fix: extract a shared `lower_spatial_namespace_call(call, ctx) -> Result<SpatialIterShape>` in a new `cg/lower/stdlib_spatial.rs`, callable from any rule lower-pass (physics, view, scoring) and from the `@spatial query` decl resolver. Standalone decls become named handles; inline calls become anonymous shapes; both flow through one schedule-aware lowering.

**Tech Stack:** Rust workspace; `dsl_compiler` (lower + emit + cg). Verification by re-running the existing fixture apps (`pp_app`, `pc_app`, `cn_app`) and checking observables don't change, plus removing one staging `_min` line that the slice unblocks.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `CgExpr::NamespaceCall { ns, method, args, ty }` at `crates/dsl_compiler/src/cg/expr.rs:886` — typed namespace-call CG variant; emit dispatches to `wgsl_fn_name` via registry.
  - `prog.namespace_registry` at `crates/dsl_compiler/src/cg/program.rs` — central per-`(ns, method)` registration with `WgslAccessForm` + `wgsl_fn_name`. Already the right shape for unification.
  - `lower_namespace_call` at `crates/dsl_compiler/src/cg/lower/expr.rs:2360` — central dispatch from `IrExpr::NamespaceCall` into either a registered `lower_registered_namespace_call` (function-call form) or a structural specialization.
  - `lower_registered_namespace_call` at `crates/dsl_compiler/src/cg/lower/expr.rs:2484` — turns registered methods into `CgExpr::NamespaceCall`. Works.
  - `crates/dsl_compiler/src/cg/lower/mask.rs:253` (`from query.nearby_agents(...)` per-pair lowering) — currently the **only** site that turns spatial-query namespace-calls into `ComputeOpKind::SpatialQuery` / `CgStmt::ForEachNeighbor*`. This is the duplication we need to break out.
  - `crates/dsl_compiler/src/cg/lower/spatial.rs` — already a shared spatial-lowering helper file but only consumed by mask. Natural home for the extraction.
  - `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:823` — `CgExpr::NamespaceCall` arm; uniform across all emit contexts. Already collapsed; no change needed here.
  - `crates/dsl_compiler/src/cg/lower/expr.rs:2213` — partial structural specialization: `Some(IrExpr::NamespaceCall { ns: NamespaceId::Spatial, .. }) => true` already detects spatial. Extension point.
  - Spec: `docs/spec/dsl.md` §2.3 (`@spatial query` decl), namespace-stdlib section (`agents.*`, `query.*`).
  - Audit reference: 2026-04-26-audit-language-stdlib.md items #2, #4 (the original characterization of the gap).

  Search method: `rg -n` on `NamespaceCall|namespace_call|query\.|agents\.`, direct `Read` on `cg/lower/{expr,mask,spatial}.rs`, `cg/emit/wgsl_body.rs`.

- **Decision:** **Extend** existing primitives (`namespace_registry`, `CgExpr::NamespaceCall`, `cg/lower/spatial.rs`). No new top-level CG IR nodes for the per-row case; one new structural-iter shape (`SpatialIterShape`) for the per-pair case, lowered into the **existing** `ComputeOpKind::SpatialQuery` + `CgStmt::ForEachNeighbor*` infrastructure. Reason: the registry + `NamespaceCall` already have the right ABI; the gap is registration completeness + extracting one duplicated lowering helper. Avoid introducing parallel IR nodes when the existing ones cover the semantics.

- **Rule-compiler touchpoints:**
  - DSL inputs edited: NONE in slices 1+2 (the unifications are emit-side; existing fixtures work unchanged). Slice 3 may touch `assets/sim/particle_collision_min.sim` to remove the staging guard now that per-pair physics lowers.
  - Generated outputs re-emitted: every `<fixture>_runtime/build.rs`-driven emit re-runs (mechanical regen, no observable change for slices 1+2; slice 3 produces a real per-pair physics kernel for `particle_collision`).

- **Hand-written downstream code:** NONE. Every change lives inside `dsl_compiler`. Per-fixture runtimes are unchanged (slices 1-2) or pick up a new compiler-emitted kernel automatically (slice 3).

- **Constitution check:**
  - P1 (Compiler-First): PASS — every change is in the DSL compiler (lower/emit/cg). No hand-written rule logic added to engine.
  - P2 (Schema-Hash on Layout): N/A — no SimState SoA changes.
  - P3 (Cross-Backend Parity): N/A for slices 1-2 (no behavioral change). Slice 3 must verify pc_app observables match the previous min-staged total (or the new total under the now-real per-pair semantics, documented in slice 3's task).
  - P4 (`EffectOp` Size Budget): N/A — no `EffectOp` variants added.
  - P5 (Determinism via Keyed PCG): PASS — no RNG paths changed.
  - P6 (Events Are the Mutation Channel): PASS — event emit shape unchanged.
  - P7 (Replayability Flagged): N/A — no event decls added.
  - P8 (AIS Required): PASS — this section.
  - P9 (Tasks Close With Verified Commit): PASS — every task closes with a `closes_commit` SHA.
  - P10 (No Runtime Panic): PASS — every new lowering returns `Result<…, LoweringError>` or `Result<…, EmitError>`; no `unwrap` on the deterministic path.
  - P11 (Reduction Determinism): N/A — no reductions added.

- **Runtime gate:** running `cargo run -p sim_app --bin pp_app && --bin pc_app && --bin cn_app` after each slice and asserting the printed view-fold totals match the pre-refactor numbers. Concretely:
  - Slice 1 gate: pp/pc/cn all emit unchanged totals (kill_count=200/slot, collision_count=200/slot, stuck_count=100/slot). Refactor should be observably invisible.
  - Slice 2 gate: same totals still hold; additionally `compile-dsl --cg-canonical` on a probe fixture that uses `query.nearby_agents` in a physics body emits a per-pair physics kernel without the `Unsupported` rejection.
  - Slice 3 gate: pc_app produces an observable that reflects real per-pair collision detection (not the placeholder single-event-per-particle emit). Exact target depends on the design-target body; documented when slice 3 lands.

- **Re-evaluation:** [x] AIS reviewed at design phase (initial fill, 2026-05-03). [ ] AIS reviewed post-design (after slice 1 lands and the pattern proves out).

## Pre-existing context (do NOT re-investigate)

1. `CgExpr::NamespaceCall` already typed via `(NamespaceId, method, ty)`. The emit-side lowering at `wgsl_body.rs:823` is uniform — every emit context gets the same `wgsl_fn_name(args)` form once a `(ns, method)` is registered.
2. The audit on 2026-05-03 (this session) confirmed the duplication is **not** "Call(string) is opaque"; it's that `(ns, method)` registration + structural-iter lowering live per-context. The unification target is the registration / extraction layer, not a new IR variant.
3. `lower_mask.rs` already lowers `from query.nearby_agents(...)` into `ComputeOpKind::SpatialQuery` + a per-pair iter. That logic transplants into a shared helper unchanged.
4. `crates/dsl_compiler/src/cg/lower/spatial.rs` already exists as a stub for shared spatial logic — extraction lands there, no new file needed.
5. Per-fixture runtimes don't see this refactor — they consume `dispatch_<kernel>` helpers whose binding shape is unchanged. The slices are emit-side only.

## Tasks

| # | Task | Files | LOC est. | Depends |
|---|---|---|---|---|
| 1 | **Slice 1: hoist `agents.<field>` registration into the namespace-registry seed pass.** Move per-context `agents.*` registration (currently scattered across `lower/mask.rs`, `lower/view.rs`, `lower/scoring.rs`) into a single `populate_namespace_registry::register_agents_namespace(reg, prog)` that runs once per Program. Every emit context inherits the same `wgsl_fn_name` + binding shape. Verify pp/pc/cn observables unchanged. | `cg/lower/expr.rs` (registration consolidation), `cg/program.rs::populate_namespace_registry` | ~150 net (mostly moves) | — |
| 2 | **Slice 2a: extract `lower_spatial_namespace_call` from mask into shared.** Move the `from query.nearby_agents(...)` → `ComputeOpKind::SpatialQuery` + `CgStmt::ForEachNeighbor*` translation out of `cg/lower/mask.rs:253` into `cg/lower/spatial.rs`. Signature: `fn lower_spatial_namespace_call(call: &IrCall, ctx: &mut LowerCtx) -> Result<SpatialIterShape, LoweringError>`. Mask lowering becomes a one-line call. | `cg/lower/spatial.rs`, `cg/lower/mask.rs` | ~100 net (move + 1-line call) | 1 |
| 3 | **Slice 2b: wire shared spatial-lowering into physics + view + scoring lower-passes.** Each `lower_<context>` walks IR and on hitting `IrExpr::NamespaceCall { ns: Spatial, .. }` calls `lower_spatial_namespace_call` instead of falling through to `Unsupported`. Verify with a probe `.sim` fixture that uses `for other in spatial.nearby_agents(self) { … }` inside a physics rule and confirms `--cg-canonical` emits a real per-pair kernel. | `cg/lower/physics.rs`, `cg/lower/view.rs`, `cg/lower/scoring.rs` | ~80 | 2a |
| 4 | **Slice 2c: collapse standalone `@spatial query` decl into shared lowering.** Today `Decl::Query` is parsed and dropped (the `Compilation` struct has no `queries` field). Wire it: resolve emits a named `SpatialIterShape` registered on `prog.named_queries`; `IrExpr::QueryRef(name)` lookups go through the same lowering helper as inline calls. Decls + inline calls produce the same WGSL shape via one path. | `cg/program.rs` (add `named_queries`), `cg/lower/spatial.rs`, `parse/decl_query.rs` (re-enable resolution) | ~120 | 2a |
| 5 | **Slice 3: per-pair physics kernel for `particle_collision`.** With slices 1+2 landed, the `particle_collision.sim` design-target's `for other in spatial.nearby_particles(self)` body now lowers. Replace the staging `particle_collision_min.sim` placeholder emit with the real per-pair Collision detection. Verify pc_app's `collision_count` total reflects real pair counts (not the per-particle self-loop placeholder). | `assets/sim/particle_collision_min.sim` (delete or shrink toward `particle_collision.sim`) | ~+50 / -50 sim LOC; ~0 compiler LOC | 3 |
| 6 | **Slice 3 (gate)**: cross-fixture determinism still passes. Run `cargo test --workspace` + the three `<fixture>_app` smoke binaries; record observables. | `crates/sim_app/tests/*` | 0 | 5 |

**Net (slices 1+2 only): ~+450 / -200 compiler LOC, observably invisible.** Slice 3 is opt-in DSL surface enablement; doesn't gate the unification.

## Sequencing

Slices 1, 2a, 2b, 2c are mostly independent on the resolve side, but slice 1 lands first to validate that per-context registration consolidation doesn't regress observables. Slice 2a extracts (no behavior change); 2b wires (unblocks the gap); 2c is independent of 2b but cleaner to land after 2b stabilizes. Slice 3 is the user-visible payoff that proves the unification is real.

If schedule pressure favors landing the unblocker fastest, do **slice 2 first then slice 1** — slice 2 alone gives `query.nearby_agents` in physics, which unblocks `particle_collision`. Slice 1 is the architectural cleanup that pays back when the next stdlib op gets added. Recommended order is 1→2 because slice 1's pattern (consolidated registration) is the load-bearing shape that slice 2 inherits.

## Out-of-scope

- **`verb` / `invariant` / `probe` / `metric` emit pipelines.** These parse + resolve to IR but have no emit paths at all. Adding them is not unification — it's wholesale new emit work. Track as a separate plan.
- **View-fold body operator-set enforcement** (`UdfInViewFoldBody` error variant). Resolver doesn't reject forbidden constructs in `@materialized` fold bodies. Related shape (validation in resolve) but orthogonal to stdlib unification. Track separately.
- **Top-K spatial query (`@top_k`) variant** — already wired via `ForEachNeighborTopK`; uses the same registry path, no extra unification needed.
- **Stdlib semantic spec ↔ impl mismatches** (`stun_remaining_ticks` naming, etc.) — fix in a docs-+-spec sweep, not in this plan.

## Anti-pattern: do NOT do this

- Do **not** add a `match call_name { … }` table in the emit-context layer. That's the duplication this plan exists to remove. Every emit context walks `CgExpr::NamespaceCall` uniformly via the registry, or walks `CgStmt::ForEachNeighbor*` uniformly via the structural-lowering pass.
- Do **not** introduce `CgExpr::AgentField`, `CgExpr::SpatialNeighborIter`, etc. as parallel IR variants alongside `CgExpr::NamespaceCall`. The registry + existing `NamespaceCall` is the one path. Two paths = the same problem with extra steps.
- Do **not** lower `agents.<field>` in physics differently than in view (e.g. inlining the SoA load in physics but going through a fn-call in view). One `wgsl_fn_name` per `(ns, method)`, registered once, called from every context.

## References

- **Audit**: 2026-05-03 in-session DSL coverage audit (this session's prior turn). Identifies items #2, #4, #10, #11 as the targets unified by this plan.
- **Prior unification work**: `2026-04-29-dsl-compute-graph-ir.md` introduced `CgExpr::NamespaceCall` + the registry. This plan finishes the migration that plan started.
- **Sibling plan (still open)**: `2026-05-01-phase-7-general-spatial-queries.md` — the spatial-query infrastructure this plan extends. Phase 7 shipped `ForEachNeighbor*`; this plan makes it usable from physics + view + scoring contexts uniformly.
- **Spec**: `docs/spec/dsl.md` §2.3 (spatial queries), namespace-stdlib section (`agents.*`, `query.*`).
- **Constitution**: P1 (this plan stays in the DSL compiler), P8 (this AIS).
