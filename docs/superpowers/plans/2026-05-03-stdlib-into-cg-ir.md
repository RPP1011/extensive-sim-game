# Stdlib Calls Into CG IR — Unify Lowering Across Emit Contexts

> **Status: DRAFT (2026-05-03, revised post-investigation).** Phase 7 + Phase 8 landed the spatial-query + event-ring + view-fold infrastructure end-to-end across `predator_prey`, `particle_collision`, `crowd_navigation`. The audit immediately after exposed what *looked* like duplicated lowering across emit contexts; pre-implementation investigation (this same session) revealed the real shape is sharper than the audit suggested.
>
> **Investigation findings (2026-05-03):**
> 1. `agents.*` registration is **not** scattered. The namespace registry is already centralized in one place: `cg/lower/driver.rs:582 populate_namespace_registry`. Only `is_hostile_to` and `engaged_with_or` are registered there; everything else (`agents.pos`, `agents.creature_type`, `agents.alive`, etc.) takes a structural fallback path in `cg/lower/expr.rs:2403` that produces `CgExpr::Read(DataHandle::AgentField{ field, target: AgentRef::Target(expr_id) })`.
> 2. The structural fallback is already shared across all emit contexts via `lower_cg_expr_to_wgsl` (one site).
> 3. The actual gap is **emit-side**: at `cg/emit/wgsl_body.rs:650`, the emit returns a B1 typed-default placeholder (`b1_default_for_field_ty(field.ty())` → `0.0`, `vec3(0.0)`) for every `AgentRef::Target(_)` read. So `agents.pos(other)` silently returns `vec3(0.0)` in every context — physics, view, scoring, fold. The audit's "works in views, broken in physics" framing was misleading; the gap is uniform, in one site.
> 4. `query.<…>` lowering is in fact only in `cg/lower/mask.rs:253`; that part of the audit + plan stands.
>
> This plan still pushes stdlib semantics into one shared site so every emit context inherits the fix — but slice 1 is now a one-site emit fix, not a registration consolidation.

**Goal:** every `agents.<field>(<expr>)` and `query.<…>(<expr>)` call lowers correctly regardless of the surrounding rule context (mask / fold / physics / scoring / view-body). The shared CG IR shape already exists; this plan finishes the emit-side and structural-lowering-side gaps so the IR's promise holds end-to-end.

**Architecture:** Two surgical fixes:

1. **Cross-agent reads (`agents.<field>(other)` for non-`Self_` targets).** The structural lowering already produces `CgExpr::Read(AgentField{ field, target: AgentRef::Target(expr_id) })`. The emit returns a typed-default placeholder. Fix: hoist `let target_expr_<N> = <lowered_target>;` into the enclosing statement scope (analogous to how `ForEachNeighbor` already pre-binds its candidate index), then emit `agent_<field>[target_expr_<N>]`. One-site fix in shared emit (`cg/emit/wgsl_body.rs`); inherits to all contexts because they all walk the same emit pipeline.

2. **Structural stdlib calls** (`query.nearby_agents`, `query.within_planar`, `query.nearby_items`, plus the standalone `@spatial query` decl path) — these are NOT per-row function calls; they're per-pair iteration shapes. Today the lowering from `IrExpr::NamespaceCall(Spatial, …)` into `ComputeOpKind::SpatialQuery` + `CgStmt::ForEachNeighbor*` lives only in `crates/dsl_compiler/src/cg/lower/mask.rs`. Fix: extract a shared `lower_spatial_namespace_call(call, ctx) -> Result<SpatialIterShape>` into the existing `cg/lower/spatial.rs`, callable from any rule lower-pass (physics, view, scoring) and from the `@spatial query` decl resolver. Standalone decls become named handles; inline calls become anonymous shapes; both flow through one schedule-aware lowering.

**Tech Stack:** Rust workspace; `dsl_compiler` (lower + emit + cg). Verification by re-running the existing fixture apps (`pp_app`, `pc_app`, `cn_app`) and checking observables don't change, plus removing one staging `_min` line that the slice unblocks.

## Architectural Impact Statement

- **Existing primitives searched:**
  - `CgExpr::NamespaceCall { ns, method, args, ty }` at `crates/dsl_compiler/src/cg/expr.rs:886` — typed namespace-call CG variant; emit dispatches to `wgsl_fn_name` via registry.
  - `prog.namespace_registry` at `crates/dsl_compiler/src/cg/program.rs:683` + populated at `cg/lower/driver.rs:582` — central per-`(ns, method)` registration with `WgslAccessForm` + `wgsl_fn_name`. **Already centralized**: only one `populate_namespace_registry` site, registers `agents.is_hostile_to`, `agents.engaged_with_or`, `query.nearest_hostile_to_or`, `world.tick`. No scattered per-context registration to consolidate.
  - `lower_namespace_call` at `crates/dsl_compiler/src/cg/lower/expr.rs:2360` — central dispatch from `IrExpr::NamespaceCall` into either a registered `lower_registered_namespace_call` (function-call form) or the `(NamespaceId::Agents, field) if args.len() == 1` structural fallback at line 2403.
  - **Structural fallback for `agents.<field>(target)`** at `cg/lower/expr.rs:2403-2453` — produces `CgExpr::Read(DataHandle::AgentField{ field, target: AgentRef::Target(expr_id) })`. Already shared across emit contexts.
  - **The bug**: `cg/emit/wgsl_body.rs:649-652` — when `target` is `AgentRef::Target(_)`, returns `b1_default_for_field_ty(field.ty()).to_string()` (a placeholder constant: `0.0`, `vec3(0.0)`, `0u`). Comment at the call site flags it as a "B1 typed-default fallback — the per-thread target index isn't yet threaded into kernel-local scope (Phase 8 follow-up)". This plan is that follow-up.
  - **Reference for the hoisting pattern**: `ForEachNeighbor*` lowering already pre-binds `per_pair_candidate` in the kernel preamble; the emitter at `wgsl_body.rs:665-686` reads it through `AgentRef::PerPairCandidate`. The fix mirrors that shape: emit a stmt-scope `let target_expr_<N> = <lowered>;` and read `agent_<field>[target_expr_<N>]`.
  - `crates/dsl_compiler/src/cg/lower/mask.rs:253` (`from query.nearby_agents(...)` per-pair lowering) — currently the **only** site that turns spatial-query namespace-calls into `ComputeOpKind::SpatialQuery` / `CgStmt::ForEachNeighbor*`. This duplication is real (slice 2 target).
  - `crates/dsl_compiler/src/cg/lower/spatial.rs` — already a shared spatial-lowering helper file. Natural home for the slice 2 extraction.
  - `crates/dsl_compiler/src/cg/emit/wgsl_body.rs:823` — `CgExpr::NamespaceCall` arm; uniform across all emit contexts. No change needed here.
  - Spec: `docs/spec/dsl.md` §2.3 (`@spatial query` decl), namespace-stdlib section (`agents.*`, `query.*`).
  - Audit reference: 2026-04-26-audit-language-stdlib.md items #2, #4 (original characterization; this plan refines it after deeper investigation).

  Search method: `rg -n` on `NamespaceCall|namespace_call|query\.|agents\.|target_expr_|AgentRef::Target`, direct `Read` on `cg/lower/{expr,driver,mask,spatial,physics}.rs`, `cg/emit/wgsl_body.rs`.

- **Decision:** **Extend** existing primitives. Slice 1 is a one-site emit fix in `lower_cg_expr_to_wgsl` that threads `let target_expr_<N>` through the statement-level emit, mirroring the existing `PerPairCandidate` pre-binding pattern. Slice 2 is a one-helper extraction from `cg/lower/mask.rs` into `cg/lower/spatial.rs`. **No new CG IR nodes**, no new emit pipelines, no new namespace-registry seed pass. The IR already has the right shapes (`CgExpr::Read(AgentField{Target(_)})`, `ComputeOpKind::SpatialQuery`, `CgStmt::ForEachNeighbor*`); this plan finishes their wiring.

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
2. **Investigation (2026-05-03)** confirmed: namespace registration is **not** scattered. `populate_namespace_registry` is the single seed site (`cg/lower/driver.rs:582`). The audit's "scattered per-context registration" framing was wrong.
3. **Investigation (2026-05-03)** revealed the real per-row gap: `cg/emit/wgsl_body.rs:649-652` returns a B1 typed-default placeholder for every `AgentRef::Target(_)` read, so `agents.pos(other)` silently returns `vec3(0.0)` everywhere. The fix is hoisting `let target_expr_<N> = <lowered>;` into stmt scope, mirroring `PerPairCandidate`'s existing pre-binding pattern at `wgsl_body.rs:665-686`.
4. `lower_mask.rs:253` does lower `from query.nearby_agents(...)` into `ComputeOpKind::SpatialQuery` + a per-pair iter. That logic transplants into a shared helper unchanged.
5. `crates/dsl_compiler/src/cg/lower/spatial.rs` already exists — extraction lands there, no new file.
6. Per-fixture runtimes don't see this refactor — they consume `dispatch_<kernel>` helpers whose binding shape is unchanged. The slices are compiler-side only.

## Tasks

| # | Task | Files | LOC est. | Depends |
|---|---|---|---|---|
| 1 | **Slice 1: hoist `AgentRef::Target(expr_id)` reads into stmt-scope `let` bindings.** Replace the B1 typed-default at `wgsl_body.rs:651` with a real lowering: lift `let target_expr_<N>: u32 = <lowered_target>;` into the enclosing statement scope (in `lower_cg_stmt`-level emit, since exprs can't introduce `let`s), then have `agent_field_access` emit `agent_<field>[target_expr_<N>]` for `AgentRef::Target`. Mirrors the existing `AgentRef::PerPairCandidate` pre-binding pattern. One-site fix; all emit contexts inherit because they share `lower_cg_expr_to_wgsl` / `lower_cg_stmt`. | `cg/emit/wgsl_body.rs` (`lower_cg_expr_to_wgsl`, `lower_cg_stmt`, `agent_field_access`); possibly `cg/emit/kernel.rs` if pre-stmt threading needs a hook | ~80–120 | — |
| 2 | **Slice 1 (gate)**: write a unit test in `cg/emit/wgsl_body.rs` asserting `Read(AgentField{Pos, Target(expr)})` emits `agent_pos[target_expr_<N>]` (not `vec3(0.0)`). Run pp/pc/cn fixture apps to confirm observables unchanged (no fixture currently exercises non-`Self_` targets in physics, so this should be a pure no-regression gate). | `cg/emit/wgsl_body.rs` (test), pp_app/pc_app/cn_app smoke run | ~30 | 1 |
| 3 | **Slice 2a: extract `lower_spatial_namespace_call` from mask into shared.** Move the `from query.nearby_agents(...)` → `ComputeOpKind::SpatialQuery` + `CgStmt::ForEachNeighbor*` translation out of `cg/lower/mask.rs:253` into `cg/lower/spatial.rs`. Signature: `fn lower_spatial_namespace_call(call: &IrCall, ctx: &mut LowerCtx) -> Result<SpatialIterShape, LoweringError>`. Mask lowering becomes a one-line call. Verify pp/pc/cn observables unchanged (mask is the only consumer today). | `cg/lower/spatial.rs`, `cg/lower/mask.rs` | ~100 net (move + 1-line call) | 1 |
| 4 | **Slice 2b: wire shared spatial-lowering into physics + view + scoring lower-passes.** Each `lower_<context>` walks IR and on hitting `IrExpr::NamespaceCall { ns: Spatial, .. }` calls `lower_spatial_namespace_call` instead of falling through to `Unsupported`. Verify with a probe `.sim` fixture that uses `for other in spatial.nearby_agents(self) { … }` inside a physics rule and confirms `--cg-canonical` emits a real per-pair kernel. | `cg/lower/physics.rs`, `cg/lower/view.rs`, `cg/lower/scoring.rs` | ~80 | 3 |
| 5 | **Slice 2c: collapse standalone `@spatial query` decl into shared lowering.** Today `Decl::Query` is parsed and dropped (the `Compilation` struct has no `queries` field). Wire it: resolve emits a named `SpatialIterShape` registered on `prog.named_queries`; `IrExpr::QueryRef(name)` lookups go through the same lowering helper as inline calls. Decls + inline calls produce the same WGSL shape via one path. | `cg/program.rs` (add `named_queries`), `cg/lower/spatial.rs`, `parse/decl_query.rs` (re-enable resolution) | ~120 | 3 |
| 6 | **Slice 3: per-pair physics kernel for `particle_collision`.** With slices 1+2 landed, the `particle_collision.sim` design-target's `for other in spatial.nearby_particles(self)` body now lowers. Replace the staging `particle_collision_min.sim` placeholder emit with real per-pair Collision detection. Verify pc_app's `collision_count` total reflects real pair counts (not the per-particle self-loop placeholder). | `assets/sim/particle_collision_min.sim` (delete or shrink toward `particle_collision.sim`) | ~+50 / -50 sim LOC; ~0 compiler LOC | 4 |
| 7 | **Slice 3 (gate)**: cross-fixture determinism still passes. Run `cargo test --workspace` + the three `<fixture>_app` smoke binaries; record observables. | `crates/sim_app/tests/*` | 0 | 6 |

**Net (slices 1+2 only): ~+380 / -180 compiler LOC, observably invisible.** Slice 3 is opt-in DSL surface enablement; doesn't gate the unification.

## Sequencing

Slices 1 and 2 are independent on the lowering side (slice 1 is emit-side; slice 2 is lower-side). Recommended order: **1 → 2a → 2b → 2c → 3**. Slice 1 lands first because it's smallest and validates the "one-site fix, all contexts inherit" hypothesis cheaply (the existing fixtures don't exercise non-`Self_` Target reads, so the gate is a no-regression run + a unit test on the emit shape). Slice 2a is a pure code move; 2b is the unblocker; 2c is the dead-decl cleanup; 3 is the user-visible payoff.

If slice 1 surfaces unexpected complexity (e.g., the pre-stmt threading needs more emit-pipeline plumbing than the `PerPairCandidate` precedent suggests), pause and split slice 1 further before continuing.

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
