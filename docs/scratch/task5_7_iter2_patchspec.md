# Task 5.7-iter2 Patch Spec — Driver-Level Ring Unification

**Date:** 2026-04-29
**HEAD at dispatch:** `65d913c7` (post Task 5.7-iter1).
**Lib + xtask test baseline:** 796 + 13 passing.
**Target:** `gpu_pipeline_smoke` under `--cg-canonical` should compile (Bucket B1).
**Stretch target:** schedule entries collapse from 9 ViewFold + N PhysicsRule per-handler kernels to 7 view-fused + 1–2 physics-fused kernels (matching legacy topology).

---

## Investigation findings

### F1 — Driver assigns 1 ring per event KIND (1:1)

`crates/dsl_compiler/src/cg/lower/driver.rs:283-316` — `populate_event_kinds`:

```rust
for (i, event) in comp.events.iter().enumerate() {
    let kind_id = EventKindId(i as u32);
    let ring_id = EventRingId(i as u32);   // <-- the 1:1 rule
    ring_ids.push(ring_id);
    ctx.register_event_kind(...);
    ctx.builder.intern_event_kind_name(kind_id, ...);
    ctx.builder.intern_event_ring_name(ring_id, ...);
}
```

**This is the root cause.** The driver allocates `EventRingId(i)` per event kind. Every ViewFold handler matching event kind `i` ends up dispatched on `PerEvent { source_ring: EventRingId(i) }`. Two handlers on the same view but different events therefore land on different `DispatchShapeKey::PerEvent(i)` projections, and `cg/schedule/fusion.rs::fusion_candidates` (line 320–423) splits at the shape boundary.

### F2 — Runtime contract is one shared `batch_events_ring` buffer

`crates/engine_gpu_rules/src/resident_context.rs:21,23,89,95` — the resident context owns ONE `batch_events_ring` + ONE `batch_events_tail` buffer. Every legacy fold kernel binds those at slots 0/1 (`crates/dsl_compiler/src/emit_view_fold_kernel.rs:142,149`):

```rust
bg_source: BgSource::Resident("batch_events_ring".into()),  // slot 0
bg_source: BgSource::Resident("batch_events_tail".into()),  // slot 1
```

The CG emitter already mirrors this: `crates/dsl_compiler/src/cg/emit/kernel.rs:1203,1208` (the 7-binding ViewFold spec). The emit layer is correct; the fusion layer never sees that the rings are unified at runtime.

### F3 — Empirical confirmation: 9 ViewFold ops, 8 distinct rings

Side-channel run: `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical` produces these fusion diagnostics:

```
op#6  per_event(ring=#1)   — fold_threat_level on AgentAttacked
op#7  per_event(ring=#26)  — fold_threat_level on EffectDamageApplied
op#8  per_event(ring=#24)  — fold_engaged_with on EngagementCommitted
op#9  per_event(ring=#25)  — fold_engaged_with on EngagementBroken
op#10 per_event(ring=#1)   — fold_my_enemies on AgentAttacked
op#11 per_event(ring=#34)  — fold_kin_fear on FearSpread
op#12 per_event(ring=#35)  — fold_pack_focus on PackAssist
op#13 per_event(ring=#36)  — fold_rally_boost on RallyCall
op#14 per_event(ring=#22)  — fold_memory on RecordMemory
```

Note especially: ops #6 and #10 both reference ring #1 (AgentAttacked) but are not consecutive in topological order, so even today's existing `fusion_candidates` walk splits them. The ring identity alone is insufficient — the emitter must recognise that ALL these rings are aliases for the same shared `batch_events_ring` buffer.

### F4 — Generated SCHEDULE has 9 fold kernels + 0 physics kernels

`crates/engine_gpu_rules/src/schedule.rs` (post `--cg-canonical`):

```rust
DispatchOp::Indirect { kernel: KernelId::FoldThreatLevelAgentAttacked, ... },
DispatchOp::Indirect { kernel: KernelId::FoldThreatLevelEffectDamageApplied, ... },
DispatchOp::Indirect { kernel: KernelId::FoldEngagedWithEngagementCommitted, ... },
DispatchOp::Indirect { kernel: KernelId::FoldEngagedWithEngagementBroken, ... },
DispatchOp::Indirect { kernel: KernelId::FoldMyEnemiesAgentAttacked, ... },
DispatchOp::Indirect { kernel: KernelId::FoldKinFearFearSpread, ... },
DispatchOp::Indirect { kernel: KernelId::FoldPackFocusPackAssist, ... },
DispatchOp::Indirect { kernel: KernelId::FoldRallyBoostRallyCall, ... },
DispatchOp::Indirect { kernel: KernelId::FoldMemoryRecordMemory, ... },
```

vs legacy (`crates/engine_gpu_rules/src/schedule.rs` pre-switchover):

```rust
DispatchOp::Kernel(KernelId::FoldThreatLevel),
DispatchOp::Kernel(KernelId::FoldEngagedWith),
... (7 total fold kernels, one per view)
DispatchOp::FixedPoint { kernel: KernelId::Physics, max_iter: 8 },
```

Engine_gpu's SCHEDULE arm dispatch (`crates/engine_gpu/src/lib.rs:828-844`) hardcodes the legacy names. **Bucket B1**: 69 build errors all stem from `engine_gpu_rules::fold_threat_level::FoldThreatLevelKernel` not existing under the per-handler naming; the per-handler modules are named `fold_threat_level_agent_attacked` etc.

### F5 — Physics rules don't lower today; B1 only blocks on fold names

CG diagnostic stream contains `physics#20 body … contains AST statement For` and `physics#21 body … contains AST statement For` for several rules. The current run produces ZERO PhysicsRule ops in the program — every physics rule defers due to AST-coverage gaps (For, Let, namespace setters). So the `KernelId::Physics` reference at engine_gpu/src/lib.rs:845 has no producer in the CG schedule at all.

The B1 unblocking work is therefore **fold-only** for now. Physics-rule fusion will become important once Phase-2 physics lowering closes its AST gaps; the spec covers it as a parallel rule for forward-compatibility.

### F6 — Legacy fold kernels are stub bodies (Stream B δ-deferred)

`crates/engine_gpu_rules/src/fold_threat_level.wgsl` is a stub that "touches every binding so naga keeps them live"; the real fold body has not been hoisted into the WGSL stub. **The legacy and CG fold WGSL bodies are equivalently stubs at runtime.** This iter does not need to migrate fold-body lowering; it only needs to align kernel names (and bindings count via fusion).

### F7 — KernelTopology has 3 variants, all match sites listed

Match sites on `KernelTopology` (full census at HEAD `65d913c7`):

- `crates/dsl_compiler/src/cg/schedule/synthesis.rs:139,150,167-198,431-510` — Display, label(), ops(), group_to_topology.
- `crates/dsl_compiler/src/cg/emit/kernel.rs:257-260` (`kernel_topology_to_spec_and_body` resolution).
- `crates/dsl_compiler/src/cg/emit/kernel.rs:970-974` (`semantic_kernel_name_for_topology`).
- `crates/dsl_compiler/src/cg/emit/cross_cutting.rs:649-689` (`classify_topology_for_schedule`).

Plus ~20 test-only constructions in `kernel.rs` and `synthesis.rs` (would each need updating if a new variant lands).

---

## Approach decision: **Option B — driver-level fix**

The original task description offered three options:

- **A:** Add a `KernelTopology::TaggedFusion` variant with internal `switch (event.tag)` dispatch + a new fusion pass.
- **B:** Driver-level fix — unify event rings so all events share one ring; existing fusion handles the rest.
- **C:** Hybrid.

**Choosing B.** Justification:

1. **Smaller diff.** Estimated 60–120 LOC across 2 files vs. 600–900 LOC across 6+ files for Option A.
2. **No new variants.** Every match site on `KernelTopology` keeps working unchanged. Schedule emit, kernel naming, body composition all still see `Fused` / `Split` / `Indirect`.
3. **Models the runtime truth.** The runtime really does have one shared `batch_events_ring`. The 1:1 EventKindId↔EventRingId rule was a Phase-1 placeholder that the comment at `driver.rs:280-282` explicitly calls "purely a convenience for the driver" — it was always meant to be revisited when fusion lit up.
4. **No WGSL body composition required.** The fold WGSL today is a stub that touches bindings (F6). Adding `switch (event.tag)` dispatch to the fold body is unnecessary work for B1 unblocking.
5. **Matches legacy fusion topology exactly.** Legacy emits 7 view-fused kernels because the legacy emitter naturally unions handlers on a single bgl spec (`fold_kernel_spec` at `emit_view_fold_kernel.rs:83-227` walks all `fold_event_names(view)` into one binding set). After driver-level ring unification, CG fusion will produce the same 7 view-fused groups for the same structural reason.
6. **Physics rules drop in for free.** Once physics-rule AST coverage closes (separate task), all PhysicsRule ops will share the same shared event ring(s), and existing fusion will fuse them without further intervention. Replayability splits become an additional driver-level decision — addressed below.

### What B unifies and what it leaves split

After Option B applies, the existing `fusion_candidates` walks WILL fuse each view's handlers when they are consecutive in topological order. They will NOT fuse handlers across different views (different ViewFold ops have different bodies and different writes — write-conflict gate fires). They will NOT automatically fuse two handlers of the same view that are SEPARATED in topological order by a different op (the consecutive-only restriction in the existing fusion analysis).

Concretely from F3:

- ops #6, #7 (both `threat_level`) are consecutive → will fuse into `fold_threat_level` (1 kernel).
- op #8, #9 (both `engaged_with`) are consecutive → will fuse into `fold_engaged_with`.
- ops #10 (`my_enemies`), #11 (`kin_fear`), #12 (`pack_focus`), #13 (`rally_boost`), #14 (`memory`) are each singleton in topological order between them; they will each become their own `fold_<view>` kernel (singleton == legacy 1-handler-per-view shape).

**Net result:** 7 view-fused kernels, matching legacy 1:1 (count + naming). Bucket B1 unblocks.

The kernel naming policy at `cg/emit/kernel.rs:1022-1046` produces `fold_<view>_<event>` for ViewFold ops with one event-kind. After fusion, the multi-event group's `semantic_kernel_name` enters the `body_ops.len() > 1` branch (line 990–1000), prefixing with `fused_` → `fused_fold_<view>_<event0>`. **This is wrong** — legacy expects `fold_<view>` (no event suffix, no `fused_` prefix). Patch 2 below addresses naming.

### Replayability split (forward-looking)

The legacy SCHEDULE has ONE physics kernel (line 845, `FixedPoint`) covering all physics rules. The plan amendment text mentioned splitting on `replayable` flag. Option B handles this naturally:

- Today, `physics_replayability` at `driver.rs:769-771` returns `Replayable` for every rule. Once it parses `@phase(post)` annotations, post-rules will become `NonReplayable`.
- The CG fusion analysis already splits on write-conflict. Two rules with the same replayability flag and no write conflict will fuse; rules with different replayability will write through different storage paths (per the engine's runtime contract) and therefore won't fuse anyway.
- If write-conflict-based separation proves insufficient, Patch 4 can extend the driver to allocate a separate `EventRingId` per `(ring_class, replayability)` tuple.

For iter-2, leave physics rule handling at "every rule shares the unified ring; existing fusion handles it." Verify by inspecting CG fusion diagnostic stream after the patch lands.

---

## Patches

All citations are line numbers at HEAD `65d913c7`.

### Patch 1 — Driver: unify event rings (`crates/dsl_compiler/src/cg/lower/driver.rs`)

**File:** `crates/dsl_compiler/src/cg/lower/driver.rs`
**Function:** `populate_event_kinds` (lines 283–316).

**Change shape:**

Replace per-event ring allocation with a single shared ring. Rename to make the contract explicit. Specifically:

```rust
fn populate_event_kinds(
    comp: &Compilation,
    ctx: &mut LoweringCtx<'_>,
    diagnostics: &mut Vec<LoweringError>,
) -> Vec<EventRingId> {
    // ALL event kinds share one ring — the runtime's
    // `batch_events_ring` carries every event tag interleaved.
    // Per-kind ring identity is preserved at the WGSL level via
    // the in-kernel `event.tag` decode; the dispatch layer drives
    // a single ring's tail count.
    let shared_ring = EventRingId(0);
    if let Err(e) = ctx.builder.intern_event_ring_name(
        shared_ring,
        "batch_events".to_string(),
    ) {
        diagnostics.push(LoweringError::BuilderRejected {
            error: e,
            span: dsl_ast::ast::Span::dummy(),
        });
    }

    let mut ring_ids = Vec::with_capacity(comp.events.len());
    for (i, event) in comp.events.iter().enumerate() {
        let kind_id = EventKindId(i as u32);
        ring_ids.push(shared_ring);  // every kind → same ring

        ctx.register_event_kind(event.name.clone(), kind_id);
        if let Err(e) = ctx.builder.intern_event_kind_name(
            kind_id, event.name.clone(),
        ) {
            diagnostics.push(LoweringError::BuilderRejected {
                error: e,
                span: event.span,
            });
        }
    }
    ring_ids
}
```

**Rationale:**
- Returns a `Vec<EventRingId>` of the same length and shape (`event_rings[i]` is what each handler resolution consumes), so caller ergonomics in `lower_all_views` / `lower_all_physics` (lines 673–763) are unchanged.
- The shared ring is `EventRingId(0)`, named `batch_events` on the interner — matches the runtime resident-context field `batch_events_ring` at `crates/engine_gpu_rules/src/resident_context.rs:21` after the runtime contract remap that Patch 1 of iter-1 (`501eab17`) installed. Concretely the BgSource still resolves through `BgSource::Resident("batch_events_ring".into())` at `cg/emit/kernel.rs:1203`, which the binding-source remapper (iter-1 A1, `cg/emit/program.rs::remap_binding_source_to_runtime_field`) routes to the resident field.
- Empty `Compilation`s never call this (the loop doesn't execute), so the empty-program test at `driver.rs:1037-1050` continues to pass — there's no Phase-1 op needing `EventRingId(0)`.

**LOC:** ~25 net lines (rewrite of one helper).

**Side effects on the program:**
- Every `ComputeOpKind::ViewFold { view, on_event, body }` op now carries `DispatchShape::PerEvent { source_ring: EventRingId(0) }`. The `view`, `on_event`, and body remain distinct.
- Every `ComputeOpKind::PhysicsRule { rule, on_event, body, replayable }` op (when physics lowering succeeds — see F5) carries the same source_ring.
- Every `IndirectArgs { ring }` data handle and every `DataHandle::EventRing { ring, kind }` records ring=#0.
- The `populate_plumbing` synthesizer (`cg/lower/plumbing.rs`) will dedupe: `PlumbingKind::SeedIndirectArgs { ring: EventRingId(0) }` and `PlumbingKind::DrainEvents { ring: EventRingId(0) }` will each be allocated once, not 38 times.

### Patch 2 — Kernel naming: drop event suffix on multi-handler ViewFold (`crates/dsl_compiler/src/cg/emit/kernel.rs`)

**File:** `crates/dsl_compiler/src/cg/emit/kernel.rs`
**Function:** `semantic_kernel_name` (lines 988–1001) and `single_op_kernel_name` (lines 1005–1050).

**Problem:** After Patch 1, two ViewFold ops on the same view fuse into one `KernelTopology::Fused { ops: [a, b], dispatch: PerEvent { source_ring: EventRingId(0) } }`. Today `semantic_kernel_name` calls `single_op_kernel_name` on the first op and prefixes `fused_`. For ViewFold the first-op name is `fold_threat_level_agent_attacked`. Prefixed: `fused_fold_threat_level_agent_attacked`. **Wrong** — legacy expects `fold_threat_level`.

**Change shape:** Add a special-case in `semantic_kernel_name` (or a helper before it) that detects "all body ops are ViewFold ops on the same view" and emits `fold_<view>` directly — dropping both the event suffix and the `fused_` prefix.

```rust
fn semantic_kernel_name(body_ops: &[&ComputeOp], prog: &CgProgram) -> String {
    debug_assert!(!body_ops.is_empty(), "semantic_kernel_name on empty ops");

    // Special case: a fused-or-singleton run of ViewFold ops on the
    // same view collapses to `fold_<view>` (no event suffix). This
    // matches the legacy `emit_view_fold_kernel` topology where one
    // kernel module owns all of a view's handlers; the in-kernel
    // body switches on `event.tag` to dispatch per-handler logic.
    if let Some(name) = view_fold_fused_kernel_name(body_ops, prog) {
        return name;
    }

    // PhysicsRule analogue: a fused-or-singleton run of PhysicsRule
    // ops with matching replayability collapses to `physics`
    // (replayable) or `physics_post` (non-replayable). Mirrors the
    // legacy `physics.rs` kernel module.
    if let Some(name) = physics_rule_fused_kernel_name(body_ops, prog) {
        return name;
    }

    if body_ops.len() == 1 {
        return single_op_kernel_name(&body_ops[0].kind, prog);
    }
    let first = single_op_kernel_name(&body_ops[0].kind, prog);
    if first.starts_with("fused_") { first } else { format!("fused_{first}") }
}

/// `Some("fold_<view>")` iff every op in `body_ops` is a `ViewFold`
/// referencing the same `view`. Returns `None` otherwise (mixed
/// kinds, different views, or zero ops). When the slice is a
/// singleton we still drop the event suffix — the legacy emitter
/// names a single-handler view's kernel `fold_<view>`, never
/// `fold_<view>_<event>`.
fn view_fold_fused_kernel_name(
    body_ops: &[&ComputeOp],
    prog: &CgProgram,
) -> Option<String> {
    let mut view = None;
    for op in body_ops {
        match &op.kind {
            ComputeOpKind::ViewFold { view: v, .. } => match view {
                None => view = Some(*v),
                Some(prev) if prev == *v => {}
                Some(_) => return None,
            },
            _ => return None,
        }
    }
    let view = view?;
    Some(match prog.interner.get_view_name(view) {
        Some(name) => format!("fold_{name}"),
        None => format!("fold_view_{}", view.0),
    })
}

/// `Some("physics")` for a homogeneous Replayable PhysicsRule group;
/// `Some("physics_post")` for a homogeneous NonReplayable group;
/// `None` otherwise.
fn physics_rule_fused_kernel_name(
    body_ops: &[&ComputeOp],
    _prog: &CgProgram,
) -> Option<String> {
    let mut flag = None;
    for op in body_ops {
        match &op.kind {
            ComputeOpKind::PhysicsRule { replayable, .. } => match flag {
                None => flag = Some(*replayable),
                Some(prev) if prev == *replayable => {}
                Some(_) => return None,
            },
            _ => return None,
        }
    }
    Some(match flag? {
        ReplayabilityFlag::Replayable => "physics".to_string(),
        ReplayabilityFlag::NonReplayable => "physics_post".to_string(),
    })
}
```

**Imports needed:** `use crate::cg::op::ReplayabilityFlag;` near the top of the file.

**Rationale:**
- The classifier runs BEFORE the generic `single_op_kernel_name` routing, so single-op ViewFold also flows through and drops its event suffix — fixing the legacy parity for views with only one handler too (`my_enemies`, `kin_fear`, `pack_focus`, `rally_boost`, `memory` from F3).
- The `_post` suffix is forward-looking; the current `ReplayabilityFlag::NonReplayable` variant exists in `crate::cg::op` (verify at the import path) but isn't produced by the driver yet (F5).
- Existing tests at `kernel.rs:2929,2967,3032,3033` assert `fold_threat_level_agent_attacked`-style names. They MUST be updated (see Patch 5 — test updates).

**LOC:** ~50 lines (two helpers + dispatch through them in `semantic_kernel_name`).

### Patch 3 — Kernel naming wrapper: same routing in `semantic_kernel_name_for_topology` (`crates/dsl_compiler/src/cg/emit/kernel.rs`)

**File:** Same file, function at lines 966–986.

The wrapper resolves topology → body ops → calls `semantic_kernel_name` already, so no change needed in the wrapper itself once Patch 2 lands. **Verify only.** Tests at `kernel.rs:3537,3538` (a fused-multi-handler test) and `cross_cutting.rs:944,953,966,975,978,1041,1081` (resident-context + schedule-synthesis tests) will need name updates — see Patch 5.

**LOC:** 0 net change (verification + test fallout).

### Patch 4 — Schedule classification: route fused ViewFold/PhysicsRule via existing arms (`crates/dsl_compiler/src/cg/emit/cross_cutting.rs`)

**File:** `crates/dsl_compiler/src/cg/emit/cross_cutting.rs`
**Function:** `classify_topology_for_schedule` (lines 649–689).

**Today's arms:**
- `KernelTopology::Indirect { .. }` → `DispatchOp::Indirect { args_buf: ResidentIndirectArgs }`.
- `KernelTopology::Split { op, .. }` if `topology_op_is_physics_rule` AND name == "physics" → `FixedPoint`. Else `Kernel`.
- `KernelTopology::Fused { ops, .. }` — same as Split but checks `ops.first()`.

**After Patch 1 + 2:**
- The fused-ViewFold groups land as `KernelTopology::Indirect { producer, consumers }` because they're `PerEvent`-shaped, and the existing `find_indirect_producer` (line 516) finds the (now single) `SeedIndirectArgs { ring: EventRingId(0) }` op for the unified ring. **Schedule emits `DispatchOp::Indirect { kernel: KernelId::Fold<View>, ... }` — but the legacy schedule emits `DispatchOp::Kernel(KernelId::Fold<View>)`.** Mismatch.
- Legacy SCHEDULE rationale: the fold kernel takes the shared `batch_events_tail` value via its cfg uniform's `event_count` field, NOT via `dispatch_workgroups_indirect`. That's why legacy emits `Kernel`, not `Indirect`. The runtime drives the dispatch as `dispatch_workgroups((agent_cap + 63) / 64, …)` per `crates/dsl_compiler/src/emit_view_fold_kernel.rs:352`.

**Change shape:** Add a special-case in `classify_topology_for_schedule` for `Indirect { consumers, .. }` topologies whose semantic kernel name starts with `fold_` — these emit as `Kernel(name)` per legacy. Symmetric: physics-rule indirect topologies whose name == "physics" emit as `FixedPoint { kernel, max_iter: 8 }`.

```rust
fn classify_topology_for_schedule(
    topology: &KernelTopology,
    kernel_name: &str,
    prog: &CgProgram,
) -> ScheduleEntry {
    use crate::cg::schedule::synthesis::KernelTopology;
    match topology {
        KernelTopology::Indirect { consumers, .. } => {
            // ViewFold: legacy emits direct `Kernel(...)`; the
            // consumer kernel reads `cfg.event_count` (populated
            // from the indirect-args buffer) and uses a regular
            // dispatch_workgroups call.
            if kernel_name.starts_with("fold_") {
                return ScheduleEntry::Kernel(kernel_name.to_string());
            }
            // PhysicsRule: legacy wraps in FixedPoint at
            // max_iter=8; the inner dispatch is also a regular
            // workgroup dispatch (cascade-physics A/B ring
            // alternation handled by the runtime).
            if kernel_name == "physics" || kernel_name == "physics_post" {
                if consumers.iter().any(|op|
                    topology_op_is_physics_rule(prog, *op)
                ) {
                    return ScheduleEntry::FixedPoint {
                        kernel: kernel_name.to_string(),
                        max_iter: 8,
                    };
                }
            }
            ScheduleEntry::Indirect {
                kernel: kernel_name.to_string(),
                args_buf: "ResidentIndirectArgs",
            }
        }
        KernelTopology::Split { op, .. } => {
            if topology_op_is_physics_rule(prog, *op) && kernel_name == "physics" {
                ScheduleEntry::FixedPoint { kernel: kernel_name.to_string(), max_iter: 8 }
            } else {
                ScheduleEntry::Kernel(kernel_name.to_string())
            }
        }
        KernelTopology::Fused { ops, .. } => {
            let primary_op_id = match ops.first() {
                Some(o) => *o,
                None => return ScheduleEntry::Kernel(kernel_name.to_string()),
            };
            if topology_op_is_physics_rule(prog, primary_op_id) && kernel_name == "physics" {
                ScheduleEntry::FixedPoint { kernel: kernel_name.to_string(), max_iter: 8 }
            } else {
                ScheduleEntry::Kernel(kernel_name.to_string())
            }
        }
    }
}
```

**Rationale:**
- The "starts with `fold_`" guard is sound: every fold kernel name (`fold_threat_level`, `fold_engaged_with`, …) starts with the prefix; no other kernel does. Plumbing kernels emit `drain_events_<id>`, `seed_indirect_<id>`, etc. — distinct prefixes.
- The PhysicsRule arm is forward-looking — it triggers only when the lowering pipeline produces PhysicsRule ops; today's run never sees one (F5), so the change is structurally correct but exercise-untested for now. Cover with the Patch 5 fixture-test that synthesises a PhysicsRule program.
- `topology_op_is_physics_rule` (lines 694–699) already handles out-of-range OpIds → false; the Indirect→consumers path never panics.

**LOC:** ~30 lines added in classifier; existing helpers untouched.

### Patch 5 — Test fixture updates (multiple files)

The renaming in Patch 2 invalidates several pinned-string assertions. Audit + update:

**5.1** `crates/dsl_compiler/src/cg/emit/kernel.rs`:
- Line 2929: `fold_threat_level_agent_attacked` → `fold_threat_level`.
- Line 2967: same.
- Lines 3032, 3033: `fold_threat_level_agent_attacked` and `fold_threat_level_effect_damage_applied` are produced by SEPARATE topologies (different ops); these tests build singleton topologies per op — under the new naming both become `fold_threat_level` (not distinct). Decide:
  - Option (a): Update tests to expect collision and assert that semantic_kernel_name CAN drop the event suffix; rely on KernelNameCollision detection to fail program emission when two such kernels are emitted unfused (which would happen only if topological order separates them, e.g. through an intervening op). Today the program emits one per op for views without same-view handlers; that's fine because each view is its own kernel.
  - Option (b): Construct the fixture to produce a single `Fused` topology covering both ops, then test the fused name.
  - Recommendation: (b) — replicates production behaviour after Patch 1.
- Lines 3441, 3517, 3537, 3564: assert `fold_view_<name>_handles` accessor names — these are NOT kernel names; the accessor naming at `cg/emit/cross_cutting.rs:334` uses `view` directly, not the kernel name; verify these don't change.
- Line 3538: distinct event kinds in the same view yield distinct kernel names — UPDATE: assert that they fuse to the same kernel name `fold_<view>` after Patch 1 lands.

**5.2** `crates/dsl_compiler/src/cg/emit/cross_cutting.rs`:
- Line 966: assert `FoldThreatLevelKernel` etc. — under new naming this becomes `FoldThreatLevelKernel` (from `fold_threat_level` snake → pascal `FoldThreatLevel`). **Same string** — no change. Verify.
- Lines 1097–1142 (test at `1097`): pinned schedule entry. Update if it now emits `Kernel` instead of `Indirect`.

**5.3** `crates/dsl_compiler/src/cg/lower/driver.rs`:
- The empty-Compilation tests at `1038-1050,1057-1080` should still pass (they don't reach `populate_event_kinds`'s loop body).
- The `wire_source_ring_reads` test at `1086-1166` constructs an op with `EventRingId(7)` directly via the builder, bypassing `populate_event_kinds`. Still passes (the wiring helper is independent of the driver's allocation rule).

**5.4** Lib-test count (currently 796): expect ±10 net (a few rewrites; one or two new tests).

**LOC:** ~80 lines test churn across 3 files.

### Patch 6 — Runtime contract + name-mapping smoke test (`tests/`)

**Optional but recommended.** Add a small test to `crates/dsl_compiler/tests/` (or to one of the cg integration tests) that:

1. Lowers a small fixture with two events on one view + one event on another view.
2. Asserts the resulting program has 2 ViewFold ops with shared `EventRingId(0)`.
3. Asserts the schedule has 2 entries: `Kernel(KernelId::Fold<ViewA>)` and `Kernel(KernelId::Fold<ViewB>)`.

Pins the contract that the fix delivers. **LOC:** ~80 lines for a focused fixture.

---

## Ripple-site checklist

For grep-based verification post-patch:

| Site | File:line | What to verify |
|------|-----------|----------------|
| Driver: `populate_event_kinds` | `driver.rs:283-316` | Returns Vec<EventRingId> all == EventRingId(0). |
| Fusion analysis | `fusion.rs:106-114, 320-423` | Unchanged — sees PerEvent shapes that all share ring=#0. |
| Schedule synthesis | `synthesis.rs:467-510` | `Indirect` topology fires when SeedIndirectArgs producer exists for ring=#0; `find_indirect_producer` finds the single producer. |
| Topology emit routing | `cg/emit/kernel.rs:257-260, 970-974` | Unchanged. |
| Kernel naming | `cg/emit/kernel.rs:988-1050` | New view/physics arms (Patch 2). |
| SCHEDULE classification | `cross_cutting.rs:649-689` | New fold/physics fast-paths under `Indirect` (Patch 4). |
| Resident accessor naming | `cross_cutting.rs:334` | Unchanged — uses `view` from interner, not kernel name. |
| `KernelId` enum emit | `cross_cutting.rs:?` (kernel-id index) | Smaller — fewer kernel names produced. |
| Engine_gpu match arms | `engine_gpu/src/lib.rs:826-844` | Now resolve: `KernelId::FoldThreatLevel` etc. exist again. |
| EventKindId constants emit | `emit_kernel_index.rs` (or analogue) | Each event kind still has its own `EventKindId` constant; only ring ids collapse. |
| Indirect-args wiring | `crates/engine_gpu/src/lib.rs:897`, `seed_indirect` | Single producer; existing wiring continues. |

No `KernelTopology::TaggedFusion` variant added → no missing match arms anywhere.

---

## What this iter does NOT do

- Does **not** add a new `KernelTopology::TaggedFusion` variant. Option B keeps the topology surface stable.
- Does **not** modify ViewFold WGSL bodies (legacy is stub; CG matches; F6).
- Does **not** thread `@phase(post)` parsing into the driver — `physics_replayability` still returns `Replayable` (`driver.rs:769-771`). The Patch 2 / 4 PhysicsRule arms are forward-looking surface; they exercise only when physics ops actually lower (gated by F5).
- Does **not** change the runtime engine_gpu dispatch logic (`engine_gpu/src/lib.rs:826-844`). The hardcoded `KernelId::Fold*` names already exist in legacy and will be regenerated by the CG pipeline post-Patches 1–4.
- Does **not** address `parity_with_cpu` test failures. Those fail on AST coverage gaps (For/Let/namespace setters), unrelated to this iter.

---

## Verification plan

After all patches land, run in order:

1. **Lib-test gate.** `cargo test -p dsl_compiler --lib` — should pass at 796 ±10 (test churn from Patch 5).

2. **CG canonical re-emit.** `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical`. Inspect the printed fusion-diagnostic stream:
   - Pre-fix: 9× "indirect dispatch group of 1 ops on shape per_event(ring=#N)" with N varying.
   - Post-fix: clusters where consecutive same-view handlers fuse into "indirect dispatch group of 2 ops on shape per_event(ring=#0)" or singleton groups all sharing ring=#0.

3. **engine_gpu_rules build.** `cargo build -p engine_gpu_rules` — should still pass (per iter-1 baseline; this iter doesn't touch engine_gpu_rules emit shape, only naming).

4. **engine_gpu build (B1 unblocking).** `cargo build -p engine_gpu --features gpu` — expected to **PASS** post-fix because:
   - Generated `KernelId` enum now contains `FoldThreatLevel` (not `FoldThreatLevelAgentAttacked`).
   - Generated `engine_gpu_rules::fold_threat_level::FoldThreatLevelKernel` exists.
   - All 9 fold dispatch arms at `engine_gpu/src/lib.rs:826-844` resolve.
   - PhysicsRule dispatch arm at line 845 still references `KernelId::Physics` — but no SCHEDULE entry references it (F5), so the unreachable branch at line 906-916 never fires. Still compiles because the dispatch macro pulls the variant in `match`-position; verify by inspection.

5. **gpu_pipeline_smoke (target).** `cargo test -p engine_gpu --features gpu --test gpu_pipeline_smoke`. Two outcomes possible:
   - PASS: B1 fully unblocks; iter-2 mission accomplished.
   - FAIL but at runtime (not build): documented for follow-up; the iter still achieves its B1 unblock goal at the build layer.

6. **parity_with_cpu (stretch).** `cargo test -p engine_gpu --features gpu --test parity_with_cpu`. Expected: still fails on AST-coverage gaps (Bucket C in `gpu_pipeline_smoke_status.md`); orthogonal.

7. **Working-tree restoration after verification.** Per iter-1 protocol (`gpu_pipeline_smoke_status.md:202-208`), the engine_gpu_rules/src/* files are NOT committed. Restore via `cargo run -p xtask --bin xtask -- compile-dsl` (legacy emit) and `git clean -f` of CG-only files.

---

## Risk assessment

| Risk | Severity | Reversibility | Mitigation |
|------|----------|---------------|------------|
| Patch 1 breaks an unaudited consumer that depends on per-kind ring distinctness | Medium | High (revert single function) | The 1:1 rule's only structural consumer is the well-formed gate (which uses `cycle_edge_key`-projected EventRingId for cycle detection — unification REDUCES false-cycle risk, not increases). Verify by running well_formed gate diagnostics on the post-Patch-1 program; expect no new cycle-fallback diagnostics. |
| Patch 2 name collision: a view named `physics_<x>` collides with the physics naming | Low | High (ProgramEmitError::KernelNameCollision fires at emit time, not at runtime) | The classifier check guards on `ComputeOpKind::PhysicsRule`, not on the name string; so a ViewFold kernel can never enter the `physics` arm. Reverse mismatch: a physics rule named `fold_<x>` is rejected by the classifier (it consults `ComputeOpKind::PhysicsRule`). |
| Patch 4 `starts_with("fold_")` heuristic mis-routes a non-fold kernel | Low | Medium (string-comparison drift) | All non-fold kernel names in the namespace audit at `cg/emit/kernel.rs:1006-1090` use distinct prefixes (`mask_`, `physics_`, `scoring`, `spatial_`, `pack_agents`, `unpack_agents`, `alive_pack`, `drain_events_`, `seed_indirect_`, `upload_sim_cfg`, `kick_snapshot`). Add a regression test in Patch 5 pinning the prefix invariant. |
| EventRingId collisions in IndirectArgs handle | Low | High | `DataHandle::IndirectArgs { ring: EventRingId(0) }` becomes the canonical shared-ring args handle. Multiple `SeedIndirectArgs { ring: EventRingId(0) }` synthesised in plumbing would WAW-conflict and split — but `synthesize_plumbing_ops` (`cg/lower/plumbing.rs`) deduplicates plumbing kinds, so only one is allocated. Verify by inspecting the post-patch program op count for SeedIndirectArgs ops. |
| Physics rule fusion diverges from legacy behaviour when (eventually) physics rules lower | Medium | Medium | Physics is currently fully deferred (F5). If future physics rules introduce write-conflicts with each other, the existing fusion gate splits at WAW; if they don't, they fuse — symmetric to the fold case. Cover with a synthetic fixture in Patch 5.6 (a hand-built program with two PhysicsRule ops sharing ring=#0 and disjoint writes). |
| Snapshot test drift: the iteration's name change touches every `fold_<view>_<event>` pinned snapshot | High footprint, Low severity | High (mechanical) | Patch 5 enumerates the assertion sites. The grep `fold_threat_level_agent_attacked\|fold_threat_level_effect_damage_applied` finds the bulk of them. |

**Reversibility summary:** each patch is independently revertible; together they only modify (a) the driver's ring-allocation function, (b) two helper functions in the kernel-naming layer, (c) one classifier in the schedule-synth layer, and (d) test fixtures. No new types, no new variants, no new modules.

---

## Estimated LOC + files touched

| Patch | File(s) | Net LOC |
|-------|---------|---------|
| 1 | `cg/lower/driver.rs` | ~25 |
| 2 | `cg/emit/kernel.rs` | ~50 |
| 3 | `cg/emit/kernel.rs` (verify; no code change) | 0 |
| 4 | `cg/emit/cross_cutting.rs` | ~30 |
| 5 | `cg/emit/kernel.rs`, `cross_cutting.rs`, `driver.rs` | ~80 |
| 6 (optional) | `tests/` | ~80 |
| **Total** | **3 source files + 1 test file** | **~265 LOC** |

This is **~3× smaller than the original Option A scope** (estimated 600–900 LOC for a `KernelTopology::TaggedFusion` variant + new fusion pass + ripples).

---

## Scope discoveries

1. **Fold WGSL bodies are runtime-equivalent stubs.** F6 (legacy `fold_threat_level.wgsl` is a stub touching every binding). Both legacy and CG fold kernels are no-op-equivalent today; the iter's "match legacy WGSL body shape" requirement (in the task description) collapses to "produce a stub that touches the right bindings" — already done in CG.

2. **Physics rules don't lower today.** F5. The PhysicsRule arms in Patches 2 and 4 are forward-looking surface; their actual behaviour can't be verified end-to-end until physics-rule AST coverage closes. Cover with synthetic fixtures.

3. **Engine_gpu's `KernelId::Physics` arm at line 845 references a kernel that has NO PRODUCER in the current CG schedule.** Compiles (the variant exists in the enum) but never fires. After Patch 1+2 the `KernelId::Physics` enum variant will continue to exist (assuming Patch 6's fixture-driven test produces it; otherwise the kernel-id index would only emit kernels that appear in the SCHEDULE). Investigate: confirm whether the kernel-id-index emitter only emits used kernels.

4. **The 1:1 EventKindId↔EventRingId rule was always a Phase-1 placeholder.** `driver.rs:280-282` literally says: "Cross-id confusion is structurally prevented by the typed newtypes; the parallel allocation is purely a convenience for the driver." Patch 1 retires that convenience for the documented runtime contract.

5. **Consecutive-only fusion limitation is acceptable.** Confirmed in F3: every same-view handler pair is consecutive in topological order today. The fusion pass's existing limitation (`fusion.rs:25-32`) does not bite this fix.

6. **Net plan-vs-task scope:** the task description anticipated ~800 LOC across 6 files; investigation discovered the driver-level fix achieves the same end-state at ~265 LOC across 3 files. The architectural-promise "the schedule synthesizer chooses optimal topology" is satisfied: by unifying the source-of-truth ring identity, the same fusion analysis that was always there now produces the same topology the legacy path produces.
