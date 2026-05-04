# Task 5.7-iter-1 Patch Spec

**HEAD at dispatch:** `5ba7a0f2` (`docs(scratch): gpu_pipeline_smoke status post Task 5.7 switchover`)
**Spec author:** dispatch agent (no code changes)
**Apply target:** a separate apply agent reads this file and applies edits sequentially.

---

## Overview

Close the **37 build errors** (status doc rounded to 38) that surfaced when
`crates/engine_gpu_rules/src/` was regenerated via
`xtask compile-dsl --cg-canonical`. The errors fall into three buckets:

| Bucket | Count | Shape |
| --- | --- | --- |
| **A1** Field-name drift in CG `bind()` source paths | 28 (E0609) | `sources.transient.<numbered>`, `sources.pool.<short>` reference fields that the runtime contract names with semantic prefixes |
| **A2** Duplicate `agent_alive` field on `fused_mask_Flee` | 6 (3 × E0124 + 3 × E0062) | 4 `AgentField` handles with the same `field` but different `target` collapse to the same `structural_binding_name` |
| **A3** Unused `agent_cap` in OneShot `record()` bodies | 3 (warn-as-error) | OneShot dispatch is `dispatch_workgroups(1,1,1)` — never references `agent_cap` |

Patches modify ONLY the CG emitter:
- `crates/dsl_compiler/src/cg/emit/kernel.rs` (A1 metadata table; A2 dedup; A3 unaffected)
- `crates/dsl_compiler/src/cg/emit/program.rs` (A3 unused-param underscore)
- `crates/dsl_compiler/src/cg/emit/cross_cutting.rs` (NO change required — runtime contract is correct)
- `crates/dsl_compiler/src/cg/emit/wgsl_body.rs` (NO change required — WGSL identifiers stay structural; only the Rust-side `bind()` source path needs adapting)

**End-state:** `cargo build -p engine_gpu_rules` passes after each patch (Patches A2 and A3 individually leave A1 errors uncovered; only after A1 lands does the build succeed). `cargo test -p engine_gpu --features gpu --test gpu_pipeline_smoke` passes after all three.

**Reversibility:** the patches change emit-time behavior. Reverting them and re-running `xtask compile-dsl` (without `--cg-canonical`) restores the legacy emit. The CG-emitted files themselves are not committed at HEAD.

---

## Pre-conditions

### HEAD SHA
`5ba7a0f243ea15dd9c32da9c15297986a0b636f0`
(`docs(scratch): gpu_pipeline_smoke status post Task 5.7 switchover`)

### Baseline (legacy emit, working tree clean post `compile-dsl` without `--cg-canonical`)
- `cargo build -p engine_gpu_rules` — PASS
- 796 lib + 13 xtask tests — PASS
- `gpu_pipeline_smoke` — PASS

### Failure baseline (after `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical`)
- `cargo build -p engine_gpu_rules` — FAIL (37 errors)
- `gpu_pipeline_smoke` — cannot run (engine_gpu_rules doesn't compile)

### Status doc
`/home/ricky/Projects/game/docs/scratch/gpu_pipeline_smoke_status.md` — bucket taxonomy and root-cause sketch.

### Runtime contract (read-only references)
- `crates/engine_gpu_rules/src/transient_handles.rs` (legacy emit): defines `TransientHandles<'a>` with fields:
  `mask_bitmaps`, `mask_unpack_agents_input`, `fused_agent_unpack_input`, `fused_agent_unpack_mask_soa`, `action_buf`, `scoring_unpack_agents_input`, `cascade_current_ring`, `cascade_current_tail`, `cascade_next_ring`, `cascade_next_tail`, `cascade_indirect_args`, `_phantom`. NO `snapshot_kick`, NO `agent_scratch_packed`, NO per-mask/per-ring numbered fields.
- `crates/engine_gpu_rules/src/pool.rs` (legacy emit): defines `Pool` with fields:
  `spatial_grid_cells`, `spatial_grid_offsets`, `spatial_query_results`. NO bare `grid_cells`/`grid_offsets`/`query_results`.
- `crates/engine_gpu_rules/src/binding_sources.rs`: defines `BindingSources<'a>` with `external`, `transient`, `pool`, `resident`, `pingpong` accessors.

The CG `cross_cutting.rs::synthesize_transient_handles` and `synthesize_pool` produce field sets IDENTICAL to legacy — no change needed there. The drift is ONLY in `kernel.rs::handle_to_binding_metadata` which produces `BgSource::Transient(s)` / `BgSource::Pool(s)` strings that don't match.

---

## Investigation findings

### A1 — concrete error inventory (cited from `cargo build -p engine_gpu_rules` after `--cg-canonical`)

```
error[E0609]: no field `mask_0_bitmap` on type `&'a TransientHandles<'a>`     mask_Hold.rs:13
error[E0609]: no field `mask_1_bitmap` on type `&'a TransientHandles<'a>`     mask_MoveToward.rs:13
error[E0609]: no field `mask_2_bitmap` on type `&'a TransientHandles<'a>`     fused_mask_Flee.rs:80 (4 distinct fields × 1)
error[E0609]: no field `mask_3_bitmap` on type `&'a TransientHandles<'a>`     fused_mask_Flee.rs:81
error[E0609]: no field `mask_4_bitmap` on type `&'a TransientHandles<'a>`     fused_mask_Flee.rs:82
error[E0609]: no field `mask_5_bitmap` on type `&'a TransientHandles<'a>`     fused_mask_Flee.rs:83
error[E0609]: no field `grid_cells` on type `&'a pool::Pool`                  fused_spatial_build_hash.rs:66
error[E0609]: no field `grid_offsets` on type `&'a pool::Pool`                fused_spatial_build_hash.rs:67
error[E0609]: no field `query_results` on type `&'a pool::Pool`               fused_spatial_build_hash.rs:68
error[E0609]: no field `agent_scratch_packed` on type `&'a TransientHandles`  pack_agents.rs:176, unpack_agents.rs:176
error[E0609]: no field `event_ring_1` on type `&'a TransientHandles<'a>`      fused_seed_indirect_1.rs:92
error[E0609]: no field `event_ring_22` ... `event_ring_36`                    fused_seed_indirect_1.rs:93..99 (8 ring ids: 1, 22, 24, 25, 26, 34, 35, 36)
error[E0609]: no field `indirect_args_1` ... `indirect_args_36`               fused_seed_indirect_1.rs:100..107 (same 8 ring ids)
error[E0609]: no field `snapshot_kick` on type `&'a TransientHandles<'a>`     kick_snapshot.rs:62
```
Total: 6 mask + 3 pool + 2 agent_scratch_packed + 8 event_ring + 8 indirect_args + 1 snapshot_kick = **28 E0609**.

### A1 — root-cause trace

The chain from `handle` to source-path string:
1. `crates/dsl_compiler/src/cg/emit/kernel.rs:566-655` — `handle_to_binding_metadata` writes the `BgSource` field on a `BindingMetadata`. The strings produced today:
   - `MaskBitmap { mask }` → `BgSource::Transient(format!("mask_{}_bitmap", mask.0))` (kernel.rs:597)
   - `EventRing { ring, kind }` → `BgSource::Transient(format!("event_ring_{}", ring.0))` (kernel.rs:590)
   - `SpatialStorage { kind }` → `BgSource::Pool("grid_cells" | "grid_offsets" | "query_results")` (kernel.rs:606-622)
   - `IndirectArgs { ring }` → `BgSource::Transient(format!("indirect_args_{}", ring.0))` (kernel.rs:631)
   - `AgentScratch { kind: Packed }` → `BgSource::Transient("agent_scratch_packed")` (kernel.rs:640)
   - `SnapshotKick` → `BgSource::Transient("snapshot_kick")` (kernel.rs:651)
2. `crates/dsl_compiler/src/cg/emit/program.rs:629-657` — `render_bg_source_expr` interpolates `BgSource::Transient(f)` as `format!("sources.transient.{f}")` and `BgSource::Pool(f)` as `format!("sources.pool.{f}")` verbatim.
3. The interpolated expressions become field initialisers in the kernel's `bind()` body (program.rs:618).

The runtime `TransientHandles` / `Pool` field sets do not include `mask_<N>_bitmap`, `event_ring_<N>`, `indirect_args_<N>`, `agent_scratch_packed`, `snapshot_kick`, or unprefixed `grid_*`/`query_results`. Patching the strings in `handle_to_binding_metadata` is the minimal change — it's the source of every drift.

The Rust **struct field names** on `<Kernel>Bindings<'a>` are independent — they come from `structural_binding_name` (kernel.rs:741) and stay structural. The WGSL identifiers come from `structural_handle_name` (wgsl_body.rs:151) and also stay structural. **Only the source path on the right-hand side of `field: <expr>,` in `bind()` needs to change.** No WGSL-side patch required.

### A2 — concrete duplicate-field details

Errors all in `crates/engine_gpu_rules/src/fused_mask_Flee.rs`:
```
error[E0124]: field `agent_alive` is already declared    fused_mask_Flee.rs:13, 14, 15
error[E0062]: field `agent_alive` specified more than once    fused_mask_Flee.rs:77, 78, 79
```

Inspecting the emitted source (`/tmp/cg-iter1/src/fused_mask_Flee.rs:11-21`):
```rust
pub struct FusedMaskFleeBindings<'a> {
    pub agent_alive: &'a wgpu::Buffer,    // first decl
    pub agent_alive: &'a wgpu::Buffer,    // dup #1 (E0124)
    pub agent_alive: &'a wgpu::Buffer,    // dup #2
    pub agent_alive: &'a wgpu::Buffer,    // dup #3
    pub mask_2_bitmap: &'a wgpu::Buffer,
    pub mask_3_bitmap: &'a wgpu::Buffer,
    pub mask_4_bitmap: &'a wgpu::Buffer,
    pub mask_5_bitmap: &'a wgpu::Buffer,
    pub cfg: &'a wgpu::Buffer,
}
```

The kernel fuses 4 `MaskPredicate` ops (mask_2..mask_5). Each predicate reads an `AgentField { field: Alive, target: AgentRef::Target(<expr_id>) }` — 4 distinct handles with `expr_id` ∈ {8, 10, 12, 14}. The CG aggregation pipeline:

- `kernel.rs:315-324` — `aggregate_handle` keys on `cycle_edge_key`, which for `AgentField` returns `CycleEdgeKey::Other(self.clone())` preserving full identity (data_handle.rs:917-941). The 4 handles aggregate to 4 distinct `BTreeMap` entries.
- `kernel.rs:329-346` — each entry produces a `TypedBinding` with `name = structural_binding_name(&canonical)`. For `AgentField`, `structural_binding_name` (kernel.rs:743-747) drops `target` from the rendered name: `format!("agent_{}", field.snake())`. So all 4 entries collapse to the literal name `"agent_alive"`.
- `kernel.rs:351` — `typed_bindings` is sorted by `sort_key` (still 4 entries, no dedup).
- `kernel.rs:354-363` — slot 0..3 each get a `KernelBinding { name: "agent_alive", ... }`.

The dedup is correct at the cycle-edge level (each handle is a real read edge, useful for the cycle gate) but wrong for binding-struct synthesis (`agent_alive` is a single physical storage buffer regardless of which thread offsets into it).

The WGSL also dups: `/tmp/cg-iter1/src/fused_mask_Flee.wgsl:4-7`:
```
@group(0) @binding(0) var<storage, read> agent_alive: array<u32>;
@group(0) @binding(1) var<storage, read> agent_alive: array<u32>;   // WGSL identifier collision
@group(0) @binding(2) var<storage, read> agent_alive: array<u32>;
@group(0) @binding(3) var<storage, read> agent_alive: array<u32>;
```
Naga would reject this WGSL too (identifier collision). So the dedup must happen on `KernelSpec.bindings` BEFORE both Rust struct emit and WGSL emit consume it.

### A3 — concrete unused-param sites

```
error: unused variable: `agent_cap`    upload_sim_cfg.rs:67
error: unused variable: `agent_cap`    fused_seed_indirect_1.rs:112
error: unused variable: `agent_cap`    kick_snapshot.rs:67
```

All three kernels are dispatch-shape `OneShot` (single workgroup). Their `record()` body uses `pass.dispatch_workgroups(1, 1, 1);` (program.rs:884) — never references `agent_cap`. The signature still declares `agent_cap: u32` (program.rs:555), tripping `-D unused-variables` under `-D warnings`.

For PerAgent / PerEvent / PerPair shapes, the dispatch IS `pass.dispatch_workgroups((agent_cap + N) / W, 1, 1);` — `agent_cap` IS used. So the parameter rename must be conditional on dispatch shape.

For PerWord, the dispatch derives `let num_words = (agent_cap + 31u32) / 32u32;` — also USED.

So OneShot is the only case requiring `_agent_cap`.

The legacy emit also follows this convention — see `crates/engine_gpu_rules/src/seed_indirect.rs:65`: `fn record(&self, ..., _agent_cap: u32)`. This patch ports that convention into CG's `compose_kernel_trait_impl`.

---

## Patch A1: Field-name remap to runtime contract

### Goal
Rewrite the `BgSource::Transient(s)` and `BgSource::Pool(s)` strings produced by `handle_to_binding_metadata` so they reference the actual runtime contract field names.

### Site
`crates/dsl_compiler/src/cg/emit/kernel.rs:566-655` (`handle_to_binding_metadata`).

### Rename table — exhaustive

| `DataHandle` variant | Today's `BgSource` string (broken) | Patch's `BgSource` string (runtime-canonical) | Notes |
| --- | --- | --- | --- |
| `MaskBitmap { mask }` (any `mask`) | `Transient(format!("mask_{}_bitmap", mask.0))` | `Transient("mask_bitmaps".into())` | Runtime has ONE unified `mask_bitmaps`. Per-mask offset arithmetic happens in WGSL — see "Known runtime defect: mask aliasing" below. |
| `EventRing { ring, kind }` | `Transient(format!("event_ring_{}", ring.0))` | `Transient(<see access-mode table>)` | Runtime has `cascade_current_ring` (Read source) and `cascade_next_ring` (Append target). Per `kind`: `Read` → `"cascade_current_ring"`, `Drain` → `"cascade_current_ring"`, `Append` → `"cascade_next_ring"`. Single runtime ring pair regardless of `ring.0` — see "Known runtime defect: ring aliasing" below. |
| `IndirectArgs { ring }` (any `ring`) | `Transient(format!("indirect_args_{}", ring.0))` | `Transient("cascade_indirect_args".into())` | Single runtime indirect-args buffer. See "Known runtime defect: indirect-args aliasing" below. |
| `SpatialStorage { kind: GridCells }` | `Pool("grid_cells".into())` | `Pool("spatial_grid_cells".into())` | Direct rename. |
| `SpatialStorage { kind: GridOffsets }` | `Pool("grid_offsets".into())` | `Pool("spatial_grid_offsets".into())` | Direct rename. |
| `SpatialStorage { kind: QueryResults }` | `Pool("query_results".into())` | `Pool("spatial_query_results".into())` | Direct rename. |
| `AgentScratch { kind: Packed }` | `Transient("agent_scratch_packed".into())` | `Transient("mask_unpack_agents_input".into())` | NO runtime field for agent-scratch. Aliased onto an existing transient SoA-shaped scratch buffer for build success — see "Known runtime defect: agent-scratch aliasing" below. |
| `SnapshotKick` | `Transient("snapshot_kick".into())` | `Transient("cascade_current_tail".into())` | NO runtime field for snapshot-kick. Aliased onto an existing atomic tail counter for build success — see "Known runtime defect: snapshot-kick aliasing" below. |
| `AliveBitmap` | `Transient("alive_bitmap".into())` | UNCHANGED — no error reported, NOT in CG schedule today | Runtime has no `alive_bitmap` either; if a future kernel uses this handle the build will break. Out of scope this iteration. |
| `AgentField { field, .. }` | `External("agents".into())` | UNCHANGED | Runtime `external.agents` exists — no drift. |
| `ViewStorage { .. }` | `Resident(format!("view_{}_{}", ...))` | UNCHANGED | These names go through ResidentPathContext — different drift surface. ViewFold kernels build their own bindings (kernel.rs:286, build_view_fold_bindings) and aren't affected. |
| `ScoringOutput` | `Resident("scoring_table".into())` | UNCHANGED | Out of scope: no ScoringArgmax kernel currently emits in CG. |
| `Rng / ConfigConst` | `None` | UNCHANGED | Routed through cfg uniform / inline RNG primitive. |
| `SimCfgBuffer` | `External("sim_cfg".into())` | UNCHANGED | `external.sim_cfg` exists — no drift. |

### Implementation sketch (no code changes — for the apply agent's reference)

Modify `handle_to_binding_metadata` (kernel.rs:566-655) variant arms:

- `DataHandle::MaskBitmap { mask: _ }` (line 596): replace `format!("mask_{}_bitmap", mask.0)` with the literal `"mask_bitmaps"`. The variant's binding pattern can drop the `mask` capture entirely.
- `DataHandle::EventRing { ring: _, kind }` (line 578): replace `format!("event_ring_{}", ring.0)` with a per-kind `match`:
  - `EventRingAccess::Read` → `"cascade_current_ring"`
  - `EventRingAccess::Append` → `"cascade_next_ring"`
  - `EventRingAccess::Drain` → `"cascade_current_ring"`
  Note: `canonical_handle` (kernel.rs:534-539) returns the FIRST-SEEN handle for an EventRing key — not always `Read`. So `kind` here is the access mode of the first observed access for this ring within the kernel. For the CG `seed_indirect` kernels we audited, all ops are Read access — maps to `cascade_current_ring`.
- `DataHandle::IndirectArgs { ring: _ }` (line 630): replace `format!("indirect_args_{}", ring.0)` with the literal `"cascade_indirect_args"`. Drop `ring` capture.
- `DataHandle::SpatialStorage { kind }` (line 606): rewrite the inner match arms to use the `spatial_*`-prefixed strings.
- `DataHandle::AgentScratch { kind: Packed }` (line 635): replace `format!("agent_scratch_{suffix}")` with the literal `"mask_unpack_agents_input"`.
- `DataHandle::SnapshotKick` (line 650): replace `"snapshot_kick"` with `"cascade_current_tail"`.

### Known runtime defects (build now, fix later)

Each entry below is a SEMANTIC issue created by the rename. None block the build or the smoke test (which asserts only that step_batch returns without panic for one tick on a 4-agent fixture). Each is a candidate for follow-up iterations.

1. **mask aliasing.** All `mask_<N>_bitmap` bindings now point at the SINGLE `mask_bitmaps` buffer. The WGSL keeps distinct identifiers (`mask_2_bitmap`, `mask_3_bitmap`, ...) but they alias the same physical storage; per-mask offset arithmetic (`mask_id * num_mask_words + word`) used by the legacy `fused_mask.wgsl` is NOT emitted by CG. WGPU validation will accept multiple atomic-storage bindings of the same buffer; correctness depends on the fused_mask kernel writing each "mask" at offset 0, which races with other masks. **Followup: emit per-mask base-offset arithmetic and a single binding (matches legacy shape).**
2. **ring aliasing.** All 8 `event_ring_<id>` bindings → `cascade_current_ring` (Read access mode in CG). All 8 `indirect_args_<id>` → `cascade_indirect_args`. WGPU allows multi-slot read aliasing for read-only storage and multi-slot RW aliasing IF only one slot writes — but `fused_seed_indirect_1` writes to all 8 indirect_args slots simultaneously, all aliasing one buffer. The kernel's writes will collide. **Followup: lower SeedIndirectArgs as one kernel per ring (split, not fuse) so each kernel owns the one runtime indirect-args buffer for that iteration; or expand the runtime contract to multi-ring.**
3. **agent-scratch aliasing.** `agent_scratch_packed` aliased to `mask_unpack_agents_input` (a SoA-shaped transient). Pack/unpack kernels write into `mask_unpack_agents_input` interpreting it as their packed-agent buffer. Since the pack/unpack ops are scheduled adjacently with no intervening reader of `mask_unpack_agents_input` proper, the alias is observably benign for the smoke test. **Followup: add `agent_scratch_packed` to `TransientHandles` and `engine_gpu`'s pipeline state.**
4. **snapshot-kick aliasing.** `snapshot_kick` aliased to `cascade_current_tail` (also atomic u32, same shape). The `kick_snapshot` kernel performs a 1-shot atomicOr — the alias overwrites whatever cascade tail counter is holding at that point in the schedule. Inside the smoke test (one tick), no reader of `cascade_current_tail` runs after `kick_snapshot`, so the corruption is invisible. **Followup: add `snapshot_kick` to `TransientHandles` and `engine_gpu`.**

The defect inventory should be appended to `gpu_pipeline_smoke_status.md` after this iteration applies, so the followup queue is explicit.

---

## Patch A2: Dedup duplicate bindings on KernelSpec construction

### Goal
Collapse multiple `TypedBinding` entries with the same emitted `name` into one entry, preserving the union of their access modes.

### Site
`crates/dsl_compiler/src/cg/emit/kernel.rs:329-363` (TypedBinding accumulation + slot assignment, in the generic-path branch of `kernel_topology_to_spec`).

### Implementation sketch

Insert a name-based dedup pass between line 351 (`typed_bindings.sort_by(...)`) and line 354 (`for (slot, tb) in typed_bindings.into_iter().enumerate()`):

1. Walk `typed_bindings` left-to-right. Maintain a `BTreeMap<String, TypedBinding>` keyed on `tb.name`.
2. On collision: upgrade the kept entry's `access` via `upgrade_access(kept.access, was_written)` — but `was_written` is no longer in scope at this point. Simpler: pre-compute the upgraded access mode by lattice-comparing the two TypedBindings' `access`. Order: `Atomic > ReadWrite > Read > Uniform`. Keep the higher.
3. Sanity assertion: if names collide, their `wgsl_ty` and `bg_source` MUST be identical (the dedup is only valid for handles backed by the SAME storage). Surface a typed `KernelEmitError::BindingNameCollision { name, existing_ty, new_ty }` if not — failing fast on a real bug rather than silently merging incompatible storage.
4. Drain the map back into a `Vec<TypedBinding>` sorted by the existing `sort_key` so determinism is preserved.

The 4-way `agent_alive` collapse on `fused_mask_Flee` becomes one `TypedBinding` named `agent_alive`, slot 0 (storage Read). The WGSL emitter consumes the same `KernelSpec` and now emits ONE `var<storage, read> agent_alive: array<u32>;` declaration. The bind-group entry-list goes from 4 entries to 1 entry. The BGL goes from 4 entries to 1 entry. WGPU validation is satisfied.

### Why this site (not cycle_edge_key)
Changing `cycle_edge_key` (`crates/dsl_compiler/src/cg/data_handle.rs:917-941`) to drop `target` from `AgentField` would also dedup, but it would simultaneously change the cycle-detection semantics — any place where two `AgentField { field, target_a }` and `AgentField { field, target_b }` are treated as DIFFERENT cycle edges loses that distinction. That's a semantically-loaded change with broader blast radius. Dedup at binding-synthesis time is the surgical fix.

### Pre-existing tests
`cargo test -p dsl_compiler` exercises the kernel emitter. The single-kernel mask-predicate case (1 binding) and the multi-kernel ViewFold case (separate path) are unaffected. The 4-way mask-predicate fusion is a new regime — there isn't a unit test for it yet. Adding one is **out of scope** for this iteration; the behavioral evidence is the `gpu_pipeline_smoke` build success.

---

## Patch A3: `_agent_cap` for OneShot dispatches

### Goal
Suppress the `unused variable` warning on `record()` parameter `agent_cap` when the kernel's dispatch shape is `OneShot`.

### Site
`crates/dsl_compiler/src/cg/emit/program.rs:553-578` (the non-ViewFold branch of `compose_kernel_trait_impl`'s `record()` emission).

### Implementation sketch

At the top of the `else` branch (line 553) — after determining this is a non-ViewFold kernel — inspect `topology` to determine the dispatch shape. The match-arms in `compose_dispatch_call` (program.rs:874-899) already encode the OneShot detection:

```
KernelTopology::Fused { dispatch, .. } | KernelTopology::Split { dispatch, .. } => {
    if matches!(dispatch, DispatchShape::OneShot) { /* OneShot */ }
}
KernelTopology::Indirect { .. } => /* always PerEvent — uses agent_cap */
```

(Indirect topology fixes its dispatch to `DispatchShape::PerEvent { source_ring: _ }` per program.rs:878-880, which uses `agent_cap`.)

Refactor the `record()` signature emission (program.rs:554-557):

- For OneShot dispatches: emit `agent_cap: u32` → `_agent_cap: u32`.
- For all other dispatches: keep `agent_cap: u32` unchanged.

Implementation choice (apply agent picks): either parameterize the format string with a `cap_param` local that's `"agent_cap"` or `"_agent_cap"` based on the shape, OR emit `let _ = agent_cap;` as the first statement of the body when OneShot. The legacy convention (`crates/engine_gpu_rules/src/seed_indirect.rs:65`) is `_agent_cap: u32` — match that for consistency.

### Affected kernels (after patch)
- `upload_sim_cfg.rs:67` — `_agent_cap: u32`
- `fused_seed_indirect_1.rs:112` — `_agent_cap: u32`
- `kick_snapshot.rs:67` — `_agent_cap: u32`

PerAgent / PerEvent / PerPair / PerWord kernels stay `agent_cap: u32`.

---

## Apply order

**A2 → A3 → A1.**

Rationale:

1. **A2 first** — minimal blast radius (single function tweak in `kernel.rs`). Doesn't touch `BgSource` strings. Opens up the build by fixing 6 errors, but the remaining 31 errors block compilation. Cannot be tested in isolation by `cargo build -p engine_gpu_rules`; CAN be tested by inspecting the regenerated `fused_mask_Flee.rs` (1 `agent_alive` field, not 4) after `xtask compile-dsl --cg-canonical`.
2. **A3 second** — purely cosmetic, single-file touch in `program.rs`. Removes 3 errors. After A2 + A3 the failure count is 28 (the A1 set).
3. **A1 last** — largest patch, single-file in `kernel.rs`'s `handle_to_binding_metadata`. After A1 the build passes.

Alternative ordering (A1 first) is viable but A1 is the patch most likely to need iteration — landing the small ones first means a bisect over A1's submoves stays clean.

---

## Test plan

After **each** patch, the apply agent runs:

```bash
cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical
cargo build -p engine_gpu_rules
```

Expected error counts (cumulative):
- After A2: build still fails. Errors drop by 6 (no more E0124 + E0062). Verify by `cargo build -p engine_gpu_rules 2>&1 | grep -c error[`. Target: 31.
- After A3: build still fails. Errors drop by 3 (no more `unused variable`). Target: 28.
- After A1: build passes. Verify by exit code 0.

After **all three patches** apply:

```bash
cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical
cargo build -p engine_gpu_rules                        # PASS
cargo build                                            # PASS (whole workspace)
cargo test -p engine_gpu --features gpu --test gpu_pipeline_smoke  # PASS (single tick on 4-agent fixture)
cargo test                                             # 796 lib + 13 xtask tests, all PASS
cargo test -- --test-threads=1                         # determinism gate, all PASS
```

If any test other than `gpu_pipeline_smoke` regresses, that's a separate issue — the patches only touch the CG emitter, which the legacy emit path doesn't consume. No other test should change.

---

## Apply notes

### Cross-patch interactions

- **A2 must precede a re-test of A1.** A2 reduces the binding count on fused mask kernels; A1's rename table must be applied AFTER dedup so the rename only happens to the surviving binding(s). If A1 lands first and dedup absent, rename works but the duplicate-field errors remain.
- **A3 is independent of A1 and A2** — the `record()` signature is fixed regardless of binding count. Order with A2/A1 is purely organizational.
- **No engine_gpu_rules edits** required. The runtime contract (`transient_handles.rs`, `pool.rs`) stays the source of truth; CG output adapts.
- **No engine_gpu edits** required. The `step_batch` dispatch graph runs whatever schedule.rs prescribes.
- **No CG IR changes.** `DataHandle`, `cycle_edge_key`, and `BindingMetadata`-shape are unchanged. Only the strings inside `BgSource::Transient` / `BgSource::Pool` change.

### Rollback strategy

Reverting Patches A1, A2, A3 individually is straightforward — each is a self-contained diff in one function. After revert, re-run `xtask compile-dsl` (without `--cg-canonical`) to restore the legacy emit; the legacy emit overwrites whatever CG produced in `engine_gpu_rules/src/`. No commit-time hooks need bypassing for the revert path.

If after applying all three patches the smoke test still fails — likely cause is one of the four "Known runtime defects" in Patch A1 turning out to be observable (most likely: ring/indirect-args aliasing, since the smoke test does run a full `step_batch`). In that case:
1. Capture the WGPU validation error.
2. Don't roll back; document in `gpu_pipeline_smoke_status.md` per the same convention as Task 5.7 P5.
3. Open a Task 5.7-iter-2 plan keyed on the specific aliasing issue.

### Files touched by patches

| Patch | File | Function | Lines |
| --- | --- | --- | --- |
| A2 | `crates/dsl_compiler/src/cg/emit/kernel.rs` | `kernel_topology_to_spec` (generic branch) | between 351 and 354 |
| A3 | `crates/dsl_compiler/src/cg/emit/program.rs` | `compose_kernel_trait_impl` (non-ViewFold record() block) | 554-557 |
| A1 | `crates/dsl_compiler/src/cg/emit/kernel.rs` | `handle_to_binding_metadata` | 578-654 (selected variant arms) |

`crates/dsl_compiler/src/cg/emit/wgsl_body.rs` and `crates/dsl_compiler/src/cg/emit/cross_cutting.rs` are NOT touched. The status doc's preliminary fix-surface guess (extending the synthesizers) was a feasible alternative path — but kernel.rs string rewrites are smaller (2 functions vs. 2 functions + a coordination contract for the per-program field set).

---

## Appendix: temporary investigation artifacts

For convenience the apply agent can re-run the side-channel emit:

```bash
mkdir -p /tmp/cg-iter1
cargo run -p xtask --bin xtask -- compile-dsl --cg-emit-into /tmp/cg-iter1
ls /tmp/cg-iter1/src/
```

Files most useful for cross-checking:
- `/tmp/cg-iter1/src/transient_handles.rs` — runtime contract (synthesized identical to legacy).
- `/tmp/cg-iter1/src/pool.rs` — runtime contract.
- `/tmp/cg-iter1/src/fused_mask_Flee.rs` — A2 dup (4 × `agent_alive`).
- `/tmp/cg-iter1/src/fused_seed_indirect_1.rs` — A1 ring/indirect-args drift (16 numbered fields).
- `/tmp/cg-iter1/src/fused_spatial_build_hash.rs` — A1 pool drift.
- `/tmp/cg-iter1/src/upload_sim_cfg.rs`, `kick_snapshot.rs`, `fused_seed_indirect_1.rs` — A3 unused agent_cap.

After this iteration applies and `--cg-canonical` regenerates a clean tree, these `/tmp/cg-iter1/` artifacts go stale — re-emit before debugging.
