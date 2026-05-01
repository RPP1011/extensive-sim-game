# gpu_pipeline_smoke status — post Task 5.7 switchover

**Date:** 2026-04-29
**HEAD at gate:** `67f57037` (Task 5.7 P4 commit)
**Switchover:** `cargo run -p xtask --bin xtask -- compile-dsl --cg-canonical`
re-emitted `crates/engine_gpu_rules/src/` via the CG pipeline.

## Result

`cargo test -p engine_gpu --features gpu --test gpu_pipeline_smoke` —
**FAIL** (38 build errors before any pipeline construction can run).

This is **Failure Bucket A** per the patch-spec taxonomy: not a runtime
panic but a build-time drift between CG-emitted artifacts and the
`engine_gpu` runtime's expected `TransientHandles` / `Pool` shapes
(plus a duplicate-field bug in pack/unpack kernel emission).

The legacy emit baseline (`compile-dsl` without `--cg-canonical`) is
known PASS (re-verified pre-apply at HEAD `1dd20f9c`, after removing
a stale `crates/engine/src/generated/` directory left over from a
prior commit).

## Working-tree state after switchover

```
# Modified files (8 + .schema_hash):
crates/engine_gpu_rules/.schema_hash
crates/engine_gpu_rules/src/binding_sources.rs
crates/engine_gpu_rules/src/external_buffers.rs
crates/engine_gpu_rules/src/lib.rs
crates/engine_gpu_rules/src/pingpong_context.rs
crates/engine_gpu_rules/src/pool.rs
crates/engine_gpu_rules/src/resident_context.rs
crates/engine_gpu_rules/src/schedule.rs
crates/engine_gpu_rules/src/transient_handles.rs

# Untracked (CG-emitted per-kernel files with new naming scheme):
crates/engine_gpu_rules/src/fold_engaged_with_engagement_broken.{rs,wgsl}
crates/engine_gpu_rules/src/fold_engaged_with_engagement_committed.{rs,wgsl}
crates/engine_gpu_rules/src/fold_kin_fear_fear_spread.{rs,wgsl}
crates/engine_gpu_rules/src/fold_memory_record_memory.{rs,wgsl}
crates/engine_gpu_rules/src/fold_my_enemies_agent_attacked.{rs,wgsl}
crates/engine_gpu_rules/src/fold_pack_focus_pack_assist.{rs,wgsl}
crates/engine_gpu_rules/src/fold_rally_boost_rally_call.{rs,wgsl}
crates/engine_gpu_rules/src/fold_threat_level_agent_attacked.{rs,wgsl}
crates/engine_gpu_rules/src/fold_threat_level_effect_damage_applied.{rs,wgsl}
crates/engine_gpu_rules/src/fused_mask_Flee.{rs,wgsl}
crates/engine_gpu_rules/src/fused_seed_indirect_1.{rs,wgsl}
crates/engine_gpu_rules/src/fused_spatial_build_hash.{rs,wgsl}
crates/engine_gpu_rules/src/kick_snapshot.{rs,wgsl}
crates/engine_gpu_rules/src/mask_Hold.{rs,wgsl}
crates/engine_gpu_rules/src/mask_MoveToward.{rs,wgsl}
crates/engine_gpu_rules/src/pack_agents.{rs,wgsl}
crates/engine_gpu_rules/src/unpack_agents.{rs,wgsl}
crates/engine_gpu_rules/src/upload_sim_cfg.{rs,wgsl}
```

xtask reported: **wrote 18 wgsl + 26 rust file(s) (18 kernels in
index)**. The legacy emit produced ~24 kernels using a
semantic-naming scheme (`fold_threat_level.rs`, `fused_mask.rs`,
`apply_actions.rs`, etc.); CG's naming scheme is more granular
(per-event, per-mask), which explains the larger untracked set.

## Failure buckets — root cause analysis

### Bucket A1 — CG emits numbered field names; runtime expects semantic names (32 errors)

The CG-emitted Rust modules reference fields like
`sources.transient.event_ring_22`, `sources.transient.mask_3_bitmap`,
`sources.transient.indirect_args_22`, `sources.pool.grid_cells`. The
runtime's `TransientHandles` and `Pool` structs (synthesized from
`cross_cutting.rs::synthesize_transient_handles` and
`synthesize_pool`) have semantic field names (`mask_bitmaps`,
`cascade_current_ring`, `cascade_indirect_args`, `spatial_grid_cells`,
`spatial_query_results`, etc.) — and that synthesis is independent of
program content (those struct field sets are fixed runtime contracts
hardcoded in `cross_cutting.rs`).

Specific missing fields:
- `TransientHandles`:
  - `mask_0_bitmap` … `mask_5_bitmap` (6) — runtime has unified `mask_bitmaps`.
  - `event_ring_1`, `event_ring_22`, `event_ring_24..26`, `event_ring_34..36` (8) — runtime has semantic names per ring class.
  - `indirect_args_1`, `indirect_args_22`, `indirect_args_24..26`, `indirect_args_34..36` (8) — runtime has `cascade_indirect_args`.
  - `agent_scratch_packed` (×2) — not in runtime contract.
  - `snapshot_kick` (×1) — not in runtime contract.
- `Pool`:
  - `grid_cells` — runtime has `spatial_grid_cells`.
  - `grid_offsets` — runtime has `spatial_grid_offsets`.
  - `query_results` — runtime has `spatial_query_results`.

**Root cause:** The per-kernel emit (under
`dsl_compiler::cg::emit::kernel`) generates `bind()` bodies that
walk the `KernelBinding` source paths verbatim. The legacy emitter
threaded a static rename table that mapped
`event_ring_<id>` → `cascade_current_ring` (etc.); the CG emit
hasn't ported that rename yet.

**Fix surface:** either (a) extend `cross_cutting.rs::synthesize_transient_handles` and `synthesize_pool` to emit numbered fields per program (matching what the per-kernel emit references), or (b) thread the legacy rename table into the CG per-kernel emit so `BgSource` lookups produce semantic field names. (b) is the smaller diff.

### Bucket A2 — duplicate `agent_alive` field (6 errors: 3 E0124 + 3 E0062)

Three CG-emitted modules (probably `pack_agents.rs`, `unpack_agents.rs`, etc.) declare a `KernelBindings` struct containing `agent_alive: …` twice, AND their `bind()` body initialises it twice. This is a same-source emit bug where the same `KernelBinding` survives the per-kernel struct synthesis without dedup.

**Fix surface:** dedupe `KernelBinding` entries by name in `lower_rust_bindings_struct_fields_one_per_binding` (or wherever the struct field list is materialised).

### Bucket A3 — unused `agent_cap` warnings promoted to errors (3 warnings)

Three modules (`upload_sim_cfg.rs`, `fused_seed_indirect_1.rs`, `kick_snapshot.rs`) have a `record()` body that doesn't reference `agent_cap`. The legacy emit either uses `agent_cap` in those bodies or emits `_agent_cap`. The CG emit forgets the underscore for OneShot-shaped kernels.

**Fix surface:** in the CG `compose_rust_module_file` (or its `record()`-body lowering), prefix `agent_cap` with `_` when the kernel topology is OneShot / OneShotN.

## Reversibility

The switchover is reversible per Patch 4's contract:

```bash
cargo run -p xtask --bin xtask -- compile-dsl
# (without --cg-canonical, which re-runs the legacy emit and
# overwrites the CG-emitted files in engine_gpu_rules/src/.)
```

The `--cg-canonical` flag stays in code; not setting it returns to
legacy. **The CG-emitted files are NOT being committed at this
HEAD** — the working tree is left dirty with the CG output for
follow-up review, and per spec the apply agent does not roll back
automatically.

## Per spec direction

Per `docs/scratch/task5_7_patchspec.md` §"What to do on failure":

> For all three: **DO NOT roll back automatically.** The user
> explicitly sized 5.7's end-state to "what works works, what doesn't
> gets documented for follow-up."

This file is the documentation. The 5.7 commits stand:

- Patch 1: `17c9564d` feat(dsl_compiler): CG resident-context view alias table
- Patch 2: `999b7364` fix(dsl_compiler): CG scoring_view_buffers_slice respects binding order
- Patch 3: `bb9af02f` feat(dsl_compiler): CG SCHEDULE emits FixedPoint/Indirect wrappers
- Patch 4: `67f57037` feat(xtask): --cg-canonical switchover flag

Follow-up iterations should target the three buckets above. Bucket
A2 (duplicate-field dedup) is the lowest-hanging fix; A3 (underscore
prefix) is one-line; A1 is the substantial work.
