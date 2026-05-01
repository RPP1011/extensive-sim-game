//! Cross-cutting module emission — Task 5.4.
//!
//! Synthesises the seven Rust modules that aggregate program-wide state
//! and define the runtime contract `engine_gpu` consumes:
//!
//! 1. [`synthesize_binding_sources`] — `BindingSources<'a>` aggregate of
//!    references into the four resident/pingpong/pool/transient/external
//!    container kinds.
//! 2. [`synthesize_resident_context`] — `ResidentPathContext` struct +
//!    `new()` + `scoring_view_buffers_slice()` + per-view
//!    `fold_view_<name>_handles()` accessors. The only module whose
//!    content varies with the program — every materialised view in the
//!    program's interner produces one `view_storage_<name>` field and
//!    one accessor.
//! 3. [`synthesize_external_buffers`] — `ExternalBuffers<'a>` (engine-
//!    owned references: agents, sim_cfg, ability_registry, tag_values).
//! 4. [`synthesize_transient_handles`] — `TransientHandles<'a>` (per-
//!    tick references: mask bitmaps, action buffers, cascade rings).
//! 5. [`synthesize_pingpong_context`] — `PingPongContext` (cascade A/B
//!    ring-buffer pair).
//! 6. [`synthesize_pool`] — `Pool` (shape-keyed scratch buffers reused
//!    across compatible kernels).
//! 7. [`synthesize_schedule`] — `DispatchOp` enum + `SCHEDULE` constant
//!    walked from the [`ComputeSchedule`].
//!
//! Each function takes only the inputs its content depends on
//! (`synthesize_binding_sources` is constant; `synthesize_resident_context`
//! takes the [`CgProgram`]; `synthesize_schedule` takes the
//! [`ComputeSchedule`]). All other modules are fixed runtime contracts
//! and accept no arguments.
//!
//! # Limitations
//!
//! - **Resident / external / transient / pingpong / pool aggregates are
//!   largely fixed runtime contracts.** The CG pipeline emits them with
//!   the same shape as the legacy emitters, modulo any minor cosmetic
//!   divergence (header comment phrasing, exact whitespace). The set of
//!   non-view fields on `ResidentPathContext` (alive_bitmap,
//!   scoring_table, batch_events_ring/tail, gold, …) is hard-coded —
//!   future plans that introduce a new resident-class buffer must
//!   extend this module's `RESIDENT_FIXED_FIELDS` table.
//! - **`ResidentPathContext::fold_view_<name>_handles()` returns
//!   `(primary, None, None)` for every view today.** Storage-hint-aware
//!   `Some(&self.<anchor_field>)` / `Some(&self.<ids_field>)` arms (for
//!   `PairMap @decay` and `SymmetricPairTopK` views) are a future
//!   refinement; the IR doesn't yet surface per-view storage-hint
//!   metadata from [`crate::cg::program::CgProgram`]. Single-storage
//!   views (`SlotMap`, dense `PairMap`) already return the right shape
//!   today.
//! - **`SCHEDULE` synthesis walks [`ComputeSchedule::stages`]
//!   deterministically.** The resulting `Vec<ScheduleEntry>` matches
//!   the legacy shape modulo kernel-name divergence (Task 5.2 already
//!   aligned the names). Fixed-point / indirect / gated wrappers are
//!   detected from each stage's [`KernelTopology`] variant; today every
//!   non-`Indirect` stage emits as `DispatchOp::Kernel(...)`. Encoding
//!   of `FixedPoint` / `GatedBy` is a future refinement.
//! - **`ExternalBuffers` / `TransientHandles` / `Pool` / `PingPongContext`
//!   field sets are pinned to the legacy contract.** New buffers must be
//!   added in tandem here and in `engine_gpu`'s consumer code.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::cg::op::ComputeOpKind;
use crate::cg::program::CgProgram;
use crate::cg::schedule::synthesis::ComputeSchedule;
use crate::kernel_binding_ir::snake_to_pascal;

use super::kernel::semantic_kernel_name_for_topology;

// ---------------------------------------------------------------------------
// Fixed runtime-contract field sets
// ---------------------------------------------------------------------------

/// Fixed (non-view-dependent) fields on `ResidentPathContext`, in
/// declaration order. Mirrors the legacy emitter's resident fields:
/// see `crates/engine_gpu_rules/src/resident_context.rs`.
const RESIDENT_FIXED_FIELDS: &[(&str, &str)] = &[
    ("alive_bitmap", "Per-agent alive bitmap (Resident; ceil(N/32) words)."),
    ("scoring_table", "Resident scoring table (per-action priors)."),
    ("per_slot_cooldown", "Per-agent per-slot cooldown counters (Resident; persists across ticks)."),
    ("chosen_ability_buf", "PickAbilityKernel output (Resident; consumed by ApplyActions next tick)."),
    ("gold", "Per-agent gold balance (Resident)."),
    ("standing_primary", "Standing view storage primary buffer (Resident)."),
    ("memory_primary", "Memory view storage primary buffer (Resident)."),
    ("batch_events_ring", "Batch event ring records (consumed by view folds + post-batch readback)."),
    ("batch_events_tail", "Batch event ring tail counter."),
];

/// View names the scoring kernel binds in `bind()` order. Mirrors the
/// legacy `scoring_view_binding_order` (the alphabetical-by-snake-name
/// sort of materialised non-Lazy views that pass
/// `emit_view_wgsl::classify_view`).
///
/// **Why hardcoded.** The CG IR's `prog.interner.views` includes
/// every view the lowering pass touched, in interner order — a
/// superset that includes aliased views (`standing`, `memory`,
/// `engaged_with`) the legacy emitter excludes. Threading the
/// classify-and-sort logic through the CG pipeline is a future
/// refinement (the IR would carry a `materialised_for_scoring: bool`
/// per-view flag); this hardcoded list closes the gap for Task 5.7
/// without that plumbing.
///
/// Adding a new scoring view to the DSL: append the snake-name here
/// AND ensure the view either (a) gets a `view_storage_<name>`
/// resident field (the default) or (b) is added to
/// `resident_primary_field_for_view`'s alias table. Failing to do
/// either yields an emit-time `compile_error` or a wgpu validation
/// panic at runtime — both are loud.
const SCORING_VIEW_BINDING_ORDER: &[&str] = &[
    "kin_fear",
    "my_enemies",
    "pack_focus",
    "rally_boost",
    "threat_level",
];

/// Resolve a materialised-view snake-name to the resident-field it
/// aliases for the `fold_view_<name>_handles()` accessor's primary
/// return.
///
/// Mirrors the legacy aliasing in `crates/xtask/src/compile_dsl_cmd.rs`
/// (the `match name.as_str()` near line 1436): the `standing`,
/// `memory`, and `engaged_with` views all live in two pre-existing
/// resident fields (`standing_primary`, `memory_primary`); the
/// remainder live under `view_storage_<name>`.
///
/// **Why this is a hardcoded table and not an IR property.** The legacy
/// emitter folds three concerns into one alias decision:
///   1. `standing` and `memory` use specialised storage shapes
///      (SymmetricPairTopK, PerEntityRing) the CG IR doesn't yet
///      surface as per-view storage-hint metadata.
///   2. `engaged_with`'s `Agent`/`AgentId` return-type mismatch causes
///      `classify_view` to reject it; the legacy fold kernel for it
///      is dead code (gated off by default).
///   3. Future plans (the `(a)` IR-level alias hint and `(c)` distinct
///      buffers per view in `engine_gpu`) supersede this table.
///
/// This patch matches the legacy contract byte-for-byte so the
/// switchover (Task 5.7 Patch 4) doesn't require an engine_gpu code
/// change. The table is the load-bearing alias surface until one of
/// the future plans lands.
fn resident_primary_field_for_view(view_name: &str) -> String {
    match view_name {
        "standing" => "standing_primary".to_string(),
        "memory" => "memory_primary".to_string(),
        // `engaged_with`'s legacy fold falls back to `standing_primary`
        // because `classify_view` rejects it (return-type mismatch).
        // The dispatch is gated off by default, so this never runs in
        // production today; the alias keeps emitted code compiling.
        "engaged_with" => "standing_primary".to_string(),
        other => format!("view_storage_{other}"),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Synthesise `binding_sources.rs` — the 5-field aggregate every
/// emitted kernel module imports as `crate::binding_sources::BindingSources`.
///
/// # Limitations
///
/// - **No program-derived content.** The struct shape is a fixed
///   runtime contract; the CG pipeline emits it byte-for-byte alike
///   across programs.
pub fn synthesize_binding_sources() -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by dsl_compiler — do not edit by hand.\n");
    out.push_str("// Regenerate with `cargo run --bin xtask -- compile-dsl`.\n");
    out.push('\n');
    out.push_str("use crate::resident_context::ResidentPathContext;\n");
    out.push_str("use crate::pingpong_context::PingPongContext;\n");
    out.push_str("use crate::pool::Pool;\n");
    out.push_str("use crate::transient_handles::TransientHandles;\n");
    out.push_str("use crate::external_buffers::ExternalBuffers;\n");
    out.push('\n');
    out.push_str("/// Aggregate of every container kernels can pull buffers from.\n");
    out.push_str("/// `Kernel::bind` takes `&BindingSources<'a>` and walks into the\n");
    out.push_str("/// right field; each kernel's emitted bind() body is real\n");
    out.push_str("/// generated code, not `unimplemented!()`.\n");
    out.push_str("pub struct BindingSources<'a> {\n");
    out.push_str("    pub resident:  &'a ResidentPathContext,\n");
    out.push_str("    pub pingpong:  &'a PingPongContext,\n");
    out.push_str("    pub pool:      &'a Pool,\n");
    out.push_str("    pub transient: &'a TransientHandles<'a>,\n");
    out.push_str("    pub external:  &'a ExternalBuffers<'a>,\n");
    out.push_str("}\n");
    out
}

/// Synthesise `resident_context.rs` — the persistent-buffer container
/// every kernel reads from. Walks the program for materialised views
/// (`ComputeOpKind::ViewFold` ops) and emits one `view_storage_<name>:
/// wgpu::Buffer` field + one `fold_view_<name>_handles()` accessor per
/// view. Non-view fields (alive_bitmap, scoring_table, gold, …) are
/// drawn from [`RESIDENT_FIXED_FIELDS`] — the legacy runtime contract.
///
/// # Limitations
///
/// - **Storage-hint metadata is not surfaced from the IR.** Every
///   accessor returns `(primary, None, None)`. Storage-hint-aware
///   `Some(&self.<anchor>)` / `Some(&self.<ids>)` arms (for
///   `PairMap @decay` / `SymmetricPairTopK` views) require future
///   plumbing of the lowering's per-view metadata into
///   [`CgProgram`]. Single-storage views already match the legacy
///   shape today.
/// - **`scoring_view_buffers_slice` lists every materialised view.**
///   The legacy emitter scoped the slice to `scoring_view_binding_order`;
///   the CG pipeline doesn't yet thread that order, so the slice today
///   contains every materialised view in interner order. Scoring's
///   bind() consumer is robust to ordering (it walks the slice
///   sequentially with the same order on both ends).
pub fn synthesize_resident_context(prog: &CgProgram) -> String {
    let view_names = collect_materialised_view_names(prog);

    let mut out = String::new();
    out.push_str("// GENERATED by dsl_compiler — do not edit by hand.\n");
    out.push_str("// Regenerate with `cargo run --bin xtask -- compile-dsl`.\n");
    out.push('\n');
    out.push_str("/// Resident-lifetime buffers — persist across ticks within a batch.\n");
    out.push_str("pub struct ResidentPathContext {\n");

    // Fixed runtime fields first, then per-view storage fields.
    for (name, doc) in RESIDENT_FIXED_FIELDS {
        writeln!(out, "    /// {doc}").expect("write to String never fails");
        writeln!(out, "    pub {name}: wgpu::Buffer,").expect("write to String");
    }
    for view in &view_names {
        // Skip per-view storage fields for views that alias to a
        // pre-existing resident field (see
        // `resident_primary_field_for_view`). Emitting both would
        // double-allocate the placeholder and diverge from the legacy
        // resident-context shape consumed by `engine_gpu`.
        if !resident_primary_field_for_view(view).starts_with("view_storage_") {
            continue;
        }
        writeln!(
            out,
            "    /// Resident view storage for `{view}`."
        ).expect("write to String");
        writeln!(out, "    pub view_storage_{view}: wgpu::Buffer,").expect("write to String");
    }

    out.push_str("    /// Cached slice over scoring view buffers — populated lazily by\n");
    out.push_str("    /// `scoring_view_buffers_slice` and re-used across `bind()` calls.\n");
    out.push_str("    /// `OnceLock` keeps it `Sync`-safe without runtime locking.\n");
    out.push_str("    scoring_view_buffers_cache: std::sync::OnceLock<Vec<&'static wgpu::Buffer>>,\n");
    out.push_str("}\n\n");

    // impl block.
    out.push_str("impl ResidentPathContext {\n");
    out.push_str("    /// Allocate every Resident-class buffer up-front. agent_cap is\n");
    out.push_str("    /// the maximum agent capacity across the batch — caps that grow\n");
    out.push_str("    /// at runtime force a context rebuild (the existing GpuBackend\n");
    out.push_str("    /// resident-rebuild path).\n");
    out.push_str("    pub fn new(_device: &wgpu::Device, _agent_cap: u32) -> Self {\n");
    out.push_str("        Self {\n");
    for (name, _doc) in RESIDENT_FIXED_FIELDS {
        emit_buffer_init(&mut out, name);
    }
    for view in &view_names {
        // Aliased views (standing, memory, engaged_with) get their
        // buffer from `RESIDENT_FIXED_FIELDS` — no per-view init.
        // Mirrors the field-emit guard above.
        let field = resident_primary_field_for_view(view);
        if !field.starts_with("view_storage_") {
            continue;
        }
        emit_buffer_init(&mut out, &field);
    }
    out.push_str("            scoring_view_buffers_cache: std::sync::OnceLock::new(),\n");
    out.push_str("        }\n");
    out.push_str("    }\n\n");

    // scoring_view_buffers_slice() helper.
    out.push_str("    /// Slice of per-view scoring buffers in `scoring_view_binding_order`.\n");
    out.push_str("    /// Used by `ScoringKernel::bind()`.\n");
    out.push_str("    pub fn scoring_view_buffers_slice<'a>(&'a self) -> &'a [&'a wgpu::Buffer] {\n");

    // Compute the intersection of `SCORING_VIEW_BINDING_ORDER` and the
    // views actually materialised in `prog`. The legacy emitter walks
    // `combined.views` and filters by `classify_view`; here we approximate
    // by intersecting the hardcoded order with the IR's
    // `view_names` set so the slice never references a view the program
    // didn't materialise.
    let materialised_set: std::collections::BTreeSet<&str> =
        view_names.iter().map(String::as_str).collect();
    let scoring_views: Vec<&str> = SCORING_VIEW_BINDING_ORDER
        .iter()
        .copied()
        .filter(|n| materialised_set.contains(n))
        .collect();

    if scoring_views.is_empty() {
        out.push_str("        &[]\n");
    } else {
        out.push_str("        let v = self.scoring_view_buffers_cache.get_or_init(|| {\n");
        out.push_str("            // SAFETY: the &wgpu::Buffer references inside `Self` live as long as\n");
        out.push_str("            // `Self` does. The OnceLock is dropped together with `Self`,\n");
        out.push_str("            // so the 'static cast is sound for as long as the cache exists.\n");
        out.push_str("            let raw: Vec<&'static wgpu::Buffer> = vec![\n");
        for view in &scoring_views {
            let primary_field = resident_primary_field_for_view(view);
            writeln!(
                out,
                "                unsafe {{ &*((&self.{primary_field}) as *const wgpu::Buffer) }},"
            ).expect("write to String");
        }
        out.push_str("            ];\n");
        out.push_str("            raw\n");
        out.push_str("        });\n");
        out.push_str("        // Re-borrow at the caller's lifetime.\n");
        out.push_str("        // Vec<&'static T> coerces to &[&T] here.\n");
        out.push_str("        unsafe { std::mem::transmute::<&[&'static wgpu::Buffer], &'a [&'a wgpu::Buffer]>(v.as_slice()) }\n");
    }
    out.push_str("    }\n");

    // Per-view fold-handle accessors.
    for view in &view_names {
        let pascal = snake_to_pascal(view);
        out.push('\n');
        writeln!(
            out,
            "    /// Returns (primary, anchor_opt, ids_opt) for the `{view}` fold kernel."
        ).expect("write to String");
        writeln!(
            out,
            "    /// Used by `Fold{pascal}Kernel::bind()`."
        ).expect("write to String");
        writeln!(
            out,
            "    pub fn fold_view_{view}_handles<'a>(&'a self) -> (&'a wgpu::Buffer, Option<&'a wgpu::Buffer>, Option<&'a wgpu::Buffer>) {{"
        ).expect("write to String");
        writeln!(
            out,
            "        // Storage-hint-aware Some(...) arms are a future refinement; see"
        ).expect("write to String");
        writeln!(
            out,
            "        // the module-level `# Limitations` on `cross_cutting.rs`."
        ).expect("write to String");
        let primary_field = resident_primary_field_for_view(view);
        writeln!(
            out,
            "        (&self.{primary_field}, None, None)"
        ).expect("write to String");
        out.push_str("    }\n");
    }

    out.push_str("}\n");
    out
}

/// Synthesise `external_buffers.rs` — engine-owned reference handles.
///
/// # Limitations
///
/// - **Field set is pinned to the legacy runtime contract** (agents,
///   sim_cfg, ability_registry, tag_values). Future buffers must be
///   added in tandem with `engine_gpu`'s consumer code.
pub fn synthesize_external_buffers() -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by dsl_compiler — do not edit by hand.\n");
    out.push_str("// Regenerate with `cargo run --bin xtask -- compile-dsl`.\n");
    out.push('\n');
    out.push_str("/// External-lifetime buffer references — engine-owned (agent SoA, sim_cfg, registries).\n");
    out.push_str("pub struct ExternalBuffers<'a> {\n");
    out.push_str("    /// Agent SoA buffer (engine-owned).\n");
    out.push_str("    pub agents: &'a wgpu::Buffer,\n");
    out.push_str("    /// SimCfg uniform/storage buffer (engine-owned).\n");
    out.push_str("    pub sim_cfg: &'a wgpu::Buffer,\n");
    out.push_str("    /// AbilityRegistry buffer (engine-owned).\n");
    out.push_str("    pub ability_registry: &'a wgpu::Buffer,\n");
    out.push_str("    /// Per-tag value table (engine-owned).\n");
    out.push_str("    pub tag_values: &'a wgpu::Buffer,\n");
    out.push_str("    pub _phantom: std::marker::PhantomData<&'a ()>,\n");
    out.push_str("}\n");
    out
}

/// Synthesise `transient_handles.rs` — per-tick scratch buffer
/// references populated each tick by `engine_gpu`.
///
/// # Limitations
///
/// - **Field set is pinned to the legacy runtime contract.** New
///   transient scratch buffers must be added in tandem with
///   `engine_gpu`'s consumer code.
pub fn synthesize_transient_handles() -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by dsl_compiler — do not edit by hand.\n");
    out.push_str("// Regenerate with `cargo run --bin xtask -- compile-dsl`.\n");
    out.push('\n');
    out.push_str("/// Transient-lifetime buffer references — populated each tick by engine_gpu.\n");
    out.push_str("pub struct TransientHandles<'a> {\n");
    out.push_str("    /// FusedMaskKernel output: ceil(N/32) words × N masks; recycled per tick.\n");
    out.push_str("    pub mask_bitmaps: &'a wgpu::Buffer,\n");
    out.push_str("    /// MaskUnpackKernel scratch: source SoA before unpack.\n");
    out.push_str("    pub mask_unpack_agents_input: &'a wgpu::Buffer,\n");
    out.push_str("    /// FusedAgentUnpackKernel scratch: source pre-unpack agent buffer.\n");
    out.push_str("    pub fused_agent_unpack_input: &'a wgpu::Buffer,\n");
    out.push_str("    /// FusedAgentUnpackKernel scratch: derived mask SoA.\n");
    out.push_str("    pub fused_agent_unpack_mask_soa: &'a wgpu::Buffer,\n");
    out.push_str("    /// ScoringKernel output (action-per-agent buffer).\n");
    out.push_str("    pub action_buf: &'a wgpu::Buffer,\n");
    out.push_str("    /// ScoringUnpackKernel scratch.\n");
    out.push_str("    pub scoring_unpack_agents_input: &'a wgpu::Buffer,\n");
    out.push_str("    /// Cascade producer-ring records for the current iteration.\n");
    out.push_str("    pub cascade_current_ring: &'a wgpu::Buffer,\n");
    out.push_str("    /// Cascade producer-ring tail counter.\n");
    out.push_str("    pub cascade_current_tail: &'a wgpu::Buffer,\n");
    out.push_str("    /// Cascade consumer-ring records (next iteration).\n");
    out.push_str("    pub cascade_next_ring: &'a wgpu::Buffer,\n");
    out.push_str("    /// Cascade consumer-ring tail counter (atomic).\n");
    out.push_str("    pub cascade_next_tail: &'a wgpu::Buffer,\n");
    out.push_str("    /// dispatch_indirect args for the next iteration.\n");
    out.push_str("    pub cascade_indirect_args: &'a wgpu::Buffer,\n");
    out.push_str("    pub _phantom: std::marker::PhantomData<&'a ()>,\n");
    out.push_str("}\n");
    out
}

/// Synthesise `pingpong_context.rs` — cascade-physics A/B ring pair.
///
/// # Limitations
///
/// - **Field set is pinned to the cascade A/B-ring contract** (record
///   buffers + tail counters). Multi-ring cascade variants would need
///   additional fields here.
pub fn synthesize_pingpong_context() -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by dsl_compiler — do not edit by hand.\n");
    out.push_str("// Regenerate with `cargo run --bin xtask -- compile-dsl`.\n");
    out.push('\n');
    out.push_str("/// PingPong-lifetime ring buffers (cascade A/B).\n");
    out.push_str("pub struct PingPongContext {\n");
    out.push_str("    /// Cascade-physics A-ring event records (write side at iter 0).\n");
    out.push_str("    pub events_a_records: wgpu::Buffer,\n");
    out.push_str("    /// Cascade-physics A-ring tail (atomic counter).\n");
    out.push_str("    pub events_a_tail: wgpu::Buffer,\n");
    out.push_str("    /// Cascade-physics B-ring event records.\n");
    out.push_str("    pub events_b_records: wgpu::Buffer,\n");
    out.push_str("    /// Cascade-physics B-ring tail.\n");
    out.push_str("    pub events_b_tail: wgpu::Buffer,\n");
    out.push_str("}\n\n");
    out.push_str("impl PingPongContext {\n");
    out.push_str("    pub fn new(_device: &wgpu::Device) -> Self {\n");
    out.push_str("        Self {\n");
    for field in [
        "events_a_records",
        "events_a_tail",
        "events_b_records",
        "events_b_tail",
    ] {
        out.push_str(&format!(
            "            {field}: _device.create_buffer(&wgpu::BufferDescriptor {{\n"
        ));
        out.push_str(&format!(
            "                label: Some(\"engine_gpu_rules::pingpong::{field}\"),\n"
        ));
        out.push_str("                size: 4,\n");
        out.push_str("                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,\n");
        out.push_str("                mapped_at_creation: false,\n");
        out.push_str("            }),\n");
    }
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

/// Synthesise `pool.rs` — shape-keyed scratch buffers.
///
/// # Limitations
///
/// - **Field set is pinned to the spatial-hash + query-result
///   contract.** Future shape-keyed scratch buffers must be added in
///   tandem with the consumer kernels.
pub fn synthesize_pool() -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by dsl_compiler — do not edit by hand.\n");
    out.push_str("// Regenerate with `cargo run --bin xtask -- compile-dsl`.\n");
    out.push('\n');
    out.push_str("/// Pooled-lifetime buffers — shape-keyed; reused across compatible kernels.\n");
    out.push_str("pub struct Pool {\n");
    out.push_str("    /// Spatial-hash cell-index buffer (Pooled).\n");
    out.push_str("    pub spatial_grid_cells: wgpu::Buffer,\n");
    out.push_str("    /// Spatial-hash cell-offsets buffer (Pooled).\n");
    out.push_str("    pub spatial_grid_offsets: wgpu::Buffer,\n");
    out.push_str("    /// Per-query result buffer (Pooled).\n");
    out.push_str("    pub spatial_query_results: wgpu::Buffer,\n");
    out.push_str("}\n\n");
    out.push_str("impl Pool {\n");
    out.push_str("    pub fn new(_device: &wgpu::Device) -> Self {\n");
    out.push_str("        Self {\n");
    for field in [
        "spatial_grid_cells",
        "spatial_grid_offsets",
        "spatial_query_results",
    ] {
        out.push_str(&format!(
            "            {field}: _device.create_buffer(&wgpu::BufferDescriptor {{\n"
        ));
        out.push_str(&format!(
            "                label: Some(\"engine_gpu_rules::pool::{field}\"),\n"
        ));
        out.push_str("                size: 4,\n");
        out.push_str("                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,\n");
        out.push_str("                mapped_at_creation: false,\n");
        out.push_str("            }),\n");
    }
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n");
    out
}

/// Synthesise `schedule.rs` — `DispatchOp` enum + `SCHEDULE` constant
/// derived deterministically from the [`ComputeSchedule`] stages.
///
/// Each stage's [`KernelTopology`] resolves to one or more dispatch-op
/// entries. Today every topology lowers to `DispatchOp::Kernel(...)`;
/// `FixedPoint` / `Indirect` / `GatedBy` wrappers are a future
/// refinement (see `# Limitations`).
///
/// # Limitations
///
/// - **All entries today emit as `DispatchOp::Kernel`.** Detecting
///   fixed-point loops requires walking the schedule's topology
///   classifications which the CG pipeline doesn't yet thread through;
///   `Indirect` topologies emit one entry per consumer (matching the
///   legacy schedule structure).
/// - **Kernel naming routes through Task 5.2's
///   [`semantic_kernel_name_for_topology`].** Mismatches between the
///   schedule entries and the per-kernel module file names are
///   structurally impossible because both call sites use the same
///   helper.
pub fn synthesize_schedule(schedule: &ComputeSchedule, prog: &CgProgram) -> String {
    let mut out = String::new();
    out.push_str("// GENERATED by dsl_compiler — do not edit by hand.\n");
    out.push_str("// Regenerate with `cargo run --bin xtask -- compile-dsl`.\n");
    out.push('\n');
    out.push_str("use crate::{KernelKind, BufferRef};\n");
    out.push('\n');
    out.push_str("#[derive(Copy, Clone, Debug)]\n");
    out.push_str("pub enum DispatchOp {\n");
    out.push_str("    Kernel(KernelKind),\n");
    out.push_str("    FixedPoint { kernel: KernelKind, max_iter: u32 },\n");
    out.push_str("    Indirect { kernel: KernelKind, args_buf: BufferRef },\n");
    out.push_str("    GatedBy { kernel: KernelKind, gate: BufferRef },\n");
    out.push_str("}\n");
    out.push('\n');

    // Walk stages to collect kernel names in schedule order.
    let mut entries: Vec<String> = Vec::new();
    for stage in &schedule.stages {
        for topology in &stage.kernels {
            // Use the same naming helper the per-kernel emit uses, so
            // schedule entries reference the actual emitted module
            // names. Errors here are unreachable in practice (the
            // per-kernel emit would have already failed); we
            // defensively skip rather than panic.
            match semantic_kernel_name_for_topology(topology, prog) {
                Some(name) => entries.push(name),
                None => continue,
            }
        }
    }

    if entries.is_empty() {
        out.push_str("pub const SCHEDULE: &[DispatchOp] = &[];\n");
        return out;
    }

    out.push_str("pub const SCHEDULE: &[DispatchOp] = &[\n");
    for name in &entries {
        let pascal = snake_to_pascal(name);
        writeln!(out, "    DispatchOp::Kernel(KernelKind::{pascal}),")
            .expect("write to String");
    }
    out.push_str("];\n");
    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Walk every [`ComputeOpKind::ViewFold`] op in `prog.ops` and collect
/// the unique view-name strings (in interner order). Views without an
/// interned name fall back to `view_<id>`. Used by
/// [`synthesize_resident_context`] to drive per-view field + accessor
/// emission.
fn collect_materialised_view_names(prog: &CgProgram) -> Vec<String> {
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for op in &prog.ops {
        if let ComputeOpKind::ViewFold { view, .. } = &op.kind {
            seen.insert(view.0);
        }
    }
    let mut out: Vec<String> = Vec::with_capacity(seen.len());
    for id in seen {
        let name = match prog.interner.views.get(&id) {
            Some(n) => n.clone(),
            None => format!("view_{id}"),
        };
        out.push(name);
    }
    out
}

/// Append a `field: device.create_buffer(...)` initialiser to `out`.
/// Single source of truth for the 256-byte placeholder descriptor used
/// by every Resident-class buffer init in [`synthesize_resident_context`].
fn emit_buffer_init(out: &mut String, field: &str) {
    out.push_str(&format!(
        "            {field}: _device.create_buffer(&wgpu::BufferDescriptor {{\n"
    ));
    out.push_str(&format!(
        "                label: Some(\"engine_gpu_rules::resident::{field}\"),\n"
    ));
    out.push_str("                size: 256,\n");
    out.push_str("                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,\n");
    out.push_str("                mapped_at_creation: false,\n");
    out.push_str("            }),\n");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{CgExprId, DataHandle, EventRingId, ViewId, ViewStorageSlot};
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{ComputeOp, EventKindId, OpId, Span};
    use crate::cg::program::CgProgram;
    use crate::cg::schedule::synthesis::{ComputeStage, KernelTopology};
    use crate::cg::stmt::{CgStmt, CgStmtId, CgStmtList, CgStmtListId};

    fn one_view_fold_program(view_id: u32, view_name: &str) -> (CgProgram, OpId) {
        let mut prog = CgProgram::default();
        prog.interner.views.insert(view_id, view_name.to_string());
        prog.interner
            .event_kinds
            .insert(7, "AgentAttacked".to_string());
        let one = CgExpr::Lit(LitValue::F32(1.0));
        let one_id = CgExprId(prog.exprs.len() as u32);
        prog.exprs.push(one);
        let assign = CgStmt::Assign {
            target: DataHandle::ViewStorage {
                view: ViewId(view_id),
                slot: ViewStorageSlot::Primary,
            },
            value: one_id,
        };
        let assign_id = CgStmtId(prog.stmts.len() as u32);
        prog.stmts.push(assign);
        let list = CgStmtList {
            stmts: vec![assign_id],
        };
        let list_id = CgStmtListId(prog.stmt_lists.len() as u32);
        prog.stmt_lists.push(list);
        let kind = ComputeOpKind::ViewFold {
            view: ViewId(view_id),
            on_event: EventKindId(7),
            body: list_id,
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = OpId(prog.ops.len() as u32);
        prog.ops.push(op);
        (prog, op_id)
    }

    /// Build a CgProgram materialising one ViewFold op per `(view_id, view_name)`
    /// in `views`. Used by Patch 2 tests to verify SCORING_VIEW_BINDING_ORDER
    /// drives the slice contents.
    fn multi_view_fold_program(views: &[(u32, &str)]) -> CgProgram {
        let mut prog = CgProgram::default();
        prog.interner
            .event_kinds
            .insert(7, "AgentAttacked".to_string());
        for (view_id, view_name) in views {
            prog.interner.views.insert(*view_id, view_name.to_string());
            let one = CgExpr::Lit(LitValue::F32(1.0));
            let one_id = CgExprId(prog.exprs.len() as u32);
            prog.exprs.push(one);
            let assign = CgStmt::Assign {
                target: DataHandle::ViewStorage {
                    view: ViewId(*view_id),
                    slot: ViewStorageSlot::Primary,
                },
                value: one_id,
            };
            let assign_id = CgStmtId(prog.stmts.len() as u32);
            prog.stmts.push(assign);
            let list = CgStmtList {
                stmts: vec![assign_id],
            };
            let list_id = CgStmtListId(prog.stmt_lists.len() as u32);
            prog.stmt_lists.push(list);
            let kind = ComputeOpKind::ViewFold {
                view: ViewId(*view_id),
                on_event: EventKindId(7),
                body: list_id,
            };
            let op = ComputeOp::new(
                OpId(prog.ops.len() as u32),
                kind,
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
                &prog,
                &prog,
                &prog,
            );
            prog.ops.push(op);
        }
        prog
    }

    // ---- 1. binding_sources ----

    #[test]
    fn binding_sources_emits_aggregate_struct() {
        let src = synthesize_binding_sources();
        assert!(src.starts_with("// GENERATED"), "header: {src}");
        assert!(src.contains("pub struct BindingSources<'a>"), "{src}");
        assert!(src.contains("pub resident:  &'a ResidentPathContext"), "{src}");
        assert!(src.contains("pub pingpong:  &'a PingPongContext"), "{src}");
        assert!(src.contains("pub pool:      &'a Pool"), "{src}");
        assert!(src.contains("pub transient: &'a TransientHandles"), "{src}");
        assert!(src.contains("pub external:  &'a ExternalBuffers"), "{src}");
    }

    // ---- 2. external_buffers ----

    #[test]
    fn external_buffers_emits_engine_owned_handles() {
        let src = synthesize_external_buffers();
        assert!(src.contains("pub struct ExternalBuffers<'a>"), "{src}");
        assert!(src.contains("pub agents: &'a wgpu::Buffer"), "{src}");
        assert!(src.contains("pub sim_cfg: &'a wgpu::Buffer"), "{src}");
        assert!(src.contains("pub ability_registry: &'a wgpu::Buffer"), "{src}");
        assert!(src.contains("pub tag_values: &'a wgpu::Buffer"), "{src}");
        assert!(src.contains("_phantom"), "{src}");
    }

    // ---- 3. transient_handles ----

    #[test]
    fn transient_handles_emits_per_tick_scratch_refs() {
        let src = synthesize_transient_handles();
        assert!(src.contains("pub struct TransientHandles<'a>"), "{src}");
        assert!(src.contains("pub mask_bitmaps:"), "{src}");
        assert!(src.contains("pub action_buf:"), "{src}");
        assert!(src.contains("pub cascade_current_ring:"), "{src}");
        assert!(src.contains("pub cascade_indirect_args:"), "{src}");
    }

    // ---- 4. pingpong_context ----

    #[test]
    fn pingpong_context_emits_ab_ring_struct_and_new() {
        let src = synthesize_pingpong_context();
        assert!(src.contains("pub struct PingPongContext"), "{src}");
        assert!(src.contains("pub events_a_records: wgpu::Buffer"), "{src}");
        assert!(src.contains("pub events_b_tail: wgpu::Buffer"), "{src}");
        assert!(src.contains("impl PingPongContext"), "{src}");
        assert!(src.contains("pub fn new(_device: &wgpu::Device) -> Self"), "{src}");
        assert!(src.contains("create_buffer"), "{src}");
    }

    // ---- 5. pool ----

    #[test]
    fn pool_emits_struct_and_new() {
        let src = synthesize_pool();
        assert!(src.contains("pub struct Pool"), "{src}");
        assert!(src.contains("pub spatial_grid_cells: wgpu::Buffer"), "{src}");
        assert!(src.contains("pub spatial_grid_offsets: wgpu::Buffer"), "{src}");
        assert!(src.contains("pub spatial_query_results: wgpu::Buffer"), "{src}");
        assert!(src.contains("impl Pool"), "{src}");
        assert!(src.contains("pub fn new"), "{src}");
    }

    // ---- 6. resident_context — fixed fields ----

    #[test]
    fn resident_context_emits_fixed_fields_with_no_views() {
        let prog = CgProgram::default();
        let src = synthesize_resident_context(&prog);
        assert!(src.contains("pub struct ResidentPathContext"), "{src}");
        // Spot-check fixed fields.
        for (field, _) in RESIDENT_FIXED_FIELDS {
            assert!(
                src.contains(&format!("pub {field}: wgpu::Buffer")),
                "field `{field}` missing from: {src}"
            );
        }
        // Empty view set → empty scoring slice.
        assert!(src.contains("scoring_view_buffers_slice"), "{src}");
        assert!(src.contains("        &[]"), "empty slice should be emitted: {src}");
        // No view_storage_* fields.
        assert!(
            !src.contains("view_storage_"),
            "no views → no view_storage fields: {src}"
        );
    }

    // ---- 7. resident_context — per-view storage + accessors ----

    #[test]
    fn resident_context_emits_one_view_storage_field_per_materialised_view() {
        let (prog, _) = one_view_fold_program(3, "threat_level");
        let src = synthesize_resident_context(&prog);
        assert!(
            src.contains("pub view_storage_threat_level: wgpu::Buffer"),
            "view storage field missing: {src}"
        );
    }

    #[test]
    fn resident_context_emits_fold_view_handles_accessor_per_view() {
        let (prog, _) = one_view_fold_program(3, "threat_level");
        let src = synthesize_resident_context(&prog);
        assert!(
            src.contains("pub fn fold_view_threat_level_handles<'a>(&'a self)"),
            "accessor missing: {src}"
        );
        assert!(
            src.contains("(&self.view_storage_threat_level, None, None)"),
            "accessor body should return placeholder triple: {src}"
        );
        // Rustdoc cross-link to the consumer kernel.
        assert!(
            src.contains("FoldThreatLevelKernel"),
            "doc cross-link missing: {src}"
        );
    }

    // ---- 7b. resident_context — view aliasing (Task 5.7 P1) ----

    #[test]
    fn resident_context_aliases_standing_to_standing_primary() {
        let (prog, _) = one_view_fold_program(11, "standing");
        let src = synthesize_resident_context(&prog);
        assert!(
            src.contains("pub fn fold_view_standing_handles<'a>"),
            "accessor must exist: {src}"
        );
        assert!(
            src.contains("(&self.standing_primary, None, None)"),
            "standing must alias to standing_primary: {src}"
        );
        assert!(
            !src.contains("pub view_storage_standing: wgpu::Buffer"),
            "standing must NOT get its own view_storage field: {src}"
        );
    }

    #[test]
    fn resident_context_aliases_memory_to_memory_primary() {
        let (prog, _) = one_view_fold_program(12, "memory");
        let src = synthesize_resident_context(&prog);
        assert!(
            src.contains("(&self.memory_primary, None, None)"),
            "memory must alias to memory_primary: {src}"
        );
        assert!(
            !src.contains("pub view_storage_memory: wgpu::Buffer"),
            "memory must NOT get its own view_storage field: {src}"
        );
    }

    #[test]
    fn resident_context_aliases_engaged_with_to_standing_primary() {
        let (prog, _) = one_view_fold_program(13, "engaged_with");
        let src = synthesize_resident_context(&prog);
        assert!(
            src.contains("(&self.standing_primary, None, None)"),
            "engaged_with must alias to standing_primary: {src}"
        );
        assert!(
            !src.contains("pub view_storage_engaged_with: wgpu::Buffer"),
            "engaged_with must NOT get its own view_storage field: {src}"
        );
    }

    #[test]
    fn resident_context_non_aliased_view_keeps_view_storage_field() {
        let (prog, _) = one_view_fold_program(14, "threat_level");
        let src = synthesize_resident_context(&prog);
        assert!(
            src.contains("pub view_storage_threat_level: wgpu::Buffer"),
            "non-aliased view must keep its view_storage_<name> field: {src}"
        );
        assert!(
            src.contains("(&self.view_storage_threat_level, None, None)"),
            "non-aliased view's accessor body unchanged: {src}"
        );
    }

    // ---- 7c. resident_context — scoring_view_buffers_slice (Task 5.7 P2) ----

    #[test]
    fn scoring_view_buffers_slice_uses_binding_order_subset() {
        // Fixture: program materialises kin_fear, threat_level, and a
        // non-scoring view some_other. Slice must list kin_fear before
        // threat_level (binding order) and skip some_other entirely (not
        // in SCORING_VIEW_BINDING_ORDER).
        let prog = multi_view_fold_program(&[
            (1, "kin_fear"),
            (2, "threat_level"),
            (99, "some_other"),
        ]);

        let src = synthesize_resident_context(&prog);

        // Locate the scoring slice body.
        let slice_start = src
            .find("pub fn scoring_view_buffers_slice")
            .expect("slice fn must exist");
        let after_slice = &src[slice_start..];
        // The body ends at the closing `}` of the function — find the
        // next `\n    }\n` boundary.
        let body_end = after_slice
            .find("\n    }\n")
            .expect("slice fn must have a closing brace");
        let body = &after_slice[..body_end];

        let kin_pos = body
            .find("&self.view_storage_kin_fear")
            .expect("kin_fear must appear in slice body");
        let threat_pos = body
            .find("&self.view_storage_threat_level")
            .expect("threat_level must appear in slice body");
        assert!(
            kin_pos < threat_pos,
            "kin_fear must precede threat_level: {body}"
        );
        assert!(
            !body.contains("&self.view_storage_some_other"),
            "some_other must NOT appear in slice body: {body}"
        );
    }

    #[test]
    fn scoring_view_buffers_slice_emits_empty_when_no_scoring_views_materialised() {
        // Only an aliased view materialised — slice should be empty
        // (standing is not in SCORING_VIEW_BINDING_ORDER).
        let (prog, _) = one_view_fold_program(11, "standing");
        let src = synthesize_resident_context(&prog);
        let slice_start = src
            .find("pub fn scoring_view_buffers_slice")
            .expect("slice fn must exist");
        let after = &src[slice_start..];
        assert!(
            after.contains("        &[]"),
            "empty slice must be emitted when only aliased views materialised: {src}"
        );
    }

    // ---- 8. schedule synthesis ----

    #[test]
    fn schedule_synthesises_dispatch_op_enum_and_const_from_stages() {
        let (prog, op_id) = one_view_fold_program(3, "threat_level");
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let schedule = ComputeSchedule {
            stages: vec![ComputeStage {
                kernels: vec![topology],
            }],
        };
        let src = synthesize_schedule(&schedule, &prog);
        assert!(src.contains("pub enum DispatchOp"), "{src}");
        assert!(src.contains("Kernel(KernelKind),"), "{src}");
        assert!(src.contains("pub const SCHEDULE: &[DispatchOp]"), "{src}");
        assert!(
            src.contains("DispatchOp::Kernel(KernelKind::"),
            "schedule entry missing: {src}"
        );
    }

    #[test]
    fn schedule_with_empty_stages_emits_empty_const() {
        let prog = CgProgram::default();
        let schedule = ComputeSchedule { stages: vec![] };
        let src = synthesize_schedule(&schedule, &prog);
        assert!(
            src.contains("pub const SCHEDULE: &[DispatchOp] = &[];"),
            "empty schedule should emit empty const: {src}"
        );
    }

    // ---- 9. Determinism ----

    #[test]
    fn cross_cutting_synthesis_is_deterministic() {
        let (prog, _) = one_view_fold_program(3, "threat_level");
        assert_eq!(synthesize_binding_sources(), synthesize_binding_sources());
        assert_eq!(synthesize_external_buffers(), synthesize_external_buffers());
        assert_eq!(synthesize_transient_handles(), synthesize_transient_handles());
        assert_eq!(synthesize_pingpong_context(), synthesize_pingpong_context());
        assert_eq!(synthesize_pool(), synthesize_pool());
        assert_eq!(
            synthesize_resident_context(&prog),
            synthesize_resident_context(&prog)
        );
    }
}
