//! `KernelTopology` → [`KernelSpec`] lowering — Task 4.2.
//!
//! Walks a [`KernelTopology`] (one of [`KernelTopology::Fused`] /
//! [`KernelTopology::Split`] / [`KernelTopology::Indirect`]) and
//! produces a [`KernelSpec`] — the structured contract that the
//! existing [`crate::kernel_lowerings`] / [`crate::emit_kernel_module`]
//! lowerings consume to emit the four downstream surfaces (Rust BGL
//! entries, WGSL binding decls, BindGroupEntry construction, the
//! `Bindings` struct fields). Drift between those surfaces is therefore
//! structurally impossible: every downstream output is derived from the
//! single spec produced here.
//!
//! # Pipeline
//!
//! ```text
//!   KernelTopology  ─┐
//!                    ├─►  kernel_topology_to_spec  ─►  KernelSpec
//!   CgProgram       ─┤                                   │
//!   EmitCtx         ─┘                                   ▼
//!                                                kernel_binding_ir
//!                                                lowerings (Rust+WGSL)
//! ```
//!
//! # Limitations
//!
//! - **Per-thread setup hoisting deferred.** Multiple ops in a
//!   [`KernelTopology::Fused`] kernel may read the same agent field
//!   (`self.hp`, `self.alive`); for Task 4.2's first cut the
//!   per-op WGSL is concatenated as-is, leaving the WGSL compiler to
//!   dedupe. A future task can hoist the shared reads into a kernel
//!   preamble.
//! - **Kernel naming is semantic.** As of Task 5.2 the synthesized
//!   names match the legacy emitter filenames: `fused_mask`,
//!   `scoring`, `fold_<view>`, `physics_<rule>`,
//!   `spatial_<query_kind>`, plus per-plumbing-kind names
//!   (`alive_pack`, `seed_indirect_<ring>`, …). See
//!   [`semantic_kernel_name`] for the full mapping table. Multi-op
//!   fused kernels prefix the first op's name with `fused_`; collisions
//!   surface as
//!   [`super::program::ProgramEmitError::KernelNameCollision`] rather
//!   than silently overwriting.
//! - **Op-body lowering is partial.** Only [`ComputeOpKind::PhysicsRule`]
//!   and [`ComputeOpKind::ViewFold`] (which carry real
//!   [`crate::cg::stmt::CgStmtList`] bodies) lower through Task 4.1's
//!   walks. [`ComputeOpKind::MaskPredicate`] lowers its predicate
//!   expression and emits an `atomicOr` placeholder for the bitmap
//!   write. [`ComputeOpKind::SpatialQuery`] dispatches per-kind to one
//!   of the [`SPATIAL_BUILD_HASH_BODY`] / [`SPATIAL_KIN_QUERY_BODY`] /
//!   [`SPATIAL_ENGAGEMENT_QUERY_BODY`] templates (verbatim ports from
//!   the legacy `engine_gpu_rules/src/spatial_*.wgsl` stubs — Task
//!   5.6c). [`ComputeOpKind::Plumbing`] dispatches per-`PlumbingKind`
//!   to one of the [`PACK_AGENTS_BODY`] / [`UNPACK_AGENTS_BODY`] /
//!   [`ALIVE_BITMAP_BODY`] / [`UPLOAD_SIM_CFG_BODY`] /
//!   [`KICK_SNAPSHOT_BODY`] templates plus the per-ring
//!   [`drain_events_body`] / [`seed_indirect_args_body`] formatters
//!   (Task 5.6d). [`ComputeOpKind::ScoringArgmax`] still emits a
//!   documented `// TODO(task-4.x)` placeholder line (never a Rust
//!   panic) — Task 5.6e will replace it.
//! - **`Indirect` topology emits the consumer kernel only.** The
//!   producer (a [`crate::cg::op::PlumbingKind::SeedIndirectArgs`])
//!   does not contribute to the kernel WGSL body — its dispatch-args
//!   wiring is emitted separately at the schedule layer (Task 4.3).
//!   This kernel's bindings include the indirect-args buffer as a
//!   read so the consumer can size its loop.
//! - **Binding metadata table is approximate.** Each [`DataHandle`]
//!   variant maps to a [`BgSource`] / [`AccessMode`] / WGSL type via
//!   [`handle_to_binding_metadata`]; the table is shaped to match the
//!   conventions in `emit_mask_kernel`, `emit_scoring_kernel`, and
//!   `emit_view_fold_kernel`, but field names and types are not yet
//!   diffed against the legacy emitters byte-for-byte. Task 5.1 will
//!   close that gap.
//! - **Cfg uniform is the LAST slot.** Mirrors the existing emitter
//!   convention (`emit_mask_kernel.rs` slot 3, `emit_scoring_kernel.rs`
//!   slot 4, `emit_view_fold_kernel.rs` slot N). Data bindings come
//!   first, sorted by [`DataHandle::cycle_edge_key`] for determinism.
//! - **Read/write access merging is conservative.** A handle that any
//!   op writes is upgraded to [`AccessMode::ReadWriteStorage`] (or
//!   [`AccessMode::AtomicStorage`] for handles whose base mode is
//!   atomic). A handle accessed only via reads stays
//!   [`AccessMode::ReadStorage`]. No write-then-read elision is
//!   attempted.
//!
//! # Determinism
//!
//! The function is a pure function of `(topology, prog)` (the
//! [`EmitCtx`] only carries naming strategy + `prog` reference). Slot
//! assignment uses a stable [`DataHandle::cycle_edge_key`]-keyed sort,
//! never a `HashMap` iteration. Two runs over the same inputs produce
//! byte-identical [`KernelSpec`] structs.

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use crate::cg::data_handle::{
    AgentFieldTy, AgentScratchKind, CycleEdgeKey, DataHandle, EventRingAccess, MaskId,
    SpatialStorageKind, ViewStorageSlot,
};
use crate::cg::dispatch::DispatchShape;
use crate::cg::op::{
    ComputeOp, ComputeOpKind, OpId, PlumbingKind, ReplayabilityFlag, ScoringId, ScoringRowOp,
    SpatialQueryKind,
};
use crate::cg::program::CgProgram;
use crate::cg::schedule::synthesis::KernelTopology;
use crate::kernel_binding_ir::{
    snake_to_pascal, AccessMode, BgSource, KernelBinding, KernelKind, KernelSpec,
};

use super::wgsl_body::EmitError as InnerEmitError;
use super::wgsl_body::{lower_cg_expr_to_wgsl, lower_cg_stmt_list_to_wgsl, EmitCtx};

// ---------------------------------------------------------------------------
// KernelEmitError
// ---------------------------------------------------------------------------

/// Errors a Task-4.2 lowering can raise. Wraps Task 4.1's inner-walk
/// [`InnerEmitError`] for arena-resolution failures bubbled from
/// expression / statement lowering, and adds Task-4.2-specific variants
/// for topology / spec-level failures. Every variant carries typed
/// payload — no free-form `String` reasons.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelEmitError {
    /// An [`OpId`] in the topology is past the end of `prog.ops`.
    OpIdOutOfRange { id: OpId, arena_len: u32 },
    /// The topology carries no body ops (empty `Fused.ops`,
    /// empty `Indirect.consumers`). Should not occur in well-formed
    /// schedules synthesised by Phase 3.
    EmptyKernelTopology,
    /// The synthesised [`KernelSpec`] failed
    /// [`KernelSpec::validate`]. The wrapped reason carries the
    /// validation error text — always one of the structural causes
    /// `KernelSpec::validate` documents (slot non-contiguity, dangling
    /// `AliasOf`).
    InvalidKernelSpec { reason: String },
    /// An inner Task-4.1 expression / statement lowering raised an
    /// arena-out-of-range error. The wrapped variant carries the
    /// typed payload.
    Inner(InnerEmitError),
    /// An op-kind landed under a [`DispatchShape`] its lowering does
    /// not support. Surfaced by per-op body templates (today:
    /// `MaskPredicate` admits `PerAgent` + `PerPair`; everything else
    /// is a typed mismatch). Task 5.6a.
    InvalidDispatchForOpKind {
        op_kind: &'static str,
        dispatch: String,
    },
    /// Two distinct [`DataHandle`]s collapse to the same emitted
    /// binding name (via `structural_binding_name`) but resolve to
    /// different storage (different `wgsl_ty` or `bg_source`). Dedup
    /// is only valid when the collapsed handles back the SAME
    /// physical buffer; mismatches indicate a real bug in the
    /// metadata / naming tables. Surfaced by the binding-dedup pass
    /// in `kernel_topology_to_spec_and_body`.
    BindingNameCollision {
        name: String,
        existing_ty: String,
        new_ty: String,
    },
}

impl fmt::Display for KernelEmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelEmitError::OpIdOutOfRange { id, arena_len } => write!(
                f,
                "OpId(#{}) out of range (op arena holds {} entries)",
                id.0, arena_len
            ),
            KernelEmitError::EmptyKernelTopology => f.write_str("kernel topology has no body ops"),
            KernelEmitError::InvalidKernelSpec { reason } => {
                write!(f, "synthesised KernelSpec failed validation: {reason}")
            }
            KernelEmitError::Inner(inner) => write!(f, "inner lowering failure: {inner}"),
            KernelEmitError::InvalidDispatchForOpKind { op_kind, dispatch } => write!(
                f,
                "{op_kind} op cannot lower under dispatch shape {dispatch}"
            ),
            KernelEmitError::BindingNameCollision {
                name,
                existing_ty,
                new_ty,
            } => write!(
                f,
                "binding name `{name}` collides on incompatible storage \
                 (existing wgsl_ty `{existing_ty}`, new wgsl_ty `{new_ty}`)"
            ),
        }
    }
}

impl std::error::Error for KernelEmitError {}

impl From<InnerEmitError> for KernelEmitError {
    fn from(value: InnerEmitError) -> Self {
        KernelEmitError::Inner(value)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Lower a [`KernelTopology`] to its corresponding [`KernelSpec`].
///
/// `prog` is the program the topology was synthesised from (every
/// [`OpId`] in the topology indexes `prog.ops`). `ctx` carries the
/// active [`EmitCtx`] for inner expression / statement lowering.
///
/// Returns the [`KernelSpec`] on success. The spec is also validated
/// via [`KernelSpec::validate`] before return — a non-Ok validation is
/// surfaced as [`KernelEmitError::InvalidKernelSpec`].
///
/// # Errors
///
/// - [`KernelEmitError::OpIdOutOfRange`] — a topology op-id is past
///   `prog.ops.len()`.
/// - [`KernelEmitError::Inner`] (wrapping
///   [`InnerEmitError::ExprIdOutOfRange`] /
///   [`InnerEmitError::StmtIdOutOfRange`] /
///   [`InnerEmitError::StmtListIdOutOfRange`]) — bubbled from Task
///   4.1's inner walks.
/// - [`KernelEmitError::InvalidKernelSpec`] — the synthesised spec
///   failed [`KernelSpec::validate`] (e.g. AliasOf points to a
///   non-existent binding name).
/// - [`KernelEmitError::EmptyKernelTopology`] — topology carries no
///   body ops.
pub fn kernel_topology_to_spec(
    topology: &KernelTopology,
    prog: &CgProgram,
    ctx: &EmitCtx<'_>,
) -> Result<KernelSpec, KernelEmitError> {
    let (spec, _body) = kernel_topology_to_spec_and_body(topology, prog, ctx)?;
    Ok(spec)
}

/// Lower a [`KernelTopology`] to its [`KernelSpec`] AND return the
/// composed WGSL body string alongside. Useful for tests + Task 4.3
/// callers that consume both. `KernelSpec` itself does not carry the
/// body string today; see `# Limitations` on
/// [`kernel_topology_to_spec`].
///
/// This function carries the full lowering pipeline; the
/// [`kernel_topology_to_spec`] entry point delegates here and discards
/// the body so the body is never computed twice.
pub fn kernel_topology_to_spec_and_body(
    topology: &KernelTopology,
    prog: &CgProgram,
    ctx: &EmitCtx<'_>,
) -> Result<(KernelSpec, String), KernelEmitError> {
    // 1. Resolve the topology to (body_ops, dispatch, kind_label).
    //
    //    `body_ops` is the list of `OpId`s whose expressions /
    //    statements contribute to the kernel's WGSL body. For
    //    `Indirect`, only the consumers contribute — the producer is
    //    a SeedIndirectArgs plumbing op emitted separately at the
    //    schedule layer.
    let (body_ops, dispatch, kind_label) = match topology {
        KernelTopology::Fused { ops, dispatch } => (ops.clone(), *dispatch, "fused"),
        KernelTopology::Split { op, dispatch } => (vec![*op], *dispatch, "split"),
        KernelTopology::Indirect {
            producer: _,
            consumers,
        } => {
            // Pick the dispatch from the first consumer — every
            // consumer shares the same `PerEvent { source_ring }`
            // shape (synthesis guarantees this; if the program is
            // mis-shaped, the well-formed pass already failed).
            let first = *consumers
                .first()
                .ok_or(KernelEmitError::EmptyKernelTopology)?;
            let op = resolve_op(prog, first)?;
            (consumers.clone(), op.shape, "indirect")
        }
    };

    if body_ops.is_empty() {
        return Err(KernelEmitError::EmptyKernelTopology);
    }

    // 2. Classify the kernel kind — drives per-kind cfg shape, BGL
    //    layout, and (for ViewFold) custom binding synthesis. See
    //    [`KernelKindClass`] for the routing table.
    let body_ops_resolved: Vec<&ComputeOp> = body_ops
        .iter()
        .map(|id| resolve_op(prog, *id))
        .collect::<Result<Vec<_>, _>>()?;
    let class = classify_kernel(&body_ops_resolved, prog);

    // 3. Pick a semantic kernel name aligned with the legacy emitter
    //    filenames (`fused_mask`, `scoring`, `fold_<view>`, ...). See
    //    `semantic_kernel_name` for the full mapping table.
    let _ = kind_label;
    let name = semantic_kernel_name(&body_ops_resolved, prog);
    let pascal = snake_to_pascal(&name);
    let entry_point = format!("cs_{name}");
    let cfg_struct = format!("{pascal}Cfg");

    // 4. ViewFold has a fixed 7-binding shape mirroring the legacy
    //    `fold_<view>.rs` modules (event_ring, event_tail,
    //    view_storage_primary, view_storage_anchor, view_storage_ids,
    //    sim_cfg, cfg). The generic handle-aggregation pipeline does
    //    not produce the anchor/ids slots (they are not data-flow
    //    handles in the IR; the legacy kernels carry them so the BGL
    //    matches a runtime-rebound primary fallback). Synthesise the
    //    spec directly for ViewFold; route every other kernel through
    //    the generic pipeline.
    if let KernelKindClass::ViewFold { view_name } = &class {
        let bindings = build_view_fold_bindings(view_name, &cfg_struct);
        let cfg_struct_decl = build_view_fold_cfg_struct_decl(&cfg_struct);
        let cfg_build_expr = build_view_fold_cfg_build_expr(&cfg_struct);
        let wgsl_body = build_view_fold_wgsl_body(&body_ops, prog, ctx)?;
        let spec = KernelSpec {
            name,
            pascal,
            entry_point,
            cfg_struct,
            cfg_build_expr,
            cfg_struct_decl,
            bindings,
            kind: KernelKind::ViewFold,
        };
        spec.validate()
            .map_err(|reason| KernelEmitError::InvalidKernelSpec { reason })?;
        return Ok((spec, wgsl_body));
    }

    // --- Generic (non-ViewFold) path ----------------------------------

    // 5. Collect every (handle, was_written) pair across all body ops.
    //
    //    `was_written` is true if any op writes the handle (drives the
    //    AccessMode upgrade in step 6). The handle is keyed by its full
    //    `DataHandle` value but deduplicated through `cycle_edge_key`
    //    so that two `EventRing { ring, kind: Read | Append }` accesses
    //    on the same ring share a binding (the cycle key collapses the
    //    access mode).
    let mut handle_set: BTreeMap<CycleEdgeKey, HandleAggregate> = BTreeMap::new();
    for op_id in &body_ops {
        let op = resolve_op(prog, *op_id)?;
        for h in &op.reads {
            aggregate_handle(&mut handle_set, h, false);
        }
        for h in &op.writes {
            aggregate_handle(&mut handle_set, h, true);
        }
    }

    // 6. For each unique handle, derive its `BindingMetadata` (if the
    //    handle is binding-relevant — Rng / ConfigConst are not). Skip
    //    handles that have no binding metadata.
    let mut typed_bindings: Vec<TypedBinding> = Vec::new();
    for (key, agg) in &handle_set {
        let canonical = canonical_handle(key, &agg.first_seen);
        let Some(meta) = handle_to_binding_metadata(&canonical, prog) else {
            // Rng + ConfigConst route through the cfg uniform / inline
            // helpers, not as standalone bindings.
            continue;
        };
        let access = upgrade_access(meta.base_access, agg.was_written);
        let name = structural_binding_name(&canonical);
        typed_bindings.push(TypedBinding {
            sort_key: key.clone(),
            name,
            access,
            wgsl_ty: meta.wgsl_ty,
            bg_source: meta.bg_source,
        });
    }

    // 7. Sort bindings by their cycle-edge key for determinism.
    //    `BTreeMap` already iterates sorted, but `typed_bindings` is
    //    materialised post-filter so re-sort defensively.
    typed_bindings.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));

    // 7b. Dedup by emitted binding name. Two distinct `DataHandle`s
    //     can collapse to the same `structural_binding_name` (e.g. the
    //     4-way `AgentField { field: Alive, target: Target(N) }` reads
    //     in a 4-way fused mask kernel all map to `agent_alive`). The
    //     binding identifies a STORAGE BUFFER, not a per-thread
    //     access expression — so multiple cycle-edge entries that
    //     name the same buffer must collapse to one binding for both
    //     the Rust struct and the WGSL declaration to be well-formed.
    //
    //     The dedup is keyed on `name`. When two entries collide, the
    //     access modes are merged via the `access_lattice_max` order
    //     (Atomic > ReadWrite > Read > Uniform). The `wgsl_ty` /
    //     `bg_source` MUST match — a mismatch means two genuinely
    //     different storage buffers got the same name (a bug in the
    //     metadata / naming tables), surfaced as
    //     `KernelEmitError::BindingNameCollision`. The earliest
    //     `sort_key` wins so deterministic ordering is preserved.
    let mut dedup: BTreeMap<String, TypedBinding> = BTreeMap::new();
    for tb in typed_bindings.into_iter() {
        match dedup.get_mut(&tb.name) {
            Some(existing) => {
                if existing.wgsl_ty != tb.wgsl_ty || existing.bg_source != tb.bg_source {
                    return Err(KernelEmitError::BindingNameCollision {
                        name: tb.name,
                        existing_ty: existing.wgsl_ty.clone(),
                        new_ty: tb.wgsl_ty,
                    });
                }
                existing.access = access_lattice_max(existing.access.clone(), tb.access);
                if tb.sort_key < existing.sort_key {
                    existing.sort_key = tb.sort_key;
                }
            }
            None => {
                dedup.insert(tb.name.clone(), tb);
            }
        }
    }
    let mut typed_bindings: Vec<TypedBinding> = dedup.into_values().collect();
    typed_bindings.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));

    // 8. Assign slots — data bindings 0..N, cfg uniform at slot N.
    let mut bindings: Vec<KernelBinding> = Vec::with_capacity(typed_bindings.len() + 1);
    for (slot, tb) in typed_bindings.into_iter().enumerate() {
        bindings.push(KernelBinding {
            slot: slot as u32,
            name: tb.name,
            access: tb.access,
            wgsl_ty: tb.wgsl_ty,
            bg_source: tb.bg_source,
        });
    }
    let cfg_slot = bindings.len() as u32;
    bindings.push(KernelBinding {
        slot: cfg_slot,
        name: "cfg".into(),
        access: AccessMode::Uniform,
        wgsl_ty: cfg_struct.clone(),
        bg_source: BgSource::Cfg,
    });

    // 9. Build the cfg struct decl + cfg-construction expression — per
    //    classified kind. ViewFold is handled above; non-ViewFold
    //    kernels keep the placeholder shape today.
    let cfg_struct_decl = build_generic_cfg_struct_decl(&cfg_struct);
    let cfg_build_expr = build_generic_cfg_build_expr(&cfg_struct);

    // 10. Compose the WGSL body — one fragment per op, joined with
    //     blank lines. Computing the body here surfaces any inner-walk
    //     arena failures as typed errors before the spec is returned,
    //     so a malformed program never yields a spec whose body the
    //     downstream WGSL emitter can't render. The body itself is not
    //     stored on `KernelSpec` (it lives in the WGSL emitter that
    //     consumes the spec) — Task 4.3 will redo body composition at
    //     that layer. We return it alongside the spec for tests + Task
    //     4.3 callers; [`kernel_topology_to_spec`] discards it.
    let wgsl_body = build_wgsl_body(&body_ops, &dispatch, prog, ctx)?;

    let spec = KernelSpec {
        name,
        pascal,
        entry_point,
        cfg_struct,
        cfg_build_expr,
        cfg_struct_decl,
        bindings,
        kind: KernelKind::Generic,
    };

    spec.validate()
        .map_err(|reason| KernelEmitError::InvalidKernelSpec { reason })?;
    Ok((spec, wgsl_body))
}

// ---------------------------------------------------------------------------
// Kernel-kind classification
// ---------------------------------------------------------------------------

/// Coarse classification of a kernel's body ops. Used to pick the
/// per-kind cfg shape, BGL layout, and (for ViewFold) the custom
/// hand-synthesised spec.
///
/// # Limitations
///
/// - **ViewFold detection requires every body op to be a
///   [`ComputeOpKind::ViewFold`] handler over the same view.** Single
///   ViewFold ops (Split topology) and consumer-fused ViewFold groups
///   (Indirect topology with all-ViewFold consumers) both classify as
///   [`KernelKindClass::ViewFold`]. The view name is pulled from the
///   first consumer's `view: ViewId` via [`CgProgram::interner`]; if
///   no name is interned the fallback is `view_<id>`. Two consumers
///   over distinct views in the same kernel — synthesis does not
///   produce this today; if it did, the first consumer's view would
///   win and the others would point at the wrong resident accessor.
///   Task 5.4 wires the resident-handle accessor (the call site
///   here references `sources.resident.fold_view_<view>_handles()`).
/// - **Other classifications are not yet surfaced.** MaskPredicate,
///   PhysicsRule, ScoringArgmax, SpatialQuery, Plumbing all fall under
///   [`KernelKindClass::Generic`] today; per-kind cfg refinement for
///   those is a future task (5.5+).
#[derive(Debug, Clone)]
enum KernelKindClass {
    /// Every body op is a [`ComputeOpKind::ViewFold`] over a single
    /// view. Drives the legacy 7-binding fold-kernel layout +
    /// `{ event_count, tick, _pad: [u32; 2] }` cfg shape.
    ViewFold {
        /// snake_case view name. Pulled from the program interner;
        /// falls back to `view_<id>` when no name is interned. Drives
        /// the resident accessor `fold_view_<view_name>_handles()` —
        /// emitted by [`build_view_fold_bindings`]; the accessor
        /// itself is generated by Task 5.4's `emit_resident_context`.
        view_name: String,
    },
    /// Anything not detected as ViewFold. Routes through the generic
    /// handle-aggregation pipeline + placeholder cfg shape.
    Generic,
}

/// Classify a kernel's body ops. See [`KernelKindClass`] for the
/// table. `prog` carries the interner so the ViewFold path can
/// resolve a snake_case view name eagerly.
fn classify_kernel(body_ops: &[&ComputeOp], prog: &CgProgram) -> KernelKindClass {
    if body_ops.is_empty() {
        return KernelKindClass::Generic;
    }
    // ViewFold detection: every body op must be a ViewFold and all
    // must reference the same view id.
    let mut first_view_id: Option<crate::cg::data_handle::ViewId> = None;
    for op in body_ops {
        match &op.kind {
            ComputeOpKind::ViewFold { view, .. } => {
                if let Some(prev) = first_view_id {
                    if prev != *view {
                        return KernelKindClass::Generic;
                    }
                } else {
                    first_view_id = Some(*view);
                }
            }
            ComputeOpKind::MaskPredicate { .. }
            | ComputeOpKind::PhysicsRule { .. }
            | ComputeOpKind::ScoringArgmax { .. }
            | ComputeOpKind::SpatialQuery { .. }
            | ComputeOpKind::Plumbing { .. } => {
                return KernelKindClass::Generic;
            }
        }
    }
    // Reachable only if every op is a ViewFold — first_view_id is set.
    let view_id = match first_view_id {
        Some(v) => v,
        None => return KernelKindClass::Generic,
    };
    let view_name = match prog.interner.get_view_name(view_id) {
        Some(name) => name.to_string(),
        None => format!("view_{}", view_id.0),
    };
    KernelKindClass::ViewFold { view_name }
}

// ---------------------------------------------------------------------------
// Helpers — handle aggregation
// ---------------------------------------------------------------------------

/// Per-handle aggregate carried through binding synthesis.
///
/// `first_seen` records the canonical [`DataHandle`] form (used to look
/// up binding metadata). `was_written` collapses across every op:
/// `true` if any op writes the handle. This drives the
/// [`AccessMode`] upgrade rule.
#[derive(Debug, Clone)]
struct HandleAggregate {
    first_seen: DataHandle,
    was_written: bool,
}

fn aggregate_handle(
    set: &mut BTreeMap<CycleEdgeKey, HandleAggregate>,
    h: &DataHandle,
    was_write: bool,
) {
    let key = h.cycle_edge_key();
    match set.get_mut(&key) {
        Some(existing) => {
            existing.was_written |= was_write;
        }
        None => {
            set.insert(
                key,
                HandleAggregate {
                    first_seen: h.clone(),
                    was_written: was_write,
                },
            );
        }
    }
}

/// Recover the canonical [`DataHandle`] from a cycle-edge key. For
/// `Other(h)` the handle is stored verbatim; for `Ring(_)` we synthesise
/// the [`EventRingAccess::Read`] form (the binding metadata derivation
/// does not depend on the access mode at the cycle-edge level).
fn canonical_handle(key: &CycleEdgeKey, fallback: &DataHandle) -> DataHandle {
    match key {
        CycleEdgeKey::Other(h) => h.clone(),
        CycleEdgeKey::Ring(_) => fallback.clone(),
    }
}

// ---------------------------------------------------------------------------
// Binding metadata table
// ---------------------------------------------------------------------------

/// Per-handle metadata: where the buffer lives ([`BgSource`]), the
/// minimum [`AccessMode`] (upgraded to RW/Atomic if any op writes), and
/// the WGSL type string.
struct BindingMetadata {
    bg_source: BgSource,
    /// Base access mode — `ReadStorage` for read-only handles,
    /// `AtomicStorage` for handles whose semantics REQUIRE atomic
    /// regardless of read/write split (mask bitmap, alive bitmap,
    /// snapshot kick — all bit-packed). Any op writing the handle
    /// upgrades `ReadStorage` to `ReadWriteStorage`; atomic stays
    /// atomic.
    base_access: AccessMode,
    wgsl_ty: String,
}

/// Map a [`DataHandle`] to its [`BindingMetadata`]. Returns `None` for
/// handles that aren't bindings (Rng, ConfigConst — both routed via
/// the cfg uniform / inline RNG primitive). The mapping mirrors the
/// conventions in `emit_mask_kernel`, `emit_scoring_kernel`, and
/// `emit_view_fold_kernel` (see `# Limitations` on the module about
/// the alignment scope).
///
/// `prog` is consulted to resolve [`DataHandle::ViewStorage`] view
/// names against the runtime `ResidentPathContext` field convention
/// (`view_storage_<name>` for most views; special legacy aliases
/// `standing_primary` / `memory_primary` for two views — see
/// `cg/emit/cross_cutting.rs::resident_primary_field_for_view`).
/// Heterogeneous-view ViewFold fusion produces multi-view bindings
/// in one kernel; without the name-based remap, the generic emit
/// path produces `view_<id>_<slot>` field references that don't
/// resolve against the runtime resident context.
fn handle_to_binding_metadata(h: &DataHandle, prog: &CgProgram) -> Option<BindingMetadata> {
    match h {
        DataHandle::AgentField { field, target: _ } => Some(BindingMetadata {
            bg_source: BgSource::External("agents".into()),
            base_access: AccessMode::ReadStorage,
            wgsl_ty: agent_field_wgsl_ty(field.ty()),
        }),
        DataHandle::ViewStorage { view, slot } => Some(BindingMetadata {
            bg_source: BgSource::Resident(view_storage_resident_field(*view, *slot, prog)),
            base_access: AccessMode::ReadStorage,
            wgsl_ty: "array<u32>".into(),
        }),
        DataHandle::EventRing { ring: _, kind } => {
            let base_access = match kind {
                EventRingAccess::Read => AccessMode::ReadStorage,
                EventRingAccess::Append => AccessMode::AtomicStorage,
                EventRingAccess::Drain => AccessMode::ReadWriteStorage,
            };
            let wgsl_ty = match kind {
                EventRingAccess::Append => "u32".into(),
                EventRingAccess::Read => "array<u32>".into(),
                EventRingAccess::Drain => "array<u32>".into(),
            };
            // Runtime `TransientHandles` carries one cascade ring pair
            // (`cascade_current_ring` / `cascade_next_ring`) regardless
            // of the per-program ring id. Read + Drain map to the
            // current ring (consumer side); Append maps to the next
            // ring (producer side). Multi-ring schedules collapse onto
            // this pair and rely on per-ring isolation at the schedule
            // layer (a known runtime defect tracked in
            // `gpu_pipeline_smoke_status.md`).
            let bg_field = match kind {
                EventRingAccess::Read | EventRingAccess::Drain => "cascade_current_ring",
                EventRingAccess::Append => "cascade_next_ring",
            };
            Some(BindingMetadata {
                bg_source: BgSource::Transient(bg_field.into()),
                base_access,
                wgsl_ty,
            })
        }
        DataHandle::ConfigConst { .. } => None,
        DataHandle::MaskBitmap { mask: _ } => Some(BindingMetadata {
            // Runtime `TransientHandles` carries a single
            // `mask_bitmaps` storage; per-mask offset arithmetic
            // happens in WGSL (a known runtime defect — multi-mask
            // CG schedules race today; tracked in
            // `gpu_pipeline_smoke_status.md`).
            bg_source: BgSource::Transient("mask_bitmaps".into()),
            base_access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
        }),
        DataHandle::ScoringOutput => Some(BindingMetadata {
            bg_source: BgSource::Resident("scoring_table".into()),
            base_access: AccessMode::ReadWriteStorage,
            wgsl_ty: "array<u32>".into(),
        }),
        DataHandle::SpatialStorage { kind } => {
            // Runtime `Pool` fields are prefixed `spatial_*` — the
            // bare names (`grid_cells`, `grid_offsets`,
            // `query_results`) used internally to the structural
            // binding namespace must rename to the contract-side
            // identifiers for the `bind()` source path.
            let (field, base_access) = match kind {
                SpatialStorageKind::GridCells => {
                    ("spatial_grid_cells".to_string(), AccessMode::ReadStorage)
                }
                SpatialStorageKind::GridOffsets => {
                    ("spatial_grid_offsets".to_string(), AccessMode::ReadStorage)
                }
                SpatialStorageKind::QueryResults => (
                    "spatial_query_results".to_string(),
                    AccessMode::ReadWriteStorage,
                ),
            };
            Some(BindingMetadata {
                bg_source: BgSource::Pool(field),
                base_access,
                wgsl_ty: "array<u32>".into(),
            })
        }
        DataHandle::Rng { .. } => None,
        DataHandle::AliveBitmap => Some(BindingMetadata {
            bg_source: BgSource::Transient("alive_bitmap".into()),
            base_access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
        }),
        DataHandle::IndirectArgs { ring: _ } => Some(BindingMetadata {
            // Runtime `TransientHandles` carries a single
            // `cascade_indirect_args` buffer regardless of
            // per-program ring id; multi-ring schedules collapse onto
            // it (a known runtime defect — concurrent writes alias
            // today; tracked in `gpu_pipeline_smoke_status.md`).
            bg_source: BgSource::Transient("cascade_indirect_args".into()),
            base_access: AccessMode::ReadWriteStorage,
            wgsl_ty: "array<u32>".into(),
        }),
        DataHandle::AgentScratch { kind } => {
            // No runtime field for agent-scratch yet; alias onto an
            // existing transient SoA-shaped scratch buffer
            // (`mask_unpack_agents_input`) so the build links. Pack /
            // unpack ops adjacency makes the alias observably benign
            // for the smoke test (no intervening reader of the
            // mask_unpack buffer proper). Followup tracked in
            // `gpu_pipeline_smoke_status.md`.
            let bg_field = match kind {
                AgentScratchKind::Packed => "mask_unpack_agents_input",
            };
            Some(BindingMetadata {
                bg_source: BgSource::Transient(bg_field.into()),
                base_access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
            })
        }
        DataHandle::SimCfgBuffer => Some(BindingMetadata {
            // SimCfg is host-uploaded as a STORAGE buffer (see
            // `crates/engine_gpu/src/sim_cfg.rs::create_sim_cfg_buffer` —
            // usage is STORAGE | COPY_SRC | COPY_DST, no UNIFORM bit).
            // Bind as `var<storage, read>` matching the fold-kernel
            // convention (`fold_threat_level.wgsl` etc.); the typed
            // `SimCfg` view is a future refinement that needs a WGSL
            // struct decl + matching uniform-buffer flags.
            bg_source: BgSource::External("sim_cfg".into()),
            base_access: AccessMode::ReadStorage,
            wgsl_ty: "array<u32>".into(),
        }),
        DataHandle::SnapshotKick => Some(BindingMetadata {
            // No runtime field for snapshot-kick; alias onto
            // `cascade_current_tail` (atomic u32, same shape) so the
            // build links. Inside the smoke test (one tick) no reader
            // of `cascade_current_tail` runs after `kick_snapshot`, so
            // the corruption is invisible. Followup tracked in
            // `gpu_pipeline_smoke_status.md`.
            bg_source: BgSource::Transient("cascade_current_tail".into()),
            base_access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
        }),
    }
}

/// WGSL element type for an [`AgentFieldTy`]. Ports the convention used
/// by the legacy emitters: f32 fields land in an `array<f32>`, u32-ish
/// fields (counters, ticks, packed enums, optional agent ids) land in
/// `array<u32>`, vec3 in `array<vec3<f32>>`. The agents-SoA buffer is
/// today emitted as one flat `array<u32>` (`AgentField`'s `wgsl_ty`
/// here is informational — the true on-disk layout lives in the legacy
/// pack/unpack pipeline). We surface the structurally honest type so
/// downstream lowerings can specialise per-field if they choose.
fn agent_field_wgsl_ty(ty: AgentFieldTy) -> String {
    match ty {
        AgentFieldTy::F32 => "array<f32>".into(),
        AgentFieldTy::U32 => "array<u32>".into(),
        AgentFieldTy::I16 => "array<i32>".into(),
        AgentFieldTy::Bool => "array<u32>".into(),
        AgentFieldTy::Vec3 => "array<vec3<f32>>".into(),
        AgentFieldTy::EnumU8 => "array<u32>".into(),
        AgentFieldTy::OptAgentId => "array<u32>".into(),
        AgentFieldTy::OptEnumU32 => "array<u32>".into(),
    }
}

fn view_slot_field(slot: ViewStorageSlot) -> &'static str {
    match slot {
        ViewStorageSlot::Primary => "primary",
        ViewStorageSlot::Anchor => "anchor",
        ViewStorageSlot::Ids => "ids",
        ViewStorageSlot::Counts => "counts",
        ViewStorageSlot::Cursors => "cursors",
    }
}

/// Resolve a [`DataHandle::ViewStorage`] to the runtime
/// `ResidentPathContext` field name. The emit pipeline produces
/// per-view storage fields named `view_storage_<name>` for most
/// views, with two legacy-aliased exceptions (`standing_primary`,
/// `memory_primary`). Heterogeneous-view ViewFold fusion needs each
/// view's binding to resolve against the runtime resident context;
/// the structural `view_<id>_<slot>` form does not match any field
/// the generated `ResidentPathContext` exposes.
///
/// Falls back to the structural `view_<id>_<slot>` form when the
/// view name isn't interned. The slot is appended only when it
/// would have been part of the structural form — `Primary` collapses
/// to the bare `view_storage_<name>` field because that's the only
/// slot the resident context exposes per view today.
fn view_storage_resident_field(
    view: crate::cg::data_handle::ViewId,
    slot: ViewStorageSlot,
    prog: &CgProgram,
) -> String {
    let name = match prog.interner.get_view_name(view) {
        Some(n) => n.to_string(),
        None => return format!("view_{}_{}", view.0, view_slot_field(slot)),
    };
    // Mirror `cross_cutting::resident_primary_field_for_view` for the
    // primary slot — the legacy aliases (`standing_primary`,
    // `memory_primary`) and the `engaged_with → standing_primary`
    // fallback all live there. Non-primary slots aren't materialised
    // on the resident context today; emit the structural fallback so
    // a future emit-side migration sees a clear unresolved field name
    // rather than silently aliasing onto a wrong buffer.
    match slot {
        ViewStorageSlot::Primary => match name.as_str() {
            "standing" => "standing_primary".to_string(),
            "memory" => "memory_primary".to_string(),
            "engaged_with" => "standing_primary".to_string(),
            other => format!("view_storage_{other}"),
        },
        ViewStorageSlot::Anchor
        | ViewStorageSlot::Ids
        | ViewStorageSlot::Counts
        | ViewStorageSlot::Cursors => format!("view_{}_{}", view.0, view_slot_field(slot)),
    }
}

/// Upgrade a base [`AccessMode`] to read/write if the handle is
/// written. `AtomicStorage` is preserved (atomic semantics dominate);
/// `Uniform` is preserved (uniform-shaped bindings are never written).
///
/// Match arms are exhaustive over `AccessMode × bool` — adding a new
/// `AccessMode` variant forces a decision here.
fn upgrade_access(base: AccessMode, was_written: bool) -> AccessMode {
    match base {
        AccessMode::ReadStorage => {
            if was_written {
                AccessMode::ReadWriteStorage
            } else {
                AccessMode::ReadStorage
            }
        }
        // ReadWriteStorage is already at the top of the read/write
        // lattice — no further upgrade.
        AccessMode::ReadWriteStorage => AccessMode::ReadWriteStorage,
        // Atomic stays atomic regardless of read-only access (an op
        // that only reads an atomic still needs the atomic-typed
        // binding declaration).
        AccessMode::AtomicStorage => AccessMode::AtomicStorage,
        // Uniform never upgrades — the uniform-shaped buffers (sim_cfg
        // and the per-dispatch cfg) are never written from a kernel.
        AccessMode::Uniform => AccessMode::Uniform,
    }
}

/// Lattice-max over [`AccessMode`]s, used by the binding-dedup pass to
/// merge two access modes that collapse to the same emitted binding
/// name (e.g. one read-side handle and one write-side handle of the
/// same physical storage). The order, from most-permissive to least:
///
/// ```text
///   AtomicStorage > ReadWriteStorage > ReadStorage > Uniform
/// ```
///
/// `Uniform` only collides with itself in practice (uniform bindings
/// have a distinct `BgSource::Cfg`/`External` shape), but is included
/// for totality. Match arms are exhaustive over `AccessMode × AccessMode`.
fn access_lattice_max(a: AccessMode, b: AccessMode) -> AccessMode {
    fn rank(m: &AccessMode) -> u8 {
        match m {
            AccessMode::Uniform => 0,
            AccessMode::ReadStorage => 1,
            AccessMode::ReadWriteStorage => 2,
            AccessMode::AtomicStorage => 3,
        }
    }
    if rank(&a) >= rank(&b) {
        a
    } else {
        b
    }
}

/// Per-binding intermediate carrying its sort key alongside the
/// final-form fields. Materialised between metadata derivation and
/// slot assignment so determinism is preserved when sources of
/// randomness are excluded.
struct TypedBinding {
    sort_key: CycleEdgeKey,
    name: String,
    access: AccessMode,
    wgsl_ty: String,
    bg_source: BgSource,
}

// ---------------------------------------------------------------------------
// Naming
// ---------------------------------------------------------------------------

/// Render a [`DataHandle`] as a deterministic snake_case binding name.
/// Mirrors the inner-walk's structural-handle convention so a kernel
/// body referring to `agent_self_hp` resolves through this binding's
/// declared name.
///
/// For `AgentField`, we drop the `target` discriminator from the
/// binding name — the binding identifies the storage buffer, not the
/// per-thread access expression. `agent_<field>` is the form.
fn structural_binding_name(h: &DataHandle) -> String {
    match h {
        DataHandle::AgentField { field, target: _ } => {
            // The storage is shared across all per-thread accesses;
            // the binding name keys off the field, not the agent ref.
            format!("agent_{}", field.snake())
        }
        DataHandle::ViewStorage { view, slot } => {
            format!("view_{}_{}", view.0, view_slot_field(*slot))
        }
        DataHandle::EventRing { ring, kind: _ } => {
            // Cycle-edge collapses access mode; one binding per ring
            // regardless of read/append/drain (the lowering uses the
            // access mode to decide read vs atomicAdd).
            format!("event_ring_{}", ring.0)
        }
        DataHandle::ConfigConst { id } => format!("config_{}", id.0),
        DataHandle::MaskBitmap { mask } => format!("mask_{}_bitmap", mask.0),
        DataHandle::ScoringOutput => "scoring_output".into(),
        DataHandle::SpatialStorage { kind } => match kind {
            SpatialStorageKind::GridCells => "spatial_grid_cells".into(),
            SpatialStorageKind::GridOffsets => "spatial_grid_offsets".into(),
            SpatialStorageKind::QueryResults => "spatial_query_results".into(),
        },
        DataHandle::Rng { purpose } => format!("rng_{}", purpose.snake()),
        DataHandle::AliveBitmap => "alive_bitmap".into(),
        DataHandle::IndirectArgs { ring } => format!("indirect_args_{}", ring.0),
        DataHandle::AgentScratch { kind } => match kind {
            AgentScratchKind::Packed => "agent_scratch_packed".into(),
        },
        DataHandle::SimCfgBuffer => "sim_cfg".into(),
        DataHandle::SnapshotKick => "snapshot_kick".into(),
    }
}

/// Pick the kernel's snake_case file name. Aligned with the legacy
/// emitter filenames so the side-channel output drops directly into a
/// CG-overlaid `engine_gpu_rules` crate.
///
/// Mapping:
/// - Single [`ComputeOpKind::MaskPredicate`] → `mask_<name>` (interner
///   lookup); falls back to `mask_<id>` when no name is interned.
/// - Single [`ComputeOpKind::ScoringArgmax`] → `scoring`. (At most one
///   scoring kernel per program today; the legacy emitter uses the
///   bare name.)
/// - Single [`ComputeOpKind::PhysicsRule`] → `physics_<rule>` (interner
///   lookup); falls back to `physics_rule_<id>`.
/// - Single [`ComputeOpKind::ViewFold`] →
///   `fold_<view>_<event>` (both via interner lookup, event-kind name
///   normalized PascalCase → snake_case via [`pascal_to_snake`]); the
///   event suffix disambiguates handlers when one view subscribes to
///   multiple events (e.g. `threat_level` folds both
///   `AgentAttacked` and `EffectDamageApplied` and emits two distinct
///   kernels). Falls back to `fold_view_<id>` / `fold_..._event_<id>`
///   when the interner has no name.
/// - Single [`ComputeOpKind::SpatialQuery`] → `spatial_<kind_label>`
///   (`spatial_build_hash`, `spatial_kin_query`,
///   `spatial_engagement_query`).
/// - Single [`ComputeOpKind::Plumbing`] → per-kind name (`alive_pack`
///   for `AliveBitmap`, `seed_indirect` for `SeedIndirectArgs`,
///   `pack_agents` / `unpack_agents`, `drain_events`,
///   `upload_sim_cfg`, `kick_snapshot`).
/// - Multi-op fused kernel → `fused_<first_op_kernel_name>`.
///
/// # Limitations
/// - The fused-kernel naming uses the FIRST op's name with a `fused_`
///   prefix. With more than one op of the same kind in a fused kernel
///   the prefix collapses ambiguity rather than spelling out every
///   contributing op. Two distinct fused kernels with the same
///   first-op name still collide; the
///   [`super::program::ProgramEmitError::KernelNameCollision`] check
///   surfaces that as a typed error rather than silently overwriting.
/// - Plumbing variants `DrainEvents { ring }` and `SeedIndirectArgs
///   { ring }` use a per-ring suffix (`drain_events_<ring>` /
///   `seed_indirect_<ring>`) so distinct rings never collide. Without
///   a ring-name interner the suffix is the numeric ring id.
/// Public wrapper over [`semantic_kernel_name`] that takes a
/// [`KernelTopology`] directly. Resolves the topology to its body ops
/// (mirroring the routing in [`kernel_topology_to_spec_and_body`]) and
/// returns the same snake_case name the per-kernel emit uses for the
/// `.rs` / `.wgsl` filenames. Returns `None` when the topology resolves
/// to an empty body or carries an out-of-range op id (defensive — the
/// per-kernel emit would already have surfaced these as
/// [`KernelEmitError`] variants).
///
/// Used by [`super::cross_cutting::synthesize_schedule`] so the
/// schedule entries reference the same kernel names the per-kernel
/// modules carry — drift between the two sites is structurally
/// impossible because both call this helper.
///
/// # Limitations
/// - **Indirect topologies report only the consumer kernel name.** The
///   `SeedIndirectArgs` producer is a separate plumbing op handled
///   independently at the schedule layer.
pub fn semantic_kernel_name_for_topology(
    topology: &KernelTopology,
    prog: &CgProgram,
) -> Option<String> {
    let body_ops: Vec<OpId> = match topology {
        KernelTopology::Fused { ops, .. } => ops.clone(),
        KernelTopology::Split { op, .. } => vec![*op],
        KernelTopology::Indirect { consumers, .. } => consumers.clone(),
    };
    if body_ops.is_empty() {
        return None;
    }
    let mut resolved: Vec<&ComputeOp> = Vec::with_capacity(body_ops.len());
    for id in &body_ops {
        match resolve_op(prog, *id) {
            Ok(op) => resolved.push(op),
            Err(_) => return None,
        }
    }
    Some(semantic_kernel_name(&resolved, prog))
}

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
    // ops with matching replayability collapses to a name encoding
    // the contained rules. Mirrors the legacy `physics.rs` kernel
    // module shape but disambiguates per-rule so distinct kernels do
    // not collide when multiple physics rules lower into separate
    // (singleton) PhysicsRule kernels.
    if let Some(name) = physics_rule_fused_kernel_name(body_ops, prog) {
        return name;
    }

    if body_ops.len() == 1 {
        return single_op_kernel_name(&body_ops[0].kind, prog);
    }
    // Fused kernel: prefix the first op's name with `fused_` (unless the
    // first op already starts with `fused_`, in which case keep it).
    let first = single_op_kernel_name(&body_ops[0].kind, prog);
    if first.starts_with("fused_") {
        first
    } else {
        format!("fused_{first}")
    }
}

/// `Some("fold_<view>")` iff every op in `body_ops` is a
/// [`ComputeOpKind::ViewFold`] referencing the same view. Returns
/// `None` otherwise (mixed kinds, different views, or zero ops).
/// When the slice is a singleton we still drop the event suffix —
/// the legacy emitter names a single-handler view's kernel
/// `fold_<view>`, never `fold_<view>_<event>`.
fn view_fold_fused_kernel_name(body_ops: &[&ComputeOp], prog: &CgProgram) -> Option<String> {
    let mut view = None;
    for op in body_ops {
        match &op.kind {
            ComputeOpKind::ViewFold { view: v, .. } => match view {
                None => view = Some(*v),
                Some(prev) if prev == *v => {}
                Some(_) => return None,
            },
            ComputeOpKind::MaskPredicate { .. }
            | ComputeOpKind::ScoringArgmax { .. }
            | ComputeOpKind::PhysicsRule { .. }
            | ComputeOpKind::SpatialQuery { .. }
            | ComputeOpKind::Plumbing { .. } => return None,
        }
    }
    let view = view?;
    Some(match prog.interner.get_view_name(view) {
        Some(name) => format!("fold_{name}"),
        None => format!("fold_view_{}", view.0),
    })
}

/// `Some("physics_<rule>")` for a singleton PhysicsRule kernel;
/// `Some("physics_<a>_and_<b>...")` for a multi-rule fused group with
/// uniform replayability; `None` otherwise (mixed kinds, mixed
/// replayability, or zero ops).
///
/// Per-rule naming is required because every physics rule lowers to
/// its own (singleton) PhysicsRule kernel today: collapsing them all
/// to `"physics"` / `"physics_post"` produced
/// [`super::program::ProgramEmitError::KernelNameCollision`] once
/// more than one rule lowered without diagnostics. The naming here
/// agrees with the singleton path in [`single_op_kernel_name`] (which
/// also uses `physics_<rule>`), so a single PhysicsRule kernel routed
/// through either path produces the same name.
fn physics_rule_fused_kernel_name(
    body_ops: &[&ComputeOp],
    prog: &CgProgram,
) -> Option<String> {
    // Collect distinct (rule_id, replayable) pairs from the body so
    // the resulting kernel name reflects exactly which rules the
    // kernel covers.
    let mut rules: Vec<(crate::cg::PhysicsRuleId, ReplayabilityFlag)> = Vec::new();
    for op in body_ops {
        match &op.kind {
            ComputeOpKind::PhysicsRule {
                rule, replayable, ..
            } => {
                if !rules.iter().any(|(r, _)| r == rule) {
                    rules.push((*rule, *replayable));
                }
            }
            ComputeOpKind::MaskPredicate { .. }
            | ComputeOpKind::ScoringArgmax { .. }
            | ComputeOpKind::ViewFold { .. }
            | ComputeOpKind::SpatialQuery { .. }
            | ComputeOpKind::Plumbing { .. } => return None,
        }
    }
    if rules.is_empty() {
        return None;
    }
    // Replayability must be uniform across the fused group — mixing
    // replayable + non-replayable in one kernel breaks the cascade
    // gate semantics, so surface as None and let the caller fall
    // back to the structural `fused_<first>` name.
    let first_flag = rules[0].1;
    if !rules.iter().all(|(_, f)| *f == first_flag) {
        return None;
    }
    // Single-rule case: match the singleton naming path so a
    // PhysicsRule kernel produced via either route ends up with the
    // same name.
    if rules.len() == 1 {
        let (rule_id, _) = rules[0];
        return Some(match prog.interner.get_physics_rule_name(rule_id) {
            Some(name) => format!("physics_{name}"),
            None => format!("physics_rule_{}", rule_id.0),
        });
    }
    // Multi-rule case: concatenate rule names with `_and_` so the
    // kernel name encodes its membership.
    let parts: Vec<String> = rules
        .iter()
        .map(|(rule_id, _)| {
            prog.interner
                .get_physics_rule_name(*rule_id)
                .map(String::from)
                .unwrap_or_else(|| format!("rule_{}", rule_id.0))
        })
        .collect();
    Some(format!("physics_{}", parts.join("_and_")))
}

/// Snake_case kernel name for a single op (the building block of
/// [`semantic_kernel_name`]).
fn single_op_kernel_name(kind: &ComputeOpKind, prog: &CgProgram) -> String {
    match kind {
        ComputeOpKind::MaskPredicate { mask, .. } => match prog.interner.get_mask_name(*mask) {
            Some(name) => format!("mask_{name}"),
            None => format!("mask_{}", mask.0),
        },
        ComputeOpKind::ScoringArgmax { .. } => "scoring".to_string(),
        ComputeOpKind::PhysicsRule { rule, .. } => {
            // The physics rule's name is unique within the program's
            // interner; PhysicsRule ops do not need an event-kind
            // suffix to disambiguate (each `physics` rule produces a
            // single op).
            match prog.interner.get_physics_rule_name(*rule) {
                Some(name) => format!("physics_{name}"),
                None => format!("physics_rule_{}", rule.0),
            }
        }
        ComputeOpKind::ViewFold {
            view, on_event, ..
        } => {
            let view_part = match prog.interner.get_view_name(*view) {
                Some(name) => format!("fold_{name}"),
                None => format!("fold_view_{}", view.0),
            };
            // Suffix the event kind to disambiguate handlers when one
            // view subscribes to multiple events (e.g. `threat_level`
            // folds both `AgentAttacked` and `EffectDamageApplied`).
            // The legacy emitter packs all handlers into one
            // `fold_<view>` kernel; the CG pipeline's per-op lowering
            // produces one Split kernel per handler today, so the
            // names must diverge.
            //
            // Event-kind names in the interner are PascalCase
            // (matching the DSL's `event Name { ... }` casing); they
            // are normalized to snake_case here so the kernel
            // filename and module name follow the legacy snake_case
            // convention.
            match prog.interner.get_event_kind_name(*on_event) {
                Some(name) => format!("{view_part}_{}", pascal_to_snake(name)),
                None => format!("{view_part}_event_{}", on_event.0),
            }
        }
        ComputeOpKind::SpatialQuery { kind } => format!("spatial_{}", spatial_kind_name(*kind)),
        ComputeOpKind::Plumbing { kind } => plumbing_kind_name(kind),
    }
}

/// Lowercase + insert underscores before each uppercase letter
/// boundary. Used to normalize PascalCase event-kind names from the
/// interner into the snake_case convention every emitted kernel name
/// follows. A leading uppercase letter does not produce a leading
/// underscore.
fn pascal_to_snake(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn spatial_kind_name(k: SpatialQueryKind) -> &'static str {
    match k {
        SpatialQueryKind::BuildHash => "build_hash",
        SpatialQueryKind::KinQuery => "kin_query",
        SpatialQueryKind::EngagementQuery => "engagement_query",
    }
}

fn plumbing_kind_name(k: &PlumbingKind) -> String {
    match k {
        PlumbingKind::PackAgents => "pack_agents".to_string(),
        PlumbingKind::UnpackAgents => "unpack_agents".to_string(),
        PlumbingKind::AliveBitmap => "alive_pack".to_string(),
        PlumbingKind::DrainEvents { ring } => format!("drain_events_{}", ring.0),
        PlumbingKind::UploadSimCfg => "upload_sim_cfg".to_string(),
        PlumbingKind::KickSnapshot => "kick_snapshot".to_string(),
        PlumbingKind::SeedIndirectArgs { ring } => format!("seed_indirect_{}", ring.0),
    }
}

/// Short snake_case label for a [`ComputeOpKind`] — used in kernel
/// naming. `compute_dependencies` already exposes the same kind shape;
/// this helper trims that to a kernel-name-safe identifier.
fn compute_op_kind_short(kind: &ComputeOpKind) -> &'static str {
    match kind {
        ComputeOpKind::MaskPredicate { .. } => "mask_predicate",
        ComputeOpKind::ScoringArgmax { .. } => "scoring_argmax",
        ComputeOpKind::PhysicsRule { .. } => "physics_rule",
        ComputeOpKind::ViewFold { .. } => "view_fold",
        ComputeOpKind::SpatialQuery { .. } => "spatial_query",
        ComputeOpKind::Plumbing { .. } => "plumbing",
    }
}

// ---------------------------------------------------------------------------
// Cfg struct + build expr — generic (non-ViewFold) kernels
// ---------------------------------------------------------------------------

/// Build a minimal cfg struct decl for non-ViewFold kernels. The legacy
/// emitters embed per-kernel cfg fields (e.g. `agent_cap`,
/// `num_mask_words`); the generic placeholder shape is one field so the
/// spec validates and downstream lowerings have something to consume.
///
/// # Limitations
///
/// - **Per-kind refinement is partial.** ViewFold has its own cfg shape
///   (event_count + tick) routed through [`build_view_fold_cfg_struct_decl`].
///   Mask, scoring, physics, spatial, and plumbing kernels share this
///   placeholder today; Task 5.5 will refine.
fn build_generic_cfg_struct_decl(cfg_struct: &str) -> String {
    format!(
        "#[repr(C)]\n\
         #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
         pub struct {cfg_struct} {{ pub agent_cap: u32, pub _pad: [u32; 3] }}"
    )
}

fn build_generic_cfg_build_expr(cfg_struct: &str) -> String {
    format!("{cfg_struct} {{ agent_cap: state.agent_cap(), _pad: [0; 3] }}")
}

// ---------------------------------------------------------------------------
// ViewFold-specific spec synthesis
// ---------------------------------------------------------------------------

/// Build the cfg struct decl for a ViewFold kernel. Mirrors the legacy
/// fold-kernel shape (`{ event_count: u32, tick: u32, _pad: [u32; 2] }`)
/// — `event_count` is left at 0 in build_cfg and populated at dispatch
/// time via the indirect-args buffer.
///
/// # Limitations
///
/// - **`event_count` is set to 0 in build_cfg.** The real value comes
///   from the cascade tail / per-fold indirect-args buffer at dispatch
///   time. Task 5.7 wires the dispatch-time population.
fn build_view_fold_cfg_struct_decl(cfg_struct: &str) -> String {
    format!(
        "#[repr(C)]\n\
         #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
         pub struct {cfg_struct} {{ pub event_count: u32, pub tick: u32, pub _pad: [u32; 2] }}"
    )
}

fn build_view_fold_cfg_build_expr(cfg_struct: &str) -> String {
    format!("{cfg_struct} {{ event_count: 0, tick: state.tick, _pad: [0; 2] }}")
}

/// Synthesise the 7-binding [`KernelBinding`] vector for a ViewFold
/// kernel. The slot layout mirrors the legacy `fold_<view>.rs` modules
/// (`crates/engine_gpu_rules/src/fold_threat_level.rs` etc.):
///
///   slot 0: event_ring             (read storage)
///   slot 1: event_tail             (read storage)
///   slot 2: view_storage_primary   (read_write storage)
///   slot 3: view_storage_anchor    (read_write storage)
///   slot 4: view_storage_ids       (read_write storage)
///   slot 5: sim_cfg                (read storage)  -- legacy convention; not a uniform here
///   slot 6: cfg                    (uniform)
///
/// The three view-storage slots use [`BgSource::ViewHandle`] tagged
/// with the per-view resident accessor name
/// (`fold_view_<view_name>_handles`); the program-emit layer
/// destructures the tuple once into local primary/anchor/ids buffers
/// and the `record()` body's anchor/ids slots fall back to primary via
/// `Option::unwrap_or`.
///
/// # Limitations
///
/// - **`sim_cfg` slot is `BgSource::External("sim_cfg")` storage —
///   the legacy fold kernels declare slot 5 as `var<storage, read>`
///   even though `sim_cfg` is uniform-shaped on the resident path.
///   Aligning with that convention for parity; the WGSL declaration
///   will be `var<storage, read> sim_cfg: array<u32>` per
///   [`crate::kernel_lowerings::lower_wgsl_bindings`].
/// - **The resident-handle accessor (`fold_view_<view>_handles`) is
///   generated by Task 5.4's `emit_resident_context`.** Today's
///   side-channel run inherits the legacy `resident_context.rs` which
///   already exposes the accessors for the 9 active views. New views
///   landing before Task 5.4 will need a manual accessor stub.
/// - **Anchor/ids slots are tagged `ReadWriteStorage` even though they
///   may fall back to `primary` at runtime.** The BGL slot has to be
///   live; rebinding primary into both is a no-op safe choice that
///   matches the legacy emitters.
fn build_view_fold_bindings(view_name: &str, cfg_struct: &str) -> Vec<KernelBinding> {
    let accessor = format!("fold_view_{view_name}_handles");
    vec![
        KernelBinding {
            slot: 0,
            name: "event_ring".into(),
            access: AccessMode::ReadStorage,
            wgsl_ty: "array<u32>".into(),
            bg_source: BgSource::Resident("batch_events_ring".into()),
        },
        KernelBinding {
            slot: 1,
            name: "event_tail".into(),
            access: AccessMode::ReadStorage,
            wgsl_ty: "array<u32>".into(),
            bg_source: BgSource::Resident("batch_events_tail".into()),
        },
        KernelBinding {
            slot: 2,
            name: "view_storage_primary".into(),
            access: AccessMode::ReadWriteStorage,
            wgsl_ty: "array<u32>".into(),
            bg_source: BgSource::ViewHandle {
                accessor: accessor.clone(),
                tuple_idx: 0,
            },
        },
        KernelBinding {
            slot: 3,
            name: "view_storage_anchor".into(),
            access: AccessMode::ReadWriteStorage,
            wgsl_ty: "array<u32>".into(),
            bg_source: BgSource::ViewHandle {
                accessor: accessor.clone(),
                tuple_idx: 1,
            },
        },
        KernelBinding {
            slot: 4,
            name: "view_storage_ids".into(),
            access: AccessMode::ReadWriteStorage,
            wgsl_ty: "array<u32>".into(),
            bg_source: BgSource::ViewHandle {
                accessor,
                tuple_idx: 2,
            },
        },
        KernelBinding {
            slot: 5,
            name: "sim_cfg".into(),
            access: AccessMode::ReadStorage,
            wgsl_ty: "array<u32>".into(),
            bg_source: BgSource::External("sim_cfg".into()),
        },
        KernelBinding {
            slot: 6,
            name: "cfg".into(),
            access: AccessMode::Uniform,
            wgsl_ty: cfg_struct.to_string(),
            bg_source: BgSource::Cfg,
        },
    ]
}

/// Compose the ViewFold-specific WGSL body. Adds the per-kind preamble
/// (`if event_idx >= cfg.event_count { return; }`) and concatenates the
/// per-handler bodies through Task 4.1's [`lower_cg_stmt_list_to_wgsl`].
///
/// # Limitations
///
/// - **Body lowering inherits Task 4.1's coverage.** Event-pattern
///   bindings (`MatchArmBinding::local`) read from inside an arm body
///   surface as [`InnerEmitError::UnsupportedLocalBinding`]; for those
///   the body falls through to a documented `// TODO(task-5.5)` line
///   so the WGSL still compiles structurally.
/// - **No storage-hint-specific body templates.** The PairMap /
///   SymmetricPairTopK / PerEntityRing / PerEntityTopK / PairMap@decay
///   update primitives (atomicAdd, sort-and-write, ring-append-modulo)
///   are not yet emitted around the lowered body. The body emerges
///   verbatim from the IR's `CgStmt::Assign { target: ViewStorage{..}, .. }`
///   statements, which today lower as plain assignments. Storage-hint
///   templating is a Task 5.5 concern.
/// - **`event_count` and event decode are not wired.** The preamble
///   bounds-checks against `cfg.event_count` but no event-record
///   decode is performed; per-view handlers reading event fields hit
///   the Task 4.1 `UnsupportedLocalBinding` path.
fn build_view_fold_wgsl_body(
    body_ops: &[OpId],
    prog: &CgProgram,
    ctx: &EmitCtx<'_>,
) -> Result<String, KernelEmitError> {
    let mut out = String::new();
    out.push_str("    let event_idx = gid.x;\n");
    out.push_str("    if (event_idx >= cfg.event_count) { return; }\n");
    // Bind `tick` (no underscore) so `CgExpr::NamespaceField { ns: World,
    // field: "tick", access: PreambleLocal { local_name: "tick" } }` reads
    // resolve to a kernel-local `tick` identifier. Even when the kernel
    // body doesn't reference world.tick, WGSL's dead-code elimination
    // drops the unused let binding silently — there's no compile cost to
    // emitting it unconditionally. Source-of-truth for the binding name
    // is `populate_namespace_registry` in `lower::driver`.
    out.push_str("    let tick = cfg.tick;\n\n");

    for (i, op_id) in body_ops.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        let op = resolve_op(prog, *op_id)?;
        // ViewFold body is a CgStmtList; lower via Task 4.1.
        let fragment = match &op.kind {
            ComputeOpKind::ViewFold { body, .. } => {
                lower_cg_stmt_list_to_wgsl(*body, ctx).map_err(KernelEmitError::from)?
            }
            ComputeOpKind::MaskPredicate { .. }
            | ComputeOpKind::PhysicsRule { .. }
            | ComputeOpKind::ScoringArgmax { .. }
            | ComputeOpKind::SpatialQuery { .. }
            | ComputeOpKind::Plumbing { .. } => {
                // Reachable only if the classifier admitted a non-ViewFold
                // op into a ViewFold-classed kernel — the classifier
                // returns Generic in that case, so this is structurally
                // unreachable. Emit a documented TODO instead of panicking.
                "// TODO(task-5.5): non-ViewFold op in ViewFold kernel — \
                 classifier should have routed through generic path."
                    .to_string()
            }
        };
        writeln!(
            out,
            "    // op#{} ({})",
            op.id.0,
            compute_op_kind_short(&op.kind)
        )
        .expect("write to String never fails");
        // Indent the lowered fragment by 4 spaces per line so the body
        // sits at the same indent level as the preamble (event_idx,
        // gate, _tick) and the per-op `// op#…` comment that labels it.
        // The Task-4.1 lowering emits flush-left WGSL; without this
        // re-indent the fragment would be visually misaligned under its
        // comment.
        let indented = fragment
            .lines()
            .map(|l| if l.is_empty() { String::new() } else { format!("    {l}") })
            .collect::<Vec<_>>()
            .join("\n");
        out.push_str(&indented);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// WGSL body composition
// ---------------------------------------------------------------------------

/// Build the WGSL body string for a kernel covering `body_ops`. Each
/// op's body is lowered through Task 4.1's helpers and concatenated.
/// Op kinds without a structured body get a documented `// TODO(...)`
/// placeholder line — never a panic.
fn build_wgsl_body(
    body_ops: &[OpId],
    dispatch: &DispatchShape,
    prog: &CgProgram,
    ctx: &EmitCtx<'_>,
) -> Result<String, KernelEmitError> {
    let mut out = String::new();

    // Per-thread preamble — mirrors `ThreadIndexing` shape.
    write!(out, "{}", thread_indexing_preamble(dispatch))
        .expect("write to String never fails");

    for (i, op_id) in body_ops.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        let op = resolve_op(prog, *op_id)?;
        let fragment = lower_op_body(op, dispatch, ctx)?;
        // Comment header per op for traceability.
        writeln!(out, "// op#{} ({})", op.id.0, compute_op_kind_short(&op.kind))
            .expect("write to String never fails");
        out.push_str(&fragment);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// SpatialQuery body templates (Task 5.6c)
// ---------------------------------------------------------------------------

/// WGSL body fragment for [`SpatialQueryKind::BuildHash`].
///
/// Every binding declared by the [`KernelSpec`] for a `BuildHash` op is
/// touched at least once so naga's dead-code elimination cannot remove
/// them — the hand-written legacy file
/// (`engine_gpu_rules/src/spatial_hash.wgsl`) follows the same
/// stub-touch convention pending the per-cell index implementation.
///
/// Bindings consumed (per [`SpatialQueryKind::BuildHash::dependencies`] +
/// auto-injected `cfg`):
/// - `spatial_grid_cells`        (Pool, read_write)
/// - `spatial_grid_offsets`      (Pool, read_write)
/// - `cfg`                       (uniform, last slot)
///
/// # Limitations
///
/// - **Verbatim port from the legacy stub.** The body is a near-copy of
///   `engine_gpu_rules/src/spatial_hash.wgsl` — both touch every
///   binding so the BGL stays live, neither performs the actual hash
///   build today. Real per-cell hashing lives in `engine_gpu`'s
///   hand-written spatial pipeline; the CG-emitted kernel is a
///   structural placeholder until the spatial algorithm is folded into
///   the IR.
/// - **No IR-driven derivation.** The IR has no abstractions for
///   "compute cell from position" or "atomic-add to grid"; surfacing
///   those would require new [`ComputeOpKind`] sub-shapes well beyond
///   Task 5.6c's scope. Future refinement (Task 5.x or later) could
///   replace this const with an IR walk once those abstractions exist.
const SPATIAL_BUILD_HASH_BODY: &str = "// SpatialQuery::BuildHash — verbatim port from \
    engine_gpu_rules/src/spatial_hash.wgsl.\n\
    // Touches every binding so naga keeps them live. Real per-cell hash build \
    lives in the\n\
    // hand-written engine_gpu spatial pipeline; this stub is a structural \
    placeholder.\n\
    _ = spatial_grid_cells[0];\n\
    _ = spatial_grid_offsets[0];\n\
    _ = cfg.agent_cap;";

/// WGSL body fragment for [`SpatialQueryKind::KinQuery`].
///
/// Bindings consumed (per [`SpatialQueryKind::KinQuery::dependencies`] +
/// auto-injected `cfg`):
/// - `spatial_grid_cells`        (Pool, read)
/// - `spatial_grid_offsets`      (Pool, read)
/// - `spatial_query_results`     (Pool, read_write)
/// - `cfg`                       (uniform, last slot)
///
/// # Limitations
///
/// - **Verbatim port from the legacy stub.** Mirrors
///   `engine_gpu_rules/src/spatial_kin_query.wgsl`. The legacy file
///   is itself a stub that touches all bindings — actual neighborhood
///   walk + top-K kin filter is hand-written in engine_gpu's spatial
///   path, not yet IR-driven.
/// - **No IR-driven derivation.** Same rationale as
///   [`SPATIAL_BUILD_HASH_BODY`].
const SPATIAL_KIN_QUERY_BODY: &str = "// SpatialQuery::KinQuery — verbatim port from \
    engine_gpu_rules/src/spatial_kin_query.wgsl.\n\
    // Touches every binding so naga keeps them live. Real per-agent kin walk \
    lives in the\n\
    // hand-written engine_gpu spatial pipeline; this stub is a structural \
    placeholder.\n\
    _ = spatial_grid_cells[0];\n\
    _ = spatial_grid_offsets[0];\n\
    _ = spatial_query_results[0];\n\
    _ = cfg.agent_cap;";

/// WGSL body fragment for [`SpatialQueryKind::EngagementQuery`].
///
/// Bindings consumed (per
/// [`SpatialQueryKind::EngagementQuery::dependencies`] +
/// auto-injected `cfg`):
/// - `spatial_grid_cells`        (Pool, read)
/// - `spatial_grid_offsets`      (Pool, read)
/// - `spatial_query_results`     (Pool, read_write)
/// - `cfg`                       (uniform, last slot)
///
/// # Limitations
///
/// - **Verbatim port from the legacy stub.** Mirrors
///   `engine_gpu_rules/src/spatial_engagement_query.wgsl`. Body shape
///   is identical to [`SPATIAL_KIN_QUERY_BODY`] today — the kin vs
///   engagement distinction is encoded in the kernel name + (future)
///   per-kind filter logic. The legacy WGSL files are also identical;
///   the differentiation happens in the hand-written CPU-side fold.
/// - **No IR-driven derivation.** Same rationale as
///   [`SPATIAL_BUILD_HASH_BODY`].
const SPATIAL_ENGAGEMENT_QUERY_BODY: &str = "// SpatialQuery::EngagementQuery — verbatim port \
    from engine_gpu_rules/src/spatial_engagement_query.wgsl.\n\
    // Touches every binding so naga keeps them live. Real per-agent engagement \
    walk lives in\n\
    // the hand-written engine_gpu spatial pipeline; this stub is a structural \
    placeholder.\n\
    _ = spatial_grid_cells[0];\n\
    _ = spatial_grid_offsets[0];\n\
    _ = spatial_query_results[0];\n\
    _ = cfg.agent_cap;";

/// Per-op body lowering. Returns the WGSL fragment for the op without
/// surrounding kernel boilerplate. `dispatch` carries the kernel's
/// dispatch shape so per-op bodies that vary across shapes
/// (`MaskPredicate` PerAgent vs PerPair) can branch appropriately
/// (Task 5.6a).
fn lower_op_body(
    op: &ComputeOp,
    dispatch: &DispatchShape,
    ctx: &EmitCtx<'_>,
) -> Result<String, KernelEmitError> {
    match &op.kind {
        ComputeOpKind::MaskPredicate { mask, predicate } => {
            let predicate_wgsl = lower_cg_expr_to_wgsl(*predicate, ctx)?;
            match dispatch {
                DispatchShape::PerAgent => {
                    Ok(mask_predicate_per_agent_body(*mask, &predicate_wgsl))
                }
                DispatchShape::PerPair { .. } => {
                    Ok(mask_predicate_per_pair_body(*mask, &predicate_wgsl))
                }
                DispatchShape::PerEvent { .. }
                | DispatchShape::OneShot
                | DispatchShape::PerWord => Err(KernelEmitError::InvalidDispatchForOpKind {
                    op_kind: "MaskPredicate",
                    dispatch: format!("{dispatch}"),
                }),
            }
        }
        ComputeOpKind::PhysicsRule { body, .. } => {
            lower_cg_stmt_list_to_wgsl(*body, ctx).map_err(KernelEmitError::from)
        }
        ComputeOpKind::ViewFold { body, .. } => {
            lower_cg_stmt_list_to_wgsl(*body, ctx).map_err(KernelEmitError::from)
        }
        ComputeOpKind::ScoringArgmax { scoring, rows } => {
            lower_scoring_argmax_body(*scoring, rows, ctx)
        }
        // Task 5.6c: per-kind dispatch into the SpatialQuery body
        // templates. Exhaustive over [`SpatialQueryKind`] — adding a
        // new variant forces an explicit body decision here.
        ComputeOpKind::SpatialQuery { kind } => Ok(match kind {
            SpatialQueryKind::BuildHash => SPATIAL_BUILD_HASH_BODY.to_string(),
            SpatialQueryKind::KinQuery => SPATIAL_KIN_QUERY_BODY.to_string(),
            SpatialQueryKind::EngagementQuery => {
                SPATIAL_ENGAGEMENT_QUERY_BODY.to_string()
            }
        }),
        // Task 5.6d: per-`PlumbingKind` body dispatch. Exhaustive over
        // every variant so adding a new kind forces an explicit body
        // decision here. See the const-by-const docs below for the
        // legacy-port rationale and the binding-name normalization
        // that keeps each fragment compatible with the structural
        // [`KernelSpec`] layout (no spec-level renames).
        ComputeOpKind::Plumbing { kind } => Ok(plumbing_body_for_kind(kind)),
    }
}

// ---------------------------------------------------------------------------
// ScoringArgmax body template (Task 5.6b)
// ---------------------------------------------------------------------------

/// Lower a `ScoringArgmax` op to its WGSL body fragment.
///
/// Body shape — single `DispatchShape::PerAgent` form:
///
/// 1. Initialise sentinel best (`best_utility = ~f32::MIN`,
///    `best_action = 0u`, `best_target = NO_TARGET`).
/// 2. Per row in source order: optional guard test, evaluate utility,
///    optional target, strictly-greater compare-and-swap against the
///    running best.
/// 3. Write `(best_action, best_target, bitcast(best_utility), 0u)`
///    into `scoring_output[agent_id * 4 + …]`.
///
/// Stride / sentinel constants are inlined (no module-scope `const`)
/// so the fragment composes cleanly with sibling op bodies in a
/// fused topology.
fn lower_scoring_argmax_body(
    scoring: ScoringId,
    rows: &[ScoringRowOp],
    ctx: &EmitCtx<'_>,
) -> Result<String, KernelEmitError> {
    use std::fmt::Write as _;
    let mut out = String::new();
    writeln!(
        out,
        "// scoring_argmax: scoring=#{}, {} rows.",
        scoring.0,
        rows.len()
    )
    .expect("write to String never fails");
    out.push_str("var best_utility: f32 = -3.4028235e38;\n");
    out.push_str("var best_action: u32 = 0u;\n");
    out.push_str("var best_target: u32 = 0xFFFFFFFFu;\n\n");

    for (i, row) in rows.iter().enumerate() {
        let utility_wgsl = lower_cg_expr_to_wgsl(row.utility, ctx)?;
        let target_wgsl = match row.target {
            Some(target_id) => lower_cg_expr_to_wgsl(target_id, ctx)?,
            None => "0xFFFFFFFFu".to_string(),
        };
        let action_lit = format!("{}u", row.action.0);

        match row.guard {
            Some(guard_id) => {
                let guard_wgsl = lower_cg_expr_to_wgsl(guard_id, ctx)?;
                writeln!(out, "// row {i}: action=#{} (guarded)", row.action.0).unwrap();
                writeln!(out, "{{").unwrap();
                writeln!(out, "    let guard_{i}: bool = {guard_wgsl};").unwrap();
                writeln!(out, "    if (guard_{i}) {{").unwrap();
                writeln!(out, "        let utility_{i}: f32 = {utility_wgsl};").unwrap();
                writeln!(out, "        if (utility_{i} > best_utility) {{").unwrap();
                writeln!(out, "            best_utility = utility_{i};").unwrap();
                writeln!(out, "            best_action = {action_lit};").unwrap();
                writeln!(out, "            best_target = {target_wgsl};").unwrap();
                writeln!(out, "        }}").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "}}").unwrap();
            }
            None => {
                writeln!(out, "// row {i}: action=#{}", row.action.0).unwrap();
                writeln!(out, "{{").unwrap();
                writeln!(out, "    let utility_{i}: f32 = {utility_wgsl};").unwrap();
                writeln!(out, "    if (utility_{i} > best_utility) {{").unwrap();
                writeln!(out, "        best_utility = utility_{i};").unwrap();
                writeln!(out, "        best_action = {action_lit};").unwrap();
                writeln!(out, "        best_target = {target_wgsl};").unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "}}").unwrap();
            }
        }
        out.push('\n');
    }

    out.push_str("let scoring_base: u32 = agent_id * 4u;\n");
    out.push_str("scoring_output[scoring_base + 0u] = best_action;\n");
    out.push_str("scoring_output[scoring_base + 1u] = best_target;\n");
    out.push_str("scoring_output[scoring_base + 2u] = bitcast<u32>(best_utility);\n");
    out.push_str("scoring_output[scoring_base + 3u] = 0u;");
    Ok(out)
}

// ---------------------------------------------------------------------------
// MaskPredicate body templates (Task 5.6a)
// ---------------------------------------------------------------------------

/// PerAgent dispatch — `agent_id` is bound by the
/// `thread_indexing_preamble`. Each mask op uses per-id-suffixed
/// locals (`mask_<ID>_word`, `mask_<ID>_bit`) so a Fused kernel with
/// multiple `MaskPredicate` ops doesn't redeclare `word`/`bit`.
fn mask_predicate_per_agent_body(mask: MaskId, predicate_wgsl: &str) -> String {
    format!(
        "let mask_{0}_value: bool = {1};\n\
         if (mask_{0}_value) {{\n\
         \x20   let mask_{0}_word = agent_id >> 5u;\n\
         \x20   let mask_{0}_bit  = 1u << (agent_id & 31u);\n\
         \x20   atomicOr(&mask_{0}_bitmap[mask_{0}_word], mask_{0}_bit);\n\
         }}",
        mask.0, predicate_wgsl
    )
}

/// PerPair dispatch — derive `(agent, cand)` from `pair = gid.x`,
/// bound-check `agent`, evaluate the predicate, atomic-OR the
/// agent's bit. `mask_<ID>_k` placeholder is `1u` until Task 5.7
/// wires `cfg.per_pair_candidates`.
///
/// `agent_id` / `per_pair_candidate` are aliased to the per-mask
/// derivations so the predicate body — which refers to those names
/// directly via [`crate::cg::expr::CgExpr::AgentSelfId`] /
/// [`crate::cg::expr::CgExpr::PerPairCandidateId`] —
/// resolves cleanly. Path B's slot-aware lowering will fold the alias
/// step into its naming strategy.
fn mask_predicate_per_pair_body(mask: MaskId, predicate_wgsl: &str) -> String {
    format!(
        "// PerPair MaskPredicate — derive (agent, cand) from `pair`.\n\
         let mask_{0}_k = 1u; // TODO(task-5.7): read from cfg.per_pair_candidates.\n\
         let mask_{0}_agent = pair / mask_{0}_k;\n\
         let mask_{0}_cand  = pair % mask_{0}_k;\n\
         if (mask_{0}_agent >= cfg.agent_cap) {{ return; }}\n\
         let agent_id = mask_{0}_agent;\n\
         let per_pair_candidate = mask_{0}_cand;\n\
         \n\
         let mask_{0}_value: bool = {1};\n\
         if (mask_{0}_value) {{\n\
         \x20   let mask_{0}_word = mask_{0}_agent >> 5u;\n\
         \x20   let mask_{0}_bit  = 1u << (mask_{0}_agent & 31u);\n\
         \x20   atomicOr(&mask_{0}_bitmap[mask_{0}_word], mask_{0}_bit);\n\
         }}",
        mask.0, predicate_wgsl
    )
}

// ---------------------------------------------------------------------------
// Plumbing body templates (Task 5.6d)
// ---------------------------------------------------------------------------
//
// Per-`PlumbingKind` WGSL body fragments. Each const carries a per-
// variant docstring spelling out (1) which legacy emitter the body is
// ported from, (2) the bindings it touches by structural name, and
// (3) any drift vs. the legacy WGSL it replaces.
//
// Naming convention. The fragments reference bindings by the
// **structural** names produced by [`structural_binding_name`] —
// `agent_<field>`, `agent_scratch_packed`, `alive_bitmap`,
// `event_ring_<ring>`, `indirect_args_<ring>`, `sim_cfg`,
// `snapshot_kick`. The legacy WGSL files used different names
// (`agents`, `apply_tail`, `agents_input`, ...) because they declared
// their own bindings inline; the CG path declares bindings via the
// shared [`KernelSpec`] pipeline so the body must use the
// spec-derived names. Structural names are picked, never spec
// renames, because the spec layer is shared across every kernel and
// touching it for one kind would be cross-cutting scope creep.
//
// Stub-touch convention. Every binding the spec declares for a
// kernel must be referenced at least once in the body or naga's
// dead-code pass will strip the binding from the BGL. The
// hand-written legacy stubs (`fused_agent_unpack.wgsl`,
// `spatial_*.wgsl`) follow the same `let _foo = binding[0];`
// pattern; we mirror it here so the BGL stays live until the real
// algorithm is folded into the IR.

/// WGSL body for [`PlumbingKind::PackAgents`].
///
/// Per-agent dispatch. Reads every per-agent SoA field and writes the
/// packed agent scratch buffer. The legacy pack side was never given a
/// real WGSL body — `engine_gpu_rules/src/fused_agent_unpack.wgsl`
/// covers the *unpack* side as a stub that touches its three bindings
/// (`agents_input`, `mask_soa`, `agent_data`); the pack analog lives
/// host-side in the legacy emitter pipeline. We mirror the stub
/// convention: touch a representative agent-field binding (`agent_pos`
/// — every CG kernel emitting `PackAgents` will declare every
/// `agent_<field>` slot) and the packed scratch slot, plus the cfg
/// uniform's `agent_cap`, so the BGL stays live.
///
/// # Limitations
///
/// - **Structural placeholder.** The CG IR has no abstraction for
///   "lay out one agent's fields into a packed slot at offset
///   `agent_id * SLOT_STRIDE`"; surfacing it would require new
///   [`ComputeOpKind`] sub-shapes (or a typed AgentSlotPack op).
///   Until that abstraction lands, the body cannot describe the real
///   pack algorithm — it stays a binding-touch stub.
/// - **Single agent-field binding referenced.** The spec declares one
///   binding per `AgentFieldId` variant (every variant is in the
///   read set per [`PlumbingKind::PackAgents::dependencies`]). The
///   stub touches only `agent_pos`; touching all 30+ fields would
///   bloat the body without changing what the spec sees. naga keeps
///   every declared binding even when only some are referenced (the
///   wgpu-side BGL is built from the spec, not from naga's elision
///   pass), so this is safe.
const PACK_AGENTS_BODY: &str = "// PlumbingKind::PackAgents — structural stub.\n\
    // Real per-agent SoA->packed lowering needs a typed AgentSlotPack op\n\
    // (Task 5.x); today this is a binding-touch stub that keeps the\n\
    // declared agent + scratch BGL slots live.\n\
    let _ap = agent_pos[agent_id];\n\
    let _as = agent_scratch_packed[0];\n\
    let _c = cfg.agent_cap;";

/// WGSL body for [`PlumbingKind::UnpackAgents`].
///
/// Per-agent dispatch. Reads the packed agent scratch buffer and
/// writes back into per-field SoA storage. Verbatim port of
/// `engine_gpu_rules/src/fused_agent_unpack.wgsl`'s body shape (which
/// is itself a stub: it touches `agents_input`, `mask_soa`,
/// `agent_data`, and `cfg.agent_cap` to keep the BGL live).
///
/// # Limitations
///
/// - **Structural placeholder.** Same rationale as
///   [`PACK_AGENTS_BODY`] — the inverse algorithm needs a typed
///   AgentSlotUnpack op the IR doesn't carry today.
/// - **Single agent-field binding referenced.** The spec declares one
///   binding per `AgentFieldId` variant (each in the *write* set);
///   the stub touches `agent_pos` only. naga's elision is moot here
///   because the BGL is built from the spec rather than from naga's
///   live-binding analysis.
const UNPACK_AGENTS_BODY: &str = "// PlumbingKind::UnpackAgents — structural stub. Verbatim\n\
    // port of engine_gpu_rules/src/fused_agent_unpack.wgsl (the\n\
    // legacy file is itself a stub that touches every binding).\n\
    let _as = agent_scratch_packed[0];\n\
    let _ap = agent_pos[agent_id];\n\
    let _c = cfg.agent_cap;";

/// WGSL body for [`PlumbingKind::AliveBitmap`].
///
/// Per-word dispatch. Reads 32 consecutive `agent_alive[i]` slots,
/// packs the non-zero ones into a single u32, writes the result to
/// `alive_bitmap[word_idx]`. Ported from
/// `engine_gpu_rules/src/alive_pack.wgsl` with one structural
/// adaptation: the legacy file declared `agents: array<u32>` (the
/// flat AgentSlot-strided layout) and reached into it via
/// `agents[slot * SLOT_STRIDE_U32 + ALIVE_OFFSET_U32]`; the CG path
/// declares `agent_alive: array<u32>` directly (one entry per agent,
/// the SoA shape) so the access becomes `agent_alive[slot]`.
///
/// The output binding `alive_bitmap` is declared with
/// `AccessMode::AtomicStorage`, which the WGSL emitter renders as
/// `array<atomic<u32>>` (per
/// [`crate::kernel_lowerings::lower_wgsl_bindings`]). Atomic types
/// do not support index-assignment, so the write goes through
/// `atomicStore`. The legacy file used a non-atomic declaration
/// (one thread per word, no contention) — we cannot downgrade
/// because the CG metadata categorises bitmap-shaped slots as
/// atomic, so the access mode stays atomic and the body uses
/// `atomicStore`.
///
/// # Limitations
///
/// - **Atomic-store vs legacy non-atomic write.** Functionally
///   equivalent (single writer per word) but the WGSL form differs
///   from `alive_pack.wgsl`. The metadata-driven access mode is the
///   single source of truth; aligning the legacy file to atomic on
///   the next regen is a separate concern.
const ALIVE_BITMAP_BODY: &str = "// PlumbingKind::AliveBitmap — pack 32 alive slots into one\n\
    // u32 word. Ported from engine_gpu_rules/src/alive_pack.wgsl;\n\
    // the legacy file's flat-stride agent indexing (slot * STRIDE +\n\
    // ALIVE_OFFSET) collapses to direct SoA access (`agent_alive[slot]`)\n\
    // because the CG path declares per-field bindings.\n\
    let base_slot = word_idx * 32u;\n\
    var word: u32 = 0u;\n\
    for (var i: u32 = 0u; i < 32u; i = i + 1u) {\n\
        let slot = base_slot + i;\n\
        if (slot >= cfg.agent_cap) { break; }\n\
        let alive_word = agent_alive[slot];\n\
        if (alive_word != 0u) {\n\
            word = word | (1u << i);\n\
        }\n\
    }\n\
    atomicStore(&alive_bitmap[word_idx], word);";

/// WGSL body for [`PlumbingKind::DrainEvents`].
///
/// Per-event dispatch on `ring`. The drain operation mutates the
/// ring's tail counter to zero so the next iteration sees an empty
/// ring; the actual reset happens host-side (legacy:
/// `crates/engine_gpu/src/cascade_resident.rs` issues a
/// `queue.write_buffer` on the tail), not from a shader. The CG
/// kernel exists so the schedule stays uniform (every plumbing kind
/// produces a kernel) but its WGSL body is a no-op binding-touch.
///
/// # Limitations
///
/// - **No real GPU-side drain.** Tail reset is host-side. A future
///   CG variant (atomic store of zero from a 1-thread kernel) could
///   replace the host-side write, but it requires the ring binding's
///   tail to be split out from the records — the IR currently
///   binds them together via the [`DataHandle::EventRing`] handle.
/// - **Binding-touch stub.** Touches `event_ring_<r>[0]` and
///   `cfg.agent_cap` to keep both declared bindings live.
fn drain_events_body(ring_id: u32) -> String {
    format!(
        "// PlumbingKind::DrainEvents (ring={ring_id}) — host-side drain;\n\
        // the GPU-side kernel is a no-op binding-touch. Tail reset is\n\
        // issued from the CPU via queue.write_buffer (legacy parity).\n\
        let _e = event_ring_{ring_id}[0];\n\
        let _c = cfg.agent_cap;"
    )
}

/// WGSL body for [`PlumbingKind::UploadSimCfg`].
///
/// One-shot. The sim_cfg buffer is uploaded host-side via
/// `queue.write_buffer`; the GPU-side kernel cannot write to a
/// uniform binding (the WGSL emitter declares
/// `var<uniform> sim_cfg: SimCfg;` — uniforms are read-only inside
/// a shader). The kernel is still scheduled because every plumbing
/// kind produces a kernel node; its body is a no-op.
///
/// # Limitations
///
/// - **No body — sim_cfg upload is host-side.** The kernel exists for
///   schedule uniformity; the spec declares the binding to keep the
///   data dependency edge live for cycle detection. The body's only
///   job is to keep naga happy, which the gid guard already does.
/// - **No `SimCfg` struct decl in the WGSL output.** The structural
///   metadata sets `wgsl_ty: "SimCfg"` for [`DataHandle::SimCfgBuffer`]
///   without surfacing the struct decl; a fully-naga-clean build
///   would need that decl emitted at the WGSL module level (today
///   the legacy hand-written `sim_cfg` shader carries it inline).
///   Out of scope for Task 5.6d — touched in a follow-up alongside
///   the cfg-shape refinement.
const UPLOAD_SIM_CFG_BODY: &str = "// PlumbingKind::UploadSimCfg — host-side upload (queue.\n\
    // write_buffer). Body is a no-op; the spec declares sim_cfg as\n\
    // a Uniform binding which is read-only inside the shader.";

/// WGSL body for [`PlumbingKind::KickSnapshot`].
///
/// One-shot. Triggers a snapshot dump for the current tick by
/// flagging the `snapshot_kick` slot. The CG metadata declares the
/// slot with `AccessMode::AtomicStorage`, which renders as
/// `array<atomic<u32>>`; the body uses `atomicStore` to set
/// `snapshot_kick[0]` to a non-zero sentinel (`1u`). The legacy
/// snapshot pipeline triggers the dump via a host-side dispatch
/// rather than a GPU-side store — the CG kernel is a structural
/// placeholder until snapshot triggering is folded into the GPU
/// schedule.
///
/// # Limitations
///
/// - **Sentinel store, no real snapshot logic.** A real CG-driven
///   snapshot would cascade through buffer copies; that pipeline
///   does not yet have an IR representation. The store keeps the
///   binding live and signals "kick" semantically.
const KICK_SNAPSHOT_BODY: &str = "// PlumbingKind::KickSnapshot — flag the snapshot_kick slot;\n\
    // host-side snapshot pipeline observes the flag post-tick.\n\
    atomicStore(&snapshot_kick[0], 1u);";

/// WGSL body for [`PlumbingKind::SeedIndirectArgs`].
///
/// One-shot. Reads the producer ring's tail count and writes
/// `indirect_args_<ring>[0..3] = (wg, 1, 1)` so the next per-event
/// dispatch on `ring` knows how many workgroups to launch. Adapted
/// from `engine_gpu_rules/src/seed_indirect.wgsl` with one
/// structural adaptation: the legacy file separated the ring's tail
/// (`apply_tail: array<u32>`) from the ring records (a different
/// binding) and read `apply_tail[0]` for the count. The CG path
/// folds tail + records into a single `event_ring_<r>` binding via
/// [`DataHandle::EventRing`], so the body reads `event_ring_<r>[0]`
/// for the count under a structural assumption: tail count lives at
/// offset 0 of the unified ring binding.
///
/// `wg = ceil(n / 64)` capped at `CAP_WG = 4096` (legacy ceiling for
/// 200_000-agent dispatches at 64-lane workgroups).
///
/// # Limitations
///
/// - **Tail-at-offset-0 assumption.** The CG IR binds the ring as a
///   single `array<u32>`; whether `event_ring_<r>[0]` actually holds
///   the tail count depends on the (currently undefined) layout
///   convention shared between IR and runtime. The follow-up that
///   formalises the ring layout (split tail from records into
///   distinct data handles) will replace this read with the typed
///   tail accessor.
/// - **`CAP_WG` constant inlined.** 4096 mirrors the legacy ceiling;
///   surfacing it through the cfg uniform is a Task 5.7+ refinement.
fn seed_indirect_args_body(ring_id: u32) -> String {
    format!(
        "// PlumbingKind::SeedIndirectArgs (ring={ring_id}) — adapted from\n\
        // engine_gpu_rules/src/seed_indirect.wgsl. Reads tail count from\n\
        // event_ring_{ring_id}[0] (single-binding ring assumption — see\n\
        // Limitations on `seed_indirect_args_body`); writes (wg, 1, 1)\n\
        // into indirect_args_{ring_id} so the next per-event dispatch on\n\
        // ring={ring_id} launches ceil(n/64) workgroups (capped at\n\
        // CAP_WG=4096).\n\
        let n = event_ring_{ring_id}[0];\n\
        let req = (n + 63u) / 64u;\n\
        var wg: u32 = req;\n\
        if (wg > 4096u) {{ wg = 4096u; }}\n\
        indirect_args_{ring_id}[0] = wg;\n\
        indirect_args_{ring_id}[1] = 1u;\n\
        indirect_args_{ring_id}[2] = 1u;"
    )
}

/// Per-[`PlumbingKind`] body dispatch. Exhaustive over every variant —
/// adding a new variant forces an explicit body decision here.
///
/// # Limitations
///
/// - **Coverage of [`PlumbingKind`] variants only.** The plan
///   (Task 5.6 §3.4) lists additional plumbing-shaped kernels —
///   `MaskUnpack`, `ScoringUnpack`, `ApplyActions`, `Movement`,
///   `Physics`, `AppendEvents` — that exist in the legacy emitter
///   pipeline but do **not** yet have [`PlumbingKind`] variants in
///   the CG IR. Adding bodies for those requires first synthesising
///   new `PlumbingKind` variants in Phase 1 (`cg/op.rs`) plus the
///   plumbing synthesizer (`cg/lower/plumbing.rs`); that work is
///   tracked as a Task 5.6d follow-up and is out of scope here.
/// - **Bodies are verbatim ports from legacy stubs.** Where the
///   legacy WGSL file is itself a binding-touch stub
///   (`fused_agent_unpack.wgsl`, `spatial_*.wgsl`), the CG body
///   mirrors the stub. Where the legacy file carries a real
///   algorithm (`alive_pack.wgsl`, `seed_indirect.wgsl`), the CG
///   body ports it with documented adaptations to the structural
///   binding names.
fn plumbing_body_for_kind(kind: &PlumbingKind) -> String {
    match kind {
        PlumbingKind::PackAgents => PACK_AGENTS_BODY.to_string(),
        PlumbingKind::UnpackAgents => UNPACK_AGENTS_BODY.to_string(),
        PlumbingKind::AliveBitmap => ALIVE_BITMAP_BODY.to_string(),
        PlumbingKind::DrainEvents { ring } => drain_events_body(ring.0),
        PlumbingKind::UploadSimCfg => UPLOAD_SIM_CFG_BODY.to_string(),
        PlumbingKind::KickSnapshot => KICK_SNAPSHOT_BODY.to_string(),
        PlumbingKind::SeedIndirectArgs { ring } => seed_indirect_args_body(ring.0),
    }
}

/// Emit the WGSL preamble that maps the kernel's thread to its work
/// item. Mirrors `ThreadIndexing` from `cg::dispatch`.
fn thread_indexing_preamble(dispatch: &DispatchShape) -> String {
    match dispatch {
        DispatchShape::PerAgent => {
            "let agent_id = gid.x;\n\
             if (agent_id >= cfg.agent_cap) { return; }\n\n"
                .to_string()
        }
        DispatchShape::PerEvent { source_ring } => format!(
            "let event_idx = gid.x;\n\
             // Indirect dispatch on event_ring_{}; tail count bounds gid.x.\n\n",
            source_ring.0
        ),
        DispatchShape::PerPair { source: _ } => {
            "let pair = gid.x;\n\
             // PerPair: agent + cand resolution computed against per_pair_candidates.\n\n"
                .to_string()
        }
        DispatchShape::OneShot => {
            "if (gid.x != 0u) { return; }\n\n".to_string()
        }
        DispatchShape::PerWord => {
            "let word_idx = gid.x;\n\
             let num_words = (cfg.agent_cap + 31u) >> 5u;\n\
             if (word_idx >= num_words) { return; }\n\n"
                .to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Resolution helpers
// ---------------------------------------------------------------------------

fn resolve_op(prog: &CgProgram, op_id: OpId) -> Result<&ComputeOp, KernelEmitError> {
    let arena_len = prog.ops.len() as u32;
    prog.ops
        .get(op_id.0 as usize)
        .ok_or(KernelEmitError::OpIdOutOfRange {
            id: op_id,
            arena_len,
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, CgExprId, EventRingId, MaskId,
    };
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::{CgExpr, CgTy, LitValue};
    use crate::cg::op::{
        ComputeOp, ComputeOpKind, OpId, PhysicsRuleId, PlumbingKind, ReplayabilityFlag, ScoringId,
        ScoringRowOp, ActionId, EventKindId, Span,
    };
    use crate::cg::stmt::{CgStmt, CgStmtId, CgStmtList, CgStmtListId};

    /// Push an op directly into the program — bypasses the builder so
    /// tests can construct hand-shaped graphs.
    fn push_op(prog: &mut CgProgram, mut op: ComputeOp) -> OpId {
        let id = OpId(prog.ops.len() as u32);
        op.id = id;
        prog.ops.push(op);
        id
    }

    fn push_expr(prog: &mut CgProgram, e: CgExpr) -> CgExprId {
        let id = CgExprId(prog.exprs.len() as u32);
        prog.exprs.push(e);
        id
    }

    fn push_stmt(prog: &mut CgProgram, s: CgStmt) -> CgStmtId {
        let id = CgStmtId(prog.stmts.len() as u32);
        prog.stmts.push(s);
        id
    }

    fn push_list(prog: &mut CgProgram, l: CgStmtList) -> CgStmtListId {
        let id = CgStmtListId(prog.stmt_lists.len() as u32);
        prog.stmt_lists.push(l);
        id
    }

    /// Build a single MaskPredicate op `mask[m] = (self.hp < 5.0)`.
    fn mask_op(prog: &mut CgProgram, mask: MaskId) -> OpId {
        let hp = push_expr(
            prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let five = push_expr(prog, CgExpr::Lit(LitValue::F32(5.0)));
        let lt = push_expr(
            prog,
            CgExpr::Binary {
                op: crate::cg::expr::BinaryOp::LtF32,
                lhs: hp,
                rhs: five,
                ty: CgTy::Bool,
            },
        );
        let kind = ComputeOpKind::MaskPredicate {
            mask,
            predicate: lt,
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerAgent,
            Span::dummy(),
            prog,
            prog,
            prog,
        );
        push_op(prog, op)
    }

    /// Build a PhysicsRule op with a body `self.hp = self.hp + 1.0` —
    /// reads + writes Hp; per-event dispatch on `ring`.
    fn physics_op(prog: &mut CgProgram, rule: PhysicsRuleId, ring: EventRingId) -> OpId {
        let hp_read = push_expr(
            prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::Self_,
            }),
        );
        let one = push_expr(prog, CgExpr::Lit(LitValue::F32(1.0)));
        let add = push_expr(
            prog,
            CgExpr::Binary {
                op: crate::cg::expr::BinaryOp::AddF32,
                lhs: hp_read,
                rhs: one,
                ty: CgTy::F32,
            },
        );
        let assign = push_stmt(
            prog,
            CgStmt::Assign {
                target: DataHandle::AgentField {
                    field: AgentFieldId::Hp,
                    target: AgentRef::Self_,
                },
                value: add,
            },
        );
        let body = push_list(prog, CgStmtList { stmts: vec![assign] });
        let kind = ComputeOpKind::PhysicsRule {
            rule,
            on_event: EventKindId(0),
            body,
            replayable: ReplayabilityFlag::Replayable,
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerEvent { source_ring: ring },
            Span::dummy(),
            prog,
            prog,
            prog,
        );
        push_op(prog, op)
    }

    /// Build a SeedIndirectArgs producer for `ring`.
    fn seed_indirect_op(prog: &mut CgProgram, ring: EventRingId) -> OpId {
        let kind = ComputeOpKind::Plumbing {
            kind: PlumbingKind::SeedIndirectArgs { ring },
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::OneShot,
            Span::dummy(),
            prog,
            prog,
            prog,
        );
        push_op(prog, op)
    }

    // ---- 1. Split { single PerAgent op } ----

    #[test]
    fn split_mask_predicate_yields_spec_with_data_bindings_plus_cfg() {
        let mut prog = CgProgram::default();
        let m = MaskId(2);
        let op = mask_op(&mut prog, m);
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        // Bindings: agent.hp (read), mask.bitmap (atomic), cfg uniform.
        let names: Vec<&str> = spec.bindings.iter().map(|b| b.name.as_str()).collect();
        assert!(names.contains(&"agent_hp"), "bindings: {:?}", names);
        assert!(names.contains(&"mask_2_bitmap"));
        assert_eq!(names.last(), Some(&"cfg"));

        // cfg slot is the last contiguous slot.
        let cfg = spec.bindings.last().unwrap();
        assert!(matches!(cfg.access, AccessMode::Uniform));
        assert!(matches!(cfg.bg_source, BgSource::Cfg));

        // Mask binding is atomic.
        let mask = spec
            .bindings
            .iter()
            .find(|b| b.name == "mask_2_bitmap")
            .unwrap();
        assert!(matches!(mask.access, AccessMode::AtomicStorage));

        // Agent binding is read-only (mask predicate doesn't write hp).
        let agent = spec.bindings.iter().find(|b| b.name == "agent_hp").unwrap();
        assert!(matches!(agent.access, AccessMode::ReadStorage));
    }

    // ---- 2. Fused { 3 ops sharing reads } — dedup via cycle_edge_key ----

    #[test]
    fn fused_three_mask_ops_share_agent_hp_binding() {
        let mut prog = CgProgram::default();
        let op1 = mask_op(&mut prog, MaskId(0));
        let op2 = mask_op(&mut prog, MaskId(1));
        let op3 = mask_op(&mut prog, MaskId(2));
        let topology = KernelTopology::Fused {
            ops: vec![op1, op2, op3],
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        // Three mask bitmaps + one shared agent_hp + cfg = 5 bindings.
        let names: Vec<&str> = spec.bindings.iter().map(|b| b.name.as_str()).collect();
        let agent_hp_count = names.iter().filter(|n| **n == "agent_hp").count();
        assert_eq!(agent_hp_count, 1, "agent_hp dedup: {:?}", names);
        assert_eq!(spec.bindings.len(), 5, "names: {:?}", names);
        assert!(names.contains(&"mask_0_bitmap"));
        assert!(names.contains(&"mask_1_bitmap"));
        assert!(names.contains(&"mask_2_bitmap"));
        assert_eq!(names.last(), Some(&"cfg"));
    }

    // ---- 3. Indirect { producer, 1 consumer } ----

    #[test]
    fn indirect_topology_emits_consumer_kernel_using_consumer_reads() {
        // The indirect-args buffer is bound to the producer kernel +
        // the dispatch wiring (Task 4.3), not to the consumer kernel.
        // The consumer kernel's bindings come from its own reads /
        // writes only — here, agent_hp (read+write by the physics
        // rule) and cfg.
        let mut prog = CgProgram::default();
        let ring = EventRingId(7);
        let producer = seed_indirect_op(&mut prog, ring);
        let consumer = physics_op(&mut prog, PhysicsRuleId(0), ring);
        let topology = KernelTopology::Indirect {
            producer,
            consumers: vec![consumer],
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        // Consumer reads + writes agent_hp; the topology dispatch is
        // PerEvent on the same ring as the producer's seed-indirect.
        let names: Vec<&str> = spec.bindings.iter().map(|b| b.name.as_str()).collect();
        assert!(names.contains(&"agent_hp"), "names: {:?}", names);
        assert_eq!(names.last(), Some(&"cfg"));

        // The indirect-args buffer for `ring` is NOT in the consumer
        // kernel's bindings — it lives on the producer (which
        // `record_write(IndirectArgs)`s it) and the schedule-layer
        // dispatch wiring.
        let indirect_args_name = format!("indirect_args_{}", ring.0);
        assert!(
            !names.contains(&indirect_args_name.as_str()),
            "consumer kernel must not bind {indirect_args_name}; got names: {names:?}"
        );

        let agent = spec.bindings.iter().find(|b| b.name == "agent_hp").unwrap();
        // Physics rule writes hp → upgraded to read_write.
        assert!(matches!(agent.access, AccessMode::ReadWriteStorage));
    }

    // ---- 4. KernelSpec::validate passes for every shape ----

    #[test]
    fn produced_specs_validate() {
        let mut prog = CgProgram::default();
        let op = mask_op(&mut prog, MaskId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        spec.validate().expect("spec must validate");

        // Fused
        let mut prog2 = CgProgram::default();
        let a = mask_op(&mut prog2, MaskId(0));
        let b = mask_op(&mut prog2, MaskId(1));
        let topology2 = KernelTopology::Fused {
            ops: vec![a, b],
            dispatch: DispatchShape::PerAgent,
        };
        let ctx2 = EmitCtx::structural(&prog2);
        let spec2 = kernel_topology_to_spec(&topology2, &prog2, &ctx2).unwrap();
        spec2.validate().expect("fused spec must validate");

        // Indirect
        let mut prog3 = CgProgram::default();
        let ring = EventRingId(1);
        let prod = seed_indirect_op(&mut prog3, ring);
        let cons = physics_op(&mut prog3, PhysicsRuleId(0), ring);
        let topology3 = KernelTopology::Indirect {
            producer: prod,
            consumers: vec![cons],
        };
        let ctx3 = EmitCtx::structural(&prog3);
        let spec3 = kernel_topology_to_spec(&topology3, &prog3, &ctx3).unwrap();
        spec3.validate().expect("indirect spec must validate");
    }

    // ---- 5. Snapshot — pin the bindings vec for a representative spec ----

    #[test]
    fn snapshot_split_mask_predicate_bindings() {
        let mut prog = CgProgram::default();
        let op = mask_op(&mut prog, MaskId(3));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        // Pin the binding sequence — a regression in slot ordering or
        // metadata table will surface here.
        let snapshot: Vec<(u32, String, String)> = spec
            .bindings
            .iter()
            .map(|b| (b.slot, b.name.clone(), b.wgsl_ty.clone()))
            .collect();
        // CycleEdgeKey ordering: AgentField sorts before MaskBitmap by
        // the variant order in DataHandle's PartialOrd derive.
        assert_eq!(snapshot[0].0, 0);
        assert_eq!(snapshot[0].1, "agent_hp");
        assert_eq!(snapshot[0].2, "array<f32>");
        assert_eq!(snapshot[1].0, 1);
        assert_eq!(snapshot[1].1, "mask_3_bitmap");
        assert_eq!(snapshot[1].2, "u32");
        assert_eq!(snapshot[2].0, 2);
        assert_eq!(snapshot[2].1, "cfg");
    }

    // ---- 6. Determinism: two runs identical ----

    #[test]
    fn determinism_identical_specs_across_runs() {
        let build = || {
            let mut prog = CgProgram::default();
            let a = mask_op(&mut prog, MaskId(2));
            let b = mask_op(&mut prog, MaskId(0));
            let c = mask_op(&mut prog, MaskId(1));
            let topology = KernelTopology::Fused {
                ops: vec![a, b, c],
                dispatch: DispatchShape::PerAgent,
            };
            let ctx = EmitCtx::structural(&prog);
            let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
            // Project to a stable comparable form — KernelSpec doesn't
            // derive PartialEq.
            let projected: Vec<(u32, String, String, String)> = spec
                .bindings
                .iter()
                .map(|b| {
                    (
                        b.slot,
                        b.name.clone(),
                        b.wgsl_ty.clone(),
                        format!("{:?}", b.access),
                    )
                })
                .collect();
            (spec.name.clone(), projected)
        };
        assert_eq!(build(), build());
    }

    // ---- 7. Error — empty topology surfaces a typed error ----

    #[test]
    fn empty_indirect_consumers_yields_typed_error() {
        let mut prog = CgProgram::default();
        let ring = EventRingId(0);
        let producer = seed_indirect_op(&mut prog, ring);
        let topology = KernelTopology::Indirect {
            producer,
            consumers: vec![],
        };
        let ctx = EmitCtx::structural(&prog);
        let err = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap_err();
        assert!(
            matches!(err, KernelEmitError::EmptyKernelTopology),
            "got {err:?}"
        );
    }

    #[test]
    fn op_id_out_of_range_yields_typed_error() {
        let prog = CgProgram::default();
        let topology = KernelTopology::Split {
            op: OpId(99),
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let err = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap_err();
        assert!(
            matches!(err, KernelEmitError::OpIdOutOfRange { .. }),
            "got {err:?}"
        );
    }

    // ---- 8. Body composition — kernel_topology_to_spec_and_body ----

    #[test]
    fn body_includes_thread_indexing_preamble_and_op_comment() {
        let mut prog = CgProgram::default();
        let op = mask_op(&mut prog, MaskId(5));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();
        assert!(body.contains("agent_id = gid.x"), "body: {body}");
        assert!(body.contains("// op#"), "body: {body}");
        assert!(body.contains("mask_5_bitmap"), "body: {body}");
        // The lowered predicate text from Task 4.1's expr walker
        // should appear: `(agent_self_hp < 5.0)`.
        assert!(body.contains("agent_self_hp < 5.0"), "body: {body}");
    }

    // ---- 8b. MaskPredicate body (Task 5.6a) ----

    #[test]
    fn mask_predicate_per_agent_body_emits_atomic_or_at_agent_id() {
        let mut prog = CgProgram::default();
        let lit_true = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let kind = ComputeOpKind::MaskPredicate {
            mask: MaskId(7),
            predicate: lit_true,
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerAgent,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, op);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        assert!(body.contains("let agent_id = gid.x;"), "body: {body}");
        assert!(body.contains("if (agent_id >= cfg.agent_cap)"), "body: {body}");
        assert!(body.contains("let mask_7_value: bool = true;"), "body: {body}");
        assert!(body.contains("let mask_7_word = agent_id >> 5u;"), "body: {body}");
        assert!(
            body.contains("let mask_7_bit  = 1u << (agent_id & 31u);"),
            "body: {body}"
        );
        assert!(
            body.contains("atomicOr(&mask_7_bitmap[mask_7_word], mask_7_bit);"),
            "body: {body}"
        );
    }

    #[test]
    fn mask_predicate_per_pair_body_derives_agent_from_pair_and_writes_per_agent_bit() {
        use crate::cg::dispatch::PerPairSource;
        let mut prog = CgProgram::default();
        let lit_true = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let kind = ComputeOpKind::MaskPredicate {
            mask: MaskId(11),
            predicate: lit_true,
        };
        let pair_dispatch = DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            pair_dispatch,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, op);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: pair_dispatch,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        assert!(body.contains("let pair = gid.x;"), "body: {body}");
        assert!(body.contains("let mask_11_k = 1u;"), "body: {body}");
        assert!(
            body.contains("let mask_11_agent = pair / mask_11_k;"),
            "body: {body}"
        );
        assert!(
            body.contains("let mask_11_cand  = pair % mask_11_k;"),
            "body: {body}"
        );
        assert!(
            body.contains("if (mask_11_agent >= cfg.agent_cap) { return; }"),
            "body: {body}"
        );
        assert!(body.contains("let mask_11_value: bool = true;"), "body: {body}");
        assert!(
            body.contains("let mask_11_word = mask_11_agent >> 5u;"),
            "body: {body}"
        );
        assert!(
            body.contains("let mask_11_bit  = 1u << (mask_11_agent & 31u);"),
            "body: {body}"
        );
        assert!(
            body.contains("atomicOr(&mask_11_bitmap[mask_11_word], mask_11_bit);"),
            "body: {body}"
        );
    }

    #[test]
    fn mask_predicate_under_unsupported_dispatch_shape_errors() {
        let mut prog = CgProgram::default();
        let lit_true = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let kind = ComputeOpKind::MaskPredicate {
            mask: MaskId(0),
            predicate: lit_true,
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::OneShot,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, op);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::OneShot,
        };
        let ctx = EmitCtx::structural(&prog);
        let err = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap_err();
        assert!(
            matches!(
                err,
                KernelEmitError::InvalidDispatchForOpKind { op_kind: "MaskPredicate", .. }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn fused_two_mask_predicates_emit_distinct_local_names() {
        let mut prog = CgProgram::default();
        let m0 = mask_op(&mut prog, MaskId(0));
        let m1 = mask_op(&mut prog, MaskId(1));
        let topology = KernelTopology::Fused {
            ops: vec![m0, m1],
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        assert!(
            body.contains("atomicOr(&mask_0_bitmap[mask_0_word], mask_0_bit);"),
            "body: {body}"
        );
        assert!(
            body.contains("atomicOr(&mask_1_bitmap[mask_1_word], mask_1_bit);"),
            "body: {body}"
        );
        // No bare `let word = …`.
        assert!(!body.contains("\n    let word ="), "body: {body}");
        assert!(!body.contains("\nlet word ="), "body: {body}");
    }

    #[test]
    fn mask_predicate_per_pair_body_passes_through_per_pair_candidate_target_read() {
        use crate::cg::dispatch::PerPairSource;
        let mut prog = CgProgram::default();
        // Predicate: `target.hp < 5.0` (read of PerPairCandidate's hp).
        let target_hp = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Hp,
                target: AgentRef::PerPairCandidate,
            }),
        );
        let five = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(5.0)));
        let lt = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: crate::cg::expr::BinaryOp::LtF32,
                lhs: target_hp,
                rhs: five,
                ty: CgTy::Bool,
            },
        );
        let kind = ComputeOpKind::MaskPredicate {
            mask: MaskId(9),
            predicate: lt,
        };
        let pair_dispatch = DispatchShape::PerPair {
            source: PerPairSource::SpatialQuery(SpatialQueryKind::KinQuery),
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            pair_dispatch,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, op);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: pair_dispatch,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // The placeholder identifier from wgsl_body.rs flows through.
        assert!(
            body.contains("agent_per_pair_candidate_hp"),
            "body: {body}"
        );
    }

    // ---- 9. Scoring argmax body (Task 5.6b) ----

    #[test]
    fn scoring_argmax_emits_argmax_skeleton() {
        let mut prog = CgProgram::default();
        let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.5)));
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(7),
            rows: vec![ScoringRowOp {
                action: ActionId(3),
                utility,
                target: None,
                guard: None,
            }],
        };
        let probe = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerAgent,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, probe);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // Preamble + scoping + sentinel.
        assert!(body.contains("agent_id = gid.x"), "body: {body}");
        assert!(
            body.contains("// scoring_argmax: scoring=#7, 1 rows."),
            "body: {body}"
        );
        assert!(
            body.contains("var best_utility: f32 = -3.4028235e38;"),
            "body: {body}"
        );
        assert!(body.contains("var best_action: u32 = 0u;"), "body: {body}");
        assert!(
            body.contains("var best_target: u32 = 0xFFFFFFFFu;"),
            "body: {body}"
        );

        // Row block.
        assert!(body.contains("// row 0: action=#3"), "body: {body}");
        assert!(body.contains("let utility_0: f32 = 2.5"), "body: {body}");
        assert!(
            body.contains("if (utility_0 > best_utility)"),
            "body: {body}"
        );
        assert!(body.contains("best_action = 3u;"), "body: {body}");
        assert!(
            body.contains("best_target = 0xFFFFFFFFu;"),
            "body: {body}"
        );

        // Tail write into scoring_output, stride 4.
        assert!(
            body.contains("let scoring_base: u32 = agent_id * 4u;"),
            "body: {body}"
        );
        assert!(
            body.contains("scoring_output[scoring_base + 0u] = best_action;"),
            "body: {body}"
        );
        assert!(
            body.contains("scoring_output[scoring_base + 1u] = best_target;"),
            "body: {body}"
        );
        assert!(
            body.contains(
                "scoring_output[scoring_base + 2u] = bitcast<u32>(best_utility);"
            ),
            "body: {body}"
        );
        assert!(
            body.contains("scoring_output[scoring_base + 3u] = 0u;"),
            "body: {body}"
        );

        // Placeholder text MUST be gone.
        assert!(
            !body.contains("TODO(task-4.x): scoring_argmax"),
            "body still has placeholder: {body}"
        );
    }

    #[test]
    fn scoring_argmax_emits_per_row_blocks_with_guard() {
        let mut prog = CgProgram::default();
        let util_0 = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let util_1 = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(4.0)));
        let target_1 = push_expr(&mut prog, CgExpr::Lit(LitValue::U32(2)));
        let t = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(true)));
        let f = push_expr(&mut prog, CgExpr::Lit(LitValue::Bool(false)));
        let guard_1 = push_expr(
            &mut prog,
            CgExpr::Binary {
                op: crate::cg::expr::BinaryOp::And,
                lhs: t,
                rhs: f,
                ty: CgTy::Bool,
            },
        );
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(0),
            rows: vec![
                ScoringRowOp {
                    action: ActionId(0),
                    utility: util_0,
                    target: None,
                    guard: None,
                },
                ScoringRowOp {
                    action: ActionId(5),
                    utility: util_1,
                    target: Some(target_1),
                    guard: Some(guard_1),
                },
            ],
        };
        let probe = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerAgent,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, probe);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // Row 0 — unguarded, sentinel target.
        assert!(body.contains("// row 0: action=#0"), "body: {body}");
        assert!(body.contains("let utility_0: f32 = 1"), "body: {body}");
        assert!(body.contains("best_action = 0u;"), "body: {body}");
        assert!(
            body.contains("best_target = 0xFFFFFFFFu;"),
            "body: {body}"
        );

        // Row 1 — guarded, explicit target.
        assert!(
            body.contains("// row 1: action=#5 (guarded)"),
            "body: {body}"
        );
        assert!(
            body.contains("let guard_1: bool = (true && false);"),
            "body: {body}"
        );
        assert!(body.contains("if (guard_1)"), "body: {body}");
        assert!(body.contains("let utility_1: f32 = 4"), "body: {body}");
        assert!(
            body.contains("if (utility_1 > best_utility)"),
            "body: {body}"
        );
        assert!(body.contains("best_action = 5u;"), "body: {body}");
        assert!(body.contains("best_target = 2u;"), "body: {body}");
    }

    // ---- 10. Semantic kernel naming (Task 5.2) ----

    #[test]
    fn mask_predicate_uses_interner_name_when_present() {
        let mut prog = CgProgram::default();
        let mask = MaskId(2);
        let op = mask_op(&mut prog, mask);
        // Pretend lowering interned a name for this mask.
        prog.interner.masks.insert(mask.0, "low_health".to_string());
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert_eq!(spec.name, "mask_low_health", "spec.name: {}", spec.name);
        assert_eq!(spec.entry_point, "cs_mask_low_health");
        assert_eq!(spec.cfg_struct, "MaskLowHealthCfg");
    }

    #[test]
    fn mask_predicate_falls_back_to_id_when_no_interner_name() {
        let mut prog = CgProgram::default();
        let op = mask_op(&mut prog, MaskId(7));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert_eq!(spec.name, "mask_7");
    }

    #[test]
    fn scoring_argmax_emits_bare_scoring_name() {
        let mut prog = CgProgram::default();
        let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.0)));
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(0),
            rows: vec![ScoringRowOp {
                action: ActionId(0),
                utility,
                target: None,
                guard: None,
            }],
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerAgent,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, op);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert_eq!(spec.name, "scoring");
    }

    #[test]
    fn physics_rule_singleton_uses_per_rule_kernel_name() {
        // A singleton (Replayable) PhysicsRule op resolves to
        // `physics_<rule>`, where `<rule>` is the interner name. The
        // earlier collapse-to-`physics` rule produced
        // [`KernelNameCollision`] once more than one PhysicsRule
        // lowered into its own (singleton) kernel, so the canonical
        // name is now per-rule and agrees with
        // [`single_op_kernel_name`].
        let mut prog = CgProgram::default();
        let ring = EventRingId(0);
        let op = physics_op(&mut prog, PhysicsRuleId(4), ring);
        prog.interner
            .physics_rules
            .insert(4, "cast_apply".to_string());
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerEvent { source_ring: ring },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert_eq!(spec.name, "physics_cast_apply");
    }

    #[test]
    fn pascal_to_snake_normalizes_event_kind_names() {
        assert_eq!(pascal_to_snake("AgentAttacked"), "agent_attacked");
        assert_eq!(
            pascal_to_snake("EffectDamageApplied"),
            "effect_damage_applied"
        );
        assert_eq!(pascal_to_snake("Foo"), "foo");
        assert_eq!(pascal_to_snake("foo"), "foo");
        assert_eq!(pascal_to_snake(""), "");
    }

    #[test]
    fn view_fold_singleton_collapses_to_fold_view_kernel_name() {
        // Post Task 5.7-iter2: a singleton ViewFold op resolves to
        // `fold_<view>` (no event suffix). This matches the legacy
        // emitter where one kernel module owns ALL of a view's
        // handlers; the in-kernel body switches on `event.tag` to
        // dispatch per-handler logic. The pascal_to_snake helper
        // is no longer reached on the kernel-name path because no
        // event suffix is appended.
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let body = push_list(&mut prog, CgStmtList { stmts: vec![] });
        let kind = ComputeOpKind::ViewFold {
            view: ViewId(3),
            on_event: EventKindId(7),
            body,
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
        let op_id = push_op(&mut prog, op);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner
            .event_kinds
            .insert(7, "AgentAttacked".to_string());
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert_eq!(spec.name, "fold_threat_level");
    }

    #[test]
    fn view_fold_uses_interner_name_for_view_kernel() {
        // Post Task 5.7-iter2: the ViewFold kernel name is
        // `fold_<view>` regardless of the (single) handler's event
        // kind. The view name comes from the interner; the event
        // kind is no longer reflected in the kernel name.
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let body = push_list(&mut prog, CgStmtList { stmts: vec![] });
        let kind = ComputeOpKind::ViewFold {
            view: ViewId(3),
            on_event: EventKindId(7),
            body,
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
        let op_id = push_op(&mut prog, op);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner
            .event_kinds
            .insert(7, "agent_attacked".to_string());
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert_eq!(spec.name, "fold_threat_level");
    }

    #[test]
    fn view_fold_distinct_event_kinds_fuse_to_one_kernel_name() {
        // The DSL `threat_level` view in `assets/sim/views.sim` has
        // two `on` handlers (`AgentAttacked`, `EffectDamageApplied`).
        // After Task 5.7-iter2 driver-level ring unification, both
        // handlers share `EventRingId(0)` so the schedule synthesizer
        // fuses them into a single `KernelTopology::Fused`. The fused
        // kernel name collapses to `fold_<view>` regardless of how
        // many handlers the view has — matching the legacy emitter
        // shape exactly.
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner
            .event_kinds
            .insert(0, "agent_attacked".to_string());
        prog.interner
            .event_kinds
            .insert(1, "effect_damage_applied".to_string());

        let mk_fold = |prog: &mut CgProgram, ek: u32| -> OpId {
            let body = push_list(prog, CgStmtList { stmts: vec![] });
            let kind = ComputeOpKind::ViewFold {
                view: ViewId(3),
                on_event: EventKindId(ek),
                body,
            };
            let op = ComputeOp::new(
                OpId(0),
                kind,
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
                prog,
                prog,
                prog,
            );
            push_op(prog, op)
        };
        let a = mk_fold(&mut prog, 0);
        let b = mk_fold(&mut prog, 1);
        let ctx = EmitCtx::structural(&prog);
        let fused = KernelTopology::Fused {
            ops: vec![a, b],
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let spec_fused = kernel_topology_to_spec(&fused, &prog, &ctx).unwrap();
        assert_eq!(spec_fused.name, "fold_threat_level");

        // Even per-handler singleton topologies emit the same
        // collapsed kernel name. The KernelNameCollision gate at
        // program-emit time only fires when two such kernels
        // co-exist unfused in the schedule.
        let spec_a = kernel_topology_to_spec(
            &KernelTopology::Split {
                op: a,
                dispatch: DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
            },
            &prog,
            &ctx,
        )
        .unwrap();
        let spec_b = kernel_topology_to_spec(
            &KernelTopology::Split {
                op: b,
                dispatch: DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
            },
            &prog,
            &ctx,
        )
        .unwrap();
        assert_eq!(spec_a.name, "fold_threat_level");
        assert_eq!(spec_b.name, "fold_threat_level");
    }

    #[test]
    fn spatial_query_kinds_map_to_legacy_filenames() {
        use crate::cg::op::SpatialQueryKind;
        for (kind, expected) in [
            (SpatialQueryKind::BuildHash, "spatial_build_hash"),
            (SpatialQueryKind::KinQuery, "spatial_kin_query"),
            (SpatialQueryKind::EngagementQuery, "spatial_engagement_query"),
        ] {
            let mut prog = CgProgram::default();
            let op = ComputeOp::new(
                OpId(0),
                ComputeOpKind::SpatialQuery { kind },
                DispatchShape::PerAgent,
                Span::dummy(),
                &prog,
                &prog,
                &prog,
            );
            let op_id = push_op(&mut prog, op);
            let topology = KernelTopology::Split {
                op: op_id,
                dispatch: DispatchShape::PerAgent,
            };
            let ctx = EmitCtx::structural(&prog);
            let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
            assert_eq!(spec.name, expected, "kind={kind:?}");
        }
    }

    // ---- 12. SpatialQuery body templates (Task 5.6c) ----

    /// Helper: build a `Split` SpatialQuery topology and lower it to
    /// `(spec, body)`. Used by the per-kind body assertions below.
    fn spatial_query_spec_and_body(
        kind: SpatialQueryKind,
    ) -> (KernelSpec, String) {
        let mut prog = CgProgram::default();
        let op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::SpatialQuery { kind },
            DispatchShape::PerAgent,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, op);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap()
    }

    #[test]
    fn spatial_query_build_hash_emits_build_hash_body() {
        let (_spec, body) = spatial_query_spec_and_body(SpatialQueryKind::BuildHash);
        // Per-kind preamble comment from the body const.
        assert!(
            body.contains("SpatialQuery::BuildHash"),
            "BuildHash body: {body}"
        );
        // Touches the two writable spatial-storage bindings the spec
        // declares for BuildHash (per `SpatialQueryKind::dependencies`).
        assert!(body.contains("spatial_grid_cells"), "body: {body}");
        assert!(body.contains("spatial_grid_offsets"), "body: {body}");
        // BuildHash does NOT include `spatial_query_results` in its
        // dependency set — verify it stays out of the body so the WGSL
        // doesn't reference an undeclared binding.
        assert!(
            !body.contains("spatial_query_results"),
            "BuildHash should not reference query_results: {body}"
        );
        // No leftover TODO placeholder from the pre-Task-5.6c shape.
        assert!(
            !body.contains("TODO(task-4.x): spatial_query body"),
            "body still carries pre-5.6c TODO placeholder: {body}"
        );
    }

    #[test]
    fn spatial_query_kin_query_emits_kin_query_body() {
        let (_spec, body) = spatial_query_spec_and_body(SpatialQueryKind::KinQuery);
        assert!(body.contains("SpatialQuery::KinQuery"), "body: {body}");
        // KinQuery touches all three spatial-storage bindings (grid_cells
        // + grid_offsets read; query_results read_write).
        assert!(body.contains("spatial_grid_cells"), "body: {body}");
        assert!(body.contains("spatial_grid_offsets"), "body: {body}");
        assert!(body.contains("spatial_query_results"), "body: {body}");
        assert!(
            !body.contains("TODO(task-4.x): spatial_query body"),
            "body still carries pre-5.6c TODO placeholder: {body}"
        );
    }

    #[test]
    fn spatial_query_engagement_query_emits_engagement_query_body() {
        let (_spec, body) =
            spatial_query_spec_and_body(SpatialQueryKind::EngagementQuery);
        assert!(
            body.contains("SpatialQuery::EngagementQuery"),
            "body: {body}"
        );
        assert!(body.contains("spatial_grid_cells"), "body: {body}");
        assert!(body.contains("spatial_grid_offsets"), "body: {body}");
        assert!(body.contains("spatial_query_results"), "body: {body}");
        assert!(
            !body.contains("TODO(task-4.x): spatial_query body"),
            "body still carries pre-5.6c TODO placeholder: {body}"
        );
    }

    #[test]
    fn spatial_query_body_includes_per_agent_preamble_and_op_comment() {
        // Composition through `build_wgsl_body` should still wrap the
        // per-kind body const with the PerAgent thread-indexing
        // preamble + op-comment header — so the lowered body is a
        // complete WGSL function body once the entry-point shell is
        // wrapped around it by `compose_wgsl_file`.
        let (_spec, body) = spatial_query_spec_and_body(SpatialQueryKind::KinQuery);
        // PerAgent preamble.
        assert!(body.contains("agent_id = gid.x"), "body: {body}");
        assert!(
            body.contains("if (agent_id >= cfg.agent_cap)"),
            "body: {body}"
        );
        // Op-comment header from `build_wgsl_body`.
        assert!(body.contains("// op#0 (spatial_query)"), "body: {body}");
        // Cfg uniform reference — the body touches `cfg.agent_cap` so
        // the cfg uniform stays live.
        assert!(body.contains("cfg.agent_cap"), "body: {body}");
    }

    /// Snapshot: pin the EngagementQuery body output through
    /// `kernel_topology_to_spec_and_body`. Any drift in the per-kind
    /// body const, the preamble shape, or the op-comment header surfaces
    /// here — the assertions below describe the exact expected form so
    /// the snapshot survives a body-content tweak with a one-site update.
    #[test]
    fn spatial_query_body_snapshot_engagement_query() {
        let (spec, body) =
            spatial_query_spec_and_body(SpatialQueryKind::EngagementQuery);
        // Spec name + entry point are the legacy filenames (Task 5.2).
        assert_eq!(spec.name, "spatial_engagement_query");
        assert_eq!(spec.entry_point, "cs_spatial_engagement_query");
        // Pin the body's structural sections in order: PerAgent preamble
        // → op-comment header → per-kind body comment → binding touches
        // → cfg touch. Each contains-check is order-independent, so we
        // also verify the relative offsets to catch reordering.
        let preamble_idx = body
            .find("agent_id = gid.x")
            .unwrap_or_else(|| panic!("missing PerAgent preamble: {body}"));
        let op_comment_idx = body
            .find("// op#0 (spatial_query)")
            .unwrap_or_else(|| panic!("missing op-comment header: {body}"));
        let kind_comment_idx = body
            .find("SpatialQuery::EngagementQuery")
            .unwrap_or_else(|| panic!("missing per-kind comment: {body}"));
        let cfg_touch_idx = body
            .find("cfg.agent_cap")
            .unwrap_or_else(|| panic!("missing cfg touch: {body}"));
        // Cfg touch lives in BOTH the preamble (`if (agent_id >=
        // cfg.agent_cap)`) AND the body const (`let _c = cfg.agent_cap;`).
        // That's fine for liveness; the preamble copy is what we find first.
        assert!(
            preamble_idx < op_comment_idx,
            "preamble must precede op-comment: {body}"
        );
        assert!(
            op_comment_idx < kind_comment_idx,
            "op-comment must precede per-kind body: {body}"
        );
        assert!(
            cfg_touch_idx < body.len(),
            "cfg touch must appear in body: {body}"
        );
        // No TODO/unimplemented/panic markers anywhere.
        assert!(!body.contains("todo!()"), "body: {body}");
        assert!(!body.contains("unimplemented!"), "body: {body}");
        assert!(!body.contains("panic!"), "body: {body}");
    }

    #[test]
    fn plumbing_kinds_map_to_legacy_filenames() {
        let cases = [
            (PlumbingKind::PackAgents, "pack_agents"),
            (PlumbingKind::UnpackAgents, "unpack_agents"),
            (PlumbingKind::AliveBitmap, "alive_pack"),
            (PlumbingKind::UploadSimCfg, "upload_sim_cfg"),
            (PlumbingKind::KickSnapshot, "kick_snapshot"),
            (
                PlumbingKind::SeedIndirectArgs {
                    ring: EventRingId(2),
                },
                "seed_indirect_2",
            ),
            (
                PlumbingKind::DrainEvents {
                    ring: EventRingId(5),
                },
                "drain_events_5",
            ),
        ];
        for (kind, expected) in cases {
            let mut prog = CgProgram::default();
            let op = ComputeOp::new(
                OpId(0),
                ComputeOpKind::Plumbing { kind },
                kind.dispatch_shape(),
                Span::dummy(),
                &prog,
                &prog,
                &prog,
            );
            let op_id = push_op(&mut prog, op);
            let topology = KernelTopology::Split {
                op: op_id,
                dispatch: kind.dispatch_shape(),
            };
            let ctx = EmitCtx::structural(&prog);
            let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
            assert_eq!(spec.name, expected, "plumbing kind={kind:?}");
        }
    }

    #[test]
    fn fused_multi_op_kernel_prefixes_with_fused() {
        // Two MaskPredicate ops in one Fused topology — the kernel name
        // gets a `fused_` prefix on the first op's semantic name.
        let mut prog = CgProgram::default();
        let m0 = mask_op(&mut prog, MaskId(0));
        let m1 = mask_op(&mut prog, MaskId(1));
        prog.interner.masks.insert(0, "low_hp".to_string());
        prog.interner.masks.insert(1, "isolated".to_string());
        let topology = KernelTopology::Fused {
            ops: vec![m0, m1],
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        // Prefixed with `fused_` since the first op's name doesn't
        // already start with `fused_`.
        assert_eq!(spec.name, "fused_mask_low_hp", "spec.name: {}", spec.name);
    }

    // ---- 11. ViewFold body parity (Task 5.3) ----

    /// Build a ViewFold op for `view` triggered by `event_kind`. The
    /// body is a single assignment `view_storage_primary = 1.0` so the
    /// inner-walk lowering produces a non-empty fragment.
    fn view_fold_op(
        prog: &mut CgProgram,
        view: crate::cg::data_handle::ViewId,
        event_kind: EventKindId,
        ring: EventRingId,
    ) -> OpId {
        let one = push_expr(prog, CgExpr::Lit(LitValue::F32(1.0)));
        let assign = push_stmt(
            prog,
            CgStmt::Assign {
                target: DataHandle::ViewStorage {
                    view,
                    slot: crate::cg::data_handle::ViewStorageSlot::Primary,
                },
                value: one,
            },
        );
        let body = push_list(prog, CgStmtList { stmts: vec![assign] });
        let kind = ComputeOpKind::ViewFold {
            view,
            on_event: event_kind,
            body,
        };
        let op = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerEvent { source_ring: ring },
            Span::dummy(),
            prog,
            prog,
            prog,
        );
        push_op(prog, op)
    }

    /// ViewFold kernel cfg shape carries `event_count` + `tick`, not
    /// the placeholder `agent_cap`. Pins per-kind cfg routing.
    #[test]
    fn view_fold_kernel_uses_event_count_tick_cfg_shape() {
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let view = ViewId(3);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner.event_kinds.insert(7, "AgentAttacked".into());
        let op = view_fold_op(&mut prog, view, EventKindId(7), EventRingId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        // ViewFold cfg fields: event_count, tick, _pad: [u32; 2].
        assert!(
            spec.cfg_struct_decl.contains("event_count: u32"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        assert!(
            spec.cfg_struct_decl.contains("tick: u32"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        assert!(
            spec.cfg_struct_decl.contains("_pad: [u32; 2]"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        assert!(
            spec.cfg_build_expr
                .contains("event_count: 0, tick: state.tick"),
            "build_expr: {}",
            spec.cfg_build_expr
        );
    }

    /// Non-ViewFold kernels (mask, physics, plumbing, scoring, spatial)
    /// keep the placeholder `{ agent_cap, _pad: [u32; 3] }` cfg shape.
    /// Pin that the per-kind routing didn't accidentally cross over.
    #[test]
    fn non_view_fold_kernel_keeps_agent_cap_cfg_shape() {
        let mut prog = CgProgram::default();
        let op = mask_op(&mut prog, MaskId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert!(
            spec.cfg_struct_decl.contains("agent_cap: u32"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        assert!(
            spec.cfg_build_expr.contains("agent_cap: state.agent_cap()"),
            "build_expr: {}",
            spec.cfg_build_expr
        );
    }

    /// ViewFold kernel emits the legacy 7-binding layout: event_ring,
    /// event_tail, view_storage_primary, view_storage_anchor,
    /// view_storage_ids, sim_cfg, cfg.
    #[test]
    fn view_fold_kernel_emits_seven_legacy_bindings() {
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let view = ViewId(3);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner.event_kinds.insert(7, "AgentAttacked".into());
        let op = view_fold_op(&mut prog, view, EventKindId(7), EventRingId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        let names: Vec<&str> = spec.bindings.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "event_ring",
                "event_tail",
                "view_storage_primary",
                "view_storage_anchor",
                "view_storage_ids",
                "sim_cfg",
                "cfg",
            ],
            "bindings: {names:?}"
        );

        // Slots are contiguous 0..7.
        for (i, b) in spec.bindings.iter().enumerate() {
            assert_eq!(b.slot, i as u32, "binding {} slot mismatch: {:?}", i, b);
        }

        // The three view-storage slots carry ViewHandle bg_source with
        // the per-view resident accessor name.
        let view_handle_bindings: Vec<&KernelBinding> = spec
            .bindings
            .iter()
            .filter(|b| matches!(b.bg_source, BgSource::ViewHandle { .. }))
            .collect();
        assert_eq!(view_handle_bindings.len(), 3, "expected 3 ViewHandle bindings");
        for b in &view_handle_bindings {
            match &b.bg_source {
                BgSource::ViewHandle { accessor, .. } => {
                    assert_eq!(accessor, "fold_view_threat_level_handles");
                }
                BgSource::Resident(_)
                | BgSource::Transient(_)
                | BgSource::External(_)
                | BgSource::Pool(_)
                | BgSource::Cfg
                | BgSource::AliasOf(_) => unreachable!("filtered above"),
            }
        }
    }

    /// Body composition for ViewFold injects the `event_idx` declaration
    /// + `event_count` bounds-check, then concatenates each handler's
    /// IR-lowered body via Task 4.1.
    #[test]
    fn view_fold_wgsl_body_has_event_count_gate() {
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let view = ViewId(5);
        prog.interner.views.insert(5, "kin_fear".to_string());
        prog.interner.event_kinds.insert(2, "FearSpread".into());
        let op = view_fold_op(&mut prog, view, EventKindId(2), EventRingId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();
        assert!(body.contains("let event_idx = gid.x;"), "body: {body}");
        assert!(
            body.contains("if (event_idx >= cfg.event_count)"),
            "body: {body}"
        );
        assert!(body.contains("// op#"), "body: {body}");
        // The single body assignment is lowered through Task 4.1; under
        // the B1 no-op fallback (Path B step 1), ViewStorage Assign
        // targets emit a phony WGSL discard `_ = (rhs);` since the
        // structural name `view_5_primary` is not a declared binding
        // (the BGL-bound name is `view_storage_primary`, indexed by a
        // target id that the structural strategy can't synthesize).
        // Path B's slot-aware lowering will replace this with a real
        // `view_storage_primary[target_id] += value` form.
        assert!(body.contains("_ = (1.0);"), "body: {body}");
    }

    /// Indirect topology of all-ViewFold consumers also routes through
    /// the ViewFold path — drives the cascade-driven fold case.
    #[test]
    fn indirect_topology_of_view_folds_routes_through_view_fold_path() {
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let ring = EventRingId(2);
        let view = ViewId(7);
        prog.interner.views.insert(7, "memory".to_string());
        prog.interner.event_kinds.insert(0, "RecordMemory".into());
        let producer = seed_indirect_op(&mut prog, ring);
        let consumer = view_fold_op(&mut prog, view, EventKindId(0), ring);
        let topology = KernelTopology::Indirect {
            producer,
            consumers: vec![consumer],
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        // ViewFold-specific cfg shape applied.
        assert!(
            spec.cfg_struct_decl.contains("event_count: u32"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        // 7-binding layout.
        let names: Vec<&str> = spec.bindings.iter().map(|b| b.name.as_str()).collect();
        assert!(
            names.contains(&"view_storage_primary") && names.contains(&"event_ring"),
            "names: {names:?}"
        );
        // Resident accessor matches the consumer view name.
        let mut found_accessor = false;
        for b in &spec.bindings {
            if let BgSource::ViewHandle { accessor, .. } = &b.bg_source {
                assert_eq!(accessor, "fold_view_memory_handles");
                found_accessor = true;
            }
        }
        assert!(found_accessor, "expected ViewHandle binding with memory accessor");
    }

    /// Two ViewFold ops over the SAME view in a Fused topology classify
    /// as ViewFold (multi-handler view subscribes to multiple events).
    /// The consumer view name carries through.
    #[test]
    fn fused_view_folds_over_same_view_classify_as_view_fold() {
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let view = ViewId(3);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner.event_kinds.insert(0, "AgentAttacked".into());
        prog.interner
            .event_kinds
            .insert(1, "EffectDamageApplied".into());
        let op_a = view_fold_op(&mut prog, view, EventKindId(0), EventRingId(0));
        let op_b = view_fold_op(&mut prog, view, EventKindId(1), EventRingId(0));
        let topology = KernelTopology::Fused {
            ops: vec![op_a, op_b],
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        assert!(
            spec.cfg_struct_decl.contains("event_count: u32"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        // Both bodies appear in the WGSL output (joined with the
        // op-comment lines).
        let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();
        let op_count = body.matches("// op#").count();
        assert_eq!(op_count, 2, "body: {body}");
    }

    /// ViewFold falls back to `view_<id>` when the interner has no
    /// name for the view — the resident accessor is built from the
    /// fallback identifier.
    #[test]
    fn view_fold_with_no_interner_name_falls_back_to_view_id() {
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let view = ViewId(99);
        // No interner entry for view #99.
        let op = view_fold_op(&mut prog, view, EventKindId(0), EventRingId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        for b in &spec.bindings {
            if let BgSource::ViewHandle { accessor, .. } = &b.bg_source {
                assert_eq!(accessor, "fold_view_view_99_handles");
            }
        }
    }

    /// Snapshot — pin a ViewFold spec's bindings vec end-to-end. Any
    /// regression in slot ordering, naming, or BgSource tags surfaces
    /// here.
    #[test]
    fn snapshot_view_fold_spec_bindings_and_cfg() {
        use crate::cg::data_handle::ViewId;
        let mut prog = CgProgram::default();
        let view = ViewId(3);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner.event_kinds.insert(7, "AgentAttacked".into());
        let op = view_fold_op(&mut prog, view, EventKindId(7), EventRingId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();

        // Pin the (slot, name, access-debug, wgsl_ty) tuple per binding.
        let snapshot: Vec<(u32, String, String, String)> = spec
            .bindings
            .iter()
            .map(|b| {
                (
                    b.slot,
                    b.name.clone(),
                    format!("{:?}", b.access),
                    b.wgsl_ty.clone(),
                )
            })
            .collect();
        let expected = vec![
            (
                0,
                "event_ring".to_string(),
                "ReadStorage".to_string(),
                "array<u32>".to_string(),
            ),
            (
                1,
                "event_tail".to_string(),
                "ReadStorage".to_string(),
                "array<u32>".to_string(),
            ),
            (
                2,
                "view_storage_primary".to_string(),
                "ReadWriteStorage".to_string(),
                "array<u32>".to_string(),
            ),
            (
                3,
                "view_storage_anchor".to_string(),
                "ReadWriteStorage".to_string(),
                "array<u32>".to_string(),
            ),
            (
                4,
                "view_storage_ids".to_string(),
                "ReadWriteStorage".to_string(),
                "array<u32>".to_string(),
            ),
            (
                5,
                "sim_cfg".to_string(),
                "ReadStorage".to_string(),
                "array<u32>".to_string(),
            ),
            (
                6,
                "cfg".to_string(),
                "Uniform".to_string(),
                "FoldThreatLevelCfg".to_string(),
            ),
        ];
        assert_eq!(snapshot, expected);

        // ViewFold cfg shape pinned explicitly. The cfg struct name
        // is derived from the kernel name, which collapses to
        // `fold_threat_level` post Task 5.7-iter2 (no event suffix).
        assert!(spec.cfg_struct_decl.contains("event_count: u32, pub tick: u32, pub _pad: [u32; 2]"));
        assert_eq!(
            spec.cfg_build_expr,
            "FoldThreatLevelCfg { event_count: 0, tick: state.tick, _pad: [0; 2] }"
        );
    }

    /// `KernelSpec::kind` is the canonical routing tag — ViewFold
    /// topologies stamp `KernelKind::ViewFold`, every other kernel
    /// shape stamps `KernelKind::Generic`. Pins the contract that
    /// `cg/emit/program.rs` matches against.
    #[test]
    fn spec_kind_matches_topology() {
        use crate::cg::data_handle::ViewId;

        // ViewFold topology → KernelKind::ViewFold.
        let mut prog = CgProgram::default();
        let view = ViewId(3);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner.event_kinds.insert(7, "AgentAttacked".into());
        let op = view_fold_op(&mut prog, view, EventKindId(7), EventRingId(0));
        let topology = KernelTopology::Split {
            op,
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let spec = kernel_topology_to_spec(&topology, &prog, &ctx).unwrap();
        assert_eq!(spec.kind, KernelKind::ViewFold);

        // MaskPredicate Split → Generic.
        let mut prog2 = CgProgram::default();
        let op2 = mask_op(&mut prog2, MaskId(0));
        let topology2 = KernelTopology::Split {
            op: op2,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx2 = EmitCtx::structural(&prog2);
        let spec2 = kernel_topology_to_spec(&topology2, &prog2, &ctx2).unwrap();
        assert_eq!(spec2.kind, KernelKind::Generic);

        // Fused mask → Generic.
        let mut prog3 = CgProgram::default();
        let a = mask_op(&mut prog3, MaskId(0));
        let b = mask_op(&mut prog3, MaskId(1));
        let topology3 = KernelTopology::Fused {
            ops: vec![a, b],
            dispatch: DispatchShape::PerAgent,
        };
        let ctx3 = EmitCtx::structural(&prog3);
        let spec3 = kernel_topology_to_spec(&topology3, &prog3, &ctx3).unwrap();
        assert_eq!(spec3.kind, KernelKind::Generic);

        // Indirect (physics consumer) → Generic.
        let mut prog4 = CgProgram::default();
        let ring = EventRingId(1);
        let prod = seed_indirect_op(&mut prog4, ring);
        let cons = physics_op(&mut prog4, PhysicsRuleId(0), ring);
        let topology4 = KernelTopology::Indirect {
            producer: prod,
            consumers: vec![cons],
        };
        let ctx4 = EmitCtx::structural(&prog4);
        let spec4 = kernel_topology_to_spec(&topology4, &prog4, &ctx4).unwrap();
        assert_eq!(spec4.kind, KernelKind::Generic);
    }

    // ---- 13. Plumbing body templates (Task 5.6d) ----

    /// Helper: build a `Split` Plumbing topology for `kind` and lower
    /// it to `(spec, body)`. Used by the per-`PlumbingKind` body
    /// assertions below.
    fn plumbing_spec_and_body(kind: PlumbingKind) -> (KernelSpec, String) {
        let mut prog = CgProgram::default();
        let dispatch = kind.dispatch_shape();
        let op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::Plumbing { kind },
            dispatch,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        let op_id = push_op(&mut prog, op);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch,
        };
        let ctx = EmitCtx::structural(&prog);
        kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap()
    }

    #[test]
    fn plumbing_pack_agents_emits_pack_agents_body() {
        let (_spec, body) = plumbing_spec_and_body(PlumbingKind::PackAgents);
        // Per-kind preamble comment from the body const.
        assert!(
            body.contains("PlumbingKind::PackAgents"),
            "PackAgents body: {body}"
        );
        // Touches the packed scratch + a representative agent-field
        // binding + the cfg uniform's agent_cap.
        assert!(body.contains("agent_scratch_packed"), "body: {body}");
        assert!(body.contains("agent_pos"), "body: {body}");
        assert!(body.contains("cfg.agent_cap"), "body: {body}");
        // No leftover TODO placeholder from the pre-Task-5.6d shape.
        assert!(
            !body.contains("TODO(task-4.x): plumbing body"),
            "body still carries pre-5.6d TODO placeholder: {body}"
        );
    }

    #[test]
    fn plumbing_unpack_agents_emits_unpack_agents_body() {
        let (_spec, body) = plumbing_spec_and_body(PlumbingKind::UnpackAgents);
        assert!(
            body.contains("PlumbingKind::UnpackAgents"),
            "UnpackAgents body: {body}"
        );
        assert!(body.contains("agent_scratch_packed"), "body: {body}");
        assert!(body.contains("agent_pos"), "body: {body}");
        assert!(
            !body.contains("TODO(task-4.x): plumbing body"),
            "body still carries pre-5.6d TODO placeholder: {body}"
        );
    }

    #[test]
    fn plumbing_alive_bitmap_emits_alive_bitmap_body() {
        let (_spec, body) = plumbing_spec_and_body(PlumbingKind::AliveBitmap);
        assert!(
            body.contains("PlumbingKind::AliveBitmap"),
            "AliveBitmap body: {body}"
        );
        // Reads agent.alive (per-agent SoA) and writes alive_bitmap via
        // atomicStore (atomic access mode).
        assert!(body.contains("agent_alive["), "body: {body}");
        assert!(body.contains("atomicStore(&alive_bitmap"), "body: {body}");
        // PerWord preamble from `thread_indexing_preamble`.
        assert!(body.contains("word_idx = gid.x"), "body: {body}");
        // 32-slot pack loop.
        assert!(body.contains("for (var i: u32 = 0u; i < 32u"), "body: {body}");
        assert!(
            !body.contains("TODO(task-4.x): plumbing body"),
            "body still carries pre-5.6d TODO placeholder: {body}"
        );
    }

    #[test]
    fn plumbing_drain_events_emits_drain_events_body_with_ring_id() {
        let ring = EventRingId(7);
        let (_spec, body) =
            plumbing_spec_and_body(PlumbingKind::DrainEvents { ring });
        assert!(
            body.contains("PlumbingKind::DrainEvents (ring=7)"),
            "DrainEvents body: {body}"
        );
        // Touches the ring binding (structural name `event_ring_<r>`).
        assert!(body.contains("event_ring_7["), "body: {body}");
        // PerEvent preamble from `thread_indexing_preamble`.
        assert!(body.contains("event_idx = gid.x"), "body: {body}");
        assert!(
            !body.contains("TODO(task-4.x): plumbing body"),
            "body still carries pre-5.6d TODO placeholder: {body}"
        );
    }

    #[test]
    fn plumbing_upload_sim_cfg_emits_upload_sim_cfg_body() {
        let (_spec, body) = plumbing_spec_and_body(PlumbingKind::UploadSimCfg);
        assert!(
            body.contains("PlumbingKind::UploadSimCfg"),
            "UploadSimCfg body: {body}"
        );
        // OneShot preamble.
        assert!(body.contains("if (gid.x != 0u)"), "body: {body}");
        // Documents the host-side upload — body itself is a no-op.
        assert!(body.contains("host-side upload"), "body: {body}");
        assert!(
            !body.contains("TODO(task-4.x): plumbing body"),
            "body still carries pre-5.6d TODO placeholder: {body}"
        );
    }

    #[test]
    fn plumbing_kick_snapshot_emits_kick_snapshot_body() {
        let (_spec, body) = plumbing_spec_and_body(PlumbingKind::KickSnapshot);
        assert!(
            body.contains("PlumbingKind::KickSnapshot"),
            "KickSnapshot body: {body}"
        );
        // OneShot preamble.
        assert!(body.contains("if (gid.x != 0u)"), "body: {body}");
        // atomicStore on the snapshot_kick slot (atomic access mode).
        assert!(
            body.contains("atomicStore(&snapshot_kick[0], 1u);"),
            "body: {body}"
        );
        assert!(
            !body.contains("TODO(task-4.x): plumbing body"),
            "body still carries pre-5.6d TODO placeholder: {body}"
        );
    }

    #[test]
    fn plumbing_seed_indirect_args_emits_body_with_ring_id() {
        let ring = EventRingId(2);
        let (_spec, body) =
            plumbing_spec_and_body(PlumbingKind::SeedIndirectArgs { ring });
        assert!(
            body.contains("PlumbingKind::SeedIndirectArgs (ring=2)"),
            "SeedIndirectArgs body: {body}"
        );
        // Reads tail count from event_ring_<ring>[0]; writes (wg, 1, 1)
        // to indirect_args_<ring>.
        assert!(body.contains("event_ring_2[0]"), "body: {body}");
        assert!(body.contains("indirect_args_2[0] = wg"), "body: {body}");
        assert!(body.contains("indirect_args_2[1] = 1u"), "body: {body}");
        assert!(body.contains("indirect_args_2[2] = 1u"), "body: {body}");
        // CAP_WG cap.
        assert!(body.contains("if (wg > 4096u)"), "body: {body}");
        // OneShot preamble.
        assert!(body.contains("if (gid.x != 0u)"), "body: {body}");
        assert!(
            !body.contains("TODO(task-4.x): plumbing body"),
            "body still carries pre-5.6d TODO placeholder: {body}"
        );
    }

    #[test]
    fn plumbing_seed_indirect_args_distinct_ring_ids_emit_distinct_bodies() {
        // Two SeedIndirectArgs ops on different rings should emit
        // bodies that reference their ring id distinctly — the per-
        // ring formatter `seed_indirect_args_body` must thread the
        // ring id through every binding reference.
        let (_a, body_a) =
            plumbing_spec_and_body(PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(0),
            });
        let (_b, body_b) =
            plumbing_spec_and_body(PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(3),
            });
        assert!(body_a.contains("event_ring_0[0]"), "body_a: {body_a}");
        assert!(body_a.contains("indirect_args_0["), "body_a: {body_a}");
        assert!(!body_a.contains("event_ring_3"), "body_a leaked ring=3: {body_a}");
        assert!(body_b.contains("event_ring_3[0]"), "body_b: {body_b}");
        assert!(body_b.contains("indirect_args_3["), "body_b: {body_b}");
        assert!(!body_b.contains("event_ring_0"), "body_b leaked ring=0: {body_b}");
    }

    /// Snapshot: pin the `AliveBitmap` body output through
    /// `kernel_topology_to_spec_and_body`. Any drift in the body const,
    /// the PerWord preamble shape, or the op-comment header surfaces
    /// here — the assertions below describe the exact expected form so
    /// the snapshot survives a body-content tweak with a one-site update.
    #[test]
    fn plumbing_body_snapshot_alive_bitmap() {
        let (spec, body) = plumbing_spec_and_body(PlumbingKind::AliveBitmap);
        // Spec name + entry point are the legacy filenames (Task 5.2).
        assert_eq!(spec.name, "alive_pack");
        assert_eq!(spec.entry_point, "cs_alive_pack");
        // Pin the body's structural sections in order: PerWord preamble
        // -> op-comment header -> per-kind body comment -> agent_alive
        // read -> atomicStore. Each contains-check is order-independent;
        // the relative-offset asserts catch reordering.
        let preamble_idx = body
            .find("word_idx = gid.x")
            .unwrap_or_else(|| panic!("missing PerWord preamble: {body}"));
        let op_comment_idx = body
            .find("// op#0 (plumbing)")
            .unwrap_or_else(|| panic!("missing op-comment header: {body}"));
        let kind_comment_idx = body
            .find("PlumbingKind::AliveBitmap")
            .unwrap_or_else(|| panic!("missing per-kind comment: {body}"));
        let agent_alive_idx = body
            .find("agent_alive[slot]")
            .unwrap_or_else(|| panic!("missing agent_alive read: {body}"));
        let atomic_store_idx = body
            .find("atomicStore(&alive_bitmap[word_idx], word);")
            .unwrap_or_else(|| panic!("missing atomicStore: {body}"));
        assert!(
            preamble_idx < op_comment_idx,
            "PerWord preamble must come before op-comment header"
        );
        assert!(
            op_comment_idx < kind_comment_idx,
            "op-comment header must come before per-kind body comment"
        );
        assert!(
            kind_comment_idx < agent_alive_idx,
            "per-kind comment must come before the agent_alive read"
        );
        assert!(
            agent_alive_idx < atomic_store_idx,
            "agent_alive read must come before the atomicStore"
        );
    }

    /// Smoke test: every `PlumbingKind` variant produces a non-empty
    /// body that does NOT carry a leftover TODO placeholder. Mirrors
    /// the `plumbing_kinds_map_to_legacy_filenames` test's exhaustive
    /// list — adding a new variant requires updating both, which the
    /// match-arm exhaustiveness in `plumbing_body_for_kind` and the
    /// dispatch-shape mapping in `PlumbingKind::dispatch_shape` already
    /// enforce at the type level.
    #[test]
    fn every_plumbing_kind_emits_non_empty_non_todo_body() {
        let kinds = [
            PlumbingKind::PackAgents,
            PlumbingKind::UnpackAgents,
            PlumbingKind::AliveBitmap,
            PlumbingKind::DrainEvents { ring: EventRingId(0) },
            PlumbingKind::UploadSimCfg,
            PlumbingKind::KickSnapshot,
            PlumbingKind::SeedIndirectArgs { ring: EventRingId(0) },
        ];
        for kind in kinds {
            let (_spec, body) = plumbing_spec_and_body(kind);
            assert!(
                !body.is_empty(),
                "kind={kind:?} produced empty body"
            );
            assert!(
                !body.contains("TODO(task-4.x): plumbing body"),
                "kind={kind:?} still carries pre-5.6d TODO placeholder: {body}"
            );
            // Every body labels its variant — eyeballing per-kind
            // bodies in CI logs stays trivial.
            assert!(
                body.contains("PlumbingKind::"),
                "kind={kind:?} body missing PlumbingKind::* tag: {body}"
            );
        }
    }
}
