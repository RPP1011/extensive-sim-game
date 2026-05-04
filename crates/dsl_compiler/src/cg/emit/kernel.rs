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

    // ViewDecay: 2-binding kernel (view_storage_primary + cfg).
    // Body is hand-synthesised from the rate constant — no statement
    // tree to walk.
    if let KernelKindClass::ViewDecay {
        view_name,
        rate_bits,
    } = &class
    {
        let bindings = build_view_decay_bindings(view_name, &cfg_struct);
        let cfg_struct_decl = build_view_decay_cfg_struct_decl(&cfg_struct);
        let cfg_build_expr = build_view_decay_cfg_build_expr(&cfg_struct);
        let wgsl_body = build_view_decay_wgsl_body(*rate_bits);
        let spec = KernelSpec {
            name,
            pascal,
            entry_point,
            cfg_struct,
            cfg_build_expr,
            cfg_struct_decl,
            bindings,
            kind: KernelKind::ViewDecay,
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
    let mut needs_event_tail = false;
    for (key, agg) in &handle_set {
        let canonical = canonical_handle(key, &agg.first_seen);
        // Any EventRing access — Read (consumer paths like
        // SeedIndirectArgs that need to know the ring depth) or
        // Append (producers that atomicAdd to acquire a slot) — needs
        // the sibling `event_tail` binding. The fold path
        // (build_view_fold_bindings) hardcodes event_tail in slot 1
        // for its own bindings; the generic path here synthesizes
        // the same binding shape post-loop.
        if let DataHandle::EventRing { .. } = &canonical {
            needs_event_tail = true;
        }
        let Some(meta) = handle_to_binding_metadata(&canonical, prog) else {
            // Rng + ConfigConst route through the cfg uniform / inline
            // helpers, not as standalone bindings.
            continue;
        };
        let access = upgrade_access(meta.base_access, agg.was_written);
        let name = structural_binding_name(&canonical, Some(prog));
        typed_bindings.push(TypedBinding {
            sort_key: key.clone(),
            name,
            access,
            wgsl_ty: meta.wgsl_ty,
            bg_source: meta.bg_source,
        });
    }

    // 6b. Synthesize a sibling `event_tail` binding when any op in
    //     this kernel does `EventRing-Append`. The resident-context
    //     field `batch_events_tail` is the source-of-truth single
    //     atomic<u32> counter the runtime allocates alongside the
    //     event ring; producer kernels atomicAdd against it to
    //     acquire write slots.
    //
    //     Sort key uses a synthetic `Ring(EventRingId(u32::MAX))` so
    //     the tail consistently sorts after any real ring-keyed
    //     binding (deterministic ordering). Name "event_tail" is the
    //     identifier the WGSL emit body in
    //     `lower_emit_to_wgsl` references unconditionally.
    if needs_event_tail {
        typed_bindings.push(TypedBinding {
            sort_key: crate::cg::data_handle::CycleEdgeKey::Ring(
                crate::cg::data_handle::EventRingId(u32::MAX),
            ),
            name: "event_tail".to_string(),
            access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".to_string(),
            bg_source: BgSource::Resident("batch_events_tail".to_string()),
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
    //    classified kind. ViewFold is handled above; ALL PerEvent-
    //    dispatched physics rules need the same `event_count + tick`
    //    cfg shape ViewFold uses, because the preamble emitted by
    //    `thread_indexing_preamble` references `cfg.event_count` for
    //    every PerEvent dispatch regardless of whether the body has an
    //    Emit op (the body just consumes events). Three fixtures
    //    independently surfaced this:
    //      - trade_market_real's `physics ApplyTrade` (consumes Trade,
    //        emits nothing) — without this gate, the classifier dropped
    //        to `Generic` (`agent_cap` cfg), and `if (event_idx >=
    //        cfg.event_count)` failed to type-check at WGSL compile.
    //      - quest_arc_real's `ApplyStageAdvance` (reads StageAdvanced,
    //        writes mana, emits nothing) — same shape.
    //      - village_day_cycle's `ApplyHarvest` (reads WorkDone, writes
    //        wealth, emits nothing) — same shape.
    //    Closing the gap by stamping PerEventEmit for every PerEvent
    //    dispatch; the second-pass `event_ring` binding upgrade below
    //    is also required so reads of the consumed event ring
    //    type-check as `array<atomic<u32>>`.

    let is_per_event_emit = matches!(dispatch, DispatchShape::PerEvent { .. });
    let (cfg_struct_decl, cfg_build_expr, kernel_kind) = if is_per_event_emit {
        (
            build_per_event_emit_cfg_struct_decl(&cfg_struct),
            build_per_event_emit_cfg_build_expr(&cfg_struct),
            KernelKind::PerEventEmit,
        )
    } else {
        (
            build_generic_cfg_struct_decl(&cfg_struct),
            build_generic_cfg_build_expr(&cfg_struct),
            KernelKind::Generic,
        )
    };

    // PerEventEmit kernels share `event_ring` between producer-side
    // `Emit` (atomicStore) and consumer-side `EventField` reads
    // (atomicLoad). Force the EventRing-Read binding (declared today
    // as `array<u32>` per `handle_to_binding_metadata`) up to
    // `array<atomic<u32>>` so both surfaces type-check against the
    // same WGSL declaration. Affects only the EventRing binding; the
    // BGL entry stays `bgl_storage(N, false)` (atomic vs non-atomic
    // is a WGSL-side distinction the BGL doesn't carry).
    if is_per_event_emit {
        for binding in bindings.iter_mut() {
            if binding.name == "event_ring"
                && matches!(binding.access, AccessMode::ReadStorage | AccessMode::ReadWriteStorage)
            {
                binding.access = AccessMode::AtomicStorage;
                binding.wgsl_ty = "u32".to_string();
            }
        }
    }

    // 10. Compose the WGSL body — one fragment per op, joined with
    //     blank lines. Computing the body here surfaces any inner-walk
    //     arena failures as typed errors before the spec is returned,
    //     so a malformed program never yields a spec whose body the
    //     downstream WGSL emitter can't render. The body itself is not
    //     stored on `KernelSpec` (it lives in the WGSL emitter that
    //     consumes the spec) — Task 4.3 will redo body composition at
    //     that layer. We return it alongside the spec for tests + Task
    //     4.3 callers; [`kernel_topology_to_spec`] discards it.
    //
    // PerEventEmit kernels declare `event_ring: array<atomic<u32>>`
    // (above), which forces every payload-word read in the body to go
    // through `atomicLoad`. Stash the flag on the emit context for the
    // duration of the body emit so `CgExpr::EventField` and the
    // per-handler tag-filter wrap both pick the atomic form, then
    // restore on exit — defensive scoping in case the same `EmitCtx`
    // instance is reused across multiple kernels.
    let prior_atomic_loads = ctx.event_ring_atomic_loads.replace(is_per_event_emit);
    let wgsl_body_result = build_wgsl_body(&body_ops, &dispatch, prog, ctx);
    ctx.event_ring_atomic_loads.set(prior_atomic_loads);
    let wgsl_body = wgsl_body_result?;

    let spec = KernelSpec {
        name,
        pascal,
        entry_point,
        cfg_struct,
        cfg_build_expr,
        cfg_struct_decl,
        bindings,
        kind: kernel_kind,
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
    /// Singleton kernel emitting a `@decay(rate, per=tick)` per-slot
    /// anchor multiplication. Carries the view's snake_case name and
    /// the rate's bit-pattern so the body composer can synthesise
    /// `view_storage_primary[k] *= rate` directly without going through
    /// the generic handle-aggregation pipeline.
    ViewDecay {
        /// snake_case view name. Falls back to `view_<id>` when no
        /// name is interned.
        view_name: String,
        /// `f32::to_bits()` of the validated decay rate, mirroring the
        /// `ComputeOpKind::ViewDecay::rate_bits` field exactly.
        rate_bits: u32,
    },
    /// Anything not detected as ViewFold or ViewDecay. Routes through
    /// the generic handle-aggregation pipeline + placeholder cfg shape.
    Generic,
}

/// Classify a kernel's body ops. See [`KernelKindClass`] for the
/// table. `prog` carries the interner so the ViewFold path can
/// resolve a snake_case view name eagerly.
fn classify_kernel(body_ops: &[&ComputeOp], prog: &CgProgram) -> KernelKindClass {
    if body_ops.is_empty() {
        return KernelKindClass::Generic;
    }
    // ViewDecay detection: a kernel is a ViewDecay precisely when it
    // carries exactly one ViewDecay op and nothing else. The
    // `cross_domain_split_decision` arm in
    // `crates/dsl_compiler/src/cg/schedule/fusion.rs` keeps ViewDecay
    // ops from fusing with anything, so this branch fires for the
    // singleton path.
    if body_ops.len() == 1 {
        if let ComputeOpKind::ViewDecay { view, rate_bits } = &body_ops[0].kind {
            let view_name = match prog.interner.get_view_name(*view) {
                Some(name) => name.to_string(),
                None => format!("view_{}", view.0),
            };
            return KernelKindClass::ViewDecay {
                view_name,
                rate_bits: *rate_bits,
            };
        }
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
            | ComputeOpKind::Plumbing { .. }
            | ComputeOpKind::ViewDecay { .. } => {
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
        DataHandle::ItemField { field, target: _ } => {
            // Per-Item SoA field — emit a bare external binding named
            // `<entity_snake>_<field_snake>`. The per-fixture runtime
            // is responsible for allocating + binding the buffer at
            // dispatch time. Read-only at this slice (no write surface
            // routed through the lowering yet); upgrades to RW would
            // happen via the same `record_write` path AgentField uses.
            let (entity_name, field_name, _) = prog
                .entity_field_catalog
                .resolve_item(*field)
                .unwrap_or(("item", "field", AgentFieldTy::U32));
            Some(BindingMetadata {
                bg_source: BgSource::External(item_field_external_name(
                    entity_name,
                    field_name,
                )),
                base_access: AccessMode::ReadStorage,
                wgsl_ty: agent_field_wgsl_ty(field.ty),
            })
        }
        DataHandle::GroupField { field, target: _ } => {
            let (entity_name, field_name, _) = prog
                .entity_field_catalog
                .resolve_group(*field)
                .unwrap_or(("group", "field", AgentFieldTy::U32));
            Some(BindingMetadata {
                bg_source: BgSource::External(item_field_external_name(
                    entity_name,
                    field_name,
                )),
                base_access: AccessMode::ReadStorage,
                wgsl_ty: agent_field_wgsl_ty(field.ty),
            })
        }
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
            //
            // GridOffsets is atomic across the whole pipeline:
            //   - BuildHash uses atomicAdd to allocate per-cell slots
            //     (bounded counting-sort variant, MAX_PER_CELL caps
            //     each cell's population).
            //   - Consumers (per-agent kernels with ForEachNeighbor)
            //     read the count via atomicLoad. Both surfaces share
            //     the same WGSL binding type `array<atomic<u32>>`.
            // GridCells is non-atomic: the atomic on offsets gives
            //   each writer a unique slot index, so the slot write
            //   itself is conflict-free.
            let (field, base_access, wgsl_ty) = match kind {
                SpatialStorageKind::GridCells => (
                    "spatial_grid_cells".to_string(),
                    AccessMode::ReadStorage,
                    "array<u32>".to_string(),
                ),
                SpatialStorageKind::GridOffsets => (
                    "spatial_grid_offsets".to_string(),
                    AccessMode::AtomicStorage,
                    "u32".to_string(),
                ),
                SpatialStorageKind::QueryResults => (
                    "spatial_query_results".to_string(),
                    AccessMode::ReadWriteStorage,
                    "array<u32>".to_string(),
                ),
                // NonemptyCells: written by the CompactNonemptyCells
                // kernel (each non-empty cell atomicAdds into the
                // count, then writes its cell index at the returned
                // slot). The atomic is on the *count* (held in
                // NonemptyCellsIndirectArgs[0]); the cells array
                // itself is conflict-free under the atomic-allocated
                // slot indices, so non-atomic ReadWrite suffices.
                // Consumer (tiled MoveBoid) reads it as a plain
                // u32 array indexed by `wgid.x`.
                SpatialStorageKind::NonemptyCells => (
                    "spatial_nonempty_cells".to_string(),
                    AccessMode::ReadWriteStorage,
                    "array<u32>".to_string(),
                ),
                // NonemptyCellsIndirectArgs holds the (count, 1, 1)
                // tuple consumed by `dispatch_workgroups_indirect`.
                // Slot 0 is atomic (CompactNonemptyCells uses
                // atomicAdd to assign slots); slots 1 & 2 are written
                // once at construction (or by the same kernel) as the
                // constants `1u`. Atomic access on the buffer is
                // necessary to satisfy WGSL's per-binding type rule.
                SpatialStorageKind::NonemptyCellsIndirectArgs => (
                    "spatial_nonempty_indirect_args".to_string(),
                    AccessMode::AtomicStorage,
                    "u32".to_string(),
                ),
                // GridStarts: prefix-scan output. The new parallel
                // scan writes it cooperatively from many lanes
                // (phase 2a writes per-chunk inclusive prefix; phase
                // 2c adds the chunk base in place). Each cell slot
                // is touched by exactly one lane in each phase, so
                // the writes are conflict-free without atomics —
                // the storage stays plain `array<u32>`. Consumers
                // (BuildHashScatter + tiled MoveBoid) only read.
                // Size on the runtime side is `num_cells + 1` u32s;
                // the WGSL type doesn't bake in the length, so the
                // per-fixture runtime is the source of truth.
                SpatialStorageKind::GridStarts => (
                    "spatial_grid_starts".to_string(),
                    AccessMode::ReadStorage,
                    "array<u32>".to_string(),
                ),
                // ChunkSums: cross-workgroup carry buffer for the
                // parallel prefix scan. Written by phase 2a (each
                // chunk's last lane stores the chunk total),
                // exclusive-scanned in place by phase 2b, read by
                // phase 2c. Non-atomic — single writer per slot in
                // every phase. Sized
                // `ceil(num_cells / PER_SCAN_CHUNK_WORKGROUP_X)` on
                // the runtime side.
                SpatialStorageKind::ChunkSums => (
                    "spatial_chunk_sums".to_string(),
                    AccessMode::ReadStorage,
                    "array<u32>".to_string(),
                ),
            };
            Some(BindingMetadata {
                bg_source: BgSource::Pool(field),
                base_access,
                wgsl_ty,
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
/// External binding name for an Item / Group SoA field. The
/// per-fixture runtime allocates a `<entity_snake>_<field_snake>`
/// buffer (e.g. `coin_weight`) and binds it on the kernel's external
/// bind group. Pulled out of the per-handle metadata so the structural
/// binding name + the BGL composer's external lookup stay in lockstep.
pub(crate) fn item_field_external_name(entity_name: &str, field_name: &str) -> String {
    format!("{}_{}", to_snake_case(entity_name), field_name)
}

/// Convert a PascalCase / camelCase identifier to snake_case. Used for
/// the entity-name half of the Item / Group binding name.
fn to_snake_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i != 0 {
            out.push('_');
        }
        for low in ch.to_lowercase() {
            out.push(low);
        }
    }
    out
}

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
/// body referring to `agent_hp[agent_id]` resolves through this binding's
/// declared name.
///
/// For `AgentField`, we drop the `target` discriminator from the
/// binding name — the binding identifies the storage buffer, not the
/// per-thread access expression. `agent_<field>` is the form.
fn structural_binding_name(h: &DataHandle, prog: Option<&CgProgram>) -> String {
    match h {
        DataHandle::AgentField { field, target: _ } => {
            // The storage is shared across all per-thread accesses;
            // the binding name keys off the field, not the agent ref.
            format!("agent_{}", field.snake())
        }
        DataHandle::ItemField { field, target: _ } => {
            // Per-Item SoA: prefer the catalog-resolved form
            // `<entity_snake>_<field_snake>` (e.g. `coin_weight`) so
            // the BGL binding name matches the WGSL body's access
            // form (set by `item_field_binding_name` in
            // `cg/emit/wgsl_body.rs`). Falls back to the opaque
            // `item_<entity>_<slot>` form when no program / catalog is
            // available (e.g. structural debug rendering).
            if let Some(p) = prog {
                if let Some((entity_name, field_name, _)) =
                    p.entity_field_catalog.resolve_item(*field)
                {
                    return item_field_external_name(entity_name, field_name);
                }
            }
            format!("item_{}_{}", field.entity, field.slot)
        }
        DataHandle::GroupField { field, target: _ } => {
            if let Some(p) = prog {
                if let Some((entity_name, field_name, _)) =
                    p.entity_field_catalog.resolve_group(*field)
                {
                    return item_field_external_name(entity_name, field_name);
                }
            }
            format!("group_{}_{}", field.entity, field.slot)
        }
        DataHandle::ViewStorage { view, slot } => {
            format!("view_{}_{}", view.0, view_slot_field(*slot))
        }
        DataHandle::EventRing { ring: _, kind: _ } => {
            // Cycle-edge collapses access mode; one binding per ring
            // regardless of read/append/drain (the lowering uses the
            // access mode to decide read vs atomicAdd). Iter-2 unified
            // every event kind to `EventRingId(0)`, so there's only
            // one ring; drop the `_<ring.0>` suffix so the binding
            // name `event_ring` matches the ViewFold preamble's
            // canonical name and the schema's buffer_name. When the
            // runtime moves to per-kind ring fanout, restore the suffix
            // (and update `populate_event_kinds::buffer_name` in tandem).
            "event_ring".to_string()
        }
        DataHandle::ConfigConst { id } => format!("config_{}", id.0),
        DataHandle::MaskBitmap { mask } => format!("mask_{}_bitmap", mask.0),
        DataHandle::ScoringOutput => "scoring_output".into(),
        DataHandle::SpatialStorage { kind } => match kind {
            SpatialStorageKind::GridCells => "spatial_grid_cells".into(),
            SpatialStorageKind::GridOffsets => "spatial_grid_offsets".into(),
            SpatialStorageKind::QueryResults => "spatial_query_results".into(),
            SpatialStorageKind::NonemptyCells => "spatial_nonempty_cells".into(),
            SpatialStorageKind::NonemptyCellsIndirectArgs => {
                "spatial_nonempty_indirect_args".into()
            }
            SpatialStorageKind::GridStarts => "spatial_grid_starts".into(),
            SpatialStorageKind::ChunkSums => "spatial_chunk_sums".into(),
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
            | ComputeOpKind::Plumbing { .. }
            | ComputeOpKind::ViewDecay { .. } => return None,
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
            | ComputeOpKind::Plumbing { .. }
            | ComputeOpKind::ViewDecay { .. } => return None,
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
        ComputeOpKind::PhysicsRule { rule, on_event, .. } => {
            // PerAgent PhysicsRule (on_event=None) is a per-agent
            // sweep — Movement is the canonical instance today
            // (Phase 6 Task 3). Name resolves to the rule's
            // interned name without the `physics_` prefix so the
            // SCHEDULE pickup matches the runtime's expected
            // `KernelId::Movement` variant. PerEvent-shaped rules
            // keep the `physics_<name>` shape (collision-safe by
            // construction; the rule name is unique).
            let name = prog
                .interner
                .get_physics_rule_name(*rule)
                .map(String::from)
                .unwrap_or_else(|| format!("rule_{}", rule.0));
            match on_event {
                Some(_) => format!("physics_{name}"),
                None => name,
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
        ComputeOpKind::ViewDecay { view, .. } => match prog.interner.get_view_name(*view) {
            Some(name) => format!("decay_{name}"),
            None => format!("decay_view_{}", view.0),
        },
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

fn spatial_kind_name(k: SpatialQueryKind) -> String {
    match k {
        SpatialQueryKind::BuildHash => String::from("build_hash"),
        SpatialQueryKind::FilteredWalk { filter } => {
            format!("filtered_walk_{}", filter.0)
        }
        SpatialQueryKind::CompactNonemptyCells => {
            String::from("compact_nonempty_cells")
        }
        SpatialQueryKind::BuildHashCount => String::from("build_hash_count"),
        SpatialQueryKind::BuildHashScanLocal => String::from("build_hash_scan_local"),
        SpatialQueryKind::BuildHashScanCarry => String::from("build_hash_scan_carry"),
        SpatialQueryKind::BuildHashScanAdd => String::from("build_hash_scan_add"),
        SpatialQueryKind::BuildHashScatter => String::from("build_hash_scatter"),
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
        ComputeOpKind::ViewDecay { .. } => "view_decay",
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
    // `tick: u32` joined the layout when PerAgent rules with `emit
    // <Event>` bodies started referencing `tick` in the per-tick
    // event payload header. `seed: u32` joined when the rng.* surface
    // wired through to GPU (stochastic_probe Gaps #1-#3 close,
    // 2026-05-04): the WGSL kernel preamble binds `let seed =
    // cfg.seed;` so any `per_agent_u32(seed, agent_id, tick, purpose)`
    // call resolves. Backwards-compat note: every runtime
    // constructs cfg manually (per-fixture lib.rs), so the renamed
    // padding (`_pad: [u32; 2]` → `seed: u32, _pad: u32`) surfaces as
    // a missing-field error at the per-fixture build site — caller
    // updates that in lockstep.
    format!(
        "#[repr(C)]\n\
         #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
         pub struct {cfg_struct} {{ pub agent_cap: u32, pub tick: u32, pub seed: u32, pub _pad: u32 }}"
    )
}

fn build_generic_cfg_build_expr(cfg_struct: &str) -> String {
    format!(
        "{cfg_struct} {{ agent_cap: state.agent_cap(), tick: state.tick as u32, seed: state.seed as u32, _pad: 0 }}"
    )
}

// ---------------------------------------------------------------------------
// PerEventEmit-specific cfg synthesis
// ---------------------------------------------------------------------------

/// Build the cfg struct decl for a [`KernelKind::PerEventEmit`] kernel —
/// `PerEvent`-dispatched physics rule whose body contains an `Emit`.
/// Mirrors the [`KernelKind::ViewFold`] cfg layout (`event_count: u32,
/// tick: u32`) so the kernel preamble's `if event_idx >= cfg.event_count
/// { return; }` early-return guard and the body's emit-side `tick`
/// header word both resolve. Two `_pad` fields preserve the 16-byte
/// uniform alignment.
fn build_per_event_emit_cfg_struct_decl(cfg_struct: &str) -> String {
    format!(
        "#[repr(C)]\n\
         #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
         pub struct {cfg_struct} {{ pub event_count: u32, pub tick: u32, pub seed: u32, pub _pad0: u32 }}"
    )
}

/// Cfg builder expression for a [`KernelKind::PerEventEmit`] kernel.
/// `event_count` defaults to 0 — the runtime overwrites it per dispatch
/// with the actual per-tick event count from the source ring's tail
/// (or an upper-bound estimate). This mirrors the
/// [`build_view_fold_cfg_build_expr`] convention.
fn build_per_event_emit_cfg_build_expr(cfg_struct: &str) -> String {
    format!(
        "{cfg_struct} {{ event_count: 0, tick: state.tick as u32, seed: state.seed as u32, _pad0: 0 }}"
    )
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
    // `second_key_pop` joined the layout when `pair_map` storage hints
    // started lowering with a real 2-D index in the fold body's RMW
    // (`view_storage_primary[k1 * cfg.second_key_pop + k2]`). For
    // single-key views the runtime sets it to 1 so the index reduces
    // to `k1 * 1 + 0` ≡ `k1` — the WGSL template uses the field
    // unconditionally, the runtime supplies the discriminator. The
    // fourth slot stays as `_pad: u32` to preserve the 16-byte uniform
    // alignment.
    format!(
        "#[repr(C)]\n\
         #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
         pub struct {cfg_struct} {{ pub event_count: u32, pub tick: u32, pub second_key_pop: u32, pub _pad: u32 }}"
    )
}

fn build_view_fold_cfg_build_expr(cfg_struct: &str) -> String {
    // `second_key_pop = 1` is the safe default for non-PairMap views;
    // PairMap-keyed runtimes overwrite this with the second-key entity
    // population (`agent_cap` for Agent×Agent, item count for
    // Agent×Item, …) when they upload the cfg per-tick. The compile-
    // time default is harmless for non-PairMap views since their fold
    // body indexes `local_<k> * 1 + 0 ≡ local_<k>`.
    format!(
        "{cfg_struct} {{ event_count: 0, tick: state.tick, second_key_pop: 1, _pad: 0 }}"
    )
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
            // B1 fix: the fold body emits a CAS loop over this slot
            // (`atomicLoad` + `atomicCompareExchangeWeak`) so the WGSL
            // type must be `array<atomic<u32>>`. AccessMode::AtomicStorage
            // wraps `wgsl_ty="u32"` into the atomic array form via
            // `lower_wgsl_bindings`. Rust BGL entry is unchanged
            // (`bgl_storage(N, false)` covers both atomic and
            // non-atomic). Anchor/ids slots remain non-atomic — they
            // alias primary's buffer via `unwrap_or(primary_buf)` at
            // record time but the WGSL declarations are independent
            // (the kernel body never indexes anchor/ids today).
            access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
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
        // For ViewFold ops, also extract the matched event kind so the
        // body can be guarded by a tag-check (event_ring offset 0 == kind).
        // Without the guard, every event in the unified ring lands in
        // every handler — multi-kind fixtures (e.g. ecosystem_cascade
        // emits PlantEaten + HerbivoreEaten) double-count any slot
        // ranges the two kinds share. Today the producer writes the
        // kind tag as the raw `EventKindId.0` (see lower_emit_to_wgsl
        // in wgsl_body.rs: `atomicStore(&event_ring[slot * stride + 0u],
        // <event_id>u)`); the consumer here mirrors that constant.
        let (fragment, fold_meta) = match &op.kind {
            ComputeOpKind::ViewFold { body, on_event, .. } => {
                let body_wgsl =
                    lower_cg_stmt_list_to_wgsl(*body, ctx).map_err(KernelEmitError::from)?;
                // Stride for the kind tag offset. The runtime today
                // packs every kind into a single ring with stride 10
                // (= 2 header + 8 payload); per-kind layouts may
                // diverge once ring fanout lands. Look up the per-kind
                // stride when registered, fall back to the runtime's
                // hard-coded constant otherwise (test harnesses that
                // build ViewFold ops directly without populating
                // event_layouts hit the fallback — same value, no
                // observable diff).
                let stride = prog
                    .event_layouts
                    .get(&on_event.0)
                    .map(|l| l.record_stride_u32)
                    .unwrap_or(EVENT_RING_DEFAULT_STRIDE_U32);
                (body_wgsl, Some((on_event.0, stride)))
            }
            ComputeOpKind::MaskPredicate { .. }
            | ComputeOpKind::PhysicsRule { .. }
            | ComputeOpKind::ScoringArgmax { .. }
            | ComputeOpKind::SpatialQuery { .. }
            | ComputeOpKind::Plumbing { .. }
            | ComputeOpKind::ViewDecay { .. } => {
                // Reachable only if the classifier admitted a non-ViewFold
                // op into a ViewFold-classed kernel — the classifier
                // returns Generic in that case, so this is structurally
                // unreachable. Emit a documented TODO instead of panicking.
                (
                    "// TODO(task-5.5): non-ViewFold op in ViewFold kernel — \
                     classifier should have routed through generic path."
                        .to_string(),
                    None,
                )
            }
        };
        writeln!(
            out,
            "    // op#{} ({})",
            op.id.0,
            compute_op_kind_short(&op.kind)
        )
        .expect("write to String never fails");
        // Wrap the per-op body in `{ ... }` so locals declared by Task 1's
        // event-pattern binding lowering (`let local_<N>: <ty> = ...;`)
        // live in their own WGSL scope. Without these braces, two ops in
        // the same fused kernel both emit `let local_0` at the function
        // top level, which naga rejects as a redefinition. WGSL handles
        // nested scopes cleanly, so siblings can each carry their own
        // `local_0` without collision.
        //
        // For ViewFold ops, also wrap the body in an `if (tag == kind)`
        // guard inside the per-op brace so each handler only processes
        // events of its declared `on Kind { ... }` variant. The Let
        // statements that bind event-pattern locals (`let local_0: u32
        // = event_ring[event_idx * 10u + 2u]`) move INSIDE the guard so
        // they only execute when the tag matches — a non-matching
        // event's payload may not be a valid AgentId and reading it
        // before the guard would index storage with garbage.
        //
        // Indent: kernel-body is at 4-space indent, the per-op brace
        // adds another 4. With a tag guard we add a third level (4
        // more) for the body inside the if-block.
        let extra_indent = if fold_meta.is_some() { 4 } else { 0 };
        let body_indent = 8 + extra_indent;
        let indented = fragment
            .lines()
            .map(|l| {
                if l.is_empty() {
                    String::new()
                } else {
                    format!("{:indent$}{l}", "", indent = body_indent)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        out.push_str("    {\n");
        if let Some((kind_tag, stride)) = fold_meta {
            // Guard the body on the per-event tag word. The producer
            // writes the kind id as a plain u32 at offset 0 (see
            // `lower_emit_to_wgsl`); the consumer reads it the same
            // way (the fold ring binding is `array<u32>`, not atomic,
            // so a plain index read is correct).
            out.push_str(&format!(
                "        if (event_ring[event_idx * {stride}u + 0u] == {kind_tag}u) {{\n"
            ));
            out.push_str(&indented);
            out.push_str("\n        }");
        } else {
            out.push_str(&indented);
        }
        out.push_str("\n    }");
    }

    Ok(out)
}

/// Default u32-words-per-event stride for the unified event ring.
/// Mirrors the schema constant in `EventLayout::record_stride_u32`
/// (today: 2 header + 8 payload = 10). Used as a fallback when a
/// ViewFold op is built without populating `event_layouts` (test
/// harnesses that bypass the lowering). Real compilation always
/// populates the layout, so the lookup hits.
const EVENT_RING_DEFAULT_STRIDE_U32: u32 = 10;

/// Per-event op-kind tag (= EventKindId.0) and per-record stride
/// extracted from `op.kind`'s `on_event` discriminator, when present.
/// Used by `build_wgsl_body` to wrap each per-op body in an
/// `if (event_ring[..tag..] == kind_tag) { ... }` filter so a single
/// PerEvent kernel handling multiple event kinds runs each handler
/// only on its declared `on Kind { ... }` variant. The shared helper
/// is also used by `build_view_fold_wgsl_body` (ViewFold path) — both
/// paths derive the same `(kind_tag, stride)` shape from the op IR.
///
/// Returns `None` for op kinds without an `on_event` (PhysicsRule
/// with `on_event = None` — pure per-agent rules — and the legacy
/// non-event-bearing op kinds), in which case the caller emits the
/// per-op body without a tag-filter wrap.
fn per_event_op_kind_tag(
    kind: &ComputeOpKind,
    prog: &CgProgram,
) -> Option<(u32, u32)> {
    let on_event = match kind {
        ComputeOpKind::ViewFold { on_event, .. } => Some(*on_event),
        ComputeOpKind::PhysicsRule { on_event, .. } => *on_event,
        ComputeOpKind::MaskPredicate { .. }
        | ComputeOpKind::ScoringArgmax { .. }
        | ComputeOpKind::SpatialQuery { .. }
        | ComputeOpKind::Plumbing { .. }
        | ComputeOpKind::ViewDecay { .. } => None,
    }?;
    let stride = prog
        .event_layouts
        .get(&on_event.0)
        .map(|l| l.record_stride_u32)
        .unwrap_or(EVENT_RING_DEFAULT_STRIDE_U32);
    Some((on_event.0, stride))
}

/// Walk every body op's statement list and return `true` if any
/// reachable [`CgStmt::Emit`] exists. Mirrors
/// `cg::lower::driver::collect_emits_in_list` (private to that
/// module) — duplicated here rather than re-exported so the emit
/// layer's helper stays a leaf with no upstream coupling.
///
/// Currently unused: the `is_per_event_emit` gate now stamps
/// PerEventEmit unconditionally for any PerEvent-dispatched kernel,
/// since the PerEvent preamble references `cfg.event_count`
/// regardless of whether the body emits (closing the trade_market_real
/// + quest_arc_real + village_day_cycle Apply-only-PerEvent cfg-
/// mismatch gaps). Kept as `#[allow(dead_code)]` because future kernel
/// kinds may want to discriminate PerAgent rules with vs without Emit
/// bodies for cfg-shape selection.
#[allow(dead_code)]
fn body_ops_have_emit(body_ops: &[OpId], prog: &CgProgram) -> bool {
    for op_id in body_ops {
        let Ok(op) = resolve_op(prog, *op_id) else {
            continue;
        };
        let body_list = match &op.kind {
            ComputeOpKind::PhysicsRule { body, .. } => Some(*body),
            ComputeOpKind::ViewFold { body, .. } => Some(*body),
            _ => None,
        };
        if let Some(list_id) = body_list {
            if stmt_list_has_emit(list_id, prog) {
                return true;
            }
        }
    }
    false
}

/// Recursive walk: `true` iff the statement list named by `list_id`
/// contains at least one [`CgStmt::Emit`], descending through `If`
/// arms, `Match` arms, and `ForEachNeighborBody`. See
/// [`body_ops_have_emit`] for the dead-code rationale.
#[allow(dead_code)]
fn stmt_list_has_emit(list_id: crate::cg::stmt::CgStmtListId, prog: &CgProgram) -> bool {
    use crate::cg::stmt::CgStmt;
    let Some(list) = prog.stmt_lists.get(list_id.0 as usize) else {
        return false;
    };
    for &stmt_id in &list.stmts {
        let Some(stmt) = prog.stmts.get(stmt_id.0 as usize) else {
            continue;
        };
        let hit = match stmt {
            CgStmt::Emit { .. } => true,
            CgStmt::If { then, else_, .. } => {
                stmt_list_has_emit(*then, prog)
                    || else_.as_ref().is_some_and(|e| stmt_list_has_emit(*e, prog))
            }
            CgStmt::Match { arms, .. } => {
                arms.iter().any(|arm| stmt_list_has_emit(arm.body, prog))
            }
            CgStmt::ForEachNeighborBody { body, .. } => stmt_list_has_emit(*body, prog),
            CgStmt::Assign { .. }
            | CgStmt::Let { .. }
            | CgStmt::ForEachAgent { .. }
            | CgStmt::ForEachNeighbor { .. } => false,
        };
        if hit {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// ViewDecay-specific spec synthesis (B2 — `@decay(rate, per=tick)`)
// ---------------------------------------------------------------------------

/// Build the cfg struct decl for a ViewDecay kernel. Mirrors the
/// generic per-agent cfg shape (`{ agent_cap, tick, _pad: [u32; 2] }`)
/// — the per-tick anchor multiplication needs to early-return past
/// `agent_cap` and reads `cfg.tick` for diagnostic / future variable-
/// rate use.
fn build_view_decay_cfg_struct_decl(cfg_struct: &str) -> String {
    // `slot_count` joined the layout when `pair_map` storage hints
    // started over-allocating ViewStorage to `agent_cap × second_pop`.
    // The decay loop early-returns past `slot_count` so 2-D pair views
    // iterate every (k1, k2) pair, while single-key views set
    // `slot_count == agent_cap` and the loop bound matches the legacy
    // shape. `agent_cap` stays in the struct for diagnostic continuity
    // (not consulted by the kernel body anymore).
    format!(
        "#[repr(C)]\n\
         #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
         pub struct {cfg_struct} {{ pub agent_cap: u32, pub tick: u32, pub slot_count: u32, pub _pad: u32 }}"
    )
}

fn build_view_decay_cfg_build_expr(cfg_struct: &str) -> String {
    // `slot_count = state.agent_cap()` is the safe default for
    // single-key views; PairMap-keyed runtimes overwrite this with
    // `agent_cap × second_pop` when they upload the decay cfg
    // per-tick.
    format!(
        "{cfg_struct} {{ agent_cap: state.agent_cap(), tick: state.tick as u32, slot_count: state.agent_cap(), _pad: 0 }}"
    )
}

/// Synthesise the 2-binding [`KernelBinding`] vector for a ViewDecay
/// kernel:
///
///   slot 0: view_storage_primary  (atomic storage — type-compat with
///                                  the ViewFold kernel binding shape
///                                  introduced by B1)
///   slot 1: cfg                   (uniform)
///
/// view_storage_primary uses [`BgSource::ViewHandle`] tagged with the
/// per-view resident accessor (`fold_view_<view_name>_handles`) — same
/// accessor the ViewFold kernel uses, so the host-side wiring already
/// exposes the right buffer.
fn build_view_decay_bindings(view_name: &str, cfg_struct: &str) -> Vec<KernelBinding> {
    let accessor = format!("fold_view_{view_name}_handles");
    vec![
        KernelBinding {
            slot: 0,
            name: "view_storage_primary".into(),
            // B1 (the ViewFold path's atomic-CAS body) declares
            // `view_storage_primary: array<atomic<u32>>`. ViewDecay's
            // dispatch is one thread per slot — there's no contention
            // across threads — but the WGSL type must match the shared
            // binding declaration the runtime hands the same buffer to,
            // hence atomic load + store.
            access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
            bg_source: BgSource::ViewHandle {
                accessor,
                tuple_idx: 0,
            },
        },
        KernelBinding {
            slot: 1,
            name: "cfg".into(),
            access: AccessMode::Uniform,
            wgsl_ty: cfg_struct.to_string(),
            bg_source: BgSource::Cfg,
        },
    ]
}

/// Compose the ViewDecay-specific WGSL body. One thread per slot;
/// reads `view_storage_primary[k]`, multiplies by the compile-time
/// decay rate, writes back. `atomicLoad` / `atomicStore` are required
/// for type compatibility with the `array<atomic<u32>>` binding shape
/// (see `build_view_decay_bindings`'s docstring); contention is
/// structurally absent (each thread owns one slot) so a non-atomic
/// path would be equally correct, but mixing atomic and non-atomic
/// access on the same WGSL binding is a validation error.
fn build_view_decay_wgsl_body(rate_bits: u32) -> String {
    let rate = f32::from_bits(rate_bits);
    let mut out = String::new();
    out.push_str("    let k = gid.x;\n");
    // `cfg.slot_count` is the over-allocated slot count for PairMap
    // views (= agent_cap × second_pop); for single-key views the
    // runtime sets `slot_count == agent_cap`, so the early-return
    // bound stays equivalent to the legacy `k >= cfg.agent_cap` form.
    out.push_str("    if (k >= cfg.slot_count) { return; }\n");
    writeln!(
        out,
        "    let old = bitcast<f32>(atomicLoad(&view_storage_primary[k]));"
    )
    .expect("write to String never fails");
    writeln!(out, "    let new_val = old * {rate:?};")
        .expect("write to String never fails");
    out.push_str(
        "    atomicStore(&view_storage_primary[k], bitcast<u32>(new_val));\n",
    );
    out
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

    // Stash the dispatch on the emit context so the body emit (and
    // any nested per-stmt emits) can pick a tile-walk WGSL form when
    // appropriate — see EmitCtx::dispatch's docstring. Restore the
    // prior value on exit so caller-context isn't clobbered (today
    // the prior value is always None, but defensive scoping makes
    // future re-entry safe).
    let prior_dispatch = ctx.dispatch.replace(Some(*dispatch));

    // Per-thread preamble — mirrors `ThreadIndexing` shape.
    write!(out, "{}", thread_indexing_preamble(dispatch))
        .expect("write to String never fails");

    // Tiled-MoveBoid preamble: when the kernel is dispatched
    // PerCell, every workgroup cooperates on a tile load before any
    // per-op body runs. The tile load is emitted ONCE here so that
    // multiple ForEachNeighbor stmts in the body all walk the same
    // pre-loaded tile. The matching `var<workgroup>` decls land at
    // module scope via `compose_wgsl_file`; this fragment only adds
    // the runtime cooperative-load loop + `workgroupBarrier` + per-
    // lane home-cell bounds check + agent_id fetch.
    if matches!(dispatch, DispatchShape::PerCell) {
        write!(out, "{}", tiled_per_cell_preamble())
            .expect("write to String never fails");
    }

    let is_per_event = matches!(dispatch, DispatchShape::PerEvent { .. });
    for (i, op_id) in body_ops.iter().enumerate() {
        if i > 0 {
            out.push_str("\n\n");
        }
        let op = resolve_op(prog, *op_id)?;
        let fragment = lower_op_body(op, dispatch, ctx)?;
        // Comment header per op for traceability.
        writeln!(out, "// op#{} ({})", op.id.0, compute_op_kind_short(&op.kind))
            .expect("write to String never fails");
        // Per-handler event-kind tag filter, mirroring
        // `build_view_fold_wgsl_body`'s wrapping. PerEvent-dispatched
        // ops carry an `on_event` discriminator (`Some(EventKindId)`
        // for ViewFold and PhysicsRule); each per-op body runs only
        // when the per-thread `event_ring[event_idx * stride + 0u]`
        // tag word matches that kind. Without the guard, every event
        // in the unified ring lands in every handler — multi-kind
        // fixtures (ActionSelected vs PrayCompleted, etc.) would
        // double-count any slot ranges the kinds share, and over-
        // dispatched workgroup-rounding threads would index garbage
        // payload words past the actual event count. Producer side
        // writes the kind id at offset 0 (see `lower_emit_to_wgsl` in
        // `wgsl_body.rs`).
        let fold_meta = if is_per_event {
            per_event_op_kind_tag(&op.kind, prog)
        } else {
            None
        };
        // Wrap the per-op body in `{ ... }` so locals declared by Task 1's
        // event-pattern binding lowering (`let local_<N>: <ty> = ...;`)
        // live in their own WGSL scope. Without these braces, two ops in
        // the same fused kernel both emit `let local_0` at the function
        // top level, which naga rejects as a redefinition. WGSL handles
        // nested scopes cleanly, so siblings can each carry their own
        // `local_0` without collision. The brace itself sits flush-left
        // (matching the comment header above it) and the body is
        // indented one 4-space level inside.
        //
        // With a tag guard we add a third level (4 more spaces) for
        // the body inside the if-block — same indent shape as the
        // ViewFold path's per-op wrapping.
        let extra_indent = if fold_meta.is_some() { 4 } else { 0 };
        let body_indent = 4 + extra_indent;
        let indented = fragment
            .lines()
            .map(|l| {
                if l.is_empty() {
                    String::new()
                } else {
                    format!("{:indent$}{l}", "", indent = body_indent)
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        out.push_str("{\n");
        if let Some((kind_tag, stride)) = fold_meta {
            // Read the per-event tag word. The producer-side `Emit`
            // writes the kind id at offset 0 of each record (see
            // `lower_emit_to_wgsl`); the consumer reads it the same
            // way. Use `atomicLoad` when the binding is atomic
            // (PerEventEmit kernels) — plain index reads on an
            // atomic-typed binding fail WGSL validation.
            let tag_read = if ctx.event_ring_atomic_loads.get() {
                format!("atomicLoad(&event_ring[event_idx * {stride}u + 0u])")
            } else {
                format!("event_ring[event_idx * {stride}u + 0u]")
            };
            out.push_str(&format!(
                "    if ({tag_read} == {kind_tag}u) {{\n"
            ));
            out.push_str(&indented);
            out.push_str("\n    }");
        } else {
            out.push_str(&indented);
        }
        out.push_str("\n}");
    }

    // Tiled-MoveBoid kernels open a `for (var _home_iter ...)` loop
    // in their preamble (see `tiled_per_cell_preamble`) so each lane
    // can process more than one home agent at high density. Close
    // the matching brace here, *after* every per-op body has been
    // emitted, so the loop body covers the full physics computation.
    if matches!(dispatch, DispatchShape::PerCell) {
        out.push_str("\n}");
    }

    // Restore prior dispatch.
    ctx.dispatch.set(prior_dispatch);

    Ok(out)
}

/// Cooperative tile-load preamble for [`DispatchShape::PerCell`]
/// kernels. Emitted once at the top of the kernel body (before any
/// per-op body) so every fold inside the body walks the same
/// pre-loaded tile of the 27-cell neighborhood.
///
/// The preamble assumes the kernel's `@compute @workgroup_size(...)`
/// + module-level `var<workgroup>` declarations have already been
/// emitted by [`super::program::compose_wgsl_file`] (see its
/// PerCell branch). It writes:
/// 1. `let home_cell = workgroup_id.x; let lane = local_invocation_id.x;`
/// 2. Decode `home_cell` into `(home_cx, home_cy, home_cz)` integer
///    cell coordinates.
/// 3. Each lane in `0..27` cooperatively loads one neighbor cell's
///    pos/vel slots into `tile_pos[lane * MAX]` / `tile_vel[lane * MAX]`,
///    plus a `tile_count[lane]` for the per-cell agent count.
/// 4. `workgroupBarrier()` so every lane sees the populated tile.
/// 5. Per-lane home-cell bounds check (early-return if the lane
///    doesn't correspond to a populated home-cell slot).
/// 6. `let agent_id = spatial_grid_cells[home_cell * MAX + lane]`.
///
/// Each lane in `27..MAX_PER_CELL` participates in the load (no-op
/// branches just skip), then the home-cell bounds check filters them
/// out before any sim work.
fn tiled_per_cell_preamble() -> String {
    // After the three-phase counting sort, `spatial_grid_starts[c]`
    // is cell `c`'s start position in `spatial_grid_cells` and
    // `starts[c+1] - starts[c]` is the cell's count. The tile-load
    // still caps at SPATIAL_MAX_PER_CELL (the workgroup-shared
    // memory budget for the tile arrays), so cells with more than
    // MAX_PER_CELL agents truncate at the tile-load boundary
    // *visible to that workgroup's neighbors* — but they're NOT
    // dropped from the spatial structure itself (the
    // counting-sort layout is uncapped). The diagnostic kernel
    // surfaces tile-truncation as the dropped_agents stat now
    // computed against MAX_PER_CELL, which is the right knob to
    // bump or split into multi-pass tile loads.
    "let home_cell = workgroup_id.x;\n\
     let lane = local_invocation_id.x;\n\
     let home_cz = home_cell / (SPATIAL_GRID_DIM * SPATIAL_GRID_DIM);\n\
     let home_cy = (home_cell / SPATIAL_GRID_DIM) % SPATIAL_GRID_DIM;\n\
     let home_cx = home_cell % SPATIAL_GRID_DIM;\n\
     // Cooperative tile load: lanes 0..27 each load one neighbor\n\
     // cell. Lanes 27..MAX_PER_CELL skip the load (no-op).\n\
     if (lane < 27u) {\n\
     \x20   let nbr = lane;\n\
     \x20   let dz = i32(nbr / 9u) - 1;\n\
     \x20   let dy = i32((nbr / 3u) % 3u) - 1;\n\
     \x20   let dx = i32(nbr % 3u) - 1;\n\
     \x20   let _nbr_cell = cell_index(\n\
     \x20       i32(home_cx) + dx,\n\
     \x20       i32(home_cy) + dy,\n\
     \x20       i32(home_cz) + dz,\n\
     \x20   );\n\
     \x20   let _nbr_start = spatial_grid_starts[_nbr_cell];\n\
     \x20   let _nbr_end = spatial_grid_starts[_nbr_cell + 1u];\n\
     \x20   let _nbr_count = min(_nbr_end - _nbr_start, SPATIAL_MAX_PER_CELL);\n\
     \x20   tile_count[nbr] = _nbr_count;\n\
     \x20   let _dst_base = nbr * SPATIAL_MAX_PER_CELL;\n\
     \x20   for (var i: u32 = 0u; i < _nbr_count; i = i + 1u) {\n\
     \x20       let _aid = spatial_grid_cells[_nbr_start + i];\n\
     \x20       tile_pos[_dst_base + i] = agent_pos[_aid];\n\
     \x20       tile_vel[_dst_base + i] = agent_vel[_aid];\n\
     \x20   }\n\
     }\n\
     workgroupBarrier();\n\
     let _home_start = spatial_grid_starts[home_cell];\n\
     let _home_end = spatial_grid_starts[home_cell + 1u];\n\
     let _home_count = _home_end - _home_start;\n\
     // Multi-iteration home-agent loop: each lane processes home\n\
     // agents at offsets `{lane, lane + WG, lane + 2*WG, ...}` up to\n\
     // _home_count. With workgroup_size = MAX_PER_CELL = 32 and a\n\
     // home cell holding (say) 94 agents, lane 0 sees agents\n\
     // {0, 32, 64}, lane 1 sees {1, 33, 65}, etc. — every home\n\
     // agent gets processed regardless of cell density. The matching\n\
     // close brace is appended by `build_wgsl_body`'s tail when\n\
     // the dispatch is PerCell.\n\
     for (var _home_iter: u32 = lane; _home_iter < _home_count; _home_iter = _home_iter + SPATIAL_MAX_PER_CELL) {\n\
     let agent_id = spatial_grid_cells[_home_start + _home_iter];\n\n"
        .to_string()
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
/// WGSL body template for [`SpatialQueryKind::BuildHash`].
///
/// **Real bounded-list counting sort** — per-agent dispatch.
///
/// Each agent computes its cell, atomic-fetch-adds into the cell's
/// count, and (if the cell hasn't overflowed `MAX_PER_CELL`) writes
/// its agent_id into that slot. Cells with more than
/// `SPATIAL_MAX_PER_CELL` agents silently drop the overflow — the
/// dropped agents are absent from spatial-query results until the
/// host clears the offsets buffer next tick.
///
/// # Pre-condition
///
/// The host MUST clear `spatial_grid_offsets` to all-zero before
/// dispatching this kernel each tick (e.g. via
/// `queue.write_buffer(spatial_grid_offsets, 0, &zeros)`). Without
/// the clear, atomicAdd accumulates across ticks and slot indices
/// diverge from per-cell counts.
///
/// # Bindings consumed
/// - `agent_pos`            (read; resolves to `agent_pos[agent_id]`)
/// - `spatial_grid_cells`   (read_write; flat slot array, sized
///   `num_cells * MAX_PER_CELL`)
/// - `spatial_grid_offsets` (atomic; per-cell count, sized
///   `num_cells`)
/// - `cfg`                  (uniform, agent_cap)
///
/// The constants + `pos_to_cell` helper land via
/// [`super::spatial::compose_spatial_prelude`] which fires whenever
/// `spatial_grid_*` appears in the body — already true here.
const SPATIAL_BUILD_HASH_BODY: &str = "let cell = pos_to_cell(agent_pos[agent_id]);\n\
    let slot = atomicAdd(&spatial_grid_offsets[cell], 1u);\n\
    if (slot < SPATIAL_MAX_PER_CELL) {\n\
    \x20   spatial_grid_cells[cell * SPATIAL_MAX_PER_CELL + slot] = agent_id;\n\
    }";

/// Real counting sort, **phase 1** (`SpatialQueryKind::BuildHashCount`).
///
/// Per-agent dispatch: each agent computes its cell and atomic-
/// increments `spatial_grid_offsets[cell]`. After this kernel the
/// offsets buffer holds the *raw count* of agents per cell. No
/// truncation — every agent contributes regardless of cell density.
///
/// **Pre-condition**: the host clears `spatial_grid_offsets` to zero
/// before dispatch (the boids_runtime `step()` does this via a
/// `copy_buffer_to_buffer` from a pre-allocated zero source).
const SPATIAL_BUILD_HASH_COUNT_BODY: &str =
    "let cell = pos_to_cell(agent_pos[agent_id]);\n\
     atomicAdd(&spatial_grid_offsets[cell], 1u);";

/// Real counting sort, **phase 2a — workgroup-local scan**
/// (`SpatialQueryKind::BuildHashScanLocal`).
///
/// PerScanChunk dispatch: each workgroup owns one
/// `PER_SCAN_CHUNK_WORKGROUP_X`-sized cell chunk. The body performs
/// a Hillis-Steele inclusive scan in workgroup-shared memory,
/// writes the per-chunk inclusive prefix to
/// `spatial_grid_starts[chunk_base + lane + 1]`, and writes the
/// chunk's total to `spatial_chunk_sums[chunk_id]`. Lane 0 of
/// workgroup 0 also writes `spatial_grid_starts[0] = 0u`.
///
/// `var<workgroup>` shared memory holds 256 × u32 = 1 KiB —
/// well under any adapter's per-workgroup memory budget.
///
/// Out-of-range cells (`chunk_base + lane >= num_cells`) read 0
/// and skip their slot writes.
fn spatial_build_hash_scan_local_body() -> String {
    let chunk = crate::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
    let n = crate::cg::emit::spatial::num_cells();
    format!(
        "// Hillis-Steele inclusive scan in workgroup-shared memory.\n\
         // The shared array's bounds + the chunk size are both\n\
         // PER_SCAN_CHUNK_WORKGROUP_X ({chunk}) so every lane has\n\
         // exactly one slot.\n\
         let cell = chunk_id * {chunk}u + lane;\n\
         // Out-of-range lanes contribute 0 to the scan but still\n\
         // participate in every barrier — the Hillis-Steele algorithm\n\
         // requires uniform control flow across the workgroup.\n\
         var v: u32 = 0u;\n\
         if (cell < {n}u) {{\n\
         \x20   v = atomicLoad(&spatial_grid_offsets[cell]);\n\
         }}\n\
         scan_shared[lane] = v;\n\
         workgroupBarrier();\n\
         // log2({chunk}) iterations of double-and-add.\n\
         for (var step: u32 = 1u; step < {chunk}u; step = step << 1u) {{\n\
         \x20   var added: u32 = scan_shared[lane];\n\
         \x20   if (lane >= step) {{\n\
         \x20       added = added + scan_shared[lane - step];\n\
         \x20   }}\n\
         \x20   workgroupBarrier();\n\
         \x20   scan_shared[lane] = added;\n\
         \x20   workgroupBarrier();\n\
         }}\n\
         // Now scan_shared[lane] = inclusive sum over lanes [0..=lane].\n\
         // Write per-chunk inclusive prefix to grid_starts[cell+1] for\n\
         // in-range cells (the +1 shift turns inclusive→exclusive at\n\
         // the next position, matching the serial scan's output).\n\
         if (cell < {n}u) {{\n\
         \x20   spatial_grid_starts[cell + 1u] = scan_shared[lane];\n\
         }}\n\
         // Lane 0 of workgroup 0 sets the leading exclusive 0.\n\
         if (chunk_id == 0u && lane == 0u) {{\n\
         \x20   spatial_grid_starts[0u] = 0u;\n\
         }}\n\
         // Last lane in each workgroup records the chunk's total.\n\
         if (lane == {chunk}u - 1u) {{\n\
         \x20   spatial_chunk_sums[chunk_id] = scan_shared[lane];\n\
         }}",
        chunk = chunk,
        n = n,
    )
}

/// Real counting sort, **phase 2b — cross-workgroup carry**
/// (`SpatialQueryKind::BuildHashScanCarry`).
///
/// OneShot dispatch (single thread). Serially exclusive-scans
/// `spatial_chunk_sums` in place: each entry overwritten with the
/// sum of all preceding chunk totals. After this kernel,
/// `chunk_sums[chunk_id]` is the global offset to add to every
/// `grid_starts` entry the chunk owns.
///
/// `num_chunks = ceil(num_cells / PER_SCAN_CHUNK_WORKGROUP_X)` —
/// 42 for boids' 22³=10 648 grid; 256 even at 16M cells. The
/// serial loop is microseconds at this scale.
fn spatial_build_hash_scan_carry_body() -> String {
    let chunk = crate::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
    let n = crate::cg::emit::spatial::num_cells();
    let num_chunks = n.div_ceil(chunk);
    format!(
        "// Serial exclusive scan over the (small) chunk_sums buffer.\n\
         var running: u32 = 0u;\n\
         for (var i: u32 = 0u; i < {num_chunks}u; i = i + 1u) {{\n\
         \x20   let v = spatial_chunk_sums[i];\n\
         \x20   spatial_chunk_sums[i] = running;\n\
         \x20   running = running + v;\n\
         }}",
        num_chunks = num_chunks,
    )
}

/// Real counting sort, **phase 2c — add per-chunk base**
/// (`SpatialQueryKind::BuildHashScanAdd`).
///
/// PerScanChunk dispatch (same shape as phase 2a). Each lane reads
/// the chunk base from `spatial_chunk_sums[chunk_id]` and adds it
/// to its `spatial_grid_starts[cell + 1]` slot, where `cell =
/// chunk_id * PER_SCAN_CHUNK_WORKGROUP_X + lane`. Also resets
/// `spatial_grid_offsets[cell]` to zero so phase 3 can reuse it as
/// a write cursor (mirrors the serial scan's combined behaviour).
///
/// Lane `cell == num_cells - 1` (the very last in-range lane) ends
/// up with `starts[num_cells] = total agent count`, which the
/// tiled-MoveBoid kernel + scatter consume.
fn spatial_build_hash_scan_add_body() -> String {
    let chunk = crate::cg::dispatch::PER_SCAN_CHUNK_WORKGROUP_X;
    let n = crate::cg::emit::spatial::num_cells();
    format!(
        "let cell = chunk_id * {chunk}u + lane;\n\
         if (cell >= {n}u) {{ return; }}\n\
         let base = spatial_chunk_sums[chunk_id];\n\
         spatial_grid_starts[cell + 1u] = spatial_grid_starts[cell + 1u] + base;\n\
         atomicStore(&spatial_grid_offsets[cell], 0u);",
        chunk = chunk,
        n = n,
    )
}

/// Real counting sort, **phase 3** (`SpatialQueryKind::BuildHashScatter`).
///
/// Per-agent dispatch: each agent computes its cell again, claims a
/// per-cell slot via `atomicAdd(&offsets[cell], 1u)` (offsets reset
/// to zero by phase 2; here it acts as a write cursor in
/// `[0 .. count)`), then writes its id at the absolute slot
/// `starts[cell] + local_slot` in `spatial_grid_cells`. After this
/// kernel `spatial_grid_cells` holds every agent id grouped by cell.
const SPATIAL_BUILD_HASH_SCATTER_BODY: &str =
    "let cell = pos_to_cell(agent_pos[agent_id]);\n\
     let local_slot = atomicAdd(&spatial_grid_offsets[cell], 1u);\n\
     spatial_grid_cells[spatial_grid_starts[cell] + local_slot] = agent_id;";

/// WGSL body template for [`SpatialQueryKind::FilteredWalk`].
///
/// Walks the per-cell neighborhood for each agent, evaluates the
/// lowered filter expression per candidate, writes accepted
/// candidates into `spatial_query_results`. The filter WGSL is
/// interpolated into the `{filter_wgsl}` slot via [`format!`] in
/// [`spatial_filtered_walk_body`].
///
/// Bindings consumed (per `SpatialQueryKind::FilteredWalk` static
/// dependencies + filter walk's collected reads):
/// - `spatial_grid_cells`        (Pool, read)
/// - `spatial_grid_offsets`      (Pool, read)
/// - `spatial_query_results`     (Pool, read_write)
/// - `agent_*`                   (per-field reads collected from the filter expression)
/// - `cfg`                       (uniform, agent_cap)
///
/// # Limitations
///
/// - **Stub per-cell walk.** Mirrors the structural shape of the
///   legacy `engine_gpu_rules/src/spatial_kin_query.wgsl` until
///   runtime BGL wiring matures. The per-cell + per-candidate
///   iteration is structural; the filter WGSL is real and the
///   accept-write is real, but agent_pos lookups + radius checks
///   are not yet emitted at this task.
/// - **No per-cell radius bounds.** The walk visits every cell
///   in the grid — quadratic in agent count for the smoke fixture.
///   Acceptable for v1; bounded radius lookup is a follow-up.
fn spatial_filtered_walk_body(filter_wgsl: &str) -> String {
    format!(
        "// SpatialQuery::FilteredWalk — per-cell walk + per-candidate filter.\n\
         // Touches every binding so naga keeps them live; structural\n\
         // walk shape mirrors the legacy spatial_kin_query.wgsl stub.\n\
         var write_cursor: u32 = 0u;\n\
         for (var cell: u32 = 0u; cell < cfg.agent_cap; cell = cell + 1u) {{\n\
         \x20   let cell_start = spatial_grid_offsets[cell];\n\
         \x20   let cell_end = spatial_grid_offsets[cell + 1u];\n\
         \x20   for (var slot: u32 = cell_start; slot < cell_end; slot = slot + 1u) {{\n\
         \x20       let candidate = spatial_grid_cells[slot];\n\
         \x20       let filter_value: bool = {filter_wgsl};\n\
         \x20       if (filter_value) {{\n\
         \x20           spatial_query_results[write_cursor] = candidate;\n\
         \x20           write_cursor = write_cursor + 1u;\n\
         \x20       }}\n\
         \x20   }}\n\
         }}\n\
         _ = cfg.agent_cap;",
        filter_wgsl = filter_wgsl
    )
}

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
            // Snapshot the pending-target-let buffer so we can detect
            // entries pushed by the predicate sub-tree (typically
            // `items.<field>(<idx>)` reads, which queue a
            // `let target_expr_<N>: u32 = <wgsl>;` pre-binding via
            // [`crate::cg::emit::wgsl_body::EmitCtx::pending_target_lets`]).
            // The mask body templates substitute the predicate WGSL
            // inline as a single expression — without draining the
            // buffer here, the predicate references `target_expr_<N>`
            // identifiers that no enclosing stmt ever hoists.
            // trade_market_real surfaced this for `items.base_price(0)`
            // inside a verb's `when` clause.
            //
            // After lowering the predicate expression we drain any new
            // entries from the buffer and prepend them as plain
            // `let target_expr_<N>: u32 = <expr>;` lines BEFORE the
            // template body — the templates' `let mask_<id>_value`
            // line then sees the bindings already in scope.
            let snapshot_len = ctx.pending_target_lets.borrow().len();
            let predicate_wgsl = lower_cg_expr_to_wgsl(*predicate, ctx)?;
            let mut pending = ctx.pending_target_lets.borrow_mut();
            let new_lets: Vec<(crate::cg::CgExprId, String)> =
                pending.drain(snapshot_len..).collect();
            drop(pending);
            // The hoisted let lines may reference `agent_id` (PerAgent
            // dispatch) or `agent_id` / `per_pair_candidate` (PerPair),
            // so they must land AFTER the template binds those ids —
            // i.e. spliced into the predicate text before the
            // `let mask_<id>_value: bool = ...;` line. We do this by
            // wrapping the predicate in a block: `{ <lets>; <pred_expr> }`
            // is illegal WGSL, but we can emit the lets as siblings of
            // `let mask_<id>_value` by injecting them via the template
            // string — see `mask_predicate_*_body` which now accepts
            // a `prefix_lets` argument.
            let lets_prefix = if new_lets.is_empty() {
                String::new()
            } else {
                let lines: Vec<String> = new_lets
                    .iter()
                    .map(|(id, w)| format!("    let target_expr_{}: u32 = {};", id.0, w))
                    .collect();
                format!("{}\n", lines.join("\n"))
            };
            match dispatch {
                DispatchShape::PerAgent => Ok(mask_predicate_per_agent_body_with_prefix(
                    *mask,
                    &lets_prefix,
                    &predicate_wgsl,
                )),
                DispatchShape::PerPair { .. } => Ok(mask_predicate_per_pair_body_with_prefix(
                    *mask,
                    &lets_prefix,
                    &predicate_wgsl,
                )),
                DispatchShape::PerEvent { .. }
                | DispatchShape::OneShot
                | DispatchShape::PerWord
                | DispatchShape::PerCell
                | DispatchShape::PerScanChunk => Err(KernelEmitError::InvalidDispatchForOpKind {
                    op_kind: "MaskPredicate",
                    dispatch: format!("{dispatch}"),
                }),
            }
        }
        ComputeOpKind::PhysicsRule { body, .. } => {
            // Both per-event (on_event = Some) and per-agent
            // (on_event = None) PhysicsRules walk their CgStmtList
            // body via the same lowering path. The kernel preamble
            // differs by dispatch shape (per-agent binds `agent_id`
            // — Phase 6 Task 3; per-event binds `event_idx` plus
            // per-event payload locals), but the body itself is just
            // a list of CG statements either way.
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
            SpatialQueryKind::FilteredWalk { filter } => {
                let filter_wgsl = lower_cg_expr_to_wgsl(*filter, ctx)?;
                spatial_filtered_walk_body(&filter_wgsl)
            }
            SpatialQueryKind::CompactNonemptyCells => {
                "// SpatialQuery::CompactNonemptyCells — placeholder; \
                 see SpatialQueryKind::CompactNonemptyCells docs.\n\
                 _ = spatial_grid_offsets;\n\
                 _ = spatial_nonempty_cells;\n\
                 _ = spatial_nonempty_indirect_args;"
                    .to_string()
            }
            SpatialQueryKind::BuildHashCount => SPATIAL_BUILD_HASH_COUNT_BODY.to_string(),
            SpatialQueryKind::BuildHashScanLocal => spatial_build_hash_scan_local_body(),
            SpatialQueryKind::BuildHashScanCarry => spatial_build_hash_scan_carry_body(),
            SpatialQueryKind::BuildHashScanAdd => spatial_build_hash_scan_add_body(),
            SpatialQueryKind::BuildHashScatter => SPATIAL_BUILD_HASH_SCATTER_BODY.to_string(),
        }),
        // Task 5.6d: per-`PlumbingKind` body dispatch. Exhaustive over
        // every variant so adding a new kind forces an explicit body
        // decision here. See the const-by-const docs below for the
        // legacy-port rationale and the binding-name normalization
        // that keeps each fragment compatible with the structural
        // [`KernelSpec`] layout (no spec-level renames).
        ComputeOpKind::Plumbing { kind } => Ok(plumbing_body_for_kind(kind)),
        // ViewDecay routes through the dedicated ViewDecay-classed
        // path in `kernel_topology_to_spec_and_body` (the body is
        // hand-synthesised from the rate constant). Reaching this arm
        // means a singleton ViewDecay topology slipped past the
        // classifier into the generic pipeline — emit a documented
        // TODO so the WGSL still compiles structurally.
        ComputeOpKind::ViewDecay { .. } => Ok(
            "// TODO(b2): ViewDecay op in generic pipeline — \
             classifier should have routed through ViewDecay path."
                .to_string(),
        ),
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
/// **Pair-field rows (Gap #4 close).** A row whose utility / target /
/// guard expressions transitively reference
/// [`crate::cg::expr::CgExpr::PerPairCandidateId`] (or read any
/// `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`) is
/// emitted with an inner candidate loop:
///
/// ```text
/// for (var per_pair_candidate: u32 = 0u;
///      per_pair_candidate < cfg.agent_cap;
///      per_pair_candidate = per_pair_candidate + 1u) {
///     <row body — utility/target/guard reference per_pair_candidate>
///     // best_target defaults to per_pair_candidate when the row's
///     // explicit target is None (the verb-synth case).
/// }
/// ```
///
/// This implements per-(actor, candidate) scoring for pair-field rows
/// like `Heal(target: Agent) score (1000.0 - target.cooldown_next_ready_tick)`
/// — every actor evaluates the score against every candidate slot and
/// argmaxes across candidates. Rows that don't reference the per-pair
/// candidate side (the conventional `self`-only utility — `Hold`,
/// `MoveToward`, etc.) skip the inner loop and behave exactly as
/// before.
///
/// The mask gate (when an action's name matches a registered mask) sits
/// OUTSIDE the inner loop — it filters at the actor level. The
/// per-pair predicate (the verb's `when` clause) is captured in the
/// mask body's PerPair dispatch (`atomicOr` sets the agent bit when
/// any candidate satisfies); the scoring kernel's inner loop simply
/// iterates every candidate slot and lets the score expression
/// determine winning candidates.
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
        // Detect whether any of the row's expressions transitively
        // reference the per-pair candidate side. If so, the row is a
        // pair-field row and its body must execute inside an inner
        // candidate loop — every candidate slot contributes one
        // utility evaluation, and the argmax runs across (actor,
        // candidate) pairs. See the module-level doc above.
        let row_is_per_pair = row_references_per_pair_candidate(row, ctx);

        let utility_wgsl = lower_cg_expr_to_wgsl(row.utility, ctx)?;
        let target_wgsl = match row.target {
            Some(target_id) => lower_cg_expr_to_wgsl(target_id, ctx)?,
            // Pair-field rows with no explicit target expression
            // (the verb-synth case — `lower_standard_row` always
            // produces `target: None`) default the winning target
            // to the inner loop's `per_pair_candidate`. Non-pair
            // rows fall through to the sentinel as before.
            None if row_is_per_pair => "per_pair_candidate".to_string(),
            None => "0xFFFFFFFFu".to_string(),
        };
        let action_lit = format!("{}u", row.action.0);

        // Verb-cascade gap #3 (probe report 2026-05-04): a row whose
        // action name matches a registered mask is gated by that
        // mask's per-agent bitmap. The mask name is the verb's
        // `verb_<Name>` synthetic name (see `cg::lower::verb_expand`)
        // — the same name the verb expander uses for both the mask
        // head and the scoring entry head, so the action-id ↔
        // mask-id bridge is exactly the shared interner name. Rows
        // with no matching mask (the conventional standard heads —
        // `Hold`, `MoveToward`, etc.) skip the gate and behave as
        // before.
        //
        // The bitmap lives on `mask_<id>_bitmap` (atomic u32 array,
        // see `BindingMetadata` for `DataHandle::MaskBitmap`); the
        // driver records the corresponding `MaskBitmap` read on the
        // ScoringArgmax op so the binding scanner declares the
        // identifier (`cg::lower::driver::wire_scoring_mask_reads`).
        //
        // The mask gate sits OUTSIDE the inner candidate loop for
        // pair-field rows — it filters at the actor level. (The
        // per-pair `when` predicate is encoded in the mask's PerPair
        // dispatch body, which sets the agent bit when any candidate
        // satisfies; the scoring kernel does not re-check the
        // per-pair predicate inside the inner loop in today's
        // landing.)
        let mask_gate_open = mask_for_action(row.action, ctx).map(|mask_id| {
            format!(
                "    let mask_{0}_word_for_row_{i} = agent_id >> 5u;\n\
                 \x20   let mask_{0}_bit_for_row_{i}  = 1u << (agent_id & 31u);\n\
                 \x20   let mask_{0}_loaded_for_row_{i} = atomicLoad(&mask_{0}_bitmap[mask_{0}_word_for_row_{i}]);\n\
                 \x20   if ((mask_{0}_loaded_for_row_{i} & mask_{0}_bit_for_row_{i}) != 0u) {{\n",
                mask_id.0,
            )
        });
        let mask_gate_close = if mask_gate_open.is_some() { "    }\n" } else { "" };

        // Pair-field rows wrap the row body in `for (var
        // per_pair_candidate: u32 = 0u; per_pair_candidate <
        // cfg.agent_cap; per_pair_candidate = per_pair_candidate +
        // 1u) { ... }`. Iteration is in slot order to satisfy P11
        // (Reduction Determinism — argmax over candidates is
        // deterministic when the iteration order is fixed).
        let (loop_open, loop_close) = if row_is_per_pair {
            (
                format!(
                    "    for (var per_pair_candidate: u32 = 0u; \
                     per_pair_candidate < cfg.agent_cap; \
                     per_pair_candidate = per_pair_candidate + 1u) {{\n"
                ),
                "    }\n".to_string(),
            )
        } else {
            (String::new(), String::new())
        };

        match row.guard {
            Some(guard_id) => {
                let guard_wgsl = lower_cg_expr_to_wgsl(guard_id, ctx)?;
                writeln!(out, "// row {i}: action=#{} (guarded)", row.action.0).unwrap();
                writeln!(out, "{{").unwrap();
                if let Some(open) = &mask_gate_open {
                    out.push_str(open);
                }
                out.push_str(&loop_open);
                writeln!(out, "    let guard_{i}: bool = {guard_wgsl};").unwrap();
                writeln!(out, "    if (guard_{i}) {{").unwrap();
                writeln!(out, "        let utility_{i}: f32 = {utility_wgsl};").unwrap();
                writeln!(out, "        if (utility_{i} > best_utility) {{").unwrap();
                writeln!(out, "            best_utility = utility_{i};").unwrap();
                writeln!(out, "            best_action = {action_lit};").unwrap();
                writeln!(out, "            best_target = {target_wgsl};").unwrap();
                writeln!(out, "        }}").unwrap();
                writeln!(out, "    }}").unwrap();
                out.push_str(&loop_close);
                out.push_str(mask_gate_close);
                writeln!(out, "}}").unwrap();
            }
            None => {
                writeln!(out, "// row {i}: action=#{}", row.action.0).unwrap();
                writeln!(out, "{{").unwrap();
                if let Some(open) = &mask_gate_open {
                    out.push_str(open);
                }
                out.push_str(&loop_open);
                writeln!(out, "    let utility_{i}: f32 = {utility_wgsl};").unwrap();
                writeln!(out, "    if (utility_{i} > best_utility) {{").unwrap();
                writeln!(out, "        best_utility = utility_{i};").unwrap();
                writeln!(out, "        best_action = {action_lit};").unwrap();
                writeln!(out, "        best_target = {target_wgsl};").unwrap();
                writeln!(out, "    }}").unwrap();
                out.push_str(&loop_close);
                out.push_str(mask_gate_close);
                writeln!(out, "}}").unwrap();
            }
        }
        out.push('\n');
    }

    out.push_str("let scoring_base: u32 = agent_id * 4u;\n");
    out.push_str("scoring_output[scoring_base + 0u] = best_action;\n");
    out.push_str("scoring_output[scoring_base + 1u] = best_target;\n");
    out.push_str("scoring_output[scoring_base + 2u] = bitcast<u32>(best_utility);\n");
    out.push_str("scoring_output[scoring_base + 3u] = 0u;\n");

    // -- ActionSelected event emit ----------------------------------------
    //
    // Emit a per-agent `ActionSelected { actor: agent_id, action_id:
    // best_action, target: best_target }` event so the verb cascade
    // physics rules (synthesised by `cg::lower::verb_expand`'s
    // `verb_chronicle_<name>` injection) have something to consume. The
    // chronicle rule's `if action_id == <verb_id>` filter handles
    // per-verb gating downstream — the scoring kernel emits one event
    // per agent per tick unconditionally, mirroring the existing
    // producer pattern (`atomicAdd(&event_tail[0], 1u)` for slot
    // ownership + `atomicStore(&event_ring[slot * stride + N], ...)`
    // per payload word; see `wgsl_body.rs::lower_emit_to_wgsl`).
    //
    // Gated on the program containing an `ActionSelected` event kind:
    // a fixture without any `verb` decl lacks the synthesised event
    // (verb_expand's `inject_action_selected_event` only fires when at
    // least one verb has a non-empty `emit`), so the scoring kernel
    // skips the emit and stays binding-clean (no event_ring /
    // event_tail bindings declared on the kernel — see
    // `compute_dependencies` for ScoringArgmax in `cg::op`).
    //
    // The driver is responsible for recording the
    // `EventRing { Append }` write on the ScoringArgmax op's writes
    // list when the ActionSelected event exists; without that, the
    // binding scanner won't emit the `event_ring` + `event_tail`
    // bindings the WGSL below references. See
    // `cg::lower::driver::wire_action_selected_writes`.
    if let Some(action_selected_kind) = find_action_selected_kind(ctx.prog) {
        let layout = ctx
            .prog
            .event_layouts
            .get(&action_selected_kind.0)
            .expect("ActionSelected layout registered alongside the event kind");
        let stride = layout.record_stride_u32;
        let header = layout.header_word_count;
        // Field offsets within the record. The verb expander pins the
        // ActionSelected payload as `{ actor: AgentId, action_id: U32,
        // target: AgentId }` in declaration order — actor is the first
        // payload word (offset = header + 0), action_id the second,
        // target the third. We hardcode the offsets here rather than
        // looking them up by field name because the verb expander
        // owns the schema and the offsets are stable contract.
        let actor_off = header;
        let action_id_off = header + 1;
        let target_off = header + 2;
        let kind_tag = action_selected_kind.0;

        out.push('\n');
        out.push_str("// emit ActionSelected { actor: agent_id, action_id: best_action, target: best_target }\n");
        // Gate the emit on `best_utility > -inf` — a row whose mask
        // never set the agent's bit (no row passed argmax) leaves
        // best_utility at the sentinel; emitting in that case
        // produces a phantom ActionSelected whose target = NO_TARGET
        // (0xFFFFFFFFu) which downstream chronicles + ApplyDamage
        // dereference as `agent_<field>[0xFFFFFFFFu]` — out-of-bounds
        // write. Surfaced by duel_1v1 (2026-05-04 discovery).
        out.push_str("if (best_utility > -3.4028235e38) {\n");
        out.push_str("    let slot = atomicAdd(&event_tail[0], 1u);\n");
        writeln!(
            out,
            "    if (slot < {}u) {{",
            crate::cg::emit::wgsl_body::default_event_ring_cap_slots(),
        )
        .unwrap();
        writeln!(
            out,
            "        atomicStore(&event_ring[slot * {stride}u + 0u], {kind_tag}u);"
        )
        .unwrap();
        writeln!(
            out,
            "        atomicStore(&event_ring[slot * {stride}u + 1u], tick);"
        )
        .unwrap();
        writeln!(
            out,
            "        atomicStore(&event_ring[slot * {stride}u + {actor_off}u], agent_id);"
        )
        .unwrap();
        writeln!(
            out,
            "        atomicStore(&event_ring[slot * {stride}u + {action_id_off}u], best_action);"
        )
        .unwrap();
        writeln!(
            out,
            "        atomicStore(&event_ring[slot * {stride}u + {target_off}u], best_target);"
        )
        .unwrap();
        out.push_str("    }\n");
        // Closing brace for the `if (best_utility > -inf) { ... }`
        // outer guard (replaces the prior unconditional emit block).
        out.push_str("}");
    }

    Ok(out)
}

/// True iff any of the row's expressions (utility / target / guard)
/// transitively references the per-pair candidate side — either the
/// bare candidate id ([`crate::cg::expr::CgExpr::PerPairCandidateId`])
/// or any `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`.
///
/// Used by [`lower_scoring_argmax_body`] to decide whether the row's
/// body must execute inside an inner candidate loop. See the doc on
/// `lower_scoring_argmax_body` for the loop shape and rationale.
fn row_references_per_pair_candidate(row: &ScoringRowOp, ctx: &EmitCtx<'_>) -> bool {
    if expr_references_per_pair_candidate(row.utility, ctx) {
        return true;
    }
    if let Some(target_id) = row.target {
        if expr_references_per_pair_candidate(target_id, ctx) {
            return true;
        }
    }
    if let Some(guard_id) = row.guard {
        if expr_references_per_pair_candidate(guard_id, ctx) {
            return true;
        }
    }
    false
}

/// Recursive walker for [`row_references_per_pair_candidate`]. Returns
/// true when the expression tree rooted at `expr_id` contains a
/// [`crate::cg::expr::CgExpr::PerPairCandidateId`] node OR a
/// `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`.
///
/// The walk is depth-first in operand order. `Read` of a non-pair
/// `AgentField` short-circuits to false at that node (the field's
/// target side is the read's identity — it doesn't synthesise a
/// per-pair reference). `Read` of any other handle (`MaskBitmap`,
/// `Rng`, `ConfigConst`, …) is structurally per-actor / global and
/// returns false. The recursion terminates at literals, bare
/// `AgentSelfId`, `ReadLocal`, `EventField`, `NamespaceField`, and
/// `Rng` — none of which carry candidate-side semantics.
fn expr_references_per_pair_candidate(
    expr_id: crate::cg::data_handle::CgExprId,
    ctx: &EmitCtx<'_>,
) -> bool {
    use crate::cg::expr::{CgExpr, ExprArena};
    let Some(node) = ExprArena::get(ctx.prog, expr_id) else {
        return false;
    };
    match node {
        CgExpr::PerPairCandidateId => true,
        CgExpr::Read(handle) => {
            if let DataHandle::AgentField {
                target: crate::cg::data_handle::AgentRef::PerPairCandidate,
                ..
            } = handle
            {
                return true;
            }
            // `AgentRef::Target(expr_id)` carries an embedded index
            // expression — recurse into it so a `Read(AgentField {
            // target: Target(per_pair_candidate_expr) })` form is
            // detected (today's `lower_standard_row` produces
            // `PerPairCandidate` directly, not `Target(...)`, but
            // covering both keeps the walker future-proof for any
            // upstream lowering that wraps the candidate in an
            // explicit `Target(expr)`).
            if let DataHandle::AgentField {
                target: crate::cg::data_handle::AgentRef::Target(target_expr_id),
                ..
            } = handle
            {
                return expr_references_per_pair_candidate(*target_expr_id, ctx);
            }
            false
        }
        CgExpr::Lit(_)
        | CgExpr::Rng { .. }
        | CgExpr::AgentSelfId
        | CgExpr::ReadLocal { .. }
        | CgExpr::EventField { .. }
        | CgExpr::NamespaceField { .. } => false,
        CgExpr::Binary { lhs, rhs, .. } => {
            expr_references_per_pair_candidate(*lhs, ctx)
                || expr_references_per_pair_candidate(*rhs, ctx)
        }
        CgExpr::Unary { arg, .. } => expr_references_per_pair_candidate(*arg, ctx),
        CgExpr::Builtin { args, .. } | CgExpr::NamespaceCall { args, .. } => {
            args.iter().any(|a| expr_references_per_pair_candidate(*a, ctx))
        }
        CgExpr::Select {
            cond, then, else_, ..
        } => {
            expr_references_per_pair_candidate(*cond, ctx)
                || expr_references_per_pair_candidate(*then, ctx)
                || expr_references_per_pair_candidate(*else_, ctx)
        }
    }
}

/// Find the [`MaskId`] whose interned name matches the given
/// action's interned name, when both exist in the program's
/// interner. Returns `None` when either side is unregistered or no
/// mask shares the action's name.
///
/// The verb expander (`cg::lower::verb_expand`) creates the
/// scoring entry head and the mask head with the same synthetic
/// name (`verb_<Name>`), so the action-id ↔ mask-id bridge is the
/// shared interner name. Standard scoring rows (`Hold`,
/// `MoveToward`, …) have no matching mask and fall through to
/// `None`, leaving them ungated — same behaviour as before this
/// change.
fn mask_for_action(
    action: crate::cg::op::ActionId,
    ctx: &EmitCtx<'_>,
) -> Option<MaskId> {
    let action_name = ctx.prog.interner.get_action_name(action)?;
    ctx.prog
        .interner
        .masks
        .iter()
        .find(|(_, name)| name.as_str() == action_name)
        .map(|(id, _)| MaskId(*id))
}

/// Find the [`EventKindId`] of the verb-expander-injected
/// `ActionSelected` event in `prog`, returning `None` when no verb
/// in the source compilation declared a non-empty `emit` (so the
/// expander never injected the event kind).
///
/// Reverses the `prog.interner.event_kinds` (id → name) map by
/// scanning for the canonical name. The name is fixed by
/// [`crate::cg::lower::verb_expand::ACTION_SELECTED_EVENT_NAME`];
/// importing the constant directly keeps the two sides
/// (verb_expand injection + scoring kernel emit) coupled at compile
/// time.
fn find_action_selected_kind(
    prog: &CgProgram,
) -> Option<crate::cg::op::EventKindId> {
    use crate::cg::lower::verb_expand::ACTION_SELECTED_EVENT_NAME;
    prog.interner
        .event_kinds
        .iter()
        .find(|(_, name)| name.as_str() == ACTION_SELECTED_EVENT_NAME)
        .map(|(id, _)| crate::cg::op::EventKindId(*id))
}

// ---------------------------------------------------------------------------
// MaskPredicate body templates (Task 5.6a)
// ---------------------------------------------------------------------------

/// PerAgent dispatch — `agent_id` is bound by the
/// `thread_indexing_preamble`. Each mask op uses per-id-suffixed
/// locals (`mask_<ID>_word`, `mask_<ID>_bit`) so a Fused kernel with
/// multiple `MaskPredicate` ops doesn't redeclare `word`/`bit`.
/// PerAgent dispatch — splices a `prefix` string between the
/// `agent_id` resolution and the predicate evaluation. Used to hoist
/// `let target_expr_<N>` bindings produced by `items.<field>(<idx>)`
/// reads inside the predicate. See the MaskPredicate arm of
/// [`lower_op_body`]. The prefix is empty when the predicate has no
/// item/group field reads (no pending lets to drain), which keeps
/// the per-agent body output bit-for-bit identical to the prior
/// wrapped form.
fn mask_predicate_per_agent_body_with_prefix(
    mask: MaskId,
    prefix: &str,
    predicate_wgsl: &str,
) -> String {
    format!(
        "{prefix}\
         let mask_{0}_value: bool = {1};\n\
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
/// agent's bit. `mask_<ID>_k` reads `cfg.agent_cap` so every (actor,
/// candidate) pair gets a thread; with `mask_k = agent_cap`, the
/// dispatch caller must enqueue `agent_cap * agent_cap` threads (the
/// runtime helper currently dispatches only `agent_cap` for direct
/// PerAgent kernels — PerPair callers must size the dispatch to the
/// pair count themselves; see `tactical_squad_5v5_runtime` and
/// `mass_battle_100v100_runtime` for the expected pattern). When the
/// caller dispatches only `agent_cap` threads (as the duel_1v1_runtime
/// originally did) the kernel degenerates to `(actor=N, cand=0)` per
/// thread for `N < agent_cap`, which matches the previous `mask_k=1u`
/// behaviour.
///
/// Originally the literal `mask_k = 1u` (TODO task-5.7) — switched to
/// `cfg.agent_cap` so pair-field mask predicates that reference
/// `target.<field>` (e.g. team-membership filters via
/// `target.level != self.level`) actually visit every candidate slot
/// rather than collapsing to candidate=0 only. The previous behaviour
/// silently miscompiled any mask predicate where the slot-0 candidate
/// failed the predicate (e.g. team Red actors scanning team Red
/// agent-0 → predicate false → entire actor's mask bit never sets).
/// The `mass_battle_100v100` sim (200 agents × 200 cand × 4 verbs)
/// is the first SCALE-UP fixture exercising the full pair grid.
///
/// `agent_id` / `per_pair_candidate` are aliased to the per-mask
/// derivations so the predicate body — which refers to those names
/// directly via [`crate::cg::expr::CgExpr::AgentSelfId`] /
/// [`crate::cg::expr::CgExpr::PerPairCandidateId`] —
/// resolves cleanly.
///
/// `prefix` is spliced AFTER the `agent_id` / `per_pair_candidate`
/// aliases are bound and BEFORE the predicate evaluation. It typically
/// carries `let target_expr_<N>: u32 = <wgsl>;` lines hoisted from
/// `items.<field>(<idx>)` / `groups.<field>(<idx>)` reads inside the
/// predicate. See the MaskPredicate arm of [`lower_op_body`]. Empty
/// when the predicate has no item/group field reads — keeps the body
/// bit-for-bit identical to the prior wrapped form.
fn mask_predicate_per_pair_body_with_prefix(
    mask: MaskId,
    prefix: &str,
    predicate_wgsl: &str,
) -> String {
    format!(
        "// PerPair MaskPredicate — derive (agent, cand) from `pair`.\n\
         let mask_{0}_k = cfg.agent_cap; // pair-field predicate: visit every (actor, candidate) pair.\n\
         let mask_{0}_agent = pair / mask_{0}_k;\n\
         let mask_{0}_cand  = pair % mask_{0}_k;\n\
         if (mask_{0}_agent >= cfg.agent_cap) {{ return; }}\n\
         let agent_id = mask_{0}_agent;\n\
         let per_pair_candidate = mask_{0}_cand;\n\
         \n\
         {prefix}\
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
    let _ = ring_id;
    // Iter-2 unification: single shared event ring named `event_ring`
    // (no ring_id suffix). Body uses the structural binding name.
    "// PlumbingKind::DrainEvents — host-side drain;\n\
        // the GPU-side kernel is a no-op binding-touch. Tail reset is\n\
        // issued from the CPU via queue.write_buffer (legacy parity).\n\
        _ = event_ring[0];\n\
        _ = cfg.agent_cap;"
        .to_string()
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
    // Iter-2 unification: single shared event ring named `event_ring`
    // (no ring_id suffix). The indirect_args buffer keeps its per-ring
    // suffix because there's still one args buffer per ring (today
    // single ring, so single buffer).
    format!(
        "// PlumbingKind::SeedIndirectArgs (ring={ring_id}) — reads tail\n\
        // count from event_tail[0] (the separate atomic counter\n\
        // producers atomicAdd against; see the EventRing-Append emit\n\
        // body in `lower_emit_to_wgsl`). The earlier convention read\n\
        // event_ring[0] under a tail-at-offset-0 assumption that\n\
        // overlapped event payload words; the separate-buffer split\n\
        // resolves the conflict. Writes (wg, 1, 1) into\n\
        // indirect_args_{ring_id} so the next per-event dispatch on\n\
        // ring={ring_id} launches ceil(n/64) workgroups (CAP_WG=4096).\n\
        let n = atomicLoad(&event_tail[0]);\n\
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
            // `tick` is bound from the cfg uniform so PerAgent rules
            // with `emit <Event>` bodies can write the tick header
            // word into the event ring (every event payload starts
            // with [tag, tick]). `seed` is bound so any
            // `per_agent_u32(seed, agent_id, tick, purpose)` call
            // (lowered from `rng.*`) resolves against the cfg uniform
            // (stochastic_probe Gap #1 close, 2026-05-04). Always
            // emitted even for non-rng-touching kernels — naga drops
            // the unused let cleanly.
            "let agent_id = gid.x;\n\
             if (agent_id >= cfg.agent_cap) { return; }\n\
             let tick = cfg.tick;\n\
             let seed = cfg.seed;\n\n"
                .to_string()
        }
        DispatchShape::PerEvent { source_ring } => format!(
            // PerEvent kernels routed through the generic pipeline
            // are always classified [`KernelKind::PerEventEmit`]
            // today (see `kernel_topology_to_spec_and_body`'s
            // `is_per_event_emit` gate), which stamps the
            // `{ event_count, tick, seed, _pad0 }` cfg shape. Both
            // `cfg.event_count` and `cfg.tick` references below
            // resolve against that shape; `cfg.seed` resolves any
            // `per_agent_u32(...)` call lowered from `rng.*`
            // (stochastic_probe Gap #1 close, 2026-05-04). The
            // ViewFold path has its own preamble
            // (`build_view_fold_wgsl_body`) and never routes through
            // this branch. If a future PerEvent generic kernel
            // without an `Emit` body surfaces, the gate above must
            // extend to stamp PerEventEmit unconditionally for any
            // PerEvent dispatch.
            "let event_idx = gid.x;\n\
             if (event_idx >= cfg.event_count) {{ return; }}\n\
             let tick = cfg.tick;\n\
             let seed = cfg.seed;\n\
             // Indirect dispatch on event_ring_{}; tail count bounds gid.x.\n\n",
            source_ring.0
        ),
        DispatchShape::PerPair { source: _ } => {
            // Bind `tick` + `seed` so PerPair mask-predicate bodies that
            // reference `world.tick` (e.g. `(world.tick % cooldown == 0u)`
            // cooldown gates) lower cleanly. Mirror the PerAgent
            // / PerEvent preamble shape — pair-keyed dispatch has the
            // same access to the simulation clock.
            //
            // `cfg._pad` is repurposed as a `pair_offset` chunk base so
            // runtimes that exceed `max_compute_workgroups_per_dimension`
            // (typically 65535 → 4_194_240 pairs at workgroup_size=64)
            // can issue chunked dispatches by overwriting `_pad` between
            // calls. All current PerPair runtimes set `_pad: 0` for
            // single-shot dispatch — adding 0 preserves prior behaviour
            // bit-for-bit. megaswarm_10000 uses this to walk the 100M
            // pair grid in ~32 chunks of ~50000 workgroups each.
            "let pair = gid.x + cfg._pad0;\n\
             let tick = cfg.tick;\n\
             let seed = cfg.seed;\n\
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
        DispatchShape::PerCell => {
            // PerCell: the entry point uses `(workgroup_id,
            // local_invocation_id)` instead of `global_invocation_id`.
            // The full preamble (cooperative tile load + barrier +
            // per-lane home-cell bounds check) is composed by the
            // tiled-MoveBoid emit path, not by the generic
            // thread_indexing_preamble — that path inlines the
            // var<workgroup> tile arrays and the spatial-grid
            // index decode. We emit nothing here so the surrounding
            // body composer doesn't double-prefix.
            String::new()
        }
        DispatchShape::PerScanChunk => {
            // PerScanChunk: chunk_id is the workgroup index, lane
            // is the local thread within the workgroup. The body
            // computes its absolute cell index as
            // `chunk_id * CHUNK + lane` and bounds-checks against
            // num_cells inside the body (not here) so the scan
            // body can keep all 256 lanes participating in the
            // shared-memory Hillis-Steele reduction even when the
            // last chunk has out-of-range lanes.
            "let chunk_id = workgroup_id.x;\n\
             let lane = local_invocation_id.x;\n\n"
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
            on_event: Some(EventKindId(0)),
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
        // should appear: `(agent_hp[agent_id] < 5.0)`.
        assert!(body.contains("agent_hp[agent_id] < 5.0"), "body: {body}");
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

    /// When the program has no `ActionSelected` event kind registered
    /// (no verb cascade in source), the scoring kernel body must NOT
    /// emit any ring-append. Regression guard: a fixture without verbs
    /// stays binding-clean — no `event_ring` / `event_tail` references
    /// in the WGSL body.
    #[test]
    fn scoring_argmax_omits_action_selected_emit_without_event_kind() {
        let mut prog = CgProgram::default();
        let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(0),
            rows: vec![ScoringRowOp {
                action: ActionId(0),
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

        assert!(
            !body.contains("ActionSelected"),
            "no ActionSelected emit comment expected when event kind absent; body: {body}"
        );
        assert!(
            !body.contains("event_tail"),
            "no event_tail reference expected without ActionSelected; body: {body}"
        );
        assert!(
            !body.contains("event_ring"),
            "no event_ring reference expected without ActionSelected; body: {body}"
        );
    }

    /// When the program HAS an `ActionSelected` event kind registered
    /// (the verb expander injected it), the scoring kernel body emits
    /// the per-agent ring-append after argmax. The emit shape must
    /// mirror `wgsl_body::lower_emit_to_wgsl` (atomicAdd-tail,
    /// atomicStore-ring with bounds check) and write the right field
    /// payload words: actor=agent_id, action_id=best_action,
    /// target=best_target.
    #[test]
    fn scoring_argmax_emits_action_selected_when_event_kind_present() {
        use crate::cg::lower::verb_expand::ACTION_SELECTED_EVENT_NAME;
        use crate::cg::program::EventLayout;
        use std::collections::BTreeMap;

        let mut prog = CgProgram::default();
        let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(2.0)));

        // Register the ActionSelected event kind on the program — the
        // shape verb_expand + populate_event_kinds produce in the live
        // pipeline. Layout matches `populate_event_kinds`'s defaults
        // (stride=10, header=2) and the verb expander's payload order
        // (`actor, action_id, target` as offsets 0/1/2 of the payload).
        let action_selected_id: u32 = 7;
        prog.interner.event_kinds.insert(
            action_selected_id,
            ACTION_SELECTED_EVENT_NAME.to_string(),
        );
        let mut fields = BTreeMap::new();
        fields.insert(
            "actor".to_string(),
            crate::cg::program::FieldLayout {
                word_offset_in_payload: 0,
                word_count: 1,
                ty: CgTy::AgentId,
            },
        );
        fields.insert(
            "action_id".to_string(),
            crate::cg::program::FieldLayout {
                word_offset_in_payload: 1,
                word_count: 1,
                ty: CgTy::U32,
            },
        );
        fields.insert(
            "target".to_string(),
            crate::cg::program::FieldLayout {
                word_offset_in_payload: 2,
                word_count: 1,
                ty: CgTy::AgentId,
            },
        );
        prog.event_layouts.insert(
            action_selected_id,
            EventLayout {
                record_stride_u32: 10,
                header_word_count: 2,
                buffer_name: "event_ring".to_string(),
                fields,
            },
        );
        // The driver wires `EventRing { Append }` on every
        // ScoringArgmax op via `wire_action_selected_writes` — mirror
        // that here so the binding scanner declares event_ring +
        // event_tail bindings on the kernel.
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(0),
            rows: vec![ScoringRowOp {
                action: ActionId(0),
                utility,
                target: None,
                guard: None,
            }],
        };
        let mut probe = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerAgent,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        probe.record_write(crate::cg::data_handle::DataHandle::EventRing {
            ring: crate::cg::data_handle::EventRingId(0),
            kind: crate::cg::data_handle::EventRingAccess::Append,
        });
        // Need an event ring name interned so structural binding namer
        // produces "event_ring" (post-iter-2 unification).
        prog.interner
            .event_rings
            .insert(0, "batch_events".to_string());
        let op_id = push_op(&mut prog, probe);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // Argmax skeleton still present.
        assert!(
            body.contains("scoring_output[scoring_base + 0u] = best_action;"),
            "body: {body}"
        );

        // ActionSelected emit shape.
        assert!(
            body.contains("// emit ActionSelected"),
            "missing emit comment; body: {body}"
        );
        assert!(
            body.contains("let slot = atomicAdd(&event_tail[0], 1u);"),
            "missing tail atomicAdd; body: {body}"
        );
        // Tag write: kind id 7, header offset 0.
        assert!(
            body.contains("atomicStore(&event_ring[slot * 10u + 0u], 7u);"),
            "missing kind-tag store; body: {body}"
        );
        // Tick: header offset 1.
        assert!(
            body.contains("atomicStore(&event_ring[slot * 10u + 1u], tick);"),
            "missing tick store; body: {body}"
        );
        // actor=agent_id @ payload offset 0 → record offset 2.
        assert!(
            body.contains("atomicStore(&event_ring[slot * 10u + 2u], agent_id);"),
            "missing actor store; body: {body}"
        );
        // action_id=best_action @ payload offset 1 → record offset 3.
        assert!(
            body.contains("atomicStore(&event_ring[slot * 10u + 3u], best_action);"),
            "missing action_id store; body: {body}"
        );
        // target=best_target @ payload offset 2 → record offset 4.
        assert!(
            body.contains("atomicStore(&event_ring[slot * 10u + 4u], best_target);"),
            "missing target store; body: {body}"
        );
        // Bounds check.
        assert!(
            body.contains("if (slot < 65536u)"),
            "missing bounds check; body: {body}"
        );
    }

    /// Verb-fire probe GAP #3 close: when a row's action name matches
    /// a registered mask name (the verb expander's `verb_<Name>`
    /// convention — same name for the synthesised mask head and the
    /// scoring entry head), the row's per-agent body must be wrapped
    /// in a mask-bit gate that consults `mask_<id>_bitmap[agent_id /
    /// 32]` for the agent's bit.
    ///
    /// Without the gate, the verb's `when` predicate would have no
    /// observable effect on argmax (the mask kernel would set the bit,
    /// but the scoring kernel would walk every row unconditionally
    /// and the predicate would silently fail to filter — the gap the
    /// 2026-05-04 verb-fire-probe surfaced).
    #[test]
    fn scoring_argmax_gates_row_by_mask_bitmap_when_action_name_matches_mask() {
        let mut prog = CgProgram::default();
        let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(1.0)));
        // Pretend the verb expander has interned the action and the
        // mask under the same synthetic name `verb_Pray` (action #0,
        // mask #3 — distinct ids, shared name). The driver wires the
        // MaskBitmap read separately (`wire_scoring_mask_reads`); we
        // mirror that wiring here so the binding-name scanner emits
        // `mask_3_bitmap` against this kernel.
        let action_id = ActionId(0);
        let mask_id = MaskId(3);
        prog.interner
            .actions
            .insert(action_id.0, "verb_Pray".to_string());
        prog.interner
            .masks
            .insert(mask_id.0, "verb_Pray".to_string());
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(0),
            rows: vec![ScoringRowOp {
                action: action_id,
                utility,
                target: None,
                guard: None,
            }],
        };
        let mut probe = ComputeOp::new(
            OpId(0),
            kind,
            DispatchShape::PerAgent,
            Span::dummy(),
            &prog,
            &prog,
            &prog,
        );
        probe.record_read(crate::cg::data_handle::DataHandle::MaskBitmap { mask: mask_id });
        let op_id = push_op(&mut prog, probe);
        let topology = KernelTopology::Split {
            op: op_id,
            dispatch: DispatchShape::PerAgent,
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // The atomic load on the mask bitmap word for this agent.
        assert!(
            body.contains("let mask_3_word_for_row_0 = agent_id >> 5u;"),
            "missing mask word derivation; body: {body}"
        );
        assert!(
            body.contains("let mask_3_bit_for_row_0  = 1u << (agent_id & 31u);"),
            "missing mask bit derivation; body: {body}"
        );
        assert!(
            body.contains(
                "let mask_3_loaded_for_row_0 = atomicLoad(&mask_3_bitmap[mask_3_word_for_row_0]);"
            ),
            "missing atomicLoad on bitmap; body: {body}"
        );
        // The gate condition itself.
        assert!(
            body.contains("if ((mask_3_loaded_for_row_0 & mask_3_bit_for_row_0) != 0u)"),
            "missing mask-bit gate condition; body: {body}"
        );
        // The gate must precede the utility evaluation (so an unset
        // bit short-circuits the row entirely, never touching
        // best_utility).
        let gate_pos = body
            .find("if ((mask_3_loaded_for_row_0 & mask_3_bit_for_row_0) != 0u)")
            .unwrap();
        let utility_pos = body.find("let utility_0: f32 = 1").unwrap();
        assert!(
            gate_pos < utility_pos,
            "mask-bit gate must precede utility evaluation; body: {body}"
        );
    }

    /// Negation: a row whose action name does NOT match any registered
    /// mask (the conventional standard heads — `Hold`, `MoveToward`,
    /// etc.) must NOT be wrapped in a mask-bit gate. The change to the
    /// scoring body must remain a strict no-op for non-verb rows.
    #[test]
    fn scoring_argmax_does_not_gate_row_when_action_name_has_no_matching_mask() {
        let mut prog = CgProgram::default();
        let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.1)));
        // Action name `Hold` has no mask companion (the standard
        // scoring head).
        prog.interner.actions.insert(0, "Hold".to_string());
        // Register an unrelated mask under a different name to ensure
        // the lookup is name-based, not id-based.
        prog.interner.masks.insert(0, "low_hp".to_string());
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(0),
            rows: vec![ScoringRowOp {
                action: ActionId(0),
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
        let (_spec, body) = kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // No mask gate of any kind on row 0.
        assert!(
            !body.contains("mask_0_bitmap"),
            "unexpected mask binding reference for ungated row; body: {body}"
        );
        assert!(
            !body.contains("mask_0_word_for_row_0"),
            "unexpected mask gate scaffolding for ungated row; body: {body}"
        );
        // Utility evaluation still lands at the top of the row block.
        assert!(
            body.contains("let utility_0: f32 = "),
            "utility row should still emit; body: {body}"
        );
    }

    /// Pair-field row (Gap #4 close) — when the row's utility expression
    /// reads `Read(AgentField { target: AgentRef::PerPairCandidate, .. })`
    /// the scoring kernel wraps the row body in an inner candidate loop
    /// (`for (var per_pair_candidate ...; per_pair_candidate <
    /// cfg.agent_cap; ...)`) and defaults the winning target to
    /// `per_pair_candidate` when the row's explicit `target` is `None`
    /// (the verb-synth shape — `lower_standard_row` always produces
    /// `target: None`).
    ///
    /// This is the emit-side companion of the gap chain pinned by
    /// `docs/superpowers/notes/2026-05-04-pair_scoring_probe.md` —
    /// per-(actor, candidate) scoring requires the inner loop to
    /// argmax across candidates per actor.
    #[test]
    fn scoring_argmax_wraps_pair_field_row_in_per_pair_candidate_loop() {
        use crate::cg::data_handle::{AgentFieldId, AgentRef, DataHandle};
        let mut prog = CgProgram::default();
        // Row utility: `agent_alive[per_pair_candidate]` — the canonical
        // pair-field shape. Lowered as a `Read` of the `Alive`
        // field with `target: PerPairCandidate`.
        let pair_alive_read = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::PerPairCandidate,
            }),
        );
        let kind = ComputeOpKind::ScoringArgmax {
            scoring: ScoringId(0),
            rows: vec![ScoringRowOp {
                action: ActionId(0),
                utility: pair_alive_read,
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
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // The inner candidate loop wraps the row body. Iterating in
        // slot order (P11 — Reduction Determinism).
        assert!(
            body.contains(
                "for (var per_pair_candidate: u32 = 0u; \
                 per_pair_candidate < cfg.agent_cap; \
                 per_pair_candidate = per_pair_candidate + 1u)"
            ),
            "missing inner candidate loop; body: {body}"
        );
        // Pair-field rows with no explicit row.target default the
        // winning target to per_pair_candidate (the slot the inner
        // loop is currently visiting).
        assert!(
            body.contains("best_target = per_pair_candidate;"),
            "pair-field row must default best_target to per_pair_candidate; body: {body}"
        );
        // The utility expression itself reads the candidate-side SoA
        // (alive is a bool-stored-as-u32, hence the `(... != 0u)`
        // coercion in `agent_field_access`).
        assert!(
            body.contains("agent_alive[per_pair_candidate]"),
            "utility expression should read candidate-side SoA; body: {body}"
        );
        // The inner loop must sit AFTER the row block opens (so the
        // running `best_*` updates compose correctly across siblings —
        // the row block opens a fresh `{ ... }` local scope but reads +
        // writes the function-scope `best_*` vars).
        let block_open = body.find("// row 0: action=#0").unwrap();
        let loop_open = body.find("for (var per_pair_candidate").unwrap();
        assert!(
            loop_open > block_open,
            "inner loop must sit after the row block opens; body: {body}"
        );
    }

    /// Negation of [`scoring_argmax_wraps_pair_field_row_in_per_pair_candidate_loop`]
    /// — a row whose utility does NOT reference the per-pair candidate
    /// side (e.g. a `self.<field>` read or a bare literal) MUST NOT be
    /// wrapped in the inner candidate loop. The emit must remain a
    /// strict no-op for non-pair rows.
    #[test]
    fn scoring_argmax_does_not_wrap_self_only_row_in_per_pair_candidate_loop() {
        let mut prog = CgProgram::default();
        let utility = push_expr(&mut prog, CgExpr::Lit(LitValue::F32(0.5)));
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
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        assert!(
            !body.contains("per_pair_candidate"),
            "self-only row must NOT reference per_pair_candidate; body: {body}"
        );
        assert!(
            !body.contains("for (var per_pair_candidate"),
            "self-only row must NOT emit an inner candidate loop; body: {body}"
        );
        // Standard sentinel target preserved.
        assert!(
            body.contains("best_target = 0xFFFFFFFFu;"),
            "self-only row must keep the sentinel target default; body: {body}"
        );
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
        // Real counting-sort populate — calls pos_to_cell + atomicAdd
        // on per-cell offsets, then writes the agent_id into the
        // computed slot. The pre-Task-5.6c stub-comment is gone.
        assert!(body.contains("pos_to_cell"), "body: {body}");
        assert!(body.contains("atomicAdd"), "body: {body}");
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




    /// Snapshot: pin the EngagementQuery body output through
    /// `kernel_topology_to_spec_and_body`. Any drift in the per-kind
    /// body const, the preamble shape, or the op-comment header surfaces
    /// here — the assertions below describe the exact expected form so
    /// the snapshot survives a body-content tweak with a one-site update.

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
        // ViewFold cfg fields: event_count, tick, second_key_pop, _pad.
        // `second_key_pop` joined the layout with the pair_map storage
        // emit gap fix (2026-05-03) — drives the 2-D `(k1 *
        // second_key_pop + k2)` index. Single-key views set the field
        // to 1 at runtime so the index reduces to `k_last`.
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
            spec.cfg_struct_decl.contains("second_key_pop: u32"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        assert!(
            spec.cfg_struct_decl.contains("_pad: u32"),
            "decl: {}",
            spec.cfg_struct_decl
        );
        assert!(
            spec.cfg_build_expr
                .contains("event_count: 0, tick: state.tick"),
            "build_expr: {}",
            spec.cfg_build_expr
        );
        assert!(
            spec.cfg_build_expr.contains("second_key_pop: 1"),
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

    /// Task 1.5 — per-op brace scoping. When two view_fold handlers fuse
    /// into one kernel, each handler's pattern-binding `Let` synthesis
    /// emits its own `let local_<N>: <ty> = ...;` at the head of its
    /// body. Without per-op brace wrapping, both bodies declare
    /// `let local_0` at the function top level — naga rejects this as a
    /// redefinition. Wrapping each per-op body in `{ ... }` puts each
    /// op's locals in its own WGSL scope, eliminating the collision.
    ///
    /// This test pins the brace structure: the rendered body for a
    /// fused two-op view_fold kernel must contain two `{ ... let local_0`
    /// occurrences (one per op), proving the scoping isolates them.
    #[test]
    fn fused_kernel_body_wraps_each_op_in_braces() {
        use crate::cg::data_handle::ViewId;
        use crate::cg::stmt::LocalId;

        // Helper — build a view_fold op whose body starts with a `Let`
        // binding `let local_0: u32 = 1u;` followed by the canonical
        // `_ = (1.0)` ViewStorage assign. Both ops in the topology bind
        // `LocalId(0)`, mirroring the way Task 1's pattern-binding
        // lowering synthesises one Let per binder per handler — fused
        // handlers thus collide on the same local id.
        fn view_fold_op_with_let(
            prog: &mut CgProgram,
            view: ViewId,
            event_kind: EventKindId,
            ring: EventRingId,
        ) -> OpId {
            let one_u = push_expr(prog, CgExpr::Lit(LitValue::U32(1)));
            let let_stmt = push_stmt(
                prog,
                CgStmt::Let {
                    local: LocalId(0),
                    value: one_u,
                    ty: CgTy::U32,
                },
            );
            let one_f = push_expr(prog, CgExpr::Lit(LitValue::F32(1.0)));
            let assign = push_stmt(
                prog,
                CgStmt::Assign {
                    target: DataHandle::ViewStorage {
                        view,
                        slot: crate::cg::data_handle::ViewStorageSlot::Primary,
                    },
                    value: one_f,
                },
            );
            let body = push_list(
                prog,
                CgStmtList {
                    stmts: vec![let_stmt, assign],
                },
            );
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

        let mut prog = CgProgram::default();
        let view = ViewId(3);
        prog.interner.views.insert(3, "threat_level".to_string());
        prog.interner.event_kinds.insert(0, "AgentAttacked".into());
        prog.interner
            .event_kinds
            .insert(1, "EffectDamageApplied".into());
        let op_a = view_fold_op_with_let(&mut prog, view, EventKindId(0), EventRingId(0));
        let op_b = view_fold_op_with_let(&mut prog, view, EventKindId(1), EventRingId(0));
        let topology = KernelTopology::Fused {
            ops: vec![op_a, op_b],
            dispatch: DispatchShape::PerEvent {
                source_ring: EventRingId(0),
            },
        };
        let ctx = EmitCtx::structural(&prog);
        let (_spec, body) =
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).unwrap();

        // Both ops emit `let local_0: u32 = 1u;`. Without brace
        // wrapping these collide. Sanity: the rendered body contains
        // exactly two `let local_0` occurrences (one per op).
        assert_eq!(
            body.matches("let local_0").count(),
            2,
            "body: {body}"
        );
        // Each occurrence must be wrapped in its own `{ ... }` block —
        // i.e. an open brace appears between the op-comment header and
        // the `let local_0` line. With the per-handler tag-check guard
        // in place (event-kind filtering for multi-kind rings), the
        // body shape is `{\n        if (...) {\n            let
        // local_0 ...`. We pin two such nested-`let local_0` lines —
        // one per op — at the 12-space indent the inner-guard scope
        // produces.
        let scoped_lets = body.matches("            let local_0").count();
        assert_eq!(
            scoped_lets, 2,
            "expected two scoped `let local_0` blocks; body:\n{body}"
        );
        // Both per-op blocks have a tag-check `if` line. The exact
        // event-kind tags differ (op_a uses kind 0, op_b uses kind 1)
        // — count the structural prefix for each.
        assert!(
            body.contains("if (event_ring[event_idx * 10u + 0u] == 0u)"),
            "expected tag-check guard for event kind 0; body:\n{body}"
        );
        assert!(
            body.contains("if (event_ring[event_idx * 10u + 0u] == 1u)"),
            "expected tag-check guard for event kind 1; body:\n{body}"
        );
        // And every per-op block closes — the body has at least two
        // `\n    }` closers (one per op).
        assert!(
            body.matches("\n    }").count() >= 2,
            "expected at least two `}}` closers; body:\n{body}"
        );
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
                // B1 fix: primary slot is AtomicStorage so the
                // CAS-loop write-back type-checks. wgsl_ty="u32"
                // is wrapped to `array<atomic<u32>>` by
                // `lower_wgsl_bindings`.
                2,
                "view_storage_primary".to_string(),
                "AtomicStorage".to_string(),
                "u32".to_string(),
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
        // `second_key_pop` joined the layout with the pair_map storage
        // emit gap fix (2026-05-03) — fold body composes
        // `[k1 * cfg.second_key_pop + k2]` for 2-D pair views.
        assert!(spec.cfg_struct_decl.contains(
            "event_count: u32, pub tick: u32, pub second_key_pop: u32, pub _pad: u32"
        ));
        assert_eq!(
            spec.cfg_build_expr,
            "FoldThreatLevelCfg { event_count: 0, tick: state.tick, second_key_pop: 1, _pad: 0 }"
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

        // Indirect (physics consumer) → PerEventEmit. Any PerEvent-
        // dispatched kernel needs the `{ event_count, tick, seed,
        // _pad0 }` cfg shape so the preamble's `event_idx >=
        // cfg.event_count` early-return resolves; the classifier
        // stamps PerEventEmit unconditionally for PerEvent dispatch.
        // trade_market_real (ApplyTrade), quest_arc_real
        // (ApplyStageAdvance), and village_day_cycle (ApplyHarvest /
        // ApplyEat / ApplyEnergyDecay) all surfaced the prior Generic-
        // with-agent_cap cfg WGSL parse failure for Apply-only PerEvent.
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
        assert_eq!(spec4.kind, KernelKind::PerEventEmit);
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
        // Iter-2 unification: body uses canonical `event_ring` binding
        // name (no ring suffix). The header comment also drops the
        // `(ring=N)` since the ring is implicit.
        assert!(
            body.contains("PlumbingKind::DrainEvents"),
            "DrainEvents body: {body}"
        );
        assert!(body.contains("event_ring["), "body: {body}");
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
        // Iter-2 unification: ring binding is `event_ring` (no suffix);
        // indirect_args buffer keeps its per-ring suffix because there's
        // still one args buffer per ring in the runtime.
        assert!(
            body.contains("PlumbingKind::SeedIndirectArgs (ring=2)"),
            "SeedIndirectArgs body: {body}"
        );
        assert!(body.contains("event_ring[0]"), "body: {body}");
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
        // bodies whose `indirect_args` references differ. The shared
        // `event_ring` (post-iter-2 unification) reads identically for
        // both — the distinction lives in the per-ring args buffer.
        let (_a, body_a) =
            plumbing_spec_and_body(PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(0),
            });
        let (_b, body_b) =
            plumbing_spec_and_body(PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(3),
            });
        // Both bodies read from the unified `event_ring`.
        assert!(body_a.contains("event_ring[0]"), "body_a: {body_a}");
        assert!(body_b.contains("event_ring[0]"), "body_b: {body_b}");
        // Per-ring args buffer distinguishes them.
        assert!(body_a.contains("indirect_args_0["), "body_a: {body_a}");
        assert!(!body_a.contains("indirect_args_3"), "body_a leaked ring=3: {body_a}");
        assert!(body_b.contains("indirect_args_3["), "body_b: {body_b}");
        assert!(!body_b.contains("indirect_args_0"), "body_b leaked ring=0: {body_b}");
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

    // ---- 13. FilteredWalk emit (Phase 7 Task 2) ----

    #[test]
    fn filtered_walk_emit_threads_filter_wgsl_into_body() {
        use crate::cg::op::SpatialQueryKind;

        let mut prog = CgProgram::default();
        // Filter: PerPairCandidate.alive — reads agent_alive[per_pair_candidate].
        let filter_id = push_expr(
            &mut prog,
            CgExpr::Read(DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::PerPairCandidate,
            }),
        );
        let op = ComputeOp::new(
            OpId(0),
            ComputeOpKind::SpatialQuery {
                kind: SpatialQueryKind::FilteredWalk { filter: filter_id },
            },
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
            kernel_topology_to_spec_and_body(&topology, &prog, &ctx).expect("emit body");

        assert!(
            body.contains("for (var cell"),
            "filtered-walk body must include per-cell walk loop, got: {body}"
        );
        assert!(
            body.contains("agent_alive[per_pair_candidate]"),
            "filter (alive read) must lower into the body, got: {body}"
        );
        assert!(
            body.contains("spatial_query_results["),
            "body must write into spatial_query_results, got: {body}"
        );
    }

    #[test]
    fn filtered_walk_snake_name_includes_filter_id() {
        use crate::cg::op::SpatialQueryKind;

        // The filter CgExprId is 0 (first expr pushed).
        let filter_id = CgExprId(0);
        let kind = SpatialQueryKind::FilteredWalk { filter: filter_id };
        assert_eq!(spatial_kind_name(kind), "filtered_walk_0");
    }

    /// Gap #2 / #3 from `2026-05-04-trade_market_probe.md` — the
    /// per-Item / per-Group external binding name must disambiguate
    /// when multiple Item / Group entities declare overlapping field
    /// names. The naming rule is `<entity_snake>_<field_snake>`, so
    /// `Wood.weight` → `wood_weight` and `Iron.weight` → `iron_weight`,
    /// not a collision on a shared `weight` binding.
    ///
    /// Pre-this-test: the existing bartering fixture exercises ONE
    /// Item field-read and ONE Group field-read, but no test pinned
    /// the multi-entity disambiguation shape. This test pins it so a
    /// future refactor that drops the entity-name half (e.g.
    /// `format!("{field_name}")`) fires immediately.
    #[test]
    fn item_field_external_name_disambiguates_overlapping_field_names() {
        // Same field name on 4 distinct Item entities — the trade-market
        // probe shape (Wood / Iron / Grain / Cloth, all with
        // `base_price: f32`).
        assert_eq!(
            item_field_external_name("Wood", "base_price"),
            "wood_base_price"
        );
        assert_eq!(
            item_field_external_name("Iron", "base_price"),
            "iron_base_price"
        );
        assert_eq!(
            item_field_external_name("Grain", "base_price"),
            "grain_base_price"
        );
        assert_eq!(
            item_field_external_name("Cloth", "base_price"),
            "cloth_base_price"
        );

        // PascalCase entity names → snake_case (the to_snake_case
        // helper). `MultiWord` becomes `multi_word`.
        assert_eq!(
            item_field_external_name("LegendaryArtifact", "weight"),
            "legendary_artifact_weight"
        );

        // Same shape works for Group entities (the function is shared
        // between item and group binding emission — the gap-#3 surface).
        assert_eq!(
            item_field_external_name("Guild", "size"),
            "guild_size"
        );
        assert_eq!(
            item_field_external_name("Faction", "size"),
            "faction_size"
        );
    }
}
