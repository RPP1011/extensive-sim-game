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
//! - **Kernel naming is structural.** `cg_<topology>_<first_op>_<count>`
//!   today (e.g. `cg_fused_mask_predicate_2`) — Task 5.1 may re-align
//!   with the legacy emitters' semantic names (`fused_mask`, `scoring`,
//!   `fold_<view>`) once the xtask wires this output and its actual
//!   shape can be diffed against legacy emit.
//! - **Op-body lowering is partial.** Only [`ComputeOpKind::PhysicsRule`]
//!   and [`ComputeOpKind::ViewFold`] (which carry real
//!   [`crate::cg::stmt::CgStmtList`] bodies) lower through Task 4.1's
//!   walks. [`ComputeOpKind::MaskPredicate`] lowers its predicate
//!   expression and emits an `atomicOr` placeholder for the bitmap
//!   write. [`ComputeOpKind::ScoringArgmax`],
//!   [`ComputeOpKind::SpatialQuery`], and
//!   [`ComputeOpKind::Plumbing`] emit a documented `// TODO(task-4.x)`
//!   placeholder line in the WGSL output (never a Rust panic) — Task
//!   5.1 will surface what's missing through the legacy-vs-new diff.
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
    AgentFieldTy, AgentScratchKind, CycleEdgeKey, DataHandle, EventRingAccess, SpatialStorageKind,
    ViewStorageSlot,
};
use crate::cg::dispatch::DispatchShape;
use crate::cg::op::{ComputeOp, ComputeOpKind, OpId};
use crate::cg::program::CgProgram;
use crate::cg::schedule::synthesis::KernelTopology;
use crate::kernel_binding_ir::{
    snake_to_pascal, AccessMode, BgSource, KernelBinding, KernelSpec,
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
/// surfaced as [`EmitError::InvalidKernelSpec`].
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

    // 2. Collect every (handle, was_written) pair across all body ops.
    //
    //    `was_written` is true if any op writes the handle (drives the
    //    AccessMode upgrade in step 3). The handle is keyed by its full
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

    // 3. For each unique handle, derive its `BindingMetadata` (if the
    //    handle is binding-relevant — Rng / ConfigConst are not). Skip
    //    handles that have no binding metadata.
    let mut typed_bindings: Vec<TypedBinding> = Vec::new();
    for (key, agg) in &handle_set {
        let canonical = canonical_handle(key, &agg.first_seen);
        let Some(meta) = handle_to_binding_metadata(&canonical) else {
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

    // 4. Sort bindings by their cycle-edge key for determinism.
    //    `BTreeMap` already iterates sorted, but `typed_bindings` is
    //    materialised post-filter so re-sort defensively.
    typed_bindings.sort_by(|a, b| a.sort_key.cmp(&b.sort_key));

    // 5. Pick a structural kernel name from the topology shape +
    //    leading op kind. See # Limitations: this is a placeholder.
    let first_op = resolve_op(prog, body_ops[0])?;
    let name = structural_kernel_name(kind_label, &first_op.kind, body_ops.len());
    let pascal = snake_to_pascal(&name);
    let entry_point = format!("cs_{name}");
    let cfg_struct = format!("{pascal}Cfg");

    // 6. Assign slots — data bindings 0..N, cfg uniform at slot N.
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

    // 7. Build the cfg struct decl + cfg-construction expression. For
    //    Task 4.2 these are the minimal valid forms; the kernel's
    //    actual cfg shape is a Task 5.1 alignment concern.
    let cfg_struct_decl = build_cfg_struct_decl(&cfg_struct);
    let cfg_build_expr = build_cfg_build_expr(&cfg_struct);

    // 8. Compose the WGSL body — one fragment per op, joined with
    //    blank lines. We invoke the body builder here for two reasons:
    //    (a) inner-walk arena failures surface as typed errors before
    //    spec construction completes, so a malformed program never
    //    yields a spec the body lowering can't render; (b) the body
    //    itself is not stored on `KernelSpec` (it lives in the WGSL
    //    emitter that consumes the spec) — Task 4.3 will redo body
    //    composition at that layer. The
    //    [`kernel_topology_to_spec_and_body`] helper exposes the body
    //    string for tests + Task 4.3 callers.
    let _wgsl_body = build_wgsl_body(&body_ops, &dispatch, prog, ctx)?;

    let spec = KernelSpec {
        name,
        pascal,
        entry_point,
        cfg_struct,
        cfg_build_expr,
        cfg_struct_decl,
        bindings,
    };

    spec.validate()
        .map_err(|reason| KernelEmitError::InvalidKernelSpec { reason })?;
    Ok(spec)
}

/// Lower a [`KernelTopology`] to its [`KernelSpec`] AND return the
/// composed WGSL body string alongside. Useful for tests + Task 4.3
/// callers that consume both. `KernelSpec` itself does not carry the
/// body string today; see `# Limitations` on
/// [`kernel_topology_to_spec`].
pub fn kernel_topology_to_spec_and_body(
    topology: &KernelTopology,
    prog: &CgProgram,
    ctx: &EmitCtx<'_>,
) -> Result<(KernelSpec, String), KernelEmitError> {
    let spec = kernel_topology_to_spec(topology, prog, ctx)?;
    let body_ops = topology_body_ops(topology, prog)?;
    let dispatch = topology_dispatch(topology, prog)?;
    let body = build_wgsl_body(&body_ops, &dispatch, prog, ctx)?;
    Ok((spec, body))
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
fn handle_to_binding_metadata(h: &DataHandle) -> Option<BindingMetadata> {
    match h {
        DataHandle::AgentField { field, target: _ } => Some(BindingMetadata {
            bg_source: BgSource::External("agents".into()),
            base_access: AccessMode::ReadStorage,
            wgsl_ty: agent_field_wgsl_ty(field.ty()),
        }),
        DataHandle::ViewStorage { view, slot } => Some(BindingMetadata {
            bg_source: BgSource::Resident(format!("view_{}_{}", view.0, view_slot_field(*slot))),
            base_access: AccessMode::ReadStorage,
            wgsl_ty: "array<u32>".into(),
        }),
        DataHandle::EventRing { ring, kind } => {
            let base_access = match kind {
                EventRingAccess::Read => AccessMode::ReadStorage,
                EventRingAccess::Append => AccessMode::AtomicStorage,
                EventRingAccess::Drain => AccessMode::ReadWriteStorage,
            };
            let wgsl_ty = match kind {
                EventRingAccess::Append => "u32".into(),
                _ => "array<u32>".into(),
            };
            Some(BindingMetadata {
                bg_source: BgSource::Transient(format!("event_ring_{}", ring.0)),
                base_access,
                wgsl_ty,
            })
        }
        DataHandle::ConfigConst { .. } => None,
        DataHandle::MaskBitmap { mask } => Some(BindingMetadata {
            bg_source: BgSource::Transient(format!("mask_{}_bitmap", mask.0)),
            base_access: AccessMode::AtomicStorage,
            wgsl_ty: "u32".into(),
        }),
        DataHandle::ScoringOutput => Some(BindingMetadata {
            bg_source: BgSource::Resident("scoring_table".into()),
            base_access: AccessMode::ReadWriteStorage,
            wgsl_ty: "array<u32>".into(),
        }),
        DataHandle::SpatialStorage { kind } => {
            let (field, base_access) = match kind {
                SpatialStorageKind::GridCells => {
                    ("grid_cells".to_string(), AccessMode::ReadStorage)
                }
                SpatialStorageKind::GridOffsets => {
                    ("grid_offsets".to_string(), AccessMode::ReadStorage)
                }
                SpatialStorageKind::QueryResults => {
                    ("query_results".to_string(), AccessMode::ReadWriteStorage)
                }
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
        DataHandle::IndirectArgs { ring } => Some(BindingMetadata {
            bg_source: BgSource::Transient(format!("indirect_args_{}", ring.0)),
            base_access: AccessMode::ReadWriteStorage,
            wgsl_ty: "array<u32>".into(),
        }),
        DataHandle::AgentScratch { kind } => {
            let suffix = match kind {
                AgentScratchKind::Packed => "packed",
            };
            Some(BindingMetadata {
                bg_source: BgSource::Transient(format!("agent_scratch_{suffix}")),
                base_access: AccessMode::ReadWriteStorage,
                wgsl_ty: "array<u32>".into(),
            })
        }
        DataHandle::SimCfgBuffer => Some(BindingMetadata {
            bg_source: BgSource::External("sim_cfg".into()),
            base_access: AccessMode::Uniform,
            wgsl_ty: "SimCfg".into(),
        }),
        DataHandle::SnapshotKick => Some(BindingMetadata {
            bg_source: BgSource::Transient("snapshot_kick".into()),
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

/// Structural kernel name. See `# Limitations` — Task 5.1 may align
/// with the legacy semantic names.
fn structural_kernel_name(kind_label: &str, first_op_kind: &ComputeOpKind, op_count: usize) -> String {
    let op_short = compute_op_kind_short(first_op_kind);
    format!("cg_{kind_label}_{op_short}_{op_count}")
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
// Cfg struct + build expr
// ---------------------------------------------------------------------------

/// Build a minimal cfg struct decl. The legacy emitters embed
/// per-kernel cfg fields (e.g. `agent_cap`, `num_mask_words`); Task 4.2
/// produces a single-field placeholder so the spec validates and the
/// downstream lowerings have something to consume. Task 5.1 will
/// refine.
fn build_cfg_struct_decl(cfg_struct: &str) -> String {
    format!(
        "#[repr(C)]\n\
         #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]\n\
         pub struct {cfg_struct} {{ pub agent_cap: u32, pub _pad: [u32; 3] }}"
    )
}

fn build_cfg_build_expr(cfg_struct: &str) -> String {
    format!("{cfg_struct} {{ agent_cap: state.agent_cap(), _pad: [0; 3] }}")
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
        let fragment = lower_op_body(op, ctx)?;
        // Comment header per op for traceability.
        writeln!(out, "// op#{} ({})", op.id.0, compute_op_kind_short(&op.kind))
            .expect("write to String never fails");
        out.push_str(&fragment);
    }

    Ok(out)
}

/// Per-op body lowering. Returns the WGSL fragment for the op without
/// surrounding kernel boilerplate.
fn lower_op_body(op: &ComputeOp, ctx: &EmitCtx<'_>) -> Result<String, KernelEmitError> {
    match &op.kind {
        ComputeOpKind::MaskPredicate { mask, predicate } => {
            let predicate_wgsl = lower_cg_expr_to_wgsl(*predicate, ctx)?;
            // Placeholder atomic-or write — the legacy mask emitter
            // packs 32 agents per word and atomicOrs the bit. Task 5.1
            // aligns the exact form.
            Ok(format!(
                "let mask_{0}_value: bool = {1};\n\
                 if (mask_{0}_value) {{\n\
                 \x20   let word = agent_id >> 5u;\n\
                 \x20   let bit = 1u << (agent_id & 31u);\n\
                 \x20   atomicOr(&mask_{0}_bitmap[word], bit);\n\
                 }}",
                mask.0, predicate_wgsl
            ))
        }
        ComputeOpKind::PhysicsRule { body, .. } => {
            lower_cg_stmt_list_to_wgsl(*body, ctx).map_err(KernelEmitError::from)
        }
        ComputeOpKind::ViewFold { body, .. } => {
            lower_cg_stmt_list_to_wgsl(*body, ctx).map_err(KernelEmitError::from)
        }
        ComputeOpKind::ScoringArgmax { scoring, rows } => Ok(format!(
            "// TODO(task-4.x): scoring_argmax kernel body — \
             scoring_id={0}, {1} rows.\n\
             // The legacy emitter (emit_scoring_wgsl.rs) computes per-row \
             utility, runs argmax, and writes (action, target, score) into scoring_output.",
            scoring.0,
            rows.len()
        )),
        ComputeOpKind::SpatialQuery { kind } => Ok(format!(
            "// TODO(task-4.x): spatial_query body for kind={kind}. \
             Legacy emitter: emit_spatial_kernel.rs."
        )),
        ComputeOpKind::Plumbing { kind } => Ok(format!(
            "// TODO(task-4.x): plumbing body for kind={}. \
             Legacy emitters: emit_alive_pack_wgsl, emit_seed_indirect_wgsl, etc.",
            kind.label()
        )),
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

fn topology_body_ops(
    topology: &KernelTopology,
    prog: &CgProgram,
) -> Result<Vec<OpId>, KernelEmitError> {
    match topology {
        KernelTopology::Fused { ops, .. } => Ok(ops.clone()),
        KernelTopology::Split { op, .. } => Ok(vec![*op]),
        KernelTopology::Indirect { consumers, .. } => {
            // Validate every consumer resolves before returning.
            for op_id in consumers {
                resolve_op(prog, *op_id)?;
            }
            if consumers.is_empty() {
                Err(KernelEmitError::EmptyKernelTopology)
            } else {
                Ok(consumers.clone())
            }
        }
    }
}

fn topology_dispatch(
    topology: &KernelTopology,
    prog: &CgProgram,
) -> Result<DispatchShape, KernelEmitError> {
    match topology {
        KernelTopology::Fused { dispatch, .. } => Ok(*dispatch),
        KernelTopology::Split { dispatch, .. } => Ok(*dispatch),
        KernelTopology::Indirect { consumers, .. } => {
            let first = *consumers
                .first()
                .ok_or(KernelEmitError::EmptyKernelTopology)?;
            let op = resolve_op(prog, first)?;
            Ok(op.shape)
        }
    }
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
        // ComputeOp::new takes the program by reference but we need to
        // push afterwards — clone the op and push.
        let cloned = ComputeOp {
            id: OpId(0),
            kind: op.kind.clone(),
            reads: op.reads.clone(),
            writes: op.writes.clone(),
            shape: op.shape,
            span: op.span,
        };
        push_op(prog, cloned)
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
        let cloned = ComputeOp {
            id: OpId(0),
            kind: op.kind.clone(),
            reads: op.reads.clone(),
            writes: op.writes.clone(),
            shape: op.shape,
            span: op.span,
        };
        push_op(prog, cloned)
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
        let cloned = ComputeOp {
            id: OpId(0),
            kind: op.kind.clone(),
            reads: op.reads.clone(),
            writes: op.writes.clone(),
            shape: op.shape,
            span: op.span,
        };
        push_op(prog, cloned)
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
    fn indirect_topology_emits_consumer_kernel_with_indirect_args_binding() {
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

    // ---- 9. Scoring placeholder body ----

    #[test]
    fn scoring_op_emits_todo_placeholder_not_panic() {
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
        assert!(body.contains("TODO(task-4.x): scoring_argmax"), "body: {body}");
    }
}
