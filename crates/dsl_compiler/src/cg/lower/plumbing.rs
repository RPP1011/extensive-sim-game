//! Plumbing lowering — synthesize the runtime "between-kernel"
//! [`PlumbingKind`] ops that the schedule needs but the DSL surface
//! does not name.
//!
//! Phase 2, Task 2.7 of the Compute-Graph IR pipeline (see
//! `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`). Unlike
//! the mask / view / physics / scoring / spatial passes, this pass does
//! NOT consume an AST IR sub-tree: plumbing ops have no DSL surface.
//! Instead the pass takes a list of [`PlumbingKind`] values (typically
//! produced by [`synthesize_plumbing_ops`] from the user-facing ops
//! already in the program) and pushes one
//! [`crate::cg::op::ComputeOp`] per entry onto the builder.
//!
//! # Two entry points
//!
//! - [`synthesize_plumbing_ops`] — walks a program's existing user-
//!   facing ops and produces the deduplicated list of plumbing kinds
//!   the schedule needs. Always-on kinds (`UploadSimCfg`,
//!   `PackAgents`, `UnpackAgents`, `KickSnapshot`) are emitted
//!   unconditionally; conditional kinds (`AliveBitmap`,
//!   `SeedIndirectArgs`, `DrainEvents`) fire only when the program
//!   surface contains a triggering op.
//! - [`lower_plumbing`] — the parallel of
//!   [`super::spatial::lower_spatial_queries`]: pushes one op per
//!   input entry, in input order, with each op's
//!   [`crate::cg::dispatch::DispatchShape`] supplied by
//!   [`PlumbingKind::dispatch_shape`].
//!
//! The driver (Task 2.8) is expected to call them in sequence: build
//! the user ops first, then `synthesize_plumbing_ops(prog)` once on
//! the in-flight program, then `lower_plumbing` to push the resulting
//! kinds. Document this contract on the driver when it lands.
//!
//! # Synthesizer semantics
//!
//! Always-on:
//!
//! - `UploadSimCfg` — sim_cfg is uploaded every tick regardless of
//!   what the DSL produces. The buffer is the host-side handle through
//!   which `sim_cfg.tick`, `sim_cfg.world_seed`, etc. reach the GPU.
//! - `PackAgents` — every tick packs the per-agent SoA into the GPU's
//!   linear scratch.
//! - `UnpackAgents` — symmetric inverse; runs after the GPU dispatch
//!   completes.
//! - `KickSnapshot` — unconditional end-of-tick snapshot dump.
//!
//! Conditional:
//!
//! - `AliveBitmap` — emitted iff any user-facing op writes
//!   [`crate::cg::data_handle::AgentFieldId::Alive`] (the bitmap is
//!   only stale when the alive field changes).
//! - `SeedIndirectArgs { ring }` — one entry per distinct
//!   [`crate::cg::data_handle::EventRingId`] referenced by a
//!   [`crate::cg::dispatch::DispatchShape::PerEvent { source_ring }`]
//!   user op (every per-event dispatch needs its indirect-args buffer
//!   seeded from the ring's tail count).
//! - `DrainEvents { ring }` — one entry per distinct ring referenced
//!   by any read with [`crate::cg::data_handle::EventRingAccess::Drain`].
//!   Today the user surface produces no `Drain` reads directly; the
//!   variant is wired for the apply-path migration that introduces
//!   drain semantics into the IR.
//!
//! # Synthesizer ordering
//!
//! The output order is: `UploadSimCfg`, `PackAgents`, conditionally
//! `AliveBitmap`, then conditional `SeedIndirectArgs` (sorted by ring
//! id), then conditional `DrainEvents` (sorted by ring id), then
//! `UnpackAgents`, `KickSnapshot`. The order is the natural
//! end-to-end shape: configure → pack → derive bitmaps + indirect
//! args → drain → unpack → snapshot. Schedule synthesis (Phase 3)
//! refines the order using the read/write graph; this pass's
//! contract is "produce the inventory in a deterministic order so
//! snapshots are reproducible."
//!
//! # Spans
//!
//! Plumbing ops have no AST source. Each op is constructed with
//! [`Span::dummy`]; future work could thread a span from the
//! triggering user op for diagnostic provenance, but the current
//! emitters don't consume span info.
//!
//! # Typed-error surface
//!
//! All defects surface as variants on the unified
//! [`super::error::LoweringError`]. The pass touches only the shared
//! [`LoweringError::BuilderRejected`] variant — plumbing ops carry
//! no expr / list ids, so [`crate::cg::program::CgProgramBuilder::validate_op_kind_refs`]'s
//! `Plumbing { .. }` arm is `Ok(())` unconditionally, and the pass
//! cannot fail under normal operation.
//!
//! # Limitations
//!
//! - **No span threading.** Every op uses [`Span::dummy`]. The
//!   triggering op's span would be a more useful provenance, but
//!   plumbing is multi-source by construction (an `AliveBitmap`
//!   refresh isn't owned by a single user op). Schedule synthesis
//!   (Phase 3) is the natural place to thread these.
//! - **Ring id sets are sourced from `op.shape` and `op.reads`
//!   only.** A plumbing kind that needs to react to a write
//!   (e.g., a hypothetical `MirrorEventRing` that fires on every
//!   `Append`) would need a separate scan. Today the conditional
//!   plumbing kinds all key off shape (`PerEvent`) or read mode
//!   (`Drain`) so the surface is sufficient.
//! - **Phase 1 amendments.** Task 2.7 introduces new
//!   [`crate::cg::data_handle::DataHandle`] variants (`AliveBitmap`,
//!   `IndirectArgs`, `AgentScratch`, `SimCfgBuffer`, `SnapshotKick`)
//!   plus the supporting [`crate::cg::data_handle::AgentScratchKind`]
//!   typed enum and a new
//!   [`crate::cg::data_handle::EventRingAccess::Drain`] variant. These
//!   are the structural reads/writes plumbing ops touch; the original
//!   Task 1.3 stub (uninhabited `PlumbingKind`) is replaced by the
//!   typed 7-variant enum in this task.
//! - **Plumbing ordering refinements** (e.g., dedup with cascade
//!   indirect-args ops registered for symmetric folds) are deferred
//!   to Phase 3 schedule synthesis.

use std::collections::BTreeSet;

use dsl_ast::ast::Span;

use crate::cg::data_handle::{AgentFieldId, DataHandle, EventRingAccess, EventRingId};
use crate::cg::dispatch::DispatchShape;
use crate::cg::op::{ComputeOpKind, OpId, PlumbingKind};
use crate::cg::program::CgProgram;

use super::error::LoweringError;
use super::expr::LoweringCtx;

// ---------------------------------------------------------------------------
// Synthesizer
// ---------------------------------------------------------------------------

/// Walk `prog`'s existing user-facing ops and produce the deduplicated
/// list of [`PlumbingKind`] values the schedule needs.
///
/// See the module docstring's "Synthesizer semantics" + "Synthesizer
/// ordering" sections for the rules. Result is deterministic — the
/// same input produces the same output, with conditional ring ids in
/// ascending order.
///
/// The function reads `prog.ops` only; it does not mutate the program
/// or the builder. The driver (Task 2.8) calls this AFTER all user
/// ops have been added but BEFORE [`lower_plumbing`] pushes the
/// plumbing ops.
///
/// # Limitations
///
/// See the module docstring's "Limitations" section. Today the
/// conditional plumbing kinds key off
/// [`crate::cg::dispatch::DispatchShape::PerEvent { source_ring }`]
/// dispatches and [`EventRingAccess::Drain`] reads only — additional
/// trigger conditions would require extending this function.
pub fn synthesize_plumbing_ops(prog: &CgProgram) -> Vec<PlumbingKind> {
    let mut needs_alive_bitmap = false;
    let mut indirect_rings: BTreeSet<EventRingId> = BTreeSet::new();
    let mut drain_rings: BTreeSet<EventRingId> = BTreeSet::new();

    for op in &prog.ops {
        // Detect alive-field writes. PlumbingKind::AliveBitmap reads
        // every agent's alive field; we only need to fire it when a
        // user op (typically an apply-path physics rule's `Emit` →
        // resolved in driver via `record_write`) actually mutates
        // alive. The auto-walker does NOT synthesize reads/writes for
        // alive on its own — they appear because lowering recorded
        // them.
        for write in &op.writes {
            if let DataHandle::AgentField {
                field: AgentFieldId::Alive,
                ..
            } = write
            {
                needs_alive_bitmap = true;
            }
        }

        // Detect PerEvent dispatches — each distinct source ring needs
        // its indirect-args buffer seeded.
        if let DispatchShape::PerEvent { source_ring } = op.shape {
            indirect_rings.insert(source_ring);
        }

        // Detect Drain reads — each ring referenced with
        // EventRingAccess::Drain needs a DrainEvents plumbing op.
        for read in &op.reads {
            if let DataHandle::EventRing {
                ring,
                kind: EventRingAccess::Drain,
            } = read
            {
                drain_rings.insert(*ring);
            }
        }
    }

    let mut kinds = Vec::new();
    // Always-on, leading.
    kinds.push(PlumbingKind::UploadSimCfg);
    kinds.push(PlumbingKind::PackAgents);

    if needs_alive_bitmap {
        kinds.push(PlumbingKind::AliveBitmap);
    }
    for ring in &indirect_rings {
        kinds.push(PlumbingKind::SeedIndirectArgs { ring: *ring });
    }
    for ring in &drain_rings {
        kinds.push(PlumbingKind::DrainEvents { ring: *ring });
    }

    // Always-on, trailing.
    kinds.push(PlumbingKind::UnpackAgents);
    kinds.push(PlumbingKind::KickSnapshot);

    kinds
}

// ---------------------------------------------------------------------------
// Lowering pass
// ---------------------------------------------------------------------------

/// Lower a list of [`PlumbingKind`] values into one
/// [`ComputeOpKind::Plumbing`] op per entry, pushed onto `ctx.builder`
/// in source order.
///
/// # Parameters
///
/// - `kinds`: the plumbing kinds the schedule needs ops for. The
///   driver (Task 2.8) typically obtains this list from
///   [`synthesize_plumbing_ops`]; tests pass it directly. This pass
///   treats the list as opaque: source order is preserved, duplicates
///   are allowed, no validation is performed against the rest of the
///   program.
/// - `ctx`: the lowering context (carries the in-flight builder). The
///   pass touches only `ctx.builder.add_op` — no expression-level
///   helpers, no view / action / variant registries.
///
/// # Returns
///
/// One [`OpId`] per input entry, in input order. Empty input returns
/// an empty `Vec` and pushes nothing.
///
/// # Errors
///
/// See [`LoweringError`] for the closed defect set. Only
/// [`LoweringError::BuilderRejected`] is reachable today — a defense-
/// in-depth wrap around [`crate::cg::program::CgProgramBuilder::add_op`]
/// for catching builder invariant drift. The op kind carries no
/// expr / list id references, so
/// [`crate::cg::program::CgProgramBuilder::validate_op_kind_refs`]'s
/// `Plumbing { .. }` arm is `Ok(())` unconditionally.
///
/// # Side effects
///
/// On success: `kinds.len()` ops pushed onto the builder; no
/// expression sub-trees are produced (plumbing kinds carry no embedded
/// `CgExpr`). Each op uses the canonical
/// [`PlumbingKind::dispatch_shape`] for its kind. On failure at index
/// `i`: the first `i` ops have been pushed; ops past the failure
/// point are not added.
///
/// # Limitations
///
/// See the module docstring's "Limitations" section. The pass takes a
/// driver-supplied list rather than calling
/// [`synthesize_plumbing_ops`] internally so callers can override the
/// inventory (tests, fixtures). All ops use [`Span::dummy`]; duplicate
/// kinds are allowed (schedule synthesis dedupes); source order is
/// preserved.
pub fn lower_plumbing(
    kinds: &[PlumbingKind],
    ctx: &mut LoweringCtx<'_>,
) -> Result<Vec<OpId>, LoweringError> {
    let mut op_ids = Vec::with_capacity(kinds.len());
    for kind in kinds {
        let computekind = ComputeOpKind::Plumbing { kind: *kind };
        let shape = kind.dispatch_shape();
        let op_id =
            ctx.builder
                .add_op(computekind, shape, Span::dummy())
                .map_err(|e| LoweringError::BuilderRejected {
                    error: e,
                    span: Span::dummy(),
                })?;
        op_ids.push(op_id);
    }
    Ok(op_ids)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, AgentScratchKind, DataHandle, EventRingAccess, EventRingId, MaskId,
    };
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{ComputeOpKind, EventKindId, PhysicsRuleId, ReplayabilityFlag, Span};
    use crate::cg::program::CgProgramBuilder;
    use crate::cg::stmt::CgStmtList;

    // ---- Helpers ----

    fn finish_with_user_ops<F>(f: F) -> CgProgram
    where
        F: FnOnce(&mut CgProgramBuilder),
    {
        let mut builder = CgProgramBuilder::new();
        f(&mut builder);
        builder.finish()
    }

    // ---- 1. Each plumbing kind: lower it, assert reads/writes ----

    #[test]
    fn lower_plumbing_pack_agents_reads_every_field_writes_packed_scratch() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ids = lower_plumbing(&[PlumbingKind::PackAgents], &mut ctx).expect("lowers");
        assert_eq!(ids, vec![OpId(0)]);
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::PerAgent);
        match &op.kind {
            ComputeOpKind::Plumbing { kind } => assert_eq!(*kind, PlumbingKind::PackAgents),
            other => panic!("unexpected kind {other:?}"),
        }
        // Auto-walker reproduces PlumbingKind::dependencies.
        let (expected_r, expected_w) = PlumbingKind::PackAgents.dependencies();
        assert_eq!(op.reads, expected_r);
        assert_eq!(op.writes, expected_w);
        // Sanity — the writes contain the packed scratch.
        assert!(op.writes.contains(&DataHandle::AgentScratch {
            kind: AgentScratchKind::Packed,
        }));
        // Reads contain alive (a representative agent field).
        assert!(op.reads.contains(&DataHandle::AgentField {
            field: AgentFieldId::Alive,
            target: AgentRef::Self_,
        }));
    }

    #[test]
    fn lower_plumbing_unpack_agents_is_inverse_of_pack() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ids = lower_plumbing(
            &[PlumbingKind::PackAgents, PlumbingKind::UnpackAgents],
            &mut ctx,
        )
        .expect("lowers");
        assert_eq!(ids, vec![OpId(0), OpId(1)]);
        let prog = builder.finish();
        // Pack reads = Unpack writes; Pack writes = Unpack reads.
        assert_eq!(prog.ops[0].reads, prog.ops[1].writes);
        assert_eq!(prog.ops[0].writes, prog.ops[1].reads);
    }

    #[test]
    fn lower_plumbing_alive_bitmap_uses_per_word_dispatch() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(&[PlumbingKind::AliveBitmap], &mut ctx).expect("lowers");
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::PerWord);
        assert_eq!(
            op.reads,
            vec![DataHandle::AgentField {
                field: AgentFieldId::Alive,
                target: AgentRef::Self_,
            }]
        );
        assert_eq!(op.writes, vec![DataHandle::AliveBitmap]);
    }

    #[test]
    fn lower_plumbing_drain_events_uses_per_event_dispatch_with_ring() {
        let ring = EventRingId(7);
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(&[PlumbingKind::DrainEvents { ring }], &mut ctx).expect("lowers");
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(
            op.shape,
            DispatchShape::PerEvent {
                source_ring: ring,
            }
        );
        assert_eq!(
            op.reads,
            vec![DataHandle::EventRing {
                ring,
                kind: EventRingAccess::Drain,
            }]
        );
        assert!(op.writes.is_empty());
    }

    #[test]
    fn lower_plumbing_upload_sim_cfg_writes_sim_cfg_buffer() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(&[PlumbingKind::UploadSimCfg], &mut ctx).expect("lowers");
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::OneShot);
        assert!(op.reads.is_empty());
        assert_eq!(op.writes, vec![DataHandle::SimCfgBuffer]);
    }

    #[test]
    fn lower_plumbing_kick_snapshot_writes_snapshot_kick() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(&[PlumbingKind::KickSnapshot], &mut ctx).expect("lowers");
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::OneShot);
        assert!(op.reads.is_empty());
        assert_eq!(op.writes, vec![DataHandle::SnapshotKick]);
    }

    #[test]
    fn lower_plumbing_seed_indirect_args_pairs_ring_with_indirect_args_buffer() {
        let ring = EventRingId(3);
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(&[PlumbingKind::SeedIndirectArgs { ring }], &mut ctx).expect("lowers");
        let prog = builder.finish();
        let op = &prog.ops[0];
        assert_eq!(op.shape, DispatchShape::OneShot);
        assert_eq!(
            op.reads,
            vec![DataHandle::EventRing {
                ring,
                kind: EventRingAccess::Read,
            }]
        );
        assert_eq!(op.writes, vec![DataHandle::IndirectArgs { ring }]);
    }

    // ---- 2. Synthesizer: empty program → always-on plumbing -----------

    #[test]
    fn synthesize_empty_program_returns_always_on_plumbing() {
        let prog = finish_with_user_ops(|_| {});
        let kinds = synthesize_plumbing_ops(&prog);
        assert_eq!(
            kinds,
            vec![
                PlumbingKind::UploadSimCfg,
                PlumbingKind::PackAgents,
                PlumbingKind::UnpackAgents,
                PlumbingKind::KickSnapshot,
            ]
        );
    }

    // ---- 3. Synthesizer: alive write triggers AliveBitmap -------------

    #[test]
    fn synthesize_alive_field_write_triggers_alive_bitmap() {
        // Build a program with a single physics-rule op whose body
        // contains no assigns, then post-construction record a write
        // to AgentField { Alive, Self_ } via `record_write` (the same
        // seam lowering uses for registry-resolved bindings).
        let mut builder = CgProgramBuilder::new();
        // Empty body list, ref'd by the rule.
        let body = builder
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("add list");
        let op_id = builder
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: Some(EventKindId(0)),
                    body,
                    replayable: ReplayabilityFlag::Replayable,
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
            )
            .expect("add op");
        // Inject an alive write — the synthesizer reads `op.writes`.
        let mut prog = builder.finish();
        prog.ops[op_id.0 as usize].record_write(DataHandle::AgentField {
            field: AgentFieldId::Alive,
            target: AgentRef::Self_,
        });
        let kinds = synthesize_plumbing_ops(&prog);
        assert!(
            kinds.contains(&PlumbingKind::AliveBitmap),
            "expected AliveBitmap to be in {kinds:?}"
        );
        // Ordering: AliveBitmap must come after PackAgents and before
        // UnpackAgents.
        let pack_idx = kinds
            .iter()
            .position(|k| *k == PlumbingKind::PackAgents)
            .unwrap();
        let alive_idx = kinds
            .iter()
            .position(|k| *k == PlumbingKind::AliveBitmap)
            .unwrap();
        let unpack_idx = kinds
            .iter()
            .position(|k| *k == PlumbingKind::UnpackAgents)
            .unwrap();
        assert!(pack_idx < alive_idx);
        assert!(alive_idx < unpack_idx);
    }

    // ---- 4. Synthesizer: PerEvent op triggers SeedIndirectArgs --------

    #[test]
    fn synthesize_per_event_dispatch_triggers_seed_indirect_args() {
        let ring = EventRingId(2);
        let mut builder = CgProgramBuilder::new();
        let body = builder
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("add list");
        builder
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: Some(EventKindId(0)),
                    body,
                    replayable: ReplayabilityFlag::Replayable,
                },
                DispatchShape::PerEvent { source_ring: ring },
                Span::dummy(),
            )
            .expect("add op");
        let prog = builder.finish();
        let kinds = synthesize_plumbing_ops(&prog);
        assert!(
            kinds.contains(&PlumbingKind::SeedIndirectArgs { ring }),
            "expected SeedIndirectArgs(ring={ring:?}) in {kinds:?}"
        );
    }

    // ---- 5. Synthesizer: multiple distinct rings → multiple seeds -----

    #[test]
    fn synthesize_multiple_rings_yield_multiple_seed_indirect_args() {
        let ring_a = EventRingId(2);
        let ring_b = EventRingId(7);
        let mut builder = CgProgramBuilder::new();
        let body = builder
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("add list");
        for ring in [ring_a, ring_b] {
            builder
                .add_op(
                    ComputeOpKind::PhysicsRule {
                        rule: PhysicsRuleId(0),
                        on_event: Some(EventKindId(0)),
                        body,
                        replayable: ReplayabilityFlag::Replayable,
                    },
                    DispatchShape::PerEvent { source_ring: ring },
                    Span::dummy(),
                )
                .expect("add op");
        }
        let prog = builder.finish();
        let kinds = synthesize_plumbing_ops(&prog);
        assert!(kinds.contains(&PlumbingKind::SeedIndirectArgs { ring: ring_a }));
        assert!(kinds.contains(&PlumbingKind::SeedIndirectArgs { ring: ring_b }));

        // Sorted ascending — ring 2 before ring 7.
        let idx_a = kinds
            .iter()
            .position(|k| *k == PlumbingKind::SeedIndirectArgs { ring: ring_a })
            .unwrap();
        let idx_b = kinds
            .iter()
            .position(|k| *k == PlumbingKind::SeedIndirectArgs { ring: ring_b })
            .unwrap();
        assert!(idx_a < idx_b, "expected ring_a sorted before ring_b");
    }

    #[test]
    fn synthesize_duplicate_ring_yields_one_seed_indirect_args() {
        let ring = EventRingId(2);
        let mut builder = CgProgramBuilder::new();
        let body = builder
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("add list");
        for _ in 0..3 {
            builder
                .add_op(
                    ComputeOpKind::PhysicsRule {
                        rule: PhysicsRuleId(0),
                        on_event: Some(EventKindId(0)),
                        body,
                        replayable: ReplayabilityFlag::Replayable,
                    },
                    DispatchShape::PerEvent { source_ring: ring },
                    Span::dummy(),
                )
                .expect("add op");
        }
        let prog = builder.finish();
        let kinds = synthesize_plumbing_ops(&prog);
        let count = kinds
            .iter()
            .filter(|k| **k == PlumbingKind::SeedIndirectArgs { ring })
            .count();
        assert_eq!(count, 1, "expected dedup on identical ring");
    }

    // ---- 6. Auto-walker test: compute_dependencies agrees -------------

    #[test]
    fn auto_walker_matches_dependencies_signature_for_each_kind() {
        // Mirrors the spatial-pass auto-walker test: every plumbing
        // kind's lowered op's reads/writes must equal the
        // `(reads, writes)` table on `PlumbingKind::dependencies`.
        for kind in [
            PlumbingKind::PackAgents,
            PlumbingKind::UnpackAgents,
            PlumbingKind::AliveBitmap,
            PlumbingKind::DrainEvents {
                ring: EventRingId(2),
            },
            PlumbingKind::UploadSimCfg,
            PlumbingKind::KickSnapshot,
            PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(3),
            },
        ] {
            let mut builder = CgProgramBuilder::new();
            let mut ctx = LoweringCtx::new(&mut builder);
            lower_plumbing(&[kind], &mut ctx).expect("lowers");
            let prog = builder.finish();
            let op = &prog.ops[0];
            let (expected_r, expected_w) = kind.dependencies();
            assert_eq!(
                op.reads, expected_r,
                "kind={kind}: reads diverged from PlumbingKind::dependencies()"
            );
            assert_eq!(
                op.writes, expected_w,
                "kind={kind}: writes diverged from PlumbingKind::dependencies()"
            );
        }
    }

    // ---- 7. Snapshot: pinned `Display` form for a lowered op ----------

    #[test]
    fn snapshot_alive_bitmap_op_display() {
        // Pins the wire format produced by `ComputeOp`'s Display impl
        // for an `AliveBitmap` plumbing op.
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(&[PlumbingKind::AliveBitmap], &mut ctx).expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=plumbing(alive_bitmap) shape=per_word reads=[agent.self.alive] writes=[alive_bitmap]"
        );
    }

    #[test]
    fn snapshot_seed_indirect_args_op_display() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(
            &[PlumbingKind::SeedIndirectArgs {
                ring: EventRingId(3),
            }],
            &mut ctx,
        )
        .expect("lowers");
        let prog = builder.finish();
        assert_eq!(
            format!("{}", prog.ops[0]),
            "op#0 kind=plumbing(seed_indirect_args(ring=#3)) shape=one_shot reads=[event_ring[#3].read] writes=[indirect_args[#3]]"
        );
    }

    // ---- 8. Empty input is Ok(vec![]) and pushes nothing --------------

    #[test]
    fn lower_plumbing_empty_input_pushes_no_ops() {
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        let ids = lower_plumbing(&[], &mut ctx).expect("lowers");
        assert!(ids.is_empty());
        let prog = builder.finish();
        assert!(prog.ops.is_empty());
    }

    // ---- 9. End-to-end: synthesize then lower; well-formed passes -----

    #[test]
    fn end_to_end_synthesize_and_lower_yields_well_formed_plumbing() {
        // Build a small program with a mask-predicate user op, then
        // synthesize plumbing on top, lower it, and assert the
        // resulting plumbing-only ops carry the expected shapes and
        // reads/writes. This is a smoke test for the synthesizer +
        // lowering integration; the driver (Task 2.8) will consolidate
        // both calls.
        let mut builder = CgProgramBuilder::new();
        // Push a trivial mask-predicate op so the program isn't empty
        // and so the auto-walker has at least one user-facing op to
        // walk past.
        let predicate = builder
            .add_expr(CgExpr::Lit(LitValue::Bool(true)))
            .expect("add expr");
        let _user = builder
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask: MaskId(0),
                    predicate,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .expect("add op");
        // Snapshot the program so synthesize sees only the user op.
        let snapshot = builder.program().clone();
        let kinds = synthesize_plumbing_ops(&snapshot);
        // Empty-program case applies — alive isn't written, no
        // PerEvent dispatches, no Drain reads. Always-on only.
        assert_eq!(kinds.len(), 4);

        // Lower onto the SAME builder.
        {
            let mut ctx = LoweringCtx::new(&mut builder);
            lower_plumbing(&kinds, &mut ctx).expect("lowers");
        }
        let prog = builder.finish();
        // 1 user op + 4 plumbing = 5.
        assert_eq!(prog.ops.len(), 5);
        // Every plumbing op's reads/writes match its kind's
        // `dependencies()`.
        for (i, kind) in kinds.iter().enumerate() {
            let op = &prog.ops[i + 1];
            let (expected_r, expected_w) = kind.dependencies();
            assert_eq!(op.reads, expected_r, "ops[{}] reads", i + 1);
            assert_eq!(op.writes, expected_w, "ops[{}] writes", i + 1);
        }
        // Note: we deliberately do NOT assert `check_well_formed` here.
        // PackAgents writes every AgentField; UnpackAgents reads the
        // packed scratch and writes every AgentField. Within a single
        // tick the read/write graph contains a cycle (pack writes
        // scratch → unpack reads scratch → unpack writes alive → pack
        // reads alive on the next iteration). Schedule synthesis
        // (Phase 3) is the layer that resolves this by sequencing the
        // ops across phase boundaries; the well-formed pass's cycle
        // detector correctly flags the unsequenced inventory as a
        // cycle. Asserting well-formed here would be asserting on the
        // schedule, not on lowering.
    }

    // ---- 10. Synthesizer: drain reads trigger DrainEvents -------------

    #[test]
    fn synthesize_drain_read_triggers_drain_events() {
        // Construct a program with a user op that reads a ring with
        // `EventRingAccess::Drain`. (No DSL-surface op produces this
        // today, but the synthesizer must react to it when lowering
        // does — defense-in-depth.)
        let mut builder = CgProgramBuilder::new();
        let body = builder
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("add list");
        let op_id = builder
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: Some(EventKindId(0)),
                    body,
                    replayable: ReplayabilityFlag::Replayable,
                },
                DispatchShape::PerEvent {
                    source_ring: EventRingId(0),
                },
                Span::dummy(),
            )
            .expect("add op");
        let mut prog = builder.finish();
        prog.ops[op_id.0 as usize].record_read(DataHandle::EventRing {
            ring: EventRingId(9),
            kind: EventRingAccess::Drain,
        });
        let kinds = synthesize_plumbing_ops(&prog);
        assert!(kinds.contains(&PlumbingKind::DrainEvents {
            ring: EventRingId(9),
        }));
    }

    // ---- 11. Synthesizer: full canonical output ordering --------------

    #[test]
    fn synthesize_full_canonical_ordering() {
        // Pin the documented end-to-end output sequence:
        //   [UploadSimCfg, PackAgents, AliveBitmap?,
        //    seed_indirect_args*, drain_events*,
        //    UnpackAgents, KickSnapshot]
        //
        // Build a single user op that triggers ALL three conditional
        // kinds simultaneously:
        //   - writes AgentField{Alive, Self_}        → AliveBitmap
        //   - shape PerEvent { source_ring: ring_a } → SeedIndirectArgs{a}
        //   - reads EventRing { ring: ring_b, Drain} → DrainEvents{b}
        //
        // A single op is enough because alive-write, shape, and Drain
        // read are independent triggers; the synthesizer must coalesce
        // them into the documented order regardless of source proximity.
        let ring_a = EventRingId(2);
        let ring_b = EventRingId(5);
        let mut builder = CgProgramBuilder::new();
        let body = builder
            .add_stmt_list(CgStmtList::new(vec![]))
            .expect("add list");
        let op_id = builder
            .add_op(
                ComputeOpKind::PhysicsRule {
                    rule: PhysicsRuleId(0),
                    on_event: Some(EventKindId(0)),
                    body,
                    replayable: ReplayabilityFlag::Replayable,
                },
                DispatchShape::PerEvent { source_ring: ring_a },
                Span::dummy(),
            )
            .expect("add op");
        let mut prog = builder.finish();
        // Inject the alive write + Drain read via the same `record_*`
        // seam lowering uses for registry-resolved bindings.
        prog.ops[op_id.0 as usize].record_write(DataHandle::AgentField {
            field: AgentFieldId::Alive,
            target: AgentRef::Self_,
        });
        prog.ops[op_id.0 as usize].record_read(DataHandle::EventRing {
            ring: ring_b,
            kind: EventRingAccess::Drain,
        });

        let kinds = synthesize_plumbing_ops(&prog);
        // Assert the EXACT vec — no `contains`, no relative-order
        // checks. This pins the canonical ordering end-to-end.
        assert_eq!(
            kinds,
            vec![
                PlumbingKind::UploadSimCfg,
                PlumbingKind::PackAgents,
                PlumbingKind::AliveBitmap,
                PlumbingKind::SeedIndirectArgs { ring: ring_a },
                PlumbingKind::DrainEvents { ring: ring_b },
                PlumbingKind::UnpackAgents,
                PlumbingKind::KickSnapshot,
            ]
        );
    }

    // ---- 12. Source-order preserved in lower_plumbing -----------------

    #[test]
    fn lower_plumbing_preserves_source_order() {
        let kinds = [
            PlumbingKind::KickSnapshot,
            PlumbingKind::UploadSimCfg,
            PlumbingKind::PackAgents,
        ];
        let mut builder = CgProgramBuilder::new();
        let mut ctx = LoweringCtx::new(&mut builder);
        lower_plumbing(&kinds, &mut ctx).expect("lowers");
        let prog = builder.finish();
        for (i, expected) in kinds.iter().enumerate() {
            match &prog.ops[i].kind {
                ComputeOpKind::Plumbing { kind } => assert_eq!(kind, expected),
                other => panic!("op#{i}: unexpected kind {other:?}"),
            }
        }
    }

}
