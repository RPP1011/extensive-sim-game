//! Regression: in-step GPU chronicle producers feed their consumers in
//! the SAME tick.
//!
//! Pre-2026-05-12 the schedule synthesizer ran chronicle consumers
//! BEFORE the producers that emit the kinds they want, because the
//! `cycle_edge_key` projection collapsed all event-ring reads/writes
//! to the ring identity and the resulting cycle-fallback Kahn used
//! source-order tie-break (which puts user-authored physics rules
//! before driver-injected `verb_chronicle_<verb>` rules). Net: every
//! `apply_ability`-driven fixture (squad_skirmish + 5 others)
//! produced events that arrived too late and got overwritten the next
//! tick. The kind-aware refinement in
//! `dsl_compiler::cg::schedule::topology::dependency_graph` resolves
//! this by matching producer.emits ⊇ {consumer.consumes} per event
//! KIND.
//!
//! This test pins the fix at the schedule level by:
//!   1. Lowering a synthetic two-rule program: a per-agent producer
//!      (emits `EffectDamageApplied` via `apply_ability`) + a
//!      chronicle consumer (`on EffectDamageApplied { … } → emit
//!      Damaged`).
//!   2. Asserting the synthesized schedule places the producer's
//!      stage BEFORE the consumer's stage — the in-tick GPU
//!      `cfg.event_count = event_tail` snapshot will now find the
//!      producer's atomicAdd visible at consumer dispatch time.

use dsl_compiler::cg::data_handle::{DataHandle, EventRingAccess, EventRingId};
use dsl_compiler::cg::dispatch::DispatchShape;
use dsl_compiler::cg::expr::CgExpr;
use dsl_compiler::cg::op::{
    ComputeOp, ComputeOpKind, EventKindId, OpId, PhysicsRuleId, ReplayabilityFlag, Span,
};
use dsl_compiler::cg::program::CgProgram;
use dsl_compiler::cg::schedule::strategy::ScheduleStrategy;
use dsl_compiler::cg::schedule::synthesis::{synthesize_schedule, KernelTopology};
use dsl_compiler::cg::stmt::{CgStmt, CgStmtId, CgStmtList, CgStmtListId};
use dsl_compiler::cg::data_handle::CgExprId;

fn build_apply_ability_producer(
    prog: &mut CgProgram,
    op_id: OpId,
    rule_id: PhysicsRuleId,
) -> ComputeOp {
    // Push an arbitrary expression so the ApplyAbility CgStmt has a
    // valid (any) CgExprId for ability/caster/target. A literal `1u`
    // is the closest-to-real placeholder.
    let expr_id = CgExprId(prog.exprs.len() as u32);
    prog.exprs.push(CgExpr::Lit(
        dsl_compiler::cg::expr::LitValue::U32(1),
    ));
    let apply_stmt_id = CgStmtId(prog.stmts.len() as u32);
    prog.stmts.push(CgStmt::ApplyAbility {
        ability: expr_id,
        caster: expr_id,
        target: expr_id,
        with_aoe_dispatch: false,
    });
    let body_id = CgStmtListId(prog.stmt_lists.len() as u32);
    prog.stmt_lists.push(CgStmtList::new(vec![apply_stmt_id]));
    ComputeOp {
        id: op_id,
        kind: ComputeOpKind::PhysicsRule {
            rule: rule_id,
            on_event: None, // PerAgent
            body: body_id,
            replayable: ReplayabilityFlag::Replayable,
        },
        // Producer writes Append on the chronicle ring. The driver's
        // `collect_emit_destination_rings` would do this for us via
        // the ApplyAbility walk; mirror it inline.
        reads: vec![],
        writes: vec![DataHandle::EventRing {
            ring: EventRingId(0),
            kind: EventRingAccess::Append,
        }],
        shape: DispatchShape::PerAgent,
        span: Span::dummy(),
    }
}

fn build_chronicle_consumer(
    prog: &mut CgProgram,
    op_id: OpId,
    rule_id: PhysicsRuleId,
    consumes_kind: EventKindId,
    emits_kind: EventKindId,
) -> ComputeOp {
    let emit_stmt_id = CgStmtId(prog.stmts.len() as u32);
    prog.stmts.push(CgStmt::Emit {
        event: emits_kind,
        fields: Vec::new(),
    });
    let body_id = CgStmtListId(prog.stmt_lists.len() as u32);
    prog.stmt_lists.push(CgStmtList::new(vec![emit_stmt_id]));
    ComputeOp {
        id: op_id,
        kind: ComputeOpKind::PhysicsRule {
            rule: rule_id,
            on_event: Some(consumes_kind),
            body: body_id,
            replayable: ReplayabilityFlag::Replayable,
        },
        // Wire source-ring read (mirroring `wire_source_ring_reads`)
        // + Append (the consumer also emits a re-emit event).
        reads: vec![DataHandle::EventRing {
            ring: EventRingId(0),
            kind: EventRingAccess::Read,
        }],
        writes: vec![DataHandle::EventRing {
            ring: EventRingId(0),
            kind: EventRingAccess::Append,
        }],
        shape: DispatchShape::PerEvent {
            source_ring: EventRingId(0),
        },
        span: Span::dummy(),
    }
}

fn collect_stage_op_order(
    schedule: &dsl_compiler::cg::schedule::synthesis::ComputeSchedule,
) -> Vec<OpId> {
    schedule
        .stages
        .iter()
        .flat_map(|stage| {
            stage.kernels.iter().flat_map(|topology| match topology {
                KernelTopology::Fused { ops, .. } => ops.clone(),
                KernelTopology::Split { op, .. } => vec![*op],
                KernelTopology::Indirect { producer, consumers } => {
                    let mut all = vec![*producer];
                    all.extend(consumers.iter().copied());
                    all
                }
            })
        })
        .collect()
}

#[test]
fn in_step_apply_ability_producer_runs_before_chronicle_consumer() {
    // Producer first (lower OpId), consumer second.
    let mut prog = CgProgram::default();
    prog.interner
        .event_kinds
        .insert(26, "EffectDamageApplied".into());
    prog.interner.event_kinds.insert(1, "Damaged".into());

    let producer = build_apply_ability_producer(
        &mut prog,
        OpId(0),
        PhysicsRuleId(0),
    );
    prog.ops.push(producer);

    let consumer = build_chronicle_consumer(
        &mut prog,
        OpId(1),
        PhysicsRuleId(1),
        EventKindId(26), // consumes EffectDamageApplied
        EventKindId(1),  // emits Damaged
    );
    prog.ops.push(consumer);

    let result = synthesize_schedule(&prog, ScheduleStrategy::Default);
    // Skipping `result.schedule_diagnostics.is_empty()` — the
    // synthetic test fixture lacks a `SeedIndirectArgs` plumbing op
    // (which the production driver injects via Phase-4 plumbing), so
    // the consumer's `KernelTopology::Indirect` falls back to
    // `Fused` with an `IndirectProducerMissing` diagnostic. That's
    // structurally fine for this ordering-focused test — the producer/
    // consumer schedule positions are what we care about.

    let stage_ops = collect_stage_op_order(&result.schedule);
    let producer_pos = stage_ops
        .iter()
        .position(|&op| op == OpId(0))
        .expect("producer op present in schedule");
    let consumer_pos = stage_ops
        .iter()
        .position(|&op| op == OpId(1))
        .expect("consumer op present in schedule");
    assert!(
        producer_pos < consumer_pos,
        "in-step producer (OpId 0) must dispatch BEFORE its kind-26 chronicle \
         consumer (OpId 1) so the consumer's `cfg.event_count = event_tail` \
         snapshot finds the producer's atomicAdd; got producer at idx \
         {producer_pos}, consumer at idx {consumer_pos}",
    );
}

#[test]
fn out_of_order_source_decl_still_orders_producer_before_consumer() {
    // Same shape, but the consumer is declared FIRST (lower OpId).
    // Without the kind-aware refinement, the ring-collapse cycle's
    // source-order tie-break would emit the consumer before the
    // producer. The fix's data-flow edge MUST still order producer
    // before consumer — this is the squad_skirmish-shape regression
    // where verb_chronicle_<verb> ops (auto-injected at the END of
    // `comp.physics`, hence higher OpId) used to land AFTER user-
    // authored ApplyDamage rules.
    let mut prog = CgProgram::default();
    prog.interner
        .event_kinds
        .insert(26, "EffectDamageApplied".into());
    prog.interner.event_kinds.insert(1, "Damaged".into());

    let consumer = build_chronicle_consumer(
        &mut prog,
        OpId(0),
        PhysicsRuleId(0),
        EventKindId(26),
        EventKindId(1),
    );
    prog.ops.push(consumer);

    let producer = build_apply_ability_producer(
        &mut prog,
        OpId(1),
        PhysicsRuleId(1),
    );
    prog.ops.push(producer);

    let result = synthesize_schedule(&prog, ScheduleStrategy::Default);
    let stage_ops = collect_stage_op_order(&result.schedule);
    let producer_pos = stage_ops
        .iter()
        .position(|&op| op == OpId(1))
        .expect("producer op#1 present in schedule");
    let consumer_pos = stage_ops
        .iter()
        .position(|&op| op == OpId(0))
        .expect("consumer op#0 present in schedule");
    assert!(
        producer_pos < consumer_pos,
        "kind-aware refinement must order producer BEFORE consumer even \
         when producer has a HIGHER OpId — this is the squad_skirmish-shape \
         regression. Got producer at idx {producer_pos}, consumer at idx \
         {consumer_pos}",
    );
}

#[test]
fn kind_mismatch_does_not_create_spurious_edge() {
    // Producer emits kind 26 (EffectDamageApplied) — but the consumer
    // subscribes to kind 7 (some unrelated event). With the kind-
    // aware refinement, NO edge is added, so the schedule order falls
    // back to source order (consumer first because it has the lower
    // OpId). The shape pins that the kind filter actually filters.
    let mut prog = CgProgram::default();
    prog.interner
        .event_kinds
        .insert(26, "EffectDamageApplied".into());
    prog.interner.event_kinds.insert(7, "Unrelated".into());
    prog.interner.event_kinds.insert(1, "Damaged".into());

    // Consumer subscribes to kind 7 — producer emits kind 26.
    let consumer = build_chronicle_consumer(
        &mut prog,
        OpId(0),
        PhysicsRuleId(0),
        EventKindId(7),
        EventKindId(1),
    );
    prog.ops.push(consumer);

    let producer = build_apply_ability_producer(
        &mut prog,
        OpId(1),
        PhysicsRuleId(1),
    );
    prog.ops.push(producer);

    let result = synthesize_schedule(&prog, ScheduleStrategy::Default);
    let stage_ops = collect_stage_op_order(&result.schedule);
    // No data-flow edge → source order wins (consumer first).
    let consumer_pos = stage_ops
        .iter()
        .position(|&op| op == OpId(0))
        .expect("consumer present");
    let producer_pos = stage_ops
        .iter()
        .position(|&op| op == OpId(1))
        .expect("producer present");
    assert!(
        consumer_pos < producer_pos,
        "kind-mismatched producer/consumer pair must NOT add a data-flow \
         edge — source order should be preserved. Got consumer at idx \
         {consumer_pos}, producer at idx {producer_pos}",
    );
}
