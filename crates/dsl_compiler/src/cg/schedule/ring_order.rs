//! Schedule validator — "no chronicle consumer before its producer".
//!
//! # Why this file exists
//!
//! The engine's same-tick event flow is a hard ordering contract: an op
//! that READS this tick's event ring (a `physics on <Event>` consumer,
//! a `view fold on <Event>`, a `merge from <Event>` belief kernel) must
//! be dispatched AFTER every op that APPENDS that event kind into the
//! ring. Break it and nothing fails — the consumer scans a ring segment
//! that has not been written yet, matches zero rows, and the feature it
//! implements simply stops existing. No panic, no WGSL error, no
//! diagnostic.
//!
//! That silence has cost this repo three debug cycles:
//!
//! * **Gap dungeon_stealth#5** — `apply_ability`'s emitted-kind table was
//!   hardcoded to `[26,27,28,29]`, so `ApplyStealthFromChronicle`
//!   (kind 54) had no producer edge and Kahn's put it first;
//!   `stealth_until_tick` stayed 0 forever.
//! * **webband_colony / S5** — a cycle stall let the force-pick emit
//!   `BeliefSocialMerge` 65 stages ahead of `PhysicsSupperGather`;
//!   supper gossip died silently. Fixed inside the sort (the forced
//!   pick now skips ring consumers with pending producers).
//! * **webband_colony / S5b** — the `merge from` lowering used a source
//!   INDEX as the event kind id, so the kind-refined ring edge never
//!   formed at all; the sort's guard cannot help when the edge is
//!   absent, and the same feature died the same silent death.
//!
//! Two of those three are edge-CONSTRUCTION bugs, not ordering-policy
//! bugs. A guard inside the sort can never catch them. So this module
//! validates the FINISHED schedule against the program's own event
//! facts, and does it on every emit — the loud check the class has been
//! missing.
//!
//! # What is checked
//!
//! 1. [`RingOrderIssueKind::ConsumerBeforeProducer`] — a producer→consumer
//!    ring edge exists in the [`DepGraph`] but the consumer is scheduled
//!    at an earlier stage. `cyclic` records whether producer and consumer
//!    share a strongly-connected component, i.e. whether a real cycle
//!    forced the break (documented + deterministic, see
//!    [`super::topology::ForcedRingBreak`]) or whether it is a scheduler
//!    bug (never expected).
//! 2. [`RingOrderIssueKind::ConsumerKindNotInterned`] — an op subscribes
//!    to an [`EventKindId`] that has no name in the program's event-kind
//!    interner. This is the S5b signature: a wrong kind id is triple
//!    silent (no name, wrong payload offset, no dep-graph edge), and it
//!    is impossible in a correctly lowered program, so it is always a
//!    compiler bug.
//! 3. [`RingOrderIssueKind::ConsumerKindHasNoProducer`] — an op
//!    subscribes to an interned kind that NO op in the program emits.
//!    Legitimate for host-injected and engine-injected kinds, so this is
//!    reported at INFO severity only and never promoted to an error.
//!
//! # Cost
//!
//! O(ring edges + ops). Runs once per `synthesize_schedule` call.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::cg::data_handle::{CycleEdgeKey, DataHandle, EventRingAccess};
use crate::cg::op::{EventKindId, OpId};
use crate::cg::program::CgProgram;

use super::topology::{compute_event_ring_kind_facts, find_cycles, DepGraph};

/// Severity of a [`RingOrderIssue`]. Callers (build scripts) print
/// everything and promote only [`RingOrderSeverity::Bug`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RingOrderSeverity {
    /// The schedule is wrong and no cycle explains it — a compiler
    /// defect. Promoted to a hard build error under
    /// `SIM_REQUIRE_ALL_RULES`.
    Bug,
    /// The schedule sacrifices a same-tick ring ordering because the
    /// ring sub-relation is genuinely cyclic. Deterministic and
    /// documented, but the affected feature WILL read a stale ring —
    /// always printed.
    Forced,
    /// Informational; may be entirely legitimate.
    Info,
}

impl RingOrderSeverity {
    /// Stable snake_case label for logs.
    pub fn label(&self) -> &'static str {
        match self {
            RingOrderSeverity::Bug => "bug",
            RingOrderSeverity::Forced => "forced",
            RingOrderSeverity::Info => "info",
        }
    }
}

/// Typed payload of a [`RingOrderIssue`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RingOrderIssueKind {
    /// `consumer` reads event kind `kind` from the ring but is
    /// scheduled before `producer`, which appends it.
    ConsumerBeforeProducer {
        /// The op reading the ring.
        consumer: OpId,
        /// The op appending the kind the consumer reads.
        producer: OpId,
        /// The subscribed kind, when the consumer carries one.
        kind: Option<EventKindId>,
        /// `true` when producer and consumer share a non-trivial SCC —
        /// the break was unavoidable.
        cyclic: bool,
    },
    /// `consumer` subscribes to an [`EventKindId`] with no entry in the
    /// program's event-kind interner. Always a lowering bug.
    ConsumerKindNotInterned {
        /// The op with the bad subscription.
        consumer: OpId,
        /// The unresolvable kind id.
        kind: EventKindId,
    },
    /// `consumer` subscribes to an interned kind no op emits. Usually a
    /// host- or engine-injected event; informational only.
    ConsumerKindHasNoProducer {
        /// The subscribing op.
        consumer: OpId,
        /// The kind nobody emits.
        kind: EventKindId,
    },
}

/// One finding from [`validate_ring_order`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RingOrderIssue {
    /// Typed payload.
    pub kind: RingOrderIssueKind,
    /// How loud the caller should be about it.
    pub severity: RingOrderSeverity,
    /// Human-readable rendering (op ids + event NAMES where known).
    pub message: String,
}

impl fmt::Display for RingOrderIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity.label(), self.message)
    }
}

/// Does this op read the chronicle ring (as opposed to draining it)?
fn reads_ring(op: &crate::cg::op::ComputeOp) -> bool {
    op.reads.iter().any(|r| {
        matches!(
            r,
            DataHandle::EventRing {
                kind: EventRingAccess::Read,
                ..
            }
        )
    })
}

fn kind_name(prog: &CgProgram, kind: EventKindId) -> Option<&str> {
    prog.interner.event_kinds.get(&kind.0).map(String::as_str)
}

fn render_kind(prog: &CgProgram, kind: EventKindId) -> String {
    match kind_name(prog, kind) {
        Some(n) => format!("{n}(#{})", kind.0),
        None => format!("<uninterned kind #{}>", kind.0),
    }
}

/// Validate that `order` (a linearization of `graph`, e.g. the stage
/// order [`super::synthesis::synthesize_schedule`] produced) respects
/// every event-ring producer→consumer edge, and that every ring
/// subscription in `prog` is resolvable.
///
/// Returns findings sorted deterministically (by typed payload), so the
/// output is byte-stable across runs and suitable for snapshot tests.
///
/// Ops in `order` that the graph does not index are ignored; ops missing
/// from `order` are treated as scheduled last (a defensive stance — a
/// dropped op cannot be "before" anything).
pub fn validate_ring_order(
    prog: &CgProgram,
    graph: &DepGraph,
    order: &[OpId],
) -> Vec<RingOrderIssue> {
    let n = graph.op_count;
    let mut pos: Vec<usize> = vec![usize::MAX; n];
    for (i, op) in order.iter().enumerate() {
        let idx = op.0 as usize;
        if idx < n {
            pos[idx] = i;
        }
    }

    // SCC id per op, so a violation can say whether a cycle explains it.
    let mut scc_of: Vec<usize> = vec![usize::MAX; n];
    for (sid, scc) in find_cycles(graph).iter().enumerate() {
        if scc.len() < 2 {
            continue;
        }
        for op in scc {
            let i = op.0 as usize;
            if i < n {
                scc_of[i] = sid;
            }
        }
    }

    let facts: Vec<_> = prog
        .ops
        .iter()
        .map(|op| compute_event_ring_kind_facts(op, prog))
        .collect();

    let mut issues: Vec<RingOrderIssue> = Vec::new();

    // --- 1. Order violations on real ring edges ------------------------
    for ((p, c), reasons) in graph.edge_reasons.iter() {
        if !reasons.iter().any(|r| matches!(r, CycleEdgeKey::Ring(_))) {
            continue;
        }
        let (pi, ci) = (p.0 as usize, c.0 as usize);
        if pi >= n || ci >= n {
            continue;
        }
        if pos[ci] >= pos[pi] {
            continue;
        }
        let cyclic = scc_of[ci] != usize::MAX && scc_of[ci] == scc_of[pi];
        let kind = facts.get(ci).and_then(|f| f.consumed);
        let kind_txt = kind
            .map(|k| render_kind(prog, k))
            .unwrap_or_else(|| "<any kind>".to_string());
        let message = format!(
            "event-ring consumer op#{} runs at stage {} but its producer op#{} runs at \
             stage {} — the same-tick read of {kind_txt} sees a ring the producer has not \
             written yet{}",
            c.0,
            pos[ci],
            p.0,
            pos[pi],
            if cyclic {
                " (producer and consumer are in the same dependency cycle; the break is \
                 forced and deterministic — re-shape the rules to remove the cycle if the \
                 same-tick flow matters)"
            } else {
                " (NO dependency cycle explains this — scheduler bug)"
            },
        );
        issues.push(RingOrderIssue {
            kind: RingOrderIssueKind::ConsumerBeforeProducer {
                consumer: *c,
                producer: *p,
                kind,
                cyclic,
            },
            severity: if cyclic {
                RingOrderSeverity::Forced
            } else {
                RingOrderSeverity::Bug
            },
            message,
        });
    }

    // --- 2/3. Subscription sanity --------------------------------------
    // Union of every kind any op appends into the ring.
    let mut all_emitted: BTreeSet<EventKindId> = BTreeSet::new();
    for f in &facts {
        all_emitted.extend(f.emitted.iter().copied());
    }
    // Group consumers by kind so one bad kind reports once per op but
    // deterministically ordered.
    let mut by_kind: BTreeMap<EventKindId, Vec<OpId>> = BTreeMap::new();
    for (i, op) in prog.ops.iter().enumerate() {
        if !reads_ring(op) {
            continue;
        }
        let Some(k) = facts[i].consumed else {
            continue;
        };
        by_kind.entry(k).or_default().push(OpId(i as u32));
    }
    for (kind, consumers) in by_kind {
        let interned = kind_name(prog, kind).is_some();
        let has_producer = all_emitted.contains(&kind);
        for consumer in consumers {
            if !interned {
                issues.push(RingOrderIssue {
                    kind: RingOrderIssueKind::ConsumerKindNotInterned { consumer, kind },
                    severity: RingOrderSeverity::Bug,
                    message: format!(
                        "op#{} subscribes to event kind #{} which has no name in the \
                         event-kind interner — a wrong kind id here is silent three times \
                         over (no kernel name, wrong payload offset, and NO dep-graph edge \
                         to its producer, so the scheduler cannot order it)",
                        consumer.0, kind.0,
                    ),
                });
            } else if !has_producer {
                issues.push(RingOrderIssue {
                    kind: RingOrderIssueKind::ConsumerKindHasNoProducer { consumer, kind },
                    severity: RingOrderSeverity::Info,
                    message: format!(
                        "op#{} subscribes to {} but no op in this program emits it \
                         (fine for host- or engine-injected events; a typo otherwise)",
                        consumer.0,
                        render_kind(prog, kind),
                    ),
                });
            }
        }
    }

    issues.sort();
    issues.dedup();
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cg::data_handle::{AgentFieldId, AgentRef, EventRingId, MaskId};
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{ComputeOpKind, Span};
    use crate::cg::program::CgProgramBuilder;
    use crate::cg::schedule::topology::dependency_graph;

    /// A PhysicsRule op with the given body, optionally subscribing to
    /// `on_event`.
    fn add_rule_with_body(
        b: &mut CgProgramBuilder,
        on_event: Option<EventKindId>,
        stmts: Vec<crate::cg::stmt::CgStmtId>,
    ) -> OpId {
        let body = b
            .add_stmt_list(crate::cg::stmt::CgStmtList::new(stmts))
            .unwrap();
        b.add_op(
            ComputeOpKind::PhysicsRule {
                rule: crate::cg::op::PhysicsRuleId(0),
                on_event,
                body,
                replayable: crate::cg::op::ReplayabilityFlag::Replayable,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap()
    }

    fn add_rule(b: &mut CgProgramBuilder, on_event: Option<EventKindId>) -> OpId {
        add_rule_with_body(b, on_event, Vec::new())
    }

    fn add_mask(b: &mut CgProgramBuilder, mask: u32) -> OpId {
        let pred = b.add_expr(CgExpr::Lit(LitValue::Bool(true))).unwrap();
        b.add_op(
            ComputeOpKind::MaskPredicate {
                mask: MaskId(mask),
                predicate: pred,
            },
            DispatchShape::PerAgent,
            Span::dummy(),
        )
        .unwrap()
    }

    #[test]
    fn consumer_after_producer_is_clean() {
        let mut b = CgProgramBuilder::new();
        let producer = add_mask(&mut b, 0);
        let consumer = add_rule(&mut b, Some(EventKindId(7)));
        let mut prog = b.finish();
        prog.interner.event_kinds.insert(7, "Supper".to_string());
        // Producer emits kind 7 into ring 0; consumer reads ring 0.
        let ring = EventRingId(0);
        prog.ops[producer.0 as usize].record_write(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Append,
        });
        prog.ops[consumer.0 as usize].record_read(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Read,
        });
        // MaskPredicate ops carry no emit set, so the kind filter treats
        // the consumer's kind as unmatched — assert on the ORDER check
        // using a producer that really emits (below). Here we only need
        // the "no producer" info issue to be the sole finding.
        let graph = dependency_graph(&prog);
        let issues = validate_ring_order(&prog, &graph, &[producer, consumer]);
        assert!(
            issues
                .iter()
                .all(|i| i.severity == RingOrderSeverity::Info),
            "{issues:?}"
        );
    }

    #[test]
    fn consumer_before_producer_without_a_cycle_is_a_bug() {
        // Two ops with a real ring edge, handed to the validator in the
        // WRONG order — exactly the shape the pre-fix scheduler
        // produced for webband_colony's supper gossip.
        let mut b = CgProgramBuilder::new();
        // Give the producer a real Emit of kind 7 so the kind-refined
        // edge forms.
        let emit = b
            .add_stmt(crate::cg::stmt::CgStmt::Emit {
                event: EventKindId(7),
                fields: Vec::new(),
            })
            .unwrap();
        let producer = add_rule_with_body(&mut b, None, vec![emit]);
        let consumer = add_rule(&mut b, Some(EventKindId(7)));
        let mut prog = b.finish();
        prog.interner.event_kinds.insert(7, "SupperTale".to_string());
        let ring = EventRingId(0);
        prog.ops[producer.0 as usize].record_write(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Append,
        });
        prog.ops[consumer.0 as usize].record_read(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Read,
        });

        let graph = dependency_graph(&prog);
        assert!(
            graph.edges.get(&producer).is_some_and(|s| s.contains(&consumer)),
            "the kind-refined ring edge must exist for this test to mean anything"
        );

        // Correct order: clean.
        let ok = validate_ring_order(&prog, &graph, &[producer, consumer]);
        assert!(
            !ok.iter().any(|i| matches!(
                i.kind,
                RingOrderIssueKind::ConsumerBeforeProducer { .. }
            )),
            "{ok:?}"
        );

        // Inverted order: one Bug-severity violation naming both ops.
        let bad = validate_ring_order(&prog, &graph, &[consumer, producer]);
        let v: Vec<_> = bad
            .iter()
            .filter(|i| {
                matches!(
                    i.kind,
                    RingOrderIssueKind::ConsumerBeforeProducer { cyclic: false, .. }
                )
            })
            .collect();
        assert_eq!(v.len(), 1, "{bad:?}");
        assert_eq!(v[0].severity, RingOrderSeverity::Bug);
        assert!(v[0].message.contains("SupperTale"), "{}", v[0].message);
        assert!(v[0].message.contains("scheduler bug"), "{}", v[0].message);
    }

    #[test]
    fn uninterned_subscription_kind_is_a_bug() {
        // The S5b signature: a consumer whose `on_event` id has no name.
        let mut b = CgProgramBuilder::new();
        let consumer = add_rule(&mut b, Some(EventKindId(49)));
        let mut prog = b.finish();
        prog.ops[consumer.0 as usize].record_read(DataHandle::EventRing {
            ring: EventRingId(0),
            kind: EventRingAccess::Read,
        });
        let graph = dependency_graph(&prog);
        let issues = validate_ring_order(&prog, &graph, &[consumer]);
        assert!(
            issues.iter().any(|i| matches!(
                i.kind,
                RingOrderIssueKind::ConsumerKindNotInterned { .. }
            ) && i.severity == RingOrderSeverity::Bug),
            "{issues:?}"
        );
    }

    #[test]
    fn non_ring_edges_are_ignored() {
        let mut b = CgProgramBuilder::new();
        let a = add_mask(&mut b, 0);
        let c = add_mask(&mut b, 1);
        let mut prog = b.finish();
        let hp = DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        };
        prog.ops[a.0 as usize].record_write(hp.clone());
        prog.ops[c.0 as usize].record_read(hp);
        let graph = dependency_graph(&prog);
        // Deliberately inverted — a field edge violation is NOT this
        // validator's business (fusion's own analysis owns those).
        let issues = validate_ring_order(&prog, &graph, &[c, a]);
        assert!(issues.is_empty(), "{issues:?}");
    }
}
