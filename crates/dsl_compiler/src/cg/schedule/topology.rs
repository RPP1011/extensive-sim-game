//! `DepGraph` + `topological_sort` — Phase-3 schedule synthesis primitives.
//!
//! Phase 2 produces a `CgProgram` whose every op carries auto-derived
//! reads + writes (with driver-injected source-ring reads and Emit
//! destination-ring writes folded in). Phase 3 needs to (a) discover
//! the precedence relation those reads/writes induce and (b) walk the
//! ops in an order that respects it.
//!
//! [`dependency_graph`] turns the op-level read/write metadata into a
//! [`DepGraph`] keyed by op-id pairs, with the `DataHandle`s that
//! caused each edge captured for diagnostics. [`topological_sort`]
//! returns a deterministic Kahn's-order linearization, surfacing any
//! cycle as a typed [`CycleError`] holding the offending SCCs.
//!
//! See `docs/superpowers/plans/2026-04-29-dsl-compute-graph-ir.md`,
//! Task 3.1, for the design rationale.
//!
//! # Limitations
//!
//! - **RAW edges only.** This first cut models read-after-write
//!   (`A writes X`, `B reads X` ⇒ `A → B`). Write-after-write (WAW)
//!   and write-after-read (WAR) hazards — which become relevant for
//!   fusion alias analysis (Task 3.2) and for serializing concurrent
//!   writers in megakernel synthesis (Task 3.3) — are deferred. The
//!   structure of [`DepGraph`] supports adding them without a breaking
//!   change to the public surface.
//! - **Cycles are allowed in the graph.** [`dependency_graph`] never
//!   fails — even when the input program contains structural cycles
//!   (e.g. the Pack/Unpack plumbing pair, which schedule synthesis
//!   resolves by sequencing across phase boundaries). Surface cycles
//!   by calling [`topological_sort`] and inspecting the
//!   [`CycleError::cycles`] payload.
//! - **Self-edges are skipped.** An op that reads a handle it also
//!   writes (the canonical event-fold pattern: read prior tick's
//!   storage, write next tick's) does NOT receive an `op → op`
//!   self-edge. This matches the `well_formed::detect_cycles`
//!   convention.
//! - **Driver-injected ring edges are honored verbatim.** Task 2.8
//!   wires source-ring reads on `PerEvent` dispatches and ring writes
//!   on `Emit` destinations directly onto each op's `reads`/`writes`.
//!   This pass treats those as it would any other handle: they
//!   participate in producer/consumer matching exactly when the driver
//!   inserted them, and not otherwise.
//! - **Edges use the [`crate::cg::data_handle::CycleEdgeKey`]
//!   projection.** EventRing producers (`Append`) match consumers
//!   (`Read` / `Drain`) on ring identity alone — the access kind is
//!   intentionally collapsed so dependency edges close across the
//!   read/append boundary.
//!
//! # The event-ring ordering invariant
//!
//! **No op that reads this tick's event ring is ever emitted before an
//! op that appends the kind it reads — unless the ring sub-relation is
//! itself cyclic, in which case the break is taken inside the cycle,
//! deterministically, and reported.**
//!
//! Ordinary Kahn pops trivially satisfy it (an op leaves the queue only
//! once every predecessor is emitted). The one way to violate it is
//! [`topological_sort_best_effort`]'s cycle-stall FORCED pick, which is
//! why that pick skips ring consumers with pending ring producers and
//! why its last-resort branch records a [`ForcedRingBreak`].
//!
//! The invariant is worth this much care because violating it is
//! **silent**: the consumer reads an unwritten ring, folds nothing, and
//! the feature it implements stops existing with no error anywhere.
//! Note the limit of what this file can promise — the guard can only
//! honour edges that EXIST, so a mis-keyed subscription (no edge at
//! all) walks straight past it. That door is closed by the schedule
//! validator in [`super::ring_order`], which checks the FINISHED order
//! against the program's own event facts on every emit.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt;

use crate::cg::data_handle::{CycleEdgeKey, DataHandle, EventRingAccess};
use crate::cg::op::{ComputeOpKind, EventKindId, OpId};
use crate::cg::program::CgProgram;
use crate::cg::stmt::{CgStmt, CgStmtListId};

// ---------------------------------------------------------------------------
// DepGraph
// ---------------------------------------------------------------------------

/// Read-after-write dependency graph for a [`CgProgram`].
///
/// One node per op (indices `0..op_count`). One directed edge
/// `producer → consumer` for every pair of ops where the producer
/// writes and the consumer reads the same projected handle. Self-edges
/// are filtered out (an op reading what it writes is the legitimate
/// event-fold pattern).
///
/// Both `edges` and `edge_reasons` use [`BTreeMap`] / [`BTreeSet`] so
/// iteration order is deterministic across runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DepGraph {
    /// Total number of ops the graph indexes. Equal to
    /// `prog.ops.len()` at the time the graph was built. Nodes are
    /// `OpId(0)..OpId(op_count as u32)`.
    pub op_count: usize,
    /// Out-edges per producer: `edges[a]` is the set of consumers `b`
    /// such that `a → b`. Empty entries (ops with no successors) are
    /// not stored.
    pub edges: BTreeMap<OpId, BTreeSet<OpId>>,
    /// For each `(producer, consumer)` edge, the projected handle keys
    /// that justify it. Sorted + deduplicated. A producer/consumer
    /// pair sharing two handles (e.g. Hp + ShieldHp) yields a
    /// two-element vector.
    pub edge_reasons: BTreeMap<(OpId, OpId), Vec<CycleEdgeKey>>,
}

impl DepGraph {
    /// Predecessors of `op` — ops whose writes feed into `op`'s reads,
    /// i.e. ops that must complete before `op` can run. Returned
    /// sorted by [`OpId`] for determinism.
    pub fn predecessors(&self, op: OpId) -> Vec<OpId> {
        let mut preds = Vec::new();
        for (producer, consumers) in &self.edges {
            if consumers.contains(&op) {
                preds.push(*producer);
            }
        }
        preds
    }

    /// Successors of `op` — ops that read what `op` writes, i.e. ops
    /// that must wait for `op`. Returned sorted by [`OpId`].
    pub fn successors(&self, op: OpId) -> Vec<OpId> {
        match self.edges.get(&op) {
            Some(succs) => succs.iter().copied().collect(),
            None => Vec::new(),
        }
    }

    /// Quick check: does the graph contain any cycle? Runs Tarjan's
    /// SCC and returns `true` on any non-trivial SCC.
    pub fn has_cycle(&self) -> bool {
        find_cycles(self).iter().any(|scc| scc.len() > 1)
    }

    /// Render the graph in a multi-line, human-readable form. One
    /// line per op listing its successors with the projected handle
    /// keys that justify each edge. Designed for logs and structured
    /// debugger output, not for round-tripping.
    pub fn display_for_debug(&self) -> String {
        let mut out = String::new();
        out.push_str("dep_graph {\n");
        out.push_str(&format!("    op_count: {},\n", self.op_count));
        if self.edges.is_empty() {
            out.push_str("    edges: [],\n");
        } else {
            out.push_str("    edges: [\n");
            for (producer, consumers) in &self.edges {
                for consumer in consumers {
                    out.push_str(&format!(
                        "        op#{} -> op#{}",
                        producer.0, consumer.0
                    ));
                    if let Some(reasons) = self.edge_reasons.get(&(*producer, *consumer)) {
                        out.push_str(" via [");
                        for (i, r) in reasons.iter().enumerate() {
                            if i > 0 {
                                out.push_str(", ");
                            }
                            out.push_str(&format_cycle_edge_key(r));
                        }
                        out.push(']');
                    }
                    out.push_str(",\n");
                }
            }
            out.push_str("    ],\n");
        }
        out.push('}');
        out
    }
}

impl fmt::Display for DepGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display_for_debug())
    }
}

/// Render a [`CycleEdgeKey`] in the same shape `DataHandle::Display`
/// uses for the wrapped variants — `Ring(#N)` for collapsed event-ring
/// keys, the inner handle's `Display` for `Other`. Kept private so the
/// shape stays consistent across `DepGraph::display_for_debug` and
/// `CycleError::Display`.
fn format_cycle_edge_key(key: &CycleEdgeKey) -> String {
    match key {
        CycleEdgeKey::Ring(ring) => format!("event_ring[#{}]", ring.0),
        CycleEdgeKey::Other(handle) => format!("{}", handle),
    }
}

// ---------------------------------------------------------------------------
// CycleError
// ---------------------------------------------------------------------------

/// Returned by [`topological_sort`] when the graph is not a DAG. Holds
/// every non-trivial strongly-connected component (size > 1) Tarjan's
/// algorithm finds. The vectors inside `cycles` are sorted by
/// [`OpId`] for deterministic output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleError {
    /// One SCC per detected cycle. Each inner [`Vec<OpId>`] is sorted
    /// by [`OpId`].
    pub cycles: Vec<Vec<OpId>>,
}

impl fmt::Display for CycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cycle in dep graph: ")?;
        if self.cycles.is_empty() {
            return f.write_str("[]");
        }
        for (i, scc) in self.cycles.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            f.write_str("[")?;
            for (j, op) in scc.iter().enumerate() {
                if j > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "op#{}", op.0)?;
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// dependency_graph
// ---------------------------------------------------------------------------

/// Build the [`DepGraph`] for a [`CgProgram`].
///
/// For every op `A` that writes a handle and op `B` that reads the
/// same projected handle (via [`crate::cg::data_handle::DataHandle::cycle_edge_key`]),
/// emit edge `A → B`. Self-edges are filtered out.
///
/// # Limitations
///
/// - Models RAW dependencies only. Write-after-write and
///   write-after-read are deferred — see the module-level
///   `# Limitations` block.
/// - Returns the graph unconditionally, even when it contains cycles.
///   Pass the graph to [`topological_sort`] to surface a [`CycleError`].
/// - Driver-injected reads/writes (Task 2.8 ring wiring) participate
///   in edge construction iff the driver placed them on the op.
/// - Edge keys use the `cycle_edge_key()` projection — EventRing
///   producers (`Append`) match consumers (`Read` / `Drain`) on ring
///   identity alone for non-EventRing handles. **EventRing edges are
///   refined by event KIND** (see below) to break the ring-as-cycle
///   collapse that ate every chronicle producer/consumer schedule.
///
/// # EventRing kind-aware refinement
///
/// The legacy ring-collapse projection ([`CycleEdgeKey::Ring`]) treated
/// every op touching a ring (Append OR Read) as a producer AND consumer
/// of the entire ring. With multiple chronicle stages on the same ring
/// (`Scoring → verb_chronicle_X → ApplyDamageFromChronicle → ApplyDamage`,
/// each stage Appending one kind and Reading another), every pair forms
/// a 2-op cycle in the dep graph, [`topological_sort_best_effort`] falls
/// back to OpId source order, and chronicle consumers run BEFORE their
/// in-tick GPU producers — so the consumer's `cfg.event_count = 0` snap
/// finds no matching kind-tag and silently no-ops. This was the
/// 6-fixture in-step GPU producer gap (hill_raid + 5 others).
///
/// The fix: refine EventRing edges by event KIND. For each op, walk its
/// body to collect the set of [`EventKindId`]s its emit statements
/// produce (`apply_ability` expands to the four engine effect kinds —
/// `EffectDamageApplied(26)`, `EffectHealApplied(27)`,
/// `EffectShieldApplied(28)`, `EffectStunApplied(29)`). Read [`OpId`]s
/// match producers via `producer.emits ⊇ {consumer.consumes}`. The
/// consumer's kind is the [`ComputeOpKind::PhysicsRule::on_event`] /
/// [`ComputeOpKind::ViewFold::on_event`] field. Producer/consumer pairs
/// that don't share a kind get NO EventRing edge — the
/// `verb_chronicle_Strike` Append of kind 26 and the `ApplyDamage` Read
/// of kind 1 (Damaged) used to form spurious cross-edges, but now
/// only `verb_chronicle_Strike → ApplyDamageFromChronicle` (kind 26
/// match) and `ApplyDamageFromChronicle → ApplyDamage` (kind 1 match)
/// fire — a clean DAG, no cycle, correct order.
pub fn dependency_graph(prog: &CgProgram) -> DepGraph {
    let mut edges: BTreeMap<OpId, BTreeSet<OpId>> = BTreeMap::new();
    let mut edge_reasons: BTreeMap<(OpId, OpId), Vec<CycleEdgeKey>> = BTreeMap::new();

    // Pre-compute per-op (emitted_kinds, consumed_kind) for the
    // EventRing kind-aware refinement (see doc comment above). Empty
    // sets / `None` for ops that don't touch the chronicle ring.
    let kind_facts: Vec<EventRingKindFacts> = prog
        .ops
        .iter()
        .map(|op| compute_event_ring_kind_facts(op, prog))
        .collect();

    // Index "what does each handle's writers list look like" first.
    // Same projection (`cycle_edge_key`) used by the consumer-side scan
    // so EventRing append/read/drain accesses match on ring identity.
    // EventRing entries get a SECOND filter via `kind_facts` below so
    // we don't over-edge across unrelated kinds on the same ring.
    let mut writers: BTreeMap<CycleEdgeKey, Vec<OpId>> = BTreeMap::new();
    for (op_index, op) in prog.ops.iter().enumerate() {
        let producer = OpId(op_index as u32);
        for w in &op.writes {
            writers.entry(w.cycle_edge_key()).or_default().push(producer);
        }
    }

    // Walk every consumer's reads, look up writers of the same
    // projected handle, add edges (skipping self-edges per the
    // event-fold convention).
    //
    // Reads INTO a SpatialQuery op are also skipped: the per-tick
    // BuildHash kernel reads the prior-tick agent positions that
    // every per-agent rule overwrote at the end of last tick. The
    // edge from "writer of agent_pos" → "BuildHash" is a cross-tick
    // edge (BuildHash sees prior-tick state), not a same-tick
    // dependency. Without this skip, any per-agent rule that writes
    // pos AND uses the spatial grid forms a 2-op cycle in the dep
    // graph; topo falls back to source order and fusion misorders
    // BuildHash relative to its consumers.
    //
    // Edges FROM `Plumbing { kind: UnpackAgents }` are similarly
    // skipped. UnpackAgents writes the entire agent SoA at the END of
    // each tick (re-hydrating from the snapshot's packed buffer), but
    // user ops next tick read those writes — a cross-tick dependency.
    // Treating it as same-tick would form a `user_op → PackAgents →
    // UnpackAgents → user_op` cycle through every `agent.self.<field>`
    // (Pack reads what user ops wrote; Unpack writes back the same
    // fields user ops read). With the cycle present, `topological_sort`
    // falls back to source order and fusion places the spatial-build
    // phases AFTER their PerAgent consumer kernels (because the
    // spatial-build ops are synthesised after user ops and therefore
    // have higher OpIds), which silently no-ops every
    // `for x in spatial.nearby(self) { ... }` walk on the first tick.
    // Skipping outgoing edges from UnpackAgents breaks the cycle
    // without affecting the remaining legitimate edges
    // (`user_op → PackAgents`, `PackAgents → UnpackAgents`).
    for (op_index, op) in prog.ops.iter().enumerate() {
        let consumer = OpId(op_index as u32);
        let consumer_is_spatial_build = matches!(
            op.kind,
            crate::cg::op::ComputeOpKind::SpatialQuery { .. }
        );
        let consumer_facts = &kind_facts[op_index];
        for r in &op.reads {
            let key = r.cycle_edge_key();
            // EventRing-shaped read: refine producer matching by event
            // KIND. The consumer's kind comes from the op-level
            // `on_event` (PhysicsRule / ViewFold). Producers must
            // declare the same kind in their emit set (which for
            // ApplyAbility expands to the four engine effect kinds).
            //
            // **Why we still consult `writers[key]` first.** EventRing
            // edges share the same `Ring(EventRingId)` projection, so
            // `writers[Ring(0)]` is the union of every Append onto that
            // ring. The kind filter then prunes non-matching pairs
            // pairwise — this is what breaks the chronicle-stage cycle.
            //
            // **Drain reads are NOT kind-filtered.** A `DrainEvents`
            // plumbing op consumes EVERY event regardless of kind; treat
            // it as a wildcard consumer (matches every producer on the
            // ring) so view-fold pipelines that read-then-drain still
            // see their producer edges.
            let is_event_ring_read = matches!(
                r,
                DataHandle::EventRing { kind: EventRingAccess::Read, .. }
            );
            let is_event_ring_drain = matches!(
                r,
                DataHandle::EventRing { kind: EventRingAccess::Drain, .. }
            );
            if let Some(producers) = writers.get(&key) {
                for &producer in producers {
                    if producer == consumer {
                        continue;
                    }
                    if consumer_is_spatial_build {
                        // Cross-tick read (see comment above).
                        continue;
                    }
                    let producer_is_unpack_agents = matches!(
                        prog.ops.get(producer.0 as usize).map(|p| &p.kind),
                        Some(crate::cg::op::ComputeOpKind::Plumbing {
                            kind: crate::cg::op::PlumbingKind::UnpackAgents,
                        })
                    );
                    if producer_is_unpack_agents {
                        // Cross-tick write (see comment above).
                        continue;
                    }
                    // EventRing kind-aware filter. See doc comment on
                    // `dependency_graph`. Drain reads bypass the filter
                    // (wildcard consumer of all kinds on the ring).
                    if is_event_ring_read && !is_event_ring_drain {
                        let producer_facts = &kind_facts[producer.0 as usize];
                        let kinds_overlap = match consumer_facts.consumed {
                            Some(ck) => producer_facts.emitted.contains(&ck),
                            // Consumer has no `on_event` (shouldn't
                            // happen for PerEvent ops in practice — the
                            // driver wires source-ring reads only on
                            // PerEvent — but be defensive: treat as
                            // wildcard so we don't drop a legitimate
                            // edge a future op kind introduces).
                            None => true,
                        };
                        if !kinds_overlap {
                            continue;
                        }
                    }
                    edges.entry(producer).or_default().insert(consumer);
                    edge_reasons
                        .entry((producer, consumer))
                        .or_default()
                        .push(key.clone());
                }
            }
        }
    }

    // Reasons may collect duplicates when a producer/consumer pair
    // shares the same handle through both reads and writes; sort +
    // dedup for deterministic output.
    for v in edge_reasons.values_mut() {
        v.sort();
        v.dedup();
    }

    DepGraph {
        op_count: prog.ops.len(),
        edges,
        edge_reasons,
    }
}

// ---------------------------------------------------------------------------
// EventRing kind-aware refinement helpers
// ---------------------------------------------------------------------------

/// What event kinds an op emits to / consumes from the chronicle ring,
/// gleaned from its op-kind metadata + body walk. Used by
/// [`dependency_graph`] to refine EventRing edges by kind so chronicle
/// producer/consumer chains form a DAG instead of a 2-op cycle.
///
/// `emitted` is the union of every `CgStmt::Emit { event }` and every
/// `CgStmt::ApplyAbility` (which expands to the four engine effect
/// kinds the apply_ability dispatcher writes — see
/// [`apply_ability_emitted_kinds`]). `consumed` is the op's
/// `on_event` field (if any).
///
/// Empty / `None` for ops that don't subscribe to or emit events.
pub(super) struct EventRingKindFacts {
    /// Set of [`EventKindId`]s this op writes to the chronicle ring,
    /// either via direct `CgStmt::Emit` statements or via the
    /// `apply_ability` dispatcher's effect-event emits.
    pub(super) emitted: BTreeSet<EventKindId>,
    /// The single [`EventKindId`] this op subscribes to (from
    /// `PhysicsRule.on_event` / `ViewFold.on_event`). `None` for ops
    /// that don't read the ring or that have no kind-tagged
    /// subscription (e.g. per-agent rules whose `on_event` lowers to
    /// `None`, plumbing ops, mask predicates).
    pub(super) consumed: Option<EventKindId>,
}

/// Engine event kinds the `apply_ability` dispatcher emits per
/// non-EMPTY effect slot. Sourced from
/// `crate::cg::emit::wgsl_body::EFFECT_KIND_TO_EVENT_KIND_ID` — the
/// authoritative `EffectOp` ordinal → `EventKindId` mapping that the
/// dispatcher's WGSL arm chain renders against. Any new EffectOp /
/// chronicle event added to that table is automatically picked up
/// here.
///
/// The exact subset depends on the ability program's per-effect
/// `kind` array, which is data the registry carries — not visible at
/// schedule synthesis time. So we conservatively claim every chronicle
/// event the dispatcher CAN emit; this over-estimates producer→consumer
/// matches but only ever ADDS edges (never spuriously suppresses them),
/// and a consumer that reads kind X will correctly find every
/// `apply_ability` op as a producer regardless of the underlying
/// ability's actual effect mix.
///
/// **Gap dungeon_stealth#5 (2026-05-12).** Previously this list was
/// hardcoded to `[26, 27, 28, 29]`, missing every extended-corpus kind
/// (Stealth=54, Charm=55, …). The `ApplyStealthFromChronicle` consumer
/// rule's kind=54 read therefore had no matching producer in the
/// dep-graph, so Kahn's topo sort placed it BEFORE `RogueStealth` (the
/// dispatcher) in the schedule. Each tick: the consumer scanned an
/// empty ring, then the dispatcher emitted records that wouldn't be
/// drained until the next tick's consumer pass — but the ring is reset
/// per tick, so the records were silently dropped, leaving
/// `stealth_until_tick` at 0 forever.
fn apply_ability_emitted_kinds() -> Vec<EventKindId> {
    crate::cg::emit::wgsl_body::EFFECT_KIND_TO_EVENT_KIND_ID
        .iter()
        .map(|(_effect_kind, event_kind)| EventKindId(*event_kind))
        .collect()
}

pub(super) fn compute_event_ring_kind_facts(
    op: &crate::cg::op::ComputeOp,
    prog: &CgProgram,
) -> EventRingKindFacts {
    let mut emitted: BTreeSet<EventKindId> = BTreeSet::new();
    let consumed: Option<EventKindId> = match &op.kind {
        ComputeOpKind::PhysicsRule { on_event, body, .. } => {
            collect_emit_kinds_in_list(*body, prog, &mut emitted);
            *on_event
        }
        ComputeOpKind::ViewFold { on_event, body, .. } => {
            collect_emit_kinds_in_list(*body, prog, &mut emitted);
            Some(*on_event)
        }
        // Mask / scoring / spatial / plumbing / decay don't carry an
        // `on_event` and don't emit chronicle records via CgStmt::Emit
        // (the scoring kernel's ActionSelected emit is inlined at the
        // emit layer rather than via CgStmt::Emit, but it does appear
        // in the writes list as Append on the ring — the kind-aware
        // filter ALWAYS lets edges fire when the consumer reads with
        // no specific kind subscription via the `consumed.is_none()`
        // wildcard branch). For a producer whose emitted set we
        // can't enumerate at this layer (e.g. scoring), we treat it
        // as a wildcard producer (`emit ALL` — matches every kind).
        ComputeOpKind::ScoringArgmax { .. } => {
            // The scoring kernel emits ActionSelected directly in the
            // emitted WGSL — claim it here so verb_chronicle ops
            // (which subscribe via `on_event = ActionSelected`) match
            // their producer correctly. ActionSelected lives in the
            // event-kind interner; resolve by name.
            for (id, name) in &prog.interner.event_kinds {
                if name
                    == crate::cg::lower::verb_expand::ACTION_SELECTED_EVENT_NAME
                {
                    emitted.insert(EventKindId(*id));
                    break;
                }
            }
            None
        }
        ComputeOpKind::MaskPredicate { .. }
        | ComputeOpKind::SpatialQuery { .. }
        | ComputeOpKind::Plumbing { .. }
        | ComputeOpKind::ViewDecay { .. } => None,
        // BeliefSocialMerge consumes an event kind (the merge fires
        // on the named event). Surface it like ViewFold's on_event
        // so the scheduler can serialize against the producer.
        ComputeOpKind::BeliefSocialMerge { on_event, .. } => Some(*on_event),
    };
    EventRingKindFacts { emitted, consumed }
}

/// Recursively walk a [`CgStmtListId`] and collect every emitted
/// [`EventKindId`] — both direct `CgStmt::Emit { event }` statements
/// and `CgStmt::ApplyAbility` dispatcher calls (which expand to every
/// chronicle event kind the dispatcher CAN emit — see
/// [`apply_ability_emitted_kinds`]).
///
/// Mirrors the shape of `crate::cg::lower::driver::collect_emits_in_list`
/// but lives here so the schedule-layer dependency analysis doesn't
/// need to reach across the module boundary. The two walkers SHOULD
/// stay structurally aligned — a future `CgStmt` variant that
/// introduces an emit-bearing body must be handled in BOTH.
fn collect_emit_kinds_in_list(
    list_id: CgStmtListId,
    prog: &CgProgram,
    out: &mut BTreeSet<EventKindId>,
) {
    let Some(list) = prog.stmt_lists.get(list_id.0 as usize) else {
        return;
    };
    for &stmt_id in &list.stmts {
        let Some(stmt) = prog.stmts.get(stmt_id.0 as usize) else {
            continue;
        };
        match stmt {
            CgStmt::Emit { event, .. } => {
                out.insert(*event);
            }
            CgStmt::ApplyAbility { .. } => {
                for k in apply_ability_emitted_kinds() {
                    out.insert(k);
                }
            }
            CgStmt::If { then, else_, .. } => {
                collect_emit_kinds_in_list(*then, prog, out);
                if let Some(else_list) = else_ {
                    collect_emit_kinds_in_list(*else_list, prog, out);
                }
            }
            CgStmt::Match { arms, .. } => {
                for arm in arms {
                    collect_emit_kinds_in_list(arm.body, prog, out);
                }
            }
            CgStmt::ForEachNeighborBody { body, .. } => {
                collect_emit_kinds_in_list(*body, prog, out);
            }
            CgStmt::ForEachAgentBody { body, .. } => {
                collect_emit_kinds_in_list(*body, prog, out);
            }
            CgStmt::Assign { .. }
            | CgStmt::Let { .. }
            | CgStmt::ForEachAgent { .. }
            | CgStmt::ForEachNeighbor { .. }
            | CgStmt::ViewStorageAppend { .. } => {
                // No emit-bearing payload.
            }
        }
    }
}

// ---------------------------------------------------------------------------
// topological_sort
// ---------------------------------------------------------------------------

/// Linearize a [`DepGraph`] using Kahn's algorithm. Ties between
/// available ops are broken by [`OpId`] (smallest first) so the order
/// is deterministic across runs.
///
/// Returns `Ok(order)` when the graph is a DAG; `order.len() ==
/// graph.op_count`. Returns `Err(CycleError { cycles })` otherwise,
/// with `cycles` populated by Tarjan's SCC over the same graph.
///
/// # Limitations
///
/// - Surfaces cycles as `Err`; does not attempt to resolve them. Phase 3
///   schedule strategies decide what to do with a cyclic graph (e.g.
///   the megakernel synthesis sequences Pack/Unpack across phase
///   boundaries).
/// - Tie-breaking is by [`OpId`] only. There is no priority hint
///   today; future passes that want to bias toward a specific op (e.g.
///   place producers as late as possible to minimize live state) will
///   add a separate scheduler that consults [`DepGraph`] directly.
pub fn topological_sort(graph: &DepGraph) -> Result<Vec<OpId>, CycleError> {
    let n = graph.op_count;

    // In-degree per node. We size for `n` nodes; ops with no
    // predecessors get `0`.
    let mut in_degree: Vec<u32> = vec![0; n];
    for succs in graph.edges.values() {
        for s in succs {
            // Defensive — `op_count` is built from `prog.ops.len()`
            // and edges only ever reference in-range OpIds, but never
            // panic if a malformed graph slips through.
            let idx = s.0 as usize;
            if idx < n {
                in_degree[idx] += 1;
            }
        }
    }

    // Min-heap on `Reverse(OpId)` — Kahn's with deterministic
    // tie-breaking.
    let mut queue: BinaryHeap<Reverse<OpId>> = BinaryHeap::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push(Reverse(OpId(i as u32)));
        }
    }

    let mut order: Vec<OpId> = Vec::with_capacity(n);
    while let Some(Reverse(op)) = queue.pop() {
        order.push(op);
        // Walk only this node's successors. Avoid `graph.successors`
        // (allocates) — read the BTreeSet directly.
        if let Some(succs) = graph.edges.get(&op) {
            for &succ in succs {
                let idx = succ.0 as usize;
                if idx < n {
                    in_degree[idx] -= 1;
                    if in_degree[idx] == 0 {
                        queue.push(Reverse(succ));
                    }
                }
            }
        }
    }

    if order.len() == n {
        Ok(order)
    } else {
        // Some nodes never reached zero in-degree: the residual
        // subgraph contains at least one cycle. Find the SCCs to
        // report.
        let mut sccs = find_cycles(graph);
        sccs.retain(|s| s.len() > 1);
        for s in &mut sccs {
            s.sort_by_key(|o| o.0);
        }
        sccs.sort();
        Err(CycleError { cycles: sccs })
    }
}

/// One place the best-effort sort had to schedule an event-ring
/// CONSUMER ahead of a still-unemitted event-ring PRODUCER of the same
/// kind, because every remaining op was in that position (a genuine
/// ring cycle). Never produced on an acyclic-in-the-ring graph — the
/// forced pick skips ring consumers with pending producers.
///
/// Surfaced so callers can turn the break into a LOUD diagnostic: a
/// same-tick chronicle read that runs before its writer silently reads
/// an empty ring, which is the single most expensive failure mode this
/// scheduler has shipped (see the `topological_sort_best_effort` doc
/// comment).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ForcedRingBreak {
    /// The ring consumer that was forced into the order early.
    pub consumer: OpId,
    /// Ring producers of `consumer` that were still unemitted at the
    /// moment of the force. Sorted by [`OpId`].
    pub pending_producers: Vec<OpId>,
    /// `true` when at least one pending producer sits in the same
    /// strongly-connected component as the consumer — i.e. the break
    /// was genuinely unavoidable. `false` means the sort ran out of
    /// legal candidates for a reason the analysis could not attribute
    /// to a cycle; that is a compiler bug and should be reported as
    /// such.
    pub cyclic: bool,
}

/// Best-effort variant of [`topological_sort`]. Always returns an order
/// of length `graph.op_count`; reports any cycles encountered as a
/// secondary signal.
///
/// Algorithm: Kahn's with the same deterministic OpId tie-break as
/// [`topological_sort`]. When the queue empties before all ops are
/// emitted, force the smallest-OpId remaining op (in source order)
/// into the order — this breaks cycles by preferring the user's source
/// order within each SCC, then continues Kahn's normally. Iterates
/// until every op is emitted.
///
/// **Why fusion needs this.** [`topological_sort`] returns `Err` on any
/// cycle, even small ones (e.g. two adjacent PerAgent rules whose
/// write/read sets cross — `LifecycleAge` writes hunger, `Reaper`
/// writes alive; each reads the other's write target). Fusion's
/// previous fallback was *full* source order — discarding every edge,
/// including the well-formed `SpatialBuildHashScatter → user_op` edge
/// the ordering depends on. Best-effort preserves all edges that don't
/// participate in the cycle and only forces source order at the SCC
/// boundary.
///
/// Returns `(order, cycles)`. `cycles` is `Some` when at least one
/// non-trivial SCC was found; the caller can surface a diagnostic
/// without re-running Tarjan's.
///
/// Callers that need to know whether the sort had to sacrifice a
/// same-tick ring ordering to break a cycle should use
/// [`topological_sort_best_effort_reporting`] — this wrapper drops that
/// third channel.
pub fn topological_sort_best_effort(graph: &DepGraph) -> (Vec<OpId>, Option<CycleError>) {
    let (order, cycles, _breaks) = topological_sort_best_effort_reporting(graph);
    (order, cycles)
}

/// [`topological_sort_best_effort`] plus the forced-ring-break report.
///
/// The third element is empty for every graph whose event-ring
/// producer/consumer sub-relation is acyclic — which is every shipped
/// fixture. A non-empty vector means the schedule contains a same-tick
/// chronicle read placed before its writer; see [`ForcedRingBreak`].
pub fn topological_sort_best_effort_reporting(
    graph: &DepGraph,
) -> (Vec<OpId>, Option<CycleError>, Vec<ForcedRingBreak>) {
    let n = graph.op_count;

    let mut in_degree: Vec<u32> = vec![0; n];
    for succs in graph.edges.values() {
        for s in succs {
            let idx = s.0 as usize;
            if idx < n {
                in_degree[idx] += 1;
            }
        }
    }

    let mut queue: BinaryHeap<Reverse<OpId>> = BinaryHeap::new();
    let mut emitted: Vec<bool> = vec![false; n];
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push(Reverse(OpId(i as u32)));
        }
    }

    let mut order: Vec<OpId> = Vec::with_capacity(n);
    let mut cycles_observed = false;
    // Lazily-computed per-op list of EVENT-RING predecessors — only
    // materialized on the first stall (cycle-free graphs never pay).
    let mut ring_pred: Option<Vec<Vec<u32>>> = None;
    // Lazily-computed SCC id per op — only materialized if the
    // ring-consumer guard ever runs out of legal candidates (i.e. a
    // real ring cycle). Cycle-free-in-the-ring graphs never pay.
    let mut scc_of: Option<Vec<usize>> = None;
    let mut ring_breaks: Vec<ForcedRingBreak> = Vec::new();

    while order.len() < n {
        // Drain the Kahn's-ready queue first.
        while let Some(Reverse(op)) = queue.pop() {
            if emitted[op.0 as usize] {
                // The forced-emit branch (below) may have already
                // popped this op out of the residual graph; skip
                // duplicates so the final order has length == n.
                continue;
            }
            emitted[op.0 as usize] = true;
            order.push(op);
            if let Some(succs) = graph.edges.get(&op) {
                for &succ in succs {
                    let idx = succ.0 as usize;
                    if idx < n && !emitted[idx] {
                        in_degree[idx] -= 1;
                        if in_degree[idx] == 0 {
                            queue.push(Reverse(succ));
                        }
                    }
                }
            }
        }

        if order.len() == n {
            break;
        }

        // No op has in_degree == 0 but ops remain → cycle. Force the
        // smallest-OpId not-yet-emitted op into the order (preserves
        // the user's source order within the SCC) and continue
        // Kahn's — EXCEPT ops that consume the EVENT RING and still
        // have an unemitted ring PRODUCER: those are skipped by the
        // forced pick.
        //
        // The ring-consumer guard is load-bearing (found 2026-07-22,
        // S5 webband_colony): the queue stalls whenever every
        // remaining op sits at or downstream of a cycle, and the
        // remaining set then contains INNOCENT ring consumers whose
        // producers are merely stuck behind the SCC. The old
        // unconditional global-smallest pick emitted webband_colony's
        // BeliefSocialMerge op 65 stages before the physics rule that
        // emits its trigger event — the same-tick live-tail merge
        // then read a ring with no SupperTale rows and supper gossip
        // went silently dead. Same-tick event flow is the one
        // ordering that can never be sacrificed to a cycle break.
        // The guard is deliberately NARROW (ring edges only): field-
        // edge order inside SCCs keeps the historic global-smallest
        // pick, because shipped fixtures' pins are calibrated to it
        // (edgeworld's 9-op hunger SCC regressed under a broader
        // force-only-cycle-members rule and was reverted).
        cycles_observed = true;
        if ring_pred.is_none() {
            // predecessor lists restricted to edges that carry an
            // EventRing reason (producer -> ring-consumer edges).
            let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
            for ((p, c), reasons) in graph.edge_reasons.iter() {
                if reasons.iter().any(|r| matches!(r, CycleEdgeKey::Ring(_))) {
                    let ci = c.0 as usize;
                    if ci < n {
                        preds[ci].push(p.0);
                    }
                }
            }
            ring_pred = Some(preds);
        }
        let preds = ring_pred.as_ref().expect("just populated");
        let legal = (0..n).find(|&i| {
            !emitted[i] && !preds[i].iter().any(|&p| !emitted[p as usize])
        });
        let forced = match legal {
            Some(i) => i,
            None => {
                // Every remaining op is a ring consumer whose producer
                // is also still pending: the event-ring sub-relation
                // itself is cyclic and SOMETHING has to give. Break it
                // deterministically and INSIDE the cycle — prefer the
                // smallest remaining op that shares an SCC with one of
                // its own pending ring producers, so an innocent
                // downstream consumer is never the one sacrificed. The
                // break is recorded in `ring_breaks` and surfaces as a
                // loud schedule diagnostic (see `ForcedRingBreak`).
                //
                // This branch is unreachable for every fixture in the
                // corpus (proved by
                // `ring_order::validate_ring_order` running on every
                // emit); it exists so a future genuinely-cyclic
                // fixture degrades loudly and reproducibly instead of
                // silently.
                if scc_of.is_none() {
                    let mut ids = vec![usize::MAX; n];
                    for (sid, scc) in find_cycles(graph).iter().enumerate() {
                        for op in scc {
                            let i = op.0 as usize;
                            if i < n {
                                ids[i] = sid;
                            }
                        }
                    }
                    scc_of = Some(ids);
                }
                let sccs = scc_of.as_ref().expect("just populated");
                let in_cycle_with_producer = (0..n).find(|&i| {
                    !emitted[i]
                        && preds[i].iter().any(|&p| {
                            !emitted[p as usize]
                                && sccs[i] != usize::MAX
                                && sccs[i] == sccs[p as usize]
                        })
                });
                let pick = in_cycle_with_producer.unwrap_or_else(|| {
                    (0..n)
                        .find(|&i| !emitted[i])
                        .expect("len < n implies at least one un-emitted op")
                });
                let sccs_ok = sccs[pick] != usize::MAX;
                let mut pending: Vec<OpId> = preds[pick]
                    .iter()
                    .filter(|&&p| !emitted[p as usize])
                    .map(|&p| OpId(p))
                    .collect();
                pending.sort();
                pending.dedup();
                let cyclic = sccs_ok
                    && pending
                        .iter()
                        .any(|p| sccs[p.0 as usize] == sccs[pick]);
                ring_breaks.push(ForcedRingBreak {
                    consumer: OpId(pick as u32),
                    pending_producers: pending,
                    cyclic,
                });
                pick
            }
        };
        let forced_op = OpId(forced as u32);
        emitted[forced] = true;
        order.push(forced_op);
        if let Some(succs) = graph.edges.get(&forced_op) {
            for &succ in succs {
                let idx = succ.0 as usize;
                if idx < n && !emitted[idx] {
                    in_degree[idx] = in_degree[idx].saturating_sub(1);
                    if in_degree[idx] == 0 {
                        queue.push(Reverse(succ));
                    }
                }
            }
        }
    }

    let cycles = if cycles_observed {
        let mut sccs = find_cycles(graph);
        sccs.retain(|s| s.len() > 1);
        for s in &mut sccs {
            s.sort_by_key(|o| o.0);
        }
        sccs.sort();
        if sccs.is_empty() {
            None
        } else {
            Some(CycleError { cycles: sccs })
        }
    } else {
        None
    };

    (order, cycles, ring_breaks)
}

// ---------------------------------------------------------------------------
// find_cycles — Tarjan's SCC over a DepGraph
// ---------------------------------------------------------------------------

/// Run Tarjan's strongly-connected-components algorithm over `graph`
/// and return every SCC (including trivial size-1 ones). Iterative
/// implementation so deep graphs don't blow the stack.
///
/// Lifted from `well_formed::tarjan_scc` and adapted to consume a
/// [`DepGraph`] directly. The two implementations stay in sync because
/// they share the algorithm; promotion to a single shared helper is a
/// Phase-3 cleanup deferred until Task 3.2 also needs it.
pub(super) fn find_cycles(graph: &DepGraph) -> Vec<Vec<OpId>> {
    let n = graph.op_count;
    if n == 0 {
        return Vec::new();
    }

    // Materialize adjacency into Vec<Vec<usize>> for fast indexed
    // iteration. Edges in `DepGraph` are sorted (BTreeSet), so the
    // resulting traversal is deterministic.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (producer, consumers) in &graph.edges {
        let p_idx = producer.0 as usize;
        if p_idx >= n {
            continue;
        }
        for c in consumers {
            let c_idx = c.0 as usize;
            if c_idx < n {
                adj[p_idx].push(c_idx);
            }
        }
    }

    let mut indices: Vec<i64> = vec![-1; n];
    let mut lowlinks: Vec<i64> = vec![0; n];
    let mut on_stack: Vec<bool> = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut sccs: Vec<Vec<OpId>> = Vec::new();
    let mut index_counter: i64 = 0;

    struct Frame {
        node: usize,
        edges: Vec<usize>,
        next_edge: usize,
    }

    for start in 0..n {
        if indices[start] != -1 {
            continue;
        }

        let mut call_stack: Vec<Frame> = Vec::new();
        indices[start] = index_counter;
        lowlinks[start] = index_counter;
        index_counter += 1;
        stack.push(start);
        on_stack[start] = true;
        call_stack.push(Frame {
            node: start,
            edges: adj[start].clone(),
            next_edge: 0,
        });

        while let Some(frame) = call_stack.last_mut() {
            if frame.next_edge < frame.edges.len() {
                let w = frame.edges[frame.next_edge];
                frame.next_edge += 1;
                if w >= n {
                    continue;
                }
                if indices[w] == -1 {
                    indices[w] = index_counter;
                    lowlinks[w] = index_counter;
                    index_counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    let edges_w = adj[w].clone();
                    call_stack.push(Frame {
                        node: w,
                        edges: edges_w,
                        next_edge: 0,
                    });
                    continue;
                } else if on_stack[w] && indices[w] < lowlinks[frame.node] {
                    lowlinks[frame.node] = indices[w];
                }
            } else {
                let v = frame.node;
                if lowlinks[v] == indices[v] {
                    let mut scc = Vec::new();
                    while let Some(w) = stack.pop() {
                        on_stack[w] = false;
                        scc.push(OpId(w as u32));
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(scc);
                }
                call_stack.pop();
                if let Some(parent) = call_stack.last_mut() {
                    if lowlinks[v] < lowlinks[parent.node] {
                        lowlinks[parent.node] = lowlinks[v];
                    }
                }
            }
        }
    }

    sccs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cg::data_handle::{
        AgentFieldId, AgentRef, DataHandle, EventRingAccess, EventRingId, MaskId,
    };
    use crate::cg::dispatch::DispatchShape;
    use crate::cg::expr::{CgExpr, LitValue};
    use crate::cg::op::{ComputeOpKind, OpId, Span};
    use crate::cg::program::{CgProgram, CgProgramBuilder};

    // --- helpers -------------------------------------------------------

    /// Build a no-op `MaskPredicate` op with no auto-derived
    /// reads/writes (the predicate expression is a literal `true`).
    /// Returns the [`OpId`]; tests inject reads/writes via
    /// `record_read` / `record_write`.
    fn add_blank_mask_op(builder: &mut CgProgramBuilder, mask: MaskId) -> OpId {
        let pred = builder.add_expr(CgExpr::Lit(LitValue::Bool(true))).unwrap();
        builder
            .add_op(
                ComputeOpKind::MaskPredicate {
                    mask,
                    predicate: pred,
                },
                DispatchShape::PerAgent,
                Span::dummy(),
            )
            .unwrap()
    }

    /// Convenience: Hp on `self`.
    fn hp_handle() -> DataHandle {
        DataHandle::AgentField {
            field: AgentFieldId::Hp,
            target: AgentRef::Self_,
        }
    }

    /// Convenience: ShieldHp on `self`.
    fn shield_handle() -> DataHandle {
        DataHandle::AgentField {
            field: AgentFieldId::ShieldHp,
            target: AgentRef::Self_,
        }
    }

    /// Convenience: Mana on `self`.
    fn mana_handle() -> DataHandle {
        DataHandle::AgentField {
            field: AgentFieldId::Mana,
            target: AgentRef::Self_,
        }
    }

    /// Build a program with `count` blank mask-predicate ops, returning
    /// the builder so the caller can inject reads/writes.
    fn program_with_blank_ops(count: u32) -> (CgProgram, Vec<OpId>) {
        let mut b = CgProgramBuilder::new();
        let mut ids = Vec::with_capacity(count as usize);
        for i in 0..count {
            ids.push(add_blank_mask_op(&mut b, MaskId(i)));
        }
        (b.finish(), ids)
    }

    // --- 1. Empty program ----------------------------------------------

    #[test]
    fn empty_program_has_empty_graph_and_topo_order() {
        let prog = CgProgram::new();
        let graph = dependency_graph(&prog);
        assert_eq!(graph.op_count, 0);
        assert!(graph.edges.is_empty());
        assert!(graph.edge_reasons.is_empty());
        assert!(!graph.has_cycle());
        assert_eq!(topological_sort(&graph), Ok(Vec::<OpId>::new()));
    }

    // --- 2. Linear chain -----------------------------------------------

    #[test]
    fn linear_chain_op0_writes_op1_reads_yields_single_edge() {
        let (mut prog, ids) = program_with_blank_ops(2);
        // Op0 writes Hp; Op1 reads Hp.
        prog.ops[ids[0].0 as usize].record_write(hp_handle());
        prog.ops[ids[1].0 as usize].record_read(hp_handle());

        let graph = dependency_graph(&prog);
        assert_eq!(graph.op_count, 2);
        assert_eq!(graph.successors(ids[0]), vec![ids[1]]);
        assert_eq!(graph.predecessors(ids[1]), vec![ids[0]]);
        assert!(graph.successors(ids[1]).is_empty());
        assert!(graph.predecessors(ids[0]).is_empty());

        // Edge reason captures the projected handle.
        let reasons = graph.edge_reasons.get(&(ids[0], ids[1])).unwrap();
        assert_eq!(reasons.len(), 1);
        assert_eq!(reasons[0], hp_handle().cycle_edge_key());

        assert_eq!(topological_sort(&graph), Ok(vec![ids[0], ids[1]]));
    }

    // --- 3. Diamond ----------------------------------------------------

    #[test]
    fn diamond_dependency_topologically_orders_root_before_sinks() {
        // Op0: writes Hp.
        // Op1: reads Hp, writes ShieldHp.
        // Op2: reads Hp, writes Mana.
        // Op3: reads ShieldHp + Mana.
        let (mut prog, ids) = program_with_blank_ops(4);
        prog.ops[ids[0].0 as usize].record_write(hp_handle());

        prog.ops[ids[1].0 as usize].record_read(hp_handle());
        prog.ops[ids[1].0 as usize].record_write(shield_handle());

        prog.ops[ids[2].0 as usize].record_read(hp_handle());
        prog.ops[ids[2].0 as usize].record_write(mana_handle());

        prog.ops[ids[3].0 as usize].record_read(shield_handle());
        prog.ops[ids[3].0 as usize].record_read(mana_handle());

        let graph = dependency_graph(&prog);

        // Edges: 0→1, 0→2, 1→3, 2→3.
        assert_eq!(graph.successors(ids[0]), vec![ids[1], ids[2]]);
        assert_eq!(graph.successors(ids[1]), vec![ids[3]]);
        assert_eq!(graph.successors(ids[2]), vec![ids[3]]);
        assert!(graph.successors(ids[3]).is_empty());

        // Topological sort: 0 first, 3 last. Tie-break by OpId
        // chooses 1 before 2 (Kahn's heap on Reverse(OpId)).
        let order = topological_sort(&graph).expect("DAG");
        assert_eq!(order, vec![ids[0], ids[1], ids[2], ids[3]]);

        // Cross-check the order respects every edge.
        for (a, succs) in &graph.edges {
            let pos_a = order.iter().position(|x| x == a).unwrap();
            for s in succs {
                let pos_s = order.iter().position(|x| x == s).unwrap();
                assert!(pos_a < pos_s, "edge {:?}->{:?} violated", a, s);
            }
        }
    }

    // --- 4. EventRing producer/consumer via cycle_edge_key projection --

    #[test]
    fn event_ring_append_and_read_match_via_cycle_edge_projection() {
        let (mut prog, ids) = program_with_blank_ops(2);
        let ring = EventRingId(7);
        prog.ops[ids[0].0 as usize].record_write(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Append,
        });
        prog.ops[ids[1].0 as usize].record_read(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Read,
        });

        let graph = dependency_graph(&prog);
        assert_eq!(graph.successors(ids[0]), vec![ids[1]]);

        // The edge reason is the projected key (`Ring(EventRingId(7))`),
        // not the raw `DataHandle` — Append and Read collapse to it.
        let reasons = graph.edge_reasons.get(&(ids[0], ids[1])).unwrap();
        assert_eq!(reasons, &vec![CycleEdgeKey::Ring(ring)]);

        assert_eq!(topological_sort(&graph), Ok(vec![ids[0], ids[1]]));
    }

    // --- 5. Self-edge skipped ------------------------------------------

    #[test]
    fn self_edge_is_skipped_because_event_fold_pattern_is_legitimate() {
        let (mut prog, ids) = program_with_blank_ops(1);
        prog.ops[ids[0].0 as usize].record_read(hp_handle());
        prog.ops[ids[0].0 as usize].record_write(hp_handle());

        let graph = dependency_graph(&prog);
        assert!(graph.edges.is_empty(), "self-edge must not be recorded");
        assert!(graph.edge_reasons.is_empty());
        assert!(!graph.has_cycle());
        assert_eq!(topological_sort(&graph), Ok(vec![ids[0]]));
    }

    // --- 6. Cycle ------------------------------------------------------

    #[test]
    fn cycle_between_two_ops_is_reported_as_cycle_error() {
        // Op0: reads Hp, writes Mana.
        // Op1: reads Mana, writes Hp.
        // Edges: 0->1 (writes Mana → reads Mana) and 1->0 (writes Hp →
        // reads Hp).
        let (mut prog, ids) = program_with_blank_ops(2);
        prog.ops[ids[0].0 as usize].record_read(hp_handle());
        prog.ops[ids[0].0 as usize].record_write(mana_handle());
        prog.ops[ids[1].0 as usize].record_read(mana_handle());
        prog.ops[ids[1].0 as usize].record_write(hp_handle());

        let graph = dependency_graph(&prog);
        assert!(graph.has_cycle());

        let err = topological_sort(&graph).expect_err("cycle must surface");
        assert_eq!(err.cycles.len(), 1);
        assert_eq!(err.cycles[0], vec![ids[0], ids[1]]);

        // Display contains both op references.
        let rendered = format!("{}", err);
        assert!(rendered.contains("op#0"));
        assert!(rendered.contains("op#1"));
    }

    // --- 7. Determinism ------------------------------------------------

    #[test]
    fn dependency_graph_is_byte_identical_across_runs() {
        let (mut prog, ids) = program_with_blank_ops(4);
        prog.ops[ids[0].0 as usize].record_write(hp_handle());
        prog.ops[ids[1].0 as usize].record_read(hp_handle());
        prog.ops[ids[1].0 as usize].record_write(shield_handle());
        prog.ops[ids[2].0 as usize].record_read(hp_handle());
        prog.ops[ids[2].0 as usize].record_write(mana_handle());
        prog.ops[ids[3].0 as usize].record_read(shield_handle());
        prog.ops[ids[3].0 as usize].record_read(mana_handle());

        let g1 = dependency_graph(&prog);
        let g2 = dependency_graph(&prog);
        assert_eq!(g1, g2);

        // And the rendered debug form is identical too.
        assert_eq!(g1.display_for_debug(), g2.display_for_debug());

        // And topological sort is identical.
        assert_eq!(topological_sort(&g1), topological_sort(&g2));
    }

    // --- 8. Edge reasons (multi-handle pair) ---------------------------

    #[test]
    fn edge_reasons_capture_every_handle_that_bridges_a_pair() {
        // Op0 writes both Hp and ShieldHp; Op1 reads both. The single
        // (op0 -> op1) edge's reasons should list both handles, sorted
        // and deduplicated.
        let (mut prog, ids) = program_with_blank_ops(2);
        prog.ops[ids[0].0 as usize].record_write(hp_handle());
        prog.ops[ids[0].0 as usize].record_write(shield_handle());
        prog.ops[ids[1].0 as usize].record_read(hp_handle());
        prog.ops[ids[1].0 as usize].record_read(shield_handle());

        let graph = dependency_graph(&prog);
        let reasons = graph.edge_reasons.get(&(ids[0], ids[1])).unwrap();
        assert_eq!(reasons.len(), 2);
        // Sorted ascending — so the smaller projected key comes first.
        let mut expected = vec![hp_handle().cycle_edge_key(), shield_handle().cycle_edge_key()];
        expected.sort();
        assert_eq!(reasons, &expected);
    }

    // --- 9. No matching reader -----------------------------------------

    #[test]
    fn op_writes_handle_no_reader_yields_no_edge() {
        let (mut prog, ids) = program_with_blank_ops(2);
        prog.ops[ids[0].0 as usize].record_write(hp_handle());
        // ids[1] reads a *different* handle — must not produce an edge.
        prog.ops[ids[1].0 as usize].record_read(mana_handle());

        let graph = dependency_graph(&prog);
        assert!(graph.edges.is_empty());
        assert!(graph.edge_reasons.is_empty());
        assert_eq!(
            topological_sort(&graph),
            Ok(vec![ids[0], ids[1]]) // smaller OpId first via Kahn's tie-break.
        );
    }

    // --- 10. Three-cycle -----------------------------------------------

    #[test]
    fn three_cycle_is_reported_with_all_three_ops() {
        // 0 writes Hp, reads Mana
        // 1 reads Hp, writes ShieldHp
        // 2 reads ShieldHp, writes Mana
        // Edges: 0->1 (Hp), 1->2 (ShieldHp), 2->0 (Mana). SCC = {0,1,2}.
        let (mut prog, ids) = program_with_blank_ops(3);
        prog.ops[ids[0].0 as usize].record_write(hp_handle());
        prog.ops[ids[0].0 as usize].record_read(mana_handle());

        prog.ops[ids[1].0 as usize].record_read(hp_handle());
        prog.ops[ids[1].0 as usize].record_write(shield_handle());

        prog.ops[ids[2].0 as usize].record_read(shield_handle());
        prog.ops[ids[2].0 as usize].record_write(mana_handle());

        let graph = dependency_graph(&prog);
        assert!(graph.has_cycle());

        let err = topological_sort(&graph).expect_err("cycle expected");
        assert_eq!(err.cycles.len(), 1);
        assert_eq!(err.cycles[0], vec![ids[0], ids[1], ids[2]]);
    }

    // --- 11. Display roundtrips human-readably -------------------------

    #[test]
    fn dep_graph_display_for_debug_lists_every_edge_with_reason() {
        let (mut prog, ids) = program_with_blank_ops(2);
        prog.ops[ids[0].0 as usize].record_write(hp_handle());
        prog.ops[ids[1].0 as usize].record_read(hp_handle());

        let graph = dependency_graph(&prog);
        let rendered = graph.display_for_debug();
        assert!(rendered.contains("op#0 -> op#1"));
        assert!(rendered.contains("agent.self.hp"));
        assert!(rendered.contains("op_count: 2"));
    }

    // --- best_effort topo sort -----------------------------------------

    #[test]
    fn best_effort_topo_matches_strict_topo_on_acyclic_graph() {
        // Diamond fixture from `diamond_dependency_topologically_orders_root_before_sinks`.
        // Best-effort must agree with the strict variant when no cycle exists,
        // and report `None` for cycles.
        let (mut prog, ids) = program_with_blank_ops(4);
        prog.ops[ids[0].0 as usize].record_write(hp_handle());
        prog.ops[ids[1].0 as usize].record_read(hp_handle());
        prog.ops[ids[1].0 as usize].record_write(shield_handle());
        prog.ops[ids[2].0 as usize].record_read(hp_handle());
        prog.ops[ids[2].0 as usize].record_write(mana_handle());
        prog.ops[ids[3].0 as usize].record_read(shield_handle());
        prog.ops[ids[3].0 as usize].record_read(mana_handle());

        let graph = dependency_graph(&prog);
        let strict = topological_sort(&graph).expect("acyclic");
        let (best, cycles) = topological_sort_best_effort(&graph);
        assert_eq!(best, strict);
        assert!(cycles.is_none(), "no cycles expected on acyclic graph");
    }

    #[test]
    fn best_effort_topo_breaks_cycle_via_source_order_and_reports_it() {
        // Op0 ↔ Op1 mutual writers (same shape as `cycle_between_two_ops_is_reported_as_cycle_error`).
        // Best-effort returns an order of length 2; smallest OpId comes first
        // because forcing breaks at the smallest remaining OpId, and
        // every edge that's NOT inside the cycle is honoured (here there
        // are none, but cycles surface as a `Some(CycleError)` signal).
        let (mut prog, ids) = program_with_blank_ops(2);
        prog.ops[ids[0].0 as usize].record_read(hp_handle());
        prog.ops[ids[0].0 as usize].record_write(mana_handle());
        prog.ops[ids[1].0 as usize].record_read(mana_handle());
        prog.ops[ids[1].0 as usize].record_write(hp_handle());

        let graph = dependency_graph(&prog);
        let (order, cycles) = topological_sort_best_effort(&graph);
        assert_eq!(order, vec![ids[0], ids[1]]);
        assert!(cycles.is_some(), "cycle must surface in the secondary signal");
        let cycles = cycles.unwrap();
        assert_eq!(cycles.cycles.len(), 1);
        assert_eq!(cycles.cycles[0], vec![ids[0], ids[1]]);
    }

    #[test]
    fn best_effort_topo_preserves_cross_scc_edges_when_one_scc_has_a_cycle() {
        // 4-op fixture: op0 ↔ op1 (cycle), op2 → op3 (acyclic chain
        // independent of the cycle). Best-effort must place op2 before
        // op3 (the cross-SCC edge survives) and forced source order
        // within the cycle: op0 then op1.
        //
        // The previous fusion fallback (`(0..n).map(OpId).collect()`)
        // gave the right shape here only by accident — when fusion
        // dispatches a real graph (e.g. trade_caravans's spatial
        // build-chain → user-op edges), source order discards the
        // ordering information the best-effort variant preserves.
        let (mut prog, ids) = program_with_blank_ops(4);
        // SCC: op0 ↔ op1 via Hp/Mana mutual writers.
        prog.ops[ids[0].0 as usize].record_read(hp_handle());
        prog.ops[ids[0].0 as usize].record_write(mana_handle());
        prog.ops[ids[1].0 as usize].record_read(mana_handle());
        prog.ops[ids[1].0 as usize].record_write(hp_handle());
        // Independent chain: op2 → op3 via ShieldHp.
        prog.ops[ids[2].0 as usize].record_write(shield_handle());
        prog.ops[ids[3].0 as usize].record_read(shield_handle());

        let graph = dependency_graph(&prog);
        let (order, cycles) = topological_sort_best_effort(&graph);

        assert_eq!(order.len(), 4);
        assert!(cycles.is_some(), "cycle present in op0/op1 SCC");

        let pos = |op: OpId| order.iter().position(|x| *x == op).unwrap();
        // Cross-SCC edge survives: op2 BEFORE op3.
        assert!(
            pos(ids[2]) < pos(ids[3]),
            "op2→op3 edge must be honoured: {:?}",
            order
        );
        // Within-SCC: source order tie-break.
        assert!(
            pos(ids[0]) < pos(ids[1]),
            "within-SCC source-order tie-break: {:?}",
            order
        );
    }

    // --- ring-cycle fallback (S10) -------------------------------------

    /// Hand-build a graph: `ring_edge(a, b)` for each pair.
    fn ring_graph(op_count: usize, ring: &[(u32, u32)]) -> DepGraph {
        let mut edges: BTreeMap<OpId, BTreeSet<OpId>> = BTreeMap::new();
        let mut reasons: BTreeMap<(OpId, OpId), Vec<CycleEdgeKey>> = BTreeMap::new();
        for &(p, c) in ring {
            edges.entry(OpId(p)).or_default().insert(OpId(c));
            reasons
                .entry((OpId(p), OpId(c)))
                .or_default()
                .push(CycleEdgeKey::Ring(crate::cg::data_handle::EventRingId(0)));
        }
        DepGraph {
            op_count,
            edges,
            edge_reasons: reasons,
        }
    }

    #[test]
    fn ring_cycle_fallback_breaks_inside_the_cycle_not_on_an_innocent_consumer() {
        // op1 ↔ op2 is a RING cycle; op0 is an innocent ring consumer of
        // op2 that happens to carry the smallest OpId. Every remaining op
        // has a pending ring producer, so the guard runs out of legal
        // candidates and the fallback decides who gets sacrificed.
        //
        // The old fallback took the global smallest — op0, the innocent
        // one, whose same-tick read would then be dead for a cycle it is
        // not even part of. The shipped fallback breaks inside the SCC.
        let graph = ring_graph(3, &[(1, 2), (2, 1), (2, 0)]);
        let (order, cycles, breaks) = topological_sort_best_effort_reporting(&graph);

        assert_eq!(order.len(), 3);
        assert!(cycles.is_some(), "op1 ↔ op2 is a cycle");
        assert_eq!(breaks.len(), 1, "exactly one forced ring break: {breaks:?}");
        assert!(
            breaks[0].cyclic,
            "the break must be attributed to the real cycle: {:?}",
            breaks[0]
        );
        assert!(
            breaks[0].consumer == OpId(1) || breaks[0].consumer == OpId(2),
            "the break must land on a cycle member, not the innocent op0: {:?}",
            breaks[0]
        );

        let pos = |op: u32| order.iter().position(|x| x.0 == op).unwrap();
        assert!(
            pos(2) < pos(0),
            "op0's producer (op2) must still precede it: {order:?}"
        );
    }

    #[test]
    fn acyclic_ring_relation_never_reports_a_forced_break() {
        // A field-only cycle (op0 ↔ op1) plus a ring chain op2 → op3.
        // The stall happens, the force-pick runs, and the ring edge is
        // still honoured with no break reported.
        let (mut prog, ids) = program_with_blank_ops(4);
        prog.ops[ids[0].0 as usize].record_read(hp_handle());
        prog.ops[ids[0].0 as usize].record_write(mana_handle());
        prog.ops[ids[1].0 as usize].record_read(mana_handle());
        prog.ops[ids[1].0 as usize].record_write(hp_handle());
        let ring = EventRingId(3);
        prog.ops[ids[2].0 as usize].record_write(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Append,
        });
        prog.ops[ids[3].0 as usize].record_read(DataHandle::EventRing {
            ring,
            kind: EventRingAccess::Read,
        });

        let graph = dependency_graph(&prog);
        let (order, cycles, breaks) = topological_sort_best_effort_reporting(&graph);
        assert!(cycles.is_some());
        assert!(breaks.is_empty(), "no ring break expected: {breaks:?}");
        let pos = |op: OpId| order.iter().position(|x| *x == op).unwrap();
        assert!(pos(ids[2]) < pos(ids[3]), "{order:?}");
    }
}
