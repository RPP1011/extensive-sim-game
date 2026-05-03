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

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt;

use crate::cg::data_handle::CycleEdgeKey;
use crate::cg::op::OpId;
use crate::cg::program::CgProgram;

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
///   identity alone.
pub fn dependency_graph(prog: &CgProgram) -> DepGraph {
    let mut edges: BTreeMap<OpId, BTreeSet<OpId>> = BTreeMap::new();
    let mut edge_reasons: BTreeMap<(OpId, OpId), Vec<CycleEdgeKey>> = BTreeMap::new();

    // Index "what does each handle's writers list look like" first.
    // Same projection (`cycle_edge_key`) used by the consumer-side scan
    // so EventRing append/read/drain accesses match on ring identity.
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
    for (op_index, op) in prog.ops.iter().enumerate() {
        let consumer = OpId(op_index as u32);
        let consumer_is_spatial_build = matches!(
            op.kind,
            crate::cg::op::ComputeOpKind::SpatialQuery { .. }
        );
        for r in &op.reads {
            let key = r.cycle_edge_key();
            if let Some(producers) = writers.get(&key) {
                for &producer in producers {
                    if producer == consumer {
                        continue;
                    }
                    if consumer_is_spatial_build {
                        // Cross-tick read (see comment above).
                        continue;
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
fn find_cycles(graph: &DepGraph) -> Vec<Vec<OpId>> {
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
}
