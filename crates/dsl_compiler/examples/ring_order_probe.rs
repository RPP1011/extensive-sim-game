//! S10 diagnostic: schedule a `.sim` and report every event-ring
//! producer→consumer ordering fact.
//!
//! Complements `sched_probe` (which dumps the dep graph): this one runs
//! the REAL schedule synthesis and prints
//! `cg::schedule::ring_order::validate_ring_order`'s findings, plus a
//! side-by-side of the shipped force-pick against the pre-S5 legacy
//! force-pick (unconditional global-smallest) so a regression in the
//! ordering policy is visible as a diff rather than as a dead feature.
//!
//! Usage:
//!   cargo run -p dsl_compiler --example ring_order_probe <fixture|path.sim>

use std::collections::{BTreeSet, BinaryHeap};
use std::cmp::Reverse;

use dsl_compiler::cg::schedule::{
    dependency_graph, ring_order::validate_ring_order, synthesize_schedule, ScheduleStrategy,
};
use dsl_compiler::cg::{CycleEdgeKey, DepGraph, OpId};

/// The pre-S5 force-pick: on a cycle stall, emit the globally smallest
/// un-emitted op with no regard for pending event-ring producers.
/// Reproduced here (and ONLY here) so the probe and the regression test
/// can measure what the shipped policy buys.
fn legacy_best_effort(graph: &DepGraph) -> Vec<OpId> {
    let n = graph.op_count;
    let mut in_degree = vec![0u32; n];
    for succs in graph.edges.values() {
        for s in succs {
            if (s.0 as usize) < n {
                in_degree[s.0 as usize] += 1;
            }
        }
    }
    let mut queue: BinaryHeap<Reverse<OpId>> = BinaryHeap::new();
    let mut emitted = vec![false; n];
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push(Reverse(OpId(i as u32)));
        }
    }
    let mut order = Vec::with_capacity(n);
    while order.len() < n {
        while let Some(Reverse(op)) = queue.pop() {
            if emitted[op.0 as usize] {
                continue;
            }
            emitted[op.0 as usize] = true;
            order.push(op);
            if let Some(succs) = graph.edges.get(&op) {
                for &s in succs {
                    let i = s.0 as usize;
                    if i < n && !emitted[i] {
                        in_degree[i] -= 1;
                        if in_degree[i] == 0 {
                            queue.push(Reverse(s));
                        }
                    }
                }
            }
        }
        if order.len() == n {
            break;
        }
        let forced = (0..n).find(|&i| !emitted[i]).expect("residual op");
        emitted[forced] = true;
        order.push(OpId(forced as u32));
        if let Some(succs) = graph.edges.get(&OpId(forced as u32)) {
            for &s in succs {
                let i = s.0 as usize;
                if i < n && !emitted[i] {
                    in_degree[i] = in_degree[i].saturating_sub(1);
                    if in_degree[i] == 0 {
                        queue.push(Reverse(s));
                    }
                }
            }
        }
    }
    order
}

fn ring_edges(graph: &DepGraph) -> Vec<(OpId, OpId)> {
    graph
        .edge_reasons
        .iter()
        .filter(|(_, r)| r.iter().any(|k| matches!(k, CycleEdgeKey::Ring(_))))
        .map(|((p, c), _)| (*p, *c))
        .collect()
}

fn violations(graph: &DepGraph, order: &[OpId]) -> Vec<(OpId, OpId)> {
    let n = graph.op_count;
    let mut pos = vec![usize::MAX; n];
    for (i, op) in order.iter().enumerate() {
        if (op.0 as usize) < n {
            pos[op.0 as usize] = i;
        }
    }
    ring_edges(graph)
        .into_iter()
        .filter(|(p, c)| pos[c.0 as usize] < pos[p.0 as usize])
        .collect()
}

fn main() {
    let arg = std::env::args()
        .nth(1)
        .expect("usage: ring_order_probe <fixture|path.sim>");
    let path = if arg.ends_with(".sim") {
        std::path::PathBuf::from(arg)
    } else {
        std::path::PathBuf::from(format!("assets/sim/{arg}.sim"))
    };
    let src = std::fs::read_to_string(&path).expect("read fixture");
    let program = dsl_compiler::parse(&src).expect("parse");
    let _ = dsl_compiler::custom_agent_fields::populate(&program);
    let comp = dsl_ast::resolve::resolve(program).expect("resolve");
    let prog = match dsl_compiler::cg::lower::lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                eprintln!("[lower diag] {d}");
            }
            o.program
        }
    };
    let deps = dependency_graph(&prog);
    let result = synthesize_schedule(&prog, ScheduleStrategy::Default);
    let stage_order: Vec<OpId> = result
        .schedule
        .stages
        .iter()
        .flat_map(|s| s.kernels.iter().flat_map(|k| k.ops()))
        .collect();

    println!(
        "{}: {} ops, {} stages, {} ring edges",
        path.display(),
        prog.ops.len(),
        result.schedule.stages.len(),
        ring_edges(&deps).len(),
    );

    let shipped = violations(&deps, &stage_order);
    let legacy_order = legacy_best_effort(&deps);
    let legacy = violations(&deps, &legacy_order);
    println!("ring-order violations: shipped={} legacy={}", shipped.len(), legacy.len());
    let shipped_set: BTreeSet<_> = shipped.iter().collect();
    for (p, c) in &legacy {
        let tag = if shipped_set.contains(&(*p, *c)) { "BOTH" } else { "LEGACY-ONLY" };
        println!(
            "  {tag}: producer op#{} ({}) -> consumer op#{} ({})",
            p.0,
            prog.ops[p.0 as usize].kind.label(),
            c.0,
            prog.ops[c.0 as usize].kind.label(),
        );
    }
    for (p, c) in &shipped {
        println!(
            "  SHIPPED-VIOLATION: producer op#{} -> consumer op#{}",
            p.0, c.0
        );
    }

    for issue in validate_ring_order(&prog, &deps, &stage_order) {
        println!("issue {issue}");
    }
}
