//! S5 diagnostic: print the dependency-graph facts for one fixture's
//! BeliefSocialMerge op vs its event producer (found while porting
//! webband_colony's raids — the supper merge kernel scheduled BEFORE
//! PhysicsSupperGather, so the same-tick live-tail read saw no
//! SupperTale rows and gossip went dead).
//!
//! Usage: cargo run -p dsl_compiler --example sched_probe <fixture>

use dsl_ast::parser::parse_program;
use dsl_ast::resolve::resolve;
use dsl_compiler::cg::lower::lower_compilation_to_cg;
use dsl_compiler::cg::{dependency_graph, topological_sort};

fn main() {
    let name = std::env::args().nth(1).expect("usage: sched_probe <fixture>");
    let path = format!("assets/sim/{name}.sim");
    let src = std::fs::read_to_string(&path).expect("read fixture");
    let program = parse_program(&src).expect("parse");
    // Custom-field registry, as build_helper does (Gap plague_city#P-A) —
    // without it every rule touching a custom field drops at lowering
    // and the probe inspects a phantom program.
    let _ = dsl_compiler::custom_agent_fields::populate(&program);
    let comp = resolve(program).expect("resolve");
    let prog = match lower_compilation_to_cg(&comp) {
        Ok(p) => p,
        Err(o) => {
            for d in &o.diagnostics {
                eprintln!("[lower diag] {d}");
            }
            o.program
        }
    };
    let deps = dependency_graph(&prog);
    let order = topological_sort(&deps);
    let labels: Vec<String> = prog.ops.iter().map(|o| o.kind.label()).collect();
    let find = |needle: &str| -> Vec<usize> {
        labels
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains(needle))
            .map(|(i, _)| i)
            .collect()
    };
    if std::env::var_os("SCHED_PROBE_DUMP_OPS").is_some() {
        for (i, l) in labels.iter().enumerate() {
            println!("OP {i}: {l}");
        }
    }
    let merges = find("belief_social_merge");
    let suppers = find("SupperGather");
    println!("merge ops: {merges:?} supper ops: {suppers:?}");
    for (i, l) in labels.iter().enumerate() {
        if merges.contains(&i) || suppers.contains(&i) {
            println!("  op {i}: {l}");
        }
    }
    match &order {
        Ok(ord) => {
            for (pos, op) in ord.iter().enumerate() {
                let i = op.0 as usize;
                if merges.contains(&i) || suppers.contains(&i) {
                    println!("order pos {pos}: op {i} ({})", labels[i]);
                }
            }
        }
        Err(e) => println!("topological_sort error: {e}"),
    }
    // Cycle members + edge reasons.
    if let Err(e) = &order {
        for scc in &e.cycles {
            println!("CYCLE:");
            for op in scc {
                println!("  op {}: {}", op.0, labels[op.0 as usize]);
            }
            for a in scc {
                for b in scc {
                    if let Some(reasons) = deps.edge_reasons.get(&(*a, *b)) {
                        println!("  edge {} -> {} via {:?}", a.0, b.0, reasons);
                    }
                }
            }
        }
    }
    // Edge presence: producer -> merge?
    for &s in &suppers {
        for &m in &merges {
            let has = deps
                .edges
                .get(&dsl_compiler::cg::OpId(s as u32))
                .map(|set| set.contains(&dsl_compiler::cg::OpId(m as u32)))
                .unwrap_or(false);
            println!("edge supper({s}) -> merge({m}): {has}");
        }
    }
    // Raw handle lists on the merge op + ring writers census.
    for &m in &merges {
        let op = &prog.ops[m];
        println!("merge op {m} reads: {:?}", op.reads);
        println!("merge op {m} writes: {:?}", op.writes);
    }
    // Which event kinds does each PhysicsRule emit? (own recursive walk)
    fn walk(
        list: dsl_compiler::cg::CgStmtListId,
        prog: &dsl_compiler::cg::CgProgram,
        out: &mut Vec<u32>,
    ) {
        let Some(l) = prog.stmt_lists.get(list.0 as usize) else { return };
        for &sid in &l.stmts {
            let Some(s) = prog.stmts.get(sid.0 as usize) else { continue };
            match s {
                dsl_compiler::cg::CgStmt::Emit { event, .. } => out.push(event.0),
                dsl_compiler::cg::CgStmt::If { then, else_, .. } => {
                    walk(*then, prog, out);
                    if let Some(e) = else_ {
                        walk(*e, prog, out);
                    }
                }
                dsl_compiler::cg::CgStmt::Match { arms, .. } => {
                    for a in arms {
                        walk(a.body, prog, out);
                    }
                }
                dsl_compiler::cg::CgStmt::ForEachNeighborBody { body, .. } => {
                    walk(*body, prog, out);
                }
                dsl_compiler::cg::CgStmt::ForEachAgentBody { body, .. } => {
                    walk(*body, prog, out);
                }
                _ => {}
            }
        }
    }
    for (i, op) in prog.ops.iter().enumerate() {
        if let dsl_compiler::cg::ComputeOpKind::PhysicsRule { body, .. } = &op.kind {
            let mut kinds = Vec::new();
            walk(*body, &prog, &mut kinds);
            if !kinds.is_empty() {
                println!("physics op {i} ({}) emits {kinds:?}", labels[i]);
            }
            // Dump the raw stmt-variant tree for ops that emit nothing
            // (looking for where per_agent emits actually live).
            if kinds.is_empty() {
                let Some(l) = prog.stmt_lists.get(body.0 as usize) else { continue };
                let mut variants: Vec<String> = Vec::new();
                for &sid in &l.stmts {
                    if let Some(s) = prog.stmts.get(sid.0 as usize) {
                        let d = format!("{s:?}");
                        variants.push(d.split(&[' ', '{'][..]).next().unwrap_or("?").to_string());
                    }
                }
                println!("physics op {i} ({}) stmt variants: {variants:?}", labels[i]);
            }
        }
    }
    let mut ring_writers = 0usize;
    for op in prog.ops.iter() {
        if op
            .writes
            .iter()
            .any(|w| matches!(w, dsl_compiler::cg::DataHandle::EventRing { .. }))
        {
            ring_writers += 1;
        }
    }
    println!("ops writing EventRing: {ring_writers}");
    // Any edges INTO the merge ops at all?
    for &m in &merges {
        let mut incoming: Vec<usize> = Vec::new();
        for (p, set) in deps.edges.iter() {
            if set.contains(&dsl_compiler::cg::OpId(m as u32)) {
                incoming.push(p.0 as usize);
            }
        }
        incoming.sort();
        println!(
            "incoming edges to merge op {m}: {:?}",
            incoming
                .iter()
                .map(|&i| format!("{i}:{}", labels[i]))
                .collect::<Vec<_>>()
        );
    }
}
