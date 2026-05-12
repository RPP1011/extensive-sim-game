// Throwaway: dumps dep graph + toposort + fusion diagnostics for the
// cooldown_probe.sim. Used to diagnose the producer/consumer inversion.
fn main() {
    let src = std::fs::read_to_string("assets/sim/cooldown_probe.sim").unwrap();
    let program = dsl_compiler::parse(&src).unwrap();
    let comp = dsl_ast::resolve::resolve(program).unwrap();
    let cg = dsl_compiler::cg::lower::lower_compilation_to_cg(&comp).unwrap();

    println!("=== ops ({} total) ===", cg.ops.len());
    for (i, op) in cg.ops.iter().enumerate() {
        println!(
            "op#{i} kind={:?}\n   reads:  {:?}\n   writes: {:?}\n",
            op.kind,
            op.reads.iter().map(|h| format!("{h}")).collect::<Vec<_>>(),
            op.writes.iter().map(|h| format!("{h}")).collect::<Vec<_>>(),
        );
    }

    let deps = dsl_compiler::cg::schedule::topology::dependency_graph(&cg);
    println!("\n=== dep graph ===\n{}", deps.display_for_debug());

    let topo = dsl_compiler::cg::schedule::topology::topological_sort(&deps);
    match topo {
        Ok(order) => {
            println!("\n=== toposort ===");
            for op in &order {
                println!("op#{}", op.0);
            }
        }
        Err(c) => println!("\n=== cycle ===\n{c}"),
    }

    let result = dsl_compiler::cg::schedule::synthesize_schedule(
        &cg,
        dsl_compiler::cg::schedule::ScheduleStrategy::Default,
    );
    println!("\n=== schedule ({} stages) ===", result.schedule.stages.len());
    for (i, stage) in result.schedule.stages.iter().enumerate() {
        for (j, kernel) in stage.kernels.iter().enumerate() {
            println!("stage#{i} kernel#{j}: {kernel:?}");
        }
    }
    println!("\n=== fusion diagnostics ===");
    for d in &result.fusion_diagnostics {
        println!("{d:?}");
    }
}
