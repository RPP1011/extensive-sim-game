//! `xtask trace` — non-interactive mask + agent-history collection.
//!
//! Mirrors `engine::debug::DebugConfig { trace_mask: true, agent_history: Some(..) }`.
//!
//! # Deviation note
//!
//! xtask has no engine / engine_rules dependency. This command simulates the
//! collection pattern with synthetic per-tick data. To wire up a real run,
//! add `engine_rules` to xtask's `[dependencies]` and replace the stub loop
//! with `engine_rules::step::step(...)` using a DebugConfig that enables
//! `trace_mask` and `agent_history`.

use std::fmt::Write as FmtWrite;
use std::process::ExitCode;

use crate::cli::TraceArgs;

pub fn run_trace(args: TraceArgs) -> ExitCode {
    println!(
        "trace: scenario={} ticks={}",
        args.scenario.display(),
        args.ticks,
    );
    println!("NOTE: running in stub mode (no engine dep). Synthetic snapshots generated.");
    println!();

    // Simulate collection of mask + agent-history snapshots.
    let mut report = String::new();
    writeln!(report, "=== Trace Report ({} ticks) ===", args.ticks).unwrap();
    writeln!(report, "scenario: {}", args.scenario.display()).unwrap();
    writeln!(report).unwrap();

    writeln!(report, "--- mask snapshots ---").unwrap();
    for tick in 0..args.ticks {
        // Synthetic: 4 agents × 8 mask kinds, all enabled.
        let n_agents = 4u32;
        let n_kinds = 8u32;
        let bits_set = n_agents * n_kinds;
        writeln!(
            report,
            "  tick={tick:>4}  n_agents={n_agents}  n_kinds={n_kinds}  bits_set={bits_set}",
        )
        .unwrap();
    }

    writeln!(report).unwrap();
    writeln!(report, "--- agent history ---").unwrap();
    for tick in 0..args.ticks {
        // Synthetic: 4 agents alive, HP decreasing linearly.
        for agent_id in 0..4u32 {
            let hp = 100.0f32 - (tick as f32) * 0.5;
            if hp <= 0.0 {
                continue;
            }
            writeln!(
                report,
                "  tick={tick:>4}  agent={agent_id}  alive=true  hp={hp:.1}  pos=({:.1},{:.1},{:.1})",
                agent_id as f32 * 2.0,
                0.0,
                agent_id as f32 * 2.0,
            )
            .unwrap();
        }
    }

    // Write or print output.
    if let Some(out_path) = &args.output {
        match std::fs::write(out_path, &report) {
            Ok(()) => println!("trace: wrote {} bytes to {}", report.len(), out_path.display()),
            Err(e) => {
                eprintln!("trace: failed to write {}: {e}", out_path.display());
                return ExitCode::FAILURE;
            }
        }
    } else {
        print!("{report}");
    }

    ExitCode::SUCCESS
}
