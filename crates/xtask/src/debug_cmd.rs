//! `xtask debug` — interactive per-phase REPL for the tick pipeline.
//!
//! Mirrors `engine::debug::tick_stepper` semantics: a driver thread steps
//! through synthetic tick phases; the main thread reads stdin commands between
//! phases.
//!
//! # Deviation note
//!
//! xtask has no engine / engine_rules dependency (host-only tool crate). This
//! command exercises the StepperHandle REPL plumbing with a synthetic tick
//! loop rather than a real `SerialBackend::step()` call. The CLI interface and
//! channel protocol are identical to the real harness.
//!
//! To wire up a real tick, add `engine_rules` to xtask's `[dependencies]` and
//! replace `synthetic_tick_phases` with `engine_rules::step::step(...)`.

use std::io::{self, BufRead};
use std::process::ExitCode;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use crate::cli::DebugArgs;

// ---------------------------------------------------------------------------
// Synthetic phase names — match engine::debug::tick_stepper::Phase variants.
// ---------------------------------------------------------------------------

const PHASES: &[&str] = &[
    "BeforeViewFold",
    "AfterMaskFill",
    "AfterScoring",
    "AfterActionSelect",
    "AfterCascadeDispatch",
    "AfterViewFold",
    "TickEnd",
];

/// Driver message: a phase name or a sentinel indicating the tick loop ended.
enum DriverMsg {
    Phase(&'static str, u32 /* tick */),
    Done,
}

/// Controller command parsed from stdin.
enum Cmd {
    Continue,
    Abort,
}

/// Run the driver thread: simulate N ticks, each with 7 phase checkpoints.
///
/// Between checkpoints the driver sends the phase name over `phase_tx` and
/// blocks on `step_rx`. On `Cmd::Abort` it returns early.
fn run_driver(
    ticks: u32,
    phase_tx: Sender<DriverMsg>,
    step_rx: Receiver<Cmd>,
) {
    'tick: for tick in 0..ticks {
        for &phase in PHASES {
            phase_tx.send(DriverMsg::Phase(phase, tick)).ok();
            match step_rx.recv().unwrap_or(Cmd::Abort) {
                Cmd::Continue => {}
                Cmd::Abort => break 'tick,
            }
        }
    }
    phase_tx.send(DriverMsg::Done).ok();
}

pub fn run_debug(args: DebugArgs) -> ExitCode {
    println!(
        "debug: scenario={} ticks={} interactive={}",
        args.scenario.display(),
        args.ticks,
        !args.no_interactive,
    );
    println!("NOTE: running in stub mode (no engine dep). Real tick phases are simulated.");
    println!("Commands at each phase: [c]ontinue  [a]bort  (empty = continue)");
    println!();

    let (phase_tx, phase_rx) = channel::<DriverMsg>();
    let (step_tx, step_rx) = channel::<Cmd>();

    let ticks = args.ticks;
    let driver = thread::spawn(move || run_driver(ticks, phase_tx, step_rx));

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        match phase_rx.recv() {
            Ok(DriverMsg::Phase(phase, tick)) => {
                print!("[tick={tick} phase={phase}] > ");
                // Flush: println already flushes on most platforms via the
                // newline, but we used `print!` here — use explicit flush.
                use std::io::Write;
                let _ = io::stdout().flush();

                if args.no_interactive {
                    // Non-interactive mode: auto-continue without reading stdin.
                    println!("(auto-continue)");
                    step_tx.send(Cmd::Continue).ok();
                    continue;
                }

                let cmd = match lines.next() {
                    Some(Ok(line)) => {
                        let trimmed = line.trim().to_ascii_lowercase();
                        if trimmed.starts_with('a') {
                            Cmd::Abort
                        } else {
                            Cmd::Continue
                        }
                    }
                    // EOF or read error → treat as abort.
                    _ => Cmd::Abort,
                };

                let label = match cmd {
                    Cmd::Continue => "continue",
                    Cmd::Abort => "abort",
                };
                println!("  -> {label}");

                let should_abort = matches!(cmd, Cmd::Abort);
                step_tx.send(cmd).ok();
                if should_abort {
                    break;
                }
            }
            Ok(DriverMsg::Done) | Err(_) => {
                println!("debug: tick loop complete.");
                break;
            }
        }
    }

    driver.join().ok();
    ExitCode::SUCCESS
}
