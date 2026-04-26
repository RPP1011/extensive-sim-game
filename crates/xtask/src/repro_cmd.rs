//! `xtask repro` — capture or replay a reproduction bundle.
//!
//! Mirrors `engine::debug::repro_bundle::ReproBundle`:
//!   capture: run scenario → capture state + causal tree → write_to(path)
//!   replay:  read_from(path) → print causal_tree_dump
//!
//! # Wire format (from engine::debug::repro_bundle)
//!
//! ```text
//! [4-byte LE length][snapshot_bytes]
//! [4-byte LE length][causal_tree_dump UTF-8]
//! [4-byte LE length][mask_trace_bytes]
//! [4-byte LE length][agent_history_bytes]
//! [32-byte schema_hash]
//! ```
//!
//! # Deviation note
//!
//! xtask has no engine / engine_rules dependency. Capture produces a
//! well-formed bundle with synthetic content using the same wire format.
//! Replay reads and validates any bundle written by this command or by
//! `engine::debug::repro_bundle::ReproBundle::write_to`.

use std::io;
use std::path::Path;
use std::process::ExitCode;

use crate::cli::ReproArgs;

// ---------------------------------------------------------------------------
// Wire-format helpers (mirror engine::debug::repro_bundle internals)
// ---------------------------------------------------------------------------

fn write_section(out: &mut Vec<u8>, data: &[u8]) {
    let len = data.len() as u32;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(data);
}

fn read_section<'a>(data: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    if *pos + 4 > data.len() {
        return None;
    }
    let len = u32::from_le_bytes(data[*pos..*pos + 4].try_into().ok()?) as usize;
    *pos += 4;
    if *pos + len > data.len() {
        return None;
    }
    let slice = &data[*pos..*pos + len];
    *pos += len;
    Some(slice)
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

fn do_capture(scenario: &Path, ticks: u32, output: &Path) -> io::Result<()> {
    println!(
        "repro capture: scenario={} ticks={} output={}",
        scenario.display(),
        ticks,
        output.display(),
    );
    println!("NOTE: running in stub mode (no engine dep). Synthetic bundle generated.");

    // Synthetic snapshot bytes.
    let snapshot_bytes: Vec<u8> = format!(
        "STUB_SNAPSHOT scenario={} ticks={} seed=42",
        scenario.display(),
        ticks
    )
    .into_bytes();

    // Synthetic causal tree dump.
    let causal_tree_dump = format!(
        "[tick=0 seq=0 kind=Attack]\n  └ [tick=0 seq=1 kind=Damage]\n    └ [tick=0 seq=2 kind=Death]\n\
         [tick=1 seq=3 kind=Heal]\n  └ [tick=1 seq=4 kind=StatusClear]\n\
         <{ticks} ticks simulated, scenario={}>\n",
        scenario.display(),
    );

    // Synthetic mask trace bytes (1 tick, 4 agents, 8 kinds, all enabled).
    let mut mask_trace_bytes = Vec::<u8>::new();
    let n_agents: u32 = 4;
    let n_kinds: u32 = 8;
    mask_trace_bytes.extend_from_slice(&0u32.to_le_bytes()); // tick 0
    mask_trace_bytes.extend_from_slice(&n_agents.to_le_bytes());
    mask_trace_bytes.extend_from_slice(&n_kinds.to_le_bytes());
    mask_trace_bytes.extend(std::iter::repeat(1u8).take((n_agents * n_kinds) as usize));

    // Synthetic agent history (text dump).
    let agent_history_bytes = format!(
        "AgentHistory {{ snapshots: [TickSnapshot {{ tick: 0, n_agents: {n_agents} }}, ...] }}"
    )
    .into_bytes();

    // Schema hash: 32-byte stub (all zeroes for synthetic bundles).
    let schema_hash = [0u8; 32];

    // Assemble bundle.
    let mut buf = Vec::<u8>::new();
    write_section(&mut buf, &snapshot_bytes);
    write_section(&mut buf, causal_tree_dump.as_bytes());
    write_section(&mut buf, &mask_trace_bytes);
    write_section(&mut buf, &agent_history_bytes);
    buf.extend_from_slice(&schema_hash);

    std::fs::write(output, &buf)?;
    println!(
        "repro capture: wrote {} bytes to {}",
        buf.len(),
        output.display()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

fn do_replay(input: &Path) -> io::Result<()> {
    println!("repro replay: input={}", input.display());

    let data = std::fs::read(input)?;
    let mut pos = 0usize;

    let snapshot_bytes = read_section(&data, &mut pos)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing snapshot section"))?;
    let tree_bytes = read_section(&data, &mut pos)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing causal-tree section"))?;
    let mask_bytes = read_section(&data, &mut pos)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing mask-trace section"))?;
    let history_bytes = read_section(&data, &mut pos).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "missing agent-history section")
    })?;

    if data.len() < pos + 32 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing schema_hash (need 32 bytes at end)",
        ));
    }
    let schema_hash = &data[pos..pos + 32];

    println!();
    println!("=== Repro Bundle Contents ===");
    println!("snapshot_bytes   : {} bytes", snapshot_bytes.len());

    let causal_tree = std::str::from_utf8(tree_bytes)
        .unwrap_or("<invalid UTF-8 in causal-tree section>");
    println!("--- causal_tree_dump ---");
    print!("{causal_tree}");

    println!("--- mask_trace ---");
    println!("  {} bytes of mask-trace data", mask_bytes.len());

    println!("--- agent_history ---");
    let history_text = std::str::from_utf8(history_bytes)
        .unwrap_or("<invalid UTF-8 in agent-history section>");
    println!("  {history_text}");

    println!("--- schema_hash ---");
    let hash_hex: String = schema_hash.iter().map(|b| format!("{b:02x}")).collect();
    println!("  {hash_hex}");

    println!();
    println!("repro replay: bundle parsed successfully.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

pub fn run_repro(args: ReproArgs) -> ExitCode {
    if let Some(replay_path) = &args.replay {
        match do_replay(replay_path) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("repro replay error: {e}");
                ExitCode::FAILURE
            }
        }
    } else {
        match do_capture(&args.scenario, args.ticks, &args.output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("repro capture error: {e}");
                ExitCode::FAILURE
            }
        }
    }
}
