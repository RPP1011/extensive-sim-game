//! `cargo run --bin xtask -- chronicle` — run an engine fixture and dump
//! the rendered chronicle.
//!
//! Two fixtures:
//!
//! * **Canonical** (default): mirrors
//!   `crates/engine/tests/wolves_and_humans_parity.rs` — 3 humans + 2 wolves
//!   on a flat plane, 100 ticks, seed `0xD00DFACE00420042`. Identical output
//!   to the parity test minus the assertion harness.
//! * **Showcase** (`--showcase`): longer curated fixture — 8 humans,
//!   8 wolves, 4 deer scattered across a 40×40 area, 500 ticks,
//!   seed `0xDEADBEEF`. Emits pretty-printed section headers and an
//!   outcome summary for demos and essays. Selectable presets via
//!   `--fixture`:
//!     * `showcase` (default): balance-biased toward humans (~98% win);
//!       the canonical demo narrative.
//!     * `balanced`: wolf-favoured counts + HP tuned to produce mixed
//!       outcomes so the balance sweep surfaces both win paths.
//!
//! Both paths walk the event ring and hand every `ChronicleEntry` to
//! `engine::chronicle::render_entry`, printing one line per event.

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use engine::cascade::CascadeRegistry;
use engine::chronicle;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::invariant::{InvariantRegistry, PoolNonOverlapInvariant};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::NullSink;
use engine::view::materialized::MaterializedView;
use glam::Vec3;

use super::cli::ChronicleArgs;

const CANONICAL_AGENT_CAP: u32 = 8;
const SHOWCASE_AGENT_CAP: u32 = 32;
const EVENT_RING_CAP: usize = 1 << 16;

const CANONICAL_DEFAULT_SEED: &str = "0xD00DFACE00420042";
const CANONICAL_DEFAULT_TICKS: u32 = 100;
const SHOWCASE_DEFAULT_SEED: &str = "0xDEADBEEF";
const SHOWCASE_DEFAULT_TICKS: u32 = 500;

pub fn run_chronicle(args: ChronicleArgs) -> ExitCode {
    // Parse `--fixture` up front: it feeds both sweep and single-run
    // paths, and a typo should fail fast regardless of mode.
    let fixture = match FixtureKind::parse(&args.fixture) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("chronicle: invalid --fixture: {e}");
            return ExitCode::FAILURE;
        }
    };

    // `--gpu` resolves to a `Backend` here so every run path below can be
    // indifferent to whether the `gpu` feature was compiled in.
    // Phase 0: GpuBackend forwards to the CPU kernel, so `--gpu` output is
    // byte-identical to the default. The flag is gated on the `gpu` feature
    // so CPU-only builds don't silently ignore it.
    let backend = match resolve_backend(args.gpu) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("chronicle: {e}");
            return ExitCode::FAILURE;
        }
    };

    // Bench mode is orthogonal to sweep/showcase: it iterates *all*
    // fixture shapes internally. Reject the combo with --sweep to avoid
    // ambiguity ("am I benching the sweep or a single run?") — the bench
    // always builds a fresh SimState per sample, no sweep needed.
    if args.bench {
        if args.sweep.is_some() {
            eprintln!("chronicle: --bench cannot be combined with --sweep");
            return ExitCode::FAILURE;
        }
        if args.csv.is_some() {
            eprintln!("chronicle: --bench does not produce CSV output");
            return ExitCode::FAILURE;
        }
        // --gpu vs --bench: Phase 0's GpuBackend is a CPU-forwarding stub,
        // so a `--bench --gpu` comparison would measure noise. Reject the
        // combo until Phase 1+ lands real GPU dispatch.
        if args.gpu {
            eprintln!("chronicle: --gpu is not supported with --bench in Phase 0 (stub backend)");
            return ExitCode::FAILURE;
        }
        let mut stdout = std::io::stdout().lock();
        return run_bench(&mut stdout);
    }

    // Sweep mode short-circuits seed resolution: it has its own base-seed
    // knob and always runs a showcase fixture (`--fixture` picks which
    // preset). `--sweep 0` is a no-op that we surface as an error so
    // users don't get silent empty output.
    if let Some(runs) = args.sweep {
        if args.gpu {
            eprintln!("chronicle: --gpu is not supported with --sweep in Phase 0 (stub backend)");
            return ExitCode::FAILURE;
        }
        return run_sweep_from_args(&args, runs, fixture);
    }

    // `--csv` is a sweep-only side-channel; without `--sweep` there are no
    // per-run rows to write, so we reject rather than silently no-op.
    if args.csv.is_some() {
        eprintln!("chronicle: --csv requires --sweep");
        return ExitCode::FAILURE;
    }

    let (seed_raw, ticks) = resolve_args(&args);
    let seed = match parse_seed(&seed_raw) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("chronicle: invalid --seed: {e}");
            return ExitCode::FAILURE;
        }
    };

    if args.showcase {
        run_showcase(fixture, seed, ticks, backend)
    } else {
        run_canonical(seed, ticks, backend)
    }
}

/// Which tick driver a chronicle run uses. Resolved once in
/// `run_chronicle` from `--gpu` + whether the `gpu` feature was compiled in,
/// then threaded down through the single-run paths so each call site does
/// one match instead of re-checking the CLI flag.
///
/// Phase 0: both variants produce byte-identical chronicles — the Gpu arm
/// dispatches through `engine_gpu::GpuBackend`, which forwards to the CPU
/// kernel. Real GPU work lands in Phase 1+.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu,
}

/// Resolve the `--gpu` flag against the `gpu` feature. Returns
/// `Backend::Gpu` only when both are set; returns an error when `--gpu` is
/// passed to a CPU-only binary so the user doesn't get a silent fallback.
fn resolve_backend(gpu_flag: bool) -> Result<Backend, String> {
    if gpu_flag {
        #[cfg(feature = "gpu")]
        { Ok(Backend::Gpu) }
        #[cfg(not(feature = "gpu"))]
        { Err(String::from(
            "--gpu requires a build with the `gpu` feature (rebuild with \
             `cargo run --bin xtask --features gpu -- chronicle --gpu ...`)",
        )) }
    } else {
        Ok(Backend::Cpu)
    }
}

fn run_sweep_from_args(args: &ChronicleArgs, runs: u32, fixture: FixtureKind) -> ExitCode {
    if runs == 0 {
        eprintln!("chronicle: --sweep requires a positive run count");
        return ExitCode::FAILURE;
    }
    let base_seed_raw = args
        .base_seed
        .clone()
        .or_else(|| args.seed.clone())
        .unwrap_or_else(|| SHOWCASE_DEFAULT_SEED.to_string());
    let base_seed = match parse_seed(&base_seed_raw) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("chronicle: invalid --base-seed: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ticks = args.ticks.unwrap_or(SHOWCASE_DEFAULT_TICKS);

    let summary = run_sweep(
        fixture,
        base_seed,
        runs,
        ticks,
        args.verbose,
        &mut std::io::stderr(),
    );
    let mut stdout = std::io::stdout().lock();
    render_sweep_summary(&summary, &mut stdout).ok();

    // CSV side-channel: aggregates already went to stdout above; the CSV
    // carries the per-run detail for plotting without polluting the report.
    if let Some(ref path) = args.csv {
        if let Err(e) = write_sweep_csv(&summary, path) {
            eprintln!("chronicle: failed to write CSV to {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
        // Note to stdout (not stderr) — it's a successful side-effect of
        // the sweep, not a warning. Matches the aggregate report tone.
        let _ = writeln!(
            stdout,
            "CSV written to {} ({} rows)",
            path.display(),
            summary.per_run.len(),
        );
    }
    ExitCode::SUCCESS
}

fn resolve_args(args: &ChronicleArgs) -> (String, u32) {
    let seed = args.seed.clone().unwrap_or_else(|| {
        if args.showcase {
            SHOWCASE_DEFAULT_SEED.to_string()
        } else {
            CANONICAL_DEFAULT_SEED.to_string()
        }
    });
    let ticks = args.ticks.unwrap_or(if args.showcase {
        SHOWCASE_DEFAULT_TICKS
    } else {
        CANONICAL_DEFAULT_TICKS
    });
    (seed, ticks)
}

fn run_canonical(seed: u64, ticks: u32, backend: Backend) -> ExitCode {
    let mut state = spawn_canonical_fixture(seed);
    let events = simulate_with(&mut state, ticks, backend);

    // Walk every pushed event (the ring only evicts when it overflows, and
    // the chosen cap is far above our per-run volume). Render one line per
    // `ChronicleEntry`; non-chronicle events are skipped by the helper.
    let lines = chronicle::render_entries(&state, events.iter());
    println!(
        "# chronicle — seed={:#x} ticks={} agents={} chronicle_entries={}",
        seed,
        ticks,
        state.agent_cap(),
        lines.len()
    );
    for line in lines {
        println!("{line}");
    }
    ExitCode::SUCCESS
}

fn run_showcase(fixture: FixtureKind, seed: u64, ticks: u32, backend: Backend) -> ExitCode {
    let (mut state, counts) = spawn_fixture(fixture, seed);
    let events = simulate_with(&mut state, ticks, backend);

    let lines = chronicle::render_entries(&state, events.iter());
    let total_events = events.iter().count();
    let alive = alive_by_type(&state);

    // Header
    println!("=== Wolves + Humans Showcase ===");
    println!("Seed: {:#x}", seed);
    println!(
        "Agents: {} humans, {} wolves, {} deer",
        counts.humans, counts.wolves, counts.deer
    );
    println!("Initial positions: scattered across ~20×20 area (humans SE, wolves NW, deer center)");
    println!();

    // Chronicle body
    println!("--- Chronicle ---");
    if lines.is_empty() {
        println!("(no chronicle entries emitted)");
    } else {
        for line in &lines {
            println!("{line}");
        }
    }
    println!();

    // Outcome
    println!("--- Outcome ---");
    println!(
        "Alive: {} humans, {} wolves, {} deer (total {})",
        alive.humans,
        alive.wolves,
        alive.deer,
        alive.humans + alive.wolves + alive.deer,
    );
    println!(
        "Total events: {} ({} chronicle entries)",
        total_events,
        lines.len(),
    );
    println!("Duration: {} ticks", ticks);

    ExitCode::SUCCESS
}

// ---------------------------------------------------------------------------
// Balance sweep
// ---------------------------------------------------------------------------

/// Per-run statistics captured by the sweep. Aggregates are computed on a
/// `Vec<RunStats>` — keeping the raw per-run values around lets us compute
/// median / min / max without a second pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunStats {
    seed:             u64,
    humans_alive:     u32,
    wolves_alive:     u32,
    deer_alive:       u32,
    total_events:     u32,
    chronicle_count:  u32,
    rout_count:       u32,
    /// `None` if no agent died during the run.
    first_death_tick: Option<u32>,
    winner:           Winner,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Winner {
    Humans,
    Wolves,
    Deer,
    Stalemate,
}

impl Winner {
    /// Winner = the species with strictly the most alive agents at the end.
    /// Ties are "stalemate" so "wolves won" in the report means wolves
    /// outnumbered everyone alive — mere survival isn't a win.
    fn from_counts(c: &SpawnCounts) -> Self {
        let (h, w, d) = (c.humans, c.wolves, c.deer);
        let max = h.max(w).max(d);
        if max == 0 {
            return Winner::Stalemate;
        }
        let tied = [h, w, d].iter().filter(|&&n| n == max).count();
        if tied > 1 {
            return Winner::Stalemate;
        }
        if h == max {
            Winner::Humans
        } else if w == max {
            Winner::Wolves
        } else {
            Winner::Deer
        }
    }
}

/// Run the showcase fixture once and return aggregate stats. Does not
/// print anything — the caller decides whether the run goes into a sweep
/// report, a verbose one-liner, or both.
fn run_showcase_stats(fixture: FixtureKind, seed: u64, ticks: u32) -> RunStats {
    let (mut state, _counts) = spawn_fixture(fixture, seed);
    let events = simulate(&mut state, ticks);

    let mut total_events: u32 = 0;
    let mut chronicle_count: u32 = 0;
    let mut rout_count: u32 = 0;
    let mut first_death_tick: Option<u32> = None;

    for ev in events.iter() {
        total_events = total_events.saturating_add(1);
        match ev {
            engine::event::Event::ChronicleEntry { template_id, .. } => {
                chronicle_count = chronicle_count.saturating_add(1);
                if *template_id == chronicle::templates::ROUT {
                    rout_count = rout_count.saturating_add(1);
                }
            }
            engine::event::Event::AgentDied { tick, .. } => {
                if first_death_tick.map_or(true, |t| *tick < t) {
                    first_death_tick = Some(*tick);
                }
            }
            _ => {}
        }
    }

    let alive = alive_by_type(&state);
    let winner = Winner::from_counts(&alive);

    RunStats {
        seed,
        humans_alive: alive.humans,
        wolves_alive: alive.wolves,
        deer_alive: alive.deer,
        total_events,
        chronicle_count,
        rout_count,
        first_death_tick,
        winner,
    }
}

/// Aggregate summary rendered by the sweep command. Separating the
/// computation from the rendering keeps the determinism test simple
/// (construct an expected `SweepSummary`, compare by field).
#[derive(Debug, Clone)]
struct SweepSummary {
    runs:        u32,
    ticks:       u32,
    base_seed:   u64,
    /// Which fixture preset produced these runs. Threaded through so the
    /// header can advertise it — the aggregate stats alone can't tell
    /// showcase from balanced.
    fixture:     FixtureKind,
    /// Retained for introspection tests and CSV export (`--csv` flag).
    /// Not read by the text renderer (which only needs the pre-computed
    /// aggregates).
    per_run:     Vec<RunStats>,
    // Pre-computed aggregates
    humans_wins: u32,
    wolves_wins: u32,
    deer_wins:   u32,
    stalemates:  u32,
    // Deer are the prey — "deer survived" (any alive at end) is more
    // meaningful than "deer won", which almost never happens.
    deer_survived_runs: u32,
    // Rout cascade: number of runs that saw any ROUT entry, plus the mean
    // ROUT count across those cascade runs (not the full sweep).
    runs_with_rout:        u32,
    cascade_mean_routs:    f64,
    // Total events: mean + min/max range
    total_events_mean: f64,
    total_events_min:  u32,
    total_events_max:  u32,
    // Chronicle entries: mean + min/max range
    chronicle_mean: f64,
    chronicle_min:  u32,
    chronicle_max:  u32,
    // Rout events across all runs (not just cascade runs)
    rout_mean: f64,
    rout_min:  u32,
    rout_max:  u32,
    // First-death tick: mean / min / max across runs that had any death
    first_death_mean: Option<f64>,
    first_death_min:  Option<u32>,
    first_death_max:  Option<u32>,
    // Survivor stats by species (mean, median, stdev)
    humans_mean:   f64,
    humans_median: f64,
    humans_stdev:  f64,
    wolves_mean:   f64,
    wolves_median: f64,
    wolves_stdev:  f64,
    deer_mean:     f64,
    deer_median:   f64,
    deer_stdev:    f64,
}

fn summarize(
    per_run: Vec<RunStats>,
    runs: u32,
    ticks: u32,
    base_seed: u64,
    fixture: FixtureKind,
) -> SweepSummary {
    let n = per_run.len() as f64;
    let mut humans_wins = 0u32;
    let mut wolves_wins = 0u32;
    let mut deer_wins = 0u32;
    let mut stalemates = 0u32;
    let mut deer_survived_runs = 0u32;
    let mut runs_with_rout = 0u32;
    let mut cascade_rout_sum: u64 = 0;
    for r in &per_run {
        match r.winner {
            Winner::Humans => humans_wins += 1,
            Winner::Wolves => wolves_wins += 1,
            Winner::Deer => deer_wins += 1,
            Winner::Stalemate => stalemates += 1,
        }
        if r.deer_alive > 0 {
            deer_survived_runs += 1;
        }
        if r.rout_count > 0 {
            runs_with_rout += 1;
            cascade_rout_sum += r.rout_count as u64;
        }
    }

    let cascade_mean_routs = if runs_with_rout > 0 {
        cascade_rout_sum as f64 / runs_with_rout as f64
    } else {
        0.0
    };

    let humans_vals: Vec<u32> = per_run.iter().map(|r| r.humans_alive).collect();
    let wolves_vals: Vec<u32> = per_run.iter().map(|r| r.wolves_alive).collect();
    let deer_vals: Vec<u32> = per_run.iter().map(|r| r.deer_alive).collect();
    let events_vals: Vec<u32> = per_run.iter().map(|r| r.total_events).collect();
    let chronicle_vals: Vec<u32> = per_run.iter().map(|r| r.chronicle_count).collect();
    let rout_vals: Vec<u32> = per_run.iter().map(|r| r.rout_count).collect();
    let first_death_vals: Vec<u32> =
        per_run.iter().filter_map(|r| r.first_death_tick).collect();

    let (humans_mean, humans_median, humans_stdev) = mean_median_stdev(&humans_vals);
    let (wolves_mean, wolves_median, wolves_stdev) = mean_median_stdev(&wolves_vals);
    let (deer_mean, deer_median, deer_stdev) = mean_median_stdev(&deer_vals);

    let total_events_mean = events_vals.iter().map(|v| *v as f64).sum::<f64>() / n.max(1.0);
    let total_events_min = *events_vals.iter().min().unwrap_or(&0);
    let total_events_max = *events_vals.iter().max().unwrap_or(&0);

    let chronicle_mean =
        chronicle_vals.iter().map(|v| *v as f64).sum::<f64>() / n.max(1.0);
    let chronicle_min = *chronicle_vals.iter().min().unwrap_or(&0);
    let chronicle_max = *chronicle_vals.iter().max().unwrap_or(&0);

    let rout_mean = rout_vals.iter().map(|v| *v as f64).sum::<f64>() / n.max(1.0);
    let rout_min = *rout_vals.iter().min().unwrap_or(&0);
    let rout_max = *rout_vals.iter().max().unwrap_or(&0);

    let (first_death_mean, first_death_min, first_death_max) = if first_death_vals.is_empty() {
        (None, None, None)
    } else {
        let sum: u64 = first_death_vals.iter().map(|v| *v as u64).sum();
        let mean = sum as f64 / first_death_vals.len() as f64;
        let min = *first_death_vals.iter().min().unwrap();
        let max = *first_death_vals.iter().max().unwrap();
        (Some(mean), Some(min), Some(max))
    };

    SweepSummary {
        runs,
        ticks,
        base_seed,
        fixture,
        per_run,
        humans_wins,
        wolves_wins,
        deer_wins,
        stalemates,
        deer_survived_runs,
        runs_with_rout,
        cascade_mean_routs,
        total_events_mean,
        total_events_min,
        total_events_max,
        chronicle_mean,
        chronicle_min,
        chronicle_max,
        rout_mean,
        rout_min,
        rout_max,
        first_death_mean,
        first_death_min,
        first_death_max,
        humans_mean,
        humans_median,
        humans_stdev,
        wolves_mean,
        wolves_median,
        wolves_stdev,
        deer_mean,
        deer_median,
        deer_stdev,
    }
}

/// Mean, median, population stdev for a slice of u32s. Population (not
/// sample) stdev — we have the full population of runs, not a sample.
fn mean_median_stdev(vals: &[u32]) -> (f64, f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = vals.len() as f64;
    let sum: u64 = vals.iter().map(|v| *v as u64).sum();
    let mean = sum as f64 / n;
    let mut sorted: Vec<u32> = vals.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    let median = if sorted.len() % 2 == 0 {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) / 2.0
    } else {
        sorted[mid] as f64
    };
    let var = vals.iter().map(|v| {
        let d = *v as f64 - mean;
        d * d
    }).sum::<f64>() / n;
    (mean, median, var.sqrt())
}

/// Drive N runs with `base_seed + k` for k = 0..N, emitting a progress dot
/// per 10 runs to stderr and optionally a per-run one-liner when verbose.
/// Returns the finished summary; rendering happens in the caller.
fn run_sweep<W: std::io::Write>(
    fixture: FixtureKind,
    base_seed: u64,
    runs: u32,
    ticks: u32,
    verbose: bool,
    progress: &mut W,
) -> SweepSummary {
    let mut per_run = Vec::with_capacity(runs as usize);
    for k in 0..runs {
        // Stepping by 1 is intentional — SimState hashes the seed before use
        // (see `SimState::new`), so neighbouring seeds produce independent
        // rng streams. No need to jitter the base_seed by 2^32.
        let seed = base_seed.wrapping_add(k as u64);
        let stats = run_showcase_stats(fixture, seed, ticks);
        if verbose {
            let _ = writeln!(progress, "{}", format_run_line(k, &stats));
        }
        if !verbose && runs >= 10 && (k + 1) % 10 == 0 {
            let _ = write!(progress, ".");
            let _ = progress.flush();
        }
        per_run.push(stats);
    }
    if !verbose && runs >= 10 {
        let _ = writeln!(progress);
    }
    summarize(per_run, runs, ticks, base_seed, fixture)
}

fn format_run_line(k: u32, s: &RunStats) -> String {
    let winner_word = match s.winner {
        Winner::Humans => "humans win",
        Winner::Wolves => "wolves win",
        Winner::Deer => "deer win",
        Winner::Stalemate => "stalemate",
    };
    format!(
        "Run {:>4} [seed {:#018x}]: {}, H{}/W{}/D{}, {} events, {} chronicle, {} routs",
        k, s.seed, winner_word,
        s.humans_alive, s.wolves_alive, s.deer_alive,
        s.total_events, s.chronicle_count, s.rout_count,
    )
}

fn render_sweep_summary<W: std::io::Write>(
    s: &SweepSummary,
    out: &mut W,
) -> std::io::Result<()> {
    let runs = s.runs as f64;
    let pct = |n: u32| -> f64 {
        if s.runs == 0 { 0.0 } else { 100.0 * n as f64 / runs }
    };

    writeln!(out, "=== Balance Sweep: {} seeds × {} ticks ===", s.runs, s.ticks)?;
    writeln!(out, "Fixture: {}", s.fixture.description())?;
    writeln!(out, "Base seed: {:#018x} (stepped by 1 per run)", s.base_seed)?;
    writeln!(out)?;

    writeln!(out, "--- Outcomes ---")?;
    writeln!(out, "Humans won:    {:>3}/{} ({:.0}%)", s.humans_wins, s.runs, pct(s.humans_wins))?;
    writeln!(out, "Wolves won:    {:>3}/{} ({:.0}%)", s.wolves_wins, s.runs, pct(s.wolves_wins))?;
    writeln!(out, "Deer won:      {:>3}/{} ({:.0}%)", s.deer_wins, s.runs, pct(s.deer_wins))?;
    writeln!(out, "Stalemate:     {:>3}/{} ({:.0}%)", s.stalemates, s.runs, pct(s.stalemates))?;
    writeln!(out, "Deer survived: {:>3}/{} ({:.0}% of runs had >= 1 deer alive)",
        s.deer_survived_runs, s.runs, pct(s.deer_survived_runs))?;
    writeln!(out)?;

    writeln!(out, "--- Survivors (mean / median / stdev) ---")?;
    writeln!(out, "Humans: {:>5.1} / {:>4.1} / {:>4.2}", s.humans_mean, s.humans_median, s.humans_stdev)?;
    writeln!(out, "Wolves: {:>5.1} / {:>4.1} / {:>4.2}", s.wolves_mean, s.wolves_median, s.wolves_stdev)?;
    writeln!(out, "Deer:   {:>5.1} / {:>4.1} / {:>4.2}", s.deer_mean, s.deer_median, s.deer_stdev)?;
    writeln!(out)?;

    writeln!(out, "--- Combat dynamics ---")?;
    match (s.first_death_mean, s.first_death_min, s.first_death_max) {
        (Some(mean), Some(min), Some(max)) => {
            writeln!(out, "First death tick: mean {:.1}, min {}, max {}", mean, min, max)?;
        }
        _ => {
            writeln!(out, "First death tick: (no deaths in any run)")?;
        }
    }
    writeln!(out, "Chronicle entries: mean {:.1} (range: {} - {})",
        s.chronicle_mean, s.chronicle_min, s.chronicle_max)?;
    writeln!(out, "Rout events (fear cascade): mean {:.2} (range: {} - {})",
        s.rout_mean, s.rout_min, s.rout_max)?;
    writeln!(out, "Runs with any rout: {}/{} ({:.0}%)",
        s.runs_with_rout, s.runs, pct(s.runs_with_rout))?;
    if s.runs_with_rout > 0 {
        writeln!(out, "Cascade depth (mean routs per cascade run): {:.2}", s.cascade_mean_routs)?;
    }
    writeln!(out)?;

    writeln!(out, "--- Total events ---")?;
    writeln!(out, "Mean {:.0} events per run (range: {} - {})",
        s.total_events_mean, s.total_events_min, s.total_events_max)?;

    // A trailing newline keeps the output tidy when piped into `| head`.
    Ok(())
}

/// CSV header for the sweep export. Kept as a const so the test can
/// assert it without reimplementing the column list.
const SWEEP_CSV_HEADER: &str =
    "seed,alive_humans,alive_wolves,alive_deer,winner,total_events,chronicle_entries,rout_count,first_death_tick";

/// Textual winner name, matching the order documented in the `--csv`
/// flag help. Keeping this as a free fn (not `Display`) keeps the CSV
/// serialization independent from any future pretty-print formatting.
fn winner_csv(w: Winner) -> &'static str {
    match w {
        Winner::Humans => "humans",
        Winner::Wolves => "wolves",
        Winner::Deer => "deer",
        Winner::Stalemate => "stalemate",
    }
}

/// Serialize one sweep run as a CSV row. All fields are numeric or from a
/// closed vocabulary (winner), so no quoting or escaping is needed; the
/// seed is hex-prefixed to match the aggregate report, and
/// `first_death_tick` is blank when the run saw no deaths (matches pandas'
/// NaN-on-read default — `pd.read_csv` turns the empty field into NaN).
fn write_sweep_csv_row<W: Write>(out: &mut W, r: &RunStats) -> std::io::Result<()> {
    let first_death = match r.first_death_tick {
        Some(t) => t.to_string(),
        None => String::new(),
    };
    writeln!(
        out,
        "{:#018x},{},{},{},{},{},{},{},{}",
        r.seed,
        r.humans_alive,
        r.wolves_alive,
        r.deer_alive,
        winner_csv(r.winner),
        r.total_events,
        r.chronicle_count,
        r.rout_count,
        first_death,
    )
}

/// Write the full sweep (header + one row per run) to `path`. We create
/// the file outright (overwriting any existing one) — the user passed a
/// path, they own it. Parent-directory existence is checked explicitly so
/// a typo like `--csv typo/out.csv` fails fast instead of creating a
/// confusing "No such file or directory" chain deep in `File::create`.
fn write_sweep_csv(summary: &SweepSummary, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        // An empty parent component ("") means "current dir" — always
        // exists; skip the check. Non-empty parents must exist.
        if !parent.as_os_str().is_empty() && !parent.is_dir() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "parent directory does not exist: {}",
                    parent.display()
                ),
            ));
        }
    }
    let file = std::fs::File::create(path)?;
    let mut buf = std::io::BufWriter::new(file);
    writeln!(buf, "{}", SWEEP_CSV_HEADER)?;
    for r in &summary.per_run {
        write_sweep_csv_row(&mut buf, r)?;
    }
    buf.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Perf benchmark
// ---------------------------------------------------------------------------

/// Ticks per timed sample. Large enough to amortise fixture-spawn cost
/// (tens of ms) but small enough that a 5-sample bench across 3 fixtures
/// finishes in a few seconds on release. Kept as a const so the test can
/// pass a smaller value via `run_bench_shape` without duplicating knobs.
const BENCH_TICKS_PER_SAMPLE: u32 = 500;
/// Throwaway runs that warm the page cache + branch predictors before
/// timing starts. Three is overkill for a ~10ms run but cheap enough to
/// keep — the first run is consistently ~20% slower than steady state.
const BENCH_WARMUP_RUNS: u32 = 3;
/// Timed samples per fixture. Odd count keeps the median well-defined
/// without a midpoint average. Five is the minimum for a meaningful
/// p5/p95 estimate; raising this would lengthen the bench without moving
/// the headline numbers much.
const BENCH_SAMPLES: u32 = 5;

/// One of the fixture shapes the bench probes. Kept as `(label, build_fn)`
/// pairs so the bench loop is oblivious to fixture layout — any new
/// preset just drops into `BENCH_SHAPES`.
struct BenchShape {
    /// Short label for the per-fixture section header.
    label:       &'static str,
    /// One-line prose description — counts, seed, HP tweaks, etc. Printed
    /// immediately under the section header so the reader can correlate
    /// "showcase" → "8+8+4 agents, seed-jittered" without cross-referencing.
    description: &'static str,
    /// Construct a fresh `SimState` for one sample. Called once per warmup
    /// and once per timed sample — the benchmark never reuses state, so
    /// per-tick numbers reflect cold-start cost without memo/cache effects.
    build:       fn() -> SimState,
}

fn build_canonical_bench() -> SimState {
    spawn_canonical_fixture(0xD00DFACE00420042)
}

fn build_showcase_bench() -> SimState {
    spawn_fixture(FixtureKind::Showcase, 0xDEADBEEF).0
}

fn build_balanced_bench() -> SimState {
    spawn_fixture(FixtureKind::Balanced, 0xDEADBEEF).0
}

/// Fixture shapes the bench iterates, in printed order. Canonical first
/// because it's the smallest/fastest — reader gets a "here's the floor"
/// number before seeing the heavier presets.
const BENCH_SHAPES: &[BenchShape] = &[
    BenchShape {
        label:       "canonical",
        description: "3 humans + 2 wolves, seed=0xD00DFACE00420042",
        build:       build_canonical_bench,
    },
    BenchShape {
        label:       "showcase",
        description: "8 humans + 8 wolves + 4 deer, seed=0xDEADBEEF (seed-jittered)",
        build:       build_showcase_bench,
    },
    BenchShape {
        label:       "balanced",
        description: "6 humans @85HP + 10 wolves @95HP + 4 deer, seed=0xDEADBEEF",
        build:       build_balanced_bench,
    },
];

/// Numeric result of one fixture's benchmark — raw enough that the
/// renderer can format the same data any way it wants. Durations live in
/// nanoseconds so the percentile math stays integer-stable.
#[derive(Debug, Clone)]
struct BenchShapeResult {
    label:          &'static str,
    description:    &'static str,
    ticks:          u32,
    /// Wall-clock nanoseconds per 500-tick sample, sorted ascending.
    /// Sample count lives implicitly in `sample_ns.len()` — kept here
    /// so the renderer can format "N samples" without threading the
    /// knob separately.
    sample_ns:      Vec<u128>,
    /// Events observed per sample (total `events.len()` after 500 ticks),
    /// sorted ascending. Parallel to `sample_ns` in bucketing only —
    /// both are sorted independently, the renderer just reports the median.
    events_sorted:  Vec<u64>,
}

impl BenchShapeResult {
    /// Linear-interp percentile on sorted samples. `frac` in [0, 1].
    /// With N=5, frac=0 → min (p0), frac=0.5 → sorted[2] (median),
    /// frac=1 → max (p100). For p5/p95 we use 0.05 / 0.95 which on N=5
    /// still lands at the endpoints after rounding, so the report calls
    /// them "p5/p95 (approx — N=5)" rather than claiming precision we don't
    /// have. Kept as linear-interp anyway so raising `BENCH_SAMPLES` just
    /// sharpens the report without touching this code.
    fn percentile_ns(&self, frac: f64) -> u128 {
        linear_interp_percentile_u128(&self.sample_ns, frac)
    }

    fn median_ns(&self) -> u128 {
        self.percentile_ns(0.5)
    }

    fn median_events(&self) -> u64 {
        // `percentile_u64` at 0.5 matches the median used for sample_ns so
        // the two columns are consistent.
        linear_interp_percentile_u64(&self.events_sorted, 0.5)
    }
}

/// Linear interpolation percentile for a pre-sorted slice of u128 values.
/// Returns 0 for empty input (the bench never produces empty samples, but
/// defending against that lets the unit test cover the edge).
///
/// Uses the "C=1" variant (index = frac * (N - 1)): no out-of-range
/// extrapolation, endpoints map exactly to min/max, and for N=5 the p5/p95
/// with frac=0.05/0.95 land at the bracketed pair — mostly the min/max with
/// a sliver of interpolation toward the next value.
fn linear_interp_percentile_u128(sorted: &[u128], frac: f64) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let f = frac.clamp(0.0, 1.0);
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = f * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let t = idx - lo as f64;
    let a = sorted[lo] as f64;
    let b = sorted[hi] as f64;
    (a + (b - a) * t).round() as u128
}

/// Mirror of [`linear_interp_percentile_u128`] for u64 event counts.
/// Separate function (not generic) because the round-trip through f64
/// on very large u128 values loses precision, and we want the u128 path
/// to stay unrounded above ~2^53 ns — a 5s-per-sample bench would trip
/// that otherwise. u64 events cap well under 2^53 so the f64 path is fine.
fn linear_interp_percentile_u64(sorted: &[u64], frac: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let f = frac.clamp(0.0, 1.0);
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = f * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let t = idx - lo as f64;
    let a = sorted[lo] as f64;
    let b = sorted[hi] as f64;
    (a + (b - a) * t).round() as u64
}

/// Run one fixture through warmup + timed samples. Each sample rebuilds
/// the `SimState` from scratch (via `shape.build()`) so per-sample timings
/// reflect realistic per-tick cost without any cross-sample memoisation.
fn run_bench_shape(shape: &BenchShape, warmup: u32, samples: u32, ticks: u32) -> BenchShapeResult {
    // Warmup: run and discard. We don't even measure — the goal is just to
    // prime page cache + branch predictors so sample #1 isn't an outlier.
    for _ in 0..warmup {
        let mut state = (shape.build)();
        let _ = simulate(&mut state, ticks);
    }

    let mut sample_ns = Vec::with_capacity(samples as usize);
    let mut events_sorted = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let mut state = (shape.build)();
        let t0 = std::time::Instant::now();
        let events = simulate(&mut state, ticks);
        let elapsed = t0.elapsed().as_nanos();
        // `events.iter().count()` walks the ring — the ring is bounded and
        // we configured EVENT_RING_CAP well above one sample's volume, so
        // this is "total events this sample" with no eviction.
        let n_events = events.iter().count() as u64;
        sample_ns.push(elapsed);
        events_sorted.push(n_events);
    }

    sample_ns.sort_unstable();
    events_sorted.sort_unstable();

    // `samples` isn't stored on the result — it's `sample_ns.len()`.
    // Silence the unused-warning by binding it explicitly.
    let _ = samples;

    BenchShapeResult {
        label: shape.label,
        description: shape.description,
        ticks,
        sample_ns,
        events_sorted,
    }
}

/// Format a ticks-per-second rate to 1 decimal with thousands separators.
/// "45231.2" → "45,231.2". Kept local (not pulled in as a dep) because
/// this is the only place in the bench that needs grouping.
fn format_tps(tps: f64) -> String {
    // Split on the decimal point; group the integer part in threes.
    let s = format!("{:.1}", tps);
    let (int_part, frac_part) = match s.split_once('.') {
        Some((i, f)) => (i, f),
        None => (s.as_str(), "0"),
    };
    // Handle negative sign if any (shouldn't occur here, but defensively).
    let (sign, digits) = if let Some(rest) = int_part.strip_prefix('-') {
        ("-", rest)
    } else {
        ("", int_part)
    };
    let bytes = digits.as_bytes();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        let remaining = bytes.len() - i;
        grouped.push(*b as char);
        if remaining > 1 && remaining % 3 == 1 {
            grouped.push(',');
        }
    }
    format!("{sign}{grouped}.{frac_part}")
}

/// Render one fixture's bench result as the per-fixture section. Takes a
/// `Write` sink so the tests can inspect the output without going through
/// stdout.
fn render_bench_shape<W: std::io::Write>(
    out: &mut W,
    r: &BenchShapeResult,
) -> std::io::Result<()> {
    let median_ns = r.median_ns();
    let p5_ns = r.percentile_ns(0.05);
    let p95_ns = r.percentile_ns(0.95);
    // ns → seconds → ticks/sec. Divide-by-zero guarded just in case a
    // sample underflows (e.g. if a future cheap fixture ran in < 1ns,
    // which this code would then misreport anyway).
    let ns_to_tps = |ns: u128| -> f64 {
        if ns == 0 {
            return 0.0;
        }
        (r.ticks as f64) * 1_000_000_000.0 / (ns as f64)
    };
    let ns_to_ms = |ns: u128| -> f64 { ns as f64 / 1_000_000.0 };

    let median_tps = ns_to_tps(median_ns);
    // Note the swap: *fastest* run (min ns, `p5_ns` on sorted asc) is the
    // p95 ticks/sec, because ticks/sec is inverse to wall-clock. We print
    // the tps column sorted so higher tps lines up with fastest sample.
    let p5_tps = ns_to_tps(p95_ns);
    let p95_tps = ns_to_tps(p5_ns);

    writeln!(out, "--- {} ({}) ---", r.label, r.description)?;
    writeln!(
        out,
        "median: {:>14} ticks/sec  ({:.1} ms/{} ticks)",
        format_tps(median_tps),
        ns_to_ms(median_ns),
        r.ticks,
    )?;
    writeln!(
        out,
        "p5/p95: {:>14} / {:<12}  ({:.1} / {:.1} ms)",
        format_tps(p5_tps),
        format_tps(p95_tps),
        ns_to_ms(p95_ns),
        ns_to_ms(p5_ns),
    )?;
    let median_evs = r.median_events();
    let evs_per_tick = median_evs as f64 / r.ticks as f64;
    writeln!(
        out,
        "events/tick: {:.1} median ({} events across {} ticks)",
        evs_per_tick, median_evs, r.ticks,
    )?;
    writeln!(out)?;
    Ok(())
}

/// Render the full bench report: preamble + per-fixture sections. Returns
/// `ExitCode::SUCCESS` on success; IO errors bubble up via `?` and are
/// converted to `FAILURE` in `run_bench` (stdout writes failing is already
/// pathological).
fn render_bench_report<W: std::io::Write>(
    out: &mut W,
    results: &[BenchShapeResult],
) -> std::io::Result<()> {
    writeln!(out, "=== Perf Benchmark ===")?;
    let build_mode = if cfg!(debug_assertions) { "DEBUG" } else { "release" };
    writeln!(
        out,
        "Rust: {} build, {} samples + {} warmup, {} ticks per run",
        build_mode, BENCH_SAMPLES, BENCH_WARMUP_RUNS, BENCH_TICKS_PER_SAMPLE,
    )?;
    if cfg!(debug_assertions) {
        writeln!(out)?;
        writeln!(
            out,
            "!! WARNING: bench in debug mode; times are 10-50x slower than release.",
        )?;
        writeln!(
            out,
            "!!          Rerun with `cargo run --bin xtask --release -- chronicle --bench`.",
        )?;
    }
    writeln!(out)?;
    for r in results {
        render_bench_shape(out, r)?;
    }
    Ok(())
}

/// Drive the full benchmark across every shape in `BENCH_SHAPES`. Writes
/// progress dots to stderr (one per sample × shape) so long benches don't
/// look hung, then prints the aggregate report to `out`.
fn run_bench<W: std::io::Write>(out: &mut W) -> ExitCode {
    let mut results = Vec::with_capacity(BENCH_SHAPES.len());
    for shape in BENCH_SHAPES {
        eprint!("benching {}... ", shape.label);
        let _ = std::io::stderr().flush();
        let r = run_bench_shape(shape, BENCH_WARMUP_RUNS, BENCH_SAMPLES, BENCH_TICKS_PER_SAMPLE);
        eprintln!("done");
        results.push(r);
    }
    if let Err(e) = render_bench_report(out, &results) {
        eprintln!("chronicle: failed to write bench report: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn simulate(state: &mut SimState, ticks: u32) -> EventRing {
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    let mut invariants = InvariantRegistry::new();
    invariants.register(Box::new(PoolNonOverlapInvariant));

    let mut views: Vec<&mut dyn MaterializedView> = Vec::new();
    let telemetry = NullSink;

    for _ in 0..ticks {
        step_full(
            state,
            &mut scratch,
            &mut events,
            &UtilityBackend,
            &cascade,
            &mut views[..],
            &invariants,
            &telemetry,
        );
    }
    events
}

/// Backend-aware variant of [`simulate`]. `Backend::Cpu` delegates to the
/// canonical `simulate` function (6-phase `step_full` with
/// `PoolNonOverlapInvariant`, empty views, `NullSink`). `Backend::Gpu`
/// drives the `engine_gpu::GpuBackend` trait impl, which in Phase 0
/// forwards to `engine::step::step` — observationally identical because
/// `step::step` runs `step_full` internally with empty views, an empty
/// invariant registry, and a `NullSink`. Invariant checks are non-mutating
/// and every registered invariant so far (`PoolNonOverlapInvariant`) only
/// emits telemetry on violation, so dropping them doesn't change the
/// event log.
fn simulate_with(state: &mut SimState, ticks: u32, backend: Backend) -> EventRing {
    match backend {
        Backend::Cpu => simulate(state, ticks),
        #[cfg(feature = "gpu")]
        // Phase 1: `GpuBackend::new()` now returns a `Result` because
        // wgpu device creation can fail (no GPU adapter / driver).
        // `expect` is fine here — the xtask chronicle CLI is a
        // developer tool; surfacing the wgpu error message via panic
        // beats silently degrading to CPU.
        Backend::Gpu => simulate_via_trait(
            state,
            ticks,
            engine_gpu::GpuBackend::new().expect("GpuBackend init"),
        ),
    }
}

/// Run `ticks` ticks through a [`engine::backend::SimBackend`] implementation.
/// Kept separate from `simulate` so the trait call path is visible — the
/// Phase 0 parity harness exercises this function indirectly via the `--gpu`
/// flag. The body mirrors `step::step`'s internal setup (empty view list,
/// no invariants, `NullSink`) rather than `simulate`'s (with
/// `PoolNonOverlapInvariant` registered); the difference is unobservable in
/// the event log because the invariant is non-mutating and the telemetry
/// sink is null.
#[cfg(feature = "gpu")]
fn simulate_via_trait<B: engine::backend::SimBackend>(
    state:       &mut SimState,
    ticks:       u32,
    mut backend: B,
) -> EventRing {
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    for _ in 0..ticks {
        backend.step(state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    events
}

/// Spawn the canonical wolves+humans fixture. Kept in sync with
/// `tests/wolves_and_humans_parity.rs::spawn_fixture` — the two fixtures
/// share a seed-parity contract (same seed + same ticks ⇒ same chronicle).
fn spawn_canonical_fixture(seed: u64) -> SimState {
    let mut state = SimState::new(CANONICAL_AGENT_CAP, seed);
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 2 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(-2.0, 0.0, 0.0),
            hp: 100.0,
            ..Default::default()
        })
        .expect("human 3 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 1 spawn");
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(-3.0, 0.0, 0.0),
            hp: 80.0,
            ..Default::default()
        })
        .expect("wolf 2 spawn");
    state
}

/// Constant mixed into the showcase seed before deriving spawn-jitter RNG.
/// Keeps the spawn PRNG stream independent from the sim's rng (which hashes
/// the raw seed) — same `seed` always yields the same spawn positions, but
/// we don't borrow bits out of the sim's rng stream.
const SPAWN_JITTER_SEED_XOR: u64 = 0x5FA57_000D_C0FFEE5;

/// Maximum per-axis jitter applied to each showcase spawn, in world units.
/// Clusters sit ~15 units apart so ±2.5 in x/z meaningfully reorders
/// who-meets-whom first without letting clusters collide at spawn.
const SPAWN_JITTER_AMPLITUDE: f32 = 2.5;

/// xorshift64* — tiny deterministic PRNG used only for spawn-position
/// jitter. We don't touch `SimState.rng_state` here: that stream drives
/// the simulation proper, and sharing it would couple spawn layout to
/// sim determinism in fragile ways.
#[derive(Debug, Clone, Copy)]
struct SpawnRng(u64);

impl SpawnRng {
    fn new(seed: u64) -> Self {
        // xorshift64 degenerates at state=0; salt any zero seed to a
        // non-zero constant so `seed ^ SPAWN_JITTER_SEED_XOR == 0` still
        // produces a usable stream.
        let mut s = seed ^ SPAWN_JITTER_SEED_XOR;
        if s == 0 {
            s = 0x9E37_79B9_7F4A_7C15;
        }
        Self(s)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f32 in [-amp, amp]. Uses the upper 24 bits of the state,
    /// which is where xorshift64 has the best statistical properties.
    fn jitter(&mut self, amp: f32) -> f32 {
        let bits = (self.next_u64() >> 40) as u32; // 24 bits
        let unit = (bits as f32) / ((1u32 << 24) as f32); // [0, 1)
        (unit * 2.0 - 1.0) * amp
    }
}

/// Which curated fixture to instantiate. Separated from `ChronicleArgs` so
/// tests can construct a `FixtureKind::Balanced` directly without building
/// a full CLI arg struct, and so adding a third preset doesn't ripple
/// through the spawn/sweep call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixtureKind {
    /// Default curated narrative: 8 humans @100HP vs 8 wolves @80HP + 4
    /// deer. Strongly human-favoured (~98% humans win at 50 seeds). The
    /// clustered layout and counts are frozen — every other fixture is a
    /// perturbation of this baseline.
    Showcase,
    /// Wolf-favoured preset tuned for mixed 40-60% outcomes across seeds.
    /// Iterated empirically on the sweep CSV (task 176). Flips the HP
    /// asymmetry (wolves 100, humans 80) and gives the pack a count
    /// advantage (10 vs 6) to counter the cluster-convergence that
    /// otherwise always favours the humans — the single lever we're
    /// allowed to pull per task spec.
    Balanced,
}

impl FixtureKind {
    /// Parse the CLI `--fixture <name>` flag. The parser lives here (not on
    /// the `clap` arg) so the vocabulary stays next to the enum.
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "showcase" => Ok(Self::Showcase),
            "balanced" => Ok(Self::Balanced),
            other => Err(format!(
                "unknown fixture {other:?}: expected 'showcase' or 'balanced'"
            )),
        }
    }

    /// Human-readable label for the sweep header so the aggregate report
    /// advertises which preset the numbers came from.
    fn description(self) -> &'static str {
        match self {
            Self::Showcase => "8 humans + 8 wolves + 4 deer (showcase)",
            Self::Balanced => "6 humans @85HP + 10 wolves @95HP + 4 deer (balanced)",
        }
    }
}

/// Declarative spec for a cluster of agents to spawn: shared creature type
/// and HP, with per-agent base positions. Each base gets seed-derived
/// x/z jitter applied at spawn time (see `SpawnRng`).
///
/// Modelling the fixture as data (rather than a hand-rolled function per
/// preset) lets us add new presets by extending one table and keeps the
/// spawn loop identical across variants — no behavioural drift.
struct ClusterSpec {
    creature: CreatureType,
    hp:       f32,
    bases:    &'static [Vec3],
    /// Short label used in panic messages if the engine rejects a spawn.
    /// Surfaces "human 3 spawn" instead of a generic index.
    label:    &'static str,
}

/// Showcase cluster bases — frozen from the original task-171 showcase so
/// `--showcase` (no `--fixture`) keeps its byte-for-byte behaviour. Humans
/// SE (+6,+6), wolves NW (-6,-6), deer centred.
const SHOWCASE_HUMAN_BASES: &[Vec3] = &[
    Vec3::new(6.0, 0.0, 6.0),
    Vec3::new(9.0, 0.0, 6.0),
    Vec3::new(6.0, 0.0, 9.0),
    Vec3::new(3.0, 0.0, 6.0),
    Vec3::new(6.0, 0.0, 3.0),
    Vec3::new(9.0, 0.0, 9.0),
    Vec3::new(3.0, 0.0, 3.0),
    Vec3::new(8.0, 0.0, 4.0),
];
const SHOWCASE_WOLF_BASES: &[Vec3] = &[
    Vec3::new(-6.0, 0.0, -6.0),
    Vec3::new(-9.0, 0.0, -6.0),
    Vec3::new(-6.0, 0.0, -9.0),
    Vec3::new(-3.0, 0.0, -6.0),
    Vec3::new(-6.0, 0.0, -3.0),
    Vec3::new(-9.0, 0.0, -9.0),
    Vec3::new(-3.0, 0.0, -3.0),
    Vec3::new(-8.0, 0.0, -4.0),
];
const SHOWCASE_DEER_BASES: &[Vec3] = &[
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(2.0, 0.0, -2.0),
    Vec3::new(-2.0, 0.0, 2.0),
    Vec3::new(1.0, 0.0, 1.0),
];

/// Balanced cluster bases — 6 humans SE, 10 wolves NW, 4 deer centred.
/// The count asymmetry plus flipped HP (wolves 100, humans 80) counters
/// the pack_focus species-symmetry that makes the showcase human-favoured.
/// Cluster shapes mirror the showcase so the jitter/convergence dynamics
/// stay recognisable.
const BALANCED_HUMAN_BASES: &[Vec3] = &[
    Vec3::new(6.0, 0.0, 6.0),
    Vec3::new(9.0, 0.0, 6.0),
    Vec3::new(6.0, 0.0, 9.0),
    Vec3::new(3.0, 0.0, 6.0),
    Vec3::new(6.0, 0.0, 3.0),
    Vec3::new(9.0, 0.0, 9.0),
];
const BALANCED_WOLF_BASES: &[Vec3] = &[
    Vec3::new(-6.0, 0.0, -6.0),
    Vec3::new(-9.0, 0.0, -6.0),
    Vec3::new(-6.0, 0.0, -9.0),
    Vec3::new(-3.0, 0.0, -6.0),
    Vec3::new(-6.0, 0.0, -3.0),
    Vec3::new(-9.0, 0.0, -9.0),
    Vec3::new(-3.0, 0.0, -3.0),
    Vec3::new(-8.0, 0.0, -4.0),
    Vec3::new(-4.0, 0.0, -8.0),
    Vec3::new(-10.0, 0.0, -10.0),
];
const BALANCED_DEER_BASES: &[Vec3] = &[
    Vec3::new(0.0, 0.0, 0.0),
    Vec3::new(2.0, 0.0, -2.0),
    Vec3::new(-2.0, 0.0, 2.0),
    Vec3::new(1.0, 0.0, 1.0),
];

/// Resolve the (ordered) cluster list for a fixture. Ordering matters:
/// `spawn_agent` returns IDs in call order, and the tests rely on humans
/// being IDs [0..H), wolves [H..H+W), etc. The canonical `Showcase` order
/// (humans → wolves → deer) is preserved across both presets.
fn fixture_clusters(kind: FixtureKind) -> [ClusterSpec; 3] {
    match kind {
        FixtureKind::Showcase => [
            ClusterSpec {
                creature: CreatureType::Human,
                hp:       100.0,
                bases:    SHOWCASE_HUMAN_BASES,
                label:    "human",
            },
            ClusterSpec {
                creature: CreatureType::Wolf,
                hp:       80.0,
                bases:    SHOWCASE_WOLF_BASES,
                label:    "wolf",
            },
            ClusterSpec {
                creature: CreatureType::Deer,
                hp:       60.0,
                bases:    SHOWCASE_DEER_BASES,
                label:    "deer",
            },
        ],
        FixtureKind::Balanced => [
            ClusterSpec {
                creature: CreatureType::Human,
                hp:       85.0,
                bases:    BALANCED_HUMAN_BASES,
                label:    "human",
            },
            ClusterSpec {
                creature: CreatureType::Wolf,
                hp:       95.0,
                bases:    BALANCED_WOLF_BASES,
                label:    "wolf",
            },
            ClusterSpec {
                creature: CreatureType::Deer,
                hp:       60.0,
                bases:    BALANCED_DEER_BASES,
                label:    "deer",
            },
        ],
    }
}

/// Spawn the showcase or balanced fixture, applying the task-174 seed-
/// derived jitter to every agent in every cluster. Cluster centres and
/// counts are fixed per preset so a sweep compares apples to apples, but
/// per-agent x/z positions are perturbed by a seed-derived xorshift64 RNG
/// (see `SpawnRng`). Same seed + same fixture ⇒ bit-identical spawn;
/// different seeds ⇒ meaningfully different initial conditions, which
/// unsticks the "first-death at tick 12 every run" artifact seen in the
/// task-171 sweep. z (vertical) stays at 0 — the simulation is 2D in the
/// xz plane; the 3D engine just carries a dummy height.
#[cfg(test)]
fn spawn_showcase_fixture(seed: u64) -> (SimState, SpawnCounts) {
    spawn_fixture(FixtureKind::Showcase, seed)
}

fn spawn_fixture(kind: FixtureKind, seed: u64) -> (SimState, SpawnCounts) {
    let mut state = SimState::new(SHOWCASE_AGENT_CAP, seed);
    let mut rng = SpawnRng::new(seed);
    let amp = SPAWN_JITTER_AMPLITUDE;

    let clusters = fixture_clusters(kind);
    let mut counts = SpawnCounts::default();

    for spec in &clusters {
        for (i, base) in spec.bases.iter().enumerate() {
            let dx = rng.jitter(amp);
            let dz = rng.jitter(amp);
            let pos = Vec3::new(base.x + dx, base.y, base.z + dz);
            state
                .spawn_agent(AgentSpawn {
                    creature_type: spec.creature,
                    pos,
                    hp: spec.hp,
                    ..Default::default()
                })
                .unwrap_or_else(|| panic!("{} {} spawn", spec.label, i + 1));
        }
        // Accumulate into the matching field. Pattern-matching on
        // creature here (vs a tagged union of counts) keeps the surface
        // area tiny — no new species has been added in years.
        let n = spec.bases.len() as u32;
        match spec.creature {
            CreatureType::Human => counts.humans = n,
            CreatureType::Wolf => counts.wolves = n,
            CreatureType::Deer => counts.deer = n,
            _ => {}
        }
    }

    (state, counts)
}

#[derive(Debug, Clone, Copy, Default)]
struct SpawnCounts {
    humans: u32,
    wolves: u32,
    deer: u32,
}

fn alive_by_type(state: &SimState) -> SpawnCounts {
    let mut out = SpawnCounts::default();
    for id in state.agents_alive() {
        match state.agent_creature_type(id) {
            Some(CreatureType::Human) => out.humans += 1,
            Some(CreatureType::Wolf) => out.wolves += 1,
            Some(CreatureType::Deer) => out.deer += 1,
            _ => {}
        }
    }
    out
}

/// Parse a 64-bit seed from either a `0x`-prefixed hex literal or a plain
/// decimal string. clap's default `u64` parser rejects hex; we accept both
/// so copy-pasting the fixture seed (`0xD00D_FACE_0042_0042`) just works.
fn parse_seed(raw: &str) -> Result<u64, String> {
    let stripped = raw.trim().trim_start_matches("0x").trim_start_matches("0X");
    if raw.trim().starts_with("0x") || raw.trim().starts_with("0X") {
        u64::from_str_radix(stripped, 16).map_err(|e| format!("hex parse: {e}"))
    } else {
        raw.trim()
            .parse::<u64>()
            .map_err(|e| format!("decimal parse: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_seed_accepts_hex_and_decimal() {
        assert_eq!(parse_seed("0xDEADBEEF").unwrap(), 0xDEADBEEF);
        assert_eq!(parse_seed("0XDEADBEEF").unwrap(), 0xDEADBEEF);
        assert_eq!(parse_seed("42").unwrap(), 42);
        assert!(parse_seed("not-a-seed").is_err());
    }

    fn default_args() -> ChronicleArgs {
        ChronicleArgs {
            ticks: None,
            seed: None,
            showcase: false,
            sweep: None,
            base_seed: None,
            verbose: false,
            csv: None,
            fixture: "showcase".to_string(),
            bench: false,
            gpu: false,
        }
    }

    #[test]
    fn resolve_args_defaults_canonical() {
        let args = default_args();
        let (seed, ticks) = resolve_args(&args);
        assert_eq!(seed, CANONICAL_DEFAULT_SEED);
        assert_eq!(ticks, CANONICAL_DEFAULT_TICKS);
    }

    #[test]
    fn resolve_args_defaults_showcase() {
        let args = ChronicleArgs {
            showcase: true,
            ..default_args()
        };
        let (seed, ticks) = resolve_args(&args);
        assert_eq!(seed, SHOWCASE_DEFAULT_SEED);
        assert_eq!(ticks, SHOWCASE_DEFAULT_TICKS);
    }

    #[test]
    fn resolve_args_override_seed_and_ticks() {
        let args = ChronicleArgs {
            ticks: Some(42),
            seed: Some("0x1234".to_string()),
            showcase: true,
            ..default_args()
        };
        let (seed, ticks) = resolve_args(&args);
        assert_eq!(seed, "0x1234");
        assert_eq!(ticks, 42);
    }

    #[test]
    fn showcase_fixture_has_expected_counts() {
        let (state, counts) = spawn_showcase_fixture(0xDEADBEEF);
        assert_eq!(counts.humans, 8);
        assert_eq!(counts.wolves, 8);
        assert_eq!(counts.deer, 4);
        // All 20 agents are alive at spawn time.
        let alive = alive_by_type(&state);
        assert_eq!(alive.humans, 8);
        assert_eq!(alive.wolves, 8);
        assert_eq!(alive.deer, 4);
    }

    #[test]
    fn fixture_kind_parse_accepts_documented_names() {
        assert_eq!(FixtureKind::parse("showcase").unwrap(), FixtureKind::Showcase);
        assert_eq!(FixtureKind::parse("balanced").unwrap(), FixtureKind::Balanced);
        // Case-insensitive — the CLI shouldn't punish "Balanced" vs "balanced".
        assert_eq!(FixtureKind::parse("BALANCED").unwrap(), FixtureKind::Balanced);
        // Surrounding whitespace is tolerated (matches `parse_seed`).
        assert_eq!(FixtureKind::parse("  showcase ").unwrap(), FixtureKind::Showcase);
        assert!(FixtureKind::parse("unknown").is_err());
    }

    #[test]
    fn balanced_fixture_has_expected_counts() {
        // 6 humans vs 10 wolves is the core balance lever — the HP
        // asymmetry (humans 85, wolves 95) was iterated on a 50-seed
        // sweep to land in the 40-60% wolf window. This test guards the
        // shape; the sweep-based test below guards the outcome.
        let (state, counts) = spawn_fixture(FixtureKind::Balanced, 0xDEADBEEF);
        assert_eq!(counts.humans, 6);
        assert_eq!(counts.wolves, 10);
        assert_eq!(counts.deer, 4);
        // All 20 agents are alive at spawn time.
        let alive = alive_by_type(&state);
        assert_eq!(alive.humans, 6);
        assert_eq!(alive.wolves, 10);
        assert_eq!(alive.deer, 4);
    }

    #[test]
    fn balanced_run_is_deterministic() {
        // Same seed + same fixture + same ticks ⇒ identical chronicle.
        // Mirrors `showcase_run_is_deterministic` — the jitter PRNG is
        // seeded identically across fixtures, so the balanced preset
        // inherits the same determinism contract.
        let (mut a, _) = spawn_fixture(FixtureKind::Balanced, 0xCAFEBABE);
        let (mut b, _) = spawn_fixture(FixtureKind::Balanced, 0xCAFEBABE);
        let ea = simulate(&mut a, 50);
        let eb = simulate(&mut b, 50);
        let la = chronicle::render_entries(&a, ea.iter());
        let lb = chronicle::render_entries(&b, eb.iter());
        assert_eq!(la, lb);
    }

    #[test]
    fn balanced_fixture_respects_cluster_identity() {
        // Humans SE, wolves NW — same cluster convention as showcase.
        // Spawn order: 6 humans, then 10 wolves, then 4 deer.
        let (state, _) = spawn_fixture(FixtureKind::Balanced, 0xCAFEBABE);
        let ids: Vec<_> = state.agents_alive().collect();
        assert_eq!(ids.len(), 20);
        for &id in ids.iter().take(6) {
            let pos = state.agent_pos(id).expect("human pos");
            assert!(pos.x > 0.0, "human x should stay in SE cluster: {:?}", pos);
            assert!(pos.z > 0.0, "human z should stay in SE cluster: {:?}", pos);
        }
        for &id in ids.iter().skip(6).take(10) {
            let pos = state.agent_pos(id).expect("wolf pos");
            assert!(pos.x < 0.0, "wolf x should stay in NW cluster: {:?}", pos);
            assert!(pos.z < 0.0, "wolf z should stay in NW cluster: {:?}", pos);
        }
    }

    #[test]
    fn sweep_with_balanced_fixture_has_mixed_outcomes() {
        // Small sweep — just enough to assert the preset isn't dominated
        // by either side. The 50-seed validation run lives in the commit
        // message; CI can't afford 50 × 500 ticks on every push, so a
        // 10-seed probe here checks the "mixed outcomes" invariant.
        //
        // Using base seed 0xDEADBEEF (same as the sweep default) keeps
        // this reproducible from the CLI — a regression here is easy to
        // explore interactively.
        // 30 seeds — at the ~24% human win rate task 176 tuned for,
        // P(zero human wins in 30 independent draws) ≈ 0.76^30 ≈ 0.04%,
        // so this test is effectively non-flaky. A 10-seed sweep had
        // ~6% false-fail rate and broke CI once.
        let mut sink = Vec::new();
        let summary = run_sweep(
            FixtureKind::Balanced,
            0xDEADBEEF,
            30,
            500,
            false,
            &mut sink,
        );
        assert!(
            summary.wolves_wins > 0,
            "balanced fixture should see wolves win at least once in 30 seeds; \
             got H{}/W{}/D{}/S{}",
            summary.humans_wins, summary.wolves_wins, summary.deer_wins, summary.stalemates,
        );
        assert!(
            summary.humans_wins > 0,
            "balanced fixture should see humans win at least once in 30 seeds; \
             got H{}/W{}/D{}/S{}",
            summary.humans_wins, summary.wolves_wins, summary.deer_wins, summary.stalemates,
        );
    }

    #[test]
    fn showcase_fixture_counts_unchanged_by_refactor() {
        // Guard: task 176 refactored `spawn_showcase_fixture` to dispatch
        // through `spawn_fixture(FixtureKind::Showcase, ...)`. The counts,
        // HP tier, and cluster layout for the default preset must be
        // preserved byte-for-byte — the showcase is our canonical demo
        // narrative, and drift here would silently break existing runs.
        let clusters = fixture_clusters(FixtureKind::Showcase);
        assert_eq!(clusters[0].creature, CreatureType::Human);
        assert_eq!(clusters[0].hp, 100.0);
        assert_eq!(clusters[0].bases.len(), 8);
        assert_eq!(clusters[1].creature, CreatureType::Wolf);
        assert_eq!(clusters[1].hp, 80.0);
        assert_eq!(clusters[1].bases.len(), 8);
        assert_eq!(clusters[2].creature, CreatureType::Deer);
        assert_eq!(clusters[2].hp, 60.0);
        assert_eq!(clusters[2].bases.len(), 4);
    }

    #[test]
    fn showcase_run_is_deterministic() {
        let (mut a, _) = spawn_showcase_fixture(0xDEADBEEF);
        let (mut b, _) = spawn_showcase_fixture(0xDEADBEEF);
        let ea = simulate(&mut a, 50);
        let eb = simulate(&mut b, 50);
        let la = chronicle::render_entries(&a, ea.iter());
        let lb = chronicle::render_entries(&b, eb.iter());
        assert_eq!(la, lb, "same seed + same ticks must produce identical chronicles");
    }

    /// Collect every alive agent's (x, z) position. Spawns happen in a
    /// fixed order (humans → wolves → deer), so the AgentId order is the
    /// same across both calls and we can compare index-by-index.
    fn alive_positions(state: &SimState) -> Vec<(f32, f32)> {
        state
            .agents_alive()
            .filter_map(|id| state.agent_pos(id).map(|p| (p.x, p.z)))
            .collect()
    }

    #[test]
    fn different_seeds_produce_different_spawns() {
        // Two different seeds must produce at least one distinct agent
        // position. Without spawn jitter this test would fail — the
        // fixture used to hardcode positions regardless of seed.
        let (a, _) = spawn_showcase_fixture(0xDEADBEEF);
        let (b, _) = spawn_showcase_fixture(0xCAFEBABE);
        let pa = alive_positions(&a);
        let pb = alive_positions(&b);
        assert_eq!(pa.len(), pb.len(), "same fixture size for both seeds");
        assert_ne!(
            pa, pb,
            "different seeds must produce different spawn positions (got identical)"
        );
    }

    #[test]
    fn same_seed_spawns_are_bit_identical() {
        // Companion to the determinism chronicle test — assert at the
        // position level so a regression in the spawn PRNG shows up even
        // if the sim would paper over it.
        let (a, _) = spawn_showcase_fixture(0xDEADBEEF);
        let (b, _) = spawn_showcase_fixture(0xDEADBEEF);
        assert_eq!(alive_positions(&a), alive_positions(&b));
    }

    #[test]
    fn spawn_jitter_respects_cluster_identity() {
        // Humans SE (+6,+6), wolves NW (-6,-6), deer centred. With a ±2.5
        // amplitude the humans' x/z must stay positive-ish and the wolves'
        // negative-ish — the sweep measures divergence within clusters,
        // not across them.
        let (state, _) = spawn_showcase_fixture(0xCAFEBABE);
        let ids: Vec<_> = state.agents_alive().collect();
        // Humans are spawned first (indices 0..8).
        for &id in ids.iter().take(8) {
            let pos = state.agent_pos(id).expect("human pos");
            assert!(pos.x > 0.0, "human x should stay in SE cluster: {:?}", pos);
            assert!(pos.z > 0.0, "human z should stay in SE cluster: {:?}", pos);
        }
        // Wolves next (indices 8..16).
        for &id in ids.iter().skip(8).take(8) {
            let pos = state.agent_pos(id).expect("wolf pos");
            assert!(pos.x < 0.0, "wolf x should stay in NW cluster: {:?}", pos);
            assert!(pos.z < 0.0, "wolf z should stay in NW cluster: {:?}", pos);
        }
    }

    #[test]
    fn sweep_with_varied_spawns_has_wider_dispersion() {
        // Small sweep — just enough to observe that spawns now vary by
        // seed. Before task 174 all first-death ticks were identical;
        // after the jitter at least two distinct values should appear.
        let mut sink = Vec::new();
        let summary = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 5, 500, false, &mut sink);
        let death_ticks: std::collections::BTreeSet<u32> = summary
            .per_run
            .iter()
            .filter_map(|r| r.first_death_tick)
            .collect();
        assert!(
            death_ticks.len() >= 2,
            "expected varied first-death ticks, got {:?}",
            death_ticks
        );
    }

    #[test]
    fn spawn_rng_is_nonzero_for_zero_seed() {
        // Salting against a zero state keeps xorshift64 live when
        // `seed ^ SPAWN_JITTER_SEED_XOR == 0`. Without the guard, every
        // `next_u64` would return 0 and every agent would sit on its
        // base position — silently defeating the task.
        let zero_seeded = SPAWN_JITTER_SEED_XOR;
        let mut rng = SpawnRng::new(zero_seeded);
        assert_ne!(rng.next_u64(), 0, "spawn rng must never stall at zero");
    }

    // --- sweep tests ---

    #[test]
    fn winner_prefers_strict_max() {
        // Strict majority → that species wins.
        assert_eq!(
            Winner::from_counts(&SpawnCounts { humans: 5, wolves: 2, deer: 1 }),
            Winner::Humans
        );
        assert_eq!(
            Winner::from_counts(&SpawnCounts { humans: 0, wolves: 4, deer: 0 }),
            Winner::Wolves
        );
        // Tie at top (both >0) → stalemate.
        assert_eq!(
            Winner::from_counts(&SpawnCounts { humans: 3, wolves: 3, deer: 1 }),
            Winner::Stalemate
        );
        // Total wipe → stalemate (no max to pick).
        assert_eq!(
            Winner::from_counts(&SpawnCounts { humans: 0, wolves: 0, deer: 0 }),
            Winner::Stalemate
        );
        // All three tied → stalemate.
        assert_eq!(
            Winner::from_counts(&SpawnCounts { humans: 2, wolves: 2, deer: 2 }),
            Winner::Stalemate
        );
    }

    #[test]
    fn mean_median_stdev_handles_empty_and_basic() {
        assert_eq!(mean_median_stdev(&[]), (0.0, 0.0, 0.0));
        let (mean, median, stdev) = mean_median_stdev(&[2, 4, 4, 4, 5, 5, 7, 9]);
        assert!((mean - 5.0).abs() < 1e-9);
        assert!((median - 4.5).abs() < 1e-9);
        // Population stdev of the textbook sample = 2.0.
        assert!((stdev - 2.0).abs() < 1e-9);
        // Odd-length median = exact middle.
        let (_, med2, _) = mean_median_stdev(&[1, 2, 3]);
        assert!((med2 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn sweep_is_deterministic() {
        // Same base seed + same N ⇒ identical per-run stats. 3 runs x 60
        // ticks keeps the test fast while still exercising the aggregation.
        let mut sink_a = Vec::new();
        let mut sink_b = Vec::new();
        let sum_a = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 3, 60, false, &mut sink_a);
        let sum_b = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 3, 60, false, &mut sink_b);
        assert_eq!(sum_a.per_run, sum_b.per_run);
        assert_eq!(sum_a.humans_wins, sum_b.humans_wins);
        assert_eq!(sum_a.wolves_wins, sum_b.wolves_wins);
        assert_eq!(sum_a.total_events_mean.to_bits(), sum_b.total_events_mean.to_bits());
        assert_eq!(sum_a.chronicle_mean.to_bits(), sum_b.chronicle_mean.to_bits());
    }

    #[test]
    fn sweep_seeds_are_stepped_by_one() {
        // Run 0 uses the base seed exactly; run k uses base + k.
        let mut sink = Vec::new();
        let summary = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 3, 20, false, &mut sink);
        assert_eq!(summary.per_run.len(), 3);
        assert_eq!(summary.per_run[0].seed, 0xDEADBEEF);
        assert_eq!(summary.per_run[1].seed, 0xDEADBEEF + 1);
        assert_eq!(summary.per_run[2].seed, 0xDEADBEEF + 2);
        assert_eq!(summary.runs, 3);
        assert_eq!(summary.ticks, 20);
        assert_eq!(summary.base_seed, 0xDEADBEEF);
    }

    #[test]
    fn sweep_outcome_buckets_sum_to_runs() {
        // Every run lands in exactly one outcome bucket.
        let mut sink = Vec::new();
        let s = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 5, 80, false, &mut sink);
        assert_eq!(
            s.humans_wins + s.wolves_wins + s.deer_wins + s.stalemates,
            s.runs
        );
    }

    #[test]
    fn verbose_sweep_emits_a_line_per_run() {
        // Verbose mode writes one line per run to the progress sink.
        let mut sink = Vec::new();
        let _ = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 3, 30, true, &mut sink);
        let text = String::from_utf8(sink).expect("utf8 sink");
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        for (k, line) in lines.iter().enumerate() {
            assert!(line.starts_with(&format!("Run {:>4}", k)),
                "expected run-index prefix, got {line:?}");
        }
    }

    // --- csv export tests ---

    /// Build a unique scratch path inside the OS temp dir. Avoids the
    /// `tempfile` crate (not a dep here) while still giving per-test
    /// isolation — each call yields a fresh filename via `thread_id + nanos`.
    fn scratch_csv_path(tag: &str) -> std::path::PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tid = format!("{:?}", std::thread::current().id());
        let tid_clean: String = tid.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        let mut p = std::env::temp_dir();
        p.push(format!("chronicle_csv_{tag}_{tid_clean}_{nanos}.csv"));
        p
    }

    #[test]
    fn csv_export_writes_expected_rows() {
        // Drive a 3-seed sweep, round-trip through the CSV writer, and
        // assert the file matches the spec: 1 header + 3 data rows, hex
        // seeds, and the documented column layout.
        let mut sink = Vec::new();
        let summary = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 3, 60, false, &mut sink);
        let path = scratch_csv_path("expected_rows");
        write_sweep_csv(&summary, &path).expect("write csv");

        let text = std::fs::read_to_string(&path).expect("read csv");
        // Best-effort cleanup — don't fail the test if the OS already
        // reaped the file.
        let _ = std::fs::remove_file(&path);

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 4, "expected header + 3 rows, got {lines:?}");
        assert_eq!(lines[0], SWEEP_CSV_HEADER);
        // Header columns must match the spec exactly, in order.
        assert_eq!(
            lines[0],
            "seed,alive_humans,alive_wolves,alive_deer,winner,total_events,chronicle_entries,rout_count,first_death_tick"
        );
        for (i, row) in lines.iter().skip(1).enumerate() {
            assert!(
                row.starts_with("0x"),
                "row {i} seed must be 0x-prefixed hex, got {row:?}",
            );
            let cols: Vec<&str> = row.split(',').collect();
            assert_eq!(cols.len(), 9, "row {i} should have 9 columns, got {row:?}");
            // Seed column is 18 chars: "0x" + 16 hex digits.
            assert_eq!(
                cols[0].len(),
                18,
                "seed should be zero-padded to 16 hex digits, got {:?}",
                cols[0],
            );
            // Winner column must be one of the 4 tokens.
            assert!(
                matches!(cols[4], "humans" | "wolves" | "deer" | "stalemate"),
                "row {i} winner token invalid: {:?}",
                cols[4],
            );
        }
    }

    #[test]
    fn csv_export_empty_first_death_when_no_deaths() {
        // The showcase fixture reliably sees deaths within a few ticks, so
        // we can't produce a no-death run via the real sweep. Instead
        // synthesise a 2-row sweep summary with one `first_death_tick:
        // None` and one `Some(...)`, then round-trip through the writer.
        let per_run = vec![
            RunStats {
                seed:             0xCAFEBABE_u64,
                humans_alive:     8,
                wolves_alive:     8,
                deer_alive:       4,
                total_events:     0,
                chronicle_count:  0,
                rout_count:       0,
                first_death_tick: None,
                winner:           Winner::Stalemate,
            },
            RunStats {
                seed:             0xDEADBEEF_u64,
                humans_alive:     3,
                wolves_alive:     0,
                deer_alive:       1,
                total_events:     42,
                chronicle_count:  7,
                rout_count:       1,
                first_death_tick: Some(17),
                winner:           Winner::Humans,
            },
        ];
        let summary = summarize(per_run, 2, 500, 0xCAFEBABE, FixtureKind::Showcase);
        let path = scratch_csv_path("empty_first_death");
        write_sweep_csv(&summary, &path).expect("write csv");

        let text = std::fs::read_to_string(&path).expect("read csv");
        let _ = std::fs::remove_file(&path);

        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 rows: {lines:?}");
        // Row 0: no death -> trailing column is empty (ends with ",").
        assert!(
            lines[1].ends_with(','),
            "no-death row must end with empty first_death_tick: {:?}",
            lines[1],
        );
        // Split preserves the empty trailing field.
        let cols0: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(cols0.len(), 9);
        assert_eq!(cols0[8], "", "first_death_tick must be blank for no-death run");
        // Row 1: death -> trailing column is "17".
        let cols1: Vec<&str> = lines[2].split(',').collect();
        assert_eq!(cols1[8], "17");
        assert_eq!(cols1[4], "humans");
    }

    #[test]
    fn csv_rejects_without_sweep() {
        // `run_chronicle` returns ExitCode::FAILURE when `--csv` is
        // passed without `--sweep`; this guards against the misuse silently
        // running a single-fixture chronicle and discarding the requested
        // CSV path.
        let args = ChronicleArgs {
            csv: Some(std::path::PathBuf::from("/tmp/should-never-be-written.csv")),
            ..default_args()
        };
        assert_eq!(run_chronicle(args), ExitCode::FAILURE);
    }

    #[test]
    fn csv_rejects_missing_parent_directory() {
        // Typo-prevention: writing to a non-existent directory fails fast
        // with a clear error instead of File::create's raw ENOENT.
        let mut sink = Vec::new();
        let summary = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 2, 20, false, &mut sink);
        let path = std::path::PathBuf::from(
            "/tmp/chronicle_csv_nonexistent_xyz123_parent/out.csv",
        );
        let err = write_sweep_csv(&summary, &path).expect_err("should fail");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn sweep_summary_render_has_expected_sections() {
        let mut progress = Vec::new();
        let summary = run_sweep(FixtureKind::Showcase, 0xDEADBEEF, 2, 40, false, &mut progress);
        let mut out = Vec::new();
        render_sweep_summary(&summary, &mut out).unwrap();
        let text = String::from_utf8(out).expect("utf8 render");
        for section in [
            "=== Balance Sweep: 2 seeds",
            "Fixture: 8 humans + 8 wolves + 4 deer",
            "Base seed:",
            "--- Outcomes ---",
            "Humans won:",
            "Wolves won:",
            "Stalemate:",
            "--- Survivors",
            "--- Combat dynamics ---",
            "Chronicle entries:",
            "Rout events",
            "--- Total events ---",
        ] {
            assert!(
                text.contains(section),
                "sweep summary missing section {section:?}; got:\n{text}",
            );
        }
    }

    // --- bench tests ---

    #[test]
    fn percentile_computation_is_correct() {
        // Endpoints map exactly to min/max.
        let v: Vec<u128> = vec![10, 20, 30, 40, 50];
        assert_eq!(linear_interp_percentile_u128(&v, 0.0), 10);
        assert_eq!(linear_interp_percentile_u128(&v, 1.0), 50);
        // Median (frac=0.5 on N=5) lands on sorted[2] exactly.
        assert_eq!(linear_interp_percentile_u128(&v, 0.5), 30);
        // Quarter-point: idx = 0.25 * 4 = 1.0 → sorted[1] exactly.
        assert_eq!(linear_interp_percentile_u128(&v, 0.25), 20);
        // Frac clamps to [0, 1] — a misuse shouldn't panic.
        assert_eq!(linear_interp_percentile_u128(&v, -0.5), 10);
        assert_eq!(linear_interp_percentile_u128(&v, 1.5), 50);
        // Empty slice returns 0 (defensive; bench never produces empty).
        assert_eq!(linear_interp_percentile_u128(&[], 0.5), 0);
        // Single element: every percentile is that element.
        assert_eq!(linear_interp_percentile_u128(&[42u128], 0.0), 42);
        assert_eq!(linear_interp_percentile_u128(&[42u128], 1.0), 42);
        // Linear interpolation between neighbours: idx = 0.125 * 4 = 0.5
        // → midpoint of sorted[0]=10 and sorted[1]=20 = 15.
        assert_eq!(linear_interp_percentile_u128(&v, 0.125), 15);

        // u64 variant mirrors the u128 one.
        let v64: Vec<u64> = vec![100, 200, 300, 400, 500];
        assert_eq!(linear_interp_percentile_u64(&v64, 0.5), 300);
        assert_eq!(linear_interp_percentile_u64(&v64, 0.0), 100);
        assert_eq!(linear_interp_percentile_u64(&v64, 1.0), 500);
        assert_eq!(linear_interp_percentile_u64(&[], 0.5), 0);
    }

    #[test]
    fn bench_produces_nonzero_samples() {
        // Tiny bench so the test finishes fast even in debug: 1 warmup,
        // 1 timed sample, 50 ticks, canonical fixture (smallest shape).
        // We assert the sample is non-zero and that per-tick cost is
        // plausibly positive — no absolute floor (CI machines vary wildly).
        let shape = &BENCH_SHAPES[0]; // canonical
        let result = run_bench_shape(shape, 1, 1, 50);
        assert_eq!(result.label, "canonical");
        assert_eq!(result.ticks, 50);
        assert_eq!(result.sample_ns.len(), 1);
        assert_eq!(result.events_sorted.len(), 1);
        assert!(
            result.sample_ns[0] > 0,
            "sample must record positive wall-clock: {:?}",
            result.sample_ns,
        );
        // Median of a single-sample result equals that sample.
        assert_eq!(result.median_ns(), result.sample_ns[0]);
        // Re-run with 2 samples to exercise sorting + non-trivial
        // percentile math (stable: same fixture, same seed, both
        // samples should be close — we only assert ordering, not bounds).
        let r2 = run_bench_shape(shape, 1, 2, 50);
        assert_eq!(r2.sample_ns.len(), 2);
        assert!(r2.sample_ns[0] <= r2.sample_ns[1], "samples must be sorted asc");
        // Events are also recorded per sample; canonical at 50 ticks
        // emits some events but the exact count is sim-state dependent
        // and seed-stable. Just assert it's non-negative and sorted.
        assert!(r2.events_sorted[0] <= r2.events_sorted[1]);
    }

    #[test]
    fn bench_report_has_expected_sections() {
        // Render a mini report from two synthetic shape results and
        // assert the headline strings are present. Doesn't run the
        // bench — the timing-heavy path is covered by
        // `bench_produces_nonzero_samples` above; this test only
        // exercises the renderer so format regressions surface in <1ms.
        let results = vec![
            BenchShapeResult {
                label:         "canonical",
                description:   "test fixture",
                ticks:         500,
                sample_ns:     vec![10_000_000, 11_000_000, 12_000_000, 13_000_000, 14_000_000],
                events_sorted: vec![100, 110, 120, 130, 140],
            },
            BenchShapeResult {
                label:         "showcase",
                description:   "test fixture 2",
                ticks:         500,
                sample_ns:     vec![50_000_000, 55_000_000, 60_000_000, 65_000_000, 70_000_000],
                events_sorted: vec![500, 550, 600, 650, 700],
            },
        ];
        let mut out = Vec::new();
        render_bench_report(&mut out, &results).expect("render");
        let text = String::from_utf8(out).expect("utf8");
        for needle in [
            "=== Perf Benchmark ===",
            "samples + 3 warmup",
            "--- canonical (test fixture) ---",
            "--- showcase (test fixture 2) ---",
            "median:",
            "p5/p95:",
            "events/tick:",
            "ticks/sec",
        ] {
            assert!(
                text.contains(needle),
                "bench report missing {needle:?}; got:\n{text}",
            );
        }
    }

    #[test]
    fn bench_rejects_combined_with_sweep() {
        // --bench and --sweep are mutually exclusive: bench owns its own
        // fixture iteration, sweep has its own output shape. Combining
        // them should fail fast rather than silently picking one.
        let args = ChronicleArgs {
            bench: true,
            sweep: Some(5),
            ..default_args()
        };
        assert_eq!(run_chronicle(args), ExitCode::FAILURE);
    }

    #[test]
    fn format_tps_groups_thousands() {
        assert_eq!(format_tps(0.0), "0.0");
        assert_eq!(format_tps(42.0), "42.0");
        assert_eq!(format_tps(1234.5), "1,234.5");
        assert_eq!(format_tps(45_231.2), "45,231.2");
        assert_eq!(format_tps(1_234_567.8), "1,234,567.8");
        // Integer portion 100 has 3 digits — no grouping comma inserted.
        assert_eq!(format_tps(100.0), "100.0");
        // 1000 is the threshold where a comma appears.
        assert_eq!(format_tps(1000.0), "1,000.0");
    }
}
