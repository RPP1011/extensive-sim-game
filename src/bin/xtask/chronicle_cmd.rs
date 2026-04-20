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
//!   outcome summary for demos and essays.
//!
//! Both paths walk the event ring and hand every `ChronicleEntry` to
//! `engine::chronicle::render_entry`, printing one line per event.

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
    // Sweep mode short-circuits seed resolution: it has its own base-seed
    // knob and always runs the showcase fixture. `--sweep 0` is a no-op
    // that we surface as an error so users don't get silent empty output.
    if let Some(runs) = args.sweep {
        return run_sweep_from_args(&args, runs);
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
        run_showcase(seed, ticks)
    } else {
        run_canonical(seed, ticks)
    }
}

fn run_sweep_from_args(args: &ChronicleArgs, runs: u32) -> ExitCode {
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

    let summary = run_sweep(base_seed, runs, ticks, args.verbose, &mut std::io::stderr());
    let mut stdout = std::io::stdout().lock();
    render_sweep_summary(&summary, &mut stdout).ok();
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

fn run_canonical(seed: u64, ticks: u32) -> ExitCode {
    let mut state = spawn_canonical_fixture(seed);
    let events = simulate(&mut state, ticks);

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

fn run_showcase(seed: u64, ticks: u32) -> ExitCode {
    let (mut state, counts) = spawn_showcase_fixture(seed);
    let events = simulate(&mut state, ticks);

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
fn run_showcase_stats(seed: u64, ticks: u32) -> RunStats {
    let (mut state, _counts) = spawn_showcase_fixture(seed);
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
    /// Retained for introspection tests + future JSON export. Not read by
    /// the text renderer (which only needs the pre-computed aggregates).
    #[allow(dead_code)]
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

fn summarize(per_run: Vec<RunStats>, runs: u32, ticks: u32, base_seed: u64) -> SweepSummary {
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
        let stats = run_showcase_stats(seed, ticks);
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
    summarize(per_run, runs, ticks, base_seed)
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
    writeln!(out, "Fixture: 8 humans + 8 wolves + 4 deer (showcase)")?;
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

/// Spawn the showcase fixture: 8 humans (SE cluster), 8 wolves (NW cluster),
/// 4 deer (center). Cluster centres and agent counts are fixed so a sweep
/// compares apples to apples, but per-agent x/z positions are perturbed by
/// a seed-derived xorshift64 RNG (see `SpawnRng`). Same seed ⇒ same spawn;
/// different seeds ⇒ meaningfully different initial conditions, which
/// unsticks the "first-death at tick 12 every run" artifact seen in the
/// task-171 sweep. z (vertical) stays at 0 — the simulation is 2D in the
/// xz plane; the 3D engine just carries a dummy height.
fn spawn_showcase_fixture(seed: u64) -> (SimState, SpawnCounts) {
    let mut state = SimState::new(SHOWCASE_AGENT_CAP, seed);
    let mut rng = SpawnRng::new(seed);

    let amp = SPAWN_JITTER_AMPLITUDE;

    // Humans: SE cluster at (+6, +6), spread ±3 plus per-agent jitter.
    let human_bases = [
        Vec3::new(6.0, 0.0, 6.0),
        Vec3::new(9.0, 0.0, 6.0),
        Vec3::new(6.0, 0.0, 9.0),
        Vec3::new(3.0, 0.0, 6.0),
        Vec3::new(6.0, 0.0, 3.0),
        Vec3::new(9.0, 0.0, 9.0),
        Vec3::new(3.0, 0.0, 3.0),
        Vec3::new(8.0, 0.0, 4.0),
    ];
    for (i, base) in human_bases.iter().enumerate() {
        let dx = rng.jitter(amp);
        let dz = rng.jitter(amp);
        let pos = Vec3::new(base.x + dx, base.y, base.z + dz);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos,
                hp: 100.0,
                ..Default::default()
            })
            .unwrap_or_else(|| panic!("human {} spawn", i + 1));
    }

    // Wolves: NW cluster at (-6, -6), spread ±3 plus per-agent jitter.
    let wolf_bases = [
        Vec3::new(-6.0, 0.0, -6.0),
        Vec3::new(-9.0, 0.0, -6.0),
        Vec3::new(-6.0, 0.0, -9.0),
        Vec3::new(-3.0, 0.0, -6.0),
        Vec3::new(-6.0, 0.0, -3.0),
        Vec3::new(-9.0, 0.0, -9.0),
        Vec3::new(-3.0, 0.0, -3.0),
        Vec3::new(-8.0, 0.0, -4.0),
    ];
    for (i, base) in wolf_bases.iter().enumerate() {
        let dx = rng.jitter(amp);
        let dz = rng.jitter(amp);
        let pos = Vec3::new(base.x + dx, base.y, base.z + dz);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos,
                hp: 80.0,
                ..Default::default()
            })
            .unwrap_or_else(|| panic!("wolf {} spawn", i + 1));
    }

    // Deer: 4 in the center plus per-agent jitter.
    let deer_bases = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, -2.0),
        Vec3::new(-2.0, 0.0, 2.0),
        Vec3::new(1.0, 0.0, 1.0),
    ];
    for (i, base) in deer_bases.iter().enumerate() {
        let dx = rng.jitter(amp);
        let dz = rng.jitter(amp);
        let pos = Vec3::new(base.x + dx, base.y, base.z + dz);
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Deer,
                pos,
                hp: 60.0,
                ..Default::default()
            })
            .unwrap_or_else(|| panic!("deer {} spawn", i + 1));
    }

    let counts = SpawnCounts {
        humans: human_bases.len() as u32,
        wolves: wolf_bases.len() as u32,
        deer: deer_bases.len() as u32,
    };
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
        let summary = run_sweep(0xDEADBEEF, 5, 500, false, &mut sink);
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
        let sum_a = run_sweep(0xDEADBEEF, 3, 60, false, &mut sink_a);
        let sum_b = run_sweep(0xDEADBEEF, 3, 60, false, &mut sink_b);
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
        let summary = run_sweep(0xDEADBEEF, 3, 20, false, &mut sink);
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
        let s = run_sweep(0xDEADBEEF, 5, 80, false, &mut sink);
        assert_eq!(
            s.humans_wins + s.wolves_wins + s.deer_wins + s.stalemates,
            s.runs
        );
    }

    #[test]
    fn verbose_sweep_emits_a_line_per_run() {
        // Verbose mode writes one line per run to the progress sink.
        let mut sink = Vec::new();
        let _ = run_sweep(0xDEADBEEF, 3, 30, true, &mut sink);
        let text = String::from_utf8(sink).expect("utf8 sink");
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        for (k, line) in lines.iter().enumerate() {
            assert!(line.starts_with(&format!("Run {:>4}", k)),
                "expected run-index prefix, got {line:?}");
        }
    }

    #[test]
    fn sweep_summary_render_has_expected_sections() {
        let mut progress = Vec::new();
        let summary = run_sweep(0xDEADBEEF, 2, 40, false, &mut progress);
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
}
