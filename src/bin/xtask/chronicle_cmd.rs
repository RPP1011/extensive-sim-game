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

/// Spawn the showcase fixture: 8 humans (SE cluster), 8 wolves (NW cluster),
/// 4 deer (center). Positions deterministic so a given seed always reproduces
/// the same chronicle. Clusters are ~15 units apart so the two predator
/// groups find each other within the first ~100 ticks and a mixed narrative
/// emerges over 500 ticks rather than degenerating into an immediate rout.
fn spawn_showcase_fixture(seed: u64) -> (SimState, SpawnCounts) {
    let mut state = SimState::new(SHOWCASE_AGENT_CAP, seed);

    // Humans: SE cluster at (+6, +6), spread ±3.
    let humans = [
        Vec3::new(6.0, 0.0, 6.0),
        Vec3::new(9.0, 0.0, 6.0),
        Vec3::new(6.0, 0.0, 9.0),
        Vec3::new(3.0, 0.0, 6.0),
        Vec3::new(6.0, 0.0, 3.0),
        Vec3::new(9.0, 0.0, 9.0),
        Vec3::new(3.0, 0.0, 3.0),
        Vec3::new(8.0, 0.0, 4.0),
    ];
    for (i, pos) in humans.iter().enumerate() {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: *pos,
                hp: 100.0,
                ..Default::default()
            })
            .unwrap_or_else(|| panic!("human {} spawn", i + 1));
    }

    // Wolves: NW cluster at (-6, -6), spread ±3.
    let wolves = [
        Vec3::new(-6.0, 0.0, -6.0),
        Vec3::new(-9.0, 0.0, -6.0),
        Vec3::new(-6.0, 0.0, -9.0),
        Vec3::new(-3.0, 0.0, -6.0),
        Vec3::new(-6.0, 0.0, -3.0),
        Vec3::new(-9.0, 0.0, -9.0),
        Vec3::new(-3.0, 0.0, -3.0),
        Vec3::new(-8.0, 0.0, -4.0),
    ];
    for (i, pos) in wolves.iter().enumerate() {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: *pos,
                hp: 80.0,
                ..Default::default()
            })
            .unwrap_or_else(|| panic!("wolf {} spawn", i + 1));
    }

    // Deer: 4 in the center.
    let deer = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, -2.0),
        Vec3::new(-2.0, 0.0, 2.0),
        Vec3::new(1.0, 0.0, 1.0),
    ];
    for (i, pos) in deer.iter().enumerate() {
        state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Deer,
                pos: *pos,
                hp: 60.0,
                ..Default::default()
            })
            .unwrap_or_else(|| panic!("deer {} spawn", i + 1));
    }

    let counts = SpawnCounts {
        humans: humans.len() as u32,
        wolves: wolves.len() as u32,
        deer: deer.len() as u32,
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

    #[test]
    fn resolve_args_defaults_canonical() {
        let args = ChronicleArgs {
            ticks: None,
            seed: None,
            showcase: false,
        };
        let (seed, ticks) = resolve_args(&args);
        assert_eq!(seed, CANONICAL_DEFAULT_SEED);
        assert_eq!(ticks, CANONICAL_DEFAULT_TICKS);
    }

    #[test]
    fn resolve_args_defaults_showcase() {
        let args = ChronicleArgs {
            ticks: None,
            seed: None,
            showcase: true,
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
}
