//! `cargo run --bin xtask -- chronicle` — run the engine's canonical
//! wolves+humans fixture and dump the rendered chronicle.
//!
//! The fixture mirrors `crates/engine/tests/wolves_and_humans_parity.rs`:
//! 3 humans + 2 wolves on a flat plane, deterministic seed, stock scoring
//! table + full 6-phase `step_full` pipeline. The only difference is this
//! command walks the event ring and hands every `ChronicleEntry` to
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

const AGENT_CAP: u32 = 8;
const EVENT_RING_CAP: usize = 1 << 16;

pub fn run_chronicle(args: ChronicleArgs) -> ExitCode {
    let seed = match parse_seed(&args.seed) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("chronicle: invalid --seed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut state = spawn_fixture(seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    let mut invariants = InvariantRegistry::new();
    invariants.register(Box::new(PoolNonOverlapInvariant));

    let mut views: Vec<&mut dyn MaterializedView> = Vec::new();
    let telemetry = NullSink;

    for _ in 0..args.ticks {
        step_full(
            &mut state,
            &mut scratch,
            &mut events,
            &UtilityBackend,
            &cascade,
            &mut views[..],
            &invariants,
            &telemetry,
        );
    }

    // Walk every pushed event (the ring only evicts when it overflows, and
    // the chosen cap is far above our per-run volume). Render one line per
    // `ChronicleEntry`; non-chronicle events are skipped by the helper.
    let lines = chronicle::render_entries(&state, events.iter());
    println!("# chronicle — seed={:#x} ticks={} agents={} chronicle_entries={}",
        seed, args.ticks, state.agent_cap(), lines.len());
    for line in lines {
        println!("{line}");
    }
    ExitCode::SUCCESS
}

/// Spawn the canonical wolves+humans fixture. Kept in sync with
/// `tests/wolves_and_humans_parity.rs::spawn_fixture` — the two fixtures
/// share a seed-parity contract (same seed + same ticks ⇒ same chronicle).
fn spawn_fixture(seed: u64) -> SimState {
    let mut state = SimState::new(AGENT_CAP, seed);
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
