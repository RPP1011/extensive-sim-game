//! Wolves + humans parity anchor — the regression fixture for the
//! DSL-owned game rules as of compiler milestones 2-6 + physics parity.
//!
//! Scenario: 3 humans + 2 wolves on a flat plane, fixed seed, driven by
//! the stock `UtilityBackend` (DSL-emitted scoring table) + the full
//! 6-phase `step_full` pipeline with engine-builtin cascade handlers
//! (DSL-emitted physics + hand-written RecordMemory). Runs 100 ticks.
//!
//! Two assertions:
//! 1. Structural invariants — event counts, kinds observed, tick ordering,
//!    hostility directionality — are in line with the wolves+humans design.
//! 2. The full event log, serialised as deterministic one-line records,
//!    matches the committed baseline at `wolves_and_humans_baseline.txt`
//!    byte-for-byte.
//!
//! A failure on (2) means something in the DSL-owned surface moved. If the
//! move is intentional, regenerate the baseline by setting
//! `WOLVES_AND_HUMANS_REGEN=1` — the test will overwrite the committed file
//! with the new log, then still fail the run so the diff lands in review.
//!
//! The baseline is captured against the *current* DSL-owned state. It is
//! not a comparison against any pre-DSL hand-written code (that code is
//! deleted). This anchor exists so future DSL edits that silently change
//! wolves+humans behaviour fail loud.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::invariant::{InvariantRegistry, PoolNonOverlapInvariant};
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch};
use engine::telemetry::NullSink;
use engine::view::materialized::MaterializedView;
use glam::Vec3;
use std::fmt::Write as _;
use std::path::Path;

const SEED: u64 = 0xD00D_FACE_0042_0042;
const TICKS: u32 = 100;
const AGENT_CAP: u32 = 8;
const EVENT_RING_CAP: usize = 1 << 16;

const BASELINE_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/wolves_and_humans_baseline.txt");

/// Spawn the canonical wolves+humans fixture:
///
/// - 3 humans at (0,0,0), (2,0,0), (-2,0,0). Middle human borders wolf A's
///   2m attack range — the first tick emits an attack immediately.
/// - 2 wolves at (3,0,0), (-3,0,0). Each sits 1 m from the nearest human.
///
/// Returns the constructed `SimState`. Agent ids are 1..=5 in spawn order.
fn spawn_fixture() -> SimState {
    let mut state = SimState::new(AGENT_CAP, SEED);
    // Humans.
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new( 0.0, 0.0, 0.0), hp: 100.0,
    }).expect("human 1 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new( 2.0, 0.0, 0.0), hp: 100.0,
    }).expect("human 2 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(-2.0, 0.0, 0.0), hp: 100.0,
    }).expect("human 3 spawn");
    // Wolves.
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new( 3.0, 0.0, 0.0), hp: 80.0,
    }).expect("wolf 1 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(-3.0, 0.0, 0.0), hp: 80.0,
    }).expect("wolf 2 spawn");
    state
}

/// Format one event as a stable one-line record. Keeps the formatting
/// explicit (rather than leaning on `Debug`) so a Rust-version bump to
/// glam / std Debug doesn't silently invalidate the baseline.
fn fmt_event(e: &Event) -> String {
    match *e {
        Event::AgentMoved { actor, from, location, tick } => format!(
            "AgentMoved(tick={tick},id={},from=({:.6},{:.6},{:.6}),to=({:.6},{:.6},{:.6}))",
            actor.raw(),
            from.x, from.y, from.z, location.x, location.y, location.z,
        ),
        Event::AgentAttacked { actor, target, damage, tick } => format!(
            "AgentAttacked(tick={tick},attacker={},target={},damage={:.6})",
            actor.raw(), target.raw(), damage,
        ),
        Event::AgentDied { agent_id, tick } => format!(
            "AgentDied(tick={tick},id={})", agent_id.raw(),
        ),
        Event::AgentFled { agent_id, from, to, tick } => format!(
            "AgentFled(tick={tick},id={},from=({:.6},{:.6},{:.6}),to=({:.6},{:.6},{:.6}))",
            agent_id.raw(),
            from.x, from.y, from.z, to.x, to.y, to.z,
        ),
        Event::AgentAte { agent_id, delta, tick } => format!(
            "AgentAte(tick={tick},id={},delta={:.6})", agent_id.raw(), delta,
        ),
        Event::AgentDrank { agent_id, delta, tick } => format!(
            "AgentDrank(tick={tick},id={},delta={:.6})", agent_id.raw(), delta,
        ),
        Event::AgentRested { agent_id, delta, tick } => format!(
            "AgentRested(tick={tick},id={},delta={:.6})", agent_id.raw(), delta,
        ),
        Event::OpportunityAttackTriggered { actor, target, tick } => format!(
            "OpportunityAttackTriggered(tick={tick},attacker={},target={})",
            actor.raw(), target.raw(),
        ),
        // Any other variant we don't currently expect — serialise a
        // generic tag so the test still captures the surprise. If one of
        // these starts appearing for real, promote it to an explicit arm.
        other => format!("OTHER({:?})", other),
    }
}

/// Run the scenario and return the event log rendered as one line per event.
fn run_scenario_log() -> String {
    let mut state = spawn_fixture();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(EVENT_RING_CAP);
    let cascade = CascadeRegistry::with_engine_builtins();

    let mut invariants = InvariantRegistry::new();
    invariants.register(Box::new(PoolNonOverlapInvariant));

    let mut views: Vec<&mut dyn MaterializedView> = Vec::new();
    let telemetry = NullSink;

    for _ in 0..TICKS {
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

    // The ring is sized well above the expected emission count (100 ticks
    // * ~5 events/tick ≪ 65 K) so `iter()` walks every pushed event.
    assert!(
        events.total_pushed() <= EVENT_RING_CAP,
        "event ring capacity {} exceeded by run ({} pushed) — raise EVENT_RING_CAP",
        EVENT_RING_CAP,
        events.total_pushed(),
    );

    let mut out = String::with_capacity(events.len() * 64);
    // Header documents the fixture shape so a diff against the baseline
    // shows intent-level metadata, not just raw event lines.
    let _ = writeln!(
        out,
        "# wolves_and_humans parity baseline — seed={:#x} ticks={} agents={}",
        SEED,
        TICKS,
        state.agent_cap(),
    );
    let _ = writeln!(out, "# total_events={}", events.total_pushed());
    for ev in events.iter() {
        let _ = writeln!(out, "{}", fmt_event(ev));
    }
    out
}

/// Count events by variant tag across the log. Used to pin structural
/// invariants (e.g. "at least one AgentDied"). Keys are stable string
/// tags emitted by `fmt_event`'s leading token.
fn count_by_tag(log: &str) -> std::collections::BTreeMap<String, usize> {
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for line in log.lines() {
        if line.is_empty() || line.starts_with('#') { continue; }
        let tag = match line.find('(') {
            Some(i) => &line[..i],
            None => line,
        };
        *counts.entry(tag.to_string()).or_default() += 1;
    }
    counts
}

#[test]
fn parity_log_is_byte_identical_to_baseline() {
    let actual = run_scenario_log();

    // Regen escape hatch: `WOLVES_AND_HUMANS_REGEN=1 cargo test ...` writes
    // the fresh log to the committed baseline path and fails the test so the
    // diff still needs human sign-off. Not a CI bypass — CI doesn't set the
    // env var, so a baseline update can only come from a reviewer.
    if std::env::var_os("WOLVES_AND_HUMANS_REGEN").is_some() {
        std::fs::write(BASELINE_PATH, &actual).expect("regen: write baseline");
        panic!(
            "WOLVES_AND_HUMANS_REGEN set: wrote {} bytes to {}; re-run without the env var",
            actual.len(),
            BASELINE_PATH,
        );
    }

    let expected = std::fs::read_to_string(Path::new(BASELINE_PATH))
        .expect("read committed baseline at tests/wolves_and_humans_baseline.txt");

    if actual != expected {
        // Print a compact diff hint — the file-level diff lives in git.
        let actual_lines: Vec<&str> = actual.lines().collect();
        let expected_lines: Vec<&str> = expected.lines().collect();
        let max = actual_lines.len().max(expected_lines.len());
        let mut shown = 0;
        for i in 0..max {
            let a = actual_lines.get(i).copied().unwrap_or("<missing>");
            let e = expected_lines.get(i).copied().unwrap_or("<missing>");
            if a != e {
                eprintln!("line {}: expected {:?}\n           actual   {:?}", i + 1, e, a);
                shown += 1;
                if shown >= 20 { eprintln!("... (truncated)"); break; }
            }
        }
        panic!(
            "wolves+humans log diverged from committed baseline (\
             {} actual lines, {} expected lines). If the change is intentional, \
             re-run with WOLVES_AND_HUMANS_REGEN=1 to refresh the baseline.",
            actual_lines.len(),
            expected_lines.len(),
        );
    }
}

#[test]
fn parity_log_has_expected_structure() {
    let log = run_scenario_log();
    let counts = count_by_tag(&log);

    // Wolves are within 1 m of the closest human at t=0, so the mask's
    // Attack bit is set on tick 0 and at least one AgentAttacked event
    // fires in the first handful of ticks.
    let attacks = *counts.get("AgentAttacked").unwrap_or(&0);
    assert!(
        attacks > 0,
        "expected at least one AgentAttacked; got counts={:?}",
        counts,
    );

    // 2 wolves × DPS 10 × 100 ticks ≥ the 100 HP per human by a wide
    // margin, so at least one human must die inside the window. If this
    // drops to zero the wolves lost their teeth somewhere upstream.
    let deaths = *counts.get("AgentDied").unwrap_or(&0);
    assert!(
        deaths >= 1,
        "expected at least one AgentDied across {} ticks; got counts={:?}",
        TICKS, counts,
    );

    // Tick-ordering invariant: every event row's tick is monotonically
    // non-decreasing in the log (the ring is stamped with tick at push
    // time, and handlers in phase 4 run inside the same tick). This
    // catches accidental out-of-order emission.
    let mut last_tick: u32 = 0;
    for line in log.lines() {
        if line.is_empty() || line.starts_with('#') { continue; }
        let start = line.find("tick=").expect("every event carries tick=");
        let after = &line[start + "tick=".len()..];
        let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
        let t: u32 = after[..end].parse().expect("numeric tick");
        assert!(
            t >= last_tick,
            "event tick went backwards: {} after {} (line={:?})",
            t, last_tick, line,
        );
        last_tick = t;
    }

    // Hostility directionality pin: *most* AgentAttacked events must cross
    // species. The scoring table gates the Attack kind via the mask (which
    // does check `is_hostile`), but `UtilityBackend::build_action` picks
    // `nearest_other` regardless of hostility — so a same-species "stray"
    // attack can land when the Attack bit is mask-allowed (because some
    // hostile was in range) but the nearest neighbour is same-species.
    //
    // This is current DSL-owned behaviour as of compiler milestones 2-6.
    // Migrating target-selection to DSL (a follow-up to milestone 6) will
    // close the hole. For now we pin the ratio rather than assert zero.
    let state = spawn_fixture();
    let mut cross_species = 0usize;
    let mut same_species = 0usize;
    for line in log.lines() {
        if !line.starts_with("AgentAttacked(") { continue; }
        let attacker = parse_kv_u32(line, "attacker=").expect("attacker id");
        let target = parse_kv_u32(line, "target=").expect("target id");
        let a = engine::ids::AgentId::new(attacker).unwrap();
        let t = engine::ids::AgentId::new(target).unwrap();
        let ca = state.agent_creature_type(a).expect("attacker creature type");
        let ct = state.agent_creature_type(t).expect("target creature type");
        if ca == ct { same_species += 1; } else { cross_species += 1; }
    }
    assert!(
        cross_species >= same_species,
        "cross-species attacks {} should dominate same-species {} (counts={:?})",
        cross_species, same_species, counts,
    );
    assert!(
        cross_species > 0,
        "expected at least one cross-species attack (hostility mask in effect)",
    );
}

/// Parse a `key=<u32>` fragment out of a one-line record. Returns None if
/// the key isn't present.
fn parse_kv_u32(line: &str, key: &str) -> Option<u32> {
    let start = line.find(key)? + key.len();
    let after = &line[start..];
    let end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
    after[..end].parse().ok()
}

#[test]
fn parity_log_is_deterministic_across_runs() {
    // Two fresh runs with the same seed must produce the exact same log.
    // This is the "seeded scenario is actually deterministic" side of the
    // contract — the baseline comparison gives us persistence, but this
    // test gives us repeatability within a single `cargo test` invocation.
    let a = run_scenario_log();
    let b = run_scenario_log();
    assert_eq!(a, b, "re-running the same seed diverged within one process");
}
