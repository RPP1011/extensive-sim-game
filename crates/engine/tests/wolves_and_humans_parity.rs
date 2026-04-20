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
        Event::EngagementCommitted { actor, target, tick } => format!(
            "EngagementCommitted(tick={tick},actor={},target={})",
            actor.raw(), target.raw(),
        ),
        Event::EngagementBroken { actor, former_target, reason, tick } => format!(
            "EngagementBroken(tick={tick},actor={},former_target={},reason={})",
            actor.raw(), former_target.raw(), reason,
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

    // Hostility directionality pin: *every* AgentAttacked event must cross
    // species. Task 138 retired the `nearest_other` fallback; target
    // selection now argmaxes over the mask-produced candidate list, and
    // the Attack mask's `from query.nearby_agents(...) when is_hostile`
    // clause only enumerates hostile candidates. Same-species "stray"
    // attacks can no longer appear in the log — if one shows up, the
    // target-mask pipeline has regressed.
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

// ---------------------------------------------------------------------------
// Behavioural parity scenarios — task 141+ follow-up.
//
// The three tests above pin the CURRENT event log against a baseline. That
// catches silent drift but not the "scoring doesn't actually do anything"
// failure mode: a fixture whose behaviour is an accident of ties in the
// argmax would still match its own regenerated baseline.
//
// The five tests below each seed a custom arrangement of creatures, run a
// short window under the stock `UtilityBackend` + 6-phase `step_full`
// pipeline, and assert load-bearing properties of the event log. Together
// they cover: hp-biased target selection, flee-when-threatened, hostility-
// matrix enforcement, apex-predator reach, and engagement stickiness.
//
// Helpers are factored so each test reads as setup → run → assert. Every
// test uses a fixed seed; the run loops are small (20–100 ticks) so
// failures point at first-tick behaviour rather than late-game drift.
// ---------------------------------------------------------------------------

/// Per-test spawn spec: creature type, starting position, starting HP.
#[derive(Clone, Copy)]
struct CreatureSpawn {
    creature_type: CreatureType,
    pos: Vec3,
    hp: f32,
}

/// Build a `SimState` with the given creatures and run it for `ticks` ticks
/// under the stock scorer + builtin cascades. Returns the final state plus
/// the full event log (in push order). The `ids` slice returned lines up
/// with the input `spawns` slice so callers can assert on specific agents.
fn run_behavioural_scenario(
    seed: u64,
    spawns: &[CreatureSpawn],
    ticks: u32,
) -> (SimState, Vec<Event>, Vec<engine::ids::AgentId>) {
    // Agent cap just needs to cover the spawn list; give a little headroom
    // for any cascade that might allocate more slots (none do today, but a
    // +4 cushion is free insurance).
    let cap = (spawns.len() as u32 + 4).max(4);
    let mut state = SimState::new(cap, seed);
    let mut ids = Vec::with_capacity(spawns.len());
    for s in spawns {
        let id = state
            .spawn_agent(AgentSpawn {
                creature_type: s.creature_type,
                pos: s.pos,
                hp: s.hp,
            })
            .expect("spawn inside agent cap");
        ids.push(id);
    }

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    // Generous ring — each test is under 200 ticks × <10 events/tick.
    let mut events = EventRing::with_cap(1 << 14);
    let cascade = CascadeRegistry::with_engine_builtins();
    let invariants = InvariantRegistry::new();
    let mut views: Vec<&mut dyn MaterializedView> = Vec::new();
    let telemetry = NullSink;

    for _ in 0..ticks {
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

    let log: Vec<Event> = events.iter().copied().collect();
    (state, log, ids)
}

/// (1) Wolves prefer wounded humans.
///
/// Setup: a single wolf at the origin with two humans both inside its 2 m
/// attack range — H1 barely alive (hp=10), H2 healthy (hp=100). Because H1
/// sits one hit from death and the engine deals fixed 10 damage/tick, the
/// very first attack the wolf lands on H1 kills it. So if the scorer /
/// target-argmax does anything sensible at all with the wounded candidate
/// — or even if it just happens to fire enough attacks before switching —
/// H1 should die first.
///
/// This test is deliberately weak: it does NOT require an hp-pct
/// preference modifier to be wired into scoring.sim. It only asserts that
/// (a) the wolf kills at least one human in the window, and (b) the first
/// death is H1. If H2 dies first (or nobody dies), the wolf is attacking
/// the healthy target preferentially, which is the opposite of
/// "intelligent" and would reveal a regression.
#[test]
fn wolves_prefer_wounded_humans() {
    let spawns = [
        // Wolf at origin — id will be 1.
        CreatureSpawn { creature_type: CreatureType::Wolf,  pos: Vec3::new(0.0, 0.0, 0.0), hp: 80.0 },
        // H1: wounded, near — id will be 2.
        CreatureSpawn { creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 10.0 },
        // H2: healthy, slightly farther but still inside 2 m attack range — id will be 3.
        CreatureSpawn { creature_type: CreatureType::Human, pos: Vec3::new(1.5, 0.0, 0.0), hp: 100.0 },
    ];
    let (_state, log, ids) = run_behavioural_scenario(0xBEEF_0001, &spawns, 20);
    let (wolf, h1, h2) = (ids[0], ids[1], ids[2]);

    // At least one human must die — sanity check, not the real assertion.
    let deaths: Vec<engine::ids::AgentId> = log
        .iter()
        .filter_map(|e| match e {
            Event::AgentDied { agent_id, .. } => Some(*agent_id),
            _ => None,
        })
        .collect();
    assert!(
        !deaths.is_empty(),
        "wolf produced zero kills in 20 ticks against two in-range humans — \
         attack pipeline regressed?",
    );
    // The first death must be H1 (hp=10 dies in one hit; if H2 dies first
    // the wolf spent ≥10 attacks on the healthy target, which is the
    // behavioural failure this test guards against).
    assert_eq!(
        deaths[0], h1,
        "first death should be the wounded human h1={} (h2={}, wolf={}); got {}",
        h1.raw(), h2.raw(), wolf.raw(), deaths[0].raw(),
    );

    // Companion check: a majority of the wolf's attacks land on H1. This
    // is the stronger form of the assertion — an hp-aware scorer should
    // concentrate attacks on the lower-hp target. Tolerance is "strict
    // majority" (> 50 %) to leave slack for tick-0 tie resolution.
    let (mut hits_h1, mut hits_h2) = (0usize, 0usize);
    for e in &log {
        if let Event::AgentAttacked { actor, target, .. } = *e {
            if actor == wolf {
                if target == h1 { hits_h1 += 1; }
                else if target == h2 { hits_h2 += 1; }
            }
        }
    }
    assert!(
        hits_h1 > hits_h2,
        "wolf hits on h1={} should exceed hits on h2={} (wolf={}); \
         got hits_h1={}, hits_h2={} — scorer may not prefer wounded targets",
        h1.raw(), h2.raw(), wolf.raw(), hits_h1, hits_h2,
    );
}

/// (2) Deer flee from wolves.
///
/// Setup: one wolf at the origin, one healthy deer 2 m away. The wolf is
/// within MoveToward radius (20 m) of the deer and hostile, so the deer
/// observing a predator nearby should produce movement AWAY from it.
///
/// Assertion: distance(wolf, deer) at tick 100 > distance at tick 0. If
/// the deer instead closed on the wolf (MoveToward picked the wolf as
/// its nearest candidate because Flee doesn't fire at full hp, or because
/// Flee's action-builder falls back to Hold), this fails and exposes the
/// regression.
#[test]
fn deer_flee_from_wolves() {
    let spawns = [
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new(0.0, 0.0, 0.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Deer, pos: Vec3::new(2.0, 0.0, 0.0), hp: 40.0 },
    ];
    let initial_distance = (spawns[1].pos - spawns[0].pos).length();
    let (state, _log, ids) = run_behavioural_scenario(0xBEEF_0002, &spawns, 100);
    let (wolf, deer) = (ids[0], ids[1]);

    // The deer might have died — a legitimate "wolves win" outcome. But
    // the stronger behavioural claim is still about what the deer did
    // *while alive*: net movement should be away from the wolf. We read
    // the deer's final position from the state; if it died mid-run the
    // final position is the last-alive pos (set_agent_pos never runs on
    // a dead agent — kill_agent clears `alive` before any later move).
    let final_wolf_pos = state.agent_pos(wolf).expect("wolf pos");
    let final_deer_pos = state.agent_pos(deer).expect("deer pos");
    let final_distance = final_deer_pos.distance(final_wolf_pos);

    assert!(
        final_distance > initial_distance,
        "deer should move AWAY from wolf; initial_distance={:.3}, \
         final_distance={:.3} (wolf at {:?}, deer at {:?}) — \
         Flee mask/scoring may not be firing for healthy prey",
        initial_distance,
        final_distance,
        final_wolf_pos,
        final_deer_pos,
    );
}

/// (3) Wolves don't attack wolves.
///
/// Setup: five wolves clustered within attack range of each other, no
/// humans / deer / dragons. Every pair is a hostile-mask rejection (the
/// predator/prey table is symmetric and Wolf doesn't prey on Wolf), so
/// the Attack mask enumerator should never emit a same-species candidate
/// and the argmax should never produce a wolf-on-wolf attack.
///
/// Assertion: zero `AgentAttacked` events where both actor and target
/// are wolves. (Any non-zero count points at a regression in `is_hostile`
/// or the mask's `when is_hostile(self, target)` clause.)
#[test]
fn wolves_dont_attack_wolves() {
    // Five wolves in a tight pentagon, all within 2 m of at least one
    // neighbour. If hostility gating broke, every pair is a valid Attack
    // candidate — the log would fill with wolf-on-wolf events.
    let spawns = [
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 0.0,  0.0,  0.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 1.0,  0.0,  0.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new(-1.0,  0.0,  0.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 0.0,  0.0,  1.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 0.0,  0.0, -1.0), hp: 80.0 },
    ];
    let (state, log, _ids) = run_behavioural_scenario(0xBEEF_0003, &spawns, 50);

    let mut same_species_attacks = 0usize;
    for e in &log {
        if let Event::AgentAttacked { actor, target, .. } = *e {
            let ca = state.agent_creature_type(actor);
            let ct = state.agent_creature_type(target);
            if ca == Some(CreatureType::Wolf) && ct == Some(CreatureType::Wolf) {
                same_species_attacks += 1;
            }
        }
    }
    assert_eq!(
        same_species_attacks, 0,
        "expected zero wolf-on-wolf attacks; got {} (hostility gate regressed?)",
        same_species_attacks,
    );
}

/// (4) Dragon attacks all.
///
/// Setup: one dragon at the origin with one human, one wolf, and one deer
/// arranged within attack range. The dragon's `preys_on` list is
/// `[Human, Wolf, Deer]` — every other creature is hostile to it, so the
/// Attack mask enumerator should emit all three as candidates.
///
/// Assertion: across 50 ticks the dragon attacks at least two of the three
/// other creatures. (Proving "all three" is a tighter claim but can fail
/// on sticky engagement — once the first target dies the engagement lock
/// drops and the dragon moves to the next, but 50 ticks may not cover the
/// third. Two out of three is the meaningful minimum: it rules out the
/// hostility matrix collapsing to a single predator/prey pair.)
#[test]
fn dragon_attacks_all() {
    let spawns = [
        CreatureSpawn { creature_type: CreatureType::Dragon, pos: Vec3::new(0.0, 0.0, 0.0), hp: 500.0 },
        CreatureSpawn { creature_type: CreatureType::Human,  pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0 },
        CreatureSpawn { creature_type: CreatureType::Wolf,   pos: Vec3::new(1.5, 0.0, 0.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Deer,   pos: Vec3::new(0.0, 0.0, 1.5), hp: 40.0 },
    ];
    let (_state, log, ids) = run_behavioural_scenario(0xBEEF_0004, &spawns, 50);
    let (dragon, human, wolf, deer) = (ids[0], ids[1], ids[2], ids[3]);

    // Track distinct targets the dragon attacked.
    let mut attacked_human = false;
    let mut attacked_wolf = false;
    let mut attacked_deer = false;
    for e in &log {
        if let Event::AgentAttacked { actor, target, .. } = *e {
            if actor == dragon {
                if target == human { attacked_human = true; }
                if target == wolf  { attacked_wolf = true; }
                if target == deer  { attacked_deer = true; }
            }
        }
    }

    let distinct = [attacked_human, attacked_wolf, attacked_deer]
        .iter()
        .filter(|&&b| b)
        .count();
    assert!(
        distinct >= 2,
        "apex dragon should attack at least two of its three possible prey \
         over 50 ticks; got distinct={} (human={}, wolf={}, deer={}) — \
         dragon predator_prey coverage regressed?",
        distinct, attacked_human, attacked_wolf, attacked_deer,
    );
}

/// (5) Engaged wolves stay committed.
///
/// Setup: two wolves (W_a at x=0, W_b at x=5) and two humans (H_1 at x=1,
/// H_2 at x=4). Each wolf is within attack + engagement range of exactly
/// one human at spawn time, so first-tick engagement assignments are
/// fully determined by geometry. We run 30 ticks and check that once an
/// `EngagementCommitted` event fires for a (wolf, human) pair, the same
/// pair persists for at least 3 ticks before any `EngagementBroken` for
/// that wolf fires.
///
/// This guards against a tick-by-tick oscillation regression: if the
/// event-driven engagement recompute fired `Broken → Committed` in the
/// same tick on every move, the scorer would bounce between candidates
/// and the log would show near-continuous churn. The current pipeline
/// only recomputes on `AgentMoved` and `AgentDied`, so a stable pairing
/// should hold until one side dies.
#[test]
fn engaged_wolves_stay_committed() {
    let spawns = [
        CreatureSpawn { creature_type: CreatureType::Wolf,  pos: Vec3::new(0.0, 0.0, 0.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Wolf,  pos: Vec3::new(5.0, 0.0, 0.0), hp: 80.0 },
        CreatureSpawn { creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0 },
        CreatureSpawn { creature_type: CreatureType::Human, pos: Vec3::new(4.0, 0.0, 0.0), hp: 100.0 },
    ];
    let (_state, log, ids) = run_behavioural_scenario(0xBEEF_0005, &spawns, 30);
    let (wolf_a, wolf_b) = (ids[0], ids[1]);

    // Walk the log per-wolf and measure commitment duration. For each
    // `EngagementCommitted { actor=wolf, target=T, tick=t }`, find the
    // next event on the same wolf that changes partner: either an
    // `EngagementBroken { actor=wolf, .. }` or a subsequent
    // `EngagementCommitted { actor=wolf, target != T, .. }`. Duration is
    // the tick delta. A legitimate short span is one that ends with
    // `EngagementBroken { reason=PARTNER_DIED }` — the target being
    // killed is not "oscillation", it's finishing the fight.
    fn max_stable_span(log: &[Event], wolf: engine::ids::AgentId) -> Option<u32> {
        let mut best: Option<u32> = None;
        for i in 0..log.len() {
            let (commit_tick, commit_target) = match log[i] {
                Event::EngagementCommitted { actor, target, tick } if actor == wolf => (tick, target),
                _ => continue,
            };
            let mut end_tick = None;
            let mut ended_by_death = false;
            for j in (i + 1)..log.len() {
                match log[j] {
                    Event::EngagementBroken { actor, reason, tick, .. } if actor == wolf => {
                        end_tick = Some(tick);
                        // reason 2 == PARTNER_DIED — see
                        // `engagement::break_reason`. Short spans ending
                        // in death are excluded from the stickiness
                        // criterion because they reflect target kills,
                        // not target churn.
                        ended_by_death = reason == 2;
                        break;
                    }
                    Event::EngagementCommitted { actor, target, tick, .. }
                        if actor == wolf && target != commit_target =>
                    {
                        end_tick = Some(tick);
                        break;
                    }
                    _ => {}
                }
            }
            let span = match end_tick {
                Some(t) => t.saturating_sub(commit_tick),
                None => 30u32.saturating_sub(commit_tick), // still engaged at end-of-run
            };
            if ended_by_death {
                // Treat as fully-sticky: the wolf didn't switch, the
                // target was eliminated.
                return Some(u32::MAX);
            }
            best = Some(best.map_or(span, |b| b.max(span)));
        }
        best
    }

    // At minimum, one of the two wolves must hold its first engagement
    // for ≥ 3 ticks (or resolve it by killing the target, which we treat
    // as a fully-sticky outcome). The 3-tick threshold is the loose
    // tolerance called out in the test plan — it rules out per-tick
    // oscillation without demanding a specific kill/switch cadence.
    let a = max_stable_span(&log, wolf_a).unwrap_or(0);
    let b = max_stable_span(&log, wolf_b).unwrap_or(0);
    assert!(
        a >= 3 || b >= 3,
        "at least one wolf should hold an engagement for ≥3 ticks before \
         switching / dying; got max_span(wolf_a={})={}, max_span(wolf_b={})={} \
         — engagement may be churning per-tick",
        wolf_a.raw(), a, wolf_b.raw(), b,
    );
}
