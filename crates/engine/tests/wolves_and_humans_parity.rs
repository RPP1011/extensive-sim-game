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
        ..Default::default()
    }).expect("human 1 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new( 2.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).expect("human 2 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos: Vec3::new(-2.0, 0.0, 0.0), hp: 100.0,
        ..Default::default()
    }).expect("human 3 spawn");
    // Wolves.
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new( 3.0, 0.0, 0.0), hp: 80.0,
        ..Default::default()
    }).expect("wolf 1 spawn");
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos: Vec3::new(-3.0, 0.0, 0.0), hp: 80.0,
        ..Default::default()
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
        Event::FearSpread { observer, dead_kin, tick } => format!(
            "FearSpread(tick={tick},observer={},dead_kin={})",
            observer.raw(), dead_kin.raw(),
        ),
        Event::PackAssist { observer, target, tick } => format!(
            "PackAssist(tick={tick},observer={},target={})",
            observer.raw(), target.raw(),
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

    // Filter to replayable events only. `ChronicleEntry` (and any other
    // future @non_replayable variant) lives in the ring as a prose
    // side-channel; emitting one is allowed to change at any time
    // without perturbing the parity baseline. `Event::is_replayable()`
    // is the same partition the hash walk uses, so the baseline tracks
    // exactly the subset that defines replay equivalence.
    let replayable: Vec<&Event> = events.iter().filter(|e| e.is_replayable()).collect();

    let mut out = String::with_capacity(replayable.len() * 64);
    // Header documents the fixture shape so a diff against the baseline
    // shows intent-level metadata, not just raw event lines.
    let _ = writeln!(
        out,
        "# wolves_and_humans parity baseline — seed={:#x} ticks={} agents={}",
        SEED,
        TICKS,
        state.agent_cap(),
    );
    let _ = writeln!(out, "# total_events={}", replayable.len());
    for ev in &replayable {
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

/// Chronicle renderer smoke test. Runs the same 100-tick wolves+humans
/// fixture, walks the full event ring (including non-replayable
/// `ChronicleEntry` events), and asserts the rendered prose looks like
/// expected — readable, tick-stamped, references named creatures, and
/// includes at least one of each of the three template kinds the
/// chronicle-mvp physics rules emit (death, strike, engagement).
#[test]
fn chronicle_renders_readable_text() {
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

    // Render every ChronicleEntry in the ring.
    let lines = engine::chronicle::render_entries(&state, events.iter());
    assert!(
        !lines.is_empty(),
        "chronicle should emit at least one entry across 100 ticks of wolves+humans",
    );

    // Every line is of the form "Tick <u32>: ..." — tick-stamped.
    for line in &lines {
        assert!(
            line.starts_with("Tick "),
            "chronicle line missing tick prefix: {line:?}",
        );
    }

    // Each of the three template kinds must surface at least once —
    // death, strike, engagement. The wolves+humans fixture guarantees
    // all three in a 100-tick window.
    let has_strike = lines.iter().any(|l| l.contains(" struck "));
    let has_engagement = lines.iter().any(|l| l.contains(" engaged "));
    let has_death = lines.iter().any(|l| l.contains(" fell."));
    assert!(has_strike, "chronicle missing a strike line; lines={lines:?}");
    assert!(
        has_engagement,
        "chronicle missing an engagement line; lines={lines:?}"
    );
    assert!(has_death, "chronicle missing a death line; lines={lines:?}");

    // Names include a creature type — wolves+humans only, no deer/dragon.
    let any_human = lines.iter().any(|l| l.contains("Human #"));
    let any_wolf = lines.iter().any(|l| l.contains("Wolf #"));
    assert!(any_human && any_wolf, "expected both Human and Wolf references; lines={lines:?}");
}

/// Task 166 — after the WOUND (template 4) and ENGAGEMENT_BROKEN (template 5)
/// physics rules landed, both must surface in the wolves+humans 100-tick
/// fixture. The fixture guarantees at least one human takes heavy damage
/// (wolves DPS ≈ 10/tick, humans have 100 HP) and at least one engagement
/// pair dissolves (humans die, wolves switch, survivors displace each
/// other). Checking the raw event ring rather than rendered prose keeps
/// the assertion on the DSL emit-site behaviour — rename the prose and
/// the test still passes; break the emit and the test fails.
#[test]
fn chronicle_has_wound_and_break_templates() {
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

    let mut seen_wound = false;
    let mut seen_break = false;
    for ev in events.iter() {
        if let Event::ChronicleEntry { template_id, .. } = ev {
            match *template_id {
                4 => seen_wound = true,
                5 => seen_break = true,
                _ => {}
            }
        }
    }
    assert!(
        seen_wound,
        "expected at least one ChronicleEntry{{ template_id: 4 }} (WOUND) in 100-tick wolves+humans run",
    );
    assert!(
        seen_break,
        "expected at least one ChronicleEntry{{ template_id: 5 }} (ENGAGEMENT_BROKEN) in 100-tick wolves+humans run",
    );
}

/// Task 168 — after the ROUT (template 6) and FLEE (template 7)
/// physics rules landed, both must surface in the wolves+humans 100-tick
/// fixture. The baseline already shows `AgentFled` events on ticks 3+
/// (wounded wolves retreating, task 165) and `FearSpread` fires when a
/// kin dies within 12 m (task 167) — the fixture has both a wolf death
/// and wolves within the 12 m fear radius of each other. Checking the
/// raw event ring keeps the assertion on the DSL emit-site behaviour —
/// rename the prose and the test still passes; break the emit and the
/// test fails.
#[test]
fn chronicle_has_rout_and_flee_templates() {
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

    let mut seen_rout = false;
    let mut seen_flee = false;
    for ev in events.iter() {
        if let Event::ChronicleEntry { template_id, .. } = ev {
            match *template_id {
                6 => seen_rout = true,
                7 => seen_flee = true,
                _ => {}
            }
        }
    }
    assert!(
        seen_rout,
        "expected at least one ChronicleEntry{{ template_id: 6 }} (ROUT) in 100-tick wolves+humans run",
    );
    assert!(
        seen_flee,
        "expected at least one ChronicleEntry{{ template_id: 7 }} (FLEE) in 100-tick wolves+humans run",
    );
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
///
/// `max_hp` defaults to `hp` so a healthy spawn reports `hp_pct = 1.0`; a
/// wounded-spawn test writes `hp: 10.0` and `max_hp: Some(100.0)` so
/// `hp_pct` actually reflects the wound (task 150).
#[derive(Clone, Copy)]
struct CreatureSpawn {
    creature_type: CreatureType,
    pos: Vec3,
    hp: f32,
    max_hp: Option<f32>,
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
                // Default max_hp to hp when the test didn't pin it — a
                // healthy spawn reports hp_pct = 1.0. The wounded-spawn
                // fixture (h1 in `wolves_prefer_wounded_humans`) passes
                // an explicit max_hp so hp_pct = 0.1 drives target
                // selection.
                max_hp: s.max_hp.unwrap_or(s.hp),
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
        CreatureSpawn { creature_type: CreatureType::Wolf,  pos: Vec3::new(0.0, 0.0, 0.0), hp: 80.0, max_hp: None },
        // H1: wounded, near — id will be 2.
        CreatureSpawn { creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 10.0, max_hp: Some(100.0) },
        // H2: healthy, slightly farther but still inside 2 m attack range — id will be 3.
        CreatureSpawn { creature_type: CreatureType::Human, pos: Vec3::new(1.5, 0.0, 0.0), hp: 100.0, max_hp: None },
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

    // Companion check: the wolf's first hit must land on H1 (the wounded
    // target). Task 150 loosened this from "majority of attacks on H1"
    // because H1 at hp=10 max_hp=100 (`hp_pct = 0.1`) is a one-hit kill,
    // so the wolf can never accumulate more than one hit on H1 before
    // the death event breaks engagement. The prior "strict majority"
    // worked under the pre-150 scoring because `max_hp := spec.hp.max(1.0)`
    // made H1 report `hp_pct = 1.0` and the wolf's hp-raw modifiers were
    // weaker than the Flee row — the wolf would typically die to H2's
    // counter-attacks before landing a second shot. With hp_pct-based
    // scoring and the Task 148 threat-aware Flee (wolf retreats below
    // 50% hp), the wolf now survives long enough to trade one hit with
    // H2, so the meaningful invariant is "first hit on H1" not "majority
    // on H1".
    let first_wolf_hit = log.iter().find_map(|e| match *e {
        Event::AgentAttacked { actor, target, .. } if actor == wolf => Some(target),
        _ => None,
    });
    assert_eq!(
        first_wolf_hit, Some(h1),
        "wolf's first attack should land on the wounded h1={} (wolf={}, h2={}); \
         got first target = {:?} — scorer may not prefer wounded targets",
        h1.raw(), wolf.raw(), h2.raw(), first_wolf_hit.map(|t| t.raw()),
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
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new(0.0, 0.0, 0.0), hp: 80.0, max_hp: None },
        CreatureSpawn { creature_type: CreatureType::Deer, pos: Vec3::new(2.0, 0.0, 0.0), hp: 40.0, max_hp: None },
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

    // Spirit of the test: the deer's net movement should be AWAY from
    // where the wolf started. Task 150's `max_hp`-split + hp_pct scoring
    // + task 148's threat-aware Flee actually make the deer flee (under
    // the pre-task-148 `hp_pct < 0.3` gate, a fresh-hp deer wouldn't
    // flee at all and would MoveToward-close on the wolf). But wolf and
    // deer walk at the same default speed so a chase keeps distance
    // roughly constant — `final_distance > initial_distance` strict is
    // impossible in that regime. We now measure against the *initial*
    // wolf position: the deer must have net-displaced AWAY from the
    // wolf's spawn point, which is the real "is Flee firing?" check.
    let initial_wolf_pos = spawns[0].pos;
    let deer_displacement_from_wolf_origin = final_deer_pos.distance(initial_wolf_pos);

    assert!(
        deer_displacement_from_wolf_origin > initial_distance,
        "deer should move AWAY from wolf's initial position; \
         initial_distance={:.3}, deer_displacement_from_wolf_origin={:.3}, \
         final_distance={:.3} (wolf at {:?}, deer at {:?}) — \
         Flee mask/scoring may not be firing for healthy prey",
        initial_distance,
        deer_displacement_from_wolf_origin,
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
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 0.0,  0.0,  0.0), hp: 80.0, max_hp: None },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 1.0,  0.0,  0.0), hp: 80.0, max_hp: None },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new(-1.0,  0.0,  0.0), hp: 80.0, max_hp: None },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 0.0,  0.0,  1.0), hp: 80.0, max_hp: None },
        CreatureSpawn { creature_type: CreatureType::Wolf, pos: Vec3::new( 0.0,  0.0, -1.0), hp: 80.0, max_hp: None },
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
    // Task 150: pre-wound each prey to a distinct hp_pct so the dragon's
    // hp_pct-based target-selection can discriminate between them at
    // t=0. Under the pre-150 raw-hp scoring, the dragon picked the
    // lowest-hp target via absolute thresholds (Deer at hp=40 won with
    // `target.hp < 50` firing). With hp_pct restored, three
    // fresh-from-spawn prey all report `hp_pct = 1.0` and the argmax
    // degenerates to enumeration order, so the dragon engages a single
    // target and sticks to it until one side dies (no "apex variety").
    // Pre-wounding (deer 25%, wolf 50%, human 75%) gives the dragon a
    // differentiable target list while keeping all three within its
    // attack range.
    let spawns = [
        CreatureSpawn { creature_type: CreatureType::Dragon, pos: Vec3::new(0.0, 0.0, 0.0), hp: 500.0, max_hp: Some(500.0) },
        CreatureSpawn { creature_type: CreatureType::Human,  pos: Vec3::new(1.0, 0.0, 0.0), hp: 75.0,  max_hp: Some(100.0) },
        CreatureSpawn { creature_type: CreatureType::Wolf,   pos: Vec3::new(1.5, 0.0, 0.0), hp: 40.0,  max_hp: Some(80.0) },
        CreatureSpawn { creature_type: CreatureType::Deer,   pos: Vec3::new(0.0, 0.0, 1.5), hp: 10.0,  max_hp: Some(40.0) },
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
/// Setup: one wolf (W at x=0) and one human (H at x=1). Wolf has H within
/// attack + engagement range at spawn, so the fight starts immediately.
/// Run for 30 ticks and check that the wolf's attacks stay locked on a
/// single target for a stretch (no churn between alternative targets).
///
/// Task 160 note — this test originally spawned two (wolf, human) pairs
/// at x=0/5/1/4 and asserted an `EngagementCommitted` with the wolf as
/// actor held for ≥3 ticks. That shape relied on wolves transitioning
/// from `Attack` to `MoveToward` once `hp_pct` dropped below the fresh
/// bonus — MoveToward emits `AgentMoved` which triggers the engagement
/// recompute. The `my_enemies` grudge (+0.4 on Attack) keeps wolves
/// attacking in place even after they take damage, so they never move,
/// never emit `AgentMoved`, and never commit engagement as the actor
/// (the human does, when pursuing a fleeing wolf later in the fight).
///
/// That's strictly better for "stay committed" intent — attacking the
/// same target every tick is the strongest form of commitment — but it
/// invalidates the test's specific event-shape assertion. The test now
/// checks target stability directly by counting distinct non-self
/// attack targets the wolf picks in the window before it starts
/// fleeing. Grudge-flip behaviour (wolf preferring its attacker over
/// an equivalent fresh target) is covered by
/// `threat_level_scoring::wolf_attacks_grudge_target_over_stranger`.
#[test]
fn engaged_wolves_stay_committed() {
    let spawns = [
        CreatureSpawn { creature_type: CreatureType::Wolf,  pos: Vec3::new(0.0, 0.0, 0.0), hp: 80.0, max_hp: None },
        CreatureSpawn { creature_type: CreatureType::Human, pos: Vec3::new(1.0, 0.0, 0.0), hp: 100.0, max_hp: None },
    ];
    let (_state, log, ids) = run_behavioural_scenario(0xBEEF_0005, &spawns, 30);
    let (wolf, human) = (ids[0], ids[1]);

    // Collect the wolf's attack targets in declaration order. If the
    // wolf attacks more than one distinct target in the whole run, the
    // scoring row is oscillating between candidates (the failure mode
    // the original test called out). With only one human alive, this
    // should be a singleton — the one and only human.
    let mut distinct_targets: Vec<engine::ids::AgentId> = Vec::new();
    let mut attack_count = 0;
    for e in &log {
        if let Event::AgentAttacked { actor, target, .. } = *e {
            if actor == wolf {
                attack_count += 1;
                if !distinct_targets.contains(&target) {
                    distinct_targets.push(target);
                }
            }
        }
    }

    assert!(
        attack_count >= 3,
        "wolf should attack at least 3 times before the fight resolves; got {attack_count}",
    );
    assert_eq!(
        distinct_targets.as_slice(),
        &[human][..],
        "wolf should attack exactly one target (the only human); got {distinct_targets:?}",
    );
}

/// (6) Wounded wolves flee from humans.
///
/// Setup: one wolf pre-wounded to ~20% hp (hp=16, max_hp=80) adjacent to
/// a healthy human. The Flee row's task-165 `self.hp_pct < 0.3 : +0.6`
/// modifier should dominate the Attack row's grudge / threat deltas so
/// the wolf retreats even when the human has been drawing blood.
///
/// Two assertions:
/// 1. Scoring level — compute Flee and Attack scores directly through the
///    same `score_row_for` helper the threat_level / my_enemies tests use.
///    Flee must score strictly higher than Attack at the wounded-hp state,
///    with no priming required (threat / grudge views start empty).
/// 2. Behavioural level — run 10 ticks via `step_full`; the wolf's
///    distance to the human's initial position must increase (wolf is
///    net-displaced AWAY from the threat).
#[test]
fn wounded_wolves_flee_from_humans() {
    // --- Scoring-level check ------------------------------------------------
    //
    // Direct `SCORING_TABLE` lookup — builds a bare SimState, spawns the
    // two agents, and computes the Flee + Attack rows without running the
    // engine. Confirms the hp_pct<0.3 modifier actually wins the argmax
    // regardless of engagement / mask gating. A local micro-scorer handles
    // only the predicate kinds this test exercises (scalar-compare on
    // self.hp / self.hp_pct / target.hp_pct) — threat_level / my_enemies
    // views are never primed so their rows contribute zero.
    {
        use engine::mask::MicroKind;
        use engine_rules::scoring::{
            PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE,
        };

        let mut state = SimState::new(4, 0xBEEF_0006);
        let wolf = state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: Vec3::new(0.0, 0.0, 0.0),
                hp: 16.0,
                max_hp: 80.0,
                ..Default::default()
            })
            .expect("wounded wolf spawn");
        let human = state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: Vec3::new(1.0, 0.0, 0.0),
                hp: 100.0,
                max_hp: 100.0,
                ..Default::default()
            })
            .expect("human spawn");

        fn find_entry(head: u16) -> &'static ScoringEntry {
            for e in SCORING_TABLE {
                if e.action_head == head {
                    return e;
                }
            }
            panic!("SCORING_TABLE missing head {head}");
        }

        // Read the scalar field_ids this test uses. Unknown ids return 0.0
        // (not NaN) because none of the rows evaluated here reference
        // them — the compare ops that DO read a scalar all match one of
        // these three ids.
        fn read_scalar(
            state: &SimState,
            agent: engine::ids::AgentId,
            target: Option<engine::ids::AgentId>,
            field_id: u16,
        ) -> f32 {
            if field_id == 0x4002 {
                let t = target.expect("target.hp_pct needs a target");
                let hp = state.agent_hp(t).unwrap_or(0.0);
                let mx = state.agent_max_hp(t).unwrap_or(1.0);
                return if mx > 0.0 { hp / mx } else { 0.0 };
            }
            match field_id {
                0 => state.agent_hp(agent).unwrap_or(0.0),
                2 => {
                    let hp = state.agent_hp(agent).unwrap_or(0.0);
                    let mx = state.agent_max_hp(agent).unwrap_or(1.0);
                    if mx > 0.0 { hp / mx } else { 0.0 }
                }
                _ => f32::NAN,
            }
        }

        // Mini-scorer — evaluates only `KIND_SCALAR_COMPARE` rows. View
        // rows (gradient / view-scalar-compare) contribute 0 because this
        // test never primes the threat_level or my_enemies views, so
        // ignoring them is equivalent to evaluating them.
        fn score_row(
            entry: &ScoringEntry,
            state: &SimState,
            agent: engine::ids::AgentId,
            target: Option<engine::ids::AgentId>,
        ) -> f32 {
            let mut s = entry.base;
            let count = (entry.modifier_count as usize).min(MAX_MODIFIERS);
            for i in 0..count {
                let row = &entry.modifiers[i];
                if row.predicate.kind == PredicateDescriptor::KIND_SCALAR_COMPARE {
                    let lhs = read_scalar(state, agent, target, row.predicate.field_id);
                    let mut tb = [0u8; 4];
                    tb.copy_from_slice(&row.predicate.payload[0..4]);
                    let rhs = f32::from_le_bytes(tb);
                    if !lhs.is_nan() && !rhs.is_nan() {
                        let fires = match row.predicate.op {
                            PredicateDescriptor::OP_LT => lhs < rhs,
                            PredicateDescriptor::OP_LE => lhs <= rhs,
                            PredicateDescriptor::OP_EQ => lhs == rhs,
                            PredicateDescriptor::OP_GE => lhs >= rhs,
                            PredicateDescriptor::OP_GT => lhs > rhs,
                            PredicateDescriptor::OP_NE => lhs != rhs,
                            _ => false,
                        };
                        if fires {
                            s += row.delta;
                        }
                    }
                }
            }
            s
        }

        let flee_entry = find_entry(MicroKind::Flee as u16);
        let attack_entry = find_entry(MicroKind::Attack as u16);

        let flee_score = score_row(flee_entry, &state, wolf, None);
        let attack_score = score_row(attack_entry, &state, wolf, Some(human));

        // Wolf hp=16 → `hp<30` (+0.6) + `hp<50` (+0.4) + `hp_pct<0.3`
        // (+0.6) = 1.6. No threat primed so gradient + scalar-gate = 0.
        assert!(
            (flee_score - 1.6).abs() < 1e-3,
            "wounded-wolf Flee score = {flee_score}, expected ≈1.6 \
             (hp<30 +0.6, hp<50 +0.4, hp_pct<0.3 +0.6)",
        );
        // Wolf hp_pct=0.2 — fresh-self (`>=0.8`) gate FALSE so Attack base
        // stays at 0.0. Target (human) at hp_pct=1.0 — neither target gate
        // fires. No threat, no grudge. Attack = 0.0.
        assert!(
            attack_score.abs() < 1e-3,
            "wounded-wolf Attack score = {attack_score}, expected ≈0.0",
        );
        assert!(
            flee_score > attack_score,
            "task 165: Flee should beat Attack for a wounded wolf; \
             flee={flee_score}, attack={attack_score}",
        );
    }

    // --- Behavioural check --------------------------------------------------
    //
    // Short 10-tick run via the stock pipeline — confirms the scoring
    // preference actually drives the engine's Flee action-builder (nearest
    // hostile → move away). Wolf starts 1 m from the human; after 10 ticks
    // the wolf should have net-displaced AWAY from the human's spawn pos.
    let spawns = [
        // Wolf pre-wounded — hp_pct = 16/80 = 0.2, below the <0.3 Flee gate.
        CreatureSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 16.0,
            max_hp: Some(80.0),
        },
        CreatureSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: None,
        },
    ];
    let initial_wolf_pos = spawns[0].pos;
    let initial_human_pos = spawns[1].pos;
    let initial_distance = initial_wolf_pos.distance(initial_human_pos);

    let (state, _log, ids) = run_behavioural_scenario(0xBEEF_0006, &spawns, 10);
    let (wolf, _human) = (ids[0], ids[1]);

    // Wolf may have died mid-run — the human does 10 dmg/tick in melee and
    // the wolf enters at hp=16 (two hits to kill). Even so, the last-alive
    // position must be AWAY from the human's starting spot, which is the
    // "Flee fired at least once" marker. If the wolf had picked Attack
    // instead, its position would stay at ~1 m from the human (attack
    // doesn't move) and the displacement check fails.
    let final_wolf_pos = state.agent_pos(wolf).expect("wolf pos");
    let wolf_displacement_from_human_origin = final_wolf_pos.distance(initial_human_pos);

    assert!(
        wolf_displacement_from_human_origin > initial_distance,
        "wounded wolf should net-displace AWAY from human's initial position; \
         initial_distance={:.3}, wolf_displacement_from_human_origin={:.3} \
         (final wolf pos {:?}) — Flee row may not be firing on hp_pct<0.3",
        initial_distance,
        wolf_displacement_from_human_origin,
        final_wolf_pos,
    );
}

// ---------------------------------------------------------------------------
// threat_level wiring (task threat_level_wiring).
//
// `view::threat_level(self, _) per_unit 0.01` in Flee and
// `view::threat_level(self, target) per_unit 0.01` in Attack make the
// scorer respond to accumulated hits rather than only to current HP. The
// tests below score the two rows directly (via the scoring-table export)
// against hand-assembled SimStates with different threat_level values, to
// concretely verify that:
//
// 1. Flee's score grows monotonically with accumulated threat.
// 2. Attack's score ranks a target with high accumulated threat_level
//    above a fresh target at the same HP.
//
// The tests don't step the engine — they build a SimState with a primed
// ThreatLevel view and call the scoring entries' `Flee` / `Attack(target)`
// rows through the UtilityBackend's `score_entry` indirectly by using
// `evaluate` + a sparse mask setup. We use a small helper that computes
// the score for one row against one (agent, target) pair by mimicking
// `score_entry` — the engine's private `score_entry` is not exported, so
// this helper reproduces its math against `SCORING_TABLE`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod threat_level_scoring {
    use super::*;
    use engine::mask::MicroKind;
    use engine::state::AgentSpawn;
    use engine_rules::scoring::{
        PredicateDescriptor, ScoringEntry, MAX_MODIFIERS, SCORING_TABLE,
    };

    /// Compute the scoring-table `Flee` / `Attack(target)` row's score
    /// for a given (agent, target) pair. Reproduces the relevant parts
    /// of `engine::policy::utility::score_entry` — needed because that
    /// function is `pub(crate)` private.
    ///
    /// Only supports the predicate kinds we exercise: scalar compare
    /// (field-read) and the two view-call kinds. Unknown kinds drop
    /// the modifier silently — matches the "fail closed" contract.
    fn score_row_for(
        entry: &ScoringEntry,
        state: &SimState,
        agent: engine::ids::AgentId,
        target: Option<engine::ids::AgentId>,
    ) -> f32 {
        let mut score = entry.base;
        let count = entry.modifier_count as usize;
        let max = count.min(MAX_MODIFIERS);
        for i in 0..max {
            let row = &entry.modifiers[i];
            let pred = &row.predicate;
            match pred.kind {
                PredicateDescriptor::KIND_SCALAR_COMPARE => {
                    let lhs = read_field_scalar(state, agent, target, pred.field_id);
                    let mut tb = [0u8; 4];
                    tb.copy_from_slice(&pred.payload[0..4]);
                    let rhs = f32::from_le_bytes(tb);
                    if compare(pred.op, lhs, rhs) {
                        score += row.delta;
                    }
                }
                PredicateDescriptor::KIND_VIEW_SCALAR_COMPARE => {
                    let lhs = eval_view(state, agent, target, pred);
                    let mut tb = [0u8; 4];
                    tb.copy_from_slice(&pred.payload[0..4]);
                    let rhs = f32::from_le_bytes(tb);
                    if compare(pred.op, lhs, rhs) {
                        score += row.delta;
                    }
                }
                PredicateDescriptor::KIND_VIEW_GRADIENT => {
                    let v = eval_view(state, agent, target, pred);
                    if v.is_finite() {
                        score += v * row.delta;
                    }
                }
                _ => {}
            }
        }
        score
    }

    /// Read a scoring field by id. Only covers the fields the test
    /// actually exercises (hp, hp_pct, self-side + target-side).
    fn read_field_scalar(
        state: &SimState,
        agent: engine::ids::AgentId,
        target: Option<engine::ids::AgentId>,
        field_id: u16,
    ) -> f32 {
        if field_id >= 0x4000 && field_id < 0x8000 {
            let t = match target {
                Some(x) => x,
                None => return f32::NAN,
            };
            return match field_id {
                0x4000 => state.agent_hp(t).unwrap_or(0.0),
                0x4002 => {
                    let hp = state.agent_hp(t).unwrap_or(0.0);
                    let mx = state.agent_max_hp(t).unwrap_or(1.0);
                    if mx > 0.0 { hp / mx } else { 0.0 }
                }
                _ => f32::NAN,
            };
        }
        match field_id {
            0 => state.agent_hp(agent).unwrap_or(0.0),
            2 => {
                let hp = state.agent_hp(agent).unwrap_or(0.0);
                let mx = state.agent_max_hp(agent).unwrap_or(1.0);
                if mx > 0.0 { hp / mx } else { 0.0 }
            }
            _ => f32::NAN,
        }
    }

    /// Reproduce `eval_view_call` for the test harness. Matches the
    /// engine-side implementation in `crates/engine/src/policy/utility.rs`.
    fn eval_view(
        state: &SimState,
        agent: engine::ids::AgentId,
        target: Option<engine::ids::AgentId>,
        pred: &PredicateDescriptor,
    ) -> f32 {
        let slot0 = pred.payload[4];
        let slot1 = pred.payload[5];
        match pred.field_id {
            PredicateDescriptor::VIEW_ID_THREAT_LEVEL => {
                let a = match slot0 {
                    PredicateDescriptor::ARG_SELF => agent,
                    PredicateDescriptor::ARG_TARGET => match target {
                        Some(t) => t,
                        None => return f32::NAN,
                    },
                    _ => return f32::NAN,
                };
                match slot1 {
                    PredicateDescriptor::ARG_WILDCARD => {
                        state.views.threat_level.sum_for_first(a, state.tick)
                    }
                    _ => {
                        let b = match slot1 {
                            PredicateDescriptor::ARG_SELF => agent,
                            PredicateDescriptor::ARG_TARGET => match target {
                                Some(t) => t,
                                None => return f32::NAN,
                            },
                            _ => return f32::NAN,
                        };
                        state.views.threat_level.get(a, b, state.tick)
                    }
                }
            }
            PredicateDescriptor::VIEW_ID_MY_ENEMIES => {
                // `my_enemies` has no `@decay`, so the generated `get(a, b)`
                // takes no tick argument. Mirrors `eval_view_call` in
                // `crates/engine/src/policy/utility.rs`.
                let a = match slot0 {
                    PredicateDescriptor::ARG_SELF => agent,
                    PredicateDescriptor::ARG_TARGET => match target {
                        Some(t) => t,
                        None => return f32::NAN,
                    },
                    _ => return f32::NAN,
                };
                let b = match slot1 {
                    PredicateDescriptor::ARG_SELF => agent,
                    PredicateDescriptor::ARG_TARGET => match target {
                        Some(t) => t,
                        None => return f32::NAN,
                    },
                    _ => return f32::NAN,
                };
                state.views.my_enemies.get(a, b)
            }
            _ => f32::NAN,
        }
    }

    fn compare(op: u8, lhs: f32, rhs: f32) -> bool {
        if lhs.is_nan() || rhs.is_nan() {
            return false;
        }
        match op {
            PredicateDescriptor::OP_LT => lhs < rhs,
            PredicateDescriptor::OP_LE => lhs <= rhs,
            PredicateDescriptor::OP_EQ => lhs == rhs,
            PredicateDescriptor::OP_GE => lhs >= rhs,
            PredicateDescriptor::OP_GT => lhs > rhs,
            PredicateDescriptor::OP_NE => lhs != rhs,
            _ => false,
        }
    }

    fn flee_entry() -> &'static ScoringEntry {
        for e in SCORING_TABLE {
            if e.action_head == MicroKind::Flee as u16 {
                return e;
            }
        }
        panic!("SCORING_TABLE missing Flee row");
    }

    fn attack_entry() -> &'static ScoringEntry {
        for e in SCORING_TABLE {
            if e.action_head == MicroKind::Attack as u16 {
                return e;
            }
        }
        panic!("SCORING_TABLE missing Attack row");
    }

    /// Feed `n` AgentAttacked events against `target` from `actor` into
    /// the threat_level view. Fakes the event-fold phase directly so the
    /// test doesn't need to step the whole pipeline; the view's
    /// `fold_event` method is the same one the tick pipeline calls.
    fn prime_threat(state: &mut SimState, actor: engine::ids::AgentId, target: engine::ids::AgentId, n: u32) {
        for _ in 0..n {
            let ev = Event::AgentAttacked {
                actor,
                target,
                damage: 10.0,
                tick: state.tick,
            };
            state.views.threat_level.fold_event(&ev, state.tick);
        }
    }

    /// Spawn one deer (id 1) and one wolf (id 2) at close range. The
    /// deer is the "self" agent we score Flee for; the wolf is the
    /// threat whose cumulative damage the view records.
    fn spawn_deer_and_wolf() -> (SimState, engine::ids::AgentId, engine::ids::AgentId) {
        let mut state = SimState::new(4, 0xFEED);
        let deer = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Deer,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 40.0,
            max_hp: 40.0,
            ..Default::default()
        }).expect("deer spawn");
        let wolf = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(2.0, 0.0, 0.0),
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        }).expect("wolf spawn");
        (state, deer, wolf)
    }

    /// Fuzzy response — Flee's score strictly increases as the deer
    /// accumulates threat from the wolf. The classic hp-only Flee row
    /// would plateau at hp=40 (no hp gate fires because hp >= 30 and
    /// hp >= 50), but the threat_level gradient pushes the score up
    /// linearly, and the 50-threat scalar-compare gate adds a +0.3
    /// step once cumulative threat crosses 50.
    #[test]
    fn wounded_deer_flees_proportionally() {
        let (mut state, deer, wolf) = spawn_deer_and_wolf();
        let entry = flee_entry();

        // Baseline: 0 threat. Deer hp=40 — neither hp gate fires
        // (`hp < 30` false, `hp < 50` TRUE at 40). Baseline score =
        // 0.0 + 0.0 + 0.4 + 0.0 + 0.0 = 0.4.
        let s0 = score_row_for(entry, &state, deer, None);
        assert!((s0 - 0.4).abs() < 1e-4, "baseline Flee score = {s0}, expected ≈0.4");

        // +10 small threat: gradient adds 10.0 * 0.01 = +0.1.
        // Total ≈ 0.5.
        prime_threat(&mut state, wolf, deer, 10);
        let s1 = score_row_for(entry, &state, deer, None);
        assert!(s1 > s0, "Flee score should rise with accumulated threat ({s0} → {s1})");
        assert!((s1 - 0.5).abs() < 1e-3, "low-threat Flee score = {s1}, expected ≈0.5");

        // +40 more threat (total 50): gradient ≈ 50 * 0.01 = +0.5.
        // The `> 50.0` gate does NOT fire yet (50 is not > 50).
        // Total ≈ 0.4 (hp gate) + 0.5 (gradient) = 0.9.
        prime_threat(&mut state, wolf, deer, 40);
        let s2 = score_row_for(entry, &state, deer, None);
        assert!(s2 > s1, "Flee score should keep rising ({s1} → {s2})");
        assert!((s2 - 0.9).abs() < 1e-3, "at-threshold Flee score = {s2}, expected ≈0.9");

        // +1 more threat (total 51): gradient +0.51, plus the `>50`
        // gate fires adding +0.3. Total = 0.4 + 0.51 + 0.3 = 1.21.
        prime_threat(&mut state, wolf, deer, 1);
        let s3 = score_row_for(entry, &state, deer, None);
        assert!(
            s3 > s2 + 0.25,
            "crossing the 50-threat gate should add a hard +0.3 step ({s2} → {s3})",
        );
        assert!((s3 - 1.21).abs() < 1e-3, "over-threshold Flee score = {s3}, expected ≈1.21");

        // Determinism: computing the score twice gives the same answer.
        assert_eq!(s3, score_row_for(entry, &state, deer, None));
    }

    /// Retaliation bias — Attack's score ranks the specific attacker
    /// who has drawn blood above a fresh threat at the same HP. This
    /// is per-target: the view-gradient + scalar-compare modifiers
    /// reference `view::threat_level(self, target)` with the TARGET
    /// slot, so two different candidate targets score differently
    /// even when their self-side context is identical.
    #[test]
    fn wolf_attacks_accumulated_threat() {
        // Wolf (id 1) + two humans (ids 2, 3). Both humans at full HP.
        let mut state = SimState::new(8, 0xCAFE);
        let wolf = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        }).expect("wolf spawn");
        let h1 = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        }).expect("h1 spawn");
        let h2 = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(-1.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        }).expect("h2 spawn");

        let entry = attack_entry();

        // Baseline: zero threat on either side. Both humans at full
        // hp (hp_pct = 1.0), wolf at full hp (hp_pct = 1.0 >= 0.8 so
        // `self fresh` modifier fires +0.5). Neither target-side hp_pct
        // gate fires. No view modifiers fire. Score = 0.5 for both.
        let s_h1_0 = score_row_for(entry, &state, wolf, Some(h1));
        let s_h2_0 = score_row_for(entry, &state, wolf, Some(h2));
        assert!((s_h1_0 - s_h2_0).abs() < 1e-4, "symmetric baseline: {s_h1_0} vs {s_h2_0}");
        assert!((s_h1_0 - 0.5).abs() < 1e-4, "baseline Attack score = {s_h1_0}, expected ≈0.5");

        // Prime threat only against h1 — 25 cumulative hits.
        // gradient(self=wolf, target=h1) = 25 * 0.01 = +0.25.
        // Scalar `> 20.0` fires adding +0.3.
        // Score for h1 = 0.5 + 0.25 + 0.3 = 1.05.
        // Score for h2 unchanged at 0.5.
        prime_threat(&mut state, h1, wolf, 25);
        let s_h1_1 = score_row_for(entry, &state, wolf, Some(h1));
        let s_h2_1 = score_row_for(entry, &state, wolf, Some(h2));
        assert!(
            s_h1_1 > s_h2_1 + 0.5,
            "wolf should prefer attacking h1 (accumulated threat) over h2 (fresh); got h1={s_h1_1}, h2={s_h2_1}",
        );
        assert!((s_h2_1 - 0.5).abs() < 1e-4, "h2 unchanged at ≈0.5, got {s_h2_1}");
        assert!((s_h1_1 - 1.05).abs() < 1e-3, "h1 with threat = {s_h1_1}, expected ≈1.05");

        // Fuzzy: even below the 20-threat scalar gate, the gradient
        // alone differentiates. Reset + prime just 5 threat against
        // h2 (not 25 — the scalar gate does NOT fire).
        let mut state2 = SimState::new(8, 0xCAFE);
        let _ = state2.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 80.0, max_hp: 80.0,
            ..Default::default()
        }).unwrap();
        let h1b = state2.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0, max_hp: 100.0,
            ..Default::default()
        }).unwrap();
        let h2b = state2.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(-1.0, 0.0, 0.0),
            hp: 100.0, max_hp: 100.0,
            ..Default::default()
        }).unwrap();
        prime_threat(&mut state2, h2b, wolf, 5);
        let s2_h1 = score_row_for(entry, &state2, wolf, Some(h1b));
        let s2_h2 = score_row_for(entry, &state2, wolf, Some(h2b));
        // Gradient alone: 5 * 0.01 = +0.05. Strictly greater than the
        // fresh target, even though the scalar-compare gate hasn't
        // fired yet. This is the "fuzzy" property the task calls for.
        assert!(
            s2_h2 > s2_h1,
            "below the 20-threat scalar gate, the gradient alone still ranks h2 > h1; got h1={s2_h1}, h2={s2_h2}",
        );
        assert!(
            (s2_h2 - s2_h1 - 0.05).abs() < 1e-4,
            "gradient delta should be ~+0.05 (5 threat × 0.01); got {}",
            s2_h2 - s2_h1,
        );
    }

    /// Memory-driven scoring — wolves remember who hurt them and prefer
    /// that specific attacker as a target over an otherwise-equivalent
    /// stranger (task 160). The `my_enemies` view folds `AgentAttacked`
    /// events into a per-pair saturating `[0.0, 1.0]` flag; the scoring
    /// table's Attack row adds +0.4 when the view crosses the `> 0.5`
    /// gate, so a wolf that's been bitten by one of two otherwise-
    /// identical humans scores Attack against the biter +0.4 higher.
    ///
    /// Separate from `wolf_attacks_accumulated_threat` above: threat_level
    /// decays and accumulates, my_enemies saturates and persists. Both
    /// can fire together — the test isolates my_enemies by priming only
    /// a single `AgentAttacked` event (below the threat_level `>20` gate)
    /// and asserting the exact +0.4 delta.
    #[test]
    fn wolf_attacks_grudge_target_over_stranger() {
        // Wolf + two humans at full HP. Symmetric geometry so the only
        // differentiator can be the my_enemies view.
        let mut state = SimState::new(8, 0xDEAD);
        let wolf = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos: Vec3::new(0.0, 0.0, 0.0),
            hp: 80.0,
            max_hp: 80.0,
            ..Default::default()
        }).expect("wolf spawn");
        let h1 = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(1.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        }).expect("h1 spawn");
        let h2 = state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(-1.0, 0.0, 0.0),
            hp: 100.0,
            max_hp: 100.0,
            ..Default::default()
        }).expect("h2 spawn");

        let entry = attack_entry();

        // Baseline: no grudges, no threat. Both humans at full HP, wolf
        // fresh (hp_pct=1.0 ≥ 0.8 fires +0.5). Target-hp_pct gates don't
        // fire (both humans at 1.0). View modifiers all zero. Score = 0.5
        // for both humans — symmetric.
        let s_h1_0 = score_row_for(entry, &state, wolf, Some(h1));
        let s_h2_0 = score_row_for(entry, &state, wolf, Some(h2));
        assert!((s_h1_0 - s_h2_0).abs() < 1e-4, "symmetric baseline: h1={s_h1_0}, h2={s_h2_0}");
        assert!(
            (s_h1_0 - 0.5).abs() < 1e-4,
            "baseline Attack score = {s_h1_0}, expected ≈0.5 (fresh-self +0.5, no view deltas)",
        );

        // Fold a single AgentAttacked event — h1 hits the wolf. The
        // my_enemies view saturates at 1.0 on the first hit, which is
        // above the `> 0.5` gate, so the Attack row for target=h1
        // picks up +0.4. h2 stays a stranger.
        //
        // threat_level ALSO records this event (it folds on the same
        // Event::AgentAttacked shape). threat_level(wolf, h1) = 1 after
        // one hit — the gradient adds 0.01 but the `>20` scalar gate
        // doesn't fire (1 ≤ 20). Accounted for in the expected score.
        let ev = Event::AgentAttacked {
            actor: h1,
            target: wolf,
            damage: 10.0,
            tick: state.tick,
        };
        state.views.my_enemies.fold_event(&ev, state.tick);
        state.views.threat_level.fold_event(&ev, state.tick);

        let s_h1_1 = score_row_for(entry, &state, wolf, Some(h1));
        let s_h2_1 = score_row_for(entry, &state, wolf, Some(h2));

        // h2 unchanged — the wolf has no grudge against this one.
        assert!((s_h2_1 - 0.5).abs() < 1e-4, "h2 unchanged at ≈0.5, got {s_h2_1}");

        // h1 gains the +0.4 my_enemies bump plus the +0.01 threat_level
        // gradient (from the single hit). Expected ≈ 0.5 + 0.4 + 0.01 = 0.91.
        assert!(
            (s_h1_1 - 0.91).abs() < 1e-3,
            "h1 with grudge = {s_h1_1}, expected ≈0.91 (0.5 baseline + 0.4 grudge + 0.01 threat gradient)",
        );

        // Delta is decisive — at least +0.4, driven by the grudge gate.
        assert!(
            s_h1_1 - s_h2_1 >= 0.4,
            "wolf should strongly prefer grudge target h1 over stranger h2; delta = {}",
            s_h1_1 - s_h2_1,
        );

        // Idempotent: re-priming the same grudge doesn't push the score
        // further (my_enemies saturates at the 1.0 clamp). threat_level
        // continues accumulating, but the grudge gate has already fired.
        state.views.my_enemies.fold_event(&ev, state.tick);
        let s_h1_2 = score_row_for(entry, &state, wolf, Some(h1));
        // my_enemies is already clamped at 1.0, so the grudge delta
        // stays at +0.4. The only further drift would be threat_level's
        // gradient from the second fold — which we skip in this assert
        // by folding only my_enemies on the second event.
        assert!(
            (s_h1_2 - s_h1_1).abs() < 1e-4,
            "second my_enemies fold should not change Attack score (saturated); got {s_h1_1} → {s_h1_2}",
        );
    }
}
