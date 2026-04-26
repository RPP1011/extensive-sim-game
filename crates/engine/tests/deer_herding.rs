#![allow(unused_mut, unused_variables, unused_imports, dead_code)]
//! Deer herding — task 177. A fleeing deer should bias its direction
//! toward same-species kin so the herd clusters rather than scatters.
//!
//! These tests drive the capability-gated flee primitive
//! `spatial::flee_direction_with_kin_bias` and its step.rs call site.
//! The capability is declared on the `Deer` entity in
//! `assets/sim/entities.sim` (`herds_when_fleeing: true`); every other
//! species leaves it `false` and takes the pre-177 pure-away flee
//! branch, so the wolves+humans parity baseline stays byte-exact
//! (guarded by `wolves_and_humans_parity.rs`).
//!
//! Test shape mirrors `fear_spread_rout.rs` / `pack_focus.rs`:
//! a per-scenario spawn helper that runs `step_full` for N ticks under
//! the stock `UtilityBackend` + builtin cascades, then asserts on
//! final agent positions. No custom scorer — we want to observe the
//! scoring + step pipeline end-to-end, the same surface the DSL
//! targets.

use engine_rules::views::ViewRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::ids::AgentId;
use engine::invariant::InvariantRegistry;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch}; // Plan B1' Task 11: step_full is unimplemented!() stub
use engine::telemetry::NullSink;
use engine::view::MaterializedView;
use glam::Vec3;

/// Spawn a scenario + run `ticks` ticks under the stock pipeline.
/// Returns the final state and the slice of spawned agent ids in spawn
/// order. Parallels `run_behavioural_scenario` in
/// `wolves_and_humans_parity.rs` but trimmed — these tests only need
/// final positions, not the event log.
fn run_scenario(
    seed: u64,
    spawns: &[(CreatureType, Vec3, f32)],
    ticks: u32,
) -> (SimState, Vec<AgentId>, Vec<Event>) {
    let cap = (spawns.len() as u32 + 4).max(4);
    let mut state = SimState::new(cap, seed);
    let mut ids = Vec::with_capacity(spawns.len());
    for &(ct, pos, hp) in spawns {
        let id = state
            .spawn_agent(AgentSpawn {
                creature_type: ct,
                pos,
                hp,
                max_hp: hp,
                ..Default::default()
            })
            .expect("spawn inside agent cap");
        ids.push(id);
    }

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1 << 14);
    let cascade = engine_rules::with_engine_builtins();
    let invariants = InvariantRegistry::<Event>::new();
    let mut views: Vec<&mut dyn MaterializedView<Event>> = Vec::new();
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
    (state, ids, log)
}

/// Mean pairwise (Euclidean, 3D) distance across a set of agents.
/// Returns 0.0 for fewer than 2 live agents (degenerate — nothing to
/// measure). Uses `state.agent_pos` (skipping dead slots) so a deer
/// that died mid-run doesn't contaminate the centroid with a stale
/// position.
fn mean_pairwise_distance(state: &SimState, ids: &[AgentId]) -> f32 {
    let live: Vec<Vec3> = ids
        .iter()
        .filter_map(|&id| {
            if state.agent_alive(id) {
                state.agent_pos(id)
            } else {
                None
            }
        })
        .collect();
    if live.len() < 2 {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for i in 0..live.len() {
        for j in (i + 1)..live.len() {
            sum += live[i].distance(live[j]);
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / (n as f32) }
}

// ---------------------------------------------------------------------------
// Test (1) — the core behavioural claim.
// ---------------------------------------------------------------------------
//
// Setup: 2 deer side-by-side (perpendicular to the wolf). Wolf at
// (0, 0, -15). Deer at (-3, 0, 0) and (3, 0, 0). Each deer's "away
// from wolf" vector points predominantly +z (away from the wolf in
// -z direction) with a small ∓x tilt due to the deer's x-offset.
// Each deer's "toward kin centroid" vector points strongly toward
// the other deer: deer[0] at (-3,0,0) points toward +x, deer[1] at
// (+3,0,0) points toward -x. The kin bias tilts both deer inward
// along x while they flee northward in z. Final pairwise distance
// in x shrinks → clustering.
//
// Why not just put the wolf directly between them on the same axis?
// That configuration is the "threat between deer" case tested
// separately below. The "wolf to one side" case is the canonical
// herding scenario: prey flee AWAY from the predator (not toward
// each other through it) while converging laterally.
//
// Collinear layouts (all deer on one ray from the wolf) do NOT
// demonstrate clustering: every deer's away-vector is parallel and
// the kin-centroid tilt cancels out (outer deer's toward-kin is
// collinear with away; inner deer's toward-kin is zero). See
// design-memo §6 for the full degenerate-geometry catalogue.
//
// Deer at hp=40 so the `self.hp < 50` Flee gate fires (scoring.sim
// line 112, +0.4 when hp < 50). Wolf aggro_range is 50 so it's seen
// as a threat.
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn deer_cluster_when_fleeing_wolf() {
    let spawns = [
        (CreatureType::Wolf, Vec3::new(0.0, 0.0, -15.0), 80.0),
        (CreatureType::Deer, Vec3::new(-3.0, 0.0, 0.0), 40.0),
        (CreatureType::Deer, Vec3::new(3.0, 0.0, 0.0), 40.0),
    ];
    let initial_positions: Vec<Vec3> = spawns.iter().map(|s| s.1).collect();
    let (state, ids, log) = run_scenario(0xDEE2_0177, &spawns, 20);
    let (wolf, deer_ids) = (ids[0], &ids[1..]);

    // Clustering invariant: final pairwise distance across the deer
    // should be strictly smaller than initial. Pre-177 (pure-away
    // flee) the three deer all move with the same velocity along +x
    // so their pairwise distances stay *exactly* at initial — any
    // cluster-vs-initial negative delta is new behaviour.
    let mut initial_state = SimState::new(4, 0xDEAD);
    for &p in &initial_positions[1..] {
        initial_state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Deer,
                pos: p,
                hp: 40.0,
                max_hp: 40.0,
                ..Default::default()
            })
            .unwrap();
    }
    let initial_pairwise =
        mean_pairwise_distance(&initial_state, &initial_state.agents_alive().collect::<Vec<_>>());
    let final_pairwise = mean_pairwise_distance(&state, deer_ids);

    // Silence unused-log warning — AgentFled is driving the motion, but
    // the assertion reads final positions directly. Counting events is
    // useful for debugging via `cargo test -- --nocapture`.
    let flee_count = log
        .iter()
        .filter(|e| matches!(e, Event::AgentFled { .. }))
        .count();
    assert!(
        flee_count > 0,
        "no AgentFled events emitted — deer may have Attack or MoveToward \
         scoring above Flee. Check Flee-row scoring gates and Flee mask.",
    );

    assert!(
        final_pairwise < initial_pairwise,
        "deer should cluster while fleeing: initial mean pairwise = {:.3}, \
         final mean pairwise = {:.3} (expected final < initial). \
         If equal, kin-flee bias is not firing — check the `herds_when_fleeing` \
         capability on the Deer entity and the Flee arm in step.rs.",
        initial_pairwise,
        final_pairwise,
    );

    // Anti-regression: even while clustering, no deer should run *into*
    // the wolf. Guard: each deer's final distance from the wolf's
    // initial position exceeds its initial distance from the wolf's
    // initial position. Matches `deer_flee_from_wolves` in
    // `wolves_and_humans_parity.rs`.
    let wolf_initial_pos = initial_positions[0];
    for (i, &deer_id) in deer_ids.iter().enumerate() {
        let initial_dist = initial_positions[i + 1].distance(wolf_initial_pos);
        // If the deer died mid-run, skip — the wolves+humans baseline
        // test already pins that death is the legitimate failure mode.
        let final_pos = match state.agent_pos(deer_id) {
            Some(p) => p,
            None => continue,
        };
        let final_dist = final_pos.distance(wolf_initial_pos);
        assert!(
            final_dist >= initial_dist,
            "deer[{}] ran toward the wolf while herding: initial_dist={:.3}, \
             final_dist={:.3} (kin bias overwhelmed threat term; check \
             combat.kin_flee_bias in config.sim).",
            i, initial_dist, final_dist,
        );
    }

    // Sanity: the wolf id is exercised via spawn; silencing unused-var
    // would hide a future regression where `ids[0]` drifts.
    let _ = wolf;

    // Print the measurement so the parent task can eyeball the
    // clustering delta in CI output (--nocapture).
    println!(
        "deer_cluster_when_fleeing_wolf: initial_pairwise={:.3} final_pairwise={:.3} \
         delta={:.3}",
        initial_pairwise,
        final_pairwise,
        final_pairwise - initial_pairwise,
    );
}

// ---------------------------------------------------------------------------
// Test (2a) — task 178: wolves DO herd now (pack cohesion).
// ---------------------------------------------------------------------------
//
// Mirror of `deer_cluster_when_fleeing_wolf` but with wolves as the
// fleeing party and a human as the threat. Two wolves at (-3, 0, 0)
// and (3, 0, 0), human threat at (0, 0, -15). Wolves are lightly
// wounded (hp=40 / max_hp=80 → hp_pct=0.5) so the `self.hp < 50`
// Flee gate fires (+0.4 on the Flee row). A human at hp=100 is a
// fresh target, but Flee with the wounded gate beats Attack on a
// fresh target for the wounded wolf.
//
// Task 178 flipped `herds_when_fleeing` from false → true on the
// Wolf entity. With that flag on, the Flee arm blends the pure-
// away vector with the toward-kin-centroid vector, producing the
// same clustering behaviour deer get. Assertion: final pairwise
// distance shrinks relative to initial.
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn wolves_cluster_when_fleeing() {
    let spawns = [
        (CreatureType::Human, Vec3::new(0.0, 0.0, -15.0), 100.0),
        (CreatureType::Wolf, Vec3::new(-3.0, 0.0, 0.0), 40.0),
        (CreatureType::Wolf, Vec3::new(3.0, 0.0, 0.0), 40.0),
    ];
    let initial_positions: Vec<Vec3> = spawns.iter().map(|s| s.1).collect();
    let (state, ids, log) = run_scenario(0xDEE2_0178_BEEF, &spawns, 20);
    let (_human, wolf_ids) = (ids[0], &ids[1..]);

    // Clustering invariant: final pairwise distance across the wolves
    // should be strictly smaller than initial. Pre-178 wolves
    // (herds_when_fleeing: false) would move in parallel away-
    // vectors and keep their pairwise distance roughly constant; now
    // the kin-bias blend tilts them inward.
    let mut initial_state = SimState::new(4, 0xDEAD);
    for &p in &initial_positions[1..] {
        initial_state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Wolf,
                pos: p,
                hp: 40.0,
                max_hp: 80.0,
                ..Default::default()
            })
            .unwrap();
    }
    let initial_pairwise =
        mean_pairwise_distance(&initial_state, &initial_state.agents_alive().collect::<Vec<_>>());
    let final_pairwise = mean_pairwise_distance(&state, wolf_ids);

    let flee_count = log
        .iter()
        .filter(|e| matches!(e, Event::AgentFled { .. }))
        .count();
    assert!(
        flee_count > 0,
        "no AgentFled events emitted — wolves may have Attack or MoveToward \
         scoring above Flee. Check Flee-row scoring gates and Flee mask for \
         hp=40/max_hp=80.",
    );

    assert!(
        final_pairwise < initial_pairwise,
        "wolves should cluster while fleeing (task 178): initial mean pairwise = {:.3}, \
         final mean pairwise = {:.3} (expected final < initial). If equal, \
         kin-flee bias isn't firing — check the `herds_when_fleeing` capability \
         on the Wolf entity in assets/sim/entities.sim.",
        initial_pairwise,
        final_pairwise,
    );

    println!(
        "wolves_cluster_when_fleeing: initial_pairwise={:.3} final_pairwise={:.3} \
         delta={:.3}",
        initial_pairwise,
        final_pairwise,
        final_pairwise - initial_pairwise,
    );
}

// ---------------------------------------------------------------------------
// Test (2b) — capability gate isolates species (humans still don't herd).
// ---------------------------------------------------------------------------
//
// Task 178 flipped `herds_when_fleeing` on for wolves but left humans
// alone. Humans (hp=16 / max_hp=100 → hp_pct=0.16 fires the `hp_pct <
// 0.3` Flee gate) fleeing a dragon should NOT cluster — they take
// the pre-177 pure-away branch, same as pre-178.
//
// The assertion is one-sided: "humans do not cluster more than a
// small tolerance". Exact pairwise-distance invariance would require
// perfectly matched velocity vectors, which can drift slightly in
// corner cases (opportunity attacks, etc.).
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn humans_dont_herd() {
    let spawns = [
        (CreatureType::Dragon, Vec3::new(0.0, 0.0, 0.0), 500.0),
        (CreatureType::Human, Vec3::new(6.0, 0.0, 0.0), 16.0),
        (CreatureType::Human, Vec3::new(0.0, 0.0, 6.0), 16.0),
        (CreatureType::Human, Vec3::new(-6.0, 0.0, 0.0), 16.0),
    ];
    let initial_positions: Vec<Vec3> = spawns.iter().map(|s| s.1).collect();
    let (state, ids, _log) = run_scenario(0xDEE2_0178, &spawns, 10);
    let human_ids = &ids[1..];

    // Build a temp state with just the humans at initial positions to
    // compute the initial pairwise baseline. (Can't use `state` pre-
    // simulation since we only have the post-sim one here.)
    let mut initial_state = SimState::new(8, 0xBEEF);
    for &p in &initial_positions[1..] {
        initial_state
            .spawn_agent(AgentSpawn {
                creature_type: CreatureType::Human,
                pos: p,
                hp: 16.0,
                max_hp: 100.0,
                ..Default::default()
            })
            .unwrap();
    }
    let initial_pairwise =
        mean_pairwise_distance(&initial_state, &initial_state.agents_alive().collect::<Vec<_>>());
    let final_pairwise = mean_pairwise_distance(&state, human_ids);

    // The humans should NOT cluster. A small epsilon tolerates the
    // tiny numerical drift from non-parallel away-vectors (6-m
    // triangle geometry + opportunity-attack micro-jitter). The
    // absolute claim is "final_pairwise should not be materially
    // smaller than initial_pairwise" — 0.5 m is well below the
    // clustering-range the Deer test produces (several meters) and
    // well above the noise floor.
    assert!(
        final_pairwise > initial_pairwise - 0.5,
        "humans should not herd (capability gate broken?): \
         initial_pairwise={:.3}, final_pairwise={:.3}. \
         Check that Human entity has `herds_when_fleeing: false` and \
         step.rs Flee arm dispatches on the capability, not CreatureType.",
        initial_pairwise,
        final_pairwise,
    );

    println!(
        "humans_dont_herd: initial_pairwise={:.3} final_pairwise={:.3}",
        initial_pairwise, final_pairwise,
    );
}

// ---------------------------------------------------------------------------
// Test (3) — single deer, no kin → pure straight-line flee.
// ---------------------------------------------------------------------------
//
// Regression guard for the degenerate-case path in
// `flee_direction_with_kin_bias`: empty `nearby_kin` list → return
// pure `away`. Equivalent to the existing `deer_flee_from_wolves`
// test (1 deer + 1 wolf) but pins the exact post-tick position —
// since kin contributes nothing, the deer at (5, 0, 0) fleeing a
// wolf at (0, 0, 0) for 1 tick must advance by exactly
// `move_speed_mps = 1.0` along +x. If the kin-bias primitive
// accidentally normalizes a zero vector into something non-zero (a
// classic `.normalize()` NaN / (1,0,0) mistake), this test catches
// it.
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn single_deer_flees_straight() {
    let spawns = [
        (CreatureType::Wolf, Vec3::new(0.0, 0.0, 0.0), 80.0),
        (CreatureType::Deer, Vec3::new(5.0, 0.0, 0.0), 40.0),
    ];
    let initial_deer_pos = spawns[1].1;
    let (state, ids, _log) = run_scenario(0xDEE2_0179, &spawns, 1);
    let deer = ids[1];
    let final_deer_pos = state.agent_pos(deer).expect("deer alive after 1 tick");

    // Exactly 1 move-speed along +x. Tolerance 1e-5 pins the value
    // past f32 rounding but below any real drift.
    let dx = final_deer_pos.x - initial_deer_pos.x;
    assert!(
        (dx - 1.0).abs() < 1e-5,
        "single deer should flee along pure +x at move_speed=1.0; got dx={:.6} \
         (no-kin path in flee_direction_with_kin_bias broken?)",
        dx,
    );
    assert!(
        (final_deer_pos.y - initial_deer_pos.y).abs() < 1e-6,
        "y drift with no kin: {:.6}",
        final_deer_pos.y,
    );
    assert!(
        (final_deer_pos.z - initial_deer_pos.z).abs() < 1e-6,
        "z drift with no kin: {:.6}",
        final_deer_pos.z,
    );
}

// ---------------------------------------------------------------------------
// Test (4) — threat between two deer; both flee AWAY, not toward each other.
// ---------------------------------------------------------------------------
//
// Deer at (-5, 0, 0) and (5, 0, 0) flanking a wolf at (0, 0, 0). If
// the kin-bias primitive did not cap kin attraction vs. threat
// avoidance, each deer would flee *toward* the other (toward kin)
// and straight into the wolf. The normalized blend with kin_weight
// = 0.5 caps the kin tilt at ~27° from pure-away; in this geometry
// the kin centroid is exactly on the opposite side of the wolf from
// the fleeing deer, so the blend can shift the direction but the
// "away from wolf" component dominates. Both deer should end up
// farther from the wolf, not closer.
    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn deer_herding_with_threat_in_middle() {
    let spawns = [
        (CreatureType::Wolf, Vec3::new(0.0, 0.0, 0.0), 80.0),
        (CreatureType::Deer, Vec3::new(-5.0, 0.0, 0.0), 40.0),
        (CreatureType::Deer, Vec3::new(5.0, 0.0, 0.0), 40.0),
    ];
    let initial_positions: Vec<Vec3> = spawns.iter().map(|s| s.1).collect();
    let (state, ids, _log) = run_scenario(0xDEE2_017A, &spawns, 5);
    let wolf_initial_pos = initial_positions[0];

    for (i, &deer_id) in ids[1..].iter().enumerate() {
        let initial_dist = initial_positions[i + 1].distance(wolf_initial_pos);
        let final_pos = match state.agent_pos(deer_id) {
            Some(p) => p,
            None => continue, // deer died; baseline test guards that path
        };
        let final_dist = final_pos.distance(wolf_initial_pos);
        assert!(
            final_dist > initial_dist,
            "deer[{}] with a wolf between it and its kin ended up closer to \
             the wolf: initial={:.3}, final={:.3}. Kin bias overwhelmed \
             threat term — lower combat.kin_flee_bias or revisit the \
             normalize-after-blend step in spatial::flee_direction_with_kin_bias.",
            i, initial_dist, final_dist,
        );
    }
}
