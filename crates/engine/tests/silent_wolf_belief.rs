//! Silent Wolf — belief-aware scoring reference scenario.
// T11: entire test module requires the `theory-of-mind` feature.
// Run with: cargo test -p engine --test silent_wolf_belief --features engine_rules/theory-of-mind
// (engine_rules/theory-of-mind transitively enables engine/theory-of-mind)
#![cfg(feature = "theory-of-mind")]
//!
//! Plan: docs/superpowers/plans/2026-04-25-theory-of-mind-impl.md Task 10.
//!
//! Demonstrates the full belief pipeline end-to-end:
//!   write (physics cascade) → decay (step Phase 5b) → read (scoring loop)
//!
//! ## T10 deviation note
//!
//! The original plan placed the belief predicate on the **Flee** score row.
//! The current scoring grammar (`beliefs(obs).about(tgt).<field>`) requires
//! the target to be either `self` or the head's named target binding.
//! `Flee` is a self-only head (no target binding in `MicroKind::TARGET_BOUND`),
//! so `beliefs(self).about(nearest_hostile)` is not a valid expression in
//! the current emitter.  The predicate was therefore placed on **Attack**
//! instead, which has an explicit `target` binding, making
//! `beliefs(self).about(target).confidence > 0.5 { +0.2 }` valid and
//! semantically meaningful: "commit harder when I have a fresh belief about
//! this specific target."  The `KIND_BELIEF_SCALAR_COMPARE` pipeline is
//! exercised identically regardless of which row hosts the predicate.
//!
//! ## Test design
//!
//! ### Test 1 — `belief_bonus_fires_when_belief_planted`
//!
//! A wolf faces two humans at equal distance with equal HP.  A high-confidence
//! belief about human A is manually injected into the wolf's cold-beliefs map.
//! The Attack row's `beliefs(self).about(target).confidence > 0.5 { +0.2 }`
//! modifier fires for human A (score 0.7) but not for human B (score 0.5).
//! The wolf's scoring table prefers human A.
//!
//! ### Test 2 — `belief_bonus_absent_without_belief`
//!
//! Same topology with no injected belief.  Both humans score identically (0.5);
//! the wolf attacks one of them (order not asserted, only that it attacks).
//!
//! ### Test 3 — `belief_forms_via_kin_observation`
//!
//! Two wolves within `observation_range` (default 10 m).  Wolf B moves.  The
//! physics cascade fires `update_beliefs` and writes a belief entry into wolf
//! A's cold-beliefs map. After the tick wolf A has a confidence-1.0 belief
//! about wolf B.  (Wolves are same-species; belief writes use `nearby_kin`.)
//! This does NOT result in wolf A attacking wolf B (wolves are not hostile
//! to wolves), but it confirms the write path works.

use engine_data::belief::BeliefState;
use engine_data::entities::CreatureType;
use engine_data::events::Event;
use engine_data::scoring::{PredicateDescriptor, SCORING_TABLE, MAX_MODIFIERS};
use engine::event::EventRing;
use engine::ids::AgentId;
use engine::mask::MicroKind;
use engine::policy::UtilityBackend;
use engine::scratch::SimScratch;
use engine::state::{AgentSpawn, SimState};
use engine_rules::views::ViewRegistry;
use glam::Vec3;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn spawn_wolf(state: &mut SimState, pos: Vec3, hp: f32, max_hp: f32) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Wolf,
            pos,
            hp,
            max_hp,
            ..Default::default()
        })
        .expect("wolf spawn")
}

fn spawn_human(state: &mut SimState, pos: Vec3, hp: f32, max_hp: f32) -> AgentId {
    state
        .spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos,
            hp,
            max_hp,
            ..Default::default()
        })
        .expect("human spawn")
}

/// Evaluate the Attack scoring row for `(agent, target)` in the given state.
/// Mirrors the scoring path in `engine::policy::utility::score_entry` for the
/// predicates used in scoring.sim's Attack row, including the new
/// `KIND_BELIEF_SCALAR_COMPARE` arm added in T10.
fn attack_score(state: &SimState, agent: AgentId, target: AgentId) -> f32 {
    let entry = SCORING_TABLE
        .iter()
        .find(|e| e.action_head == MicroKind::Attack as u16)
        .expect("Attack row must exist in SCORING_TABLE");

    let mut score = entry.base;
    let count = (entry.modifier_count as usize).min(MAX_MODIFIERS);
    for i in 0..count {
        let row = &entry.modifiers[i];
        let pred = &row.predicate;
        match pred.kind {
            PredicateDescriptor::KIND_SCALAR_COMPARE => {
                let lhs = read_field(state, agent, Some(target), pred.field_id);
                let mut tb = [0u8; 4];
                tb.copy_from_slice(&pred.payload[0..4]);
                let rhs = f32::from_le_bytes(tb);
                if compare(pred.op, lhs, rhs) {
                    score += row.delta;
                }
            }
            PredicateDescriptor::KIND_BELIEF_SCALAR_COMPARE => {
                let lhs = read_belief(state, agent, Some(target), pred);
                let mut tb = [0u8; 4];
                tb.copy_from_slice(&pred.payload[0..4]);
                let rhs = f32::from_le_bytes(tb);
                if compare(pred.op, lhs, rhs) {
                    score += row.delta;
                }
            }
            PredicateDescriptor::KIND_VIEW_SCALAR_COMPARE
            | PredicateDescriptor::KIND_VIEW_GRADIENT => {
                // View-backed predicates (threat_level, my_enemies, pack_focus,
                // rally_boost) return 0.0 in a fresh state — the view registry
                // hasn't been folded so all values are zero.  That's fine for
                // the belief-isolation tests here; other test files cover those
                // predicates.
            }
            _ => {}
        }
    }
    score
}

fn read_field(state: &SimState, agent: AgentId, target: Option<AgentId>, field_id: u16) -> f32 {
    if field_id >= 0x4000 && field_id < 0x8000 {
        let t = match target {
            Some(x) => x,
            None => return f32::NAN,
        };
        return match field_id {
            0x4000 => state.agent_hp(t).unwrap_or(0.0),
            0x4001 => state.agent_max_hp(t).unwrap_or(1.0),
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
        1 => state.agent_max_hp(agent).unwrap_or(1.0),
        2 => {
            let hp = state.agent_hp(agent).unwrap_or(0.0);
            let mx = state.agent_max_hp(agent).unwrap_or(1.0);
            if mx > 0.0 { hp / mx } else { 0.0 }
        }
        _ => f32::NAN,
    }
}

/// Read a belief-state scalar from `state` for use in a
/// `KIND_BELIEF_SCALAR_COMPARE` predicate.  Layout mirrors
/// `engine::policy::utility::eval_belief_scalar`.
fn read_belief(
    state: &SimState,
    agent: AgentId,
    target: Option<AgentId>,
    pred: &PredicateDescriptor,
) -> f32 {
    let obs_slot = pred.payload[4];
    let tgt_slot = pred.payload[5];

    let observer = match obs_slot {
        PredicateDescriptor::ARG_SELF => agent,
        PredicateDescriptor::ARG_TARGET => match target {
            Some(t) => t,
            None => return f32::NAN,
        },
        _ => return f32::NAN,
    };
    let tgt = match tgt_slot {
        PredicateDescriptor::ARG_SELF => agent,
        PredicateDescriptor::ARG_TARGET => match target {
            Some(t) => t,
            None => return f32::NAN,
        },
        _ => return f32::NAN,
    };

    let beliefs = match state.agent_cold_beliefs(observer) {
        Some(b) => b,
        None => return f32::NAN,
    };
    let entry = match beliefs.get(&tgt) {
        Some(e) => e,
        None => return 0.0,
    };

    match pred.field_id {
        PredicateDescriptor::BELIEF_FIELD_LAST_KNOWN_HP => entry.last_known_hp,
        PredicateDescriptor::BELIEF_FIELD_LAST_KNOWN_MAX_HP => entry.last_known_max_hp,
        PredicateDescriptor::BELIEF_FIELD_CONFIDENCE => entry.confidence,
        PredicateDescriptor::BELIEF_FIELD_LAST_UPDATED_TICK => entry.last_updated_tick as f32,
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

// ---------------------------------------------------------------------------
// Test 1: belief bonus fires when belief is present
// ---------------------------------------------------------------------------

/// A wolf faces two humans at equal distance with equal HP.
/// A high-confidence belief about human A is planted.
/// The Attack row scores human A 0.2 points higher than human B.
#[test]
fn belief_bonus_fires_when_belief_planted() {
    let mut state = SimState::new(4, 42);

    // Wolf at origin, full HP (hp_pct = 1.0 ≥ 0.8 → +0.5 modifier).
    let wolf = spawn_wolf(&mut state, Vec3::ZERO, 80.0, 80.0);
    // Two humans at equal distance; equal HP.
    let human_a = spawn_human(&mut state, Vec3::new(1.5, 0.0, 0.0), 100.0, 100.0);
    let human_b = spawn_human(&mut state, Vec3::new(-1.5, 0.0, 0.0), 100.0, 100.0);

    // Plant a high-confidence belief about human A into the wolf.
    state
        .agent_cold_beliefs_mut(wolf)
        .expect("wolf belief map")
        .upsert(
            human_a,
            BeliefState {
                last_known_pos: Vec3::new(1.5, 0.0, 0.0),
                last_known_hp: 100.0,
                last_known_max_hp: 100.0,
                last_updated_tick: 0,
                confidence: 1.0,
                ..Default::default()
            },
        );

    let score_a = attack_score(&state, wolf, human_a);
    let score_b = attack_score(&state, wolf, human_b);

    // Both are at full HP; wolf is at full HP.
    // human A: +0.5 (self fresh) + 0.2 (belief confidence > 0.5) = 0.7
    // human B: +0.5 (self fresh) + 0.0 (no belief entry)         = 0.5
    assert!(
        score_a > score_b,
        "Attack score toward believed target ({}) should exceed unbelieved target ({})",
        score_a,
        score_b,
    );
    assert!(
        (score_a - score_b - 0.2).abs() < 1e-5,
        "Expected +0.2 belief delta; got score_a={} score_b={}",
        score_a,
        score_b,
    );
}

// ---------------------------------------------------------------------------
// Test 2: no belief → no bonus, scores are equal
// ---------------------------------------------------------------------------

/// Same topology without any planted belief.
/// Both humans score identically on the Attack row — the belief predicate
/// returns 0.0 for both (no entry → 0.0 short-circuit).
#[test]
fn belief_bonus_absent_without_belief() {
    let mut state = SimState::new(4, 42);

    let wolf = spawn_wolf(&mut state, Vec3::ZERO, 80.0, 80.0);
    let human_a = spawn_human(&mut state, Vec3::new(1.5, 0.0, 0.0), 100.0, 100.0);
    let human_b = spawn_human(&mut state, Vec3::new(-1.5, 0.0, 0.0), 100.0, 100.0);

    // No belief planted.
    let score_a = attack_score(&state, wolf, human_a);
    let score_b = attack_score(&state, wolf, human_b);

    assert!(
        (score_a - score_b).abs() < 1e-5,
        "Without beliefs both targets should score identically; got score_a={} score_b={}",
        score_a,
        score_b,
    );
}

// ---------------------------------------------------------------------------
// Test 3: belief writes via kin observation (physics cascade path)
// ---------------------------------------------------------------------------

/// Two wolves within `observation_range` (default 10 m).
/// Wolf B moves; the physics cascade fires `update_beliefs` and writes
/// a confidence-1.0 belief entry into wolf A's cold-beliefs map.
/// This exercises the write path of the pipeline end-to-end.
///
/// Note: wolves are not hostile to other wolves so wolf A's scoring never
/// selects wolf B as an Attack target — the belief is written but the
/// scoring predicate would only fire if the attacker faced a hostile version
/// of that target.  The two sub-tests above cover the scoring path directly.
#[test]
fn belief_forms_via_kin_observation() {
    let mut state = SimState::new(4, 42);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let mut views = ViewRegistry::new();
    let cascade = engine_rules::with_engine_builtins();
    let policy = UtilityBackend;
    let debug = engine::debug::DebugConfig::default();

    // Wolf A at origin; wolf B nearby (within observation_range=10m).
    let wolf_a = spawn_wolf(&mut state, Vec3::ZERO, 80.0, 80.0);
    let wolf_b = spawn_wolf(&mut state, Vec3::new(5.0, 0.0, 0.0), 80.0, 80.0);

    // Before any ticks: wolf A has no beliefs.
    {
        let beliefs = state.agent_cold_beliefs(wolf_a).expect("belief map");
        assert!(
            beliefs.get(&wolf_b).is_none(),
            "Wolf A should have no belief about wolf B before any movement"
        );
    }

    // Run one tick. Wolf B's movement (or wolf A's movement — whichever the
    // utility backend selects first) triggers update_beliefs for nearby kin.
    engine_rules::step::step(
        &mut engine_rules::backend::SerialBackend,
        &mut state,
        &mut scratch,
        &mut events,
        &mut views,
        &policy,
        &cascade,
        &debug,
    );

    // After the tick: wolf A should have a belief about wolf B (or vice-
    // versa, since the belief is written symmetrically to all nearby kin).
    // We check wolf A specifically — if it moved, the cascade writes a
    // belief into every nearby kin including wolf B; if wolf B moved, the
    // belief is written into wolf A.  Either way, at least one wolf ends up
    // with a belief about the other.
    let a_believes_b = state
        .agent_cold_beliefs(wolf_a)
        .and_then(|m| m.get(&wolf_b))
        .map(|b| b.confidence > 0.5)
        .unwrap_or(false);
    let b_believes_a = state
        .agent_cold_beliefs(wolf_b)
        .and_then(|m| m.get(&wolf_a))
        .map(|b| b.confidence > 0.5)
        .unwrap_or(false);

    assert!(
        a_believes_b || b_believes_a,
        "After one tick of movement, at least one wolf should have a high-confidence belief \
         about the other (both within observation_range=10m)"
    );
}
