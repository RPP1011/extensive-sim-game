//! Combat Foundation Task 4 — engagement-aware MoveToward/Flee +
//! OpportunityAttackTriggered cascade.
//!
//! Tests bypass `step_full` and drive the `apply_actions` + cascade path
//! directly by manually setting `agent_engaged_with` on the SoA SoT,
//! emitting a `MoveToward` / `Flee` action, and running the cascade with
//! `engine_rules::with_engine_builtins()` to get the OA damage cascade.

use engine::cascade::MAX_CASCADE_ITERATIONS;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::policy::{Action, ActionKind, MicroTarget, PolicyBackend};
use engine::mask::{MaskBuffer, MicroKind};
use engine::state::{AgentSpawn, SimState};
use engine::step::{step_full, SimScratch}; // Plan B1' Task 11: step_full is unimplemented!() stub
use engine_data::config::Config;
use glam::Vec3;

/// Policy backend that scripts a single predetermined action per tick.
struct ScriptedBackend {
    action: std::sync::Mutex<Option<Action>>,
}
impl ScriptedBackend {
    fn new(a: Action) -> Self { Self { action: std::sync::Mutex::new(Some(a)) } }
}
impl PolicyBackend for ScriptedBackend {
    fn evaluate(&self, _state: &SimState, _mask: &MaskBuffer, _target_mask: &engine::mask::TargetMask, out: &mut Vec<Action>) {
        if let Some(a) = self.action.lock().unwrap().take() {
            out.push(a);
        }
    }
}

fn spawn_human(state: &mut SimState, pos: Vec3, hp: f32) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human, pos, hp,
        ..Default::default()
    }).unwrap()
}
fn spawn_wolf(state: &mut SimState, pos: Vec3, hp: f32) -> engine::ids::AgentId {
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf, pos, hp,
        ..Default::default()
    }).unwrap()
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn engaged_move_away_from_engager_is_slowed() {
    // Human A and Wolf B, placed 1m apart so tick_start locks them into
    // mutual engagement. A is scripted to MoveToward +X (away from B, which
    // is at +X direction — wait, that's *toward* B. Put B at +X, A at origin,
    // and move A toward -X so the direction is away from engager.)
    let mut state = SimState::new(4, 42);
    let a = spawn_human(&mut state, Vec3::ZERO, 100.0);
    let b = spawn_wolf(&mut state, Vec3::new(1.0, 0.0, 0.0), 100.0);

    // Seed engagement.
    state.set_agent_engaged_with(a, Some(b));
    state.set_agent_engaged_with(b, Some(a));

    // A moves toward -X (away from engager at +X direction).
    let dest = Vec3::new(-10.0, 0.0, 0.0);
    let action = Action {
        agent: a,
        kind: ActionKind::Micro {
            kind: MicroKind::MoveToward,
            target: MicroTarget::Position(dest),
        },
    };

    let backend = ScriptedBackend::new(action);
    let cascade = engine_rules::with_engine_builtins();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);

    let before = state.agent_pos(a).unwrap();
    step_full(
        &mut state, &mut scratch, &mut events,
        &backend, &cascade, &mut [], &engine::invariant::InvariantRegistry::<Event>::new(),
        &engine::telemetry::NullSink,
    );
    // After step_full, tick_start has re-run engagement update. Since A has
    // moved a slowed step toward -X, distance is 1 + 0.3 = 1.3m — still ≤
    // engagement_range (2.0), so still engaged.
    let after = state.agent_pos(a).unwrap();
    let delta = (after - before).length();
    let cfg = Config::default();
    let expected = cfg.movement.move_speed_mps * cfg.combat.engagement_slow_factor;
    assert!(
        (delta - expected).abs() < 1e-5,
        "engaged-away move expected {}m, got {}m", expected, delta,
    );
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn engaged_move_toward_engager_is_full_speed() {
    // Same setup but A moves toward engager (+X direction, where B is).
    let mut state = SimState::new(4, 42);
    let a = spawn_human(&mut state, Vec3::ZERO, 100.0);
    let b = spawn_wolf(&mut state, Vec3::new(1.0, 0.0, 0.0), 100.0);
    state.set_agent_engaged_with(a, Some(b));
    state.set_agent_engaged_with(b, Some(a));

    // A moves toward +X (same side as engager).
    let dest = Vec3::new(10.0, 0.0, 0.0);
    let action = Action {
        agent: a,
        kind: ActionKind::Micro {
            kind: MicroKind::MoveToward,
            target: MicroTarget::Position(dest),
        },
    };

    let backend = ScriptedBackend::new(action);
    let cascade = engine_rules::with_engine_builtins();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let before = state.agent_pos(a).unwrap();
    step_full(
        &mut state, &mut scratch, &mut events,
        &backend, &cascade, &mut [], &engine::invariant::InvariantRegistry::<Event>::new(),
        &engine::telemetry::NullSink,
    );
    let after = state.agent_pos(a).unwrap();
    let delta = (after - before).length();
    let mps = Config::default().movement.move_speed_mps;
    assert!(
        (delta - mps).abs() < 1e-5,
        "engaged-toward move expected {}m, got {}m", mps, delta,
    );
    // No OpportunityAttackTriggered should have been emitted.
    let oa_count = events.iter().filter(|e| matches!(e, Event::OpportunityAttackTriggered { .. })).count();
    assert_eq!(oa_count, 0, "moving toward engager should not trigger OA");
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn flee_while_engaged_triggers_opportunity_attack() {
    // Human A (hp 100) engaged with Wolf B. A flees -> OA fires -> A takes
    // `config.combat.attack_damage` (default 10.0) damage.
    let mut state = SimState::new(4, 42);
    let a = spawn_human(&mut state, Vec3::ZERO, 100.0);
    let b = spawn_wolf(&mut state, Vec3::new(1.0, 0.0, 0.0), 100.0);
    state.set_agent_engaged_with(a, Some(b));
    state.set_agent_engaged_with(b, Some(a));

    let action = Action {
        agent: a,
        kind: ActionKind::Micro {
            kind: MicroKind::Flee,
            target: MicroTarget::Agent(b),
        },
    };
    let backend = ScriptedBackend::new(action);
    let cascade = engine_rules::with_engine_builtins();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);

    step_full(
        &mut state, &mut scratch, &mut events,
        &backend, &cascade, &mut [], &engine::invariant::InvariantRegistry::<Event>::new(),
        &engine::telemetry::NullSink,
    );

    let oa_count = events.iter().filter(|e| matches!(e, Event::OpportunityAttackTriggered { .. })).count();
    assert_eq!(oa_count, 1, "one OA should have been triggered");
    let attacked_count = events.iter().filter(|e| matches!(
        e, Event::AgentAttacked { target, .. } if *target == a
    )).count();
    assert_eq!(attacked_count, 1, "cascade should have emitted one AgentAttacked");
    // A took `config.combat.attack_damage` damage (default 10.0).
    let dmg = Config::default().combat.attack_damage;
    assert_eq!(state.agent_hp(a), Some(100.0 - dmg));
    // Sanity: we haven't blown the cascade bound (OA handler doesn't re-emit).
    let _ = MAX_CASCADE_ITERATIONS;
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn opportunity_attack_can_kill_and_cascade_emits_agentdied() {
    // Human A at hp=5 (dies on OA hit). Engaged with Wolf B. Flee → kill.
    let mut state = SimState::new(4, 42);
    let a = spawn_human(&mut state, Vec3::ZERO, 5.0);
    let b = spawn_wolf(&mut state, Vec3::new(1.0, 0.0, 0.0), 100.0);
    state.set_agent_engaged_with(a, Some(b));
    state.set_agent_engaged_with(b, Some(a));

    let action = Action {
        agent: a,
        kind: ActionKind::Micro {
            kind: MicroKind::Flee,
            target: MicroTarget::Agent(b),
        },
    };
    let backend = ScriptedBackend::new(action);
    let cascade = engine_rules::with_engine_builtins();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);

    step_full(
        &mut state, &mut scratch, &mut events,
        &backend, &cascade, &mut [], &engine::invariant::InvariantRegistry::<Event>::new(),
        &engine::telemetry::NullSink,
    );

    assert!(!state.agent_alive(a), "A should have been killed by OA");
    let died = events.iter().filter(|e| matches!(
        e, Event::AgentDied { agent_id, .. } if *agent_id == a
    )).count();
    assert_eq!(died, 1, "one AgentDied event expected");
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn unengaged_move_away_is_full_speed_and_no_oa() {
    // Baseline: an agent with engaged_with == None moves at full speed and
    // triggers no OA. Pins the guard condition on the engagement check.
    let mut state = SimState::new(4, 42);
    let a = spawn_human(&mut state, Vec3::ZERO, 100.0);
    // Place wolf out of engagement_range so tick_start leaves A unengaged.
    let _b = spawn_wolf(&mut state, Vec3::new(5.0, 0.0, 0.0), 100.0);

    let dest = Vec3::new(-10.0, 0.0, 0.0);
    let action = Action {
        agent: a,
        kind: ActionKind::Micro {
            kind: MicroKind::MoveToward,
            target: MicroTarget::Position(dest),
        },
    };

    let backend = ScriptedBackend::new(action);
    let cascade = engine_rules::with_engine_builtins();
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(256);
    let before = state.agent_pos(a).unwrap();
    step_full(
        &mut state, &mut scratch, &mut events,
        &backend, &cascade, &mut [], &engine::invariant::InvariantRegistry::<Event>::new(),
        &engine::telemetry::NullSink,
    );
    let after = state.agent_pos(a).unwrap();
    let delta = (after - before).length();
    let mps = Config::default().movement.move_speed_mps;
    assert!((delta - mps).abs() < 1e-5,
        "unengaged move expected {}m, got {}m", mps, delta);
    let oa_count = events.iter().filter(|e| matches!(e, Event::OpportunityAttackTriggered { .. })).count();
    assert_eq!(oa_count, 0);
    // The opportunity_attack dispatcher is registered by
    // `engine_rules::with_engine_builtins()` even though it never fires
    // here — the mask path is separately covered by `per_agent_combat_stats.rs`.
}
