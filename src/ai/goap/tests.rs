//! Unit tests for the GOAP system.

use super::*;
use super::action::{GoapAction, IntentTemplate};
use super::dsl::{parse_goap, GoapDef};
use super::goal::{CompOp, Goal, InsistenceFn, Precondition};
use super::party::PartyCulture;
use super::plan_cache::UnitPlanState;
use super::planner;
use super::world_state::*;
use super::target::Target;

// ---------------------------------------------------------------------------
// World state
// ---------------------------------------------------------------------------

#[test]
fn world_state_default_is_zeroed() {
    let ws = WorldState::default();
    assert_eq!(ws.get(SELF_HP_PCT).as_float(), 0.0);
    assert!(!ws.get(SELF_IS_CASTING).as_bool());
}

#[test]
fn world_state_set_get_roundtrip() {
    let mut ws = WorldState::default();
    ws.set(SELF_HP_PCT, PropValue::Float(0.75));
    ws.set(IN_ATTACK_RANGE, PropValue::Bool(true));
    ws.set(TARGET_ID, PropValue::Id(Some(42)));

    assert!((ws.get(SELF_HP_PCT).as_float() - 0.75).abs() < f32::EPSILON);
    assert!(ws.get(IN_ATTACK_RANGE).as_bool());
    assert_eq!(ws.get(TARGET_ID), PropValue::Id(Some(42)));
}

#[test]
fn prop_index_lookup() {
    assert_eq!(prop_index("self_hp_pct"), Some(SELF_HP_PCT));
    assert_eq!(prop_index("hp_pct"), Some(SELF_HP_PCT));
    assert_eq!(prop_index("in_attack_range"), Some(IN_ATTACK_RANGE));
    assert_eq!(prop_index("ability0_ready"), Some(ABILITY_0_READY));
    assert_eq!(prop_index("nonexistent"), None);
}

// ---------------------------------------------------------------------------
// Goals & insistence
// ---------------------------------------------------------------------------

fn make_engage_goal() -> Goal {
    Goal {
        name: "engage".to_string(),
        desired: vec![(ENEMY_COUNT, Precondition { op: CompOp::Eq, value: PropValue::Float(0.0) })],
        insistence: InsistenceFn::Fixed(0.8),
    }
}

fn make_heal_goal() -> Goal {
    Goal {
        name: "protect_ally".to_string(),
        desired: vec![(LOWEST_ALLY_HP_PCT, Precondition { op: CompOp::Gt, value: PropValue::Float(0.4) })],
        insistence: InsistenceFn::Linear {
            prop: LOWEST_ALLY_HP_PCT,
            scale: -1.2,
            offset: 1.0,
        },
    }
}

#[test]
fn goal_satisfied_check() {
    let goal = make_engage_goal();
    let mut ws = WorldState::default();
    ws.set(ENEMY_COUNT, PropValue::Float(0.0));
    assert!(goal.is_satisfied(&ws));

    ws.set(ENEMY_COUNT, PropValue::Float(3.0));
    assert!(!goal.is_satisfied(&ws));
}

#[test]
fn insistence_fixed() {
    let ws = WorldState::default();
    let insistence = InsistenceFn::Fixed(0.8);
    assert!((insistence.evaluate(&ws) - 0.8).abs() < f32::EPSILON);
}

#[test]
fn insistence_linear() {
    let mut ws = WorldState::default();
    let insistence = InsistenceFn::Linear { prop: LOWEST_ALLY_HP_PCT, scale: -1.2, offset: 1.0 };

    ws.set(LOWEST_ALLY_HP_PCT, PropValue::Float(0.3));
    let score = insistence.evaluate(&ws);
    assert!((score - 0.64).abs() < 0.01); // -1.2 * 0.3 + 1.0 = 0.64

    ws.set(LOWEST_ALLY_HP_PCT, PropValue::Float(1.0));
    let score = insistence.evaluate(&ws);
    assert!((score - 0.0).abs() < 0.01); // -1.2 * 1.0 + 1.0 = -0.2, clamped to 0
}

#[test]
fn insistence_threshold() {
    let mut ws = WorldState::default();
    let insistence = InsistenceFn::Threshold {
        prop: ENEMY_IS_CASTING,
        op: CompOp::Gt,
        threshold: 0.0,
        value: 0.95,
    };

    ws.set(ENEMY_IS_CASTING, PropValue::Bool(false));
    assert!((insistence.evaluate(&ws) - 0.0).abs() < f32::EPSILON);

    ws.set(ENEMY_IS_CASTING, PropValue::Bool(true));
    assert!((insistence.evaluate(&ws) - 0.95).abs() < f32::EPSILON);
}

#[test]
fn goal_selection_hysteresis() {
    let goals = vec![make_engage_goal(), make_heal_goal()];
    let mut ws = WorldState::default();
    ws.set(ENEMY_COUNT, PropValue::Float(2.0));
    ws.set(LOWEST_ALLY_HP_PCT, PropValue::Float(0.3)); // heal insistence = 0.64

    // With no current goal, pick highest insistence (engage at 0.8 > heal at 0.64)
    let selected = goal::select_goal(&goals, &ws, None, 0.15);
    assert_eq!(selected.unwrap().name, "engage");

    // If we're on engage (0.8), heal (0.64) doesn't beat it even without hysteresis
    let selected = goal::select_goal(&goals, &ws, Some("engage"), 0.15);
    assert_eq!(selected.unwrap().name, "engage");

    // Lower ally HP to make heal more urgent
    ws.set(LOWEST_ALLY_HP_PCT, PropValue::Float(0.05)); // heal insistence = -1.2 * 0.05 + 1.0 = 0.94
    let selected = goal::select_goal(&goals, &ws, Some("engage"), 0.15);
    // 0.94 > 0.8 + 0.15 = 0.95 — just barely not enough
    // Actually 0.94 < 0.95, so we stick with engage
    assert_eq!(selected.unwrap().name, "engage");

    ws.set(LOWEST_ALLY_HP_PCT, PropValue::Float(0.0)); // heal insistence = 1.0
    let selected = goal::select_goal(&goals, &ws, Some("engage"), 0.15);
    // 1.0 > 0.95 — now heal wins
    assert_eq!(selected.unwrap().name, "protect_ally");
}

// ---------------------------------------------------------------------------
// Actions & planner
// ---------------------------------------------------------------------------

fn make_test_actions() -> Vec<GoapAction> {
    vec![
        GoapAction {
            name: "attack".to_string(),
            cost: 1.0,
            preconditions: vec![
                (IN_ATTACK_RANGE, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
            ],
            effects: vec![
                (ENEMY_COUNT, PropValue::Float(0.0)), // simplified: attack kills enemy
            ],
            intent: IntentTemplate::AttackTarget(Target::NearestEnemy),
            duration_ticks: 1,
        },
        GoapAction {
            name: "close_gap".to_string(),
            cost: 1.5,
            preconditions: vec![
                (TARGET_IS_ALIVE, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
            ],
            effects: vec![
                (IN_ATTACK_RANGE, PropValue::Bool(true)),
            ],
            intent: IntentTemplate::ChaseTarget(Target::NearestEnemy),
            duration_ticks: 4,
        },
    ]
}

#[test]
fn planner_single_step_plan() {
    let mut ws = WorldState::default();
    ws.set(IN_ATTACK_RANGE, PropValue::Bool(true));
    ws.set(ENEMY_COUNT, PropValue::Float(2.0));
    ws.set(TARGET_IS_ALIVE, PropValue::Bool(true));

    let goal = make_engage_goal();
    let actions = make_test_actions();

    let plan = planner::plan(&ws, &goal, &actions, 0, None);
    assert!(plan.is_some());
    let plan = plan.unwrap();
    assert_eq!(plan.actions.len(), 1);
    assert_eq!(plan.actions[0], 0); // attack
}

#[test]
fn planner_multi_step_plan() {
    let mut ws = WorldState::default();
    ws.set(IN_ATTACK_RANGE, PropValue::Bool(false));
    ws.set(ENEMY_COUNT, PropValue::Float(2.0));
    ws.set(TARGET_IS_ALIVE, PropValue::Bool(true));

    let goal = make_engage_goal();
    let actions = make_test_actions();

    let plan = planner::plan(&ws, &goal, &actions, 0, None);
    assert!(plan.is_some());
    let plan = plan.unwrap();
    assert_eq!(plan.actions.len(), 2);
    assert_eq!(plan.actions[0], 1); // close_gap first
    assert_eq!(plan.actions[1], 0); // then attack
}

#[test]
fn planner_no_plan_when_goal_satisfied() {
    let mut ws = WorldState::default();
    ws.set(ENEMY_COUNT, PropValue::Float(0.0));

    let goal = make_engage_goal();
    let actions = make_test_actions();

    let plan = planner::plan(&ws, &goal, &actions, 0, None);
    assert!(plan.is_none());
}

// ---------------------------------------------------------------------------
// Plan cache / replan
// ---------------------------------------------------------------------------

#[test]
fn plan_cache_advance_step() {
    let mut ps = UnitPlanState::default();
    ps.current_plan = Some(planner::Plan {
        actions: vec![1, 0],
        goal_name: "engage".to_string(),
        created_tick: 0,
    });
    ps.current_step = 0;

    assert_eq!(ps.current_action_idx(), Some(1));
    assert!(ps.advance_step());
    assert_eq!(ps.current_action_idx(), Some(0));
    assert!(!ps.advance_step());
    assert_eq!(ps.current_action_idx(), None);
}

#[test]
fn oscillation_guard_triggers() {
    let mut ps = UnitPlanState::default();

    let ws = WorldState::default();
    let goals = vec![make_engage_goal()];
    let actions = make_test_actions();

    // Rapid replanning
    for tick in 0..5 {
        ps.replan(&ws, &goals, &actions, tick, 0.15, None);
    }

    // After 4+ replans in quick succession, forced hold should activate
    assert!(ps.forced_hold_until > 0);
}

// ---------------------------------------------------------------------------
// DSL parser
// ---------------------------------------------------------------------------

#[test]
fn parse_simple_goap() {
    let input = r#"
goap "tank" {
    role_hint melee_engage

    goal "engage" {
        desire enemy_count == 0
        insistence fixed 0.8
    }

    action "attack" {
        cost 1.0
        precondition in_attack_range == true
        effect enemy_count = 0
        intent attack nearest_enemy
        duration 1
    }

    action "close_gap" {
        cost 1.5
        precondition target_is_alive == true
        effect in_attack_range = true
        intent chase nearest_enemy
        duration 4
    }
}
"#;

    let def = parse_goap(input).unwrap();
    assert_eq!(def.name, "tank");
    assert_eq!(def.role_hint, Some(dsl::RoleHint::MeleeEngage));
    assert_eq!(def.goals.len(), 1);
    assert_eq!(def.goals[0].name, "engage");
    assert_eq!(def.actions.len(), 2);
    assert_eq!(def.actions[0].name, "attack");
    assert_eq!(def.actions[1].name, "close_gap");
    assert!((def.actions[0].cost - 1.0).abs() < f32::EPSILON);
    assert_eq!(def.actions[1].duration_ticks, 4);
}

#[test]
fn parse_goap_with_linear_insistence() {
    let input = r#"
goap "healer" {
    goal "protect_ally" {
        desire lowest_ally_hp_pct > 0.4
        insistence linear lowest_ally_hp_pct scale -1.2 offset 1.0
    }

    action "heal" {
        cost 0.5
        precondition has_heal_target == true
        precondition ability0_ready == true
        effect lowest_ally_hp_pct = 0.7
        intent cast_if_ready ability0 on lowest_hp_ally
        duration 2
    }
}
"#;

    let def = parse_goap(input).unwrap();
    assert_eq!(def.goals[0].name, "protect_ally");
    assert_eq!(def.actions[0].preconditions.len(), 2);
}

#[test]
fn parse_goap_with_threshold_insistence() {
    let input = r#"
goap "interrupter" {
    goal "interrupt" {
        desire enemy_is_casting == false
        insistence threshold enemy_is_casting > 0 value 0.95
    }

    action "cc_interrupt" {
        cost 0.2
        precondition ability1_ready == true
        precondition enemy_is_casting == true
        effect enemy_is_casting = false
        intent cast_if_ready ability1 on casting_enemy
        duration 1
    }
}
"#;

    let def = parse_goap(input).unwrap();
    assert_eq!(def.goals[0].name, "interrupt");
    assert_eq!(def.actions[0].name, "cc_interrupt");
    assert!((def.actions[0].cost - 0.2).abs() < f32::EPSILON);
}


// ---------------------------------------------------------------------------
// Party culture
// ---------------------------------------------------------------------------

#[test]
fn party_culture_generation_produces_valid_traits() {
    let mut counter = 0u64;
    let mut rng = || { counter = counter.wrapping_add(1); counter.wrapping_mul(6364136223846793005).wrapping_add(1) };

    let mut names = std::collections::HashSet::new();
    for _ in 0..100 {
        let culture = PartyCulture::generate(&mut rng);

        // Validate trait bounds
        assert!(culture.traits.aggression_bias >= -0.3 && culture.traits.aggression_bias <= 0.3,
            "aggression_bias out of range: {}", culture.traits.aggression_bias);
        assert!(culture.traits.coordination >= 0.0 && culture.traits.coordination <= 1.0);
        assert!(culture.traits.ability_eagerness >= 0.0 && culture.traits.ability_eagerness <= 1.0);
        assert!(culture.traits.retreat_threshold >= 0.0 && culture.traits.retreat_threshold <= 1.0);
        assert!(culture.traits.protect_instinct >= 0.0 && culture.traits.protect_instinct <= 1.0);

        assert!(!culture.name.is_empty());
        names.insert(culture.name.clone());
    }

    // With 15 adjectives × 15 nouns = 225 possible names, 100 samples should produce variety
    assert!(names.len() > 10, "Expected name variety, got {} unique names", names.len());
}

#[test]
fn party_culture_from_toml() {
    let culture = PartyCulture::from_toml_fields(
        Some("Bloodsworn"),
        Some(0.2),
        Some(0.8),
        Some(0.7),
        Some(0.15),
        Some(0.6),
    );
    assert_eq!(culture.name, "Bloodsworn");
    assert!((culture.traits.aggression_bias - 0.2).abs() < f32::EPSILON);
    assert!((culture.traits.coordination - 0.8).abs() < f32::EPSILON);
}

// ---------------------------------------------------------------------------
// Verify
// ---------------------------------------------------------------------------

#[test]
fn verify_detects_excessive_replanning() {
    let mut goap = GoapAiState {
        defs: HashMap::new(),
        plans: HashMap::new(),
        world_cache: HashMap::new(),
        culture: None,
    };

    let def = GoapDef {
        name: "test".to_string(),
        role_hint: None,
        goals: vec![],
        actions: vec![],
    };

    goap.defs.insert(1, def);
    let mut ps = UnitPlanState::default();
    ps.replan_count = 6;
    goap.plans.insert(1, ps);

    // Need a minimal SimState with unit 1 alive
    let state = crate::ai::core::SimState {
        tick: 10,
        rng_state: 0,
        units: crate::ai::core::UnitStore::new(vec![crate::ai::core::UnitState {
            id: 1,
            team: crate::ai::core::Team::Hero,
            hp: 100,
            max_hp: 100,
            position: crate::ai::core::SimVec2 { x: 0.0, y: 0.0 },
            move_speed_per_sec: 3.0,
            attack_damage: 10,
            attack_range: 1.5,
            attack_cooldown_ms: 1000,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 0,
            ability_range: 0.0,
            ability_cooldown_ms: 0,
            ability_cast_time_ms: 0,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: vec![],
            passives: vec![],
            status_effects: vec![],
            shield_hp: 0,
            resistance_tags: Default::default(),
            state_history: Default::default(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0,
            owner_id: None,
            directed: false,
            total_healing_done: 0,
            total_damage_done: 0,
            armor: 0.0,
            magic_resist: 0.0,
            cover_bonus: 0.0,
            elevation: 0.0,
        }]),
        projectiles: vec![],
        passive_trigger_depth: 0,
        zones: vec![],
        tethers: vec![],
        grid_nav: None,
    };

    let violations = verify::verify_goap(&goap, &state);
    assert!(violations.iter().any(|v| matches!(v, verify::GoapViolation::ExcessiveReplanning { .. })));
}

// ---------------------------------------------------------------------------
// Benchmarks (run with `cargo test --lib goap::tests::bench -- --nocapture`)
// ---------------------------------------------------------------------------

/// Build a realistic set of actions similar to frontline.goap (4 actions, mixed preconditions).
fn make_frontline_actions() -> Vec<GoapAction> {
    vec![
        GoapAction {
            name: "attack".into(),
            cost: 1.0,
            preconditions: vec![
                (IN_ATTACK_RANGE, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
            ],
            effects: vec![(ENEMY_COUNT, PropValue::Float(0.0))],
            intent: IntentTemplate::AttackTarget(Target::NearestEnemy),
            duration_ticks: 1,
        },
        GoapAction {
            name: "close_gap".into(),
            cost: 1.5,
            preconditions: vec![
                (TARGET_IS_ALIVE, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
            ],
            effects: vec![(IN_ATTACK_RANGE, PropValue::Bool(true))],
            intent: IntentTemplate::ChaseTarget(Target::NearestEnemy),
            duration_ticks: 4,
        },
        GoapAction {
            name: "cc_interrupt".into(),
            cost: 0.2,
            preconditions: vec![
                (ABILITY_1_READY, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
                (ENEMY_IS_CASTING, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
            ],
            effects: vec![(ENEMY_IS_CASTING, PropValue::Bool(false))],
            intent: IntentTemplate::CastIfReady(1, Target::CastingEnemy),
            duration_ticks: 1,
        },
        GoapAction {
            name: "protect_heal".into(),
            cost: 0.8,
            preconditions: vec![
                (ABILITY_0_READY, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
                (HAS_HEAL_TARGET, Precondition { op: CompOp::Eq, value: PropValue::Bool(true) }),
            ],
            effects: vec![(LOWEST_ALLY_HP_PCT, PropValue::Float(0.7))],
            intent: IntentTemplate::CastIfReady(0, Target::LowestHpAlly),
            duration_ticks: 2,
        },
    ]
}

fn make_frontline_goals() -> Vec<Goal> {
    vec![
        Goal {
            name: "engage".into(),
            desired: vec![(ENEMY_COUNT, Precondition { op: CompOp::Eq, value: PropValue::Float(0.0) })],
            insistence: InsistenceFn::Fixed(0.8),
        },
        Goal {
            name: "protect_ally".into(),
            desired: vec![(LOWEST_ALLY_HP_PCT, Precondition { op: CompOp::Gt, value: PropValue::Float(0.4) })],
            insistence: InsistenceFn::Linear { prop: LOWEST_ALLY_HP_PCT, scale: -1.2, offset: 1.0 },
        },
        Goal {
            name: "interrupt".into(),
            desired: vec![(ENEMY_IS_CASTING, Precondition { op: CompOp::Eq, value: PropValue::Bool(false) })],
            insistence: InsistenceFn::Threshold { prop: ENEMY_IS_CASTING, op: CompOp::Gt, threshold: 0.0, value: 0.95 },
        },
    ]
}

#[test]
fn bench_planner_single_step() {
    let actions = make_frontline_actions();
    let goals = make_frontline_goals();

    // World state: in attack range, enemies exist → single-step "attack" plan
    let mut ws = WorldState::default();
    ws.set(IN_ATTACK_RANGE, PropValue::Bool(true));
    ws.set(ENEMY_COUNT, PropValue::Float(3.0));
    ws.set(TARGET_IS_ALIVE, PropValue::Bool(true));

    let iters = 100_000;
    let start = std::time::Instant::now();
    for i in 0..iters {
        let plan = planner::plan(&ws, &goals[0], &actions, i as u64, None);
        std::hint::black_box(&plan);
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() / iters as u128;
    eprintln!(
        "bench_planner_single_step: {}ns/call ({:.1}us), {:.1}M calls/sec",
        per_call_ns,
        per_call_ns as f64 / 1000.0,
        1_000_000_000.0 / per_call_ns as f64 / 1_000_000.0,
    );
}

#[test]
fn bench_planner_multi_step() {
    let actions = make_frontline_actions();
    let goals = make_frontline_goals();

    // World state: NOT in attack range → needs close_gap then attack (2-step)
    let mut ws = WorldState::default();
    ws.set(IN_ATTACK_RANGE, PropValue::Bool(false));
    ws.set(ENEMY_COUNT, PropValue::Float(3.0));
    ws.set(TARGET_IS_ALIVE, PropValue::Bool(true));

    let iters = 100_000;
    let start = std::time::Instant::now();
    for i in 0..iters {
        let plan = planner::plan(&ws, &goals[0], &actions, i as u64, None);
        std::hint::black_box(&plan);
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() / iters as u128;
    eprintln!(
        "bench_planner_multi_step: {}ns/call ({:.1}us), {:.1}M calls/sec",
        per_call_ns,
        per_call_ns as f64 / 1000.0,
        1_000_000_000.0 / per_call_ns as f64 / 1_000_000.0,
    );
}

#[test]
fn bench_world_state_extract() {
    // Build a minimal 8-unit SimState
    let mut units = Vec::new();
    for i in 0..8 {
        units.push(crate::ai::core::UnitState {
            id: i,
            team: if i < 4 { crate::ai::core::Team::Hero } else { crate::ai::core::Team::Enemy },
            hp: 100,
            max_hp: 100,
            position: crate::ai::core::SimVec2 { x: i as f32 * 2.0, y: 5.0 },
            move_speed_per_sec: 3.0,
            attack_damage: 10,
            attack_range: 1.5,
            attack_cooldown_ms: 1000,
            attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0,
            ability_damage: 20,
            ability_range: 5.0,
            ability_cooldown_ms: 3000,
            ability_cast_time_ms: 500,
            ability_cooldown_remaining_ms: 0,
            heal_amount: 0,
            heal_range: 0.0,
            heal_cooldown_ms: 0,
            heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0,
            control_range: 0.0,
            control_duration_ms: 0,
            control_cooldown_ms: 0,
            control_cast_time_ms: 0,
            control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0,
            casting: None,
            abilities: vec![],
            passives: vec![],
            status_effects: vec![],
            shield_hp: 0,
            resistance_tags: Default::default(),
            state_history: Default::default(),
            channeling: None,
            resource: 0,
            max_resource: 0,
            resource_regen_per_sec: 0.0,
            owner_id: None,
            directed: false,
            total_healing_done: 0,
            total_damage_done: 0,
            armor: 0.0,
            magic_resist: 0.0,
            cover_bonus: 0.0,
            elevation: 0.0,
        });
    }
    let state = crate::ai::core::SimState {
        tick: 100,
        rng_state: 0,
        units: crate::ai::core::UnitStore::new(units),
        projectiles: vec![],
        passive_trigger_depth: 0,
        zones: vec![],
        tethers: vec![],
        grid_nav: None,
    };

    let iters = 100_000;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let ws = WorldState::extract(&state, 0);
        std::hint::black_box(&ws);
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() / iters as u128;
    eprintln!(
        "bench_world_state_extract: {}ns/call ({:.1}us), {:.1}M calls/sec",
        per_call_ns,
        per_call_ns as f64 / 1000.0,
        1_000_000_000.0 / per_call_ns as f64 / 1_000_000.0,
    );
}

#[test]
fn bench_goal_selection() {
    let goals = make_frontline_goals();
    let mut ws = WorldState::default();
    ws.set(ENEMY_COUNT, PropValue::Float(3.0));
    ws.set(LOWEST_ALLY_HP_PCT, PropValue::Float(0.5));
    ws.set(ENEMY_IS_CASTING, PropValue::Bool(false));

    let iters = 1_000_000;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let g = goal::select_goal(&goals, &ws, Some("engage"), 0.15);
        std::hint::black_box(&g);
    }
    let elapsed = start.elapsed();
    let per_call_ns = elapsed.as_nanos() / iters as u128;
    eprintln!(
        "bench_goal_selection: {}ns/call, {:.1}M calls/sec",
        per_call_ns,
        1_000_000_000.0 / per_call_ns as f64 / 1_000_000.0,
    );
}
