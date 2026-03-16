//! Curriculum stages for self-play training.
//!
//! Stage 1: Move — 1 unit navigates to a target point (dummy enemy).
//! Stage 2: Kill — 1v1, reduce enemy HP to 0.
//! Stage 3: Team2 — 2v2 combat.
//! Stage 4: Team4 — 4v4 combat (uses existing scenario files).

use super::{SimState, SimVec2, Team, UnitState, UnitStore, sim_vec2};

/// Curriculum stage identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Move,
    Kill,
    Team2,
    Team4,
}

impl Stage {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "move" => Some(Stage::Move),
            "kill" => Some(Stage::Kill),
            "2v2" => Some(Stage::Team2),
            "4v4" => Some(Stage::Team4),
            _ => None,
        }
    }

    pub fn max_ticks(self) -> u64 {
        match self {
            Stage::Move => 500,   // 5 seconds — should reach target quickly
            Stage::Kill => 2000,  // 20 seconds — 1v1 should resolve
            Stage::Team2 => 3000, // 30 seconds
            Stage::Team4 => 5000, // 50 seconds
        }
    }

    pub fn step_interval(self) -> u64 {
        match self {
            Stage::Move => 5,
            Stage::Kill => 5,
            Stage::Team2 => 10,
            Stage::Team4 => 10,
        }
    }
}

fn make_unit(id: u32, team: Team, pos: SimVec2) -> UnitState {
    UnitState {
        id,
        team,
        hp: 1000,
        max_hp: 1000,
        position: pos,
        move_speed_per_sec: 3.0,
        attack_damage: 50,
        attack_range: 2.0,
        attack_cooldown_ms: 1000,
        attack_cast_time_ms: 200,
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
        abilities: Vec::new(),
        passives: Vec::new(),
        status_effects: Vec::new(),
        shield_hp: 0,
        resistance_tags: Default::default(),
        state_history: Default::default(),
        channeling: None,
        resource: 0,
        max_resource: 0,
        resource_regen_per_sec: 0.0,
        owner_id: None,
        directed: false,
        armor: 0.0,
        magic_resist: 0.0,
        cover_bonus: 0.0,
        elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
    }
}

/// Generate a Stage::Move scenario — hero needs to reach a target position.
/// The "target" is a dummy enemy with 1 HP at the destination.
pub fn generate_move(rng: &mut u64) -> (SimState, String) {
    let angle = lcg_f32(rng) * std::f32::consts::TAU;
    let dist = 8.0 + lcg_f32(rng) * 8.0; // 8-16 units away
    let target = sim_vec2(angle.cos() * dist, angle.sin() * dist);

    let mut hero = make_unit(1, Team::Hero, sim_vec2(0.0, 0.0));
    hero.attack_damage = 1; // minimal damage, this stage is about movement

    // Dummy target: 1 HP, no attack, stationary
    let mut dummy = make_unit(2, Team::Enemy, target);
    dummy.attack_damage = 0;
    dummy.move_speed_per_sec = 0.0;
    dummy.hp = 1;
    dummy.max_hp = 1;

    let sim = SimState {
        tick: 0,
        rng_state: *rng,
        units: UnitStore::new(vec![hero, dummy]),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    };
    (sim, format!("curriculum_move_{:.0}_{:.0}", target.x, target.y))
}

/// Generate a Stage::Kill scenario — 1v1 combat.
pub fn generate_kill(rng: &mut u64) -> (SimState, String) {
    let spacing = 4.0 + lcg_f32(rng) * 4.0; // 4-8 units apart

    let hero = make_unit(1, Team::Hero, sim_vec2(-spacing / 2.0, 0.0));
    let enemy = make_unit(2, Team::Enemy, sim_vec2(spacing / 2.0, 0.0));

    let sim = SimState {
        tick: 0,
        rng_state: *rng,
        units: UnitStore::new(vec![hero, enemy]),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    };
    (sim, "curriculum_kill_1v1".to_string())
}

/// Generate a Stage::Team2 scenario — 2v2 combat.
pub fn generate_team2(rng: &mut u64) -> (SimState, String) {
    let spacing = 6.0 + lcg_f32(rng) * 4.0;
    let spread = 1.5 + lcg_f32(rng) * 2.0;

    let h1 = make_unit(1, Team::Hero, sim_vec2(-spacing / 2.0, -spread));
    let h2 = make_unit(2, Team::Hero, sim_vec2(-spacing / 2.0, spread));
    let e1 = make_unit(3, Team::Enemy, sim_vec2(spacing / 2.0, -spread));
    let e2 = make_unit(4, Team::Enemy, sim_vec2(spacing / 2.0, spread));

    let sim = SimState {
        tick: 0,
        rng_state: *rng,
        units: UnitStore::new(vec![h1, h2, e1, e2]),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    };
    (sim, "curriculum_team_2v2".to_string())
}

/// Generate a Stage::Team2 variant with stat asymmetry to make targeting matter.
pub fn generate_team2_asymmetric(rng: &mut u64) -> (SimState, String) {
    let spacing = 6.0 + lcg_f32(rng) * 4.0;
    let spread = 1.5 + lcg_f32(rng) * 2.0;

    // Hero team: one tanky, one glass cannon
    let mut h_tank = make_unit(1, Team::Hero, sim_vec2(-spacing / 2.0, -spread));
    h_tank.hp = 1500;
    h_tank.max_hp = 1500;
    h_tank.attack_damage = 30;

    let mut h_dps = make_unit(2, Team::Hero, sim_vec2(-spacing / 2.0, spread));
    h_dps.hp = 600;
    h_dps.max_hp = 600;
    h_dps.attack_damage = 80;

    // Enemy: same composition
    let mut e_tank = make_unit(3, Team::Enemy, sim_vec2(spacing / 2.0, -spread));
    e_tank.hp = 1500;
    e_tank.max_hp = 1500;
    e_tank.attack_damage = 30;

    let mut e_dps = make_unit(4, Team::Enemy, sim_vec2(spacing / 2.0, spread));
    e_dps.hp = 600;
    e_dps.max_hp = 600;
    e_dps.attack_damage = 80;

    let sim = SimState {
        tick: 0,
        rng_state: *rng,
        units: UnitStore::new(vec![h_tank, h_dps, e_tank, e_dps]),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    };
    (sim, "curriculum_team_2v2_asym".to_string())
}

/// Simple LCG for procedural generation.
fn lcg_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*state >> 33) as f32 / (1u64 << 31) as f32
}
