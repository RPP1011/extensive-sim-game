use crate::ai::core::{SimState, SimVec2, Team, UnitState};
use std::collections::{HashMap, VecDeque};
use crate::game_core::RoomType;

use super::templates::{EnemyTemplate, build_enemy_unit, DEFAULT_ATTACK_CAST_MS};

// ---------------------------------------------------------------------------
// Wave builders
// ---------------------------------------------------------------------------

/// Builds a `Vec<UnitState>` from a slice of (template, position) pairs.
///
/// The `id_offset` is added to the 0-based loop index to produce each unit's
/// `id`, so callers can avoid colliding with hero IDs.
pub fn build_enemy_wave(templates: &[(EnemyTemplate, SimVec2)], id_offset: u32) -> Vec<UnitState> {
    templates
        .iter()
        .enumerate()
        .map(|(i, (template, position))| {
            build_enemy_unit(*template, id_offset + i as u32, *position)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// LCG-based balanced wave generator
// ---------------------------------------------------------------------------

/// Template distribution weights for Phase 1 balanced waves:
///   Grunt  50 %  (indices 0-49)
///   Brute  30 %  (indices 50-79)
///   Mystic 20 %  (indices 80-99)
fn lcg_template(rng: &mut u64) -> EnemyTemplate {
    // Classic LCG parameters (Numerical Recipes).
    *rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let bucket = (*rng >> 33) % 100; // 0..99
    if bucket < 50 {
        EnemyTemplate::Grunt
    } else if bucket < 80 {
        EnemyTemplate::Brute
    } else {
        EnemyTemplate::Mystic
    }
}

/// Generates a balanced Phase 1 enemy wave.
///
/// - Templates are selected via a seeded LCG (Grunt 50 %, Brute 30 %, Mystic 20 %).
/// - Positions are assigned cyclically from `spawn_positions` (wraps around if
///   there are fewer positions than units).
/// - Unit IDs start at 1000 so they do not collide with hero IDs (which
///   `build_default_sim` assigns starting at 1).
///
/// If `spawn_positions` is empty a single fallback position at (15.0, 15.0)
/// is used.
pub fn default_enemy_wave(
    count: usize,
    room_seed: u64,
    spawn_positions: &[SimVec2],
) -> Vec<UnitState> {
    const ID_OFFSET: u32 = 1000;

    let fallback = [SimVec2 { x: 15.0, y: 15.0 }];
    let positions: &[SimVec2] = if spawn_positions.is_empty() {
        &fallback
    } else {
        spawn_positions
    };

    let mut rng = room_seed;
    (0..count)
        .map(|i| {
            let template = lcg_template(&mut rng);
            let position = positions[i % positions.len()];
            build_enemy_unit(template, ID_OFFSET + i as u32, position)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Boss encounter
// ---------------------------------------------------------------------------

/// Metadata describing a climax-room boss unit.
///
/// The fields mirror the key stats written into the spawned `UnitState`; they
/// are kept together here so callers can inspect them without digging into the
/// sim unit directly.
#[derive(Debug, Clone)]
pub struct BossTemplate {
    pub name: String,
    pub hp: i32,
    pub max_hp: i32,
    pub attack_damage: i32,
    pub move_speed: f32,
}

/// The fixed unit ID used for the climax-room boss.
///
/// Chosen to avoid collisions with hero IDs (starting at 1) and regular
/// enemy wave IDs (starting at 1000 or 2000).
pub const BOSS_UNIT_ID: u32 = 9000;

/// A pool of 12 thematic boss names selected via the seeded LCG.
const BOSS_NAMES: &[&str; 12] = &[
    "The Hollow Sovereign",
    "Ashveil the Undying",
    "Krath, Fang of Ruin",
    "The Iron Confessor",
    "Mordechai, Dusk's Hand",
    "Veleth the Unbound",
    "Sorrow-Wrought Cain",
    "The Pale Archivist",
    "Graven Herald",
    "Ixcara the Sunken",
    "Dreadwarden Noss",
    "The Abyssal Seer",
];

/// Generates a boss `UnitState` for the climax room.
pub fn generate_boss(flashpoint_id: u64, difficulty: u32) -> UnitState {
    let seed = flashpoint_id
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add((difficulty as u64).wrapping_mul(0x6c62272e07bb0142));

    let mut rng = seed;
    rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let name_index = ((rng >> 33) as usize) % BOSS_NAMES.len();
    let name = BOSS_NAMES[name_index].to_string();

    let hp = 400 + difficulty as i32 * 20;
    let attack_damage = 25 + difficulty as i32 * 2;

    let _template = BossTemplate {
        name: name.clone(),
        hp,
        max_hp: hp,
        attack_damage,
        move_speed: 1.5,
    };

    UnitState {
        id: BOSS_UNIT_ID,
        team: Team::Enemy,
        hp,
        max_hp: hp,
        position: SimVec2 { x: 15.0, y: 15.0 },
        move_speed_per_sec: 1.5,
        attack_damage,
        attack_range: 2.0,
        attack_cooldown_ms: 1200,
        attack_cast_time_ms: DEFAULT_ATTACK_CAST_MS,
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
        resistance_tags: HashMap::new(),
        state_history: VecDeque::new(),
        channeling: None,
        resource: 0,
        max_resource: 0,
        resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
    }
}

/// Returns `true` when the given room type is the climax room.
pub fn is_climax_room(room_type: &RoomType) -> bool {
    matches!(room_type, RoomType::Climax)
}

// ---------------------------------------------------------------------------
// Integration helper: SimState with templated enemies
// ---------------------------------------------------------------------------

/// Constructs a `SimState` with `hero_count` default hero units and a
/// caller-supplied enemy wave produced by `default_enemy_wave` or
/// `build_enemy_wave`.
///
/// Delegates to `sim_bridge::build_sim_with_templates`.
pub fn build_sim_with_templates(
    hero_count: usize,
    enemy_wave: Vec<UnitState>,
    seed: u64,
) -> SimState {
    crate::mission::sim_bridge::build_sim_with_templates(hero_count, enemy_wave, seed)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::SimVec2;

    fn origin() -> SimVec2 {
        SimVec2 { x: 0.0, y: 0.0 }
    }

    #[test]
    fn grunt_stats_match_spec() {
        let u = build_enemy_unit(EnemyTemplate::Grunt, 1000, origin());
        assert_eq!(u.hp, 80);
        assert_eq!(u.max_hp, 80);
        assert_eq!((u.move_speed_per_sec * 10.0).round() as i32, 25);
        assert_eq!(u.attack_damage, 20);
        assert_eq!((u.attack_range * 10.0).round() as i32, 15);
        assert_eq!(u.attack_cooldown_ms, 900);
        assert_eq!(u.team, Team::Enemy);
    }

    #[test]
    fn brute_stats_match_spec() {
        let u = build_enemy_unit(EnemyTemplate::Brute, 1001, origin());
        assert_eq!(u.hp, 160);
        assert_eq!(u.attack_damage, 30);
        assert_eq!((u.attack_range * 10.0).round() as i32, 12);
        assert_eq!(u.attack_cooldown_ms, 1400);
    }

    #[test]
    fn mystic_has_heal_and_control() {
        let u = build_enemy_unit(EnemyTemplate::Mystic, 1002, origin());
        assert_eq!(u.hp, 60);
        assert_eq!(u.heal_amount, 25);
        assert_eq!((u.heal_range * 10.0).round() as i32, 35);
        assert_eq!(u.heal_cooldown_ms, 2000);
        assert_eq!(u.control_duration_ms, 2000);
        assert!((u.control_range - 3.5).abs() < 0.01);
    }

    #[test]
    fn sentinel_has_long_range() {
        let u = build_enemy_unit(EnemyTemplate::Sentinel, 1003, origin());
        assert_eq!(u.hp, 55);
        assert_eq!(u.attack_damage, 35);
        assert!((u.attack_range - 6.0).abs() < 0.01);
        assert_eq!(u.attack_cooldown_ms, 1100);
    }

    #[test]
    fn berserker_stats_match_spec() {
        let u = build_enemy_unit(EnemyTemplate::Berserker, 1004, origin());
        assert_eq!(u.hp, 90);
        assert!((u.move_speed_per_sec - 3.5).abs() < 0.01);
        assert_eq!(u.attack_damage, 25);
        assert_eq!(u.attack_cooldown_ms, 800);
    }

    #[test]
    fn summoner_stats_match_spec() {
        let u = build_enemy_unit(EnemyTemplate::Summoner, 1005, origin());
        assert_eq!(u.hp, 50);
        assert_eq!(u.attack_damage, 8);
        assert_eq!(u.attack_cooldown_ms, 1500);
    }

    #[test]
    fn build_enemy_wave_assigns_sequential_ids() {
        let pairs = vec![
            (EnemyTemplate::Grunt, SimVec2 { x: 1.0, y: 1.0 }),
            (EnemyTemplate::Brute, SimVec2 { x: 2.0, y: 2.0 }),
            (EnemyTemplate::Mystic, SimVec2 { x: 3.0, y: 3.0 }),
        ];
        let wave = build_enemy_wave(&pairs, 1000);
        assert_eq!(wave.len(), 3);
        assert_eq!(wave[0].id, 1000);
        assert_eq!(wave[1].id, 1001);
        assert_eq!(wave[2].id, 1002);
    }

    #[test]
    fn build_enemy_wave_preserves_positions() {
        let pos = SimVec2 { x: 7.5, y: 12.0 };
        let pairs = vec![(EnemyTemplate::Grunt, pos)];
        let wave = build_enemy_wave(&pairs, 500);
        assert!((wave[0].position.x - 7.5).abs() < 0.001);
        assert!((wave[0].position.y - 12.0).abs() < 0.001);
    }

    #[test]
    fn default_enemy_wave_length_matches_count() {
        let positions = vec![SimVec2 { x: 10.0, y: 10.0 }, SimVec2 { x: 12.0, y: 10.0 }];
        let wave = default_enemy_wave(5, 42, &positions);
        assert_eq!(wave.len(), 5);
    }

    #[test]
    fn default_enemy_wave_ids_start_at_1000() {
        let positions = vec![SimVec2 { x: 10.0, y: 10.0 }];
        let wave = default_enemy_wave(3, 99, &positions);
        assert_eq!(wave[0].id, 1000);
        assert_eq!(wave[1].id, 1001);
        assert_eq!(wave[2].id, 1002);
    }

    #[test]
    fn default_enemy_wave_is_deterministic() {
        let positions = vec![SimVec2 { x: 10.0, y: 10.0 }, SimVec2 { x: 12.0, y: 10.0 }];
        let a = default_enemy_wave(8, 7, &positions);
        let b = default_enemy_wave(8, 7, &positions);
        let ids_a: Vec<u32> = a.iter().map(|u| u.id).collect();
        let ids_b: Vec<u32> = b.iter().map(|u| u.id).collect();
        assert_eq!(ids_a, ids_b);
        let hp_a: Vec<i32> = a.iter().map(|u| u.hp).collect();
        let hp_b: Vec<i32> = b.iter().map(|u| u.hp).collect();
        assert_eq!(hp_a, hp_b);
    }

    #[test]
    fn default_enemy_wave_bias_roughly_holds() {
        let positions = vec![SimVec2 { x: 10.0, y: 10.0 }];
        let wave = default_enemy_wave(200, 1234, &positions);
        let grunts = wave.iter().filter(|u| u.hp == 80).count();
        let brutes = wave.iter().filter(|u| u.hp == 160).count();
        let mystics = wave.iter().filter(|u| u.hp == 60).count();
        assert!(grunts >= 70 && grunts <= 130, "Grunt count {grunts} out of range");
        assert!(brutes >= 45 && brutes <= 75, "Brute count {brutes} out of range");
        assert!(mystics >= 25 && mystics <= 55, "Mystic count {mystics} out of range");
    }

    #[test]
    fn default_enemy_wave_cycles_positions() {
        let positions = vec![
            SimVec2 { x: 1.0, y: 0.0 },
            SimVec2 { x: 2.0, y: 0.0 },
            SimVec2 { x: 3.0, y: 0.0 },
        ];
        let wave = default_enemy_wave(6, 0, &positions);
        assert!((wave[0].position.x - 1.0).abs() < 0.001);
        assert!((wave[1].position.x - 2.0).abs() < 0.001);
        assert!((wave[2].position.x - 3.0).abs() < 0.001);
        assert!((wave[3].position.x - 1.0).abs() < 0.001);
        assert!((wave[4].position.x - 2.0).abs() < 0.001);
        assert!((wave[5].position.x - 3.0).abs() < 0.001);
    }

    #[test]
    fn all_units_are_enemy_team() {
        let positions = vec![SimVec2 { x: 10.0, y: 10.0 }];
        let wave = default_enemy_wave(10, 0, &positions);
        assert!(wave.iter().all(|u| u.team == Team::Enemy));
    }

    #[test]
    fn build_sim_with_templates_has_correct_unit_counts() {
        let positions = vec![SimVec2 { x: 12.0, y: 15.0 }];
        let enemy_wave = default_enemy_wave(4, 0, &positions);
        let sim = build_sim_with_templates(3, enemy_wave, 1);
        let hero_count = sim.units.iter().filter(|u| u.team == Team::Hero).count();
        let enemy_count = sim.units.iter().filter(|u| u.team == Team::Enemy).count();
        assert_eq!(hero_count, 3);
        assert_eq!(enemy_count, 4);
    }
}
