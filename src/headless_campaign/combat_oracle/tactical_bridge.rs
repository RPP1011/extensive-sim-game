//! Tactical sim bridge — runs actual deterministic combat for campaign battles.
//!
//! Converts campaign `Adventurer` data into tactical sim `UnitState` units,
//! builds enemies scaled to the quest threat level, and runs the full combat
//! loop to completion. Generated abilities are applied as real ability slots
//! so they affect combat outcomes.

use std::collections::VecDeque;

use tactical_sim::effects::defs::{AbilityDef, AbilitySlot};
use tactical_sim::effects::types::Tags;
use tactical_sim::sim::{
    sim_vec2, step, SimState, SimVec2, Team, UnitState, UnitStore, FIXED_TICK_MS,
};
use tactical_sim::squad::SquadAiState;

use super::CombatOracleResult;
use crate::headless_campaign::state::{Adventurer, CampaignState, UnlockCategory, UnlockInstance};

/// Maximum ticks before declaring a timeout (draw).
const MAX_COMBAT_TICKS: u64 = 5000;

/// Result of running a tactical sim fight.
#[derive(Debug, Clone)]
pub struct TacticalResult {
    /// True if the hero team won.
    pub victory: bool,
    /// Per-hero HP remaining as fraction of max (0-1). Same order as input.
    pub hp_remaining: Vec<f32>,
    /// Number of hero casualties (hp <= 0).
    pub casualties: usize,
    /// Ticks elapsed.
    pub ticks: u64,
}

// ---------------------------------------------------------------------------
// Adventurer → UnitState conversion
// ---------------------------------------------------------------------------

/// Convert a campaign Adventurer into a tactical sim UnitState.
///
/// Uses the adventurer's archetype to look up a base hero template, then
/// scales stats based on the adventurer's level and stats. Generated abilities
/// from progression are attached as real AbilitySlot entries.
fn adventurer_to_unit(
    adventurer: &Adventurer,
    id: u32,
    position: SimVec2,
    state: &CampaignState,
) -> UnitState {
    // Try to load the hero template for this archetype
    let base_template = try_load_hero_template(&adventurer.archetype);

    // Scale stats from the campaign adventurer
    let level_scale = 1.0 + (adventurer.level as f32 - 1.0) * 0.1;
    let hp = (adventurer.stats.max_hp * level_scale) as i32;
    let attack_damage = (adventurer.stats.attack * level_scale) as i32;

    // Collect generated abilities for this adventurer from progression
    let mut ability_slots = Vec::new();
    for prog in &state.pending_progression {
        if prog.adventurer_id == Some(adventurer.id)
            && prog.kind == crate::headless_campaign::state::ProgressionKind::Ability
        {
            if let Some(def) = try_parse_ability_dsl(&prog.content) {
                ability_slots.push(AbilitySlot::new(def));
            }
        }
    }

    // Also check completed progression (unlocks with ActiveAbility category)
    // These are guild-wide, so they apply to all adventurers.
    // Individual ability slots from template take priority.

    // Start from template if available, otherwise build from stats
    let mut unit = if let Some(template) = base_template {
        let mut u = crate::mission::hero_templates::hero_toml_to_unit(
            &template, id, Team::Hero, position,
        );
        // Override HP/attack with campaign stats (which include level scaling)
        u.hp = hp.max(1);
        u.max_hp = hp.max(1);
        u.attack_damage = attack_damage.max(1);
        // Merge generated abilities with template abilities
        u.abilities.extend(ability_slots);
        u
    } else {
        // Fallback: build a minimal unit from raw stats
        let mut u = build_unit_from_stats(adventurer, id, position);
        u.abilities = ability_slots;
        u
    };

    // Apply buff effects from active unlocks
    apply_unlock_buffs(&mut unit, &state.unlocks);

    // Apply condition modifiers (injury/fatigue reduce effectiveness)
    let condition = 1.0 - (adventurer.injury / 200.0 + adventurer.fatigue / 200.0);
    let condition = condition.max(0.1);
    unit.hp = ((unit.hp as f32) * condition) as i32;
    unit.max_hp = ((unit.max_hp as f32) * condition) as i32;
    unit.attack_damage = ((unit.attack_damage as f32) * condition) as i32;

    unit
}

/// Try to load a hero template by archetype name.
fn try_load_hero_template(archetype: &str) -> Option<tactical_sim::effects::HeroToml> {
    use crate::mission::hero_templates::{load_embedded_templates, parse_hero_toml_with_dsl};

    // Check embedded templates first
    let lower = archetype.to_lowercase();
    let embedded = load_embedded_templates();
    for (template, toml) in &embedded {
        if template.file_name().trim_end_matches(".toml").eq_ignore_ascii_case(&lower) {
            return Some(toml.clone());
        }
    }

    // Try loading from disk
    let paths = [
        format!("dataset/abilities/hero_templates/{}.toml", lower),
        format!("dataset/abilities/lol_heroes/{}.toml", lower),
    ];
    for path in &paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            let dsl_content = std::path::Path::new(path)
                .with_extension("ability")
                .to_str()
                .and_then(|p| std::fs::read_to_string(p).ok());
            if let Ok(toml) = parse_hero_toml_with_dsl(&content, dsl_content.as_deref()) {
                return Some(toml);
            }
        }
    }

    None
}

/// Build a minimal UnitState from campaign AdventurerStats.
fn build_unit_from_stats(adv: &Adventurer, id: u32, position: SimVec2) -> UnitState {
    let level_scale = 1.0 + (adv.level as f32 - 1.0) * 0.1;
    UnitState {
        id,
        team: Team::Hero,
        hp: (adv.stats.max_hp * level_scale) as i32,
        max_hp: (adv.stats.max_hp * level_scale) as i32,
        position,
        move_speed_per_sec: adv.stats.speed.max(3.0),
        attack_damage: (adv.stats.attack * level_scale) as i32,
        attack_range: 2.0,
        attack_cooldown_ms: 1000,
        attack_cast_time_ms: 200,
        cooldown_remaining_ms: 0,
        ability_damage: (adv.stats.ability_power * level_scale) as i32,
        ability_range: 5.0,
        ability_cooldown_ms: 5000,
        ability_cast_time_ms: 300,
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
        resistance_tags: Tags::default(),
        state_history: VecDeque::new(),
        channeling: None,
        resource: 0,
        max_resource: 0,
        resource_regen_per_sec: 0.0,
        owner_id: None,
        directed: false,
        armor: adv.stats.defense * 0.5,
        magic_resist: adv.stats.defense * 0.3,
        cover_bonus: 0.0,
        elevation: 0.0,
        total_healing_done: 0,
        total_damage_done: 0,
    }
}

/// Try to parse ability DSL text into an AbilityDef.
/// Returns the first ability defined in the DSL text.
fn try_parse_ability_dsl(dsl_text: &str) -> Option<AbilityDef> {
    tactical_sim::effects::dsl::parse_abilities(dsl_text)
        .ok()
        .and_then(|(abilities, _passives)| abilities.into_iter().next())
}

/// Apply active unlock buffs to a unit's stats.
///
/// PassiveBuff unlocks modify stats multiplicatively:
/// - magnitude is treated as a percentage bonus (e.g. 0.1 = +10%)
/// ActiveAbility unlocks with shields add initial shield HP.
fn apply_unlock_buffs(unit: &mut UnitState, unlocks: &[UnlockInstance]) {
    for unlock in unlocks {
        if !unlock.active {
            continue;
        }
        match unlock.category {
            UnlockCategory::PassiveBuff => {
                let bonus = unlock.properties.magnitude;
                // Apply as multiplier to attack and HP
                unit.attack_damage =
                    ((unit.attack_damage as f32) * (1.0 + bonus)) as i32;
                unit.hp = ((unit.hp as f32) * (1.0 + bonus * 0.5)) as i32;
                unit.max_hp = ((unit.max_hp as f32) * (1.0 + bonus * 0.5)) as i32;
            }
            UnlockCategory::ActiveAbility => {
                // Active abilities with magnitude add shield HP at fight start
                let shield = (unlock.properties.magnitude * 50.0) as i32;
                unit.shield_hp += shield;
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Enemy generation scaled to threat level
// ---------------------------------------------------------------------------

/// Build enemy units for a given threat level and party size.
///
/// Uses the standard enemy wave generator, then scales stats to match threat.
fn build_enemies(
    threat_level: f32,
    party_size: usize,
    seed: u64,
) -> Vec<UnitState> {
    // Enemy count scales with threat: 1-2 at low threat, up to party_size+1 at high
    let enemy_count = ((threat_level / 25.0) as usize + 1).min(party_size + 2).max(1);

    let spawn_positions: Vec<SimVec2> = (0..enemy_count)
        .map(|i| sim_vec2(12.0 + (i as f32) * 2.0, 15.0))
        .collect();

    let mut enemies =
        crate::mission::enemy_templates::default_enemy_wave(enemy_count, seed, &spawn_positions);

    // Scale enemy stats based on threat level
    // threat_level 0-100 maps to stat multiplier 0.5x-3.0x
    let scale = 0.5 + (threat_level / 100.0) * 2.5;
    for enemy in &mut enemies {
        enemy.hp = ((enemy.hp as f32) * scale) as i32;
        enemy.max_hp = ((enemy.max_hp as f32) * scale) as i32;
        enemy.attack_damage = ((enemy.attack_damage as f32) * scale) as i32;
    }

    enemies
}

// ---------------------------------------------------------------------------
// Combat runner
// ---------------------------------------------------------------------------

/// Run a full tactical combat simulation.
///
/// Returns a `TacticalResult` with victory/loss, HP remaining, and casualties.
pub fn run_tactical_combat(
    party_members: &[&Adventurer],
    threat_level: f32,
    seed: u64,
    state: &CampaignState,
) -> TacticalResult {
    if party_members.is_empty() {
        return TacticalResult {
            victory: false,
            hp_remaining: Vec::new(),
            casualties: 0,
            ticks: 0,
        };
    }

    // Build hero units from adventurers
    let mut units: Vec<UnitState> = Vec::new();
    for (i, adv) in party_members.iter().enumerate() {
        let pos = sim_vec2(2.0 + (i as f32) * 2.0, 5.0);
        units.push(adventurer_to_unit(adv, (i + 1) as u32, pos, state));
    }

    let hero_count = units.len();
    let hero_ids: Vec<u32> = units.iter().map(|u| u.id).collect();

    // Build enemies
    let enemies = build_enemies(threat_level, party_members.len(), seed);
    units.extend(enemies);

    // Build SimState
    let mut sim = SimState {
        tick: 0,
        rng_state: seed,
        units: UnitStore::new(units),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    };

    // Build AI
    let mut squad_ai = SquadAiState::new_inferred(&sim);

    // Run combat loop
    let mut ticks: u64 = 0;
    loop {
        let intents =
            tactical_sim::squad::generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        let (new_sim, _events) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;
        ticks += 1;

        // Check win/loss conditions
        let heroes_alive = sim
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0)
            .count();
        let enemies_alive = sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0)
            .count();

        if enemies_alive == 0 {
            // Victory
            let hp_remaining: Vec<f32> = hero_ids
                .iter()
                .map(|id| {
                    sim.units
                        .iter()
                        .find(|u| u.id == *id)
                        .map(|u| (u.hp as f32 / u.max_hp as f32).max(0.0))
                        .unwrap_or(0.0)
                })
                .collect();
            let casualties = hero_count - heroes_alive;
            return TacticalResult {
                victory: true,
                hp_remaining,
                casualties,
                ticks,
            };
        }

        if heroes_alive == 0 {
            // Defeat
            let hp_remaining = vec![0.0; hero_count];
            return TacticalResult {
                victory: false,
                hp_remaining,
                casualties: hero_count,
                ticks,
            };
        }

        if ticks >= MAX_COMBAT_TICKS {
            // Timeout — whoever has more total HP% wins
            let hero_hp_total: f32 = sim
                .units
                .iter()
                .filter(|u| u.team == Team::Hero && u.hp > 0)
                .map(|u| u.hp as f32 / u.max_hp as f32)
                .sum();
            let enemy_hp_total: f32 = sim
                .units
                .iter()
                .filter(|u| u.team == Team::Enemy && u.hp > 0)
                .map(|u| u.hp as f32 / u.max_hp as f32)
                .sum();

            let victory = hero_hp_total > enemy_hp_total;
            let hp_remaining: Vec<f32> = hero_ids
                .iter()
                .map(|id| {
                    sim.units
                        .iter()
                        .find(|u| u.id == *id)
                        .map(|u| (u.hp as f32 / u.max_hp as f32).max(0.0))
                        .unwrap_or(0.0)
                })
                .collect();
            let casualties = hero_count - heroes_alive;
            return TacticalResult {
                victory,
                hp_remaining,
                casualties,
                ticks,
            };
        }
    }
}

/// Convert a TacticalResult into a CombatOracleResult for compatibility
/// with the existing battle system.
pub fn tactical_to_oracle_result(result: &TacticalResult, _party_size: usize) -> CombatOracleResult {
    let avg_hp = if result.hp_remaining.is_empty() {
        0.0
    } else {
        result.hp_remaining.iter().sum::<f32>() / result.hp_remaining.len() as f32
    };

    CombatOracleResult {
        victory_probability: if result.victory { 1.0 } else { 0.0 },
        expected_hp_remaining: avg_hp,
        expected_ticks: result.ticks,
        expected_casualties: result.casualties as f32,
    }
}

// ---------------------------------------------------------------------------
// Enhanced heuristic oracle — factors in unlock power
// ---------------------------------------------------------------------------

/// Compute a power bonus from active unlocks.
///
/// This makes generated abilities affect even the sigmoid oracle's
/// prediction, so abilities are never purely cosmetic.
pub fn unlock_power_bonus(unlocks: &[UnlockInstance]) -> f32 {
    let mut bonus = 0.0;
    for unlock in unlocks {
        if !unlock.active {
            continue;
        }
        match unlock.category {
            UnlockCategory::PassiveBuff => {
                bonus += unlock.properties.magnitude * 20.0;
            }
            UnlockCategory::ActiveAbility => {
                bonus += unlock.properties.magnitude * 10.0;
            }
            UnlockCategory::Information => {
                // Intel gives a small power bonus (better prepared)
                bonus += unlock.properties.magnitude * 5.0;
            }
            UnlockCategory::Economic => {
                // Economic unlocks don't directly affect combat
            }
        }
    }
    bonus
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::state::*;

    fn test_adventurer(id: u32, archetype: &str, level: u32, attack: f32, defense: f32) -> Adventurer {
        Adventurer {
            id,
            name: format!("Test_{}", id),
            archetype: archetype.into(),
            level,
            xp: 0,
            stats: AdventurerStats {
                max_hp: 100.0,
                attack,
                defense,
                speed: 10.0,
                ability_power: 10.0,
            },
            equipment: Equipment::default(),
            traits: Vec::new(),
            status: AdventurerStatus::Idle,
            loyalty: 70.0,
            stress: 10.0,
            fatigue: 10.0,
            injury: 0.0,
            resolve: 60.0,
            morale: 80.0,
            party_id: None,
            guild_relationship: 50.0,
            leadership_role: None,
            is_player_character: false,
            faction_id: None,
            rallying_to: None,
            tier_status: Default::default(),
            history_tags: Default::default(),
            backstory: None,
            deeds: Vec::new(),
            hobbies: Vec::new(),
            disease_status: crate::headless_campaign::state::DiseaseStatus::Healthy,

            mood_state: crate::headless_campaign::state::MoodState::default(),

            fears: Vec::new(),

            personal_goal: None,

            journal: Vec::new(),

            equipped_items: Vec::new(),
            nicknames: Vec::new(),
            secret_past: None,
            wounds: Vec::new(),
            potion_dependency: 0.0,
            withdrawal_severity: 0.0,
            ticks_since_last_potion: 0,
            total_potions_consumed: 0,
            behavior_ledger: BehaviorLedger::default(),
            classes: Vec::new(),
            skill_state: Default::default(),
            gold: 0.0,
            home_location_id: None,
            economic_intent: Default::default(),
            ticks_since_income: 0,
        }
    }

    #[test]
    fn test_tactical_combat_strong_party_wins() {
        let state = CampaignState::default_test_campaign(42);
        let a1 = test_adventurer(1, "warrior", 5, 25.0, 15.0);
        let a2 = test_adventurer(2, "mage", 5, 20.0, 10.0);
        let a3 = test_adventurer(3, "cleric", 5, 15.0, 12.0);
        let members: Vec<&Adventurer> = vec![&a1, &a2, &a3];

        let result = run_tactical_combat(&members, 20.0, 42, &state);
        // A 3-person party of level 5 heroes should beat threat 20
        assert!(
            result.victory,
            "Strong party should win against low threat: {:?}",
            result
        );
    }

    #[test]
    fn test_tactical_combat_weak_party_loses() {
        let state = CampaignState::default_test_campaign(42);
        let a1 = test_adventurer(1, "warrior", 1, 8.0, 5.0);
        let members: Vec<&Adventurer> = vec![&a1];

        let result = run_tactical_combat(&members, 80.0, 42, &state);
        // A single level 1 warrior should not beat threat 80
        assert!(
            !result.victory,
            "Weak party should lose against high threat: {:?}",
            result
        );
    }

    #[test]
    fn test_tactical_combat_empty_party() {
        let state = CampaignState::default_test_campaign(42);
        let result = run_tactical_combat(&[], 50.0, 42, &state);
        assert!(!result.victory);
        assert_eq!(result.ticks, 0);
    }

    #[test]
    fn test_unlock_power_bonus() {
        let unlocks = vec![
            UnlockInstance {
                id: 1,
                category: UnlockCategory::PassiveBuff,
                properties: UnlockProperties {
                    cooldown_ms: 0,
                    target_type: TargetType::Party,
                    magnitude: 0.1,
                    duration_ms: 0,
                    resource_cost: 0.0,
                    is_passive: true,
                    category_embedding: [0.0; 8],
                },
                name: "Battle Hardened".into(),
                description: "test".into(),
                active: true,
                cooldown_remaining_ms: 0,
            },
        ];
        let bonus = unlock_power_bonus(&unlocks);
        assert!((bonus - 2.0).abs() < 0.01, "0.1 magnitude * 20 = 2.0 power bonus, got {}", bonus);
    }

    #[test]
    fn test_adventurer_to_unit_applies_buffs() {
        let mut state = CampaignState::default_test_campaign(42);
        state.unlocks.push(UnlockInstance {
            id: 1,
            category: UnlockCategory::PassiveBuff,
            properties: UnlockProperties {
                cooldown_ms: 0,
                target_type: TargetType::Party,
                magnitude: 0.5, // +50% attack, +25% HP
                duration_ms: 0,
                resource_cost: 0.0,
                is_passive: true,
                category_embedding: [0.0; 8],
            },
            name: "Mighty Buff".into(),
            description: "test".into(),
            active: true,
            cooldown_remaining_ms: 0,
        });

        let adv = test_adventurer(1, "warrior", 1, 20.0, 10.0);
        let unit = adventurer_to_unit(&adv, 1, sim_vec2(5.0, 5.0), &state);

        // Base attack would be 20, with 50% buff = 30
        // (condition factor also applies but injury=0, fatigue=10 → condition=0.95)
        assert!(
            unit.attack_damage > 20,
            "Unlock buff should increase attack: got {}",
            unit.attack_damage
        );
    }
}
