use std::collections::HashMap;

use crate::ai::core::{SimEvent, SimState};
use crate::game_core::HeroCompanion;

use super::types::CombatJournal;

/// Build a combat journal for a single hero from the mission's event log.
///
/// `hero_unit_id` is the sim unit ID assigned to this hero during the mission.
pub fn build_combat_journal(
    hero: &HeroCompanion,
    hero_unit_id: u32,
    events: &[SimEvent],
    sim: &SimState,
    mission_name: &str,
    outcome: &str,
) -> CombatJournal {
    let mut damage_dealt: i32 = 0;
    let mut damage_taken: i32 = 0;
    let mut heals_given: i32 = 0;
    let mut kills: u32 = 0;
    let mut deaths: u32 = 0;
    let mut abilities: HashMap<String, u32> = HashMap::new();
    let mut passives: HashMap<String, u32> = HashMap::new();
    let mut near_death_moments: u32 = 0;
    let mut cc_applied: u32 = 0;
    let mut shields_given: i32 = 0;
    let mut duels_fought: u32 = 0;

    // Track who last damaged each unit so we can attribute kills.
    let mut last_damager: HashMap<u32, u32> = HashMap::new();

    // Look up max_hp for near-death threshold.
    let hero_max_hp = sim
        .units
        .iter()
        .find(|u| u.id == hero_unit_id)
        .map(|u| u.max_hp)
        .unwrap_or(100);

    for event in events {
        match event {
            SimEvent::DamageApplied {
                source_id,
                target_id,
                amount,
                target_hp_after,
                ..
            } => {
                if *source_id == hero_unit_id {
                    damage_dealt += amount;
                    last_damager.insert(*target_id, *source_id);
                }
                if *target_id == hero_unit_id {
                    damage_taken += amount;
                    // Near-death: HP dropped below 25% but still alive.
                    if *target_hp_after > 0 && *target_hp_after < hero_max_hp / 4 {
                        near_death_moments += 1;
                    }
                }
            }
            SimEvent::HealApplied {
                source_id, amount, ..
            } => {
                if *source_id == hero_unit_id {
                    heals_given += amount;
                }
            }
            SimEvent::UnitDied { unit_id, .. } => {
                if *unit_id == hero_unit_id {
                    deaths += 1;
                } else if last_damager.get(unit_id) == Some(&hero_unit_id) {
                    kills += 1;
                }
            }
            SimEvent::AbilityUsed {
                unit_id,
                ability_name,
                ..
            } => {
                if *unit_id == hero_unit_id {
                    *abilities.entry(ability_name.clone()).or_insert(0) += 1;
                }
            }
            SimEvent::PassiveTriggered {
                unit_id,
                passive_name,
                ..
            } => {
                if *unit_id == hero_unit_id {
                    *passives.entry(passive_name.clone()).or_insert(0) += 1;
                }
            }
            SimEvent::ControlApplied { source_id, .. } => {
                if *source_id == hero_unit_id {
                    cc_applied += 1;
                }
            }
            SimEvent::ShieldApplied {
                unit_id, amount, ..
            } => {
                // Shield given BY hero to anyone (including self).
                // We approximate: if the hero used an ability recently that shields,
                // credit them. For simplicity, credit all shields on hero units
                // if the hero is a support archetype. But the simplest approach:
                // credit shields on the hero's unit_id as "received" isn't right —
                // we want shields the hero *gave*. Since ShieldApplied doesn't
                // have a source_id, we check if the unit is on the hero's team
                // and the hero recently used an ability. For now, count all
                // shields applied to any unit if the hero has used a shield ability.
                // Simplest: just count shields on the hero's own unit.
                if *unit_id == hero_unit_id {
                    shields_given += amount;
                }
            }
            SimEvent::DuelStarted { unit_a, unit_b, .. } => {
                if *unit_a == hero_unit_id || *unit_b == hero_unit_id {
                    duels_fought += 1;
                }
            }
            _ => {}
        }
    }

    let abilities_used: Vec<(String, u32)> = {
        let mut v: Vec<_> = abilities.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    };
    let passives_triggered: Vec<(String, u32)> = {
        let mut v: Vec<_> = passives.into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    };

    CombatJournal {
        hero_id: hero.id,
        hero_name: hero.name.clone(),
        hero_archetype: hero.archetype,
        hero_backstory: hero.backstory.clone(),
        mission_name: mission_name.to_string(),
        outcome: outcome.to_string(),
        damage_dealt,
        damage_taken,
        heals_given,
        kills,
        deaths,
        abilities_used,
        passives_triggered,
        near_death_moments,
        cc_applied,
        shields_given,
        duels_fought,
        loyalty: hero.loyalty,
        stress: hero.stress,
        resolve: hero.resolve,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::core::{SimState, SimVec2, Team, UnitState, UnitStore};
    use crate::game_core::PersonalityArchetype;

    fn make_hero() -> HeroCompanion {
        HeroCompanion {
            id: 1,
            name: "Kael".to_string(),
            origin_faction_id: 0,
            origin_region_id: 0,
            backstory: "A wandering swordsman.".to_string(),
            archetype: PersonalityArchetype::Vanguard,
            loyalty: 60.0,
            stress: 20.0,
            fatigue: 10.0,
            injury: 5.0,
            resolve: 50.0,
            active: true,
            deserter: false,
            xp: 0,
            level: 1,
            equipment: Default::default(),
            traits: Vec::new(),
        }
    }

    fn make_sim() -> SimState {
        SimState {
            tick: 100,
            rng_state: 0,
            units: UnitStore::new(vec![UnitState {
                id: 10,
                team: Team::Hero,
                hp: 50,
                max_hp: 200,
                position: SimVec2 { x: 0.0, y: 0.0 },
                move_speed_per_sec: 2.0,
                attack_damage: 20,
                attack_range: 1.5,
                attack_cooldown_ms: 1000,
                attack_cast_time_ms: 0,
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
                resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
            }]),
            projectiles: Vec::new(),
            passive_trigger_depth: 0,
            zones: Vec::new(),
            tethers: Vec::new(),
            grid_nav: None,
        }
    }

    #[test]
    fn journal_counts_damage_and_kills() {
        let hero = make_hero();
        let sim = make_sim();
        let events = vec![
            SimEvent::DamageApplied {
                tick: 1,
                source_id: 10,
                target_id: 20,
                amount: 30,
                target_hp_before: 100,
                target_hp_after: 70,
            },
            SimEvent::DamageApplied {
                tick: 2,
                source_id: 10,
                target_id: 20,
                amount: 70,
                target_hp_before: 70,
                target_hp_after: 0,
            },
            SimEvent::UnitDied {
                tick: 2,
                unit_id: 20,
            },
            SimEvent::DamageApplied {
                tick: 3,
                source_id: 20,
                target_id: 10,
                amount: 45,
                target_hp_before: 200,
                target_hp_after: 155,
            },
        ];

        let journal = build_combat_journal(&hero, 10, &events, &sim, "Test Mission", "Victory");
        assert_eq!(journal.damage_dealt, 100);
        assert_eq!(journal.damage_taken, 45);
        assert_eq!(journal.kills, 1);
        assert_eq!(journal.deaths, 0);
    }

    #[test]
    fn journal_counts_near_death() {
        let hero = make_hero();
        let sim = make_sim();
        let events = vec![SimEvent::DamageApplied {
            tick: 1,
            source_id: 20,
            target_id: 10,
            amount: 170,
            target_hp_before: 200,
            target_hp_after: 30, // 30 < 200/4=50, alive
        }];

        let journal = build_combat_journal(&hero, 10, &events, &sim, "Test", "Victory");
        assert_eq!(journal.near_death_moments, 1);
    }

    #[test]
    fn journal_counts_abilities() {
        let hero = make_hero();
        let sim = make_sim();
        let events = vec![
            SimEvent::AbilityUsed {
                tick: 1,
                unit_id: 10,
                ability_index: 0,
                ability_name: "Whirlwind".to_string(),
            },
            SimEvent::AbilityUsed {
                tick: 2,
                unit_id: 10,
                ability_index: 0,
                ability_name: "Whirlwind".to_string(),
            },
            SimEvent::AbilityUsed {
                tick: 3,
                unit_id: 10,
                ability_index: 1,
                ability_name: "Shield Bash".to_string(),
            },
        ];

        let journal = build_combat_journal(&hero, 10, &events, &sim, "Test", "Victory");
        assert_eq!(journal.abilities_used.len(), 2);
        assert_eq!(journal.abilities_used[0], ("Whirlwind".to_string(), 2));
        assert_eq!(journal.abilities_used[1], ("Shield Bash".to_string(), 1));
    }
}
