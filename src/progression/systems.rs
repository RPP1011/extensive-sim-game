use bevy::prelude::*;

use crate::ai::core::{SimEvent, Team, UnitStore};
use crate::game_core::{
    AssignedHero, CampaignEventLog, CampaignRoster, EquipmentItem, HeroCompanion,
    HeroTrait, ItemRarity, MissionBoard, MissionData, MissionProgress, MissionResult,
    RunState, push_campaign_event,
};
use crate::mission::sim_bridge::{MissionEventLog, MissionSimState, SimEventBuffer};

use super::journal::build_combat_journal;
use super::lfm_bridge::spawn_lfm_progression_request;
use super::types::*;

// ---------------------------------------------------------------------------
// 1. Dispatch: after mission consequences are recorded, build journals and
//    spawn LFM subprocess requests.
// ---------------------------------------------------------------------------

pub fn dispatch_narrative_progression_system(
    run_state: Res<RunState>,
    mission_query: Query<(&MissionData, &MissionProgress, &AssignedHero)>,
    board: Res<MissionBoard>,
    roster: Res<CampaignRoster>,
    event_log: Option<Res<MissionEventLog>>,
    sim_state: Option<Res<MissionSimState>>,
    mut progression: ResMut<NarrativeProgressionState>,
) {
    if run_state.global_turn == 0 {
        return;
    }

    // Only process when there are newly-recorded outcomes.
    for &entity in &board.entities {
        let Ok((data, progress, assigned)) = mission_query.get(entity) else {
            continue;
        };
        // We want missions that just had their outcome recorded this turn.
        // outcome_recorded is set by resolve_mission_consequences_system.
        // We detect "just recorded" by checking outcome_recorded && result != InProgress
        // and ensuring we haven't already dispatched for this hero+turn.
        if !progress.outcome_recorded || progress.result == MissionResult::InProgress {
            continue;
        }

        let Some(hero_id) = assigned.hero_id.or_else(|| {
            // After consequence system, hero_id may already be cleared.
            // Check if any in-flight request already covers this.
            None
        }) else {
            continue;
        };

        // Skip if we already have an in-flight request for this hero.
        if progression
            .in_flight
            .iter()
            .any(|(id, _)| *id == hero_id)
        {
            continue;
        }
        // Skip if we already have a pending reward for this hero from this turn.
        if progression
            .pending
            .iter()
            .any(|p| p.hero_id == hero_id && p.source_turn == run_state.global_turn)
        {
            continue;
        }

        let Some(hero) = roster.heroes.iter().find(|h| h.id == hero_id) else {
            continue;
        };

        let events = event_log.as_ref().map(|log| log.all_events.as_slice()).unwrap_or(&[]);
        let sim_ref = sim_state.as_ref().map(|s| &s.sim);

        // We need a SimState for max_hp lookup. If not available, create a minimal one.
        let empty_sim = crate::ai::core::SimState {
            tick: 0,
            rng_state: 0,
            units: UnitStore::new(Vec::new()),
            projectiles: Vec::new(),
            passive_trigger_depth: 0,
            zones: Vec::new(),
            tethers: Vec::new(),
            grid_nav: None,
        };
        let sim = sim_ref.unwrap_or(&empty_sim);

        // The hero's sim unit ID: in the real mission, hero units typically
        // use IDs 1..N for heroes. We scan the sim for hero-team units.
        // If the sim is empty (mission already cleaned up), we use hero_id as fallback.
        let hero_unit_id = sim
            .units
            .iter()
            .find(|u| u.team == Team::Hero && u.id == hero_id)
            .map(|u| u.id)
            .unwrap_or(hero_id);

        let outcome_str = match progress.result {
            MissionResult::Victory => "Victory",
            MissionResult::Defeat => "Defeat",
            MissionResult::InProgress => "InProgress",
        };

        let journal = build_combat_journal(
            hero,
            hero_unit_id,
            events,
            sim,
            &data.mission_name,
            outcome_str,
        );

        let journal_json = match serde_json::to_string(&journal) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("[progression] Failed to serialize journal: {e}");
                continue;
            }
        };

        let handle = spawn_lfm_progression_request(journal_json, hero_id);
        progression.in_flight.push((hero_id, handle));
        eprintln!(
            "[progression] Dispatched LFM request for hero {} ({})",
            hero.name, hero_id
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Collect: poll in-flight LFM results and create pending progressions.
// ---------------------------------------------------------------------------

pub fn collect_narrative_progression_system(
    run_state: Res<RunState>,
    mut progression: ResMut<NarrativeProgressionState>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    let mut completed_indices = Vec::new();
    let mut new_pendings = Vec::new();

    for (i, (hero_id, shared)) in progression.in_flight.iter().enumerate() {
        let maybe_result = if let Ok(mut slot) = shared.lock() {
            slot.take()
        } else {
            None
        };

        let Some(result) = maybe_result else {
            continue;
        };

        completed_indices.push(i);

        if !result.success {
            eprintln!(
                "[progression] LFM failed for hero {}: {:?}",
                hero_id, result.error
            );
            continue;
        }

        match serde_json::from_str::<ProgressionReward>(&result.reward_json) {
            Ok(reward) => {
                eprintln!(
                    "[progression] Reward queued for hero {}: {}",
                    hero_id,
                    result.narrative_text.chars().take(80).collect::<String>()
                );
                if let Some(ref mut log) = event_log {
                    push_campaign_event(
                        log,
                        run_state.global_turn,
                        format!(
                            "A revelation awaits hero {} from their trials.",
                            hero_id
                        ),
                    );
                }
                new_pendings.push(PendingProgression {
                    hero_id: *hero_id,
                    reward,
                    narrative_text: result.narrative_text.clone(),
                    source_turn: run_state.global_turn,
                });
            }
            Err(e) => {
                eprintln!(
                    "[progression] Failed to parse reward JSON for hero {}: {e}\nRaw: {}",
                    hero_id,
                    result.reward_json.chars().take(200).collect::<String>()
                );
            }
        }
    }

    // Remove completed entries (iterate in reverse to preserve indices).
    for i in completed_indices.into_iter().rev() {
        progression.in_flight.remove(i);
    }
    progression.pending.extend(new_pendings);
}

// ---------------------------------------------------------------------------
// 3. Apply: when a hero goes unconscious during a mission, grant pending rewards.
// ---------------------------------------------------------------------------

pub fn apply_progression_on_unconscious_system(
    event_buf: Res<SimEventBuffer>,
    mut sim_state: Option<ResMut<MissionSimState>>,
    mut roster: ResMut<CampaignRoster>,
    mut progression: ResMut<NarrativeProgressionState>,
    run_state: Res<RunState>,
    mut event_log: Option<ResMut<CampaignEventLog>>,
) {
    if progression.pending.is_empty() {
        return;
    }

    // Find hero UnitDied events this frame.
    let mut hero_deaths: Vec<u32> = Vec::new();
    for event in &event_buf.events {
        if let SimEvent::UnitDied { unit_id, .. } = event {
            // Check if this unit is on Team::Hero in the sim.
            if let Some(ref sim) = sim_state {
                if sim
                    .sim
                    .units
                    .iter()
                    .any(|u| u.id == *unit_id && u.team == Team::Hero)
                {
                    hero_deaths.push(*unit_id);
                }
            }
        }
    }

    if hero_deaths.is_empty() {
        return;
    }

    // For each dead hero unit, check if we have a pending reward.
    // The mapping from sim unit_id to hero_id: we check the roster for
    // heroes whose id matches the unit_id (common convention), or we
    // look in pending for any hero_id that matches.
    let mut applied_indices = Vec::new();

    for dead_unit_id in &hero_deaths {
        // Find pending rewards where hero_id matches the dead unit.
        // In mission setup, hero unit IDs typically match hero.id.
        for (idx, pending) in progression.pending.iter().enumerate() {
            if pending.hero_id == *dead_unit_id && !applied_indices.contains(&idx) {
                applied_indices.push(idx);

                // Apply the reward to the hero's persistent state.
                if let Some(hero) = roster.heroes.iter_mut().find(|h| h.id == pending.hero_id) {
                    apply_reward_to_hero(hero, &pending.reward);
                    eprintln!(
                        "[progression] REVELATION: {} receives reward on unconscious! {}",
                        hero.name,
                        pending.narrative_text.chars().take(80).collect::<String>()
                    );
                }

                // Apply to live sim unit (revive/buff).
                if let Some(ref mut sim) = sim_state {
                    apply_reward_to_sim_unit(&mut sim.sim, *dead_unit_id, &pending.reward);
                }

                if let Some(ref mut log) = event_log {
                    push_campaign_event(
                        log,
                        run_state.global_turn,
                        format!(
                            "Revelation! {}",
                            pending.narrative_text.chars().take(100).collect::<String>()
                        ),
                    );
                }
            }
        }
    }

    // Remove applied pendings (reverse order).
    applied_indices.sort_unstable();
    for idx in applied_indices.into_iter().rev() {
        progression.pending.remove(idx);
    }
}

// ---------------------------------------------------------------------------
// Reward application helpers
// ---------------------------------------------------------------------------

fn apply_reward_to_hero(hero: &mut HeroCompanion, reward: &ProgressionReward) {
    match reward {
        ProgressionReward::Equipment {
            name,
            slot,
            rarity,
            attack_bonus,
            hp_bonus,
            speed_bonus,
            cooldown_mult,
            ..
        } => {
            let item_rarity = if rarity.to_lowercase().contains("rare") {
                ItemRarity::Rare
            } else {
                ItemRarity::Standard
            };
            let item = EquipmentItem {
                name: name.clone(),
                rarity: item_rarity,
                attack_bonus: *attack_bonus,
                hp_bonus: *hp_bonus,
                speed_bonus: *speed_bonus,
                cooldown_mult: *cooldown_mult,
            };
            // Equip to the specified slot (or first empty).
            match slot.to_lowercase().as_str() {
                "weapon" => hero.equipment.weapon = Some(item),
                "offhand" => hero.equipment.offhand = Some(item),
                "chest" => hero.equipment.chest = Some(item),
                "boots" => hero.equipment.boots = Some(item),
                "accessory" => hero.equipment.accessory = Some(item),
                _ => {
                    // Find first empty slot.
                    if hero.equipment.weapon.is_none() {
                        hero.equipment.weapon = Some(item);
                    } else if hero.equipment.offhand.is_none() {
                        hero.equipment.offhand = Some(item);
                    } else if hero.equipment.chest.is_none() {
                        hero.equipment.chest = Some(item);
                    } else if hero.equipment.boots.is_none() {
                        hero.equipment.boots = Some(item);
                    } else if hero.equipment.accessory.is_none() {
                        hero.equipment.accessory = Some(item);
                    }
                    // If all slots full, replace accessory.
                    else {
                        hero.equipment.accessory = Some(item);
                    }
                }
            }
        }
        ProgressionReward::StatBoost {
            stat, amount, ..
        } => {
            match stat.to_lowercase().as_str() {
                "loyalty" => hero.loyalty = (hero.loyalty + amount).clamp(0.0, 100.0),
                "stress" => hero.stress = (hero.stress + amount).clamp(0.0, 100.0),
                "resolve" => hero.resolve += amount,
                "fatigue" => hero.fatigue = (hero.fatigue + amount).clamp(0.0, 100.0),
                "injury" => hero.injury = (hero.injury + amount).clamp(0.0, 100.0),
                "xp" => hero.xp += *amount as u32,
                _ => {
                    // Default: apply to resolve.
                    hero.resolve += amount;
                }
            }
        }
        ProgressionReward::Trait {
            name, description, ..
        } => {
            hero.traits.push(HeroTrait {
                name: name.clone(),
                description: description.clone(),
                passive_toml: None,
            });
        }
        ProgressionReward::Ability { .. } => {
            // Ability rewards are applied to the live sim unit, not persistent hero state.
            // The TOML is stored but not directly on HeroCompanion in this version.
        }
        ProgressionReward::Quest { .. } => {
            // Quest rewards would create a CompanionQuest. For now, log it.
            // Full quest integration deferred to avoid coupling complexity.
        }
    }
}

fn apply_reward_to_sim_unit(
    sim: &mut crate::ai::core::SimState,
    unit_id: u32,
    reward: &ProgressionReward,
) {
    let Some(unit) = sim.units.iter_mut().find(|u| u.id == unit_id) else {
        return;
    };

    // Revive the unit with a portion of HP as the "revelation" mechanic.
    if unit.hp <= 0 {
        let revive_hp = unit.max_hp / 3;
        unit.hp = revive_hp.max(1);
    }

    match reward {
        ProgressionReward::Equipment {
            attack_bonus,
            hp_bonus,
            speed_bonus,
            cooldown_mult,
            ..
        } => {
            unit.attack_damage += attack_bonus;
            unit.max_hp += hp_bonus;
            unit.hp += hp_bonus;
            unit.move_speed_per_sec += speed_bonus;
            if *cooldown_mult > 0.0 && *cooldown_mult < 1.0 {
                unit.attack_cooldown_ms =
                    (unit.attack_cooldown_ms as f32 * cooldown_mult) as u32;
            }
        }
        ProgressionReward::StatBoost { stat, amount, .. } => {
            match stat.to_lowercase().as_str() {
                "hp" | "max_hp" => {
                    let bonus = *amount as i32;
                    unit.max_hp += bonus;
                    unit.hp += bonus;
                }
                "attack" | "damage" => {
                    unit.attack_damage += *amount as i32;
                }
                "speed" | "move_speed" => {
                    unit.move_speed_per_sec += amount;
                }
                _ => {
                    // Emotional stats don't affect sim unit directly.
                }
            }
        }
        ProgressionReward::Ability { toml_content, .. } => {
            // Try to parse TOML as an ability and add it.
            if let Ok(hero_toml) = crate::mission::hero_templates::parse_hero_toml(toml_content) {
                for ability_def in &hero_toml.abilities {
                    unit.abilities
                        .push(crate::ai::effects::AbilitySlot::new(ability_def.clone()));
                }
                for passive_def in &hero_toml.passives {
                    unit.passives
                        .push(crate::ai::effects::PassiveSlot::new(passive_def.clone()));
                }
            }
        }
        ProgressionReward::Trait { passive_toml, .. } => {
            if let Some(toml_content) = passive_toml {
                if let Ok(hero_toml) =
                    crate::mission::hero_templates::parse_hero_toml(toml_content)
                {
                    for passive_def in &hero_toml.passives {
                        unit.passives
                            .push(crate::ai::effects::PassiveSlot::new(passive_def.clone()));
                    }
                }
            }
        }
        ProgressionReward::Quest { .. } => {
            // Quest rewards give an item — handled in persistent state.
        }
    }
}
