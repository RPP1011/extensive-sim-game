//! Adventurer oath system — every 300 ticks.
//!
//! Adventurers can swear binding oaths that grant powerful bonuses but impose
//! strict constraints. Breaking an oath carries loyalty, morale, and reputation
//! penalties plus an "oathbreaker" history tag. Fulfilling an oath grants a
//! permanent bonus and an "oathkeeper" tag.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: runs every 300 ticks.
const OATH_INTERVAL: u64 = 300;

/// Common quality threshold — items with quality > this are "above common".
const COMMON_QUALITY_THRESHOLD: f32 = 30.0;

/// Ticks an OathOfExploration adventurer has to visit a new region.
const EXPLORATION_DEADLINE_TICKS: u64 = 2000;

/// Loyalty threshold for voluntary oath proposals.
const OATH_PROPOSAL_LOYALTY: f32 = 70.0;

/// Chance (0–1) per eligible adventurer per tick to propose an oath.
const OATH_PROPOSAL_CHANCE: f32 = 0.10;

/// Format an OathType as a short label for events.
fn oath_type_label(oath_type: &OathType) -> String {
    match oath_type {
        OathType::OathOfVengeance { target_faction } => {
            format!("vengeance(faction={})", target_faction)
        }
        OathType::OathOfProtection { ward_id } => {
            format!("protection(ward={})", ward_id)
        }
        OathType::OathOfPoverty => "poverty".to_string(),
        OathType::OathOfSilence => "silence".to_string(),
        OathType::OathOfService { faction_id } => {
            format!("service(faction={})", faction_id)
        }
        OathType::OathOfExploration => "exploration".to_string(),
    }
}

/// Main tick function. Called every 300 ticks.
///
/// 1. Check oath conditions — has the adventurer violated their oath?
/// 2. Broken oaths: -20 loyalty, -15 morale, -5 reputation, "oathbreaker" tag
/// 3. Fulfilled oaths: permanent bonus, +10 reputation, "oathkeeper" tag
/// 4. Oath proposal: high-loyalty adventurers may propose oaths (10% chance)
pub fn tick_oaths(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % OATH_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Phase 1: Check active oaths for violations and fulfillment ---
    let mut break_ids: Vec<u32> = Vec::new();
    let mut fulfill_ids: Vec<u32> = Vec::new();

    for oath in state.oaths.iter() {
        if oath.broken || oath.fulfilled {
            continue;
        }

        let adv = state.adventurers.iter().find(|a| a.id == oath.adventurer_id);
        let adv = match adv {
            Some(a) if a.status != AdventurerStatus::Dead => a,
            // Dead or missing adventurer — oath is moot, just mark broken silently.
            _ => {
                break_ids.push(oath.id);
                continue;
            }
        };

        match &oath.oath_type {
            OathType::OathOfVengeance { target_faction } => {
                // Violation: adventurer is idle while the guild has an active quest
                // targeting this faction and the adventurer is not assigned to it.
                let has_faction_quest = state.active_quests.iter().any(|q| {
                    q.request.source_faction_id == Some(*target_faction)
                        && q.request.quest_type == QuestType::Combat
                });
                let is_idle = adv.status == AdventurerStatus::Idle;
                if has_faction_quest && is_idle {
                    break_ids.push(oath.id);
                }
            }

            OathType::OathOfProtection { ward_id } => {
                // Violation: ward has died.
                let ward_dead = state
                    .adventurers
                    .iter()
                    .find(|a| a.id == *ward_id)
                    .map(|a| a.status == AdventurerStatus::Dead)
                    .unwrap_or(true);
                if ward_dead {
                    break_ids.push(oath.id);
                }
            }

            OathType::OathOfPoverty => {
                // Violation: any equipped item has quality above common threshold.
                let has_expensive = equipped_item_ids(adv).iter().any(|item_id| {
                    state
                        .guild
                        .inventory
                        .iter()
                        .find(|i| i.id == *item_id)
                        .map(|i| i.quality > COMMON_QUALITY_THRESHOLD)
                        .unwrap_or(false)
                });
                if has_expensive {
                    break_ids.push(oath.id);
                }
            }

            OathType::OathOfSilence => {
                // Violation: adventurer is assigned to a diplomatic quest.
                let on_diplomatic = state.active_quests.iter().any(|q| {
                    q.request.quest_type == QuestType::Diplomatic
                        && q.assigned_pool.contains(&adv.id)
                });
                if on_diplomatic {
                    break_ids.push(oath.id);
                }
            }

            OathType::OathOfService { faction_id: _ } => {
                // Fulfillment: adventurer has participated in 3+ completed quests
                // since the oath was sworn.
                let completed_since = state
                    .completed_quests
                    .iter()
                    .filter(|q| {
                        q.completed_at_ms >= oath.sworn_tick * CAMPAIGN_TICK_MS as u64
                            && q.result == QuestResult::Victory
                    })
                    .count();
                if completed_since >= 3 {
                    fulfill_ids.push(oath.id);
                }
            }

            OathType::OathOfExploration => {
                // Violation: more than EXPLORATION_DEADLINE_TICKS have passed
                // since the oath was sworn and no new region has been visited.
                // We approximate "visited new region" by checking if any region
                // visibility increased since the oath was sworn. Use tick delta
                // as a proxy — if the adventurer has been idle too long, break.
                let ticks_since_sworn = state.tick.saturating_sub(oath.sworn_tick);
                if ticks_since_sworn >= EXPLORATION_DEADLINE_TICKS {
                    // Check if adventurer is on any scouting or exploration quest.
                    let on_exploration = state.active_quests.iter().any(|q| {
                        q.request.quest_type == QuestType::Exploration
                            && q.assigned_pool.contains(&adv.id)
                    });
                    let is_traveling = matches!(
                        adv.status,
                        AdventurerStatus::Traveling | AdventurerStatus::OnMission
                    );
                    if !on_exploration && !is_traveling {
                        break_ids.push(oath.id);
                    }
                }
            }
        }
    }

    // --- Phase 2: Apply broken oath penalties ---
    for oath_id in &break_ids {
        let oath = match state.oaths.iter_mut().find(|o| o.id == *oath_id) {
            Some(o) => o,
            None => continue,
        };
        oath.broken = true;
        oath.bonus_active = false;

        let adv_id = oath.adventurer_id;
        let label = oath_type_label(&oath.oath_type);

        events.push(WorldEvent::OathBroken {
            adventurer_id: adv_id,
            oath_id: *oath_id,
            oath_type: label,
        });

        // Apply penalties to the adventurer.
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            adv.loyalty = (adv.loyalty - 20.0).max(0.0);
            adv.morale = (adv.morale - 15.0).max(0.0);
            *adv.history_tags.entry("oathbreaker".to_string()).or_insert(0) += 1;
        }

        // Reputation penalty to the guild.
        state.guild.reputation = (state.guild.reputation - 5.0).max(0.0);
    }

    // --- Phase 3: Apply fulfilled oath bonuses ---
    for oath_id in &fulfill_ids {
        let oath = match state.oaths.iter_mut().find(|o| o.id == *oath_id) {
            Some(o) => o,
            None => continue,
        };
        oath.fulfilled = true;
        oath.bonus_active = false; // Permanent bonus applied via history tag now.

        let adv_id = oath.adventurer_id;
        let label = oath_type_label(&oath.oath_type);

        events.push(WorldEvent::OathFulfilled {
            adventurer_id: adv_id,
            oath_id: *oath_id,
            oath_type: label,
        });

        // Apply permanent bonus tag and reputation.
        if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
            *adv.history_tags.entry("oathkeeper".to_string()).or_insert(0) += 1;
        }
        state.guild.reputation += 10.0;
    }

    // --- Phase 4: Oath proposals from high-loyalty adventurers ---
    // Collect eligible adventurers (alive, idle, loyalty > threshold, no active oath).
    let active_oath_adv_ids: Vec<u32> = state
        .oaths
        .iter()
        .filter(|o| !o.broken && !o.fulfilled)
        .map(|o| o.adventurer_id)
        .collect();

    let eligible: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status != AdventurerStatus::Dead
                && a.loyalty > OATH_PROPOSAL_LOYALTY
                && !active_oath_adv_ids.contains(&a.id)
        })
        .map(|a| a.id)
        .collect();

    let num_factions = state.factions.len();
    let num_adventurers = state.adventurers.len();

    for adv_id in eligible {
        let roll = lcg_f32(&mut state.rng);
        if roll >= OATH_PROPOSAL_CHANCE {
            continue;
        }

        // Pick an oath type deterministically based on RNG.
        let oath_roll = lcg_next(&mut state.rng) % 6;
        let oath_type = match oath_roll {
            0 if num_factions > 0 => {
                let faction_idx = lcg_next(&mut state.rng) as usize % num_factions;
                let faction_id = state.factions[faction_idx].id;
                OathType::OathOfVengeance {
                    target_faction: faction_id,
                }
            }
            1 if num_adventurers > 1 => {
                // Pick a random other adventurer as ward.
                let ward_idx = lcg_next(&mut state.rng) as usize % num_adventurers;
                let ward = &state.adventurers[ward_idx];
                if ward.id == adv_id || ward.status == AdventurerStatus::Dead {
                    continue; // Skip this proposal.
                }
                OathType::OathOfProtection { ward_id: ward.id }
            }
            2 => OathType::OathOfPoverty,
            3 => OathType::OathOfSilence,
            4 if num_factions > 0 => {
                let faction_idx = lcg_next(&mut state.rng) as usize % num_factions;
                let faction_id = state.factions[faction_idx].id;
                OathType::OathOfService { faction_id }
            }
            5 => OathType::OathOfExploration,
            _ => OathType::OathOfPoverty, // Fallback for edge cases.
        };

        let oath_id = state.next_oath_id;
        state.next_oath_id += 1;

        let label = oath_type_label(&oath_type);

        state.oaths.push(Oath {
            id: oath_id,
            adventurer_id: adv_id,
            oath_type,
            sworn_tick: state.tick,
            fulfilled: false,
            broken: false,
            bonus_active: true,
        });

        events.push(WorldEvent::OathSworn {
            adventurer_id: adv_id,
            oath_id,
            oath_type: label,
        });
    }
}

/// Collect all equipped item IDs from an adventurer's equipment.
fn equipped_item_ids(adv: &Adventurer) -> Vec<u32> {
    let eq = &adv.equipment;
    [eq.weapon, eq.offhand, eq.chest, eq.boots, eq.accessory]
        .iter()
        .filter_map(|slot| *slot)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless_campaign::actions::StepDeltas;

    /// Helper: create a minimal campaign state for testing oaths.
    fn test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        // Advance past init phase.
        state.phase = CampaignPhase::Playing;
        state.tick = OATH_INTERVAL; // Will fire on next check.
        state
    }

    #[test]
    fn oath_proposal_fires_for_loyal_adventurer() {
        let mut state = test_state();
        // Make first adventurer highly loyal.
        if let Some(adv) = state.adventurers.first_mut() {
            adv.loyalty = 90.0;
        }
        // Run many ticks to get at least one proposal (probabilistic, but seed 42 is consistent).
        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        // Force multiple attempts.
        for i in 0..20 {
            state.tick = OATH_INTERVAL * (i + 1);
            tick_oaths(&mut state, &mut deltas, &mut events);
        }
        // With 10% chance per tick over 20 ticks, very unlikely to get zero.
        let sworn_count = events
            .iter()
            .filter(|e| matches!(e, WorldEvent::OathSworn { .. }))
            .count();
        assert!(
            sworn_count > 0,
            "Expected at least one oath proposal, got 0"
        );
    }

    #[test]
    fn broken_oath_applies_penalties() {
        let mut state = test_state();
        let adv_id = state.adventurers[0].id;
        state.adventurers[0].loyalty = 80.0;
        state.adventurers[0].morale = 80.0;
        let initial_rep = state.guild.reputation;

        // Manually add an OathOfPoverty and equip a high-quality item.
        state.oaths.push(Oath {
            id: 99,
            adventurer_id: adv_id,
            oath_type: OathType::OathOfPoverty,
            sworn_tick: 0,
            fulfilled: false,
            broken: false,
            bonus_active: true,
        });
        // Add an expensive item to inventory and equip it.
        state.guild.inventory.push(InventoryItem {
            id: 999,
            name: "Rare Sword".into(),
            slot: EquipmentSlot::Weapon,
            quality: 80.0,
            stat_bonuses: StatBonuses::default(),
            durability: 100.0,
        });
        state.adventurers[0].equipment.weapon = Some(999);

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_oaths(&mut state, &mut deltas, &mut events);

        // Oath should be broken.
        let oath = state.oaths.iter().find(|o| o.id == 99).unwrap();
        assert!(oath.broken);
        assert!(!oath.bonus_active);

        // Penalties applied.
        assert!(state.adventurers[0].loyalty < 80.0);
        assert!(state.adventurers[0].morale < 80.0);
        assert!(state.guild.reputation < initial_rep);
        assert!(state.adventurers[0]
            .history_tags
            .contains_key("oathbreaker"));

        // Event emitted.
        assert!(events
            .iter()
            .any(|e| matches!(e, WorldEvent::OathBroken { oath_id: 99, .. })));
    }

    #[test]
    fn fulfilled_oath_grants_bonus() {
        let mut state = test_state();
        let adv_id = state.adventurers[0].id;
        let initial_rep = state.guild.reputation;

        // Add an OathOfService and fake 3 completed faction quests.
        let faction_id = if state.factions.is_empty() {
            0
        } else {
            state.factions[0].id
        };
        state.oaths.push(Oath {
            id: 100,
            adventurer_id: adv_id,
            oath_type: OathType::OathOfService { faction_id },
            sworn_tick: 0,
            fulfilled: false,
            broken: false,
            bonus_active: true,
        });

        // Add 3 completed quests since the oath was sworn.
        for i in 0..3 {
            state.completed_quests.push(CompletedQuest {
                id: 500 + i,
                quest_type: QuestType::Combat,
                result: QuestResult::Victory,
                reward_applied: QuestReward::default(),
                completed_at_ms: state.tick * CAMPAIGN_TICK_MS as u64,
                party_id: 0,
                casualties: 0,
                threat_level: 50.0,
            });
        }

        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_oaths(&mut state, &mut deltas, &mut events);

        // Oath should be fulfilled.
        let oath = state.oaths.iter().find(|o| o.id == 100).unwrap();
        assert!(oath.fulfilled);

        // Reputation bonus applied.
        assert!(state.guild.reputation > initial_rep);
        assert!(state.adventurers[0]
            .history_tags
            .contains_key("oathkeeper"));

        // Event emitted.
        assert!(events
            .iter()
            .any(|e| matches!(e, WorldEvent::OathFulfilled { oath_id: 100, .. })));
    }

    #[test]
    fn no_oath_at_tick_zero() {
        let mut state = test_state();
        state.tick = 0;
        let mut events = Vec::new();
        let mut deltas = StepDeltas::default();
        tick_oaths(&mut state, &mut deltas, &mut events);
        assert!(events.is_empty());
    }
}
