//! Treasure hunt system — multi-step treasure map quests.
//!
//! Maps are found as loot on quest completion (5% chance for threat>40 quests)
//! or purchased from merchants. Each map has 2-4 steps across different regions,
//! with escalating rewards culminating in a major final reward.
//!
//! Fires every 200 ticks. Active hunts advance when the assigned party reaches
//! the target region. Maps expire after 5000 ticks if not started.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often the treasure hunt system ticks.
const TICK_INTERVAL: u64 = 7;

/// Maximum number of active (started) treasure hunts.
const MAX_ACTIVE_HUNTS: usize = 3;

/// Ticks before an unstarted map expires.
const MAP_EXPIRY_TICKS: u64 = 167;

/// Chance of a treasure map dropping on quest completion (threat > 40).
const MAP_DROP_CHANCE: f32 = 0.05;

/// Distance threshold (squared) for a party to be considered "in" a region.
/// We compute a region center from its locations and check against this.
const REGION_PROXIMITY_SQ: f32 = 400.0;

// ---------------------------------------------------------------------------
// Name tables for map generation
// ---------------------------------------------------------------------------

const COMBAT_MAP_NAMES: &[&str] = &[
    "Blood-Stained Map",
    "Warlord's Treasure Chart",
    "Fallen Champion's Map",
    "Siege Master's Hoard Map",
    "Battlefield Relic Map",
];

const RUIN_MAP_NAMES: &[&str] = &[
    "Ancient Cartograph",
    "Faded Ruin Scroll",
    "Explorer's Lost Chart",
    "Cartographer's Final Map",
    "Dusty Expedition Journal",
];

const COMBAT_CLUES: &[&str] = &[
    "Follow the trail of broken shields",
    "Seek the cairn of fallen soldiers",
    "The treasure lies beneath scorched earth",
    "Where ravens circle, dig deep",
    "Find the rusted blade that marks the spot",
    "The old watchtower's shadow points the way",
];

const RUIN_CLUES: &[&str] = &[
    "The ancient glyph reveals the path",
    "Where roots crack stone, look below",
    "Follow the mosaic to its broken edge",
    "The starlight through the dome shows the vault",
    "Read the inscription on the fallen column",
    "The underground river leads to riches",
];

const ARTIFACT_NAMES: &[&str] = &[
    "Crown of the First King",
    "Starforged Amulet",
    "Blade of the Ancients",
    "Orb of the Deep",
    "Scepter of Ages",
    "Dragonscale Codex",
    "Voidheart Gem",
    "Titan's Signet",
];

// ---------------------------------------------------------------------------
// Public tick entry point
// ---------------------------------------------------------------------------

/// Advance treasure hunts. Called every tick; gates internally on `TICK_INTERVAL`.
pub fn tick_treasure_hunts(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire unstarted maps ---
    expire_old_maps(state, events);

    // --- Advance active hunts ---
    advance_active_hunts(state, events);
}

// ---------------------------------------------------------------------------
// Map generation (called from quest completion)
// ---------------------------------------------------------------------------

/// Attempt to generate a treasure map from a completed quest.
///
/// Called by quest lifecycle when a quest completes with victory.
/// `quest_type` determines the map theme: combat quests produce combat-themed
/// maps, exploration/gather produce ruin-themed maps.
pub fn maybe_generate_map_from_quest(
    state: &mut CampaignState,
    quest_type: QuestType,
    threat_level: f32,
    events: &mut Vec<WorldEvent>,
) {
    if threat_level < 40.0 {
        return;
    }

    let roll = lcg_f32(&mut state.rng);
    if roll > MAP_DROP_CHANCE {
        return;
    }

    let is_combat = matches!(quest_type, QuestType::Combat | QuestType::Rescue | QuestType::Escort);
    generate_map(state, is_combat, threat_level, events);
}

/// Generate a treasure map (called from quest completion or merchant purchase).
fn generate_map(
    state: &mut CampaignState,
    is_combat_themed: bool,
    quality_hint: f32,
    events: &mut Vec<WorldEvent>,
) {
    let num_regions = state.overworld.regions.len();
    if num_regions < 2 {
        return; // Need at least 2 regions for a multi-step hunt
    }

    let map_id = state.next_treasure_map_id;
    state.next_treasure_map_id += 1;

    // 2-4 steps
    let num_steps = 2 + (lcg_next(&mut state.rng) % 3) as usize; // 2, 3, or 4
    let num_steps = num_steps.min(num_regions); // Can't exceed number of regions

    // Pick distinct regions for each step
    let mut used_regions: Vec<usize> = Vec::new();
    let mut steps: Vec<TreasureStep> = Vec::new();

    let clue_table = if is_combat_themed { COMBAT_CLUES } else { RUIN_CLUES };

    let mut total_reward = 0.0f32;

    for i in 0..num_steps {
        // Pick a region not yet used
        let mut region_id = lcg_next(&mut state.rng) as usize % num_regions;
        let mut attempts = 0;
        while used_regions.contains(&region_id) && attempts < 20 {
            region_id = lcg_next(&mut state.rng) as usize % num_regions;
            attempts += 1;
        }
        if used_regions.contains(&region_id) {
            // Fallback: find any unused region
            if let Some(r) = (0..num_regions).find(|r| !used_regions.contains(r)) {
                region_id = r;
            } else {
                break; // No more unique regions
            }
        }
        used_regions.push(region_id);

        let clue_idx = lcg_next(&mut state.rng) as usize % clue_table.len();
        let clue = clue_table[clue_idx].to_string();

        // Escalating rewards: later steps give more
        let step_multiplier = (i as f32 + 1.0) / num_steps as f32;
        let base_reward = 30.0 + quality_hint * 0.5;
        let reward = base_reward * step_multiplier;
        total_reward += reward;

        steps.push(TreasureStep {
            region_id,
            clue,
            completed: false,
            reward,
        });
    }

    // Final step gets a large bonus
    if let Some(last) = steps.last_mut() {
        let final_bonus = 100.0 + quality_hint * 2.0;
        last.reward += final_bonus;
        total_reward += final_bonus;
    }

    // Pick map name
    let names = if is_combat_themed { COMBAT_MAP_NAMES } else { RUIN_MAP_NAMES };
    let name_idx = lcg_next(&mut state.rng) as usize % names.len();
    let name = names[name_idx].to_string();

    let map = TreasureMap {
        id: map_id,
        name: name.clone(),
        steps,
        current_step: 0,
        found_tick: state.tick,
        assigned_party_id: None,
        total_reward,
        is_combat_themed,
    };

    state.treasure_maps.push(map);

    events.push(WorldEvent::TreasureMapFound {
        map_id,
        name,
        num_steps: num_steps as u32,
    });
}

// ---------------------------------------------------------------------------
// Start treasure hunt action
// ---------------------------------------------------------------------------

/// Start a treasure hunt by assigning a party to follow a map.
///
/// Returns an error string if invalid, or None on success.
pub fn start_treasure_hunt(
    state: &mut CampaignState,
    map_id: u32,
    party_id: u32,
    events: &mut Vec<WorldEvent>,
) -> Result<String, String> {
    // Count active hunts
    let active_count = state.treasure_maps.iter().filter(|m| m.assigned_party_id.is_some()).count();
    if active_count >= MAX_ACTIVE_HUNTS {
        return Err("Maximum active treasure hunts reached (3)".into());
    }

    let map = state.treasure_maps.iter().find(|m| m.id == map_id);
    match map {
        None => return Err(format!("Treasure map {} not found", map_id)),
        Some(m) if m.assigned_party_id.is_some() => {
            return Err("Treasure map already has an assigned party".into())
        }
        Some(m) if m.current_step >= m.steps.len() => {
            return Err("Treasure hunt already completed".into())
        }
        _ => {}
    }

    // Verify party exists and is idle or returning
    let party = state.parties.iter().find(|p| p.id == party_id);
    match party {
        None => return Err(format!("Party {} not found", party_id)),
        Some(p) if !matches!(p.status, PartyStatus::Idle | PartyStatus::Returning) => {
            return Err("Party is not available (must be idle or returning)".into())
        }
        _ => {}
    }

    // Assign party and set destination to first step's region
    let map = state.treasure_maps.iter_mut().find(|m| m.id == map_id).unwrap();
    map.assigned_party_id = Some(party_id);

    let first_region_id = map.steps[map.current_step].region_id;
    let map_name = map.name.clone();

    // Compute target position from region
    let target_pos = region_center(state, first_region_id);

    if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
        party.destination = Some(target_pos);
        party.status = PartyStatus::Traveling;
    }

    events.push(WorldEvent::CampaignMilestone {
        description: format!(
            "Party {} begins treasure hunt: {}",
            party_id, map_name
        ),
    });

    Ok(format!("Started treasure hunt '{}' with party {}", map_name, party_id))
}

// ---------------------------------------------------------------------------
// Internal: advance active hunts
// ---------------------------------------------------------------------------

fn advance_active_hunts(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
) {
    // Collect hunt data to avoid borrow issues
    let hunt_data: Vec<(usize, u32, u32, usize, usize)> = state
        .treasure_maps
        .iter()
        .enumerate()
        .filter_map(|(idx, map)| {
            let party_id = map.assigned_party_id?;
            if map.current_step >= map.steps.len() {
                return None; // Already completed
            }
            let target_region = map.steps[map.current_step].region_id;
            Some((idx, map.id, party_id, map.current_step, target_region))
        })
        .collect();

    for (map_idx, map_id, party_id, step_idx, target_region_id) in hunt_data {
        // Check if party is in the target region
        let party_pos = match state.parties.iter().find(|p| p.id == party_id) {
            Some(p) => p.position,
            None => {
                // Party disbanded — cancel hunt
                if map_idx < state.treasure_maps.len() {
                    state.treasure_maps[map_idx].assigned_party_id = None;
                }
                continue;
            }
        };

        let target_pos = region_center(state, target_region_id);
        let dx = party_pos.0 - target_pos.0;
        let dy = party_pos.1 - target_pos.1;
        let dist_sq = dx * dx + dy * dy;

        if dist_sq > REGION_PROXIMITY_SQ {
            continue; // Party not there yet
        }

        // --- Complete the current step ---
        let map = &mut state.treasure_maps[map_idx];
        map.steps[step_idx].completed = true;
        let step_reward = map.steps[step_idx].reward;
        let step_clue = map.steps[step_idx].clue.clone();
        let map_name = map.name.clone();
        let is_final = step_idx + 1 >= map.steps.len();

        // Award gold
        state.guild.gold += step_reward;

        // Award XP to party members
        let member_ids: Vec<u32> = state
            .parties
            .iter()
            .find(|p| p.id == party_id)
            .map(|p| p.member_ids.clone())
            .unwrap_or_default();

        let xp_per_member = (step_reward * 0.5) as u32;
        for &mid in &member_ids {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == mid) {
                adv.xp += xp_per_member;
            }
        }

        events.push(WorldEvent::TreasureStepCompleted {
            map_id,
            step_index: step_idx as u32,
            clue: step_clue,
            reward: step_reward,
        });
        events.push(WorldEvent::GoldChanged {
            amount: step_reward,
            reason: format!("Treasure hunt '{}' step {}", map_name, step_idx + 1),
        });

        if is_final {
            // --- Hunt completed ---
            // Generate artifact item as final reward
            let artifact_name = {
                let idx = lcg_next(&mut state.rng) as usize % ARTIFACT_NAMES.len();
                ARTIFACT_NAMES[idx].to_string()
            };

            let item_id = lcg_next(&mut state.rng);
            let artifact = InventoryItem {
                id: item_id,
                name: artifact_name.clone(),
                slot: EquipmentSlot::Accessory,
                quality: 0.9 + lcg_f32(&mut state.rng) * 0.1, // 0.9-1.0
                stat_bonuses: StatBonuses {
                    hp_bonus: 30.0,
                    attack_bonus: 15.0,
                    defense_bonus: 15.0,
                    speed_bonus: 10.0,
                },
                durability: 100.0,
                appraised: false,
            };
            state.guild.inventory.push(artifact);

            // Add bonus supplies
            state.guild.supplies += 25.0;

            let total_reward = state.treasure_maps[map_idx].total_reward;
            state.treasure_maps[map_idx].assigned_party_id = None; // Mark as done

            events.push(WorldEvent::TreasureHuntCompleted {
                map_id,
                total_reward,
                artifact_name,
            });
            events.push(WorldEvent::SupplyChanged {
                amount: 25.0,
                reason: format!("Treasure hunt '{}' completion", map_name),
            });

            // Boost reputation
            state.guild.reputation = (state.guild.reputation + 5.0).min(100.0);
        } else {
            // Advance to next step
            let map = &mut state.treasure_maps[map_idx];
            map.current_step = step_idx + 1;
            let next_region = map.steps[map.current_step].region_id;
            let next_clue = map.steps[map.current_step].clue.clone();

            // Redirect party to next region
            let next_pos = region_center(state, next_region);
            if let Some(party) = state.parties.iter_mut().find(|p| p.id == party_id) {
                party.destination = Some(next_pos);
                party.status = PartyStatus::Traveling;
            }

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "Treasure hunt '{}': next clue — \"{}\"",
                    map_name, next_clue
                ),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Internal: expire old maps
// ---------------------------------------------------------------------------

fn expire_old_maps(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    state.treasure_maps.retain(|map| {
        // Keep if started (has assigned party) or not yet expired
        map.assigned_party_id.is_some()
            || map.current_step > 0
            || state.tick.saturating_sub(map.found_tick) < MAP_EXPIRY_TICKS
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a rough center position for a region.
///
/// Uses the average position of all locations owned by the region's faction.
/// Falls back to a deterministic position derived from the region index.
fn region_center(state: &CampaignState, region_id: usize) -> (f32, f32) {
    let region = state.overworld.regions.get(region_id);
    let faction_id = region.map(|r| r.owner_faction_id);

    if let Some(fid) = faction_id {
        let locs: Vec<(f32, f32)> = state
            .overworld
            .locations
            .iter()
            .filter(|l| l.faction_owner == Some(fid))
            .map(|l| l.position)
            .collect();

        if !locs.is_empty() {
            let sum_x: f32 = locs.iter().map(|p| p.0).sum();
            let sum_y: f32 = locs.iter().map(|p| p.1).sum();
            return (sum_x / locs.len() as f32, sum_y / locs.len() as f32);
        }
    }

    // Fallback: spread regions evenly across the map
    let angle = (region_id as f32) * 2.0 * std::f32::consts::PI / 6.0;
    (50.0 + 30.0 * angle.cos(), 50.0 + 30.0 * angle.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_generation_deterministic() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        let mut events = Vec::new();

        // Force a map generation
        generate_map(&mut state, true, 60.0, &mut events);

        assert_eq!(state.treasure_maps.len(), 1);
        let map = &state.treasure_maps[0];
        assert!(map.steps.len() >= 2 && map.steps.len() <= 4);
        assert!(map.total_reward > 0.0);
        assert!(map.assigned_party_id.is_none());
        assert!(events.iter().any(|e| matches!(e, WorldEvent::TreasureMapFound { .. })));

        // Same seed should produce the same map
        let mut state2 = CampaignState::default_test_campaign(42);
        state2.phase = CampaignPhase::Playing;
        let mut events2 = Vec::new();
        generate_map(&mut state2, true, 60.0, &mut events2);

        assert_eq!(state.treasure_maps[0].name, state2.treasure_maps[0].name);
        assert_eq!(state.treasure_maps[0].steps.len(), state2.treasure_maps[0].steps.len());
    }

    #[test]
    fn test_map_expiry() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        let mut events = Vec::new();

        generate_map(&mut state, false, 50.0, &mut events);
        assert_eq!(state.treasure_maps.len(), 1);

        // Advance tick past expiry
        state.tick = MAP_EXPIRY_TICKS + 1;
        expire_old_maps(&mut state, &mut events);
        assert_eq!(state.treasure_maps.len(), 0, "Unstarted map should expire");
    }

    #[test]
    fn test_map_not_expired_when_started() {
        let mut state = CampaignState::default_test_campaign(42);
        state.phase = CampaignPhase::Playing;
        let mut events = Vec::new();

        generate_map(&mut state, false, 50.0, &mut events);
        state.treasure_maps[0].assigned_party_id = Some(1);

        state.tick = MAP_EXPIRY_TICKS + 1;
        expire_old_maps(&mut state, &mut events);
        assert_eq!(state.treasure_maps.len(), 1, "Started map should not expire");
    }

    #[test]
    fn test_escalating_rewards() {
        let mut state = CampaignState::default_test_campaign(123);
        state.phase = CampaignPhase::Playing;
        let mut events = Vec::new();

        generate_map(&mut state, true, 80.0, &mut events);
        let map = &state.treasure_maps[0];

        if map.steps.len() >= 2 {
            // Last step should have the biggest reward (includes final bonus)
            let last_reward = map.steps.last().unwrap().reward;
            let first_reward = map.steps[0].reward;
            assert!(
                last_reward > first_reward,
                "Final step reward ({}) should exceed first step reward ({})",
                last_reward, first_reward
            );
        }
    }
}
