//! Nemesis system — recurring faction champions that grow stronger.
//!
//! When an adventurer dies in battle against a faction, there is a 20% chance
//! a nemesis spawns from that faction. Nemeses grow stronger over time, roam
//! regions where guild parties operate, and engage parties in combat. Defeating
//! a nemesis is a major event granting reputation and morale.
//!
//! Ticks every 200 ticks.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::{
    lcg_f32, lcg_next, AdventurerStatus, BattleStatus, CampaignState, Nemesis, PartyStatus,
};

/// Maximum number of active (undefeated) nemeses at any time.
const MAX_ACTIVE_NEMESES: usize = 3;

/// Distance in tiles for a party to encounter a nemesis.
const ENCOUNTER_DISTANCE: f32 = 3.0;

/// Name templates for generated nemeses.
const NEMESIS_TITLES: &[&str] = &[
    "the Cruel",
    "the Unyielding",
    "Dreadfang",
    "Ironmaw",
    "the Butcher",
    "Shadowbane",
    "the Relentless",
    "Bloodreaver",
    "the Undying",
    "Grimclaw",
    "the Merciless",
    "Bonecrusher",
];

const NEMESIS_FIRST_NAMES: &[&str] = &[
    "Krath", "Vorn", "Sethak", "Morvus", "Dragan", "Helvara", "Zarek", "Thessa", "Ulric",
    "Balgor", "Nyx", "Rathgar",
];

/// Called from quest_lifecycle when an adventurer dies in battle.
/// `faction_id` is the faction that caused the death. May be `None` if
/// the quest had no source faction.
pub fn on_adventurer_killed_by_faction(
    state: &mut CampaignState,
    _adventurer_id: u32,
    faction_id: usize,
    events: &mut Vec<WorldEvent>,
) {
    // Cap active nemeses
    let active_count = state.nemeses.iter().filter(|n| !n.defeated).count();
    if active_count >= MAX_ACTIVE_NEMESES {
        return;
    }

    // 20% chance to spawn a nemesis
    let roll = lcg_f32(&mut state.rng);
    if roll >= 0.2 {
        return;
    }

    // Generate a name
    let first_idx = (lcg_next(&mut state.rng) as usize) % NEMESIS_FIRST_NAMES.len();
    let title_idx = (lcg_next(&mut state.rng) as usize) % NEMESIS_TITLES.len();
    let name = format!("{} {}", NEMESIS_FIRST_NAMES[first_idx], NEMESIS_TITLES[title_idx]);

    // Determine region: use the faction's controlled region if available
    let region_id = state
        .overworld
        .regions
        .iter()
        .find(|r| r.owner_faction_id == faction_id)
        .map(|r| r.id);

    let faction_name = state
        .factions
        .iter()
        .find(|f| f.id == faction_id)
        .map(|f| f.name.clone())
        .unwrap_or_else(|| format!("Faction {}", faction_id));

    let nemesis = Nemesis {
        id: state.next_event_id,
        name: name.clone(),
        faction_id,
        strength: 10.0,
        kills: 1, // they killed the adventurer that spawned them
        created_tick: state.tick,
        region_id,
        defeated: false,
    };
    state.next_event_id += 1;
    state.nemeses.push(nemesis);

    events.push(WorldEvent::NemesisAppeared {
        name,
        faction: faction_name,
    });
}

/// Main tick function. Called every tick; gates internally on tick % 200.
pub fn tick_nemesis(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 {
        return;
    }

    // Collect active nemesis indices for processing
    let active_indices: Vec<usize> = state
        .nemeses
        .iter()
        .enumerate()
        .filter(|(_, n)| !n.defeated)
        .map(|(i, _)| i)
        .collect();

    if active_indices.is_empty() {
        return;
    }

    // --- 1. Grow stronger ---
    for &idx in &active_indices {
        state.nemeses[idx].strength += 1.0;
        events.push(WorldEvent::NemesisGrew {
            name: state.nemeses[idx].name.clone(),
            new_strength: state.nemeses[idx].strength,
        });
    }

    // --- 2. Move toward regions where guild parties operate ---
    // Collect regions where guild parties are active
    let party_regions: Vec<usize> = state
        .parties
        .iter()
        .filter(|p| {
            matches!(
                p.status,
                PartyStatus::Traveling | PartyStatus::OnMission | PartyStatus::Fighting
            )
        })
        .filter_map(|_p| {
            // Find closest region to party position
            state
                .overworld
                .regions
                .iter()
                .min_by(|a, b| {
                    // Use region id as proxy for distance (regions don't have positions,
                    // so use owner faction's territory)
                    a.id.cmp(&b.id)
                })
                .map(|r| r.id)
        })
        .collect();

    for &idx in &active_indices {
        if !party_regions.is_empty() {
            let region_idx = (lcg_next(&mut state.rng) as usize) % party_regions.len();
            state.nemeses[idx].region_id = Some(party_regions[region_idx]);
        }
    }

    // --- 3. Check for encounters with guild parties ---
    // Collect encounter pairs: (nemesis_idx, party_id)
    let mut encounters: Vec<(usize, u32)> = Vec::new();

    for &nem_idx in &active_indices {
        let nem_region = match state.nemeses[nem_idx].region_id {
            Some(r) => r,
            None => continue,
        };

        for party in &state.parties {
            if !matches!(
                party.status,
                PartyStatus::Traveling | PartyStatus::OnMission
            ) {
                continue;
            }

            // Check if party is in the same region or nearby
            // Use region ownership as proximity: party near a region controlled
            // by the nemesis's faction or the nemesis's current region
            let party_region = state
                .overworld
                .regions
                .iter()
                .enumerate()
                .min_by_key(|(_, r)| {
                    let dx = party.position.0 - (r.id as f32 * 10.0);
                    let dy = party.position.1 - (r.id as f32 * 10.0);
                    ((dx * dx + dy * dy) * 100.0) as i64
                })
                .map(|(_, r)| r.id);

            if party_region == Some(nem_region) {
                // Encounter roll — not guaranteed, 40% chance per encounter check
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.4 {
                    encounters.push((nem_idx, party.id));
                    break; // one encounter per nemesis per tick
                }
            }
        }
    }

    // --- 4. Resolve encounters ---
    for (nem_idx, party_id) in encounters {
        let nem_strength = state.nemeses[nem_idx].strength;
        let nem_name = state.nemeses[nem_idx].name.clone();

        // Calculate party power
        let party_power: f32 = state
            .adventurers
            .iter()
            .filter(|a| a.party_id == Some(party_id) && a.status != AdventurerStatus::Dead)
            .map(|a| a.stats.attack + a.stats.defense + a.stats.max_hp / 10.0)
            .sum();

        // Combat resolution: compare party power vs nemesis strength
        let combat_roll = lcg_f32(&mut state.rng);
        let party_advantage = party_power / (party_power + nem_strength);
        let victory = combat_roll < party_advantage;

        if victory {
            // --- Nemesis defeated ---
            state.nemeses[nem_idx].defeated = true;

            // +15 reputation
            state.guild.reputation = (state.guild.reputation + 15.0).min(100.0);

            // +20 morale to all adventurers
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.morale = (adv.morale + 20.0).min(100.0);
                }
            }

            // Find the adventurer who "defeated" the nemesis (highest attack in party)
            let slayer_id = state
                .adventurers
                .iter()
                .filter(|a| a.party_id == Some(party_id) && a.status != AdventurerStatus::Dead)
                .max_by(|a, b| a.stats.attack.partial_cmp(&b.stats.attack).unwrap_or(std::cmp::Ordering::Equal))
                .map(|a| a.id)
                .unwrap_or(0);

            events.push(WorldEvent::NemesisDefeated {
                name: nem_name,
                adventurer_id: slayer_id,
            });

            // Party takes some damage from the fight
            for adv in &mut state.adventurers {
                if adv.party_id == Some(party_id) && adv.status != AdventurerStatus::Dead {
                    let dmg = nem_strength * 0.3;
                    adv.injury = (adv.injury + dmg).min(100.0);
                }
            }
        } else {
            // --- Nemesis wins ---
            state.nemeses[nem_idx].strength += 5.0;
            state.nemeses[nem_idx].kills += 1;

            // Party takes heavy damage
            for adv in &mut state.adventurers {
                if adv.party_id == Some(party_id) && adv.status != AdventurerStatus::Dead {
                    let dmg = nem_strength * 0.6;
                    adv.injury = (adv.injury + dmg).min(100.0);
                    adv.morale = (adv.morale - 15.0).max(0.0);
                }
            }

            // Morale penalty to all non-dead adventurers (fear effect)
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.morale = (adv.morale - 5.0).max(0.0);
                }
            }

            // Create a battle event for tracking
            let battle_id = state.next_battle_id;
            state.next_battle_id += 1;

            state.active_battles.push(crate::state::BattleState {
                id: battle_id,
                quest_id: 0, // no quest — nemesis encounter
                party_id,
                location: state
                    .parties
                    .iter()
                    .find(|p| p.id == party_id)
                    .map(|p| p.position)
                    .unwrap_or((0.0, 0.0)),
                party_health_ratio: 0.3,
                enemy_health_ratio: 0.0,
                enemy_strength: nem_strength,
                elapsed_ticks: 1,
                predicted_outcome: -0.5,
                status: BattleStatus::Defeat,
                runner_sent: false,
                mercenary_hired: false,
                rescue_called: false,
            });

            events.push(WorldEvent::BattleEnded {
                battle_id,
                result: BattleStatus::Defeat,
            });
        }
    }
}
