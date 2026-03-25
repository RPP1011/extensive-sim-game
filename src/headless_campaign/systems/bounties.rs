//! Bounty board system — factions and NPCs post bounties on specific targets.
//!
//! Fires every 300 ticks. Maintains 2–6 active bounties at a time.
//! Bounty types: monster hunts, faction enemies, nemesis kills, bandit clearance,
//! resource deliveries, and escort missions.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the bounty system ticks (in campaign ticks).
const BOUNTY_INTERVAL: u64 = 10;

/// Maximum active (unclaimed + claimed-but-incomplete) bounties.
const MAX_ACTIVE_BOUNTIES: usize = 6;

/// Minimum active bounties before we try to generate more.
const MIN_ACTIVE_BOUNTIES: usize = 2;

/// Bounty expiry duration in ticks from when it was posted.
const BOUNTY_EXPIRY_TICKS: u64 = 3000;

/// Tick the bounty board system.
///
/// Every `BOUNTY_INTERVAL` ticks:
/// 1. Expire stale bounties.
/// 2. Auto-complete bounties whose conditions are met.
/// 3. Generate new bounties if below the minimum.
pub fn tick_bounties(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % BOUNTY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire old bounties ---
    expire_bounties(state, events);

    // --- Auto-complete check ---
    auto_complete_bounties(state, events);

    // --- Generate new bounties if needed ---
    let active_count = state
        .bounty_board
        .iter()
        .filter(|b| !is_bounty_expired(b, state.tick) && !b.completed)
        .count();

    if active_count < MIN_ACTIVE_BOUNTIES {
        let to_generate = (MIN_ACTIVE_BOUNTIES - active_count)
            .max(1)
            .min(MAX_ACTIVE_BOUNTIES - active_count);
        for _ in 0..to_generate {
            if let Some(bounty) = generate_bounty(state) {
                events.push(WorldEvent::BountyPosted {
                    description: bounty.target_description.clone(),
                    reward: bounty.reward_gold,
                });
                state.bounty_board.push(bounty);
            }
        }
    }
}

/// Check if a bounty has expired based on its posted tick and the current tick.
fn is_bounty_expired(bounty: &Bounty, current_tick: u64) -> bool {
    current_tick >= bounty.expires_tick
}

/// Remove expired unclaimed bounties and emit events.
fn expire_bounties(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let mut expired_descs = Vec::new();

    for bounty in &mut state.bounty_board {
        if !bounty.claimed && !bounty.completed && is_bounty_expired(bounty, tick) {
            expired_descs.push(bounty.target_description.clone());
        }
    }

    for desc in &expired_descs {
        events.push(WorldEvent::BountyExpired {
            description: desc.clone(),
        });
    }

    // Remove expired unclaimed bounties
    state
        .bounty_board
        .retain(|b| b.completed || b.claimed || !is_bounty_expired(b, tick));
}

/// Scan recent game state to auto-complete claimed bounties.
fn auto_complete_bounties(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect completion info first to avoid borrow conflicts
    let mut completions: Vec<(usize, f32, f32)> = Vec::new();

    for (idx, bounty) in state.bounty_board.iter().enumerate() {
        if !bounty.claimed || bounty.completed {
            continue;
        }

        let completed = match &bounty.target_type {
            BountyTarget::MonsterHunt { species: _, count } => {
                // Check if enough combat quests have been completed recently
                let recent_victories = state
                    .completed_quests
                    .iter()
                    .filter(|q| {
                        q.result == QuestResult::Victory
                            && q.completed_at_ms > bounty.posted_tick * CAMPAIGN_TURN_SECS as u64 * 1000
                    })
                    .count() as u32;
                recent_victories >= *count
            }
            BountyTarget::FactionEnemy { faction_id } => {
                // Faction relationship dropped below -50 (defeated or weakened)
                state
                    .factions
                    .iter()
                    .find(|f| f.id == *faction_id)
                    .map(|f| f.military_strength < 20.0)
                    .unwrap_or(false)
            }
            BountyTarget::NemesisKill { nemesis_id } => {
                // The nemesis adventurer is dead
                state
                    .adventurers
                    .iter()
                    .find(|a| a.id == *nemesis_id)
                    .map(|a| a.status == AdventurerStatus::Dead)
                    .unwrap_or(true) // if not found, assume completed
            }
            BountyTarget::BanditClearance { region_id } => {
                // Region unrest below 20
                state
                    .overworld
                    .regions
                    .iter()
                    .find(|r| r.id == *region_id)
                    .map(|r| r.unrest < 20.0)
                    .unwrap_or(false)
            }
            BountyTarget::ResourceDelivery { resource: _, amount } => {
                // Guild has enough supplies (simplified — treats all resources as supplies)
                state.guild.supplies >= *amount
            }
            BountyTarget::EscortMission {
                destination_region: _,
            } => {
                // Check if any quest of type Escort was completed recently
                state.completed_quests.iter().any(|q| {
                    q.quest_type == QuestType::Escort
                        && q.result == QuestResult::Victory
                        && q.completed_at_ms > bounty.posted_tick * CAMPAIGN_TURN_SECS as u64 * 1000
                })
            }
        };

        if completed {
            completions.push((idx, bounty.reward_gold, bounty.reward_reputation));
        }
    }

    // Apply completions in reverse order to preserve indices
    for &(idx, gold, rep) in completions.iter().rev() {
        let bounty = &mut state.bounty_board[idx];
        bounty.completed = true;
        let desc = bounty.target_description.clone();
        let claimer = bounty.claimed_by;

        // Grant rewards
        state.guild.gold += gold;
        state.guild.reputation = (state.guild.reputation + rep).min(100.0);

        events.push(WorldEvent::BountyCompleted {
            description: desc.clone(),
            reward_gold: gold,
        });

        events.push(WorldEvent::GoldChanged {
            amount: gold,
            reason: format!("Bounty completed: {}", desc),
        });

        // If a specific adventurer claimed it, grant XP
        if let Some(adv_id) = claimer {
            if let Some(adv) = state.adventurers.iter_mut().find(|a| a.id == adv_id) {
                adv.xp += 100;
            }
        }
    }
}

/// Generate a single new bounty based on current game state.
fn generate_bounty(state: &mut CampaignState) -> Option<Bounty> {
    // Build weighted candidate pool
    let mut candidates: Vec<(BountyCandidate, f32)> = Vec::new();

    // --- Hostile factions post bounties against their enemies ---
    for faction in &state.factions {
        if faction.relationship_to_guild < -20.0
            || faction.diplomatic_stance == DiplomaticStance::AtWar
        {
            // Post bounty against this hostile faction
            candidates.push((
                BountyCandidate::FactionEnemy {
                    poster_faction_id: None, // guild-posted against hostile
                    target_faction_id: faction.id,
                    faction_name: faction.name.clone(),
                    strength: faction.military_strength,
                },
                12.0,
            ));
        }
    }

    // --- Friendly factions post resource/escort bounties ---
    for faction in &state.factions {
        if faction.relationship_to_guild > 20.0 {
            candidates.push((
                BountyCandidate::ResourceDelivery {
                    poster_faction_id: Some(faction.id),
                    faction_name: faction.name.clone(),
                },
                8.0,
            ));
            candidates.push((
                BountyCandidate::EscortMission {
                    poster_faction_id: Some(faction.id),
                    faction_name: faction.name.clone(),
                },
                6.0,
            ));
        }
    }

    // --- High-threat regions trigger monster hunt bounties ---
    for region in &state.overworld.regions {
        if region.threat_level > 50.0 {
            candidates.push((
                BountyCandidate::MonsterHunt {
                    region_id: region.id,
                    region_name: region.name.clone(),
                    threat: region.threat_level,
                },
                10.0,
            ));
        }
    }

    // --- High-unrest regions trigger bandit clearance ---
    for region in &state.overworld.regions {
        if region.unrest > 40.0 {
            candidates.push((
                BountyCandidate::BanditClearance {
                    region_id: region.id,
                    region_name: region.name.clone(),
                    unrest: region.unrest,
                },
                10.0,
            ));
        }
    }

    // --- Dead/hostile adventurers with faction_id serve as nemesis targets ---
    for adv in &state.adventurers {
        if adv.faction_id.is_some()
            && adv.status != AdventurerStatus::Dead
            && adv.guild_relationship < -30.0
        {
            candidates.push((
                BountyCandidate::NemesisKill {
                    nemesis_id: adv.id,
                    nemesis_name: adv.name.clone(),
                },
                15.0, // Nemeses always weighted high
            ));
        }
    }

    if candidates.is_empty() {
        // Fallback: always offer a generic monster hunt
        candidates.push((
            BountyCandidate::GenericMonsterHunt,
            10.0,
        ));
    }

    // Weighted random selection
    let total_weight: f32 = candidates.iter().map(|(_, w)| w).sum();
    let pick = lcg_f32(&mut state.rng) * total_weight;
    let mut cumulative = 0.0;
    let mut chosen_idx = 0;
    for (i, (_, w)) in candidates.iter().enumerate() {
        cumulative += w;
        if pick < cumulative {
            chosen_idx = i;
            break;
        }
    }

    let bounty_id = state.next_bounty_id;
    state.next_bounty_id += 1;

    let (candidate, _) = &candidates[chosen_idx];
    Some(candidate_to_bounty(candidate, bounty_id, state))
}

/// Internal enum for bounty generation candidates.
#[derive(Clone)]
enum BountyCandidate {
    MonsterHunt {
        region_id: usize,
        region_name: String,
        threat: f32,
    },
    FactionEnemy {
        poster_faction_id: Option<usize>,
        target_faction_id: usize,
        faction_name: String,
        strength: f32,
    },
    NemesisKill {
        nemesis_id: u32,
        nemesis_name: String,
    },
    BanditClearance {
        region_id: usize,
        region_name: String,
        unrest: f32,
    },
    ResourceDelivery {
        poster_faction_id: Option<usize>,
        faction_name: String,
    },
    EscortMission {
        poster_faction_id: Option<usize>,
        faction_name: String,
    },
    GenericMonsterHunt,
}

/// Convert a candidate into a concrete Bounty.
fn candidate_to_bounty(
    candidate: &BountyCandidate,
    bounty_id: u32,
    state: &mut CampaignState,
) -> Bounty {
    let tick = state.tick;

    match candidate {
        BountyCandidate::MonsterHunt {
            region_id,
            region_name,
            threat,
        } => {
            let count = 1 + (lcg_next(&mut state.rng) % 3); // 1–3
            let species_list = ["wolves", "drakes", "ghouls", "trolls", "wyverns"];
            let species_idx = (lcg_next(&mut state.rng) as usize) % species_list.len();
            let species = species_list[species_idx].to_string();
            let reward_gold = 30.0 + threat * 0.5 + count as f32 * 10.0;
            let reward_rep = 3.0 + threat * 0.05;

            Bounty {
                id: bounty_id,
                poster_faction_id: None,
                target_description: format!(
                    "Hunt {} {} in {}",
                    count, species, region_name
                ),
                target_type: BountyTarget::MonsterHunt { species, count },
                reward_gold,
                reward_reputation: reward_rep,
                region_id: Some(*region_id),
                posted_tick: tick,
                expires_tick: tick + BOUNTY_EXPIRY_TICKS,
                claimed: false,
                claimed_by: None,
                completed: false,
            }
        }

        BountyCandidate::FactionEnemy {
            poster_faction_id,
            target_faction_id,
            faction_name,
            strength,
        } => {
            let reward_gold = 50.0 + strength * 0.8;
            let reward_rep = 5.0 + strength * 0.1;

            Bounty {
                id: bounty_id,
                poster_faction_id: *poster_faction_id,
                target_description: format!("Weaken the {}", faction_name),
                target_type: BountyTarget::FactionEnemy {
                    faction_id: *target_faction_id,
                },
                reward_gold,
                reward_reputation: reward_rep,
                region_id: None,
                posted_tick: tick,
                expires_tick: tick + BOUNTY_EXPIRY_TICKS,
                claimed: false,
                claimed_by: None,
                completed: false,
            }
        }

        BountyCandidate::NemesisKill {
            nemesis_id,
            nemesis_name,
        } => {
            let reward_gold = 80.0 + (lcg_next(&mut state.rng) % 41) as f32; // 80–120
            let reward_rep = 8.0;

            Bounty {
                id: bounty_id,
                poster_faction_id: None,
                target_description: format!("Eliminate {}", nemesis_name),
                target_type: BountyTarget::NemesisKill {
                    nemesis_id: *nemesis_id,
                },
                reward_gold,
                reward_reputation: reward_rep,
                region_id: None,
                posted_tick: tick,
                expires_tick: tick + BOUNTY_EXPIRY_TICKS,
                claimed: false,
                claimed_by: None,
                completed: false,
            }
        }

        BountyCandidate::BanditClearance {
            region_id,
            region_name,
            unrest,
        } => {
            let reward_gold = 25.0 + unrest * 0.5;
            let reward_rep = 4.0 + unrest * 0.05;

            Bounty {
                id: bounty_id,
                poster_faction_id: None,
                target_description: format!("Clear bandits from {}", region_name),
                target_type: BountyTarget::BanditClearance {
                    region_id: *region_id,
                },
                reward_gold,
                reward_reputation: reward_rep,
                region_id: Some(*region_id),
                posted_tick: tick,
                expires_tick: tick + BOUNTY_EXPIRY_TICKS,
                claimed: false,
                claimed_by: None,
                completed: false,
            }
        }

        BountyCandidate::ResourceDelivery {
            poster_faction_id,
            faction_name,
        } => {
            let amount = 20.0 + (lcg_next(&mut state.rng) % 31) as f32; // 20–50
            let resource_list = ["iron", "timber", "herbs", "grain", "stone"];
            let res_idx = (lcg_next(&mut state.rng) as usize) % resource_list.len();
            let resource = resource_list[res_idx].to_string();
            let reward_gold = 20.0 + amount * 0.8;
            let reward_rep = 2.0;

            Bounty {
                id: bounty_id,
                poster_faction_id: *poster_faction_id,
                target_description: format!(
                    "Deliver {:.0} {} to the {}",
                    amount, resource, faction_name
                ),
                target_type: BountyTarget::ResourceDelivery { resource, amount },
                reward_gold,
                reward_reputation: reward_rep,
                region_id: None,
                posted_tick: tick,
                expires_tick: tick + BOUNTY_EXPIRY_TICKS,
                claimed: false,
                claimed_by: None,
                completed: false,
            }
        }

        BountyCandidate::EscortMission {
            poster_faction_id,
            faction_name,
        } => {
            let dest_region = if !state.overworld.regions.is_empty() {
                let idx = (lcg_next(&mut state.rng) as usize) % state.overworld.regions.len();
                state.overworld.regions[idx].id
            } else {
                0
            };
            let reward_gold = 40.0 + (lcg_next(&mut state.rng) % 31) as f32; // 40–70
            let reward_rep = 3.0;

            Bounty {
                id: bounty_id,
                poster_faction_id: *poster_faction_id,
                target_description: format!(
                    "Escort {} caravan to safety",
                    faction_name
                ),
                target_type: BountyTarget::EscortMission {
                    destination_region: dest_region,
                },
                reward_gold,
                reward_reputation: reward_rep,
                region_id: Some(dest_region),
                posted_tick: tick,
                expires_tick: tick + BOUNTY_EXPIRY_TICKS,
                claimed: false,
                claimed_by: None,
                completed: false,
            }
        }

        BountyCandidate::GenericMonsterHunt => {
            let count = 1 + (lcg_next(&mut state.rng) % 2); // 1–2
            let species = "beasts".to_string();
            let reward_gold = 25.0 + count as f32 * 8.0;
            let reward_rep = 2.0;

            let region_id = if !state.overworld.regions.is_empty() {
                let idx = (lcg_next(&mut state.rng) as usize) % state.overworld.regions.len();
                Some(state.overworld.regions[idx].id)
            } else {
                None
            };

            Bounty {
                id: bounty_id,
                poster_faction_id: None,
                target_description: format!("Hunt {} {}", count, species),
                target_type: BountyTarget::MonsterHunt { species, count },
                reward_gold,
                reward_reputation: reward_rep,
                region_id,
                posted_tick: tick,
                expires_tick: tick + BOUNTY_EXPIRY_TICKS,
                claimed: false,
                claimed_by: None,
                completed: false,
            }
        }
    }
}
