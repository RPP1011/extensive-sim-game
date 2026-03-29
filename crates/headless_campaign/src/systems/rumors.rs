//! Tavern rumors and intelligence gathering system.
//!
//! Returning parties and NPCs bring information fragments that reveal
//! hidden quests and predict crises. Fires every 300 ticks (~30s game time).

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often to check for new rumors (in ticks).
const RUMOR_INTERVAL: u64 = 10;

/// Chance per returning party to bring a rumor (30%).
const RUMOR_CHANCE: f32 = 0.30;

/// Rumors expire after this many ticks if not acted on.
const RUMOR_EXPIRY_TICKS: u64 = 67;

/// Maximum active (unrevealed) rumors at once.
const MAX_ACTIVE_RUMORS: usize = 5;

/// Check for new rumors from returning parties and expire old ones.
pub fn tick_rumors(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % RUMOR_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire old rumors ---
    state.rumors.retain(|r| {
        if r.revealed {
            return true; // keep investigated rumors
        }
        state.tick.saturating_sub(r.source_tick) < RUMOR_EXPIRY_TICKS
    });

    // --- Generate rumors from returning parties ---
    // A party that just returned (status == Idle, was recently traveling/on mission)
    // is detected by checking parties with Idle status at guild base position.
    // We use a simpler heuristic: any idle party that exists can source rumors.
    let returning_party_ids: Vec<u32> = state
        .parties
        .iter()
        .filter(|p| p.status == PartyStatus::Idle)
        .map(|p| p.id)
        .collect();

    for party_id in &returning_party_ids {
        // Cap active rumors
        let active_count = state.rumors.iter().filter(|r| !r.revealed).count();
        if active_count >= MAX_ACTIVE_RUMORS {
            break;
        }

        // Roll for rumor
        let roll = lcg_f32(&mut state.rng);
        if roll > RUMOR_CHANCE {
            continue;
        }

        // Compute accuracy from party's average level and region visibility
        let party = match state.parties.iter().find(|p| p.id == *party_id) {
            Some(p) => p,
            None => continue,
        };

        let avg_level = compute_party_avg_level(state, party);
        let level_factor = (avg_level as f32 / 10.0).clamp(0.2, 1.0);

        // Pick a rumor type
        let type_roll = lcg_next(&mut state.rng) % 6;
        let rumor_type = match type_roll {
            0 => RumorType::HiddenQuest,
            1 => RumorType::CrisisWarning,
            2 => RumorType::FactionPlan,
            3 => RumorType::TreasureLocation,
            4 => RumorType::AmbushThreat,
            _ => RumorType::TradeOpportunity,
        };

        // Pick target region/faction
        let (target_region_id, target_faction_id, region_vis) =
            pick_rumor_targets(state, &rumor_type);

        // Accuracy scales with level and visibility of target region
        let vis_factor = region_vis.clamp(0.1, 1.0);
        let accuracy = (level_factor * 0.6 + vis_factor * 0.4).clamp(0.0, 1.0);

        let text = generate_rumor_text(state, &rumor_type, target_region_id, target_faction_id);

        let rumor_id = state.next_event_id;
        state.next_event_id += 1;

        let rumor = Rumor {
            id: rumor_id,
            text: text.clone(),
            rumor_type,
            accuracy,
            source_tick: state.tick,
            revealed: false,
            target_region_id,
            target_faction_id,
        };

        let type_name = rumor_type_name(&rumor.rumor_type);

        state.rumors.push(rumor);

        events.push(WorldEvent::RumorReceived {
            text,
            rumor_type: type_name,
        });
    }

    // --- Prune to max if somehow over limit (oldest unrevealed first) ---
    while state.rumors.iter().filter(|r| !r.revealed).count() > MAX_ACTIVE_RUMORS {
        // Remove oldest unrevealed
        if let Some(pos) = state.rumors.iter().position(|r| !r.revealed) {
            state.rumors.remove(pos);
        } else {
            break;
        }
    }
}

/// Investigate a rumor: mark as revealed and apply outcome.
/// Returns an outcome description string.
pub fn investigate_rumor(
    state: &mut CampaignState,
    rumor_id: u32,
    events: &mut Vec<WorldEvent>,
) -> String {
    let rumor = match state.rumors.iter_mut().find(|r| r.id == rumor_id) {
        Some(r) => r,
        None => return "Rumor not found.".into(),
    };

    if rumor.revealed {
        return "Rumor already investigated.".into();
    }

    rumor.revealed = true;
    let accuracy = rumor.accuracy;
    let rumor_type = rumor.rumor_type;
    let target_region_id = rumor.target_region_id;
    let target_faction_id = rumor.target_faction_id;

    // Roll against accuracy to determine if the rumor was true
    let truth_roll = lcg_f32(&mut state.rng);
    let is_accurate = truth_roll < accuracy;

    let outcome = match rumor_type {
        RumorType::HiddenQuest => {
            if is_accurate {
                // Spawn a bonus quest with better rewards
                spawn_bonus_quest(state, target_region_id, events);
                "Investigation reveals a hidden quest with exceptional rewards!".into()
            } else {
                "The rumored quest turned out to be a dead end.".into()
            }
        }
        RumorType::CrisisWarning => {
            if is_accurate {
                // Boost visibility of target region (advance warning)
                if let Some(rid) = target_region_id {
                    if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == rid) {
                        region.visibility = (region.visibility + 0.3).min(1.0);
                    }
                }
                "The warning proves credible. Your scouts now have better intel on the region."
                    .into()
            } else {
                "The crisis warning was overblown — nothing unusual found.".into()
            }
        }
        RumorType::FactionPlan => {
            if is_accurate {
                // Reveal faction intent by boosting visibility
                if let Some(fid) = target_faction_id {
                    // Boost visibility of all regions owned by this faction
                    let faction_regions: Vec<usize> = state
                        .overworld
                        .regions
                        .iter()
                        .filter(|r| r.owner_faction_id == fid)
                        .map(|r| r.id)
                        .collect();
                    for rid in faction_regions {
                        if let Some(region) =
                            state.overworld.regions.iter_mut().find(|r| r.id == rid)
                        {
                            region.visibility = (region.visibility + 0.2).min(1.0);
                        }
                    }
                }
                "Intelligence confirms the faction's plans. Visibility improved.".into()
            } else {
                "The faction intelligence was outdated — plans have changed.".into()
            }
        }
        RumorType::TreasureLocation => {
            if is_accurate {
                let gold_reward = 80.0 + (lcg_next(&mut state.rng) % 121) as f32; // 80-200
                state.guild.gold += gold_reward;
                events.push(WorldEvent::GoldChanged {
                    amount: gold_reward,
                    reason: "Treasure from investigated rumor".into(),
                });
                format!(
                    "The treasure was real! Your party returns with {:.0} gold.",
                    gold_reward
                )
            } else {
                "The treasure map was a forgery — nothing found.".into()
            }
        }
        RumorType::AmbushThreat => {
            if is_accurate {
                // Reduce threat in target region
                if let Some(rid) = target_region_id {
                    if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == rid) {
                        region.threat_level = (region.threat_level - 15.0).max(0.0);
                    }
                }
                "Your scouts neutralized the ambush threat. Regional danger reduced.".into()
            } else {
                "No ambush was found — the warning was false.".into()
            }
        }
        RumorType::TradeOpportunity => {
            if is_accurate {
                // Temporary gold + supply bonus
                let gold_bonus = 30.0 + (lcg_next(&mut state.rng) % 51) as f32; // 30-80
                let supply_bonus = 15.0 + (lcg_next(&mut state.rng) % 26) as f32; // 15-40
                state.guild.gold += gold_bonus;
                state.guild.supplies += supply_bonus;
                events.push(WorldEvent::GoldChanged {
                    amount: gold_bonus,
                    reason: "Trade opportunity from rumor".into(),
                });
                events.push(WorldEvent::SupplyChanged {
                    amount: supply_bonus,
                    reason: "Trade opportunity from rumor".into(),
                });
                format!(
                    "Trade deal secured! +{:.0} gold, +{:.0} supplies.",
                    gold_bonus, supply_bonus
                )
            } else {
                "The trade opportunity fell through — merchant never arrived.".into()
            }
        }
    };

    events.push(WorldEvent::RumorInvestigated {
        rumor_id,
        outcome: outcome.clone(),
    });

    outcome
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_party_avg_level(state: &CampaignState, party: &Party) -> u32 {
    if party.member_ids.is_empty() {
        return 1;
    }
    let total: u32 = party
        .member_ids
        .iter()
        .filter_map(|id| state.adventurers.iter().find(|a| a.id == *id))
        .map(|a| a.level)
        .sum();
    total / party.member_ids.len().max(1) as u32
}

fn pick_rumor_targets(
    state: &mut CampaignState,
    rumor_type: &RumorType,
) -> (Option<usize>, Option<usize>, f32) {
    let mut region_vis = 0.3_f32;

    let target_region_id = if !state.overworld.regions.is_empty() {
        let idx = (lcg_next(&mut state.rng) as usize) % state.overworld.regions.len();
        let region = &state.overworld.regions[idx];
        region_vis = region.visibility;
        Some(region.id)
    } else {
        None
    };

    let target_faction_id = match rumor_type {
        RumorType::FactionPlan => {
            if !state.factions.is_empty() {
                let idx = (lcg_next(&mut state.rng) as usize) % state.factions.len();
                Some(state.factions[idx].id)
            } else {
                None
            }
        }
        _ => None,
    };

    (target_region_id, target_faction_id, region_vis)
}

fn generate_rumor_text(
    state: &mut CampaignState,
    rumor_type: &RumorType,
    target_region_id: Option<usize>,
    target_faction_id: Option<usize>,
) -> String {
    let region_name = target_region_id
        .and_then(|id| state.overworld.regions.iter().find(|r| r.id == id))
        .map(|r| r.name.clone())
        .unwrap_or_else(|| "an unknown land".into());

    let faction_name = target_faction_id
        .and_then(|id| state.factions.iter().find(|f| f.id == id))
        .map(|f| f.name.clone())
        .unwrap_or_else(|| "a shadowy group".into());

    let templates: &[&str] = match rumor_type {
        RumorType::HiddenQuest => &[
            "Travelers speak of a hidden challenge in {}.",
            "A mysterious patron seeks adventurers to investigate {}.",
            "An old map hints at unexplored ruins near {}.",
        ],
        RumorType::CrisisWarning => &[
            "Dark omens gather over {}. Something stirs.",
            "Refugees from {} warn of growing danger.",
            "A seer predicts calamity will strike {} soon.",
        ],
        RumorType::FactionPlan => &[
            "Spies report {} is mobilizing forces.",
            "Merchants whisper that {} plans a bold move.",
            "{} has been stockpiling weapons — war may come.",
        ],
        RumorType::TreasureLocation => &[
            "A drunk adventurer babbles about treasure buried near {}.",
            "An ancient tome describes a hidden vault in {}.",
            "A dying merchant reveals a cache of gold somewhere in {}.",
        ],
        RumorType::AmbushThreat => &[
            "Bandits are setting up an ambush along the roads of {}.",
            "Scouts report hostile forces massing near {}.",
            "Travelers have gone missing on the way to {}.",
        ],
        RumorType::TradeOpportunity => &[
            "A merchant caravan heading to {} offers favorable terms.",
            "Surplus goods in {} have driven prices down — buy now!",
            "A rare trade route through {} has opened briefly.",
        ],
    };

    let idx = (lcg_next(&mut state.rng) as usize) % templates.len();
    let template = templates[idx];

    // For FactionPlan, substitute faction name; otherwise region name
    match rumor_type {
        RumorType::FactionPlan => template.replacen("{}", &faction_name, 1),
        _ => template.replacen("{}", &region_name, 1),
    }
}

fn spawn_bonus_quest(
    state: &mut CampaignState,
    target_region_id: Option<usize>,
    events: &mut Vec<WorldEvent>,
) {
    let quest_id = state.next_quest_id;
    state.next_quest_id += 1;

    let target_position = target_region_id
        .and_then(|id| state.overworld.regions.iter().find(|r| r.id == id))
        .map(|_| {
            let x = (lcg_next(&mut state.rng) % 100) as f32;
            let y = (lcg_next(&mut state.rng) % 100) as f32;
            (x, y)
        })
        .unwrap_or((50.0, 50.0));

    let distance = ((target_position.0 - state.guild.base.position.0).powi(2)
        + (target_position.1 - state.guild.base.position.1).powi(2))
    .sqrt();

    let quest = QuestRequest {
        id: quest_id,
        source_faction_id: None,
        source_area_id: target_region_id,
        quest_type: QuestType::Exploration,
        threat_level: 30.0 + (lcg_next(&mut state.rng) % 31) as f32, // 30-60
        reward: QuestReward {
            gold: 100.0 + (lcg_next(&mut state.rng) % 151) as f32, // 100-250
            reputation: 10.0,
            relation_faction_id: None,
            relation_change: 0.0,
            supply_reward: 20.0,
            potential_loot: true,
        },
        distance,
        target_position,
        deadline_ms: state.elapsed_ms + 200_000, // generous deadline
        description: "A hidden quest revealed by tavern rumors!".into(),
        arrived_at_ms: state.elapsed_ms,
    };

    events.push(WorldEvent::QuestRequestArrived {
        request_id: quest_id,
        quest_type: QuestType::Exploration,
        threat_level: quest.threat_level,
    });

    state.request_board.push(quest);
}

fn rumor_type_name(rt: &RumorType) -> String {
    match rt {
        RumorType::HiddenQuest => "HiddenQuest".into(),
        RumorType::CrisisWarning => "CrisisWarning".into(),
        RumorType::FactionPlan => "FactionPlan".into(),
        RumorType::TreasureLocation => "TreasureLocation".into(),
        RumorType::AmbushThreat => "AmbushThreat".into(),
        RumorType::TradeOpportunity => "TradeOpportunity".into(),
    }
}
