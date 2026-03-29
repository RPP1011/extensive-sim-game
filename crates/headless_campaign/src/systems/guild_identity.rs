//! Guild identity/specialization system — fires every 500 ticks.
//!
//! The guild develops a distinct identity based on its actions across five axes:
//! martial, mercantile, scholarly, diplomatic, shadowy. Each ranges 0–100 and
//! decays toward 10 (neutral) when unexercised.
//!
//! When the highest axis exceeds 50, the guild acquires a dominant identity
//! type (WarriorsGuild, MerchantCompany, etc.) which unlocks unique bonuses
//! and attracts matching recruits. Opposing identities (martial↔diplomatic,
//! mercantile↔shadowy) create internal tension that slightly suppresses both.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often the identity system ticks (in ticks).
const IDENTITY_INTERVAL: u64 = 17;

/// Influence gain per matching action detected in the window.
const ACTION_INFLUENCE_GAIN: f32 = 2.0;

/// Passive decay rate toward neutral each identity tick.
const DECAY_RATE: f32 = 0.5;

/// Neutral resting value for all identity axes.
const NEUTRAL: f32 = 10.0;

/// Threshold for an axis to be considered dominant.
const DOMINANT_THRESHOLD: f32 = 50.0;

/// Tension suppression applied to opposing identity pairs each tick.
const TENSION_SUPPRESSION: f32 = 0.3;

pub fn tick_guild_identity(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % IDENTITY_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let old_dominant = state.guild_identity.dominant;
    let window_start_ms =
        state.elapsed_ms.saturating_sub(IDENTITY_INTERVAL * CAMPAIGN_TURN_SECS as u64 * 1000);

    // --- Phase 1: Tally identity-relevant actions in the window ---
    let mut martial: f32 = 0.0;
    let mut mercantile: f32 = 0.0;
    let mut scholarly: f32 = 0.0;
    let mut diplomatic: f32 = 0.0;
    let mut shadowy: f32 = 0.0;

    // Completed quests in window
    for quest in &state.completed_quests {
        if quest.completed_at_ms < window_start_ms {
            continue;
        }
        if quest.result != QuestResult::Victory {
            continue;
        }
        match quest.quest_type {
            QuestType::Combat | QuestType::Rescue => martial += ACTION_INFLUENCE_GAIN,
            QuestType::Escort | QuestType::Gather => mercantile += ACTION_INFLUENCE_GAIN,
            QuestType::Exploration => scholarly += ACTION_INFLUENCE_GAIN,
            QuestType::Diplomatic => diplomatic += ACTION_INFLUENCE_GAIN,
        }
    }

    // Active battles contribute to martial
    martial += state.active_battles.len() as f32 * ACTION_INFLUENCE_GAIN * 0.5;

    // Active trade routes / caravans contribute to mercantile
    mercantile += state.trade_routes.len() as f32 * ACTION_INFLUENCE_GAIN * 0.3;
    mercantile += state.caravans.len() as f32 * ACTION_INFLUENCE_GAIN * 0.2;

    // Active festivals contribute to diplomatic
    diplomatic += state.active_festivals.len() as f32 * ACTION_INFLUENCE_GAIN * 0.5;

    // Active spies / black market deals contribute to shadowy
    shadowy += state.spies.len() as f32 * ACTION_INFLUENCE_GAIN * 0.5;
    if state.black_market.heat > 20.0 {
        shadowy += ACTION_INFLUENCE_GAIN;
    }

    // Dungeon exploration contributes to scholarly
    for dungeon in &state.dungeons {
        if dungeon.explored > 0.0 && dungeon.explored < 1.0 {
            scholarly += ACTION_INFLUENCE_GAIN * 0.3;
        }
    }

    // --- Phase 2: Apply gains, decay, and tension ---
    let id = &mut state.guild_identity;

    // Apply gains
    id.martial_identity = (id.martial_identity + martial).min(100.0);
    id.mercantile_identity = (id.mercantile_identity + mercantile).min(100.0);
    id.scholarly_identity = (id.scholarly_identity + scholarly).min(100.0);
    id.diplomatic_identity = (id.diplomatic_identity + diplomatic).min(100.0);
    id.shadowy_identity = (id.shadowy_identity + shadowy).min(100.0);

    // Decay toward neutral for axes that had no activity
    if martial == 0.0 {
        id.martial_identity = decay_toward(id.martial_identity, NEUTRAL, DECAY_RATE);
    }
    if mercantile == 0.0 {
        id.mercantile_identity = decay_toward(id.mercantile_identity, NEUTRAL, DECAY_RATE);
    }
    if scholarly == 0.0 {
        id.scholarly_identity = decay_toward(id.scholarly_identity, NEUTRAL, DECAY_RATE);
    }
    if diplomatic == 0.0 {
        id.diplomatic_identity = decay_toward(id.diplomatic_identity, NEUTRAL, DECAY_RATE);
    }
    if shadowy == 0.0 {
        id.shadowy_identity = decay_toward(id.shadowy_identity, NEUTRAL, DECAY_RATE);
    }

    // Opposing identity tension: martial↔diplomatic, mercantile↔shadowy
    if id.martial_identity > DOMINANT_THRESHOLD && id.diplomatic_identity > DOMINANT_THRESHOLD {
        id.martial_identity -= TENSION_SUPPRESSION;
        id.diplomatic_identity -= TENSION_SUPPRESSION;
    }
    if id.mercantile_identity > DOMINANT_THRESHOLD && id.shadowy_identity > DOMINANT_THRESHOLD {
        id.mercantile_identity -= TENSION_SUPPRESSION;
        id.shadowy_identity -= TENSION_SUPPRESSION;
    }

    // Clamp all to [0, 100]
    id.martial_identity = id.martial_identity.clamp(0.0, 100.0);
    id.mercantile_identity = id.mercantile_identity.clamp(0.0, 100.0);
    id.scholarly_identity = id.scholarly_identity.clamp(0.0, 100.0);
    id.diplomatic_identity = id.diplomatic_identity.clamp(0.0, 100.0);
    id.shadowy_identity = id.shadowy_identity.clamp(0.0, 100.0);

    // --- Phase 3: Determine dominant identity ---
    let axes = [
        (IdentityType::WarriorsGuild, id.martial_identity),
        (IdentityType::MerchantCompany, id.mercantile_identity),
        (IdentityType::ScholarOrder, id.scholarly_identity),
        (IdentityType::DiplomaticCorps, id.diplomatic_identity),
        (IdentityType::ShadowNetwork, id.shadowy_identity),
    ];

    let best = axes
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    id.dominant = if best.1 > DOMINANT_THRESHOLD {
        Some(best.0)
    } else {
        None
    };

    // --- Phase 4: Emit events ---
    if id.dominant != old_dominant {
        events.push(WorldEvent::IdentityShift {
            old: old_dominant.map(|i| format!("{:?}", i)),
            new: id.dominant.map(|i| format!("{:?}", i)),
        });

        // Emit bonus unlocked when gaining a new dominant identity
        if let Some(new_id) = id.dominant {
            let bonus = match new_id {
                IdentityType::WarriorsGuild => "+15% combat power, military contracts",
                IdentityType::MerchantCompany => "+20% trade income, exclusive deals",
                IdentityType::ScholarOrder => "+20% XP gain, research topics",
                IdentityType::DiplomaticCorps => "+15% relation gains, peace missions",
                IdentityType::ShadowNetwork => "+20% spy effectiveness, black market discount",
            };
            events.push(WorldEvent::IdentityBonusUnlocked {
                identity: format!("{:?}", new_id),
                bonus: bonus.to_string(),
            });
        }
    }

    // --- Phase 5: Attract matching recruits ---
    // Deterministic chance of attracting a recruit matching the dominant identity.
    if let Some(dominant) = id.dominant {
        let mut rng = state.rng;
        let roll = lcg_f32(&mut rng);
        state.rng = rng;

        // ~20% chance per identity tick (every 500 ticks) when dominant
        if roll < 0.20 {
            let recruit_type = match dominant {
                IdentityType::WarriorsGuild => "warrior",
                IdentityType::MerchantCompany => "merchant",
                IdentityType::ScholarOrder => "scholar",
                IdentityType::DiplomaticCorps => "diplomat",
                IdentityType::ShadowNetwork => "rogue",
            };
            events.push(WorldEvent::IdentityRecruitAttracted {
                identity: format!("{:?}", dominant),
                recruit_type: recruit_type.to_string(),
            });
        }
    }

    // --- Phase 6: Apply identity bonuses to game state ---
    apply_identity_bonuses(state);
}

/// Decay a value toward `target` by `rate`.
fn decay_toward(value: f32, target: f32, rate: f32) -> f32 {
    if value > target {
        (value - rate).max(target)
    } else if value < target {
        (value + rate).min(target)
    } else {
        value
    }
}

/// Apply passive bonuses based on the guild's dominant identity.
fn apply_identity_bonuses(state: &mut CampaignState) {
    let dominant = match state.guild_identity.dominant {
        Some(d) => d,
        None => return,
    };

    match dominant {
        IdentityType::MerchantCompany => {
            // +20% trade income: small gold bonus per trade route
            let trade_bonus = state.trade_routes.len() as f32 * 0.2;
            state.guild.gold += trade_bonus;
        }
        IdentityType::ScholarOrder => {
            // +20% XP: small xp bump for living adventurers
            let xp_bump = 1;
            for adv in &mut state.adventurers {
                if adv.status != AdventurerStatus::Dead {
                    adv.xp += xp_bump;
                }
            }
        }
        IdentityType::DiplomaticCorps => {
            // +15% relation gains: small faction relation improvement
            for faction in &mut state.factions {
                if faction.relationship_to_guild > 0.0 && faction.relationship_to_guild < 100.0 {
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild + 0.15).min(100.0);
                }
            }
        }
        IdentityType::ShadowNetwork => {
            // Black market heat decays faster (discount effect)
            if state.black_market.heat > 0.0 {
                state.black_market.heat = (state.black_market.heat - 1.0).max(0.0);
            }
        }
        IdentityType::WarriorsGuild => {
            // Combat bonus is applied passively by other systems reading guild_identity.
            // No direct mutation needed here.
        }
    }
}

/// Public helper: returns the combat power multiplier from guild identity.
/// Called by battle systems to apply the WarriorsGuild bonus.
pub fn identity_combat_bonus(state: &CampaignState) -> f32 {
    if state.guild_identity.dominant == Some(IdentityType::WarriorsGuild) {
        0.15
    } else {
        0.0
    }
}

/// Public helper: returns the spy effectiveness bonus from guild identity.
/// Called by espionage systems to apply the ShadowNetwork bonus.
pub fn identity_spy_bonus(state: &CampaignState) -> f32 {
    if state.guild_identity.dominant == Some(IdentityType::ShadowNetwork) {
        0.20
    } else {
        0.0
    }
}
