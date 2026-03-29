//! Seasonal festival system — fires every 500 ticks.
//!
//! Each faction hosts festivals based on the current season. Festivals last
//! 500 ticks and provide economic, diplomatic, and morale effects. The guild
//! can attend festivals of friendly factions for relation bonuses and rewards.
//! Not attending a friendly faction's festival incurs a small relation penalty.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often to check for new festivals and expire old ones (in ticks).
const FESTIVAL_INTERVAL: u64 = 17;

/// How long each festival lasts (in ticks).
const FESTIVAL_DURATION: u64 = 17;

/// Relation boost for attending a festival.
const ATTEND_RELATION_BOOST: f32 = 5.0;

/// Relation penalty for ignoring a friendly faction's festival.
const IGNORE_RELATION_PENALTY: f32 = 2.0;

/// Tick festivals: spawn new ones, apply active effects, expire old ones.
pub fn tick_festivals(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % FESTIVAL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Expire completed festivals ---
    expire_festivals(state, events);

    // --- Spawn new festivals ---
    spawn_festivals(state, events);

    // --- Apply active festival effects ---
    apply_festival_effects(state, events);
}

/// Remove festivals that have exceeded their duration.
/// If the guild didn't attend a friendly faction's festival, apply a relation penalty.
fn expire_festivals(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let tick = state.tick;
    let mut expired = Vec::new();

    for (i, festival) in state.active_festivals.iter().enumerate() {
        if tick >= festival.started_tick + festival.duration {
            expired.push(i);
        }
    }

    // Process in reverse to preserve indices.
    for &i in expired.iter().rev() {
        let festival = state.active_festivals.remove(i);

        // Penalty for not attending a friendly faction's festival.
        if !festival.attended {
            if let Some(faction) = state.factions.iter_mut().find(|f| f.id == festival.faction_id) {
                if faction.relationship_to_guild > 20.0 {
                    faction.relationship_to_guild =
                        (faction.relationship_to_guild - IGNORE_RELATION_PENALTY).max(-100.0);
                    events.push(WorldEvent::FactionRelationChanged {
                        faction_id: festival.faction_id,
                        old: faction.relationship_to_guild + IGNORE_RELATION_PENALTY,
                        new: faction.relationship_to_guild,
                    });
                }
            }
        }
    }
}

/// Spawn new festivals for factions based on season and faction characteristics.
fn spawn_festivals(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let season = state.overworld.season;
    let faction_count = state.factions.len();

    if faction_count == 0 {
        return;
    }

    // Each faction has a chance to host a festival based on season.
    // Use deterministic RNG to decide which factions host and what type.
    let roll = lcg_next(&mut state.rng);
    // Roughly 1-2 factions host per cycle: pick one faction deterministically.
    let host_idx = (roll as usize) % faction_count;
    let host_faction_id = state.factions[host_idx].id;
    let host_faction_name = state.factions[host_idx].name.clone();
    let host_stance = state.factions[host_idx].diplomatic_stance;

    // Don't spawn a festival if this faction already has one active.
    let already_hosting = state
        .active_festivals
        .iter()
        .any(|f| f.faction_id == host_faction_id);
    if already_hosting {
        return;
    }

    // Pick festival type based on season + faction stance.
    let festival_type = pick_festival_type(season, host_stance, &mut state.rng);
    let festival_name = format!(
        "{} {} of {}",
        season_adjective(season),
        festival_type.display_name(),
        host_faction_name
    );

    let festival_id = state.next_event_id;
    state.next_event_id += 1;

    let festival = Festival {
        id: festival_id,
        name: festival_name.clone(),
        faction_id: host_faction_id,
        festival_type,
        started_tick: state.tick,
        duration: FESTIVAL_DURATION,
        attended: false,
    };

    state.active_festivals.push(festival);

    events.push(WorldEvent::FestivalStarted {
        name: festival_name.clone(),
        faction: host_faction_name,
    });

    // Create a choice event for the guild to attend or ignore.
    let choice_id = state.next_event_id;
    state.next_event_id += 1;

    let reward_desc = festival_type.reward_description();

    state.pending_choices.push(ChoiceEvent {
        id: choice_id,
        source: ChoiceSource::WorldEvent,
        prompt: format!(
            "The {} has begun! Will the guild send representatives? {}",
            festival_name, reward_desc
        ),
        options: vec![
            ChoiceOption {
                label: "Attend the festival".to_string(),
                description: format!(
                    "Send representatives to gain +{} faction relation and {}.",
                    ATTEND_RELATION_BOOST as u32, reward_desc
                ),
                effects: vec![
                    ChoiceEffect::FactionRelation {
                        faction_id: host_faction_id,
                        delta: ATTEND_RELATION_BOOST,
                    },
                    ChoiceEffect::Narrative(format!(
                        "The guild attends the {}.",
                        festival_name
                    )),
                    ChoiceEffect::AttendFestival(festival_id),
                ],
            },
            ChoiceOption {
                label: "Decline invitation".to_string(),
                description: "Skip the festival. Friendly factions may take offense.".to_string(),
                effects: vec![ChoiceEffect::Narrative(
                    "The guild declines the invitation.".to_string(),
                )],
            },
        ],
        default_option: 1,
        deadline_ms: Some(state.elapsed_ms + FESTIVAL_DURATION as u64 * CAMPAIGN_TURN_SECS as u64 * 1000),
        created_at_ms: state.elapsed_ms,
    });

    events.push(WorldEvent::ChoicePresented {
        choice_id,
        prompt: format!("{} has begun!", festival_name),
        num_options: 2,
    });
}

/// Apply passive effects from active festivals.
fn apply_festival_effects(state: &mut CampaignState, _events: &mut Vec<WorldEvent>) {
    // Collect festival data to avoid borrow issues.
    let festival_info: Vec<(usize, FestivalType)> = state
        .active_festivals
        .iter()
        .map(|f| (f.faction_id, f.festival_type))
        .collect();

    for (faction_id, festival_type) in &festival_info {
        match festival_type {
            FestivalType::HarvestFeast => {
                // +20% trade income boost: add a small gold bonus for regions
                // owned by this faction.
                let region_count = state
                    .overworld
                    .regions
                    .iter()
                    .filter(|r| r.owner_faction_id == *faction_id)
                    .count();
                // Small per-region gold bonus to represent trade income boost.
                let bonus = region_count as f32 * 0.5;
                if bonus > 0.0 && state.diplomacy.guild_faction_id == *faction_id {
                    state.guild.gold += bonus;
                    state.guild.total_trade_income += bonus;
                }
            }
            FestivalType::WarMemorial => {
                // +10 morale for faction units, applied via adventurers in that faction.
                // For guild adventurers, apply if guild is allied with faction.
                if let Some(faction) = state.factions.iter().find(|f| f.id == *faction_id) {
                    if faction.relationship_to_guild > 40.0 {
                        for adv in &mut state.adventurers {
                            if adv.status != AdventurerStatus::Dead {
                                adv.morale = (adv.morale + 0.02).min(100.0);
                            }
                        }
                    }
                }
            }
            FestivalType::TradeFair => {
                // Improve market prices temporarily: reduce supply multiplier slightly.
                if state.diplomacy.guild_faction_id == *faction_id
                    || state
                        .factions
                        .iter()
                        .any(|f| f.id == *faction_id && f.relationship_to_guild > 30.0)
                {
                    state.guild.market_prices.supply_multiplier =
                        (state.guild.market_prices.supply_multiplier - 0.01).max(0.7);
                }
            }
            FestivalType::ReligiousCeremony => {
                // +10 civilian morale = reduce unrest in faction regions.
                for region in &mut state.overworld.regions {
                    if region.owner_faction_id == *faction_id {
                        region.unrest = (region.unrest - 0.2).max(0.0);
                    }
                }
            }
            FestivalType::ArtisanExpo => {
                // Small chance to find rare materials: guild supply boost.
                if state.diplomacy.guild_faction_id == *faction_id {
                    state.guild.supplies += 0.1;
                }
            }
            FestivalType::MartialTournament => {
                // Adventurer XP trickle for idle adventurers (training opportunities).
                // Only if guild is friendly with the faction.
                if let Some(faction) = state.factions.iter().find(|f| f.id == *faction_id) {
                    if faction.relationship_to_guild > 30.0 {
                        for adv in &mut state.adventurers {
                            if adv.status == AdventurerStatus::Idle {
                                adv.resolve = (adv.resolve + 0.05).min(100.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Pick a festival type based on the current season and faction stance.
fn pick_festival_type(
    season: Season,
    stance: DiplomaticStance,
    rng: &mut u64,
) -> FestivalType {
    // Season-weighted candidates.
    let candidates: &[FestivalType] = match season {
        Season::Autumn => &[
            FestivalType::HarvestFeast,
            FestivalType::HarvestFeast,
            FestivalType::TradeFair,
            FestivalType::ArtisanExpo,
        ],
        Season::Spring => &[
            FestivalType::ReligiousCeremony,
            FestivalType::TradeFair,
            FestivalType::ArtisanExpo,
            FestivalType::MartialTournament,
        ],
        Season::Summer => &[
            FestivalType::MartialTournament,
            FestivalType::MartialTournament,
            FestivalType::TradeFair,
            FestivalType::HarvestFeast,
        ],
        Season::Winter => &[
            FestivalType::WarMemorial,
            FestivalType::ReligiousCeremony,
            FestivalType::ReligiousCeremony,
            FestivalType::ArtisanExpo,
        ],
    };

    // Stance override: hostile/at-war factions lean toward WarMemorial.
    let idx = (lcg_next(rng) as usize) % candidates.len();
    match stance {
        DiplomaticStance::Hostile | DiplomaticStance::AtWar => {
            let roll = lcg_f32(rng);
            if roll < 0.5 {
                FestivalType::WarMemorial
            } else {
                candidates[idx]
            }
        }
        _ => candidates[idx],
    }
}

/// Returns a seasonal adjective for festival naming.
fn season_adjective(season: Season) -> &'static str {
    match season {
        Season::Spring => "Vernal",
        Season::Summer => "Midsummer",
        Season::Autumn => "Harvest",
        Season::Winter => "Midwinter",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::StepDeltas;

    fn make_test_state() -> CampaignState {
        let mut state = CampaignState::default_test_campaign(42);
        // Force into playing state.
        state.phase = CampaignPhase::Playing;
        state.tick = 499; // Will become 500 on next step
        state
    }

    #[test]
    fn festival_spawns_on_interval() {
        let mut state = make_test_state();
        state.tick = 500;
        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        tick_festivals(&mut state, &mut deltas, &mut events);

        // Should have spawned a festival if there are factions.
        if !state.factions.is_empty() {
            assert!(
                !state.active_festivals.is_empty(),
                "Expected a festival to be spawned"
            );
            assert!(
                events.iter().any(|e| matches!(e, WorldEvent::FestivalStarted { .. })),
                "Expected FestivalStarted event"
            );
        }
    }

    #[test]
    fn festival_type_display_names() {
        assert_eq!(FestivalType::HarvestFeast.display_name(), "Harvest Feast");
        assert_eq!(FestivalType::WarMemorial.display_name(), "War Memorial");
        assert_eq!(FestivalType::TradeFair.display_name(), "Trade Fair");
        assert_eq!(
            FestivalType::ReligiousCeremony.display_name(),
            "Religious Ceremony"
        );
        assert_eq!(FestivalType::ArtisanExpo.display_name(), "Artisan Expo");
        assert_eq!(
            FestivalType::MartialTournament.display_name(),
            "Martial Tournament"
        );
    }

    #[test]
    fn festival_expires_after_duration() {
        let mut state = make_test_state();
        state.tick = 500;

        // Manually add an expired festival.
        state.active_festivals.push(Festival {
            id: 99,
            name: "Test Festival".into(),
            faction_id: 0,
            festival_type: FestivalType::TradeFair,
            started_tick: 0,
            duration: 500,
            attended: true,
        });

        let mut deltas = StepDeltas::default();
        let mut events = Vec::new();

        expire_festivals(&mut state, &mut events);

        assert!(
            state.active_festivals.iter().all(|f| f.id != 99),
            "Expired festival should be removed"
        );
    }
}
