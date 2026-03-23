//! Endgame crisis system — supports multiple simultaneous crises.
//!
//! Each crisis type has its own escalation mechanics.
//! Multiple crises can be active at once — the world can face a
//! Sleeping King AND a dungeon breach simultaneously.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Tick all active crises. Runs every tick.
pub fn tick_crisis(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.overworld.active_crises.is_empty() {
        return;
    }

    // Process each crisis — collect updates to avoid borrow issues
    let crises = state.overworld.active_crises.clone();
    let mut updated_crises = Vec::new();

    for crisis in crises {
        let updated = match crisis {
            ActiveCrisis::SleepingKing { .. } => {
                tick_sleeping_king(state, events, crisis)
            }
            ActiveCrisis::Breach { .. } => {
                tick_breach(state, events, crisis)
            }
            ActiveCrisis::Corruption { .. } => {
                tick_corruption(state, events, crisis)
            }
            ActiveCrisis::Decline { .. } => {
                tick_decline(state, events, crisis)
            }
            ActiveCrisis::Unifier { .. } => {
                tick_unifier(state, events, crisis)
            }
        };
        if let Some(c) = updated {
            updated_crises.push(c);
        }
        // None = crisis resolved/removed
    }

    state.overworld.active_crises = updated_crises;
}

/// Activate a crisis from a CalamityType. Called by threat.rs when
/// campaign progress crosses the threshold.
pub fn activate_crisis(
    state: &mut CampaignState,
    calamity: &CalamityType,
    events: &mut Vec<WorldEvent>,
) {
    // Don't duplicate crisis types
    let already_active = state.overworld.active_crises.iter().any(|c| {
        matches!(
            (c, calamity),
            (ActiveCrisis::SleepingKing { .. }, CalamityType::AggressiveFaction { .. })
            | (ActiveCrisis::Breach { .. }, CalamityType::MajorMonster { .. })
            | (ActiveCrisis::Corruption { .. }, CalamityType::CrisisFlood)
            | (ActiveCrisis::Decline { .. }, CalamityType::Conquest)
        )
    });
    if already_active {
        return;
    }

    match calamity {
        CalamityType::AggressiveFaction { faction_id } => {
            // Spawn 7 champion adventurers scattered across the map
            let mut champion_ids = Vec::new();
            let champion_names = [
                ("Sera the Strategist", "knight", 7, LeadershipBuff::AttackMultiplier(1.5)),
                ("Vorn the Smith", "knight", 6, LeadershipBuff::DefenseBonus(20.0)),
                ("Kira Shadowstep", "rogue", 8, LeadershipBuff::GoldIncome(5.0)),
                ("Brother Aldous", "cleric", 6, LeadershipBuff::RecoveryRate(2.0)),
                ("Mira Farsight", "ranger", 7, LeadershipBuff::RecruitBonus(0.5)),
                ("Dax Ironwall", "knight", 9, LeadershipBuff::MilitaryStrength(50.0)),
                ("The Whisperer", "mage", 8, LeadershipBuff::DiplomacyBonus(20.0)),
            ];

            for (i, (name, archetype, level, buff)) in champion_names.iter().enumerate() {
                let id = 9000 + i as u32;
                let pos = (
                    (i as f32 * 17.0 - 50.0),
                    (i as f32 * 13.0 - 40.0),
                );

                let champ = Adventurer {
                    id,
                    name: name.to_string(),
                    archetype: archetype.to_string(),
                    level: *level,
                    xp: 0,
                    stats: AdventurerStats {
                        max_hp: 100.0 + *level as f32 * 15.0,
                        attack: 10.0 + *level as f32 * 4.0,
                        defense: 8.0 + *level as f32 * 3.0,
                        speed: 10.0,
                        ability_power: 8.0 + *level as f32 * 3.0,
                    },
                    equipment: Equipment::default(),
                    traits: vec!["champion".into()],
                    status: AdventurerStatus::Idle,
                    loyalty: 100.0,
                    stress: 0.0,
                    fatigue: 0.0,
                    injury: 0.0,
                    resolve: 90.0,
                    morale: 95.0,
                    party_id: None,
                    guild_relationship: 30.0, // Neutral — player may interact
                    leadership_role: Some(LeadershipRole {
                        title: format!("Champion of the King"),
                        buff: buff.clone(),
                    }),
                    is_player_character: false,
                    faction_id: None, // Unaffiliated until activated
                    rallying_to: None,
                };

                champion_ids.push(id);
                state.adventurers.push(champ);
            }

            let crisis = ActiveCrisis::SleepingKing {
                king_faction_id: *faction_id,
                champion_ids,
                champions_arrived: 0,
                next_activation_tick: state.tick + 3000, // First activation in ~5 min
                activation_interval: 2000, // Then every ~3.3 min
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: "The Sleeping King stirs! Ancient champions across the land feel the call...".into(),
            });
        }

        CalamityType::MajorMonster { name, strength } => {
            // Find a dungeon location as the source
            let source = state.overworld.locations.iter()
                .find(|l| l.location_type == LocationType::Dungeon)
                .map(|l| l.id)
                .unwrap_or(0);

            let crisis = ActiveCrisis::Breach {
                source_location_id: source,
                wave_number: 0,
                wave_strength: *strength * 0.3,
                next_wave_tick: state.tick + 2000,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: format!("The ground trembles... {} stirs beneath the earth!", name),
            });
        }

        CalamityType::CrisisFlood => {
            // Find the most unstable region as origin
            let origin = state.overworld.regions.iter()
                .max_by(|a, b| a.unrest.partial_cmp(&b.unrest).unwrap_or(std::cmp::Ordering::Equal))
                .map(|r| r.id)
                .unwrap_or(0);

            let crisis = ActiveCrisis::Corruption {
                origin_region_id: origin,
                corrupted_regions: vec![origin],
                spread_rate_ticks: 5000,
                next_spread_tick: state.tick + 5000,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: "A strange blight spreads across the land...".into(),
            });
        }

        CalamityType::Conquest => {
            let crisis = ActiveCrisis::Decline {
                severity: 1.0,
                tick_started: state.tick,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: "The world grows weary. Resources dwindle, morale falters...".into(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Sleeping King
// ---------------------------------------------------------------------------

fn tick_sleeping_king(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    crisis: ActiveCrisis,
) -> Option<ActiveCrisis> {
    let (king_faction_id, champion_ids, mut champions_arrived, mut next_activation_tick, activation_interval) =
        match crisis {
            ActiveCrisis::SleepingKing {
                king_faction_id, champion_ids, champions_arrived, next_activation_tick, activation_interval,
            } => (king_faction_id, champion_ids, champions_arrived, next_activation_tick, activation_interval),
            _ => return Some(crisis),
        };

    // Activate next dormant champion on schedule
    if state.tick >= next_activation_tick {
        if let Some(champ) = state.adventurers.iter_mut().find(|a| {
            champion_ids.contains(&a.id)
                && a.rallying_to.is_none()
                && a.faction_id != Some(king_faction_id)
                && a.status != AdventurerStatus::Dead
        }) {
            let king_pos = state.overworld.regions.iter()
                .find(|r| r.owner_faction_id == king_faction_id)
                .map(|r| ((r.id as f32 * 20.0) - 30.0, (r.id as f32 * 15.0) - 20.0))
                .unwrap_or((50.0, 50.0));

            let name = champ.name.clone();
            champ.rallying_to = Some(RallyTarget {
                faction_id: king_faction_id,
                destination: king_pos,
                speed: 3.0,
            });
            champ.status = AdventurerStatus::Traveling;

            events.push(WorldEvent::CampaignMilestone {
                description: format!("{} has heard the King's call and begins their journey!", name),
            });

            next_activation_tick = state.tick + activation_interval;
        }
    }

    // Check for arrivals (simplified: arrive after activation_interval ticks of travel)
    let mut newly_arrived = Vec::new();
    for champ in &mut state.adventurers {
        if !champion_ids.contains(&champ.id) || champ.status == AdventurerStatus::Dead {
            continue;
        }
        if let Some(ref rally) = champ.rallying_to {
            if rally.faction_id == king_faction_id {
                champ.faction_id = Some(king_faction_id);
                champ.rallying_to = None;
                champ.status = AdventurerStatus::Idle;
                newly_arrived.push(champ.name.clone());
            }
        }
    }

    for name in &newly_arrived {
        champions_arrived += 1;

        if let Some(faction) = state.factions.iter_mut().find(|f| f.id == king_faction_id) {
            // QUADRATIC snowball: power = 25 * n^2
            let power_boost = 25.0 * (champions_arrived as f32) * (champions_arrived as f32);
            faction.military_strength += power_boost;
            faction.max_military_strength += power_boost;

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} has joined the Sleeping King! ({}/{} champions, +{:.0} strength, total {:.0})",
                    name, champions_arrived, champion_ids.len(),
                    power_boost, faction.military_strength
                ),
            });

            if champions_arrived >= 5 && faction.diplomatic_stance != DiplomaticStance::AtWar {
                faction.diplomatic_stance = DiplomaticStance::AtWar;
                if !faction.at_war_with.contains(&state.diplomacy.guild_faction_id) {
                    faction.at_war_with.push(state.diplomacy.guild_faction_id);
                }
                faction.relationship_to_guild = -100.0;
                events.push(WorldEvent::CampaignMilestone {
                    description: "The Sleeping King declares war on all who oppose them!".into(),
                });
            }
        }
    }

    // Crisis resolved if king faction destroyed (all regions lost, strength < 10)
    let king_alive = state.factions.iter()
        .find(|f| f.id == king_faction_id)
        .map(|f| f.military_strength > 10.0 || f.territory_size > 0)
        .unwrap_or(false);

    if !king_alive {
        events.push(WorldEvent::CampaignMilestone {
            description: "The Sleeping King has been defeated!".into(),
        });
        return None; // Remove crisis
    }

    Some(ActiveCrisis::SleepingKing {
        king_faction_id,
        champion_ids,
        champions_arrived,
        next_activation_tick,
        activation_interval,
    })
}

// ---------------------------------------------------------------------------
// Breach
// ---------------------------------------------------------------------------

fn tick_breach(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    crisis: ActiveCrisis,
) -> Option<ActiveCrisis> {
    let (source_location_id, mut wave_number, mut wave_strength, next_wave_tick) = match crisis {
        ActiveCrisis::Breach { source_location_id, wave_number, wave_strength, next_wave_tick } =>
            (source_location_id, wave_number, wave_strength, next_wave_tick),
        _ => return Some(crisis),
    };

    if state.tick < next_wave_tick {
        return Some(ActiveCrisis::Breach { source_location_id, wave_number, wave_strength, next_wave_tick });
    }

    wave_number += 1;
    wave_strength *= 1.3;

    if let Some(region) = state.overworld.regions.iter_mut()
        .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
        .min_by(|a, b| a.control.partial_cmp(&b.control).unwrap_or(std::cmp::Ordering::Equal))
    {
        region.control = (region.control - wave_strength * 0.5).max(0.0);
        region.unrest = (region.unrest + wave_strength * 0.3).min(100.0);
    }

    events.push(WorldEvent::CampaignMilestone {
        description: format!("Breach wave {} erupts! (strength {:.0})", wave_number, wave_strength),
    });

    let wave_interval = 3000u64.saturating_sub(wave_number as u64 * 200);
    Some(ActiveCrisis::Breach {
        source_location_id,
        wave_number,
        wave_strength,
        next_wave_tick: state.tick + wave_interval.max(500),
    })
}

// ---------------------------------------------------------------------------
// Corruption
// ---------------------------------------------------------------------------

fn tick_corruption(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    crisis: ActiveCrisis,
) -> Option<ActiveCrisis> {
    let (origin_region_id, mut corrupted_regions, spread_rate_ticks, next_spread_tick) = match crisis {
        ActiveCrisis::Corruption { origin_region_id, corrupted_regions, spread_rate_ticks, next_spread_tick } =>
            (origin_region_id, corrupted_regions, spread_rate_ticks, next_spread_tick),
        _ => return Some(crisis),
    };

    // Ongoing damage to corrupted regions
    for &rid in &corrupted_regions {
        if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == rid) {
            region.unrest = (region.unrest + 0.5).min(100.0);
            region.control = (region.control - 0.3).max(0.0);
        }
    }

    if state.tick >= next_spread_tick {
        // Spread to adjacent
        let mut spread_to = None;
        for &rid in &corrupted_regions {
            if let Some(region) = state.overworld.regions.iter().find(|r| r.id == rid) {
                for &neighbor in &region.neighbors {
                    if !corrupted_regions.contains(&neighbor) {
                        spread_to = Some(neighbor);
                        break;
                    }
                }
            }
            if spread_to.is_some() { break; }
        }

        if let Some(new_rid) = spread_to {
            corrupted_regions.push(new_rid);
            let name = state.overworld.regions.iter()
                .find(|r| r.id == new_rid)
                .map(|r| r.name.clone())
                .unwrap_or_else(|| format!("Region {}", new_rid));
            events.push(WorldEvent::CampaignMilestone {
                description: format!("The corruption spreads to {}! ({}/{} regions)",
                    name, corrupted_regions.len(), state.overworld.regions.len()),
            });
        }

        // Resolved if all regions corrupted or all corrupted regions purged
        if corrupted_regions.len() >= state.overworld.regions.len() {
            events.push(WorldEvent::CampaignMilestone {
                description: "The corruption has consumed the entire realm!".into(),
            });
        }

        return Some(ActiveCrisis::Corruption {
            origin_region_id,
            corrupted_regions,
            spread_rate_ticks,
            next_spread_tick: state.tick + spread_rate_ticks,
        });
    }

    Some(ActiveCrisis::Corruption { origin_region_id, corrupted_regions, spread_rate_ticks, next_spread_tick })
}

// ---------------------------------------------------------------------------
// Unifier
// ---------------------------------------------------------------------------

fn tick_unifier(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    crisis: ActiveCrisis,
) -> Option<ActiveCrisis> {
    let (unifier_faction_id, mut absorbed_factions) = match crisis {
        ActiveCrisis::Unifier { unifier_faction_id, absorbed_factions } =>
            (unifier_faction_id, absorbed_factions),
        _ => return Some(crisis),
    };

    // Every 5000 ticks, the unifier absorbs the weakest non-guild faction
    if state.tick % 5000 == 0 {
        let weakest = state.factions.iter()
            .filter(|f| {
                f.id != unifier_faction_id
                    && f.id != state.diplomacy.guild_faction_id
                    && !absorbed_factions.contains(&f.id)
            })
            .min_by(|a, b| a.military_strength.partial_cmp(&b.military_strength).unwrap_or(std::cmp::Ordering::Equal))
            .map(|f| f.id);

        if let Some(target_id) = weakest {
            absorbed_factions.push(target_id);

            // Transfer regions
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == target_id {
                    region.owner_faction_id = unifier_faction_id;
                }
            }

            // Absorb military strength
            let target_strength = state.factions.iter()
                .find(|f| f.id == target_id)
                .map(|f| f.military_strength)
                .unwrap_or(0.0);
            if let Some(unifier) = state.factions.iter_mut().find(|f| f.id == unifier_faction_id) {
                unifier.military_strength += target_strength * 0.7;
                unifier.max_military_strength += target_strength * 0.5;
            }

            let target_name = state.factions.iter()
                .find(|f| f.id == target_id)
                .map(|f| f.name.clone())
                .unwrap_or_default();

            events.push(WorldEvent::CampaignMilestone {
                description: format!("The Unifier has absorbed {}! ({} factions remain independent)",
                    target_name,
                    state.factions.len() - absorbed_factions.len() - 1 // -1 for the unifier itself
                ),
            });
        }
    }

    Some(ActiveCrisis::Unifier { unifier_faction_id, absorbed_factions })
}

// ---------------------------------------------------------------------------
// Decline
// ---------------------------------------------------------------------------

fn tick_decline(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    crisis: ActiveCrisis,
) -> Option<ActiveCrisis> {
    let (severity, tick_started) = match crisis {
        ActiveCrisis::Decline { severity, tick_started } => (severity, tick_started),
        _ => return Some(crisis),
    };

    let elapsed = state.tick.saturating_sub(tick_started);
    let severity = 1.0 + (elapsed as f32 / 10000.0);

    if state.tick % 100 == 0 {
        state.guild.gold = (state.guild.gold - severity * 0.5).max(0.0);
        state.guild.supplies = (state.guild.supplies - severity * 0.3).max(0.0);

        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale - severity * 0.1).max(0.0);
                adv.stress = (adv.stress + severity * 0.05).min(100.0);
            }
        }

        for region in &mut state.overworld.regions {
            region.control = (region.control - severity * 0.1).max(0.0);
            region.unrest = (region.unrest + severity * 0.05).min(100.0);
        }
    }

    Some(ActiveCrisis::Decline { severity, tick_started })
}
