//! Endgame crisis system — supports multiple simultaneous crises.
//!
//! Each crisis type has its own escalation mechanics.
//! Multiple crises can be active at once — the world can face a
//! Sleeping King AND a dungeon breach simultaneously.
//!
//! All content (champion rosters, escalation rates, etc.) comes from
//! TOML templates in `assets/crises/`. This module is pure mechanics.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::crisis_templates::{
    compute_power_boost, parse_leadership_buff, parse_location_type, CrisisConfig, CrisisTemplate,
};
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

/// Activate a crisis from a template. Called by threat.rs when
/// campaign progress crosses the template's threshold.
pub fn activate_crisis_from_template(
    state: &mut CampaignState,
    template: &CrisisTemplate,
    events: &mut Vec<WorldEvent>,
) {
    // Don't duplicate crisis types
    let crisis_type = template.crisis_type.as_str();
    let already_active = state.overworld.active_crises.iter().any(|c| {
        matches!(
            (c, crisis_type),
            (ActiveCrisis::SleepingKing { .. }, "sleeping_king")
                | (ActiveCrisis::Breach { .. }, "breach")
                | (ActiveCrisis::Corruption { .. }, "corruption")
                | (ActiveCrisis::Unifier { .. }, "unifier")
                | (ActiveCrisis::Decline { .. }, "decline")
        )
    });
    if already_active {
        return;
    }

    match &template.config {
        CrisisConfig::SleepingKing {
            king_faction_name,
            champions,
            first_activation_ticks,
            activation_interval_ticks,
            power_formula,
            war_threshold,
        } => {
            // Find the faction by name, or pick the strongest hostile faction
            let king_faction_id = state
                .factions
                .iter()
                .find(|f| f.name == *king_faction_name)
                .map(|f| f.id)
                .or_else(|| {
                    state
                        .factions
                        .iter()
                        .filter(|f| {
                            matches!(
                                f.diplomatic_stance,
                                DiplomaticStance::Hostile | DiplomaticStance::AtWar
                            )
                        })
                        .max_by(|a, b| {
                            a.military_strength
                                .partial_cmp(&b.military_strength)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|f| f.id)
                })
                .unwrap_or(1); // fallback

            let mut champion_ids = Vec::new();

            for (i, champ_def) in champions.iter().enumerate() {
                let id = 9000 + i as u32;
                let buff = parse_leadership_buff(&champ_def.buff_type, champ_def.buff_value);

                let champ = Adventurer {
                    id,
                    name: champ_def.name.clone(),
                    archetype: champ_def.archetype.clone(),
                    level: champ_def.level,
                    xp: 0,
                    stats: AdventurerStats {
                        max_hp: 100.0 + champ_def.level as f32 * 15.0,
                        attack: 10.0 + champ_def.level as f32 * 4.0,
                        defense: 8.0 + champ_def.level as f32 * 3.0,
                        speed: 10.0,
                        ability_power: 8.0 + champ_def.level as f32 * 3.0,
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
                    guild_relationship: 30.0,
                    leadership_role: Some(LeadershipRole {
                        title: "Champion of the King".to_string(),
                        buff,
                    }),
                    is_player_character: false,
                    faction_id: None,
                    rallying_to: None,
                    tier_status: Default::default(),
                    history_tags: Default::default(),
                };

                champion_ids.push(id);
                state.adventurers.push(champ);
            }

            let crisis = ActiveCrisis::SleepingKing {
                king_faction_id,
                champion_ids,
                champions_arrived: 0,
                next_activation_tick: state.tick + first_activation_ticks,
                activation_interval: *activation_interval_ticks,
                power_formula: power_formula.clone(),
                war_threshold: *war_threshold,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{}: {}",
                    template.name, template.description
                ),
            });
        }

        CrisisConfig::Breach {
            source_location_type,
            initial_strength,
            strength_multiplier,
            initial_wave_interval,
            wave_acceleration,
            min_wave_interval,
        } => {
            let loc_type = parse_location_type(source_location_type);
            let source = state
                .overworld
                .locations
                .iter()
                .find(|l| l.location_type == loc_type)
                .map(|l| l.id)
                .unwrap_or(0);

            // Scale initial strength by world threat level
            let base_strength = state.overworld.global_threat_level.max(50.0);
            let wave_strength = base_strength * initial_strength;

            let crisis = ActiveCrisis::Breach {
                source_location_id: source,
                wave_number: 0,
                wave_strength,
                next_wave_tick: state.tick + initial_wave_interval,
                strength_multiplier: *strength_multiplier,
                wave_acceleration: *wave_acceleration,
                min_wave_interval: *min_wave_interval,
                initial_wave_interval: *initial_wave_interval,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{}: {}",
                    template.name, template.description
                ),
            });
        }

        CrisisConfig::Corruption {
            spread_interval_ticks,
            control_damage_per_tick,
            unrest_damage_per_tick,
        } => {
            let origin = state
                .overworld
                .regions
                .iter()
                .max_by(|a, b| {
                    a.unrest
                        .partial_cmp(&b.unrest)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|r| r.id)
                .unwrap_or(0);

            let crisis = ActiveCrisis::Corruption {
                origin_region_id: origin,
                corrupted_regions: vec![origin],
                spread_rate_ticks: *spread_interval_ticks,
                next_spread_tick: state.tick + spread_interval_ticks,
                control_damage_per_tick: *control_damage_per_tick,
                unrest_damage_per_tick: *unrest_damage_per_tick,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{}: {}",
                    template.name, template.description
                ),
            });
        }

        CrisisConfig::Unifier {
            absorb_interval_ticks,
            strength_absorption_rate,
        } => {
            // Pick the strongest hostile faction as the unifier
            let unifier_faction_id = state
                .factions
                .iter()
                .filter(|f| {
                    f.id != state.diplomacy.guild_faction_id
                        && matches!(
                            f.diplomatic_stance,
                            DiplomaticStance::Hostile | DiplomaticStance::AtWar
                        )
                })
                .max_by(|a, b| {
                    a.military_strength
                        .partial_cmp(&b.military_strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|f| f.id)
                .unwrap_or(1);

            let crisis = ActiveCrisis::Unifier {
                unifier_faction_id,
                absorbed_factions: Vec::new(),
                absorb_interval_ticks: *absorb_interval_ticks,
                strength_absorption_rate: *strength_absorption_rate,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{}: {}",
                    template.name, template.description
                ),
            });
        }

        CrisisConfig::Decline {
            gold_drain_per_tick,
            supply_drain_per_tick,
            morale_drain_per_tick,
            control_drain_per_tick,
            severity_growth_rate,
        } => {
            let crisis = ActiveCrisis::Decline {
                severity: 1.0,
                tick_started: state.tick,
                gold_drain_per_tick: *gold_drain_per_tick,
                supply_drain_per_tick: *supply_drain_per_tick,
                morale_drain_per_tick: *morale_drain_per_tick,
                control_drain_per_tick: *control_drain_per_tick,
                severity_growth_rate: *severity_growth_rate,
            };

            state.overworld.active_crises.push(crisis);

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{}: {}",
                    template.name, template.description
                ),
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
    let (
        king_faction_id,
        champion_ids,
        mut champions_arrived,
        mut next_activation_tick,
        activation_interval,
        power_formula,
        war_threshold,
    ) = match crisis {
        ActiveCrisis::SleepingKing {
            king_faction_id,
            champion_ids,
            champions_arrived,
            next_activation_tick,
            activation_interval,
            power_formula,
            war_threshold,
        } => (
            king_faction_id,
            champion_ids,
            champions_arrived,
            next_activation_tick,
            activation_interval,
            power_formula,
            war_threshold,
        ),
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
            let king_pos = state
                .overworld
                .regions
                .iter()
                .find(|r| r.owner_faction_id == king_faction_id)
                .map(|r| {
                    (
                        (r.id as f32 * 20.0) - 30.0,
                        (r.id as f32 * 15.0) - 20.0,
                    )
                })
                .unwrap_or((50.0, 50.0));

            let name = champ.name.clone();
            champ.rallying_to = Some(RallyTarget {
                faction_id: king_faction_id,
                destination: king_pos,
                speed: 3.0,
            });
            champ.status = AdventurerStatus::Traveling;

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} has heard the King's call and begins their journey!",
                    name
                ),
            });

            next_activation_tick = state.tick + activation_interval;
        }
    }

    // Check for arrivals
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
            let power_boost = compute_power_boost(&power_formula, champions_arrived);
            faction.military_strength += power_boost;
            faction.max_military_strength += power_boost;

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "{} has joined the Sleeping King! ({}/{} champions, +{:.0} strength, total {:.0})",
                    name,
                    champions_arrived,
                    champion_ids.len(),
                    power_boost,
                    faction.military_strength
                ),
            });

            if champions_arrived >= war_threshold
                && faction.diplomatic_stance != DiplomaticStance::AtWar
            {
                faction.diplomatic_stance = DiplomaticStance::AtWar;
                if !faction
                    .at_war_with
                    .contains(&state.diplomacy.guild_faction_id)
                {
                    faction
                        .at_war_with
                        .push(state.diplomacy.guild_faction_id);
                }
                faction.relationship_to_guild = -100.0;
                events.push(WorldEvent::CampaignMilestone {
                    description: "The Sleeping King declares war on all who oppose them!"
                        .into(),
                });
            }
        }
    }

    // Crisis resolved if king faction destroyed
    let king_alive = state
        .factions
        .iter()
        .find(|f| f.id == king_faction_id)
        .map(|f| f.military_strength > 10.0 || f.territory_size > 0)
        .unwrap_or(false);

    if !king_alive {
        events.push(WorldEvent::CampaignMilestone {
            description: "The Sleeping King has been defeated!".into(),
        });
        return None;
    }

    Some(ActiveCrisis::SleepingKing {
        king_faction_id,
        champion_ids,
        champions_arrived,
        next_activation_tick,
        activation_interval,
        power_formula,
        war_threshold,
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
    let (
        source_location_id,
        mut wave_number,
        mut wave_strength,
        next_wave_tick,
        strength_multiplier,
        wave_acceleration,
        min_wave_interval,
        initial_wave_interval,
    ) = match crisis {
        ActiveCrisis::Breach {
            source_location_id,
            wave_number,
            wave_strength,
            next_wave_tick,
            strength_multiplier,
            wave_acceleration,
            min_wave_interval,
            initial_wave_interval,
        } => (
            source_location_id,
            wave_number,
            wave_strength,
            next_wave_tick,
            strength_multiplier,
            wave_acceleration,
            min_wave_interval,
            initial_wave_interval,
        ),
        _ => return Some(crisis),
    };

    if state.tick < next_wave_tick {
        return Some(ActiveCrisis::Breach {
            source_location_id,
            wave_number,
            wave_strength,
            next_wave_tick,
            strength_multiplier,
            wave_acceleration,
            min_wave_interval,
            initial_wave_interval,
        });
    }

    wave_number += 1;
    wave_strength *= strength_multiplier;

    if let Some(region) = state
        .overworld
        .regions
        .iter_mut()
        .filter(|r| r.owner_faction_id == state.diplomacy.guild_faction_id)
        .min_by(|a, b| {
            a.control
                .partial_cmp(&b.control)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        region.control = (region.control - wave_strength * 0.5).max(0.0);
        region.unrest = (region.unrest + wave_strength * 0.3).min(100.0);
    }

    events.push(WorldEvent::CampaignMilestone {
        description: format!(
            "Breach wave {} erupts! (strength {:.0})",
            wave_number, wave_strength
        ),
    });

    let next_interval =
        initial_wave_interval.saturating_sub(wave_number as u64 * wave_acceleration);
    Some(ActiveCrisis::Breach {
        source_location_id,
        wave_number,
        wave_strength,
        next_wave_tick: state.tick + next_interval.max(min_wave_interval),
        strength_multiplier,
        wave_acceleration,
        min_wave_interval,
        initial_wave_interval,
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
    let (
        origin_region_id,
        mut corrupted_regions,
        spread_rate_ticks,
        next_spread_tick,
        control_damage_per_tick,
        unrest_damage_per_tick,
    ) = match crisis {
        ActiveCrisis::Corruption {
            origin_region_id,
            corrupted_regions,
            spread_rate_ticks,
            next_spread_tick,
            control_damage_per_tick,
            unrest_damage_per_tick,
        } => (
            origin_region_id,
            corrupted_regions,
            spread_rate_ticks,
            next_spread_tick,
            control_damage_per_tick,
            unrest_damage_per_tick,
        ),
        _ => return Some(crisis),
    };

    // Ongoing damage to corrupted regions — rates from template
    for &rid in &corrupted_regions {
        if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == rid) {
            region.unrest = (region.unrest + unrest_damage_per_tick).min(100.0);
            region.control = (region.control - control_damage_per_tick).max(0.0);
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
            if spread_to.is_some() {
                break;
            }
        }

        if let Some(new_rid) = spread_to {
            corrupted_regions.push(new_rid);
            let name = state
                .overworld
                .regions
                .iter()
                .find(|r| r.id == new_rid)
                .map(|r| r.name.clone())
                .unwrap_or_else(|| format!("Region {}", new_rid));
            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "The corruption spreads to {}! ({}/{} regions)",
                    name,
                    corrupted_regions.len(),
                    state.overworld.regions.len()
                ),
            });
        }

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
            control_damage_per_tick,
            unrest_damage_per_tick,
        });
    }

    Some(ActiveCrisis::Corruption {
        origin_region_id,
        corrupted_regions,
        spread_rate_ticks,
        next_spread_tick,
        control_damage_per_tick,
        unrest_damage_per_tick,
    })
}

// ---------------------------------------------------------------------------
// Unifier
// ---------------------------------------------------------------------------

fn tick_unifier(
    state: &mut CampaignState,
    events: &mut Vec<WorldEvent>,
    crisis: ActiveCrisis,
) -> Option<ActiveCrisis> {
    let (unifier_faction_id, mut absorbed_factions, absorb_interval_ticks, strength_absorption_rate) =
        match crisis {
            ActiveCrisis::Unifier {
                unifier_faction_id,
                absorbed_factions,
                absorb_interval_ticks,
                strength_absorption_rate,
            } => (
                unifier_faction_id,
                absorbed_factions,
                absorb_interval_ticks,
                strength_absorption_rate,
            ),
            _ => return Some(crisis),
        };

    // Absorb on the template-defined interval
    if absorb_interval_ticks > 0 && state.tick % absorb_interval_ticks == 0 {
        let weakest = state
            .factions
            .iter()
            .filter(|f| {
                f.id != unifier_faction_id
                    && f.id != state.diplomacy.guild_faction_id
                    && !absorbed_factions.contains(&f.id)
            })
            .min_by(|a, b| {
                a.military_strength
                    .partial_cmp(&b.military_strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|f| f.id);

        if let Some(target_id) = weakest {
            absorbed_factions.push(target_id);

            // Transfer regions
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == target_id {
                    region.owner_faction_id = unifier_faction_id;
                }
            }

            // Absorb military strength at template-defined rate
            let target_strength = state
                .factions
                .iter()
                .find(|f| f.id == target_id)
                .map(|f| f.military_strength)
                .unwrap_or(0.0);
            if let Some(unifier) = state
                .factions
                .iter_mut()
                .find(|f| f.id == unifier_faction_id)
            {
                unifier.military_strength += target_strength * strength_absorption_rate;
                unifier.max_military_strength +=
                    target_strength * (strength_absorption_rate * 0.7);
            }

            let target_name = state
                .factions
                .iter()
                .find(|f| f.id == target_id)
                .map(|f| f.name.clone())
                .unwrap_or_default();

            events.push(WorldEvent::CampaignMilestone {
                description: format!(
                    "The Unifier has absorbed {}! ({} factions remain independent)",
                    target_name,
                    state.factions.len() - absorbed_factions.len() - 1
                ),
            });
        }
    }

    Some(ActiveCrisis::Unifier {
        unifier_faction_id,
        absorbed_factions,
        absorb_interval_ticks,
        strength_absorption_rate,
    })
}

// ---------------------------------------------------------------------------
// Decline
// ---------------------------------------------------------------------------

fn tick_decline(
    state: &mut CampaignState,
    _events: &mut Vec<WorldEvent>,
    crisis: ActiveCrisis,
) -> Option<ActiveCrisis> {
    let (
        _severity,
        tick_started,
        gold_drain_per_tick,
        supply_drain_per_tick,
        morale_drain_per_tick,
        control_drain_per_tick,
        severity_growth_rate,
    ) = match crisis {
        ActiveCrisis::Decline {
            severity,
            tick_started,
            gold_drain_per_tick,
            supply_drain_per_tick,
            morale_drain_per_tick,
            control_drain_per_tick,
            severity_growth_rate,
        } => (
            severity,
            tick_started,
            gold_drain_per_tick,
            supply_drain_per_tick,
            morale_drain_per_tick,
            control_drain_per_tick,
            severity_growth_rate,
        ),
        _ => return Some(crisis),
    };

    let elapsed = state.tick.saturating_sub(tick_started);
    let base_severity = 1.0 + (elapsed as f32 / severity_growth_rate);

    // Recovery: high reputation and quest completions push back against decline
    let quest_recovery = (state.completed_quests.len() as f32 * 0.05).min(3.0);
    let rep_recovery = (state.guild.reputation / 50.0).max(0.0); // rep 50 → 1.0 recovery
    let severity = (base_severity - quest_recovery - rep_recovery).max(0.5);

    if state.tick % 100 == 0 {
        state.guild.gold = (state.guild.gold - severity * gold_drain_per_tick).max(0.0);
        state.guild.supplies = (state.guild.supplies - severity * supply_drain_per_tick).max(0.0);

        for adv in &mut state.adventurers {
            if adv.status != AdventurerStatus::Dead {
                adv.morale = (adv.morale - severity * morale_drain_per_tick).max(0.0);
                adv.stress = (adv.stress + severity * (morale_drain_per_tick * 0.5)).min(100.0);
            }
        }

        for region in &mut state.overworld.regions {
            region.control = (region.control - severity * control_drain_per_tick).max(0.0);
            region.unrest = (region.unrest + severity * (control_drain_per_tick * 0.5)).min(100.0);
        }
    }

    Some(ActiveCrisis::Decline {
        severity,
        tick_started,
        gold_drain_per_tick,
        supply_drain_per_tick,
        morale_drain_per_tick,
        control_drain_per_tick,
        severity_growth_rate,
    })
}
