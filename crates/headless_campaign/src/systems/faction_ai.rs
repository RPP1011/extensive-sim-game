//! Faction AI — every 600 ticks (~60s).
//!
//! Each faction evaluates its situation and picks from multiple possible
//! actions based on scoring. Factions interact with each other, not just
//! the guild. War costs strength. Coalition members defend allies.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

pub fn tick_faction_ai(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    let cfg = state.config.faction_ai.clone();
    if state.tick % cfg.decision_interval_ticks != 0 || state.tick == 0 {
        return;
    }

    let n_factions = state.factions.len();
    if n_factions == 0 {
        return;
    }

    let guild_faction_id = state.diplomacy.guild_faction_id;
    let max_recent = cfg.max_recent_actions;

    for fi in 0..n_factions {
        let _strength = state.factions[fi].military_strength;
        let max_strength = state.factions[fi].max_military_strength;
        let stance = state.factions[fi].diplomatic_stance;
        let _relation = state.factions[fi].relationship_to_guild;
        let _is_coalition = state.factions[fi].coalition_member;

        let action_desc = match stance {
            DiplomaticStance::AtWar => {
                faction_at_war(state, fi, guild_faction_id, &cfg)
            }
            DiplomaticStance::Hostile => {
                faction_hostile(state, fi, &cfg)
            }
            DiplomaticStance::Neutral => {
                faction_neutral(state, fi, &cfg)
            }
            DiplomaticStance::Friendly => {
                faction_friendly(state, fi, guild_faction_id, &cfg)
            }
            DiplomaticStance::Coalition => {
                faction_coalition(state, fi, guild_faction_id, &cfg)
            }
        };

        // Natural strength regeneration (factions rebuild over time)
        let regen = (max_strength - state.factions[fi].military_strength).max(0.0) * 0.02;
        state.factions[fi].military_strength =
            (state.factions[fi].military_strength + regen).min(max_strength);

        // Relationship drift — war erodes relations, peace improves them
        match state.factions[fi].diplomatic_stance {
            DiplomaticStance::AtWar => {
                state.factions[fi].relationship_to_guild =
                    (state.factions[fi].relationship_to_guild - 3.0).max(-100.0);
            }
            DiplomaticStance::Hostile => {
                state.factions[fi].relationship_to_guild =
                    (state.factions[fi].relationship_to_guild - 1.0).max(-100.0);
            }
            DiplomaticStance::Coalition | DiplomaticStance::Friendly => {
                // Slight positive drift for allies
                state.factions[fi].relationship_to_guild =
                    (state.factions[fi].relationship_to_guild + 0.5).min(100.0);
            }
            _ => {}
        }

        // Update territory_size
        state.factions[fi].territory_size = state
            .overworld
            .regions
            .iter()
            .filter(|r| r.owner_faction_id == fi)
            .count();

        state.factions[fi].recent_actions.push(FactionActionRecord {
            tick: state.tick,
            action: action_desc.clone(),
        });
        if state.factions[fi].recent_actions.len() > max_recent {
            state.factions[fi].recent_actions.remove(0);
        }

        events.push(WorldEvent::FactionActionTaken {
            faction_id: fi,
            action: action_desc,
        });
    }

    // Faction-to-faction wars (factions fight each other, not just the guild)
    tick_faction_wars(state, events);
}

/// Faction at war: attacks guild territory, but war COSTS strength.
fn faction_at_war(
    state: &mut CampaignState,
    fi: usize,
    guild_faction_id: usize,
    cfg: &crate::config::FactionAiConfig,
) -> String {
    let strength = state.factions[fi].military_strength;

    // War costs resources — factions need minimum strength to keep attacking
    if strength < 20.0 {
        // Too weak to attack — forced ceasefire
        state.factions[fi].diplomatic_stance = DiplomaticStance::Hostile;
        state.factions[fi].at_war_with.retain(|&id| id != guild_faction_id);
        return "Forced ceasefire — military exhausted".into();
    }

    // Attack costs a flat amount + small percentage (stronger armies sustain longer)
    let attack_cost = 3.0 + strength * 0.03;
    state.factions[fi].military_strength -= attack_cost;

    // Find guild region to attack (prefer weakest control first)
    let target_idx = state.overworld.regions.iter()
        .enumerate()
        .filter(|(_, r)| r.owner_faction_id == guild_faction_id && r.control > 0.0)
        .min_by(|(_, a), (_, b)| a.control.partial_cmp(&b.control).unwrap())
        .map(|(i, _)| i);

    if let Some(idx) = target_idx {
        let attack_power = strength * cfg.attack_power_fraction;
        let region = &mut state.overworld.regions[idx];
        region.control = (region.control - attack_power).max(0.0);
        region.unrest = (region.unrest + attack_power * 0.5).min(100.0);

        if region.control <= 0.0 {
            // Conquest! Faction gains power from captured territory
            region.owner_faction_id = fi;
            region.control = cfg.territory_capture_control;
            region.unrest = 20.0;
            // War spoils: conquering boosts military strength (modest)
            state.factions[fi].military_strength += 5.0;
            format!("Captured region {} — strength grows to {:.0}!",
                region.name, state.factions[fi].military_strength)
        } else {
            format!("Attacked region {} (control {:.0}, strength {:.0})",
                region.name, region.control, state.factions[fi].military_strength)
        }
    } else {
        // All territory conquered — faction rebuilds
        state.factions[fi].military_strength += cfg.hostile_strength_gain;
        "No remaining targets — consolidating power".into()
    }
}

/// Hostile faction: builds up military, may declare war if strong enough.
fn faction_hostile(
    state: &mut CampaignState,
    fi: usize,
    cfg: &crate::config::FactionAiConfig,
) -> String {
    state.factions[fi].military_strength += cfg.hostile_strength_gain;

    if state.factions[fi].military_strength > cfg.war_declaration_threshold
        && state.factions[fi].relationship_to_guild < -20.0
    {
        state.factions[fi].diplomatic_stance = DiplomaticStance::AtWar;
        state.factions[fi].at_war_with.push(state.diplomacy.guild_faction_id);
        state.factions[fi].relationship_to_guild =
            (state.factions[fi].relationship_to_guild - cfg.war_declaration_penalty).max(-100.0);
        "Declared war on the guild!".into()
    } else {
        format!("Recruited forces (strength {:.0})", state.factions[fi].military_strength)
    }
}

/// Neutral faction: defends territory, trades, may drift toward friendly or hostile.
fn faction_neutral(
    state: &mut CampaignState,
    fi: usize,
    cfg: &crate::config::FactionAiConfig,
) -> String {
    // Defend owned territory
    for region in &mut state.overworld.regions {
        if region.owner_faction_id == fi {
            region.control = (region.control + cfg.neutral_control_gain).min(100.0);
        }
    }

    // Drift based on relationship
    let rel = state.factions[fi].relationship_to_guild;
    if rel > 40.0 {
        state.factions[fi].diplomatic_stance = DiplomaticStance::Friendly;
        return "Relations improved — now friendly toward guild".into();
    } else if rel < -30.0 {
        state.factions[fi].diplomatic_stance = DiplomaticStance::Hostile;
        return "Relations deteriorated — now hostile toward guild".into();
    }

    "Fortified borders".into()
}

/// Friendly faction: trades, supports guild, may join coalition.
fn faction_friendly(
    state: &mut CampaignState,
    fi: usize,
    guild_faction_id: usize,
    cfg: &crate::config::FactionAiConfig,
) -> String {
    let rel = state.factions[fi].relationship_to_guild;

    // Friendly factions defend their own territory AND shared borders
    for region in &mut state.overworld.regions {
        if region.owner_faction_id == fi {
            region.control = (region.control + cfg.neutral_control_gain * 0.5).min(100.0);
        }
    }

    // If guild regions are under attack, friendly factions provide support
    let guild_under_pressure = state
        .overworld
        .regions
        .iter()
        .any(|r| r.owner_faction_id == guild_faction_id && r.control < 40.0);

    if guild_under_pressure && state.factions[fi].military_strength > 30.0 {
        // Send military aid — boost guild region control
        if let Some(region) = state
            .overworld
            .regions
            .iter_mut()
            .find(|r| r.owner_faction_id == guild_faction_id && r.control < 40.0)
        {
            let aid = state.factions[fi].military_strength * 0.03;
            region.control = (region.control + aid).min(100.0);
            state.factions[fi].military_strength -= aid * 0.5; // Costs them some strength
            return format!("Sent military aid to {} (control +{:.0})", region.name, aid);
        }
    }

    // Relationship improvement
    state.factions[fi].relationship_to_guild =
        (rel + cfg.friendly_relationship_gain * 0.5).min(100.0);

    // Check if should drift to neutral
    if rel < 20.0 {
        state.factions[fi].diplomatic_stance = DiplomaticStance::Neutral;
        return "Relations cooled — now neutral".into();
    }

    "Trade and diplomatic exchange".into()
}

/// Coalition member: actively defends guild, joint operations, shares resources.
fn faction_coalition(
    state: &mut CampaignState,
    fi: usize,
    guild_faction_id: usize,
    cfg: &crate::config::FactionAiConfig,
) -> String {
    // Coalition members defend guild territory as their own
    for region in &mut state.overworld.regions {
        if region.owner_faction_id == guild_faction_id || region.owner_faction_id == fi {
            region.control = (region.control + cfg.neutral_control_gain).min(100.0);
            region.unrest = (region.unrest - 0.5).max(0.0);
        }
    }

    // Counter-attack factions at war with the guild
    for other_fi in 0..state.factions.len() {
        if other_fi == fi {
            continue;
        }
        if state.factions[other_fi].diplomatic_stance == DiplomaticStance::AtWar {
            // Attack the enemy's regions
            if let Some(region) = state
                .overworld
                .regions
                .iter_mut()
                .find(|r| r.owner_faction_id == other_fi && r.control > 10.0)
            {
                let attack = state.factions[fi].military_strength * 0.05;
                region.control = (region.control - attack).max(0.0);
                state.factions[fi].military_strength -= attack * 0.3;
                return format!(
                    "Coalition counter-attack on {} territory (region {})",
                    state.factions[other_fi].name, region.name
                );
            }
        }
    }

    // Provide supplies to guild
    "Coalition support — defending shared territory".into()
}

/// Factions fight each other over territory (not just the guild).
fn tick_faction_wars(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n = state.factions.len();
    for fi in 0..n {
        let war_targets = state.factions[fi].at_war_with.clone();
        for &target in &war_targets {
            if target >= n || target == state.diplomacy.guild_faction_id {
                continue; // Guild wars handled above
            }

            let attacker_strength = state.factions[fi].military_strength;
            let defender_strength = state.factions[target].military_strength;

            if attacker_strength < 10.0 {
                continue; // Too weak
            }

            // Attack target's regions
            if let Some(region) = state
                .overworld
                .regions
                .iter_mut()
                .find(|r| r.owner_faction_id == target && r.control > 5.0)
            {
                let attack = attacker_strength * 0.04;
                let defense = defender_strength * 0.02;
                let net = (attack - defense).max(0.0);

                region.control = (region.control - net).max(0.0);
                state.factions[fi].military_strength -= attack * 0.3;

                if region.control <= 0.0 {
                    region.owner_faction_id = fi;
                    region.control = 30.0;
                    events.push(WorldEvent::RegionOwnerChanged {
                        region_id: region.id,
                        old_owner: target,
                        new_owner: fi,
                    });
                }
            }
        }
    }
}
