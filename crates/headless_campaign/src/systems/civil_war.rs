//! Civil war and coup system — internal faction instability.
//!
//! Every 200 ticks, checks whether factions should erupt into civil war.
//! During a civil war the faction splits strength between loyalists and rebels,
//! cannot declare new external wars, and regions gain unrest. The guild can
//! intervene by supporting either side via `SupportFactionSide`.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// Cadenced system — fires every 200 ticks.
pub fn tick_civil_wars(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % 7 != 0 || state.tick == 0 {
        return;
    }

    // --- Trigger new civil wars ---
    try_trigger_civil_wars(state, events);

    // --- Tick active civil wars ---
    tick_active_civil_wars(state, events);
}

// ---------------------------------------------------------------------------
// Trigger logic
// ---------------------------------------------------------------------------

fn try_trigger_civil_wars(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    let n = state.factions.len();
    let guild_faction_id = state.diplomacy.guild_faction_id;

    for fi in 0..n {
        // Skip guild faction and factions already in civil war
        if fi == guild_faction_id {
            continue;
        }
        if state.civil_wars.iter().any(|cw| cw.faction_id == fi) {
            continue;
        }

        let faction = &state.factions[fi];

        // Determine cause (if any condition is met)
        let cause = determine_civil_war_cause(faction, state, fi);
        if cause.is_none() {
            continue;
        }
        let cause = cause.unwrap();

        // 5% chance per check when conditions met
        let roll = lcg_f32(&mut state.rng);
        if roll > 0.05 {
            continue;
        }

        // Pick a rebel leader name deterministically
        let rebel_names = [
            "General Voss",
            "Commander Reth",
            "Lord Kaine",
            "Marshal Dren",
            "Captain Sura",
            "Warlord Thane",
            "Prefect Mala",
            "Admiral Bron",
        ];
        let name_idx = (lcg_next(&mut state.rng) as usize) % rebel_names.len();
        let rebel_leader = rebel_names[name_idx].to_string();

        // Split faction strength: rebels get 30-50% based on cause severity
        let rebel_fraction = 0.3 + lcg_f32(&mut state.rng) * 0.2;
        let total_strength = state.factions[fi].military_strength;
        let rebel_strength = total_strength * rebel_fraction;
        let loyalist_strength = total_strength - rebel_strength;

        // Faction's effective military strength drops to loyalist portion
        state.factions[fi].military_strength = loyalist_strength;

        state.civil_wars.push(CivilWarState {
            faction_id: fi,
            rebel_strength,
            loyalist_strength,
            started_tick: state.tick,
            rebel_leader_name: rebel_leader.clone(),
            cause: cause.clone(),
            guild_supported_side: None,
        });

        events.push(WorldEvent::CivilWarStarted {
            faction_id: fi,
            cause: format!("{:?}", cause),
        });
    }
}

fn determine_civil_war_cause(
    faction: &FactionState,
    state: &CampaignState,
    fi: usize,
) -> Option<CivilWarCause> {
    // Check succession crisis: faction lost >50% strength recently
    // (current strength < 50% of max)
    let strength_ratio = faction.military_strength / faction.max_military_strength;
    if strength_ratio < 0.5 {
        // Check if recent actions suggest military defeat
        let recent_defeats = faction
            .recent_actions
            .iter()
            .filter(|a| {
                a.tick + 2000 >= state.tick
                    && (a.action.contains("ceasefire") || a.action.contains("exhausted"))
            })
            .count();
        if recent_defeats > 0 {
            return Some(CivilWarCause::MilitaryDefeat);
        }
        return Some(CivilWarCause::SuccessionCrisis);
    }

    // Check corruption: hostile to guild but strong
    if faction.relationship_to_guild < -50.0 && faction.military_strength > 60.0 {
        return Some(CivilWarCause::Corruption);
    }

    // Check high unrest across faction regions
    let faction_regions: Vec<&Region> = state
        .overworld
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == fi)
        .collect();

    if !faction_regions.is_empty() {
        let avg_unrest: f32 =
            faction_regions.iter().map(|r| r.unrest).sum::<f32>() / faction_regions.len() as f32;
        if avg_unrest > 60.0 {
            // Pick a random cause from the more exotic options
            let causes = [
                CivilWarCause::ForeignInfluence,
                CivilWarCause::ReligiousSplit,
                CivilWarCause::Corruption,
            ];
            // Use faction id + tick to pick deterministically without consuming rng
            let idx = (fi + state.tick as usize) % causes.len();
            return Some(causes[idx].clone());
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Active civil war processing
// ---------------------------------------------------------------------------

fn tick_active_civil_wars(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Process civil wars; collect indices to remove
    let mut resolved_indices = Vec::new();
    let n_wars = state.civil_wars.len();

    for i in 0..n_wars {
        let cw = &state.civil_wars[i];
        let fi = cw.faction_id;
        let elapsed = state.tick.saturating_sub(cw.started_tick);

        // Apply unrest to faction regions (+20 per check, clamped)
        for region in &mut state.overworld.regions {
            if region.owner_faction_id == fi {
                region.unrest = (region.unrest + 4.0).min(100.0); // +4 per 200 ticks = +20 per 1000
            }
        }

        // Strength attrition: both sides lose some strength from fighting
        let attrition = 1.0 + lcg_f32(&mut state.rng) * 2.0;
        state.civil_wars[i].loyalist_strength =
            (state.civil_wars[i].loyalist_strength - attrition).max(0.0);
        state.civil_wars[i].rebel_strength =
            (state.civil_wars[i].rebel_strength - attrition).max(0.0);

        let loyalist = state.civil_wars[i].loyalist_strength;
        let rebel = state.civil_wars[i].rebel_strength;

        // Check resolution conditions
        if loyalist > 0.0 && rebel > 0.0 && loyalist >= rebel * 2.0 {
            // Loyalist victory
            resolved_indices.push((i, CivilWarResolution::LoyalistWin));
        } else if rebel > 0.0 && loyalist > 0.0 && rebel >= loyalist * 2.0 {
            // Rebel victory
            resolved_indices.push((i, CivilWarResolution::RebelWin));
        } else if elapsed >= 5000 {
            // Stalemate timeout
            resolved_indices.push((i, CivilWarResolution::Stalemate));
        }
    }

    // Resolve in reverse order to keep indices valid
    resolved_indices.sort_by(|a, b| b.0.cmp(&a.0));
    for (idx, resolution) in resolved_indices {
        let cw = state.civil_wars.remove(idx);
        resolve_civil_war(state, cw, resolution, events);
    }
}

#[derive(Clone, Debug)]
enum CivilWarResolution {
    LoyalistWin,
    RebelWin,
    Stalemate,
}

fn resolve_civil_war(
    state: &mut CampaignState,
    cw: CivilWarState,
    resolution: CivilWarResolution,
    events: &mut Vec<WorldEvent>,
) {
    let fi = cw.faction_id;

    match resolution {
        CivilWarResolution::LoyalistWin => {
            // Faction weakened but stable, relation unchanged
            let surviving_strength = cw.loyalist_strength * 0.8; // Some losses
            state.factions[fi].military_strength = surviving_strength;

            // Apply guild support consequences
            if let Some(supported_rebels) = cw.guild_supported_side {
                if supported_rebels {
                    // Supported loser (rebels) → -30 relation
                    state.factions[fi].relationship_to_guild =
                        (state.factions[fi].relationship_to_guild - 30.0).max(-100.0);
                } else {
                    // Supported winner (loyalists) → +20 relation
                    state.factions[fi].relationship_to_guild =
                        (state.factions[fi].relationship_to_guild + 20.0).min(100.0);
                }
            }

            // Reduce unrest in faction regions after stability returns
            for region in &mut state.overworld.regions {
                if region.owner_faction_id == fi {
                    region.unrest = (region.unrest - 10.0).max(0.0);
                }
            }

            events.push(WorldEvent::CivilWarResolved {
                faction_id: fi,
                rebels_won: false,
                description: format!(
                    "Loyalists prevailed in {}. {} was defeated.",
                    state.factions[fi].name, cw.rebel_leader_name
                ),
            });
        }

        CivilWarResolution::RebelWin => {
            // Faction leadership changes, relation reset to 0, new stance
            let surviving_strength = cw.rebel_strength * 0.8;
            state.factions[fi].military_strength = surviving_strength;

            // Apply guild support consequences
            if let Some(supported_rebels) = cw.guild_supported_side {
                if supported_rebels {
                    // Supported winner (rebels) → +20 relation
                    state.factions[fi].relationship_to_guild = 20.0;
                } else {
                    // Supported loser (loyalists) → -30 relation
                    state.factions[fi].relationship_to_guild = -30.0;
                }
            } else {
                // Neutral → reset to 0
                state.factions[fi].relationship_to_guild = 0.0;
            }

            // New leadership resets diplomatic stance to neutral
            state.factions[fi].diplomatic_stance = DiplomaticStance::Neutral;
            state.factions[fi].coalition_member = false;
            // Clear any wars — new government starts fresh
            state.factions[fi].at_war_with.clear();

            events.push(WorldEvent::CivilWarResolved {
                faction_id: fi,
                rebels_won: true,
                description: format!(
                    "{} seized control of {}. New leadership established.",
                    cw.rebel_leader_name, state.factions[fi].name
                ),
            });
        }

        CivilWarResolution::Stalemate => {
            // Faction splits into two if room permits (< 6 factions)
            if state.factions.len() < 6 {
                // Create splinter faction from rebels
                let new_id = state.factions.len();
                let original_name = state.factions[fi].name.clone();
                let new_name = format!("{} Separatists", original_name);

                // Split territory: give rebels some faction regions
                let faction_region_ids: Vec<usize> = state
                    .overworld
                    .regions
                    .iter()
                    .filter(|r| r.owner_faction_id == fi)
                    .map(|r| r.id)
                    .collect();

                // Give roughly half the regions to the new faction
                let split_count = faction_region_ids.len() / 2;
                for (i, &region_id) in faction_region_ids.iter().enumerate() {
                    if i < split_count {
                        if let Some(region) = state.overworld.regions.iter_mut().find(|r| r.id == region_id) {
                            region.owner_faction_id = new_id;
                        }
                    }
                }

                let new_faction = FactionState {
                    id: new_id,
                    name: new_name.clone(),
                    relationship_to_guild: 0.0,
                    military_strength: cw.rebel_strength * 0.7,
                    max_military_strength: state.factions[fi].max_military_strength * 0.6,
                    territory_size: split_count,
                    diplomatic_stance: DiplomaticStance::Neutral,
                    coalition_member: false,
                    at_war_with: vec![fi], // Hostile to original faction
                    has_guild: false,
                    guild_adventurer_count: 0,
                    recent_actions: Vec::new(),
                    relation: 0.0,
                    coup_risk: 0.0,
                    coup_cooldown: 0,
                    escalation_level: 0,
                    patrol_losses: 0,
                    escalation_cooldown: 0,
                    last_patrol_loss_tick: 0,
                    skill_modifiers: Default::default(),
                };

                state.factions.push(new_faction);

                // Original faction retains loyalist strength
                state.factions[fi].military_strength = cw.loyalist_strength * 0.7;
                state.factions[fi].at_war_with.push(new_id);

                // Expand diplomacy matrix
                let n = state.diplomacy.relations.len();
                for row in &mut state.diplomacy.relations {
                    row.push(0);
                }
                let mut new_row = vec![0i32; n + 1];
                new_row[fi] = -50; // Hostile to parent
                state.diplomacy.relations.push(new_row);
                state.diplomacy.relations[fi][new_id] = -50;

                events.push(WorldEvent::FactionSplit {
                    original_id: fi,
                    new_faction_name: new_name,
                });
            } else {
                // Can't split — loyalists win by default (exhaustion)
                state.factions[fi].military_strength =
                    (cw.loyalist_strength + cw.rebel_strength) * 0.4;

                events.push(WorldEvent::CivilWarResolved {
                    faction_id: fi,
                    rebels_won: false,
                    description: format!(
                        "Civil war in {} ended in exhaustion. Loyalists retain control.",
                        state.factions[fi].name
                    ),
                });
            }

            // Apply guild support consequences for stalemate
            if let Some(supported_rebels) = cw.guild_supported_side {
                // In a stalemate/split, supporting either side has mild negative effect
                // with the opposing side
                if supported_rebels {
                    state.factions[fi].relationship_to_guild =
                        (state.factions[fi].relationship_to_guild - 15.0).max(-100.0);
                } else {
                    // If faction split, rebels are now a new faction — they dislike us
                    if state.factions.len() > fi + 1 {
                        let last = state.factions.len() - 1;
                        state.factions[last].relationship_to_guild = -20.0;
                    }
                }
            }
        }
    }
}
