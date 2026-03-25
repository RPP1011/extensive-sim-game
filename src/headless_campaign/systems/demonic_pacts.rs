//! Demonic pacts system — contracts with demons accrue debt per tick with
//! escalating consequences, creating temptation/risk management dynamics.
//!
//! Fires every 200 ticks. Pacts are offered randomly to adventurers with low
//! morale or high ambition. Debt grows by interest rate each cycle and triggers
//! escalating effects at threshold levels.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the demonic pacts system ticks (in ticks).
const PACT_INTERVAL: u64 = 200;

/// Chance per eligible adventurer per cycle to be offered a pact.
const OFFER_CHANCE: f32 = 0.05;

/// Morale threshold below which an adventurer is eligible for a pact offer.
const LOW_MORALE_THRESHOLD: f32 = 40.0;

/// Debt threshold for minor side effects (nightmares, morale damage to allies).
const DEBT_MINOR: f32 = 0.3;

/// Debt threshold for moderate effects (visible corruption, faction standing loss).
const DEBT_MODERATE: f32 = 0.6;

/// Debt threshold for severe effects (risk of possession, ally attacks).
const DEBT_SEVERE: f32 = 0.9;

/// Debt threshold at which the demon collector arrives.
const DEBT_COLLECTOR: f32 = 1.0;

/// Morale penalty to nearby allies at minor debt level.
const MINOR_MORALE_PENALTY: f32 = 5.0;

/// Faction standing penalty at moderate debt level.
const MODERATE_FACTION_PENALTY: f32 = 10.0;

/// Chance of possession (attacking allies) at severe debt level per cycle.
const POSSESSION_CHANCE: f32 = 0.15;

/// Debt reduction from completing a purification quest.
pub const PURIFICATION_DEBT_REDUCTION: f32 = 0.3;

/// Gold cost per 0.1 debt reduction via sacrifice.
pub const SACRIFICE_GOLD_PER_POINT: f32 = 100.0;

/// Tick the demonic pacts system. Called every tick; internally gates on interval.
pub fn tick_demonic_pacts(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % PACT_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Offer new pacts to eligible adventurers ---
    offer_pacts(state, events);

    // --- Accrue debt on existing pacts ---
    accrue_debt(state);

    // --- Apply escalating consequences ---
    apply_consequences(state, events);
}

/// Offer pacts to adventurers with low morale (5% chance each per cycle).
fn offer_pacts(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect eligible adventurer ids first to avoid borrow issues.
    let eligible: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.status != AdventurerStatus::Dead
                && a.morale < LOW_MORALE_THRESHOLD
                // Don't offer if they already have an active (non-renounced) pact
                && !state
                    .demonic_pacts
                    .iter()
                    .any(|p| p.adventurer_id == a.id && !p.renounced)
        })
        .map(|a| a.id)
        .collect();

    for adv_id in eligible {
        let roll = lcg_f32(&mut state.rng);
        if roll < OFFER_CHANCE {
            // Power granted: 10-30% combat bonus
            let power = 10.0 + lcg_f32(&mut state.rng) * 20.0;
            // Interest rate: 0.02-0.08 per cycle
            let interest = 0.02 + lcg_f32(&mut state.rng) * 0.06;

            events.push(WorldEvent::DemonicPactOffered {
                adventurer_id: adv_id,
                power,
            });

            // Auto-accept with probability based on how low morale is.
            // Lower morale = higher acceptance chance.
            let adv_morale = state
                .adventurers
                .iter()
                .find(|a| a.id == adv_id)
                .map(|a| a.morale)
                .unwrap_or(50.0);
            let accept_chance = 1.0 - (adv_morale / LOW_MORALE_THRESHOLD);
            let accept_roll = lcg_f32(&mut state.rng);

            if accept_roll < accept_chance {
                let pact_id = state.next_pact_id;
                state.next_pact_id += 1;

                state.demonic_pacts.push(DemonicPact {
                    pact_id,
                    adventurer_id: adv_id,
                    power_granted: power,
                    debt: 0.0,
                    interest_rate: interest,
                    created_tick: state.tick as u32,
                    renounced: false,
                });

                events.push(WorldEvent::DemonicPactAccepted {
                    adventurer_id: adv_id,
                    pact_id,
                });
            }
        }
    }
}

/// Accrue debt on all active (non-renounced) pacts.
fn accrue_debt(state: &mut CampaignState) {
    for pact in &mut state.demonic_pacts {
        if !pact.renounced {
            pact.debt += pact.interest_rate;
        }
    }
}

/// Apply escalating consequences based on debt thresholds.
fn apply_consequences(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Work on indices to satisfy borrow checker.
    let pact_count = state.demonic_pacts.len();
    for i in 0..pact_count {
        let pact = &state.demonic_pacts[i];
        if pact.renounced {
            continue;
        }

        let debt = pact.debt;
        let adv_id = pact.adventurer_id;

        // --- Collector arrives at debt >= 1.0 ---
        if debt >= DEBT_COLLECTOR {
            events.push(WorldEvent::DemonicCollectorArrived {
                adventurer_id: adv_id,
            });

            // Spawn a nemesis-tier enemy targeting this adventurer
            let nemesis_id = state.nemeses.len() as u32;
            state.nemeses.push(Nemesis {
                id: nemesis_id,
                name: format!("Debt Collector of {}", adventurer_name(state, adv_id)),
                faction_id: 0,
                strength: 0.9,
                kills: 0,
                created_tick: state.tick,
                region_id: None,
                defeated: false,
            });

            // Mark pact as renounced (collector takes over, debt frozen)
            state.demonic_pacts[i].renounced = true;
            continue;
        }

        // --- Severe: risk of possession ---
        if debt >= DEBT_SEVERE {
            let roll = lcg_f32(&mut state.rng);
            if roll < POSSESSION_CHANCE {
                events.push(WorldEvent::DemonicDebtEscalated {
                    adventurer_id: adv_id,
                    debt_level: debt,
                    effect: "Possessed — attacking allies!".to_string(),
                });

                // Possession: injure a random party member
                if let Some(party) = state
                    .parties
                    .iter()
                    .find(|p| p.member_ids.contains(&adv_id))
                {
                    let allies: Vec<u32> = party
                        .member_ids
                        .iter()
                        .copied()
                        .filter(|&id| id != adv_id)
                        .collect();
                    if !allies.is_empty() {
                        let idx = (lcg_next(&mut state.rng) as usize) % allies.len();
                        let target_id = allies[idx];
                        if let Some(target) =
                            state.adventurers.iter_mut().find(|a| a.id == target_id)
                        {
                            target.injury = (target.injury + 20.0).min(100.0);
                            target.morale = (target.morale - 15.0).max(0.0);
                        }
                    }
                }
            } else {
                events.push(WorldEvent::DemonicDebtEscalated {
                    adventurer_id: adv_id,
                    debt_level: debt,
                    effect: "Severe corruption — risk of possession".to_string(),
                });
            }
        }
        // --- Moderate: faction standing loss ---
        else if debt >= DEBT_MODERATE {
            events.push(WorldEvent::DemonicDebtEscalated {
                adventurer_id: adv_id,
                debt_level: debt,
                effect: "Visible corruption — religious orders hostile".to_string(),
            });

            // Damage faction relations with religious-aligned factions
            for faction in &mut state.factions {
                faction.relationship_to_guild =
                    (faction.relationship_to_guild - MODERATE_FACTION_PENALTY / 10.0).max(-100.0);
            }

            // Reduce temple devotion
            for temple in &mut state.temples {
                temple.devotion = (temple.devotion - 2.0).max(-100.0);
            }
        }
        // --- Minor: nightmares, morale penalty to nearby allies ---
        else if debt >= DEBT_MINOR {
            events.push(WorldEvent::DemonicDebtEscalated {
                adventurer_id: adv_id,
                debt_level: debt,
                effect: "Nightmares — morale damage to nearby allies".to_string(),
            });

            // Morale penalty to party members
            if let Some(party) = state
                .parties
                .iter()
                .find(|p| p.member_ids.contains(&adv_id))
            {
                let allies: Vec<u32> = party
                    .member_ids
                    .iter()
                    .copied()
                    .filter(|&id| id != adv_id)
                    .collect();
                for ally_id in allies {
                    if let Some(ally) = state.adventurers.iter_mut().find(|a| a.id == ally_id) {
                        ally.morale = (ally.morale - MINOR_MORALE_PENALTY).max(0.0);
                    }
                }
            }
        }
    }
}

/// Helper to get an adventurer's name by id.
fn adventurer_name(state: &CampaignState, id: u32) -> String {
    state
        .adventurers
        .iter()
        .find(|a| a.id == id)
        .map(|a| a.name.clone())
        .unwrap_or_else(|| format!("Adventurer #{}", id))
}

/// Get the combat power multiplier for an adventurer with a demonic pact.
/// Returns 1.0 if no active pact.
pub fn demonic_combat_bonus(state: &CampaignState, adventurer_id: u32) -> f32 {
    state
        .demonic_pacts
        .iter()
        .find(|p| p.adventurer_id == adventurer_id && !p.renounced)
        .map(|p| 1.0 + p.power_granted / 100.0)
        .unwrap_or(1.0)
}

/// Reduce debt on a pact by completing a purification quest.
pub fn apply_purification(state: &mut CampaignState, adventurer_id: u32) {
    if let Some(pact) = state
        .demonic_pacts
        .iter_mut()
        .find(|p| p.adventurer_id == adventurer_id && !p.renounced)
    {
        pact.debt = (pact.debt - PURIFICATION_DEBT_REDUCTION).max(0.0);
    }
}

/// Reduce debt by sacrificing gold. Costs 100g per 0.1 debt reduced.
pub fn sacrifice_gold_for_debt(state: &mut CampaignState, adventurer_id: u32) -> f32 {
    let debt_reduction = 0.1;
    if state.guild.gold >= SACRIFICE_GOLD_PER_POINT {
        if let Some(pact) = state
            .demonic_pacts
            .iter_mut()
            .find(|p| p.adventurer_id == adventurer_id && !p.renounced)
        {
            state.guild.gold -= SACRIFICE_GOLD_PER_POINT;
            pact.debt = (pact.debt - debt_reduction).max(0.0);
            return debt_reduction;
        }
    }
    0.0
}

/// Renounce a pact — lose all power bonus, debt is frozen.
pub fn renounce_pact(state: &mut CampaignState, adventurer_id: u32) -> bool {
    if let Some(pact) = state
        .demonic_pacts
        .iter_mut()
        .find(|p| p.adventurer_id == adventurer_id && !p.renounced)
    {
        pact.renounced = true;
        true
    } else {
        false
    }
}
