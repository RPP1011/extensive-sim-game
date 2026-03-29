//! Coup probability engine — fires every 300 ticks.
//!
//! Tracks running coup-risk scores for each faction based on internal
//! instability factors. When the accumulated risk crosses 0.7 and a
//! random roll succeeds, a coup is triggered: the leader is replaced,
//! faction relations reset, and the faction suffers temporary instability.
//! The guild's standing shifts depending on its relationship with the
//! old or new leadership.

use crate::actions::{StepDeltas, WorldEvent};
use crate::state::*;

/// How often the coup engine evaluates risk (in ticks).
const COUP_CHECK_INTERVAL: u64 = 10;

/// Risk threshold above which a coup can fire.
const COUP_RISK_THRESHOLD: f32 = 0.7;

/// Ticks of instability after a coup (−20% effectiveness).
const COUP_INSTABILITY_DURATION: u32 = 1000;

/// Cadenced system — fires every 300 ticks.
pub fn tick_coup_engine(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % COUP_CHECK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    let n_factions = state.factions.len();
    let guild_faction_id = state.diplomacy.guild_faction_id;

    for fi in 0..n_factions {
        // Skip the guild faction.
        if fi == guild_faction_id {
            continue;
        }

        // Tick down cooldown.
        if state.factions[fi].coup_cooldown > 0 {
            let decrement = COUP_CHECK_INTERVAL as u32;
            state.factions[fi].coup_cooldown =
                state.factions[fi].coup_cooldown.saturating_sub(decrement);
            continue;
        }

        // Skip factions in active civil war (separate system handles that).
        if state.civil_wars.iter().any(|cw| cw.faction_id == fi) {
            continue;
        }

        // Compute coup risk factors (each contributes 0.0–0.2, max total 1.0).
        let risk = compute_coup_risk(state, fi);
        state.factions[fi].coup_risk = risk;

        // Emit rising-risk warning at notable thresholds.
        if risk >= 0.5 {
            events.push(WorldEvent::CoupRiskRising {
                faction_id: fi,
                risk_level: risk,
            });
        }

        // Attempt coup if risk exceeds threshold.
        if risk > COUP_RISK_THRESHOLD {
            let roll = lcg_f32(&mut state.rng);
            // Probability scales with excess risk: (risk - 0.7) mapped to 0-0.3 window.
            let coup_chance = (risk - COUP_RISK_THRESHOLD).min(0.3);
            if roll < coup_chance {
                execute_coup(state, fi, events);
            }
        }
    }
}

/// Compute the aggregate coup-risk score for a faction.
///
/// Five factors, each contributing 0.0–0.2:
///   1. Leader unpopularity (low relationship_to_guild as proxy for internal dissent)
///   2. Military concentration (single-general-too-powerful heuristic)
///   3. Treasury deficit (negative gold trend — low military strength as proxy)
///   4. Recent defeats (lost battles / forced ceasefires)
///   5. Vassal resentment (high unrest in owned regions)
fn compute_coup_risk(state: &CampaignState, fi: usize) -> f32 {
    let faction = &state.factions[fi];
    let mut risk = 0.0_f32;

    // 1. Leader unpopularity — low internal cohesion proxied by guild relation
    //    being very negative (faction is badly managed / internationally isolated).
    //    Map relation from [-100, 100] → risk [0.2, 0.0].
    let unpopularity = ((-faction.relationship_to_guild - 20.0) / 80.0).clamp(0.0, 1.0);
    risk += unpopularity * 0.2;

    // 2. Military concentration — if strength is very high relative to max,
    //    generals feel emboldened. Also triggers if strength is very low (power vacuum).
    let strength_ratio = if faction.max_military_strength > 0.0 {
        faction.military_strength / faction.max_military_strength
    } else {
        0.5
    };
    // Risk peaks at extremes: very strong generals (>0.9) or power vacuum (<0.3).
    let concentration = if strength_ratio > 0.9 {
        (strength_ratio - 0.9) * 10.0 // 0.0–1.0
    } else if strength_ratio < 0.3 {
        (0.3 - strength_ratio) / 0.3 // 0.0–1.0
    } else {
        0.0
    };
    risk += concentration.clamp(0.0, 1.0) * 0.2;

    // 3. Treasury deficit — factions with low military strength are under-funded.
    //    Map strength [0, 30] → risk [0.2, 0.0].
    let deficit = ((30.0 - faction.military_strength) / 30.0).clamp(0.0, 1.0);
    risk += deficit * 0.2;

    // 4. Recent defeats — forced ceasefires or military exhaustion.
    let recent_defeats = faction
        .recent_actions
        .iter()
        .filter(|a| {
            a.tick + 2000 >= state.tick
                && (a.action.contains("ceasefire")
                    || a.action.contains("exhausted")
                    || a.action.contains("Captured region"))
        })
        .count();
    let defeat_factor = (recent_defeats as f32 / 3.0).clamp(0.0, 1.0);
    risk += defeat_factor * 0.2;

    // 5. Vassal resentment — average unrest across owned regions.
    let faction_regions: Vec<&Region> = state
        .overworld
        .regions
        .iter()
        .filter(|r| r.owner_faction_id == fi)
        .collect();
    let avg_unrest = if !faction_regions.is_empty() {
        faction_regions.iter().map(|r| r.unrest).sum::<f32>() / faction_regions.len() as f32
    } else {
        0.0
    };
    let resentment = (avg_unrest / 100.0).clamp(0.0, 1.0);
    risk += resentment * 0.2;

    risk.clamp(0.0, 1.0)
}

/// Execute a coup in the given faction.
fn execute_coup(
    state: &mut CampaignState,
    fi: usize,
    events: &mut Vec<WorldEvent>,
) {
    // Determine success — coups with very high risk are more likely to succeed.
    let success_roll = lcg_f32(&mut state.rng);
    let success = success_roll < 0.6 + (state.factions[fi].coup_risk - COUP_RISK_THRESHOLD) * 0.5;

    // Pick a new leader ID deterministically.
    let new_leader_id = (lcg_next(&mut state.rng) as usize) % 1000;

    if success {
        let old_relation = state.factions[fi].relationship_to_guild;

        // Leader replaced — relations reset to 0 (new government).
        state.factions[fi].relationship_to_guild = 0.0;

        // Diplomatic stance resets to neutral.
        state.factions[fi].diplomatic_stance = DiplomaticStance::Neutral;
        state.factions[fi].coalition_member = false;
        state.factions[fi].at_war_with.clear();

        // Military strength reduced by instability.
        state.factions[fi].military_strength *= 0.8;

        // Set cooldown to prevent rapid successive coups.
        state.factions[fi].coup_cooldown = COUP_INSTABILITY_DURATION;

        // Reset coup risk after successful coup.
        state.factions[fi].coup_risk = 0.0;

        // Apply unrest to faction regions (temporary instability).
        for region in &mut state.overworld.regions {
            if region.owner_faction_id == fi {
                region.unrest = (region.unrest + 15.0).min(100.0);
                // −20% control effectiveness during instability.
                region.control = (region.control * 0.8).max(0.0);
            }
        }

        // Guild involvement consequences.
        if old_relation > 30.0 {
            // Guild was allied with old leader — lose standing with new regime.
            state.factions[fi].relationship_to_guild = -15.0;
        }

        events.push(WorldEvent::CoupAttempted {
            faction_id: fi,
            success: true,
            new_leader_id,
        });
    } else {
        // Failed coup — faction cracks down, becomes more authoritarian.
        // Relations worsen slightly (paranoia), but faction stabilizes.
        state.factions[fi].relationship_to_guild =
            (state.factions[fi].relationship_to_guild - 10.0).max(-100.0);

        // Military purge costs some strength.
        state.factions[fi].military_strength *= 0.9;

        // Cooldown is shorter for a failed attempt.
        state.factions[fi].coup_cooldown = COUP_INSTABILITY_DURATION / 2;

        // Risk partially reduced (dissidents purged).
        state.factions[fi].coup_risk *= 0.5;

        // Unrest increases from the crackdown.
        for region in &mut state.overworld.regions {
            if region.owner_faction_id == fi {
                region.unrest = (region.unrest + 8.0).min(100.0);
            }
        }

        events.push(WorldEvent::CoupAttempted {
            faction_id: fi,
            success: false,
            new_leader_id,
        });
    }
}
