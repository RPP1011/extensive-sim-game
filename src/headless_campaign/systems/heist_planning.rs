//! Heist planning system — fires every 300 ticks.
//!
//! Multi-tick heist preparation with blueprint/scouting/roles, creating
//! sequential decision problems with risk/reward tradeoffs. Each heist
//! progresses through five phases (Planning → Scouting → Infiltration →
//! Execution → Escape), with success depending on preparation quality,
//! crew skill, and risk level.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// How often the heist system ticks (in ticks).
const HEIST_INTERVAL: u64 = 300;

/// Ticks each phase lasts before advancing.
const PHASE_DURATION: u64 = 500;

/// Maximum number of concurrent active heists.
const MAX_ACTIVE_HEISTS: usize = 2;

/// Minimum tick before heists can spawn.
const MIN_HEIST_TICK: u64 = 1500;

/// Base prep score gained per phase tick.
const BASE_PREP_PER_PHASE: f32 = 0.1;

/// Faction relation penalty on heist failure.
const FAILURE_RELATION_PENALTY: f32 = 15.0;

/// Heat increase on successful heist.
const SUCCESS_HEAT_INCREASE: f32 = 20.0;

/// Tick the heist planning system every `HEIST_INTERVAL` ticks.
pub fn tick_heist_planning(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % HEIST_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.tick < MIN_HEIST_TICK {
        return;
    }

    // --- Progress active heists ---
    progress_heists(state, events);

    // --- Maybe spawn a new heist opportunity ---
    if state.active_heists.len() < MAX_ACTIVE_HEISTS {
        maybe_spawn_heist(state, events);
    }
}

/// Progress all active heists through their phases.
fn progress_heists(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Collect updates to avoid borrow conflicts.
    let mut completed: Vec<(usize, bool)> = Vec::new(); // (index, succeeded)
    let mut phase_advances: Vec<(usize, HeistPhase)> = Vec::new();

    for (idx, heist) in state.active_heists.iter().enumerate() {
        let ticks_in_phase = state.tick.saturating_sub(heist.started_tick);

        // Calculate crew skill factor from adventurers
        let crew_skill = compute_crew_skill(state, &heist.crew_ids);

        // Check if phase should advance
        let phase_complete = ticks_in_phase >= PHASE_DURATION;

        if phase_complete {
            match heist.phase {
                HeistPhase::Planning
                | HeistPhase::Scouting
                | HeistPhase::Infiltration => {
                    let next = heist.phase.next();
                    phase_advances.push((idx, next));
                }
                HeistPhase::Execution => {
                    // Resolve the heist
                    let success_prob =
                        heist.prep_score * (1.0 - heist.risk_level) * crew_skill;
                    let roll = lcg_f32(&mut state.rng);
                    completed.push((idx, roll < success_prob));
                }
                HeistPhase::Escape => {
                    // Escape phase resolves as success (already passed execution)
                    completed.push((idx, true));
                }
            }
        }
    }

    // Apply phase advances
    for &(idx, ref new_phase) in &phase_advances {
        let crew_ids = state.active_heists[idx].crew_ids.clone();
        let crew_skill = compute_crew_skill(state, &crew_ids);
        let heist = &mut state.active_heists[idx];

        // Accumulate prep score based on phase
        let phase_bonus = match new_phase {
            HeistPhase::Scouting => 0.15,      // scouting adds moderate prep
            HeistPhase::Infiltration => 0.20,   // infiltration adds more
            HeistPhase::Execution => 0.10,      // execution is the test
            _ => 0.05,
        };
        heist.prep_score = (heist.prep_score + BASE_PREP_PER_PHASE + phase_bonus * crew_skill)
            .min(1.0);
        heist.phase = *new_phase;
        heist.started_tick = state.tick;

        events.push(WorldEvent::HeistPhaseAdvanced {
            heist_id: heist.heist_id,
            new_phase: *new_phase,
        });
    }

    // Resolve completed heists (process in reverse to safely remove)
    let mut resolved: Vec<(u32, bool, f32, Vec<u32>, usize)> = Vec::new();
    for &(idx, succeeded) in completed.iter().rev() {
        let heist = &state.active_heists[idx];
        resolved.push((
            heist.heist_id,
            succeeded,
            heist.reward_estimate,
            heist.crew_ids.clone(),
            heist.target_faction,
        ));
    }

    for (heist_id, succeeded, reward, crew_ids, target_faction) in resolved {
        state.active_heists.retain(|h| h.heist_id != heist_id);

        if succeeded {
            // Success: gain gold and increase heat
            state.guild.gold += reward;
            state.black_market.heat += SUCCESS_HEAT_INCREASE;

            events.push(WorldEvent::HeistSucceeded {
                heist_id,
                loot_value: reward,
            });
        } else {
            // Failure: some crew may be captured
            let mut captured_ids = Vec::new();
            for &cid in &crew_ids {
                let roll = lcg_f32(&mut state.rng);
                if roll < 0.4 {
                    // 40% chance each crew member is captured
                    captured_ids.push(cid);
                    if let Some(adv) = state
                        .adventurers
                        .iter_mut()
                        .find(|a| a.id == cid)
                    {
                        adv.status = AdventurerStatus::Dead;
                    }
                    // Track as captured
                    if !state.captured_adventurers.contains(&cid) {
                        state.captured_adventurers.push(cid);
                    }
                }
            }

            // Damage faction relations
            if let Some(faction) = state
                .factions
                .iter_mut()
                .find(|f| f.id == target_faction)
            {
                faction.relationship_to_guild =
                    (faction.relationship_to_guild - FAILURE_RELATION_PENALTY).max(-100.0);
            }

            events.push(WorldEvent::HeistFailed {
                heist_id,
                captured_ids,
            });
        }
    }
}

/// Maybe spawn a new heist opportunity based on espionage intel or black market contacts.
fn maybe_spawn_heist(state: &mut CampaignState, events: &mut Vec<WorldEvent>) {
    // Heist opportunities arise from: high intel on a faction, or black market contacts
    let has_intel = state.spies.iter().any(|s| s.intel_gathered > 50.0);
    let has_contacts = state.black_market.total_profit > 100.0;

    if !has_intel && !has_contacts {
        return;
    }

    // Spawn chance: 20% per tick when conditions met
    let roll = lcg_f32(&mut state.rng);
    if roll > 0.2 {
        return;
    }

    // Pick a target faction (prefer ones we have intel on)
    if state.factions.is_empty() {
        return;
    }

    let target_idx = if has_intel {
        // Target the faction with the most intel
        let best_spy = state
            .spies
            .iter()
            .max_by(|a, b| a.intel_gathered.partial_cmp(&b.intel_gathered).unwrap());
        match best_spy {
            Some(spy) => spy.target_faction_id,
            None => (lcg_next(&mut state.rng) as usize) % state.factions.len(),
        }
    } else {
        (lcg_next(&mut state.rng) as usize) % state.factions.len()
    };

    let target_faction = state
        .factions
        .get(target_idx)
        .map(|f| f.id)
        .unwrap_or(0);

    // Select crew: pick up to 3 idle adventurers, prefer rogues
    let mut crew_ids: Vec<u32> = Vec::new();
    let mut candidates: Vec<(u32, u32, bool)> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Idle)
        .map(|a| {
            let is_rogue = a.archetype.to_lowercase().contains("rogue");
            (a.id, a.level, is_rogue)
        })
        .collect();

    // Sort: rogues first, then by level descending
    candidates.sort_by(|a, b| {
        b.2.cmp(&a.2)
            .then_with(|| b.1.cmp(&a.1))
    });

    for (id, _, _) in candidates.into_iter().take(3) {
        crew_ids.push(id);
    }

    if crew_ids.is_empty() {
        return; // No available crew
    }

    // Calculate reward based on faction wealth/power and guild tier
    let faction_strength = state
        .factions
        .get(target_idx)
        .map(|f| f.military_strength)
        .unwrap_or(50.0);
    let tier_mult = 1.0 + (state.guild.reputation / 100.0) * 0.5;
    let base_reward = 80.0 + faction_strength * 0.5;
    let reward_estimate = base_reward * tier_mult;

    // Risk scales with faction hostility and military power
    let faction_hostile = state
        .factions
        .get(target_idx)
        .map(|f| {
            matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            )
        })
        .unwrap_or(false);
    let base_risk = if faction_hostile { 0.5 } else { 0.3 };
    let risk_level = (base_risk + faction_strength / 500.0).min(0.9);

    let heist_id = state.next_heist_id;
    state.next_heist_id += 1;

    state.active_heists.push(HeistPlan {
        heist_id,
        target_faction,
        phase: HeistPhase::Planning,
        prep_score: 0.0,
        crew_ids,
        reward_estimate,
        risk_level,
        started_tick: state.tick,
    });

    events.push(WorldEvent::HeistPhaseAdvanced {
        heist_id,
        new_phase: HeistPhase::Planning,
    });
}

/// Compute aggregate crew skill factor (0.5 - 1.5) from adventurer levels and archetypes.
fn compute_crew_skill(state: &CampaignState, crew_ids: &[u32]) -> f32 {
    if crew_ids.is_empty() {
        return 0.5;
    }

    let mut total_skill = 0.0;
    let mut count = 0u32;

    for &cid in crew_ids {
        if let Some(adv) = state.adventurers.iter().find(|a| a.id == cid) {
            let level_bonus = (adv.level as f32 - 1.0) * 0.05; // +5% per level above 1
            let rogue_bonus = if adv.archetype.to_lowercase().contains("rogue") {
                0.3
            } else {
                0.0
            };
            total_skill += 1.0 + level_bonus + rogue_bonus;
            count += 1;
        }
    }

    if count == 0 {
        return 0.5;
    }

    (total_skill / count as f32).clamp(0.5, 1.5)
}
