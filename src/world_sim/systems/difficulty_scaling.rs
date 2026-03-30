#![allow(unused)]
//! Dynamic difficulty scaling — every 10 ticks.
//!
//! Monitors guild power (entity count, levels, economy) and adjusts pressure.
//! If dominant: escalation (Damage to treasury, boost hostile threat).
//! If struggling: relief (Heal, TransferGold aid, reduce threat).
//!
//! Ported from `crates/headless_campaign/src/systems/difficulty_scaling.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{WorldState, EntityKind, WorldTeam};

//              consecutive_wins, consecutive_losses, scaling_events_triggered }

/// How often (in ticks) the difficulty system evaluates.
const SCALING_INTERVAL: u64 = 10;

/// Rate at which current_pressure moves toward target per evaluation.
const PRESSURE_SMOOTHING: f32 = 2.0;

/// Power rating above which escalation triggers.
const DOMINANT_THRESHOLD: f32 = 70.0;

/// Power rating below which relief triggers.
const STRUGGLING_THRESHOLD: f32 = 30.0;

pub fn compute_difficulty_scaling(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % SCALING_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Compute guild power rating (0-100) ---
    let alive_npcs: Vec<&crate::world_sim::state::Entity> = state
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Npc && e.alive && e.team == WorldTeam::Friendly)
        .collect();

    let npc_count = alive_npcs.len() as f32;
    if npc_count == 0.0 {
        // No friendly NPCs — cannot compute power, skip.
        return;
    }

    let avg_level = alive_npcs.iter().map(|e| e.level as f32).sum::<f32>() / npc_count;

    // Manpower score: count * avg_level, normalized to 0-30
    let manpower_score = (npc_count * avg_level / 40.0 * 30.0).min(30.0);

    // Economy score: sum of settlement treasuries, normalized to 0-20
    let total_treasury: f32 = state.settlements.iter().map(|s| s.treasury).sum();
    let gold_score = (total_treasury / 500.0 * 20.0).min(20.0);

    // Territory score: fraction of regions with low threat, normalized to 0-20
    let safe_regions = state
        .regions
        .iter()
        .filter(|r| r.threat_level < 30.0)
        .count() as f32;
    let total_regions = state.regions.len().max(1) as f32;
    let territory_score = (safe_regions / total_regions) * 20.0;

    // Population score: settlement populations, normalized to 0-15
    let total_pop: u32 = state.settlements.iter().map(|s| s.population).sum();
    let pop_score = (total_pop as f32 / 200.0 * 15.0).min(15.0);

    // Monster density penalty: high monster density lowers power rating
    let avg_monster_density = if state.regions.is_empty() {
        0.0
    } else {
        state.regions.iter().map(|r| r.monster_density).sum::<f32>() / state.regions.len() as f32
    };
    let monster_penalty = (avg_monster_density * 15.0).min(15.0);

    let power_rating =
        (manpower_score + gold_score + territory_score + pop_score - monster_penalty)
            .clamp(0.0, 100.0);

    // --- Compute target pressure ---
    let mut target = 50.0_f32;

    if power_rating > DOMINANT_THRESHOLD {
        let excess = (power_rating - DOMINANT_THRESHOLD) / (100.0 - DOMINANT_THRESHOLD);
        target += excess * 30.0;
    } else if power_rating < STRUGGLING_THRESHOLD {
        let deficit = (STRUGGLING_THRESHOLD - power_rating) / STRUGGLING_THRESHOLD;
        target -= deficit * 30.0;
    }

    let target_pressure = target.clamp(0.0, 100.0);

    // --- Apply escalation or relief via deltas ---
    // We cannot read difficulty_scaling.current_pressure from state (field doesn't exist yet),
    // so we use the power rating directly to decide whether to escalate or relieve.

    if power_rating > DOMINANT_THRESHOLD {
        apply_escalation(state, power_rating, out);
    } else if power_rating < STRUGGLING_THRESHOLD {
        apply_relief(state, power_rating, out);
    }
}

/// Escalation: guild is dominant, increase pressure via economic penalties.
fn apply_escalation(state: &WorldState, power_rating: f32, out: &mut Vec<WorldDelta>) {
    // Trade disruption: drain 10% from each settlement treasury (only positive treasury)
    for settlement in &state.settlements {
        if settlement.treasury <= 0.0 {
            continue;
        }
        let loss = (settlement.treasury * 0.1).min(80.0);
        if loss > 0.0 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: -loss,
            });
        }
    }

    // Damage to the weakest friendly NPC (represents increased monster aggression)
    let weakest = state
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::Npc && e.alive && e.team == WorldTeam::Friendly)
        .min_by(|a, b| a.hp.partial_cmp(&b.hp).unwrap_or(std::cmp::Ordering::Equal));

    if let Some(npc) = weakest {
        let escalation_damage = (power_rating - DOMINANT_THRESHOLD) * 0.5;
        if escalation_damage > 0.0 {
            out.push(WorldDelta::Damage {
                target_id: npc.id,
                amount: escalation_damage,
                source_id: 0, // system-generated
            });
        }
    }
}

/// Relief: guild is struggling, provide aid via healing and gold.
fn apply_relief(state: &WorldState, power_rating: f32, out: &mut Vec<WorldDelta>) {
    // Heal all injured friendly NPCs
    for entity in &state.entities {
        if entity.kind == EntityKind::Npc
            && entity.alive
            && entity.team == WorldTeam::Friendly
            && entity.hp < entity.max_hp
        {
            let heal = (entity.max_hp - entity.hp) * 0.2; // Heal 20% of missing HP
            if heal > 0.0 {
                out.push(WorldDelta::Heal {
                    target_id: entity.id,
                    amount: heal,
                    source_id: 0, // system-generated
                });
            }
        }
    }

    // Gold aid to struggling settlements
    let relief_amount = (STRUGGLING_THRESHOLD - power_rating) * 1.5;
    for settlement in &state.settlements {
        if settlement.treasury < 50.0 && relief_amount > 0.0 {
            out.push(WorldDelta::UpdateTreasury {
                location_id: settlement.id,
                delta: relief_amount,
            });
        }
    }
}
