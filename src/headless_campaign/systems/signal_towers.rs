//! Signal tower system — every 200 ticks.
//!
//! Chains of signal towers relay beacon codes across regions. Towers can be
//! captured, destroyed, or fed false signals by enemy espionage.
//!
//! Coverage bonus: regions with an operational tower get +20% scouting accuracy
//! and -30% surprise attack chance.
//!
//! Sabotage: enemy espionage can compromise a tower (feeds false signals for
//! 500 ticks before detected).
//!
//! Destruction: battles in a tower's region have a 30% chance of damaging it.
//!
//! Repair: costs gold per tower, takes 300 ticks.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: runs every 200 ticks.
const SIGNAL_TOWER_INTERVAL: u64 = 7;

/// Chance that a battle in a tower's region destroys the tower (0–1).
const BATTLE_DAMAGE_CHANCE: f32 = 0.30;

/// Gold cost to repair a destroyed tower.
const REPAIR_COST: f32 = 25.0;

/// Ticks required to complete a repair.
const REPAIR_TICKS: u64 = 10;

/// Ticks a compromised tower feeds false signals before detection.
const COMPROMISE_DURATION: u64 = 17;

/// Chance per tick that an enemy spy compromises an operational tower (0–1).
const COMPROMISE_CHANCE: f32 = 0.05;

/// Scouting accuracy bonus for regions with an operational tower.
pub const SCOUTING_BONUS: f32 = 0.20;

/// Surprise attack chance reduction for regions with an operational tower.
pub const SURPRISE_REDUCTION: f32 = 0.30;

pub fn tick_signal_towers(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % SIGNAL_TOWER_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.signal_towers.is_empty() {
        return;
    }

    // --- Battle damage: check for active battles in tower regions ---
    // Collect region ids that have active battles (using battle location mapped
    // to the closest region by index).
    {
        let battle_region_ids: Vec<u32> = state
            .active_battles
            .iter()
            .map(|b| {
                // Map battle location to nearest region by minimising squared distance
                // to region id. Since Region has no center coordinate, we approximate
                // by using the battle's quest party destination or just use the
                // region id modulo the number of regions based on battle id.
                // Simpler: iterate regions and match by owner faction proximity.
                // Best simple heuristic: hash battle location into a region index.
                let num_regions = state.overworld.regions.len().max(1) as f32;
                let hashed = ((b.location.0.abs() + b.location.1.abs()) * 7.0) as u32;
                hashed % (num_regions as u32)
            })
            .collect();

        let mut destroyed_ids = Vec::new();
        for tower in &mut state.signal_towers {
            if !tower.operational {
                continue;
            }
            if battle_region_ids.contains(&tower.region_id) {
                let roll = lcg_f32(&mut state.rng);
                if roll < BATTLE_DAMAGE_CHANCE {
                    tower.operational = false;
                    tower.compromised = false;
                    destroyed_ids.push((tower.tower_id, tower.region_id));
                }
            }
        }

        for (tower_id, region_id) in destroyed_ids {
            events.push(WorldEvent::TowerDestroyed {
                tower_id,
                region_id,
            });
        }
    }

    // --- Espionage compromise: hostile factions can compromise towers ---
    {
        let has_hostile_factions = state.factions.iter().any(|f| {
            matches!(
                f.diplomatic_stance,
                DiplomaticStance::Hostile | DiplomaticStance::AtWar
            )
        });

        if has_hostile_factions {
            let mut compromised_events = Vec::new();
            for tower in &mut state.signal_towers {
                if !tower.operational || tower.compromised {
                    continue;
                }

                // Check if the tower's region is owned by a hostile faction.
                let region_hostile = state
                    .overworld
                    .regions
                    .get(tower.region_id as usize)
                    .map(|r| {
                        state.factions.iter().any(|f| {
                            f.id == r.owner_faction_id
                                && matches!(
                                    f.diplomatic_stance,
                                    DiplomaticStance::Hostile | DiplomaticStance::AtWar
                                )
                        })
                    })
                    .unwrap_or(false);

                if region_hostile {
                    let roll = lcg_f32(&mut state.rng);
                    if roll < COMPROMISE_CHANCE {
                        tower.compromised = true;
                        tower.compromised_at = Some(state.tick);
                        // Find the hostile faction responsible.
                        let by_faction = state
                            .factions
                            .iter()
                            .find(|f| {
                                matches!(
                                    f.diplomatic_stance,
                                    DiplomaticStance::Hostile | DiplomaticStance::AtWar
                                )
                            })
                            .map(|f| f.id)
                            .unwrap_or(0);
                        compromised_events.push((tower.tower_id, by_faction));
                    }
                }
            }

            for (tower_id, by_faction) in compromised_events {
                events.push(WorldEvent::TowerCompromised {
                    tower_id,
                    by_faction,
                });
            }
        }
    }

    // --- False signal detection: compromised towers detected after 500 ticks ---
    {
        let mut detected_ids = Vec::new();
        for tower in &mut state.signal_towers {
            if tower.compromised {
                if let Some(compromised_at) = tower.compromised_at {
                    if state.tick >= compromised_at + COMPROMISE_DURATION {
                        tower.compromised = false;
                        tower.compromised_at = None;
                        detected_ids.push(tower.tower_id);
                    }
                }
            }
        }

        for tower_id in detected_ids {
            events.push(WorldEvent::FalseSignalDetected { tower_id });
        }
    }

    // --- Repair: towers under repair progress toward completion ---
    {
        let mut repaired_ids = Vec::new();
        for tower in &mut state.signal_towers {
            if !tower.operational {
                if let Some(repair_start) = tower.repair_started_at {
                    if state.tick >= repair_start + REPAIR_TICKS {
                        tower.operational = true;
                        tower.repair_started_at = None;
                        repaired_ids.push(tower.tower_id);
                    }
                }
            }
        }

        for tower_id in repaired_ids {
            events.push(WorldEvent::SignalRelayed {
                from_tower: tower_id,
                to_tower: tower_id,
                signal_type: SignalType::Repaired,
            });
        }
    }

    // --- Signal relay: operational towers relay signals along chains ---
    // Two towers can relay signals if they are in the same region or in
    // neighboring regions (within `range` hops via the region neighbor graph).
    // For simplicity, we check direct region adjacency (range=1 hop) or same
    // region. The tower's `range` field scales: range >= region distance.
    {
        // Pre-collect neighbor lists to avoid borrow conflicts.
        let region_neighbors: Vec<Vec<usize>> = state
            .overworld
            .regions
            .iter()
            .map(|r| r.neighbors.clone())
            .collect();

        let towers_snapshot: Vec<SignalTower> = state.signal_towers.clone();

        for tower in &towers_snapshot {
            if !tower.operational {
                continue;
            }

            let t_rid = tower.region_id as usize;

            for other in &towers_snapshot {
                if other.tower_id == tower.tower_id || !other.operational {
                    continue;
                }
                // Must be same faction.
                if other.owner_faction != tower.owner_faction {
                    continue;
                }

                let o_rid = other.region_id as usize;

                // Check if in range: same region or within `range` neighbor hops.
                // Simple check: same region (distance 0) or adjacent (distance 1).
                // Tower range > 1 allows one extra hop per range unit above 1.
                let in_range = if t_rid == o_rid {
                    true
                } else if let Some(neighbors) = region_neighbors.get(t_rid) {
                    if neighbors.contains(&o_rid) {
                        true // Adjacent region, range >= 1 is enough.
                    } else if tower.range > 1 {
                        // Check 2-hop neighbors for extended range.
                        neighbors.iter().any(|&n| {
                            region_neighbors
                                .get(n)
                                .map(|nn| nn.contains(&o_rid))
                                .unwrap_or(false)
                        })
                    } else {
                        false
                    }
                } else {
                    false
                };

                if in_range {
                    let signal = if tower.compromised {
                        SignalType::FalseAlert
                    } else {
                        SignalType::EarlyWarning
                    };
                    events.push(WorldEvent::SignalRelayed {
                        from_tower: tower.tower_id,
                        to_tower: other.tower_id,
                        signal_type: signal,
                    });
                }
            }
        }
    }

    // --- Coverage bonus: boost scouting for regions with operational towers ---
    {
        for tower in &state.signal_towers {
            if tower.operational && !tower.compromised {
                if let Some(region) = state
                    .overworld
                    .regions
                    .get_mut(tower.region_id as usize)
                {
                    region.visibility = (region.visibility + SCOUTING_BONUS).min(1.0);
                }
            }
        }
    }
}

/// Start repairing a destroyed tower. Returns true if repair was initiated.
pub fn start_repair(state: &mut CampaignState, tower_id: u32) -> bool {
    if let Some(tower) = state
        .signal_towers
        .iter_mut()
        .find(|t| t.tower_id == tower_id)
    {
        if tower.operational || tower.repair_started_at.is_some() {
            return false; // Already operational or already repairing.
        }
        if state.guild.gold < REPAIR_COST {
            return false; // Not enough gold.
        }
        state.guild.gold -= REPAIR_COST;
        tower.repair_started_at = Some(state.tick);
        true
    } else {
        false
    }
}
