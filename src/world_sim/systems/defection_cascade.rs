#![allow(unused)]
//! Defection cascade system — every 7 ticks.
//!
//! When a high-status adventurer defects from a faction, it triggers a cascade
//! through loyalty graphs affecting morale, bonds, and faction standing.
//! Cascades are depth-limited to 3 hops to prevent total faction collapse.
//!
//! Ported from `crates/headless_campaign/src/systems/defection_cascade.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::WorldState;
use crate::world_sim::state::entity_hash_f32;

// NEEDS STATE: factions: Vec<FactionState> on WorldState
//   FactionState { id, relationship_to_guild }
// NEEDS STATE: adventurers: Vec<AdventurerState> on WorldState
//   AdventurerState { id, faction_id: Option<u32>, loyalty: f32, morale: f32,
//                     level: u32, status: AdventurerStatus, deeds: Vec<LegendaryDeed> }
// NEEDS STATE: adventurer_bonds: BondGraph on WorldState
//   bond_strength(bonds, a_id, b_id) -> f32
// NEEDS STATE: diplomacy: DiplomacyState on WorldState
//   diplomacy.relations: Vec<Vec<i32>>, guild_faction_id: u32
// NEEDS STATE: defection_events: Vec<DefectionEvent> on WorldState

// NEEDS DELTA: SetFaction { adventurer_id: u32, faction_id: Option<u32> }
// NEEDS DELTA: AdjustMorale { adventurer_id: u32, delta: f32 }
// NEEDS DELTA: AdjustRelationship { faction_id: u32, delta: f32 }
// NEEDS DELTA: RecordDefection { adventurer_id: u32, from_faction: u32, to_faction: u32,
//              cascade_count: u32, tick: u64 }

/// Maximum cascade depth to prevent total faction collapse.
const MAX_CASCADE_DEPTH: u32 = 3;

/// Cadence: fires every 7 ticks.
const DEFECTION_INTERVAL: u64 = 7;


pub fn compute_defection_cascade(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DEFECTION_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // Once state.adventurers, state.factions, state.adventurer_bonds,
    // state.diplomacy exist, enable this.

    /*
    let n_factions = state.factions.len();
    if n_factions < 2 {
        return;
    }

    // --- Phase 1: Identify candidates ---
    // Adventurers with: faction_id set (not guild), loyalty < 20,
    // faction relation < -30, and a "pull" faction with relation > 40.
    struct DefectionCandidate {
        adv_id: u32,
        from_faction: u32,
        to_faction: u32,
    }

    let mut candidates: Vec<DefectionCandidate> = Vec::new();

    for adv in &state.adventurers {
        if adv.status == AdventurerStatus::Dead {
            continue;
        }
        let from_faction = match adv.faction_id {
            Some(fid) => fid,
            None => continue,
        };

        if adv.loyalty >= 20.0 {
            continue;
        }

        let faction = match state.factions.iter().find(|f| f.id == from_faction) {
            Some(f) => f,
            None => continue,
        };
        if faction.relationship_to_guild >= -30.0 {
            continue;
        }

        // Find a "pull" faction with relation > 40
        let mut best_pull: Option<u32> = None;
        for other in &state.factions {
            if other.id == from_faction {
                continue;
            }
            let pull_relation = if (from_faction as usize) < state.diplomacy.relations.len()
                && (other.id as usize) < state.diplomacy.relations[from_faction as usize].len()
            {
                state.diplomacy.relations[other.id as usize][from_faction as usize]
            } else {
                other.relationship_to_guild as i32
            };
            if pull_relation > 40 {
                best_pull = Some(other.id);
                break;
            }
        }

        if let Some(to_faction) = best_pull {
            candidates.push(DefectionCandidate {
                adv_id: adv.id,
                from_faction,
                to_faction,
            });
        }
    }

    // --- Phase 2: Process each defection ---
    for candidate in &candidates {
        // Move the defector.
        out.push(WorldDelta::SetFaction {
            adventurer_id: candidate.adv_id,
            faction_id: Some(candidate.to_faction),
        });

        // Check if high-status (level > 5 or has legendary deeds).
        let high_status = state
            .adventurers
            .iter()
            .find(|a| a.id == candidate.adv_id)
            .map(|a| a.level > 5 || !a.deeds.is_empty())
            .unwrap_or(false);

        let mut cascade_count = 0u32;

        if high_status {
            // Cascade: bonded allies with bond > 50 and loyalty < 40 also defect.
            let cascade_ids = compute_cascade(
                state,
                candidate.adv_id,
                candidate.from_faction,
                candidate.to_faction,
                0,
                out,
            );
            cascade_count = cascade_ids.len() as u32;

            // Remaining adventurers in losing faction lose 5-15 morale.
            let morale_loss = 5.0
                + entity_hash_f32(candidate.adv_id, state.tick, 10 as u64) * 10.0;
            for adv in &state.adventurers {
                if adv.faction_id == Some(candidate.from_faction)
                    && adv.status != AdventurerStatus::Dead
                    && adv.id != candidate.adv_id
                    && !cascade_ids.contains(&adv.id)
                {
                    out.push(WorldDelta::AdjustMorale {
                        adventurer_id: adv.id,
                        delta: -morale_loss,
                    });
                }
            }

            // Faction standing drops by 10.
            out.push(WorldDelta::AdjustRelationship {
                faction_id: candidate.from_faction,
                delta: -10.0,
            });
        }

        // Record the defection.
        out.push(WorldDelta::RecordDefection {
            adventurer_id: candidate.adv_id,
            from_faction: candidate.from_faction,
            to_faction: candidate.to_faction,
            cascade_count,
            tick: state.tick,
        });
    }
    */
}

/*
/// Recursively compute cascade defections through bonded allies.
/// Returns the IDs of all adventurers who defected in the cascade.
/// Depth-limited to MAX_CASCADE_DEPTH (3) hops.
fn compute_cascade(
    state: &WorldState,
    trigger_id: u32,
    from_faction: u32,
    to_faction: u32,
    depth: u32,
    out: &mut Vec<WorldDelta>,
) -> Vec<u32> {
    if depth >= MAX_CASCADE_DEPTH {
        return Vec::new();
    }

    let cascade_candidates: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| {
            a.id != trigger_id
                && a.faction_id == Some(from_faction)
                && a.status != AdventurerStatus::Dead
                && a.loyalty < 40.0
                && bond_strength(&state.adventurer_bonds, trigger_id, a.id) > 50.0
        })
        .map(|a| a.id)
        .collect();

    let mut all_cascaded = Vec::new();

    for cid in cascade_candidates {
        out.push(WorldDelta::SetFaction {
            adventurer_id: cid,
            faction_id: Some(to_faction),
        });
        all_cascaded.push(cid);

        let sub = compute_cascade(state, cid, from_faction, to_faction, depth + 1, out);
        all_cascaded.extend(sub);
    }

    all_cascaded
}
*/
