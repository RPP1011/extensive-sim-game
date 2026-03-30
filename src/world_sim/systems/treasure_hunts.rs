//! Treasure hunt system — every 7 ticks.
//!
//! Multi-step treasure map quests. NPCs traveling across regions discover
//! treasure steps, earning escalating gold rewards. Final step yields a
//! large gold bonus and commodity goods (representing artifacts).
//!
//! **Gold conservation:** Rewards are paid from the NPC's home settlement
//! treasury. If the settlement cannot afford the reward, no gold is paid.
//!
//! Ported from `crates/headless_campaign/src/systems/treasure_hunts.rs`.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::{WorldState, WorldTeam};
use crate::world_sim::state::entity_hash;

/// How often the treasure hunt system ticks.
const TICK_INTERVAL: u64 = 7;

/// Distance threshold (squared) for discovering a treasure step.
const DISCOVERY_RADIUS_SQ: f32 = 400.0; // 20 units

/// Base gold reward per treasure discovery (scales with distance traveled).
const BASE_STEP_REWARD: f32 = 30.0;

/// Final step bonus multiplier.
const FINAL_STEP_MULTIPLIER: f32 = 3.0;

/// Commodity index for treasure goods (artifact equivalent).
const TREASURE_COMMODITY: usize = 7; // Last commodity slot

pub fn compute_treasure_hunts(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    // --- Treasure discovery via NPC exploration ---
    // Without explicit treasure map state, we model treasure hunts as:
    // NPCs traveling far from their home settlement discover treasure
    // proportional to the distance traveled. This captures the multi-step
    // nature of the original system (further = more reward).

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        for entity in &state.entities[range] {
            if !entity.alive || entity.team != WorldTeam::Friendly {
                continue;
            }

            let _npc = match &entity.npc {
                Some(n) => n,
                None => continue,
            };

            // Skip NPCs on combat grids (fighting, not treasure hunting)
            if let Some(gid) = entity.grid_id {
                if state
                    .grid(gid)
                    .map(|g| g.fidelity == crate::world_sim::fidelity::Fidelity::High)
                    .unwrap_or(false)
                {
                    continue;
                }
            }

            let home_id = settlement.id;

            let home_pos = settlement.pos;

            let dist_from_home_sq = dist_sq(entity.pos, home_pos);

            // NPCs far from home are "on treasure hunts"
            // Step rewards increase with distance (escalating like the original)
            if dist_from_home_sq < 625.0 {
                // 25^2: too close to home
                continue;
            }

            // Deterministic treasure discovery roll (entity id + tick based)
            let roll = (entity_hash(entity.id, state.tick, 0x78EA) % 1000) as f32 / 1000.0;

            // 5% chance per interval to find treasure
            if roll > 0.05 {
                continue;
            }

            // Reward scales with distance from home (escalating steps)
            let distance = dist_from_home_sq.sqrt();
            let distance_multiplier = (distance / 25.0).min(4.0); // cap at 4x
            let step_reward = BASE_STEP_REWARD * distance_multiplier;

            // Gold reward paid from home settlement treasury
            if settlement.treasury > step_reward {
                out.push(WorldDelta::TransferGold {
                    from_entity: settlement.id,
                    to_entity: entity.id,
                    amount: step_reward,
                });
            }

            // At max distance, also award treasure commodity (artifact equivalent)
            if distance_multiplier >= 3.0 {
                // Final step: large bonus + treasure goods
                let final_bonus = step_reward * FINAL_STEP_MULTIPLIER;
                if settlement.treasury > step_reward + final_bonus {
                    out.push(WorldDelta::TransferGold {
                        from_entity: settlement.id,
                        to_entity: entity.id,
                        amount: final_bonus,
                    });
                }

                // Award treasure commodity to NPC's home settlement
                out.push(WorldDelta::UpdateStockpile {
                    settlement_id: home_id,
                    commodity: TREASURE_COMMODITY,
                    delta: 5.0, // artifact-equivalent goods
                });
            }

            // Exploration near settlements: deliver findings to nearest settlement
            let nearest_settlement = state.settlements.iter().min_by(|a, b| {
                let da = dist_sq(entity.pos, a.pos);
                let db = dist_sq(entity.pos, b.pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(nearest) = nearest_settlement {
                let settlement_dist = dist_sq(entity.pos, nearest.pos);
                if settlement_dist <= DISCOVERY_RADIUS_SQ {
                    // NPC is near a settlement: deposit treasure findings
                    out.push(WorldDelta::UpdateTreasury {
                        settlement_id: nearest.id,
                        delta: step_reward * 0.1, // 10% goes to settlement
                    });
                }
            }
        }
    }
}

fn dist_sq(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}
