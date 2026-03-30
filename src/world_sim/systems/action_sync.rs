//! Action sync — updates NpcAction from current state each tick.
//!
//! Derives the visible action from WorkState, GoalStack, movement, and combat.
//! This is a read-from-state, write-to-action pass — no deltas needed.
//!
//! Cadence: every tick.

use crate::world_sim::state::*;

/// Sync NpcAction from current NPC state.
/// Called post-apply from runtime.rs.
pub fn sync_npc_actions(state: &mut WorldState) {
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // Derive action from work state (highest fidelity source).
        let new_action = match &npc.work_state {
            WorkState::TravelingToWork { target_pos } => {
                NpcAction::Walking { destination: *target_pos }
            }
            WorkState::Working { building_id, ticks_remaining } => {
                let activity = npc.work_building_id
                    .and_then(|_wid| {
                        // Can't look up building type here (no state access for entities),
                        // so infer from behavior tags.
                        let farming = npc.behavior_value(tags::FARMING);
                        let mining = npc.behavior_value(tags::MINING);
                        let smithing = npc.behavior_value(tags::SMITHING);
                        let woodwork = npc.behavior_value(tags::WOODWORK);
                        let alchemy = npc.behavior_value(tags::ALCHEMY);
                        let research = npc.behavior_value(tags::RESEARCH);

                        let max = farming.max(mining).max(smithing).max(woodwork).max(alchemy).max(research);
                        if max <= 0.0 { return None; }
                        if max == farming { Some(WorkActivity::Farming) }
                        else if max == mining { Some(WorkActivity::Mining) }
                        else if max == smithing { Some(WorkActivity::Smithing) }
                        else if max == woodwork { Some(WorkActivity::Logging) }
                        else if max == alchemy { Some(WorkActivity::Brewing) }
                        else if max == research { Some(WorkActivity::Researching) }
                        else { None }
                    })
                    .unwrap_or(WorkActivity::Crafting);

                NpcAction::Working {
                    ticks_remaining: *ticks_remaining,
                    building_id: *building_id,
                    activity,
                }
            }
            WorkState::CarryingToStorage { commodity, amount, .. } => {
                NpcAction::Hauling { commodity: *commodity, amount: *amount }
            }
            WorkState::Idle => {
                // Check goal stack for non-work actions.
                match npc.goal_stack.current_kind() {
                    GoalKind::Eat => {
                        NpcAction::Eating {
                            ticks_remaining: 5,
                            building_id: npc.goal_stack.current()
                                .and_then(|g| g.target_entity)
                                .unwrap_or(0),
                        }
                    }
                    GoalKind::Fight => {
                        NpcAction::Fighting { target_id: 0 } // target resolved elsewhere
                    }
                    GoalKind::Socialize => {
                        NpcAction::Socializing { partner_id: 0, ticks_remaining: 10 }
                    }
                    GoalKind::Rest => {
                        NpcAction::Resting { ticks_remaining: 20 }
                    }
                    GoalKind::Flee { .. } => NpcAction::Fleeing,
                    GoalKind::Trade { .. } => {
                        NpcAction::Trading { ticks_remaining: 10 }
                    }
                    GoalKind::Build { building_id } => {
                        NpcAction::Building { building_id: *building_id, ticks_remaining: 30 }
                    }
                    _ => {
                        // Check if walking (has cached path).
                        if !npc.cached_path.is_empty() {
                            let dest = npc.goal_stack.current()
                                .and_then(|g| g.target_pos)
                                .unwrap_or(entity.pos);
                            NpcAction::Walking { destination: dest }
                        } else {
                            NpcAction::Idle
                        }
                    }
                }
            }
        };

        npc.action = new_action;
    }
}
