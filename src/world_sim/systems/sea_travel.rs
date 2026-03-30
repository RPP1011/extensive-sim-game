//! Sea travel — NPCs at coastal settlements embark on voyages to other
//! coastal settlements. Faster than overland travel but risk of sea monsters.
//!
//! A voyage is a multi-tick journey: NPC sets EconomicIntent::Travel with
//! the destination coastal settlement's position. Movement speed is 2× overland.
//! Each tick at sea, there's a small chance of sea monster attack (damage).
//!
//! Eligibility: NPC at a coastal settlement with Trade or Relocate goal,
//! destination is also coastal. Voyage takes ~half the overland time.
//!
//! Cadence: every 20 ticks (voyage initiation), every tick (voyage progress).

use crate::world_sim::state::*;
use crate::world_sim::DT_SEC;

const VOYAGE_CHECK_INTERVAL: u64 = 20;
/// Sea travel speed multiplier vs overland.
const SEA_SPEED_MULT: f32 = 2.0;
/// Chance per tick of sea monster attack during voyage (0-1).
const SEA_MONSTER_ATTACK_CHANCE: f32 = 0.005;
/// Damage from sea monster encounter.
const SEA_MONSTER_DAMAGE: f32 = 15.0;
/// Max distance to consider "arrived" at coastal destination.
const ARRIVAL_DIST_SQ: f32 = 400.0;

/// Advance sea voyages. Called post-apply from runtime.rs.
pub fn advance_sea_travel(state: &mut WorldState) {
    let tick = state.tick;

    // --- Phase 1: Initiate voyages (every 20 ticks) ---
    // NPCs with Trade goal at coastal settlements heading to another coastal
    // settlement should take the sea route.
    if tick % VOYAGE_CHECK_INTERVAL == 0 && tick > 0 {
        // Collect coastal settlement IDs and positions.
        let coastal_settlements: Vec<(u32, (f32, f32))> = state.settlements.iter()
            .filter(|s| {
                // Settlement is coastal if its region is coastal.
                state.regions.iter().any(|r| {
                    r.is_coastal && state.settlements.iter().any(|s2| s2.id == s.id)
                })
            })
            .map(|s| (s.id, s.pos))
            .collect();

        if coastal_settlements.len() >= 2 {
            for entity in &mut state.entities {
                if !entity.alive || entity.kind != EntityKind::Npc { continue; }
                let npc = match &mut entity.npc { Some(n) => n, None => continue };

                // Must be at a coastal settlement.
                let home_sid = match npc.home_settlement_id { Some(id) => id, None => continue };
                let is_home_coastal = coastal_settlements.iter().any(|(id, _)| *id == home_sid);
                if !is_home_coastal { continue; }

                // Must have a Trade or Travel intent to another settlement.
                let dest_sid = match &npc.economic_intent {
                    EconomicIntent::Trade { destination_settlement_id } => Some(*destination_settlement_id),
                    _ => continue,
                };
                let dest_sid = match dest_sid { Some(id) => id, None => continue };

                // Destination must also be coastal.
                let dest_coastal = coastal_settlements.iter().find(|(id, _)| *id == dest_sid);
                let dest_pos = match dest_coastal { Some((_, pos)) => *pos, None => continue };

                // Already heading there? Check if voyage is faster.
                let home_pos = coastal_settlements.iter()
                    .find(|(id, _)| *id == home_sid)
                    .map(|(_, p)| *p)
                    .unwrap_or(entity.pos);

                let overland_dist = {
                    let dx = dest_pos.0 - home_pos.0;
                    let dy = dest_pos.1 - home_pos.1;
                    (dx * dx + dy * dy).sqrt()
                };

                // Sea route is worth it for distances > 30 units.
                if overland_dist > 30.0 {
                    // Mark as sea voyage by setting a unique goal target.
                    if let Some(goal) = npc.goal_stack.current_mut() {
                        goal.target_pos = Some(dest_pos);
                    }
                    // Chronicle: departure.
                    if entity_hash_f32(entity.id, tick, 0x5EA1) < 0.1 { // 10% get chronicled
                        // Will be added when we have access to names.
                    }
                }
            }
        }
    }

    // --- Phase 2: Move voyaging NPCs at sea speed ---
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        // Check if NPC is on a sea voyage (Trade intent + coastal home + distant destination).
        let dest = match &npc.economic_intent {
            EconomicIntent::Trade { destination_settlement_id } => {
                state.settlements.iter()
                    .find(|s| s.id == *destination_settlement_id)
                    .map(|s| s.pos)
            }
            _ => continue,
        };
        let dest = match dest { Some(d) => d, None => continue };

        // Only apply sea speed if NPC is far from both home and destination
        // (i.e., "at sea" — not near any settlement).
        let near_any_settlement = state.settlements.iter().any(|s| {
            let dx = s.pos.0 - entity.pos.0;
            let dy = s.pos.1 - entity.pos.1;
            dx * dx + dy * dy < 900.0 // within 30 units
        });

        if near_any_settlement { continue; } // still near land, normal movement

        // Accumulate seafaring tags while at sea.
        if tick % 10 == 0 {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::SEAFARING, 1.0);
                a.add(tags::NAVIGATION, 0.5);
                a.add(tags::SURVIVAL, 0.3);
                a
            });
        }

        // At sea: move at 2× speed.
        let dx = dest.0 - entity.pos.0;
        let dy = dest.1 - entity.pos.1;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < 1.0 { continue; }

        let speed = entity.move_speed * DT_SEC * SEA_SPEED_MULT;
        entity.pos.0 += dx / dist * speed;
        entity.pos.1 += dy / dist * speed;

        // Sea monster attack chance.
        let roll = entity_hash_f32(entity.id, tick, 0x5EA2);
        if roll < SEA_MONSTER_ATTACK_CHANCE {
            entity.hp -= SEA_MONSTER_DAMAGE;
            if entity.hp <= 0.0 {
                entity.alive = false;
                // Chronicle: lost at sea.
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Death,
                    text: format!("{} was lost at sea, claimed by the deep.",
                        npc.name),
                    entity_ids: vec![entity.id],
                });
            } else {
                // Survived a sea monster attack — record memory + seafaring tags.
                crate::world_sim::systems::agent_inner::record_npc_event(
                    npc, MemEventType::WasAttacked,
                    entity.pos, vec![], -0.5, tick,
                );
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::SEAFARING, 3.0);   // sea battle experience
                    a.add(tags::COMBAT, 2.0);
                    a.add(tags::SURVIVAL, 2.0);
                    a.add(tags::RESILIENCE, 1.0);
                    a
                });
            }
        }
    }
}
