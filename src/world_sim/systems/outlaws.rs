#![allow(unused)]
//! Outlaw system — exiled NPCs form bandit camps, raid caravans, seek redemption.
//!
//! Hostile NPCs (from betrayal or faction conflict) without a settlement become
//! outlaws. Outlaws:
//! - Cluster together to form bandit camps (3+ outlaws within 30 units)
//! - Raid nearby trading NPCs (steal carried goods)
//! - Can seek redemption: if they kill a monster or return gold, they may
//!   petition a settlement for re-acceptance
//!
//! Cadence: every 50 ticks.

use crate::world_sim::state::*;
use crate::world_sim::systems::agent_inner::record_npc_event;

const OUTLAW_INTERVAL: u64 = 50;
const RAID_RANGE_SQ: f32 = 400.0; // 20 units
const RAID_CHANCE: f32 = 0.1;
const CAMP_RANGE_SQ: f32 = 900.0; // 30 units
const REDEMPTION_GOLD_THRESHOLD: f32 = 50.0;
const REDEMPTION_CHANCE: f32 = 0.03;

pub fn advance_outlaws(state: &mut WorldState) {
    if state.tick % OUTLAW_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Collect outlaw info: (entity_idx, pos, gold).
    let outlaws: Vec<(usize, u32, (f32, f32))> = state.entities.iter().enumerate()
        .filter(|(_, e)| e.alive && e.kind == EntityKind::Npc && e.team == WorldTeam::Hostile)
        .filter(|(_, e)| e.npc.as_ref().map(|n| n.home_settlement_id.is_none()).unwrap_or(false))
        .map(|(i, e)| (i, e.id, e.pos))
        .collect();

    if outlaws.is_empty() { return; }

    // --- Phase 1: Raiding — outlaws attack nearby traders ---
    let mut raids: Vec<(usize, usize)> = Vec::new(); // (outlaw_idx, victim_idx)

    for &(oi, oid, opos) in &outlaws {
        let roll = entity_hash_f32(oid, tick, 0x8A1D);
        if roll > RAID_CHANCE { continue; }

        // Find nearest trading NPC.
        for (vi, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if entity.team != WorldTeam::Friendly { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };
            if !matches!(npc.economic_intent, EconomicIntent::Trade { .. }) { continue; }

            let dx = entity.pos.0 - opos.0;
            let dy = entity.pos.1 - opos.1;
            if dx * dx + dy * dy < RAID_RANGE_SQ {
                raids.push((oi, vi));
                break;
            }
        }
    }

    // Execute raids — collect data first, then mutate.
    let raid_data: Vec<(usize, usize, u32, u32, f32, f32)> = raids.iter()
        .map(|&(oi, vi)| {
            let outlaw_id = state.entities[oi].id;
            let victim_id = state.entities[vi].id;
            let stolen = state.entities[vi].npc.as_ref()
                .map(|n| (n.gold * 0.3).min(20.0)).unwrap_or(0.0);
            let dmg = state.entities[oi].attack_damage * 0.5;
            (oi, vi, outlaw_id, victim_id, stolen, dmg)
        })
        .collect();

    for (oi, vi, outlaw_id, _victim_id, stolen, dmg) in raid_data {
        let victim_pos = state.entities[vi].pos;
        if stolen > 0.0 {
            if let Some(npc) = state.entities[vi].npc.as_mut() {
                npc.gold -= stolen;
                record_npc_event(npc, MemEventType::WasAttacked,
                    victim_pos, vec![outlaw_id], -0.5, tick);
                npc.memory.beliefs.push(Belief {
                    belief_type: BeliefType::Grudge(outlaw_id),
                    confidence: 1.0,
                    formed_tick: tick,
                });
            }
            if let Some(npc) = state.entities[oi].npc.as_mut() {
                npc.gold += stolen;
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::DECEPTION, 1.0);
                    a.add(tags::STEALTH, 1.0);
                    a
                });
            }
        }
        state.entities[vi].hp -= dmg;
    }

    // --- Phase 2: Bandit camp detection ---
    // When 3+ outlaws cluster, record a chronicle event (once per cluster).
    if tick % 200 == 0 {
        let mut counted: Vec<u32> = Vec::new();
        for &(_, oid, opos) in &outlaws {
            if counted.contains(&oid) { continue; }
            let nearby: Vec<u32> = outlaws.iter()
                .filter(|&&(_, nid, npos)| {
                    if nid == oid { return false; }
                    let dx = npos.0 - opos.0;
                    let dy = npos.1 - opos.1;
                    dx * dx + dy * dy < CAMP_RANGE_SQ
                })
                .map(|&(_, nid, _)| nid)
                .collect();

            if nearby.len() >= 2 { // 3+ total including self
                counted.push(oid);
                counted.extend(&nearby);
                state.chronicle.push(ChronicleEntry {
                    tick,
                    category: ChronicleCategory::Narrative,
                    text: format!("A bandit camp of {} outlaws has formed in the wilderness.",
                        nearby.len() + 1),
                    entity_ids: std::iter::once(oid).chain(nearby.into_iter()).collect(),
                });
                break; // one camp event per tick
            }
        }
    }

    // --- Phase 3: Redemption — outlaws with enough gold seek re-acceptance ---
    for &(oi, oid, opos) in &outlaws {
        let npc = match &state.entities[oi].npc { Some(n) => n, None => continue };
        if npc.gold < REDEMPTION_GOLD_THRESHOLD { continue; }

        let roll = entity_hash_f32(oid, tick, 0xDE3D);
        if roll > REDEMPTION_CHANCE { continue; }

        // Find nearest friendly settlement to petition.
        let nearest_settlement = state.settlements.iter()
            .min_by_key(|s| {
                let dx = s.pos.0 - opos.0;
                let dy = s.pos.1 - opos.1;
                (dx * dx + dy * dy) as u32
            });

        let settlement = match nearest_settlement { Some(s) => s, None => continue };
        let sid = settlement.id;
        let settlement_name = settlement.name.clone();

        // Redemption: pay gold, switch back to friendly, join settlement.
        let redemption_cost = REDEMPTION_GOLD_THRESHOLD;
        let entity = &mut state.entities[oi];
        entity.team = WorldTeam::Friendly;
        if let Some(npc) = &mut entity.npc {
            npc.gold -= redemption_cost;
            npc.home_settlement_id = Some(sid);
            npc.economic_intent = EconomicIntent::Produce;
            npc.emotions.grief = (npc.emotions.grief + 0.3).min(1.0); // regret
            npc.emotions.joy = (npc.emotions.joy + 0.5).min(1.0); // relief
            npc.personality.compassion = (npc.personality.compassion + 0.1).min(1.0); // reformed

            // Accumulate redemption tags.
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::RESILIENCE, 3.0);
                a.add(tags::COMPASSION_TAG, 2.0);
                a.add(tags::SURVIVAL, 2.0);
                a
            });

            let outlaw_name = npc.name.clone();
            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Narrative,
                text: format!("{} has been redeemed and accepted into {}. Their outlaw days are over.",
                    outlaw_name, settlement_name),
                entity_ids: vec![oid],
            });
        }
    }
}
