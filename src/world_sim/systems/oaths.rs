//! Oath system — NPCs swear, fulfill, and break oaths.
//!
//! Oath types: Loyalty (to settlement), Vengeance (against grudge target),
//! Protection (defend settlement). Stored on NpcData.oaths.
//!
//! Fulfilling oaths → faith+discipline tags → Oathkeeper class.
//! Breaking oaths (betrayal) → deception tags → Oathbreaker class.
//!
//! Cadence: every 200 ticks (post-apply).

use serde::{Serialize, Deserialize};
use crate::world_sim::state::*;

const OATH_INTERVAL: u64 = 200;
const OATH_CHANCE: f32 = 0.05;

/// An oath sworn by an NPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Oath {
    pub kind: OathKind,
    pub target_id: u32,
    pub sworn_tick: u64,
    pub fulfilled: bool,
    pub broken: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OathKind {
    Loyalty,
    Vengeance,
    Protection,
}

/// Process oath swearing, fulfillment, and breaking. Called post-apply.
pub fn advance_oaths(state: &mut WorldState) {
    if state.tick % OATH_INTERVAL != 0 || state.tick == 0 { return; }
    let tick = state.tick;

    let entity_count = state.entities.len();

    // --- Phase 1: Swear new oaths ---
    for i in 0..entity_count {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if npc.oaths.len() >= 3 { continue; }
        let sid = match npc.home_settlement_id { Some(s) => s, None => continue };

        let roll = entity_hash_f32(entity.id, tick, 0x0A7B);
        if roll > OATH_CHANCE { continue; }

        let oath_type = entity_hash(entity.id, tick, 0x0A7C) % 3;
        let new_oath = match oath_type {
            0 => {
                let already = npc.oaths.iter().any(|o| o.kind == OathKind::Loyalty && !o.fulfilled && !o.broken);
                if already { continue; }
                Some(Oath { kind: OathKind::Loyalty, target_id: sid, sworn_tick: tick, fulfilled: false, broken: false })
            }
            1 => {
                let grudge_target = npc.memory.beliefs.iter()
                    .find_map(|b| match &b.belief_type { BeliefType::Grudge(tid) => Some(*tid), _ => None });
                let already = npc.oaths.iter().any(|o| o.kind == OathKind::Vengeance && !o.fulfilled && !o.broken);
                if already { continue; }
                grudge_target.map(|tid| Oath { kind: OathKind::Vengeance, target_id: tid, sworn_tick: tick, fulfilled: false, broken: false })
            }
            _ => {
                let already = npc.oaths.iter().any(|o| o.kind == OathKind::Protection && !o.fulfilled && !o.broken);
                if already { continue; }
                Some(Oath { kind: OathKind::Protection, target_id: sid, sworn_tick: tick, fulfilled: false, broken: false })
            }
        };

        if let Some(oath) = new_oath {
            let npc = state.entities[i].npc.as_mut().unwrap();
            npc.oaths.push(oath);
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::FAITH, 1.0);
                a.add(tags::DISCIPLINE, 0.5);
                a
            });
        }
    }

    // --- Phase 2: Check fulfillment ---
    for i in 0..state.entities.len() {
        let entity = &state.entities[i];
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        let mut to_fulfill: Vec<usize> = Vec::new();
        for (oi, oath) in npc.oaths.iter().enumerate() {
            if oath.fulfilled || oath.broken { continue; }
            let fulfilled = match &oath.kind {
                OathKind::Vengeance => !state.entities.iter().any(|e| e.id == oath.target_id && e.alive),
                OathKind::Protection => tick.saturating_sub(oath.sworn_tick) >= 1000,
                OathKind::Loyalty => tick.saturating_sub(oath.sworn_tick) >= 1500
                    && npc.home_settlement_id == Some(oath.target_id),
            };
            if fulfilled { to_fulfill.push(oi); }
        }

        if to_fulfill.is_empty() { continue; }

        let entity_id = entity.id;
        let npc_name = npc.name.clone();
        let npc = state.entities[i].npc.as_mut().unwrap();

        for &oi in &to_fulfill {
            npc.oaths[oi].fulfilled = true;
            npc.emotions.pride = (npc.emotions.pride + 0.6).min(1.0);
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::FAITH, 3.0);
                a.add(tags::DISCIPLINE, 2.0);
                a.add(tags::RESILIENCE, 1.0);
                a
            });
        }

        let total_fulfilled = npc.oaths.iter().filter(|o| o.fulfilled).count();
        if total_fulfilled >= 3 && to_fulfill.len() > 0 {
            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Achievement,
                text: format!("{} has fulfilled three oaths — a true Oathkeeper.", npc_name),
                entity_ids: vec![entity_id],
            });
        }
    }

    // --- Phase 3: Detect broken oaths (hostile betrayers) ---
    for entity in &mut state.entities {
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        if entity.team != WorldTeam::Hostile { continue; }
        let npc = match &mut entity.npc { Some(n) => n, None => continue };

        let mut any_broken = false;
        for oath in &mut npc.oaths {
            if oath.fulfilled || oath.broken { continue; }
            if oath.kind == OathKind::Loyalty {
                oath.broken = true;
                any_broken = true;
            }
        }
        if any_broken {
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::DECEPTION, 3.0);
                a
            });
        }
    }
}

// Keep stub for backward compat with compute_all_systems.
pub fn compute_oaths(_state: &WorldState, _out: &mut Vec<crate::world_sim::delta::WorldDelta>) {
    // No-op — oaths handled by advance_oaths (post-apply).
}
