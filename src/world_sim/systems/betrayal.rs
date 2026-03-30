//! Betrayal system — NPCs with treacherous profiles become villains.
//!
//! NPCs with stealth/deception tags > 80 and compassion < 0.3 may:
//! - Steal gold from settlement treasury
//! - Betray their adventuring party (steal loot, flee)
//! - Switch factions to a hostile one
//! - Become outlaws with bounties
//!
//! Betrayal creates grudges across the settlement, chronicle entries,
//! and the Betrayer/Villain class path.
//!
//! Cadence: every 200 ticks.

use crate::world_sim::state::*;

const BETRAYAL_INTERVAL: u64 = 200;
/// Minimum stealth+deception tags to consider betrayal.
const TREACHERY_THRESHOLD: f32 = 50.0;
/// Compassion below which betrayal is possible.
const COMPASSION_CAP: f32 = 0.35;
/// Base chance of betrayal per eligible NPC per check.
const BETRAYAL_CHANCE: f32 = 0.05;
/// Gold stolen as fraction of settlement treasury.
const THEFT_FRACTION: f32 = 0.1;

pub fn advance_betrayal(state: &mut WorldState) {
    if state.tick % BETRAYAL_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;
    let mut betrayals: Vec<(usize, u32, String)> = Vec::new(); // (entity_idx, settlement_id, npc_name)

    for (i, entity) in state.entities.iter().enumerate() {
        if betrayals.len() >= 2 { break; } // max 2 per tick
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        let stealth = npc.behavior_value(tags::STEALTH);
        let deception = npc.behavior_value(tags::DECEPTION);
        let treachery = stealth + deception;
        if treachery < TREACHERY_THRESHOLD { continue; }
        if npc.personality.compassion >= COMPASSION_CAP { continue; }

        let sid = match npc.home_settlement_id { Some(id) => id, None => continue };

        // Higher treachery + lower compassion = higher chance.
        let chance = BETRAYAL_CHANCE
            * (treachery / TREACHERY_THRESHOLD)
            * (1.0 - npc.personality.compassion);

        let roll = entity_hash_f32(entity.id, tick, 0xBE78);
        if roll < chance {
            betrayals.push((i, sid, npc.name.clone()));
        }
    }

    for (entity_idx, sid, betrayer_name) in &betrayals {
        let entity_id = state.entities[*entity_idx].id;
        let _entity_pos = state.entities[*entity_idx].pos;

        // Steal gold from treasury.
        let stolen = state.settlements.iter()
            .find(|s| s.id == *sid)
            .map(|s| (s.treasury * THEFT_FRACTION).max(0.0).min(100.0))
            .unwrap_or(0.0);

        if stolen > 0.0 {
            if let Some(settlement) = state.settlements.iter_mut().find(|s| s.id == *sid) {
                settlement.treasury -= stolen;
            }
        }

        let settlement_name = state.settlements.iter()
            .find(|s| s.id == *sid)
            .map(|s| s.name.clone())
            .unwrap_or_default();

        // Modify the betrayer.
        let entity = &mut state.entities[*entity_idx];
        entity.team = WorldTeam::Hostile;
        if let Some(npc) = &mut entity.npc {
            npc.gold += stolen;
            npc.faction_id = None;
            npc.party_id = None;
            npc.home_settlement_id = None;
            npc.home_building_id = None;
            npc.work_building_id = None;
            npc.work_state = WorkState::Idle;
            npc.economic_intent = EconomicIntent::Idle;
            npc.emotions.pride = (npc.emotions.pride + 0.3).min(1.0);
            npc.emotions.anxiety = (npc.emotions.anxiety + 0.5).min(1.0);
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::DECEPTION, 5.0);
                a.add(tags::STEALTH, 3.0);
                a
            });
        }

        // Chronicle.
        state.chronicle.push(ChronicleEntry {
            tick,
            category: ChronicleCategory::Narrative,
            text: format!(
                "{} of {} has betrayed the settlement! They stole {:.0} gold and fled into the wilderness.",
                betrayer_name, settlement_name, stolen),
            entity_ids: vec![entity_id],
        });
    }

    // Apply grudges to settlement residents (separate loop to avoid borrow conflict).
    for (_, sid, _) in &betrayals {
        let betrayer_ids: Vec<u32> = betrayals.iter()
            .filter(|(_, s, _)| s == sid)
            .map(|(idx, _, _)| state.entities[*idx].id)
            .collect();

        for entity in &mut state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };
            if npc.home_settlement_id != Some(*sid) { continue; }

            for &bid in &betrayer_ids {
                if entity.id == bid { continue; }
                npc.emotions.anger = (npc.emotions.anger + 0.4).min(1.0);
                let has_grudge = npc.memory.beliefs.iter().any(|b|
                    matches!(b.belief_type, BeliefType::Grudge(gid) if gid == bid));
                if !has_grudge {
                    npc.memory.beliefs.push(Belief {
                        belief_type: BeliefType::Grudge(bid),
                        confidence: 1.0,
                        formed_tick: tick,
                    });
                }
            }
        }
    }
}
