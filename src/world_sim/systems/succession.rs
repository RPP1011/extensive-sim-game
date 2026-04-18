//! Succession crisis — when a settlement's strongest NPC dies, a power struggle.
//!
//! Each settlement's de facto leader is its highest-level NPC. When they die:
//! 1. A succession crisis begins (200 ticks)
//! 2. Top 2-3 candidates compete based on leadership+diplomacy tags
//! 3. Settlement morale drops during the crisis
//! 4. The winner becomes the new leader (gains leadership tags)
//! 5. The runner-up either accepts (loyalty oath) or rebels (betrayal path)
//!
//! Cadence: every 100 ticks.

use crate::world_sim::state::*;

const SUCCESSION_INTERVAL: u64 = 100;

pub fn advance_succession(state: &mut WorldState) {
    if state.tick % SUCCESSION_INTERVAL != 0 || state.tick == 0 { return; }

    let tick = state.tick;

    // Build a set of settlements that recently lost a high-level leader.
    // Previously this was re-scanned per settlement — O(S × W × E) where
    // S=settlements, W=world_events, E=entities. Now it's O(W) with O(1)
    // entity lookup via `state.entity()`.
    let mut settlements_with_leader_death: std::collections::HashSet<u32, ahash::RandomState> =
        std::collections::HashSet::default();
    for e in &state.world_events {
        if let WorldEvent::EntityDied { entity_id, .. } = e {
            if let Some(ent) = state.entity(*entity_id) {
                if !ent.alive {
                    if let Some(npc) = &ent.npc {
                        let total_level: u32 = npc.classes.iter().map(|c| c.level as u32).sum();
                        if total_level >= 10 {
                            if let Some(sid) = npc.home_settlement_id {
                                settlements_with_leader_death.insert(sid);
                            }
                        }
                    }
                }
            }
        }
    }
    if settlements_with_leader_death.is_empty() { return; }

    for si in 0..state.settlements.len() {
        let sid = state.settlements[si].id;
        if !settlements_with_leader_death.contains(&sid) { continue; }

        // Find all alive NPCs at this settlement, sorted by total class levels.
        let mut npcs_at_settlement: Vec<(usize, u32, u32)> = Vec::new(); // (entity_idx, entity_id, total_level)
        for (i, entity) in state.entities.iter().enumerate() {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &entity.npc { Some(n) => n, None => continue };
            if npc.home_settlement_id != Some(sid) { continue; }

            let total_level: u32 = npc.classes.iter().map(|c| c.level as u32).sum();
            npcs_at_settlement.push((i, entity.id, total_level));
        }

        if npcs_at_settlement.len() < 3 { continue; }

        // Sort by total level descending.
        npcs_at_settlement.sort_by(|a, b| b.2.cmp(&a.2));

        // --- Succession crisis ---
        // Top candidate becomes leader.
        let (winner_idx, winner_id, _winner_level) = npcs_at_settlement[0];
        let (runner_idx, runner_id, _runner_level) = npcs_at_settlement[1];

        let winner_name = state.entities[winner_idx].npc.as_ref()
            .map(|n| n.name.clone()).unwrap_or_default();
        let runner_name = state.entities[runner_idx].npc.as_ref()
            .map(|n| n.name.clone()).unwrap_or_default();
        let settlement_name = state.settlements[si].name.clone();

        // Winner gains leadership.
        if let Some(npc) = state.entities[winner_idx].npc.as_mut() {
            npc.emotions.pride = (npc.emotions.pride + 0.7).min(1.0);
            npc.needs.purpose = 100.0;
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::LEADERSHIP, 10.0);
                a.add(tags::DIPLOMACY, 5.0);
                a.add(tags::DISCIPLINE, 3.0);
                a
            });
        }

        // Runner-up decision: accept or rebel.
        let runner_compassion = state.entities[runner_idx].npc.as_ref()
            .map(|n| n.personality.compassion).unwrap_or(0.5);
        let runner_ambition = state.entities[runner_idx].npc.as_ref()
            .map(|n| n.personality.ambition).unwrap_or(0.5);

        let rebels = runner_ambition > 0.6 && runner_compassion < 0.4;

        if rebels {
            // Runner-up rebels! Becomes hostile.
            state.entities[runner_idx].team = WorldTeam::Hostile;
            if let Some(npc) = state.entities[runner_idx].npc.as_mut() {
                npc.home_settlement_id = None;
                npc.emotions.anger = 1.0;
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::DECEPTION, 3.0);
                    a.add(tags::COMBAT, 3.0);
                    a
                });
            }

            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Narrative,
                text: format!(
                    "Succession crisis in {}! {} claims leadership, but {} rebels and flees into exile!",
                    settlement_name, winner_name, runner_name),
                entity_ids: vec![winner_id, runner_id],
            });
        } else {
            // Runner-up accepts gracefully.
            if let Some(npc) = state.entities[runner_idx].npc.as_mut() {
                npc.emotions.grief = (npc.emotions.grief + 0.2).min(1.0);
                npc.needs.esteem -= 10.0;
                npc.accumulate_tags(&{
                    let mut a = ActionTags::empty();
                    a.add(tags::DISCIPLINE, 2.0);
                    a.add(tags::COMPASSION_TAG, 1.0);
                    a
                });
            }

            state.chronicle.push(ChronicleEntry {
                tick,
                category: ChronicleCategory::Achievement,
                text: format!(
                    "{} assumes leadership of {} after a peaceful succession. {} supports the transition.",
                    winner_name, settlement_name, runner_name),
                entity_ids: vec![winner_id, runner_id],
            });
        }

        // Settlement morale impact during transition.
        for entity in &mut state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &mut entity.npc {
                if npc.home_settlement_id == Some(sid) {
                    npc.morale = (npc.morale - 5.0).max(0.0);
                    npc.emotions.anxiety = (npc.emotions.anxiety + 0.2).min(1.0);
                }
            }
        }
    }
}
