//! Adventurer secret past system.
//!
//! Some NPCs have hidden histories that emerge over time based on their
//! behavior profile and circumstances. Since we don't store explicit
//! secret state, secrets are derived deterministically from NPC id and
//! behavior patterns. When conditions align, a "reveal" fires — a one-time
//! event that modifies the NPC and records a chronicle entry.
//!
//! Cadence: every 100 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::*;

const SECRETS_INTERVAL: u64 = 100;

/// A "secret" fires when an NPC crosses a behavior threshold that indicates
/// a hidden history. The reveal window is narrow so it fires once.
const REVEAL_WINDOW: f32 = 10.0;


pub fn compute_secrets(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % SECRETS_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_secrets_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_secrets_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % SECRETS_INTERVAL != 0 || state.tick == 0 { return; }

    for entity in entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        // Derive secret type from entity id (deterministic — always the same).
        let secret_type = entity.id % 8;

        // Check reveal conditions based on behavior profile thresholds.
        // Each secret type has a different trigger.
        match secret_type {
            0 => {
                // "Former Assassin" — revealed when stealth + combat > 300
                let stealth = npc.behavior_value(tags::STEALTH);
                let combat = npc.behavior_value(tags::COMBAT);
                let total = stealth + combat;
                if total >= 300.0 && total < 300.0 + REVEAL_WINDOW {
                    // Combat skill boost from hidden training.
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: entity.id,
                        field: EntityField::Morale,
                        value: -5.0, // guilt/shame
                    });
                    let mut action = ActionTags::empty();
                    action.add(tags::COMBAT, 5.0);
                    action.add(tags::STEALTH, 3.0);
                    action.add(tags::TACTICS, 2.0);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: entity.id,
                        tags: action.tags,
                        count: action.count,
                    });
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Discovery,
                            text: format!("{} revealed a past as a trained assassin", entity_display_name(entity)),
                            entity_ids: vec![entity.id],
                        },
                    });
                }
            }
            1 => {
                // "Exiled Noble" — revealed when diplomacy + leadership > 200
                let diplomacy = npc.behavior_value(tags::DIPLOMACY);
                let leadership = npc.behavior_value(tags::LEADERSHIP);
                let total = diplomacy + leadership;
                if total >= 200.0 && total < 200.0 + REVEAL_WINDOW {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: entity.id,
                        field: EntityField::Morale,
                        value: 10.0,
                    });
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Discovery,
                            text: format!("{} was revealed as an exiled noble", entity_display_name(entity)),
                            entity_ids: vec![entity.id],
                        },
                    });
                }
            }
            2 => {
                // "Cursed Bloodline" — revealed when resilience + faith > 250
                let resilience = npc.behavior_value(tags::RESILIENCE);
                let faith = npc.behavior_value(tags::FAITH);
                let total = resilience + faith;
                if total >= 250.0 && total < 250.0 + REVEAL_WINDOW {
                    // Powerful but destabilizing.
                    let mut action = ActionTags::empty();
                    action.add(tags::RESILIENCE, 10.0);
                    action.add(tags::COMBAT, 3.0);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: entity.id,
                        tags: action.tags,
                        count: action.count,
                    });
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: entity.id,
                        field: EntityField::Morale,
                        value: -8.0, // burden of the curse
                    });
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Discovery,
                            text: format!("{} bears the mark of a cursed bloodline", entity_display_name(entity)),
                            entity_ids: vec![entity.id],
                        },
                    });
                }
            }
            3 => {
                // "Hidden Mage" — revealed when research + lore > 300
                let research = npc.behavior_value(tags::RESEARCH);
                let lore = npc.behavior_value(tags::LORE);
                let total = research + lore;
                if total >= 300.0 && total < 300.0 + REVEAL_WINDOW {
                    let mut action = ActionTags::empty();
                    action.add(tags::RESEARCH, 5.0);
                    action.add(tags::LORE, 5.0);
                    action.add(tags::ALCHEMY, 3.0);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: entity.id,
                        tags: action.tags,
                        count: action.count,
                    });
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Discovery,
                            text: format!("{} revealed hidden magical talents", entity_display_name(entity)),
                            entity_ids: vec![entity.id],
                        },
                    });
                }
            }
            4 => {
                // "Deep Cover Spy" — revealed when stealth + deception > 200 and stress > 50
                let stealth = npc.behavior_value(tags::STEALTH);
                let deception = npc.behavior_value(tags::DECEPTION);
                let total = stealth + deception;
                if total >= 200.0 && total < 200.0 + REVEAL_WINDOW && npc.stress > 50.0 {
                    // Betrayal: steal settlement gold.
                    let stolen = (npc.gold * 0.3).min(50.0);
                    if stolen > 1.0 {
                        out.push(WorldDelta::UpdateTreasury {
                            settlement_id: settlement_id,
                            delta: -stolen,
                        });
                    }
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: entity.id,
                        field: EntityField::Morale,
                        value: -15.0,
                    });
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Crisis,
                            text: format!("{} was unmasked as a spy and fled with stolen gold!", entity_display_name(entity)),
                            entity_ids: vec![entity.id],
                        },
                    });
                }
            }
            5 => {
                // "Fallen Paladin" — revealed when faith > 200 and combat > 150
                let faith = npc.behavior_value(tags::FAITH);
                let combat = npc.behavior_value(tags::COMBAT);
                if faith >= 200.0 && faith < 200.0 + REVEAL_WINDOW && combat > 150.0 {
                    out.push(WorldDelta::UpdateEntityField {
                        entity_id: entity.id,
                        field: EntityField::Morale,
                        value: 15.0, // redemption
                    });
                    let mut action = ActionTags::empty();
                    action.add(tags::FAITH, 5.0);
                    action.add(tags::RESILIENCE, 5.0);
                    action.add(tags::COMBAT, 3.0);
                    out.push(WorldDelta::AddBehaviorTags {
                        entity_id: entity.id,
                        tags: action.tags,
                        count: action.count,
                    });
                    out.push(WorldDelta::RecordChronicle {
                        entry: ChronicleEntry {
                            tick: state.tick,
                            category: ChronicleCategory::Achievement,
                            text: format!("{} reclaimed their oath as a paladin", entity_display_name(entity)),
                            entity_ids: vec![entity.id],
                        },
                    });
                }
            }
            _ => {} // no secret for types 6, 7
        }
    }
}
