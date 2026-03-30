//! Legendary deeds — milestone achievements based on behavior tag thresholds.
//!
//! NPCs who accumulate enough tags in a category earn a one-time deed.
//! Deeds grant XP, morale, and chronicle entries. Since we don't store
//! explicit deed state, we use the same narrow-window trick as secrets:
//! deeds fire when tag value crosses a threshold within a 50-unit window.
//!
//! Cadence: every 100 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::naming::entity_display_name;
use crate::world_sim::state::*;

const DEED_INTERVAL: u64 = 100;
const WINDOW: f32 = 20.0; // narrow enough to fire ~once per threshold crossing

struct DeedDef {
    tag: u32,
    threshold: f32,
    title: &'static str,
    xp: u32,
    morale: f32,
    category: ChronicleCategory,
}

const DEEDS: &[DeedDef] = &[
    DeedDef { tag: tags::COMBAT,     threshold: 2000.0, title: "Slayer",          xp: 30, morale: 5.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::MELEE,      threshold: 1500.0, title: "Blade Master",    xp: 25, morale: 4.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::DEFENSE,    threshold: 1000.0, title: "Shield Bearer",   xp: 20, morale: 3.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::TACTICS,    threshold: 1000.0, title: "Tactician",       xp: 20, morale: 3.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::TRADE,      threshold: 1500.0, title: "Master Merchant", xp: 20, morale: 4.0, category: ChronicleCategory::Economy },
    DeedDef { tag: tags::DIPLOMACY,  threshold: 800.0,  title: "Peacemaker",      xp: 25, morale: 5.0, category: ChronicleCategory::Diplomacy },
    DeedDef { tag: tags::LEADERSHIP, threshold: 1500.0, title: "Commander",       xp: 30, morale: 5.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::RESEARCH,   threshold: 1000.0, title: "Scholar",         xp: 25, morale: 3.0, category: ChronicleCategory::Discovery },
    DeedDef { tag: tags::FAITH,      threshold: 1000.0, title: "Devout",          xp: 20, morale: 4.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::RESILIENCE, threshold: 2000.0, title: "Survivor",        xp: 30, morale: 5.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::MINING,     threshold: 2000.0, title: "Deep Delver",     xp: 20, morale: 3.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::FARMING,    threshold: 2000.0, title: "Master Farmer",   xp: 20, morale: 3.0, category: ChronicleCategory::Economy },
    DeedDef { tag: tags::STEALTH,    threshold: 1000.0, title: "Shadow Walker",   xp: 25, morale: 3.0, category: ChronicleCategory::Achievement },
    DeedDef { tag: tags::EXPLORATION,threshold: 1000.0, title: "Explorer",        xp: 25, morale: 4.0, category: ChronicleCategory::Discovery },
    DeedDef { tag: tags::TEACHING,   threshold: 1000.0, title: "Grand Mentor",    xp: 20, morale: 4.0, category: ChronicleCategory::Achievement },
];

pub fn compute_legendary_deeds(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % DEED_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_legendary_deeds_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_legendary_deeds_for_settlement(
    state: &WorldState,
    _settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % DEED_INTERVAL != 0 || state.tick == 0 { return; }

    for entity in entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        for deed in DEEDS {
            let value = npc.behavior_value(deed.tag);
            // Fire once: only in the window [threshold, threshold + WINDOW).
            if value >= deed.threshold && value < deed.threshold + WINDOW {
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: entity.id,
                    field: EntityField::Morale,
                    value: deed.morale,
                });
                out.push(WorldDelta::RecordChronicle {
                    entry: ChronicleEntry {
                        tick: state.tick,
                        category: deed.category.clone(),
                        text: format!("{} earned the title '{}'", entity_display_name(entity), deed.title),
                        entity_ids: vec![entity.id],
                    },
                });
            }

            // Second tier: double the threshold.
            let tier2 = deed.threshold * 2.0;
            if value >= tier2 && value < tier2 + WINDOW {
                out.push(WorldDelta::UpdateEntityField {
                    entity_id: entity.id,
                    field: EntityField::Morale,
                    value: deed.morale * 1.5,
                });
                out.push(WorldDelta::RecordChronicle {
                    entry: ChronicleEntry {
                        tick: state.tick,
                        category: deed.category.clone(),
                        text: format!("{} became a legendary {}", entity_display_name(entity), deed.title),
                        entity_ids: vec![entity.id],
                    },
                });
            }
        }
    }
}
