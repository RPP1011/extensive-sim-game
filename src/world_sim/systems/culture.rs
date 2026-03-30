#![allow(unused)]
//! Regional culture — emerges from NPC behavior at settlements.
//!
//! Four culture axes per settlement, derived from accumulated behavior tags:
//! - Martial: from combat/melee/defense/tactics tags
//! - Mercantile: from trade/negotiation/crafting tags
//! - Scholarly: from research/lore/medicine/herbalism tags
//! - Spiritual: from faith/ritual/resilience tags
//!
//! Culture affects settlement bonuses via UpdateSettlementField deltas.
//! High martial → higher threat tolerance. High mercantile → better prices.
//! High scholarly → faster progression. High spiritual → better morale.
//!
//! Cadence: every 50 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const CULTURE_INTERVAL: u64 = 50;

pub fn compute_culture(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % CULTURE_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        let range = state.group_index.settlement_entities(settlement.id);
        compute_culture_for_settlement(state, settlement.id, &state.entities[range], out);
    }
}

pub fn compute_culture_for_settlement(
    state: &WorldState,
    settlement_id: u32,
    entities: &[Entity],
    out: &mut Vec<WorldDelta>,
) {
    if state.tick % CULTURE_INTERVAL != 0 || state.tick == 0 { return; }

    let settlement = match state.settlement(settlement_id) {
        Some(s) => s,
        None => return,
    };

    // Sum behavior tags across all NPCs at this settlement.
    let mut martial = 0.0f32;
    let mut mercantile = 0.0f32;
    let mut scholarly = 0.0f32;
    let mut spiritual = 0.0f32;

    for entity in entities {
        if !entity.alive { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };

        martial += npc.behavior_value(tags::MELEE)
            + npc.behavior_value(tags::COMBAT)
            + npc.behavior_value(tags::DEFENSE)
            + npc.behavior_value(tags::TACTICS);

        mercantile += npc.behavior_value(tags::TRADE)
            + npc.behavior_value(tags::NEGOTIATION)
            + npc.behavior_value(tags::CRAFTING)
            + npc.behavior_value(tags::SMITHING);

        scholarly += npc.behavior_value(tags::RESEARCH)
            + npc.behavior_value(tags::LORE)
            + npc.behavior_value(tags::MEDICINE)
            + npc.behavior_value(tags::HERBALISM);

        spiritual += npc.behavior_value(tags::FAITH)
            + npc.behavior_value(tags::RITUAL)
            + npc.behavior_value(tags::RESILIENCE);
    }

    // Normalize: convert raw sums to 0-100 scale.
    let total = (martial + mercantile + scholarly + spiritual).max(1.0);
    let martial_pct = (martial / total * 100.0).min(100.0);
    let mercantile_pct = (mercantile / total * 100.0).min(100.0);
    let scholarly_pct = (scholarly / total * 100.0).min(100.0);
    let spiritual_pct = (spiritual / total * 100.0).min(100.0);

    // Apply culture effects to settlement.

    // Martial culture > 30%: settlement tolerates more threat.
    if martial_pct > 30.0 {
        let threat_tolerance = (martial_pct - 30.0) * 0.01;
        out.push(WorldDelta::UpdateSettlementField {
            settlement_id,
            field: SettlementField::ThreatLevel,
            value: -threat_tolerance, // reduce perceived threat
        });
    }

    // Mercantile culture > 30%: better prices (infrastructure bonus).
    if mercantile_pct > 30.0 {
        let infra_bonus = (mercantile_pct - 30.0) * 0.001;
        out.push(WorldDelta::UpdateSettlementField {
            settlement_id,
            field: SettlementField::InfrastructureLevel,
            value: infra_bonus,
        });
    }

    // Scholarly culture > 30%: XP bonus to all NPCs.
    if scholarly_pct > 30.0 {
        let xp_bonus = ((scholarly_pct - 30.0) * 0.1) as u32;
        if xp_bonus > 0 {
            for entity in entities {
                if !entity.alive { continue; }
                out.push(WorldDelta::AddXp {
                    entity_id: entity.id,
                    amount: xp_bonus.min(3),
                });
            }
        }
    }

    // Spiritual culture > 30%: morale boost.
    if spiritual_pct > 30.0 {
        let morale_boost = (spiritual_pct - 30.0) * 0.01;
        for entity in entities {
            if !entity.alive { continue; }
            out.push(WorldDelta::UpdateEntityField {
                entity_id: entity.id,
                field: EntityField::Morale,
                value: morale_boost,
            });
        }
    }

    // Chronicle: record dominant culture (rarely — only when culture is very strong).
    if state.tick % 2000 == 0 {
        let max_pct = [martial_pct, mercantile_pct, scholarly_pct, spiritual_pct]
            .iter().cloned().fold(0.0f32, f32::max);
        if max_pct < 40.0 { return; } // only record notable cultural identity

        let dominant = if martial_pct >= mercantile_pct && martial_pct >= scholarly_pct && martial_pct >= spiritual_pct {
            "martial"
        } else if mercantile_pct >= scholarly_pct && mercantile_pct >= spiritual_pct {
            "mercantile"
        } else if scholarly_pct >= spiritual_pct {
            "scholarly"
        } else {
            "spiritual"
        };

        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Narrative,
                text: format!("{} has developed a {} culture ({:.0}%)",
                    settlement.name, dominant,
                    [martial_pct, mercantile_pct, scholarly_pct, spiritual_pct]
                        .iter().cloned().fold(0.0f32, f32::max)),
                entity_ids: vec![],
            },
        });
    }
}
