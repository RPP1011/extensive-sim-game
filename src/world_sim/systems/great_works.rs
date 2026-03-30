#![allow(unused)]
//! Great works — prosperous settlements commission monuments and institutions.
//!
//! When a settlement's treasury exceeds a threshold and population is healthy,
//! it "builds" a great work — a named structure that provides bonuses and
//! generates a narrative chronicle entry.
//!
//! Cadence: every 500 ticks.

use crate::world_sim::delta::WorldDelta;
use crate::world_sim::state::*;

const GREAT_WORK_INTERVAL: u64 = 500;

/// Minimum treasury to attempt a great work.
const MIN_TREASURY: f32 = 2000.0;

/// Minimum population.
const MIN_POP: u32 = 100;

/// Treasury cost of a great work.
const WORK_COST: f32 = 1500.0;

/// Max great works per settlement (tracked via chronicle scan).
const MAX_WORKS_PER_SETTLEMENT: usize = 3;

static WORK_TYPES: &[(&str, &str)] = &[
    ("Grand Library", "A center of learning that attracts scholars from distant lands"),
    ("Great Forge", "A masterwork foundry producing legendary weapons and armor"),
    ("Cathedral", "A towering monument to faith that inspires the faithful"),
    ("Colosseum", "An arena where warriors test their mettle and earn glory"),
    ("Market Palace", "A magnificent trading hall that draws merchants from every settlement"),
    ("Walls of Stone", "Impenetrable fortifications that shield the settlement from harm"),
    ("Academy of War", "A prestigious institution training the finest commanders"),
    ("Herbalist's Garden", "A vast botanical garden yielding rare medicines and reagents"),
    ("Observatory", "A tower reaching toward the stars, unlocking arcane secrets"),
    ("Monument of Heroes", "A grand memorial honoring the fallen and inspiring the living"),
];


pub fn compute_great_works(state: &WorldState, out: &mut Vec<WorldDelta>) {
    if state.tick % GREAT_WORK_INTERVAL != 0 || state.tick == 0 { return; }

    for settlement in &state.settlements {
        if settlement.treasury < MIN_TREASURY || settlement.population < MIN_POP {
            continue;
        }

        // Check how many great works this settlement already has.
        let existing_works = state.chronicle.iter().filter(|e| {
            e.category == ChronicleCategory::Achievement
                && e.text.contains(&settlement.name)
                && e.text.contains("completed")
        }).count();
        if existing_works >= MAX_WORKS_PER_SETTLEMENT { continue; }

        // Deterministic chance: 20% per eligible settlement per check.
        if entity_hash(settlement.id, state.tick, 0xC0FFEE) % 100 >= 20 { continue; }

        // Pick a work type based on settlement specialty + hash.
        let work_idx = match settlement.specialty {
            SettlementSpecialty::ScholarCity => [0, 8, 9], // Library, Observatory, Monument
            SettlementSpecialty::CraftingGuild => [1, 7, 9], // Forge, Garden, Monument
            SettlementSpecialty::MilitaryOutpost => [3, 5, 6], // Colosseum, Walls, Academy
            SettlementSpecialty::TradeHub => [4, 0, 9], // Market, Library, Monument
            SettlementSpecialty::FarmingVillage => [7, 2, 9], // Garden, Cathedral, Monument
            SettlementSpecialty::MiningTown => [1, 5, 9], // Forge, Walls, Monument
            SettlementSpecialty::PortTown => [4, 8, 9], // Market, Observatory, Monument
            SettlementSpecialty::General => [9, 2, 0], // Monument, Cathedral, Library
        };
        let pick = work_idx[existing_works % work_idx.len()];
        let (name, desc) = WORK_TYPES[pick];

        // Deduct cost.
        out.push(WorldDelta::UpdateTreasury {
            settlement_id: settlement.id,
            delta: -WORK_COST,
        });

        // Record chronicle.
        out.push(WorldDelta::RecordChronicle {
            entry: ChronicleEntry {
                tick: state.tick,
                category: ChronicleCategory::Achievement,
                text: format!("{} completed the {}. {}", settlement.name, name, desc),
                entity_ids: vec![],
            },
        });
    }
}
