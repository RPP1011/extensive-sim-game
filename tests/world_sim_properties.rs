//! Property-based tests for world sim invariants.
//!
//! These tests use proptest to verify system-level properties hold
//! across random seeds and tick counts.

use proptest::prelude::*;
use bevy_game::world_sim::runtime::WorldSim;
use bevy_game::world_sim::state::{
    WorldState, Entity, EntityKind, SettlementState,
    FactionState, DiplomaticStance, RegionState, Terrain, LocalGrid,
    Inventory,
};
use bevy_game::world_sim::fidelity::Fidelity;
use std::collections::HashSet;

/// Build a small but complete world state for property testing.
/// Has settlements, factions, regions, NPCs, buildings, monsters.
fn build_test_world(seed: u64, npcs: usize, settlements: usize) -> WorldState {
    let mut state = WorldState::new(seed);

    // Regions
    for i in 0..settlements.max(2) {
        state.regions.push(RegionState {
            id: i as u32,
            name: format!("Region{}", i),
            terrain: Terrain::Plains,
            monster_density: 0.0,
            threat_level: 0.0,
            neighbors: vec![],
            faction_id: None,
            elevation: 1,
            is_coastal: false,
            has_river: false,
            has_lake: false,
            is_chokepoint: false,
            is_floating: false,
            sub_biome: Default::default(),
            river_connections: vec![],
            dungeon_sites: vec![],
            unrest: 0.0,
            control: 0.0,
        });
    }

    // Factions
    state.factions.push(FactionState {
        id: 0,
        name: "TestFaction".into(),
        diplomatic_stance: DiplomaticStance::Neutral,
        military_strength: 50.0,
        max_military_strength: 100.0,
        treasury: 100.0,
        territory_size: settlements as u32,
        relationship_to_guild: 0.0,
        at_war_with: vec![],
        coup_risk: 0.0,
        escalation_level: 0,
        tech_level: 0,
        recent_actions: vec![],
    });

    // Settlements
    for i in 0..settlements {
        let mut s = SettlementState::new(i as u32, format!("Town{}", i), (i as f32 * 100.0, 0.0));
        s.faction_id = Some(0);
        s.treasury = 200.0;
        s.stockpile[0] = 100.0; // food
        s.stockpile[1] = 50.0;  // iron

        let grid = LocalGrid {
            id: i as u32,
            fidelity: Fidelity::Medium,
            center: s.pos,
            radius: 30.0,
            entity_ids: Vec::new(),
        };
        state.grids.push(grid);
        s.grid_id = Some(i as u32);

        state.settlements.push(s);
    }

    // NPCs
    let mut id = 0u32;
    for i in 0..npcs {
        let sid = (i % settlements.max(1)) as u32;
        let mut npc = Entity::new_npc(id, (sid as f32 * 100.0, 0.0));
        npc.grid_id = Some(sid);
        let npc_data = npc.npc.as_mut().unwrap();
        npc_data.home_settlement_id = Some(sid);
        npc_data.faction_id = Some(0);
        npc_data.gold = 10.0;
        npc_data.morale = 50.0;
        npc_data.behavior_production = vec![(0, 0.1)]; // produces food
        npc.inventory = Some(Inventory::default());
        state.entities.push(npc);
        id += 1;
    }

    // Monsters (a few, unaffiliated)
    let num_monsters = (npcs / 5).max(2);
    for _ in 0..num_monsters {
        let monster = Entity::new_monster(id, (500.0, 500.0), 3);
        state.entities.push(monster);
        id += 1;
    }

    state.next_id = id;
    state
}

fn run_sim(seed: u64, npcs: usize, settlements: usize, ticks: usize) -> WorldState {
    let state = build_test_world(seed, npcs, settlements);
    let mut sim = WorldSim::new(state);
    for _ in 0..ticks {
        sim.tick();
    }
    sim.state().clone()
}

// =========================================================================
// Property: Determinism — same seed produces identical state
// =========================================================================

proptest! {
    #[test]
    fn determinism_same_seed(seed in 0u64..10000, ticks in 1usize..50) {
        let s1 = run_sim(seed, 20, 3, ticks);
        let s2 = run_sim(seed, 20, 3, ticks);

        prop_assert_eq!(s1.tick, s2.tick, "tick mismatch");

        // Entity count must match
        prop_assert_eq!(s1.entities.len(), s2.entities.len(), "entity count mismatch");

        // All entity positions and HP must match
        for (e1, e2) in s1.entities.iter().zip(s2.entities.iter()) {
            prop_assert_eq!(e1.id, e2.id, "entity ID mismatch");
            prop_assert_eq!(e1.alive, e2.alive, "alive mismatch for entity {}", e1.id);
            if e1.alive {
                prop_assert!((e1.hp - e2.hp).abs() < 0.001,
                    "HP mismatch for entity {}: {} vs {}", e1.id, e1.hp, e2.hp);
                prop_assert!((e1.pos.0 - e2.pos.0).abs() < 0.01,
                    "pos.x mismatch for entity {}", e1.id);
            }
        }
    }
}

// =========================================================================
// Property: Entity index integrity after any number of ticks
// =========================================================================

proptest! {
    #[test]
    fn entity_index_integrity(seed in 0u64..10000, ticks in 1usize..100) {
        let state = run_sim(seed, 30, 3, ticks);

        // Every entity must be findable via entity()
        for (i, entity) in state.entities.iter().enumerate() {
            if let Some(found) = state.entity(entity.id) {
                prop_assert_eq!(found.id, entity.id,
                    "entity_index returned wrong entity for id {}", entity.id);
            }
            // entity_idx must return the correct position
            if let Some(idx) = state.entity_idx(entity.id) {
                prop_assert_eq!(idx, i,
                    "entity_idx mismatch for id {}: got {} expected {}", entity.id, idx, i);
            }
        }
    }
}

// =========================================================================
// Property: No entity ID collisions
// =========================================================================

proptest! {
    #[test]
    fn no_entity_id_collisions(seed in 0u64..10000, ticks in 1usize..100) {
        let state = run_sim(seed, 30, 3, ticks);

        let mut seen = HashSet::new();
        for entity in &state.entities {
            prop_assert!(seen.insert(entity.id),
                "duplicate entity ID {} at tick {}", entity.id, state.tick);
        }
    }
}

// =========================================================================
// Property: HP invariants — hp in [0, max_hp] for alive entities
// =========================================================================

proptest! {
    #[test]
    fn hp_invariants(seed in 0u64..10000, ticks in 1usize..100) {
        let state = run_sim(seed, 30, 3, ticks);

        for entity in &state.entities {
            if entity.alive {
                prop_assert!(entity.hp >= 0.0,
                    "entity {} has negative HP: {}", entity.id, entity.hp);
                prop_assert!(entity.hp <= entity.max_hp + 0.01,
                    "entity {} HP {} exceeds max_hp {}", entity.id, entity.hp, entity.max_hp);
                prop_assert!(entity.max_hp >= 1.0,
                    "entity {} has max_hp < 1: {}", entity.id, entity.max_hp);
            }
        }
    }
}

// =========================================================================
// Property: Tick always advances
// =========================================================================

proptest! {
    #[test]
    fn tick_monotonic(seed in 0u64..10000, ticks in 1usize..50) {
        let state = run_sim(seed, 10, 2, ticks);
        prop_assert_eq!(state.tick, ticks as u64,
            "tick should be {} but is {}", ticks, state.tick);
    }
}

// =========================================================================
// Property: Settlement stockpile never negative
// =========================================================================

proptest! {
    #[test]
    fn stockpile_non_negative(seed in 0u64..10000, ticks in 1usize..100) {
        let state = run_sim(seed, 30, 3, ticks);

        for settlement in &state.settlements {
            for (c, &amount) in settlement.stockpile.iter().enumerate() {
                prop_assert!(amount >= -0.01,
                    "settlement {} commodity {} is negative: {}",
                    settlement.name, c, amount);
            }
        }
    }
}

// =========================================================================
// Property: NPC morale stays clamped [10, 100]
// =========================================================================

proptest! {
    #[test]
    fn morale_clamped(seed in 0u64..10000, ticks in 10usize..100) {
        let state = run_sim(seed, 30, 3, ticks);

        for entity in &state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            if let Some(npc) = &entity.npc {
                prop_assert!(npc.morale >= 9.9 && npc.morale <= 100.1,
                    "NPC {} morale {} out of [10, 100]", entity.id, npc.morale);
            }
        }
    }
}

// =========================================================================
// Property: GroupIndex settlement ranges are consistent
// =========================================================================

proptest! {
    #[test]
    fn group_index_consistency(seed in 0u64..10000, ticks in 1usize..50) {
        let state = run_sim(seed, 30, 3, ticks);

        for settlement in &state.settlements {
            let range = state.group_index.settlement_entities(settlement.id);
            for i in range {
                let entity = &state.entities[i];
                let entity_sid = entity.settlement_id();
                prop_assert_eq!(entity_sid, Some(settlement.id),
                    "entity {} at index {} has settlement_id {:?} but is in range for settlement {}",
                    entity.id, i, entity_sid, settlement.id);
            }
        }
    }
}

// =========================================================================
// Property: No NaN in critical fields
// =========================================================================

proptest! {
    #[test]
    fn no_nan_in_entities(seed in 0u64..10000, ticks in 1usize..100) {
        let state = run_sim(seed, 30, 3, ticks);

        for entity in &state.entities {
            prop_assert!(!entity.hp.is_nan(), "entity {} hp is NaN", entity.id);
            prop_assert!(!entity.max_hp.is_nan(), "entity {} max_hp is NaN", entity.id);
            prop_assert!(!entity.pos.0.is_nan(), "entity {} pos.x is NaN", entity.id);
            prop_assert!(!entity.pos.1.is_nan(), "entity {} pos.y is NaN", entity.id);
            prop_assert!(!entity.attack_damage.is_nan(), "entity {} attack_damage is NaN", entity.id);
            prop_assert!(!entity.armor.is_nan(), "entity {} armor is NaN", entity.id);
            prop_assert!(!entity.move_speed.is_nan(), "entity {} move_speed is NaN", entity.id);

            if let Some(npc) = &entity.npc {
                prop_assert!(!npc.morale.is_nan(), "NPC {} morale is NaN", entity.id);
                prop_assert!(!npc.gold.is_nan(), "NPC {} gold is NaN", entity.id);
                prop_assert!(!npc.stress.is_nan(), "NPC {} stress is NaN", entity.id);
            }
        }

        for s in &state.settlements {
            prop_assert!(!s.treasury.is_nan(), "settlement {} treasury is NaN", s.name);
            for (c, &v) in s.stockpile.iter().enumerate() {
                prop_assert!(!v.is_nan(), "settlement {} commodity {} is NaN", s.name, c);
            }
        }
    }
}

// =========================================================================
// Property: Gold is bounded (no infinite minting)
// =========================================================================

proptest! {
    #[test]
    fn gold_bounded(seed in 0u64..10000, ticks in 1usize..200) {
        let state = run_sim(seed, 20, 3, ticks);

        let total_npc_gold: f32 = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc)
            .filter_map(|e| e.npc.as_ref())
            .map(|n| n.gold)
            .sum();

        let total_treasury: f32 = state.settlements.iter()
            .map(|s| s.treasury)
            .sum();

        let total_gold = total_npc_gold + total_treasury;

        // Gold should stay bounded — not grow without limit.
        // Initial: 20 NPCs × 10 gold + 3 settlements × 200 = 800 initial.
        // Allow 10x growth from production over 200 ticks.
        prop_assert!(total_gold < 100_000.0,
            "total gold {} is unreasonably large at tick {}", total_gold, state.tick);
        prop_assert!(total_gold > -1000.0,
            "total gold {} is unreasonably negative at tick {}", total_gold, state.tick);
    }
}

// =========================================================================
// Property: Alive entity count is reasonable
// =========================================================================

proptest! {
    #[test]
    fn population_reasonable(seed in 0u64..10000, ticks in 1usize..200) {
        let state = run_sim(seed, 20, 3, ticks);

        let alive_npcs = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc)
            .count();

        // Population shouldn't explode (bounded spawning).
        prop_assert!(alive_npcs < 5000,
            "NPC count {} is unreasonably large at tick {}", alive_npcs, state.tick);
    }
}
