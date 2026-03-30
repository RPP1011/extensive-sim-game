#![allow(unused)]
//! Settlement founding — overcrowded settlements send colonists to empty regions.
//!
//! When population exceeds 1.5× housing AND an NPC with leadership+exploration
//! exists, a founding expedition launches. Leader + colonists travel to the
//! nearest settleable region without a settlement and create a new one.
//!
//! Cadence: every 500 ticks (post-apply, needs &mut WorldState).

use crate::world_sim::state::*;
use crate::world_sim::naming;
use crate::world_sim::city_grid::{CityGrid, InfluenceMap};

const FOUNDING_INTERVAL: u64 = 500;
const OVERCROWDING_MULT: f32 = 1.5;
const MIN_POP_FOR_FOUNDING: usize = 40;
const COLONIST_COUNT: usize = 8;
const FOUNDER_TAG_THRESHOLD: f32 = 5.0;
const MAX_SETTLEMENTS: usize = 20;

/// Process settlement founding. Called post-apply from runtime.rs.
pub fn advance_settlement_founding(state: &mut WorldState) {
    if state.tick % FOUNDING_INTERVAL != 0 || state.tick == 0 { return; }
    if state.settlements.len() >= MAX_SETTLEMENTS { return; }

    let tick = state.tick;

    // Find an overcrowded settlement with a viable founder.
    let mut founding: Option<(u32, usize, u32)> = None;

    for si in 0..state.settlements.len() {
        let sid = state.settlements[si].id;
        let settlement_pos = state.settlements[si].pos;

        let pop = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Npc
                && e.npc.as_ref().map(|n| n.home_settlement_id == Some(sid)).unwrap_or(false))
            .count();

        if pop < MIN_POP_FOR_FOUNDING { continue; }

        let housing: u32 = state.entities.iter()
            .filter(|e| e.alive && e.kind == EntityKind::Building)
            .filter_map(|e| e.building.as_ref())
            .filter(|b| b.settlement_id == Some(sid) && b.construction_progress >= 1.0)
            .map(|b| b.residential_capacity as u32)
            .sum();

        if (pop as f32) < housing.max(1) as f32 * OVERCROWDING_MULT { continue; }

        // Find founder.
        let founder_idx = state.entities.iter().enumerate()
            .filter(|(_, e)| e.alive && e.kind == EntityKind::Npc)
            .filter(|(_, e)| e.npc.as_ref().map(|n| n.home_settlement_id == Some(sid)).unwrap_or(false))
            .filter(|(_, e)| {
                let npc = e.npc.as_ref().unwrap();
                let tags = npc.behavior_value(tags::LEADERSHIP) + npc.behavior_value(tags::EXPLORATION);
                tags > FOUNDER_TAG_THRESHOLD && npc.personality.curiosity > 0.3
            })
            .map(|(i, _)| i)
            .next();

        let founder_idx = match founder_idx { Some(i) => i, None => continue };

        // Find target region — nearest settleable without a settlement.
        let occupied: Vec<(f32, f32)> = state.settlements.iter().map(|s| s.pos).collect();
        let target = state.regions.iter()
            .filter(|r| r.terrain.is_settleable())
            .filter(|r| {
                let rx = (r.id as f32 * 137.5).sin() * 150.0;
                let ry = (r.id as f32 * 73.1).cos() * 150.0;
                // No existing settlement within 80 units.
                !occupied.iter().any(|&(sx, sy)| {
                    let dx = sx - rx;
                    let dy = sy - ry;
                    dx * dx + dy * dy < 6400.0
                })
            })
            .min_by_key(|r| {
                let rx = (r.id as f32 * 137.5).sin() * 150.0;
                let ry = (r.id as f32 * 73.1).cos() * 150.0;
                let dx = settlement_pos.0 - rx;
                let dy = settlement_pos.1 - ry;
                (dx * dx + dy * dy) as u32
            });

        if let Some(region) = target {
            founding = Some((sid, founder_idx, region.id));
            break;
        }
    }

    let (source_sid, founder_idx, target_rid) = match founding {
        Some(f) => f,
        None => return,
    };

    // Compute position.
    let target_terrain = state.regions.iter()
        .find(|r| r.id == target_rid)
        .map(|r| r.terrain)
        .unwrap_or(Terrain::Plains);
    let target_name = state.regions.iter()
        .find(|r| r.id == target_rid)
        .map(|r| r.name.clone())
        .unwrap_or_default();
    let new_pos = (
        (target_rid as f32 * 137.5).sin() * 150.0,
        (target_rid as f32 * 73.1).cos() * 150.0,
    );

    // Create settlement.
    let new_sid = state.settlements.iter().map(|s| s.id).max().unwrap_or(0) + 1;
    let new_name = naming::generate_settlement_name_from_seed(tick ^ target_rid as u64);
    let faction_id = state.entities[founder_idx].npc.as_ref().and_then(|n| n.faction_id);

    let mut new_settlement = SettlementState::new(new_sid, new_name.clone(), new_pos);
    new_settlement.faction_id = faction_id;
    new_settlement.treasury = 50.0;
    new_settlement.stockpile[crate::world_sim::commodity::FOOD] = 20.0;

    // City grid.
    let grid = CityGrid::new(128, 128, new_sid, target_terrain.name(), tick);
    let influence = InfluenceMap::new(128, 128);
    let grid_idx = state.city_grids.len();
    state.city_grids.push(grid);
    state.influence_maps.push(influence);
    new_settlement.city_grid_idx = Some(grid_idx);

    state.settlements.push(new_settlement);

    // Select colonists.
    let mut colonists: Vec<usize> = vec![founder_idx];
    for (i, entity) in state.entities.iter().enumerate() {
        if colonists.len() >= COLONIST_COUNT { break; }
        if i == founder_idx { continue; }
        if !entity.alive || entity.kind != EntityKind::Npc { continue; }
        let npc = match &entity.npc { Some(n) => n, None => continue };
        if npc.home_settlement_id != Some(source_sid) { continue; }
        if npc.spouse_id.is_some() || npc.party_id.is_some() { continue; }
        colonists.push(i);
    }

    // Move colonists.
    let founder_name = state.entities[founder_idx].npc.as_ref()
        .map(|n| n.name.clone()).unwrap_or_default();

    for &ci in &colonists {
        state.entities[ci].pos = new_pos;
        if let Some(npc) = state.entities[ci].npc.as_mut() {
            npc.home_settlement_id = Some(new_sid);
            npc.home_building_id = None;
            npc.work_building_id = None;
            npc.work_state = WorkState::Idle;
            npc.goal_stack = GoalStack::default();
            npc.needs.purpose = 100.0;
            npc.emotions.pride = (npc.emotions.pride + 0.8).min(1.0);
            npc.accumulate_tags(&{
                let mut a = ActionTags::empty();
                a.add(tags::LEADERSHIP, 3.0);
                a.add(tags::EXPLORATION, 3.0);
                a
            });
        }
    }

    let source_name = state.settlements.iter()
        .find(|s| s.id == source_sid)
        .map(|s| s.name.clone())
        .unwrap_or_default();

    state.chronicle.push(ChronicleEntry {
        tick,
        category: ChronicleCategory::Narrative,
        text: format!("{} leads {} colonists from {} to found {} in the {} of {}.",
            founder_name, colonists.len() - 1, source_name, new_name,
            target_terrain.name().to_lowercase(), target_name),
        entity_ids: colonists.iter().map(|&i| state.entities[i].id).collect(),
    });
}

// Keep the old delta-based stub for backward compat (called by compute_all_systems).
pub fn compute_settlement_founding(state: &WorldState, out: &mut Vec<crate::world_sim::delta::WorldDelta>) {
    // No-op — founding is handled by advance_settlement_founding (post-apply).
}
