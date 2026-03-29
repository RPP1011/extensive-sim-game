//! Zero-allocation runtime for the world simulation.
//!
//! All buffers are pre-allocated at init. Ticking reuses capacity via dirty-list
//! clears — no heap allocations after `WorldSim::new()`. Merge uses flat arrays
//! indexed by entity/settlement ID instead of HashMaps.

use std::time::Instant;

use super::delta::WorldDelta;
use super::fidelity::Fidelity;
use super::spatial::SpatialIndex;
use super::state::{WorldState, EntityKind};
use super::tick::{TickProfile, ProfileAccumulator};
use super::NUM_COMMODITIES;
use super::apply::ApplyProfile;

// ---------------------------------------------------------------------------
// FlatMergedDeltas — flat-array merge accumulator, zero HashMap overhead
// ---------------------------------------------------------------------------

/// Flat-array merge accumulator. All per-entity fields are indexed by entity ID.
/// Per-settlement fields indexed by settlement ID. Dirty lists track which
/// indices were touched so clear is O(touched) not O(capacity).
struct FlatMergedDeltas {
    // --- Per-entity (indexed by entity ID) ---
    damage: Vec<f32>,
    heals: Vec<f32>,
    shields: Vec<f32>,
    force_x: Vec<f32>,
    force_y: Vec<f32>,
    dead: Vec<bool>,
    entity_dirty: Vec<u32>,

    // --- Per-settlement (indexed by settlement ID) ---
    production: Vec<[f32; NUM_COMMODITIES]>,
    consumption: Vec<[f32; NUM_COMMODITIES]>,
    stockpile_adj: Vec<[f32; NUM_COMMODITIES]>,
    treasury_adj: Vec<f32>,
    settlement_dirty: Vec<u32>,

    // --- Collected (Vecs, cleared each tick) ---
    new_statuses: Vec<(u32, super::state::StatusEffect)>,
    remove_statuses: Vec<(u32, u8)>,
    gold_transfers: Vec<(u32, u32, f32)>,
    goods_transfers: Vec<(u32, u32, usize, f32)>,
    grid_enters: Vec<(u32, u32)>,
    grid_leaves: Vec<(u32, u32)>,
    fidelity_changes: Vec<(u32, Fidelity)>,
    price_reports: Vec<(u32, u32, super::state::PriceReport)>,
    price_updates: Vec<(u32, [f32; NUM_COMMODITIES])>,

    /// Campaign system deltas that need the MergedDeltas path.
    overflow: Vec<WorldDelta>,
}

impl FlatMergedDeltas {
    fn new(max_entities: usize, max_settlements: usize) -> Self {
        FlatMergedDeltas {
            damage: vec![0.0; max_entities],
            heals: vec![0.0; max_entities],
            shields: vec![0.0; max_entities],
            force_x: vec![0.0; max_entities],
            force_y: vec![0.0; max_entities],
            dead: vec![false; max_entities],
            entity_dirty: Vec::with_capacity(max_entities),

            production: vec![[0.0; NUM_COMMODITIES]; max_settlements],
            consumption: vec![[0.0; NUM_COMMODITIES]; max_settlements],
            stockpile_adj: vec![[0.0; NUM_COMMODITIES]; max_settlements],
            treasury_adj: vec![0.0; max_settlements],
            settlement_dirty: Vec::with_capacity(max_settlements),

            new_statuses: Vec::with_capacity(64),
            remove_statuses: Vec::with_capacity(64),
            gold_transfers: Vec::with_capacity(64),
            goods_transfers: Vec::with_capacity(64),
            grid_enters: Vec::with_capacity(32),
            grid_leaves: Vec::with_capacity(32),
            fidelity_changes: Vec::with_capacity(16),
            price_reports: Vec::with_capacity(64),
            price_updates: Vec::with_capacity(16),
            overflow: Vec::with_capacity(64),
        }
    }

    fn clear(&mut self) {
        for &id in &self.entity_dirty {
            let i = id as usize;
            self.damage[i] = 0.0;
            self.heals[i] = 0.0;
            self.shields[i] = 0.0;
            self.force_x[i] = 0.0;
            self.force_y[i] = 0.0;
            self.dead[i] = false;
        }
        self.entity_dirty.clear();

        for &id in &self.settlement_dirty {
            let i = id as usize;
            self.production[i] = [0.0; NUM_COMMODITIES];
            self.consumption[i] = [0.0; NUM_COMMODITIES];
            self.stockpile_adj[i] = [0.0; NUM_COMMODITIES];
            self.treasury_adj[i] = 0.0;
        }
        self.settlement_dirty.clear();

        self.new_statuses.clear();
        self.remove_statuses.clear();
        self.gold_transfers.clear();
        self.goods_transfers.clear();
        self.grid_enters.clear();
        self.grid_leaves.clear();
        self.fidelity_changes.clear();
        self.price_reports.clear();
        self.price_updates.clear();
        self.overflow.clear();
    }

    #[inline]
    fn mark_entity(&mut self, id: u32) {
        let i = id as usize;
        // Only add to dirty list on first touch (damage==0 && heals==0 && ... is the init state).
        // Cheaper to just always push and dedup isn't needed — clear handles duplicates fine.
        if self.damage[i] == 0.0 && self.heals[i] == 0.0 && self.shields[i] == 0.0
            && self.force_x[i] == 0.0 && self.force_y[i] == 0.0 && !self.dead[i]
        {
            self.entity_dirty.push(id);
        }
    }

    #[inline]
    fn mark_settlement(&mut self, id: u32) {
        let i = id as usize;
        if self.production[i] == [0.0; NUM_COMMODITIES]
            && self.consumption[i] == [0.0; NUM_COMMODITIES]
            && self.stockpile_adj[i] == [0.0; NUM_COMMODITIES]
            && self.treasury_adj[i] == 0.0
        {
            self.settlement_dirty.push(id);
        }
    }

    fn merge_one(&mut self, delta: WorldDelta) {
        match delta {
            WorldDelta::Damage { target_id, amount, .. } => {
                self.mark_entity(target_id);
                self.damage[target_id as usize] += amount;
            }
            WorldDelta::Heal { target_id, amount, .. } => {
                self.mark_entity(target_id);
                self.heals[target_id as usize] += amount;
            }
            WorldDelta::Shield { target_id, amount, .. } => {
                self.mark_entity(target_id);
                self.shields[target_id as usize] += amount;
            }
            WorldDelta::Move { entity_id, force } => {
                self.mark_entity(entity_id);
                self.force_x[entity_id as usize] += force.0;
                self.force_y[entity_id as usize] += force.1;
            }
            WorldDelta::Die { entity_id } => {
                self.mark_entity(entity_id);
                self.dead[entity_id as usize] = true;
            }
            WorldDelta::ApplyStatus { target_id, status } => {
                self.new_statuses.push((target_id, status));
            }
            WorldDelta::RemoveStatus { target_id, status_discriminant } => {
                self.remove_statuses.push((target_id, status_discriminant));
            }
            WorldDelta::ProduceCommodity { location_id, commodity, amount } => {
                self.mark_settlement(location_id);
                self.production[location_id as usize][commodity] += amount;
            }
            WorldDelta::ConsumeCommodity { location_id, commodity, amount } => {
                self.mark_settlement(location_id);
                self.consumption[location_id as usize][commodity] += amount;
            }
            WorldDelta::TransferGold { from_id, to_id, amount } => {
                self.gold_transfers.push((from_id, to_id, amount));
            }
            WorldDelta::TransferGoods { from_id, to_id, commodity, amount } => {
                self.goods_transfers.push((from_id, to_id, commodity, amount));
            }
            WorldDelta::UpdateStockpile { location_id, commodity, delta } => {
                self.mark_settlement(location_id);
                self.stockpile_adj[location_id as usize][commodity] += delta;
            }
            WorldDelta::UpdateTreasury { location_id, delta } => {
                self.mark_settlement(location_id);
                self.treasury_adj[location_id as usize] += delta;
            }
            WorldDelta::UpdatePrices { location_id, prices } => {
                self.price_updates.push((location_id, prices));
            }
            WorldDelta::EntityEntersGrid { entity_id, grid_id } => {
                self.grid_enters.push((entity_id, grid_id));
            }
            WorldDelta::EntityLeavesGrid { entity_id, grid_id } => {
                self.grid_leaves.push((entity_id, grid_id));
            }
            WorldDelta::EscalateFidelity { grid_id, new_fidelity } => {
                self.fidelity_changes.push((grid_id, new_fidelity));
            }
            WorldDelta::SharePriceReport { from_id, to_id, report } => {
                self.price_reports.push((from_id, to_id, report));
            }
            WorldDelta::TickCooldown { .. } => {}

            // Campaign system deltas — not handled in the flat runtime path.
            // These go through the MergedDeltas → apply_campaign_deltas path instead.
            WorldDelta::UpdateEntityField { .. }
            | WorldDelta::SetEntityMood { .. }
            | WorldDelta::AddXp { .. }
            | WorldDelta::UpdateFaction { .. }
            | WorldDelta::UpdateRegion { .. }
            | WorldDelta::UpdateSettlementField { .. }
            | WorldDelta::UpdateRelation { .. }
            | WorldDelta::SpawnEntity { .. }
            | WorldDelta::RecordEvent { .. }
            | WorldDelta::RecordChronicle { .. }
            | WorldDelta::QuestUpdate { .. }
            | WorldDelta::UpdateGuildGold { .. }
            | WorldDelta::UpdateGuildSupplies { .. }
            | WorldDelta::UpdateGuildReputation { .. } => {
                // Fall through to overflow path which uses MergedDeltas.
                self.overflow.push(delta);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flat apply — reads FlatMergedDeltas arrays directly
// ---------------------------------------------------------------------------

const DT_SEC: f32 = 0.1;

fn apply_flat(state: &mut WorldState, m: &FlatMergedDeltas) -> ApplyProfile {
    let mut p = ApplyProfile::default();

    // HP changes
    let t = Instant::now();
    for &id in &m.entity_dirty {
        let i = id as usize;
        let damage = m.damage[i];
        let heal = m.heals[i];
        let shield_add = m.shields[i];
        if damage == 0.0 && heal == 0.0 && shield_add == 0.0 { continue; }

        if let Some(e) = state.entities.iter_mut().find(|e| e.id == id) {
            if !e.alive { continue; }
            let shield_absorb = damage.min(e.shield_hp);
            e.shield_hp = e.shield_hp - shield_absorb + shield_add;
            let remaining = damage - shield_absorb;
            e.hp = (e.hp + heal - remaining).clamp(0.0, e.max_hp);
        }
    }
    p.hp_us = t.elapsed().as_micros() as u64;

    // Movement
    let t = Instant::now();
    for &id in &m.entity_dirty {
        let i = id as usize;
        let fx = m.force_x[i];
        let fy = m.force_y[i];
        if fx == 0.0 && fy == 0.0 { continue; }

        if let Some(e) = state.entities.iter_mut().find(|e| e.id == id) {
            if !e.alive { continue; }
            let mag = (fx * fx + fy * fy).sqrt();
            let max_speed = e.move_speed * DT_SEC;
            let (dx, dy) = if mag > max_speed && mag > 0.001 {
                (fx / mag * max_speed, fy / mag * max_speed)
            } else {
                (fx, fy)
            };
            e.pos.0 += dx;
            e.pos.1 += dy;
        }
    }
    p.movement_us = t.elapsed().as_micros() as u64;

    // Status effects
    let t = Instant::now();
    for &(target_id, disc) in &m.remove_statuses {
        if let Some(e) = state.entity_mut(target_id) {
            e.status_effects.retain(|s| s.kind.discriminant() != disc);
        }
    }
    for (target_id, status) in &m.new_statuses {
        if let Some(e) = state.entity_mut(*target_id) {
            let disc = status.kind.discriminant();
            if let Some(existing) = e.status_effects.iter_mut()
                .find(|s| s.kind.discriminant() == disc)
            {
                if status.remaining_ms > existing.remaining_ms {
                    *existing = status.clone();
                }
            } else {
                e.status_effects.push(status.clone());
            }
        }
    }
    p.status_us = t.elapsed().as_micros() as u64;

    // Economy
    let t = Instant::now();
    for &id in &m.settlement_dirty {
        let i = id as usize;
        if let Some(s) = state.settlements.iter_mut().find(|s| s.id == id) {
            // Stockpile adjustments
            for c in 0..NUM_COMMODITIES {
                s.stockpile[c] += m.stockpile_adj[i][c];
            }
            // Treasury
            s.treasury += m.treasury_adj[i];
            // Production
            for c in 0..NUM_COMMODITIES {
                s.stockpile[c] += m.production[i][c];
            }
            // Consumption with fair rationing
            for c in 0..NUM_COMMODITIES {
                let consumed = m.consumption[i][c];
                if consumed <= 0.0 { continue; }
                if consumed > s.stockpile[c] {
                    s.stockpile[c] = 0.0;
                } else {
                    s.stockpile[c] -= consumed;
                }
            }
        }
    }
    for &(loc_id, ref prices) in &m.price_updates {
        if let Some(s) = state.settlements.iter_mut().find(|s| s.id == loc_id) {
            s.prices = *prices;
        }
    }
    p.economy_us = t.elapsed().as_micros() as u64;

    // Deaths
    let t = Instant::now();
    for entity in &mut state.entities {
        if !entity.alive { continue; }
        if m.dead[entity.id as usize]
            || (entity.hp <= 0.0 && entity.kind != EntityKind::Building)
        {
            entity.alive = false;
        }
    }
    p.deaths_us = t.elapsed().as_micros() as u64;

    // Grid transitions
    let t = Instant::now();
    for &(entity_id, grid_id) in &m.grid_leaves {
        if let Some(grid) = state.grids.iter_mut().find(|g| g.id == grid_id) {
            grid.entity_ids.retain(|&id| id != entity_id);
        }
        if let Some(e) = state.entity_mut(entity_id) {
            if e.grid_id == Some(grid_id) {
                e.grid_id = None;
                e.local_pos = None;
            }
        }
    }
    for &(entity_id, grid_id) in &m.grid_enters {
        if let Some(grid) = state.grids.iter_mut().find(|g| g.id == grid_id) {
            if !grid.entity_ids.contains(&entity_id) {
                grid.entity_ids.push(entity_id);
            }
        }
        if let Some(e) = state.entity_mut(entity_id) {
            e.grid_id = Some(grid_id);
        }
    }
    p.grid_us = t.elapsed().as_micros() as u64;

    // Fidelity
    let t = Instant::now();
    for &(grid_id, new_fidelity) in &m.fidelity_changes {
        if let Some(grid) = state.grids.iter_mut().find(|g| g.id == grid_id) {
            grid.fidelity = new_fidelity;
        }
    }
    p.fidelity_us = t.elapsed().as_micros() as u64;

    // Price reports
    let t = Instant::now();
    for (_, to_id, report) in &m.price_reports {
        if let Some(e) = state.entity_mut(*to_id) {
            if let Some(npc) = e.npc.as_mut() {
                if let Some(existing) = npc.price_knowledge.iter_mut()
                    .find(|r| r.settlement_id == report.settlement_id)
                {
                    if report.tick_observed > existing.tick_observed {
                        *existing = report.clone();
                    }
                } else {
                    npc.price_knowledge.push(report.clone());
                }
            }
        }
    }
    p.price_reports_us = t.elapsed().as_micros() as u64;

    // Campaign system overflow — merge into MergedDeltas and apply.
    if !m.overflow.is_empty() {
        let t = Instant::now();
        let merged = super::merge_deltas(m.overflow.iter().cloned());
        super::apply::apply_campaign_deltas(state, &merged);
        p.campaign_us = t.elapsed().as_micros() as u64;
    }

    p
}

// ---------------------------------------------------------------------------
// WorldSim — zero-alloc runtime
// ---------------------------------------------------------------------------

/// Pre-allocated world simulation runtime.
pub struct WorldSim {
    /// Single state, mutated in-place.
    state: WorldState,

    /// Reusable delta buffer.
    delta_buf: Vec<WorldDelta>,

    /// Flat-array merge accumulator.
    merged: FlatMergedDeltas,

    /// Reusable spatial index.
    spatial: SpatialIndex,

    /// Accumulated profiling stats.
    pub profile_acc: ProfileAccumulator,
}

impl WorldSim {
    pub fn new(initial: WorldState) -> Self {
        let max_entity_id = initial.entities.iter().map(|e| e.id).max().unwrap_or(0) as usize + 1;
        let max_settlement_id = initial.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;

        WorldSim {
            delta_buf: Vec::with_capacity(initial.entities.len() * 4),
            merged: FlatMergedDeltas::new(max_entity_id, max_settlement_id),
            spatial: SpatialIndex::new(),
            profile_acc: ProfileAccumulator::default(),
            state: initial,
        }
    }

    pub fn state(&self) -> &WorldState {
        &self.state
    }

    pub fn tick(&mut self) -> TickProfile {
        let tick_start = Instant::now();
        let mut profile = TickProfile::default();

        // SPATIAL INDEX
        self.spatial.rebuild(&self.state.entities);

        // COMPUTE
        self.delta_buf.clear();
        let compute_start = Instant::now();

        for i in 0..self.state.entities.len() {
            if !self.state.entities[i].alive { continue; }
            let fid = entity_fidelity_idx(i, &self.state);
            match fid {
                Fidelity::High => {
                    profile.high_count += 1;
                    super::compute_high::compute_entity_deltas_into(
                        &self.state.entities[i], &self.state, &mut self.delta_buf,
                    );
                }
                Fidelity::Medium => {
                    profile.medium_count += 1;
                    super::compute_medium::compute_entity_deltas_into(
                        &self.state.entities[i], &self.state, &mut self.delta_buf,
                    );
                }
                Fidelity::Low => {
                    profile.low_count += 1;
                    super::compute_low::compute_entity_deltas_into(
                        &self.state.entities[i], &self.state, &mut self.delta_buf,
                    );
                }
                Fidelity::Background => {
                    profile.background_count += 1;
                }
            }
        }

        // Campaign systems (all 122, each gated by its own cadence).
        super::systems::compute_all_systems(&self.state, &mut self.delta_buf);

        let grid_start = Instant::now();
        for i in 0..self.state.grids.len() {
            compute_grid_deltas_into(&self.state.grids[i], &self.spatial, &mut self.delta_buf);
        }
        profile.compute_grid_us = grid_start.elapsed().as_micros() as u64;
        profile.compute_us = compute_start.elapsed().as_micros() as u64;
        profile.delta_count = self.delta_buf.len();
        profile.entities_processed = profile.high_count + profile.medium_count
            + profile.low_count + profile.background_count;

        // MERGE (flat arrays, no HashMap)
        let merge_start = Instant::now();
        self.merged.clear();
        for delta in self.delta_buf.drain(..) {
            self.merged.merge_one(delta);
        }
        profile.merge_us = merge_start.elapsed().as_micros() as u64;

        // APPLY (in-place, reads flat arrays)
        let apply_start = Instant::now();
        let apply_profile = apply_flat(&mut self.state, &self.merged);
        self.state.tick += 1;
        profile.apply_us = apply_start.elapsed().as_micros() as u64;
        profile.apply_hp_us = apply_profile.hp_us;
        profile.apply_movement_us = apply_profile.movement_us;
        profile.apply_status_us = apply_profile.status_us;
        profile.apply_economy_us = apply_profile.economy_us;
        profile.apply_deaths_us = apply_profile.deaths_us;
        profile.apply_grid_us = apply_profile.grid_us;
        profile.apply_fidelity_us = apply_profile.fidelity_us;
        profile.apply_price_reports_us = apply_profile.price_reports_us;

        profile.total_us = tick_start.elapsed().as_micros() as u64;
        self.profile_acc.record(&profile);
        profile
    }
}

fn entity_fidelity_idx(idx: usize, state: &WorldState) -> Fidelity {
    let entity = &state.entities[idx];
    if let Some(grid_id) = entity.grid_id {
        state.grid(grid_id)
            .map(|g| g.fidelity)
            .unwrap_or(Fidelity::Low)
    } else {
        Fidelity::Low
    }
}

fn compute_grid_deltas_into(
    grid: &super::state::LocalGrid,
    spatial: &SpatialIndex,
    out: &mut Vec<WorldDelta>,
) {
    let has_hostiles = spatial.has_hostiles_in_radius(grid.center, grid.radius);
    let has_friendlies = spatial.has_friendlies_in_radius(grid.center, grid.radius);

    let desired = if has_hostiles && has_friendlies {
        Fidelity::High
    } else if has_friendlies {
        Fidelity::Medium
    } else if has_hostiles {
        Fidelity::Low
    } else {
        Fidelity::Background
    };

    if desired != grid.fidelity {
        out.push(WorldDelta::EscalateFidelity {
            grid_id: grid.id,
            new_fidelity: desired,
        });
    }
}
