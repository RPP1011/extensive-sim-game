//! Zero-allocation runtime for the world simulation.
//!
//! All buffers are pre-allocated at init. Ticking reuses capacity via dirty-list
//! clears — no heap allocations after `WorldSim::new()`. Merge uses flat arrays
//! indexed by entity/settlement ID instead of HashMaps.

use std::time::Instant;

use super::delta::WorldDelta;
use super::fidelity::Fidelity;
use super::state::Entity;
use super::spatial::SpatialIndex;
use super::state::{WorldState, EntityKind};
use super::tick::{TickProfile, ProfileAccumulator};
use super::NUM_COMMODITIES;
use super::apply::ApplyProfile;

// ---------------------------------------------------------------------------
// FlatMergedDeltas — flat-array merge accumulator, zero HashMap overhead
// ---------------------------------------------------------------------------

const NUM_ENTITY_FIELDS: usize = 18; // 17 original + Level
const NUM_FACTION_FIELDS: usize = 6;
const NUM_REGION_FIELDS: usize = 4;
const NUM_SETTLEMENT_FIELDS: usize = 4;

/// Flat-array merge accumulator. All per-entity fields are indexed by entity ID.
/// Per-settlement fields indexed by settlement ID. Dirty lists track which
/// indices were touched so clear is O(touched) not O(capacity).
struct FlatMergedDeltas {
    // --- Per-entity combat (indexed by entity ID) ---
    damage: Vec<f32>,
    heals: Vec<f32>,
    shields: Vec<f32>,
    force_x: Vec<f32>,
    force_y: Vec<f32>,
    dead: Vec<bool>,
    entity_dirty: Vec<u32>,

    // --- Per-entity campaign fields (indexed by entity_id * NUM_ENTITY_FIELDS + field) ---
    entity_fields: Vec<f32>,
    entity_xp: Vec<u32>,
    entity_mood: Vec<u8>,
    entity_mood_set: Vec<bool>, // whether mood was set this tick

    // --- Per-settlement (indexed by settlement ID) ---
    production: Vec<[f32; NUM_COMMODITIES]>,
    consumption: Vec<[f32; NUM_COMMODITIES]>,
    stockpile_adj: Vec<[f32; NUM_COMMODITIES]>,
    treasury_adj: Vec<f32>,
    settlement_dirty: Vec<u32>,

    // --- Per-settlement campaign fields (indexed by settlement_id * NUM_SETTLEMENT_FIELDS + field) ---
    settlement_fields: Vec<f32>,

    // --- Per-faction (indexed by faction_id * NUM_FACTION_FIELDS + field) ---
    faction_fields: Vec<f32>,
    faction_dirty: Vec<u32>,
    max_factions: usize,

    // --- Per-region (indexed by region_id * NUM_REGION_FIELDS + field) ---
    region_fields: Vec<f32>,
    region_dirty: Vec<u32>,
    max_regions: usize,

    // --- Guild scalars ---
    guild_gold_delta: f32,
    guild_supplies_delta: f32,
    guild_reputation_delta: f32,

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

    // --- Behavior tags (collected) ---
    behavior_tag_deltas: Vec<(u32, [(u32, f32); 8], u8)>,

    // --- Intent changes (last-writer-wins, collected) ---
    intent_changes: Vec<(u32, super::state::EconomicIntent)>,

    // --- Campaign collected (no flat-array equivalent) ---
    relation_deltas: Vec<(u32, u32, u8, f32)>,
    spawns: Vec<(super::state::EntityKind, (f32, f32), super::state::WorldTeam, u32)>,
    events: Vec<super::state::WorldEvent>,
    chronicles: Vec<super::state::ChronicleEntry>,
    quest_updates: Vec<(u32, super::state::QuestDelta)>,
}

impl FlatMergedDeltas {
    fn new(max_entities: usize, max_settlements: usize) -> Self {
        let max_factions = 16; // pre-allocate for up to 16 factions
        let max_regions = 16;
        FlatMergedDeltas {
            damage: vec![0.0; max_entities],
            heals: vec![0.0; max_entities],
            shields: vec![0.0; max_entities],
            force_x: vec![0.0; max_entities],
            force_y: vec![0.0; max_entities],
            dead: vec![false; max_entities],
            entity_dirty: Vec::with_capacity(max_entities),

            entity_fields: vec![0.0; max_entities * NUM_ENTITY_FIELDS],
            entity_xp: vec![0; max_entities],
            entity_mood: vec![0; max_entities],
            entity_mood_set: vec![false; max_entities],

            production: vec![[0.0; NUM_COMMODITIES]; max_settlements],
            consumption: vec![[0.0; NUM_COMMODITIES]; max_settlements],
            stockpile_adj: vec![[0.0; NUM_COMMODITIES]; max_settlements],
            treasury_adj: vec![0.0; max_settlements],
            settlement_dirty: Vec::with_capacity(max_settlements),
            settlement_fields: vec![0.0; max_settlements * NUM_SETTLEMENT_FIELDS],

            faction_fields: vec![0.0; max_factions * NUM_FACTION_FIELDS],
            faction_dirty: Vec::with_capacity(max_factions),
            max_factions,

            region_fields: vec![0.0; max_regions * NUM_REGION_FIELDS],
            region_dirty: Vec::with_capacity(max_regions),
            max_regions,

            guild_gold_delta: 0.0,
            guild_supplies_delta: 0.0,
            guild_reputation_delta: 0.0,

            new_statuses: Vec::with_capacity(64),
            remove_statuses: Vec::with_capacity(64),
            gold_transfers: Vec::with_capacity(64),
            goods_transfers: Vec::with_capacity(64),
            grid_enters: Vec::with_capacity(32),
            grid_leaves: Vec::with_capacity(32),
            fidelity_changes: Vec::with_capacity(16),
            price_reports: Vec::with_capacity(64),
            price_updates: Vec::with_capacity(16),

            behavior_tag_deltas: Vec::with_capacity(64),
            intent_changes: Vec::with_capacity(64),
            relation_deltas: Vec::with_capacity(64),
            spawns: Vec::with_capacity(16),
            events: Vec::with_capacity(32),
            chronicles: Vec::with_capacity(16),
            quest_updates: Vec::with_capacity(16),
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
            // Clear entity campaign fields
            let base = i * NUM_ENTITY_FIELDS;
            for f in 0..NUM_ENTITY_FIELDS { self.entity_fields[base + f] = 0.0; }
            self.entity_xp[i] = 0;
            self.entity_mood_set[i] = false;
        }
        self.entity_dirty.clear();

        for &id in &self.settlement_dirty {
            let i = id as usize;
            self.production[i] = [0.0; NUM_COMMODITIES];
            self.consumption[i] = [0.0; NUM_COMMODITIES];
            self.stockpile_adj[i] = [0.0; NUM_COMMODITIES];
            self.treasury_adj[i] = 0.0;
            let base = i * NUM_SETTLEMENT_FIELDS;
            for f in 0..NUM_SETTLEMENT_FIELDS { self.settlement_fields[base + f] = 0.0; }
        }
        self.settlement_dirty.clear();

        for &id in &self.faction_dirty {
            let base = id as usize * NUM_FACTION_FIELDS;
            for f in 0..NUM_FACTION_FIELDS { self.faction_fields[base + f] = 0.0; }
        }
        self.faction_dirty.clear();

        for &id in &self.region_dirty {
            let base = id as usize * NUM_REGION_FIELDS;
            for f in 0..NUM_REGION_FIELDS { self.region_fields[base + f] = 0.0; }
        }
        self.region_dirty.clear();

        self.guild_gold_delta = 0.0;
        self.guild_supplies_delta = 0.0;
        self.guild_reputation_delta = 0.0;

        self.new_statuses.clear();
        self.remove_statuses.clear();
        self.gold_transfers.clear();
        self.goods_transfers.clear();
        self.grid_enters.clear();
        self.grid_leaves.clear();
        self.fidelity_changes.clear();
        self.price_reports.clear();
        self.price_updates.clear();

        self.behavior_tag_deltas.clear();
        self.intent_changes.clear();
        self.relation_deltas.clear();
        self.spawns.clear();
        self.events.clear();
        self.chronicles.clear();
        self.quest_updates.clear();
    }

    #[inline]
    fn mark_entity(&mut self, id: u32) {
        let i = id as usize;
        if self.damage[i] == 0.0 && self.heals[i] == 0.0 && self.shields[i] == 0.0
            && self.force_x[i] == 0.0 && self.force_y[i] == 0.0 && !self.dead[i]
            && self.entity_xp[i] == 0 && !self.entity_mood_set[i]
        {
            // Check if any entity_fields are nonzero
            let base = i * NUM_ENTITY_FIELDS;
            let fields_clean = (0..NUM_ENTITY_FIELDS).all(|f| self.entity_fields[base + f] == 0.0);
            if fields_clean {
                self.entity_dirty.push(id);
            }
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
            let base = i * NUM_SETTLEMENT_FIELDS;
            let fields_clean = (0..NUM_SETTLEMENT_FIELDS).all(|f| self.settlement_fields[base + f] == 0.0);
            if fields_clean {
                self.settlement_dirty.push(id);
            }
        }
    }

    #[inline]
    fn mark_faction(&mut self, id: u32) {
        let i = id as usize;
        if i < self.max_factions {
            let base = i * NUM_FACTION_FIELDS;
            let clean = (0..NUM_FACTION_FIELDS).all(|f| self.faction_fields[base + f] == 0.0);
            if clean { self.faction_dirty.push(id); }
        }
    }

    #[inline]
    fn mark_region(&mut self, id: u32) {
        let i = id as usize;
        if i < self.max_regions {
            let base = i * NUM_REGION_FIELDS;
            let clean = (0..NUM_REGION_FIELDS).all(|f| self.region_fields[base + f] == 0.0);
            if clean { self.region_dirty.push(id); }
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

            // --- Campaign deltas: flat-array path ---
            WorldDelta::UpdateEntityField { entity_id, field, value } => {
                self.mark_entity(entity_id);
                self.entity_fields[entity_id as usize * NUM_ENTITY_FIELDS + field as usize] += value;
            }
            WorldDelta::SetEntityMood { entity_id, mood } => {
                self.mark_entity(entity_id);
                self.entity_mood[entity_id as usize] = mood;
                self.entity_mood_set[entity_id as usize] = true;
            }
            WorldDelta::AddXp { entity_id, amount } => {
                self.mark_entity(entity_id);
                self.entity_xp[entity_id as usize] = self.entity_xp[entity_id as usize].saturating_add(amount);
            }
            WorldDelta::UpdateFaction { faction_id, field, value } => {
                self.mark_faction(faction_id);
                let i = faction_id as usize;
                if i < self.max_factions {
                    self.faction_fields[i * NUM_FACTION_FIELDS + field as usize] += value;
                }
            }
            WorldDelta::UpdateRegion { region_id, field, value } => {
                self.mark_region(region_id);
                let i = region_id as usize;
                if i < self.max_regions {
                    self.region_fields[i * NUM_REGION_FIELDS + field as usize] += value;
                }
            }
            WorldDelta::UpdateSettlementField { settlement_id, field, value } => {
                self.mark_settlement(settlement_id);
                self.settlement_fields[settlement_id as usize * NUM_SETTLEMENT_FIELDS + field as usize] += value;
            }
            WorldDelta::UpdateRelation { entity_a, entity_b, kind, delta } => {
                self.relation_deltas.push((entity_a, entity_b, kind as u8, delta));
            }
            WorldDelta::SpawnEntity { kind, pos, team, level } => {
                self.spawns.push((kind, pos, team, level));
            }
            WorldDelta::RecordEvent { event } => {
                self.events.push(event);
            }
            WorldDelta::RecordChronicle { entry } => {
                self.chronicles.push(entry);
            }
            WorldDelta::QuestUpdate { quest_id, update } => {
                self.quest_updates.push((quest_id, update));
            }
            WorldDelta::UpdateGuildGold { delta } => {
                self.guild_gold_delta += delta;
            }
            WorldDelta::UpdateGuildSupplies { delta } => {
                self.guild_supplies_delta += delta;
            }
            WorldDelta::UpdateGuildReputation { delta } => {
                self.guild_reputation_delta += delta;
            }
            WorldDelta::AddBehaviorTags { entity_id, tags, count } => {
                self.behavior_tag_deltas.push((entity_id, tags, count));
            }
            WorldDelta::SetIntent { entity_id, intent } => {
                self.intent_changes.push((entity_id, intent));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flat apply — reads FlatMergedDeltas arrays directly
// ---------------------------------------------------------------------------


fn apply_flat(state: &mut WorldState, m: &FlatMergedDeltas) -> ApplyProfile {
    let mut p = ApplyProfile::default();

    // Helper: O(1) entity index lookup (inline to avoid borrow issues).
    let idx = &state.entity_index;
    let ents = &mut state.entities;

    // HP changes
    let t = Instant::now();
    for &id in &m.entity_dirty {
        let i = id as usize;
        let damage = m.damage[i];
        let heal = m.heals[i];
        let shield_add = m.shields[i];
        if damage == 0.0 && heal == 0.0 && shield_add == 0.0 { continue; }

        if i < idx.len() {
            let ei = idx[i] as usize;
            if ei < ents.len() {
                let e = &mut ents[ei];
                // Revive dead entities that receive healing (entity pool recycling).
                if !e.alive {
                    if heal > 0.0 && damage == 0.0 {
                        e.alive = true;
                        e.hp = heal.min(e.max_hp);
                    }
                    continue;
                }
                let shield_absorb = damage.min(e.shield_hp);
                e.shield_hp = e.shield_hp - shield_absorb + shield_add;
                let remaining = damage - shield_absorb;
                e.hp = (e.hp + heal - remaining).clamp(0.0, e.max_hp);
            }
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

        if i < idx.len() {
            let ei = idx[i] as usize;
            if ei < ents.len() {
                let e = &mut ents[ei];
                if !e.alive { continue; }
                let mag = (fx * fx + fy * fy).sqrt();
                let max_speed = e.move_speed * crate::world_sim::DT_SEC;
                let (dx, dy) = if mag > max_speed && mag > 0.001 {
                    (fx / mag * max_speed, fy / mag * max_speed)
                } else {
                    (fx, fy)
                };
                e.pos.0 += dx;
                e.pos.1 += dy;
            }
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

    // Campaign system deltas — flat-array apply
    let t = Instant::now();

    // Entity field deltas + XP + mood
    for &id in &m.entity_dirty {
        let i = id as usize;
        let base = i * NUM_ENTITY_FIELDS;
        let has_fields = (0..NUM_ENTITY_FIELDS).any(|f| m.entity_fields[base + f] != 0.0);
        let has_xp = m.entity_xp[i] > 0;
        let has_mood = m.entity_mood_set[i];

        if !has_fields && !has_xp && !has_mood { continue; }

        let ei = if i < state.entity_index.len() { state.entity_index[i] as usize } else { usize::MAX };
        if ei < state.entities.len() {
            let entity = &mut state.entities[ei];
            if has_fields {
                super::apply::apply_entity_field_delta(entity, 0, m.entity_fields[base + 0]); // Morale
                super::apply::apply_entity_field_delta(entity, 1, m.entity_fields[base + 1]); // Stress
                super::apply::apply_entity_field_delta(entity, 2, m.entity_fields[base + 2]); // Fatigue
                super::apply::apply_entity_field_delta(entity, 3, m.entity_fields[base + 3]); // Loyalty
                super::apply::apply_entity_field_delta(entity, 4, m.entity_fields[base + 4]); // Injury
                super::apply::apply_entity_field_delta(entity, 5, m.entity_fields[base + 5]); // Resolve
                super::apply::apply_entity_field_delta(entity, 6, m.entity_fields[base + 6]); // GuildRelationship
                super::apply::apply_entity_field_delta(entity, 7, m.entity_fields[base + 7]); // Gold
                super::apply::apply_entity_field_delta(entity, 8, m.entity_fields[base + 8]); // Hp
                super::apply::apply_entity_field_delta(entity, 9, m.entity_fields[base + 9]); // MaxHp
                super::apply::apply_entity_field_delta(entity, 10, m.entity_fields[base + 10]); // ShieldHp
                super::apply::apply_entity_field_delta(entity, 11, m.entity_fields[base + 11]); // Armor
                super::apply::apply_entity_field_delta(entity, 12, m.entity_fields[base + 12]); // MagicResist
                super::apply::apply_entity_field_delta(entity, 13, m.entity_fields[base + 13]); // AttackDamage
                super::apply::apply_entity_field_delta(entity, 14, m.entity_fields[base + 14]); // AttackRange
                super::apply::apply_entity_field_delta(entity, 15, m.entity_fields[base + 15]); // MoveSpeed
                super::apply::apply_entity_field_delta(entity, 16, m.entity_fields[base + 16]); // Level
            }
            if has_xp {
                if let Some(npc) = entity.npc.as_mut() {
                    npc.xp = npc.xp.saturating_add(m.entity_xp[i]);
                }
            }
            if has_mood {
                if let Some(npc) = entity.npc.as_mut() {
                    npc.mood = m.entity_mood[i];
                }
            }
        }
    }

    // Faction field deltas
    for &id in &m.faction_dirty {
        let base = id as usize * NUM_FACTION_FIELDS;
        if let Some(faction) = state.faction_mut(id) {
            super::apply::apply_faction_field_delta(faction, 0, m.faction_fields[base + 0]);
            super::apply::apply_faction_field_delta(faction, 1, m.faction_fields[base + 1]);
            super::apply::apply_faction_field_delta(faction, 2, m.faction_fields[base + 2]);
            super::apply::apply_faction_field_delta(faction, 3, m.faction_fields[base + 3]);
            super::apply::apply_faction_field_delta(faction, 4, m.faction_fields[base + 4]);
            super::apply::apply_faction_field_delta(faction, 5, m.faction_fields[base + 5]);
        }
    }

    // Region field deltas
    for &id in &m.region_dirty {
        let base = id as usize * NUM_REGION_FIELDS;
        if let Some(region) = state.region_mut(id) {
            super::apply::apply_region_field_delta(region, 0, m.region_fields[base + 0]);
            super::apply::apply_region_field_delta(region, 1, m.region_fields[base + 1]);
            super::apply::apply_region_field_delta(region, 2, m.region_fields[base + 2]);
            super::apply::apply_region_field_delta(region, 3, m.region_fields[base + 3]);
        }
    }

    // Settlement field deltas
    for &id in &m.settlement_dirty {
        let base = id as usize * NUM_SETTLEMENT_FIELDS;
        if let Some(s) = state.settlement_mut(id) {
            super::apply::apply_settlement_field_delta(s, 0, m.settlement_fields[base + 0]);
            super::apply::apply_settlement_field_delta(s, 1, m.settlement_fields[base + 1]);
            super::apply::apply_settlement_field_delta(s, 2, m.settlement_fields[base + 2]);
            super::apply::apply_settlement_field_delta(s, 3, m.settlement_fields[base + 3]);
        }
    }

    // Relations
    for &(a, b, kind, delta) in &m.relation_deltas {
        let entry = state.relations.entry((a, b, kind)).or_insert(0.0);
        *entry += delta;
    }

    // Spawns
    for &(kind, pos, team, level) in &m.spawns {
        let id = state.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
        let mut entity = match kind {
            EntityKind::Npc => Entity::new_npc(id, pos),
            EntityKind::Monster => Entity::new_monster(id, pos, level),
            EntityKind::Building => Entity::new_building(id, pos),
            EntityKind::Projectile => Entity::new_monster(id, pos, 0),
        };
        entity.team = team;
        entity.level = level;
        state.entities.push(entity);
    }

    // Events
    state.world_events.extend(m.events.iter().cloned());
    if state.world_events.len() > 1000 {
        let drain = state.world_events.len() - 1000;
        state.world_events.drain(..drain);
    }

    // Chronicles
    state.chronicle.extend(m.chronicles.iter().cloned());
    if state.chronicle.len() > 500 {
        let drain = state.chronicle.len() - 500;
        state.chronicle.drain(..drain);
    }

    // Quest updates
    for (quest_id, update) in &m.quest_updates {
        if let Some(quest) = state.quest_mut(*quest_id) {
            match update {
                super::state::QuestDelta::AdvanceProgress { amount } => {
                    quest.progress = (quest.progress + amount).min(1.0);
                }
                super::state::QuestDelta::SetStatus { status } => {
                    quest.status = *status;
                }
                super::state::QuestDelta::AddMember { entity_id } => {
                    if !quest.party_member_ids.contains(entity_id) {
                        quest.party_member_ids.push(*entity_id);
                    }
                }
                super::state::QuestDelta::RemoveMember { entity_id } => {
                    quest.party_member_ids.retain(|id| id != entity_id);
                }
                super::state::QuestDelta::Complete => {
                    quest.status = super::state::QuestStatus::Completed;
                    quest.progress = 1.0;
                }
                super::state::QuestDelta::Fail => {
                    quest.status = super::state::QuestStatus::Failed;
                }
            }
        }
    }

    // Guild
    state.guild.gold += m.guild_gold_delta;
    state.guild.supplies += m.guild_supplies_delta;
    state.guild.reputation += m.guild_reputation_delta;

    // Behavior tag accumulation
    for &(entity_id, tags, count) in &m.behavior_tag_deltas {
        let ei = if (entity_id as usize) < state.entity_index.len() {
            state.entity_index[entity_id as usize] as usize
        } else { continue };
        if ei >= state.entities.len() { continue; }
        if let Some(npc) = state.entities[ei].npc.as_mut() {
            let action = crate::world_sim::state::ActionTags { tags, count };
            npc.accumulate_tags(&action);
        }
    }

    // Intent changes (last-writer-wins: iterate in reverse, skip already-set)
    for (entity_id, intent) in m.intent_changes.iter().rev() {
        let ei = if (*entity_id as usize) < state.entity_index.len() {
            state.entity_index[*entity_id as usize] as usize
        } else { continue };
        if ei >= state.entities.len() { continue; }
        if let Some(npc) = state.entities[ei].npc.as_mut() {
            npc.economic_intent = intent.clone();
        }
    }

    p.campaign_us = t.elapsed().as_micros() as u64;
    p
}

// ---------------------------------------------------------------------------
// WorldSim — zero-alloc runtime
// ---------------------------------------------------------------------------

/// Pre-allocated world simulation runtime.
pub struct WorldSim {
    /// Single state, mutated in-place.
    state: WorldState,

    /// Reusable delta buffer (entity-level compute + global systems).
    delta_buf: Vec<WorldDelta>,

    /// Flat-array merge accumulator.
    merged: FlatMergedDeltas,

    /// Reusable spatial index.
    spatial: SpatialIndex,

    /// Dedicated thread pool — stays warm across ticks, no wake latency.
    pool: Option<rayon::ThreadPool>,

    /// Per-thread reusable delta buffers. One per rayon thread.
    /// Avoids Vec allocation per settlement per tick.
    thread_bufs: Vec<std::sync::Mutex<Vec<WorldDelta>>>,

    /// Accumulated profiling stats.
    pub profile_acc: ProfileAccumulator,

    /// Swappable class generator (matches behavior profiles to class templates).
    pub class_gen: Box<dyn super::class_gen::ClassGenerator>,

    /// Swappable ability generator (produces abilities on class level-up).
    pub ability_gen: Box<dyn super::class_gen::AbilityGenerator>,

    /// Async naming queue + resolved name cache for LFM model integration.
    pub naming: super::naming::NamingService,
}

impl WorldSim {
    pub fn new(mut initial: WorldState) -> Self {
        let max_entity_id = initial.entities.iter().map(|e| e.id).max().unwrap_or(0) as usize + 1;
        let max_settlement_id = initial.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;

        // Sort entities by settlement/party, build hot/cold + all indices.
        initial.rebuild_all_indices();

        // Create dedicated thread pool for large worlds.
        let num_threads = rayon::current_num_threads();
        let (pool, thread_bufs) = if initial.entities.len() > 10_000 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .ok();
            let bufs: Vec<std::sync::Mutex<Vec<WorldDelta>>> = (0..num_threads)
                .map(|_| std::sync::Mutex::new(Vec::with_capacity(initial.entities.len() * 4 / num_threads)))
                .collect();
            (pool, bufs)
        } else {
            (None, Vec::new())
        };

        WorldSim {
            delta_buf: Vec::with_capacity(initial.entities.len() * 4),
            merged: FlatMergedDeltas::new(max_entity_id, max_settlement_id),
            spatial: SpatialIndex::new(),
            pool,
            thread_bufs,
            profile_acc: ProfileAccumulator::default(),
            state: initial,
            class_gen: Box::new(super::class_gen::DefaultClassGenerator::new()),
            ability_gen: Box::new(super::class_gen::DefaultAbilityGenerator),
            naming: super::naming::NamingService::new(),
        }
    }

    pub fn state(&self) -> &WorldState {
        &self.state
    }

    /// Run campaign systems with per-settlement parallel dispatch.
    /// Uses dedicated thread pool + pre-allocated per-thread buffers.
    fn compute_campaign_systems_par(&mut self) {
        use rayon::prelude::*;

        let state = &self.state;
        let thread_bufs = &self.thread_bufs;

        // Clear all thread buffers (O(threads), not O(settlements)).
        for buf in thread_bufs.iter() {
            buf.lock().unwrap().clear();
        }

        let do_parallel = |settlements: &[super::state::SettlementState]| {
            settlements.par_iter().for_each(|settlement| {
                let range = state.group_index.settlement_entities(settlement.id);
                let entities = &state.entities[range];

                // Get this thread's buffer. rayon assigns threads by index.
                let thread_idx = rayon::current_thread_index().unwrap_or(0);
                let buf_idx = thread_idx % thread_bufs.len().max(1);
                let mut buf = thread_bufs[buf_idx].lock().unwrap();

                run_settlement_systems(state, settlement.id, entities, &mut buf);
            });
        };

        // Run on dedicated pool if available, else global pool.
        if let Some(pool) = &self.pool {
            pool.install(|| do_parallel(&state.settlements));
        } else {
            do_parallel(&state.settlements);
        }

        // Drain all thread buffers into main delta_buf.
        for buf in thread_bufs.iter() {
            let mut b = buf.lock().unwrap();
            self.delta_buf.extend(b.drain(..));
        }

        // Global systems (sequential).
        super::systems::compute_global_systems(state, &mut self.delta_buf);
    }
}

/// Run all settlement-scoped systems for one settlement.
/// Extracted so both parallel and sequential paths can call it.
fn run_settlement_systems(
    state: &super::state::WorldState,
    sid: u32,
    entities: &[super::state::Entity],
    buf: &mut Vec<WorldDelta>,
) {
    use super::systems::*;
    economy::compute_economy_for_settlement(state, sid, entities, buf);
    food::compute_food_for_settlement(state, sid, entities, buf);
    population::compute_population_for_settlement(state, sid, entities, buf);
    mentorship::compute_mentorship_for_settlement(state, sid, entities, buf);
    adventurer_condition::compute_adventurer_condition_for_settlement(state, sid, entities, buf);
    adventurer_recovery::compute_adventurer_recovery_for_settlement(state, sid, entities, buf);
    progression::compute_progression_for_settlement(state, sid, entities, buf);
    class_progression::compute_class_progression_for_settlement(state, sid, entities, buf);
    recruitment::compute_recruitment_for_settlement(state, sid, entities, buf);
    npc_decisions::compute_npc_decisions_for_settlement(state, sid, entities, buf);
    price_discovery::compute_price_discovery_for_settlement(state, sid, entities, buf);
    quests::compute_quests_for_settlement(state, sid, entities, buf);
    retirement::compute_retirement_for_settlement(state, sid, entities, buf);
    hobbies::compute_hobbies_for_settlement(state, sid, entities, buf);
    fears::compute_fears_for_settlement(state, sid, entities, buf);
    personal_goals::compute_personal_goals_for_settlement(state, sid, entities, buf);
    journals::compute_journals_for_settlement(state, sid, entities, buf);
    wound_persistence::compute_wound_persistence_for_settlement(state, sid, entities, buf);
    addiction::compute_addiction_for_settlement(state, sid, entities, buf);
    equipment_durability::compute_equipment_durability_for_settlement(state, sid, entities, buf);
    moods::compute_moods_for_settlement(state, sid, entities, buf);
    bonds::compute_bonds_for_settlement(state, sid, entities, buf);
    npc_relationships::compute_npc_relationships_for_settlement(state, sid, entities, buf);
    npc_reputation::compute_npc_reputation_for_settlement(state, sid, entities, buf);
    romance::compute_romance_for_settlement(state, sid, entities, buf);
    rivalries::compute_rivalries_for_settlement(state, sid, entities, buf);
    companions::compute_companions_for_settlement(state, sid, entities, buf);
    party_chemistry::compute_party_chemistry_for_settlement(state, sid, entities, buf);
    nicknames::compute_nicknames_for_settlement(state, sid, entities, buf);
    legendary_deeds::compute_legendary_deeds_for_settlement(state, sid, entities, buf);
    folk_hero::compute_folk_hero_for_settlement(state, sid, entities, buf);
    memorials::compute_memorials_for_settlement(state, sid, entities, buf);
    trophies::compute_trophies_for_settlement(state, sid, entities, buf);
    awakening::compute_awakening_for_settlement(state, sid, entities, buf);
    visions::compute_visions_for_settlement(state, sid, entities, buf);
    bloodlines::compute_bloodlines_for_settlement(state, sid, entities, buf);
    divine_favor::compute_divine_favor_for_settlement(state, sid, entities, buf);
    religion::compute_religion_for_settlement(state, sid, entities, buf);
    demonic_pacts::compute_demonic_pacts_for_settlement(state, sid, entities, buf);
    legacy_weapons::compute_legacy_weapons_for_settlement(state, sid, entities, buf);
    cooldowns::compute_cooldowns_for_settlement(state, sid, entities, buf);
    battles::compute_battles_for_settlement(state, sid, entities, buf);
    loot::compute_loot_for_settlement(state, sid, entities, buf);
    last_stand::compute_last_stand_for_settlement(state, sid, entities, buf);
    skill_challenges::compute_skill_challenges_for_settlement(state, sid, entities, buf);
    dungeons::compute_dungeons_for_settlement(state, sid, entities, buf);
    escalation_protocol::compute_escalation_protocol_for_settlement(state, sid, entities, buf);
    trade_goods::compute_trade_goods_for_settlement(state, sid, entities, buf);
    infrastructure::compute_infrastructure_for_settlement(state, sid, entities, buf);
    crafting::compute_crafting_for_settlement(state, sid, entities, buf);
    buildings::compute_buildings_for_settlement(state, sid, entities, buf);
    guild_rooms::compute_guild_rooms_for_settlement(state, sid, entities, buf);
    guild_tiers::compute_guild_tiers_for_settlement(state, sid, entities, buf);
    festivals::compute_festivals_for_settlement(state, sid, entities, buf);
    supply::compute_supply_for_settlement(state, sid, entities, buf);
    exploration::compute_exploration_for_settlement(state, sid, entities, buf);
}

impl WorldSim {
    pub fn tick(&mut self) -> TickProfile {
        let tick_start = Instant::now();
        let mut profile = TickProfile::default();

        // SPATIAL INDEX — rebuild from hot entities (position data).
        self.spatial.rebuild(&self.state.entities);

        // COMPUTE — iterate hot array for fast filtering.
        self.delta_buf.clear();
        let compute_start = Instant::now();

        for i in 0..self.state.hot.len() {
            let h = &self.state.hot[i];
            if !h.alive { continue; }
            let fid = hot_entity_fidelity(h, &self.state);
            // Systems still receive full Entity refs for compatibility.
            // The hot iteration is just for the filter loop — the actual compute
            // still reads from state.entities[i].
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

        // Campaign systems: parallel above 10K entities, sequential below.
        if self.state.entities.len() > 10_000 {
            self.compute_campaign_systems_par();
        } else {
            super::systems::compute_all_systems(&self.state, &mut self.delta_buf);
        }

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

        // POST-APPLY: process world events into state changes.
        self.process_world_events();

        // CLASS MATCHING — run after apply so behavior tags are up to date.
        if self.state.tick % 50 == 0 && self.state.tick > 0 {
            self.run_class_matching();
        }

        // Sync hot array from entities (fast — scalar copy only).
        // Full rebuild only if entities were spawned (array length changed).
        if self.state.entities.len() != self.state.hot.len() {
            // Entities spawned/removed — full rebuild with re-sort.
            self.state.rebuild_all_indices();
        } else {
            self.state.sync_hot_from_entities();
        }

        profile.total_us = tick_start.elapsed().as_micros() as u64;
        self.profile_acc.record(&profile);
        profile
    }

    /// Match NPC behavior profiles against class templates and grant new classes.
    /// Runs after apply so behavior_tags are up to date.
    /// Convert world events into state changes (quest postings, etc.).
    fn process_world_events(&mut self) {
        use super::state::{WorldEvent, QuestPosting, QuestType};

        let mut next_quest_id = self.state.quest_board.iter()
            .map(|q| q.id)
            .max()
            .unwrap_or(0) + 1;

        // Process events that create quest postings.
        for event in &self.state.world_events {
            match event {
                WorldEvent::QuestPosted { settlement_id, threat_level, reward_gold } => {
                    // Don't duplicate.
                    if self.state.quest_board.iter().any(|q| q.settlement_id == *settlement_id) {
                        continue;
                    }
                    if self.state.quest_board.len() >= 20 { break; }

                    let settlement_pos = self.state.settlement(*settlement_id)
                        .map(|s| s.pos)
                        .unwrap_or((0.0, 0.0));

                    self.state.quest_board.push(QuestPosting {
                        id: next_quest_id,
                        name: format!("Clear threat near settlement {}", settlement_id),
                        quest_type: QuestType::Hunt,
                        settlement_id: *settlement_id,
                        destination: settlement_pos,
                        threat_level: *threat_level,
                        reward_gold: *reward_gold,
                        reward_xp: (*threat_level * 5.0) as u32,
                        expires_tick: self.state.tick + 500,
                    });
                    next_quest_id += 1;
                }
                _ => {}
            }
        }

        // Expire old quest postings.
        self.state.quest_board.retain(|q| q.expires_tick > self.state.tick);
    }

    fn run_class_matching(&mut self) {
        use super::state::{EntityKind, ClassSlot};

        let min_behavior_sum = 10.0_f32;

        for entity in &mut self.state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };

            // Skip NPCs with insufficient behavior.
            let behavior_sum: f32 = npc.behavior_values.iter().sum();
            if behavior_sum < min_behavior_sum { continue; }

            // Match against class templates.
            let matches = self.class_gen.match_classes(&npc.behavior_tags, &npc.behavior_values);

            let entity_seed = self.state.tick.wrapping_mul(2654435761) ^ entity.id as u64;

            for class_match in &matches {
                // Skip if NPC already has this class.
                if npc.classes.iter().any(|c| c.class_name_hash == class_match.class_name_hash) {
                    continue;
                }

                // Generate procedural name from behavior profile.
                let display_name = super::naming::procedural_class_name(
                    &class_match.display_name,
                    &npc.behavior_tags,
                    &npc.behavior_values,
                    entity_seed ^ class_match.class_name_hash as u64,
                );

                // Grant the class.
                npc.classes.push(ClassSlot {
                    class_name_hash: class_match.class_name_hash,
                    level: 1,
                    xp: 0.0,
                    display_name,
                });
            }

            // If no templates matched and behavior is high, try unique class.
            if matches.is_empty() && behavior_sum > 500.0 && npc.classes.is_empty() {
                let seed = self.state.tick ^ entity.id as u64;
                if let Some(class_def) = self.class_gen.generate_unique_class(
                    &npc.behavior_tags, &npc.behavior_values, seed,
                ) {
                    let display_name = super::naming::procedural_class_name(
                        &class_def.display_name,
                        &npc.behavior_tags,
                        &npc.behavior_values,
                        seed,
                    );
                    npc.classes.push(ClassSlot {
                        class_name_hash: class_def.name_hash,
                        level: 1,
                        xp: 0.0,
                        display_name,
                    });
                }
            }
        }
    }
}

fn hot_entity_fidelity(h: &super::state::HotEntity, state: &WorldState) -> Fidelity {
    if let Some(grid_id) = h.grid_id {
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
