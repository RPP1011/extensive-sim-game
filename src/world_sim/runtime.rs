//! Low-allocation runtime for the world simulation.
//!
//! Most buffers are pre-allocated at init. Ticking reuses capacity via dirty-list
//! clears. The merge phase uses flat arrays indexed by entity/settlement ID for
//! commutative fields (damage, movement, economy), plus a small HashMap for
//! last-damage-source attribution. Vecs accumulate non-commutative items
//! (status effects, spawns, transfers).

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
const NUM_SETTLEMENT_FIELDS: usize = 5;

/// How often (in ticks) to update grid membership for combat proximity.
const GRID_MEMBERSHIP_INTERVAL: u64 = 10;
/// How often (in ticks) to run class matching against behavior tags.
const CLASS_MATCHING_INTERVAL: u64 = 50;
/// How often (in ticks) to compact dead entities from the array.
const ENTITY_COMPACTION_INTERVAL: u64 = 500;

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
    grid_enters: Vec<(u32, u32)>,
    grid_leaves: Vec<(u32, u32)>,
    fidelity_changes: Vec<(u32, Fidelity)>,
    price_reports: Vec<(u32, u32, super::state::PriceReport)>,
    price_updates: Vec<(u32, [f32; NUM_COMMODITIES])>,

    // --- Behavior tags (collected) ---
    behavior_tag_deltas: Vec<(u32, [(u32, f32); 8], u8)>,

    // --- Intent changes (last-writer-wins, collected) ---
    intent_changes: Vec<(u32, super::state::EconomicIntent)>,

    // --- Last damage source per target (for death attribution) ---
    last_damage_source: std::collections::HashMap<u32, u32>,

    // --- Item system ---
    item_spawns: Vec<((f32, f32), super::state::ItemData)>,
    equip_requests: Vec<(u32, u32)>,
    unequip_requests: Vec<(u32, u32)>,

    // --- Inventory transfers ---
    inventory_transfers: Vec<(u32, u32, usize, f32)>,
    inventory_gold_transfers: Vec<(u32, u32, f32)>,

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
            grid_enters: Vec::with_capacity(32),
            grid_leaves: Vec::with_capacity(32),
            fidelity_changes: Vec::with_capacity(16),
            price_reports: Vec::with_capacity(64),
            price_updates: Vec::with_capacity(16),

            behavior_tag_deltas: Vec::with_capacity(64),
            intent_changes: Vec::with_capacity(64),
            last_damage_source: std::collections::HashMap::with_capacity(64),
            item_spawns: Vec::with_capacity(32),
            equip_requests: Vec::with_capacity(32),
            unequip_requests: Vec::with_capacity(32),
            inventory_transfers: Vec::with_capacity(64),
            inventory_gold_transfers: Vec::with_capacity(32),
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
        self.grid_enters.clear();
        self.grid_leaves.clear();
        self.fidelity_changes.clear();
        self.price_reports.clear();
        self.price_updates.clear();

        self.behavior_tag_deltas.clear();
        self.intent_changes.clear();
        self.last_damage_source.clear();
        self.item_spawns.clear();
        self.equip_requests.clear();
        self.unequip_requests.clear();
        self.inventory_transfers.clear();
        self.inventory_gold_transfers.clear();
        self.relation_deltas.clear();
        self.spawns.clear();
        self.events.clear();
        self.chronicles.clear();
        self.quest_updates.clear();
    }

    /// Grow flat arrays to accommodate new entity IDs (after building spawns).
    fn ensure_capacity(&mut self, new_max: usize) {
        if new_max <= self.damage.len() { return; }
        self.damage.resize(new_max, 0.0);
        self.heals.resize(new_max, 0.0);
        self.shields.resize(new_max, 0.0);
        self.force_x.resize(new_max, 0.0);
        self.force_y.resize(new_max, 0.0);
        self.dead.resize(new_max, false);
        self.entity_fields.resize(new_max * NUM_ENTITY_FIELDS, 0.0);
        self.entity_mood_set.resize(new_max, false);
    }

    /// Mark entity as dirty if not already. Returns false if ID exceeds capacity.
    #[inline]
    fn mark_entity(&mut self, id: u32) -> bool {
        let i = id as usize;
        if i >= self.damage.len() { return false; }
        if self.damage[i] == 0.0 && self.heals[i] == 0.0 && self.shields[i] == 0.0
            && self.force_x[i] == 0.0 && self.force_y[i] == 0.0 && !self.dead[i]
            && !self.entity_mood_set[i]
        {
            // Check if any entity_fields are nonzero
            let base = i * NUM_ENTITY_FIELDS;
            let fields_clean = (0..NUM_ENTITY_FIELDS).all(|f| self.entity_fields[base + f] == 0.0);
            if fields_clean {
                self.entity_dirty.push(id);
            }
        }
        true
    }

    #[inline]
    fn mark_settlement(&mut self, id: u32) -> bool {
        let i = id as usize;
        if i >= self.production.len() { return false; }
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
        true
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
            WorldDelta::Damage { target_id, amount, source_id } => {
                if self.mark_entity(target_id) {
                    self.damage[target_id as usize] += amount;
                    self.last_damage_source.insert(target_id, source_id);
                }
            }
            WorldDelta::Heal { target_id, amount, .. } => {
                if self.mark_entity(target_id) {
                    self.heals[target_id as usize] += amount;
                }
            }
            WorldDelta::Shield { target_id, amount, .. } => {
                if self.mark_entity(target_id) {
                    self.shields[target_id as usize] += amount;
                }
            }
            WorldDelta::Move { entity_id, force } => {
                if self.mark_entity(entity_id) {
                    self.force_x[entity_id as usize] += force.0;
                    self.force_y[entity_id as usize] += force.1;
                }
            }
            WorldDelta::Die { entity_id } => {
                if self.mark_entity(entity_id) {
                    self.dead[entity_id as usize] = true;
                }
            }
            WorldDelta::ApplyStatus { target_id, status } => {
                self.new_statuses.push((target_id, status));
            }
            WorldDelta::RemoveStatus { target_id, status_discriminant } => {
                self.remove_statuses.push((target_id, status_discriminant));
            }
            WorldDelta::ProduceCommodity { settlement_id, commodity, amount } => {
                if self.mark_settlement(settlement_id) {
                    self.production[settlement_id as usize][commodity] += amount;
                }
            }
            WorldDelta::ConsumeCommodity { settlement_id, commodity, amount } => {
                if self.mark_settlement(settlement_id) {
                    self.consumption[settlement_id as usize][commodity] += amount;
                }
            }
            WorldDelta::TransferGold { from_entity, to_entity, amount } => {
                self.gold_transfers.push((from_entity, to_entity, amount));
            }
            // TransferGoods removed — use TransferCommodity instead.
            WorldDelta::UpdateStockpile { settlement_id, commodity, delta } => {
                if self.mark_settlement(settlement_id) {
                    self.stockpile_adj[settlement_id as usize][commodity] += delta;
                }
            }
            WorldDelta::UpdateTreasury { settlement_id, delta } => {
                if self.mark_settlement(settlement_id) {
                    self.treasury_adj[settlement_id as usize] += delta;
                }
            }
            WorldDelta::UpdatePrices { settlement_id, prices } => {
                self.price_updates.push((settlement_id, prices));
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
                if self.mark_entity(entity_id) {
                    self.entity_fields[entity_id as usize * NUM_ENTITY_FIELDS + field as usize] += value;
                }
            }
            WorldDelta::SetEntityMood { entity_id, mood } => {
                if self.mark_entity(entity_id) {
                    self.entity_mood[entity_id as usize] = mood;
                    self.entity_mood_set[entity_id as usize] = true;
                }
            }
            WorldDelta::AddXp { .. } => {
                // Vestigial: npc.xp is no longer used (level derives from class levels).
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
                if self.mark_settlement(settlement_id) {
                    self.settlement_fields[settlement_id as usize * NUM_SETTLEMENT_FIELDS + field as usize] += value;
                }
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
            WorldDelta::SpawnItem { pos, item_data } => {
                self.item_spawns.push((pos, item_data));
            }
            WorldDelta::EquipItem { npc_id, item_id } => {
                self.equip_requests.push((npc_id, item_id));
            }
            WorldDelta::UnequipItem { npc_id, item_id } => {
                self.unequip_requests.push((npc_id, item_id));
            }
            WorldDelta::TransferCommodity { from_entity, to_entity, commodity, amount } => {
                self.inventory_transfers.push((from_entity, to_entity, commodity, amount));
            }
            WorldDelta::TransferInventoryGold { from_entity, to_entity, amount } => {
                self.inventory_gold_transfers.push((from_entity, to_entity, amount));
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
        if let Some(s) = state.settlement_mut(id) {
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
        if let Some(s) = state.settlement_mut(loc_id) {
            s.prices = *prices;
        }
    }
    p.economy_us = t.elapsed().as_micros() as u64;

    // Deaths
    let t = Instant::now();
    {
        let mut death_records: Vec<(u32, String, Option<u32>, Option<u32>, u32, EntityKind)> = Vec::new();
        // Collect items to drop from dead NPCs: (item_id, death_pos, settlement_id).
        let mut item_drops: Vec<(u32, (f32, f32), Option<u32>)> = Vec::new();
        for entity in &mut state.entities {
            if !entity.alive { continue; }
            let is_dead = (entity.id as usize) < m.dead.len() && m.dead[entity.id as usize];
            if is_dead
                || (entity.hp <= 0.0 && entity.kind != EntityKind::Building && entity.kind != EntityKind::Item)
            {
                let victim_name = super::naming::entity_display_name(entity);
                let home_settlement_id = entity.npc.as_ref().and_then(|n| n.home_settlement_id);
                let killer_id = m.last_damage_source.get(&entity.id).copied();

                // Drop equipped items on death.
                if let Some(npc) = &mut entity.npc {
                    for item_id in [npc.equipped_items.weapon_id,
                                    npc.equipped_items.armor_id,
                                    npc.equipped_items.accessory_id].iter().flatten() {
                        item_drops.push((*item_id, entity.pos, home_settlement_id));
                    }
                    npc.equipped_items = super::state::EquippedItems::default();
                }

                death_records.push((entity.id, victim_name, home_settlement_id, killer_id, entity.level, entity.kind));
                entity.alive = false;
            }
        }

        // Record kills on killer's weapon (item provenance).
        for &(_victim_id, ref victim_name, _, killer_id, victim_level, _) in &death_records {
            if let Some(kid) = killer_id {
                // Find killer's weapon and record the kill.
                let weapon_id = state.entity(kid)
                    .filter(|e| e.alive)
                    .and_then(|e| e.npc.as_ref())
                    .and_then(|n| n.equipped_items.weapon_id);

                if let Some(wid) = weapon_id {
                    let current_tick = state.tick;
                    let mut legend_entry: Option<super::state::ChronicleEntry> = None;

                    if let Some(item_entity) = state.entity_mut(wid) {
                        if let Some(item) = &mut item_entity.item {
                            item.history.push(super::state::ItemEvent {
                                tick: current_tick,
                                kind: super::state::ItemEventKind::Kill {
                                    victim_name: victim_name.clone(),
                                    victim_level,
                                },
                            });
                            if item.history.len() > 20 { item.history.drain(..item.history.len() - 20); }

                            if !item.is_legendary {
                                let kill_count = item.history.iter()
                                    .filter(|e| matches!(e.kind, super::state::ItemEventKind::Kill { .. }))
                                    .count();
                                if kill_count >= 3 {
                                    item.is_legendary = true;
                                    let legendary_name = generate_legendary_name(
                                        item.slot, kill_count, current_tick, wid);
                                    let old_name = item.name.clone();
                                    item.name = legendary_name.clone();
                                    legend_entry = Some(super::state::ChronicleEntry {
                                        tick: current_tick,
                                        category: super::state::ChronicleCategory::Achievement,
                                        text: format!("The {} has earned the name \"{}\" after {} kills.",
                                            old_name, legendary_name, kill_count),
                                        entity_ids: vec![wid, kid],
                                    });
                                }
                            }
                        }
                    }
                    if let Some(entry) = legend_entry {
                        state.chronicle.push(entry);
                    }
                }
            }
        }

        // Process item drops: clear owner, set position to death site.
        for (item_id, death_pos, settlement_id) in item_drops {
            if let Some(item_entity) = state.entity_mut(item_id) {
                if let Some(item) = &mut item_entity.item {
                    item.owner_id = None;
                    item.settlement_id = settlement_id;
                }
                item_entity.pos = death_pos;
            }
        }

        let mut death_chronicle_count = 0u32;
        const MAX_DEATH_CHRONICLES_PER_TICK: u32 = 2;
        for (entity_id, victim_name, home_settlement_id, killer_id, level, kind) in death_records {
            let settlement_name = home_settlement_id
                .and_then(|sid| state.settlement(sid))
                .map(|s| s.name.clone());
            let killer_name = killer_id
                .filter(|&kid| kid != entity_id) // exclude self-damage
                .and_then(|kid| {
                    state.entity(kid).map(|e| super::naming::entity_display_name(e))
                });

            let text = match (&settlement_name, &killer_name) {
                (Some(sname), Some(kname)) => {
                    format!("{} of {} was slain by {}", victim_name, sname, kname)
                }
                (Some(sname), None) => {
                    format!("{} of {} fell in battle", victim_name, sname)
                }
                (None, Some(kname)) => {
                    format!("{} was slain by {}", victim_name, kname)
                }
                (None, None) => {
                    format!("{} fell in battle", victim_name)
                }
            };

            // Only record notable deaths, capped per tick.
            if (level >= 40 || kind == EntityKind::Monster) && death_chronicle_count < MAX_DEATH_CHRONICLES_PER_TICK {
                death_chronicle_count += 1;
                state.chronicle.push(super::state::ChronicleEntry {
                    tick: state.tick,
                    category: super::state::ChronicleCategory::Death,
                    text,
                    entity_ids: {
                        let mut ids = vec![entity_id];
                        if let Some(kid) = killer_id {
                            ids.push(kid);
                        }
                        ids
                    },
                });
            }

            // Legendary death: entities with 20+ chronicle mentions get a special narrative.
            // Legendary death: first time only (check no existing "legendary" narrative for this entity).
            if kind == EntityKind::Npc {
                let already_legendary = state.chronicle.iter().any(|e| {
                    e.category == super::state::ChronicleCategory::Narrative
                        && e.entity_ids.contains(&entity_id)
                        && e.text.contains("legendary")
                });
                if !already_legendary {
                    let mention_count = state.chronicle.iter()
                        .filter(|e| e.entity_ids.contains(&entity_id))
                        .count();
                    if mention_count >= 20 {
                        let settlement_text = settlement_name.as_deref().unwrap_or("the wilderness");
                        state.chronicle.push(super::state::ChronicleEntry {
                            tick: state.tick,
                            category: super::state::ChronicleCategory::Narrative,
                            text: format!("The legendary {} of {} has fallen. {} chronicle entries tell their story.",
                                victim_name, settlement_text, mention_count),
                            entity_ids: vec![entity_id],
                        });
                    }
                }
            }

            // Always record ALL deaths to world_events.
            state.world_events.push(super::state::WorldEvent::EntityDied {
                entity_id,
                cause: "combat".to_string(),
            });
        }
    }

    // Record death memories on nearby NPCs (witnesses).
    // Includes grudge formation when the killer is known.
    let recent_deaths_with_killer: Vec<(u32, Option<u32>, Option<u32>)> = state.world_events.iter()
        .filter_map(|e| match e {
            super::state::WorldEvent::EntityDied { entity_id, .. } => {
                let sid = state.entity(*entity_id).and_then(|e| e.settlement_id());
                let killer = m.last_damage_source.get(entity_id).copied();
                Some((*entity_id, sid, killer))
            }
            _ => None,
        })
        .collect();
    for (dead_id, home_sid, killer_id) in &recent_deaths_with_killer {
        let kind = state.entity(*dead_id).map(|e| e.kind).unwrap_or(EntityKind::Npc);
        if kind != EntityKind::Npc { continue; }
        let dead_pos = state.entity(*dead_id).map(|e| e.pos).unwrap_or((0.0, 0.0));
        let sid = match *home_sid { Some(s) => s, None => continue };
        for entity in &mut state.entities {
            if entity.id == *dead_id || !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };
            if npc.home_settlement_id != Some(sid) { continue; }
            // Rate limit: max 1 FriendDied per NPC per 500 ticks.
            let recent_grief = npc.memory.events.iter().rev().take(5).any(|e| {
                matches!(e.event_type, super::state::MemEventType::FriendDied(_))
                    && state.tick.saturating_sub(e.tick) < 500
            });
            if recent_grief { continue; }
            let dx = entity.pos.0 - dead_pos.0;
            let dy = entity.pos.1 - dead_pos.1;
            if dx * dx + dy * dy > 2500.0 { continue; }
            super::systems::agent_inner::record_npc_event(
                npc,
                super::state::MemEventType::FriendDied(*dead_id),
                dead_pos,
                vec![*dead_id],
                -0.7,
                state.tick,
            );

            // Form grudge against the killer if known.
            if let Some(kid) = killer_id {
                // Don't grudge yourself or dead entities.
                if *kid != entity.id {
                    // Check if we already have a grudge against this entity.
                    let has_grudge = npc.memory.beliefs.iter().any(|b| {
                        matches!(b.belief_type, super::state::BeliefType::Grudge(gid) if gid == *kid)
                    });
                    if !has_grudge {
                        npc.memory.beliefs.push(super::state::Belief {
                            belief_type: super::state::BeliefType::Grudge(*kid),
                            confidence: 1.0,
                            formed_tick: state.tick,
                        });
                        // Grudge anger spike.
                        npc.emotions.anger = (npc.emotions.anger + 0.5).min(1.0);
                        // Accumulate vengeance-related tags.
                        npc.accumulate_tags(&{
                            let mut a = super::state::ActionTags::empty();
                            a.add(super::state::tags::COMBAT, 2.0);
                            a.add(super::state::tags::RESILIENCE, 1.0);
                            a
                        });
                    }
                }
            }
        }
    }

    p.deaths_us = t.elapsed().as_micros() as u64;

    // Grid transitions
    let t = Instant::now();
    for &(entity_id, grid_id) in &m.grid_leaves {
        if let Some(grid) = state.grid_mut(grid_id) {
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
        if let Some(grid) = state.grid_mut(grid_id) {
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
        if let Some(grid) = state.grid_mut(grid_id) {
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
        let has_mood = m.entity_mood_set[i];

        if !has_fields && !has_mood { continue; }

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
            for f in 0..NUM_SETTLEMENT_FIELDS {
                let delta = m.settlement_fields[base + f];
                if delta.abs() > 0.0001 {
                    super::apply::apply_settlement_field_delta(s, f as u8, delta);
                }
            }
        }
    }

    // Relations
    for &(a, b, kind, delta) in &m.relation_deltas {
        let entry = state.relations.entry((a, b, kind)).or_insert(0.0);
        *entry += delta;
    }

    // Spawns
    for &(kind, pos, team, level) in &m.spawns {
        let id = state.next_entity_id();
        let mut entity = match kind {
            EntityKind::Npc => Entity::new_npc(id, pos),
            EntityKind::Monster => Entity::new_monster(id, pos, level),
            EntityKind::Building => Entity::new_building(id, pos),
            EntityKind::Projectile => Entity::new_monster(id, pos, 0),
            EntityKind::Item => Entity::new_building(id, pos), // items spawned via SpawnItem
        };
        entity.team = team;
        entity.level = level;
        state.entities.push(entity);
    }

    // Item spawns
    for (pos, item_data) in &m.item_spawns {
        let id = state.next_entity_id();
        state.entities.push(Entity::new_item(id, *pos, item_data.clone()));
    }

    // Unequip requests (process before equips so slot is freed)
    for &(npc_id, item_id) in &m.unequip_requests {
        // Extract item info immutably first.
        let item_info = state.entity(item_id)
            .and_then(|e| e.item.as_ref())
            .map(|i| (i.slot, i.attack_bonus(), i.armor_bonus(), i.hp_bonus(), i.speed_bonus()));
        if let Some((slot, attack_loss, armor_loss, hp_loss, speed_loss)) = item_info {
            // Remove stat bonuses from NPC.
            if let Some(npc_entity) = state.entity_mut(npc_id) {
                npc_entity.attack_damage = (npc_entity.attack_damage - attack_loss).max(0.0);
                npc_entity.armor = (npc_entity.armor - armor_loss).max(0.0);
                npc_entity.max_hp = (npc_entity.max_hp - hp_loss).max(1.0);
                npc_entity.hp = npc_entity.hp.min(npc_entity.max_hp);
                npc_entity.move_speed = (npc_entity.move_speed - speed_loss).max(0.5);
                if let Some(npc) = &mut npc_entity.npc {
                    npc.equipped_items.set_slot(slot, None);
                }
            }
            // Clear item owner.
            if let Some(item_entity) = state.entity_mut(item_id) {
                if let Some(item) = &mut item_entity.item {
                    item.owner_id = None;
                }
            }
        }
    }

    // Equip requests
    for &(npc_id, item_id) in &m.equip_requests {
        // Get item data.
        let item_info = state.entity(item_id)
            .and_then(|e| e.item.as_ref())
            .map(|i| (i.slot, i.attack_bonus(), i.armor_bonus(), i.hp_bonus(), i.speed_bonus()));
        if let Some((slot, attack, armor_bonus, hp, speed)) = item_info {
            // Apply stat bonuses to NPC.
            if let Some(npc_entity) = state.entity_mut(npc_id) {
                npc_entity.attack_damage += attack;
                npc_entity.armor += armor_bonus;
                npc_entity.max_hp += hp;
                npc_entity.hp += hp;
                npc_entity.move_speed += speed;
                if let Some(npc) = &mut npc_entity.npc {
                    npc.equipped_items.set_slot(slot, Some(item_id));
                }
            }
            if let Some(item_entity) = state.entity_mut(item_id) {
                if let Some(item) = &mut item_entity.item {
                    item.owner_id = Some(npc_id);
                }
            }
        }
    }

    // Inventory commodity transfers
    for &(from_id, to_id, commodity, amount) in &m.inventory_transfers {
        let withdrawn = state.entity_mut(from_id)
            .and_then(|e| e.inventory.as_mut())
            .map(|inv| inv.withdraw(commodity, amount))
            .unwrap_or(0.0);
        if withdrawn > 0.0 {
            if let Some(target) = state.entity_mut(to_id) {
                if let Some(inv) = &mut target.inventory {
                    inv.deposit(commodity, withdrawn);
                }
            }
        }
    }

    for &(from_id, to_id, amount) in &m.inventory_gold_transfers {
        let taken = state.entity_mut(from_id)
            .and_then(|e| e.inventory.as_mut())
            .map(|inv| { let t = amount.min(inv.gold).max(0.0); inv.gold -= t; t })
            .unwrap_or(0.0);
        if taken > 0.0 {
            if let Some(target) = state.entity_mut(to_id) {
                if let Some(inv) = &mut target.inventory {
                    inv.gold += taken;
                }
            }
        }
    }

    // Events
    const MAX_WORLD_EVENTS: usize = 1000;
    const MAX_CHRONICLE_ENTRIES: usize = 2000;

    state.world_events.extend(m.events.iter().cloned());
    if state.world_events.len() > MAX_WORLD_EVENTS {
        let drain = state.world_events.len() - MAX_WORLD_EVENTS;
        state.world_events.drain(..drain);
    }

    // Chronicles
    state.chronicle.extend(m.chronicles.iter().cloned());
    if state.chronicle.len() > MAX_CHRONICLE_ENTRIES {
        let drain = state.chronicle.len() - MAX_CHRONICLE_ENTRIES;
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
        // Sync monotonic ID counter from externally created entities.
        initial.sync_next_id();

        // Ensure every settlement has a Treasury building.
        initial.ensure_treasury_buildings();

        // Sort entities by settlement/party, build hot/cold + all indices.
        initial.rebuild_all_indices();

        // Compute max IDs AFTER treasury buildings are spawned.
        let max_entity_id = initial.entities.iter().map(|e| e.id).max().unwrap_or(0) as usize + 1;
        let max_settlement_id = initial.settlements.iter().map(|s| s.id).max().unwrap_or(0) as usize + 1;

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
            ability_gen: Box::new(super::class_gen::DefaultAbilityGenerator::new()),
            naming: super::naming::NamingService::new(),
        }
    }

    pub fn state_mut(&mut self) -> &mut WorldState {
        &mut self.state
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
    culture::compute_culture_for_settlement(state, sid, entities, buf);
    equipping::compute_equipping_for_settlement(state, sid, entities, buf);
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
        // Ensure merger capacity covers any entities spawned this tick.
        self.merged.ensure_capacity(self.state.next_id as usize + 1);
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

        // GRID ASSIGNMENT — entities near grids get assigned to them.
        // This enables combat: monsters entering a settlement grid trigger
        // fidelity escalation → High → compute_high runs combat.
        if self.state.tick % GRID_MEMBERSHIP_INTERVAL == 0 {
            self.update_grid_membership();
        }

        // CLASS MATCHING — run after apply so behavior tags are up to date.
        if self.state.tick % CLASS_MATCHING_INTERVAL == 0 && self.state.tick > 0 {
            self.run_class_matching();
        }

        let postapply_start = Instant::now();

        super::systems::death_consequences::advance_death_consequences(&mut self.state);
        super::systems::agent_inner::update_agent_inner_states(&mut self.state);
        super::systems::goal_eval::evaluate_goals(&mut self.state);
        super::systems::world_goap::evaluate_world_goap(&mut self.state);
        super::systems::pathfollow::advance_pathfinding(&mut self.state);

        // WORK STATE — advance NPC work state machine.
        super::systems::work::advance_work_states(&mut self.state);

        // EATING — hungry NPCs walk to food and eat.
        super::systems::work::advance_eating(&mut self.state);

        // SOCIAL GATHERING — NPCs meet at taverns/temples for conversations.
        super::systems::social_gathering::advance_social_gatherings(&mut self.state);

        // ADVENTURING — form parties, take quests, explore wilderness.
        super::systems::adventuring::advance_adventuring(&mut self.state);

        // SEA TRAVEL — coastal voyages with sea monster risk.
        super::systems::sea_travel::advance_sea_travel(&mut self.state);

        // ITEM DURABILITY — degrade equipped items, unequip broken items.
        super::systems::equipment_durability::advance_item_durability(&mut self.state);

        // TITLES — NPCs earn honorifics from their deeds.
        super::systems::titles::advance_titles(&mut self.state);

        // HAUNTED — mass death sites become supernaturally dangerous.
        super::systems::haunted::advance_haunted(&mut self.state);

        // WORLD AGES — name historical eras from event density.
        super::systems::world_ages::advance_world_ages(&mut self.state);

        // OATHS — swear, fulfill, and break oaths.
        super::systems::oaths::advance_oaths(&mut self.state);

        // MONSTER NAMES — monsters that kill NPCs gain names and notoriety.
        super::systems::monster_names::advance_monster_naming(&mut self.state);

        // CULTURAL IDENTITY — settlements develop emergent culture.
        super::systems::cultural_identity::advance_cultural_identity(&mut self.state);

        // WARFARE — faction wars, declarations, and peace treaties.
        super::systems::warfare::advance_warfare(&mut self.state);

        // SUCCESSION — power struggle when settlement leaders die.
        super::systems::succession::advance_succession(&mut self.state);

        // LEGENDS — detect and maintain heroic legends.
        super::systems::legends::advance_legends(&mut self.state);

        // PROPHECY — check and fulfill world prophecies.
        super::systems::prophecy::advance_prophecies(&mut self.state);

        // OUTLAWS — bandit camps, caravan raids, redemption.
        super::systems::outlaws::advance_outlaws(&mut self.state);

        // TRADE GUILDS — merchants form guilds, set prices, fund caravans.
        super::systems::trade_guilds::advance_trade_guilds(&mut self.state);

        // SETTLEMENT FOUNDING — overcrowded settlements send colonists.
        super::systems::settlement_founding::advance_settlement_founding(&mut self.state);

        // BETRAYAL — treacherous NPCs steal gold and become outlaws.
        super::systems::betrayal::advance_betrayal(&mut self.state);

        // FAMILY — marriages and births.
        super::systems::family::advance_family(&mut self.state);

        // STOCKPILE SYNC — rebuild settlement stockpiles from building inventories.
        super::systems::work::sync_stockpiles_from_buildings(&mut self.state);

        // INVENTORY SYNC — bridge legacy carried_goods/building.storage ↔ entity.inventory.
        for entity in &mut self.state.entities {
            // Cap NPC gold to prevent infinity.
            if let Some(npc) = &mut entity.npc {
                if !npc.gold.is_finite() || npc.gold > 10000.0 {
                    npc.gold = 10000.0;
                }
            }
            entity.sync_inventory_from_carried_goods();
            entity.sync_inventory_from_building_storage();
        }

        // BUILDING INTERIORS — NPCs enter buildings and occupy rooms.
        super::systems::interiors::advance_interiors(&mut self.state);

        // ACTION SYNC — derive visible NPC action from current state.
        super::systems::action_sync::sync_npc_actions(&mut self.state);

        profile.postapply_us = postapply_start.elapsed().as_micros() as u64;

        // CITY GROWTH — CA-driven building placement on city grids.
        self.grow_cities();

        // ENTITY COMPACTION — remove long-dead items/buildings every 500 ticks.
        if self.state.tick % ENTITY_COMPACTION_INTERVAL == 0 && self.state.tick > 0 {
            self.state.compact_dead_entities();
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
    /// Runs after apply so behavior_profile is up to date.
    /// Assign entities to grids based on proximity.
    fn update_grid_membership(&mut self) {
        let num_grids = self.state.grids.len();

        // Phase 1: compute new membership for each grid.
        // Collect (grid_index, new_entity_ids) without borrowing conflicts.
        let mut new_memberships: Vec<Vec<u32>> = Vec::with_capacity(num_grids);

        for grid in &self.state.grids {
            let r2 = grid.radius * grid.radius;
            let mut members = Vec::new();

            for entity in &self.state.entities {
                if !entity.alive { continue; }
                // Items and buildings don't enter combat grids.
                if entity.kind == super::state::EntityKind::Item
                    || entity.kind == super::state::EntityKind::Building { continue; }
                let dx = entity.pos.0 - grid.center.0;
                let dy = entity.pos.1 - grid.center.1;
                if dx * dx + dy * dy > r2 { continue; }

                // Workers/traders stay inside the settlement — not on the battlefield.
                // Only idle/adventuring NPCs and monsters enter combat grids.
                if entity.kind == super::state::EntityKind::Npc {
                    if let Some(npc) = &entity.npc {
                        let is_combatant = matches!(npc.economic_intent,
                            super::state::EconomicIntent::Idle
                            | super::state::EconomicIntent::Adventuring { .. }
                        );
                        if !is_combatant { continue; }
                    }
                }

                members.push(entity.id);
            }
            new_memberships.push(members);
        }

        // Phase 2: apply.
        for (i, grid) in self.state.grids.iter_mut().enumerate() {
            grid.entity_ids = std::mem::take(&mut new_memberships[i]);
        }

        // Phase 3: update entity grid_id.
        // Build a quick lookup: entity_id → grid_id.
        let mut entity_grid: Vec<Option<u32>> = vec![None; self.state.max_entity_id as usize + 1];
        for grid in &self.state.grids {
            for &eid in &grid.entity_ids {
                if (eid as usize) < entity_grid.len() {
                    entity_grid[eid as usize] = Some(grid.id);
                }
            }
        }
        for entity in &mut self.state.entities {
            entity.grid_id = if (entity.id as usize) < entity_grid.len() {
                entity_grid[entity.id as usize]
            } else {
                None
            };
        }
    }

    /// Convert world events into state changes (quest postings, etc.).
    fn process_world_events(&mut self) {
        use super::state::{WorldEvent, QuestPosting, QuestType};

        let mut next_quest_id = self.state.quest_board.iter()
            .map(|q| q.id)
            .max()
            .unwrap_or(0) + 1;
        let mut conquests: Vec<(u32, u32)> = Vec::new();

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
                WorldEvent::SettlementConquered { settlement_id, new_faction_id } => {
                    conquests.push((*settlement_id, *new_faction_id));
                }
                _ => {}
            }
        }

        // Apply settlement conquests (deferred to avoid borrow conflict).
        for (sid, fid) in conquests {
            if let Some(s) = self.state.settlement_mut(sid) {
                s.faction_id = Some(fid);
            }
        }

        // Expire old quest postings.
        self.state.quest_board.retain(|q| q.expires_tick > self.state.tick);
    }

    fn run_class_matching(&mut self) {
        use super::state::{EntityKind, ClassSlot};

        let min_behavior_sum = 10.0_f32;
        let mut abilities_generated = 0u32;
        let mut hero_milestones: Vec<(u32, String, String, String)> = Vec::new();

        for entity in &mut self.state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };

            // Skip NPCs with insufficient behavior.
            let behavior_sum: f32 = npc.behavior_profile.iter().map(|&(_, v)| v).sum();
            if behavior_sum < min_behavior_sum { continue; }

            // Soft cap: XP gain scales down as total level approaches 100.
            // At 100 total, gain is 10% of normal. At 200, gain is ~1%.
            const SOFT_CAP_LEVEL: f32 = 100.0;

            // Match against class templates.
            let matches = self.class_gen.match_classes(&npc.behavior_profile);

            let entity_seed = super::state::entity_hash(entity.id, self.state.tick, 0xC1A5) as u64;

            let total_level: u16 = npc.classes.iter().map(|c| c.level).sum();

            // Past 80 total: only the highest class can gain levels.
            // Past 100 total: hard cap, no more levels.
            let global_factor = 1.0 + (total_level as f32 / SOFT_CAP_LEVEL).powi(2);

            for class_match in &matches {
                // No new classes past 80 total level.
                if total_level >= 80 { break; }

                // Score must exceed threshold × global factor to gain a new class.
                if class_match.score < 0.3 * global_factor { continue; }

                // Skip if NPC already has this class.
                if npc.classes.iter().any(|c| c.class_name_hash == class_match.class_name_hash) {
                    continue;
                }

                // Generate procedural name from behavior profile.
                let display_name = super::naming::procedural_class_name(
                    &class_match.display_name,
                    &npc.behavior_profile,
                    entity_seed ^ class_match.class_name_hash as u64,
                );

                // Grant the class.
                let class_display = display_name.clone();
                npc.classes.push(ClassSlot {
                    class_name_hash: class_match.class_name_hash,
                    level: 1,
                    xp: 0.0,
                    display_name,
                });

                // Hero milestone: 5th class marks a rising hero (deferred to avoid borrow).
                if npc.classes.len() == 5 {
                    let hero_name = if !npc.name.is_empty() { npc.name.clone() }
                        else { format!("Entity #{}", entity.id) };
                    hero_milestones.push((entity.id, hero_name, npc.home_settlement_id.unwrap_or(0).to_string(), class_display));
                }
            }

            // If no templates matched and behavior is high, try unique class.
            if matches.is_empty() && behavior_sum > 500.0 && npc.classes.is_empty() {
                let seed = self.state.tick ^ entity.id as u64;
                if let Some(class_def) = self.class_gen.generate_unique_class(
                    &npc.behavior_profile, seed,
                ) {
                    let display_name = super::naming::procedural_class_name(
                        &class_def.display_name,
                        &npc.behavior_profile,
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

            // --- Per-class XP allocation + level-up ---
            // Each class gains XP proportional to behavior alignment with its template.
            // The ClassGenerator.match_classes() returns scores — use those.
            let class_matches = self.class_gen.match_classes(&npc.behavior_profile);

            let total_level: u16 = npc.classes.iter().map(|c| c.level).sum();

            // Hard cap: no levels past 100 total.
            if total_level >= 100 { continue; }

            // At 80+: consolidate all classes into the single strongest one.
            if total_level >= 80 && npc.classes.len() > 1 {
                npc.classes.sort_by(|a, b| b.level.cmp(&a.level));
                let absorbed_levels: u16 = npc.classes[1..].iter().map(|c| c.level).sum();
                let absorbed_xp: f32 = npc.classes[1..].iter().map(|c| c.xp).sum();
                npc.classes.truncate(1);
                npc.classes[0].level += absorbed_levels;
                npc.classes[0].xp += absorbed_xp;
                // Re-check: don't exceed 100.
                if npc.classes[0].level > 100 { npc.classes[0].level = 100; }
            }

            for class_slot in &mut npc.classes {
                if class_slot.level >= 100 { continue; }

                // Find score for this class from template matching.
                let score = class_matches.iter()
                    .find(|m| m.class_name_hash == class_slot.class_name_hash)
                    .map(|m| m.score)
                    .unwrap_or(0.1);

                let xp_gain = score * 0.5;
                class_slot.xp += xp_gain;

                // Level-up XP floor: per-class exponential + global progression term.
                // base = 50 × e^(class_level × 0.1)
                // global = 1 + (total_level / 100)^2
                // Effect: leveling slows as total approaches 100.
                let base_xp = 50.0 * (class_slot.level as f32 * 0.1).exp();
                let global_factor = 1.0 + (total_level as f32 / SOFT_CAP_LEVEL).powi(2);
                let xp_to_next = base_xp * global_factor;
                if class_slot.xp >= xp_to_next {
                    class_slot.xp -= xp_to_next;
                    class_slot.level += 1;

                    // Generate ability on level-up at tier thresholds.
                    let tier = match class_slot.level {
                        2..=4 => Some(1),
                        5..=7 => Some(2),
                        10..=12 => Some(3),
                        20..=22 => Some(4),
                        35..=37 => Some(5),
                        50..=52 => Some(6),
                        70..=72 => Some(7),
                        _ => None,
                    };
                    if let Some(tier) = tier {
                        // Rate-limit ability generation to avoid tick spikes.
                        // Max 10 abilities generated per class-assignment pass.
                        if abilities_generated < 10 {
                            let ability_seed = super::state::entity_hash(
                                entity.id, self.state.tick,
                                class_slot.class_name_hash as u64,
                            ) as u64;
                            let generated = self.ability_gen.generate_ability(
                                class_slot.class_name_hash,
                                &npc.archetype,
                                tier,
                                &npc.behavior_profile,
                                ability_seed,
                            );
                            npc.class_tags.push(generated.dsl_text);
                            abilities_generated += 1;
                        }
                    }
                }
            }
        }

        // --- Class pruning: consolidate fragmented classes ---
        // If an NPC has more than 5 classes, drop the weakest (level < 5)
        // and redistribute XP to the strongest class.
        for entity in &mut self.state.entities {
            if !entity.alive || entity.kind != EntityKind::Npc { continue; }
            let npc = match &mut entity.npc { Some(n) => n, None => continue };
            if npc.classes.len() <= 5 { continue; }

            // Sort by level descending, keep top 5 + anything >= L5.
            npc.classes.sort_by(|a, b| b.level.cmp(&a.level));
            let mut pruned_xp: f32 = 0.0;
            let mut keep = Vec::new();
            for c in npc.classes.drain(..) {
                if keep.len() < 5 || c.level >= 5 {
                    keep.push(c);
                } else {
                    pruned_xp += c.xp + c.level as f32 * 10.0;
                }
            }
            npc.classes = keep;

            // Redistribute pruned XP to the highest-level class.
            if pruned_xp > 0.0 {
                if let Some(best) = npc.classes.first_mut() {
                    best.xp += pruned_xp;
                }
            }
        }

        // Apply deferred hero milestones to chronicle.
        for (eid, hero_name, home_sid_str, class_name) in hero_milestones {
            let home = home_sid_str.parse::<u32>().ok()
                .and_then(|sid| self.state.settlement(sid))
                .map(|s| s.name.clone())
                .unwrap_or_else(|| "unknown".into());
            self.state.chronicle.push(super::state::ChronicleEntry {
                tick: self.state.tick,
                category: super::state::ChronicleCategory::Achievement,
                text: format!("{} of {} has become a hero of five disciplines, earning the {} class.",
                    hero_name, home, class_name),
                entity_ids: vec![eid],
            });
        }
    }

    /// CA-driven city growth on settlement grids.
    /// Called post-apply so it can mutate city grids directly.
    fn grow_cities(&mut self) {
        let old_count = self.state.entities.len();
        super::systems::buildings::grow_cities(&mut self.state);

        // If building entities were added, resize flat merge arrays.
        if self.state.entities.len() > old_count {
            let new_max = self.state.entities.iter().map(|e| e.id).max().unwrap_or(0) as usize + 1;
            self.merged.ensure_capacity(new_max);
        }
    }
}

/// Generate a legendary name for an item with enough kills.
fn generate_legendary_name(slot: super::state::ItemSlot, kills: usize, tick: u64, item_id: u32) -> String {
    use super::state::ItemSlot;
    let h = super::state::entity_hash(item_id, tick, 0x1E6D);

    let prefix = match kills {
        3..=5 => ["Blood", "Shadow", "Iron", "Storm"][h as usize % 4],
        6..=9 => ["Dread", "Doom", "Death", "Wrath"][h as usize % 4],
        _ => ["Soul", "World", "God", "Eternal"][h as usize % 4],
    };

    let suffix = match slot {
        ItemSlot::Weapon => ["bringer", "cleaver", "fang", "edge", "reaper"][h as usize / 4 % 5],
        ItemSlot::Armor => ["ward", "aegis", "bastion", "shell", "mantle"][h as usize / 4 % 5],
        ItemSlot::Accessory => ["step", "sight", "whisper", "charm", "veil"][h as usize / 4 % 5],
    };

    format!("{}{}", prefix, suffix)
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
