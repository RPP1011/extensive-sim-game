use std::collections::HashMap;
use std::time::Instant;

use super::delta::MergedDeltas;
use super::state::{
    Entity, EntityField, EntityKind, QuestDelta, QuestStatus, WorldState,
};
use super::NUM_COMMODITIES;

/// Sub-phase timing for the apply phase.
#[derive(Debug, Clone, Default)]
pub struct ApplyProfile {
    pub clone_us: u64,
    pub hp_us: u64,
    pub movement_us: u64,
    pub status_us: u64,
    pub economy_us: u64,
    pub transfers_us: u64,
    pub deaths_us: u64,
    pub grid_us: u64,
    pub fidelity_us: u64,
    pub price_reports_us: u64,
    pub campaign_us: u64,
}

/// Apply merged deltas to produce the next world state.
pub fn apply_deltas(old: &WorldState, merged: &MergedDeltas) -> WorldState {
    apply_deltas_profiled(old, merged).0
}

/// Apply merged deltas with sub-phase profiling.
pub fn apply_deltas_profiled(old: &WorldState, merged: &MergedDeltas) -> (WorldState, ApplyProfile) {
    let mut p = ApplyProfile::default();

    let t = Instant::now();
    let mut next = old.clone();
    p.clone_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_hp_changes(&mut next, merged);
    p.hp_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_movement(&mut next, merged);
    p.movement_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_status_effects(&mut next, merged);
    p.status_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_economy(&mut next, merged);
    apply_gold_transfers(&mut next, merged);
    apply_goods_transfers(&mut next, merged);
    p.economy_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_deaths(&mut next, merged);
    p.deaths_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_grid_transitions(&mut next, merged);
    p.grid_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_fidelity(&mut next, merged);
    p.fidelity_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_price_reports(&mut next, merged);
    p.price_reports_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_campaign_deltas(&mut next, merged);
    p.campaign_us = t.elapsed().as_micros() as u64;

    next.tick += 1;
    (next, p)
}

/// Apply merged deltas in-place (no clone — caller owns the state).
/// Used by the zero-alloc runtime where state was already clone_from'd.
pub fn apply_deltas_in_place(state: &mut WorldState, merged: &MergedDeltas) -> ApplyProfile {
    let mut p = ApplyProfile::default();

    let t = Instant::now();
    apply_hp_changes(state, merged);
    p.hp_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_movement(state, merged);
    p.movement_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_status_effects(state, merged);
    p.status_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_economy(state, merged);
    apply_gold_transfers(state, merged);
    apply_goods_transfers(state, merged);
    p.economy_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_deaths(state, merged);
    p.deaths_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_grid_transitions(state, merged);
    p.grid_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_fidelity(state, merged);
    p.fidelity_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_price_reports(state, merged);
    p.price_reports_us = t.elapsed().as_micros() as u64;

    let t = Instant::now();
    apply_campaign_deltas(state, merged);
    p.campaign_us = t.elapsed().as_micros() as u64;

    p
}

// ---------------------------------------------------------------------------
// HP changes: simultaneous damage/heal/shield resolution
// ---------------------------------------------------------------------------

fn apply_hp_changes(state: &mut WorldState, merged: &MergedDeltas) {
    for entity in &mut state.entities {
        if !entity.alive { continue; }

        let damage = merged.damage_by_target.get(&entity.id).copied().unwrap_or(0.0);
        let heal = merged.heals_by_target.get(&entity.id).copied().unwrap_or(0.0);
        let shield_add = merged.shields_by_target.get(&entity.id).copied().unwrap_or(0.0);

        if damage == 0.0 && heal == 0.0 && shield_add == 0.0 { continue; }

        // Shield absorbs damage first.
        let shield_absorb = damage.min(entity.shield_hp);
        entity.shield_hp = entity.shield_hp - shield_absorb + shield_add;
        let remaining_damage = damage - shield_absorb;

        // Net HP change: heal - remaining damage.
        entity.hp = (entity.hp + heal - remaining_damage).clamp(0.0, entity.max_hp);
    }
}

// ---------------------------------------------------------------------------
// Movement: force vectors → position update
// ---------------------------------------------------------------------------

/// Tick duration in seconds (100ms fixed tick).
const DT_SEC: f32 = 0.1;

fn apply_movement(state: &mut WorldState, merged: &MergedDeltas) {
    for entity in &mut state.entities {
        if !entity.alive { continue; }

        if let Some(&(fx, fy)) = merged.forces_by_entity.get(&entity.id) {
            let mag = (fx * fx + fy * fy).sqrt();
            let max_speed = entity.move_speed * DT_SEC;
            let (dx, dy) = if mag > max_speed && mag > 0.001 {
                (fx / mag * max_speed, fy / mag * max_speed)
            } else {
                (fx, fy)
            };
            entity.pos.0 += dx;
            entity.pos.1 += dy;
        }
    }
}

// ---------------------------------------------------------------------------
// Status effects: add new, remove expired
// ---------------------------------------------------------------------------

fn apply_status_effects(state: &mut WorldState, merged: &MergedDeltas) {
    // Remove requested statuses.
    for &(target_id, disc) in &merged.remove_statuses {
        if let Some(e) = state.entity_mut(target_id) {
            e.status_effects.retain(|s| s.kind.discriminant() != disc);
        }
    }

    // Add new statuses (dedup: keep longest duration per discriminant per target).
    for (target_id, status) in &merged.new_statuses {
        if let Some(e) = state.entity_mut(*target_id) {
            let disc = status.kind.discriminant();
            if let Some(existing) = e.status_effects.iter_mut()
                .find(|s| s.kind.discriminant() == disc)
            {
                // Keep the one with longer remaining duration.
                if status.remaining_ms > existing.remaining_ms {
                    *existing = status.clone();
                }
            } else {
                e.status_effects.push(status.clone());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Economy: production, consumption with fair rationing
// ---------------------------------------------------------------------------

fn apply_economy(state: &mut WorldState, merged: &MergedDeltas) {
    // Direct stockpile deltas (additive).
    for (&loc_id, deltas) in &merged.stockpile_deltas {
        if let Some(s) = state.settlements.iter_mut().find(|s| s.id == loc_id) {
            for i in 0..NUM_COMMODITIES {
                s.stockpile[i] += deltas[i];
            }
        }
    }

    // Treasury deltas (additive).
    for (&loc_id, &delta) in &merged.treasury_deltas {
        if let Some(s) = state.settlements.iter_mut().find(|s| s.id == loc_id) {
            s.treasury += delta;
        }
    }

    // Production: add to stockpile.
    for (&loc_id, produced) in &merged.production_by_settlement {
        if let Some(s) = state.settlements.iter_mut().find(|s| s.id == loc_id) {
            for i in 0..NUM_COMMODITIES {
                s.stockpile[i] += produced[i];
            }
        }
    }

    // Consumption with fair rationing.
    for (&loc_id, consumed) in &merged.consumption_by_settlement {
        if let Some(s) = state.settlements.iter_mut().find(|s| s.id == loc_id) {
            for i in 0..NUM_COMMODITIES {
                if consumed[i] <= 0.0 { continue; }
                if consumed[i] > s.stockpile[i] {
                    // Demand exceeds supply — stockpile is fully consumed.
                    // Individual consumers would each get (stockpile / consumed) fraction.
                    s.stockpile[i] = 0.0;
                } else {
                    s.stockpile[i] -= consumed[i];
                }
            }
        }
    }

    // Price updates.
    for (&loc_id, prices) in &merged.price_updates {
        if let Some(s) = state.settlements.iter_mut().find(|s| s.id == loc_id) {
            s.prices = *prices;
        }
    }
}

// ---------------------------------------------------------------------------
// Gold transfers with proportional scaling for insufficient funds
// ---------------------------------------------------------------------------

fn apply_gold_transfers(state: &mut WorldState, merged: &MergedDeltas) {
    if merged.gold_transfers.is_empty() { return; }

    // Compute total outgoing per sender.
    let mut outgoing: HashMap<u32, f32> = HashMap::new();
    for &(from, _, amount) in &merged.gold_transfers {
        *outgoing.entry(from).or_default() += amount;
    }

    // Apply each transfer with proportional scaling.
    for &(from, to, amount) in &merged.gold_transfers {
        let sender_gold = state.entity(from)
            .and_then(|e| e.npc.as_ref())
            .map(|n| n.gold)
            .unwrap_or(0.0);
        let total_out = outgoing.get(&from).copied().unwrap_or(0.0);
        let scale = if total_out > sender_gold && total_out > 0.0 {
            sender_gold / total_out
        } else {
            1.0
        };
        let actual = amount * scale;

        if let Some(sender) = state.entity_mut(from) {
            if let Some(npc) = sender.npc.as_mut() {
                npc.gold -= actual;
            }
        }
        if let Some(receiver) = state.entity_mut(to) {
            if let Some(npc) = receiver.npc.as_mut() {
                npc.gold += actual;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Goods transfers with proportional scaling
// ---------------------------------------------------------------------------

fn apply_goods_transfers(state: &mut WorldState, merged: &MergedDeltas) {
    if merged.goods_transfers.is_empty() { return; }

    // Compute total outgoing per (sender, commodity).
    let mut outgoing: HashMap<(u32, usize), f32> = HashMap::new();
    for &(from, _, commodity, amount) in &merged.goods_transfers {
        *outgoing.entry((from, commodity)).or_default() += amount;
    }

    for &(from, to, commodity, amount) in &merged.goods_transfers {
        let sender_stock = state.entity(from)
            .and_then(|e| e.npc.as_ref())
            .map(|n| n.carried_goods[commodity])
            .unwrap_or(0.0);
        let total_out = outgoing.get(&(from, commodity)).copied().unwrap_or(0.0);
        let scale = if total_out > sender_stock && total_out > 0.0 {
            sender_stock / total_out
        } else {
            1.0
        };
        let actual = amount * scale;

        if let Some(sender) = state.entity_mut(from) {
            if let Some(npc) = sender.npc.as_mut() {
                npc.carried_goods[commodity] -= actual;
            }
        }
        if let Some(receiver) = state.entity_mut(to) {
            if let Some(npc) = receiver.npc.as_mut() {
                npc.carried_goods[commodity] += actual;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deaths: mark entities with hp <= 0 as dead
// ---------------------------------------------------------------------------

fn apply_deaths(state: &mut WorldState, merged: &MergedDeltas) {
    for entity in &mut state.entities {
        if !entity.alive { continue; }
        // Explicit Die deltas or hp depleted.
        if merged.deaths.contains(&entity.id)
            || (entity.hp <= 0.0 && entity.kind != EntityKind::Building)
        {
            entity.alive = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Grid transitions
// ---------------------------------------------------------------------------

fn apply_grid_transitions(state: &mut WorldState, merged: &MergedDeltas) {
    // Process leaves first.
    for &(entity_id, grid_id) in &merged.grid_leaves {
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

    // Then enters.
    for &(entity_id, grid_id) in &merged.grid_enters {
        if let Some(grid) = state.grids.iter_mut().find(|g| g.id == grid_id) {
            if !grid.entity_ids.contains(&entity_id) {
                grid.entity_ids.push(entity_id);
            }
        }
        if let Some(e) = state.entity_mut(entity_id) {
            e.grid_id = Some(grid_id);
        }
    }
}

// ---------------------------------------------------------------------------
// Fidelity changes
// ---------------------------------------------------------------------------

fn apply_fidelity(state: &mut WorldState, merged: &MergedDeltas) {
    for (&grid_id, &new_fidelity) in &merged.fidelity_changes {
        if let Some(grid) = state.grids.iter_mut().find(|g| g.id == grid_id) {
            grid.fidelity = new_fidelity;
        }
    }
}

// ---------------------------------------------------------------------------
// Price report sharing
// ---------------------------------------------------------------------------

fn apply_price_reports(state: &mut WorldState, merged: &MergedDeltas) {
    for (_, to_id, report) in &merged.price_reports {
        if let Some(e) = state.entity_mut(*to_id) {
            if let Some(npc) = e.npc.as_mut() {
                // Replace existing report for this settlement, or add new.
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
}

// ---------------------------------------------------------------------------
// Campaign system deltas
// ---------------------------------------------------------------------------

pub(super) fn apply_campaign_deltas(state: &mut WorldState, merged: &MergedDeltas) {
    // --- Entity field deltas (additive) ---
    for (&(entity_id, field_disc), &delta) in &merged.entity_field_deltas {
        if let Some(entity) = state.entity_mut(entity_id) {
            apply_entity_field_delta(entity, field_disc, delta);
        }
    }

    // --- Entity mood sets (last-write-wins) ---
    for (&entity_id, &mood) in &merged.entity_mood_sets {
        if let Some(entity) = state.entity_mut(entity_id) {
            if let Some(npc) = entity.npc.as_mut() {
                npc.mood = mood;
            }
        }
    }

    // --- XP additions ---
    for (&entity_id, &xp) in &merged.xp_additions {
        if let Some(entity) = state.entity_mut(entity_id) {
            if let Some(npc) = entity.npc.as_mut() {
                npc.xp = npc.xp.saturating_add(xp);
            }
        }
    }

    // --- Faction field deltas ---
    for (&(faction_id, field_disc), &delta) in &merged.faction_field_deltas {
        if let Some(faction) = state.faction_mut(faction_id) {
            apply_faction_field_delta(faction, field_disc, delta);
        }
    }

    // --- Region field deltas ---
    for (&(region_id, field_disc), &delta) in &merged.region_field_deltas {
        if let Some(region) = state.region_mut(region_id) {
            apply_region_field_delta(region, field_disc, delta);
        }
    }

    // --- Settlement field deltas ---
    for (&(settlement_id, field_disc), &delta) in &merged.settlement_field_deltas {
        if let Some(settlement) = state.settlement_mut(settlement_id) {
            apply_settlement_field_delta(settlement, field_disc, delta);
        }
    }

    // --- Relation deltas ---
    for (&(a, b, _kind_disc), &delta) in &merged.relation_deltas {
        let key = (a, b, _kind_disc);
        let entry = state.relations.entry(key).or_insert(0.0);
        *entry += delta;
    }

    // --- Spawns ---
    for &(kind, pos, team, level) in &merged.spawns {
        let id = state.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
        let entity = match kind {
            EntityKind::Npc => {
                let mut e = Entity::new_npc(id, pos);
                e.team = team;
                e.level = level;
                e
            }
            EntityKind::Monster => {
                let mut e = Entity::new_monster(id, pos, level);
                e.team = team;
                e
            }
            EntityKind::Building => {
                let mut e = Entity::new_building(id, pos);
                e.team = team;
                e
            }
            EntityKind::Projectile => {
                // Projectiles reuse monster template with minimal HP.
                let mut e = Entity::new_monster(id, pos, 0);
                e.kind = EntityKind::Projectile;
                e.team = team;
                e
            }
        };
        state.entities.push(entity);
    }

    // --- World events ---
    state.world_events.extend(merged.recorded_events.iter().cloned());
    // Bound world events to last 1000 entries.
    if state.world_events.len() > 1000 {
        let drain = state.world_events.len() - 1000;
        state.world_events.drain(..drain);
    }

    // --- Chronicle entries ---
    state.chronicle.extend(merged.recorded_chronicles.iter().cloned());
    // Bound chronicle to last 500 entries.
    if state.chronicle.len() > 500 {
        let drain = state.chronicle.len() - 500;
        state.chronicle.drain(..drain);
    }

    // --- Quest updates ---
    for (quest_id, update) in &merged.quest_updates {
        if let Some(quest) = state.quest_mut(*quest_id) {
            match update {
                QuestDelta::AdvanceProgress { amount } => {
                    quest.progress = (quest.progress + amount).min(1.0);
                }
                QuestDelta::SetStatus { status } => {
                    quest.status = *status;
                }
                QuestDelta::AddMember { entity_id } => {
                    if !quest.party_member_ids.contains(entity_id) {
                        quest.party_member_ids.push(*entity_id);
                    }
                }
                QuestDelta::RemoveMember { entity_id } => {
                    quest.party_member_ids.retain(|id| id != entity_id);
                }
                QuestDelta::Complete => {
                    quest.status = QuestStatus::Completed;
                    quest.progress = 1.0;
                }
                QuestDelta::Fail => {
                    quest.status = QuestStatus::Failed;
                }
            }
        }
    }

    // --- Guild updates ---
    if merged.guild_gold_delta != 0.0 {
        state.guild.gold += merged.guild_gold_delta;
    }
    if merged.guild_supplies_delta != 0.0 {
        state.guild.supplies += merged.guild_supplies_delta;
    }
    if merged.guild_reputation_delta != 0.0 {
        state.guild.reputation = (state.guild.reputation + merged.guild_reputation_delta).clamp(0.0, 100.0);
    }
}

/// Apply an additive delta to the correct field on an entity.
fn apply_entity_field_delta(entity: &mut Entity, field_disc: u8, delta: f32) {
    // Fields that live on Entity directly.
    match field_disc {
        d if d == EntityField::Hp as u8 => {
            entity.hp = (entity.hp + delta).clamp(0.0, entity.max_hp);
        }
        d if d == EntityField::MaxHp as u8 => {
            entity.max_hp = (entity.max_hp + delta).max(1.0);
        }
        d if d == EntityField::ShieldHp as u8 => {
            entity.shield_hp = (entity.shield_hp + delta).max(0.0);
        }
        d if d == EntityField::Armor as u8 => {
            entity.armor += delta;
        }
        d if d == EntityField::MagicResist as u8 => {
            entity.magic_resist += delta;
        }
        d if d == EntityField::AttackDamage as u8 => {
            entity.attack_damage = (entity.attack_damage + delta).max(0.0);
        }
        d if d == EntityField::AttackRange as u8 => {
            entity.attack_range = (entity.attack_range + delta).max(0.0);
        }
        d if d == EntityField::MoveSpeed as u8 => {
            entity.move_speed = (entity.move_speed + delta).max(0.0);
        }
        _ => {
            // NPC-specific fields.
            if let Some(npc) = entity.npc.as_mut() {
                match field_disc {
                    d if d == EntityField::Morale as u8 => {
                        npc.morale = (npc.morale + delta).clamp(0.0, 100.0);
                    }
                    d if d == EntityField::Stress as u8 => {
                        npc.stress = (npc.stress + delta).clamp(0.0, 100.0);
                    }
                    d if d == EntityField::Fatigue as u8 => {
                        npc.fatigue = (npc.fatigue + delta).clamp(0.0, 100.0);
                    }
                    d if d == EntityField::Loyalty as u8 => {
                        npc.loyalty = (npc.loyalty + delta).clamp(0.0, 100.0);
                    }
                    d if d == EntityField::Injury as u8 => {
                        npc.injury = (npc.injury + delta).clamp(0.0, 100.0);
                    }
                    d if d == EntityField::Resolve as u8 => {
                        npc.resolve = (npc.resolve + delta).clamp(0.0, 100.0);
                    }
                    d if d == EntityField::GuildRelationship as u8 => {
                        npc.guild_relationship = (npc.guild_relationship + delta).clamp(-100.0, 100.0);
                    }
                    d if d == EntityField::Gold as u8 => {
                        npc.gold += delta;
                    }
                    _ => {} // Unknown field — ignore silently.
                }
            }
        }
    }
}

/// Apply an additive delta to the correct field on a faction.
fn apply_faction_field_delta(faction: &mut super::state::FactionState, field_disc: u8, delta: f32) {
    use super::state::FactionField;
    match field_disc {
        d if d == FactionField::RelationshipToGuild as u8 => {
            faction.relationship_to_guild = (faction.relationship_to_guild + delta).clamp(-100.0, 100.0);
        }
        d if d == FactionField::MilitaryStrength as u8 => {
            faction.military_strength = (faction.military_strength + delta).max(0.0);
        }
        d if d == FactionField::Treasury as u8 => {
            faction.treasury += delta;
        }
        d if d == FactionField::CoupRisk as u8 => {
            faction.coup_risk = (faction.coup_risk + delta).clamp(0.0, 1.0);
        }
        d if d == FactionField::EscalationLevel as u8 => {
            faction.escalation_level = (faction.escalation_level as f32 + delta).clamp(0.0, 5.0) as u32;
        }
        d if d == FactionField::TechLevel as u8 => {
            faction.tech_level = (faction.tech_level as f32 + delta).max(0.0) as u32;
        }
        _ => {}
    }
}

/// Apply an additive delta to the correct field on a region.
fn apply_region_field_delta(region: &mut super::state::RegionState, field_disc: u8, delta: f32) {
    use super::state::RegionField;
    match field_disc {
        d if d == RegionField::MonsterDensity as u8 => {
            region.monster_density = (region.monster_density + delta).max(0.0);
        }
        d if d == RegionField::ThreatLevel as u8 => {
            region.threat_level = (region.threat_level + delta).max(0.0);
        }
        d if d == RegionField::Unrest as u8 => {
            region.unrest = (region.unrest + delta).clamp(0.0, 1.0);
        }
        d if d == RegionField::Control as u8 => {
            region.control = (region.control + delta).clamp(0.0, 1.0);
        }
        _ => {}
    }
}

/// Apply an additive delta to the correct field on a settlement.
fn apply_settlement_field_delta(settlement: &mut super::state::SettlementState, field_disc: u8, delta: f32) {
    use super::state::SettlementField;
    match field_disc {
        d if d == SettlementField::Treasury as u8 => {
            settlement.treasury += delta;
        }
        d if d == SettlementField::Population as u8 => {
            settlement.population = (settlement.population as f32 + delta).max(0.0) as u32;
        }
        d if d == SettlementField::ThreatLevel as u8 => {
            settlement.threat_level = (settlement.threat_level + delta).clamp(0.0, 1.0);
        }
        d if d == SettlementField::InfrastructureLevel as u8 => {
            settlement.infrastructure_level = (settlement.infrastructure_level + delta).clamp(0.0, 5.0);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_sim::delta::{WorldDelta, merge_deltas};
    use crate::world_sim::state::*;

    fn test_state() -> WorldState {
        let mut s = WorldState::new(42);
        s.entities.push(Entity::new_npc(1, (0.0, 0.0)));
        s.entities.push(Entity::new_npc(2, (5.0, 0.0)));
        s.entities.push(Entity::new_monster(3, (10.0, 0.0), 1));
        s.entities[0].hp = 100.0;
        s.entities[0].max_hp = 100.0;
        s.entities[1].hp = 60.0;
        s.entities[1].max_hp = 100.0;
        s.entities[2].hp = 70.0;
        s.entities[2].max_hp = 70.0;
        s.settlements.push(SettlementState::new(10, "TestVille".into(), (0.0, 0.0)));
        s.settlements[0].stockpile[0] = 100.0; // 100 food
        s
    }

    #[test]
    fn simultaneous_damage() {
        let state = test_state();
        let merged = merge_deltas(vec![
            WorldDelta::Damage { target_id: 1, amount: 30.0, source_id: 2 },
            WorldDelta::Damage { target_id: 1, amount: 50.0, source_id: 3 },
        ]);
        let next = apply_deltas(&state, &merged);
        assert_eq!(next.entity(1).unwrap().hp, 20.0); // 100 - 80
    }

    #[test]
    fn heal_and_damage_simultaneous() {
        let state = test_state();
        let merged = merge_deltas(vec![
            WorldDelta::Damage { target_id: 2, amount: 40.0, source_id: 3 },
            WorldDelta::Heal { target_id: 2, amount: 15.0, source_id: 1 },
        ]);
        let next = apply_deltas(&state, &merged);
        assert_eq!(next.entity(2).unwrap().hp, 35.0); // 60 - 40 + 15
    }

    #[test]
    fn shield_absorbs_damage() {
        let mut state = test_state();
        state.entity_mut(1).unwrap().shield_hp = 20.0;
        let merged = merge_deltas(vec![
            WorldDelta::Damage { target_id: 1, amount: 50.0, source_id: 3 },
        ]);
        let next = apply_deltas(&state, &merged);
        let e = next.entity(1).unwrap();
        assert_eq!(e.shield_hp, 0.0);
        assert_eq!(e.hp, 70.0); // 100 - (50 - 20)
    }

    #[test]
    fn damage_causes_death() {
        let state = test_state();
        let merged = merge_deltas(vec![
            WorldDelta::Damage { target_id: 2, amount: 100.0, source_id: 3 },
        ]);
        let next = apply_deltas(&state, &merged);
        assert!(!next.entity(2).unwrap().alive);
    }

    #[test]
    fn mutual_kill() {
        let mut state = test_state();
        state.entity_mut(1).unwrap().hp = 30.0;
        state.entity_mut(3).unwrap().hp = 30.0;
        let merged = merge_deltas(vec![
            WorldDelta::Damage { target_id: 1, amount: 50.0, source_id: 3 },
            WorldDelta::Damage { target_id: 3, amount: 50.0, source_id: 1 },
        ]);
        let next = apply_deltas(&state, &merged);
        assert!(!next.entity(1).unwrap().alive);
        assert!(!next.entity(3).unwrap().alive);
    }

    #[test]
    fn force_movement() {
        let state = test_state();
        let merged = merge_deltas(vec![
            WorldDelta::Move { entity_id: 1, force: (1.0, 0.0) },
        ]);
        let next = apply_deltas(&state, &merged);
        let e = next.entity(1).unwrap();
        assert!((e.pos.0 - 1.0).abs() < 1e-6 || e.pos.0 > 0.0); // moved right
    }

    #[test]
    fn fair_rationing_consumption() {
        let state = test_state();
        // Two consumers want 60 food each from 100 stockpile.
        let merged = merge_deltas(vec![
            WorldDelta::ConsumeCommodity { location_id: 10, commodity: 0, amount: 60.0 },
            WorldDelta::ConsumeCommodity { location_id: 10, commodity: 0, amount: 60.0 },
        ]);
        let next = apply_deltas(&state, &merged);
        // Total demand 120 > 100 stockpile → stockpile drained to 0.
        assert_eq!(next.settlement(10).unwrap().stockpile[0], 0.0);
    }

    #[test]
    fn production_before_consumption() {
        let mut state = test_state();
        state.settlements[0].stockpile[0] = 0.0; // empty
        let merged = merge_deltas(vec![
            WorldDelta::ProduceCommodity { location_id: 10, commodity: 0, amount: 50.0 },
            WorldDelta::ConsumeCommodity { location_id: 10, commodity: 0, amount: 30.0 },
        ]);
        let next = apply_deltas(&state, &merged);
        // Production adds first, then consumption subtracts.
        assert_eq!(next.settlement(10).unwrap().stockpile[0], 20.0);
    }

    #[test]
    fn gold_transfer_scaling() {
        let mut state = test_state();
        state.entity_mut(1).unwrap().npc.as_mut().unwrap().gold = 10.0;
        // Two transfers from entity 1: total 20 but only has 10.
        let merged = merge_deltas(vec![
            WorldDelta::TransferGold { from_id: 1, to_id: 2, amount: 10.0 },
            WorldDelta::TransferGold { from_id: 1, to_id: 3, amount: 10.0 },
        ]);
        let next = apply_deltas(&state, &merged);
        let sender = next.entity(1).unwrap().npc.as_ref().unwrap();
        // Sender should have ~0 gold (scaled down to 50% each).
        assert!((sender.gold).abs() < 1e-4);
    }

    #[test]
    fn tick_increments() {
        let state = test_state();
        let merged = merge_deltas(Vec::<WorldDelta>::new());
        let next = apply_deltas(&state, &merged);
        assert_eq!(next.tick, state.tick + 1);
    }
}
