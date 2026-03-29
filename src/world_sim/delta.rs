use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::fidelity::Fidelity;
use super::state::{PriceReport, StatusEffect};
use super::NUM_COMMODITIES;

// ---------------------------------------------------------------------------
// WorldDelta — a single atomic change to the world state
// ---------------------------------------------------------------------------

/// Every change to world state is expressed as a delta.
/// Deltas are plain data — no closures, no references to mutable state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorldDelta {
    // --- Unit state changes ---
    Damage {
        target_id: u32,
        amount: f32,
        source_id: u32,
    },
    Heal {
        target_id: u32,
        amount: f32,
        source_id: u32,
    },
    Shield {
        target_id: u32,
        amount: f32,
        source_id: u32,
    },
    ApplyStatus {
        target_id: u32,
        status: StatusEffect,
    },
    RemoveStatus {
        target_id: u32,
        /// Discriminant from StatusEffectKind::discriminant().
        status_discriminant: u8,
    },
    /// Force-based movement: (dx, dy) velocity vector to add to entity.
    Move {
        entity_id: u32,
        force: (f32, f32),
    },
    Die {
        entity_id: u32,
    },

    // --- Economy ---
    ProduceCommodity {
        location_id: u32,
        commodity: usize,
        amount: f32,
    },
    ConsumeCommodity {
        location_id: u32,
        commodity: usize,
        amount: f32,
    },
    TransferGold {
        from_id: u32,
        to_id: u32,
        amount: f32,
    },
    TransferGoods {
        from_id: u32,
        to_id: u32,
        commodity: usize,
        amount: f32,
    },

    // --- Settlement ---
    UpdateStockpile {
        location_id: u32,
        commodity: usize,
        delta: f32,
    },
    UpdateTreasury {
        location_id: u32,
        delta: f32,
    },
    UpdatePrices {
        location_id: u32,
        prices: [f32; NUM_COMMODITIES],
    },

    // --- World ---
    EntityEntersGrid {
        entity_id: u32,
        grid_id: u32,
    },
    EntityLeavesGrid {
        entity_id: u32,
        grid_id: u32,
    },
    EscalateFidelity {
        grid_id: u32,
        new_fidelity: Fidelity,
    },

    // --- Information ---
    SharePriceReport {
        from_id: u32,
        to_id: u32,
        report: PriceReport,
    },

    // --- Cooldown ---
    TickCooldown {
        entity_id: u32,
        dt_ms: u32,
    },
}

// ---------------------------------------------------------------------------
// MergedDeltas — accumulated deltas ready for apply phase
// ---------------------------------------------------------------------------

/// Intermediate structure holding commutatively-merged deltas.
/// All fields are accumulated via commutative operations (sum, set union, etc.).
#[derive(Debug, Clone, Default)]
pub struct MergedDeltas {
    /// Per-entity accumulated damage (commutative sum).
    pub damage_by_target: HashMap<u32, f32>,
    /// Per-entity accumulated heals (commutative sum).
    pub heals_by_target: HashMap<u32, f32>,
    /// Per-entity accumulated shields (commutative sum).
    pub shields_by_target: HashMap<u32, f32>,

    /// Per-entity movement forces (commutative vector sum).
    pub forces_by_entity: HashMap<u32, (f32, f32)>,

    /// Deaths (idempotent set).
    pub deaths: HashSet<u32>,

    /// Status effects to apply (collected, dedup by kind per target on apply).
    pub new_statuses: Vec<(u32, StatusEffect)>,
    /// Status effects to remove (target_id, discriminant).
    pub remove_statuses: Vec<(u32, u8)>,

    /// Per-settlement commodity production (commutative sums).
    pub production_by_settlement: HashMap<u32, [f32; NUM_COMMODITIES]>,
    /// Per-settlement commodity consumption (commutative sums).
    pub consumption_by_settlement: HashMap<u32, [f32; NUM_COMMODITIES]>,

    /// Gold transfers: (from, to, amount). Applied with proportional scaling.
    pub gold_transfers: Vec<(u32, u32, f32)>,
    /// Goods transfers: (from, to, commodity, amount).
    pub goods_transfers: Vec<(u32, u32, usize, f32)>,

    /// Direct stockpile adjustments (commutative sums per settlement per commodity).
    pub stockpile_deltas: HashMap<u32, [f32; NUM_COMMODITIES]>,
    /// Treasury adjustments (commutative sums per settlement).
    pub treasury_deltas: HashMap<u32, f32>,
    /// Price updates — last-write-wins per settlement (deterministic: lowest location_id wins ties).
    pub price_updates: HashMap<u32, [f32; NUM_COMMODITIES]>,

    /// Grid membership changes.
    pub grid_enters: Vec<(u32, u32)>,
    pub grid_leaves: Vec<(u32, u32)>,

    /// Fidelity escalation requests.
    pub fidelity_changes: HashMap<u32, Fidelity>,

    /// Price reports to share.
    pub price_reports: Vec<(u32, u32, PriceReport)>,

    /// Cooldown ticks per entity (sum of dt_ms).
    pub cooldown_ticks: HashMap<u32, u32>,
}

impl MergedDeltas {
    /// Clear all accumulators without deallocating. HashMap::clear() preserves capacity.
    pub fn clear(&mut self) {
        self.damage_by_target.clear();
        self.heals_by_target.clear();
        self.shields_by_target.clear();
        self.forces_by_entity.clear();
        self.deaths.clear();
        self.new_statuses.clear();
        self.remove_statuses.clear();
        self.production_by_settlement.clear();
        self.consumption_by_settlement.clear();
        self.gold_transfers.clear();
        self.goods_transfers.clear();
        self.stockpile_deltas.clear();
        self.treasury_deltas.clear();
        self.price_updates.clear();
        self.grid_enters.clear();
        self.grid_leaves.clear();
        self.fidelity_changes.clear();
        self.price_reports.clear();
        self.cooldown_ticks.clear();
    }

    /// Merge a single delta into this accumulator (in-place, no alloc).
    pub fn merge_one(&mut self, delta: WorldDelta) {
        merge_one(self, delta);
    }
}

// ---------------------------------------------------------------------------
// merge_deltas — fold a bag of deltas into MergedDeltas (order-invariant)
// ---------------------------------------------------------------------------

/// Merge all deltas from all sources into a single MergedDeltas.
/// Order of input does not affect the result (commutative + associative).
pub fn merge_deltas(deltas: impl IntoIterator<Item = WorldDelta>) -> MergedDeltas {
    let mut m = MergedDeltas::default();
    for delta in deltas {
        merge_one(&mut m, delta);
    }
    m
}

fn merge_one(m: &mut MergedDeltas, delta: WorldDelta) {
    match delta {
        WorldDelta::Damage { target_id, amount, .. } => {
            *m.damage_by_target.entry(target_id).or_default() += amount;
        }
        WorldDelta::Heal { target_id, amount, .. } => {
            *m.heals_by_target.entry(target_id).or_default() += amount;
        }
        WorldDelta::Shield { target_id, amount, .. } => {
            *m.shields_by_target.entry(target_id).or_default() += amount;
        }
        WorldDelta::ApplyStatus { target_id, status } => {
            m.new_statuses.push((target_id, status));
        }
        WorldDelta::RemoveStatus { target_id, status_discriminant } => {
            m.remove_statuses.push((target_id, status_discriminant));
        }
        WorldDelta::Move { entity_id, force } => {
            let entry = m.forces_by_entity.entry(entity_id).or_default();
            entry.0 += force.0;
            entry.1 += force.1;
        }
        WorldDelta::Die { entity_id } => {
            m.deaths.insert(entity_id);
        }
        WorldDelta::ProduceCommodity { location_id, commodity, amount } => {
            m.production_by_settlement
                .entry(location_id)
                .or_insert([0.0; NUM_COMMODITIES])[commodity] += amount;
        }
        WorldDelta::ConsumeCommodity { location_id, commodity, amount } => {
            m.consumption_by_settlement
                .entry(location_id)
                .or_insert([0.0; NUM_COMMODITIES])[commodity] += amount;
        }
        WorldDelta::TransferGold { from_id, to_id, amount } => {
            m.gold_transfers.push((from_id, to_id, amount));
        }
        WorldDelta::TransferGoods { from_id, to_id, commodity, amount } => {
            m.goods_transfers.push((from_id, to_id, commodity, amount));
        }
        WorldDelta::UpdateStockpile { location_id, commodity, delta } => {
            m.stockpile_deltas
                .entry(location_id)
                .or_insert([0.0; NUM_COMMODITIES])[commodity] += delta;
        }
        WorldDelta::UpdateTreasury { location_id, delta } => {
            *m.treasury_deltas.entry(location_id).or_default() += delta;
        }
        WorldDelta::UpdatePrices { location_id, prices } => {
            m.price_updates.insert(location_id, prices);
        }
        WorldDelta::EntityEntersGrid { entity_id, grid_id } => {
            m.grid_enters.push((entity_id, grid_id));
        }
        WorldDelta::EntityLeavesGrid { entity_id, grid_id } => {
            m.grid_leaves.push((entity_id, grid_id));
        }
        WorldDelta::EscalateFidelity { grid_id, new_fidelity } => {
            m.fidelity_changes.insert(grid_id, new_fidelity);
        }
        WorldDelta::SharePriceReport { from_id, to_id, report } => {
            m.price_reports.push((from_id, to_id, report));
        }
        WorldDelta::TickCooldown { entity_id, dt_ms } => {
            *m.cooldown_ticks.entry(entity_id).or_default() += dt_ms;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: merge deltas in one order, then reverse, assert merged results match.
    fn assert_commutative(deltas: Vec<WorldDelta>) {
        let forward = merge_deltas(deltas.iter().cloned());
        let mut reversed: Vec<WorldDelta> = deltas.into_iter().rev().collect();
        let backward = merge_deltas(reversed.drain(..));

        // Commutative sums must match exactly.
        assert_eq!(forward.damage_by_target, backward.damage_by_target, "damage not commutative");
        assert_eq!(forward.heals_by_target, backward.heals_by_target, "heals not commutative");
        assert_eq!(forward.shields_by_target, backward.shields_by_target, "shields not commutative");
        assert_eq!(forward.forces_by_entity, backward.forces_by_entity, "forces not commutative");
        assert_eq!(forward.deaths, backward.deaths, "deaths not commutative");

        // Economy sums.
        assert_eq!(forward.production_by_settlement, backward.production_by_settlement);
        assert_eq!(forward.consumption_by_settlement, backward.consumption_by_settlement);
        assert_eq!(forward.stockpile_deltas, backward.stockpile_deltas);
        assert_eq!(forward.treasury_deltas, backward.treasury_deltas);
        assert_eq!(forward.cooldown_ticks, backward.cooldown_ticks);
        assert_eq!(forward.fidelity_changes, backward.fidelity_changes);
    }

    #[test]
    fn damage_commutative() {
        assert_commutative(vec![
            WorldDelta::Damage { target_id: 1, amount: 30.0, source_id: 2 },
            WorldDelta::Damage { target_id: 1, amount: 50.0, source_id: 3 },
            WorldDelta::Damage { target_id: 2, amount: 20.0, source_id: 1 },
        ]);
    }

    #[test]
    fn heal_commutative() {
        assert_commutative(vec![
            WorldDelta::Heal { target_id: 1, amount: 10.0, source_id: 2 },
            WorldDelta::Heal { target_id: 1, amount: 25.0, source_id: 3 },
        ]);
    }

    #[test]
    fn damage_sums_correctly() {
        let merged = merge_deltas(vec![
            WorldDelta::Damage { target_id: 1, amount: 30.0, source_id: 2 },
            WorldDelta::Damage { target_id: 1, amount: 50.0, source_id: 3 },
        ]);
        assert_eq!(merged.damage_by_target[&1], 80.0);
    }

    #[test]
    fn movement_forces_sum() {
        let merged = merge_deltas(vec![
            WorldDelta::Move { entity_id: 1, force: (3.0, 0.0) },
            WorldDelta::Move { entity_id: 1, force: (-5.0, -2.0) },
        ]);
        let (fx, fy) = merged.forces_by_entity[&1];
        assert!((fx - (-2.0)).abs() < 1e-6);
        assert!((fy - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn movement_commutative() {
        assert_commutative(vec![
            WorldDelta::Move { entity_id: 1, force: (3.0, 0.0) },
            WorldDelta::Move { entity_id: 1, force: (-5.0, -2.0) },
            WorldDelta::Move { entity_id: 2, force: (1.0, 1.0) },
        ]);
    }

    #[test]
    fn death_idempotent() {
        let merged = merge_deltas(vec![
            WorldDelta::Die { entity_id: 1 },
            WorldDelta::Die { entity_id: 1 },
            WorldDelta::Die { entity_id: 1 },
        ]);
        assert_eq!(merged.deaths.len(), 1);
        assert!(merged.deaths.contains(&1));
    }

    #[test]
    fn economy_commutative() {
        assert_commutative(vec![
            WorldDelta::ProduceCommodity { location_id: 10, commodity: 0, amount: 5.0 },
            WorldDelta::ProduceCommodity { location_id: 10, commodity: 0, amount: 3.0 },
            WorldDelta::ConsumeCommodity { location_id: 10, commodity: 0, amount: 4.0 },
            WorldDelta::ConsumeCommodity { location_id: 10, commodity: 1, amount: 2.0 },
            WorldDelta::UpdateStockpile { location_id: 10, commodity: 2, delta: 7.0 },
            WorldDelta::UpdateTreasury { location_id: 10, delta: -15.0 },
            WorldDelta::UpdateTreasury { location_id: 10, delta: 5.0 },
        ]);
    }

    #[test]
    fn production_sums() {
        let merged = merge_deltas(vec![
            WorldDelta::ProduceCommodity { location_id: 10, commodity: 0, amount: 5.0 },
            WorldDelta::ProduceCommodity { location_id: 10, commodity: 0, amount: 3.0 },
        ]);
        assert_eq!(merged.production_by_settlement[&10][0], 8.0);
    }

    #[test]
    fn shield_commutative() {
        assert_commutative(vec![
            WorldDelta::Shield { target_id: 1, amount: 20.0, source_id: 2 },
            WorldDelta::Shield { target_id: 1, amount: 10.0, source_id: 3 },
        ]);
    }

    #[test]
    fn mixed_deltas_commutative() {
        assert_commutative(vec![
            WorldDelta::Damage { target_id: 1, amount: 30.0, source_id: 2 },
            WorldDelta::Heal { target_id: 1, amount: 10.0, source_id: 3 },
            WorldDelta::Shield { target_id: 1, amount: 20.0, source_id: 3 },
            WorldDelta::Move { entity_id: 1, force: (1.0, 2.0) },
            WorldDelta::Move { entity_id: 2, force: (-1.0, 0.0) },
            WorldDelta::Die { entity_id: 5 },
            WorldDelta::ProduceCommodity { location_id: 10, commodity: 0, amount: 5.0 },
            WorldDelta::ConsumeCommodity { location_id: 10, commodity: 0, amount: 3.0 },
            WorldDelta::UpdateTreasury { location_id: 10, delta: 100.0 },
            WorldDelta::TickCooldown { entity_id: 1, dt_ms: 100 },
            WorldDelta::TickCooldown { entity_id: 1, dt_ms: 100 },
        ]);
    }

    #[test]
    fn empty_merge() {
        let merged = merge_deltas(Vec::<WorldDelta>::new());
        assert!(merged.damage_by_target.is_empty());
        assert!(merged.forces_by_entity.is_empty());
        assert!(merged.deaths.is_empty());
    }
}
