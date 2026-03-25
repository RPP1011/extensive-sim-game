//! Guild buildings system — every 100 ticks (~10s).
//!
//! The guild can spend gold on permanent structures that provide passive bonuses.
//! Each building has 3 upgrade tiers. Auto-upgrades when the guild has enough gold
//! and prioritizes by cost-efficiency.

use serde::{Deserialize, Serialize};

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::CampaignState;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// Tracks the tier (0–3) of each guild building.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GuildBuildings {
    /// +10/20/30% XP gain for all adventurers.
    pub training_grounds: u8,
    /// Scout range +1/2/3, threat warning earlier.
    pub watchtower: u8,
    /// +10/20/30% quest gold rewards.
    pub trade_post: u8,
    /// Max adventurers +2/4/6.
    pub barracks: u8,
    /// Recovery speed ×1.5/2.0/3.0.
    pub infirmary: u8,
    /// Party size +1/2/3, formation bonuses.
    pub war_room: u8,
}

/// A building type that can be upgraded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuildingType {
    TrainingGrounds,
    Watchtower,
    TradePost,
    Barracks,
    Infirmary,
    WarRoom,
}

impl BuildingType {
    /// All building types in priority order (cheapest first for auto-upgrade).
    pub const ALL: [BuildingType; 6] = [
        BuildingType::Watchtower,
        BuildingType::TrainingGrounds,
        BuildingType::Infirmary,
        BuildingType::TradePost,
        BuildingType::Barracks,
        BuildingType::WarRoom,
    ];

    /// Gold cost to upgrade to the given tier (1, 2, or 3).
    pub fn tier_cost(self, tier: u8) -> f32 {
        match (self, tier) {
            (BuildingType::TrainingGrounds, 1) => 100.0,
            (BuildingType::TrainingGrounds, 2) => 250.0,
            (BuildingType::TrainingGrounds, 3) => 500.0,

            (BuildingType::Watchtower, 1) => 80.0,
            (BuildingType::Watchtower, 2) => 200.0,
            (BuildingType::Watchtower, 3) => 400.0,

            (BuildingType::TradePost, 1) => 120.0,
            (BuildingType::TradePost, 2) => 300.0,
            (BuildingType::TradePost, 3) => 600.0,

            (BuildingType::Barracks, 1) => 150.0,
            (BuildingType::Barracks, 2) => 350.0,
            (BuildingType::Barracks, 3) => 700.0,

            (BuildingType::Infirmary, 1) => 100.0,
            (BuildingType::Infirmary, 2) => 250.0,
            (BuildingType::Infirmary, 3) => 500.0,

            (BuildingType::WarRoom, 1) => 200.0,
            (BuildingType::WarRoom, 2) => 400.0,
            (BuildingType::WarRoom, 3) => 800.0,

            _ => f32::INFINITY, // tier 0 or >3 — no cost
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            BuildingType::TrainingGrounds => "Training Grounds",
            BuildingType::Watchtower => "Watchtower",
            BuildingType::TradePost => "Trade Post",
            BuildingType::Barracks => "Barracks",
            BuildingType::Infirmary => "Infirmary",
            BuildingType::WarRoom => "War Room",
        }
    }
}

impl GuildBuildings {
    /// Get the current tier of a building.
    pub fn tier(&self, building: BuildingType) -> u8 {
        match building {
            BuildingType::TrainingGrounds => self.training_grounds,
            BuildingType::Watchtower => self.watchtower,
            BuildingType::TradePost => self.trade_post,
            BuildingType::Barracks => self.barracks,
            BuildingType::Infirmary => self.infirmary,
            BuildingType::WarRoom => self.war_room,
        }
    }

    /// Set the tier of a building.
    pub fn set_tier(&mut self, building: BuildingType, tier: u8) {
        let field = match building {
            BuildingType::TrainingGrounds => &mut self.training_grounds,
            BuildingType::Watchtower => &mut self.watchtower,
            BuildingType::TradePost => &mut self.trade_post,
            BuildingType::Barracks => &mut self.barracks,
            BuildingType::Infirmary => &mut self.infirmary,
            BuildingType::WarRoom => &mut self.war_room,
        };
        *field = tier.min(3);
    }

    /// Whether all buildings are at max tier.
    pub fn all_maxed(&self) -> bool {
        self.training_grounds >= 3
            && self.watchtower >= 3
            && self.trade_post >= 3
            && self.barracks >= 3
            && self.infirmary >= 3
            && self.war_room >= 3
    }

    // --- Bonus accessors ---

    /// XP multiplier from Training Grounds (1.0 = no bonus).
    pub fn xp_multiplier(&self) -> f32 {
        1.0 + self.training_grounds as f32 * 0.1
    }

    /// Scout range bonus from Watchtower.
    pub fn scout_range_bonus(&self) -> u8 {
        self.watchtower
    }

    /// Gold reward multiplier from Trade Post (1.0 = no bonus).
    pub fn gold_reward_multiplier(&self) -> f32 {
        1.0 + self.trade_post as f32 * 0.1
    }

    /// Extra adventurer slots from Barracks.
    pub fn extra_adventurer_slots(&self) -> usize {
        self.barracks as usize * 2
    }

    /// Recovery speed multiplier from Infirmary (1.0 = no bonus).
    pub fn recovery_multiplier(&self) -> f32 {
        match self.infirmary {
            0 => 1.0,
            1 => 1.5,
            2 => 2.0,
            _ => 3.0,
        }
    }

    /// Extra party size from War Room.
    pub fn extra_party_size(&self) -> u8 {
        self.war_room
    }
}

// ---------------------------------------------------------------------------
// Tick system
// ---------------------------------------------------------------------------

/// Building tick interval in campaign ticks (100 ticks = 10s game time).
const BUILDING_TICK_INTERVAL: u64 = 3;

/// Run the buildings system: auto-upgrade buildings when gold is available.
///
/// Called every tick but only does work every `BUILDING_TICK_INTERVAL` ticks.
/// Bonuses are applied passively by other systems reading `guild_buildings`.
pub fn tick_buildings(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % BUILDING_TICK_INTERVAL != 0 {
        return;
    }

    // Skip if all buildings are maxed
    if state.guild_buildings.all_maxed() {
        return;
    }

    // Auto-upgrade: try to upgrade the cheapest available building.
    // Only one upgrade per tick to avoid spending all gold at once.
    let mut best: Option<(BuildingType, f32)> = None;

    for &building in &BuildingType::ALL {
        let current_tier = state.guild_buildings.tier(building);
        if current_tier >= 3 {
            continue;
        }
        let next_tier = current_tier + 1;
        let cost = building.tier_cost(next_tier);
        if cost <= state.guild.gold {
            if best.is_none() || cost < best.unwrap().1 {
                best = Some((building, cost));
            }
        }
    }

    if let Some((building, cost)) = best {
        let new_tier = state.guild_buildings.tier(building) + 1;
        state.guild.gold -= cost;
        state.guild_buildings.set_tier(building, new_tier);

        events.push(WorldEvent::GoldChanged {
            amount: -cost,
            reason: format!(
                "Upgraded {} to tier {}",
                building.name(),
                new_tier,
            ),
        });

        events.push(WorldEvent::BuildingUpgraded {
            building: building.name().to_string(),
            new_tier,
            cost,
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_costs_are_increasing() {
        for &b in &BuildingType::ALL {
            let c1 = b.tier_cost(1);
            let c2 = b.tier_cost(2);
            let c3 = b.tier_cost(3);
            assert!(c1 < c2, "{:?} tier 1 ({}) should be less than tier 2 ({})", b, c1, c2);
            assert!(c2 < c3, "{:?} tier 2 ({}) should be less than tier 3 ({})", b, c2, c3);
        }
    }

    #[test]
    fn bonus_values() {
        let mut b = GuildBuildings::default();
        assert_eq!(b.xp_multiplier(), 1.0);
        assert_eq!(b.gold_reward_multiplier(), 1.0);
        assert_eq!(b.recovery_multiplier(), 1.0);
        assert_eq!(b.extra_adventurer_slots(), 0);
        assert_eq!(b.extra_party_size(), 0);

        b.training_grounds = 2;
        assert!((b.xp_multiplier() - 1.2).abs() < 0.001);

        b.trade_post = 3;
        assert!((b.gold_reward_multiplier() - 1.3).abs() < 0.001);

        b.infirmary = 2;
        assert!((b.recovery_multiplier() - 2.0).abs() < 0.001);

        b.barracks = 3;
        assert_eq!(b.extra_adventurer_slots(), 6);

        b.war_room = 1;
        assert_eq!(b.extra_party_size(), 1);
    }

    #[test]
    fn all_maxed() {
        let mut b = GuildBuildings::default();
        assert!(!b.all_maxed());

        b.training_grounds = 3;
        b.watchtower = 3;
        b.trade_post = 3;
        b.barracks = 3;
        b.infirmary = 3;
        b.war_room = 3;
        assert!(b.all_maxed());
    }

    #[test]
    fn set_tier_caps_at_3() {
        let mut b = GuildBuildings::default();
        b.set_tier(BuildingType::Barracks, 5);
        assert_eq!(b.barracks, 3);
    }
}
