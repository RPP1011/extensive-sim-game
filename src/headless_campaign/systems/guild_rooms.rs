//! Guild room system — every 500 ticks.
//!
//! The guild base has customizable rooms that provide passive bonuses. Rooms
//! can be built, upgraded (level 1–3), and optionally staffed with an idle
//! adventurer for a 50% effectiveness boost.

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A room installed in the guild base.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GuildRoom {
    pub id: u32,
    pub room_type: RoomType,
    /// Level 1–3. Higher levels provide stronger passive effects.
    pub level: u32,
    /// An idle adventurer assigned to boost the room's effect by 50%.
    pub assigned_adventurer_id: Option<u32>,
    /// Tick at which the room was built.
    pub built_tick: u64,
}

/// The type of room that can be built in the guild.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoomType {
    /// Leadership bonus: +reputation per level, +morale to all.
    ThroneRoom,
    /// Combat planning: +counter-intel, helps battle outcomes.
    WarRoom,
    /// Gold security: +0.1% gold interest per tick per level.
    Treasury,
    /// Equipment maintenance: −10% equipment degradation per level.
    Armory,
    /// Food quality: +morale for adventurers per level.
    Kitchen,
    /// Stress relief: −2 stress per tick for idle adventurers, +herb growth.
    Garden,
    /// Vision frequency: +20% scout chance per level, +scouting range.
    Observatory,
    /// Prisoner capacity: +2 per level, +escape prevention.
    Dungeon,
    /// Crafting speed: +crafting speed, +10% quality per level.
    Workshop,
    /// NPC reputation: +5 reputation per level, +merchant visits.
    GuestQuarters,
}

impl RoomType {
    /// All room types for enumeration.
    pub const ALL: [RoomType; 10] = [
        RoomType::ThroneRoom,
        RoomType::WarRoom,
        RoomType::Treasury,
        RoomType::Armory,
        RoomType::Kitchen,
        RoomType::Garden,
        RoomType::Observatory,
        RoomType::Dungeon,
        RoomType::Workshop,
        RoomType::GuestQuarters,
    ];

    /// Gold cost to build a new room of this type.
    pub fn build_cost(self) -> f32 {
        match self {
            RoomType::ThroneRoom => 100.0,
            RoomType::WarRoom => 80.0,
            RoomType::Treasury => 90.0,
            RoomType::Armory => 70.0,
            RoomType::Kitchen => 50.0,
            RoomType::Garden => 50.0,
            RoomType::Observatory => 80.0,
            RoomType::Dungeon => 60.0,
            RoomType::Workshop => 75.0,
            RoomType::GuestQuarters => 65.0,
        }
    }

    /// Gold cost to upgrade a room to the given level.
    /// Level 2 = level*75 = 150, Level 3 = 225.
    pub fn upgrade_cost(self, target_level: u32) -> f32 {
        target_level as f32 * 75.0
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            RoomType::ThroneRoom => "Throne Room",
            RoomType::WarRoom => "War Room",
            RoomType::Treasury => "Treasury",
            RoomType::Armory => "Armory",
            RoomType::Kitchen => "Kitchen",
            RoomType::Garden => "Garden",
            RoomType::Observatory => "Observatory",
            RoomType::Dungeon => "Dungeon",
            RoomType::Workshop => "Workshop",
            RoomType::GuestQuarters => "Guest Quarters",
        }
    }

    /// Parse from a string name (case-insensitive).
    pub fn from_str_name(s: &str) -> Option<RoomType> {
        match s.to_lowercase().as_str() {
            "throneroom" | "throne_room" => Some(RoomType::ThroneRoom),
            "warroom" | "war_room" => Some(RoomType::WarRoom),
            "treasury" => Some(RoomType::Treasury),
            "armory" => Some(RoomType::Armory),
            "kitchen" => Some(RoomType::Kitchen),
            "garden" => Some(RoomType::Garden),
            "observatory" => Some(RoomType::Observatory),
            "dungeon" => Some(RoomType::Dungeon),
            "workshop" => Some(RoomType::Workshop),
            "guestquarters" | "guest_quarters" => Some(RoomType::GuestQuarters),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Room capacity
// ---------------------------------------------------------------------------

/// Maximum rooms the guild can have. Base 5, +1 per barracks tier.
pub fn max_room_count(state: &CampaignState) -> usize {
    5 + state.guild_buildings.barracks as usize
}

// ---------------------------------------------------------------------------
// Tick system
// ---------------------------------------------------------------------------

/// Guild rooms tick interval (500 ticks = 50s game time).
const GUILD_ROOMS_TICK_INTERVAL: u64 = 17;

pub fn tick_guild_rooms(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % GUILD_ROOMS_TICK_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.guild_rooms.is_empty() {
        return;
    }

    // Validate assigned adventurers: if they're no longer idle, unassign them.
    let idle_ids: Vec<u32> = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Idle)
        .map(|a| a.id)
        .collect();

    for room in &mut state.guild_rooms {
        if let Some(adv_id) = room.assigned_adventurer_id {
            if !idle_ids.contains(&adv_id) {
                room.assigned_adventurer_id = None;
            }
        }
    }

    // Collect room effects to apply (avoid borrow issues).
    let room_effects: Vec<(RoomType, u32, bool)> = state
        .guild_rooms
        .iter()
        .map(|r| (r.room_type, r.level, r.assigned_adventurer_id.is_some()))
        .collect();

    for (room_type, level, has_assignee) in &room_effects {
        let level = *level;
        let multiplier = if *has_assignee { 1.5 } else { 1.0 };

        match room_type {
            RoomType::ThroneRoom => {
                // +2 reputation per level, +3 morale to all idle adventurers
                let rep_bonus = 2.0 * level as f32 * multiplier;
                state.guild.reputation =
                    (state.guild.reputation + rep_bonus).min(100.0);

                let morale_bonus = 3.0 * multiplier;
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.morale = (adv.morale + morale_bonus).min(100.0);
                    }
                }
            }
            RoomType::WarRoom => {
                // Boost battle outcomes: reduce enemy health ratio in active battles.
                // Modeled as +1% defensive strength per level.
                let bonus = 1.0 * level as f32 * multiplier;
                state.guild.base.defensive_strength += bonus;
            }
            RoomType::Treasury => {
                // +0.1% gold interest per level per tick.
                let interest_rate = 0.001 * level as f32 * multiplier;
                let interest = state.guild.gold * interest_rate;
                if interest > 0.01 {
                    state.guild.gold += interest;
                }
            }
            RoomType::Armory => {
                // -10% equipment degradation per level.
                // Modeled as slight quality boost to inventory items.
                let quality_boost = 0.01 * level as f32 * multiplier;
                for item in &mut state.guild.inventory {
                    item.quality = (item.quality + quality_boost).min(100.0);
                }
            }
            RoomType::Kitchen => {
                // +morale per level for all non-dead adventurers.
                let morale_bonus = 2.0 * level as f32 * multiplier;
                for adv in &mut state.adventurers {
                    if adv.status != AdventurerStatus::Dead {
                        adv.morale = (adv.morale + morale_bonus).min(100.0);
                    }
                }
            }
            RoomType::Garden => {
                // -2 stress per level for idle adventurers.
                let stress_relief = 2.0 * level as f32 * multiplier;
                for adv in &mut state.adventurers {
                    if adv.status == AdventurerStatus::Idle {
                        adv.stress = (adv.stress - stress_relief).max(0.0);
                    }
                }
                // +herb production to resources.
                let herb_amount = 1.0 * level as f32 * multiplier;
                *state
                    .resources
                    .entry(ResourceType::Herbs)
                    .or_insert(0.0) += herb_amount;
            }
            RoomType::Observatory => {
                // +20% vision chance per level: add visions via RNG.
                let chance = 0.2 * level as f32 * multiplier;
                let roll = lcg_f32(&mut state.rng);
                if roll < chance.min(0.9) {
                    let vision_id = state.next_vision_id;
                    state.next_vision_id += 1;

                    // Pick a random adventurer for the vision.
                    let alive: Vec<u32> = state
                        .adventurers
                        .iter()
                        .filter(|a| a.status != AdventurerStatus::Dead)
                        .map(|a| a.id)
                        .collect();

                    if let Some(&adv_id) = alive.first() {
                        state.visions.push(Vision {
                            id: vision_id,
                            adventurer_id: adv_id,
                            vision_type: VisionType::TreasureReveal,
                            text: "The observatory reveals distant lands.".into(),
                            accuracy: 0.7,
                            tick: state.tick,
                            fulfilled: false,
                            faded: false,
                        });
                        events.push(WorldEvent::VisionReceived {
                            adventurer_id: adv_id,
                            vision_type: "observatory_scouting".into(),
                            text: "The observatory reveals distant lands.".into(),
                        });
                    }
                }
            }
            RoomType::Dungeon => {
                // Reduce prisoner escape chance: we model this by adding time
                // to prisoner capture (resetting their escape escalation).
                let prevention_per_level = 0.01 * level as f32 * multiplier;
                for prisoner in &mut state.prisoners {
                    prisoner.escape_chance =
                        (prisoner.escape_chance - prevention_per_level).max(0.01);
                }
            }
            RoomType::Workshop => {
                // +crafting speed: add small amounts of crafting resources.
                let amount = 0.5 * level as f32 * multiplier;
                *state
                    .resources
                    .entry(ResourceType::Iron)
                    .or_insert(0.0) += amount;
                *state
                    .resources
                    .entry(ResourceType::Wood)
                    .or_insert(0.0) += amount;
            }
            RoomType::GuestQuarters => {
                // +5 NPC reputation per level.
                let rep_bonus = 5.0 * level as f32 * multiplier;
                for npc in &mut state.npc_relationships {
                    npc.relationship_score =
                        (npc.relationship_score + rep_bonus).min(100.0);
                }
            }
        }

        events.push(WorldEvent::RoomBonusApplied {
            room_type: room_type.name().to_string(),
            level,
            boosted: *has_assignee,
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
    fn room_type_parse_roundtrip() {
        for rt in &RoomType::ALL {
            let name = rt.name().replace(' ', "_").to_lowercase();
            let parsed = RoomType::from_str_name(&name);
            assert_eq!(parsed, Some(*rt), "Failed roundtrip for {:?}", rt);
        }
    }

    #[test]
    fn build_costs_are_positive() {
        for rt in &RoomType::ALL {
            assert!(rt.build_cost() > 0.0);
        }
    }

    #[test]
    fn upgrade_costs_scale_with_level() {
        for rt in &RoomType::ALL {
            let c2 = rt.upgrade_cost(2);
            let c3 = rt.upgrade_cost(3);
            assert!(c2 < c3);
        }
    }

    #[test]
    fn max_rooms_scales_with_barracks() {
        let mut state = CampaignState::default_test_campaign(42);
        assert_eq!(max_room_count(&state), 5);
        state.guild_buildings.barracks = 2;
        assert_eq!(max_room_count(&state), 7);
        state.guild_buildings.barracks = 3;
        assert_eq!(max_room_count(&state), 8);
    }
}
