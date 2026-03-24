//! Tick systems for the headless campaign simulator.
//!
//! Each system is a pure function: `fn(state: &mut CampaignState, deltas: &mut StepDeltas, events: &mut Vec<WorldEvent>)`
//! Systems fire at different cadences (all multiples of the 100ms base tick).

pub mod choices;
pub mod crisis;
pub mod travel;
pub mod supply;
pub mod battles;
pub mod quest_lifecycle;
pub mod quest_generation;
pub mod quest_expiry;
pub mod adventurer_condition;
pub mod adventurer_recovery;
pub mod faction_ai;
pub mod npc_relationships;
pub mod economy;
pub mod progression;
pub mod progression_triggers;
pub mod cooldowns;
pub mod recruitment;
pub mod threat;
pub mod bonds;
pub mod verify;
pub mod interception;
pub mod seasons;
