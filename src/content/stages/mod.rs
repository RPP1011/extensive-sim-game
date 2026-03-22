//! World-building generation stages for the AOT pipeline.
//!
//! Each stage implements [`GenerationStage`] and can use an optional
//! [`ModelClient`] for text generation, falling back to deterministic
//! procedural generation when no model is available.

mod theme;
mod factions;
mod geography;
mod settlements;
mod npcs;
mod quests;
mod events;
mod items;
mod narrative;

pub use theme::ThemeStage;
pub use factions::FactionStage;
pub use geography::GeographyStage;
pub use settlements::SettlementStage;
pub use npcs::NpcStage;
pub use quests::QuestStage;
pub use events::EventStage;
pub use items::ItemStage;
pub use narrative::NarrativeStage;
