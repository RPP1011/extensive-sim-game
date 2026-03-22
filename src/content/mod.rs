//! Content schema layer with typed registry, generation cache, and validation.
//!
//! Three tiers of content:
//! - **Static (Tier A):** Hero stats, abilities, sim parameters from TOML/DSL
//! - **AOT-generated (Tier B):** Lore, factions, quests — generated once per campaign, cached
//! - **Runtime-generated (Tier C):** Dialogue, encounter text — on-demand

mod registry;
mod schema;
mod cache;
mod validation;
pub mod aot_pipeline;
pub mod stages;

pub use registry::{ContentRegistry, ContentId, ContentNamespace, ContentKind, ContentEntry, ContentTier};
pub use schema::{
    ContentData,
    // Tier 2 content types (Issue #15)
    ThemeContent, RegionContent, EventContent, ItemContent, NarrativeArcContent,
};
pub use cache::ContentCache;
pub use validation::{ValidationError, validate_entry};
