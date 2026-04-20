//! CreatureType + Capabilities — now DSL-emitted. See `assets/sim/entities.sim`.
//!
//! Milestone 6 (2026-04-19) moved ownership of this vocabulary to
//! `engine_rules::entities`. The engine re-exports from there so existing
//! `use crate::creature::{CreatureType, Capabilities, LanguageId}` call sites
//! compile unchanged — the same pattern established by milestone 2 for the
//! `Event` enum.

pub use engine_rules::entities::{Capabilities, CreatureType};
pub use engine_rules::types::LanguageId;
