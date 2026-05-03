//! CreatureType + Capabilities — now DSL-emitted. See `assets/sim/entities.sim`.
//!
//! Milestone 6 (2026-04-19) moved ownership of this vocabulary to
//! `engine_data::entities`; `LanguageId` is in `engine_data::types`.
//! Task 4 (Plan B1') dropped the re-export shims from this module; callers
//! now import directly from `engine_data`.
//
// (re-exports moved to engine_data; types live there now)
