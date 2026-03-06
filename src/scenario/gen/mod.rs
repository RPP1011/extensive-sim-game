//! Scenario generation engine.
//!
//! Coverage-driven, constraint-based generation that exhaustively explores the
//! space of hero compositions, enemy waves, room types, and difficulty levels.
//! Simulation is fast enough (~50-250K scenarios/min parallel) that we generate
//! everything meaningful and deduplicate rather than artificially capping count.

mod metadata;
mod coverage;
mod strategies;

pub use metadata::{Role, HeroMeta, ALL_HEROES, ALL_LOL_HEROES, ROOM_TYPES, heroes_by_role};
pub use strategies::{generate, write_scenarios};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

pub struct GenConfig {
    pub seed: u64,
    pub seed_variants: u32,
    pub include_synergy_pairs: bool,
    pub include_stress_archetypes: bool,
    pub include_difficulty_ladders: bool,
    pub include_room_aware: bool,
    pub include_size_spectrum: bool,
    pub include_hero_vs_hero: bool,
    pub hvh_count: usize,
    pub extra_random: usize,
    pub verbose: bool,
}

impl Default for GenConfig {
    fn default() -> Self {
        Self {
            seed: 2026,
            seed_variants: 3,
            include_synergy_pairs: true,
            include_stress_archetypes: true,
            include_difficulty_ladders: true,
            include_room_aware: true,
            include_size_spectrum: true,
            include_hero_vs_hero: true,
            hvh_count: 200,
            extra_random: 200,
            verbose: false,
        }
    }
}
