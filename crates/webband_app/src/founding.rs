//! The founding pipeline — the port of `newGuild`'s generation half
//! (`F:\MB\src\guild\state.ts`). One seeded pass in the FROZEN draw order:
//!
//!   name → cast → world → goals → colony
//!
//! then the scenario is stamped LAST with zero draws (the comparability law).
//! The TS appends factions → ambition draws AFTER the colony roll; those
//! systems belong to a later slice, and appending to the draw order is safe
//! (the TS itself documents this: "Appending to the draw order is safe;
//! inserting anywhere above would desynchronise every seed").
//!
//! The rng's (seed, counter) pair is persisted on the founding so a save can
//! resume the exact stream (`resume_rng`).

use serde::{Deserialize, Serialize};

use crate::castgen::{roll_cast, Cast};
use crate::error::GenError;
use crate::rng::{rng_float, rng_pick, RngState};
use crate::scenario::{apply_scenario, ScenarioId};
use crate::worldgen::{generate_world, roll_band_goals, BandState, WorldState};

/// The playable colony board's hex radius (`state.ts COLONY_R`).
pub const COLONY_R: i32 = 60;
/// The founding ground: cache, muster, and ward (`state.ts CACHE_GROUND`).
pub const CACHE_GROUND: (i32, i32) = (0, 0);

// ---------- Guild name ----------

const NAME_FIRST: [&str; 8] = [
    "Grey", "Black", "Gilded", "Wandering", "Iron", "Broken", "Silver", "Crimson",
];
const NAME_SECOND: [&str; 8] = [
    "Banner", "Lantern", "Wolf", "Compass", "Crown", "Road", "Spear", "Raven",
];

pub fn roll_guild_name(g: &mut RngState) -> String {
    format!("{} {}", rng_pick(g, &NAME_FIRST), rng_pick(g, &NAME_SECOND))
}

// ---------- Colony terrain ----------

/// The colony's country, rolled ONCE from the guild rng and stored
/// (`scene/terrain.ts rollTerrain` with `{ river: false, kind: null }` —
/// exactly five draws; the river branch and barrow mounds never fire for the
/// colony board). Field geometry (fbm relief) is derived at render/sim time
/// from these params, never stored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TerrainParams {
    pub hill_ox: f64,
    pub hill_oz: f64,
    pub track_theta: f64,
    pub track_off: f64,
    pub palette_idx: u8,
    pub water_level: f64,
    pub shrub_bar: f64,
    pub river_on: bool,
}

pub fn roll_colony_terrain(g: &mut RngState) -> TerrainParams {
    let hill_ox = rng_float(g) * 512.0;
    let hill_oz = rng_float(g) * 512.0;
    let track_theta = rng_float(g) * std::f64::consts::PI;
    let track_off = (rng_float(g) - 0.5) * 12.0;
    let palette_idx = [0u8, 0, 1, 2][(rng_float(g) * 4.0).floor() as usize];
    TerrainParams {
        hill_ox,
        hill_oz,
        track_theta,
        track_off,
        palette_idx,
        water_level: -2.9,
        shrub_bar: 0.65,
        river_on: false,
    }
}

// ---------- The founding ----------

/// A stray item stack on the colony ground (the scenario's resource cache).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stack {
    pub item: String,
    pub count: u32,
    pub q: i32,
    pub r: i32,
    /// Absolute day the stack spoils, stamped at drop time (TS idiom).
    pub spoil_day: Option<i64>,
}

/// The generation half of `GuildState` v3 — everything a founding rolls,
/// plus the scenario stamp. Campaign-runtime state (roster needs, colony
/// buildings, chronicle, director) belongs to later slices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Founding {
    pub name: String,
    pub seed: u32,
    /// The stream position after all founding draws — resume from here.
    pub rng_counter: u32,
    pub day: i64,
    pub gold: i64,
    pub renown: i64,
    /// Founder companion ids (the founders band, seated at founding).
    pub roster: Vec<String>,
    pub cast: Cast,
    pub world: WorldState,
    /// Band id → state, in cast band order (the TS Record's insertion order).
    pub band_states: Vec<(String, BandState)>,
    pub colony_terrain: TerrainParams,
    pub scenario: ScenarioId,
    /// The resource cache: stray stacks at CACHE_GROUND.
    pub stacks: Vec<Stack>,
    /// Days of food the dawn provisioner keeps bought per mouth (0 = none).
    pub provisioning: u32,
    /// The scenario's standing offset toward the country's powers.
    pub standing: i64,
}

impl Founding {
    /// Resume the seeded stream exactly where founding left it.
    pub fn resume_rng(&self) -> RngState {
        RngState { seed: self.seed, rng_counter: self.rng_counter }
    }
}

/// Roll a founding: the frozen draw order, then the zero-draw scenario stamp.
pub fn new_founding(seed: u32, legacy_renown: i64, scenario: ScenarioId) -> Result<Founding, GenError> {
    let mut rng = RngState::new(seed);
    // Fixed draw order: name → cast → world → goals → colony. The cast must
    // roll BEFORE the world (hook binding iterates it).
    let name = roll_guild_name(&mut rng);
    let cast = roll_cast(&mut rng)?;
    let founders_band = cast
        .bands
        .iter()
        .find(|b| b.founders)
        .unwrap_or(&cast.bands[0])
        .id
        .clone();
    let roster: Vec<String> = cast
        .companions
        .iter()
        .filter(|c| c.band.as_deref() == Some(founders_band.as_str()))
        .map(|c| c.id.clone())
        .collect();
    let world = generate_world(&mut rng, seed, &cast);
    let band_states = roll_band_goals(&mut rng, &cast, &world, &roster);
    let colony_terrain = roll_colony_terrain(&mut rng);

    let counter_after_rolls = rng.rng_counter;
    let mut f = Founding {
        name,
        seed,
        rng_counter: counter_after_rolls,
        day: 1,
        gold: 150, // newGuild's initial value; the scenario overwrites it
        renown: legacy_renown,
        roster,
        cast,
        world,
        band_states,
        colony_terrain,
        scenario,
        stacks: Vec::new(),
        provisioning: 0,
        standing: 0,
    };
    // The starting condition is stamped LAST and takes NO draws — the same
    // seed rolls the same cast, country and goals in all four conditions.
    apply_scenario(&mut f, scenario);
    debug_assert_eq!(f.rng_counter, counter_after_rolls, "scenario must take zero draws");
    Ok(f)
}
