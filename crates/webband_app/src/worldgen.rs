//! World generation — port of `F:\MB\src\guild\worldgen.ts`. ~12 named
//! landmarks placed terrain-aware against a seed-offset fbm field (the TS
//! radius/angle rejection scheme, ported whole — it IS the simple scheme),
//! plus the narrative ties (companion/band hook tags bound to fitting
//! landmarks, iterated bands-then-companions in cast order) and the band-goal
//! parameterization. Every draw comes from the guild's seeded rng.
//!
//! THE ONE-CITY GUARANTEE: the kind roster always leads with exactly one
//! `city` slot (it claims the best open ground), and substitutions never add
//! another.

use serde::{Deserialize, Serialize};

use crate::castgen::{Cast, GoalKind};
use crate::noise::{fbm, hash};
use crate::rng::{rng_float, rng_int, rng_pick, RngState};

// ---------- The shared terrain field ----------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WorldOffset {
    pub offset_x: f64,
    pub offset_y: f64,
}

pub const WATER_LEVEL: f64 = 0.38;
pub const HILL_LEVEL: f64 = 0.6;

/// Derive the founding's country from the seed. Integer offsets stepped 8
/// apart so two seeds' sampling windows can never overlap.
pub fn roll_world_offset(seed: u32) -> WorldOffset {
    WorldOffset {
        offset_x: 40.0 + (hash(f64::from(seed), 11.0) * 4096.0).floor() * 8.0,
        offset_y: 40.0 + (hash(f64::from(seed), 23.0) * 4096.0).floor() * 8.0,
    }
}

/// THE chart terrain field — landmark placement samples this and nothing else.
pub fn world_field(off: WorldOffset, x: f64, z: f64) -> f64 {
    fbm(x * 0.02 + off.offset_x, z * 0.02 + off.offset_y, 4)
}

/// The tree-scatter detail field (the chart's second fbm), same offsets.
pub fn world_detail(off: WorldOffset, x: f64, z: f64) -> f64 {
    fbm(x * 0.09 + off.offset_x, z * 0.09 + off.offset_y, 2)
}

// ---------- Landmark roster ----------

const X_MIN: f64 = -104.0;
const X_MAX: f64 = 76.0;
const Z_MIN: f64 = -40.0;
const Z_MAX: f64 = 62.0;
const MIN_APART: f64 = 17.0;
const MIN_FROM_HALL: f64 = 12.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkKind {
    City,
    Village,
    Crossroads,
    Ford,
    Port,
    Fen,
    Pass,
    Abbey,
    Ruin,
    Barrow,
    Mill,
}

impl LandmarkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            LandmarkKind::City => "city",
            LandmarkKind::Village => "village",
            LandmarkKind::Crossroads => "crossroads",
            LandmarkKind::Ford => "ford",
            LandmarkKind::Port => "port",
            LandmarkKind::Fen => "fen",
            LandmarkKind::Pass => "pass",
            LandmarkKind::Abbey => "abbey",
            LandmarkKind::Ruin => "ruin",
            LandmarkKind::Barrow => "barrow",
            LandmarkKind::Mill => "mill",
        }
    }
}

/// Hook vocabulary per kind — what stories a place can hold.
pub fn kind_tags(kind: LandmarkKind) -> &'static [&'static str] {
    match kind {
        LandmarkKind::City => &["trade", "roads", "hearth"],
        LandmarkKind::Village => &["trade", "hearth"],
        LandmarkKind::Crossroads => &["trade", "roads", "dead"],
        LandmarkKind::Ford => &["roads", "water"],
        LandmarkKind::Port => &["sea", "trade"],
        LandmarkKind::Fen => &["marsh"],
        LandmarkKind::Pass => &["mountain", "high"],
        LandmarkKind::Abbey => &["faith", "learning"],
        LandmarkKind::Ruin => &["fallen-house", "war"],
        LandmarkKind::Barrow => &["dead", "old"],
        LandmarkKind::Mill => &["trade", "hearth"],
    }
}

const ALL_KINDS: [LandmarkKind; 11] = [
    LandmarkKind::City, LandmarkKind::Village, LandmarkKind::Crossroads,
    LandmarkKind::Ford, LandmarkKind::Port, LandmarkKind::Fen, LandmarkKind::Pass,
    LandmarkKind::Abbey, LandmarkKind::Ruin, LandmarkKind::Barrow, LandmarkKind::Mill,
];

/// The flat hook vocabulary (castgen's assert checks hooks against it).
pub fn hook_vocab() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for k in ALL_KINDS {
        for t in kind_tags(k) {
            if !v.contains(t) {
                v.push(t);
            }
        }
    }
    v
}

#[derive(Debug, Clone, Copy)]
struct KindSpec {
    kind: LandmarkKind,
    lo: f64,
    hi: f64,
    needs_water: bool,
}

const OPEN_LO: f64 = 0.42;
const OPEN_HI: f64 = 0.58;

fn open(kind: LandmarkKind) -> KindSpec {
    KindSpec { kind, lo: OPEN_LO, hi: OPEN_HI, needs_water: false }
}

fn spec(kind: LandmarkKind, lo: f64, hi: f64) -> KindSpec {
    KindSpec { kind, lo, hi, needs_water: false }
}

/// Terrain-adaptive roster: 13 slots (the city + 12), constrained kinds
/// first. EXACTLY ONE CITY per country.
fn kind_roster(has_water: bool, has_hills: bool) -> Vec<KindSpec> {
    let mut slots: Vec<KindSpec> = Vec::new();
    slots.push(spec(LandmarkKind::City, 0.44, 0.6));
    if has_water {
        slots.push(KindSpec { kind: LandmarkKind::Port, lo: 0.38, hi: 0.45, needs_water: true });
        slots.push(KindSpec { kind: LandmarkKind::Ford, lo: 0.38, hi: 0.48, needs_water: true });
    }
    if has_hills {
        slots.push(spec(LandmarkKind::Pass, 0.62, 0.9));
        slots.push(spec(LandmarkKind::Ruin, 0.55, 0.68));
    }
    if has_water {
        slots.push(spec(LandmarkKind::Fen, 0.38, 0.43));
    }
    slots.push(spec(LandmarkKind::Abbey, 0.48, 0.62));
    slots.push(open(LandmarkKind::Village));
    slots.push(open(LandmarkKind::Village));
    slots.push(open(LandmarkKind::Village));
    slots.push(open(LandmarkKind::Mill));
    slots.push(spec(LandmarkKind::Barrow, 0.44, 0.62));
    slots.push(open(LandmarkKind::Crossroads));
    // Substitutions keep the count at 13 whatever the country looks like.
    while slots.len() < 13 {
        let s = if slots.len() % 2 == 0 {
            open(LandmarkKind::Village)
        } else if has_hills {
            spec(LandmarkKind::Pass, 0.62, 0.9)
        } else {
            spec(LandmarkKind::Barrow, 0.44, 0.62)
        };
        slots.push(s);
    }
    slots.truncate(13);
    slots
}

// ---------- Names ----------

const STEMS: [&str; 16] = [
    "Oster", "Harlow", "Wolf", "Thorn", "Ash", "Grey", "Redd", "Stane",
    "Marle", "Dunn", "Fenn", "Salt", "Raven", "Elm", "Crow", "Bracken",
];
const VILLAGE_ENDS: [&str; 6] = ["by", "wick", "stead", "ton", "field", "mere"];

fn name_for(g: &mut RngState, kind: LandmarkKind, used: &mut Vec<String>) -> String {
    let mut stem = (*rng_pick(g, &STEMS)).to_string();
    let mut guard = 40;
    while used.contains(&stem) && guard > 0 {
        stem = (*rng_pick(g, &STEMS)).to_string();
        guard -= 1;
    }
    used.push(stem.clone());
    match kind {
        LandmarkKind::City => format!("{stem}burgh"),
        LandmarkKind::Village => format!("{stem}{}", rng_pick(g, &VILLAGE_ENDS)),
        LandmarkKind::Ford => format!("{stem} Ford"),
        LandmarkKind::Port => {
            if rng_float(g) < 0.5 {
                format!("Port {stem}")
            } else {
                format!("{stem} Quay")
            }
        }
        LandmarkKind::Pass => format!("the {stem} Pass"),
        LandmarkKind::Fen => format!("the {stem} Fens"),
        LandmarkKind::Abbey => format!("{stem} Abbey"),
        LandmarkKind::Ruin => format!("{stem} Keep"),
        LandmarkKind::Mill => format!("{stem} Mill"),
        LandmarkKind::Barrow => format!("the {stem} Barrow"),
        LandmarkKind::Crossroads => format!("{stem} Cross"),
    }
}

// ---------- Generation ----------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Landmark {
    pub id: String,
    pub name: String,
    pub kind: LandmarkKind,
    pub x: f64,
    pub z: f64,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorldState {
    /// Offsets are STORED, never re-derived — a derivation change can't
    /// redraw a save's chart.
    pub offset_x: f64,
    pub offset_y: f64,
    pub landmarks: Vec<Landmark>,
    /// companionId or `band:<id>` → landmarkId, in BIND ORDER (bands first,
    /// then companions, cast order). A missing entry is honest — the sea may
    /// not exist this founding.
    pub homes: Vec<(String, String)>,
}

impl WorldState {
    pub fn home_of(&self, key: &str) -> Option<&str> {
        self.homes.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn landmark_by_id(&self, id: &str) -> Option<&Landmark> {
        self.landmarks.iter().find(|l| l.id == id)
    }
}

pub fn generate_world(g: &mut RngState, seed: u32, cast: &Cast) -> WorldState {
    let off = roll_world_offset(seed);

    // Coarse survey of the country: does it hold water, does it hold hills?
    let mut has_water = false;
    let mut has_hills = false;
    let mut z = Z_MIN;
    while z <= Z_MAX {
        let mut x = X_MIN;
        while x <= X_MAX {
            let n = world_field(off, x, z);
            if n < 0.37 {
                has_water = true;
            }
            if n > 0.63 {
                has_hills = true;
            }
            x += 10.0;
        }
        z += 10.0;
    }

    let near_water = |x: f64, z: f64| -> bool {
        world_field(off, x - 6.0, z)
            .min(world_field(off, x + 6.0, z))
            .min(world_field(off, x, z - 6.0))
            .min(world_field(off, x, z + 6.0))
            < 0.365
    };

    let mut landmarks: Vec<Landmark> = Vec::new();
    let mut used: Vec<String> = Vec::new();
    let mut counts: Vec<(LandmarkKind, u32)> = Vec::new();
    for spec in kind_roster(has_water, has_hills) {
        let mid = (spec.lo + spec.hi) / 2.0;
        let mut best: Option<(f64, f64, f64)> = None; // (x, z, score)
        for _ in 0..40 {
            let angle = rng_float(g) * std::f64::consts::PI * 2.0;
            // The country is WIDE; gentle near-bias keeps a settled heartland.
            let radius = 20.0 + rng_float(g).powf(1.15) * 96.0;
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            if x < X_MIN || x > X_MAX || z < Z_MIN || z > Z_MAX {
                continue;
            }
            if (x * x + z * z).sqrt() < MIN_FROM_HALL {
                continue;
            }
            if landmarks
                .iter()
                .any(|l| ((x - l.x).powi(2) + (z - l.z).powi(2)).sqrt() < MIN_APART)
            {
                continue;
            }
            let n = world_field(off, x, z);
            let fits = n >= spec.lo && n <= spec.hi && (!spec.needs_water || near_water(x, z));
            if fits {
                best = Some((x, z, 0.0));
                break;
            }
            // Remember the least-wrong spot so a slot never fails to place.
            let score = (n - mid).abs() + if spec.needs_water && !near_water(x, z) { 0.5 } else { 0.0 };
            match best {
                Some((_, _, bs)) if score >= bs => {}
                _ => best = Some((x, z, score)),
            }
        }
        let Some((x, z, _)) = best else {
            continue; // pathological (all tries out of bounds) — drop the slot
        };
        let ordinal = if let Some(pos) = counts.iter().position(|(k, _)| *k == spec.kind) {
            counts[pos].1 += 1;
            counts[pos].1
        } else {
            counts.push((spec.kind, 1));
            1
        };
        let name = name_for(g, spec.kind, &mut used);
        landmarks.push(Landmark {
            id: format!("lm-{}-{}", spec.kind.as_str(), ordinal),
            name,
            kind: spec.kind,
            x,
            z,
            tags: kind_tags(spec.kind).iter().map(|t| (*t).to_string()).collect(),
        });
    }

    // Narrative ties: bands first, then companions, in cast order. First hook
    // with a match wins; no match anywhere → no home this founding.
    let mut homes: Vec<(String, String)> = Vec::new();
    {
        let mut bind = |g: &mut RngState, key: String, hooks: &[String]| {
            for hook in hooks {
                let fits: Vec<usize> = landmarks
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.tags.iter().any(|t| t == hook))
                    .map(|(i, _)| i)
                    .collect();
                if !fits.is_empty() {
                    let pick = fits[crate::rng::rng_pick_idx(g, fits.len())];
                    homes.push((key, landmarks[pick].id.clone()));
                    return;
                }
            }
        };
        for band in &cast.bands {
            bind(g, format!("band:{}", band.id), &band.hooks);
        }
        for c in &cast.companions {
            if let Some(hooks) = &c.hooks {
                bind(g, c.id.clone(), hooks);
            }
        }
    }

    WorldState { offset_x: off.offset_x, offset_y: off.offset_y, landmarks, homes }
}

// ---------- Band goals (generation-time) ----------

pub const ELITE_NAMES: [&str; 8] = [
    "Kradus", "Vorn", "Sarga", "Hetha", "Drogan", "Ulfric", "Mazok", "Irena",
];
pub const ELITE_EPITHETS: [&str; 7] = [
    "the Red", "Iron-Hand", "the Cruel", "Wolfskin", "Half-Ear", "the Burned", "Oathbreaker",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BandGoal {
    Guild,
    Deed { landmark_id: String },
    Prosperity { need_renown: i64, need_gold: i64 },
    Debt { landmark_id: String, foe_name: String },
    Poach { target_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BandStatus {
    Signed,
    Camped,
    /// Signed, but their goal has gone unserved too long: two days and they
    /// ride out unless something changes (`goals.ts` — the bands slice, S11).
    /// Rolled foundings never start here; only `tick_band_goals` sets it.
    Notice,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandState {
    pub goal: BandGoal,
    pub want: Option<BandGoal>,
    pub status: BandStatus,
    pub camp_landmark_id: String,
    pub patience: i64,
    pub times_departed: i64,
}

/// Every band's goal, parameterized against the generated world. Runs right
/// after `generate_world`; the draw order (name → cast → world → goals →
/// colony) is frozen forever. A band already rostered (the founders) starts
/// `Signed`.
pub fn roll_band_goals(
    g: &mut RngState,
    cast: &Cast,
    world: &WorldState,
    roster: &[String],
) -> Vec<(String, BandState)> {
    let mut states: Vec<(String, BandState)> = Vec::new();
    for band in &cast.bands {
        let key = format!("band:{}", band.id);
        let camp = match world.home_of(&key) {
            Some(id) => id.to_string(),
            None => rng_pick(g, &world.landmarks).id.clone(),
        };
        let goal = match band.goal_kind {
            GoalKind::Guild => BandGoal::Guild,
            GoalKind::Deed => BandGoal::Deed { landmark_id: camp.clone() },
            GoalKind::Prosperity => BandGoal::Prosperity {
                need_renown: rng_int(g, 6, 10) * 25,
                need_gold: rng_int(g, 8, 12) * 25,
            },
            GoalKind::Debt => BandGoal::Debt {
                landmark_id: camp.clone(),
                foe_name: format!("{} {}", rng_pick(g, &ELITE_NAMES), rng_pick(g, &ELITE_EPITHETS)),
            },
        };
        let want = band
            .want_poach_id
            .as_ref()
            .map(|t| BandGoal::Poach { target_id: t.clone() });
        let rostered = cast
            .companions
            .iter()
            .filter(|c| c.band.as_deref() == Some(band.id.as_str()) && roster.iter().any(|r| r == &c.id))
            .count();
        states.push((
            band.id.clone(),
            BandState {
                goal,
                want,
                status: if rostered >= 2 { BandStatus::Signed } else { BandStatus::Camped },
                camp_landmark_id: camp,
                patience: if band.founders { 100 } else { 60 },
                times_departed: 0,
            },
        ));
    }
    states
}
