//! FACTIONS — the demand side of the country (`F:\MB\src\guild\factions.ts`).
//!
//! Bands are the LABOUR market (patience, goals of their own — see
//! [`crate::bands`]); factions are the DEMAND side: they want things done and
//! they remember whether you did them. Keeping that split clean is deliberate
//! — one opinion number per band was already enough.
//!
//! LAWS ENCODED HERE (each pinned by a test in `tests_politics.rs`):
//! 1. **No beliefs, only a ledger.** Nothing in this module may touch minds
//!    state. Structurally enforced: the module imports nothing from the
//!    campaign's mind seams, and a [`FactionState`] is four counters. The one
//!    place politics becomes personal is `petitions::home_feeling`, which
//!    writes a THOUGHT through the same injection channel every other mood
//!    source uses.
//! 2. **Powers are GENERATED from the landmarks the country actually rolled**
//!    — a coast founding gets merchants, an inland one may not.
//! 3. **Holds are PERSISTED, never re-derived**, so changing the claim rule
//!    cannot silently redraw an old save's map.
//! 4. **The wild power authors raids** (`raids::SpawnOpts::faction_id`), which
//!    turns a raid from weather into a chapter of one running quarrel.
//! 5. **Hostility is a LATCH** (`hostile_since`) with exactly TWO doors out:
//!    paying a `tribute` petition, or beating their raid. Never time.

use serde::{Deserialize, Serialize};

use crate::rng::{rng_float, rng_int, rng_pick, RngState};
use crate::worldgen::{LandmarkKind, WorldState};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactionKind {
    Crown,
    Church,
    Mercantile,
    Wild,
}

impl FactionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FactionKind::Crown => "crown",
            FactionKind::Church => "church",
            FactionKind::Mercantile => "mercantile",
            FactionKind::Wild => "wild",
        }
    }
}

/// Which landmark kinds each power grows out of, and how it reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactionKindSpec {
    pub kind: FactionKind,
    /// Seats it will take, best first.
    pub seats: &'static [LandmarkKind],
    /// What it wants of you, for generated prose.
    pub wants: &'static str,
    pub color: u32,
    /// A wild power never petitions — it only threatens.
    pub petitions: bool,
}

/// `factions.ts:47-64`, verbatim. THE ORDER IS THE DRAW ORDER: generation
/// walks crown → church → mercantile → wild.
pub const FACTION_KINDS: [FactionKindSpec; 4] = [
    FactionKindSpec {
        kind: FactionKind::Crown,
        seats: &[LandmarkKind::City, LandmarkKind::Village, LandmarkKind::Crossroads],
        wants: "levies, tolls, and to be seen to be obeyed",
        color: 0x9c_5a_4a,
        petitions: true,
    },
    FactionKindSpec {
        kind: FactionKind::Church,
        seats: &[LandmarkKind::Abbey],
        wants: "tithes, and the appearance of virtue",
        color: 0xd8_cf_ae,
        petitions: true,
    },
    FactionKindSpec {
        kind: FactionKind::Mercantile,
        seats: &[LandmarkKind::Port, LandmarkKind::Mill, LandmarkKind::Ford],
        wants: "open roads and escorted carts",
        color: 0x6f_8f_6a,
        petitions: true,
    },
    FactionKindSpec {
        kind: FactionKind::Wild,
        seats: &[
            LandmarkKind::Ruin,
            LandmarkKind::Barrow,
            LandmarkKind::Fen,
            LandmarkKind::Pass,
        ],
        wants: "nothing you can give it",
        color: 0x4a_3a_52,
        petitions: false,
    },
];

pub fn faction_kind_spec(kind: FactionKind) -> &'static FactionKindSpec {
    FACTION_KINDS
        .iter()
        .find(|s| s.kind == kind)
        .expect("every FactionKind has a spec")
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Faction {
    pub id: String,
    pub name: String,
    pub kind: FactionKind,
    /// The landmark they sit at.
    pub seat_landmark_id: String,
    pub color: u32,
    /// Landmarks claimed at founding — PERSISTED, never re-derived.
    pub hold_ids: Vec<String>,
}

/// Per-power ledger. STANDING ITSELF IS NOT HERE (it lives in
/// [`crate::petitions::StandingLedger`] and drifts at read) so this stays a
/// pure record of WHAT HAPPENED, with no clock of its own to keep in sync.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactionState {
    pub served: i64,
    pub refused: i64,
    /// A LATCH: set when they turn on you, cleared by an EVENT (paying their
    /// tribute, or beating their raid), never merely by time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostile_since: Option<i64>,
    pub last_petition_day: i64,
}

impl Default for FactionState {
    fn default() -> Self {
        // `factions.ts:198` — `{served: 0, refused: 0, lastPetitionDay: -99}`.
        FactionState { served: 0, refused: 0, hostile_since: None, last_petition_day: -99 }
    }
}

/// `g.factionStates` — the lazily-created per-power record map, as an
/// insertion-ordered association list (the JS `Record`'s own shape).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct FactionLedger {
    pub entries: Vec<(String, FactionState)>,
}

impl FactionLedger {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A read that never creates (the TS `factionState` creates on read; the
    /// borrow discipline here splits it into [`Self::get`] / [`Self::at`]).
    pub fn get(&self, id: &str) -> FactionState {
        self.entries
            .iter()
            .find(|(k, _)| k == id)
            .map_or_else(FactionState::default, |(_, s)| s.clone())
    }

    /// `factions.ts factionState` — create-on-read, insertion ordered.
    pub fn at(&mut self, id: &str) -> &mut FactionState {
        if !self.entries.iter().any(|(k, _)| k == id) {
            self.entries.push((id.to_string(), FactionState::default()));
        }
        self.entries
            .iter_mut()
            .find(|(k, _)| k == id)
            .map(|(_, s)| s)
            .expect("inserted above")
    }

    /// THE LATCH, read side. Never decays with time — see the module laws.
    pub fn is_hostile(&self, id: &str) -> bool {
        self.get(id).hostile_since.is_some()
    }

    /// DOOR ONE and DOOR TWO both land here (tribute paid / their raid
    /// beaten). There is deliberately no third caller and no time-based path.
    pub fn clear_hostility(&mut self, id: &str) -> bool {
        let e = self.at(id);
        e.hostile_since.take().is_some()
    }
}

// ---------- generation ----------

const CROWN_NAMES: [&str; 3] = ["the Warden of ", "the Reeve of ", "the House of "];
const CHURCH_NAMES: [&str; 3] = ["the Chapter of ", "the Order of ", "the Almoners of "];
const TRADE_NAMES: [&str; 3] = ["the Factors of ", "the Carters of ", "the Guild of "];
const WILD_NAMES: [&str; 3] = ["the Things in ", "what walks in ", "the Old Blood of "];

fn name_book(kind: FactionKind) -> &'static [&'static str; 3] {
    match kind {
        FactionKind::Crown => &CROWN_NAMES,
        FactionKind::Church => &CHURCH_NAMES,
        FactionKind::Mercantile => &TRADE_NAMES,
        FactionKind::Wild => &WILD_NAMES,
    }
}

fn name_for(rng: &mut RngState, kind: FactionKind, seat_name: &str) -> String {
    format!("{}{seat_name}", rng_pick(rng, name_book(kind)))
}

/// Roll the country's powers. Called at the END of the founding draw order
/// (name → cast → world → goals → colony → FACTIONS), per the frozen-order
/// law: appending is safe, inserting would desynchronise every old seed.
///
/// DRAW SHAPE (must not move): per seated power, seat pick then name pick;
/// then the single-power fallback's two picks if it fires; then the two
/// unconditional draws the TS spends so the stream advances identically
/// whether or not the tie-breaks branched.
pub fn roll_factions(rng: &mut RngState, world: &WorldState) -> Vec<Faction> {
    let landmarks = &world.landmarks;
    if landmarks.is_empty() {
        return Vec::new();
    }
    let mut taken: Vec<String> = Vec::new();
    let mut out: Vec<Faction> = Vec::new();

    // One power per kind the country can actually seat — the world decides
    // which politics exist.
    for spec in &FACTION_KINDS {
        let options: Vec<usize> = landmarks
            .iter()
            .enumerate()
            .filter(|(_, l)| spec.seats.contains(&l.kind) && !taken.contains(&l.id))
            .map(|(i, _)| i)
            .collect();
        if options.is_empty() {
            continue;
        }
        let seat = &landmarks[*rng_pick(rng, &options)];
        taken.push(seat.id.clone());
        let name = name_for(rng, spec.kind, &seat.name);
        out.push(Faction {
            id: format!("f_{}_{}", spec.kind.as_str(), seat.id),
            name,
            kind: spec.kind,
            seat_landmark_id: seat.id.clone(),
            color: spec.color,
            hold_ids: vec![seat.id.clone()],
        });
    }

    // A country with only one power has no politics; give it a wild rival
    // rather than leaving the map a monologue.
    if out.len() == 1 {
        let spare: Vec<usize> = landmarks
            .iter()
            .enumerate()
            .filter(|(_, l)| !taken.contains(&l.id))
            .map(|(i, _)| i)
            .collect();
        if !spare.is_empty() {
            let seat = &landmarks[*rng_pick(rng, &spare)];
            taken.push(seat.id.clone());
            let name = name_for(rng, FactionKind::Wild, &seat.name);
            out.push(Faction {
                id: format!("f_wild_{}", seat.id),
                name,
                kind: FactionKind::Wild,
                seat_landmark_id: seat.id.clone(),
                color: faction_kind_spec(FactionKind::Wild).color,
                hold_ids: vec![seat.id.clone()],
            });
        }
    }

    // Everything else falls to whoever sits nearest. Claims are PERSISTED
    // from here on; the rule may change without redrawing old saves.
    for l in landmarks {
        if taken.contains(&l.id) {
            continue;
        }
        let mut best: Option<usize> = None;
        let mut best_d = f64::INFINITY;
        for (i, f) in out.iter().enumerate() {
            let Some(seat) = world.landmark_by_id(&f.seat_landmark_id) else { continue };
            let d = ((seat.x - l.x).powi(2) + (seat.z - l.z).powi(2)).sqrt();
            if d < best_d {
                best_d = d;
                best = Some(i);
            }
        }
        if let Some(i) = best {
            out[i].hold_ids.push(l.id.clone());
        }
    }

    // Consume a draw either way so the stream advances identically whether or
    // not the tie-breaks above happened to branch (`factions.ts:166-167`).
    let _ = rng_int(rng, 0, 3);
    let _ = rng_float(rng);
    out
}

// ---------- reads ----------

pub fn faction_by_id<'a>(factions: &'a [Faction], id: &str) -> Option<&'a Faction> {
    factions.iter().find(|f| f.id == id)
}

/// Who holds this ground. The transitive step that makes politics personal:
/// `world.homes` binds a companion to a landmark, and this binds that landmark
/// to a power.
pub fn faction_holding<'a>(factions: &'a [Faction], landmark_id: &str) -> Option<&'a Faction> {
    factions.iter().find(|f| f.hold_ids.iter().any(|h| h == landmark_id))
}

/// The powers that will actually ask you for things.
pub fn petitioners(factions: &[Faction]) -> Vec<&Faction> {
    factions
        .iter()
        .filter(|f| faction_kind_spec(f.kind).petitions)
        .collect()
}

/// The power that authors raids, if this country has one.
pub fn wild_power(factions: &[Faction]) -> Option<&Faction> {
    factions.iter().find(|f| f.kind == FactionKind::Wild)
}

/// What they want, in a sentence — GENERATED from the kind spec (the
/// no-authored-prose law).
pub fn describe_faction(world: &WorldState, f: &Faction) -> String {
    let seat = world.landmark_by_id(&f.seat_landmark_id);
    let holds = f.hold_ids.len();
    format!(
        "{} — seated {}, holding {}. They want {}.",
        f.name,
        seat.map_or_else(
            || "somewhere in the country".to_string(),
            |l| format!("at {}", l.name)
        ),
        if holds == 1 { "one place".to_string() } else { format!("{holds} places") },
        faction_kind_spec(f.kind).wants,
    )
}
