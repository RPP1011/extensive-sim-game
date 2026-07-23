//! The storyteller — the port of `F:\MB\src\guild\director.ts` (v2, the
//! colony-pivot rebuild). A points budget accrues with the colony's size and
//! fortune; the storyteller commits to a PLAN (a weighted seeded draw) and
//! SAVES TOWARD it instead of firing whatever is cheapest — without that,
//! cheap tropes drain the pool and a raid never accumulates. Mercy gates,
//! relief windows, a global cooldown, one storm at a time.
//!
//! Tropes RESOLVE to typed [`CampaignEvent`]s — pure data the campaign loop
//! later translates into fixture injections. No side effects escape the
//! campaign state; nothing here touches a sim.
//!
//! THE DRAW LAW (director.ts:284-287, kept to the letter): every rng draw
//! happens INSIDE the committed trope's case, never in eligibility — or the
//! counter shifts on days when nothing fires and the whole stream
//! desynchronises.

use serde::{Deserialize, Serialize};

use crate::campaign::{chronicle, Campaign, ChronicleKind};
use crate::defs::js_round;
use crate::factions::{wild_power, FactionKind};
use crate::petitions::{
    describe_petition, Petition, PetitionKind, PETITIONS, PETITION_DAYS,
};
use crate::noise::{hash, id_seed};
use crate::raids::{comp_power, roll_comp, spawn_raid, ActiveRaid, PartyComp, SpawnOpts};
use crate::rng::{rng_float, rng_int, rng_pick, rng_pick_idx};
use crate::scenario::scenario_spec;
use crate::worldgen::{BandGoal, BandStatus};

pub const POINT_CAP: i64 = 120;
pub const COOLDOWN_DAYS: i64 = 3;

// ---------- State ----------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Guest {
    pub id: String,
    pub leaves_day: i64,
}

/// `state.ts DirectorState`, init `{points: 0, lastEventDay: 0, reliefUntil:
/// 0, raidsWon: 0}` (state.ts:650).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectorState {
    pub points: i64,
    pub last_event_day: i64,
    /// No drama before this day (post-tension-peak breather).
    pub relief_until: i64,
    pub guest: Option<Guest>,
    /// The trope the storyteller is SAVING TOWARD: committed on a seeded
    /// draw, fired when the pool affords it.
    pub plan: Option<TropeKind>,
    /// The last day a raid resolved (bands count being fielded as service).
    pub last_raid_day: Option<i64>,
    /// Raids the colony has actually won (the ambition's settle stage reads
    /// this — ambition is a later slice; the counter keeps its meaning now).
    pub raids_won: i64,
}

impl Default for DirectorState {
    fn default() -> Self {
        DirectorState {
            points: 0,
            last_event_day: 0,
            relief_until: 0,
            guest: None,
            plan: None,
            last_raid_day: None,
            raids_won: 0,
        }
    }
}

// ---------- The trope table ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TropeKind {
    Raid,
    Petition,
    Warband,
    Wanderer,
    Feud,
    CauseRaid,
    RefugeeBand,
    Festival,
    Blight,
    Windfall,
    Caravan,
}

/// (kind, cost, weight) in the TS `COSTS` key order — pool construction
/// iterates this order, so it is part of the seeded stream's shape.
/// Costs: director.ts:42-54; weights: director.ts:132-135.
pub const TROPES: [(TropeKind, i64, u32); 11] = [
    (TropeKind::Raid, 60, 3),
    (TropeKind::Petition, 20, 3),
    (TropeKind::Warband, 50, 2),
    (TropeKind::Wanderer, 35, 1),
    (TropeKind::Feud, 30, 3),
    (TropeKind::CauseRaid, 30, 3),
    (TropeKind::RefugeeBand, 30, 1),
    (TropeKind::Festival, 30, 1),
    (TropeKind::Blight, 25, 1),
    (TropeKind::Windfall, 25, 1),
    (TropeKind::Caravan, 25, 2),
];

pub fn trope_cost(kind: TropeKind) -> i64 {
    TROPES.iter().find(|(k, _, _)| *k == kind).expect("trope table").1
}

// ---------- Campaign-side world objects the tropes create ----------

/// A warband on the move: spawned at the country's edge, walking toward the
/// colony. Visible warning; arrival converts to an [`ActiveRaid`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreatState {
    pub id: String,
    pub name: String,
    pub comp: PartyComp,
    pub tier: i64,
    pub target_landmark_id: String,
    /// The walked path (world coords) and how far along it they are.
    pub path: Vec<(f64, f64)>,
    pub step_idx: usize,
    /// Settlements this band has stripped on the march — settlement life is a
    /// later slice; the field keeps the save shape.
    pub pillaged: Vec<String>,
}

/// `knowledge.ts ThreatReport` — the guild's KNOWLEDGE of a warband (a
/// report's day and place, never a fact table; the fog-of-the-country law).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreatReport {
    pub day: i64,
    pub x: f64,
    pub z: f64,
    pub power: i64,
    pub exact: bool,
}

/// A finite-purse trader camped at the rim for two days (`state.ts` caravan).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Caravan {
    pub id: String,
    pub faction_id: Option<String>,
    pub arrived_day: i64,
    pub leaves_day: i64,
    /// The trader's purse — what selling to them is capped by.
    pub gold: i64,
    /// Wares (item, count) in draw order.
    pub goods: Vec<(String, u32)>,
    /// Any business done? Remembered kindly by their power at departure.
    pub traded: bool,
}

/// An open vendetta (minds — later slice feeds these; the parameter seam is
/// live now). `director.ts gatherFeuds`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feud {
    pub a: String,
    pub foe: String,
}

// ---------- Typed events (pure data out) ----------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ItemDrop {
    pub item: String,
    pub count: u32,
    pub q: i32,
    pub r: i32,
}

/// What a fired trope means for the world. The campaign loop translates the
/// fixture-facing variants into sim injections; the campaign-side variants
/// have already mutated campaign state when the event is returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CampaignEvent {
    /// raid/feud/cause_raid tropes and warband arrivals. The raid is also on
    /// `campaign.raid`; the copy here is the announcement.
    RaidIncoming { raid: ActiveRaid },
    /// A warband gathered at the rim (pushed to `campaign.threats`); the
    /// drums filed the first (inexact) report.
    WarbandGathers { threat_id: String, reported_power: i64 },
    /// A freelancer at the fire (recorded on `director.guest`); signing is
    /// the Companies page's business (bands slice).
    WandererArrives { id: String, leaves_day: i64 },
    /// A camped band will sign cheap until `desperate_until` (×0.4 — the
    /// price lives in goals.ts, bands slice).
    RefugeeBand { band_id: String, desperate_until: i64 },
    /// FIXTURE INJECTION: kill these sown growth cells.
    Blight { killed_cells: Vec<String> },
    /// FIXTURE INJECTION: everyone gets the `festival` thought and cheer = 1.
    Festival,
    /// FIXTURE INJECTION: drop the bundles at the rim (haul work). The gold
    /// is already applied campaign-side.
    Windfall { drops: Vec<ItemDrop>, gold: i64 },
    /// A trader camped (recorded on `campaign.caravan`); trade verdicts are
    /// the trade slice's business.
    CaravanArrives { caravan: Caravan },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectorEvent {
    pub kind: TropeKind,
    pub title: String,
    pub text: String,
    pub payload: CampaignEvent,
}

// ---------- Daily inputs ----------

/// The colony-derived numbers the storyteller reads each dawn. Assembled by
/// `campaign::dawn_fold` from the fixture snapshot; tests may hand-build one.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DirectorView {
    /// `colonyWealth` (defs::colony_wealth) — the difficulty axis.
    pub wealth: i64,
    /// Average roster mood (0-100).
    pub avg_mood: f64,
    /// Sown growth-cell keys (blight's target pool, in the fixture's
    /// insertion order — the TS `Object.keys(g.colony.growth)`).
    pub sown_cells: Vec<String>,
    /// Open vendettas against living foes (minds — later slice feeds this).
    pub feuds: Vec<Feud>,
    /// Petition machinery (factions/petitions — later slice; 0 keeps the
    /// petition trope faithfully ineligible, exactly as the TS with no
    /// factions rolled).
    pub petitioner_count: usize,
    pub available_hands: usize,
}

// ---------- Helpers ----------

/// A band whose cause the player took up (`goals.requestCause` — the flag
/// lives on the campaign's live band state). Cast band order.
fn cause_band(c: &Campaign) -> Option<String> {
    for b in &c.founding.cast.bands {
        if let Some((_, live)) = c.band_states.iter().find(|(id, _)| *id == b.id) {
            if live.cause_requested {
                return Some(b.id.clone());
            }
        }
    }
    None
}

/// Freelancers not yet rostered — the wanderer pool. Empty when the scenario
/// closed recruiting (the sacked-home clause joins with settlement life).
fn guest_pool(c: &Campaign) -> Vec<String> {
    if !scenario_spec(c.founding.scenario).recruiting {
        return Vec::new();
    }
    c.founding
        .cast
        .companions
        .iter()
        .filter(|comp| comp.band.is_none() && !c.roster.iter().any(|id| id == &comp.id))
        .map(|comp| comp.id.clone())
        .collect()
}

/// `knowledge.ts fileThreatReport` — rough reports mis-count
/// deterministically (hash by threat id + day, no rng draw).
pub fn file_threat_report(
    c: &mut Campaign,
    threat_id: &str,
    comp: &PartyComp,
    at: (f64, f64),
    exact: bool,
) -> ThreatReport {
    let truth = comp_power(comp);
    let power = if exact {
        js_round(truth)
    } else {
        js_round(truth * (0.75 + hash(f64::from(id_seed(threat_id)), c.day as f64 * 1.7) * 0.6)).max(2)
    };
    let rep = ThreatReport { day: c.day, x: at.0, z: at.1, power, exact };
    match c.threat_intel.iter_mut().find(|(id, _)| id == threat_id) {
        Some((_, r)) => *r = rep.clone(),
        None => c.threat_intel.push((threat_id.to_string(), rep.clone())),
    }
    rep
}

/// The warband's march route. The TS walks `owPath` (world-map tile A*, a
/// later slice); until it lands, the straight route is sampled at the OW
/// tile spacing (~10 world units) so the ~2-steps-per-day pacing still gives
/// days of visible warning — the TS's own fallback for pathless ground is
/// exactly this line with two points. Point count draws NOTHING, so the
/// seeded stream is unaffected by the substitution.
fn synth_path(x: f64, z: f64) -> Vec<(f64, f64)> {
    let dist = (x * x + z * z).sqrt();
    let n = (dist / 10.0).ceil().max(1.0) as usize;
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (x + (0.0 - x) * t, z + (0.0 - z) * t)
        })
        .collect()
}

/// The per-dawn eligibility snapshot — `director.ts`'s `eligible()` gate as
/// plain flags read before any mutation. NO rng in eligibility, ever.
struct EligFlags {
    mercied: bool,
    threats_empty: bool,
    feuds: bool,
    cause: bool,
    guest_open: bool,
    camped_recruitable: bool,
    sown: usize,
    caravan_none: bool,
    /// No mercantile power, or none that has turned on you.
    roads_open: bool,
    petition_open_slot: bool,
}

impl EligFlags {
    fn eligible(&self, kind: TropeKind) -> bool {
        match kind {
            TropeKind::Raid => !self.mercied,
            TropeKind::Warband => !self.mercied && self.threats_empty,
            TropeKind::Feud => self.feuds && !self.mercied,
            TropeKind::CauseRaid => self.cause,
            TropeKind::Wanderer => self.guest_open,
            TropeKind::RefugeeBand => self.camped_recruitable,
            TropeKind::Blight => self.sown >= 8,
            // One trader at a time — AND a mercantile power you have made an
            // enemy of closes the roads: no caravans until tribute or blood
            // clears the latch. No powers = open roads (the TS's `!merc ||`).
            TropeKind::Caravan => self.caravan_none && self.roads_open,
            TropeKind::Petition => self.petition_open_slot,
            TropeKind::Festival | TropeKind::Windfall => true,
        }
    }
}

// ---------- The tick ----------

/// Everything one storyteller tick can produce. `event` is the
/// fixture-facing half; `petition` is campaign-side only and deliberately NOT
/// a [`CampaignEvent`] variant — the fixture has nothing to inject for an ask,
/// and widening that enum would break every existing exhaustive consumer.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DirectorTick {
    pub event: Option<DirectorEvent>,
    pub petition: Option<Petition>,
}

/// One tick per day (a dawn-fold step): accrue, gate, maybe fire ONE trope.
/// The narrow legacy view — byte-compatible with every pre-S11 caller.
pub fn tick_director(c: &mut Campaign, view: &DirectorView) -> Option<DirectorEvent> {
    tick_director_full(c, view).event
}

/// Port of `director.ts tickDirector`, structured identically.
pub fn tick_director_full(c: &mut Campaign, view: &DirectorView) -> DirectorTick {
    let wealth = view.wealth;
    let mood = view.avg_mood;
    let roster_len = c.roster.len();

    // Accrual (director.ts:84-86): 2 + ceil(roster/2) + floor(wealth/800)
    // (+2 in good spirits), capped at 120.
    let accrual = 2
        + (roster_len as i64 + 1) / 2
        + wealth.div_euclid(800)
        + if mood > 60.0 { 2 } else { 0 };
    c.director.points = (c.director.points + accrual).min(POINT_CAP);

    if c.day < c.director.relief_until {
        return DirectorTick::default();
    }
    if c.day - c.director.last_event_day < COOLDOWN_DAYS {
        return DirectorTick::default();
    }
    if c.raid.is_some() {
        return DirectorTick::default(); // one storm at a time
    }

    // MERCY: a colony on its knees is not raided (unless it is rich enough
    // to be worth the trouble anyway).
    let mercied = (roster_len <= 2 || mood < 30.0) && wealth <= 2000;

    // Eligibility flags, all read BEFORE any mutation (the draw law needs
    // no rng here; the borrow checker needs no live closure over `c`).
    let cause = cause_band(c);
    let elig = EligFlags {
        mercied,
        threats_empty: c.threats.is_empty(),
        feuds: !view.feuds.is_empty(),
        cause: cause.is_some(),
        guest_open: c.director.guest.is_none() && !guest_pool(c).is_empty(),
        camped_recruitable: scenario_spec(c.founding.scenario).recruiting
            && c.band_states.iter().any(|(_, live)| live.state.status == BandStatus::Camped),
        sown: view.sown_cells.len(),
        caravan_none: c.caravan.is_none(),
        roads_open: !c
            .factions
            .iter()
            .any(|f| f.kind == FactionKind::Mercantile && c.faction_ledger.is_hostile(&f.id)),
        // Petitions/factions land in a later slice: with no petitioner
        // powers, `petitioner_count` stays 0 and the trope is faithfully
        // ineligible — the TS's own behavior with no factions rolled, not a
        // simplification of the gate.
        // S13 FIX — THE GATE IS THE ASK, NOT A FLAG. `c.petition_open` was
        // set true when a petition opened and never cleared by any of the four
        // answers (`answer_petition` / `resolve_sent_petition` /
        // `lapse_petitions` all clear `c.petition` and leave the flag set), so
        // ONE petition could ever fire per campaign. Nothing had noticed
        // because nothing drove a political campaign for more than one ask.
        // The live ask itself is the honest gate — `petition_open` survives as
        // a save field (shape unchanged) and is kept in step below.
        petition_open_slot: c.petition.is_none()
            && !mercied
            && view.petitioner_count > 0
            && view.available_hands >= 2,
    };
    let eligible = |kind: TropeKind| elig.eligible(kind);

    // The running story jumps the queue: an open feud (or a taken-up cause)
    // fires the moment it is affordable. Otherwise, commit to a weighted
    // seeded draw and SAVE TOWARD it.
    if eligible(TropeKind::CauseRaid) && c.director.points >= trope_cost(TropeKind::CauseRaid) {
        c.director.plan = Some(TropeKind::CauseRaid);
    } else if !view.feuds.is_empty()
        && eligible(TropeKind::Feud)
        && c.director.points >= trope_cost(TropeKind::Feud)
    {
        c.director.plan = Some(TropeKind::Feud);
    }
    if c.director.plan.is_none() || !eligible(c.director.plan.expect("checked")) {
        let mut pool: Vec<TropeKind> = Vec::new();
        for &(kind, _, weight) in &TROPES {
            if !eligible(kind) {
                continue;
            }
            for _ in 0..weight {
                pool.push(kind);
            }
        }
        if pool.is_empty() {
            return DirectorTick::default();
        }
        c.director.plan = Some(*rng_pick(&mut c.rng, &pool));
    }
    let plan = c.director.plan.expect("plan committed above");
    if c.director.points < trope_cost(plan) {
        return DirectorTick::default(); // saving up
    }
    let kind = plan;
    c.director.plan = None;
    c.director.points -= trope_cost(kind);
    c.director.last_event_day = c.day;

    // EVERY draw below happens inside its committed case (the draw law).
    let mut petition: Option<Petition> = None;
    let event = match kind {
        TropeKind::Raid => {
            // WHO is coming: an angry power, else the WILD POWER (which is why
            // the wild power authors raids), else weather. Zero draws — the
            // author is a find, not a pick. With no factions rolled both
            // branches miss and the prose is the TS's own authorless fallback.
            let angry = c
                .factions
                .iter()
                .find(|f| c.faction_ledger.is_hostile(&f.id))
                .map(|f| (f.id.clone(), f.name.clone()));
            let author = angry.clone().or_else(|| {
                wild_power(&c.factions).map(|f| (f.id.clone(), f.name.clone()))
            });
            let raid = spawn_raid(
                &mut c.rng,
                c.day,
                wealth,
                roster_len,
                c.founding.scenario,
                SpawnOpts { faction_id: author.as_ref().map(|(id, _)| id.clone()), ..Default::default() },
            );
            c.raid = Some(raid.clone());
            c.director.relief_until = c.day + 4;
            let warband = raid
                .elite_name
                .as_ref()
                .map_or("a warband".to_string(), |n| format!("{n}'s warband"));
            // A raid with a name behind it is a chapter of something.
            let whose = match &author {
                Some((_, name)) => format!(
                    "{name} {}",
                    if angry.is_some() { "has stopped asking" } else { "has stirred" }
                ),
                None => "Riders were seen circling the valley".to_string(),
            };
            Some(DirectorEvent {
                kind,
                title: "Smoke on the road".to_string(),
                text: format!(
                    "{whose} — {warband} will be at the fences by dawn. One night \
                     to make ready."
                ),
                payload: CampaignEvent::RaidIncoming { raid },
            })
        }
        TropeKind::Feud => {
            let f = rng_pick(&mut c.rng, &view.feuds).clone();
            let raid = spawn_raid(
                &mut c.rng,
                c.day,
                wealth,
                roster_len,
                c.founding.scenario,
                SpawnOpts {
                    elite_name: Some(f.foe.clone()),
                    saga_ref: Some(format!("vendetta:{}:{}", f.a, f.foe)),
                    ..Default::default()
                },
            );
            c.raid = Some(raid.clone());
            c.director.relief_until = c.day + 4;
            Some(DirectorEvent {
                kind,
                title: "An old score rides in".to_string(),
                text: format!(
                    "{} has not forgotten either. Their band will be at the \
                     fences by dawn — the score settles one way or the other.",
                    f.foe
                ),
                payload: CampaignEvent::RaidIncoming { raid },
            })
        }
        TropeKind::CauseRaid => {
            let band_id = cause.expect("eligibility checked");
            let foe_name = c
                .band_states
                .iter()
                .find(|(id, _)| *id == band_id)
                .and_then(|(_, live)| match &live.state.goal {
                    BandGoal::Debt { foe_name, .. } => Some(foe_name.clone()),
                    _ => None,
                });
            let band_name = c
                .founding
                .cast
                .bands
                .iter()
                .find(|b| b.id == band_id)
                .map_or("the band".to_string(), |b| b.name.clone());
            let raid = spawn_raid(
                &mut c.rng,
                c.day,
                wealth,
                roster_len,
                c.founding.scenario,
                SpawnOpts {
                    elite_name: foe_name.clone(),
                    band_ref: Some(band_id),
                    ..Default::default()
                },
            );
            c.raid = Some(raid.clone());
            c.director.relief_until = c.day + 4;
            Some(DirectorEvent {
                kind,
                title: "The cause comes to a head".to_string(),
                text: format!(
                    "{band_name}'s matter has found the colony: {} will be at \
                     the fences by dawn. Win, and they are repaid.",
                    foe_name.as_deref().unwrap_or("what haunts their old ground")
                ),
                payload: CampaignEvent::RaidIncoming { raid },
            })
        }
        TropeKind::Warband => {
            // Gathers at the edge, walks the world map toward the colony —
            // days of visible warning, then a heavier raid.
            let a = rng_float(&mut c.rng) * std::f64::consts::PI * 2.0;
            let x = a.sin() * 85.0;
            let z = a.cos() * 85.0;
            let path = synth_path(x, z);
            let name = format!(
                "{}'s war-band",
                rng_pick(&mut c.rng, &["Kradus", "Sarga", "Drogan", "Mazok", "Irena"])
            );
            let id = format!("th{}", rng_int(&mut c.rng, 1, 1 << 30));
            // The escalation clock rides here too.
            let comp = roll_comp(&mut c.rng, 20.0 + wealth as f64 * 0.02 + c.day as f64 * 0.3);
            let t = ThreatState {
                id: id.clone(),
                name: name.clone(),
                comp: comp.clone(),
                tier: 1,
                target_landmark_id: "hall".to_string(),
                path,
                step_idx: 0,
                pillaged: Vec::new(),
            };
            c.threats.push(t);
            // The drums ARE the first report — rough, hash-mis-counted.
            let rep = file_threat_report(c, &id, &comp, (x, z), false);
            Some(DirectorEvent {
                kind,
                title: "Drums in the hills".to_string(),
                text: format!(
                    "{name} has gathered at the edge of the country and turned \
                     toward the colony — travellers guess {} spears, but nobody \
                     stayed to count. A scout would know more.",
                    rep.power
                ),
                payload: CampaignEvent::WarbandGathers { threat_id: id, reported_power: rep.power },
            })
        }
        TropeKind::Wanderer => {
            let pool = guest_pool(c);
            let id = rng_pick(&mut c.rng, &pool).clone();
            c.director.guest = Some(Guest { id: id.clone(), leaves_day: c.day + 2 });
            let (name, title) = c
                .founding
                .cast
                .companions
                .iter()
                .find(|comp| comp.id == id)
                .map_or((id.clone(), String::new()), |d| (d.name.clone(), d.title.clone()));
            Some(DirectorEvent {
                kind,
                title: "A stranger at the fire".to_string(),
                text: format!(
                    "{name} {title} walked in with the dusk and asked for a \
                     place at the fire. They'll wait two days for an answer — \
                     the Companies page has their terms."
                ),
                payload: CampaignEvent::WandererArrives { id, leaves_day: c.day + 2 },
            })
        }
        TropeKind::RefugeeBand => {
            let camped: Vec<String> = c
                .founding
                .cast
                .bands
                .iter()
                .filter(|b| {
                    c.band_states
                        .iter()
                        .any(|(id, live)| *id == b.id && live.state.status == BandStatus::Camped)
                })
                .map(|b| b.id.clone())
                .collect();
            let band_id = rng_pick(&mut c.rng, &camped).clone();
            let until = c.day + 3;
            if let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| *id == band_id) {
                live.desperate_until = Some(until);
            }
            let band_name = c
                .founding
                .cast
                .bands
                .iter()
                .find(|b| b.id == band_id)
                .map_or("the band".to_string(), |b| b.name.clone());
            Some(DirectorEvent {
                kind,
                title: "A band at a low ebb".to_string(),
                text: format!(
                    "Word comes that {band_name} have had a hard season — for \
                     the next three days they would sign for very little."
                ),
                payload: CampaignEvent::RefugeeBand { band_id, desperate_until: until },
            })
        }
        TropeKind::Blight => {
            // Picks WITHOUT replacement (the TS re-reads Object.keys after
            // each delete); fixture kills the cells on injection.
            let mut sown = view.sown_cells.clone();
            let toll = js_round(sown.len() as f64 * 0.4).max(1) as usize;
            let mut killed = Vec::with_capacity(toll);
            for _ in 0..toll {
                if sown.is_empty() {
                    break;
                }
                let idx = rng_pick_idx(&mut c.rng, sown.len());
                killed.push(sown.remove(idx));
            }
            let n = killed.len();
            Some(DirectorEvent {
                kind,
                title: "Blight in the rows".to_string(),
                text: format!(
                    "A grey rot came through the fields overnight — {n} sown \
                     beds lost. The rest will want tending."
                ),
                payload: CampaignEvent::Blight { killed_cells: killed },
            })
        }
        TropeKind::Festival => Some(DirectorEvent {
            kind,
            title: "A festival day".to_string(),
            text: "A saint's day by the old calendar — work set aside by \
                   afternoon, tables out, and the colony the lighter for it."
                .to_string(),
            payload: CampaignEvent::Festival,
        }),
        TropeKind::Petition => {
            // EVERY draw happens HERE, inside the committed case — never in
            // the eligibility gate, or rng_counter shifts on days when nothing
            // fires and the whole seeded stream desynchronises.
            let asking: Vec<String> = crate::factions::petitioners(&c.factions)
                .into_iter()
                .map(|f| f.id.clone())
                .collect();
            let fid = rng_pick(&mut c.rng, &asking).clone();
            // A power that has TURNED ON YOU does not ask nicely — it demands,
            // and paying is one of the two things that clears the latch.
            let hostile = c.faction_ledger.is_hostile(&fid);
            let pk = if hostile {
                PetitionKind::Tribute
            } else {
                let kinds: Vec<PetitionKind> = PETITIONS
                    .iter()
                    .map(|s| s.kind)
                    .filter(|k| *k != PetitionKind::Tribute)
                    .collect();
                *rng_pick(&mut c.rng, &kinds)
            };
            let f = crate::factions::faction_by_id(&c.factions, &fid).expect("picked above");
            let f_name = f.name.clone();
            let seat = f.seat_landmark_id.clone();
            // `?? rngPick(landmarks)` short-circuits in the TS: the seat
            // always exists, so this branch draws nothing.
            let lm_id = match c.founding.world.landmark_by_id(&seat) {
                Some(lm) => lm.id.clone(),
                None => rng_pick(&mut c.rng, &c.founding.world.landmarks).id.clone(),
            };
            let spec = crate::petitions::petition_spec(pk);
            let roster = c.roster.len() as f64;
            let hands = js_round(roster * spec.hands).max(2) as u32;
            let p = Petition {
                id: format!("p{}_{fid}", c.day),
                faction_id: fid.clone(),
                kind: pk,
                landmark_id: Some(lm_id),
                need_hands: hands,
                need_provisions: hands * spec.days as u32,
                need_gold: i64::from(hands) * spec.gold_per_hand,
                posted_day: c.day,
                expires_day: c.day + PETITION_DAYS,
                chosen: None,
                afield_id: None,
            };
            c.petition = Some(p.clone());
            c.petition_open = true;
            let day = c.day;
            c.faction_ledger.at(&fid).last_petition_day = day;
            let text = describe_petition(c, &p);
            // Campaign-side only: chronicled HERE because it carries no
            // `CampaignEvent` (see [`DirectorTick`]).
            chronicle(
                c,
                ChronicleKind::Director,
                text,
                Vec::new(),
                Some(format!("{f_name} asks")),
            );
            petition = Some(p);
            None
        }
        TropeKind::Caravan => {
            // The wares, their counts, and the purse all roll HERE.
            const WARES: [(&str, i64, i64); 6] = [
                ("grain", 18, 30),
                ("timber", 10, 20),
                ("meal", 6, 10),
                ("herbs", 5, 10),
                ("poultice", 2, 4),
                ("plank", 8, 14),
            ];
            let mut pool: Vec<(&str, i64, i64)> = WARES.to_vec();
            let mut goods: Vec<(String, u32)> = Vec::new();
            for _ in 0..3 {
                let idx = rng_pick_idx(&mut c.rng, pool.len());
                let (item, lo, hi) = pool.remove(idx);
                goods.push((item.to_string(), rng_int(&mut c.rng, lo, hi) as u32));
            }
            let merc = c
                .factions
                .iter()
                .find(|f| f.kind == FactionKind::Mercantile)
                .map(|f| f.id.clone());
            let caravan = Caravan {
                id: format!("c{}", c.day),
                faction_id: merc,
                arrived_day: c.day,
                leaves_day: c.day + 2,
                gold: 60 + rng_int(&mut c.rng, 0, 80),
                goods,
                traded: false,
            };
            c.caravan = Some(caravan.clone());
            Some(DirectorEvent {
                kind,
                title: "A trader at the rim".to_string(),
                text: "A wandering trader has drawn up at the edge of the yard \
                       and will camp two days — coin for your surplus, wares \
                       for your coin."
                    .to_string(),
                payload: CampaignEvent::CaravanArrives { caravan },
            })
        }
        TropeKind::Windfall => {
            // A lord's gratitude: bundles at the rim — even good luck makes
            // haul work. Gold lands campaign-side now; the drops are the
            // fixture's injection.
            c.gold += 40;
            Some(DirectorEvent {
                kind,
                title: "A lord's gratitude".to_string(),
                text: "A wagon left bundles at the north rim before dawn — \
                       meals, timber, and a purse of 40 gold. Someone should \
                       haul it in."
                    .to_string(),
                payload: CampaignEvent::Windfall {
                    drops: vec![
                        ItemDrop { item: "meal".to_string(), count: 6, q: 10, r: -10 },
                        ItemDrop { item: "timber".to_string(), count: 8, q: 11, r: -10 },
                    ],
                    gold: 40,
                },
            })
        }
    };
    DirectorTick { event, petition }
}

/// Warbands walk ~2 steps per day; arrival becomes tomorrow's raid (heavier
/// for the gathering, `tierBump: 1`). An engaged warband — one the current
/// raid references — holds its ground. Port of `director.ts advanceThreats`;
/// the pillage-on-the-march sub-step is a NO-OP until settlement life lands
/// (guild/life.ts — later slice; `pillaged` keeps the save shape ready).
pub fn advance_threats(c: &mut Campaign, wealth: i64) -> Vec<DirectorEvent> {
    let mut events = Vec::new();
    let mut threats = std::mem::take(&mut c.threats);
    for t in &mut threats {
        if c.raid.as_ref().is_some_and(|r| r.threat_ref.as_deref() == Some(t.id.as_str())) {
            continue;
        }
        t.step_idx += 2;
        // (pillage sweep lives here in the TS — settlement life, later slice)
        if t.step_idx >= t.path.len().saturating_sub(1) {
            if c.raid.is_none() {
                let elite = match t.name.strip_suffix("'s war-band") {
                    Some(stem) => format!("{stem} the Grim"),
                    None => t.name.clone(),
                };
                let raid = spawn_raid(
                    &mut c.rng,
                    c.day,
                    wealth,
                    c.roster.len(),
                    c.founding.scenario,
                    SpawnOpts {
                        threat_ref: Some(t.id.clone()),
                        tier_bump: 1,
                        elite_name: Some(elite),
                        ..Default::default()
                    },
                );
                c.raid = Some(raid.clone());
                chronicle(
                    c,
                    ChronicleKind::Director,
                    format!(
                        "{} has reached the valley — they will be at the fences by dawn.",
                        t.name
                    ),
                    Vec::new(),
                    Some("The war-band arrives".to_string()),
                );
                events.push(DirectorEvent {
                    kind: TropeKind::Warband,
                    title: "The war-band arrives".to_string(),
                    text: format!(
                        "{} has reached the valley — they will be at the fences by dawn.",
                        t.name
                    ),
                    payload: CampaignEvent::RaidIncoming { raid },
                });
            }
            // They stand at the gates either way; the raid reference keeps them.
            t.step_idx = t.path.len().saturating_sub(1);
        }
    }
    c.threats = threats;
    events
}

/// `knowledge.ts sweepThreatIntel` — reports of warbands that no longer walk
/// fall away with them.
pub fn sweep_threat_intel(c: &mut Campaign) {
    let alive: Vec<String> = c.threats.iter().map(|t| t.id.clone()).collect();
    c.threat_intel.retain(|(id, _)| alive.iter().any(|a| a == id));
}
