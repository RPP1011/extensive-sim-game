//! The campaign root and the dawn fold — the host-side orchestration of
//! Webband's day turn (`F:\MB\src\colony\colony-day.ts dawnFold`, whose step
//! ORDER is load-bearing) around one persistent [`Campaign`] struct (the
//! `GuildState` v3 analog: S7a's `Founding` + the living campaign fields +
//! the storyteller + the chronicle + the resumable rng).
//!
//! THE SPLIT (documented per step in [`dawn_fold`]): needs, spoilage,
//! healing, regrowth and the work itself live IN THE FIXTURE (S3 owns them
//! in-sim; those steps here are documented pass-throughs reading the
//! fixture's snapshot). Campaign-side money/politics/story — the
//! provisioner, trade income, rent, caravan sweep, starvation-exodus
//! bookkeeping, the storyteller — are implemented here.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::afield::{is_afield, AfieldParty};
use crate::ambition::{
    ambition_progress, current_stage, describe_stage, roll_ambition, stage_met, AmbitionStep,
    AmbitionTruth, Epilogue, EpilogueLine, FoundersAmbition,
};
use crate::bands::{tick_band_goals, BandTickInput, BandTickReport};
use crate::defs::{colony_wealth, food_days, InventorySnapshot, StackView};
use crate::factions::{
    faction_holding, petitioners, roll_factions, Faction, FactionLedger,
};
use crate::petitions::{
    lapse_petitions, standing_with, PetitionCapacity, PetitionLapse, Petition, StandingLedger,
    ThoughtInjections,
};
use crate::director::{
    advance_threats, sweep_threat_intel, tick_director_full, Caravan, DirectorEvent, DirectorState,
    DirectorView, Feud, ThreatReport, ThreatState,
};
use crate::founding::Founding;
use crate::raids::{plunder, victory_renown, ActiveRaid, PlunderOutcome, PlunderSeverity};
use crate::rng::RngState;
use crate::scenario::scenario_spec;
use crate::worldgen::{BandState, BandStatus};

/// Bump on any breaking save-shape change. A mismatched load is an `Err` and
/// the caller founds fresh — the TS discard rule (`loadGuild` returns null
/// for any save that isn't the current shape).
pub const SAVE_VERSION: u32 = 1;

pub const CHRONICLE_ENTRY_CAP: usize = 200;

// ---------- Chronicle ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChronicleKind {
    Raid,
    Director,
    Social,
    Guild,
    Colony,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChronicleEntry {
    pub day: i64,
    pub kind: ChronicleKind,
    pub text: String,
    pub actors: Vec<String>,
    pub headline: Option<String>,
}

pub fn chronicle(
    c: &mut Campaign,
    kind: ChronicleKind,
    text: String,
    actors: Vec<String>,
    headline: Option<String>,
) {
    c.chronicle.push(ChronicleEntry { day: c.day, kind, text, actors, headline });
    if c.chronicle.len() > CHRONICLE_ENTRY_CAP {
        let excess = c.chronicle.len() - CHRONICLE_ENTRY_CAP;
        c.chronicle.drain(..excess);
    }
}

// ---------- Live band state ----------

/// The campaign's LIVING view of a band: the founding's rolled [`BandState`]
/// plus the runtime flags later slices mutate (cause requests, desperation
/// windows, notice clocks). Wrapping — rather than widening S7a's struct —
/// keeps the generation record pure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandLive {
    pub state: BandState,
    /// `goals.requestCause` set this; the cause_raid trope consumes it.
    #[serde(default)]
    pub cause_requested: bool,
    /// The refugee_band trope's cheap-signing window.
    #[serde(default)]
    pub desperate_until: Option<i64>,
    /// A departure held (whole-roster or afield guard) waits here.
    #[serde(default)]
    pub notice_day: Option<i64>,
    /// S11 (bands): the day their PRIMARY goal was settled. A satisfied goal
    /// pins patience forever.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub satisfied_day: Option<i64>,
    /// S11: they signed for coin — the goal is POSTPONED, never deleted.
    #[serde(default, skip_serializing_if = "is_false")]
    pub deferred_by_coin: bool,
    /// S11: the day their secondary poach want came true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub want_met_day: Option<i64>,
}

/// serde helper: skip a `false` flag so adding it left every existing save's
/// JSON byte-identical (the S6 soak digest hashes the serialized campaign).
fn is_false(b: &bool) -> bool {
    !*b
}

// ---------- The campaign root ----------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Campaign {
    pub version: u32,
    /// The immutable generation record (S7a). Live values below start as
    /// copies of its fields and drift from there.
    pub founding: Founding,
    pub day: i64,
    pub gold: i64,
    pub renown: i64,
    /// Standing-at-large placeholder: rent/caravan effects land here until
    /// the factions slice gives every power its own ledger.
    pub standing: i64,
    /// Days of food the dawn provisioner keeps bought per mouth.
    pub provisioning: u32,
    /// Current member ids (campaign membership truth; the fixture mirrors it).
    pub roster: Vec<String>,
    /// Band id → live state, in cast band order.
    pub band_states: Vec<(String, BandLive)>,
    /// The live seeded stream (resumed from the founding's counter).
    pub rng: RngState,
    pub director: DirectorState,
    pub chronicle: Vec<ChronicleEntry>,
    pub raid: Option<ActiveRaid>,
    pub threats: Vec<ThreatState>,
    /// The guild's aging KNOWLEDGE of warbands (fog-of-the-country).
    pub threat_intel: Vec<(String, ThreatReport)>,
    pub caravan: Option<Caravan>,
    /// One ask at a time — petitions land in a later slice; the flag keeps
    /// the director's gate honest now and the save shape stable.
    #[serde(default)]
    pub petition_open: bool,

    // ---------- S11: THE GUILD LAYER (politics) ----------
    //
    // EVERY field below is `skip_serializing_if`-empty, so a campaign founded
    // through `Campaign::new` serializes BYTE-IDENTICALLY to before this slice
    // — which is what keeps the S6 soak's cross-process digest valid.
    /// The politics layer is LIVE (set by [`Campaign::new_political`]). When
    /// false, `dawn_fold` skips every politics step and behaves exactly as it
    /// did before S11.
    #[serde(default, skip_serializing_if = "is_false")]
    pub politics_enabled: bool,
    /// The country's powers, GENERATED at founding. Holds are persisted here,
    /// never re-derived.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub factions: Vec<Faction>,
    /// Per-power record of what happened (served/refused/the hostility latch).
    #[serde(default, skip_serializing_if = "FactionLedger::is_empty")]
    pub faction_ledger: FactionLedger,
    /// Per-power opinion, drifting AT READ (never a per-day pass).
    #[serde(default, skip_serializing_if = "StandingLedger::is_empty")]
    pub standing_ledger: StandingLedger,
    /// ONE OPEN ASK AT A TIME.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub petition: Option<Petition>,
    /// The founders' long arc — rolled last of all, ends the campaign.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ambition: Option<FoundersAmbition>,
    /// Companies on the road.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub afield: Vec<AfieldParty>,
    /// party id → a tracker rides with them (the perk gate for scout reports).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub afield_tracker: Vec<(String, bool)>,
    /// landmark id → the day someone last asked after its market.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub market_intel: Vec<(String, i64)>,
    /// The provisioner's named source (only a KNOWN market may be named).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provision_source: Option<String>,
    /// Landmarks put to the torch — a sacked place keeps no market.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sacked: Vec<String>,
}

impl Campaign {
    pub fn new(founding: Founding) -> Campaign {
        let rng = founding.resume_rng();
        let band_states = founding
            .band_states
            .iter()
            .map(|(id, st)| {
                (
                    id.clone(),
                    BandLive {
                        state: st.clone(),
                        cause_requested: false,
                        desperate_until: None,
                        notice_day: None,
                        satisfied_day: None,
                        deferred_by_coin: false,
                        want_met_day: None,
                    },
                )
            })
            .collect();
        Campaign {
            version: SAVE_VERSION,
            day: founding.day,
            gold: founding.gold,
            renown: founding.renown,
            standing: founding.standing,
            provisioning: founding.provisioning,
            roster: founding.roster.clone(),
            band_states,
            rng,
            director: DirectorState::default(),
            chronicle: Vec::new(),
            raid: None,
            threats: Vec::new(),
            threat_intel: Vec::new(),
            caravan: None,
            petition_open: false,
            politics_enabled: false,
            factions: Vec::new(),
            faction_ledger: FactionLedger::default(),
            standing_ledger: StandingLedger::default(),
            petition: None,
            ambition: None,
            afield: Vec::new(),
            afield_tracker: Vec::new(),
            market_intel: Vec::new(),
            provision_source: None,
            sacked: Vec::new(),
            founding,
        }
    }

    /// THE FULL FOUNDING — `new` plus the politics roll APPENDED to the seeded
    /// stream, in the TS's own order (…→ colony → FACTIONS → AMBITION), then
    /// the scenario's standing offset stamped onto the powers that now exist.
    ///
    /// WHY A SECOND CONSTRUCTOR RATHER THAN A CHANGE TO `new_founding`:
    /// appending to the founding draw order is seed-safe for the FOUNDING
    /// ROLLS (name/cast/world/goals/colony are byte-identical — pinned by
    /// `politics_roll_is_append_only`), but it MOVES the stream position every
    /// post-founding draw resumes from, which would shift every storyteller
    /// draw in an existing campaign. `Campaign::new` therefore stays exactly
    /// as it was, and the politics layer is opt-in per campaign.
    pub fn new_political(founding: Founding) -> Campaign {
        let mut c = Campaign::new(founding);
        roll_politics(&mut c);
        c
    }

    /// A companion's display name (or the raw id if the cast has none).
    pub fn companion_display_name(&self, id: &str) -> String {
        self.companion_name(id)
    }

    /// "A, B and C" is not this game's voice — the TS joins with ", ".
    pub fn member_names(&self, ids: &[String]) -> String {
        ids.iter().map(|id| self.companion_name(id)).collect::<Vec<_>>().join(", ")
    }

    fn companion_name(&self, id: &str) -> String {
        self.founding
            .cast
            .companions
            .iter()
            .find(|c| c.id == id)
            .map_or_else(|| id.to_string(), |c| c.name.clone())
    }

    fn founders_band_id(&self) -> Option<&str> {
        self.founding.cast.bands.iter().find(|b| b.founders).map(|b| b.id.as_str())
    }

    fn is_founder(&self, id: &str) -> bool {
        let Some(founders) = self.founders_band_id() else { return false };
        self.founding
            .cast
            .companions
            .iter()
            .any(|c| c.id == id && c.band.as_deref() == Some(founders))
    }

    fn band_of(&self, id: &str) -> Option<&str> {
        self.founding
            .cast
            .companions
            .iter()
            .find(|c| c.id == id)
            .and_then(|c| c.band.as_deref())
    }
}

/// The politics roll, appended to the founding stream in the TS's own order:
/// FACTIONS, then the scenario's standing offset over the powers that now
/// exist, then the founders' AMBITION (which reads the factions).
///
/// `applyScenario` in the TS moves standing for every petitioner AFTER the
/// factions roll (`scenario.ts:154-156`); S7a could only record the offset as
/// a scalar because no powers existed yet — this is where it lands.
pub fn roll_politics(c: &mut Campaign) {
    c.factions = roll_factions(&mut c.rng, &c.founding.world);
    let offset = crate::scenario::scenario_spec(c.founding.scenario).standing;
    if offset != 0 {
        let ids: Vec<String> =
            petitioners(&c.factions).into_iter().map(|f| f.id.clone()).collect();
        for id in ids {
            crate::petitions::move_standing(c, &id, offset as f64);
        }
    }
    c.ambition = roll_ambition(&mut c.rng, &c.factions, &c.roster, &c.founding.world);
    c.politics_enabled = true;
}

// ---------- The colony snapshot (fixture → host seam) ----------

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MemberView {
    pub id: String,
    /// 0-100 (the fixture's mood field; needs.ts formula).
    pub mood: f64,
    /// Consecutive days with nothing to eat — the exodus clock.
    pub starving_days: i64,
    /// `status === 'ready'` (afield members are false once afield lands).
    pub ready: bool,
}

/// What the fixture reports at dawn. Plain data — the host never reaches
/// into a sim; a test hand-builds one.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ColonySnapshot {
    pub inventory: InventorySnapshot,
    /// Sown growth-cell keys, fixture insertion order (blight's pool).
    pub sown_cells: Vec<String>,
    pub members: Vec<MemberView>,
    /// Nutrition units lost to rot in the fixture's own dawn sweep (S3 owns
    /// spoilage; the host only chronicles it).
    pub rotted_food_units: f64,
    /// Who ate nothing at the fixture's eat-or-starve resolution.
    pub starved_today: Vec<String>,
}

// ---------- Campaign outcome ----------

/// The terminal state of the campaign day loop (S6). The colony ends only
/// when its people are gone — `dawn_fold` reports `fell` the dawn the roster
/// empties, and the driver folds that into this terminal outcome (legacy
/// renown survives a fall, exactly as the TS "Found anew").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CampaignOutcome {
    Ongoing,
    Fell { day: i64 },
    /// S11: the founders' arc closed its last stage. The campaign ends — and
    /// is WON. Structurally the fall's twin, but where the fall SCATTERS the
    /// stories this one SPENDS them ([`Epilogue`]). Legacy renown carries
    /// exactly as it does through a fall.
    Achieved { day: i64 },
}

// ---------- Dawn outcome ----------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProvisionOrder {
    /// FIXTURE INJECTION: drop this many of `item` at the cache ground.
    pub item: String,
    pub count: u32,
    /// Already deducted from the campaign purse.
    pub gold_spent: i64,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DawnOutcome {
    /// The roster emptied — the colony ends (legacy survives).
    pub fell: bool,
    /// A raid arrives at dawn tomorrow (the staging warning).
    pub raid_tomorrow: Option<ActiveRaid>,
    /// The storyteller's event, if one fired this morning.
    pub event: Option<DirectorEvent>,
    /// Warband arrivals converted to raids this dawn (advance_threats).
    pub arrivals: Vec<DirectorEvent>,
    /// The provisioner's purchase, if any (a fixture injection).
    pub provision: Option<ProvisionOrder>,
    /// Members who walked out hungry this dawn.
    pub departed: Vec<String>,
    pub trade_income: i64,
    pub rent_unpaid: bool,
    pub caravan_departed: bool,

    // ---------- S11: the guild layer ----------
    /// A power opened an ask this morning (campaign-side; deliberately NOT a
    /// [`CampaignEvent`] variant — the fixture has nothing to inject for a
    /// petition, and widening that enum would break every existing consumer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub petition_opened: Option<Petition>,
    /// An ask fell due unanswered. Costs 1.5× a refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub petition_lapsed: Option<PetitionLapse>,
    /// What the band clocks did this dawn.
    #[serde(default)]
    pub bands: BandTickReport,
    /// A stage of the founders' arc closed (the last one ends the campaign).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ambition: Option<AmbitionStep>,
    /// The arc's last stage fell — the campaign is over and won.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epilogue: Option<Epilogue>,
    /// FIXTURE INJECTION: `addThought(id, key)` from the politics steps.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thoughts: ThoughtInjections,
}

/// The dawn fold — Webband's LOAD-BEARING ORDER, step by step. Each step
/// states where it lives in the port:
///
///  1. `day += 1`                          — host (here).
///  2. syncAfield                          — DEFERRED (afield slice); no-op.
///  3. advanceThreats                      — host (implemented; the pillage
///     sub-step waits on settlement life).
///  4. spoil sweep                         — FIXTURE (S3 owns spoilage
///     in-sim); host chronicles `snap.rotted_food_units`.
///  5. the provisioner                     — host (implemented: campaign
///     gold, fixture drop injection).
///  6. work fold → progression            — DEFERRED (classes/progression
///     slice); the /120 divisor and the 60-minute floor land with it.
///  7. regrow / growth ticks               — FIXTURE (S3); host prunes only
///     its own threat intel (sweepThreatIntel).
///  8. resolveNeeds (eat/rest/mood)        — FIXTURE (S3); host chronicles
///     `snap.starved_today`.
///  9. rollBreaks                          — DEFERRED (minds slice — breaks
///     read beliefs through the port).
/// 10. healColonists                       — FIXTURE (S3).
/// 11. supper scene + gossip               — DEFERRED (minds/S4).
/// 12. minds day tick (`port.onDayEnd`)    — DEFERRED (minds slice).
/// 13. starvation exodus                   — host (implemented: 3/6
///     hungry-day thresholds, signed bands walk together).
/// 14. guest expiry                        — host (implemented).
/// 15. tickSettlementLife                  — DEFERRED (settlement slice).
/// 16. tradeIncome                         — host (implemented).
/// 17. collectRent                         — host (implemented).
/// 18. caravan departure sweep             — host (implemented).
/// 19. tickBandGoals                       — DEFERRED (bands slice).
/// 20. checkAmbition                       — DEFERRED (ambition was not
///     rolled in S7a; it appends seed-safely later).
/// 21. lapsePetitions                      — DEFERRED (petitions slice);
///     ORDER NOTE: it must run BEFORE the storyteller so today's expiry can
///     never be papered over by today's event.
/// 22. tickDirector                        — host (implemented).
/// 23. work-accumulator reset / clearJobs  — FIXTURE (S3).
/// 24. fall check + tomorrow's raid        — host (implemented).
pub fn dawn_fold(c: &mut Campaign, snap: &ColonySnapshot, feuds: &[Feud]) -> DawnOutcome {
    dawn_fold_inner(c, snap, feuds, None).0
}

/// The dawn fold WITH step 2 (`syncAfield`) live. The errand fight needs a
/// battle runtime the host layer deliberately does not own, so the resolver
/// comes from the caller; everything else about the road — rations, the
/// turn-back rule, arrival errands, homecoming — is this crate's.
pub fn dawn_fold_political(
    c: &mut Campaign,
    snap: &ColonySnapshot,
    feuds: &[Feud],
    fight: &mut dyn FnMut(&Campaign, &AfieldParty) -> crate::afield::ErrandFight,
) -> (DawnOutcome, crate::afield::AfieldReport) {
    dawn_fold_inner(c, snap, feuds, Some(fight))
}

fn dawn_fold_inner(
    c: &mut Campaign,
    snap: &ColonySnapshot,
    feuds: &[Feud],
    afield: Option<&mut dyn FnMut(&Campaign, &AfieldParty) -> crate::afield::ErrandFight>,
) -> (DawnOutcome, crate::afield::AfieldReport) {
    let mut out = DawnOutcome::default();
    let mut road = crate::afield::AfieldReport::default();

    // 1. A new day.
    c.day += 1;

    // 2. syncAfield — the road, resolved off the NEW day (progress derives
    // from the calendar, so this is idempotent and jump-safe).
    if let Some(fight) = afield {
        road = crate::afield::sync_afield(c, fight);
        out.thoughts.extend(road.thoughts.clone());
    }

    // 3. Warbands walk the world map toward the colony.
    let wealth_at_dawn = colony_wealth(c.gold, c.roster.len(), &snap.inventory);
    out.arrivals = advance_threats(c, wealth_at_dawn);

    // 4. Spoilage is FIXTURE truth; the host writes the chronicle line.
    let rotted = snap.rotted_food_units.round() as i64;
    if rotted >= 3 {
        chronicle(
            c,
            ChronicleKind::Colony,
            format!("The damp took the stores — {rotted} of food spoiled."),
            Vec::new(),
            None,
        );
    }

    // 5. THE PROVISIONER (guild economy): where a market exists, the guild
    // buys its bread before anyone goes hungry. Buys only the shortfall
    // against the standing order, only while the purse holds, capped 30/day.
    // Remote provision sources (guild/markets.ts) are a later slice — the
    // home market price is the whole table for now.
    let unit_price = scenario_spec(c.founding.scenario).meal_price;
    let ready_mouths = snap.members.iter().filter(|m| m.ready).count().max(1);
    let provision_target = i64::from(c.provisioning) * ready_mouths as i64;
    if unit_price > 0 && provision_target > 0 {
        let mut bought = 0u32;
        let mut food = food_days(&snap.inventory.stacks);
        while bought < 30 && c.gold >= unit_price && food < provision_target as f64 {
            c.gold -= unit_price;
            food += 1.0; // one meal = one nutrition unit, the TS re-read
            bought += 1;
        }
        if bought > 0 {
            out.provision = Some(ProvisionOrder {
                item: "meal".to_string(),
                count: bought,
                gold_spent: i64::from(bought) * unit_price,
            });
        }
    }

    // 6. Work fold → progression — DEFERRED (classes slice).

    // 7. Regrowth is fixture truth; the host sweeps only its own knowledge.
    sweep_threat_intel(c);

    // 8. resolveNeeds is FIXTURE truth; chronicle the hungry.
    if !snap.starved_today.is_empty() {
        let text = if snap.starved_today.len() == 1 {
            format!("{} went to sleep hungry.", c.companion_name(&snap.starved_today[0]))
        } else {
            format!("{} went to sleep hungry — the larder is bare.", snap.starved_today.len())
        };
        chronicle(c, ChronicleKind::Colony, text, Vec::new(), None);
    }

    // 9. rollBreaks — DEFERRED (minds slice).
    // 10. healColonists — FIXTURE (S3).
    // 11-12. Supper gossip + the minds day tick — DEFERRED (S4/minds).

    // 13. Starvation exodus: hunger empties the colony long before it kills
    // anyone. Non-founders walk after 3 hungry days (a signed band walks
    // TOGETHER); founders hold to 6. Afield members are skipped once the
    // afield slice lands (nobody rides out from the middle of the road).
    let ids: Vec<String> = c.roster.clone();
    for id in ids {
        if !c.roster.iter().any(|r| r == &id) {
            continue; // a band already took them
        }
        // S13: nobody walks out from the middle of the road. Step 13's own
        // comment promised this guard "once the afield slice lands" and it was
        // never added — removing a member mid-journey strands an
        // `AfieldParty::member_ids`, which is exactly what `depart_band`'s
        // twin guard exists to prevent. (`is_afield` is always false in an
        // apolitical campaign, so no pinned behaviour moves.)
        if is_afield(c, &id) {
            continue;
        }
        let starving = snap
            .members
            .iter()
            .find(|m| m.id == id)
            .map_or(0, |m| m.starving_days);
        let is_founder = c.is_founder(&id);
        let bar = if is_founder { 6 } else { 3 };
        if starving < bar {
            continue;
        }
        let band = c.band_of(&id).map(str::to_string);
        let signed_band = band.as_ref().filter(|b| {
            !is_founder
                && c.band_states
                    .iter()
                    .any(|(bid, live)| bid == *b && live.state.status == BandStatus::Signed)
        });
        if let Some(band_id) = signed_band {
            let walked = depart_band(c, &band_id.clone());
            out.departed.extend(walked);
        } else {
            c.roster.retain(|r| r != &id);
            let name = c.companion_name(&id);
            chronicle(
                c,
                ChronicleKind::Colony,
                format!(
                    "{name} went to find bread the colony could not give — gone down the road."
                ),
                vec![id.clone()],
                Some(format!("{name} walks out")),
            );
            out.departed.push(id);
        }
    }

    // 14. A waiting wanderer stops waiting.
    if c.director.guest.as_ref().is_some_and(|g| g.leaves_day < c.day) {
        c.director.guest = None;
    }

    // 15. tickSettlementLife — DEFERRED (settlement slice).

    // 16. The local trade, where there is one. (The sacked-home zero joins
    // with settlement life.)
    out.trade_income = scenario_spec(c.founding.scenario).trade_per_day;
    c.gold += out.trade_income;

    // 17. The landlord's man: rent paid silently while the purse holds;
    // unpaid rent bleeds STANDING (the holder power's ledger once factions
    // land), never a scripted eviction.
    let rent = scenario_spec(c.founding.scenario).rent_per_day;
    if rent > 0 {
        if c.gold >= rent {
            c.gold -= rent;
        } else {
            c.standing -= 3;
            out.rent_unpaid = true;
            chronicle(
                c,
                ChronicleKind::Colony,
                "The landlord's man went away unpaid — word of it will travel.".to_string(),
                Vec::new(),
                Some("Rent unpaid".to_string()),
            );
        }
    }

    // 18. A camped caravan breaks camp on its day — remembered kindly by its
    // power if any business was done (+4 standing; the per-power ledger is
    // the factions slice).
    if c.caravan.as_ref().is_some_and(|cv| c.day >= cv.leaves_day) {
        let cv = c.caravan.take().expect("checked");
        if cv.traded {
            c.standing += 4;
        }
        chronicle(
            c,
            ChronicleKind::Colony,
            if cv.traded {
                "The trader broke camp at first light, the better for the business done."
            } else {
                "The trader broke camp at first light — nothing sold, nothing bought."
            }
            .to_string(),
            Vec::new(),
            Some("The caravan moves on".to_string()),
        );
        out.caravan_departed = true;
    }

    // 19-21. THE GUILD LAYER (S11), only where politics is live. With
    // `politics_enabled == false` every step below is skipped and this fold
    // behaves exactly as it did before the slice.
    if c.politics_enabled {
        // 19. Band goals: patience drains while unserved; notice → 2 days →
        // they ride out (the founders exempt).
        let beds = snap
            .inventory
            .buildings
            .iter()
            .filter(|b| b.built && b.kind == "bed")
            .count();
        out.bands = tick_band_goals(
            c,
            BandTickInput {
                hungry_day: !snap.starved_today.is_empty(),
                wealth: colony_wealth(c.gold, c.roster.len(), &snap.inventory),
                beds,
            },
        );
        out.departed.extend(out.bands.departed.clone());
        out.thoughts.extend(out.bands.thoughts.clone());

        // 20. The founders' arc: stages close IN ORDER, zero draws.
        let truth = AmbitionTruth {
            roster_len: c.roster.len(),
            wealth: colony_wealth(c.gold, c.roster.len(), &snap.inventory),
            raids_won: c.director.raids_won,
        };
        out.ambition = check_ambition(c, truth);
        if matches!(out.ambition, Some(AmbitionStep::Achieved { .. })) {
            out.epilogue = Some(build_epilogue(c, truth.wealth));
        }

        // 21. THE DEADLINE SWEEP — BEFORE the storyteller, so today's expiry
        // can never be papered over by today's event.
        out.petition_lapsed = lapse_petitions(c);
        if let Some(l) = &out.petition_lapsed {
            out.thoughts.extend(l.thoughts.clone());
        }
    }

    // 22. The storyteller.
    let view = DirectorView {
        wealth: colony_wealth(c.gold, c.roster.len(), &snap.inventory),
        avg_mood: if snap.members.is_empty() {
            0.0
        } else {
            snap.members.iter().map(|m| m.mood).sum::<f64>() / snap.members.len() as f64
        },
        sown_cells: snap.sown_cells.clone(),
        feuds: feuds.to_vec(),
        // With no powers rolled these stay 0 and the petition trope is
        // faithfully ineligible — the TS's own behavior, not a stub.
        petitioner_count: if c.politics_enabled { petitioners(&c.factions).len() } else { 0 },
        available_hands: if c.politics_enabled {
            petition_capacity(c, snap).available_hands
        } else {
            0
        },
    };
    let tick = tick_director_full(c, &view);
    out.event = tick.event;
    out.petition_opened = tick.petition;
    if let Some(ev) = &out.event {
        chronicle(c, ChronicleKind::Director, ev.text.clone(), Vec::new(), Some(ev.title.clone()));
    }

    // 23. Work-accumulator reset / clearJobs — FIXTURE (S3).

    // 24. The colony ends only when its people are gone.
    out.fell = c.roster.is_empty();
    out.raid_tomorrow = c
        .raid
        .as_ref()
        .filter(|r| r.arrives_day == c.day + 1)
        .cloned();
    (out, road)
}

// ---------- The guild layer's campaign-side reads ----------

/// Who could actually ride out today: present, whole, and not already on the
/// road (`petitions.ts availableHands`). Injury and break state is FIXTURE
/// truth — it arrives as [`MemberView::ready`].
pub fn petition_capacity(c: &Campaign, snap: &ColonySnapshot) -> PetitionCapacity {
    let available_hands = c
        .roster
        .iter()
        .filter(|id| {
            !is_afield(c, id)
                && snap.members.iter().find(|m| &&m.id == id).is_none_or(|m| m.ready)
        })
        .count();
    // `stockOf('meal') + stockOf('grain')` — a COUNT of units, not nutrition.
    let meals: u32 = snap
        .inventory
        .stacks
        .iter()
        .filter(|s| s.item == "meal" || s.item == "grain")
        .map(|s| s.count)
        .sum();
    PetitionCapacity { available_hands, meals }
}

/// THE AMBITION SWEEP. Closes at most ONE stage per dawn, IN ORDER, with zero
/// rng draws (`ambition.ts checkAmbition`).
pub fn check_ambition(c: &mut Campaign, truth: AmbitionTruth) -> Option<AmbitionStep> {
    let a = c.ambition.as_ref()?;
    if a.achieved_day.is_some() {
        return None;
    }
    let stage = current_stage(a)?.clone();
    let day = c.day;
    let factions = c.factions.clone();
    let standing = |id: &str| c.standing_ledger.get(day, id);
    let hostile = |id: &str| c.faction_ledger.is_hostile(id);
    if !stage_met(&stage, truth, &factions, &standing, &hostile) {
        return None;
    }
    let line = {
        let name = |id: &str| c.companion_display_name(id);
        describe_stage(&c.factions, &c.founding.world, &name, &stage)
    };
    let (title, done, total) = {
        let a = c.ambition.as_mut().expect("checked");
        let s = a.stages.iter_mut().find(|s| s.done_day.is_none()).expect("checked");
        s.done_day = Some(day);
        let (done, total) = ambition_progress(a);
        (a.title.clone(), done, total)
    };
    if done >= total {
        c.ambition.as_mut().expect("checked").achieved_day = Some(day);
        chronicle(
            c,
            ChronicleKind::Guild,
            format!(
                "{title} — it is done. What the founders set out to do, this company \
                 has done, and the country knows it."
            ),
            Vec::new(),
            Some(title.clone()),
        );
        return Some(AmbitionStep::Achieved { title });
    }
    chronicle(
        c,
        ChronicleKind::Guild,
        format!(
            "A step toward {title}: {} — achieved.",
            line.strip_suffix('.').unwrap_or(&line)
        ),
        Vec::new(),
        Some(format!("{done} of {total}")),
    );
    Some(AmbitionStep::Stage { done, total, line })
}

/// The ending, as DATA: every companion still standing walked out on their own
/// record (band, the ground worldgen tied them to, who holds it now, and how
/// that power regards the guild). The presentation layer writes the sentences.
pub fn build_epilogue(c: &Campaign, wealth: i64) -> Epilogue {
    let heir = c.ambition.as_ref().and_then(|a| {
        a.stages
            .iter()
            .find(|s| s.kind == crate::ambition::StageKind::Repaid)
            .and_then(|s| s.member_id.clone())
    });
    let lines = c
        .roster
        .iter()
        .map(|id| {
            let def = c.founding.cast.companions.iter().find(|comp| &comp.id == id);
            let home_id = c.founding.world.home_of(id).map(str::to_string);
            let holder =
                home_id.as_deref().and_then(|lm| faction_holding(&c.factions, lm));
            EpilogueLine {
                member_id: id.clone(),
                name: c.companion_display_name(id),
                band: def.and_then(|d| d.band.clone()).and_then(|b| {
                    c.founding.cast.bands.iter().find(|x| x.id == b).map(|x| x.name.clone())
                }),
                home_landmark: home_id
                    .as_deref()
                    .and_then(|lm| c.founding.world.landmark_by_id(lm))
                    .map(|l| l.name.clone()),
                home_holder: holder.map(|f| f.name.clone()),
                home_standing: holder.map_or(0.0, |f| standing_with(c, &f.id)),
                was_heir: heir.as_deref() == Some(id.as_str()),
            }
        })
        .collect();
    Epilogue {
        title: c.ambition.as_ref().map_or_else(String::new, |a| a.title.clone()),
        day: c.day,
        renown: c.renown,
        gold: c.gold,
        wealth,
        standings: c
            .factions
            .iter()
            .map(|f| (f.name.clone(), standing_with(c, &f.id)))
            .collect(),
        lines,
    }
}

/// `goals.ts departBand`, the campaign-side core: the whole current band
/// leaves together. Guards: never strand the guild (a band that IS the
/// roster holds its notice instead); the afield guard joins with that slice.
/// Returns the ids who walked.
pub fn depart_band(c: &mut Campaign, band_id: &str) -> Vec<String> {
    let leaving: Vec<String> = c
        .roster
        .iter()
        .filter(|id| c.band_of(id) == Some(band_id))
        .cloned()
        .collect();
    if leaving.is_empty() {
        return Vec::new();
    }
    if leaving.len() >= c.roster.len() {
        if let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| id == band_id) {
            live.notice_day = Some(c.day);
        }
        return Vec::new();
    }
    // NOBODY RIDES OUT FROM THE MIDDLE OF THE ROAD: a band with people afield
    // holds its notice until they are home. Removing them mid-journey would
    // strand an `AfieldParty::member_ids`.
    if leaving.iter().any(|id| is_afield(c, id)) {
        if let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| id == band_id) {
            live.notice_day = Some(c.day);
        }
        return Vec::new();
    }
    c.roster.retain(|id| !leaving.contains(id));
    if let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| id == band_id) {
        live.state.status = BandStatus::Camped;
        live.notice_day = None;
        live.desperate_until = None;
        live.state.times_departed += 1;
        live.state.patience = 50;
    }
    let band_name = c
        .founding
        .cast
        .bands
        .iter()
        .find(|b| b.id == band_id)
        .map_or_else(|| band_id.to_string(), |b| b.name.clone());
    chronicle(
        c,
        ChronicleKind::Director,
        format!(
            "{band_name} furl their banner and ride out — back to their own matters. \
             Their fire burns again in the marches."
        ),
        leaving.clone(),
        Some(format!("{band_name} ride out")),
    );
    leaving
}

// ---------- Raid resolution (the campaign-side fold) ----------

/// What the battle (fixture, S5) reports back — plain data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RaidResultView {
    pub victory: bool,
    /// Loot gold on victory (`BattleResult.gold`).
    pub gold_looted: i64,
    /// Fielded member id → homecoming hp fraction.
    pub member_hp: Vec<(String, f64)>,
}

/// Fixture-facing consequences of a resolved raid.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RaidOutcome {
    /// FIXTURE INJECTION: set these members' hp fractions (homecoming).
    pub set_hp: Vec<(String, f64)>,
    /// FIXTURE INJECTION: `injuryDays = max(injuryDays, days)` per member
    /// (hp under 0.4 beds them for 2).
    pub injuries: Vec<(String, i64)>,
    /// FIXTURE INJECTION: `addThought(id, key)` — victory / defeat /
    /// came_home_to_ashes per the TS.
    pub thoughts: Vec<(String, String)>,
    /// Defeat only: what the raiders took and burnt.
    pub plunder: Option<PlunderOutcome>,
}

/// `raids.ts resolveRaid` + `plunder`, campaign side. The battle's own fold
/// (progression/minds/loot into `result`) is fixture/minds business; this
/// lands the raid's consequences on campaign state and returns the fixture
/// injections. Deferred inside (documented): band-cause signing uses a
/// sign-lite (camped → signed; the full signBand/markGoalSatisfied machinery
/// is the bands slice); faction hostility clearing waits on factions.
pub fn resolve_raid(
    c: &mut Campaign,
    raid: &ActiveRaid,
    result: &RaidResultView,
    inv: &InventorySnapshot,
) -> RaidOutcome {
    let mut out = RaidOutcome::default();

    // Homecoming: hp + the infirmary.
    for (id, frac) in &result.member_hp {
        if !c.roster.iter().any(|r| r == id) {
            continue;
        }
        out.set_hp.push((id.clone(), *frac));
        if *frac < 0.4 {
            out.injuries.push((id.clone(), 2));
        }
    }
    c.raid = None;
    c.director.last_raid_day = Some(c.day);

    // Fielded bands were USED — that is service (patience mends); a resolved
    // cause raid clears the request either way.
    let fielded: Vec<&String> = result.member_hp.iter().map(|(id, _)| id).collect();
    for band in &c.founding.cast.bands {
        let members_fielded = c
            .founding
            .cast
            .companions
            .iter()
            .any(|comp| comp.band.as_deref() == Some(band.id.as_str())
                && fielded.iter().any(|f| **f == comp.id));
        if let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| *id == band.id) {
            if raid.band_ref.as_deref() == Some(band.id.as_str()) {
                live.cause_requested = false;
            }
            if live.state.status == BandStatus::Signed && members_fielded {
                live.state.patience = (live.state.patience + 8).min(100);
            }
        }
    }

    if result.victory {
        c.gold += result.gold_looted;
        c.renown += victory_renown(raid.tier);
        if let Some(t) = &raid.threat_ref {
            c.threats.retain(|th| &th.id != t);
        }
        if let Some(band_id) = &raid.band_ref.clone() {
            // Their cause was won: the band SIGNS and the goal is satisfied
            // (S11 — the full `signBand` + `markGoalSatisfied` fold; the
            // pre-S11 sign-lite survives as the no-recruiting fallback).
            let camped = c
                .band_states
                .iter()
                .any(|(id, live)| id == band_id && live.state.status == BandStatus::Camped);
            if camped && !crate::bands::sign_band(c, band_id, crate::bands::SignHow::Cause) {
                if let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| id == band_id) {
                    live.state.status = BandStatus::Signed;
                }
            }
            out.thoughts.extend(crate::bands::mark_goal_satisfied(c, band_id));
        }
        // DOOR TWO out of the hostility latch: beating the raid a power sent.
        // Tribute is the price a rich guild pays; blood is the price a poor
        // one pays — without this second door a broke colony that angered a
        // power could never recover.
        if let Some(fid) = &raid.faction_id {
            c.faction_ledger.clear_hostility(fid);
        }
        c.director.raids_won += 1;
        for id in &c.roster {
            out.thoughts.push((id.clone(), "victory".to_string()));
        }
        let who = raid
            .elite_name
            .as_ref()
            .map_or("The raiders".to_string(), |n| format!("{n}'s warband"));
        chronicle(
            c,
            ChronicleKind::Raid,
            format!(
                "The colony held. {who} broke on the yard and left {} gold behind.",
                result.gold_looted
            ),
            Vec::new(),
            Some("The colony holds".to_string()),
        );
    } else {
        out.plunder = Some(apply_plunder(c, raid, inv, PlunderSeverity::Beaten, &mut out.thoughts));
    }
    out
}

/// The defeat/undefended branch: compute the pure plunder math, then land
/// its campaign-side deltas (renown, storyteller relief, threat removal,
/// thoughts, chronicle). The stack/building mutations stay in the returned
/// outcome — fixture injections.
pub fn apply_plunder(
    c: &mut Campaign,
    raid: &ActiveRaid,
    inv: &InventorySnapshot,
    severity: PlunderSeverity,
    thoughts: &mut Vec<(String, String)>,
) -> PlunderOutcome {
    let p = plunder(inv, c.roster.len(), raid.tier, severity);
    let heavy = severity == PlunderSeverity::Undefended;

    c.renown = (c.renown - p.renown_loss).max(0);
    if let Some(t) = &raid.threat_ref {
        c.threats.retain(|th| &th.id != t);
    }
    c.raid = None;
    c.director.points = (c.director.points - p.director_points_loss).max(0);
    c.director.relief_until = c.day + p.relief_days;
    // Those who were here take the defeat; the road party comes home to
    // ashes (all present until the afield slice lands).
    for id in &c.roster {
        thoughts.push((id.clone(), "defeat".to_string()));
    }
    let burnt = p.burnt_building_ids.len();
    let taken = p.taken_total;
    let text = if heavy {
        format!(
            "There was no one to hold the fences. {} walked in, took {taken} of the stores{}, \
             and were gone before anyone could ride back.",
            raid.elite_name.as_deref().unwrap_or("The raiders"),
            if burnt > 0 { format!(" and burned {burnt} of the outbuildings") } else { String::new() },
        )
    } else {
        format!(
            "The line broke. The raiders took {taken} of the stores{} — but everyone will mend.",
            if burnt > 0 { " and left char where the outbuildings stood" } else { "" },
        )
    };
    chronicle(
        c,
        ChronicleKind::Raid,
        text,
        Vec::new(),
        Some(if heavy { "Sacked".to_string() } else { "Plundered".to_string() }),
    );
    p
}

// ---------- Save / load ----------

#[derive(Debug)]
pub enum CampaignError {
    Io(std::io::Error),
    Format(serde_json::Error),
    /// The TS discard rule: a save from another shape founds fresh.
    Version { found: u32, want: u32 },
}

impl fmt::Display for CampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CampaignError::Io(e) => write!(f, "save io: {e}"),
            CampaignError::Format(e) => write!(f, "save format: {e}"),
            CampaignError::Version { found, want } => {
                write!(f, "save version {found} (this build reads {want}) — found anew")
            }
        }
    }
}

impl std::error::Error for CampaignError {}

impl From<std::io::Error> for CampaignError {
    fn from(e: std::io::Error) -> Self {
        CampaignError::Io(e)
    }
}

impl From<serde_json::Error> for CampaignError {
    fn from(e: serde_json::Error) -> Self {
        CampaignError::Format(e)
    }
}

pub fn save_campaign(c: &Campaign, path: &Path) -> Result<(), CampaignError> {
    let json = serde_json::to_string(c)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_campaign(path: &Path) -> Result<Campaign, CampaignError> {
    let json = std::fs::read_to_string(path)?;
    // Version first, against a shape-tolerant probe — a future save must
    // fail on VERSION, not on whatever field happened to change.
    #[derive(Deserialize)]
    struct VersionProbe {
        #[serde(default)]
        version: u32,
    }
    let probe: VersionProbe = serde_json::from_str(&json)?;
    if probe.version != SAVE_VERSION {
        return Err(CampaignError::Version { found: probe.version, want: SAVE_VERSION });
    }
    let c: Campaign = serde_json::from_str(&json)?;
    Ok(c)
}

/// Convenience for drivers/tests: a snapshot whose stacks are the founding
/// cache (the fixture will replace this with live truth).
pub fn founding_snapshot(c: &Campaign, mood: f64) -> ColonySnapshot {
    ColonySnapshot {
        inventory: InventorySnapshot {
            stacks: c
                .founding
                .stacks
                .iter()
                .enumerate()
                .map(|(i, s)| StackView { id: format!("s{i}"), item: s.item.clone(), count: s.count })
                .collect(),
            buildings: Vec::new(),
        },
        sown_cells: Vec::new(),
        members: c
            .roster
            .iter()
            .map(|id| MemberView { id: id.clone(), mood, starving_days: 0, ready: true })
            .collect(),
        rotted_food_units: 0.0,
        starved_today: Vec::new(),
    }
}
