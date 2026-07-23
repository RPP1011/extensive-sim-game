//! AFIELD — the guild's reach (`F:\MB\src\colony\afield.ts`), and the thing
//! that makes the colony a SUPPLY BASE rather than a diorama.
//!
//! A dispatched party leaves the roster's working strength, carries rations
//! drawn from the larder, and eats them day by day on the road. Run dry and
//! they turn back hungry.
//!
//! LAWS ENCODED HERE:
//! 1. **THE CAP IS THE ROSTER.** No maximum number of parties: send everyone
//!    and nobody works, nobody defends. The sim states the cost and lets you
//!    pay it.
//! 2. **Progress DERIVES from the day** ([`afield_phase`]) — jump-proof and
//!    idempotent. `sync_afield` may be run on a day that jumped several
//!    mornings and each resolves in order; nothing here is a countdown.
//! 3. **Away members are excluded from colony work and raid musters.** The
//!    campaign-side seam is [`is_afield`] plus [`AfieldReport::away_ids`]: the
//!    fixture flags those bodies `away` and its job masks then budget them
//!    nothing BY CONSTRUCTION — there is no penalty code anywhere.
//! 4. **Rations are real inventory**: dispatch TAKES them out of the larder
//!    (meals first — they travel best), and the homecoming DROPS the leftovers
//!    back. Both are returned as fixture injections, never applied here.
//! 5. **The petition is marked SENT at the dispatch seam**, so the lapse sweep
//!    spares it while they ride and the homecoming gratitude finds it.
//!
//! SIMPLIFIED, documented: road cost uses the TS's own pathless fallback
//! (`owTravelDays` → `max(1, round(hypot/20))`) because the world-map tile A*
//! (`guild/owtiles.ts`) is not ported; `tileWear` is likewise deferred with
//! it. Fighting errands resolve through a caller-supplied closure — the battle
//! itself is fixture business (S5).

use serde::{Deserialize, Serialize};

use crate::campaign::{chronicle, Campaign, ChronicleKind};
use crate::defs::{item_nutrition, StackView};
use crate::markets::{describe_market, market_at};
use crate::petitions::{resolve_sent_petition, PetitionChoiceKind, ThoughtInjections};
use crate::raids::PartyComp;
use crate::rng::rng_int;

/// A day's rations for one person (colony/defs nutrition units).
pub const RATION: f64 = 1.0;
/// Dispatch draws this much slack over the round trip — a party that cuts it
/// exactly to the day is a party that starves on any delay.
pub const RATION_MARGIN: i64 = 2;

/// A peaceful errand with a PURPOSE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Errand {
    /// Ask after the target's market and ride home with the prices.
    Inquire,
    /// Find a warband and file a fresh report (exact when a tracker rides).
    Scout,
}

/// A company on the road. Everything about where they are in the journey
/// derives from `c.day` — nothing here is a countdown a day-jump could
/// desynchronize.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AfieldParty {
    pub id: String,
    pub member_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_landmark_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub petition_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threat_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_ref: Option<String>,
    /// What they must beat when they arrive (absent = a peaceful errand).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comp: Option<PartyComp>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elite_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errand: Option<Errand>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scout_threat_ref: Option<String>,
    pub depart_day: i64,
    /// One-way road cost in days, priced at dispatch.
    pub travel_days: i64,
    /// Nutrition-days of rations carried; burned one per member per day.
    pub provisions: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<AfieldOutcome>,
    /// Homecoming hp fractions from a fighting errand — their wounds ride home
    /// with them, and the WARD is applied at the gate (the TS split:
    /// `resolveErrand` sets hp, `comeHome` beds the badly hurt).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub member_hp: Vec<(String, f64)>,
    /// Days the party has gone without food on the road.
    #[serde(default)]
    pub hungry_days: i64,
    /// Where they are bound (world coords), for the chart.
    pub x: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AfieldOutcome {
    Won,
    Lost,
    TurnedBack,
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AfieldPhase {
    Outbound,
    Arrived,
    Homeward,
    Home,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DispatchOpts {
    pub target_landmark_id: Option<String>,
    pub x: Option<f64>,
    pub z: Option<f64>,
    pub comp: Option<PartyComp>,
    pub elite_name: Option<String>,
    pub petition_id: Option<String>,
    pub threat_ref: Option<String>,
    pub band_ref: Option<String>,
    pub errand: Option<Errand>,
    pub scout_threat_ref: Option<String>,
    /// A tracker rides — the perk's colony-era re-key: a scout's count is
    /// EXACT when one does.
    pub tracker_aboard: bool,
}

/// What the colony can put behind a journey. Assembled from the dawn snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DispatchContext<'a> {
    pub stacks: &'a [StackView],
    /// Members the colony reports as unable to ride (laid up, will not stir).
    /// Afield members are excluded here without being listed — [`is_afield`]
    /// is authoritative for those.
    pub unavailable: &'a [String],
}

#[derive(Debug, Clone, PartialEq)]
pub struct DispatchCost {
    pub travel_days: i64,
    pub provisions: u32,
    pub have_provisions: f64,
    /// Set when the sim refuses — the panel prints this and never works out
    /// affordability for itself (the `canPlace` law, applied to dispatch).
    pub blocked: Option<String>,
}

/// Every nutrition-day sitting in the colony's stacks.
pub fn larder_nutrition(stacks: &[StackView]) -> f64 {
    stacks.iter().map(|s| item_nutrition(&s.item) * f64::from(s.count)).sum()
}

/// `owTravelDays`'s pathless fallback: the tile A* is not ported (see the
/// module header), so the road is priced by distance at 20 units a day.
pub fn travel_days_to(x: f64, z: f64) -> i64 {
    (((x * x + z * z).sqrt() / 20.0).round() as i64).max(1)
}

fn target_xz(c: &Campaign, opts: &DispatchOpts) -> (f64, f64) {
    let lm = opts
        .target_landmark_id
        .as_deref()
        .and_then(|id| c.founding.world.landmark_by_id(id));
    (
        opts.x.or_else(|| lm.map(|l| l.x)).unwrap_or(0.0),
        opts.z.or_else(|| lm.map(|l| l.z)).unwrap_or(0.0),
    )
}

/// What a journey would cost before you commit to it.
pub fn dispatch_cost(
    c: &Campaign,
    member_ids: &[String],
    opts: &DispatchOpts,
    ctx: DispatchContext<'_>,
) -> DispatchCost {
    let (x, z) = target_xz(c, opts);
    let travel_days = travel_days_to(x, z);
    let provisions =
        (member_ids.len() as f64 * (travel_days * 2 + RATION_MARGIN) as f64 * RATION) as u32;
    let have_provisions = larder_nutrition(ctx.stacks);
    let blocked = if member_ids.is_empty() {
        Some("nobody to send".to_string())
    } else if member_ids.iter().any(|id| {
        !c.roster.iter().any(|r| r == id)
            || is_afield(c, id)
            || ctx.unavailable.iter().any(|u| u == id)
    }) {
        Some("someone on that list cannot ride".to_string())
    } else if have_provisions < f64::from(provisions) {
        Some(format!(
            "the larder holds {} of {provisions} days of rations",
            have_provisions.floor() as i64
        ))
    } else {
        None
    };
    DispatchCost { travel_days, provisions, have_provisions, blocked }
}

/// Draw `want` nutrition-days from the larder, MEALS FIRST (they travel best),
/// then raw. Returns the packed nutrition and the per-item take (a fixture
/// injection — the host never mutates a sim).
pub fn pack_rations(stacks: &[StackView], want: f64) -> (f64, Vec<(String, u32)>) {
    let mut packed = 0.0;
    let mut take: Vec<(String, u32)> = Vec::new();
    for item in ["meal", "grain", "berries"] {
        let per = item_nutrition(item);
        if per <= 0.0 || packed >= want {
            continue;
        }
        let need = ((want - packed) / per).ceil() as u32;
        let have: u32 = stacks.iter().filter(|s| s.item == item).map(|s| s.count).sum();
        let got = need.min(have);
        if got > 0 {
            take.push((item.to_string(), got));
            packed += f64::from(got) * per;
        }
    }
    (packed, take)
}

/// The dispatch's fixture injections, alongside the party itself.
#[derive(Debug, Clone, PartialEq)]
pub struct Dispatch {
    pub party: AfieldParty,
    /// FIXTURE INJECTION: remove these from the stacks (the packed rations).
    pub take_items: Vec<(String, u32)>,
    /// FIXTURE INJECTION: flag these bodies `away` — the job masks then budget
    /// them nothing and raids muster without them, both by construction.
    pub away_ids: Vec<String>,
}

/// THE dispatch mutation. Returns `None` if the sim refuses.
pub fn dispatch_party(
    c: &mut Campaign,
    member_ids: &[String],
    opts: DispatchOpts,
    ctx: DispatchContext<'_>,
) -> Option<Dispatch> {
    let cost = dispatch_cost(c, member_ids, &opts, ctx);
    if cost.blocked.is_some() {
        return None;
    }
    let (x, z) = target_xz(c, &opts);
    let (provisions, take_items) = pack_rations(ctx.stacks, f64::from(cost.provisions));

    let party = AfieldParty {
        id: format!("af{}", rng_int(&mut c.rng, 1, 1 << 30)),
        member_ids: member_ids.to_vec(),
        target_landmark_id: opts.target_landmark_id.clone(),
        petition_id: opts.petition_id.clone(),
        threat_ref: opts.threat_ref.clone(),
        band_ref: opts.band_ref.clone(),
        comp: opts.comp.clone(),
        elite_name: opts.elite_name.clone(),
        errand: opts.errand,
        scout_threat_ref: opts.scout_threat_ref.clone(),
        depart_day: c.day,
        travel_days: cost.travel_days,
        provisions,
        outcome: None,
        member_hp: Vec::new(),
        hungry_days: 0,
        x,
        z,
    };
    c.afield.push(party.clone());
    c.afield_tracker.push((party.id.clone(), opts.tracker_aboard));

    // A company dispatched to answer an ask IS the answer: mark the petition
    // SENT here, at the sim seam, so the lapse sweep spares it while they ride
    // and the homecoming gratitude finds it.
    if let (Some(pid), Some(open)) = (opts.petition_id.as_deref(), c.petition.as_mut()) {
        if open.id == pid {
            open.chosen = Some(PetitionChoiceKind::Send);
            open.afield_id = Some(party.id.clone());
        }
    }

    let lm_name = opts
        .target_landmark_id
        .as_deref()
        .and_then(|id| c.founding.world.landmark_by_id(id))
        .map(|l| l.name.clone());
    let names = c.member_names(member_ids);
    chronicle(
        c,
        ChronicleKind::Guild,
        format!(
            "{names} rode out{} with {} days of rations.",
            lm_name.map_or(String::new(), |n| format!(" for {n}")),
            provisions.round() as i64
        ),
        member_ids.to_vec(),
        None,
    );
    Some(Dispatch { party, take_items, away_ids: member_ids.to_vec() })
}

/// Where a party is today, DERIVED from the calendar.
pub fn afield_phase(day: i64, p: &AfieldParty) -> AfieldPhase {
    let elapsed = day - p.depart_day;
    if elapsed < p.travel_days {
        AfieldPhase::Outbound
    } else if elapsed == p.travel_days {
        AfieldPhase::Arrived
    } else if elapsed < p.travel_days * 2 {
        AfieldPhase::Homeward
    } else {
        AfieldPhase::Home
    }
}

/// Is this companion currently on the road? `depart_band` and the starvation
/// exodus must never delete someone mid-journey — removing them would strand
/// an [`AfieldParty::member_ids`].
pub fn is_afield(c: &Campaign, id: &str) -> bool {
    c.afield.iter().any(|p| p.member_ids.iter().any(|m| m == id))
}

/// What the caller reports back for a fighting errand (the battle is fixture
/// business; this is its result).
#[derive(Debug, Clone, PartialEq)]
pub struct ErrandFight {
    pub victory: bool,
    pub gold: i64,
    /// Fielded member id → homecoming hp fraction.
    pub member_hp: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct AfieldReport {
    /// FIXTURE INJECTION: `addThought(id, key)`.
    pub thoughts: ThoughtInjections,
    /// FIXTURE INJECTION: SET hp fractions (homecoming wounds — absolute).
    pub set_hp: Vec<(String, f64)>,
    /// FIXTURE INJECTION: SUBTRACT this much hp fraction, floored at 0.1 (a
    /// hungry day on the road; `hpFrac = max(0.1, hpFrac - 0.1)`).
    pub hp_drain: Vec<(String, f64)>,
    /// FIXTURE INJECTION: `injuryDays = max(injuryDays, days)`.
    pub injuries: Vec<(String, i64)>,
    /// FIXTURE INJECTION: drop this many meals back on the pile by the door.
    pub rations_returned: u32,
    /// FIXTURE INJECTION: these bodies are home — clear their `away` flag.
    pub home_ids: Vec<String>,
    /// Members currently on the road AFTER this sweep (the away set).
    pub away_ids: Vec<String>,
    /// Parties that came home this dawn, with how it went.
    pub arrivals: Vec<(String, AfieldOutcome)>,
}

/// Advance every party by the day: eat, arrive and do the errand, come home.
/// Idempotent and jump-safe — progress is read off `c.day`, never counted
/// down. `fight` is called only for a fighting errand on its arrival day.
pub fn sync_afield(
    c: &mut Campaign,
    mut fight: impl FnMut(&Campaign, &AfieldParty) -> ErrandFight,
) -> AfieldReport {
    let mut report = AfieldReport::default();
    let mut still: Vec<AfieldParty> = Vec::new();
    let parties = std::mem::take(&mut c.afield);

    for mut p in parties {
        // ---- rations ----
        let eaten = p.member_ids.len() as f64 * RATION;
        if p.provisions >= eaten {
            p.provisions -= eaten;
        } else {
            p.provisions = 0.0;
            p.hungry_days += 1;
            for id in &p.member_ids {
                report.thoughts.push((id.clone(), "hungry_road".to_string()));
                report.hp_drain.push((id.clone(), 0.1));
            }
            // Two days empty on the road and they turn for home, errand undone.
            if p.hungry_days >= 2 && p.outcome.is_none() {
                p.outcome = Some(AfieldOutcome::TurnedBack);
                let names = c.member_names(&p.member_ids);
                chronicle(
                    c,
                    ChronicleKind::Guild,
                    format!(
                        "{names}'s company ran out of food on the road and turned for \
                         home with the errand undone."
                    ),
                    p.member_ids.clone(),
                    Some("A company turns back".to_string()),
                );
            }
        }

        let phase = afield_phase(c.day, &p);
        if phase == AfieldPhase::Arrived && p.outcome.is_none() {
            resolve_errand(c, &mut p, &mut fight, &mut report);
        }
        let turned_home = p.outcome == Some(AfieldOutcome::TurnedBack)
            && phase == AfieldPhase::Homeward
            && c.day - p.depart_day >= p.travel_days;
        if phase == AfieldPhase::Home || turned_home {
            come_home(c, &p, &mut report);
            continue;
        }
        still.push(p);
    }
    c.afield = still;
    let live: Vec<String> = c.afield.iter().map(|p| p.id.clone()).collect();
    c.afield_tracker.retain(|(id, _)| live.contains(id));
    report.away_ids = c.afield.iter().flat_map(|p| p.member_ids.clone()).collect();
    report
}

/// The errand itself, the morning they arrive.
fn resolve_errand(
    c: &mut Campaign,
    p: &mut AfieldParty,
    fight: &mut impl FnMut(&Campaign, &AfieldParty) -> ErrandFight,
    report: &mut AfieldReport,
) {
    // A SCOUT finds the warband's trail and files a FRESH report at their TRUE
    // position — exact numbers when a tracker rides, a soldier's estimate
    // otherwise (guild/knowledge.ts).
    if p.errand == Some(Errand::Scout) {
        if let Some(threat_id) = p.scout_threat_ref.clone() {
            let found = c.threats.iter().find(|t| t.id == threat_id).map(|t| {
                let idx = t.step_idx.min(t.path.len().saturating_sub(1));
                (t.name.clone(), t.comp.clone(), t.path.get(idx).copied().unwrap_or((0.0, 0.0)))
            });
            if let Some((name, comp, at)) = found {
                let exact =
                    c.afield_tracker.iter().find(|(id, _)| *id == p.id).is_some_and(|(_, t)| *t);
                crate::director::file_threat_report(c, &threat_id, &comp, at, exact);
                let word = crate::knowledge::threat_strength_word(c, &threat_id);
                chronicle(
                    c,
                    ChronicleKind::Guild,
                    format!("The scouts found {name}'s trail: {word}."),
                    p.member_ids.clone(),
                    Some("Word from the scouts".to_string()),
                );
            } else {
                chronicle(
                    c,
                    ChronicleKind::Guild,
                    "The scouts came up empty — whatever walked there walks no more."
                        .to_string(),
                    p.member_ids.clone(),
                    Some("Word from the scouts".to_string()),
                );
            }
        }
        p.outcome = Some(AfieldOutcome::Done);
        return;
    }
    // A MARKET INQUIRY: the errand IS the learning — intel home with them.
    if p.errand == Some(Errand::Inquire) {
        if let Some(lm_id) = p.target_landmark_id.clone() {
            let lm_name = c.founding.world.landmark_by_id(&lm_id).map(|l| l.name.clone());
            if let Some(name) = lm_name {
                if market_at(c, &lm_id, crate::markets::market_epoch(c.day)).is_some() {
                    let day = c.day;
                    match c.market_intel.iter_mut().find(|(id, _)| *id == lm_id) {
                        Some((_, d)) => *d = day,
                        None => c.market_intel.push((lm_id.clone(), day)),
                    }
                    let desc = describe_market(c, &lm_id, crate::markets::market_epoch(c.day));
                    chronicle(
                        c,
                        ChronicleKind::Guild,
                        format!("At {name} the company asked after the market: {desc}."),
                        p.member_ids.clone(),
                        Some("Word of the market".to_string()),
                    );
                } else {
                    chronicle(
                        c,
                        ChronicleKind::Guild,
                        format!(
                            "{name} keeps no market worth the name — the road was the answer."
                        ),
                        p.member_ids.clone(),
                        Some("Word of the market".to_string()),
                    );
                }
            }
        }
        p.outcome = Some(AfieldOutcome::Done);
        return;
    }
    if p.comp.is_none() {
        p.outcome = Some(AfieldOutcome::Done);
        return;
    }
    if p.member_ids.iter().all(|id| !c.roster.iter().any(|r| r == id)) {
        p.outcome = Some(AfieldOutcome::Lost);
        return;
    }
    let result = fight(c, p);
    p.outcome = Some(if result.victory { AfieldOutcome::Won } else { AfieldOutcome::Lost });
    // Their wounds ride home with them (the homecoming beds the badly hurt).
    for (id, frac) in &result.member_hp {
        report.set_hp.push((id.clone(), *frac));
    }
    p.member_hp = result.member_hp.clone();
    if result.victory {
        c.gold += result.gold;
    }
    let where_ = p
        .target_landmark_id
        .as_deref()
        .and_then(|id| c.founding.world.landmark_by_id(id))
        .map(|l| l.name.clone());
    let text = match (&where_, result.victory) {
        (Some(n), true) => format!(
            "At {n}, the company carried the day{}.",
            p.elite_name.as_deref().map_or(String::new(), |e| format!(" — {e} broken"))
        ),
        (None, true) => format!(
            "The company carried the day{}.",
            p.elite_name.as_deref().map_or(String::new(), |e| format!(" — {e} broken"))
        ),
        (Some(n), false) => format!("At {n}, the company was beaten back."),
        (None, false) => "The company was beaten back.".to_string(),
    };
    chronicle(
        c,
        ChronicleKind::Guild,
        text,
        p.member_ids.clone(),
        Some(if result.victory { "The road answered" } else { "A hard road" }.to_string()),
    );
}

/// Back through the gates: rank restored, wounds to the ward, leftovers to the
/// larder.
fn come_home(c: &mut Campaign, p: &AfieldParty, report: &mut AfieldReport) {
    for id in &p.member_ids {
        if !c.roster.iter().any(|r| r == id) {
            continue;
        }
        report.home_ids.push(id.clone());
        if let Some((_, frac)) = p.member_hp.iter().find(|(m, _)| m == id) {
            if *frac < 0.4 {
                report.injuries.push((id.clone(), 2));
            }
        }
        match p.outcome {
            Some(AfieldOutcome::Won) => report.thoughts.push((id.clone(), "victory".to_string())),
            Some(AfieldOutcome::Lost) => report.thoughts.push((id.clone(), "defeat".to_string())),
            _ => {}
        }
    }
    // Unspent rations go back on the pile by the door.
    let left = p.provisions.floor().max(0.0) as u32;
    report.rations_returned += left;
    let names = c.member_names(&p.member_ids);
    chronicle(
        c,
        ChronicleKind::Guild,
        format!(
            "{names} came home{}.",
            match p.outcome {
                Some(AfieldOutcome::Won) => " with the work done",
                Some(AfieldOutcome::TurnedBack) => " empty-handed",
                _ => "",
            }
        ),
        p.member_ids.clone(),
        None,
    );
    // If they went to answer an ask, the power that asked finds out NOW — the
    // gratitude arrives with the party, not with the promise.
    if p.petition_id.is_some() {
        let won = !matches!(p.outcome, Some(AfieldOutcome::TurnedBack) | Some(AfieldOutcome::Lost));
        report.thoughts.extend(resolve_sent_petition(c, won));
    }
    report.arrivals.push((p.id.clone(), p.outcome.unwrap_or(AfieldOutcome::Done)));
}
