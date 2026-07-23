//! BANDS — the labour market's runtime (`F:\MB\src\guild\goals.ts`).
//!
//! Bands join the guild so long as membership furthers their own goal — and
//! coin can buy a POSTPONEMENT, never a deletion. Pure functions over
//! [`Campaign`].
//!
//! LAWS ENCODED HERE:
//! 1. **Patience drains only while UNSERVED.** Served = fielded in a raid
//!    within 5 days, a standing cause (hope), standing ≥ 25 with whoever holds
//!    their old ground, or — for prosperity bands — simply being fed.
//! 2. **Notice → 2 days → departure**, and the countdown FREEZES the moment
//!    they are served again ([`tick_band_goals`]).
//! 3. **The founders never leave** (`goal_kind: Guild` is exempt BY KIND — the
//!    debt-spiral guild-fall is their exit), and a satisfied goal pins
//!    patience.
//! 4. **Coin postpones, never deletes**: [`sign_price`] scales by goal
//!    PROXIMITY (mid-quest is dear), and a coin signing sets
//!    `deferred_by_coin` with the goal still open.
//! 5. **Nobody rides out from the middle of the road** — a band with people
//!    afield holds its notice ([`crate::campaign::depart_band`]).
//! 6. **`cause_requested` is settable** ([`request_cause`]) — the storyteller's
//!    `cause_raid` trope was unreachable until this landed (S7b flagged it).
//! 7. Epistemic split: everything here writes guild-layer state ONLY (gold,
//!    roster, band state, chronicle, thought injections).

use serde::{Deserialize, Serialize};

use crate::campaign::{chronicle, depart_band, Campaign, ChronicleKind};
use crate::castgen::{FreelancerWant, GoalKind};
use crate::defs::js_round;
use crate::factions::faction_holding;
use crate::petitions::{standing_with, ThoughtInjections};
use crate::scenario::scenario_spec;
use crate::worldgen::{BandGoal, BandStatus};

/// How a band came to sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignHow {
    /// The guild rode for their cause.
    Cause,
    /// A purse persuaded them to set their own matter aside — for now.
    Coin,
    /// The colony simply became what they were looking for.
    Goal,
}

// ---------- reads ----------

pub fn camped_bands(c: &Campaign) -> Vec<String> {
    c.founding
        .cast
        .bands
        .iter()
        .filter(|b| {
            c.band_states
                .iter()
                .any(|(id, live)| id == &b.id && live.state.status == BandStatus::Camped)
        })
        .map(|b| b.id.clone())
        .collect()
}

/// The band's ids in cast order.
pub fn band_members(c: &Campaign, band_id: &str) -> Vec<String> {
    c.founding
        .cast
        .companions
        .iter()
        .filter(|comp| comp.band.as_deref() == Some(band_id))
        .map(|comp| comp.id.clone())
        .collect()
}

/// `parties.ts bandCost` — the sum of the band's hire costs.
pub fn band_cost(c: &Campaign, band_id: &str) -> i64 {
    c.founding
        .cast
        .companions
        .iter()
        .filter(|comp| comp.band.as_deref() == Some(band_id))
        .map(|comp| comp.hire_cost)
        .sum()
}

/// Prosperity, colony-keyed: the generated renown/gold thresholds derive ONE
/// wealth bar (same draws, RE-READ — never re-rolled). `goals.ts needWealth`,
/// with the TS's own `?? 200 / ?? 150` fallbacks for goals that carry neither.
pub fn need_wealth(goal: &BandGoal) -> i64 {
    let (need_renown, need_gold) = match goal {
        BandGoal::Prosperity { need_renown, need_gold } => (*need_renown, *need_gold),
        _ => (150, 200),
    };
    need_gold * 3 + need_renown * 5
}

/// A paymaster worth the name has walls, beds, and a full ledger.
pub fn prosperity_met(goal: &BandGoal, wealth: i64, beds: usize, roster_len: usize) -> bool {
    wealth >= need_wealth(goal) && beds >= roster_len
}

/// The coin that POSTPONES a goal. Mid-quest is dear; desperation is cheap; a
/// band that walked once costs more to bring back (`goals.ts signPrice`).
pub fn sign_price(c: &Campaign, band_id: &str) -> i64 {
    let Some((_, live)) = c.band_states.iter().find(|(id, _)| id == band_id) else {
        return band_cost(c, band_id);
    };
    let mut proximity = 0.3;
    if let BandGoal::Prosperity { need_renown, need_gold } = &live.state.goal {
        let r = (c.renown as f64 / (*need_renown).max(1) as f64).min(1.0);
        let go = (c.gold.max(0) as f64 / (*need_gold).max(1) as f64).min(1.0);
        proximity = (r + go) / 2.0;
    }
    let desperate = live.desperate_until.is_some_and(|d| c.day <= d);
    let local = scenario_spec(c.founding.scenario).sign_discount;
    js_round(
        local
            * band_cost(c, band_id) as f64
            * (0.5 + proximity)
            * if desperate { 0.4 } else { 1.0 }
            * (1.0 + 0.25 * live.state.times_departed as f64),
    )
}

pub fn douceur_cost(c: &Campaign, band_id: &str) -> i64 {
    js_round(sign_price(c, band_id) as f64 * 0.5)
}

/// A freelancer's price, DERIVED from what they want (no new state): the
/// generated want is met → half price (`goals.ts guestPrice`).
pub fn guest_price(c: &Campaign, id: &str) -> i64 {
    let Some(def) = c.founding.cast.companions.iter().find(|comp| comp.id == id) else {
        return 0;
    };
    let met = match &def.want {
        None => false,
        Some(FreelancerWant::Renown { need }) => c.renown >= *need,
        // Ground wants read the dead mission board in the TS too — still false.
        Some(FreelancerWant::Ground { .. }) => false,
        Some(FreelancerWant::Band { band_id }) => c
            .band_states
            .iter()
            .any(|(bid, live)| bid == band_id && live.state.status == BandStatus::Signed),
    };
    js_round(def.hire_cost as f64 * if met { 0.5 } else { 1.0 })
}

/// One-word weather for a signed band (`goals.ts patienceMood`).
pub fn patience_mood(c: &Campaign, band_id: &str) -> &'static str {
    let Some((_, live)) = c.band_states.iter().find(|(id, _)| id == band_id) else {
        return "content";
    };
    let goal_is_guild = matches!(live.state.goal, BandGoal::Guild);
    if goal_is_guild || live.satisfied_day.is_some() {
        return "content";
    }
    if live.state.status == BandStatus::Notice {
        return "a foot out the door";
    }
    match live.state.patience {
        p if p > 70 => "content",
        p if p > 40 => "steady",
        p if p > 20 => "restless",
        _ => "close to riding out",
    }
}

// ---------- mutations ----------

/// Some starts have no labour market at all — every door in is shut HERE at
/// the sim seam rather than by hiding buttons.
pub fn can_recruit(c: &Campaign) -> bool {
    scenario_spec(c.founding.scenario).recruiting
}

/// THE ONE SIGNING MUTATION: the band arrives as its own standing company.
/// Callers own the money (a coin signing deducts before calling).
///
/// Simplification vs the TS, documented: this campaign models the ROSTER, not
/// `g.parties` (S7b's choice) — the band's members join the roster in cast
/// order and the party record is a presentation concern.
pub fn sign_band(c: &mut Campaign, band_id: &str, how: SignHow) -> bool {
    let ids = band_members(c, band_id);
    let Some((_, live)) = c.band_states.iter().find(|(id, _)| id == band_id) else {
        return false;
    };
    if ids.is_empty() || live.state.status == BandStatus::Signed || !can_recruit(c) {
        return false;
    }
    for id in &ids {
        if !c.roster.iter().any(|r| r == id) {
            c.roster.push(id.clone());
        }
    }
    let times_departed = {
        let (_, live) = c.band_states.iter_mut().find(|(id, _)| id == band_id).expect("checked");
        live.state.status = BandStatus::Signed;
        live.notice_day = None;
        live.deferred_by_coin = how == SignHow::Coin;
        let base: i64 = if how == SignHow::Coin { 40 } else { 60 };
        live.state.patience = (base - 10 * live.state.times_departed).max(30);
        live.state.times_departed
    };
    let _ = times_departed;
    let band_name = band_name(c, band_id);
    let why = match how {
        SignHow::Coin => "Coin persuaded them to set their own matters aside — for now.",
        SignHow::Cause => "The guild rode for their cause, and they ride for the guild.",
        SignHow::Goal => "The guild is what they were looking for.",
    };
    chronicle(
        c,
        ChronicleKind::Director,
        format!(
            "{band_name} sign the guild's book — {} names at one stroke. {why}",
            ids.len()
        ),
        ids,
        Some(format!("{band_name} take the guild's colors")),
    );
    true
}

/// A satisfied goal pins patience and pays out ONCE.
pub fn mark_goal_satisfied(c: &mut Campaign, band_id: &str) -> ThoughtInjections {
    let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| id == band_id) else {
        return Vec::new();
    };
    if live.satisfied_day.is_some() {
        return Vec::new();
    }
    let day = c.day;
    let (_, live) = c.band_states.iter_mut().find(|(id, _)| id == band_id).expect("checked");
    live.satisfied_day = Some(day);
    live.state.patience = 100;
    live.deferred_by_coin = false;
    let was_out = live.state.status == BandStatus::Signed || live.state.status == BandStatus::Notice;
    if !was_out {
        return Vec::new();
    }
    let (_, live) = c.band_states.iter_mut().find(|(id, _)| id == band_id).expect("checked");
    live.state.status = BandStatus::Signed;
    live.notice_day = None;
    let ids = band_members(c, band_id);
    let thoughts: ThoughtInjections = ids
        .iter()
        .filter(|id| c.roster.iter().any(|r| r == *id))
        .map(|id| (id.clone(), "goal_served".to_string()))
        .collect();
    let band_name = band_name(c, band_id);
    chronicle(
        c,
        ChronicleKind::Director,
        format!(
            "{band_name}'s own matter is settled at last — they stand easier under \
             the guild's banner."
        ),
        ids,
        Some(format!("{band_name} are repaid")),
    );
    thoughts
}

/// The player takes up a band's cause: the storyteller's next pressure trope
/// becomes THEIR matter (a cause raid with `band_ref`), and hope holds their
/// patience while it stands. Player-confirmed, never automatic.
///
/// THIS IS THE FLAG S7b FLAGGED: nothing set it, so `cause_raid` could never
/// fire. Only deed/debt goals have a cause to take up.
pub fn request_cause(c: &mut Campaign, band_id: &str) -> bool {
    let Some((_, live)) = c.band_states.iter_mut().find(|(id, _)| id == band_id) else {
        return false;
    };
    if live.cause_requested {
        return false;
    }
    if !matches!(live.state.goal, BandGoal::Deed { .. } | BandGoal::Debt { .. }) {
        return false;
    }
    live.cause_requested = true;
    live.state.patience = (live.state.patience + 15).min(100); // hope holds
    let band_name = band_name(c, band_id);
    chronicle(
        c,
        ChronicleKind::Guild,
        format!("The guild takes up {band_name}'s cause — their matter is the colony's now."),
        Vec::new(),
        Some("A cause taken up".to_string()),
    );
    true
}

/// A douceur steadies a restless band without settling anything.
pub fn pay_douceur(c: &mut Campaign, band_id: &str) -> bool {
    let cost = douceur_cost(c, band_id);
    let Some((_, live)) = c.band_states.iter().find(|(id, _)| id == band_id) else {
        return false;
    };
    if (live.state.status != BandStatus::Notice && live.state.patience >= 25) || c.gold < cost {
        return false;
    }
    c.gold -= cost;
    let (_, live) = c.band_states.iter_mut().find(|(id, _)| id == band_id).expect("checked");
    live.state.patience = live.state.patience.max(40);
    live.state.status = BandStatus::Signed;
    live.notice_day = None;
    let band_name = band_name(c, band_id);
    chronicle(
        c,
        ChronicleKind::Guild,
        format!(
            "A purse changes hands and {band_name} unpack — their own matters can \
             wait a little longer."
        ),
        Vec::new(),
        Some("A douceur, quietly paid".to_string()),
    );
    true
}

// ---------- the daily tick ----------

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BandTickInput {
    /// Anyone went to sleep hungry — hunger wears on everyone.
    pub hungry_day: bool,
    pub wealth: i64,
    /// Built beds standing.
    pub beds: usize,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BandTickReport {
    pub signed: Vec<String>,
    pub gave_notice: Vec<String>,
    pub departed: Vec<String>,
    pub thoughts: ThoughtInjections,
}

/// The daily goal tick, colony-keyed. Zero rng draws.
pub fn tick_band_goals(c: &mut Campaign, input: BandTickInput) -> BandTickReport {
    let mut report = BandTickReport::default();
    let band_ids: Vec<String> = c.founding.cast.bands.iter().map(|b| b.id.clone()).collect();

    for band_id in band_ids {
        let Some((_, live)) = c.band_states.iter().find(|(id, _)| id == &band_id) else { continue };
        let status = live.state.status;
        let goal = live.state.goal.clone();
        let want = live.state.want.clone();
        let satisfied = live.satisfied_day.is_some();
        let cause_requested = live.cause_requested;
        let want_met = live.want_met_day.is_some();

        if status == BandStatus::Camped {
            // Prosperity: the colony became what they were looking for.
            if matches!(goal, BandGoal::Prosperity { .. })
                && prosperity_met(&goal, input.wealth, input.beds, c.roster.len())
                && sign_band(c, &band_id, SignHow::Goal)
            {
                report.signed.push(band_id.clone());
                report.thoughts.extend(mark_goal_satisfied(c, &band_id));
                continue;
            }
            // A poach want: the one they watch now lives under the roof.
            if let Some(BandGoal::Poach { target_id }) = &want {
                if c.roster.iter().any(|m| m == target_id) && sign_band(c, &band_id, SignHow::Goal) {
                    report.signed.push(band_id.clone());
                    let who = c.companion_display_name(target_id);
                    let band_name = band_name(c, &band_id);
                    chronicle(
                        c,
                        ChronicleKind::Guild,
                        format!(
                            "{band_name} did not come for the colony — they came for \
                             {who}. Time will tell."
                        ),
                        vec![target_id.clone()],
                        None,
                    );
                }
            }
            continue;
        }

        // Signed or on notice. The founders never leave, and a satisfied goal
        // pins patience.
        if matches!(goal, BandGoal::Guild) || satisfied {
            continue;
        }

        // Late satisfaction for goals met after a coin signing.
        if matches!(goal, BandGoal::Prosperity { .. })
            && prosperity_met(&goal, input.wealth, input.beds, c.roster.len())
        {
            report.thoughts.extend(mark_goal_satisfied(c, &band_id));
            continue;
        }
        if let Some(BandGoal::Poach { target_id }) = &want {
            if !want_met && c.roster.iter().any(|m| m == target_id) {
                let day = c.day;
                if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band_id) {
                    l.want_met_day = Some(day);
                    l.state.patience = (l.state.patience + 20).min(100);
                }
                let who = c.companion_display_name(target_id);
                let band_name = band_name(c, &band_id);
                chronicle(
                    c,
                    ChronicleKind::Guild,
                    format!(
                        "{who} takes {band_name}'s colors — what they came for, come \
                         to pass."
                    ),
                    vec![target_id.clone()],
                    None,
                );
            }
        }

        // SERVED? A standing cause is hope; a recent raid was use; a
        // prosperity band asks only to be fed. And standing with whoever holds
        // their old ground is service in their eyes.
        let fought_lately = c.director.last_raid_day.unwrap_or(-99) >= c.day - 5;
        let home_holder = c
            .founding
            .world
            .home_of(&format!("band:{band_id}"))
            .and_then(|lm| faction_holding(&c.factions, lm))
            .map(|f| f.id.clone());
        let home_standing = home_holder.as_deref().map(|id| standing_with(c, id));
        let home_served = home_standing.is_some_and(|v| v >= 25.0);
        let home_slighted = home_standing.is_some_and(|v| v <= -25.0);
        if home_slighted {
            if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band_id) {
                l.state.patience = (l.state.patience - 3).max(0);
            }
        }
        let prosperity_fed = matches!(goal, BandGoal::Prosperity { .. }) && !input.hungry_day;
        if cause_requested || fought_lately || home_served || prosperity_fed {
            // Freeze the countdown — being served again buys time.
            if status == BandStatus::Notice {
                let day = c.day;
                if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band_id) {
                    l.notice_day = Some(day);
                }
            }
            continue;
        }

        // The unserved goal wears on them; hunger wears on everyone.
        let mut drain = 2;
        if input.hungry_day {
            drain += 6;
        }
        if matches!(want, Some(BandGoal::Poach { .. })) && !want_met {
            drain += 1;
        }
        let (patience, notice_day) = {
            let (_, l) = c.band_states.iter_mut().find(|(id, _)| id == &band_id).expect("found");
            l.state.patience = (l.state.patience - drain).max(0);
            (l.state.patience, l.notice_day)
        };

        if status == BandStatus::Signed && patience <= 0 {
            let day = c.day;
            if let Some((_, l)) = c.band_states.iter_mut().find(|(id, _)| id == &band_id) {
                l.state.status = BandStatus::Notice;
                l.notice_day = Some(day);
            }
            let band_name = band_name(c, &band_id);
            let ids = band_members(c, &band_id);
            chronicle(
                c,
                ChronicleKind::Director,
                format!(
                    "{band_name} give notice: the colony no longer serves what they \
                     came for. Two days and they ride out — unless something changes."
                ),
                ids,
                Some(format!("{band_name} give notice")),
            );
            report.gave_notice.push(band_id.clone());
        } else if status == BandStatus::Notice && c.day - notice_day.unwrap_or(c.day) >= 2 {
            let walked = depart_band(c, &band_id);
            if !walked.is_empty() {
                report.departed.extend(walked);
            }
        }
    }
    report
}

fn band_name(c: &Campaign, band_id: &str) -> String {
    c.founding
        .cast
        .bands
        .iter()
        .find(|b| b.id == band_id)
        .map_or_else(|| band_id.to_string(), |b| b.name.clone())
}

/// The generated goal-kind of a band (used by tests and UI alike).
pub fn goal_kind_of(c: &Campaign, band_id: &str) -> Option<GoalKind> {
    c.founding.cast.bands.iter().find(|b| b.id == band_id).map(|b| b.goal_kind)
}
