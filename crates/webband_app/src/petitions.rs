//! PETITIONS — the answer verb (`F:\MB\src\guild\petitions.ts`).
//!
//! The colony makes capacity; petitions are what spends it. A power asks, and
//! every way of answering costs a different thing:
//!
//! | answer | costs | credit |
//! |---|---|---|
//! | SEND | hands + days + rations (and PAYS on return) | full |
//! | PAY | coin | half |
//! | REFUSE | nothing today | −weight standing, and PLEASES THEIR RIVALS |
//! | LAPSE | nothing — it is the clock | −1.5×weight: silence is contempt |
//!
//! LAWS ENCODED HERE:
//! 1. **Lapse costs 1.5× a refusal** — that is what stops the panel being
//!    dismissible ([`lapse_petitions`], pinned).
//! 2. **Refusing pleases rivals** ([`please_rivals`], +0.25×weight to every
//!    OTHER petitioner) — the one field that turns a standing table into
//!    politics.
//! 3. **Standing DRIFTS AT READ, asymmetrically** ([`StandingLedger::get`]):
//!    0.5/day up from negative, 0.25/day down from positive. No per-day pass,
//!    no stored tier. Hatred fades twice as fast as love — that asymmetry IS
//!    the anti-death-spiral, and it costs a constant.
//! 4. **The choices are a SIM SEAM** ([`petition_choices`]): each option
//!    carries its cost and a `blocked` reason. A UI renders verdicts and never
//!    recomputes affordability (the `canPlace` law).
//! 5. **The deadline sweep runs BEFORE the storyteller** — see the ordering
//!    note in [`crate::campaign::dawn_fold`]; today's expiry can never be
//!    papered over by today's event.

use serde::{Deserialize, Serialize};

use crate::campaign::{chronicle, Campaign, ChronicleKind};
use crate::defs::js_round;
use crate::factions::{faction_by_id, faction_holding, petitioners};

// ---------- the catalog ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PetitionKind {
    Levy,
    Escort,
    Tithe,
    Relief,
    Arbitration,
    Tribute,
}

impl PetitionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            PetitionKind::Levy => "levy",
            PetitionKind::Escort => "escort",
            PetitionKind::Tithe => "tithe",
            PetitionKind::Relief => "relief",
            PetitionKind::Arbitration => "arbitration",
            PetitionKind::Tribute => "tribute",
        }
    }
}

/// What each kind of ask wants, and how it reads. Descriptions are GENERATED
/// from these numbers — the catalog law: no authored per-petition prose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PetitionSpec {
    pub kind: PetitionKind,
    /// Hands asked for, as a share of a healthy roster (rounded, floored at 2).
    pub hands: f64,
    /// Days of provisions per hand.
    pub days: i64,
    /// Gold asked INSTEAD, as a multiple of the hands demanded.
    pub gold_per_hand: i64,
    /// THE WAGE: gold per hand per day, paid when a sent company comes home
    /// having done the thing. Zero = the ask is charity or extraction.
    pub pay_per_hand: i64,
    /// Standing paid on a full answer. Pay earns half; refusing costs this.
    pub weight: f64,
    pub asks: &'static str,
    pub because: &'static str,
}

/// `petitions.ts:48-73`, verbatim. KEY ORDER IS DRAW ORDER: the storyteller
/// picks uniformly among the non-tribute kinds in this order.
pub const PETITIONS: [PetitionSpec; 6] = [
    PetitionSpec {
        kind: PetitionKind::Levy,
        hands: 0.4, days: 3, gold_per_hand: 60, pay_per_hand: 8, weight: 12.0,
        asks: "swords for a levy",
        because: "they are short of men and long of enemies",
    },
    PetitionSpec {
        kind: PetitionKind::Escort,
        hands: 0.3, days: 3, gold_per_hand: 45, pay_per_hand: 7, weight: 9.0,
        asks: "an escort for their carts",
        because: "the road has stopped being safe",
    },
    PetitionSpec {
        kind: PetitionKind::Tithe,
        hands: 0.2, days: 2, gold_per_hand: 70, pay_per_hand: 0, weight: 8.0,
        asks: "a tithe of hands and grain",
        because: "the season has gone against them",
    },
    PetitionSpec {
        kind: PetitionKind::Relief,
        hands: 0.5, days: 4, gold_per_hand: 40, pay_per_hand: 6, weight: 14.0,
        asks: "relief for a place in trouble",
        because: "there is no one else near enough",
    },
    PetitionSpec {
        kind: PetitionKind::Arbitration,
        hands: 0.2, days: 2, gold_per_hand: 55, pay_per_hand: 10, weight: 7.0,
        asks: "someone impartial to judge a quarrel",
        because: "both sides trust an outsider more than each other",
    },
    PetitionSpec {
        kind: PetitionKind::Tribute,
        hands: 0.0, days: 0, gold_per_hand: 0, pay_per_hand: 0, weight: 16.0,
        asks: "tribute, and no argument",
        because: "they consider the matter already settled",
    },
];

pub fn petition_spec(kind: PetitionKind) -> &'static PetitionSpec {
    PETITIONS.iter().find(|s| s.kind == kind).expect("every PetitionKind has a spec")
}

/// How long an ask waits before it becomes an insult.
pub const PETITION_DAYS: i64 = 6;

/// The open ask (`state.ts Petition`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Petition {
    pub id: String,
    pub faction_id: String,
    pub kind: PetitionKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub landmark_id: Option<String>,
    pub need_hands: u32,
    pub need_provisions: u32,
    pub need_gold: i64,
    pub posted_day: i64,
    /// The one good mechanic the mission board had: a deadline.
    pub expires_day: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<PetitionChoiceKind>,
    /// The dispatched company, while it is out answering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub afield_id: Option<String>,
}

/// What a won send pays: the wage × the hands asked × the days of the errand.
pub fn petition_pay(p: &Petition) -> i64 {
    let spec = petition_spec(p.kind);
    spec.pay_per_hand * i64::from(p.need_hands) * spec.days
}

// ---------- standing ----------

/// From negative, back toward zero.
pub const UP_PER_DAY: f64 = 0.5;
/// From positive, back toward zero.
pub const DOWN_PER_DAY: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StandingRec {
    pub value: f64,
    pub day: i64,
}

/// `g.standing` — the per-power opinion, as an insertion-ordered association
/// list (the JS `Record`'s own shape). NOTHING here is decayed by a pass:
/// every read derives the drift from the stamped day.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct StandingLedger {
    pub entries: Vec<(String, StandingRec)>,
}

impl StandingLedger {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// THE ASYMMETRIC LAZY DRIFT (`petitions.ts:104-110`). Being liked must be
    /// maintained; being hated does not compound forever.
    pub fn get(&self, day: i64, faction_id: &str) -> f64 {
        let Some((_, rec)) = self.entries.iter().find(|(k, _)| k == faction_id) else {
            return 0.0;
        };
        let days = (day - rec.day).max(0) as f64;
        if rec.value < 0.0 {
            (rec.value + days * UP_PER_DAY).min(0.0)
        } else {
            (rec.value - days * DOWN_PER_DAY).max(0.0)
        }
    }

    /// Move it, clamped to ±100, re-stamped to today (so the drift restarts).
    pub fn move_by(&mut self, day: i64, faction_id: &str, delta: f64) {
        let now = self.get(day, faction_id);
        let rec = StandingRec { value: (now + delta).clamp(-100.0, 100.0), day };
        match self.entries.iter_mut().find(|(k, _)| k == faction_id) {
            Some((_, r)) => *r = rec,
            None => self.entries.push((faction_id.to_string(), rec)),
        }
    }

    /// How the country as a whole regards the guild — DISPLAY ONLY. Renown is
    /// untouched and remains reputation-at-large.
    pub fn at_large(&self, day: i64) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.entries.iter().map(|(id, _)| self.get(day, id)).sum();
        sum / self.entries.len() as f64
    }
}

/// The tier as a WORD — derived, never stored (the `moodWord` pattern).
pub fn standing_word(v: f64) -> &'static str {
    if v >= 60.0 {
        "sworn friends"
    } else if v >= 25.0 {
        "well thought of"
    } else if v > -25.0 {
        "known to them"
    } else if v > -60.0 {
        "out of favour"
    } else {
        "hated"
    }
}

/// Campaign-level convenience for the ledger read.
pub fn standing_with(c: &Campaign, faction_id: &str) -> f64 {
    c.standing_ledger.get(c.day, faction_id)
}

pub fn move_standing(c: &mut Campaign, faction_id: &str, delta: f64) {
    let day = c.day;
    c.standing_ledger.move_by(day, faction_id, delta);
}

pub fn standing_at_large(c: &Campaign) -> f64 {
    c.standing_ledger.at_large(c.day)
}

// ---------- reading a petition ----------

/// Who is asking, as a name.
pub fn petitioner_name(c: &Campaign, p: &Petition) -> String {
    if let Some(f) = faction_by_id(&c.factions, &p.faction_id) {
        return f.name.clone();
    }
    if let Some(lm) = p.landmark_id.as_deref().and_then(|id| c.founding.world.landmark_by_id(id)) {
        return lm.name.clone();
    }
    "a power in the country".to_string()
}

/// The ask, in prose generated from the spec and the rolled numbers.
pub fn describe_petition(c: &Campaign, p: &Petition) -> String {
    let spec = petition_spec(p.kind);
    let who = petitioner_name(c, p);
    let left = p.expires_day - c.day;
    let when = if left <= 0 {
        "They are done waiting.".to_string()
    } else if left == 1 {
        "They want an answer by tomorrow.".to_string()
    } else {
        format!("They will wait {left} more days.")
    };
    format!("{who} asks for {} — {}. {when}", spec.asks, spec.because)
}

// ---------- the choices (THE SIM SEAM) ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PetitionChoiceKind {
    Send,
    Pay,
    Refuse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PetitionChoice {
    pub choice: PetitionChoiceKind,
    pub label: String,
    pub detail: String,
    /// Set when the choice cannot be taken, and why. A front-end prints this
    /// and disables; it never works out affordability for itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked: Option<String>,
}

/// What the colony can actually put behind an answer today. Assembled from the
/// dawn snapshot by [`crate::campaign::petition_capacity`] — hands are ready,
/// un-afield members; `meals` is `stockOf('meal') + stockOf('grain')`
/// (petitions.ts:182 — a COUNT of units, not nutrition).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PetitionCapacity {
    pub available_hands: usize,
    pub meals: u32,
}

pub fn petition_choices(c: &Campaign, p: &Petition, cap: PetitionCapacity) -> Vec<PetitionChoice> {
    let hands = p.need_hands;
    let prov = p.need_provisions;
    let gold = p.need_gold;
    let have = cap.available_hands as u32;

    let mut out: Vec<PetitionChoice> = Vec::new();

    if hands > 0 {
        let blocked = if have < hands {
            Some(if have == 0 {
                "there is no one here to send".to_string()
            } else {
                format!(
                    "only {have} of {hands} can ride — the rest are away, laid up, \
                     or will not stir"
                )
            })
        } else if cap.meals < prov {
            Some(format!("the larder cannot spare {prov} — there is {}", cap.meals))
        } else {
            None
        };
        let pay = petition_pay(p);
        out.push(PetitionChoice {
            choice: PetitionChoiceKind::Send,
            label: format!("Send {hands}"),
            detail: format!(
                "{hands} hands and {prov} rations, gone some days. Full credit{}",
                if pay > 0 { format!(", and {pay} gold when it is done.") } else { ".".to_string() }
            ),
            blocked,
        });
    }

    out.push(PetitionChoice {
        choice: PetitionChoiceKind::Pay,
        label: format!("Pay {gold} gold"),
        detail: "Coin instead of people. They will remember it, but only by half.".to_string(),
        blocked: if c.gold < gold { Some(format!("the coffer holds {}", c.gold)) } else { None },
    });

    out.push(PetitionChoice {
        choice: PetitionChoiceKind::Refuse,
        label: "Refuse them".to_string(),
        detail: "Costs nothing today. They will not forget, and their rivals will \
                 hear of it."
            .to_string(),
    blocked: None,
    });

    out
}

// ---------- answering ----------

/// Thought injections a politics step produced (`addThought(id, key)` — the
/// fixture applies them, exactly like [`crate::campaign::RaidOutcome`]'s).
pub type ThoughtInjections = Vec<(String, String)>;

/// Standing and renown for a full answer, scaled by the kind's weight.
fn pay_off(c: &mut Campaign, faction_id: &str, kind: PetitionKind, share: f64) -> ThoughtInjections {
    let w = petition_spec(kind).weight;
    move_standing(c, faction_id, w * share);
    c.renown += js_round(w * share * 0.5);
    c.faction_ledger.at(faction_id).served += 1;
    if share >= 1.0 {
        home_feeling(c, faction_id, true)
    } else {
        Vec::new()
    }
}

/// Take a choice. `None` = the sim refused it — the caller never decides that
/// for itself (the blocked-verdict law). `Some(thoughts)` = it landed.
pub fn answer_petition(
    c: &mut Campaign,
    choice: PetitionChoiceKind,
    cap: PetitionCapacity,
) -> Option<ThoughtInjections> {
    let p = c.petition.clone()?;
    if p.chosen.is_some() {
        return None;
    }
    let opt = petition_choices(c, &p, cap).into_iter().find(|o| o.choice == choice)?;
    if opt.blocked.is_some() {
        return None;
    }
    let who = petitioner_name(c, &p);

    match choice {
        PetitionChoiceKind::Pay => {
            c.gold -= p.need_gold;
            // DOOR ONE out of the hostility latch: tribute paid. An EVENT
            // clears it, never the passage of time.
            if p.kind == PetitionKind::Tribute {
                c.faction_ledger.clear_hostility(&p.faction_id);
            }
            let thoughts = pay_off(c, &p.faction_id, p.kind, 0.5);
            chronicle(
                c,
                ChronicleKind::Guild,
                format!(
                    "{who} asked, and the guild sent coin rather than swords. It was \
                     taken, and noted."
                ),
                Vec::new(),
                Some("Paid, not answered".to_string()),
            );
            c.petition = None;
            c.petition_open = false;
            Some(thoughts)
        }
        PetitionChoiceKind::Refuse => {
            move_standing(c, &p.faction_id, -petition_spec(p.kind).weight);
            c.faction_ledger.at(&p.faction_id).refused += 1;
            please_rivals(c, &p);
            let thoughts = home_feeling(c, &p.faction_id, false);
            chronicle(
                c,
                ChronicleKind::Guild,
                format!("{who} asked, and the guild said no. Word of it will travel."),
                Vec::new(),
                Some("The guild refuses".to_string()),
            );
            c.petition = None;
            c.petition_open = false;
            Some(thoughts)
        }
        PetitionChoiceKind::Send => {
            // SEND is completed by the dispatch (`crate::afield`) — the party
            // has to get there and back before anyone is grateful. The
            // petition stays open, FLAGGED, until `resolve_sent_petition`.
            if let Some(open) = c.petition.as_mut() {
                open.chosen = Some(PetitionChoiceKind::Send);
            }
            Some(Vec::new())
        }
    }
}

/// The company sent to answer came home. Truth arrives late, which is the
/// point: sending is the answer that takes time.
pub fn resolve_sent_petition(c: &mut Campaign, won: bool) -> ThoughtInjections {
    let Some(p) = c.petition.clone() else { return Vec::new() };
    if p.chosen != Some(PetitionChoiceKind::Send) {
        return Vec::new();
    }
    let who = petitioner_name(c, &p);
    let thoughts = if won {
        let pay = petition_pay(&p);
        c.gold += pay;
        let t = pay_off(c, &p.faction_id, p.kind, 1.0);
        chronicle(
            c,
            ChronicleKind::Guild,
            format!(
                "The company came back from {who}. What was asked was done{}",
                if pay > 0 {
                    format!(", and {pay} gold came home with them.")
                } else {
                    ".".to_string()
                }
            ),
            Vec::new(),
            Some("The guild answers".to_string()),
        );
        t
    } else {
        let t = pay_off(c, &p.faction_id, p.kind, 0.25);
        chronicle(
            c,
            ChronicleKind::Guild,
            format!(
                "The company came back from {who} with little to show. They were \
                 seen to try, at least."
            ),
            Vec::new(),
            None,
        );
        t
    };
    c.petition = None;
    c.petition_open = false;
    thoughts
}

/// Refusing is CHOOSING A SIDE — that is what turns a standing table into
/// politics. Everyone else who asks things of you is quietly pleased; the wild
/// power, which asks nothing, is indifferent.
fn please_rivals(c: &mut Campaign, p: &Petition) {
    let w = petition_spec(p.kind).weight;
    let others: Vec<String> = petitioners(&c.factions)
        .into_iter()
        .filter(|f| f.id != p.faction_id)
        .map(|f| f.id.clone())
        .collect();
    for id in others {
        move_standing(c, &id, w * 0.25);
    }
}

/// Politics, made personal — the ONE line that reaches the colonists.
///
/// `world.homes` ties a companion to a landmark and the factions hold those
/// landmarks, so refusing the abbey lands on the abbey-born as a THOUGHT. This
/// is the only coupling between the faction ledger and how people feel, and it
/// deliberately goes through the thought channel like every other mood source
/// rather than through the minds layer — FACTIONS HAVE NO BELIEFS.
pub fn home_feeling(c: &Campaign, faction_id: &str, good: bool) -> ThoughtInjections {
    let key = if good { "home_served" } else { "home_refused" };
    let mut out = Vec::new();
    for id in &c.roster {
        let Some(home) = c.founding.world.home_of(id) else { continue };
        if faction_holding(&c.factions, home).map(|f| f.id.as_str()) != Some(faction_id) {
            continue;
        }
        out.push((id.clone(), key.to_string()));
    }
    out
}

/// What a lapse cost, for the caller's report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PetitionLapse {
    pub faction_id: String,
    pub kind: PetitionKind,
    /// The standing paid — always 1.5× a refusal of the same ask.
    pub standing_cost: f64,
    /// The lapse pushed them over the edge into open hostility.
    pub turned_hostile: bool,
    pub thoughts: ThoughtInjections,
}

/// THE DEADLINE SWEEP. Runs in the dawn fold BEFORE the storyteller, so
/// today's expiry can never be papered over by today's event. Zero rng draws.
///
/// A refusal is an answer; silence is contempt — hence the 1.5×.
pub fn lapse_petitions(c: &mut Campaign) -> Option<PetitionLapse> {
    let p = c.petition.clone()?;
    if p.chosen == Some(PetitionChoiceKind::Send) {
        return None;
    }
    if p.chosen.is_some() || c.day < p.expires_day {
        return None;
    }
    let cost = petition_spec(p.kind).weight * 1.5;
    move_standing(c, &p.faction_id, -cost);
    c.faction_ledger.at(&p.faction_id).refused += 1;
    // Ignored too often and they stop asking and start deciding.
    let turned_hostile = if standing_with(c, &p.faction_id) <= -60.0
        && !c.faction_ledger.is_hostile(&p.faction_id)
    {
        let day = c.day;
        c.faction_ledger.at(&p.faction_id).hostile_since = Some(day);
        true
    } else {
        false
    };
    let thoughts = home_feeling(c, &p.faction_id, false);
    let who = petitioner_name(c, &p);
    chronicle(
        c,
        ChronicleKind::Guild,
        format!("{who} waited on the guild for an answer that never came."),
        Vec::new(),
        Some("Kept waiting".to_string()),
    );
    c.petition = None;
    c.petition_open = false;
    Some(PetitionLapse {
        faction_id: p.faction_id,
        kind: p.kind,
        standing_cost: cost,
        turned_hostile,
        thoughts,
    })
}
