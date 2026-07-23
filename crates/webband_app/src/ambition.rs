//! THE FOUNDERS' AMBITION — what the colony is FOR
//! (`F:\MB\src\guild\ambition.ts`).
//!
//! The founders' band carried `goalKind: 'guild'` as a placeholder; this is
//! what belongs there. A long arc GENERATED at founding from the country that
//! actually rolled, with stages whose conditions are POLITICAL rather than
//! domestic. Worldgen has already tied these people to this ground, so the
//! endgame is about holding it.
//!
//! LAWS ENCODED HERE:
//! 1. **Everything is generated** — stage text comes from the condition plus
//!    the real names of powers and places ([`describe_stage`]).
//! 2. **Checking is a ZERO-DRAW sweep over ground truth** ([`check_ambition`]);
//!    the only draws are in [`roll_ambition`], at the very end of the founding
//!    order (…→ colony → factions → AMBITION).
//! 3. **Stages close IN ORDER.** An arc is a story, not a checklist: a colony
//!    that stumbles into stage four early is still made to walk through three.
//! 4. **The last stage ENDS THE CAMPAIGN** — [`crate::campaign::CampaignOutcome`]
//!    gains a terminal twin to the fall. Where the fall scatters the stories,
//!    this one SPENDS them: [`Epilogue`] walks each companion out on a line
//!    drawn from their real record (structured data, never authored prose).

use serde::{Deserialize, Serialize};

use crate::factions::{faction_by_id, faction_holding, petitioners, wild_power, Faction};
use crate::rng::{rng_pick, RngState};
use crate::worldgen::WorldState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StageKind {
    Favour,
    Settle,
    Prosper,
    Company,
    Repaid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbitionStage {
    pub kind: StageKind,
    /// The power or place the stage turns on, when it turns on one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faction_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub landmark_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub member_id: Option<String>,
    /// The bar to clear (standing, wealth, or heads, by kind).
    pub target: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_day: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundersAmbition {
    /// Generated from the founders' own ground — the thing they came to do.
    pub title: String,
    pub stages: Vec<AmbitionStage>,
    /// Set when the last stage falls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub achieved_day: Option<i64>,
}

// ---------- generation ----------

const ARCS: [&str; 5] = [
    "A Seat at the Table",
    "The Long Peace",
    "What We Came For",
    "A Name That Holds",
    "The Root and the Rock",
];

/// Roll the founders' long arc. Called at the very END of the founding draw
/// order, AFTER the factions (it reads them). Returns `None` — taking ZERO
/// draws — when the country seated no power that asks for things, exactly as
/// the TS's early return does.
pub fn roll_ambition(
    rng: &mut RngState,
    factions: &[Faction],
    roster: &[String],
    world: &WorldState,
) -> Option<FoundersAmbition> {
    let asking: Vec<String> = petitioners(factions).into_iter().map(|f| f.id.clone()).collect();
    if asking.is_empty() {
        return None;
    }

    let patron = rng_pick(rng, &asking).clone();
    let wild = wild_power(factions).map(|f| f.id.clone());
    let title = (*rng_pick(rng, &ARCS)).to_string();

    // A founder with a home in this country is the human thread through it —
    // the ambition is partly about seeing THEM put right.
    let rooted: Vec<String> =
        roster.iter().filter(|id| world.home_of(id).is_some()).cloned().collect();
    let heir = if rooted.is_empty() { None } else { Some(rng_pick(rng, &rooted).clone()) };

    let mut stages = vec![
        // 1. Stand on your own feet.
        AmbitionStage {
            kind: StageKind::Company,
            faction_id: None,
            landmark_id: None,
            member_id: None,
            target: (roster.len() as i64 + 2).max(5),
            done_day: None,
        },
        // 2. Be worth something to somebody.
        AmbitionStage {
            kind: StageKind::Favour,
            faction_id: Some(patron),
            landmark_id: None,
            member_id: None,
            target: 40,
            done_day: None,
        },
        // 3. Make the colony a place rather than a camp.
        AmbitionStage {
            kind: StageKind::Prosper,
            faction_id: None,
            landmark_id: None,
            member_id: None,
            target: 2600,
            done_day: None,
        },
    ];
    // 4. Deal with whatever the country is afraid of.
    if let Some(w) = wild {
        stages.push(AmbitionStage {
            kind: StageKind::Settle,
            faction_id: Some(w),
            landmark_id: None,
            member_id: None,
            target: 1,
            done_day: None,
        });
    }
    // 5. And see your own put right.
    if let Some(h) = heir {
        let home = world.home_of(&h).map(str::to_string);
        stages.push(AmbitionStage {
            kind: StageKind::Repaid,
            faction_id: None,
            landmark_id: home,
            member_id: Some(h),
            target: 25,
            done_day: None,
        });
    }
    Some(FoundersAmbition { title, stages, achieved_day: None })
}

// ---------- reading ----------

fn faction_name(factions: &[Faction], id: Option<&str>) -> String {
    id.and_then(|i| faction_by_id(factions, i))
        .map_or_else(|| "the powers of this country".to_string(), |f| f.name.clone())
}

/// A stage in words, GENERATED from its condition and the real names.
pub fn describe_stage(
    factions: &[Faction],
    world: &WorldState,
    member_name: &dyn Fn(&str) -> String,
    s: &AmbitionStage,
) -> String {
    match s.kind {
        StageKind::Company => format!("Be {} strong, and keep them.", s.target),
        StageKind::Favour => {
            format!("Stand high with {}.", faction_name(factions, s.faction_id.as_deref()))
        }
        StageKind::Prosper => {
            "Build something worth defending — a holding of real substance.".to_string()
        }
        StageKind::Settle => format!(
            "Settle with {}, whatever that takes.",
            faction_name(factions, s.faction_id.as_deref())
        ),
        StageKind::Repaid => {
            let lm = s
                .landmark_id
                .as_deref()
                .and_then(|id| world.landmark_by_id(id))
                .map_or_else(|| "the place they came from".to_string(), |l| l.name.clone());
            let who = s
                .member_id
                .as_deref()
                .map_or_else(|| "one of our own".to_string(), member_name);
            format!("See {who}'s own ground — {lm} — in friendly hands.")
        }
    }
}

/// Everything the ground-truth sweep needs, gathered by the caller so the
/// check itself is a pure read (`ambition.ts stageMet`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AmbitionTruth {
    pub roster_len: usize,
    pub wealth: i64,
    pub raids_won: i64,
}

/// The stage the guild is working on now (`None` once it is all done).
pub fn current_stage(a: &FoundersAmbition) -> Option<&AmbitionStage> {
    a.stages.iter().find(|s| s.done_day.is_none())
}

pub fn ambition_progress(a: &FoundersAmbition) -> (usize, usize) {
    (a.stages.iter().filter(|s| s.done_day.is_some()).count(), a.stages.len())
}

/// What closing a stage produced — the caller headlines it and, on the last
/// one, ends the campaign.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AmbitionStep {
    /// A stage fell: (done, total) and its generated line.
    Stage { done: usize, total: usize, line: String },
    /// The last one fell. The campaign is over — and won.
    Achieved { title: String },
}

/// One walked-out companion: their real record, structured. The presentation
/// layer turns this into a sentence; the port keeps it as DATA (the
/// no-authored-prose law applies to the ending too).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpilogueLine {
    pub member_id: String,
    pub name: String,
    /// Their band's name, when they came with one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<String>,
    /// The landmark worldgen tied them to, when it tied them to one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home_landmark: Option<String>,
    /// Who holds that ground at the end.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub home_holder: Option<String>,
    /// Standing with that holder at the end (the "put right" measure).
    pub home_standing: f64,
    /// They were named in the arc's `repaid` stage.
    pub was_heir: bool,
}

/// The ending, as data: the arc, the country's final ledger, and every
/// companion still standing walked out on their own record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Epilogue {
    pub title: String,
    pub day: i64,
    pub renown: i64,
    pub gold: i64,
    pub wealth: i64,
    /// (faction name, standing) at the end, in generation order.
    pub standings: Vec<(String, f64)>,
    pub lines: Vec<EpilogueLine>,
}

/// Has this stage's condition come true? A pure read of ground truth.
pub(crate) fn stage_met(
    s: &AmbitionStage,
    truth: AmbitionTruth,
    factions: &[Faction],
    standing: &dyn Fn(&str) -> f64,
    hostile: &dyn Fn(&str) -> bool,
) -> bool {
    match s.kind {
        StageKind::Company => truth.roster_len as i64 >= s.target,
        StageKind::Favour => s
            .faction_id
            .as_deref()
            .is_some_and(|id| standing(id) >= s.target as f64),
        StageKind::Prosper => truth.wealth >= s.target,
        StageKind::Settle => {
            // "Settled" means the wild power has been beaten back at least
            // once and is not currently at your throat — the raids ledger,
            // read honestly. (DOOR TWO out of the latch feeds this.)
            s.faction_id
                .as_deref()
                .is_some_and(|id| truth.raids_won >= s.target && !hostile(id))
        }
        StageKind::Repaid => s
            .landmark_id
            .as_deref()
            .and_then(|lm| faction_holding(factions, lm))
            .is_some_and(|holder| standing(&holder.id) >= s.target as f64),
    }
}
