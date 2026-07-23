//! THE FOG OF THE COUNTRY (`F:\MB\src\guild\knowledge.ts`) — what the GUILD
//! believes about the world, as opposed to what is true.
//!
//! This is the epistemic split's third layer: the engine reads truth,
//! companions hold social beliefs (minds — people only, never places), and the
//! guild holds WORLD-knowledge that AGES.
//!
//! LAWS ENCODED HERE:
//! 1. **State is a REPORT, never a parallel fact table**: a
//!    [`crate::director::ThreatReport`] is a day, a place, a counted strength
//!    and whether the count was exact. Market knowledge is just the DAY
//!    someone last looked ([`crate::markets`] derives what they saw).
//! 2. **An unreported warband is INVISIBLE, not fuzzy** — a chart draws
//!    reports, never truth; [`describe_threat_report`] says so in words.
//! 3. **Staleness IS uncertainty**: [`threat_strength_word`] prints a rough
//!    count as a range that WIDENS with the report's age.
//! 4. Truth forces itself at the fences — raids are never fogged (nothing in
//!    the raid path consults this module).
//!
//! `file_threat_report` and `sweep_threat_intel` live in
//! [`crate::director`] (S7b ported them there with the warband trope); they
//! are the write and housekeeping halves of this same layer.

use crate::campaign::Campaign;

/// The printed strength: exact counts print plainly; rough ones print as a
/// range that widens with the report's age (`knowledge.ts:54-63`).
pub fn threat_strength_word(c: &Campaign, threat_id: &str) -> String {
    let Some((_, rep)) = c.threat_intel.iter().find(|(id, _)| id == threat_id) else {
        return "strength unknown".to_string();
    };
    let age = (c.day - rep.day).max(0);
    if rep.exact && age <= 1 {
        return format!("{} spears, counted", rep.power);
    }
    let spread = 0.12 + 0.05 * age as f64 + if rep.exact { 0.0 } else { 0.1 };
    let lo = crate::defs::js_round(rep.power as f64 * (1.0 - spread)).max(1);
    let hi = crate::defs::js_round(rep.power as f64 * (1.0 + spread));
    format!("{lo} spears, or {hi}")
}

fn nearest_landmark_name(c: &Campaign, x: f64, z: f64) -> Option<String> {
    let mut best: Option<(&str, f64)> = None;
    for lm in &c.founding.world.landmarks {
        let d = ((lm.x - x).powi(2) + (lm.z - z).powi(2)).sqrt();
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((lm.name.as_str(), d));
        }
    }
    best.filter(|(_, d)| *d < 55.0).map(|(n, _)| n.to_string())
}

/// One sentence of what the guild KNOWS about a warband — from the report,
/// never the truth.
pub fn describe_threat_report(c: &Campaign, threat_id: &str) -> String {
    let name = c
        .threats
        .iter()
        .find(|t| t.id == threat_id)
        .map_or_else(|| threat_id.to_string(), |t| t.name.clone());
    let Some((_, rep)) = c.threat_intel.iter().find(|(id, _)| id == threat_id) else {
        return format!("No word of {name} — nobody has seen them.");
    };
    let age = (c.day - rep.day).max(0);
    let when = match age {
        0 => "seen today".to_string(),
        1 => "seen yesterday".to_string(),
        n => format!("seen {n} days ago"),
    };
    let near = nearest_landmark_name(c, rep.x, rep.z)
        .map_or(String::new(), |n| format!(" near {n}"));
    format!(
        "{name} — {}, {when}{near}, making for the colony.",
        threat_strength_word(c, threat_id)
    )
}

/// Can this warband even be SPOKEN of? An unreported one cannot: the fog is
/// total, not partial (the command layer's `knownThreat` gate).
pub fn known_threat(c: &Campaign, threat_id: &str) -> bool {
    c.threat_intel.iter().any(|(id, _)| id == threat_id)
}
