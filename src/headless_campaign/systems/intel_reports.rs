//! Intelligence reports system — every 1000 ticks (~100s).
//!
//! Generates periodic strategic summaries that rotate through five report types:
//! faction power rankings, threat assessments, economic outlook, military
//! readiness, and diplomatic landscape. Report accuracy scales with scouting
//! visibility (more fog = less accurate data).

use crate::headless_campaign::actions::{StepDeltas, WorldEvent};
use crate::headless_campaign::state::*;

/// Cadence: one report every 1000 ticks.
const INTEL_INTERVAL: u64 = 1000;

/// Maximum number of reports kept in state.
const MAX_REPORTS: usize = 5;

/// Number of report types we rotate through.
const REPORT_TYPE_COUNT: u32 = 5;

pub fn tick_intel_reports(
    state: &mut CampaignState,
    _deltas: &mut StepDeltas,
    events: &mut Vec<WorldEvent>,
) {
    if state.tick % INTEL_INTERVAL != 0 || state.tick == 0 {
        return;
    }

    if state.factions.is_empty() {
        return;
    }

    // Rotate through report types based on tick cycle.
    let cycle = (state.tick / INTEL_INTERVAL) as u32;
    let report_type = match cycle % REPORT_TYPE_COUNT {
        0 => ReportType::FactionPowerRanking,
        1 => ReportType::ThreatAssessment,
        2 => ReportType::EconomicOutlook,
        3 => ReportType::MilitaryReadiness,
        _ => ReportType::DiplomaticLandscape,
    };

    // Compute average visibility across all regions for accuracy scaling.
    let avg_visibility = if state.overworld.regions.is_empty() {
        0.5
    } else {
        let sum: f32 = state.overworld.regions.iter().map(|r| r.visibility).sum();
        sum / state.overworld.regions.len() as f32
    };

    let (summary, data) = match report_type {
        ReportType::FactionPowerRanking => generate_power_ranking(state, avg_visibility),
        ReportType::ThreatAssessment => generate_threat_assessment(state, avg_visibility),
        ReportType::EconomicOutlook => generate_economic_outlook(state, avg_visibility),
        ReportType::MilitaryReadiness => generate_military_readiness(state, avg_visibility),
        ReportType::DiplomaticLandscape => generate_diplomatic_landscape(state, avg_visibility),
    };

    let report_id = state.next_report_id;
    state.next_report_id += 1;

    let report = IntelReport {
        id: report_id,
        tick: state.tick,
        report_type: report_type.clone(),
        summary: summary.clone(),
        data,
    };

    state.intel_reports.push(report);

    // Prune oldest reports beyond the cap.
    while state.intel_reports.len() > MAX_REPORTS {
        state.intel_reports.remove(0);
    }

    events.push(WorldEvent::IntelReportGenerated {
        report_type,
        summary,
    });
}

// ---------------------------------------------------------------------------
// Report generators
// ---------------------------------------------------------------------------

/// Rank all factions by a composite score of military + territory + economy.
fn generate_power_ranking(
    state: &CampaignState,
    avg_visibility: f32,
) -> (String, ReportData) {
    let mut rankings: Vec<(usize, f32)> = state
        .factions
        .iter()
        .map(|f| {
            let territory_score = f.territory_size as f32 * 10.0;
            let raw_score = f.military_strength + territory_score;
            // Apply visibility noise: lower visibility = noisier score.
            let noise_factor = 1.0 + (1.0 - avg_visibility) * 0.3;
            (f.id, raw_score * noise_factor)
        })
        .collect();

    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let strongest = rankings.first().map(|(id, _)| *id);
    let weakest = rankings.last().map(|(id, _)| *id);

    let top_name = strongest
        .and_then(|id| state.factions.iter().find(|f| f.id == id))
        .map(|f| f.name.clone())
        .unwrap_or_else(|| "Unknown".into());

    let summary = format!(
        "Faction power rankings updated. {} leads with score {:.0}.",
        top_name,
        rankings.first().map(|(_, s)| *s).unwrap_or(0.0)
    );

    let data = ReportData {
        faction_rankings: rankings,
        strongest_faction: strongest,
        weakest_faction: weakest,
        ..ReportData::default()
    };

    (summary, data)
}

/// Analyze global threat trends and predict next crisis escalation.
fn generate_threat_assessment(
    state: &CampaignState,
    avg_visibility: f32,
) -> (String, ReportData) {
    let global_threat = state.overworld.global_threat_level;

    // Compute threat trend from region-level data.
    let avg_region_threat = if state.overworld.regions.is_empty() {
        0.0
    } else {
        let sum: f32 = state
            .overworld
            .regions
            .iter()
            .map(|r| r.threat_level)
            .sum();
        sum / state.overworld.regions.len() as f32
    };

    // Positive means rising, negative means falling.
    let threat_trend = avg_region_threat - global_threat;

    // Scale by visibility — low visibility overestimates threats.
    let reported_trend = threat_trend + (1.0 - avg_visibility) * 5.0;

    // Identify imminent threats from high-unrest regions.
    let imminent: Vec<String> = state
        .overworld
        .regions
        .iter()
        .filter(|r| r.unrest > 60.0 || r.threat_level > 70.0)
        .filter(|r| r.visibility >= 0.3) // Need some visibility to detect
        .map(|r| {
            format!(
                "{}: unrest {:.0}, threat {:.0}",
                r.name, r.unrest, r.threat_level
            )
        })
        .collect();

    let trend_label = if reported_trend > 5.0 {
        "rising sharply"
    } else if reported_trend > 0.0 {
        "rising"
    } else if reported_trend < -5.0 {
        "falling sharply"
    } else {
        "stable"
    };

    let summary = format!(
        "Threat level {:.0}, trend {}. {} regions flagged.",
        global_threat,
        trend_label,
        imminent.len()
    );

    let data = ReportData {
        threat_trend: reported_trend,
        imminent_threats: imminent,
        ..ReportData::default()
    };

    (summary, data)
}

/// Guild gold trend, trade income projection, supply sustainability.
fn generate_economic_outlook(
    state: &CampaignState,
    avg_visibility: f32,
) -> (String, ReportData) {
    let gold = state.guild.gold;
    let supplies = state.guild.supplies;
    let trade_income = state.guild.total_trade_income;

    // Simple projection: current gold + trade trend per 1000 ticks.
    let projected = gold + trade_income * 0.1;

    // Apply visibility noise.
    let noise = 1.0 + (1.0 - avg_visibility) * 0.2;
    let gold_projection = projected * noise;

    let outlook = if gold_projection > gold * 1.2 {
        "positive"
    } else if gold_projection < gold * 0.8 {
        "concerning"
    } else {
        "stable"
    };

    let summary = format!(
        "Economic outlook: {}. Gold {:.0}, supplies {:.0}, projected {:.0}.",
        outlook, gold, supplies, gold_projection
    );

    let data = ReportData {
        gold_projection,
        ..ReportData::default()
    };

    (summary, data)
}

/// Party strength assessment, adventurer fatigue/injury levels.
fn generate_military_readiness(
    state: &CampaignState,
    _avg_visibility: f32, // Readiness is internal — no fog penalty.
) -> (String, ReportData) {
    let total = state.adventurers.len();
    let alive: Vec<&Adventurer> = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .collect();

    let idle_count = alive
        .iter()
        .filter(|a| a.status == AdventurerStatus::Idle)
        .count();
    let injured_count = alive
        .iter()
        .filter(|a| a.status == AdventurerStatus::Injured)
        .count();
    let fighting_count = alive
        .iter()
        .filter(|a| a.status == AdventurerStatus::Fighting)
        .count();

    let avg_fatigue = if alive.is_empty() {
        0.0
    } else {
        alive.iter().map(|a| a.fatigue).sum::<f32>() / alive.len() as f32
    };

    let avg_morale = if alive.is_empty() {
        0.0
    } else {
        alive.iter().map(|a| a.morale).sum::<f32>() / alive.len() as f32
    };

    let mut threats = Vec::new();
    if avg_fatigue > 60.0 {
        threats.push(format!("High average fatigue: {:.0}", avg_fatigue));
    }
    if avg_morale < 40.0 {
        threats.push(format!("Low average morale: {:.0}", avg_morale));
    }
    if injured_count > total / 3 {
        threats.push(format!("{} adventurers injured", injured_count));
    }

    let readiness = if avg_fatigue < 30.0 && avg_morale > 60.0 && injured_count == 0 {
        "high"
    } else if avg_fatigue > 60.0 || avg_morale < 30.0 {
        "low"
    } else {
        "moderate"
    };

    let summary = format!(
        "Military readiness: {}. {}/{} idle, {} injured, {} in combat. Fatigue {:.0}, morale {:.0}.",
        readiness,
        idle_count,
        alive.len(),
        injured_count,
        fighting_count,
        avg_fatigue,
        avg_morale
    );

    let data = ReportData {
        imminent_threats: threats,
        ..ReportData::default()
    };

    (summary, data)
}

/// Faction relations summary, alliance opportunities, war risks.
fn generate_diplomatic_landscape(
    state: &CampaignState,
    avg_visibility: f32,
) -> (String, ReportData) {
    let guild_faction_id = state.diplomacy.guild_faction_id;

    let mut threats = Vec::new();
    let mut strongest: Option<(usize, f32)> = None;
    let mut weakest: Option<(usize, f32)> = None;

    for faction in &state.factions {
        if faction.id == guild_faction_id {
            continue;
        }

        let rel = faction.relationship_to_guild;

        // Track extremes.
        match &strongest {
            None => strongest = Some((faction.id, rel)),
            Some((_, best)) if rel > *best => strongest = Some((faction.id, rel)),
            _ => {}
        }
        match &weakest {
            None => weakest = Some((faction.id, rel)),
            Some((_, worst)) if rel < *worst => weakest = Some((faction.id, rel)),
            _ => {}
        }

        // Detect war risks (only if we have visibility).
        if rel < -50.0 && avg_visibility > 0.3 {
            threats.push(format!(
                "{} hostile (relation {:.0})",
                faction.name, rel
            ));
        }

        // Alliance opportunities.
        if rel > 50.0 && !faction.coalition_member {
            threats.push(format!(
                "{} potential ally (relation {:.0})",
                faction.name, rel
            ));
        }

        // Active wars.
        if !faction.at_war_with.is_empty() {
            let war_targets: Vec<String> = faction
                .at_war_with
                .iter()
                .filter_map(|&tid| state.factions.iter().find(|f| f.id == tid))
                .map(|f| f.name.clone())
                .collect();
            if !war_targets.is_empty() {
                threats.push(format!(
                    "{} at war with {}",
                    faction.name,
                    war_targets.join(", ")
                ));
            }
        }
    }

    let summary = format!(
        "Diplomatic landscape: {} observations. Visibility {:.0}%.",
        threats.len(),
        avg_visibility * 100.0
    );

    let data = ReportData {
        strongest_faction: strongest.map(|(id, _)| id),
        weakest_faction: weakest.map(|(id, _)| id),
        imminent_threats: threats,
        ..ReportData::default()
    };

    (summary, data)
}
