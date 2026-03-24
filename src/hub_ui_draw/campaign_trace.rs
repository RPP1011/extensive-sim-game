//! Campaign trace viewer UI rendering.
//!
//! Side panel: guild stats, active quests, timeline, controls.
//! Center panel: scrollable event feed with colored entries.

use bevy_egui::egui;

use bevy_game::headless_campaign::actions::WorldEvent;
use bevy_game::headless_campaign::state::*;
use bevy_game::headless_campaign::trace::CampaignTrace;
use bevy_game::headless_campaign::trace_viewer::CampaignTraceViewerState;

use super::common;

// ---------------------------------------------------------------------------
// Side panel
// ---------------------------------------------------------------------------

pub fn draw_trace_side_panel(ui: &mut egui::Ui, viewer: &CampaignTraceViewerState) {
    let state = &viewer.current_state;
    let trace = &viewer.trace;

    // Header
    ui.label(
        egui::RichText::new("▶ CAMPAIGN TRACE")
            .strong()
            .color(egui::Color32::from_rgb(140, 200, 255)),
    );
    common::ascii_separator(ui);

    // Seed & outcome
    let outcome_str = match trace.outcome {
        Some(CampaignOutcome::Victory) => "Victory",
        Some(CampaignOutcome::Defeat) => "Defeat",
        Some(CampaignOutcome::Timeout) => "Timeout",
        None => "In Progress",
    };
    let outcome_color = match trace.outcome {
        Some(CampaignOutcome::Victory) => egui::Color32::from_rgb(80, 200, 80),
        Some(CampaignOutcome::Defeat) => egui::Color32::from_rgb(255, 90, 80),
        _ => egui::Color32::from_rgb(200, 180, 80),
    };
    ui.colored_label(
        egui::Color32::from_rgb(120, 135, 155),
        format!("Seed: {}", trace.seed),
    );
    ui.colored_label(outcome_color, format!("Outcome: {}", outcome_str));
    common::ascii_separator(ui);

    // Time
    let current_time = CampaignTrace::format_tick(viewer.current_tick);
    let total_time = CampaignTrace::format_tick(trace.total_ticks);
    ui.colored_label(
        egui::Color32::from_rgb(180, 190, 210),
        format!("Tick: {} / {}", viewer.current_tick, trace.total_ticks),
    );
    ui.colored_label(
        egui::Color32::from_rgb(140, 155, 175),
        format!("Time: {} / {}", current_time, total_time),
    );
    common::ascii_separator(ui);

    // Guild stats
    ui.label(
        egui::RichText::new("── Guild ──")
            .small()
            .color(egui::Color32::from_rgb(80, 90, 105)),
    );
    ui.colored_label(
        egui::Color32::from_rgb(220, 200, 80),
        format!("Gold: {:.0}", state.guild.gold),
    );
    ui.colored_label(
        egui::Color32::from_rgb(140, 180, 140),
        format!("Supplies: {:.0}", state.guild.supplies),
    );
    ui.colored_label(
        egui::Color32::from_rgb(140, 160, 200),
        format!("Reputation: {:.0}", state.guild.reputation),
    );

    let alive = state
        .adventurers
        .iter()
        .filter(|a| a.status != AdventurerStatus::Dead)
        .count();
    let injured = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Injured)
        .count();
    let fighting = state
        .adventurers
        .iter()
        .filter(|a| a.status == AdventurerStatus::Fighting)
        .count();

    let adv_text = if injured > 0 {
        format!("Adventurers: {} ({} injured)", alive, injured)
    } else if fighting > 0 {
        format!("Adventurers: {} ({} fighting)", alive, fighting)
    } else {
        format!("Adventurers: {}", alive)
    };
    ui.colored_label(egui::Color32::from_rgb(160, 175, 195), adv_text);
    common::ascii_separator(ui);

    // Active quests
    ui.label(
        egui::RichText::new("── Quests ──")
            .small()
            .color(egui::Color32::from_rgb(80, 90, 105)),
    );
    if state.active_quests.is_empty() {
        ui.colored_label(
            egui::Color32::from_rgb(100, 110, 125),
            "  No active quests",
        );
    } else {
        for quest in &state.active_quests {
            let status = match quest.status {
                ActiveQuestStatus::Preparing => "Preparing",
                ActiveQuestStatus::Dispatched => "Dispatched",
                ActiveQuestStatus::InProgress => "In Progress",
                ActiveQuestStatus::InCombat => "In Combat!",
                ActiveQuestStatus::Returning => "Returning",
                ActiveQuestStatus::NeedsSupport => "Needs Support!",
            };
            let color = match quest.status {
                ActiveQuestStatus::InCombat | ActiveQuestStatus::NeedsSupport => {
                    egui::Color32::from_rgb(255, 140, 80)
                }
                ActiveQuestStatus::Returning => egui::Color32::from_rgb(80, 200, 80),
                _ => egui::Color32::from_rgb(160, 175, 195),
            };
            ui.colored_label(
                color,
                format!(
                    "[{:?}] {} (threat {:.0})",
                    quest.request.quest_type, status, quest.request.threat_level
                ),
            );
        }
    }
    common::ascii_separator(ui);

    // Battles
    if !state.active_battles.is_empty() {
        ui.label(
            egui::RichText::new("── Battles ──")
                .small()
                .color(egui::Color32::from_rgb(80, 90, 105)),
        );
        for battle in &state.active_battles {
            let color = if battle.party_health_ratio < 0.3 {
                egui::Color32::from_rgb(255, 80, 80)
            } else {
                egui::Color32::from_rgb(255, 180, 80)
            };
            ui.colored_label(
                color,
                format!(
                    "Battle #{}: Party {:.0}% / Enemy {:.0}%",
                    battle.id,
                    battle.party_health_ratio * 100.0,
                    battle.enemy_health_ratio * 100.0
                ),
            );
        }
        common::ascii_separator(ui);
    }

    // Parties
    if !state.parties.is_empty() {
        ui.label(
            egui::RichText::new("── Parties ──")
                .small()
                .color(egui::Color32::from_rgb(80, 90, 105)),
        );
        for party in &state.parties {
            let status = match party.status {
                PartyStatus::Idle => "Idle",
                PartyStatus::Traveling => "Traveling",
                PartyStatus::OnMission => "On Mission",
                PartyStatus::Fighting => "Fighting!",
                PartyStatus::Returning => "Returning",
            };
            ui.colored_label(
                egui::Color32::from_rgb(140, 155, 175),
                format!(
                    "Party #{}: {} ({} members, {:.0}% supply)",
                    party.id,
                    status,
                    party.member_ids.len(),
                    party.supply_level
                ),
            );
        }
        common::ascii_separator(ui);
    }

    // Timeline scrub bar
    ui.label(
        egui::RichText::new("── Timeline ──")
            .small()
            .color(egui::Color32::from_rgb(80, 90, 105)),
    );
    {
        let bar_w = 20usize;
        let progress = if trace.total_ticks > 0 {
            (viewer.current_tick as f32 / trace.total_ticks as f32).min(1.0)
        } else {
            0.0
        };
        let filled = (progress * bar_w as f32) as usize;
        let empty = bar_w.saturating_sub(filled);

        let font = egui::FontId::monospace(13.0);
        let mut job = egui::text::LayoutJob::default();
        job.append(
            "[",
            0.0,
            egui::TextFormat {
                font_id: font.clone(),
                color: egui::Color32::from_rgb(70, 80, 95),
                ..Default::default()
            },
        );
        job.append(
            &"█".repeat(filled),
            0.0,
            egui::TextFormat {
                font_id: font.clone(),
                color: egui::Color32::from_rgb(100, 160, 255),
                ..Default::default()
            },
        );
        job.append(
            &"░".repeat(empty),
            0.0,
            egui::TextFormat {
                font_id: font.clone(),
                color: egui::Color32::from_rgb(50, 65, 85),
                ..Default::default()
            },
        );
        job.append(
            "]",
            0.0,
            egui::TextFormat {
                font_id: font.clone(),
                color: egui::Color32::from_rgb(70, 80, 95),
                ..Default::default()
            },
        );
        ui.label(job);
    }
    common::ascii_separator(ui);

    // Controls
    ui.label(
        egui::RichText::new("── Controls ──")
            .small()
            .color(egui::Color32::from_rgb(80, 90, 105)),
    );
    let kc = egui::Color32::from_rgb(140, 155, 175);
    ui.colored_label(kc, "Space      Play / Pause");
    ui.colored_label(kc, "←→         Scrub ±100 ticks");
    ui.colored_label(kc, "Shift+←→   Scrub ±1000 ticks");
    ui.colored_label(kc, "+/-        Speed control");
    ui.colored_label(kc, "F          Fork & save state");
    ui.colored_label(kc, "Esc        Exit viewer");
    common::ascii_separator(ui);

    // Speed
    let speed_mult = viewer.tick_speed / 10.0;
    let paused_str = if viewer.paused { " (PAUSED)" } else { "" };
    ui.colored_label(
        egui::Color32::from_rgb(120, 140, 165),
        format!("Speed: {:.0}x{}", speed_mult, paused_str),
    );
}

// ---------------------------------------------------------------------------
// Center panel (event feed)
// ---------------------------------------------------------------------------

pub fn draw_trace_center_panel(ui: &mut egui::Ui, viewer: &CampaignTraceViewerState) {
    ui.heading("Campaign Event Log");
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for trace_event in &viewer.visible_events {
                let time_str = CampaignTrace::format_tick(trace_event.tick);
                let (text, color) = format_world_event(&trace_event.event);
                let is_current = trace_event.tick == viewer.current_tick;

                let prefix_color = if is_current {
                    egui::Color32::from_rgb(255, 220, 100)
                } else {
                    egui::Color32::from_rgb(90, 100, 115)
                };

                ui.horizontal(|ui| {
                    ui.colored_label(prefix_color, format!("[{}]", time_str));
                    ui.colored_label(color, text);
                });
            }

            if viewer.visible_events.is_empty() {
                ui.colored_label(
                    egui::Color32::from_rgb(100, 110, 125),
                    "No events near current tick",
                );
            }
        });
}

/// Format a WorldEvent for display.
fn format_world_event(event: &WorldEvent) -> (String, egui::Color32) {
    let green = egui::Color32::from_rgb(80, 200, 80);
    let red = egui::Color32::from_rgb(255, 90, 80);
    let yellow = egui::Color32::from_rgb(220, 200, 80);
    let blue = egui::Color32::from_rgb(100, 160, 255);
    let gray = egui::Color32::from_rgb(160, 170, 185);

    match event {
        WorldEvent::QuestRequestArrived {
            quest_type,
            threat_level,
            ..
        } => (
            format!("New {:?} quest available (threat {:.0})", quest_type, threat_level),
            blue,
        ),
        WorldEvent::QuestRequestExpired { request_id } => (
            format!("Quest #{} expired", request_id),
            gray,
        ),
        WorldEvent::QuestDispatched {
            quest_id,
            member_count,
            ..
        } => (
            format!("Quest #{} dispatched ({} members)", quest_id, member_count),
            blue,
        ),
        WorldEvent::QuestCompleted { quest_id, result } => {
            let (r, c) = match result {
                bevy_game::headless_campaign::state::QuestResult::Victory => ("completed", green),
                bevy_game::headless_campaign::state::QuestResult::Defeat => ("failed", red),
                bevy_game::headless_campaign::state::QuestResult::Abandoned => ("abandoned", yellow),
            };
            (format!("Quest #{} {}", quest_id, r), c)
        }
        WorldEvent::BattleStarted {
            battle_id,
            enemy_strength,
            ..
        } => (
            format!("Battle #{} started (enemy strength: {:.0})", battle_id, enemy_strength),
            yellow,
        ),
        WorldEvent::BattleUpdate {
            battle_id,
            party_health_ratio,
            enemy_health_ratio,
        } => (
            format!(
                "Battle #{}: Party {:.0}% / Enemy {:.0}%",
                battle_id,
                party_health_ratio * 100.0,
                enemy_health_ratio * 100.0
            ),
            gray,
        ),
        WorldEvent::BattleEnded { battle_id, result } => {
            let (r, c) = match result {
                BattleStatus::Victory => ("won!", green),
                BattleStatus::Defeat => ("lost!", red),
                BattleStatus::Retreat => ("retreat", yellow),
                BattleStatus::Active => ("ongoing", gray),
            };
            (format!("Battle #{} {}", battle_id, r), c)
        }
        WorldEvent::PartyFormed {
            party_id,
            member_ids,
        } => (
            format!("Party #{} formed ({} members)", party_id, member_ids.len()),
            blue,
        ),
        WorldEvent::PartyArrived { party_id, .. } => (
            format!("Party #{} arrived at destination", party_id),
            blue,
        ),
        WorldEvent::PartyReturned { party_id } => (
            format!("Party #{} returned to base", party_id),
            green,
        ),
        WorldEvent::PartySupplyLow {
            party_id,
            supply_level,
        } => (
            format!("Party #{} supply low ({:.0}%)", party_id, supply_level),
            yellow,
        ),
        WorldEvent::AdventurerLevelUp {
            adventurer_id,
            new_level,
        } => (
            format!("Adventurer #{} leveled up to {}", adventurer_id, new_level),
            green,
        ),
        WorldEvent::AdventurerInjured {
            adventurer_id,
            injury_level,
        } => (
            format!(
                "Adventurer #{} injured ({:.0}%)",
                adventurer_id, injury_level
            ),
            red,
        ),
        WorldEvent::AdventurerRecovered { adventurer_id } => (
            format!("Adventurer #{} recovered", adventurer_id),
            green,
        ),
        WorldEvent::AdventurerDeserted {
            adventurer_id,
            reason,
        } => (
            format!("Adventurer #{} deserted: {}", adventurer_id, reason),
            red,
        ),
        WorldEvent::AdventurerDied {
            adventurer_id,
            cause,
        } => (
            format!("Adventurer #{} died: {}", adventurer_id, cause),
            red,
        ),
        WorldEvent::RunnerSent { party_id, cost, .. } => (
            format!("Runner sent to Party #{} ({:.0}g)", party_id, cost),
            blue,
        ),
        WorldEvent::MercenaryHired { quest_id, cost } => (
            format!("Mercenary hired for Quest #{} ({:.0}g)", quest_id, cost),
            blue,
        ),
        WorldEvent::RescueCalled {
            battle_id, cost, ..
        } => (
            format!("Rescue called for Battle #{} ({:.0}g)", battle_id, cost),
            yellow,
        ),
        WorldEvent::ScoutReport {
            location_id,
            threat_level,
        } => (
            format!(
                "Scout report: Location #{} (threat {:.0})",
                location_id, threat_level
            ),
            blue,
        ),
        WorldEvent::FactionActionTaken {
            faction_id, action, ..
        } => (
            format!("Faction #{}: {}", faction_id, action),
            gray,
        ),
        WorldEvent::FactionRelationChanged {
            faction_id,
            old,
            new,
        } => (
            format!(
                "Faction #{} relation: {:.0} → {:.0}",
                faction_id, old, new
            ),
            if *new > *old { green } else { red },
        ),
        WorldEvent::RegionOwnerChanged {
            region_id,
            old_owner,
            new_owner,
        } => (
            format!(
                "Region #{} ownership: #{} → #{}",
                region_id, old_owner, new_owner
            ),
            yellow,
        ),
        WorldEvent::ProgressionUnlocked { name, category, .. } => (
            format!("Unlock: {} ({:?})", name, category),
            green,
        ),
        WorldEvent::GoldChanged { amount, reason } => {
            let sign = if *amount > 0.0 { "+" } else { "" };
            (
                format!("{}{:.0}g: {}", sign, amount, reason),
                if *amount > 0.0 { green } else { red },
            )
        }
        WorldEvent::SupplyChanged { amount, reason } => (
            format!("+{:.0} supplies: {}", amount, reason),
            blue,
        ),
        WorldEvent::ChoicePresented { prompt, num_options, .. } => (
            format!("CHOICE: {} ({} options)", prompt, num_options),
            egui::Color32::from_rgb(220, 180, 255),
        ),
        WorldEvent::ChoiceResolved { label, was_default, .. } => {
            let prefix = if *was_default { "Auto-chose" } else { "Chose" };
            (format!("{}: {}", prefix, label), egui::Color32::from_rgb(180, 160, 220))
        },
        WorldEvent::RandomEvent { name, description } => (
            format!("[{}] {}", name, description),
            yellow,
        ),
        WorldEvent::CalamityWarning { description } => (description.clone(), red),
        WorldEvent::CampaignMilestone { description } => (description.clone(), green),
        WorldEvent::RegionScoutReport { region_id, details } => (
            format!(
                "Scout report: {} (threat {:.0}, unrest {:.0}, {} quests)",
                details.region_name, details.threat_level, details.unrest, details.quest_opportunities
            ),
            blue,
        ),
        WorldEvent::ChampionIntercepted { champion_id, .. } => (
            format!("Champion {} intercepted!", champion_id),
            yellow,
        ),
        WorldEvent::ChampionArrived { champion_name, .. } => (
            format!("{} arrived at the King's side", champion_name),
            red,
        ),
        WorldEvent::BuildingUpgraded { building, new_tier, cost } => (
            format!("{} upgraded to tier {} (-{:.0}g)", building, new_tier, cost),
            green,
        ),
        WorldEvent::BondGrief { adventurer_id, dead_id, bond_strength } => (
            format!("Adventurer {} grieves for {} (bond: {:.0})", adventurer_id, dead_id, bond_strength),
            red,
        ),
        WorldEvent::SeasonChanged { new_season } => (
            format!("Season changed to {:?}", new_season),
            yellow,
        ),
    }
}
