use super::generation::*;
use super::overworld_types::*;
use super::roster_types::*;
use super::types::*;

pub(crate) const FLASHPOINT_TRIGGER_PRESSURE: f32 = 86.0;
pub(crate) const FLASHPOINT_TOTAL_STAGES: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashpointCompanionHookKind {
    Homefront,
    Aggressor,
    Defender,
}

pub(crate) fn flashpoint_companion_hook_kind(
    chain: &FlashpointChain,
    roster: &CampaignRoster,
) -> Option<(u32, String, FlashpointCompanionHookKind)> {
    let hero_id = chain.companion_hook_hero_id?;
    let hero = roster.heroes.iter().find(|h| h.id == hero_id)?;
    if hero.origin_region_id == chain.region_id {
        return Some((
            hero_id,
            hero.name.clone(),
            FlashpointCompanionHookKind::Homefront,
        ));
    }
    if hero.origin_faction_id == chain.attacker_faction_id {
        return Some((
            hero_id,
            hero.name.clone(),
            FlashpointCompanionHookKind::Aggressor,
        ));
    }
    if hero.origin_faction_id == chain.defender_faction_id {
        return Some((
            hero_id,
            hero.name.clone(),
            FlashpointCompanionHookKind::Defender,
        ));
    }
    None
}

pub(crate) fn apply_flashpoint_companion_hook(
    mission: &mut MissionSnapshot,
    chain: &FlashpointChain,
    roster: &CampaignRoster,
) -> Option<String> {
    let (_hero_id, hero_name, kind) = flashpoint_companion_hook_kind(chain, roster)?;
    match kind {
        FlashpointCompanionHookKind::Homefront => {
            mission.turns_remaining = mission.turns_remaining.saturating_add(2);
            mission.sabotage_progress = (mission.sabotage_progress + 8.0).clamp(0.0, 100.0);
            mission.reactor_integrity = (mission.reactor_integrity + 5.0).clamp(0.0, 100.0);
        }
        FlashpointCompanionHookKind::Aggressor => {
            mission.turns_remaining = mission.turns_remaining.saturating_sub(2).max(8);
            mission.alert_level = (mission.alert_level + 9.0).clamp(0.0, 100.0);
            mission.sabotage_progress = (mission.sabotage_progress + 6.0).clamp(0.0, 100.0);
        }
        FlashpointCompanionHookKind::Defender => {
            mission.alert_level = (mission.alert_level - 7.0).clamp(0.0, 100.0);
            mission.reactor_integrity = (mission.reactor_integrity + 9.0).clamp(0.0, 100.0);
        }
    }
    Some(hero_name)
}

pub(crate) fn flashpoint_stage_label(stage: u8) -> &'static str {
    match stage {
        1 => "Recon Sweep",
        2 => "Sabotage Push",
        _ => "Decisive Assault",
    }
}

pub(crate) fn flashpoint_intent_label(intent: FlashpointIntent) -> &'static str {
    match intent {
        FlashpointIntent::StealthPush => "Stealth Push",
        FlashpointIntent::DirectAssault => "Direct Assault",
        FlashpointIntent::CivilianFirst => "Civilian First",
    }
}

pub(crate) fn flashpoint_hook_objective_suffix(
    chain: &FlashpointChain,
    roster: &CampaignRoster,
) -> Option<String> {
    let (_hero_id, hero_name, kind) = flashpoint_companion_hook_kind(chain, roster)?;
    let detail = match kind {
        FlashpointCompanionHookKind::Homefront => "evacuate districts and secure local allies",
        FlashpointCompanionHookKind::Aggressor => "break enemy command relays before counterfire",
        FlashpointCompanionHookKind::Defender => "hold relief corridors and preserve defenses",
    };
    Some(format!("{hero_name}: {detail}"))
}

pub(crate) fn flashpoint_projection_suffix(
    chain: &FlashpointChain,
    overworld: &OverworldMap,
) -> (String, String) {
    let attacker = overworld
        .factions
        .get(chain.attacker_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Attackers");
    let defender = overworld
        .factions
        .get(chain.defender_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Defenders");
    (
        format!("win=>{} border+recruit", attacker),
        format!("lose=>{} hold-line", defender),
    )
}

pub(crate) fn rewrite_flashpoint_mission_name(
    mission: &mut MissionSnapshot,
    chain: &FlashpointChain,
    overworld: &OverworldMap,
    roster: Option<&CampaignRoster>,
) {
    let stage = chain.stage.clamp(1, FLASHPOINT_TOTAL_STAGES);
    let region_name = overworld
        .regions
        .iter()
        .find(|r| r.id == chain.region_id)
        .map(|r| r.name.as_str())
        .unwrap_or("Unknown Region");
    let attacker_name = overworld
        .factions
        .get(chain.attacker_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Rival host");
    let defender_name = overworld
        .factions
        .get(chain.defender_faction_id)
        .map(|f| f.name.as_str())
        .unwrap_or("Defenders");
    let (win_proj, lose_proj) = flashpoint_projection_suffix(chain, overworld);
    let hook = roster
        .and_then(|r| flashpoint_hook_objective_suffix(chain, r))
        .unwrap_or_else(|| "No companion hook".to_string());
    mission.mission_name = format!(
        "Flashpoint {}/{} [{}|{}]: {} ({} vs {}) | obj={} | {} {}",
        stage,
        FLASHPOINT_TOTAL_STAGES,
        flashpoint_stage_label(stage),
        flashpoint_intent_label(chain.intent),
        region_name,
        attacker_name,
        defender_name,
        hook,
        win_proj,
        lose_proj
    );
}

pub(crate) fn pick_flashpoint_attacker(region: &OverworldRegion, overworld: &OverworldMap) -> Option<usize> {
    let defender = region.owner_faction_id;
    let mut best = None;
    let mut best_score = f32::MIN;
    for neighbor in &region.neighbors {
        let Some(other) = overworld.regions.get(*neighbor) else {
            continue;
        };
        let attacker = other.owner_faction_id;
        if attacker == defender {
            continue;
        }
        let hostility = overworld
            .factions
            .get(attacker)
            .and_then(|f| f.war_goal_faction_id)
            .map(|goal| if goal == defender { 22.0 } else { 0.0 })
            .unwrap_or(0.0);
        let score = overworld
            .factions
            .get(attacker)
            .map(|f| f.strength + f.war_focus * 0.55 + f.cohesion * 0.25 + hostility)
            .unwrap_or(0.0);
        if score > best_score {
            best_score = score;
            best = Some(attacker);
        }
    }
    best
}

pub(crate) fn configure_flashpoint_stage_mission(
    mission: &mut MissionSnapshot,
    chain: &FlashpointChain,
    _overworld: &OverworldMap,
    seed: u64,
) {
    let stage = chain.stage.clamp(1, FLASHPOINT_TOTAL_STAGES);
    let stage_ix = stage as usize - 1;
    let jitter = (rand01(
        seed ^ chain.id as u64,
        chain.region_id as u64 * 97 + stage as u64 * 17,
    ) - 0.5)
        * 6.0;
    let turn_bases = [24.0, 20.0, 16.0];
    let alert_bases = [42.0, 58.0, 74.0];
    let integrity_bases = [84.0, 72.0, 58.0];
    let progress_bases = [20.0, 34.0, 46.0];
    mission.turns_remaining = (turn_bases[stage_ix] + jitter).clamp(10.0, 30.0) as u32;
    mission.alert_level = (alert_bases[stage_ix] + jitter * 1.4).clamp(12.0, 96.0);
    mission.reactor_integrity = (integrity_bases[stage_ix] - jitter * 1.3).clamp(20.0, 100.0);
    mission.sabotage_progress = (progress_bases[stage_ix] + jitter).clamp(0.0, 72.0);
    mission.tactical_mode = match stage {
        1 => TacticalMode::Defensive,
        2 => TacticalMode::Balanced,
        _ => TacticalMode::Aggressive,
    };
    mission.bound_region_id = Some(chain.region_id);
    mission.mission_active = true;
    mission.result = MissionResult::InProgress;
    mission.command_cooldown_turns = 0;
    mission.unattended_turns = 0;
    mission.outcome_recorded = false;
}

pub(crate) fn apply_flashpoint_intent(mission: &mut MissionSnapshot, chain: &FlashpointChain) {
    let stage_scale = 1.0 + (chain.stage as f32 - 1.0) * 0.2;
    match chain.intent {
        FlashpointIntent::StealthPush => {
            mission.alert_level = (mission.alert_level - 7.0 * stage_scale).clamp(0.0, 100.0);
            mission.sabotage_progress =
                (mission.sabotage_progress + 6.0 * stage_scale).clamp(0.0, 100.0);
        }
        FlashpointIntent::DirectAssault => {
            mission.turns_remaining = mission.turns_remaining.saturating_sub(2).max(8);
            mission.alert_level = (mission.alert_level + 9.0 * stage_scale).clamp(0.0, 100.0);
            mission.sabotage_progress =
                (mission.sabotage_progress + 8.0 * stage_scale).clamp(0.0, 100.0);
        }
        FlashpointIntent::CivilianFirst => {
            mission.turns_remaining = mission.turns_remaining.saturating_add(2);
            mission.alert_level = (mission.alert_level - 4.0 * stage_scale).clamp(0.0, 100.0);
            mission.reactor_integrity =
                (mission.reactor_integrity + 8.0 * stage_scale).clamp(0.0, 100.0);
        }
    }
}

pub(crate) fn inject_recruit_for_faction(
    roster: &mut CampaignRoster,
    overworld: &OverworldMap,
    faction_id: usize,
    seed: u64,
) {
    let id = roster.next_id.max(
        roster
            .recruit_pool
            .iter()
            .map(|r| r.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1),
    );
    roster.next_id = id.saturating_add(1);

    let mut recruit = super::roster_gen::generate_recruit_for_overworld(seed, id, overworld);
    if recruit.origin_faction_id != faction_id {
        let fallback_region = overworld.regions.first().map(|r| r.id).unwrap_or(0);
        let region_id = overworld
            .regions
            .iter()
            .find(|r| r.owner_faction_id == faction_id)
            .map(|r| r.id)
            .unwrap_or(fallback_region);
        let region = overworld
            .regions
            .iter()
            .find(|r| r.id == region_id)
            .or_else(|| overworld.regions.first());
        let faction_name = overworld
            .factions
            .get(faction_id)
            .map(|f| f.name.as_str())
            .unwrap_or("Unaligned House");
        let (region_name, unrest, control) = region
            .map(|r| (r.name.as_str(), r.unrest, r.control))
            .unwrap_or(("Unknown March", 50.0, 50.0));
        recruit.origin_faction_id = faction_id;
        recruit.origin_region_id = region_id;
        recruit.backstory = backstory_for_recruit(
            &recruit.codename,
            recruit.archetype,
            faction_name,
            region_name,
            unrest,
            control,
        );
    }
    roster.recruit_pool.push(recruit);
}

pub(crate) fn build_pressure_mission_snapshot(
    seed: u64,
    turn: u32,
    slot: usize,
    region: &OverworldRegion,
    owner: &FactionState,
    pressure: f32,
) -> MissionSnapshot {
    let salt = splitmix64(
        seed ^ (turn as u64).wrapping_mul(97) ^ (region.id as u64).wrapping_mul(131) ^ slot as u64,
    );
    let variant = (salt % 5) as usize;
    let labels = [
        "Frontline Clash",
        "Insurgent Network",
        "Supply Collapse",
        "Arcane Breach",
        "Border Uprising",
    ];
    let mode = match variant {
        0 => TacticalMode::Aggressive,
        1 => TacticalMode::Defensive,
        2 => TacticalMode::Balanced,
        3 => TacticalMode::Defensive,
        _ => TacticalMode::Aggressive,
    };
    let reveal = if region.intel_level < 30.0 {
        "Unconfirmed Crisis"
    } else {
        labels[variant]
    };
    let mission_name = format!("{}: {} [{}]", reveal, region.name, owner.name);
    let variance = (rand01(salt, 5) - 0.5) * 8.0;
    let turns_remaining =
        (28.0 - pressure * 0.18 - owner.war_focus * 0.04 + variance).clamp(10.0, 32.0) as u32;
    let alert_level =
        (16.0 + pressure * 0.55 + owner.war_focus * 0.12 + variance * 0.6).clamp(8.0, 96.0);
    let reactor_integrity =
        (100.0 - pressure * 0.45 - owner.war_focus * 0.06 + variance * 0.5).clamp(12.0, 100.0);
    let sabotage_progress = (rand01(salt, 17) * 16.0 + region.intel_level * 0.06).clamp(0.0, 45.0);

    MissionSnapshot {
        mission_name,
        bound_region_id: Some(region.id),
        mission_active: true,
        result: MissionResult::InProgress,
        turns_remaining,
        reactor_integrity,
        sabotage_progress,
        sabotage_goal: 100.0,
        alert_level,
        room_index: (salt % 6) as usize,
        tactical_mode: mode,
        command_cooldown_turns: 0,
        unattended_turns: 0,
        outcome_recorded: false,
    }
}
