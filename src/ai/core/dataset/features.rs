//! Per-unit feature extraction (115 features).

use super::super::{distance, is_alive, SimState, SimVec2};
use crate::ai::effects::StatusKind;
use crate::ai::squad::SquadAiState;

pub const FEATURE_DIM: usize = 115;

/// Encode an ai_hint string to a float for the feature vector.
fn encode_hint(hint: &str) -> f32 {
    match hint {
        "damage" => 0.2,
        "heal" => 0.4,
        "crowd_control" => 0.6,
        "defense" => 0.8,
        "utility" => 1.0,
        _ => 0.0,
    }
}

/// Compute DPS estimate for a unit.
fn unit_dps(u: &super::super::UnitState) -> f32 {
    if u.attack_cooldown_ms > 0 {
        u.attack_damage as f32 / (u.attack_cooldown_ms as f32 / 1000.0)
    } else {
        0.0
    }
}

/// Check if a unit has healing capability.
fn is_healer(u: &super::super::UnitState) -> bool {
    u.heal_amount > 0 || u.abilities.iter().any(|a| a.def.ai_hint == "heal")
}

/// Extract 115 features for a single unit in context of the battle.
pub fn extract_unit_features(
    state: &SimState,
    squad_ai: &SquadAiState,
    unit_id: u32,
) -> [f32; FEATURE_DIM] {
    let mut f = [0.0f32; FEATURE_DIM];
    let unit = match state.units.iter().find(|u| u.id == unit_id) {
        Some(u) => u,
        None => return f,
    };

    let allies: Vec<_> = state.units.iter().filter(|u| u.team == unit.team && is_alive(u)).collect();
    let enemies: Vec<_> = state.units.iter().filter(|u| u.team != unit.team && is_alive(u)).collect();

    // =====================================================================
    // Block 1: Self state (0-9) — 10 features
    // =====================================================================
    f[0] = unit.hp as f32 / unit.max_hp.max(1) as f32;
    f[1] = unit.shield_hp as f32 / unit.max_hp.max(1) as f32;
    f[2] = unit.move_speed_per_sec / 5.0;
    f[3] = unit.attack_range / 10.0;
    f[4] = unit.attack_damage as f32 / 50.0;
    f[5] = if unit.attack_cooldown_ms > 0 {
        unit.cooldown_remaining_ms as f32 / unit.attack_cooldown_ms as f32
    } else {
        0.0
    };
    // Casting state: 0=idle, encoding of what we're casting
    f[6] = match &unit.casting {
        None => 0.0,
        Some(c) => match c.kind {
            super::super::CastKind::Attack => 0.2,
            super::super::CastKind::Ability => 0.4,
            super::super::CastKind::Heal => 0.6,
            super::super::CastKind::Control => 0.8,
            super::super::CastKind::HeroAbility(_) => 1.0,
        },
    };
    f[7] = unit.casting.as_ref().map(|c| c.remaining_ms as f32 / 2000.0).unwrap_or(0.0);
    f[8] = if unit.max_resource > 0 {
        unit.resource as f32 / unit.max_resource as f32
    } else {
        0.0
    };
    f[9] = if unit.channeling.is_some() { 1.0 } else { 0.0 };

    // =====================================================================
    // Block 2: Per-ability slots (10-49) — 8 slots × 5 features = 40
    // =====================================================================
    for (i, slot) in unit.abilities.iter().take(8).enumerate() {
        let base = 10 + i * 5;
        // cd_frac: 0 = ready, 1 = full cooldown
        f[base] = if slot.def.cooldown_ms > 0 {
            slot.cooldown_remaining_ms as f32 / slot.def.cooldown_ms as f32
        } else {
            0.0
        };
        f[base + 1] = encode_hint(&slot.def.ai_hint);
        f[base + 2] = slot.def.range / 10.0;
        f[base + 3] = match slot.def.targeting {
            crate::ai::effects::AbilityTargeting::SelfAoe => 1.0,
            crate::ai::effects::AbilityTargeting::GroundTarget => 0.5,
            _ => 0.0,
        };
        f[base + 4] = if slot.def.resource_cost > 0 && unit.resource < slot.def.resource_cost {
            0.0 // can't afford
        } else if slot.cooldown_remaining_ms == 0 {
            1.0 // ready and affordable
        } else {
            0.0
        };
    }

    // =====================================================================
    // Block 3: Self status effects (50-58) — 9 features
    // =====================================================================
    for se in &unit.status_effects {
        match &se.kind {
            StatusKind::Dot { .. } => f[50] = 1.0,
            StatusKind::Hot { .. } => f[51] = 1.0,
            StatusKind::Silence => f[52] = 1.0,
            StatusKind::Root => f[53] = 1.0,
            StatusKind::Stealth { .. } => f[54] = 1.0,
            StatusKind::Reflect { .. } => f[55] = 1.0,
            StatusKind::Lifesteal { .. } => f[56] = 1.0,
            StatusKind::Blind { .. } => f[57] = 1.0,
            StatusKind::Shield { .. } | StatusKind::AbsorbShield { .. } => f[58] = 1.0,
            _ => {}
        }
    }

    // =====================================================================
    // Block 4: Top-3 enemies (59-82) — 3 enemies × 8 features = 24
    // =====================================================================
    let mut sorted_enemies: Vec<_> = enemies.iter()
        .map(|e| (distance(unit.position, e.position), *e))
        .collect();
    sorted_enemies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let focus_target = squad_ai
        .blackboard_for_team(unit.team)
        .and_then(|b| b.focus_target);

    for (i, &(dist, enemy)) in sorted_enemies.iter().take(3).enumerate() {
        let base = 59 + i * 8;
        f[base] = dist / 10.0;
        f[base + 1] = enemy.hp as f32 / enemy.max_hp.max(1) as f32;
        f[base + 2] = unit_dps(enemy) / 30.0;
        f[base + 3] = if is_healer(enemy) { 1.0 } else { 0.0 };
        f[base + 4] = if enemy.control_remaining_ms > 0 { 1.0 } else { 0.0 };
        f[base + 5] = if focus_target == Some(enemy.id) { 1.0 } else { 0.0 };
        f[base + 6] = if enemy.status_effects.iter().any(|s| matches!(s.kind, StatusKind::Reflect { .. })) { 1.0 } else { 0.0 };
        f[base + 7] = if enemy.status_effects.iter().any(|s| matches!(s.kind, StatusKind::Stealth { .. })) { 1.0 } else { 0.0 };
    }

    // =====================================================================
    // Block 5: Weakest ally (83-85) — 3 features
    // =====================================================================
    let weakest_ally = allies.iter()
        .filter(|a| a.id != unit_id)
        .min_by(|a, b| {
            let ha = a.hp as f32 / a.max_hp.max(1) as f32;
            let hb = b.hp as f32 / b.max_hp.max(1) as f32;
            ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(wa) = weakest_ally {
        f[83] = wa.hp as f32 / wa.max_hp.max(1) as f32;
        f[84] = distance(unit.position, wa.position) / 10.0;
        f[85] = if is_healer(wa) { 1.0 } else { 0.0 };
    } else {
        f[83] = 1.0; // no ally = full HP sentinel
    }

    // =====================================================================
    // Block 6: Team context (86-97) — 12 features
    // =====================================================================
    f[86] = allies.len() as f32 / 8.0;
    f[87] = enemies.len() as f32 / 8.0;
    let ally_hp_avg: f32 = if !allies.is_empty() {
        allies.iter().map(|u| u.hp as f32 / u.max_hp.max(1) as f32).sum::<f32>() / allies.len() as f32
    } else { 0.0 };
    f[88] = ally_hp_avg;
    let enemy_hp_avg: f32 = if !enemies.is_empty() {
        enemies.iter().map(|u| u.hp as f32 / u.max_hp.max(1) as f32).sum::<f32>() / enemies.len() as f32
    } else { 0.0 };
    f[89] = enemy_hp_avg;

    // Role composition: fraction of each role type on my team
    let ally_tanks = allies.iter().filter(|a| a.attack_range < 2.0 && a.max_hp > 80).count();
    let ally_healers = allies.iter().filter(|a| is_healer(a)).count();
    let ally_ranged = allies.iter().filter(|a| a.attack_range > 3.0).count();
    f[90] = ally_tanks as f32 / allies.len().max(1) as f32;
    f[91] = ally_healers as f32 / allies.len().max(1) as f32;
    f[92] = ally_ranged as f32 / allies.len().max(1) as f32;

    // Team spread (std dev of ally positions from centroid)
    let centroid = if !allies.is_empty() {
        let cx: f32 = allies.iter().map(|a| a.position.x).sum::<f32>() / allies.len() as f32;
        let cy: f32 = allies.iter().map(|a| a.position.y).sum::<f32>() / allies.len() as f32;
        SimVec2 { x: cx, y: cy }
    } else {
        unit.position
    };
    let team_spread = if allies.len() > 1 {
        let var: f32 = allies.iter()
            .map(|a| {
                let dx = a.position.x - centroid.x;
                let dy = a.position.y - centroid.y;
                dx * dx + dy * dy
            })
            .sum::<f32>() / allies.len() as f32;
        var.sqrt()
    } else { 0.0 };
    f[93] = team_spread / 5.0;
    f[94] = distance(unit.position, centroid) / 5.0;

    // Healer distance to centroid (relevant for frontline/backline awareness)
    let healer_centroid_dist = allies.iter()
        .filter(|a| is_healer(a))
        .map(|a| distance(a.position, centroid))
        .fold(0.0f32, f32::max);
    f[95] = healer_centroid_dist / 5.0;

    // Enemies in range / threats targeting me
    let enemies_in_range = enemies.iter()
        .filter(|e| distance(unit.position, e.position) <= unit.attack_range)
        .count();
    f[96] = enemies_in_range as f32 / 4.0;
    let threats = enemies.iter()
        .filter(|e| distance(e.position, unit.position) <= e.attack_range)
        .count();
    f[97] = threats as f32 / 4.0;

    // =====================================================================
    // Block 7: Squad coordination (98-107) — 10 features
    // =====================================================================
    let bb = squad_ai.blackboard_for_team(unit.team);
    // Formation mode: Hold=0, Advance=0.5, Retreat=1
    f[98] = match bb.map(|b| b.mode) {
        Some(crate::ai::squad::FormationMode::Hold) => 0.0,
        Some(crate::ai::squad::FormationMode::Advance) => 0.5,
        Some(crate::ai::squad::FormationMode::Retreat) => 1.0,
        None => 0.0,
    };
    // Is my current attack target the focus target?
    f[99] = if let (Some(ft), Some(casting)) = (focus_target, &unit.casting) {
        if casting.target_id == ft { 1.0 } else { 0.0 }
    } else { 0.0 };
    // How many allies share the same target as me?
    let my_target = unit.casting.as_ref().map(|c| c.target_id);
    f[100] = if let Some(mt) = my_target {
        allies.iter()
            .filter(|a| a.id != unit_id && a.casting.as_ref().map(|c| c.target_id) == Some(mt))
            .count() as f32 / allies.len().max(1) as f32
    } else { 0.0 };
    // How many allies are attacking the focus target?
    f[101] = if let Some(ft) = focus_target {
        allies.iter()
            .filter(|a| a.casting.as_ref().map(|c| c.target_id) == Some(ft))
            .count() as f32 / allies.len().max(1) as f32
    } else { 0.0 };

    // Self role heuristic: tank (0.2), healer (0.4), melee_dps (0.6), ranged_dps (0.8)
    f[102] = if is_healer(unit) {
        0.4
    } else if unit.attack_range < 2.0 && unit.max_hp > 80 {
        0.2  // tank
    } else if unit.attack_range < 2.0 {
        0.6  // melee dps
    } else {
        0.8  // ranged dps
    };

    // Distance to nearest ally
    f[103] = allies.iter()
        .filter(|a| a.id != unit_id)
        .map(|a| distance(unit.position, a.position))
        .fold(f32::MAX, f32::min)
        .min(10.0) / 10.0;

    // Frontline pressure: am I the closest ally to any enemy?
    f[104] = if !enemies.is_empty() {
        let is_frontline = enemies.iter().any(|e| {
            let my_dist = distance(unit.position, e.position);
            allies.iter().all(|a| distance(a.position, e.position) >= my_dist - 0.01)
        });
        if is_frontline { 1.0 } else { 0.0 }
    } else { 0.0 };

    // Enemy healer fraction
    let enemy_healers = enemies.iter().filter(|e| is_healer(e)).count();
    f[105] = enemy_healers as f32 / enemies.len().max(1) as f32;

    // Nearest enemy DPS
    f[106] = sorted_enemies.first().map(|(_, e)| unit_dps(e) / 30.0).unwrap_or(0.0);

    // Am I CC'd?
    f[107] = if unit.control_remaining_ms > 0 { 1.0 } else { 0.0 };

    // =====================================================================
    // Block 8: Environment (108-112) — 5 features
    // =====================================================================
    f[108] = unit.cover_bonus;
    f[109] = unit.elevation / 5.0;
    f[110] = sorted_enemies.first()
        .map(|(_, e)| (unit.elevation - e.elevation) / 5.0)
        .unwrap_or(0.0);
    let hostile_zones = state.zones.iter()
        .filter(|z| z.source_team != unit.team && distance(unit.position, z.position) < 3.0)
        .count();
    f[111] = hostile_zones as f32 / 3.0;
    let friendly_zones = state.zones.iter()
        .filter(|z| z.source_team == unit.team && distance(unit.position, z.position) < 3.0)
        .count();
    f[112] = friendly_zones as f32 / 3.0;

    // =====================================================================
    // Block 9: Game phase (113-114) — 2 features
    // =====================================================================
    f[113] = (state.tick as f32 / 500.0).min(2.0);
    f[114] = (allies.len() as f32 - enemies.len() as f32) / 4.0;

    f
}
