use crate::ai::core::{distance, is_alive, SimState, UnitState};
use crate::ai::effects::StatusKind;

// ---------------------------------------------------------------------------
// Per-category feature extraction — helpers and targeted extractors
// ---------------------------------------------------------------------------

/// Helper: compute DPS estimate for a unit.
pub(crate) fn unit_dps(u: &UnitState) -> f32 {
    if u.attack_cooldown_ms > 0 {
        u.attack_damage as f32 / (u.attack_cooldown_ms as f32 / 1000.0)
    } else {
        0.0
    }
}

/// Helper: check if a unit is a healer.
pub(crate) fn is_healer(u: &UnitState) -> bool {
    u.heal_amount > 0 || u.abilities.iter().any(|a| a.def.ai_hint == "heal")
}

/// Helper: check if unit has a specific status kind.
pub(crate) fn has_status(u: &UnitState, check: fn(&StatusKind) -> bool) -> bool {
    u.status_effects.iter().any(|s| check(&s.kind))
}

/// Terrain features for a unit (4 features).
pub(crate) fn terrain_features(state: &SimState, unit: &UnitState) -> [f32; 4] {
    let hostile_zones = state.zones.iter()
        .filter(|z| z.source_team != unit.team && distance(unit.position, z.position) < 3.0)
        .count();
    let friendly_zones = state.zones.iter()
        .filter(|z| z.source_team == unit.team && distance(unit.position, z.position) < 3.0)
        .count();
    [
        unit.cover_bonus,                    // 0.0-0.5 damage reduction from cover
        unit.elevation / 5.0,                // normalized elevation
        hostile_zones as f32 / 3.0,          // nearby enemy zones
        friendly_zones as f32 / 3.0,         // nearby friendly zones
    ]
}

/// Terrain features for a target position (3 features).
pub(crate) fn terrain_features_at_pos(state: &SimState, pos: crate::ai::core::SimVec2, team: crate::ai::core::Team) -> [f32; 3] {
    let hostile_zones = state.zones.iter()
        .filter(|z| z.source_team != team && distance(pos, z.position) < 3.0)
        .count();
    let friendly_zones = state.zones.iter()
        .filter(|z| z.source_team == team && distance(pos, z.position) < 3.0)
        .count();
    // Check if position has nav-mesh blocking (walls nearby)
    let blocked_nearby = state.grid_nav.as_ref().map_or(0.0, |nav| {
        let cells = nav.cells_in_rect(pos, 2.0, 2.0);
        let blocked = cells.iter().filter(|c| nav.blocked.contains(c)).count();
        blocked as f32 / cells.len().max(1) as f32
    });
    [
        hostile_zones as f32 / 3.0,
        friendly_zones as f32 / 3.0,
        blocked_nearby,
    ]
}

/// Count owned summons alive for a unit.
pub(crate) fn count_summons(state: &SimState, owner_id: u32) -> usize {
    state.units.iter()
        .filter(|u| u.owner_id == Some(owner_id) && is_alive(u))
        .count()
}

/// Team healing context features (4 features).
/// Helps evaluators understand healing saturation — whether the team needs
/// more healing or whether this unit should damage-cycle instead.
pub(crate) fn team_healing_context(state: &SimState, unit: &UnitState) -> [f32; 4] {
    let allies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .collect();
    let ally_count = allies.len().max(1) as f32;

    // How many allies are healers (have heal abilities)
    let healer_count = allies.iter()
        .filter(|a| is_healer(a))
        .count();

    // Team HP deficit: how much healing is actually needed (0 = all full, 1 = all dead)
    let hp_deficit = allies.iter()
        .map(|a| 1.0 - (a.hp as f32 / a.max_hp.max(1) as f32))
        .sum::<f32>() / ally_count;

    // Healing saturation: healers per ally (high = redundant healing)
    let healer_fraction = healer_count as f32 / ally_count;

    // Whether this unit is the primary healer (only healer on team)
    let is_sole_healer = is_healer(unit) && healer_count <= 1;

    [
        healer_fraction,                          // 0=no healers, 0.5=half the team heals
        hp_deficit,                               // 0=everyone full HP, 1=everyone dead
        if is_sole_healer { 1.0 } else { 0.0 },  // sole healer should prioritize healing
        (healer_count as f32 - 1.0).max(0.0) / 3.0, // other healers on team (0-3 normalized)
    ]
}

/// Extract features for a DamageUnit ability evaluation.
/// Features (28): self context + terrain + per-candidate-target features (with cover/elev)
pub fn extract_damage_unit_features(
    state: &SimState,
    unit: &UnitState,
    ability_idx: usize,
) -> (Vec<f32>, Vec<u32>) {
    let slot = &unit.abilities[ability_idx];
    let enemies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();

    let mut target_features = Vec::new();
    let mut target_ids = Vec::new();

    // Sort enemies by distance
    let mut sorted: Vec<_> = enemies.iter()
        .map(|e| (distance(unit.position, e.position), *e))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let allies: Vec<_> = state.units.iter().filter(|u| u.team == unit.team && is_alive(u)).collect();
    let in_range_count = sorted.iter().filter(|&&(d, _)| d <= slot.def.range).count();
    let self_terrain = terrain_features(state, unit);

    // Base features (10): self context + terrain
    let base = vec![
        unit.hp as f32 / unit.max_hp.max(1) as f32,
        if unit.max_resource > 0 { unit.resource as f32 / unit.max_resource as f32 } else { 1.0 },
        slot.def.range / 10.0,
        slot.def.cast_time_ms as f32 / 2000.0,
        in_range_count as f32 / 4.0,
        (allies.len() as f32 - enemies.len() as f32) / 4.0,
        self_terrain[0], self_terrain[1], self_terrain[2], self_terrain[3],
    ];

    // Per-target features (top 3 enemies): 8 features each = 24 (added cover + elevation advantage)
    for i in 0..3 {
        if let Some(&(dist, enemy)) = sorted.get(i) {
            target_ids.push(enemy.id);
            target_features.extend_from_slice(&[
                dist / 10.0,
                enemy.hp as f32 / enemy.max_hp.max(1) as f32,
                unit_dps(enemy) / 30.0,
                if is_healer(enemy) { 1.0 } else { 0.0 },
                if enemy.casting.is_some() { 1.0 } else { 0.0 },
                if has_status(enemy, |s| matches!(s, StatusKind::Reflect { .. })) { 1.0 } else { 0.0 },
                enemy.cover_bonus,                                // target's cover
                (unit.elevation - enemy.elevation) / 5.0,         // elevation advantage
            ]);
        } else {
            target_ids.push(0);
            target_features.extend_from_slice(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
    }

    let mut features = base;
    features.extend(target_features);
    (features, target_ids)
}

/// Extract features for a CcUnit ability evaluation.
/// Features (28): self context + terrain + per-candidate-target features with CC-specific info
pub fn extract_cc_unit_features(
    state: &SimState,
    unit: &UnitState,
    ability_idx: usize,
) -> (Vec<f32>, Vec<u32>) {
    let slot = &unit.abilities[ability_idx];
    let enemies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    let allies: Vec<_> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .collect();

    let mut sorted: Vec<_> = enemies.iter()
        .map(|e| (distance(unit.position, e.position), *e))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let team_hp_avg = if !allies.is_empty() {
        allies.iter().map(|a| a.hp as f32 / a.max_hp.max(1) as f32).sum::<f32>() / allies.len() as f32
    } else { 0.0 };
    let ally_critical = allies.iter().any(|a| (a.hp as f32 / a.max_hp.max(1) as f32) < 0.3);
    let self_terrain = terrain_features(state, unit);

    // Base features (10): self context + terrain
    let base = vec![
        unit.hp as f32 / unit.max_hp.max(1) as f32,
        slot.def.range / 10.0,
        slot.def.cast_time_ms as f32 / 2000.0,
        team_hp_avg,
        if ally_critical { 1.0 } else { 0.0 },
        (allies.len() as f32 - enemies.len() as f32) / 4.0,
        self_terrain[0], self_terrain[1], self_terrain[2], self_terrain[3],
    ];

    // Per-target features (top 3): 6 features each = 18
    let mut target_features = Vec::new();
    let mut target_ids = Vec::new();

    for i in 0..3 {
        if let Some(&(dist, enemy)) = sorted.get(i) {
            target_ids.push(enemy.id);
            target_features.extend_from_slice(&[
                dist / 10.0,
                unit_dps(enemy) / 30.0,
                if is_healer(enemy) { 1.0 } else { 0.0 },
                if enemy.casting.is_some() { 1.0 } else { 0.0 },
                if enemy.control_remaining_ms > 0 { 1.0 } else { 0.0 }, // already CC'd
                enemy.hp as f32 / enemy.max_hp.max(1) as f32,
            ]);
        } else {
            target_ids.push(0);
            target_features.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        }
    }

    let mut features = base;
    features.extend(target_features);
    (features, target_ids)
}

/// Extract features for a HealUnit ability evaluation.
/// Features (24): self context + terrain + per-ally features (with terrain exposure)
pub fn extract_heal_unit_features(
    state: &SimState,
    unit: &UnitState,
    ability_idx: usize,
) -> (Vec<f32>, Vec<u32>) {
    let slot = &unit.abilities[ability_idx];
    let allies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u) && u.id != unit.id)
        .collect();
    let enemies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();

    let mut sorted_allies: Vec<_> = allies.iter()
        .map(|a| (a.hp as f32 / a.max_hp.max(1) as f32, *a))
        .collect();
    sorted_allies.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let team_hp_avg = if !allies.is_empty() {
        allies.iter().map(|a| a.hp as f32 / a.max_hp.max(1) as f32).sum::<f32>() / allies.len() as f32
    } else { 1.0 };

    let self_in_danger = enemies.iter().any(|e| distance(unit.position, e.position) <= e.attack_range);
    let self_terrain = terrain_features(state, unit);

    // Base features (10)
    let base = vec![
        unit.hp as f32 / unit.max_hp.max(1) as f32,
        slot.def.range / 10.0,
        slot.def.cast_time_ms as f32 / 2000.0,
        team_hp_avg,
        enemies.len() as f32 / 8.0,
        if self_in_danger { 1.0 } else { 0.0 },
        self_terrain[0], self_terrain[1], self_terrain[2], self_terrain[3],
    ];

    // Per-ally features (top 3 lowest HP): 5 features each = 15 (added hostile zone exposure)
    let mut ally_features = Vec::new();
    let mut target_ids = Vec::new();

    for i in 0..3 {
        if let Some(&(hp_pct, ally)) = sorted_allies.get(i) {
            target_ids.push(ally.id);
            let has_hot = has_status(ally, |s| matches!(s, StatusKind::Hot { .. }));
            let threats_on_ally = enemies.iter()
                .filter(|e| distance(e.position, ally.position) <= e.attack_range)
                .count();
            let ally_hostile_zones = state.zones.iter()
                .filter(|z| z.source_team != ally.team && distance(ally.position, z.position) < 3.0)
                .count();
            ally_features.extend_from_slice(&[
                hp_pct,
                distance(unit.position, ally.position) / 10.0,
                if has_hot { 1.0 } else { 0.0 },
                threats_on_ally as f32 / 4.0,
                ally_hostile_zones as f32 / 3.0, // ally standing in enemy zone = more urgent
            ]);
        } else {
            target_ids.push(0);
            ally_features.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        }
    }

    let mut features = base;
    features.extend(ally_features);
    (features, target_ids)
}
