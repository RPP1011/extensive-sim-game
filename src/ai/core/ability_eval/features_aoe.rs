use crate::ai::core::{distance, is_alive, SimState, SimVec2, UnitState};

use super::features::{
    count_summons, terrain_features, terrain_features_at_pos, unit_dps,
};

/// Extract features for AoE damage abilities (SelfAoe, GroundTarget).
/// Features (22): self context + terrain + spatial clustering + target position terrain
pub fn extract_damage_aoe_features(
    state: &SimState,
    unit: &UnitState,
    ability_idx: usize,
) -> (Vec<f32>, Vec<SimVec2>) {
    let slot = &unit.abilities[ability_idx];
    let enemies: Vec<&UnitState> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();
    let allies: Vec<_> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .collect();

    let mut candidate_positions = Vec::new();

    if !enemies.is_empty() {
        let cx = enemies.iter().map(|e| e.position.x).sum::<f32>() / enemies.len() as f32;
        let cy = enemies.iter().map(|e| e.position.y).sum::<f32>() / enemies.len() as f32;
        candidate_positions.push(SimVec2 { x: cx, y: cy });
    }

    let mut sorted: Vec<_> = enemies.iter()
        .map(|e| (distance(unit.position, e.position), *e))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for &(_, e) in sorted.iter().take(3) {
        candidate_positions.push(e.position);
    }

    let aoe_radius = 2.0f32;
    let best_pos_idx = candidate_positions.iter().enumerate()
        .max_by_key(|(_, pos)| {
            enemies.iter().filter(|e| distance(e.position, **pos) <= aoe_radius).count()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_pos = candidate_positions.get(best_pos_idx).copied()
        .unwrap_or(unit.position);
    let enemies_at_best = enemies.iter()
        .filter(|e| distance(e.position, best_pos) <= aoe_radius)
        .count();
    let allies_at_best = allies.iter()
        .filter(|a| distance(a.position, best_pos) <= aoe_radius)
        .count();

    let nearest_enemy_dist = sorted.first().map(|(d, _)| *d).unwrap_or(10.0);
    let self_terrain = terrain_features(state, unit);
    let pos_terrain = terrain_features_at_pos(state, best_pos, unit.team);

    // Average cover of enemies at best position
    let avg_enemy_cover_at_best = {
        let enemies_there: Vec<_> = enemies.iter()
            .filter(|e| distance(e.position, best_pos) <= aoe_radius)
            .collect();
        if !enemies_there.is_empty() {
            enemies_there.iter().map(|e| e.cover_bonus).sum::<f32>() / enemies_there.len() as f32
        } else { 0.0 }
    };

    let features = vec![
        unit.hp as f32 / unit.max_hp.max(1) as f32,
        if unit.max_resource > 0 { unit.resource as f32 / unit.max_resource as f32 } else { 1.0 },
        slot.def.range / 10.0,
        slot.def.cast_time_ms as f32 / 2000.0,
        nearest_enemy_dist / 10.0,
        enemies.len() as f32 / 8.0,
        enemies_at_best as f32 / 4.0,
        allies_at_best as f32 / 4.0,
        (allies.len() as f32 - enemies.len() as f32) / 4.0,
        distance(unit.position, best_pos) / 10.0,
        // Per-target features for top 2 enemies
        sorted.first().map(|(_, e)| e.hp as f32 / e.max_hp.max(1) as f32).unwrap_or(1.0),
        sorted.first().map(|(_, e)| unit_dps(e) / 30.0).unwrap_or(0.0),
        sorted.get(1).map(|(_, e)| e.hp as f32 / e.max_hp.max(1) as f32).unwrap_or(1.0),
        sorted.get(1).map(|(_, e)| unit_dps(e) / 30.0).unwrap_or(0.0),
        // Enemy clustering
        if enemies.len() > 1 {
            let cx = enemies.iter().map(|e| e.position.x).sum::<f32>() / enemies.len() as f32;
            let cy = enemies.iter().map(|e| e.position.y).sum::<f32>() / enemies.len() as f32;
            let var: f32 = enemies.iter().map(|e| {
                let dx = e.position.x - cx;
                let dy = e.position.y - cy;
                dx * dx + dy * dy
            }).sum::<f32>() / enemies.len() as f32;
            (var.sqrt() / 5.0).min(1.0)
        } else { 1.0 },
        // Terrain: self
        self_terrain[0], self_terrain[1], self_terrain[2], self_terrain[3],
        // Terrain at target position
        pos_terrain[0], pos_terrain[1], avg_enemy_cover_at_best,
    ];

    (features, candidate_positions)
}

/// Extract features for HealAoe / Defense / Utility (simpler evaluators).
/// Features (14): generic self + team context + terrain
pub fn extract_simple_features(
    state: &SimState,
    unit: &UnitState,
    ability_idx: usize,
) -> Vec<f32> {
    let slot = &unit.abilities[ability_idx];
    let allies: Vec<_> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .collect();
    let enemies: Vec<_> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();

    let team_hp_avg = if !allies.is_empty() {
        allies.iter().map(|a| a.hp as f32 / a.max_hp.max(1) as f32).sum::<f32>() / allies.len() as f32
    } else { 0.0 };
    let team_hp_min = allies.iter()
        .map(|a| a.hp as f32 / a.max_hp.max(1) as f32)
        .fold(1.0f32, f32::min);
    let nearest_enemy_dist = enemies.iter()
        .map(|e| distance(unit.position, e.position))
        .fold(f32::MAX, f32::min)
        .min(10.0);
    let threats = enemies.iter()
        .filter(|e| distance(e.position, unit.position) <= e.attack_range)
        .count();
    let self_terrain = terrain_features(state, unit);

    vec![
        unit.hp as f32 / unit.max_hp.max(1) as f32,
        if unit.max_resource > 0 { unit.resource as f32 / unit.max_resource as f32 } else { 1.0 },
        slot.def.range / 10.0,
        slot.def.cast_time_ms as f32 / 2000.0,
        team_hp_avg,
        team_hp_min,
        nearest_enemy_dist / 10.0,
        threats as f32 / 4.0,
        allies.len() as f32 / 8.0,
        enemies.len() as f32 / 8.0,
        self_terrain[0], self_terrain[1], self_terrain[2], self_terrain[3],
    ]
}

/// Extract features for Summon abilities.
/// Features (16): self context + terrain + existing summons + enemy/ally counts
pub fn extract_summon_features(
    state: &SimState,
    unit: &UnitState,
    ability_idx: usize,
) -> Vec<f32> {
    let slot = &unit.abilities[ability_idx];
    let allies: Vec<_> = state.units.iter()
        .filter(|u| u.team == unit.team && is_alive(u))
        .collect();
    let enemies: Vec<_> = state.units.iter()
        .filter(|u| u.team != unit.team && is_alive(u))
        .collect();

    let team_hp_avg = if !allies.is_empty() {
        allies.iter().map(|a| a.hp as f32 / a.max_hp.max(1) as f32).sum::<f32>() / allies.len() as f32
    } else { 0.0 };
    let nearest_enemy_dist = enemies.iter()
        .map(|e| distance(unit.position, e.position))
        .fold(f32::MAX, f32::min)
        .min(10.0);
    let existing_summons = count_summons(state, unit.id);
    let self_terrain = terrain_features(state, unit);

    vec![
        unit.hp as f32 / unit.max_hp.max(1) as f32,
        if unit.max_resource > 0 { unit.resource as f32 / unit.max_resource as f32 } else { 1.0 },
        slot.def.range / 10.0,
        slot.def.cast_time_ms as f32 / 2000.0,
        team_hp_avg,
        nearest_enemy_dist / 10.0,
        allies.len() as f32 / 8.0,
        enemies.len() as f32 / 8.0,
        existing_summons as f32 / 4.0,                // don't over-summon
        (allies.len() as f32 - enemies.len() as f32) / 4.0, // numeric advantage
        // Summons are more valuable when outnumbered
        if enemies.len() > allies.len() { 1.0 } else { 0.0 },
        if unit.control_remaining_ms > 0 { 1.0 } else { 0.0 }, // can't summon while CC'd
        self_terrain[0], self_terrain[1], self_terrain[2], self_terrain[3],
    ]
}

/// Extract features for Obstacle/Wall creation abilities.
/// Features (20): self context + terrain + enemy positions relative to wall placement
pub fn extract_obstacle_features(
    state: &SimState,
    unit: &UnitState,
    ability_idx: usize,
) -> (Vec<f32>, Vec<SimVec2>) {
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

    // Candidate positions for wall placement
    let mut candidate_positions = Vec::new();

    // Between self and nearest enemies (cut off approach)
    if let Some(&(_, nearest_enemy)) = sorted.first() {
        let midx = (unit.position.x + nearest_enemy.position.x) / 2.0;
        let midy = (unit.position.y + nearest_enemy.position.y) / 2.0;
        candidate_positions.push(SimVec2 { x: midx, y: midy });
    }

    // Between enemies and lowest HP ally (protect retreat)
    let weakest_ally = allies.iter()
        .filter(|a| a.id != unit.id)
        .min_by(|a, b| {
            let ha = a.hp as f32 / a.max_hp.max(1) as f32;
            let hb = b.hp as f32 / b.max_hp.max(1) as f32;
            ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
        });
    if let (Some(&(_, nearest_enemy)), Some(ally)) = (sorted.first(), weakest_ally) {
        let midx = (ally.position.x + nearest_enemy.position.x) / 2.0;
        let midy = (ally.position.y + nearest_enemy.position.y) / 2.0;
        candidate_positions.push(SimVec2 { x: midx, y: midy });
    }

    // Enemy positions (wall on top of enemy = zoning)
    for &(_, e) in sorted.iter().take(2) {
        candidate_positions.push(e.position);
    }

    let self_terrain = terrain_features(state, unit);
    let team_hp_avg = if !allies.is_empty() {
        allies.iter().map(|a| a.hp as f32 / a.max_hp.max(1) as f32).sum::<f32>() / allies.len() as f32
    } else { 0.0 };
    let ally_critical = allies.iter().any(|a| (a.hp as f32 / a.max_hp.max(1) as f32) < 0.3);

    // How many enemies are approaching (moving toward allies)?
    let nearest_enemy_dist = sorted.first().map(|(d, _)| *d).unwrap_or(10.0);

    // Best wall position terrain
    let best_pos = candidate_positions.first().copied().unwrap_or(unit.position);
    let pos_terrain = terrain_features_at_pos(state, best_pos, unit.team);

    // Existing walls/blocked cells nearby
    let existing_obstacles = state.grid_nav.as_ref().map_or(0, |nav| {
        let cells = nav.cells_in_rect(unit.position, 6.0, 6.0);
        cells.iter().filter(|c| nav.blocked.contains(c)).count()
    });

    let features = vec![
        unit.hp as f32 / unit.max_hp.max(1) as f32,
        if unit.max_resource > 0 { unit.resource as f32 / unit.max_resource as f32 } else { 1.0 },
        slot.def.range / 10.0,
        slot.def.cast_time_ms as f32 / 2000.0,
        team_hp_avg,
        if ally_critical { 1.0 } else { 0.0 },
        nearest_enemy_dist / 10.0,
        enemies.len() as f32 / 8.0,
        allies.len() as f32 / 8.0,
        (allies.len() as f32 - enemies.len() as f32) / 4.0,
        existing_obstacles as f32 / 10.0,    // avoid over-walling
        // Terrain context
        self_terrain[0], self_terrain[1], self_terrain[2], self_terrain[3],
        pos_terrain[0], pos_terrain[1], pos_terrain[2],
        // Enemy clustering near best wall position
        sorted.iter().filter(|&&(d, _)| d <= 3.0).count() as f32 / 4.0,
        // Can wall split enemy group?
        if enemies.len() > 1 {
            let cx = enemies.iter().map(|e| e.position.x).sum::<f32>() / enemies.len() as f32;
            let cy = enemies.iter().map(|e| e.position.y).sum::<f32>() / enemies.len() as f32;
            let spread: f32 = enemies.iter().map(|e| {
                let dx = e.position.x - cx;
                let dy = e.position.y - cy;
                (dx * dx + dy * dy).sqrt()
            }).sum::<f32>() / enemies.len() as f32;
            (spread / 5.0).min(1.0) // high spread = wall can split
        } else { 0.0 },
    ];

    (features, candidate_positions)
}
