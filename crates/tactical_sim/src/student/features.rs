use std::collections::HashMap;

use crate::sim::{distance, SimState, Team};
use crate::personality::PersonalityProfile;
use crate::roles::Role;
use crate::squad::FormationMode;

use super::model::{StudentMLP, StudentOutput};

// ---------------------------------------------------------------------------
// Feature extraction (mirrors Python student.py extract_features)
// ---------------------------------------------------------------------------

/// Extract 60 input features from game state (40 aggregate + 20 spatial).
pub fn extract_features(
    state: &SimState,
    roles: &HashMap<u32, Role>,
    current_personality: &HashMap<u32, PersonalityProfile>,
    current_formation: FormationMode,
) -> Vec<f32> {
    let heroes: Vec<_> = state.units.iter().filter(|u| u.hp > 0 && u.team == Team::Hero).collect();
    let enemies: Vec<_> = state.units.iter().filter(|u| u.hp > 0 && u.team == Team::Enemy).collect();

    let mut f = Vec::with_capacity(60);

    // Team HP aggregates (8)
    let hero_hp_sum: f32 = heroes.iter().map(|u| u.hp.max(0) as f32).sum();
    let hero_max_hp: f32 = heroes.iter().map(|u| u.max_hp.max(1) as f32).sum();
    let hero_avg_hp_pct = if hero_max_hp > 0.0 { hero_hp_sum / hero_max_hp } else { 0.0 };
    let hero_min_hp_pct = heroes
        .iter()
        .map(|u| u.hp.max(0) as f32 / u.max_hp.max(1) as f32)
        .fold(1.0_f32, f32::min);

    let enemy_hp_sum: f32 = enemies.iter().map(|u| u.hp.max(0) as f32).sum();
    let enemy_max_hp: f32 = enemies.iter().map(|u| u.max_hp.max(1) as f32).sum();
    let enemy_avg_hp_pct = if enemy_max_hp > 0.0 { enemy_hp_sum / enemy_max_hp } else { 0.0 };
    let enemy_min_hp_pct = enemies
        .iter()
        .map(|u| u.hp.max(0) as f32 / u.max_hp.max(1) as f32)
        .fold(1.0_f32, f32::min);

    f.extend_from_slice(&[
        hero_avg_hp_pct,
        hero_min_hp_pct,
        enemy_avg_hp_pct,
        enemy_min_hp_pct,
        heroes.len() as f32 / 8.0,
        enemies.len() as f32 / 8.0,
        hero_avg_hp_pct,
        enemy_avg_hp_pct,
    ]);

    // DPS/HPS availability (6)
    let hero_atk_ready = heroes.iter().filter(|u| u.cooldown_remaining_ms == 0).count();
    let hero_abi_ready = heroes.iter().filter(|u| u.ability_cooldown_remaining_ms == 0).count();
    let hero_heal_ready = heroes
        .iter()
        .filter(|u| {
            u.heal_cooldown_remaining_ms == 0
                && roles.get(&u.id) == Some(&Role::Healer)
        })
        .count();
    let enemy_atk_ready = enemies.iter().filter(|u| u.cooldown_remaining_ms == 0).count();
    let enemy_abi_ready = enemies.iter().filter(|u| u.ability_cooldown_remaining_ms == 0).count();
    let enemy_heal_ready = enemies
        .iter()
        .filter(|u| {
            u.heal_cooldown_remaining_ms == 0
                && roles.get(&u.id) == Some(&Role::Healer)
        })
        .count();

    let hn = heroes.len().max(1) as f32;
    let en = enemies.len().max(1) as f32;
    f.extend_from_slice(&[
        hero_atk_ready as f32 / hn,
        hero_abi_ready as f32 / hn,
        hero_heal_ready as f32 / hn,
        enemy_atk_ready as f32 / en,
        enemy_abi_ready as f32 / en,
        enemy_heal_ready as f32 / en,
    ]);

    // CC availability (4)
    let hero_cc_ready = heroes
        .iter()
        .filter(|u| u.control_cooldown_remaining_ms == 0 && roles.get(&u.id) == Some(&Role::Tank))
        .count();
    let enemy_cc_ready = enemies
        .iter()
        .filter(|u| u.control_cooldown_remaining_ms == 0 && roles.get(&u.id) == Some(&Role::Tank))
        .count();
    let hero_controlled = heroes.iter().filter(|u| u.control_remaining_ms > 0).count();
    let enemy_controlled = enemies.iter().filter(|u| u.control_remaining_ms > 0).count();

    f.extend_from_slice(&[
        hero_cc_ready as f32 / hn,
        enemy_cc_ready as f32 / en,
        hero_controlled as f32 / hn,
        enemy_controlled as f32 / en,
    ]);

    // Focus target features (4)
    let mut enemy_by_hp: Vec<_> = enemies.iter().collect();
    enemy_by_hp.sort_by(|a, b| {
        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(weakest) = enemy_by_hp.first() {
        let hp_pct = weakest.hp as f32 / weakest.max_hp.max(1) as f32;
        let is_healer = roles.get(&weakest.id) == Some(&Role::Healer);
        let is_controlled = weakest.control_remaining_ms > 0;
        f.extend_from_slice(&[
            hp_pct,
            if is_healer { 1.0 } else { 0.0 },
            if is_controlled { 1.0 } else { 0.0 },
            enemies.len() as f32 / 8.0,
        ]);
    } else {
        f.extend_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    }

    // Current personality weights (6) — average across hero units
    let mut avg_personality = [0.5_f32; 6];
    let mut p_count = 0;
    for u in &heroes {
        if let Some(p) = current_personality.get(&u.id) {
            avg_personality[0] += p.aggression;
            avg_personality[1] += p.risk_tolerance;
            avg_personality[2] += p.discipline;
            avg_personality[3] += p.control_bias;
            avg_personality[4] += p.altruism;
            avg_personality[5] += p.patience;
            p_count += 1;
        }
    }
    if p_count > 0 {
        for v in &mut avg_personality {
            // We started at 0.5, so subtract the initial 0.5 and divide
            *v = (*v - 0.5) / p_count as f32;
        }
    }
    f.extend_from_slice(&avg_personality);

    // Formation one-hot (3)
    f.push(if current_formation == FormationMode::Hold { 1.0 } else { 0.0 });
    f.push(if current_formation == FormationMode::Advance { 1.0 } else { 0.0 });
    f.push(if current_formation == FormationMode::Retreat { 1.0 } else { 0.0 });

    // Game phase indicators (4)
    let tick = state.tick as f32;
    f.push(tick / 320.0);
    f.push(if tick < 50.0 { 1.0 } else { 0.0 });
    f.push(if (50.0..200.0).contains(&tick) { 1.0 } else { 0.0 });
    f.push(if tick >= 200.0 { 1.0 } else { 0.0 });

    // Role composition (5)
    let hero_tanks = heroes.iter().filter(|u| roles.get(&u.id) == Some(&Role::Tank)).count();
    let hero_dps = heroes.iter().filter(|u| roles.get(&u.id) == Some(&Role::Dps)).count();
    let hero_healers = heroes.iter().filter(|u| roles.get(&u.id) == Some(&Role::Healer)).count();
    let enemy_tanks = enemies.iter().filter(|u| roles.get(&u.id) == Some(&Role::Tank)).count();
    let enemy_healers = enemies.iter().filter(|u| roles.get(&u.id) == Some(&Role::Healer)).count();

    f.extend_from_slice(&[
        hero_tanks as f32 / hn,
        hero_dps as f32 / hn,
        hero_healers as f32 / hn,
        enemy_tanks as f32 / en,
        enemy_healers as f32 / en,
    ]);

    // ===================================================================
    // Spatial features (20) — appended after the 40 aggregate features
    // ===================================================================

    // --- Distance & Positioning (6) ---

    // [41] Avg hero-to-nearest-enemy distance / 10.0
    // [42] Min hero-to-nearest-enemy distance (frontline gap) / 10.0
    let mut sum_nearest = 0.0_f32;
    let mut min_nearest = f32::MAX;
    for h in &heroes {
        let nearest = enemies
            .iter()
            .map(|e| distance(h.position, e.position))
            .fold(f32::MAX, f32::min);
        sum_nearest += nearest;
        min_nearest = min_nearest.min(nearest);
    }
    let avg_nearest = if !heroes.is_empty() { sum_nearest / heroes.len() as f32 } else { 0.0 };
    if min_nearest == f32::MAX { min_nearest = 0.0; }
    f.push(avg_nearest / 10.0);
    f.push(min_nearest / 10.0);

    // [43] Hero centroid X / 20.0
    // [44] Hero centroid Y / 20.0
    let hero_cx = if !heroes.is_empty() {
        heroes.iter().map(|u| u.position.x).sum::<f32>() / heroes.len() as f32
    } else { 0.0 };
    let hero_cy = if !heroes.is_empty() {
        heroes.iter().map(|u| u.position.y).sum::<f32>() / heroes.len() as f32
    } else { 0.0 };
    f.push(hero_cx / 20.0);
    f.push(hero_cy / 20.0);

    // [45] Inter-team centroid distance / 20.0
    let enemy_cx = if !enemies.is_empty() {
        enemies.iter().map(|u| u.position.x).sum::<f32>() / enemies.len() as f32
    } else { 0.0 };
    let enemy_cy = if !enemies.is_empty() {
        enemies.iter().map(|u| u.position.y).sum::<f32>() / enemies.len() as f32
    } else { 0.0 };
    let centroid_dist = ((hero_cx - enemy_cx).powi(2) + (hero_cy - enemy_cy).powi(2)).sqrt();
    f.push(centroid_dist / 20.0);

    // [46] Healer distance to hero centroid / 10.0
    let healer_dist = heroes
        .iter()
        .filter(|u| roles.get(&u.id) == Some(&Role::Healer))
        .map(|u| ((u.position.x - hero_cx).powi(2) + (u.position.y - hero_cy).powi(2)).sqrt())
        .fold(0.0_f32, f32::max);
    f.push(healer_dist / 10.0);

    // --- Engagement & Range (4) ---

    // [47] Hero attack engagement ratio: heroes with any enemy in attack_range / count
    let hero_atk_engaged = heroes
        .iter()
        .filter(|h| enemies.iter().any(|e| distance(h.position, e.position) <= h.attack_range))
        .count();
    f.push(hero_atk_engaged as f32 / hn);

    // [48] Hero ability engagement ratio: heroes with any enemy in ability_range / count
    let hero_abi_engaged = heroes
        .iter()
        .filter(|h| enemies.iter().any(|e| distance(h.position, e.position) <= h.ability_range))
        .count();
    f.push(hero_abi_engaged as f32 / hn);

    // [49] Enemy attack engagement ratio: enemies with any hero in their attack_range / count
    let enemy_atk_engaged = enemies
        .iter()
        .filter(|e| heroes.iter().any(|h| distance(e.position, h.position) <= e.attack_range))
        .count();
    f.push(enemy_atk_engaged as f32 / en);

    // [50] Focus target distance to nearest hero / 10.0
    let focus_dist = if let Some(weakest) = enemy_by_hp.first() {
        heroes
            .iter()
            .map(|h| distance(h.position, weakest.position))
            .fold(f32::MAX, f32::min)
    } else {
        0.0
    };
    f.push(if focus_dist == f32::MAX { 0.0 } else { focus_dist / 10.0 });

    // --- Clustering & Spread (4) ---

    // [51] Hero team spread (std dev of positions) / 5.0
    let hero_spread = if heroes.len() > 1 {
        let var_x = heroes.iter().map(|u| (u.position.x - hero_cx).powi(2)).sum::<f32>() / heroes.len() as f32;
        let var_y = heroes.iter().map(|u| (u.position.y - hero_cy).powi(2)).sum::<f32>() / heroes.len() as f32;
        (var_x + var_y).sqrt()
    } else { 0.0 };
    f.push(hero_spread / 5.0);

    // [52] Enemy team spread / 5.0
    let enemy_spread = if enemies.len() > 1 {
        let var_x = enemies.iter().map(|u| (u.position.x - enemy_cx).powi(2)).sum::<f32>() / enemies.len() as f32;
        let var_y = enemies.iter().map(|u| (u.position.y - enemy_cy).powi(2)).sum::<f32>() / enemies.len() as f32;
        (var_x + var_y).sqrt()
    } else { 0.0 };
    f.push(enemy_spread / 5.0);

    // [53] Enemy Y-span / 5.0 (mirrors advanced.rs tight-cluster detection)
    let enemy_y_span = if !enemies.is_empty() {
        let min_y = enemies.iter().map(|u| u.position.y).fold(f32::MAX, f32::min);
        let max_y = enemies.iter().map(|u| u.position.y).fold(f32::MIN, f32::max);
        max_y - min_y
    } else { 0.0 };
    f.push(enemy_y_span / 5.0);

    // [54] Local pressure ratio at frontline / 5.0
    // Find the frontline hero (closest to any enemy), count enemies within 3.4 of it,
    // count allies in same radius, ratio = enemies / max(allies, 1)
    let frontline_pressure = if !heroes.is_empty() && !enemies.is_empty() {
        // Find hero closest to any enemy
        let frontline = heroes
            .iter()
            .min_by(|a, b| {
                let da = enemies.iter().map(|e| distance(a.position, e.position)).fold(f32::MAX, f32::min);
                let db = enemies.iter().map(|e| distance(b.position, e.position)).fold(f32::MAX, f32::min);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let enemies_near = enemies.iter().filter(|e| distance(frontline.position, e.position) <= 3.4).count();
        let allies_near = heroes.iter().filter(|h| distance(frontline.position, h.position) <= 3.4).count();
        enemies_near as f32 / allies_near.max(1) as f32
    } else { 0.0 };
    f.push(frontline_pressure / 5.0);

    // --- Spatial Role Compliance (3) ---

    // [55] Tank frontline ratio: tanks that are closest hero to any enemy / tank count
    let tank_frontline = if hero_tanks > 0 {
        let tanks_at_front = heroes
            .iter()
            .filter(|u| roles.get(&u.id) == Some(&Role::Tank))
            .filter(|tank| {
                // Is this tank the closest hero to any enemy?
                enemies.iter().any(|e| {
                    let tank_dist = distance(tank.position, e.position);
                    heroes.iter().all(|h| distance(h.position, e.position) >= tank_dist - 0.01)
                })
            })
            .count();
        tanks_at_front as f32 / hero_tanks as f32
    } else { 0.0 };
    f.push(tank_frontline);

    // [56] Healer rear ratio: healers behind hero centroid / healer count
    // "behind" = further from enemy centroid than hero centroid
    let healer_rear = if hero_healers > 0 {
        let healers_behind = heroes
            .iter()
            .filter(|u| roles.get(&u.id) == Some(&Role::Healer))
            .filter(|healer| {
                let healer_to_enemy = ((healer.position.x - enemy_cx).powi(2) + (healer.position.y - enemy_cy).powi(2)).sqrt();
                healer_to_enemy > centroid_dist
            })
            .count();
        healers_behind as f32 / hero_healers as f32
    } else { 0.0 };
    f.push(healer_rear);

    // [57] DPS ability engagement ratio: DPS with any enemy in ability_range / DPS count
    let dps_engaged = if hero_dps > 0 {
        let engaged = heroes
            .iter()
            .filter(|u| roles.get(&u.id) == Some(&Role::Dps))
            .filter(|d| enemies.iter().any(|e| distance(d.position, e.position) <= d.ability_range))
            .count();
        engaged as f32 / hero_dps as f32
    } else { 0.0 };
    f.push(dps_engaged);

    // --- Threat Geometry (3) ---

    // [58] Max enemies threatening one hero / 4.0
    let max_threats = heroes
        .iter()
        .map(|h| enemies.iter().filter(|e| distance(e.position, h.position) <= e.attack_range).count())
        .max()
        .unwrap_or(0);
    f.push(max_threats as f32 / 4.0);

    // [59] Hero casting count / hn
    let hero_casting = heroes.iter().filter(|u| u.casting.is_some()).count();
    f.push(hero_casting as f32 / hn);

    // [60] Enemy casting count / en
    let enemy_casting = enemies.iter().filter(|u| u.casting.is_some()).count();
    f.push(enemy_casting as f32 / en);

    f
}

impl StudentMLP {
    /// Extract features from game state and run inference in one call.
    pub fn decide(
        &self,
        state: &SimState,
        roles: &HashMap<u32, Role>,
        current_personality: &HashMap<u32, PersonalityProfile>,
        current_formation: FormationMode,
    ) -> StudentOutput {
        let features = extract_features(state, roles, current_personality, current_formation);
        self.predict(&features)
    }
}
