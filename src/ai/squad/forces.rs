use std::cmp::Ordering;

use crate::ai::core::{distance, SimState, SimVec2, UnitState};

use super::personality::Personality;
use super::state::{SquadBlackboard, TickContext};

// ---------------------------------------------------------------------------
// Forces -- what the SITUATION calls for (9 tactical impulses)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct RawForces {
    pub attack: f32,
    pub heal: f32,
    pub retreat: f32,
    pub control: f32,
    pub focus: f32,
    pub protect: f32,
    pub pursue: f32,
    pub regroup: f32,
    pub position: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DominantForce {
    Attack,
    Heal,
    Retreat,
    Control,
    Focus,
    Protect,
    Pursue,
    Regroup,
    Position,
}

/// Personality -> Force weight matrix.
/// Each row is a personality trait; columns are forces in order:
/// Attack, Heal, Retreat, Control, Focus, Protect, Pursue, Regroup, Position
const WEIGHT_MATRIX: [[f32; 9]; 7] = [
    // Aggression
    [ 0.8,  0.0, -0.2,  0.0,  0.2,  0.0,  0.6, -0.1,  0.0],
    // Compassion
    [ 0.0,  0.9,  0.1,  0.0,  0.0,  0.8,  0.0,  0.3,  0.0],
    // Caution
    [-0.2,  0.2,  0.8,  0.1,  0.0,  0.0, -0.2,  0.5,  0.4],
    // Discipline
    [ 0.1,  0.0,  0.0,  0.0,  0.9,  0.2,  0.0,  0.5,  0.2],
    // Cunning
    [ 0.2,  0.0,  0.0,  0.8,  0.2,  0.0,  0.3,  0.0,  0.5],
    // Tenacity
    [ 0.3,  0.0, -0.3,  0.0,  0.4,  0.0,  0.8, -0.1,  0.0],
    // Patience
    [-0.2,  0.1,  0.0,  0.3,  0.0,  0.1, -0.1,  0.2,  0.3],
];

pub(super) fn weighted_forces(raw: &RawForces, personality: &Personality) -> RawForces {
    let traits = [
        personality.aggression,
        personality.compassion,
        personality.caution,
        personality.discipline,
        personality.cunning,
        personality.tenacity,
        personality.patience,
    ];
    let raw_arr = [
        raw.attack, raw.heal, raw.retreat, raw.control, raw.focus,
        raw.protect, raw.pursue, raw.regroup, raw.position,
    ];
    let mut out = [0.0f32; 9];
    for (t_idx, &trait_val) in traits.iter().enumerate() {
        for (f_idx, &raw_val) in raw_arr.iter().enumerate() {
            out[f_idx] += trait_val * WEIGHT_MATRIX[t_idx][f_idx] * raw_val;
        }
    }
    RawForces {
        attack: out[0],
        heal: out[1],
        retreat: out[2],
        control: out[3],
        focus: out[4],
        protect: out[5],
        pursue: out[6],
        regroup: out[7],
        position: out[8],
    }
}

pub(super) fn dominant_force(wf: &RawForces) -> DominantForce {
    let forces = [
        (DominantForce::Attack, wf.attack),
        (DominantForce::Heal, wf.heal),
        (DominantForce::Retreat, wf.retreat),
        (DominantForce::Control, wf.control),
        (DominantForce::Focus, wf.focus),
        (DominantForce::Protect, wf.protect),
        (DominantForce::Pursue, wf.pursue),
        (DominantForce::Regroup, wf.regroup),
        (DominantForce::Position, wf.position),
    ];
    // Heal wins ties.
    forces
        .iter()
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    // Heal wins ties
                    if matches!(a.0, DominantForce::Heal) {
                        Ordering::Greater
                    } else if matches!(b.0, DominantForce::Heal) {
                        Ordering::Less
                    } else {
                        Ordering::Equal
                    }
                })
        })
        .map(|(f, _)| *f)
        .unwrap_or(DominantForce::Attack)
}

pub(super) fn compute_raw_forces(
    state: &SimState,
    unit: &UnitState,
    board: &SquadBlackboard,
    ctx: &TickContext,
    personality: &Personality,
) -> RawForces {
    let allies = ctx.allies(unit.team);
    let enemies = ctx.enemies_of(unit.team);

    if enemies.is_empty() {
        return RawForces::default();
    }

    let hp_pct = unit.hp as f32 / unit.max_hp.max(1) as f32;

    // --- Attack: base engagement impulse + enemies in range + number advantage ---
    let mut enemies_in_range = 0u32;
    let mut nearest_enemy_dist = f32::MAX;
    let range_threshold = unit.attack_range * 1.5;
    for &eid in enemies {
        if let Some(e) = ctx.unit(state, eid) {
            let d = distance(unit.position, e.position);
            if d < nearest_enemy_dist { nearest_enemy_dist = d; }
            if d <= range_threshold { enemies_in_range += 1; }
        }
    }
    let number_advantage = allies.len() as f32 / enemies.len().max(1) as f32;
    let attack = (5.0 + enemies_in_range as f32 * 2.0 + number_advantage * 2.0).min(10.0);

    // --- Heal: ally missing HP x heal ability readiness x triage urgency ---
    let has_hero_heal = unit.abilities.iter().any(|s| s.def.ai_hint == "heal");
    let can_heal = unit.heal_amount > 0 || has_hero_heal;
    let heal = if can_heal {
        let mut worst_ally_hp_pct = 1.0f32;
        for &aid in allies {
            if let Some(a) = ctx.unit(state, aid) {
                let pct = a.hp as f32 / a.max_hp.max(1) as f32;
                if pct < worst_ally_hp_pct { worst_ally_hp_pct = pct; }
            }
        }
        let urgency = (1.0 - worst_ally_hp_pct) * 10.0;
        let readiness = if has_hero_heal {
            if unit.abilities.iter().any(|s| s.def.ai_hint == "heal" && s.cooldown_remaining_ms == 0) {
                1.0
            } else {
                0.3
            }
        } else if unit.heal_cooldown_remaining_ms == 0 {
            1.0
        } else {
            0.3
        };
        urgency * readiness
    } else {
        0.0
    };

    // --- Retreat: own danger -- low HP%, enemy proximity when wounded ---
    let retreat = if hp_pct < 0.5 {
        let danger = (1.0 - hp_pct) * 8.0;
        let proximity_bonus = if nearest_enemy_dist < 3.0 { 3.0 } else { 0.0 };
        danger + proximity_bonus
    } else {
        0.0
    };

    // --- Control: CC ability ready + uncontrolled priority target ---
    let has_cc = unit.control_duration_ms > 0 || unit.abilities.iter().any(|s| s.def.ai_hint == "control");
    let control = if has_cc {
        let cc_ready = if unit.control_duration_ms > 0 {
            unit.control_cooldown_remaining_ms == 0
        } else {
            unit.abilities.iter().any(|s| s.def.ai_hint == "control" && s.cooldown_remaining_ms == 0)
        };
        if cc_ready {
            let uncontrolled_priority = enemies.iter()
                .any(|&eid| ctx.unit(state, eid).map_or(false, |e| e.control_remaining_ms == 0));
            if uncontrolled_priority { 7.0 } else { 0.0 }
        } else {
            0.0
        }
    } else {
        0.0
    };

    // --- Focus: blackboard focus target alive and wounded ---
    let focus = if let Some(focus_id) = board.focus_target {
        if let Some(ft) = ctx.unit(state, focus_id).filter(|u| u.hp > 0) {
            let ft_hp_pct = ft.hp as f32 / ft.max_hp.max(1) as f32;
            5.0 + (1.0 - ft_hp_pct) * 5.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    // --- Protect: fragile ally threatened by nearby enemy ---
    let mut protect = 0.0f32;
    for &aid in allies {
        if aid == unit.id { continue; }
        let Some(ally) = ctx.unit(state, aid) else { continue };
        let ally_hp_pct = ally.hp as f32 / ally.max_hp.max(1) as f32;
        if ally_hp_pct > 0.6 { continue; }
        let threat_nearby = enemies.iter()
            .any(|&eid| ctx.unit(state, eid).map_or(false, |e| distance(ally.position, e.position) < 3.0));
        if threat_nearby {
            let score = (1.0 - ally_hp_pct) * 8.0;
            if score > protect { protect = score; }
        }
    }

    // --- Pursue: wounded enemy fleeing or nearly dead ---
    let mut pursue = 0.0f32;
    for &eid in enemies {
        if let Some(e) = ctx.unit(state, eid) {
            let e_hp_pct = e.hp as f32 / e.max_hp.max(1) as f32;
            if e_hp_pct < 0.3 {
                let score = (1.0 - e_hp_pct) * 8.0;
                if score > pursue { pursue = score; }
            }
        }
    }

    // --- Regroup: isolation from allies ---
    let regroup = if allies.len() > 1 {
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut count = 0u32;
        for &aid in allies {
            if let Some(a) = ctx.unit(state, aid) {
                cx += a.position.x;
                cy += a.position.y;
                count += 1;
            }
        }
        if count > 0 {
            cx /= count as f32;
            cy /= count as f32;
            let dist_to_centroid = distance(unit.position, SimVec2 { x: cx, y: cy });
            if dist_to_centroid > 5.0 {
                (dist_to_centroid - 5.0).min(5.0)
            } else {
                0.0
            }
        } else {
            0.0
        }
    } else {
        0.0
    };

    // --- Position: not at preferred range (only when already in combat proximity) ---
    let profile = super::state::personality_movement_profile(personality, unit);
    let preferred = (profile.preferred_range_min + profile.preferred_range_max) * 0.5;
    let position = if nearest_enemy_dist < f32::MAX && nearest_enemy_dist < preferred * 3.0 {
        let range_diff = (nearest_enemy_dist - preferred).abs();
        if range_diff > 1.0 {
            (range_diff - 1.0).min(4.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    RawForces {
        attack,
        heal,
        retreat,
        control,
        focus,
        protect,
        pursue,
        regroup,
        position,
    }
}
