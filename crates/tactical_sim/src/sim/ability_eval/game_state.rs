use crate::sim::{distance, is_alive, SimState, Team, UnitState};
use crate::effects::effect_enum::Effect;
use crate::goap::spatial::VisibilityMap;
use crate::pathing::GridNav;
/// Compute DPS from attack damage and cooldown.
pub(crate) fn unit_dps(u: &UnitState) -> f32 {
    if u.attack_cooldown_ms > 0 {
        u.attack_damage as f32 / (u.attack_cooldown_ms as f32 / 1000.0)
    } else {
        0.0
    }
}

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Rich game state representation for entity encoder pre-training
// ---------------------------------------------------------------------------
//
// 7 entity slots × 34 features = 238 floats total.
// Entity order: [self, enemy0, enemy1, enemy2, ally0, ally1, ally2]
// Entity types: 0=self, 1=enemy, 2=ally
//
// Per-entity features (34):
//   -- Vitals --
//   0: hp_pct
//   1: shield_pct (shield_hp / max_hp)
//   2: resource_pct
//   3: armor / 200
//   4: magic_resist / 200
//   -- Position / terrain --
//   5: position_x / 20 (absolute, centered)
//   6: position_y / 20
//   7: distance_from_caster / 10 (0 for self)
//   8: cover_bonus
//   9: elevation / 5
//  10: n_hostile_zones_nearby / 3
//  11: n_friendly_zones_nearby / 3
//  -- Combat stats --
//  12: auto_dps / 30
//  13: attack_range / 10
//  14: attack_cd_remaining_pct (0 = ready, 1 = full CD)
//  -- Ability readiness (strongest ability) --
//  15: ability_damage / 50
//  16: ability_range / 10
//  17: ability_cd_remaining_pct
//  -- Healing --
//  18: heal_amount / 50
//  19: heal_range / 10
//  20: heal_cd_remaining_pct
//  -- CC capability --
//  21: control_range / 10
//  22: control_duration / 2000
//  23: control_cd_remaining_pct
//  -- Current state --
//  24: is_casting (0/1)
//  25: cast_progress (0-1, how far through current cast)
//  26: cc_remaining / 2000 (how long until CC wears off)
//  27: move_speed / 5
//  -- Cumulative --
//  28: total_damage_done / 1000
//  29: exists (1.0 if slot occupied, 0.0 if padding)
//  -- Spatial summary (V6) --
//  30: visible_corner_count / 16
//  31: avg_passage_width (normalized)
//  32: min_passage_width (normalized)
//  33: avg_corner_distance / 20
//
// ---------------------------------------------------------------------------
// Context features: 4 floats — situational awareness
// ---------------------------------------------------------------------------
//
//   0: n_enemies_nearby / 8  — enemies within range 5.0
//   1: n_allies_nearby / 8   — allies within range 5.0
//   2: n_enemies_total / 8   — all living enemies
//   3: n_allies_total / 8    — all living allies (excluding self)
//
// ---------------------------------------------------------------------------
// Threat slots: 8 slots × 8 features = 64 floats
// ---------------------------------------------------------------------------
//
// Threats are incoming dangers relative to the unit being encoded.
// Sources: in-flight projectiles, hostile zones, enemy casts targeting this
// unit or its position, ground-target ability casts (warning zones).
//
// Sorted by urgency (time_to_impact ascending).
// Up to 8 threats tracked (supports large team fights).
//
// Per-threat features (8):
//   0: dx / 10       — relative x offset from unit to threat impact point
//   1: dy / 10       — relative y offset from unit to threat impact point
//   2: distance / 10 — distance from unit to threat impact point
//   3: radius / 5    — AoE radius (0 for single-target projectiles)
//   4: time_to_impact / 2000 — ms until damage lands (cast remaining or travel time)
//   5: damage_ratio  — incoming_damage / unit.hp (>1.0 = lethal)
//   6: has_cc        — 0/1, whether threat includes stun/root/silence
//   7: exists        — 1.0 if slot occupied, 0.0 if padding

pub const ENTITY_FEATURE_DIM: usize = 34;
pub const MAX_ENEMIES: usize = 3;
pub const MAX_ALLIES: usize = 3;
pub const NUM_ENTITY_SLOTS: usize = 1 + MAX_ENEMIES + MAX_ALLIES; // 7

pub const THREAT_FEATURE_DIM: usize = 8;
pub const POSITION_FEATURE_DIM: usize = 8;
pub const MAX_POSITION_TOKENS: usize = 8;

pub(super) const EMPTY_ENTITY: [f32; ENTITY_FEATURE_DIM] = [0.0; ENTITY_FEATURE_DIM];
pub(super) const EMPTY_THREAT: [f32; THREAT_FEATURE_DIM] = [0.0; THREAT_FEATURE_DIM];

/// Extract 34-dim entity features with spatial features zeroed.
/// Use `rich_entity_features_spatial()` to fill in the 4 spatial features.
pub(super) fn rich_entity_features(
    state: &SimState,
    caster: &UnitState,
    unit: &UnitState,
    is_self: bool,
) -> [f32; ENTITY_FEATURE_DIM] {
    let max_hp = unit.max_hp.max(1) as f32;

    // Zone proximity
    let hostile_zones = state.zones.iter()
        .filter(|z| z.source_team != unit.team && distance(unit.position, z.position) < 3.0)
        .count();
    let friendly_zones = state.zones.iter()
        .filter(|z| z.source_team == unit.team && distance(unit.position, z.position) < 3.0)
        .count();

    // Cast progress: remaining_ms normalized (lower = closer to firing)
    let (is_casting, cast_remaining_norm) = match &unit.casting {
        Some(cs) => (1.0, cs.remaining_ms as f32 / 2000.0),
        None => (0.0, 0.0),
    };

    // Cooldown remaining fractions
    let attack_cd_pct = if unit.attack_cooldown_ms > 0 {
        unit.cooldown_remaining_ms as f32 / unit.attack_cooldown_ms as f32
    } else { 0.0 };

    // Derive ability/heal/CC stats from DSL abilities (or legacy fields as fallback)
    let abil = summarize_abilities(unit);

    [
        // Vitals (0-4)
        unit.hp as f32 / max_hp,
        unit.shield_hp as f32 / max_hp,
        if unit.max_resource > 0 { unit.resource as f32 / unit.max_resource as f32 } else { 1.0 },
        unit.armor / 200.0,
        unit.magic_resist / 200.0,
        // Position / terrain (5-11)
        unit.position.x / 20.0,
        unit.position.y / 20.0,
        if is_self { 0.0 } else { distance(caster.position, unit.position) / 10.0 },
        unit.cover_bonus,
        unit.elevation / 5.0,
        hostile_zones as f32 / 3.0,
        friendly_zones as f32 / 3.0,
        // Combat stats (12-14)
        unit_dps(unit) / 30.0,
        unit.attack_range / 10.0,
        attack_cd_pct,
        // Ability (15-17)
        abil.ability_damage / 50.0,
        abil.ability_range / 10.0,
        abil.ability_cd_pct,
        // Healing (18-20)
        abil.heal_amount / 50.0,
        abil.heal_range / 10.0,
        abil.heal_cd_pct,
        // CC capability (21-23)
        abil.control_range / 10.0,
        abil.control_duration_ms / 2000.0,
        abil.control_cd_pct,
        // Current state (24-27)
        is_casting,
        cast_remaining_norm,
        unit.control_remaining_ms as f32 / 2000.0,
        unit.move_speed_per_sec / 5.0,
        // Cumulative (28-29)
        unit.total_damage_done as f32 / 1000.0,
        1.0, // exists
        // Spatial summary (30-33) — zeroed; use rich_entity_features_spatial() to fill
        0.0,
        0.0,
        0.0,
        0.0,
    ]
}

/// Extract 34-dim entity features with spatial summary from VisibilityMap.
pub(super) fn rich_entity_features_spatial(
    state: &SimState,
    caster: &UnitState,
    unit: &UnitState,
    is_self: bool,
    vis_map: &VisibilityMap,
    nav: &GridNav,
) -> [f32; ENTITY_FEATURE_DIM] {
    let mut feats = rich_entity_features(state, caster, unit, is_self);
    let summary = vis_map.visibility_summary(nav, unit.position);
    feats[30] = summary.visible_corner_count as f32 / 16.0;
    feats[31] = (summary.avg_passage_width / 10.0).min(1.0);
    feats[32] = if summary.min_passage_width < f32::MAX {
        (summary.min_passage_width / 10.0).min(1.0)
    } else {
        0.0
    };
    feats[33] = (summary.avg_corner_distance / 20.0).min(1.0);
    feats
}

/// Summary of a unit's strongest ability/heal/CC derived from AbilitySlot DSL effects.
pub(super) struct AbilitySummary {
    /// Highest single-hit damage from any damage ability
    pub ability_damage: f32,
    /// Range of that ability
    pub ability_range: f32,
    /// Cooldown fraction remaining (0 = ready) of strongest damage ability
    pub ability_cd_pct: f32,
    /// Highest single-hit heal from any heal ability
    pub heal_amount: f32,
    /// Range of that heal ability
    pub heal_range: f32,
    /// Cooldown fraction remaining of strongest heal ability
    pub heal_cd_pct: f32,
    /// Range of strongest CC ability
    pub control_range: f32,
    /// Duration of strongest CC (stun/root/silence)
    pub control_duration_ms: f32,
    /// Cooldown fraction remaining of strongest CC ability
    pub control_cd_pct: f32,
}

/// Scan a unit's abilities to extract strongest damage/heal/CC stats.
///
/// Falls back to legacy flat fields if the unit has no DSL abilities
/// (e.g. old-style PvE enemies).
pub(super) fn summarize_abilities(unit: &UnitState) -> AbilitySummary {
    let mut best_dmg = 0i32;
    let mut dmg_range = 0.0f32;
    let mut dmg_cd_pct = 0.0f32;

    let mut best_heal = 0i32;
    let mut heal_range = 0.0f32;
    let mut heal_cd_pct = 0.0f32;

    let mut best_cc_dur = 0u32;
    let mut cc_range = 0.0f32;
    let mut cc_cd_pct = 0.0f32;

    for slot in &unit.abilities {
        let hint = slot.def.ai_hint.as_str();
        let cd_pct = if slot.def.cooldown_ms > 0 {
            slot.cooldown_remaining_ms as f32 / slot.def.cooldown_ms as f32
        } else {
            0.0
        };

        match hint {
            "damage" | "damage_unit" | "damage_aoe" => {
                let dmg = max_damage_from_effects(&slot.def.effects);
                if dmg > best_dmg {
                    best_dmg = dmg;
                    dmg_range = slot.def.range;
                    dmg_cd_pct = cd_pct;
                }
            }
            "heal" | "heal_unit" | "heal_aoe" => {
                let heal = max_heal_from_effects(&slot.def.effects);
                if heal > best_heal {
                    best_heal = heal;
                    heal_range = slot.def.range;
                    heal_cd_pct = cd_pct;
                }
            }
            "control" | "cc" | "crowd_control" => {
                let dur = max_cc_duration_from_effects(&slot.def.effects);
                if dur > best_cc_dur {
                    best_cc_dur = dur;
                    cc_range = slot.def.range;
                    cc_cd_pct = cd_pct;
                }
            }
            _ => {
                // For untagged abilities, check effects for any damage/heal/CC
                let dmg = max_damage_from_effects(&slot.def.effects);
                if dmg > best_dmg {
                    best_dmg = dmg;
                    dmg_range = slot.def.range;
                    dmg_cd_pct = cd_pct;
                }
                let heal = max_heal_from_effects(&slot.def.effects);
                if heal > best_heal {
                    best_heal = heal;
                    heal_range = slot.def.range;
                    heal_cd_pct = cd_pct;
                }
                let dur = max_cc_duration_from_effects(&slot.def.effects);
                if dur > best_cc_dur {
                    best_cc_dur = dur;
                    cc_range = slot.def.range;
                    cc_cd_pct = cd_pct;
                }
            }
        }
    }

    // Fall back to legacy flat fields if no DSL abilities found
    if unit.abilities.is_empty() {
        let ability_cd_pct = if unit.ability_cooldown_ms > 0 {
            unit.ability_cooldown_remaining_ms as f32 / unit.ability_cooldown_ms as f32
        } else { 0.0 };
        let legacy_heal_cd = if unit.heal_cooldown_ms > 0 {
            unit.heal_cooldown_remaining_ms as f32 / unit.heal_cooldown_ms as f32
        } else { 0.0 };
        let legacy_cc_cd = if unit.control_cooldown_ms > 0 {
            unit.control_cooldown_remaining_ms as f32 / unit.control_cooldown_ms as f32
        } else { 0.0 };

        return AbilitySummary {
            ability_damage: unit.ability_damage as f32,
            ability_range: unit.ability_range,
            ability_cd_pct,
            heal_amount: unit.heal_amount as f32,
            heal_range: unit.heal_range,
            heal_cd_pct: legacy_heal_cd,
            control_range: unit.control_range,
            control_duration_ms: unit.control_duration_ms as f32,
            control_cd_pct: legacy_cc_cd,
        };
    }

    AbilitySummary {
        ability_damage: best_dmg as f32,
        ability_range: dmg_range,
        ability_cd_pct: dmg_cd_pct,
        heal_amount: best_heal as f32,
        heal_range,
        heal_cd_pct,
        control_range: cc_range,
        control_duration_ms: best_cc_dur as f32,
        control_cd_pct: cc_cd_pct,
    }
}

pub(super) fn max_damage_from_effects(effects: &[crate::effects::types::ConditionalEffect]) -> i32 {
    effects.iter().filter_map(|ce| match &ce.effect {
        Effect::Damage { amount, .. } => Some(*amount),
        _ => None,
    }).max().unwrap_or(0)
}

pub(super) fn max_heal_from_effects(effects: &[crate::effects::types::ConditionalEffect]) -> i32 {
    effects.iter().filter_map(|ce| match &ce.effect {
        Effect::Heal { amount, .. } => Some(*amount),
        _ => None,
    }).max().unwrap_or(0)
}

pub(super) fn max_cc_duration_from_effects(effects: &[crate::effects::types::ConditionalEffect]) -> u32 {
    effects.iter().filter_map(|ce| match &ce.effect {
        Effect::Stun { duration_ms } => Some(*duration_ms),
        Effect::Root { duration_ms } => Some(*duration_ms),
        Effect::Silence { duration_ms } => Some(*duration_ms),
        _ => None,
    }).max().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Outcome prediction dataset
// ---------------------------------------------------------------------------

/// A game state snapshot labeled with the final fight outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSample {
    /// Flat entity features: [self(34), enemy0(34)..enemy2(34), ally0(34)..ally2(34)]
    pub game_state: Vec<f32>,
    /// 1.0 = hero team wins, 0.0 = enemy team wins
    pub hero_wins: f32,
    /// Hero team HP fraction at fight end (0-1, richer signal than binary)
    pub hero_hp_remaining: f32,
    /// How far through the fight (0=start, 1=end)
    pub fight_progress: f32,
    /// Scenario name for debugging
    pub scenario: String,
    pub tick: u64,
}

/// Generate outcome prediction training data from a simulation run.
///
/// Runs the sim to completion, then walks back through snapshots labeling
/// each with the final outcome. Samples every `sample_interval` ticks
/// to control dataset size.
pub fn generate_outcome_dataset(
    initial_sim: SimState,
    initial_squad_ai: crate::squad::SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    sample_interval: u64,
) -> Vec<OutcomeSample> {
    use crate::sim::{step, FIXED_TICK_MS};
    use crate::squad::generate_intents;

    let mut sim = initial_sim;
    let mut squad_ai = initial_squad_ai;
    let mut snapshots: Vec<(u64, Vec<Vec<f32>>)> = Vec::new();

    // Run simulation, collecting snapshots
    for tick in 0..max_ticks {
        let intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);

        // Sample game state from each hero's perspective (34-dim per entity slot)
        if tick % sample_interval == 0 {
            let hero_states: Vec<Vec<f32>> = sim.units.iter()
                .filter(|u| u.team == Team::Hero && is_alive(u))
                .map(|u| {
                    let mut features = Vec::with_capacity(NUM_ENTITY_SLOTS * ENTITY_FEATURE_DIM);
                    features.extend_from_slice(&rich_entity_features(&sim, u, u, true));
                    let mut enemies: Vec<&UnitState> = sim.units.iter()
                        .filter(|e| e.team != u.team && is_alive(e))
                        .collect();
                    enemies.sort_by(|a, b| {
                        distance(u.position, a.position)
                            .partial_cmp(&distance(u.position, b.position))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for i in 0..MAX_ENEMIES {
                        if let Some(enemy) = enemies.get(i) {
                            features.extend_from_slice(&rich_entity_features(&sim, u, enemy, false));
                        } else {
                            features.extend_from_slice(&EMPTY_ENTITY);
                        }
                    }
                    let mut allies: Vec<&UnitState> = sim.units.iter()
                        .filter(|a| a.team == u.team && is_alive(a) && a.id != u.id)
                        .collect();
                    allies.sort_by(|a, b| {
                        let ha = a.hp as f32 / a.max_hp.max(1) as f32;
                        let hb = b.hp as f32 / b.max_hp.max(1) as f32;
                        ha.partial_cmp(&hb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for i in 0..MAX_ALLIES {
                        if let Some(ally) = allies.get(i) {
                            features.extend_from_slice(&rich_entity_features(&sim, u, ally, false));
                        } else {
                            features.extend_from_slice(&EMPTY_ENTITY);
                        }
                    }
                    features
                })
                .collect();
            if !hero_states.is_empty() {
                snapshots.push((tick, hero_states));
            }
        }

        let (new_sim, _) = step(sim, &intents, FIXED_TICK_MS);
        sim = new_sim;

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if enemies_alive == 0 || heroes_alive == 0 {
            break;
        }
    }

    // Determine outcome
    let final_hero_hp: f32 = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| (u.hp.max(0) as f32) / u.max_hp.max(1) as f32)
        .sum::<f32>()
        / sim.units.iter().filter(|u| u.team == Team::Hero).count().max(1) as f32;
    let final_enemy_hp: f32 = sim.units.iter()
        .filter(|u| u.team == Team::Enemy)
        .map(|u| (u.hp.max(0) as f32) / u.max_hp.max(1) as f32)
        .sum::<f32>()
        / sim.units.iter().filter(|u| u.team == Team::Enemy).count().max(1) as f32;
    let hero_wins = if final_hero_hp > final_enemy_hp { 1.0 } else { 0.0 };
    let total_ticks = sim.tick.max(1) as f32;

    // Label each snapshot with outcome
    let mut samples = Vec::new();
    for (tick, hero_states) in snapshots {
        for gs in hero_states {
            samples.push(OutcomeSample {
                game_state: gs,
                hero_wins,
                hero_hp_remaining: final_hero_hp,
                fight_progress: tick as f32 / total_ticks,
                scenario: scenario_name.to_string(),
                tick,
            });
        }
    }

    samples
}

/// Write outcome samples as JSONL.
pub fn write_outcome_dataset(
    samples: &[OutcomeSample],
    path: &std::path::Path,
) -> std::io::Result<()> {
    use std::io::Write;
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    for sample in samples {
        serde_json::to_writer(&mut writer, sample).unwrap();
        writeln!(writer)?;
    }
    Ok(())
}
