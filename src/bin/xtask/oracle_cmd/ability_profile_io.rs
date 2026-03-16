//! Outcome structures, vectorization, NPZ writing, and ability loading
//! for the ability profiler.
//!
//! Split from `ability_profile.rs` to keep files under 500 lines.

use std::collections::HashMap;

use bevy_game::ai::core::sim_vec2;
use bevy_game::ai::effects::{
    AbilityDef, AbilitySlot, StatusKind,
};
use bevy_game::ai::effects::dsl;
use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;
use bevy_game::ai::core::{SimState, SimVec2, Team, UnitState, UnitStore};

pub const MAX_HP: i32 = 100;
pub const CASTER_ID: u32 = 0;
pub const MAX_TARGETS: usize = 4;

// Condition grid
pub const HP_PCTS: &[f32] = &[0.2, 0.5, 0.8, 1.0];
pub const DISTANCES: &[f32] = &[1.0, 3.0, 5.0, 8.0];
pub const TARGET_COUNTS: &[usize] = &[1, 2, 4];
pub const ARMORS: &[f32] = &[0.0, 10.0, 25.0];

/// Condition vector for a single trial.
#[derive(Debug, Clone)]
pub struct TrialCondition {
    pub target_hp_pct: f32,
    pub distance: f32,
    pub n_targets: usize,
    pub armor: f32,
}

/// Per-target outcome delta recorded after ability resolves.
#[derive(Debug, Clone, Default)]
pub struct TargetOutcome {
    pub delta_hp: f32,
    pub delta_shield: f32,
    pub delta_x: f32,
    pub delta_y: f32,
    pub killed: bool,
    pub stun_dur: f32,
    pub slow_dur: f32,
    pub slow_factor: f32,
    pub root_dur: f32,
    pub silence_dur: f32,
    pub fear_dur: f32,
    pub taunt_dur: f32,
    pub blind_dur: f32,
    pub polymorph_dur: f32,
    pub suppress_dur: f32,
    pub grounded_dur: f32,
    pub charm_dur: f32,
    pub buff_dur: f32,
    pub debuff_dur: f32,
    pub dot_dur: f32,
    pub hot_dur: f32,
    pub shield_amount: f32,
    pub damage_modify_dur: f32,
}

/// Aggregated outcome for one trial.
#[derive(Debug, Clone)]
pub struct TrialOutcome {
    pub total_damage: f32,
    pub total_heal: f32,
    pub n_targets_hit: u32,
    pub n_targets_killed: u32,
    pub per_target: Vec<TargetOutcome>,
    pub caster: TargetOutcome,
}

/// A single profiling sample: condition + outcome for one ability trial.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProfileSample {
    pub ability_idx: u32,
    pub ability_name: String,
    pub condition: Vec<f32>,
    pub outcome: Vec<f32>,
}

impl TargetOutcome {
    pub fn has_any_status(&self) -> bool {
        self.stun_dur > 0.0 || self.slow_dur > 0.0 || self.root_dur > 0.0
            || self.silence_dur > 0.0 || self.fear_dur > 0.0 || self.taunt_dur > 0.0
            || self.blind_dur > 0.0 || self.polymorph_dur > 0.0 || self.suppress_dur > 0.0
            || self.grounded_dur > 0.0 || self.charm_dur > 0.0
            || self.buff_dur > 0.0 || self.debuff_dur > 0.0
            || self.dot_dur > 0.0 || self.hot_dur > 0.0
            || self.shield_amount > 0.0 || self.damage_modify_dur > 0.0
    }
}

/// Scan a unit's status effects and update peak durations in the outcome.
pub fn scan_status_effects(unit: &UnitState, out: &mut TargetOutcome) {
    if unit.control_remaining_ms > 0 {
        let dur_s = unit.control_remaining_ms as f32 / 1000.0;
        out.stun_dur = out.stun_dur.max(dur_s);
    }
    if unit.shield_hp > 0 {
        out.shield_amount = out.shield_amount.max(unit.shield_hp as f32);
    }
    for se in &unit.status_effects {
        let dur_s = se.remaining_ms as f32 / 1000.0;
        match &se.kind {
            StatusKind::Stun => out.stun_dur = out.stun_dur.max(dur_s),
            StatusKind::Slow { factor } => {
                out.slow_dur = out.slow_dur.max(dur_s);
                out.slow_factor = out.slow_factor.max(*factor);
            }
            StatusKind::Root => out.root_dur = out.root_dur.max(dur_s),
            StatusKind::Silence => out.silence_dur = out.silence_dur.max(dur_s),
            StatusKind::Fear { .. } => out.fear_dur = out.fear_dur.max(dur_s),
            StatusKind::Taunt { .. } => out.taunt_dur = out.taunt_dur.max(dur_s),
            StatusKind::Blind { .. } => out.blind_dur = out.blind_dur.max(dur_s),
            StatusKind::Polymorph => out.polymorph_dur = out.polymorph_dur.max(dur_s),
            StatusKind::Suppress => out.suppress_dur = out.suppress_dur.max(dur_s),
            StatusKind::Grounded => out.grounded_dur = out.grounded_dur.max(dur_s),
            StatusKind::Charm { .. } => out.charm_dur = out.charm_dur.max(dur_s),
            StatusKind::Buff { .. } => out.buff_dur = out.buff_dur.max(dur_s),
            StatusKind::Debuff { .. } => out.debuff_dur = out.debuff_dur.max(dur_s),
            StatusKind::Dot { .. } => out.dot_dur = out.dot_dur.max(dur_s),
            StatusKind::Hot { .. } => out.hot_dur = out.hot_dur.max(dur_s),
            StatusKind::DamageModify { .. } => out.damage_modify_dur = out.damage_modify_dur.max(dur_s),
            StatusKind::Shield { .. } => out.shield_amount = out.shield_amount.max(se.remaining_ms as f32),
            _ => {}
        }
    }
}

/// Per-target outcome dimension count.
pub const PER_TARGET_DIM: usize = 23;
/// Total outcome vector dimension.
pub const OUTCOME_DIM: usize = MAX_TARGETS * PER_TARGET_DIM + PER_TARGET_DIM + 4;

pub fn push_target_outcome(v: &mut Vec<f32>, t: &TargetOutcome) {
    v.push(t.delta_hp);
    v.push(t.delta_shield);
    v.push(t.delta_x);
    v.push(t.delta_y);
    v.push(if t.killed { 1.0 } else { 0.0 });
    v.push(t.stun_dur);
    v.push(t.slow_dur);
    v.push(t.slow_factor);
    v.push(t.root_dur);
    v.push(t.silence_dur);
    v.push(t.fear_dur);
    v.push(t.taunt_dur);
    v.push(t.blind_dur);
    v.push(t.polymorph_dur);
    v.push(t.suppress_dur);
    v.push(t.grounded_dur);
    v.push(t.charm_dur);
    v.push(t.buff_dur);
    v.push(t.debuff_dur);
    v.push(t.dot_dur);
    v.push(t.hot_dur);
    v.push(t.shield_amount);
    v.push(t.damage_modify_dur);
}

/// Flatten a TrialOutcome into a fixed-size outcome vector.
pub fn outcome_to_vec(outcome: &TrialOutcome) -> Vec<f32> {
    let mut v = Vec::with_capacity(OUTCOME_DIM);
    for i in 0..MAX_TARGETS {
        if let Some(t) = outcome.per_target.get(i) {
            push_target_outcome(&mut v, t);
        } else {
            v.extend_from_slice(&[0.0; PER_TARGET_DIM]);
        }
    }
    push_target_outcome(&mut v, &outcome.caster);
    v.push(outcome.total_damage);
    v.push(outcome.total_heal);
    v.push(outcome.n_targets_hit as f32);
    v.push(outcome.n_targets_killed as f32);
    v
}

/// Flatten a TrialCondition into a fixed-size condition vector.
pub fn condition_to_vec(cond: &TrialCondition) -> Vec<f32> {
    vec![cond.target_hp_pct, cond.distance, cond.n_targets as f32, cond.armor]
}

/// Build a hero unit with one ability attached.
pub fn caster_unit(ability: AbilityDef) -> UnitState {
    UnitState {
        id: CASTER_ID,
        team: Team::Hero,
        hp: MAX_HP / 2,
        max_hp: MAX_HP,
        position: sim_vec2(0.0, 0.0),
        move_speed_per_sec: 3.0,
        attack_damage: 10,
        attack_range: 1.4,
        attack_cooldown_ms: 700,
        attack_cast_time_ms: 300,
        cooldown_remaining_ms: 0,
        ability_damage: 0,
        ability_range: 0.0,
        ability_cooldown_ms: 0,
        ability_cast_time_ms: 0,
        ability_cooldown_remaining_ms: 0,
        heal_amount: 0,
        heal_range: 0.0,
        heal_cooldown_ms: 0,
        heal_cast_time_ms: 0,
        heal_cooldown_remaining_ms: 0,
        control_range: 0.0,
        control_duration_ms: 0,
        control_cooldown_ms: 0,
        control_cast_time_ms: 0,
        control_cooldown_remaining_ms: 0,
        control_remaining_ms: 0,
        casting: None,
        abilities: vec![AbilitySlot::new(ability)],
        passives: Vec::new(),
        status_effects: Vec::new(),
        shield_hp: 0,
        resistance_tags: HashMap::new(),
        state_history: std::collections::VecDeque::new(),
        channeling: None,
        resource: 0,
        max_resource: 0,
        resource_regen_per_sec: 0.0,
        owner_id: None,
        directed: false,
        armor: 0.0,
        magic_resist: 0.0,
        cover_bonus: 0.0,
        elevation: 0.0,
        total_healing_done: 0,
        total_damage_done: 0,
    }
}

/// Build a target unit at a given position with given HP and armor.
pub fn target_unit(id: u32, team: Team, pos: SimVec2, hp: i32, armor: f32) -> UnitState {
    UnitState {
        id,
        team,
        hp,
        max_hp: MAX_HP,
        position: pos,
        move_speed_per_sec: 3.0,
        attack_damage: 10,
        attack_range: 1.4,
        attack_cooldown_ms: 999999,
        attack_cast_time_ms: 300,
        cooldown_remaining_ms: 0,
        ability_damage: 0,
        ability_range: 0.0,
        ability_cooldown_ms: 0,
        ability_cast_time_ms: 0,
        ability_cooldown_remaining_ms: 0,
        heal_amount: 0,
        heal_range: 0.0,
        heal_cooldown_ms: 0,
        heal_cast_time_ms: 0,
        heal_cooldown_remaining_ms: 0,
        control_range: 0.0,
        control_duration_ms: 0,
        control_cooldown_ms: 0,
        control_cast_time_ms: 0,
        control_cooldown_remaining_ms: 0,
        control_remaining_ms: 0,
        casting: None,
        abilities: Vec::new(),
        passives: Vec::new(),
        status_effects: Vec::new(),
        shield_hp: 0,
        resistance_tags: HashMap::new(),
        state_history: std::collections::VecDeque::new(),
        channeling: None,
        resource: 0,
        max_resource: 0,
        resource_regen_per_sec: 0.0,
        owner_id: None,
        directed: false,
        armor,
        magic_resist: 0.0,
        cover_bonus: 0.0,
        elevation: 0.0,
        total_healing_done: 0,
        total_damage_done: 0,
    }
}

pub fn make_sim(units: Vec<UnitState>) -> SimState {
    SimState {
        tick: 0,
        rng_state: 42,
        units: UnitStore::new(units),
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

/// Recursively find all `.ability` files under a directory.
pub fn find_ability_files(dir: &str) -> Vec<std::path::PathBuf> {
    let mut result = Vec::new();
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, out);
                } else if path.extension().and_then(|e| e.to_str()) == Some("ability") {
                    out.push(path);
                }
            }
        }
    }
    walk(std::path::Path::new(dir), &mut result);
    result.sort();
    result
}

/// Load all unique abilities from dataset/abilities/, assets/hero_templates/, assets/lol_heroes/.
pub fn load_all_abilities() -> Vec<(String, AbilityDef, String)> {
    let mut abilities: Vec<(String, AbilityDef, String)> = Vec::new();
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut add_ability = |def: AbilityDef, dsl_text: String| {
        if seen_names.insert(def.name.clone()) {
            let name = def.name.clone();
            abilities.push((name, def, dsl_text));
        }
    };

    for path in find_ability_files("dataset/abilities") {
        if let Ok(content) = std::fs::read_to_string(&path) {
            match dsl::parse_abilities(&content) {
                Ok((defs, _)) => {
                    for def in defs {
                        let dsl_text = emit_ability_dsl(&def);
                        add_ability(def, dsl_text);
                    }
                }
                Err(e) => eprintln!("Warning: DSL parse error in {}: {e}", path.display()),
            }
        }
    }

    load_abilities_from_toml_dir("assets/hero_templates", &mut seen_names, &mut abilities);
    load_abilities_from_toml_dir("assets/lol_heroes", &mut seen_names, &mut abilities);

    eprintln!("Loaded {} unique abilities", abilities.len());
    abilities
}

fn load_abilities_from_toml_dir(
    dir: &str,
    seen: &mut std::collections::HashSet<String>,
    out: &mut Vec<(String, AbilityDef, String)>,
) {
    let dir_path = std::path::Path::new(dir);
    if !dir_path.is_dir() {
        return;
    }

    let mut paths: Vec<_> = std::fs::read_dir(dir_path)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
        .collect();
    paths.sort();

    for path in paths {
        if let Ok(content) = std::fs::read_to_string(&path) {
            let dsl_path = path.with_extension("ability");
            let dsl_content = std::fs::read_to_string(&dsl_path).ok();

            let toml_result = if let Some(ref dsl_str) = dsl_content {
                bevy_game::mission::hero_templates::parse_hero_toml_with_dsl(&content, Some(dsl_str))
            } else {
                bevy_game::mission::hero_templates::parse_hero_toml(&content)
            };

            if let Ok(hero) = toml_result {
                for def in hero.abilities {
                    if seen.insert(def.name.clone()) {
                        let dsl_text = emit_ability_dsl(&def);
                        out.push((def.name.clone(), def, dsl_text));
                    }
                }
            }
        }
    }
}

/// Write profile samples to NPZ format.
pub fn write_profile_npz(
    path: &std::path::Path,
    samples: &[ProfileSample],
    abilities: &[(String, AbilityDef, String)],
) {
    use ndarray::{Array1, Array2};
    use ndarray_npy::NpzWriter;

    let n = samples.len();
    if n == 0 { return; }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let file = std::fs::File::create(path).expect("Failed to create output npz");
    let mut npz = NpzWriter::new(file);

    let ability_ids: Array1<i32> = Array1::from_vec(
        samples.iter().map(|s| s.ability_idx as i32).collect(),
    );
    npz.add_array("ability_id", &ability_ids).unwrap();

    let cond_dim = 4;
    let conditions: Array2<f32> = Array2::from_shape_vec(
        (n, cond_dim),
        samples.iter().flat_map(|s| s.condition.iter().copied()).collect(),
    ).unwrap();
    npz.add_array("condition", &conditions).unwrap();

    let outcome_dim = OUTCOME_DIM;
    let outcomes: Array2<f32> = Array2::from_shape_vec(
        (n, outcome_dim),
        samples.iter().flat_map(|s| s.outcome.iter().copied()).collect(),
    ).unwrap();
    npz.add_array("outcome", &outcomes).unwrap();

    let names_str: String = abilities.iter().map(|(n, _, _)| n.as_str()).collect::<Vec<_>>().join("\n");
    let names_bytes: Array1<u8> = Array1::from_vec(names_str.into_bytes());
    npz.add_array("ability_names", &names_bytes).unwrap();

    let dsl_str: String = abilities.iter().map(|(_, _, dsl)| dsl.as_str()).collect::<Vec<_>>().join("\n---SEPARATOR---\n");
    let dsl_bytes: Array1<u8> = Array1::from_vec(dsl_str.into_bytes());
    npz.add_array("dsl_texts", &dsl_bytes).unwrap();

    npz.finish().expect("Failed to finalize npz");
}
