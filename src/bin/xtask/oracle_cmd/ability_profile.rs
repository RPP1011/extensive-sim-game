//! Behavioral ability profiling — controlled sim experiments for embedding generation.
//!
//! For each ability, spawns minimal sims (1 caster + N targets) under varied conditions,
//! casts the ability, and records clean before/after outcome deltas.
//! Outputs `dataset/ability_profiles.npz` for training behavioral embeddings.

use std::collections::HashMap;
use std::process::ExitCode;
use std::sync::Mutex;

use bevy_game::ai::core::{
    sim_vec2, step, SimState, SimVec2, Team, UnitState, UnitIntent, IntentAction, FIXED_TICK_MS,
};
use bevy_game::ai::effects::{
    AbilityDef, AbilitySlot, AbilityTarget, AbilityTargeting, StatusKind,
};
use bevy_game::ai::effects::dsl;
use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;
use rayon::prelude::*;

/// Condition vector for a single trial.
#[derive(Debug, Clone)]
struct TrialCondition {
    target_hp_pct: f32,   // 0-1 fraction of max_hp
    distance: f32,        // distance from caster to first target
    n_targets: usize,     // number of targets
    armor: f32,           // target armor value
}

/// Per-target outcome delta recorded after ability resolves.
/// Each status field is the peak duration (seconds) observed across all ticks.
#[derive(Debug, Clone, Default)]
struct TargetOutcome {
    delta_hp: f32,
    delta_shield: f32,
    delta_x: f32,
    delta_y: f32,
    killed: bool,
    // Status effects — peak duration in seconds (0 = not applied)
    stun_dur: f32,
    slow_dur: f32,
    slow_factor: f32,       // peak slow factor (0-1)
    root_dur: f32,
    silence_dur: f32,
    fear_dur: f32,
    taunt_dur: f32,
    blind_dur: f32,
    polymorph_dur: f32,
    suppress_dur: f32,
    grounded_dur: f32,
    charm_dur: f32,
    // Positive effects
    buff_dur: f32,          // any buff
    debuff_dur: f32,        // any debuff
    dot_dur: f32,           // damage-over-time
    hot_dur: f32,           // heal-over-time
    shield_amount: f32,     // peak shield gained
    damage_modify_dur: f32, // damage amplification
}

/// Aggregated outcome for one trial.
#[derive(Debug, Clone)]
struct TrialOutcome {
    total_damage: f32,
    total_heal: f32,
    n_targets_hit: u32,
    n_targets_killed: u32,
    per_target: Vec<TargetOutcome>,
    caster: TargetOutcome, // caster self-state changes (for self-buffs, dashes, self-heals)
}

/// A single profiling sample: condition + outcome for one ability trial.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ProfileSample {
    ability_idx: u32,
    ability_name: String,
    condition: Vec<f32>,
    outcome: Vec<f32>,
}

const MAX_HP: i32 = 100;
const CASTER_ID: u32 = 0;
const MAX_TARGETS: usize = 4;

// Condition grid
const HP_PCTS: &[f32] = &[0.2, 0.5, 0.8, 1.0];
const DISTANCES: &[f32] = &[1.0, 3.0, 5.0, 8.0];
const TARGET_COUNTS: &[usize] = &[1, 2, 4];
const ARMORS: &[f32] = &[0.0, 10.0, 25.0];

/// Build a hero unit with one ability attached.
fn caster_unit(ability: AbilityDef) -> UnitState {
    UnitState {
        id: CASTER_ID,
        team: Team::Hero,
        hp: MAX_HP / 2, // Start at 50% so self-heals have room to show effect
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
        // Very high attack cooldown so auto-attacks don't fire
        // (already set above, but let's ensure it's high)
    }
}

/// Build a target unit at a given position with given HP and armor.
fn target_unit(id: u32, team: Team, pos: SimVec2, hp: i32, armor: f32) -> UnitState {
    UnitState {
        id,
        team,
        hp,
        max_hp: MAX_HP,
        position: pos,
        move_speed_per_sec: 3.0,
        attack_damage: 10,
        attack_range: 1.4,
        attack_cooldown_ms: 999999, // no auto-attacks
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

fn make_sim(units: Vec<UnitState>) -> SimState {
    SimState {
        tick: 0,
        rng_state: 42,
        units,
        projectiles: Vec::new(),
        passive_trigger_depth: 0,
        zones: Vec::new(),
        tethers: Vec::new(),
        grid_nav: None,
    }
}

impl TargetOutcome {
    fn has_any_status(&self) -> bool {
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
fn scan_status_effects(unit: &UnitState, out: &mut TargetOutcome) {
    // Stun/suppress from control_remaining_ms
    if unit.control_remaining_ms > 0 {
        let dur_s = unit.control_remaining_ms as f32 / 1000.0;
        out.stun_dur = out.stun_dur.max(dur_s);
    }
    // Shield from shield_hp
    if unit.shield_hp > 0 {
        out.shield_amount = out.shield_amount.max(unit.shield_hp as f32);
    }
    // Scan status_effects list
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
            StatusKind::Shield { .. } => out.shield_amount = out.shield_amount.max(se.remaining_ms as f32), // remaining_ms as proxy
            _ => {} // Reflect, Lifesteal, Stealth, etc. — not tracked
        }
    }
}

/// Determine the correct AbilityTarget based on the ability's targeting type.
fn make_target(targeting: &AbilityTargeting, first_target_id: u32, first_target_pos: SimVec2) -> AbilityTarget {
    match targeting {
        AbilityTargeting::TargetEnemy => AbilityTarget::Unit(first_target_id),
        AbilityTargeting::TargetAlly => AbilityTarget::Unit(first_target_id),
        AbilityTargeting::SelfCast => AbilityTarget::None,
        // SelfAoe: use Position so cones/lines get a direction toward targets
        AbilityTargeting::SelfAoe => AbilityTarget::Position(first_target_pos),
        AbilityTargeting::GroundTarget | AbilityTargeting::Direction | AbilityTargeting::Vector => {
            AbilityTarget::Position(first_target_pos)
        }
        AbilityTargeting::Global => AbilityTarget::None,
    }
}

/// Run a single trial: set up sim, cast ability, tick until resolved, record outcomes.
/// Tracks peak CC duration and cumulative HP changes across all ticks.
fn run_trial(ability: &AbilityDef, cond: &TrialCondition) -> TrialOutcome {
    let target_hp = (cond.target_hp_pct * MAX_HP as f32).max(1.0) as i32;
    // SelfAoe/SelfCast still hit enemies (AoE centered on self) or buff self
    // Only TargetAlly explicitly targets allies
    let is_ally = matches!(ability.targeting, AbilityTargeting::TargetAlly);
    let target_team = if is_ally { Team::Hero } else { Team::Enemy };

    // Place targets in a cluster around the target distance
    let mut units = vec![caster_unit(ability.clone())];
    units[0].attack_cooldown_ms = 999999;

    let first_target_id = 1u32;
    for i in 0..cond.n_targets {
        let angle = (i as f32) * std::f32::consts::TAU / (cond.n_targets as f32);
        let spread = 0.5;
        let x = cond.distance + angle.cos() * spread * (i as f32).min(1.0);
        let y = angle.sin() * spread * (i as f32).min(1.0);
        units.push(target_unit(
            first_target_id + i as u32,
            target_team,
            sim_vec2(x, y),
            target_hp,
            cond.armor,
        ));
    }

    // Record pre-state (targets + caster)
    let pre_hp: Vec<i32> = units[1..].iter().map(|u| u.hp).collect();
    let pre_shield: Vec<i32> = units[1..].iter().map(|u| u.shield_hp).collect();
    let pre_pos: Vec<(f32, f32)> = units[1..].iter().map(|u| (u.position.x, u.position.y)).collect();
    let caster_pre_hp = units[0].hp;
    let caster_pre_shield = units[0].shield_hp;
    let caster_pre_pos = (units[0].position.x, units[0].position.y);

    let first_target_pos = units[1].position;
    let mut sim = make_sim(units);

    // Track peak status effects per target + caster across all ticks
    let mut peak_outcomes: Vec<TargetOutcome> = (0..cond.n_targets).map(|_| TargetOutcome::default()).collect();
    let mut caster_outcome = TargetOutcome::default();

    // Cast the ability on tick 1
    let target = make_target(&ability.targeting, first_target_id, first_target_pos);
    let intents = vec![UnitIntent {
        unit_id: CASTER_ID,
        action: IntentAction::UseAbility { ability_index: 0, target },
    }];

    let (new_sim, _events) = step(sim, &intents, FIXED_TICK_MS);
    sim = new_sim;

    // Tick until ability fully resolves
    let max_resolve_ticks = (ability.cast_time_ms / FIXED_TICK_MS).max(1) + 100;
    for _ in 0..max_resolve_ticks {
        // Sample status effects each tick — record peak durations
        for i in 0..cond.n_targets {
            let tid = first_target_id + i as u32;
            if let Some(unit) = sim.units.iter().find(|u| u.id == tid) {
                scan_status_effects(unit, &mut peak_outcomes[i]);
            }
        }
        if let Some(caster) = sim.units.iter().find(|u| u.id == CASTER_ID) {
            scan_status_effects(caster, &mut caster_outcome);
        }
        let (new_sim, _events) = step(sim, &[], FIXED_TICK_MS);
        sim = new_sim;
    }

    // Final sample
    for i in 0..cond.n_targets {
        let tid = first_target_id + i as u32;
        if let Some(unit) = sim.units.iter().find(|u| u.id == tid) {
            scan_status_effects(unit, &mut peak_outcomes[i]);
        }
    }
    if let Some(caster) = sim.units.iter().find(|u| u.id == CASTER_ID) {
        scan_status_effects(caster, &mut caster_outcome);
    }

    // Record post-state and compute deltas
    let mut per_target = Vec::new();
    let mut total_damage = 0.0f32;
    let mut total_heal = 0.0f32;
    let mut n_hit = 0u32;
    let mut n_killed = 0u32;

    for i in 0..cond.n_targets {
        let target_id = first_target_id + i as u32;
        let mut outcome = std::mem::take(&mut peak_outcomes[i]);

        if let Some(unit) = sim.units.iter().find(|u| u.id == target_id) {
            outcome.delta_hp = unit.hp as f32 - pre_hp[i] as f32;
            outcome.delta_shield = unit.shield_hp as f32 - pre_shield[i] as f32;
            outcome.delta_x = unit.position.x - pre_pos[i].0;
            outcome.delta_y = unit.position.y - pre_pos[i].1;
            outcome.killed = unit.hp <= 0;

            let has_effect = outcome.delta_hp.abs() > 0.01
                || outcome.delta_shield.abs() > 0.01
                || outcome.delta_x.abs() > 0.01
                || outcome.delta_y.abs() > 0.01
                || outcome.has_any_status();
            if has_effect { n_hit += 1; }
            if outcome.delta_hp < 0.0 { total_damage += -outcome.delta_hp; }
            if outcome.delta_hp > 0.0 { total_heal += outcome.delta_hp; }
            if outcome.killed { n_killed += 1; }
        } else {
            // Unit removed (dead)
            outcome.delta_hp = -(pre_hp[i] as f32);
            outcome.killed = true;
            total_damage += -outcome.delta_hp;
            n_hit += 1;
            n_killed += 1;
        }
        per_target.push(outcome);
    }

    // Fill caster deltas
    if let Some(caster) = sim.units.iter().find(|u| u.id == CASTER_ID) {
        caster_outcome.delta_hp = caster.hp as f32 - caster_pre_hp as f32;
        caster_outcome.delta_shield = caster.shield_hp as f32 - caster_pre_shield as f32;
        caster_outcome.delta_x = caster.position.x - caster_pre_pos.0;
        caster_outcome.delta_y = caster.position.y - caster_pre_pos.1;
    }

    TrialOutcome { total_damage, total_heal, n_targets_hit: n_hit, n_targets_killed: n_killed, per_target, caster: caster_outcome }
}

/// Per-target outcome dimension count.
const PER_TARGET_DIM: usize = 23;
/// Total outcome vector dimension: per_target(4 × 23) + caster(23) + 4 aggregates = 119
const OUTCOME_DIM: usize = MAX_TARGETS * PER_TARGET_DIM + PER_TARGET_DIM + 4;

fn push_target_outcome(v: &mut Vec<f32>, t: &TargetOutcome) {
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
/// Layout: [per_target(4 × 23), caster(23), total_damage, total_heal, n_targets_hit, n_targets_killed] = 119
fn outcome_to_vec(outcome: &TrialOutcome) -> Vec<f32> {
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
/// Layout: [target_hp_pct, distance, n_targets, armor] = 4 dims
fn condition_to_vec(cond: &TrialCondition) -> Vec<f32> {
    vec![cond.target_hp_pct, cond.distance, cond.n_targets as f32, cond.armor]
}

/// Recursively find all `.ability` files under a directory.
fn find_ability_files(dir: &str) -> Vec<std::path::PathBuf> {
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
fn load_all_abilities() -> Vec<(String, AbilityDef, String)> {
    let mut abilities: Vec<(String, AbilityDef, String)> = Vec::new(); // (name, def, dsl_text)
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut add_ability = |def: AbilityDef, dsl_text: String| {
        if seen_names.insert(def.name.clone()) {
            let name = def.name.clone();
            abilities.push((name, def, dsl_text));
        }
    };

    // 1. Load from dataset/abilities/**/*.ability
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

    // 2. Load from assets/hero_templates/*.toml (inline abilities)
    load_abilities_from_toml_dir("assets/hero_templates", &mut seen_names, &mut abilities);

    // 3. Load from assets/lol_heroes/*.toml
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
            // Check for companion .ability file
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

pub fn run_ability_profile(args: crate::cli::AbilityProfileArgs) -> ExitCode {
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .ok();
    }

    let abilities = load_all_abilities();
    if abilities.is_empty() {
        eprintln!("No abilities found!");
        return ExitCode::from(1);
    }

    // Build the full condition grid
    let mut conditions = Vec::new();
    for &hp_pct in HP_PCTS {
        for &distance in DISTANCES {
            for &n_targets in TARGET_COUNTS {
                for &armor in ARMORS {
                    conditions.push(TrialCondition { target_hp_pct: hp_pct, distance, n_targets, armor });
                }
            }
        }
    }
    let n_conditions = conditions.len();
    eprintln!("Condition grid: {} combinations ({}×{}×{}×{} = hp×dist×targets×armor)",
        n_conditions, HP_PCTS.len(), DISTANCES.len(), TARGET_COUNTS.len(), ARMORS.len());

    // Subsample if needed to hit samples_per_ability target
    // Full grid = 4×4×3×3 = 144. If samples_per_ability < 144, sample randomly.
    // If > 144, repeat grid with different seeds (not implemented yet, just use full grid).
    let n_trials_per_ability = n_conditions.min(args.samples_per_ability);

    eprintln!("Profiling {} abilities × {} trials = {} total trials",
        abilities.len(), n_trials_per_ability, abilities.len() * n_trials_per_ability);

    let all_samples: Mutex<Vec<ProfileSample>> = Mutex::new(Vec::new());
    let done = std::sync::atomic::AtomicUsize::new(0);
    let n_abilities = abilities.len();

    // Ability-level parallelism
    abilities.par_iter().enumerate().for_each(|(abl_idx, (name, def, _dsl_text))| {
        let mut local_samples = Vec::new();

        for cond in conditions.iter().take(n_trials_per_ability) {
            // Skip conditions where distance > range for targeted abilities
            // (the ability won't fire, producing only zeros — not useful training data)
            let effective_range = def.range;
            if effective_range > 0.0 && cond.distance > effective_range + 1.0 {
                // Still record it — the embedding should learn "nothing happens at this range"
            }

            let outcome = run_trial(def, cond);
            local_samples.push(ProfileSample {
                ability_idx: abl_idx as u32,
                ability_name: name.clone(),
                condition: condition_to_vec(cond),
                outcome: outcome_to_vec(&outcome),
            });
        }

        all_samples.lock().unwrap().extend(local_samples);

        let completed = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if completed % 20 == 0 || completed == n_abilities {
            eprintln!("  [{completed}/{n_abilities}] {name}");
        }
    });

    let samples = all_samples.into_inner().unwrap();
    let n = samples.len();
    eprintln!("\nTotal: {n} profile samples from {n_abilities} abilities");

    if n == 0 {
        eprintln!("No samples generated.");
        return ExitCode::from(1);
    }

    // Print summary stats
    let agg_offset = MAX_TARGETS * PER_TARGET_DIM + PER_TARGET_DIM; // after caster slot
    let mut damage_count = 0u32;
    let mut heal_count = 0u32;
    let mut cc_count = 0u32;
    let mut status_count = 0u32;
    let mut displacement_count = 0u32;
    let mut kill_count = 0u32;
    let mut zero_effect_count = 0u32;

    for s in &samples {
        let total_dmg = s.outcome[agg_offset];
        let total_heal = s.outcome[agg_offset + 1];
        let n_killed = s.outcome[agg_offset + 3];

        // Check per-target status effects and displacement
        let mut any_cc = false;
        let mut any_status = false;
        let mut any_disp = false;
        for i in 0..MAX_TARGETS {
            let base = i * PER_TARGET_DIM;
            // displacement: delta_x (idx 2), delta_y (idx 3)
            if s.outcome[base + 2].abs() > 0.01 || s.outcome[base + 3].abs() > 0.01 {
                any_disp = true;
            }
            // hard CC: stun(5), root(8), fear(10), taunt(11), polymorph(13), suppress(14), charm(16)
            if s.outcome[base + 5] > 0.0 || s.outcome[base + 8] > 0.0
                || s.outcome[base + 10] > 0.0 || s.outcome[base + 11] > 0.0
                || s.outcome[base + 13] > 0.0 || s.outcome[base + 14] > 0.0
                || s.outcome[base + 16] > 0.0 {
                any_cc = true;
            }
            // any status effect (indices 5..22)
            if (5..PER_TARGET_DIM).any(|j| s.outcome[base + j] > 0.0) {
                any_status = true;
            }
        }

        if total_dmg > 0.0 { damage_count += 1; }
        if total_heal > 0.0 { heal_count += 1; }
        if any_cc { cc_count += 1; }
        if any_status { status_count += 1; }
        if any_disp { displacement_count += 1; }
        if n_killed > 0.0 { kill_count += 1; }
        // Check caster slot too (index = MAX_TARGETS * PER_TARGET_DIM)
        let caster_base = MAX_TARGETS * PER_TARGET_DIM;
        let caster_has_effect = (0..PER_TARGET_DIM).any(|j| s.outcome[caster_base + j].abs() > 0.01);

        if total_dmg == 0.0 && total_heal == 0.0 && !any_status && !any_disp && !caster_has_effect {
            zero_effect_count += 1;
        }
    }

    eprintln!("\nOutcome distribution:");
    eprintln!("  Damage:       {damage_count:>6} ({:.1}%)", damage_count as f32 / n as f32 * 100.0);
    eprintln!("  Heal:         {heal_count:>6} ({:.1}%)", heal_count as f32 / n as f32 * 100.0);
    eprintln!("  Hard CC:      {cc_count:>6} ({:.1}%)", cc_count as f32 / n as f32 * 100.0);
    eprintln!("  Any status:   {status_count:>6} ({:.1}%)", status_count as f32 / n as f32 * 100.0);
    eprintln!("  Displacement: {displacement_count:>6} ({:.1}%)", displacement_count as f32 / n as f32 * 100.0);
    eprintln!("  Kill:         {kill_count:>6} ({:.1}%)", kill_count as f32 / n as f32 * 100.0);
    eprintln!("  Zero-effect:  {zero_effect_count:>6} ({:.1}%)", zero_effect_count as f32 / n as f32 * 100.0);

    // Write npz
    write_profile_npz(&args.output, &samples, &abilities);
    eprintln!("\nWritten to: {}", args.output.display());

    ExitCode::SUCCESS
}

fn write_profile_npz(
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

    // ability_id: (N,)
    let ability_ids: Array1<i32> = Array1::from_vec(
        samples.iter().map(|s| s.ability_idx as i32).collect(),
    );
    npz.add_array("ability_id", &ability_ids).unwrap();

    // condition: (N, 4)
    let cond_dim = 4;
    let conditions: Array2<f32> = Array2::from_shape_vec(
        (n, cond_dim),
        samples.iter().flat_map(|s| s.condition.iter().copied()).collect(),
    ).unwrap();
    npz.add_array("condition", &conditions).unwrap();

    // outcome: (N, OUTCOME_DIM)
    let outcome_dim = OUTCOME_DIM;
    let outcomes: Array2<f32> = Array2::from_shape_vec(
        (n, outcome_dim),
        samples.iter().flat_map(|s| s.outcome.iter().copied()).collect(),
    ).unwrap();
    npz.add_array("outcome", &outcomes).unwrap();

    // ability_names: store as a flat string with newline separator for Python to split
    let names_str: String = abilities.iter().map(|(n, _, _)| n.as_str()).collect::<Vec<_>>().join("\n");
    let names_bytes: Array1<u8> = Array1::from_vec(names_str.into_bytes());
    npz.add_array("ability_names", &names_bytes).unwrap();

    // dsl_texts: ability DSL text for debugging
    let dsl_str: String = abilities.iter().map(|(_, _, dsl)| dsl.as_str()).collect::<Vec<_>>().join("\n---SEPARATOR---\n");
    let dsl_bytes: Array1<u8> = Array1::from_vec(dsl_str.into_bytes());
    npz.add_array("dsl_texts", &dsl_bytes).unwrap();

    npz.finish().expect("Failed to finalize npz");
}
