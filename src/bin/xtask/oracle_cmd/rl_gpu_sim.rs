//! ActiveSim type and construction helpers for GPU-multiplexed episode generation.

use super::transformer_rl::{
    RlEpisode, RlStep,
    load_behavior_trees,
};
use super::rl_policies::hp_fraction;

/// Per-hero snapshot taken before sim step for action-specific reward computation.
#[derive(Clone)]
pub(crate) struct HeroPreStepState {
    pub(crate) unit_id: u32,
    pub(crate) position: bevy_game::ai::core::SimVec2,
    pub(crate) hp: i32,
    pub(crate) nearest_enemy_dist: f32,
    pub(crate) move_dir: usize,
    pub(crate) combat_type: usize,
}

pub(crate) struct PendingUnit {
    pub(crate) unit_id: u32,
    pub(crate) token: bevy_game::ai::core::ability_transformer::gpu_client::InferenceToken,
    pub(crate) gs_v2: bevy_game::ai::core::ability_eval::GameStateV2,
    pub(crate) mask_vec: Vec<bool>,
    pub(crate) n_abilities: usize,
    pub(crate) step_reward: f32,
    pub(crate) resolved: bool,
    pub(crate) is_hero: bool,
}

#[derive(PartialEq)]
pub(crate) enum SimPhase {
    NeedsTick,
    WaitingGpu,
}

pub(crate) struct ActiveSim {
    pub(crate) sim: bevy_game::ai::core::SimState,
    pub(crate) squad_ai: bevy_game::ai::squad::SquadAiState,
    pub(crate) scenario_name: String,
    pub(crate) max_ticks: u64,
    pub(crate) rng: u64,
    pub(crate) steps: Vec<RlStep>,
    pub(crate) unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>>,
    pub(crate) unit_ability_names: std::collections::HashMap<u32, Vec<String>>,
    pub(crate) cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>>,
    pub(crate) hero_ids: Vec<u32>,
    pub(crate) enemy_ids: Vec<u32>,
    pub(crate) prev_hero_hp: i32,
    pub(crate) prev_enemy_hp: i32,
    pub(crate) avg_unit_hp: f32,
    pub(crate) initial_enemy_count: f32,
    pub(crate) initial_hero_count: f32,
    pub(crate) pending_event_reward: f32,
    pub(crate) hero_pre_step: Vec<HeroPreStepState>,
    pub(crate) steps_recorded_this_tick: Vec<usize>,
    pub(crate) tick: u64,
    pub(crate) step_interval: u64,
    pub(crate) temperature: f32,
    pub(crate) intents: Vec<bevy_game::ai::core::UnitIntent>,
    pub(crate) pending_units: Vec<PendingUnit>,
    pub(crate) phase: SimPhase,
    pub(crate) self_play_gpu: bool,
    pub(crate) hidden_states: std::collections::HashMap<u32, Vec<f32>>,
    pub(crate) drill_objective_type: Option<String>,
    pub(crate) drill_target_position: Option<[f32; 2]>,
    pub(crate) drill_target_radius: Option<f32>,
    pub(crate) drill_objective_reached: bool,
    pub(crate) action_mask: Option<String>,
    pub(crate) behavior_trees: std::collections::HashMap<u32, bevy_game::ai::behavior::BehaviorTree>,
    /// Precomputed visibility map for corner tokens (V6 spatial cross-attention).
    /// Built once from sim.grid_nav; None if no nav grid is available.
    pub(crate) visibility_map: Option<bevy_game::ai::goap::spatial::VisibilityMap>,
}

impl ActiveSim {
    pub(crate) fn new(
        sim: bevy_game::ai::core::SimState,
        squad_ai: bevy_game::ai::squad::SquadAiState,
        scenario_name: String,
        max_ticks: u64,
        rng_seed: u64,
        step_interval: u64,
        temperature: f32,
        tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
        embedding_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    ) -> Self {
        use bevy_game::ai::core::Team;
        use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;

        let hero_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.id).collect();
        let enemy_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.id).collect();

        let mut unit_abilities = std::collections::HashMap::new();
        let mut unit_ability_names = std::collections::HashMap::new();
        let mut cls_cache = std::collections::HashMap::new();

        let all_ids: Vec<u32> = hero_ids.iter().chain(enemy_ids.iter()).copied().collect();
        for &uid in &all_ids {
            if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
                let mut ability_tokens_list = Vec::new();
                let mut ability_names_list = Vec::new();
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    let dsl = emit_ability_dsl(&slot.def);
                    let tokens = tokenizer.encode_with_cls(&dsl);
                    let safe_name = slot.def.name.replace(' ', "_");
                    if let Some(reg) = embedding_registry {
                        if let Some(reg_cls) = reg.get(&safe_name) {
                            cls_cache.insert((uid, idx), reg_cls.to_vec());
                        }
                    }
                    ability_tokens_list.push(tokens);
                    ability_names_list.push(slot.def.name.clone());
                }
                unit_abilities.insert(uid, ability_tokens_list);
                unit_ability_names.insert(uid, ability_names_list);
            }
        }

        let prev_hero_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
        let prev_enemy_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
        let n_units = sim.units.iter().filter(|u| u.hp > 0).count().max(1) as f32;
        let avg_unit_hp = (prev_hero_hp + prev_enemy_hp) as f32 / n_units;
        let initial_enemy_count = sim.units.iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0).count() as f32;
        let initial_hero_count = sim.units.iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0).count() as f32;

        let visibility_map = sim.grid_nav.as_ref()
            .map(bevy_game::ai::goap::spatial::VisibilityMap::build);

        ActiveSim {
            sim, squad_ai, scenario_name, max_ticks,
            rng: rng_seed, steps: Vec::new(),
            unit_abilities, unit_ability_names, cls_cache,
            hero_ids, enemy_ids,
            prev_hero_hp, prev_enemy_hp, avg_unit_hp,
            initial_enemy_count, initial_hero_count,
            pending_event_reward: 0.0,
            hero_pre_step: Vec::new(),
            steps_recorded_this_tick: Vec::new(),
            tick: 0, step_interval, temperature,
            intents: Vec::new(), pending_units: Vec::new(),
            phase: SimPhase::NeedsTick,
            self_play_gpu: false,
            hidden_states: std::collections::HashMap::new(),
            drill_objective_type: None,
            drill_target_position: None,
            drill_target_radius: None,
            drill_objective_reached: false,
            action_mask: None,
            behavior_trees: std::collections::HashMap::new(),
            visibility_map,
        }
    }

    pub(crate) fn is_done(&mut self) -> bool {
        use bevy_game::ai::core::Team;
        let heroes_alive = self.sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = self.sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        if heroes_alive == 0 { return true; }
        if enemies_alive == 0 && self.drill_objective_type.is_none() { return true; }
        if let Some(ref obj_type) = self.drill_objective_type {
            if obj_type == "reach_position" {
                if let (Some(target), Some(radius)) = (self.drill_target_position, self.drill_target_radius) {
                    for unit in self.sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0) {
                        let dx = unit.position.x - target[0]; let dy = unit.position.y - target[1];
                        if (dx * dx + dy * dy).sqrt() <= radius { self.drill_objective_reached = true; return true; }
                    }
                }
            } else if obj_type == "reach_entity" {
                let radius = self.drill_target_radius.unwrap_or(1.5);
                for hero in self.sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0) {
                    for enemy in self.sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0) {
                        let dx = hero.position.x - enemy.position.x; let dy = hero.position.y - enemy.position.y;
                        if (dx * dx + dy * dy).sqrt() <= radius { self.drill_objective_reached = true; return true; }
                    }
                }
            } else if obj_type == "kill_all" || obj_type == "kill_target" {
                if enemies_alive == 0 { self.drill_objective_reached = true; return true; }
            } else if obj_type == "survive" {
                if self.tick >= self.max_ticks { self.drill_objective_reached = heroes_alive > 0; return true; }
                return false;
            }
        }
        self.tick >= self.max_ticks
    }

    pub(crate) fn into_episode(self) -> RlEpisode {
        use bevy_game::ai::core::Team;
        let heroes_alive = self.sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = self.sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();
        let (outcome, reward) = if self.drill_objective_type.is_some() {
            if self.drill_objective_reached { ("Victory".to_string(), 1.0) }
            else if heroes_alive == 0 { ("Defeat".to_string(), -1.0) }
            else { ("Timeout".to_string(), -0.5) }
        } else if enemies_alive == 0 { ("Victory".to_string(), 1.0) }
        else if heroes_alive == 0 { ("Defeat".to_string(), -1.0) }
        else {
            let hero_frac = hp_fraction(&self.sim, Team::Hero);
            let enemy_frac = hp_fraction(&self.sim, Team::Enemy);
            ("Timeout".to_string(), (enemy_frac - hero_frac).clamp(-1.0, 1.0) * 0.5)
        };
        RlEpisode {
            scenario: self.scenario_name, outcome, reward,
            ticks: self.sim.tick, unit_abilities: self.unit_abilities,
            unit_ability_names: self.unit_ability_names, steps: self.steps,
        }
    }
}

/// Precomputed per-scenario data. Built once, cloned+reseeded per episode.
pub(crate) struct PrecomputedScenario {
    pub(crate) sim: bevy_game::ai::core::SimState,
    pub(crate) squad_ai: bevy_game::ai::squad::SquadAiState,
    pub(crate) scenario_name: String,
    pub(crate) max_ticks: u64,
    pub(crate) unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>>,
    pub(crate) unit_ability_names: std::collections::HashMap<u32, Vec<String>>,
    pub(crate) cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>>,
    pub(crate) hero_ids: Vec<u32>,
    pub(crate) enemy_ids: Vec<u32>,
    pub(crate) initial_hero_hp: i32,
    pub(crate) initial_enemy_hp: i32,
    pub(crate) avg_unit_hp: f32,
    pub(crate) initial_hero_count: f32,
    pub(crate) initial_enemy_count: f32,
    pub(crate) drill_objective_type: Option<String>,
    pub(crate) drill_target_position: Option<[f32; 2]>,
    pub(crate) drill_target_radius: Option<f32>,
    pub(crate) action_mask: Option<String>,
    pub(crate) behavior_trees: std::collections::HashMap<u32, bevy_game::ai::behavior::BehaviorTree>,
    pub(crate) objective: Option<bevy_game::scenario::ObjectiveDef>,
}

/// Build all scenario templates once. Returns one `PrecomputedScenario` per scenario.
pub(crate) fn precompute_scenarios(
    scenario_files: &[bevy_game::scenario::ScenarioFile],
    max_ticks_override: Option<u64>,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
) -> Vec<PrecomputedScenario> {
    use bevy_game::ai::core::Team;
    use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;
    use bevy_game::scenario::run_scenario_to_state_with_room;

    scenario_files.iter().map(|sf| {
        let cfg = &sf.scenario;
        let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);
        let (mut sim, squad_ai, nav) = run_scenario_to_state_with_room(cfg);
        sim.grid_nav = Some(nav);

        // Tokenize abilities + build CLS cache (same as ActiveSim::new)
        let hero_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.id).collect();
        let enemy_ids: Vec<u32> = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.id).collect();

        let mut unit_abilities = std::collections::HashMap::new();
        let mut unit_ability_names = std::collections::HashMap::new();
        let mut cls_cache = std::collections::HashMap::new();

        for &uid in hero_ids.iter().chain(enemy_ids.iter()) {
            if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
                let mut tokens_list = Vec::new();
                let mut names_list = Vec::new();
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    let dsl = emit_ability_dsl(&slot.def);
                    tokens_list.push(tokenizer.encode_with_cls(&dsl));
                    let safe_name = slot.def.name.replace(' ', "_");
                    if let Some(reg) = registry {
                        if let Some(reg_cls) = reg.get(&safe_name) {
                            cls_cache.insert((uid, idx), reg_cls.to_vec());
                        }
                    }
                    names_list.push(slot.def.name.clone());
                }
                unit_abilities.insert(uid, tokens_list);
                unit_ability_names.insert(uid, names_list);
            }
        }

        let initial_hero_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
        let initial_enemy_hp: i32 = sim.units.iter()
            .filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
        let n_units = sim.units.iter().filter(|u| u.hp > 0).count().max(1) as f32;
        let avg_unit_hp = (initial_hero_hp + initial_enemy_hp) as f32 / n_units;
        let initial_hero_count = sim.units.iter()
            .filter(|u| u.team == Team::Hero && u.hp > 0).count() as f32;
        let initial_enemy_count = sim.units.iter()
            .filter(|u| u.team == Team::Enemy && u.hp > 0).count() as f32;

        // Drill / objective config
        let (drill_objective_type, drill_target_position, drill_target_radius) =
            if let Some(ref obj) = cfg.objective {
                (Some(obj.objective_type.clone()), obj.position, obj.radius)
            } else if cfg.drill_type.is_some() {
                (cfg.drill_type.clone(), cfg.target_position, Some(1.0))
            } else {
                (None, None, None)
            };

        let behavior_trees = load_behavior_trees(&sim, cfg);

        PrecomputedScenario {
            sim, squad_ai, scenario_name: cfg.name.clone(), max_ticks,
            unit_abilities, unit_ability_names, cls_cache,
            hero_ids, enemy_ids,
            initial_hero_hp, initial_enemy_hp, avg_unit_hp,
            initial_hero_count, initial_enemy_count,
            drill_objective_type, drill_target_position, drill_target_radius,
            action_mask: cfg.action_mask.clone(),
            behavior_trees,
            objective: cfg.objective.clone(),
        }
    }).collect()
}

/// Spawn an ActiveSim from a precomputed scenario (clone + reseed).
pub(crate) fn active_sim_from_precomputed(
    pre: &PrecomputedScenario,
    si: usize, ei: usize,
    step_interval: u64, temperature: f32,
    self_play_gpu: bool,
) -> ActiveSim {
    let seed = (si as u64 * 1000 + ei as u64) ^ 0xDEADBEEF;
    let mut sim = pre.sim.clone();
    sim.rng_state = seed;

    let visibility_map = sim.grid_nav.as_ref()
        .map(bevy_game::ai::goap::spatial::VisibilityMap::build);

    ActiveSim {
        sim,
        squad_ai: pre.squad_ai.clone(),
        scenario_name: pre.scenario_name.clone(),
        max_ticks: pre.max_ticks,
        rng: seed,
        steps: Vec::new(),
        unit_abilities: pre.unit_abilities.clone(),
        unit_ability_names: pre.unit_ability_names.clone(),
        cls_cache: pre.cls_cache.clone(),
        hero_ids: pre.hero_ids.clone(),
        enemy_ids: pre.enemy_ids.clone(),
        prev_hero_hp: pre.initial_hero_hp,
        prev_enemy_hp: pre.initial_enemy_hp,
        avg_unit_hp: pre.avg_unit_hp,
        initial_enemy_count: pre.initial_enemy_count,
        initial_hero_count: pre.initial_hero_count,
        pending_event_reward: 0.0,
        hero_pre_step: Vec::new(),
        steps_recorded_this_tick: Vec::new(),
        tick: 0, step_interval, temperature,
        intents: Vec::new(), pending_units: Vec::new(),
        phase: SimPhase::NeedsTick,
        self_play_gpu,
        hidden_states: std::collections::HashMap::new(),
        drill_objective_type: pre.drill_objective_type.clone(),
        drill_target_position: pre.drill_target_position,
        drill_target_radius: pre.drill_target_radius,
        drill_objective_reached: false,
        action_mask: pre.action_mask.clone(),
        behavior_trees: pre.behavior_trees.clone(),
        visibility_map,
    }
}

/// Legacy: build from scenario file directly (no precompute). Still used by non-GPU path fallback.
pub(crate) fn make_active_sim(
    scenario_file: &bevy_game::scenario::ScenarioFile,
    si: usize, ei: usize,
    max_ticks_override: Option<u64>,
    temperature: f32, step_interval: u64,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    self_play_gpu: bool,
) -> Option<ActiveSim> {
    use bevy_game::scenario::run_scenario_to_state_with_room;
    let cfg = &scenario_file.scenario;
    let max_ticks = max_ticks_override.unwrap_or(cfg.max_ticks);
    let (sim, squad_ai, nav) = run_scenario_to_state_with_room(cfg);
    let mut sim = sim; sim.grid_nav = Some(nav);
    let seed = (si as u64 * 1000 + ei as u64) ^ 0xDEADBEEF;
    let mut active = ActiveSim::new(sim, squad_ai, cfg.name.clone(), max_ticks, seed, step_interval, temperature, tokenizer, registry);
    active.self_play_gpu = self_play_gpu;
    if let Some(ref obj) = cfg.objective {
        active.drill_objective_type = Some(obj.objective_type.clone());
        active.drill_target_position = obj.position;
        active.drill_target_radius = obj.radius;
    } else if cfg.drill_type.is_some() {
        active.drill_objective_type = cfg.drill_type.clone();
        active.drill_target_position = cfg.target_position;
        active.drill_target_radius = Some(1.0);
    }
    active.action_mask = cfg.action_mask.clone();
    active.behavior_trees = load_behavior_trees(&active.sim, cfg);
    Some(active)
}
