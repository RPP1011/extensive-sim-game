//! CombatEnv — wraps the tactical combat simulator as an rl4burn `Env`.

use std::collections::{HashMap, VecDeque};

use rand::{RngExt, SeedableRng};
use rl4burn::{Env, Space, Step};

use tactical_sim::effects::{AbilitySlot, HeroToml, PassiveSlot};
use tactical_sim::sim::ability_eval::{extract_game_state, GAME_STATE_DIM};
use tactical_sim::sim::self_play::actions::{action_mask, action_to_intent};
use tactical_sim::sim::{
    is_alive, sim_vec2, step, SimState, SimVec2, Team, UnitIntent, UnitState, UnitStore,
    FIXED_TICK_MS,
};
use tactical_sim::squad::{generate_intents, Personality, SquadAiState};

/// Number of discrete actions in the flat action space.
const NUM_ACTIONS: usize = 14;

/// Maximum ticks before truncation.
const MAX_TICKS: u64 = 3000;

/// Observation dimension: 7 entity slots x 30 features = 210.
const OBS_DIM: usize = GAME_STATE_DIM; // 210

/// Combat environment implementing the rl4burn `Env` trait.
///
/// The agent controls a single unit on the Hero team. All other units
/// (allies and enemies) are driven by the built-in squad AI.
pub struct CombatEnv {
    sim: SimState,
    squad_ai: SquadAiState,
    controlled_unit_id: u32,
    hero_templates: Vec<HeroToml>,
    enemy_templates: Vec<HeroToml>,
    max_ticks: u64,
    current_tick: u64,
    rng: rand::rngs::SmallRng,
    /// HP snapshot from previous tick for reward shaping.
    prev_ally_hp: i32,
    prev_enemy_hp: i32,
    prev_enemy_count: usize,
    prev_ally_count: usize,
}

impl CombatEnv {
    pub fn new(
        hero_templates: Vec<HeroToml>,
        enemy_templates: Vec<HeroToml>,
        seed: u64,
    ) -> Self {
        let mut env = Self {
            sim: SimState {
                tick: 0,
                rng_state: seed,
                units: UnitStore::new(vec![]),
                projectiles: Vec::new(),
                passive_trigger_depth: 0,
                zones: Vec::new(),
                tethers: Vec::new(),
                grid_nav: None,
            },
            squad_ai: SquadAiState::new(
                &SimState {
                    tick: 0,
                    rng_state: 0,
                    units: UnitStore::new(vec![]),
                    projectiles: Vec::new(),
                    passive_trigger_depth: 0,
                    zones: Vec::new(),
                    tethers: Vec::new(),
                    grid_nav: None,
                },
                HashMap::new(),
            ),
            controlled_unit_id: 1,
            hero_templates,
            enemy_templates,
            max_ticks: MAX_TICKS,
            current_tick: 0,
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
            prev_ally_hp: 0,
            prev_enemy_hp: 0,
            prev_enemy_count: 0,
            prev_ally_count: 0,
        };
        env.reset_internal();
        env
    }

    /// Build a fresh combat scenario by sampling random team compositions.
    fn reset_internal(&mut self) {
        let n_heroes = self.rng.random_range(2u32..=4);
        let n_enemies = self.rng.random_range(2u32..=4);

        let mut units = Vec::new();
        let mut id = 1u32;

        // Sample hero team
        for i in 0..n_heroes {
            let idx = self.rng.random_range(0..self.hero_templates.len());
            let x = -4.0 + (i as f32) * 2.0;
            let y = self.rng.random_range(-2.0f32..2.0);
            let unit = hero_toml_to_unit(&self.hero_templates[idx], id, Team::Hero, sim_vec2(x, y));
            units.push(unit);
            id += 1;
        }

        // The first hero unit is the one controlled by the agent
        let controlled_id = 1;

        // Sample enemy team
        for i in 0..n_enemies {
            let idx = self.rng.random_range(0..self.enemy_templates.len());
            let x = 4.0 + (i as f32) * 2.0;
            let y = self.rng.random_range(-2.0f32..2.0);
            let unit =
                hero_toml_to_unit(&self.enemy_templates[idx], id, Team::Enemy, sim_vec2(x, y));
            units.push(unit);
            id += 1;
        }

        units.sort_by_key(|u| u.id);

        let sim_seed = self.rng.random::<u64>();
        self.sim = SimState {
            tick: 0,
            rng_state: sim_seed,
            units: UnitStore::new(units),
            projectiles: Vec::new(),
            passive_trigger_depth: 0,
            zones: Vec::new(),
            tethers: Vec::new(),
            grid_nav: None,
        };

        // Build squad AI with default personalities for all units
        let personalities: HashMap<u32, Personality> = self
            .sim
            .units
            .iter()
            .map(|u| (u.id, Personality::default()))
            .collect();
        self.squad_ai = SquadAiState::new(&self.sim, personalities);

        self.controlled_unit_id = controlled_id;
        self.current_tick = 0;

        // Snapshot initial HP totals
        self.prev_ally_hp = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && is_alive(u))
            .map(|u| u.hp)
            .sum();
        self.prev_enemy_hp = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u))
            .map(|u| u.hp)
            .sum();
        self.prev_ally_count = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && is_alive(u))
            .count();
        self.prev_enemy_count = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u))
            .count();
    }

    fn observe(&self) -> Vec<f32> {
        match self
            .sim
            .units
            .iter()
            .find(|u| u.id == self.controlled_unit_id)
        {
            Some(unit) if is_alive(unit) => extract_game_state(&self.sim, unit),
            _ => vec![0.0; OBS_DIM],
        }
    }

    fn compute_mask(&self) -> [bool; NUM_ACTIONS] {
        action_mask(&self.sim, self.controlled_unit_id)
    }
}

impl Env for CombatEnv {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.reset_internal();
        self.observe()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let action_idx = action[0] as usize;

        // Generate intents for all units via squad AI
        let mut intents = generate_intents(&self.sim, &mut self.squad_ai, FIXED_TICK_MS);

        // Override the controlled unit's intent with the agent's action
        let controlled_alive = self
            .sim
            .units
            .iter()
            .find(|u| u.id == self.controlled_unit_id)
            .map_or(false, |u| is_alive(u));

        if controlled_alive {
            let agent_intent = action_to_intent(action_idx, self.controlled_unit_id, &self.sim);
            // Remove any existing intent for the controlled unit
            intents.retain(|i| i.unit_id != self.controlled_unit_id);
            intents.push(UnitIntent {
                unit_id: self.controlled_unit_id,
                action: agent_intent,
            });
        }

        // Step the simulation
        let (new_sim, _events) = step(self.sim.clone(), &intents, FIXED_TICK_MS);
        self.sim = new_sim;
        self.current_tick += 1;

        // Compute reward
        let cur_ally_hp: i32 = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && is_alive(u))
            .map(|u| u.hp)
            .sum();
        let cur_enemy_hp: i32 = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u))
            .map(|u| u.hp)
            .sum();
        let cur_ally_count = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Hero && is_alive(u))
            .count();
        let cur_enemy_count = self
            .sim
            .units
            .iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u))
            .count();

        let mut reward = 0.0f32;

        // +1.0 per enemy killed this tick
        if cur_enemy_count < self.prev_enemy_count {
            reward += (self.prev_enemy_count - cur_enemy_count) as f32;
        }
        // -1.0 per ally killed this tick
        if cur_ally_count < self.prev_ally_count {
            reward -= (self.prev_ally_count - cur_ally_count) as f32;
        }
        // +0.01 per damage dealt to enemies
        let enemy_dmg = (self.prev_enemy_hp - cur_enemy_hp).max(0);
        reward += enemy_dmg as f32 * 0.01;
        // -0.01 per damage taken by allies
        let ally_dmg = (self.prev_ally_hp - cur_ally_hp).max(0);
        reward -= ally_dmg as f32 * 0.01;

        // Check terminal conditions
        let heroes_alive = cur_ally_count > 0;
        let enemies_alive = cur_enemy_count > 0;
        let terminated;
        let truncated;

        if !enemies_alive {
            // Victory
            reward += 5.0;
            terminated = true;
            truncated = false;
        } else if !heroes_alive {
            // Defeat
            reward -= 5.0;
            terminated = true;
            truncated = false;
        } else if self.current_tick >= self.max_ticks {
            // Timeout
            truncated = true;
            terminated = false;
        } else {
            terminated = false;
            truncated = false;
        }

        // Update snapshots
        self.prev_ally_hp = cur_ally_hp;
        self.prev_enemy_hp = cur_enemy_hp;
        self.prev_ally_count = cur_ally_count;
        self.prev_enemy_count = cur_enemy_count;

        let obs = if terminated || truncated {
            self.reset()
        } else {
            self.observe()
        };

        Step {
            observation: obs,
            reward,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0; OBS_DIM],
            high: vec![1.0; OBS_DIM],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(NUM_ACTIONS)
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        let mask = self.compute_mask();
        Some(mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect())
    }
}

// ---------------------------------------------------------------------------
// Hero template conversion (mirrors bevy_game::mission::hero_templates)
// ---------------------------------------------------------------------------

/// Convert a HeroToml into a UnitState ready for simulation.
fn hero_toml_to_unit(toml: &HeroToml, id: u32, team: Team, position: SimVec2) -> UnitState {
    let atk = toml.attack.clone().unwrap_or_default();
    let abilities: Vec<AbilitySlot> = toml
        .abilities
        .iter()
        .map(|def| AbilitySlot::new(def.clone()))
        .collect();
    let passives: Vec<PassiveSlot> = toml
        .passives
        .iter()
        .map(|def| PassiveSlot::new(def.clone()))
        .collect();

    UnitState {
        id,
        team,
        hp: toml.stats.hp,
        max_hp: toml.stats.hp,
        position,
        move_speed_per_sec: toml.stats.move_speed,
        attack_damage: atk.damage,
        attack_range: atk.range,
        attack_cooldown_ms: atk.cooldown,
        attack_cast_time_ms: atk.cast_time,
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
        abilities,
        passives,
        status_effects: Vec::new(),
        shield_hp: 0,
        resistance_tags: toml.stats.tags.clone(),
        state_history: VecDeque::new(),
        channeling: None,
        resource: toml.stats.resource,
        max_resource: toml.stats.max_resource,
        resource_regen_per_sec: toml.stats.resource_regen_per_sec,
        owner_id: None,
        directed: false,
        armor: toml.stats.armor,
        magic_resist: toml.stats.magic_resist,
        cover_bonus: 0.0,
        elevation: 0.0,
        total_healing_done: 0,
        total_damage_done: 0,
    }
}

/// Load hero templates from a directory containing `.toml` + `.ability` file pairs.
pub fn load_hero_templates_from_dir(dir: &std::path::Path) -> Vec<HeroToml> {
    let mut templates = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("Warning: could not read directory {}", dir.display());
        return templates;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "toml") {
            let toml_str = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            // Check for a matching .ability file
            let ability_path = path.with_extension("ability");
            let dsl_content = std::fs::read_to_string(&ability_path).ok();

            let mut hero: HeroToml = match toml::from_str(&toml_str) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("Warning: failed to parse {}: {e}", path.display());
                    continue;
                }
            };

            // Parse DSL abilities if available
            if let Some(dsl_str) = &dsl_content {
                match tactical_sim::effects::dsl::parse_abilities(dsl_str) {
                    Ok((abilities, passives)) => {
                        hero.abilities = abilities;
                        hero.passives = passives;
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: failed to parse DSL {}: {e}",
                            ability_path.display()
                        );
                    }
                }
            }

            templates.push(hero);
        }
    }

    templates
}
