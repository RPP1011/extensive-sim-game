//! CombatEnv — wraps the tactical combat simulator as an rl4burn `Env`.
//!
//! Uses parameter-shared independent PPO (IPPO): each hero gets its own
//! observation and action, but all share the same policy network.
//! The env exposes one hero at a time in round-robin order.

use std::collections::{HashMap, VecDeque};

use rand::{RngExt, SeedableRng};
use rl4burn::{Env, Space, Step};

use tactical_sim::effects::{AbilitySlot, HeroToml, PassiveSlot};
use tactical_sim::sim::ability_eval::{
    extract_game_state_v2, ENTITY_FEATURE_DIM, NUM_ENTITY_SLOTS,
    THREAT_FEATURE_DIM,
};
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

/// Parameter-shared combat environment (IPPO style).
///
/// Each hero takes a turn providing observations and receiving actions.
/// After ALL heroes have acted, the sim advances one tick. This way
/// a single shared policy network learns to play any hero position,
/// and coordination emerges from seeing allies' states.
pub struct CombatEnv {
    sim: SimState,
    squad_ai: SquadAiState,
    hero_ids: Vec<u32>,
    hero_templates: Vec<HeroToml>,
    enemy_templates: Vec<HeroToml>,
    max_ticks: u64,
    current_tick: u64,
    rng: rand::rngs::SmallRng,
    /// Which hero's turn it is (0..n_heroes, then tick advances)
    current_hero_idx: usize,
    /// Accumulated intents for this tick (filled as each hero acts)
    pending_intents: Vec<UnitIntent>,
    /// HP snapshots for reward shaping
    prev_ally_hp: i32,
    prev_enemy_hp: i32,
    prev_enemy_count: usize,
    prev_ally_count: usize,
    /// Per-hero reward accumulator (shared reward at tick boundary)
    tick_reward: f32,
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
            hero_ids: Vec::new(),
            hero_templates,
            enemy_templates,
            max_ticks: MAX_TICKS,
            current_tick: 0,
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
            current_hero_idx: 0,
            pending_intents: Vec::new(),
            prev_ally_hp: 0,
            prev_enemy_hp: 0,
            prev_enemy_count: 0,
            prev_ally_count: 0,
            tick_reward: 0.0,
        };
        env.reset_internal();
        env
    }

    fn reset_internal(&mut self) {
        let n_heroes = self.rng.random_range(2u32..=4);
        let n_enemies = self.rng.random_range(2u32..=4);

        let mut units = Vec::new();
        let mut id = 1u32;

        for i in 0..n_heroes {
            let idx = self.rng.random_range(0..self.hero_templates.len());
            let x = -4.0 + (i as f32) * 2.0;
            let y = self.rng.random_range(-2.0f32..2.0);
            let unit = hero_toml_to_unit(&self.hero_templates[idx], id, Team::Hero, sim_vec2(x, y));
            units.push(unit);
            id += 1;
        }

        self.hero_ids = (1..=n_heroes).collect();

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

        let personalities: HashMap<u32, Personality> = self
            .sim
            .units
            .iter()
            .map(|u| (u.id, Personality::default()))
            .collect();
        self.squad_ai = SquadAiState::new(&self.sim, personalities);

        self.current_hero_idx = 0;
        self.current_tick = 0;
        self.pending_intents.clear();
        self.tick_reward = 0.0;

        self.prev_ally_hp = self.sim.units.iter()
            .filter(|u| u.team == Team::Hero && is_alive(u))
            .map(|u| u.hp).sum();
        self.prev_enemy_hp = self.sim.units.iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u))
            .map(|u| u.hp).sum();
        self.prev_ally_count = self.sim.units.iter()
            .filter(|u| u.team == Team::Hero && is_alive(u)).count();
        self.prev_enemy_count = self.sim.units.iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u)).count();
    }

    fn current_hero_id(&self) -> u32 {
        self.hero_ids[self.current_hero_idx % self.hero_ids.len()]
    }

    fn observe_current_hero(&self) -> Vec<f32> {
        let uid = self.current_hero_id();
        match self.sim.units.iter().find(|u| u.id == uid) {
            Some(unit) if is_alive(unit) => extract_game_state(&self.sim, unit),
            _ => vec![0.0; OBS_DIM],
        }
    }

    fn advance_tick(&mut self) -> (f32, bool, bool) {
        // Generate enemy intents via squad AI
        let mut intents = generate_intents(&self.sim, &mut self.squad_ai, FIXED_TICK_MS);
        // Remove any enemy-generated intents for hero units (we control them)
        for &hid in &self.hero_ids {
            intents.retain(|i| i.unit_id != hid);
        }
        // Add our accumulated hero intents
        intents.extend(self.pending_intents.drain(..));

        // Step the simulation
        let (new_sim, _events) = step(self.sim.clone(), &intents, FIXED_TICK_MS);
        self.sim = new_sim;
        self.current_tick += 1;

        // Compute tick reward
        let cur_ally_hp: i32 = self.sim.units.iter()
            .filter(|u| u.team == Team::Hero && is_alive(u))
            .map(|u| u.hp).sum();
        let cur_enemy_hp: i32 = self.sim.units.iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u))
            .map(|u| u.hp).sum();
        let cur_ally_count = self.sim.units.iter()
            .filter(|u| u.team == Team::Hero && is_alive(u)).count();
        let cur_enemy_count = self.sim.units.iter()
            .filter(|u| u.team == Team::Enemy && is_alive(u)).count();

        let mut reward = 0.0f32;
        if cur_enemy_count < self.prev_enemy_count {
            reward += (self.prev_enemy_count - cur_enemy_count) as f32 * 2.0;
        }
        if cur_ally_count < self.prev_ally_count {
            reward -= (self.prev_ally_count - cur_ally_count) as f32 * 1.0;
        }
        let enemy_dmg = (self.prev_enemy_hp - cur_enemy_hp).max(0);
        reward += enemy_dmg as f32 * 0.01;
        let ally_dmg = (self.prev_ally_hp - cur_ally_hp).max(0);
        reward -= ally_dmg as f32 * 0.005;

        let heroes_alive = cur_ally_count > 0;
        let enemies_alive = cur_enemy_count > 0;

        let (terminated, truncated) = if !heroes_alive {
            reward -= 5.0;
            (true, false)
        } else if !enemies_alive {
            reward += 10.0;
            (true, false)
        } else if self.current_tick >= self.max_ticks {
            (false, true)
        } else {
            (false, false)
        };

        self.prev_ally_hp = cur_ally_hp;
        self.prev_enemy_hp = cur_enemy_hp;
        self.prev_ally_count = cur_ally_count;
        self.prev_enemy_count = cur_enemy_count;

        (reward, terminated, truncated)
    }
}

impl Env for CombatEnv {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.reset_internal();
        self.observe_current_hero()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let action_idx = action[0] as usize;
        let hero_id = self.current_hero_id();

        // Record this hero's intent
        let alive = self.sim.units.iter()
            .find(|u| u.id == hero_id)
            .map_or(false, |u| is_alive(u));
        if alive {
            let intent = action_to_intent(action_idx, hero_id, &self.sim);
            self.pending_intents.push(UnitIntent {
                unit_id: hero_id,
                action: intent,
            });
        }

        // Move to next hero
        self.current_hero_idx += 1;

        // If all heroes have acted, advance the sim tick
        let (reward, terminated, truncated) = if self.current_hero_idx >= self.hero_ids.len() {
            self.current_hero_idx = 0;
            let (r, t, tr) = self.advance_tick();
            self.tick_reward = r;
            (r, t, tr)
        } else {
            // Not all heroes have acted yet — no tick advance, no reward
            (0.0, false, false)
        };

        let obs = if terminated || truncated {
            self.reset()
        } else {
            self.observe_current_hero()
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
        let hero_id = self.current_hero_id();
        let mask = action_mask(&self.sim, hero_id);
        Some(mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect())
    }
}

// ---------------------------------------------------------------------------
// Hero template loading
// ---------------------------------------------------------------------------

pub fn load_hero_templates_from_dir(dir: &std::path::Path) -> Vec<HeroToml> {
    let mut templates = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return templates,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let toml_str = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        // Try to load companion .ability file
        let ability_path = path.with_extension("ability");
        let dsl_content = std::fs::read_to_string(&ability_path).ok();

        let toml: Result<HeroToml, String> = (|| -> Result<HeroToml, String> {
            let mut parsed: HeroToml = toml::from_str(&toml_str)
                .map_err(|e| format!("parse error: {e}"))?;
            if let Some(ref dsl) = dsl_content {
                if let Ok((abilities, passives)) = tactical_sim::effects::dsl::parse_abilities(dsl) {
                    parsed.abilities = abilities;
                    parsed.passives = passives;
                }
            }
            Ok(parsed)
        })();

        match toml {
            Ok(t) => templates.push(t),
            Err(_) => {} // skip unparseable templates
        }
    }
    templates
}

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
        position,
        hp: toml.stats.hp,
        max_hp: toml.stats.hp,
        attack_damage: atk.damage,
        attack_range: atk.range,
        attack_cooldown_ms: atk.cooldown,
        attack_cast_time_ms: atk.cast_time,
        move_speed_per_sec: toml.stats.move_speed,
        abilities,
        passives,
        resistance_tags: toml.stats.tags.clone(),
        armor: toml.stats.armor,
        magic_resist: toml.stats.magic_resist,
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
        status_effects: Vec::new(),
        shield_hp: 0,
        state_history: VecDeque::new(),
        channeling: None,
        resource: toml.stats.resource,
        max_resource: toml.stats.max_resource,
        resource_regen_per_sec: toml.stats.resource_regen_per_sec,
        owner_id: None,
        directed: false,
        total_healing_done: 0,
        total_damage_done: 0,
        cover_bonus: 0.0,
        elevation: 0.0,
    }
}
