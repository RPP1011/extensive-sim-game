//! GPU-multiplexed episode generation for transformer RL.
//!
//! Runs multiple sims per thread with non-blocking GPU inference.

use super::transformer_rl::{
    RlEpisode, RlStep,
    apply_action_mask, apply_behavior_overrides,
    MAX_ABILITIES,
};
use super::rl_gpu_sim::{ActiveSim, PendingUnit, SimPhase, HeroPreStepState, active_sim_from_precomputed};

static DUMMY_ATOMIC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl ActiveSim {
    pub(crate) fn prepare_and_submit(
        &mut self,
        gpu: &dyn bevy_game::ai::core::ability_transformer::gpu_client::InferenceClient,
    ) -> Result<(), String> {
        self.prepare_and_submit_profiled(gpu, &DUMMY_ATOMIC, &DUMMY_ATOMIC, &DUMMY_ATOMIC, &DUMMY_ATOMIC)
    }

    pub(crate) fn prepare_and_submit_profiled(
        &mut self,
        gpu: &dyn bevy_game::ai::core::ability_transformer::gpu_client::InferenceClient,
        p_intent: &std::sync::atomic::AtomicU64,
        p_extract: &std::sync::atomic::AtomicU64,
        p_submit: &std::sync::atomic::AtomicU64,
        p_serialize: &std::sync::atomic::AtomicU64,
    ) -> Result<(), String> {
        use bevy_game::ai::core::{is_alive, FIXED_TICK_MS};
        use bevy_game::ai::core::ability_eval::{extract_game_state_v2, extract_game_state_v2_with_objectives, extract_game_state_v2_cached, extract_game_state_v2_cached_spatial, ExtractionCache, ZoneObjective};
        use bevy_game::ai::core::self_play::actions::action_mask;
        use bevy_game::ai::core::ability_transformer::gpu_client::InferenceRequest;
        use bevy_game::ai::squad::generate_intents;
        use std::sync::atomic::Ordering;

        let t0 = std::time::Instant::now();
        self.intents = generate_intents(&self.sim, &mut self.squad_ai, FIXED_TICK_MS);
        apply_behavior_overrides(&mut self.intents, &self.behavior_trees, &self.sim, self.tick);
        p_intent.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // Build extraction cache once per tick (shared across all hero extractions)
        let extraction_cache = ExtractionCache::build(&self.sim);

        let record = self.tick % self.step_interval == 0;
        let has_drill_movement = self.drill_target_position.is_some()
            || self.drill_objective_type.as_deref() == Some("reach_entity");
        let step_r = if record {
            if has_drill_movement {
                // Movement drills: base reward is 0 — all shaping comes from step_sim()
                let event_r = self.pending_event_reward;
                self.pending_event_reward = 0.0;
                event_r
            } else {
                // Combat drills: HP differential reward
                let cur_hero_hp: i32 = self.sim.units.iter()
                    .filter(|u| u.team == bevy_game::ai::core::Team::Hero).map(|u| u.hp.max(0)).sum();
                let cur_enemy_hp: i32 = self.sim.units.iter()
                    .filter(|u| u.team == bevy_game::ai::core::Team::Enemy).map(|u| u.hp.max(0)).sum();
                let enemy_dmg = (self.prev_enemy_hp - cur_enemy_hp).max(0) as f32;
                let hero_dmg = (self.prev_hero_hp - cur_hero_hp).max(0) as f32;
                let hp_reward = (enemy_dmg - hero_dmg) / self.avg_unit_hp.max(1.0);
                self.prev_hero_hp = cur_hero_hp;
                self.prev_enemy_hp = cur_enemy_hp;
                let event_r = self.pending_event_reward;
                self.pending_event_reward = 0.0;
                hp_reward + event_r - 0.01
            }
        } else { 0.0 };

        self.pending_units.clear();
        self.hero_pre_step.clear();
        self.steps_recorded_this_tick.clear();

        // Submit hero units
        for &uid in &self.hero_ids.clone() {
            let unit = match self.sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                Some(u) => u, None => continue,
            };
            if record {
                let nearest_enemy_dist = self.sim.units.iter()
                    .filter(|e| e.team == bevy_game::ai::core::Team::Enemy && e.hp > 0)
                    .map(|e| bevy_game::ai::core::distance(unit.position, e.position))
                    .fold(f32::MAX, f32::min);
                self.hero_pre_step.push(HeroPreStepState {
                    unit_id: uid, position: unit.position, hp: unit.hp,
                    nearest_enemy_dist, move_dir: 0, combat_type: 0,
                });
            }
            if unit.casting.is_some() || unit.control_remaining_ms > 0 { continue; }

            let t_ext = std::time::Instant::now();
            let mask_arr = action_mask(&self.sim, uid);
            let mask_vec: Vec<bool> = mask_arr.to_vec();
            let objectives: Vec<ZoneObjective> = if let Some(pos) = self.drill_target_position {
                vec![ZoneObjective { position: pos, radius: self.drill_target_radius.unwrap_or(1.0) }]
            } else {
                Vec::new()
            };
            let gs_v2 = extract_game_state_v2_cached_spatial(
                &self.sim, unit, &extraction_cache, &objectives,
                self.visibility_map.as_ref(), self.sim.grid_nav.as_ref(),
            );
            let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
            p_extract.fetch_add(t_ext.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let mut combat_mask_vec = vec![false; 10];
            combat_mask_vec[0] = gs_v2.entity_types.iter().any(|&t| t == 1);
            combat_mask_vec[1] = true;
            for idx in 0..n_abilities {
                if mask_arr[3 + idx] { combat_mask_vec[2 + idx] = true; }
            }
            apply_action_mask(&mut combat_mask_vec, self.action_mask.as_deref());

            let ability_cls_for_req: Vec<Option<Vec<f32>>> = (0..MAX_ABILITIES)
                .map(|i| {
                    if i < n_abilities && unit.abilities[i].cooldown_remaining_ms == 0 && mask_arr[3 + i] {
                        self.cls_cache.get(&(uid, i)).cloned()
                    } else { None }
                }).collect();

            let t_ser = std::time::Instant::now();
            let hidden = self.hidden_states.get(&uid).cloned().unwrap_or_default();
            let mut agg = gs_v2.aggregate_features.clone();
            if let Some(target) = self.drill_target_position {
                if agg.len() >= 16 {
                    agg[14] = (target[0] - unit.position.x) / 20.0;
                    agg[15] = (target[1] - unit.position.y) / 20.0;
                }
            } else if agg.len() >= 16 {
                if let Some(nearest) = self.sim.units.iter()
                    .filter(|u| u.team == bevy_game::ai::core::Team::Enemy && u.hp > 0)
                    .min_by(|a, b| {
                        let da = (a.position.x - unit.position.x).powi(2) + (a.position.y - unit.position.y).powi(2);
                        let db = (b.position.x - unit.position.x).powi(2) + (b.position.y - unit.position.y).powi(2);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                { agg[14] = (nearest.position.x - unit.position.x) / 20.0; agg[15] = (nearest.position.y - unit.position.y) / 20.0; }
            }
            // Extract corner tokens from visibility map (V6 spatial cross-attention)
            let corner_tokens: Vec<Vec<f32>> = match (&self.visibility_map, &self.sim.grid_nav) {
                (Some(vis), Some(nav)) => vis
                    .spatial_tokens_for_unit(nav, unit.position, 8)
                    .into_iter()
                    .map(|tok| tok.to_vec())
                    .collect(),
                _ => Vec::new(),
            };
            let req = InferenceRequest {
                entities: gs_v2.entities.clone(), entity_types: gs_v2.entity_types.clone(),
                threats: gs_v2.threats.clone(), positions: gs_v2.positions.clone(),
                zones: gs_v2.zones.clone(),
                combat_mask: combat_mask_vec, ability_cls: ability_cls_for_req,
                hidden_state: hidden, aggregate_features: agg,
                corner_tokens,
            };
            p_serialize.fetch_add(t_ser.elapsed().as_nanos() as u64, Ordering::Relaxed);

            let t_sub = std::time::Instant::now();
            let token = gpu.submit(req)?;
            p_submit.fetch_add(t_sub.elapsed().as_nanos() as u64, Ordering::Relaxed);

            self.pending_units.push(PendingUnit {
                unit_id: uid, token, gs_v2, mask_vec, n_abilities,
                step_reward: step_r, resolved: false, is_hero: true,
            });
        }

        // Submit enemy units for self-play
        if self.self_play_gpu {
            for &uid in &self.enemy_ids.clone() {
                let unit = match self.sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                    Some(u) => u, None => continue,
                };
                if unit.casting.is_some() || unit.control_remaining_ms > 0 { continue; }
                let mask_arr = action_mask(&self.sim, uid);
                let mask_vec: Vec<bool> = mask_arr.to_vec();
                let gs_v2 = extract_game_state_v2(&self.sim, unit);
                let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
                let mut combat_mask_vec = vec![false; 10];
                combat_mask_vec[0] = gs_v2.entity_types.iter().any(|&t| t == 1);
                combat_mask_vec[1] = true;
                for idx in 0..n_abilities {
                    if mask_arr[3 + idx] { combat_mask_vec[2 + idx] = true; }
                }
                let ability_cls_for_req: Vec<Option<Vec<f32>>> = (0..MAX_ABILITIES)
                    .map(|i| {
                        if i < n_abilities && unit.abilities[i].cooldown_remaining_ms == 0 && mask_arr[3 + i] {
                            self.cls_cache.get(&(uid, i)).cloned()
                        } else { None }
                    }).collect();
                let hidden = self.hidden_states.get(&uid).cloned().unwrap_or_default();
                let corner_tokens: Vec<Vec<f32>> = match (&self.visibility_map, &self.sim.grid_nav) {
                    (Some(vis), Some(nav)) => vis
                        .spatial_tokens_for_unit(nav, unit.position, 8)
                        .into_iter()
                        .map(|tok| tok.to_vec())
                        .collect(),
                    _ => Vec::new(),
                };
                let req = InferenceRequest {
                    entities: gs_v2.entities.clone(), entity_types: gs_v2.entity_types.clone(),
                    threats: gs_v2.threats.clone(), positions: gs_v2.positions.clone(),
                    zones: gs_v2.zones.clone(),
                    combat_mask: combat_mask_vec, ability_cls: ability_cls_for_req,
                    hidden_state: hidden, aggregate_features: gs_v2.aggregate_features.clone(),
                    corner_tokens,
                };
                if let Ok(token) = gpu.submit(req) {
                    self.pending_units.push(PendingUnit {
                        unit_id: uid, token, gs_v2, mask_vec, n_abilities,
                        step_reward: 0.0, resolved: false, is_hero: false,
                    });
                }
            }
        }

        self.phase = if self.pending_units.is_empty() { SimPhase::NeedsTick } else { SimPhase::WaitingGpu };
        if self.pending_units.is_empty() { self.step_sim(); }
        Ok(())
    }

    pub(crate) fn poll_gpu(
        &mut self,
        gpu: &dyn bevy_game::ai::core::ability_transformer::gpu_client::InferenceClient,
    ) -> Result<bool, String> {
        use bevy_game::ai::core::UnitIntent;
        use bevy_game::ai::core::ability_eval::extract_game_state;
        use bevy_game::ai::core::self_play::actions::{
            move_dir_to_intent, combat_action_to_intent, build_token_infos,
        };

        let record = self.tick % self.step_interval == 0;
        let mut all_done = true;

        for pu in &mut self.pending_units {
            if pu.resolved { continue; }
            match gpu.try_recv(pu.token) {
                Ok(Some(result)) => {
                    pu.resolved = true;
                    let uid = pu.unit_id;
                    if !result.hidden_state_out.is_empty() {
                        self.hidden_states.insert(uid, result.hidden_state_out.clone());
                    }
                    let combat_type = result.combat_type as usize;
                    let target_idx = result.target_idx as usize;
                    // Continuous target position from GPU
                    let target_pos = bevy_game::ai::core::sim_vec2(result.move_dx, result.move_dy);
                    let unit_pos = self.sim.units.iter().find(|u| u.id == uid)
                        .map(|u| u.position).unwrap_or(target_pos);
                    let step_size = self.sim.units.iter().find(|u| u.id == uid)
                        .map(|u| u.move_speed_per_sec * 0.1).unwrap_or(0.32);
                    let move_intent = if bevy_game::ai::core::distance(unit_pos, target_pos) > 0.1 {
                        let next = bevy_game::ai::core::move_towards(unit_pos, target_pos, step_size);
                        bevy_game::ai::core::IntentAction::MoveTo { position: next }
                    } else {
                        bevy_game::ai::core::IntentAction::Hold
                    };
                    let move_dir = 8usize; // legacy — not used for continuous
                    let token_infos = build_token_infos(&self.sim, uid, &pu.gs_v2.entity_types, &pu.gs_v2.positions);
                    let combat_intent = combat_action_to_intent(combat_type, target_idx, uid, &self.sim, &token_infos);
                    let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) { combat_intent } else { move_intent };
                    self.intents.retain(|i| i.unit_id != uid);
                    self.intents.push(UnitIntent { unit_id: uid, action: final_intent });

                    if record && pu.is_hero {
                        if let Some(unit) = self.sim.units.iter().find(|u| u.id == uid) {
                            let game_state = extract_game_state(&self.sim, unit);
                            let composite_lp = result.lp_move + result.lp_combat + result.lp_pointer;
                            let step_idx = self.steps.len();
                            let mut agg = pu.gs_v2.aggregate_features.clone();
                            if let Some(target) = self.drill_target_position {
                                if agg.len() >= 16 { agg[14] = (target[0] - unit.position.x) / 20.0; agg[15] = (target[1] - unit.position.y) / 20.0; }
                            } else if agg.len() >= 16 {
                                if let Some(nearest) = self.sim.units.iter()
                                    .filter(|u| u.team == bevy_game::ai::core::Team::Enemy && u.hp > 0)
                                    .min_by(|a, b| {
                                        let da = (a.position.x - unit.position.x).powi(2) + (a.position.y - unit.position.y).powi(2);
                                        let db = (b.position.x - unit.position.x).powi(2) + (b.position.y - unit.position.y).powi(2);
                                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                                    })
                                { agg[14] = (nearest.position.x - unit.position.x) / 20.0; agg[15] = (nearest.position.y - unit.position.y) / 20.0; }
                            }
                            self.steps.push(RlStep {
                                tick: self.tick, unit_id: uid,
                                game_state: game_state.to_vec(),
                                action: combat_type, log_prob: composite_lp,
                                mask: pu.mask_vec.clone(), step_reward: pu.step_reward,
                                entities: Some(pu.gs_v2.entities.clone()),
                                entity_types: Some(pu.gs_v2.entity_types.clone()),
                                threats: Some(pu.gs_v2.threats.clone()),
                                positions: Some(pu.gs_v2.positions.clone()),
                                zones: Some(pu.gs_v2.zones.clone()),
                                action_type: Some(combat_type), target_idx: Some(target_idx),
                                move_dir: Some(move_dir), combat_type: Some(combat_type),
                                lp_move: Some(result.lp_move), lp_combat: Some(result.lp_combat),
                                lp_pointer: Some(result.lp_pointer),
                                aggregate_features: if agg.is_empty() { None } else { Some(agg) },
                                target_move_pos: Some([result.move_dx, result.move_dy]),
                                teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
                            });
                            self.steps_recorded_this_tick.push(step_idx);
                            if let Some(ps) = self.hero_pre_step.iter_mut().find(|p| p.unit_id == uid) {
                                ps.move_dir = move_dir; ps.combat_type = combat_type;
                            }
                        }
                    }
                }
                Ok(None) => { all_done = false; }
                Err(e) => { eprintln!("GPU inference error for unit {}: {e}", pu.unit_id); pu.resolved = true; }
            }
        }

        if all_done { self.pending_units.clear(); self.step_sim(); Ok(true) } else { Ok(false) }
    }

    pub(crate) fn step_sim(&mut self) {
        use bevy_game::ai::core::{step, Team, FIXED_TICK_MS, distance};

        let (new_sim, events) = step(self.sim.clone(), &self.intents, FIXED_TICK_MS);
        for ev in &events {
            if let bevy_game::ai::core::SimEvent::UnitDied { unit_id, .. } = ev {
                if let Some(dead_unit) = new_sim.units.iter().find(|u| u.id == *unit_id) {
                    if dead_unit.team == Team::Enemy { self.pending_event_reward += 0.3 / self.initial_enemy_count.max(1.0); }
                    else if dead_unit.team == Team::Hero { self.pending_event_reward -= 0.4 / self.initial_hero_count.max(1.0); }
                }
            }
        }

        for &step_idx in &self.steps_recorded_this_tick {
            let uid = self.steps[step_idx].unit_id;
            if let Some(pre) = self.hero_pre_step.iter().find(|p| p.unit_id == uid) {
                let mut action_reward: f32 = 0.0;
                if let Some(post_unit) = new_sim.units.iter().find(|u| u.id == uid) {
                    // --- Drill objective reward (dense, per-step) ---
                    if let Some(target) = self.drill_target_position {
                        let target_pos = bevy_game::ai::core::sim_vec2(target[0], target[1]);
                        let pre_dist = distance(pre.position, target_pos);
                        let post_dist = distance(post_unit.position, target_pos);

                        // Reward = distance reduction toward target, normalized by initial distance
                        // This gives ~+1.0 total if the unit goes straight to target
                        let dist_delta = pre_dist - post_dist;
                        let initial_dist = pre_dist.max(1.0); // avoid div by zero
                        action_reward += dist_delta / initial_dist * 3.0;

                        // Big bonus for reaching target
                        let radius = self.drill_target_radius.unwrap_or(1.0);
                        if post_dist <= radius {
                            action_reward += 5.0;
                        }
                    }

                    // --- Reach-entity objective ---
                    if self.drill_objective_type.as_deref() == Some("reach_entity") {
                        if let Some(nearest_enemy) = new_sim.units.iter()
                            .filter(|e| e.team == Team::Enemy && e.hp > 0)
                            .min_by(|a, b| {
                                let da = distance(post_unit.position, a.position);
                                let db = distance(post_unit.position, b.position);
                                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                            })
                        {
                            let pre_dist = pre.nearest_enemy_dist;
                            let post_dist = distance(post_unit.position, nearest_enemy.position);
                            if pre_dist < f32::MAX {
                                action_reward += 1.0 * (pre_dist - post_dist);
                            }
                            let radius = self.drill_target_radius.unwrap_or(1.5);
                            if post_dist <= radius { action_reward += 2.0; }
                        }
                    }

                    // --- Combat reward shaping (for drills with enemies) ---
                    let post_nearest = new_sim.units.iter()
                        .filter(|e| e.team == Team::Enemy && e.hp > 0)
                        .map(|e| distance(post_unit.position, e.position))
                        .fold(f32::MAX, f32::min);
                    if self.drill_target_position.is_none() && self.drill_objective_type.as_deref() != Some("reach_entity") {
                        // Only apply combat-approach reward when there's no specific movement target
                        if pre.nearest_enemy_dist < f32::MAX && post_nearest < f32::MAX {
                            action_reward += 0.002 * (pre.nearest_enemy_dist - post_nearest);
                        }
                        if post_nearest < 100.0 { action_reward += 0.01; }
                    }
                }
                if self.drill_target_position.is_none() {
                    // Combat type bonuses only when not doing movement drills
                    if pre.combat_type == 1 && pre.nearest_enemy_dist < 150.0 { action_reward -= 0.02; }
                    if pre.combat_type == 0 || pre.combat_type >= 2 { action_reward += 0.01; }
                }
                self.steps[step_idx].step_reward += action_reward;
            }
        }
        self.steps_recorded_this_tick.clear();
        self.sim = new_sim;
        self.tick += 1;
        self.phase = SimPhase::NeedsTick;
    }
}

/// Run GPU-multiplexed episode generation.
pub(crate) fn run_gpu_multiplexed(
    gpu: &std::sync::Arc<dyn bevy_game::ai::core::ability_transformer::gpu_client::InferenceClient>,
    precomputed: std::sync::Arc<Vec<super::rl_gpu_sim::PrecomputedScenario>>,
    episode_tasks: &[(usize, usize)],
    threads: usize, sims_per_thread: usize,
    temperature: f32, step_interval: u64,
    self_play_gpu: bool,
) -> Vec<RlEpisode> {
    use crossbeam_channel::{bounded, Sender, Receiver};
    use std::sync::atomic::{AtomicU64, Ordering};

    let (task_tx, task_rx): (Sender<(usize, usize)>, Receiver<(usize, usize)>) = bounded(episode_tasks.len());
    for &task in episode_tasks { task_tx.send(task).unwrap(); }
    drop(task_tx);

    let results: std::sync::Arc<std::sync::Mutex<Vec<RlEpisode>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::with_capacity(episode_tasks.len())));

    let precomputed_arc = precomputed;

    let prof_intent_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_extract_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_submit_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_poll_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_ticks = std::sync::Arc::new(AtomicU64::new(0));
    let prof_polls = std::sync::Arc::new(AtomicU64::new(0));
    let prof_poll_misses = std::sync::Arc::new(AtomicU64::new(0));
    let prof_gpu_wait_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_make_sim_ns = std::sync::Arc::new(AtomicU64::new(0));
    let prof_serialize_ns = std::sync::Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..threads).map(|_| {
        let gpu = gpu.clone();
        let task_rx = task_rx.clone();
        let results = results.clone();
        let pre_arc = precomputed_arc.clone();
        let p_intent = prof_intent_ns.clone();
        let p_extract = prof_extract_ns.clone();
        let p_submit = prof_submit_ns.clone();
        let p_poll = prof_poll_ns.clone();
        let p_ticks = prof_ticks.clone();
        let p_polls = prof_polls.clone();
        let p_poll_misses = prof_poll_misses.clone();
        let p_gpu_wait = prof_gpu_wait_ns.clone();
        let p_make_sim = prof_make_sim_ns.clone();
        let p_serialize = prof_serialize_ns.clone();

        std::thread::spawn(move || {
            let mut active: Vec<ActiveSim> = Vec::with_capacity(sims_per_thread);
            while active.len() < sims_per_thread {
                match task_rx.try_recv() {
                    Ok((si, ei)) => {
                        let t0 = std::time::Instant::now();
                        let asim = active_sim_from_precomputed(&pre_arc[si], si, ei, step_interval, temperature, self_play_gpu);
                        active.push(asim);
                        p_make_sim.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    Err(_) => break,
                }
            }
            while !active.is_empty() {
                let mut completed_indices = Vec::new();
                for (i, asim) in active.iter_mut().enumerate() {
                    match asim.phase {
                        SimPhase::NeedsTick => {
                            if asim.is_done() { completed_indices.push(i); continue; }
                            p_ticks.fetch_add(1, Ordering::Relaxed);
                            if let Err(e) = asim.prepare_and_submit_profiled(&*gpu, &p_intent, &p_extract, &p_submit, &p_serialize) {
                                eprintln!("GPU submit error: {e}"); completed_indices.push(i);
                            }
                        }
                        SimPhase::WaitingGpu => {
                            let t0 = std::time::Instant::now();
                            p_polls.fetch_add(1, Ordering::Relaxed);
                            match asim.poll_gpu(&*gpu) {
                                Ok(true) => { p_poll.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed); if asim.is_done() { completed_indices.push(i); } }
                                Ok(false) => { p_poll_misses.fetch_add(1, Ordering::Relaxed); p_gpu_wait.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed); }
                                Err(e) => { eprintln!("GPU poll error: {e}"); completed_indices.push(i); }
                            }
                        }
                    }
                }
                completed_indices.sort_unstable(); completed_indices.dedup();
                let mut result_batch = Vec::new();
                for &idx in completed_indices.iter().rev() { result_batch.push(active.swap_remove(idx).into_episode()); }
                if !result_batch.is_empty() { results.lock().unwrap().extend(result_batch); }
                while active.len() < sims_per_thread {
                    match task_rx.try_recv() {
                        Ok((si, ei)) => {
                            let t0 = std::time::Instant::now();
                            let asim = active_sim_from_precomputed(&pre_arc[si], si, ei, step_interval, temperature, self_play_gpu);
                            active.push(asim);
                            p_make_sim.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        Err(_) => break,
                    }
                }
                if active.iter().all(|s| s.phase == SimPhase::WaitingGpu) {
                    let epoch = gpu.batch_epoch();
                    gpu.wait_for_batch(epoch);
                }
            }
        })
    }).collect();

    for h in handles { h.join().unwrap(); }

    let ticks = prof_ticks.load(Ordering::Relaxed).max(1);
    let polls = prof_polls.load(Ordering::Relaxed).max(1);
    let poll_misses = prof_poll_misses.load(Ordering::Relaxed);
    eprintln!("\n--- GPU Multiplexed Profiling ({} ticks, {} polls, {} poll misses) ---", ticks, polls, poll_misses);
    let ms = |v: &AtomicU64| v.load(Ordering::Relaxed) as f64 / 1e6;
    eprintln!("  make_sim:    {:>8.1}ms total", ms(&prof_make_sim_ns));
    eprintln!("  intents:     {:>8.1}ms total", ms(&prof_intent_ns));
    eprintln!("  extract:     {:>8.1}ms total", ms(&prof_extract_ns));
    eprintln!("  serialize:   {:>8.1}ms total", ms(&prof_serialize_ns));
    eprintln!("  submit:      {:>8.1}ms total", ms(&prof_submit_ns));
    eprintln!("  poll (hit):  {:>8.1}ms total", ms(&prof_poll_ns));
    eprintln!("  gpu_wait:    {:>8.1}ms total ({} misses)", ms(&prof_gpu_wait_ns), poll_misses);

    std::sync::Arc::try_unwrap(results).unwrap().into_inner().unwrap()
}
