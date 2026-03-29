//! Episode runner for transformer RL: runs a single scenario with a policy.

use std::sync::atomic::{AtomicU64, Ordering};

/// Global profiling counters (aggregated across all par_iter threads).
pub(crate) static PROF_INTENTS_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_POLICY_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_STEP_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_TERRAIN_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_SETUP_NS: AtomicU64 = AtomicU64::new(0);
pub(crate) static PROF_TICKS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn reset_profiling() {
    PROF_INTENTS_NS.store(0, Ordering::Relaxed);
    PROF_POLICY_NS.store(0, Ordering::Relaxed);
    PROF_STEP_NS.store(0, Ordering::Relaxed);
    PROF_TERRAIN_NS.store(0, Ordering::Relaxed);
    PROF_SETUP_NS.store(0, Ordering::Relaxed);
    PROF_TICKS.store(0, Ordering::Relaxed);
}

use super::transformer_rl::{
    Policy, RlEpisode, RlStep,
    MAX_ABILITIES,
};
use super::rl_policies::{
    apply_random_policy,
    apply_v5_policy, check_drill_objective,
};

// ---------------------------------------------------------------------------
// Episode runner
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_rl_episode(
    initial_sim: bevy_game::ai::core::SimState,
    initial_squad_ai: bevy_game::ai::squad::SquadAiState,
    scenario_name: &str,
    max_ticks: u64,
    policy: &Policy,
    tokenizer: &bevy_game::ai::core::ability_transformer::tokenizer::AbilityTokenizer,
    temperature: f32,
    rng_seed: u64,
    step_interval: u64,
    grid_nav: Option<bevy_game::ai::pathing::GridNav>,
    embedding_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    enemy_policy: Option<&Policy>,
    enemy_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    drill_objective: Option<&bevy_game::scenario::ObjectiveDef>,
    scenario_action_mask: Option<&str>,
) -> RlEpisode {
    use bevy_game::ai::core::{is_alive, step, distance, Team, FIXED_TICK_MS};
    use bevy_game::ai::core::ability_eval::{extract_game_state_v2, extract_game_state_v2_spatial, extract_game_state_v2_with_objectives, ZoneObjective};
    use bevy_game::ai::core::self_play::actions::{action_mask, intent_to_action};
    use bevy_game::ai::effects::dsl::emit::emit_ability_dsl;
    use bevy_game::ai::goap::spatial::VisibilityMap;
    use bevy_game::ai::squad::generate_intents;

    let mut sim = initial_sim;
    if let Some(nav) = grid_nav {
        sim.grid_nav = Some(nav);
    }

    // Build spatial visibility map once at episode start (if room geometry available)
    let vis_map: Option<VisibilityMap> = sim.grid_nav.as_ref().map(VisibilityMap::build);
    let mut squad_ai = initial_squad_ai;
    let mut rng = rng_seed;
    let mut steps = Vec::new();

    // Build zone objectives from drill objective
    let zone_objectives: Vec<ZoneObjective> = if let Some(obj) = drill_objective {
        if let Some(pos) = obj.position {
            vec![ZoneObjective { position: pos, radius: obj.radius.unwrap_or(1.0) }]
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let hero_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Hero)
        .map(|u| u.id)
        .collect();

    // Pre-tokenize and cache CLS embeddings per hero ability
    let mut unit_abilities: std::collections::HashMap<u32, Vec<Vec<u32>>> =
        std::collections::HashMap::new();
    let mut unit_ability_names: std::collections::HashMap<u32, Vec<String>> =
        std::collections::HashMap::new();
    let mut cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>> =
        std::collections::HashMap::new();

    for &uid in &hero_ids {
        if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
            let mut ability_tokens_list = Vec::new();
            let mut ability_names_list = Vec::new();
            for (idx, slot) in unit.abilities.iter().enumerate() {
                let dsl = emit_ability_dsl(&slot.def);
                let tokens = tokenizer.encode_with_cls(&dsl);

                let safe_name = slot.def.name.replace(' ', "_");
                if let Some(reg) = embedding_registry {
                    if let Some(reg_cls) = reg.get(&safe_name) {
                        let projected = policy.project_external_cls(reg_cls);
                        cls_cache.insert((uid, idx), projected);
                    } else if policy.needs_transformer() {
                        let cls = policy.encode_cls(&tokens);
                        cls_cache.insert((uid, idx), cls);
                    }
                } else if policy.needs_transformer() {
                    let cls = policy.encode_cls(&tokens);
                    cls_cache.insert((uid, idx), cls);
                }

                ability_tokens_list.push(tokens);
                ability_names_list.push(slot.def.name.clone());
            }
            unit_abilities.insert(uid, ability_tokens_list);
            unit_ability_names.insert(uid, ability_names_list);
        }
    }

    // Self-play: set up enemy policy CLS cache
    let enemy_ids: Vec<u32> = sim.units.iter()
        .filter(|u| u.team == Team::Enemy)
        .map(|u| u.id)
        .collect();
    let mut enemy_cls_cache: std::collections::HashMap<(u32, usize), Vec<f32>> =
        std::collections::HashMap::new();
    if let Some(ep) = enemy_policy {
        for &uid in &enemy_ids {
            if let Some(unit) = sim.units.iter().find(|u| u.id == uid) {
                for (idx, slot) in unit.abilities.iter().enumerate() {
                    let dsl = emit_ability_dsl(&slot.def);
                    let tokens = tokenizer.encode_with_cls(&dsl);
                    let safe_name = slot.def.name.replace(' ', "_");
                    if let Some(reg) = enemy_registry {
                        if let Some(reg_cls) = reg.get(&safe_name) {
                            let projected = ep.project_external_cls(reg_cls);
                            enemy_cls_cache.insert((uid, idx), projected);
                        } else if ep.needs_transformer() {
                            let cls = ep.encode_cls(&tokens);
                            enemy_cls_cache.insert((uid, idx), cls);
                        }
                    } else if ep.needs_transformer() {
                        let cls = ep.encode_cls(&tokens);
                        enemy_cls_cache.insert((uid, idx), cls);
                    }
                }
            }
        }
    }

    // Dense reward tracking
    let mut prev_hero_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Hero).map(|u| u.hp).sum();
    let mut prev_enemy_hp: i32 = sim.units.iter()
        .filter(|u| u.team == Team::Enemy).map(|u| u.hp).sum();
    let n_units = sim.units.iter().filter(|u| u.hp > 0).count().max(1) as f32;
    let avg_unit_hp = (prev_hero_hp + prev_enemy_hp) as f32 / n_units;
    let initial_enemy_count = sim.units.iter()
        .filter(|u| u.team == Team::Enemy && u.hp > 0).count() as f32;
    let mut pending_event_reward: f32 = 0.0;

    let mut t_intents_ns: u64 = 0;
    let mut t_policy_ns: u64 = 0;
    let mut t_step_ns: u64 = 0;
    let mut t_terrain_ns: u64 = 0;
    // Flush per-episode counters to global atomics on any exit path
    macro_rules! flush_prof {
        () => {
            PROF_INTENTS_NS.fetch_add(t_intents_ns, Ordering::Relaxed);
            PROF_POLICY_NS.fetch_add(t_policy_ns, Ordering::Relaxed);
            PROF_STEP_NS.fetch_add(t_step_ns, Ordering::Relaxed);
            PROF_TERRAIN_NS.fetch_add(t_terrain_ns, Ordering::Relaxed);
        };
    }

    for tick in 0..max_ticks {
        let t0 = std::time::Instant::now();
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        t_intents_ns += t0.elapsed().as_nanos() as u64;
        let record = tick % step_interval == 0;
        let t0 = std::time::Instant::now();

        // Compute dense step reward
        let step_r = if record {
            let cur_hero_hp: i32 = sim.units.iter()
                .filter(|u| u.team == Team::Hero).map(|u| u.hp.max(0)).sum();
            let cur_enemy_hp: i32 = sim.units.iter()
                .filter(|u| u.team == Team::Enemy).map(|u| u.hp.max(0)).sum();
            let enemy_dmg = (prev_enemy_hp - cur_enemy_hp).max(0) as f32;
            let hero_dmg = (prev_hero_hp - cur_hero_hp).max(0) as f32;
            let hp_reward = (enemy_dmg - hero_dmg) / avg_unit_hp.max(1.0);
            prev_hero_hp = cur_hero_hp;
            prev_enemy_hp = cur_enemy_hp;
            let event_r = pending_event_reward;
            pending_event_reward = 0.0;
            hp_reward + event_r
        } else {
            0.0
        };

        // Combined policy path: use squad AI as-is, just record
        if matches!(policy, Policy::Combined) {
            if record {
                use bevy_game::ai::core::self_play::actions::{
                    build_token_infos, intent_to_v3_action, intent_to_v4_action,
                };
                for &uid in &hero_ids {
                    let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                        Some(u) => u,
                        None => continue,
                    };
                    if unit.casting.is_some() || unit.control_remaining_ms > 0 { continue; }
                    let mask_arr = action_mask(&sim, uid);
                    let intent_action = intents.iter()
                        .find(|i| i.unit_id == uid)
                        .map(|i| &i.action)
                        .cloned()
                        .unwrap_or(bevy_game::ai::core::IntentAction::Hold);
                    let action = intent_to_action(&intent_action, uid, &sim);
                    let gs_v2 = if !zone_objectives.is_empty() {
                        extract_game_state_v2_with_objectives(
                            &sim, unit, vis_map.as_ref(), sim.grid_nav.as_ref(), &zone_objectives,
                        )
                    } else {
                        match (&vis_map, sim.grid_nav.as_ref()) {
                            (Some(vm), Some(nav)) => extract_game_state_v2_spatial(&sim, unit, vm, nav),
                            _ => extract_game_state_v2(&sim, unit),
                        }
                    };
                    let token_infos = build_token_infos(
                        &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                    );
                    let (v3_action_type, v3_target_idx) = intent_to_v3_action(
                        &intent_action, uid, &sim, &token_infos,
                    ).unwrap_or((2, 0));
                    let (v4_move_dir, v4_combat_type, _v4_target_idx) = intent_to_v4_action(
                        &intent_action, uid, &sim, &token_infos,
                    ).unwrap_or((8, 1, 0));
                    // Derive target_move_pos from intent for V6 movement head training
                    let target_move_pos = match &intent_action {
                        bevy_game::ai::core::IntentAction::MoveTo { position } => Some([position.x, position.y]),
                        bevy_game::ai::core::IntentAction::Skulk { objective } => Some([objective.x, objective.y]),
                        bevy_game::ai::core::IntentAction::Attack { target_id }
                        | bevy_game::ai::core::IntentAction::CastAbility { target_id }
                        | bevy_game::ai::core::IntentAction::CastHeal { target_id }
                        | bevy_game::ai::core::IntentAction::CastControl { target_id } => {
                            sim.units.iter().find(|u| u.id == *target_id)
                                .map(|t| [t.position.x, t.position.y])
                        }
                        bevy_game::ai::core::IntentAction::UseAbility { target, .. } => {
                            match target {
                                bevy_game::ai::effects::AbilityTarget::Unit(tid) =>
                                    sim.units.iter().find(|u| u.id == *tid).map(|t| [t.position.x, t.position.y]),
                                bevy_game::ai::effects::AbilityTarget::Position(p) => Some([p.x, p.y]),
                                bevy_game::ai::effects::AbilityTarget::None => Some([unit.position.x, unit.position.y]),
                            }
                        }
                        _ => Some([unit.position.x, unit.position.y]), // Hold/Hide → stay at current position
                    };
                    steps.push(RlStep {
                        tick, unit_id: uid,
                        game_state: vec![],
                        action, log_prob: 0.0,
                        mask: mask_arr.to_vec(),
                        step_reward: step_r,
                        entities: Some(gs_v2.entities),
                        entity_types: Some(gs_v2.entity_types),
                        threats: Some(gs_v2.threats),
                        positions: Some(gs_v2.positions),
                        zones: Some(gs_v2.zones),
                        action_type: Some(v3_action_type),
                        target_idx: Some(v3_target_idx),
                        move_dir: Some(v4_move_dir),
                        combat_type: Some(v4_combat_type),
                        lp_move: None, lp_combat: None,
                        lp_pointer: None,
                        aggregate_features: if gs_v2.aggregate_features.is_empty() { None } else { Some(gs_v2.aggregate_features) },
                        target_move_pos,
                        teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
                    });
                }
            }
        } else {
            // V5/Random policies: override hero intents with policy output
            for &uid in &hero_ids {
                let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                    Some(u) => u,
                    None => continue,
                };
                if unit.casting.is_some() || unit.control_remaining_ms > 0 { continue; }
                let mask_arr = action_mask(&sim, uid);
                let mask_vec: Vec<bool> = mask_arr.to_vec();

                // Random policy
                if matches!(policy, Policy::Random) {
                    apply_random_policy(
                        &sim, unit, uid, &mask_arr, &mask_vec,
                        scenario_action_mask, record, step_r, tick,
                        &mut rng, &mut intents, &mut steps,
                        vis_map.as_ref(), sim.grid_nav.as_ref(),
                    );
                    continue;
                }

                let gs_v2 = if !zone_objectives.is_empty() {
                    extract_game_state_v2_with_objectives(
                        &sim, unit, vis_map.as_ref(), sim.grid_nav.as_ref(), &zone_objectives,
                    )
                } else {
                    match (&vis_map, sim.grid_nav.as_ref()) {
                        (Some(vm), Some(nav)) => extract_game_state_v2_spatial(&sim, unit, vm, nav),
                        _ => extract_game_state_v2(&sim, unit),
                    }
                };
                let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
                let mut ability_cls_refs: Vec<Option<&[f32]>> = vec![None; MAX_ABILITIES];
                for idx in 0..n_abilities {
                    if unit.abilities[idx].cooldown_remaining_ms == 0 && mask_arr[3 + idx] {
                        if let Some(cls) = cls_cache.get(&(uid, idx)) {
                            ability_cls_refs[idx] = Some(cls.as_slice());
                        }
                    }
                }

                // V5 dual-head policy
                if let Policy::ActorCriticV5(ac) = policy {
                    apply_v5_policy(
                        ac, &sim, unit, uid, &gs_v2, &ability_cls_refs,
                        &mask_arr, &mask_vec, scenario_action_mask,
                        temperature, record, step_r, tick,
                        &mut rng, &mut intents, &mut steps,
                    );
                    continue;
                }

                // V6 Burn GPU inference
                #[cfg(feature = "burn-gpu")]
                if let Policy::BurnServerV6(client) = policy {
                    use bevy_game::ai::core::burn_model::inference::{InferenceRequest, InferenceClient};
                    use bevy_game::ai::core::{UnitIntent, Team};

                    let h_dim = client.h_dim();
                    let req = InferenceRequest {
                        entities: gs_v2.entities.clone(),
                        entity_types: gs_v2.entity_types.clone(),
                        zones: gs_v2.zones.clone(),
                        combat_mask: mask_vec.clone(),
                        ability_cls: vec![None; MAX_ABILITIES],
                        hidden_state: vec![0.0; h_dim],
                        aggregate_features: gs_v2.aggregate_features.clone(),
                        corner_tokens: vec![],
                    };

                    if let Ok(result) = client.infer(req) {
                        // Convert model output to intent
                        let combat_type = result.combat_type as usize;
                        let target_idx = result.target_idx as usize;
                        let target_pos = [
                            unit.position.x + result.move_dx * 20.0,
                            unit.position.y + result.move_dy * 20.0,
                        ];

                        let combat_intent = if combat_type == 0 {
                            // Attack: find target entity
                            let enemy_entities: Vec<&bevy_game::ai::core::UnitState> = sim.units.iter()
                                .filter(|u| u.team == Team::Enemy && u.hp > 0).collect();
                            if let Some(target) = enemy_entities.get(target_idx.min(enemy_entities.len().saturating_sub(1))) {
                                bevy_game::ai::core::IntentAction::Attack { target_id: target.id }
                            } else {
                                bevy_game::ai::core::IntentAction::Hold
                            }
                        } else if combat_type == 1 {
                            bevy_game::ai::core::IntentAction::Hold
                        } else {
                            // Ability usage
                            let ability_idx = combat_type - 2;
                            if ability_idx < unit.abilities.len() && mask_arr.get(3 + ability_idx).copied().unwrap_or(false) {
                                let enemy = sim.units.iter()
                                    .filter(|u| u.team == Team::Enemy && u.hp > 0).next();
                                if let Some(e) = enemy {
                                    bevy_game::ai::core::IntentAction::UseAbility {
                                        ability_index: ability_idx,
                                        target: bevy_game::ai::effects::AbilityTarget::Unit(e.id),
                                    }
                                } else {
                                    bevy_game::ai::core::IntentAction::Hold
                                }
                            } else {
                                bevy_game::ai::core::IntentAction::Hold
                            }
                        };

                        let move_intent = bevy_game::ai::core::IntentAction::MoveTo {
                            position: bevy_game::ai::core::SimVec2 { x: target_pos[0], y: target_pos[1] },
                        };
                        let final_intent = if !matches!(combat_intent, bevy_game::ai::core::IntentAction::Hold) {
                            combat_intent
                        } else {
                            move_intent
                        };

                        intents.retain(|i| i.unit_id != uid);
                        intents.push(UnitIntent { unit_id: uid, action: final_intent });

                        if record {
                            steps.push(RlStep {
                                tick, unit_id: uid,
                                game_state: vec![],
                                action: combat_type, log_prob: result.lp_combat + result.lp_pointer,
                                mask: mask_vec.clone(), step_reward: step_r,
                                entities: Some(gs_v2.entities.clone()),
                                entity_types: Some(gs_v2.entity_types.clone()),
                                threats: Some(gs_v2.threats.clone()),
                                positions: Some(gs_v2.positions.clone()),
                                zones: Some(gs_v2.zones.clone()),
                                action_type: Some(combat_type), target_idx: Some(target_idx),
                                move_dir: None, combat_type: Some(combat_type),
                                lp_move: Some(result.lp_move), lp_combat: Some(result.lp_combat),
                                lp_pointer: Some(result.lp_pointer),
                                aggregate_features: Some(gs_v2.aggregate_features.clone()),
                                target_move_pos: Some(target_pos),
                                teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
                            });
                        }
                    }
                    continue;
                }
            }
        }

        // Self-play: override enemy intents with enemy policy
        if let Some(ep) = enemy_policy {
            if let Policy::ActorCriticV5(ac) = ep {
                for &uid in &enemy_ids {
                    let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                        Some(u) => u,
                        None => continue,
                    };
                    if unit.casting.is_some() || unit.control_remaining_ms > 0 { continue; }
                    let mask_arr = action_mask(&sim, uid);
                    let mask_vec: Vec<bool> = mask_arr.to_vec();
                    let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
                    let mut ability_cls_refs: Vec<Option<&[f32]>> = vec![None; MAX_ABILITIES];
                    for idx in 0..n_abilities {
                        if unit.abilities[idx].cooldown_remaining_ms == 0 && mask_arr[3 + idx] {
                            if let Some(cls) = enemy_cls_cache.get(&(uid, idx)) {
                                ability_cls_refs[idx] = Some(cls.as_slice());
                            }
                        }
                    }
                    let gs_v2 = match (&vis_map, sim.grid_nav.as_ref()) {
                        (Some(vm), Some(nav)) => extract_game_state_v2_spatial(&sim, unit, vm, nav),
                        _ => extract_game_state_v2(&sim, unit),
                    };
                    apply_v5_policy(
                        ac, &sim, unit, uid, &gs_v2, &ability_cls_refs,
                        &mask_arr, &mask_vec, scenario_action_mask,
                        temperature, false, 0.0, tick,
                        &mut rng, &mut intents, &mut steps,
                    );
                }
            }
        }

        t_policy_ns += t0.elapsed().as_nanos() as u64;
        let t0 = std::time::Instant::now();
        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);
        t_step_ns += t0.elapsed().as_nanos() as u64;

        // Dense event-based rewards
        for ev in &events {
            match ev {
                bevy_game::ai::core::SimEvent::UnitDied { unit_id, .. } => {
                    if let Some(dead_unit) = new_sim.units.iter().find(|u| u.id == *unit_id) {
                        if dead_unit.team == Team::Enemy {
                            pending_event_reward += 0.3 / initial_enemy_count.max(1.0);
                        }
                    }
                }
                bevy_game::ai::core::SimEvent::AbilityUsed { unit_id, .. }
                | bevy_game::ai::core::SimEvent::AbilityCastStarted { unit_id, .. } => {
                    if let Some(u) = new_sim.units.iter().find(|u| u.id == *unit_id) {
                        if u.team == Team::Hero {
                            pending_event_reward += 0.05;
                        }
                    }
                }
                bevy_game::ai::core::SimEvent::ControlApplied { target_id, .. } => {
                    if let Some(t) = new_sim.units.iter().find(|u| u.id == *target_id) {
                        if t.team == Team::Enemy {
                            pending_event_reward += 0.1;
                        }
                    }
                }
                bevy_game::ai::core::SimEvent::HealApplied { target_id, amount, .. } => {
                    if *amount > 0 {
                        if let Some(t) = new_sim.units.iter().find(|u| u.id == *target_id) {
                            if t.team == Team::Hero {
                                pending_event_reward += 0.05;
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        sim = new_sim;

        // Update terrain-derived properties (cover_bonus, elevation)
        let t0 = std::time::Instant::now();
        if let Some(ref nav) = sim.grid_nav.clone() {
            use bevy_game::ai::pathing::cover_factor;
            let unit_count = sim.units.len();
            for i in 0..unit_count {
                if sim.units[i].hp <= 0 {
                    sim.units[i].cover_bonus = 0.0;
                    sim.units[i].elevation = 0.0;
                    continue;
                }
                sim.units[i].elevation = nav.elevation_at_pos(sim.units[i].position);
                let pos = sim.units[i].position;
                let team = sim.units[i].team;
                let mut nearest_enemy_pos = None;
                let mut nearest_dist = f32::INFINITY;
                for j in 0..unit_count {
                    if sim.units[j].hp <= 0 || sim.units[j].team == team { continue; }
                    let d = distance(pos, sim.units[j].position);
                    if d < nearest_dist {
                        nearest_dist = d;
                        nearest_enemy_pos = Some(sim.units[j].position);
                    }
                }
                sim.units[i].cover_bonus = match nearest_enemy_pos {
                    Some(ep) => cover_factor(&nav, pos, ep),
                    None => 0.0,
                };
            }
        }

        t_terrain_ns += t0.elapsed().as_nanos() as u64;
        PROF_TICKS.fetch_add(1, Ordering::Relaxed);

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();

        // Check drill objective completion
        if let Some(done) = check_drill_objective(drill_objective, &sim, heroes_alive, enemies_alive) {
            flush_prof!();
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: done.0, reward: done.1,
                ticks: sim.tick, unit_abilities, unit_ability_names, steps,
            };
        }
        if drill_objective.is_none() && enemies_alive == 0 {
            flush_prof!();
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: "Victory".to_string(), reward: 0.5,
                ticks: sim.tick, unit_abilities, unit_ability_names, steps,
            };
        }
        if heroes_alive == 0 {
            flush_prof!();
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: "Defeat".to_string(), reward: 0.0,
                ticks: sim.tick, unit_abilities, unit_ability_names, steps,
            };
        }
    }

    // Timeout
    let (outcome, reward) = if let Some(obj) = drill_objective {
        match obj.objective_type.as_str() {
            "survive" => {
                let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
                if heroes_alive > 0 { ("Victory".to_string(), 1.0) }
                else { ("Defeat".to_string(), -1.0) }
            }
            _ => ("Timeout".to_string(), -0.5),
        }
    } else {
        let hero_hp_pct = sim.units.iter()
            .filter(|u| u.team == Team::Hero)
            .map(|u| u.hp.max(0)).sum::<i32>() as f32
            / sim.units.iter().filter(|u| u.team == Team::Hero)
                .map(|u| u.max_hp).sum::<i32>().max(1) as f32;
        let enemy_hp_pct = sim.units.iter()
            .filter(|u| u.team == Team::Enemy)
            .map(|u| u.hp.max(0)).sum::<i32>() as f32
            / sim.units.iter().filter(|u| u.team == Team::Enemy)
                .map(|u| u.max_hp).sum::<i32>().max(1) as f32;

        if hero_hp_pct > enemy_hp_pct + 0.02 {
            ("Victory".to_string(), 0.5)
        } else if enemy_hp_pct > hero_hp_pct + 0.02 {
            ("Defeat".to_string(), -0.5)
        } else {
            ("Timeout".to_string(), 0.0)
        }
    };

    flush_prof!();
    RlEpisode {
        scenario: scenario_name.to_string(),
        outcome, reward, ticks: sim.tick,
        unit_abilities, unit_ability_names, steps,
    }
}
