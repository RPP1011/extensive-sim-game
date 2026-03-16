//! Episode runner for transformer RL: runs a single scenario with a policy.

use super::transformer_rl::{
    Policy, RlEpisode, RlStep,
    lcg_f32, masked_softmax_sample, load_behavior_trees,
    apply_behavior_overrides,
    MAX_ABILITIES,
};
use super::rl_policies::{
    apply_random_policy, apply_v3_policy, apply_gpu_policy, apply_v4_policy,
    apply_v5_policy, check_drill_objective, hp_fraction,
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
    student_weights: &Option<std::sync::Arc<super::training::StudentWeights>>,
    grid_nav: Option<bevy_game::ai::pathing::GridNav>,
    embedding_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    enemy_policy: Option<&Policy>,
    enemy_registry: Option<&bevy_game::ai::core::ability_transformer::EmbeddingRegistry>,
    drill_objective: Option<&bevy_game::scenario::ObjectiveDef>,
    scenario_action_mask: Option<&str>,
    behaviors: &std::collections::HashMap<u32, bevy_game::ai::behavior::BehaviorTree>,
) -> RlEpisode {
    use bevy_game::ai::core::{is_alive, step, distance, move_towards, position_at_range, Team, UnitIntent, FIXED_TICK_MS};
    use bevy_game::ai::core::ability_eval::{extract_game_state, extract_game_state_v2, extract_game_state_v2_spatial};
    use bevy_game::ai::core::self_play::actions::{action_mask, action_to_intent, intent_to_action};
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
    let initial_hero_count = sim.units.iter()
        .filter(|u| u.team == Team::Hero && u.hp > 0).count() as f32;
    let mut pending_event_reward: f32 = 0.0;

    for tick in 0..max_ticks {
        // Pre-set hero focus target before generate_intents for better coordination
        if matches!(policy, Policy::Combined) {
            let tt = super::training::compute_team_target(&sim);
            squad_ai.set_focus_target(Team::Hero, tt);
        }
        let mut intents = generate_intents(&sim, &mut squad_ai, FIXED_TICK_MS);
        apply_behavior_overrides(&mut intents, behaviors, &sim, tick);
        let record = tick % step_interval == 0;

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

        // Combined policy path: coordinated tactical AI
        if matches!(policy, Policy::Combined) {
            let team_target = super::training::compute_team_target(&sim);

            // Set the hero team's focus target to our coordinated target
            // This ensures the squad AI focuses all heroes on the same enemy.
            squad_ai.set_focus_target(Team::Hero, team_target);

            for &uid in &hero_ids {
                if !sim.units.iter().any(|u| u.id == uid && is_alive(u)) { continue; }
                if let Some(u) = sim.units.iter().find(|u| u.id == uid) {
                    if u.casting.is_some() || u.control_remaining_ms > 0 { continue; }
                }

                // Tactical override: coordinated targeting + abilities + kill-securing
                if let Some(action) = super::training::tactical_hero_override(&sim, uid, team_target) {
                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action });
                }
            }

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
                    let gs_v2 = match (&vis_map, sim.grid_nav.as_ref()) {
                        (Some(vm), Some(nav)) => extract_game_state_v2_spatial(&sim, unit, vm, nav),
                        _ => extract_game_state_v2(&sim, unit),
                    };
                    let game_state = extract_game_state(&sim, unit);
                    let token_infos = build_token_infos(
                        &sim, uid, &gs_v2.entity_types, &gs_v2.positions,
                    );
                    let (v3_action_type, v3_target_idx) = intent_to_v3_action(
                        &intent_action, uid, &sim, &token_infos,
                    ).unwrap_or((2, 0));
                    let (v4_move_dir, v4_combat_type, _v4_target_idx) = intent_to_v4_action(
                        &intent_action, uid, &sim, &token_infos,
                    ).unwrap_or((8, 1, 0));
                    steps.push(RlStep {
                        tick, unit_id: uid,
                        game_state: game_state.to_vec(),
                        action, log_prob: 0.0,
                        mask: mask_arr.to_vec(),
                        step_reward: step_r,
                        entities: Some(gs_v2.entities),
                        entity_types: Some(gs_v2.entity_types),
                        threats: Some(gs_v2.threats),
                        positions: Some(gs_v2.positions),
                        action_type: Some(v3_action_type),
                        target_idx: Some(v3_target_idx),
                        move_dir: Some(v4_move_dir),
                        combat_type: Some(v4_combat_type),
                        lp_move: None, lp_combat: None,
                        lp_pointer: None,
                        aggregate_features: if gs_v2.aggregate_features.is_empty() { None } else { Some(gs_v2.aggregate_features) },
                        teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
                    });
                }
            }
        } else {
            // Transformer/AC policies: override hero intents with policy output
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

                let gs_v2 = match (&vis_map, sim.grid_nav.as_ref()) {
                    (Some(vm), Some(nav)) => extract_game_state_v2_spatial(&sim, unit, vm, nav),
                    _ => extract_game_state_v2(&sim, unit),
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

                // V3 pointer policy
                if let Policy::ActorCriticV3(ac) = policy {
                    apply_v3_policy(
                        ac, &sim, unit, uid, &gs_v2, &ability_cls_refs,
                        &mask_arr, &mask_vec, temperature, record, step_r, tick,
                        &mut rng, &mut intents, &mut steps,
                    );
                    continue;
                }

                // GPU server policy
                if let Policy::GpuServer(gpu) = policy {
                    apply_gpu_policy(
                        gpu, &sim, unit, uid, &gs_v2, &ability_cls_refs,
                        &mask_arr, &mask_vec, scenario_action_mask,
                        record, step_r, tick,
                        &mut intents, &mut steps,
                    );
                    continue;
                }

                // V5 dual-head policy (d=128, aggregate token, 34-dim entities, 10-dim threats)
                if let Policy::ActorCriticV5(ac) = policy {
                    apply_v5_policy(
                        ac, &sim, unit, uid, &gs_v2, &ability_cls_refs,
                        &mask_arr, &mask_vec, scenario_action_mask,
                        temperature, record, step_r, tick,
                        &mut rng, &mut intents, &mut steps,
                    );
                    continue;
                }

                // V4 dual-head policy
                if let Policy::ActorCriticV4(ac) = policy {
                    apply_v4_policy(
                        ac, &sim, unit, uid, &gs_v2, &ability_cls_refs,
                        &mask_arr, &mask_vec, scenario_action_mask,
                        temperature, record, step_r, tick,
                        &mut rng, &mut intents, &mut steps,
                    );
                    continue;
                }

                // V1/V2/Legacy flat action space
                let logits: Vec<f32> = match policy {
                    Policy::ActorCritic(ac) => {
                        let game_state = extract_game_state(&sim, unit);
                        let ent_state = ac.encode_entities(&game_state);
                        ac.action_logits(&ent_state, &ability_cls_refs).to_vec()
                    }
                    Policy::ActorCriticV2(ac) => {
                        let ent_refs: Vec<&[f32]> = gs_v2.entities.iter().map(|e| e.as_slice()).collect();
                        let type_refs: Vec<usize> = gs_v2.entity_types.iter().map(|&t| t as usize).collect();
                        let threat_refs: Vec<&[f32]> = gs_v2.threats.iter().map(|t| t.as_slice()).collect();
                        let ent_state = ac.encode_entities_v2(&ent_refs, &type_refs, &threat_refs);
                        ac.action_logits(&ent_state, &ability_cls_refs).to_vec()
                    }
                    Policy::Legacy(tw) => {
                        let game_state = extract_game_state(&sim, unit);
                        let mut logits = vec![0.0f32; super::transformer_rl::NUM_ACTIONS];
                        if let Some(entities) = tw.encode_entities(&game_state) {
                            for idx in 0..n_abilities {
                                if let Some(cls) = cls_cache.get(&(uid, idx)) {
                                    let output = tw.predict_from_cls(cls, Some(&entities));
                                    let u = output.urgency.clamp(0.001, 0.999);
                                    logits[3 + idx] = (u / (1.0 - u)).ln();
                                }
                            }
                        }
                        logits
                    }
                    Policy::ActorCriticV3(_) | Policy::ActorCriticV4(_) | Policy::ActorCriticV5(_) | Policy::GpuServer(_) | Policy::Combined | Policy::Random => unreachable!(),
                };

                let (action, log_prob) = masked_softmax_sample(
                    &logits, &mask_arr, temperature, &mut rng,
                );
                let intent_action = action_to_intent(action, uid, &sim);
                intents.retain(|i| i.unit_id != uid);
                intents.push(UnitIntent { unit_id: uid, action: intent_action });

                if record {
                    let game_state = extract_game_state(&sim, unit);
                    steps.push(RlStep {
                        tick, unit_id: uid,
                        game_state: game_state.to_vec(),
                        action, log_prob,
                        mask: mask_vec,
                        step_reward: step_r,
                        entities: Some(gs_v2.entities.clone()),
                        entity_types: Some(gs_v2.entity_types.clone()),
                        threats: Some(gs_v2.threats.clone()),
                        positions: None, action_type: None, target_idx: None,
                        move_dir: None, combat_type: None,
                        lp_move: None, lp_combat: None,
                        lp_pointer: None, aggregate_features: None,
                        teacher_move_dir: None, teacher_combat_type: None, teacher_target_idx: None,
                    });
                }
            }
        }

        // Self-play: override enemy intents with enemy policy
        if let Some(ep) = enemy_policy {
            if let Policy::ActorCritic(ac) = ep {
                for &uid in &enemy_ids {
                    let unit = match sim.units.iter().find(|u| u.id == uid && is_alive(u)) {
                        Some(u) => u,
                        None => continue,
                    };
                    if unit.casting.is_some() || unit.control_remaining_ms > 0 { continue; }
                    let mask_arr = action_mask(&sim, uid);
                    let n_abilities = unit.abilities.len().min(MAX_ABILITIES);
                    let mut ability_cls_refs: Vec<Option<&[f32]>> = vec![None; MAX_ABILITIES];
                    for idx in 0..n_abilities {
                        if unit.abilities[idx].cooldown_remaining_ms == 0 && mask_arr[3 + idx] {
                            if let Some(cls) = enemy_cls_cache.get(&(uid, idx)) {
                                ability_cls_refs[idx] = Some(cls.as_slice());
                            }
                        }
                    }
                    let game_state = extract_game_state(&sim, unit);
                    let ent_state = ac.encode_entities(&game_state);
                    let raw = ac.action_logits(&ent_state, &ability_cls_refs);
                    let logits: Vec<f32> = raw.to_vec();
                    let (action, _log_prob) = masked_softmax_sample(
                        &logits, &mask_arr, temperature, &mut rng,
                    );
                    let intent_action = action_to_intent(action, uid, &sim);
                    intents.retain(|i| i.unit_id != uid);
                    intents.push(UnitIntent { unit_id: uid, action: intent_action });
                }
            }
        }

        let (new_sim, events) = step(sim, &intents, FIXED_TICK_MS);

        // Dense event-based rewards: reward GOOD PLAY, not winning
        for ev in &events {
            match ev {
                bevy_game::ai::core::SimEvent::UnitDied { unit_id, .. } => {
                    if let Some(dead_unit) = new_sim.units.iter().find(|u| u.id == *unit_id) {
                        if dead_unit.team == Team::Enemy {
                            // Secured a kill — good play
                            pending_event_reward += 0.3 / initial_enemy_count.max(1.0);
                        }
                        // Don't penalize hero deaths — asymmetric comps make losses inevitable
                    }
                }
                bevy_game::ai::core::SimEvent::AbilityUsed { unit_id, .. }
                | bevy_game::ai::core::SimEvent::AbilityCastStarted { unit_id, .. } => {
                    // Reward ability usage (heroes using their kit)
                    if let Some(u) = new_sim.units.iter().find(|u| u.id == *unit_id) {
                        if u.team == Team::Hero {
                            pending_event_reward += 0.05;
                        }
                    }
                }
                bevy_game::ai::core::SimEvent::ControlApplied { target_id, .. } => {
                    // Reward applying CC to enemies
                    if let Some(t) = new_sim.units.iter().find(|u| u.id == *target_id) {
                        if t.team == Team::Enemy {
                            pending_event_reward += 0.1;
                        }
                    }
                }
                bevy_game::ai::core::SimEvent::HealApplied { target_id, amount, .. } => {
                    // Reward effective healing (not overhealing)
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

        let heroes_alive = sim.units.iter().filter(|u| u.team == Team::Hero && u.hp > 0).count();
        let enemies_alive = sim.units.iter().filter(|u| u.team == Team::Enemy && u.hp > 0).count();

        // Check drill objective completion
        if let Some(done) = check_drill_objective(drill_objective, &sim, heroes_alive, enemies_alive) {
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: done.0, reward: done.1,
                ticks: sim.tick, unit_abilities, unit_ability_names, steps,
            };
        }
        if drill_objective.is_none() && enemies_alive == 0 {
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: "Victory".to_string(), reward: 0.5,  // mild bonus, not +1
                ticks: sim.tick, unit_abilities, unit_ability_names, steps,
            };
        }
        if heroes_alive == 0 {
            // No penalty for losing — asymmetric comps make losses inevitable.
            // Dense rewards already captured the quality of play.
            return RlEpisode {
                scenario: scenario_name.to_string(),
                outcome: "Defeat".to_string(), reward: 0.0,
                ticks: sim.tick, unit_abilities, unit_ability_names, steps,
            };
        }
    }

    // Timeout: if the hero team has more HP remaining, count as a win.
    // This prevents stalemates from penalizing the hero team when it was
    // winning on HP differential. In many combat games, the team with
    // more HP remaining when time expires is considered the winner.
    let (outcome, reward) = if let Some(obj) = drill_objective {
        match obj.objective_type.as_str() {
            "survive" => {
                let heroes_alive = sim.units.iter().filter(|u| u.team == bevy_game::ai::core::Team::Hero && u.hp > 0).count();
                if heroes_alive > 0 { ("Victory".to_string(), 1.0) }
                else { ("Defeat".to_string(), -1.0) }
            }
            _ => ("Timeout".to_string(), -0.5),
        }
    } else {
        let hero_hp_total: i32 = sim.units.iter()
            .filter(|u| u.team == bevy_game::ai::core::Team::Hero)
            .map(|u| u.hp.max(0)).sum();
        let enemy_hp_total: i32 = sim.units.iter()
            .filter(|u| u.team == bevy_game::ai::core::Team::Enemy)
            .map(|u| u.hp.max(0)).sum();
        let hero_hp_pct = hero_hp_total as f32 / sim.units.iter()
            .filter(|u| u.team == bevy_game::ai::core::Team::Hero)
            .map(|u| u.max_hp).sum::<i32>().max(1) as f32;
        let enemy_hp_pct = enemy_hp_total as f32 / sim.units.iter()
            .filter(|u| u.team == bevy_game::ai::core::Team::Enemy)
            .map(|u| u.max_hp).sum::<i32>().max(1) as f32;

        // Hero team wins the timeout if they have more HP remaining.
        // Use a small threshold (2%) to avoid calling truly even fights.
        if hero_hp_pct > enemy_hp_pct + 0.02 {
            ("Victory".to_string(), 0.5)
        } else if enemy_hp_pct > hero_hp_pct + 0.02 {
            ("Defeat".to_string(), -0.5)
        } else {
            ("Timeout".to_string(), 0.0)
        }
    };

    RlEpisode {
        scenario: scenario_name.to_string(),
        outcome, reward, ticks: sim.tick,
        unit_abilities, unit_ability_names, steps,
    }
}

