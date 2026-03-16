#[cfg(test)]
mod tests_stress {
    use std::collections::HashMap;
    use std::collections::VecDeque;
    use crate::ai::core::*;
    use crate::ai::core::determinism::hash_sim_state;
    use crate::ai::effects::{
        AbilityDef, AbilityTargeting, ConditionalEffect, DamageType, Effect,
        AbilitySlot, PassiveSlot, StatusKind, Stacking,
        PassiveDef, Area, Delivery, Trigger,
    };
    use crate::ai::squad::{self as squad_ai, SquadAiState};
    use crate::ai::roles::Role;

    fn hero_unit(id: u32, team: Team, pos: (f32, f32)) -> UnitState {
        UnitState {
            id, team, hp: 100, max_hp: 100,
            position: sim_vec2(pos.0, pos.1),
            move_speed_per_sec: 3.0, attack_damage: 10,
            attack_range: 1.4, attack_cooldown_ms: 700, attack_cast_time_ms: 300,
            cooldown_remaining_ms: 0, ability_damage: 0, ability_range: 0.0,
            ability_cooldown_ms: 0, ability_cast_time_ms: 0, ability_cooldown_remaining_ms: 0,
            heal_amount: 0, heal_range: 0.0, heal_cooldown_ms: 0, heal_cast_time_ms: 0,
            heal_cooldown_remaining_ms: 0, control_range: 0.0, control_duration_ms: 0,
            control_cooldown_ms: 0, control_cast_time_ms: 0, control_cooldown_remaining_ms: 0,
            control_remaining_ms: 0, casting: None, abilities: Vec::new(), passives: Vec::new(),
            status_effects: Vec::new(), shield_hp: 0, resistance_tags: HashMap::new(),
            state_history: VecDeque::new(), channeling: None, resource: 0, max_resource: 0,
            resource_regen_per_sec: 0.0, owner_id: None, directed: false, armor: 0.0, magic_resist: 0.0, cover_bonus: 0.0, elevation: 0.0, total_healing_done: 0, total_damage_done: 0,
        }
    }

    fn stress_unit(id: u32, team: Team, pos: (f32, f32)) -> UnitState {
        let mut u = hero_unit(id, team, pos);
        u.max_hp = 200; u.hp = 200; u.attack_damage = 8;
        u.resource = 100; u.max_resource = 100; u.resource_regen_per_sec = 10.0;

        match id % 4 {
            0 => {
                u.abilities.push(AbilitySlot::new(AbilityDef {
                    name: "PoisonCloud".into(), targeting: AbilityTargeting::GroundTarget,
                    range: 4.0, cooldown_ms: 4000, cast_time_ms: 200, ai_hint: "damage".into(),
                    effects: vec![ConditionalEffect {
                        effect: Effect::ApplyStacks { name: "Venom".into(), count: 1, max_stacks: 4, duration_ms: 5000 },
                        condition: None, area: Some(Area::Circle { radius: 2.0 }),
                        tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                    }],
                    delivery: Some(Delivery::Zone { duration_ms: 3000, tick_interval_ms: 500 }),
                    resource_cost: 20, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
                }));
                u.passives.push(PassiveSlot::new(PassiveDef {
                    name: "VenomBurst".into(),
                    trigger: Trigger::OnStackReached { name: "Venom".into(), count: 3 },
                    cooldown_ms: 0,
                    effects: vec![ConditionalEffect {
                        effect: Effect::Damage { amount: 30, amount_per_tick: 0, tick_interval_ms: 0, duration_ms: 0, scaling_stat: None, scaling_percent: 0.0, damage_type: DamageType::Physical, bonus: vec![] },
                        condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                    }],
                    range: 0.0,
                }));
            }
            1 => {
                u.heal_amount = 1; u.heal_range = 5.0; u.heal_cooldown_ms = 99999;
                u.abilities.push(AbilitySlot::new(AbilityDef {
                    name: "Tranquility".into(), targeting: AbilityTargeting::TargetAlly,
                    range: 5.0, cooldown_ms: 6000, cast_time_ms: 0, ai_hint: "heal".into(),
                    effects: vec![ConditionalEffect {
                        effect: Effect::Heal { amount: 15, amount_per_tick: 0, tick_interval_ms: 0, duration_ms: 0, scaling_stat: None, scaling_percent: 0.0, bonus: vec![] },
                        condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                    }],
                    delivery: Some(Delivery::Channel { duration_ms: 2000, tick_interval_ms: 500 }),
                    resource_cost: 30, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
                }));
                u.abilities.push(AbilitySlot::new(AbilityDef {
                    name: "Bash".into(), targeting: AbilityTargeting::TargetEnemy,
                    range: 1.5, cooldown_ms: 5000, cast_time_ms: 200, ai_hint: "crowd_control".into(),
                    effects: vec![ConditionalEffect {
                        effect: Effect::Stun { duration_ms: 800 },
                        condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                    }],
                    delivery: None, resource_cost: 0, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
                }));
            }
            2 => {
                u.abilities.push(AbilitySlot::new(AbilityDef {
                    name: "PierceShot".into(), targeting: AbilityTargeting::Direction,
                    range: 6.0, cooldown_ms: 3000, cast_time_ms: 100, ai_hint: "damage".into(),
                    effects: vec![],
                    delivery: Some(Delivery::Projectile {
                        speed: 10.0, pierce: true, width: 0.4,
                        on_hit: vec![ConditionalEffect {
                            effect: Effect::Damage { amount: 25, amount_per_tick: 0, tick_interval_ms: 0, duration_ms: 0, scaling_stat: None, scaling_percent: 0.0, damage_type: DamageType::Physical, bonus: vec![] },
                            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                        }],
                        on_arrival: vec![],
                    }),
                    resource_cost: 15, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
                }));
                u.abilities.push(AbilitySlot::new(AbilityDef {
                    name: "SoulLeash".into(), targeting: AbilityTargeting::TargetEnemy,
                    range: 4.0, cooldown_ms: 8000, cast_time_ms: 200, ai_hint: "crowd_control".into(),
                    effects: vec![ConditionalEffect {
                        effect: Effect::Slow { factor: 0.4, duration_ms: 1500 },
                        condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                    }],
                    delivery: Some(Delivery::Tether {
                        max_range: 5.0, tick_interval_ms: 500,
                        on_complete: vec![ConditionalEffect {
                            effect: Effect::Stun { duration_ms: 1000 },
                            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                        }],
                    }),
                    resource_cost: 25, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
                }));
            }
            3 | _ => {
                u.abilities.push(AbilitySlot::new(AbilityDef {
                    name: "Strike".into(), targeting: AbilityTargeting::TargetEnemy,
                    range: 1.5, cooldown_ms: 2000, cast_time_ms: 100, ai_hint: "damage".into(),
                    effects: vec![
                        ConditionalEffect {
                            effect: Effect::Damage { amount: 20, amount_per_tick: 0, tick_interval_ms: 0, duration_ms: 0, scaling_stat: None, scaling_percent: 0.0, damage_type: DamageType::Physical, bonus: vec![] },
                            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                        },
                        ConditionalEffect {
                            effect: Effect::Dash { to_target: true, distance: 2.0, to_position: false, is_blink: false },
                            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                        },
                    ],
                    delivery: None, resource_cost: 10,
                    morph_into: Some(Box::new(AbilityDef {
                        name: "PowerStrike".into(), targeting: AbilityTargeting::TargetEnemy,
                        range: 1.5, cooldown_ms: 1000, cast_time_ms: 100, ai_hint: "damage".into(),
                        effects: vec![ConditionalEffect {
                            effect: Effect::Damage { amount: 40, amount_per_tick: 0, tick_interval_ms: 0, duration_ms: 0, scaling_stat: None, scaling_percent: 0.0, damage_type: DamageType::Physical, bonus: vec![] },
                            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                        }],
                        delivery: None, resource_cost: 10, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
                    })),
                    morph_duration_ms: 3000, zone_tag: None, ..Default::default()
                }));
                u.abilities.push(AbilitySlot::new(AbilityDef {
                    name: "BearTrap".into(), targeting: AbilityTargeting::GroundTarget,
                    range: 3.0, cooldown_ms: 10000, cast_time_ms: 200, ai_hint: "crowd_control".into(),
                    effects: vec![
                        ConditionalEffect { effect: Effect::Stun { duration_ms: 1200 }, condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![] },
                        ConditionalEffect {
                            effect: Effect::Damage { amount: 35, amount_per_tick: 0, tick_interval_ms: 0, duration_ms: 0, scaling_stat: None, scaling_percent: 0.0, damage_type: DamageType::Physical, bonus: vec![] },
                            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
                        },
                    ],
                    delivery: Some(Delivery::Trap { duration_ms: 15000, trigger_radius: 1.5, arm_time_ms: 500 }),
                    resource_cost: 20, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
                }));
            }
        }
        u
    }

    fn build_stress_state(seed: u64) -> SimState {
        let heroes = [
            stress_unit(100, Team::Hero, (-3.0, -1.5)),
            stress_unit(101, Team::Hero, (-4.0, 0.0)),
            stress_unit(102, Team::Hero, (-3.0, 1.5)),
            stress_unit(103, Team::Hero, (-2.0, 0.0)),
        ];
        let enemies = [
            stress_unit(200, Team::Enemy, (3.0, -1.5)),
            stress_unit(201, Team::Enemy, (4.0, 0.0)),
            stress_unit(202, Team::Enemy, (3.0, 1.5)),
            stress_unit(203, Team::Enemy, (2.0, 0.0)),
        ];
        let mut units: Vec<UnitState> = heroes.into_iter().chain(enemies).collect();
        units.sort_by_key(|u| u.id);
        SimState { tick: 0, rng_state: seed, units: UnitStore::new(units), projectiles: Vec::new(), passive_trigger_depth: 0, zones: Vec::new(), tethers: Vec::new(), grid_nav: None }
    }

    fn run_stress_fight(seed: u64, ticks: u32) -> (SimState, Vec<SimEvent>) {
        let initial = build_stress_state(seed);
        let mut role_map = HashMap::new();
        for u in &initial.units {
            let role = match u.id % 4 { 0 => Role::Dps, 1 => Role::Healer, 2 => Role::Dps, _ => Role::Tank };
            role_map.insert(u.id, role);
        }
        let mut ai = SquadAiState::new_from_roles(&initial, role_map);
        let mut state = initial;
        let mut all_events = Vec::new();
        for _ in 0..ticks {
            let intents = squad_ai::generate_intents(&state, &mut ai, FIXED_TICK_MS);
            let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
            state = new_state; all_events.extend(events);
        }
        (state, all_events)
    }

    #[test]
    fn stress_test_all_mechanics_deterministic() {
        for seed in 0..10 {
            let (a, events_a) = run_stress_fight(seed, 250);
            let (b, events_b) = run_stress_fight(seed, 250);
            assert_eq!(hash_sim_state(&a), hash_sim_state(&b), "seed {seed}: determinism violated");
            assert_eq!(events_a.len(), events_b.len(), "seed {seed}: event count mismatch");
        }
    }

    #[test]
    fn stress_test_invariants_hold() {
        for seed in 0..10 {
            let (state, events) = run_stress_fight(seed, 250);
            for u in &state.units {
                if u.hp > 0 { assert!(u.hp <= u.max_hp, "seed {seed}: unit {} hp {} > max_hp {}", u.id, u.hp, u.max_hp); }
            }
            for u in &state.units {
                if u.max_resource > 0 {
                    assert!(u.resource >= 0, "seed {seed}: unit {} resource {} < 0", u.id, u.resource);
                    assert!(u.resource <= u.max_resource, "seed {seed}: unit {} resource {} > max {}", u.id, u.resource, u.max_resource);
                }
            }
            for u in &state.units {
                if u.hp <= 0 { assert!(u.channeling.is_none(), "seed {seed}: dead unit {} still channeling", u.id); }
            }
            let dead_ids: Vec<u32> = state.units.iter().filter(|u| u.hp <= 0).map(|u| u.id).collect();
            for tether in &state.tethers {
                assert!(!dead_ids.contains(&tether.source_id) && !dead_ids.contains(&tether.target_id),
                    "seed {seed}: tether between dead units ({} -> {})", tether.source_id, tether.target_id);
            }
            for u in &state.units {
                for se in &u.status_effects {
                    if let StatusKind::Stacks { ref name, count, max_stacks } = se.kind {
                        assert!(count <= max_stacks, "seed {seed}: unit {} stacks '{}' count {} > max {}", u.id, name, count, max_stacks);
                    }
                }
            }
            assert!(events.len() > 100, "seed {seed}: only {} events", events.len());
        }
    }

    #[test]
    fn stress_test_mechanics_exercised() {
        let seeds = [42u64, 7, 13, 99, 123, 200, 314, 999];
        let mut zone_seen = false;
        let mut channel_seen = false;
        let mut tether_seen = false;
        let mut stacks_seen = false;
        let mut projectile_seen = false;
        let mut morph_seen = false;
        let mut resource_spent_seen = false;

        for &seed in &seeds {
            let (st, ev) = run_stress_fight(seed, 250);
            zone_seen |= ev.iter().any(|e| matches!(e, SimEvent::ZoneCreated { .. }));
            channel_seen |= ev.iter().any(|e| matches!(e, SimEvent::ChannelStarted { .. }));
            tether_seen |= ev.iter().any(|e| matches!(e, SimEvent::TetherFormed { .. }));
            stacks_seen |= ev.iter().any(|e| matches!(e, SimEvent::StacksApplied { .. }));
            projectile_seen |= ev.iter().any(|e| matches!(e, SimEvent::ProjectileSpawned { .. }));
            morph_seen |= ev.iter().any(|e| matches!(e, SimEvent::AbilityUsed { ability_name, .. } if ability_name == "PowerStrike"));
            resource_spent_seen |= st.units.iter().any(|u| u.max_resource > 0 && u.resource < u.max_resource);
            if zone_seen && channel_seen && tether_seen && stacks_seen && projectile_seen && morph_seen && resource_spent_seen { break; }
        }

        assert!(zone_seen, "No zones created across {} seeds", seeds.len());
        assert!(channel_seen, "No channels started across {} seeds", seeds.len());
        assert!(tether_seen, "No tethers formed across {} seeds", seeds.len());
        assert!(stacks_seen, "No stacks applied across {} seeds", seeds.len());
        assert!(projectile_seen, "No projectiles launched across {} seeds", seeds.len());
        assert!(morph_seen, "No morphed abilities used across {} seeds", seeds.len());
        assert!(resource_spent_seen, "No resources spent across {} seeds", seeds.len());
    }

    /// Benchmark: O(1) idx_of vs O(N) linear scan at various unit counts,
    /// plus sim step throughput.
    #[test]
    fn bench_unit_store_lookup_and_sim_throughput() {
        use std::time::Instant;
        use std::hint::black_box;

        eprintln!();
        eprintln!("=== UnitStore Benchmark ===");
        eprintln!();

        // --- Lookup microbenchmark at various unit counts ---
        for &n_units in &[8usize, 16, 32, 64] {
            let units: Vec<UnitState> = (0..n_units as u32)
                .map(|i| stress_unit(100 + i, if i < n_units as u32 / 2 { Team::Hero } else { Team::Enemy }, (i as f32, 0.0)))
                .collect();
            let store = UnitStore::new(units);
            let ids: Vec<u32> = store.iter().map(|u| u.id).collect();

            let n_lookups = 2_000_000;

            // O(1) HashMap
            let t0 = Instant::now();
            for i in 0..n_lookups {
                black_box(store.idx_of(ids[i % ids.len()]));
            }
            let idx_of_ns = t0.elapsed().as_nanos() as f64 / n_lookups as f64;

            // O(N) linear scan
            let t1 = Instant::now();
            for i in 0..n_lookups {
                let id = ids[i % ids.len()];
                black_box(store.iter().position(|u| u.id == id));
            }
            let linear_ns = t1.elapsed().as_nanos() as f64 / n_lookups as f64;

            let speedup = linear_ns / idx_of_ns;
            eprintln!("  {n_units:>2} units: idx_of={idx_of_ns:.1}ns  linear={linear_ns:.1}ns  {speedup:.1}x");
        }

        // --- Sim step throughput (8-unit stress fights) ---
        let n_fights = 100u64;
        let ticks_per = 200u32;
        let t2 = Instant::now();
        for seed in 0..n_fights {
            let _ = run_stress_fight(seed, ticks_per);
        }
        let total_steps = n_fights * ticks_per as u64;
        let us_per_step = t2.elapsed().as_micros() as f64 / total_steps as f64;

        // --- Sim step throughput with squad AI (matches IMPALA make_sim path) ---
        let n_fights = 200u64;
        let ticks_per = 300u32;
        let t2 = Instant::now();
        for seed in 0..n_fights {
            let _ = run_stress_fight(seed, ticks_per);
        }
        let total_steps = n_fights * ticks_per as u64;
        let us_per_step = t2.elapsed().as_micros() as f64 / total_steps as f64;
        let ms_total = t2.elapsed().as_millis();

        eprintln!();
        eprintln!("  sim step (8 units, squad AI):");
        eprintln!("    {total_steps} steps in {ms_total}ms ({us_per_step:.1} µs/step)");
        eprintln!("    ({n_fights} fights × {ticks_per} ticks)");

        // --- Extrapolate to IMPALA scale ---
        // IMPALA: 474 episodes × ~175 ticks avg = ~83K steps, with ~96 find_unit_idx per step
        let impala_steps = 83_000u64;
        let projected_ms = impala_steps as f64 * us_per_step / 1000.0;
        eprintln!();
        eprintln!("  IMPALA projected (83K steps): {projected_ms:.0}ms sim time");
        eprintln!("  (baseline make_sim was ~10.5s for 474 episodes)");
        eprintln!("===========================");
    }
}
