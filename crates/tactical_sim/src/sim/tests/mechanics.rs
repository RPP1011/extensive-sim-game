use super::*;

#[test]
fn shield_absorbs_damage() {
    let mut attacker = hero_unit(1, Team::Hero, (0.0, 0.0));
    attacker.attack_range = 5.0;
    let mut defender = hero_unit(2, Team::Enemy, (1.0, 0.0));
    defender.shield_hp = 50;
    defender.status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Shield { amount: 50 },
        source_id: 2, remaining_ms: 5000, tags: HashMap::new(), stacking: Stacking::Refresh,
    });
    let mut state = make_state(vec![attacker, defender], 42);
    let intents = vec![UnitIntent { unit_id: 1, action: IntentAction::Attack { target_id: 2 } }];
    for _ in 0..4 { let (s, _) = step(state, &intents, FIXED_TICK_MS); state = s; }
    let defender = state.units.iter().find(|u| u.id == 2).unwrap();
    assert!(defender.hp >= 90, "Shield should absorb some damage, hp={}", defender.hp);
}

#[test]
fn dot_ticks_damage_over_time() {
    let mut unit = hero_unit(1, Team::Hero, (0.0, 0.0));
    unit.status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Dot { amount_per_tick: 10, tick_interval_ms: 100, tick_elapsed_ms: 0 },
        source_id: 99, remaining_ms: 500, tags: HashMap::new(), stacking: Stacking::Refresh,
    });
    let mut state = make_state(vec![unit], 1);
    for _ in 0..5 { let (s, _) = step(state, &[], FIXED_TICK_MS); state = s; }
    assert!(state.units[0].hp <= 50, "DoT should deal damage, hp={}", state.units[0].hp);
    assert!(state.units[0].hp > 0);
    assert!(state.units[0].status_effects.is_empty());
}

#[test]
fn hot_heals_over_time() {
    let mut unit = hero_unit(1, Team::Hero, (0.0, 0.0));
    unit.hp = 60;
    unit.status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Hot { amount_per_tick: 10, tick_interval_ms: 100, tick_elapsed_ms: 0 },
        source_id: 99, remaining_ms: 300, tags: HashMap::new(), stacking: Stacking::Refresh,
    });
    let mut state = make_state(vec![unit], 1);
    for _ in 0..3 { let (s, _) = step(state, &[], FIXED_TICK_MS); state = s; }
    assert!(state.units[0].hp >= 80, "HoT should heal, hp={}", state.units[0].hp);
}

#[test]
fn reflect_status_reflects_damage_to_attacker() {
    // Unit 1 (defender) has Reflect(50%), unit 2 (attacker) attacks unit 1
    let mut defender = hero_unit(1, Team::Hero, (0.0, 0.0));
    defender.status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Reflect { percent: 50.0 },
        source_id: 1, remaining_ms: 5000, tags: HashMap::new(), stacking: Stacking::Refresh,
    });
    let mut attacker = hero_unit(2, Team::Enemy, (1.0, 0.0));
    attacker.attack_range = 5.0;
    attacker.attack_damage = 20;
    attacker.attack_cast_time_ms = 0;
    attacker.attack_cooldown_ms = 100;
    let state = make_state(vec![defender, attacker], 42);
    let intents = vec![UnitIntent { unit_id: 2, action: IntentAction::Attack { target_id: 1 } }];

    // Run enough ticks for an attack to land
    let mut state = state;
    let mut found_reflect = false;
    for _ in 0..10 {
        let (s, events) = step(state, &intents, FIXED_TICK_MS);
        state = s;
        if events.iter().any(|e| matches!(e, SimEvent::ReflectDamage { .. })) {
            found_reflect = true;
        }
    }
    assert!(found_reflect, "Reflect status should cause ReflectDamage events");
    let attacker_hp = state.units.iter().find(|u| u.id == 2).unwrap().hp;
    assert!(attacker_hp < 100, "Attacker should take reflect damage, hp={}", attacker_hp);
}

#[test]
fn passive_triggers_on_damage_taken() {
    let mut unit = hero_unit(1, Team::Hero, (0.0, 0.0));
    unit.passives.push(PassiveSlot::new(PassiveDef {
        name: "Iron Skin".into(), trigger: Trigger::OnDamageTaken, cooldown_ms: 5000,
        effects: vec![ConditionalEffect {
            effect: Effect::Shield { amount: 20, duration_ms: 3000 },
            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::Refresh, chance: 0.0, else_effects: vec![], targeting_filter: None,
        }],
        range: 0.0,
    }));
    let mut enemy = hero_unit(2, Team::Enemy, (1.0, 0.0));
    enemy.attack_range = 5.0;
    let mut state = make_state(vec![unit, enemy], 42);
    let intents = vec![UnitIntent { unit_id: 2, action: IntentAction::Attack { target_id: 1 } }];
    for _ in 0..5 { let (s, _) = step(state, &intents, FIXED_TICK_MS); state = s; }
    assert_eq!(state.units.iter().find(|u| u.id == 1).unwrap().passives.len(), 1);
}
