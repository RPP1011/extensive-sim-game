use super::*;

#[test]
fn hero_ability_use_ability_intent() {
    let mut attacker = hero_unit(1, Team::Hero, (0.0, 0.0));
    attacker.abilities.push(AbilitySlot::new(AbilityDef {
        name: "Test Strike".into(), targeting: AbilityTargeting::TargetEnemy,
        range: 5.0, cooldown_ms: 3000, cast_time_ms: 0, ai_hint: "damage".into(),
        effects: vec![ConditionalEffect {
            effect: Effect::Damage { amount: 30, amount_per_tick: 0, duration_ms: 0, tick_interval_ms: 0, scaling_stat: None, scaling_percent: 0.0, damage_type: DamageType::Physical, bonus: vec![] },
            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::Refresh, chance: 0.0, else_effects: vec![],
        }],
        delivery: None, resource_cost: 0, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
    }));
    let defender = hero_unit(2, Team::Enemy, (1.0, 0.0));
    let state = make_state(vec![attacker, defender], 42);
    let intents = vec![UnitIntent { unit_id: 1, action: IntentAction::UseAbility { ability_index: 0, target: AbilityTarget::Unit(2) } }];
    let (state, events) = step(state, &intents, FIXED_TICK_MS);
    assert!(events.iter().any(|e| matches!(e, SimEvent::AbilityUsed { .. })));
    assert!(state.units.iter().find(|u| u.id == 2).unwrap().hp < 100);
}

#[test]
fn hero_ability_cooldown_applied() {
    let mut attacker = hero_unit(1, Team::Hero, (0.0, 0.0));
    attacker.abilities.push(AbilitySlot::new(AbilityDef {
        name: "Test Strike".into(), targeting: AbilityTargeting::TargetEnemy,
        range: 5.0, cooldown_ms: 3000, cast_time_ms: 0, ai_hint: "damage".into(),
        effects: vec![ConditionalEffect {
            effect: Effect::Damage { amount: 30, amount_per_tick: 0, duration_ms: 0, tick_interval_ms: 0, scaling_stat: None, scaling_percent: 0.0, damage_type: DamageType::Physical, bonus: vec![] },
            condition: None, area: None, tags: HashMap::new(), stacking: Stacking::Refresh, chance: 0.0, else_effects: vec![],
        }],
        delivery: None, resource_cost: 0, morph_into: None, morph_duration_ms: 0, zone_tag: None, ..Default::default()
    }));
    let defender = hero_unit(2, Team::Enemy, (1.0, 0.0));
    let state = make_state(vec![attacker, defender], 42);
    let intents = vec![UnitIntent { unit_id: 1, action: IntentAction::UseAbility { ability_index: 0, target: AbilityTarget::Unit(2) } }];
    let (state, _) = step(state, &intents, FIXED_TICK_MS);
    assert!(state.units.iter().find(|u| u.id == 1).unwrap().abilities[0].cooldown_remaining_ms > 0);
}

#[test]
fn hero_cooldowns_decrement_each_tick() {
    let mut unit = hero_unit(1, Team::Hero, (0.0, 0.0));
    unit.abilities.push(AbilitySlot {
        def: AbilityDef {
            name: "Test".into(), targeting: AbilityTargeting::TargetEnemy,
            range: 5.0, cooldown_ms: 3000, cast_time_ms: 0, ai_hint: "damage".into(),
            effects: Vec::new(), delivery: None, resource_cost: 0, morph_into: None, morph_duration_ms: 0, zone_tag: None,
            ..Default::default()
        },
        cooldown_remaining_ms: 1000, base_def: None, morph_remaining_ms: 0,
        ..Default::default()
    });
    let state = make_state(vec![unit], 1);
    let (state, _) = step(state, &[], FIXED_TICK_MS);
    assert_eq!(state.units[0].abilities[0].cooldown_remaining_ms, 900);
}

// --- Arcanist zone-reaction tests ---

fn arcanist_zone_ability(name: &str, tag: &str) -> AbilitySlot {
    AbilitySlot::new(AbilityDef {
        name: name.into(),
        targeting: AbilityTargeting::GroundTarget,
        range: 6.0,
        cooldown_ms: 6000,
        cast_time_ms: 0, // instant for test convenience
        ai_hint: "damage".into(),
        effects: vec![ConditionalEffect {
            effect: Effect::Damage {
                amount: 10, amount_per_tick: 0, tick_interval_ms: 0,
                duration_ms: 0, scaling_stat: None, scaling_percent: 0.0,
                damage_type: DamageType::Physical,
                bonus: vec![],
            },
            condition: None,
            area: Some(Area::Circle { radius: 2.5 }),
            tags: HashMap::new(),
            stacking: Stacking::default(), chance: 0.0, else_effects: vec![],
        }],
        delivery: Some(Delivery::Zone { duration_ms: 5000, tick_interval_ms: 500 }),
        resource_cost: 0,
        morph_into: None,
        morph_duration_ms: 0,
        zone_tag: Some(tag.into()),
        ..Default::default()
    })
}

fn arcanist_unit(id: u32, team: Team, pos: (f32, f32), abilities: Vec<AbilitySlot>) -> UnitState {
    let mut u = hero_unit(id, team, pos);
    u.abilities = abilities;
    u
}

#[test]
fn fire_frost_zone_reaction_creates_steam_cloud() {
    let caster = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FireRing", "fire"),
        arcanist_zone_ability("FrostField", "frost"),
    ]);
    let enemy = hero_unit(2, Team::Enemy, (10.0, 0.0));
    let state = make_state(vec![caster, enemy], 42);

    // Cast fire at (3, 0)
    let intent1 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 0,
            target: AbilityTarget::Position(sim_vec2(3.0, 0.0)),
        },
    }];
    let (mut state, events1) = step(state, &intent1, FIXED_TICK_MS);
    assert!(events1.iter().any(|e| matches!(e, SimEvent::ZoneCreated { .. })), "fire zone not created");
    assert_eq!(state.zones.len(), 1);

    // Reset cooldown so we can cast frost immediately
    state.units[0].abilities[1].cooldown_remaining_ms = 0;

    // Cast frost overlapping at (3, 0) — same spot triggers reaction
    let intent2 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 1,
            target: AbilityTarget::Position(sim_vec2(3.0, 0.0)),
        },
    }];
    let (state, events2) = step(state, &intent2, FIXED_TICK_MS);

    // Should have fire + frost + combo zone = 3 zones
    assert_eq!(state.zones.len(), 3, "expected 3 zones (fire + frost + combo), got {}", state.zones.len());
    assert!(events2.iter().any(|e| matches!(e, SimEvent::ZoneReaction { combo_name, .. } if combo_name == "SteamCloud")),
        "SteamCloud reaction event not found");
    // Combo zone should have the combo tag
    let combo = state.zones.iter().find(|z| z.zone_tag.as_deref() == Some("combo_steamcloud"));
    assert!(combo.is_some(), "combo zone with tag 'combo_steamcloud' not found");
}

#[test]
fn fire_lightning_zone_reaction_creates_plasma_storm() {
    let caster = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FireRing", "fire"),
        arcanist_zone_ability("LightningStorm", "lightning"),
    ]);
    let enemy = hero_unit(2, Team::Enemy, (10.0, 0.0));
    let state = make_state(vec![caster, enemy], 7);

    let intent1 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 0,
            target: AbilityTarget::Position(sim_vec2(4.0, 0.0)),
        },
    }];
    let (mut state, _) = step(state, &intent1, FIXED_TICK_MS);
    state.units[0].abilities[1].cooldown_remaining_ms = 0;

    let intent2 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 1,
            target: AbilityTarget::Position(sim_vec2(4.0, 0.0)),
        },
    }];
    let (state, events2) = step(state, &intent2, FIXED_TICK_MS);

    assert_eq!(state.zones.len(), 3);
    assert!(events2.iter().any(|e| matches!(e, SimEvent::ZoneReaction { combo_name, .. } if combo_name == "PlasmaStorm")));
}

#[test]
fn frost_lightning_zone_reaction_creates_shatter() {
    let caster = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FrostField", "frost"),
        arcanist_zone_ability("LightningStorm", "lightning"),
    ]);
    let enemy = hero_unit(2, Team::Enemy, (10.0, 0.0));
    let state = make_state(vec![caster, enemy], 13);

    let intent1 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 0,
            target: AbilityTarget::Position(sim_vec2(5.0, 0.0)),
        },
    }];
    let (mut state, _) = step(state, &intent1, FIXED_TICK_MS);
    state.units[0].abilities[1].cooldown_remaining_ms = 0;

    let intent2 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 1,
            target: AbilityTarget::Position(sim_vec2(5.0, 0.0)),
        },
    }];
    let (state, events2) = step(state, &intent2, FIXED_TICK_MS);

    assert_eq!(state.zones.len(), 3);
    assert!(events2.iter().any(|e| matches!(e, SimEvent::ZoneReaction { combo_name, .. } if combo_name == "Shatter")));
}

#[test]
fn same_element_zones_do_not_react() {
    let caster = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FireRing1", "fire"),
        arcanist_zone_ability("FireRing2", "fire"),
    ]);
    let enemy = hero_unit(2, Team::Enemy, (10.0, 0.0));
    let state = make_state(vec![caster, enemy], 99);

    let intent1 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 0,
            target: AbilityTarget::Position(sim_vec2(3.0, 0.0)),
        },
    }];
    let (mut state, _) = step(state, &intent1, FIXED_TICK_MS);
    state.units[0].abilities[1].cooldown_remaining_ms = 0;

    let intent2 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 1,
            target: AbilityTarget::Position(sim_vec2(3.0, 0.0)),
        },
    }];
    let (state, events2) = step(state, &intent2, FIXED_TICK_MS);

    // Only 2 zones (both fire), no combo
    assert_eq!(state.zones.len(), 2);
    assert!(!events2.iter().any(|e| matches!(e, SimEvent::ZoneReaction { .. })),
        "same-element zones should not react");
}

#[test]
fn distant_zones_do_not_react() {
    let caster = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FireRing", "fire"),
        arcanist_zone_ability("FrostField", "frost"),
    ]);
    let enemy = hero_unit(2, Team::Enemy, (20.0, 0.0));
    let state = make_state(vec![caster, enemy], 55);

    // Place fire at (0, 0)
    let intent1 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 0,
            target: AbilityTarget::Position(sim_vec2(0.0, 0.0)),
        },
    }];
    let (mut state, _) = step(state, &intent1, FIXED_TICK_MS);
    state.units[0].abilities[1].cooldown_remaining_ms = 0;

    // Place frost far away at (15, 0) — well beyond overlap range
    let intent2 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 1,
            target: AbilityTarget::Position(sim_vec2(15.0, 0.0)),
        },
    }];
    let (state, events2) = step(state, &intent2, FIXED_TICK_MS);

    assert_eq!(state.zones.len(), 2, "should only have 2 zones, no combo");
    assert!(!events2.iter().any(|e| matches!(e, SimEvent::ZoneReaction { .. })),
        "distant zones should not react");
}

#[test]
fn different_casters_do_not_react() {
    let caster1 = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FireRing", "fire"),
    ]);
    let caster2 = arcanist_unit(3, Team::Hero, (1.0, 0.0), vec![
        arcanist_zone_ability("FrostField", "frost"),
    ]);
    let enemy = hero_unit(2, Team::Enemy, (10.0, 0.0));
    let state = make_state(vec![caster1, caster2, enemy], 77);

    // Caster 1 places fire
    let intent1 = vec![UnitIntent {
        unit_id: 1,
        action: IntentAction::UseAbility {
            ability_index: 0,
            target: AbilityTarget::Position(sim_vec2(3.0, 0.0)),
        },
    }];
    let (state, _) = step(state, &intent1, FIXED_TICK_MS);

    // Caster 2 places frost on top
    let intent2 = vec![UnitIntent {
        unit_id: 3,
        action: IntentAction::UseAbility {
            ability_index: 0,
            target: AbilityTarget::Position(sim_vec2(3.0, 0.0)),
        },
    }];
    let (state, events2) = step(state, &intent2, FIXED_TICK_MS);

    assert_eq!(state.zones.len(), 2, "should only have 2 zones, no combo from different casters");
    assert!(!events2.iter().any(|e| matches!(e, SimEvent::ZoneReaction { .. })),
        "zones from different casters should not react");
}

#[test]
fn arcanist_combos_zones_in_full_fight() {
    // Run an arcanist vs enemies for enough ticks to see zone reactions fire.
    use crate::squad::{self as squad_ai, SquadAiState};
    use crate::roles::Role;

    let mut caster = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FireRing", "fire"),
        arcanist_zone_ability("FrostField", "frost"),
        arcanist_zone_ability("LightningStorm", "lightning"),
    ]);
    caster.hp = 200;
    caster.max_hp = 200;
    let e1 = hero_unit(10, Team::Enemy, (4.0, 0.0));
    let e2 = hero_unit(11, Team::Enemy, (4.5, 0.5));
    let mut state = make_state(vec![caster, e1, e2], 42);

    let mut role_map = HashMap::new();
    role_map.insert(1, Role::Dps);
    let mut ai = SquadAiState::new_from_roles(&state, role_map);

    let mut all_events = Vec::new();
    for _ in 0..200 {
        let intents = squad_ai::generate_intents(&state, &mut ai, FIXED_TICK_MS);
        let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
        state = new_state;
        all_events.extend(events);
    }

    let zone_count = all_events.iter().filter(|e| matches!(e, SimEvent::ZoneCreated { .. })).count();
    assert!(zone_count >= 2, "expected at least 2 zones created, got {zone_count}");

    let reaction_count = all_events.iter().filter(|e| matches!(e, SimEvent::ZoneReaction { .. })).count();
    assert!(reaction_count >= 1,
        "expected at least 1 zone reaction from arcanist combo, got {reaction_count}. \
         Zones created: {zone_count}");
}

#[test]
fn arcanist_prefers_combo_zone_over_non_combo() {
    // When a fire zone exists, the AI should pick frost/lightning (combo) over
    // casting another fire zone (no combo). We verify by running a few ticks
    // after manually placing a fire zone.
    use crate::squad::{self as squad_ai, SquadAiState};
    use crate::roles::Role;

    let caster = arcanist_unit(1, Team::Hero, (0.0, 0.0), vec![
        arcanist_zone_ability("FireRing", "fire"),        // index 0
        arcanist_zone_ability("FrostField", "frost"),     // index 1 - combo
        arcanist_zone_ability("LightningStorm", "lightning"), // index 2 - combo
    ]);
    let enemy = hero_unit(10, Team::Enemy, (4.0, 0.0));
    let mut state = make_state(vec![caster, enemy], 7);

    // Pre-place a fire zone at the enemy
    state.zones.push(ActiveZone {
        id: 999,
        source_id: 1,
        source_team: Team::Hero,
        position: sim_vec2(4.0, 0.0),
        area: Area::Circle { radius: 2.5 },
        effects: vec![],
        remaining_ms: 4000,
        tick_interval_ms: 500,
        tick_elapsed_ms: 0,
        trigger_on_enter: false,
        invisible: false,
        triggered: false,
        arm_time_ms: 0,
        blocked_cells: Vec::new(),
        zone_tag: Some("fire".into()),
    });

    let mut role_map = HashMap::new();
    role_map.insert(1, Role::Dps);
    let mut ai = SquadAiState::new_from_roles(&state, role_map);

    // Run a few ticks — the first ability used should trigger a zone reaction
    let mut reaction_fired = false;
    for _ in 0..30 {
        let intents = squad_ai::generate_intents(&state, &mut ai, FIXED_TICK_MS);
        let (new_state, events) = step(state, &intents, FIXED_TICK_MS);
        state = new_state;
        if events.iter().any(|e| matches!(e, SimEvent::ZoneReaction { .. })) {
            reaction_fired = true;
            break;
        }
    }
    assert!(reaction_fired,
        "arcanist should have triggered a zone reaction within 30 ticks when fire zone exists");
}

#[test]
fn arcanist_template_parses_and_has_zone_tags() {
    let toml_str = include_str!("../../../../../assets/hero_templates/arcanist.toml");
    let hero: crate::effects::defs::HeroToml = toml::from_str(toml_str).expect("arcanist template should parse");
    assert_eq!(hero.abilities.len(), 6);

    let fire = hero.abilities.iter().find(|a| a.name == "FireRing").unwrap();
    assert_eq!(fire.zone_tag.as_deref(), Some("fire"));

    let frost = hero.abilities.iter().find(|a| a.name == "FrostField").unwrap();
    assert_eq!(frost.zone_tag.as_deref(), Some("frost"));

    let lightning = hero.abilities.iter().find(|a| a.name == "LightningStorm").unwrap();
    assert_eq!(lightning.zone_tag.as_deref(), Some("lightning"));
}
