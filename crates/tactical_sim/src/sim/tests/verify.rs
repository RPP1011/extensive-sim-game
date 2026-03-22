use super::*;
use crate::sim::verify::*;
use crate::sim::simulation::sample_duel_state;

fn make_ability_def() -> AbilityDef {
    AbilityDef::default()
}

#[test]
fn clean_state_passes_verification() {
    let state = sample_duel_state(42);
    let report = verify_tick(&state);
    assert!(report.is_ok(), "Expected no violations, got: {:?}", report.violations);
}

#[test]
fn detects_hp_exceeds_max() {
    let mut state = sample_duel_state(42);
    state.units[0].hp = 150; // max_hp is 100
    let report = verify_tick(&state);
    assert!(!report.is_ok());
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::HpExceedsMax { unit_id: 1, hp: 150, max_hp: 100 }
    )));
}

#[test]
fn detects_dead_unit_casting() {
    let mut state = sample_duel_state(42);
    state.units[0].hp = 0;
    state.units[0].casting = Some(CastState {
        target_id: 2,
        target_pos: None,
        remaining_ms: 100,
        kind: CastKind::Attack,
        area: None,
        ability_index: None,
        effect_hint: CastEffectHint::Unknown,
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::DeadUnitCasting { unit_id: 1 }
    )));
}

#[test]
fn detects_negative_shield() {
    let mut state = sample_duel_state(42);
    state.units[0].shield_hp = -5;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::NegativeShield { unit_id: 1, shield_hp: -5 }
    )));
}

#[test]
fn detects_invalid_position() {
    let mut state = sample_duel_state(42);
    state.units[0].position = sim_vec2(f32::NAN, 0.0);
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::InvalidPosition { unit_id: 1, .. }
    )));
}

#[test]
fn detects_duplicate_unit_ids() {
    let mut state = sample_duel_state(42);
    state.units[1].id = 1; // same as units[0]
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::DuplicateUnitId { unit_id: 1 }
    )));
}

#[test]
fn detects_resource_exceeds_max() {
    let mut state = sample_duel_state(42);
    state.units[0].max_resource = 100;
    state.units[0].resource = 120;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::ResourceExceedsMax { unit_id: 1, resource: 120, max_resource: 100 }
    )));
}

#[test]
fn detects_negative_resource() {
    let mut state = sample_duel_state(42);
    state.units[0].resource = -10;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::NegativeResource { unit_id: 1, resource: -10 }
    )));
}

#[test]
fn full_simulation_passes_verification() {
    use crate::sim::simulation::{step, sample_duel_script};
    let mut state = sample_duel_state(42);
    let script = sample_duel_script(200);
    for tick in 0..200 {
        let intents = &script[tick];
        let (new_state, _events) = step(state, intents, FIXED_TICK_MS);
        state = new_state;
        let report = verify_tick(&state);
        assert!(
            report.is_ok(),
            "Violation at tick {}: {:?}",
            report.tick,
            report.violations
        );
    }
}

#[test]
fn detects_cooldown_exceeds_base() {
    let mut state = sample_duel_state(42);
    state.units[0].cooldown_remaining_ms = 5000; // base is 700
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::CooldownExceedsBase { unit_id: 1, field: "attack", .. }
    )));
}

// --- New runtime verification tests ---

#[test]
fn detects_casting_and_channeling() {
    let mut state = sample_duel_state(42);
    state.units[0].casting = Some(CastState {
        target_id: 2,
        target_pos: None,
        remaining_ms: 100,
        kind: CastKind::Attack,
        area: None,
        ability_index: None,
        effect_hint: CastEffectHint::Unknown,
    });
    state.units[0].channeling = Some(ChannelState {
        ability_index: 0,
        target_id: 2,
        target_pos: None,
        remaining_ms: 500,
        tick_interval_ms: 100,
        tick_elapsed_ms: 0,
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::CastingAndChanneling { unit_id: 1 }
    )));
}

#[test]
fn detects_dead_unit_controlled() {
    let mut state = sample_duel_state(42);
    state.units[0].hp = 0;
    state.units[0].control_remaining_ms = 500;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::DeadUnitControlled { unit_id: 1, control_remaining_ms: 500 }
    )));
}

#[test]
fn detects_ability_charges_exceed_max() {
    let mut state = sample_duel_state(42);
    let mut def = make_ability_def();
    def.max_charges = 3;
    let mut slot = AbilitySlot::new(def);
    slot.charges = 5;
    state.units[0].abilities.push(slot);
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::AbilityChargesExceedMax { unit_id: 1, ability_index: 0, charges: 5, max_charges: 3 }
    )));
}

#[test]
fn detects_morph_without_base_def() {
    let mut state = sample_duel_state(42);
    let def = make_ability_def();
    let mut slot = AbilitySlot::new(def);
    slot.morph_remaining_ms = 1000;
    // base_def is None by default
    state.units[0].abilities.push(slot);
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::MorphWithoutBaseDef { unit_id: 1, ability_index: 0 }
    )));
}

#[test]
fn detects_recast_window_without_recasts() {
    let mut state = sample_duel_state(42);
    let def = make_ability_def();
    let mut slot = AbilitySlot::new(def);
    slot.recast_window_remaining_ms = 500;
    slot.recasts_remaining = 0;
    state.units[0].abilities.push(slot);
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::RecastWindowWithoutRecasts { unit_id: 1, ability_index: 0 }
    )));
}

#[test]
fn detects_directed_without_owner() {
    let mut state = sample_duel_state(42);
    state.units[0].directed = true;
    state.units[0].owner_id = None;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::DirectedWithoutOwner { unit_id: 1 }
    )));
}

#[test]
fn detects_status_effect_orphaned_link() {
    let mut state = sample_duel_state(42);
    state.units[0].status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Link { partner_id: 999, share_percent: 0.5 },
        source_id: 2,
        remaining_ms: 1000,
        tags: Default::default(),
        stacking: Stacking::default(),
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::StatusEffectOrphanedRef { unit_id: 1, kind: "Link", referenced_id: 999 }
    )));
}

#[test]
fn detects_status_effect_dead_redirect_protector() {
    let mut state = sample_duel_state(42);
    state.units[1].hp = 0; // kill protector
    state.units[0].status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Redirect { protector_id: 2, charges: 3 },
        source_id: 2,
        remaining_ms: 1000,
        tags: Default::default(),
        stacking: Stacking::default(),
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::StatusEffectDeadRef { unit_id: 1, kind: "Redirect", referenced_id: 2 }
    )));
}

#[test]
fn detects_cover_bonus_out_of_range() {
    let mut state = sample_duel_state(42);
    state.units[0].cover_bonus = 1.5;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::CoverBonusOutOfRange { unit_id: 1, .. }
    )));
}

#[test]
fn valid_ability_slots_pass() {
    let mut state = sample_duel_state(42);
    let mut def = make_ability_def();
    def.max_charges = 3;
    let mut slot = AbilitySlot::new(def);
    slot.charges = 2;
    state.units[0].abilities.push(slot);
    let report = verify_tick(&state);
    assert!(report.is_ok(), "Expected no violations, got: {:?}", report.violations);
}

#[test]
fn detects_projectile_orphaned_source() {
    use crate::effects::Projectile;
    let mut state = sample_duel_state(42);
    state.projectiles.push(Projectile {
        source_id: 999,
        target_id: 2,
        position: sim_vec2(1.0, 1.0),
        direction: sim_vec2(1.0, 0.0),
        speed: 10.0,
        pierce: false,
        width: 0.5,
        on_hit: Vec::new(),
        on_arrival: Vec::new(),
        already_hit: Vec::new(),
        target_position: sim_vec2(5.0, 0.0),
        max_travel_distance: 0.0,
        distance_traveled: 0.0,
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::ProjectileOrphanedSource { projectile_index: 0, source_id: 999 }
    )));
}

#[test]
fn detects_projectile_invalid_position() {
    use crate::effects::Projectile;
    let mut state = sample_duel_state(42);
    state.projectiles.push(Projectile {
        source_id: 1,
        target_id: 2,
        position: sim_vec2(f32::NAN, 0.0),
        direction: sim_vec2(1.0, 0.0),
        speed: 10.0,
        pierce: false,
        width: 0.5,
        on_hit: Vec::new(),
        on_arrival: Vec::new(),
        already_hit: Vec::new(),
        target_position: sim_vec2(5.0, 0.0),
        max_travel_distance: 0.0,
        distance_traveled: 0.0,
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::ProjectileInvalidPosition { projectile_index: 0, .. }
    )));
}

#[test]
fn detects_projectile_invalid_speed() {
    use crate::effects::Projectile;
    let mut state = sample_duel_state(42);
    state.projectiles.push(Projectile {
        source_id: 1,
        target_id: 2,
        position: sim_vec2(1.0, 1.0),
        direction: sim_vec2(1.0, 0.0),
        speed: 0.0,
        pierce: false,
        width: 0.5,
        on_hit: Vec::new(),
        on_arrival: Vec::new(),
        already_hit: Vec::new(),
        target_position: sim_vec2(5.0, 0.0),
        max_travel_distance: 0.0,
        distance_traveled: 0.0,
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::ProjectileInvalidSpeed { projectile_index: 0, .. }
    )));
}

#[test]
fn detects_slow_factor_out_of_range() {
    let mut state = sample_duel_state(42);
    state.units[0].status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Slow { factor: 1.5 },
        source_id: 2,
        remaining_ms: 1000,
        tags: Default::default(),
        stacking: Stacking::default(),
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::StatusEffectValueOutOfRange { unit_id: 1, kind: "Slow", .. }
    )));
}

#[test]
fn detects_stacks_exceed_max() {
    let mut state = sample_duel_state(42);
    state.units[0].status_effects.push(ActiveStatusEffect {
        kind: StatusKind::Stacks { name: "Bleed".into(), count: 10, max_stacks: 5 },
        source_id: 2,
        remaining_ms: 1000,
        tags: Default::default(),
        stacking: Stacking::default(),
    });
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::StatusEffectStacksExceedMax { unit_id: 1, count: 10, max_stacks: 5 }
    )));
}

#[test]
fn detects_summon_owner_missing() {
    let mut state = sample_duel_state(42);
    state.units[0].owner_id = Some(999);
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::SummonOwnerMissing { unit_id: 1, owner_id: 999 }
    )));
}

#[test]
fn detects_negative_armor() {
    let mut state = sample_duel_state(42);
    state.units[0].armor = -10.0;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::NegativeResist { unit_id: 1, field: "armor", .. }
    )));
}

#[test]
fn detects_negative_magic_resist() {
    let mut state = sample_duel_state(42);
    state.units[0].magic_resist = -5.0;
    let report = verify_tick(&state);
    assert!(report.violations.iter().any(|v| matches!(v,
        Violation::NegativeResist { unit_id: 1, field: "magic_resist", .. }
    )));
}
