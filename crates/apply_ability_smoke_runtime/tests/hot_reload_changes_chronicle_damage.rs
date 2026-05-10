//! Plan I-step-3 — GPU end-to-end hot-reload pin.
//!
//! The earlier hot-reload tests proved the CPU swap mechanic works
//! and that `PackedAbilityRegistry::pack` reflects the swap in its
//! byte payload. This test closes the chain on real GPU: dispatch a
//! damage ability, observe the chronicle record, hot-reload the
//! program with a different damage amount, dispatch again, observe
//! the new amount in the chronicle.
//!
//! Why apply_ability_smoke_runtime: it's the smallest fixture that
//! actually drives the full registry → pack → upload → GPU dispatch
//! → chronicle-emit chain. The chronicle records carry the damage
//! amount in payload_a (`bitcast<u32>(amount)`), so a host-side
//! readback gives byte-equal evidence the new program reached the
//! GPU.

use apply_ability_smoke_runtime::ApplyAbilitySmokeState;
use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

fn damage_program(amount: f32) -> AbilityProgram {
    AbilityProgram::new_single_target(
        /*range*/ 5.0,
        Gate { cooldown_ticks: 0, hostile_only: false, line_of_sight: false },
        [EffectOp::Damage { amount }],
    )
}

#[test]
fn hot_reload_changes_emitted_chronicle_damage_amount() {
    const DAMAGE_KIND: u32 = 26; // EventKindId::EffectDamageApplied

    // --- v1: build with damage 30.
    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(damage_program(30.0));
    assert_eq!(id, AbilityId::new(1).unwrap());
    let registry_v1 = builder.build();

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        /*n_agents*/ 1,
        &registry_v1,
        /*per_agent_levels*/ &[id.raw()],
        /*per_agent_stats*/ &[Default::default()],
    ) {
        Some(s) => s,
        None => {
            eprintln!(
                "[hot_reload_chronicle_pin] skipping: no wgpu adapter on host. \
                 The CPU-side hot-reload chain is still validated by the \
                 dsl_compiler tests."
            );
            return;
        }
    };

    // --- Dispatch v1; expect chronicle damage = 30.
    state.step(0);
    let tail_v1 = state.read_event_tail();
    assert_eq!(
        tail_v1, 1,
        "v1 dispatch must emit exactly one chronicle record; got {tail_v1}"
    );
    let records_v1 = state.read_event_ring(tail_v1);
    let r1 = records_v1[0];
    assert_eq!(r1[0], DAMAGE_KIND, "v1 record[0] kind must be EffectDamageApplied=26");
    assert_eq!(
        r1[4],
        30.0_f32.to_bits(),
        "v1 record[0] payload_a must be bitcast<u32>(30.0); got 0x{:08X}",
        r1[4],
    );

    // --- Hot-reload to v2 (damage 50). New CPU registry via the
    // immutable swap primitive; runtime re-packs + re-uploads.
    let registry_v2 = registry_v1
        .with_program_replaced(id, damage_program(50.0))
        .expect("known id");
    state.hot_reload_registry(&registry_v2);
    state.reset_event_tail();

    // --- Dispatch v2; expect chronicle damage = 50.
    state.step(1);
    let tail_v2 = state.read_event_tail();
    assert_eq!(
        tail_v2, 1,
        "v2 dispatch must emit exactly one chronicle record; got {tail_v2}"
    );
    let records_v2 = state.read_event_ring(tail_v2);
    let r2 = records_v2[0];
    assert_eq!(r2[0], DAMAGE_KIND, "v2 record[0] kind still EffectDamageApplied=26");
    assert_eq!(
        r2[4],
        50.0_f32.to_bits(),
        "v2 record[0] payload_a must reflect the hot-reloaded value bitcast<u32>(50.0); got 0x{:08X}",
        r2[4],
    );

    // --- The behavioural delta IS the proof: same agent, same
    // dispatch path, same AbilityId, but the chronicle damage went
    // from 30 to 50 because the registry was reloaded mid-run.
    assert_ne!(
        r1[4], r2[4],
        "hot reload must produce a chronicle-level behavioural change on GPU"
    );
}

/// Live `.ability` source → live GPU registry, in one runtime call.
/// The caller-friendly bridge over the parse → lower → swap →
/// re-pack → re-upload chain. Models how a file-watch loop would
/// invoke hot-reload: read the changed source, hand it off, dispatch.
#[test]
fn hot_reload_from_ability_source_string_changes_chronicle_damage() {
    const DAMAGE_KIND: u32 = 26;

    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(damage_program(7.0));
    let registry_v1 = builder.build();

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        1,
        &registry_v1,
        &[id.raw()],
        &[Default::default()],
    ) {
        Some(s) => s,
        None => {
            eprintln!("[hot_reload_source_pin] skipping: no wgpu adapter on host.");
            return;
        }
    };

    state.step(0);
    let r_initial = state.read_event_ring(state.read_event_tail())[0];
    assert_eq!(
        r_initial[4],
        7.0_f32.to_bits(),
        "v1 chronicle damage = 7.0; got 0x{:08X}",
        r_initial[4],
    );

    // Hand the runtime an .ability source verbatim — the same shape
    // an editor would save to disk on a hot-reload trigger.
    let updated_source = "ability HotReloadFromSource {\n    \
                          target: enemy\n    \
                          range: 5.0\n    \
                          damage 99\n}\n";
    let _registry_v2 = state
        .reload_from_ability_source(&registry_v1, id, updated_source)
        .expect("source must parse + lower + swap cleanly");
    state.reset_event_tail();

    state.step(1);
    let r_after = state.read_event_ring(state.read_event_tail())[0];
    assert_eq!(r_after[0], DAMAGE_KIND, "post-reload kind still EffectDamageApplied");
    assert_eq!(
        r_after[4],
        99.0_f32.to_bits(),
        "post-reload chronicle damage must be 99.0 from the source string; got 0x{:08X}",
        r_after[4],
    );
}

/// Negative pin: malformed source surfaces an error and leaves GPU
/// state untouched. Catches a future regression where a bad source
/// string corrupts the live registry.
#[test]
fn reload_from_invalid_source_keeps_gpu_state_intact() {
    let mut builder = AbilityRegistryBuilder::new();
    let id = builder.register(damage_program(13.0));
    let registry_v1 = builder.build();

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        1,
        &registry_v1,
        &[id.raw()],
        &[Default::default()],
    ) {
        Some(s) => s,
        None => {
            eprintln!("[hot_reload_invalid_pin] skipping: no wgpu adapter on host.");
            return;
        }
    };

    let bad_source = "this is not a valid .ability file at all";
    let result = state.reload_from_ability_source(&registry_v1, id, bad_source);
    assert!(
        result.is_err(),
        "malformed source must return Err, not silently overwrite GPU state"
    );

    // Confirm GPU state is unchanged: dispatch still emits damage 13.
    state.step(0);
    let tail = state.read_event_tail();
    let r = state.read_event_ring(tail)[0];
    assert_eq!(
        r[4],
        13.0_f32.to_bits(),
        "after failed reload, original damage 13.0 must still be live; got 0x{:08X}",
        r[4],
    );
}
