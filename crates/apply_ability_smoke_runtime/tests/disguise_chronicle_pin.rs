//! Wave 3 ToM Phase 4 behavioral pin: the GPU dispatcher's chronicle
//! arm for `EffectOp::Disguise` (kind 36 → 67) writes the expected
//! (kind, actor, target, packed_payload_a) tuple for a single self-cast.
//!
//! ## Shape
//!
//! `EffectOp::Disguise { fake_type: u8, duration_ticks: u32 }` — caster
//! poses as `fake_type` for `duration_ticks`. The dispatcher writes a
//! chronicle record (kind=67 EffectDisguiseApplied):
//!
//!   - slot 0 = 67 (kind tag)
//!   - slot 1 = tick
//!   - slot 2 = caster_slot
//!   - slot 3 = target_slot       (= caster for self-cast — implicit-target rule)
//!   - slot 4 = (duration_ticks << 8) | fake_type
//!                               (consumer splits via `& 0xFFu` and `>> 8u`)
//!   - slot 5 = 0
//!   - slots 6..9 = zero
//!
//! The actual write into the per-agent `disguise_expires_at_tick` /
//! `disguise_fake_type` SoA columns lives in a downstream consumer rule
//! (`tom_probe.sim`); this pin only verifies the chronicle record shape.

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

#[test]
fn disguise_chronicle_record_carries_correct_kind_and_payloads() {
    const FAKE_TYPE: u8 = 7;
    const DURATION_TICKS: u32 = 200;
    const PACKED_PAYLOAD_A: u32 = (DURATION_TICKS << 8) | (FAKE_TYPE as u32);
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 1;

    let mut builder = AbilityRegistryBuilder::new();
    let disguise_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Disguise {
            fake_type: FAKE_TYPE,
            duration_ticks: DURATION_TICKS,
        }],
    ));
    let registry = builder.build();

    let per_agent_levels = vec![disguise_id.raw()];
    let per_agent_stats = vec![PerAgentStats::default(); N_AGENTS as usize];

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &per_agent_levels,
        &per_agent_stats,
    ) {
        Some(s) => s,
        None => {
            eprintln!("[disguise_chronicle_pin] skipping: no wgpu adapter available");
            return;
        }
    };

    state.step(TICK);
    let tail = state.read_event_tail();
    let records = state.read_event_ring(tail);

    assert_eq!(
        records.len(),
        N_AGENTS as usize,
        "expected one chronicle record per agent (got {})",
        records.len(),
    );

    let r = &records[0];
    assert_eq!(r[0], 67, "Disguise: kind tag — EffectDisguiseApplied");
    assert_eq!(r[1], TICK, "Disguise: tick");
    assert_eq!(r[2], 0, "Disguise: actor slot — caster_slot for agent 0");
    assert_eq!(
        r[3], 0,
        "Disguise: target slot — implicit-target rule routes target = caster",
    );
    assert_eq!(
        r[4], PACKED_PAYLOAD_A,
        "Disguise: packed payload_a (= duration<<8 | fake_type) at slot 4 — \
         consumer splits via `payload_a & 0xFFu` (fake_type) and \
         `payload_a >> 8u` (duration)",
    );
    assert_eq!(r[5], 0, "Disguise: slot 5 must be zero (payload_b unused)");
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Disguise: tail word {i} must be zero");
    }
}
