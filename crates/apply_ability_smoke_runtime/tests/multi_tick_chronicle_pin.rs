//! Wave 1.5+ behavioral pin: the GPU dispatcher's chronicle arms for
//! DamageOverTime / HealOverTime / TimedShield write the expected
//! (kind, actor, target, amount, duration_ticks) tuple for a single
//! self-cast.
//!
//! The parity sweep test (`parity_apply_program_sweep.rs`) covers the
//! CPU↔GPU byte-equality matrix across all 21 abilities × 5 ticks. This
//! pin is a smaller, targeted assertion: build a registry with one
//! ability per multi-tick op (`Burn=DamageOverTime`, `Regen=HealOverTime`,
//! `Aegis=TimedShield`), pre-seed three agents, dispatch at tick=10,
//! and confirm the resulting event_ring carries kinds 51..53 with the
//! correct payload layout:
//!   - amount at payload word 2 (= ring slot offset 4) as bitcast f32→u32.
//!   - duration_ticks at payload word 3 (= ring slot offset 5) as raw u32.
//!
//! Distinct from the parity sweep (which only proves CPU and GPU agree),
//! this pin asserts the absolute amount + duration values round-trip
//! correctly through the bitcast<f32> + raw-u32 pipeline — a regression
//! that drops the duration write or swaps the slot offset would fire
//! here. Distinct from the legacy DoT/HoT shapes which used
//! `expires_at_tick = tick + duration`, the multi-tick chronicle stores
//! the raw `duration_ticks`; future consumer rules will compute
//! per-tick re-emission from the (cast_tick, duration) pair.

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

/// Sanity-pin for the three Wave 1.5+ multi-tick EffectOps.
///
/// We register three abilities (Burn=DamageOverTime, Regen=HealOverTime,
/// Aegis=TimedShield) into a single registry and assign one ability to
/// each of three agents. After one dispatch tick, the event_ring
/// should hold exactly three chronicle records, one per agent — kinds
/// 51..53 in the order each agent's level routes to its registered
/// AbilityId. The shared 5-payload-word shape gets verified per-record:
///   - slot 0 = kind tag (51 / 52 / 53).
///   - slot 1 = tick.
///   - slot 2 = caster_slot (actor).
///   - slot 3 = target_slot (implicit-target self-cast: == caster).
///   - slot 4 = bitcast<u32>(amount).
///   - slot 5 = duration_ticks (raw u32).
///   - slots 6..9 = zero.
#[test]
fn multi_tick_chronicle_records_carry_correct_kind_amount_and_duration() {
    const BURN_AMOUNT: f32 = 6.0;
    const BURN_DURATION: u32 = 30;
    const REGEN_AMOUNT: f32 = 4.0;
    const REGEN_DURATION: u32 = 50;
    const AEGIS_AMOUNT: f32 = 25.0;
    const AEGIS_DURATION: u32 = 100;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 3;

    // Build a registry with the three multi-tick abilities. Each
    // single-target program emits one
    // Effect{DamageOverTime,HealOverTime,TimedShield} record per cast.
    let mut builder = AbilityRegistryBuilder::new();
    let burn_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::DamageOverTime { amount: BURN_AMOUNT, duration_ticks: BURN_DURATION }],
    ));
    let regen_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::HealOverTime { amount: REGEN_AMOUNT, duration_ticks: REGEN_DURATION }],
    ));
    let aegis_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::TimedShield { amount: AEGIS_AMOUNT, duration_ticks: AEGIS_DURATION }],
    ));
    let registry = builder.build();

    // Per-agent levels = AbilityId.raw() to dispatch — agent 0 fires
    // Burn (DoT), agent 1 fires Regen (HoT), agent 2 fires Aegis
    // (TimedShield). The dispatcher reads agent_level[caster_slot] to
    // pick the AbilityId.
    let per_agent_levels = vec![burn_id.raw(), regen_id.raw(), aegis_id.raw()];
    let per_agent_stats  = vec![PerAgentStats::default(); N_AGENTS as usize];

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &per_agent_levels,
        &per_agent_stats,
    ) {
        Some(s) => s,
        None => {
            eprintln!(
                "[multi_tick_chronicle_pin] skipping: no wgpu adapter available",
            );
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

    // Sort by kind tag — atomicAdd ring slot ordering is workgroup-
    // schedule-dependent. After sorting, sorted[0..3] should be
    // (kind=51), (kind=52), (kind=53) in ascending order.
    let mut sorted: Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> = records.clone();
    sorted.sort_by_key(|r| r[0]);

    // Per-record expectations: (kind, actor_slot, amount, duration).
    // Agent 0 → Burn (kind 51), agent 1 → Regen (kind 52), agent 2 →
    // Aegis (kind 53). The implicit-target rule writes
    // caster_slot == target_slot for every record.
    let expected: [(u32, u32, f32, u32); 3] = [
        (51, 0, BURN_AMOUNT,  BURN_DURATION),  // DamageOverTime @ agent 0
        (52, 1, REGEN_AMOUNT, REGEN_DURATION), // HealOverTime   @ agent 1
        (53, 2, AEGIS_AMOUNT, AEGIS_DURATION), // TimedShield    @ agent 2
    ];
    for (i, &(expected_kind, expected_slot, expected_amount, expected_duration)) in
        expected.iter().enumerate()
    {
        let r = &sorted[i];
        assert_eq!(
            r[0], expected_kind,
            "record {i}: kind tag mismatch (got {}, expected {})",
            r[0], expected_kind,
        );
        assert_eq!(r[1], TICK, "record {i}: tick");
        assert_eq!(
            r[2], expected_slot,
            "record {i}: actor slot — caster_slot (got {}, expected {})",
            r[2], expected_slot,
        );
        assert_eq!(
            r[3], expected_slot,
            "record {i}: target slot — implicit-target rule writes \
             caster_slot == target_slot (got {}, expected {})",
            r[3], expected_slot,
        );
        assert_eq!(
            r[4], expected_amount.to_bits(),
            "record {i}: amount at payload word 2 (got 0x{:08x}, expected 0x{:08x})",
            r[4], expected_amount.to_bits(),
        );
        assert_eq!(
            r[5], expected_duration,
            "record {i}: duration_ticks at payload word 3 — raw u32 \
             (got {}, expected {})",
            r[5], expected_duration,
        );
        // Tail words zeroed (slots 6..9).
        for j in 6..CHRONICLE_STRIDE_U32 as usize {
            assert_eq!(r[j], 0, "record {i}: tail word {j} must be zero");
        }
    }
}
