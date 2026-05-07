//! Extended-status behavioral pin: the GPU dispatcher's chronicle arms
//! for Stealth/Charm/Grounded/Suppress write the expected (kind, actor,
//! target, duration_ticks) tuple for a single self-cast.
//!
//! The parity sweep test (`parity_apply_program_sweep.rs`) covers the
//! CPU↔GPU byte-equality matrix across all 25 abilities × 5 ticks. This
//! pin is a smaller, targeted assertion: build a registry with one
//! ability per extended-status op (`Vanish=Stealth`, `Allure=Charm`,
//! `Tether=Grounded`, `Hush=Suppress`), pre-seed four agents, dispatch
//! at tick=10, and confirm the resulting event_ring carries kinds 54..57
//! with the correct per-shape duration offsets:
//!   - Stealth: duration_ticks at payload word 1 (= ring slot offset 3),
//!     no target slot in the engine event (caster-self status).
//!   - Charm/Grounded/Suppress: duration_ticks at payload word 2 (=
//!     ring slot offset 4), target at slot 3 (target-cast statuses).
//!
//! Distinct from the parity sweep (which only proves CPU and GPU agree),
//! this pin asserts the absolute duration value round-trips correctly
//! through the raw-u32 pipeline — a regression that drops the duration
//! write or swaps the slot offset would fire here. Distinct from the
//! legacy Stun/Root/Silence/Fear/Taunt shapes which fold the deadline
//! (`expires_at_tick = tick + duration_ticks`), the extended-status
//! arms store the raw `duration_ticks` so a future consumer rule can
//! compute its own per-tick re-emission window — same convention as
//! the multi-tick effect family (DoT/HoT/TimedShield, kinds 51..53).

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

/// Sanity-pin for the four extended-corpus status EffectOps.
///
/// We register four abilities (Vanish=Stealth, Allure=Charm, Tether=
/// Grounded, Hush=Suppress) into a single registry and assign one
/// ability to each of four agents. After one dispatch tick, the
/// event_ring should hold exactly four chronicle records, one per agent
/// — kinds 54..57 in the order each agent's level routes to its
/// registered AbilityId. The per-shape duration offsets get verified
/// per-record:
///   - Kind 54 (Stealth): duration lands at slot 3 (no target).
///   - Kinds 55/56/57 (Charm/Grounded/Suppress): duration lands at
///     slot 4, target at slot 3.
#[test]
fn extended_status_chronicle_records_carry_correct_kind_and_duration() {
    const VANISH_DURATION: u32 = 50;
    const ALLURE_DURATION: u32 = 30;
    const TETHER_DURATION: u32 = 25;
    const HUSH_DURATION: u32 = 40;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 4;

    // Build a registry with the four extended-status abilities. Each
    // single-target program emits one
    // Effect{Stealth,Charm,Grounded,Suppress} record per cast.
    let mut builder = AbilityRegistryBuilder::new();
    let vanish_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Stealth { duration_ticks: VANISH_DURATION }],
    ));
    let allure_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Charm { duration_ticks: ALLURE_DURATION }],
    ));
    let tether_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Grounded { duration_ticks: TETHER_DURATION }],
    ));
    let hush_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Suppress { duration_ticks: HUSH_DURATION }],
    ));
    let registry = builder.build();

    // Per-agent levels = AbilityId.raw() to dispatch — agent 0 fires
    // Vanish (Stealth), agent 1 fires Allure (Charm), agent 2 fires
    // Tether (Grounded), agent 3 fires Hush (Suppress). The dispatcher
    // reads agent_level[caster_slot] to pick the AbilityId.
    let per_agent_levels = vec![
        vanish_id.raw(),
        allure_id.raw(),
        tether_id.raw(),
        hush_id.raw(),
    ];
    let per_agent_stats = vec![PerAgentStats::default(); N_AGENTS as usize];

    let mut state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &per_agent_levels,
        &per_agent_stats,
    ) {
        Some(s) => s,
        None => {
            eprintln!(
                "[extended_status_chronicle_pin] skipping: no wgpu adapter available",
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
    // schedule-dependent. After sorting, sorted[0..4] should be
    // (kind=54), (kind=55), (kind=56), (kind=57) in ascending order.
    let mut sorted: Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> = records.clone();
    sorted.sort_by_key(|r| r[0]);

    // Per-record expectations: (kind, actor_slot, duration, is_target_cast).
    // The agent slot that fired each ability is determined by
    // `per_agent_levels` order: agent 0 → Vanish (kind 54), agent 1 →
    // Allure (kind 55), agent 2 → Tether (kind 56), agent 3 → Hush
    // (kind 57).
    let expected: [(u32, u32, u32, bool); 4] = [
        (54, 0, VANISH_DURATION, false), // Stealth @ agent 0 — caster-self
        (55, 1, ALLURE_DURATION, true),  // Charm    @ agent 1 — target-cast
        (56, 2, TETHER_DURATION, true),  // Grounded @ agent 2 — target-cast
        (57, 3, HUSH_DURATION,   true),  // Suppress @ agent 3 — target-cast
    ];
    for (i, &(expected_kind, expected_slot, expected_duration, is_target_cast)) in
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

        if is_target_cast {
            // Charm/Grounded/Suppress: target at slot 3 (caster==target
            // for implicit-target self-cast), duration_ticks at slot 4.
            assert_eq!(
                r[3], expected_slot,
                "record {i} (target-cast): target slot — implicit-target \
                 rule writes caster_slot == target_slot (got {}, expected {})",
                r[3], expected_slot,
            );
            assert_eq!(
                r[4], expected_duration,
                "record {i} (target-cast): duration_ticks at payload word 2 \
                 (got {}, expected {})",
                r[4], expected_duration,
            );
            // Tail words zeroed.
            for j in 5..CHRONICLE_STRIDE_U32 as usize {
                assert_eq!(r[j], 0, "record {i} (target-cast): tail word {j} must be zero");
            }
        } else {
            // Stealth: duration_ticks at slot 3 (no target field).
            assert_eq!(
                r[3], expected_duration,
                "record {i} (caster-self): duration_ticks at payload word 1 \
                 (got {}, expected {})",
                r[3], expected_duration,
            );
            // Tail words zeroed (slots 4..9).
            for j in 4..CHRONICLE_STRIDE_U32 as usize {
                assert_eq!(r[j], 0, "record {i} (caster-self): tail word {j} must be zero");
            }
        }
    }
}
