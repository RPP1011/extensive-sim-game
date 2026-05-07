//! Wave 2 piece 2 behavioral pin: the GPU dispatcher's chronicle arms
//! for Dash/Blink/Knockback/Pull write the expected (kind, actor,
//! target, distance) tuple for a single self-cast.
//!
//! The parity sweep test (`parity_apply_program_sweep.rs`) covers the
//! CPU↔GPU byte-equality matrix across all 18 abilities × 5 ticks. This
//! pin is a smaller, targeted assertion: build a registry with one
//! ability per movement op (`Lunge=Dash`, `Phase=Blink`, `Smash=Knockback`,
//! `Hook=Pull`), pre-seed four agents, dispatch at tick=10, and confirm
//! the resulting event_ring carries kinds 47..50 with the correct
//! per-shape distance offsets:
//!   - Dash/Blink: distance at payload word 1 (= ring slot offset 3),
//!     no target slot in the engine event.
//!   - Knockback/Pull: distance at payload word 2 (= ring slot offset
//!     4), target at slot 3.
//!
//! Distinct from the parity sweep (which only proves CPU and GPU agree),
//! this pin asserts the absolute distance value round-trips correctly
//! through the bitcast<f32> pipeline — a regression that drops the
//! distance write or swaps the slot offset would fire here.

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

/// Sanity-pin for the four Wave 2 piece 2 movement EffectOps.
///
/// We register four abilities (Lunge=Dash, Phase=Blink, Smash=Knockback,
/// Hook=Pull) into a single registry and assign one ability to each of
/// four agents. After one dispatch tick, the event_ring should hold
/// exactly four chronicle records, one per agent — kinds 47..50 in the
/// order each agent's level routes to its registered AbilityId. The
/// per-shape distance offsets get verified per-record:
///   - Kinds 47/48 (Dash/Blink): distance lands at slot 3 (no target).
///   - Kinds 49/50 (Knockback/Pull): distance lands at slot 4, target
///     at slot 3.
#[test]
fn movement_chronicle_records_carry_correct_kind_and_distance() {
    const DASH_DISTANCE: f32 = 12.0;
    const BLINK_DISTANCE: f32 = 8.0;
    const KNOCKBACK_DISTANCE: f32 = 5.0;
    const PULL_DISTANCE: f32 = 3.0;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 4;

    // Build a registry with the four movement abilities. Each
    // single-target program emits one Effect{Dash,Blink,Knockback,Pull}
    // record per cast.
    let mut builder = AbilityRegistryBuilder::new();
    let lunge_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
        [EffectOp::Dash { distance: DASH_DISTANCE }],
    ));
    let phase_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: false, line_of_sight: false },
        [EffectOp::Blink { distance: BLINK_DISTANCE }],
    ));
    let smash_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Knockback { distance: KNOCKBACK_DISTANCE }],
    ));
    let hook_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Pull { distance: PULL_DISTANCE }],
    ));
    let registry = builder.build();

    // Per-agent levels = AbilityId.raw() to dispatch — agent 0 fires
    // Lunge (Dash), agent 1 fires Phase (Blink), etc. The dispatcher
    // reads agent_level[caster_slot] to pick the AbilityId.
    let per_agent_levels = vec![lunge_id.raw(), phase_id.raw(), smash_id.raw(), hook_id.raw()];
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
                "[movement_chronicle_pin] skipping: no wgpu adapter available",
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
    // (kind=47), (kind=48), (kind=49), (kind=50) in ascending order.
    let mut sorted: Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> = records.clone();
    sorted.sort_by_key(|r| r[0]);

    // Per-record expectations: (kind, actor_slot, distance, is_forced_motion).
    // The agent slot that fired each ability is determined by
    // `per_agent_levels` order: agent 0 → Lunge (kind 47), agent 1 →
    // Phase (kind 48), agent 2 → Smash (kind 49), agent 3 → Hook (kind 50).
    let expected: [(u32, u32, f32, bool); 4] = [
        (47, 0, DASH_DISTANCE,      false), // Dash @ agent 0 — caster-self
        (48, 1, BLINK_DISTANCE,     false), // Blink @ agent 1 — caster-self
        (49, 2, KNOCKBACK_DISTANCE, true),  // Knockback @ agent 2 — forced motion
        (50, 3, PULL_DISTANCE,      true),  // Pull @ agent 3 — forced motion
    ];
    for (i, &(expected_kind, expected_slot, expected_distance, is_forced)) in
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

        if is_forced {
            // Knockback/Pull: target at slot 3 (caster==target for
            // implicit-target self-cast), distance bitcast at slot 4.
            assert_eq!(
                r[3], expected_slot,
                "record {i} (forced motion): target slot — implicit-target \
                 rule writes caster_slot == target_slot (got {}, expected {})",
                r[3], expected_slot,
            );
            assert_eq!(
                r[4], expected_distance.to_bits(),
                "record {i} (forced motion): distance at payload word 2 \
                 (got 0x{:08x}, expected 0x{:08x})",
                r[4], expected_distance.to_bits(),
            );
            // Tail words zeroed.
            for j in 5..CHRONICLE_STRIDE_U32 as usize {
                assert_eq!(r[j], 0, "record {i} (forced motion): tail word {j} must be zero");
            }
        } else {
            // Dash/Blink: distance bitcast at slot 3 (no target field).
            assert_eq!(
                r[3], expected_distance.to_bits(),
                "record {i} (caster-self): distance at payload word 1 \
                 (got 0x{:08x}, expected 0x{:08x})",
                r[3], expected_distance.to_bits(),
            );
            // Tail words zeroed (slots 4..9).
            for j in 4..CHRONICLE_STRIDE_U32 as usize {
                assert_eq!(r[j], 0, "record {i} (caster-self): tail word {j} must be zero");
            }
        }
    }
}
