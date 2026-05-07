//! Wave 2 piece 1 behavioral pin: the GPU dispatcher's chronicle arms
//! for Root/Silence/Fear/Taunt write the expected (kind, expires_at_tick)
//! tuple for a single self-cast.
//!
//! The parity sweep test (`parity_apply_program_sweep.rs`) covers the
//! CPU↔GPU byte-equality matrix across all 14 abilities × 5 ticks. This
//! pin is a smaller, targeted assertion: build a registry with one
//! `Stop` ability that has `EffectOp::Root { duration_ticks: 50 }`,
//! pre-seed agent 0, dispatch at tick=10, and confirm the resulting
//! event_ring slot 0 carries kind=43 (`EffectRootApplied`), target=0
//! (slot 0 self-cast), and payload = tick(10) + duration(50) = 60. The
//! same shape pins the other three control statuses (Silence/Fear/Taunt)
//! within one test by registering all four and indexing per agent slot.

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

/// Sanity-pin for the four Wave 2 piece 1 control statuses.
///
/// We register four abilities (Stop=Root, Mute=Silence, Terrify=Fear,
/// Provoke=Taunt) into a single registry and assign one ability to each
/// of four agents. After one dispatch tick, the event_ring should hold
/// exactly four chronicle records, one per agent — kinds 43..46 in the
/// order each agent's level routes to its registered AbilityId.
///
/// `expires_at_tick = tick + duration` is verified for every record so
/// a regression that drops the tick-add (or hardcodes the wrong base)
/// would fire here, distinct from the byte-equality parity sweep
/// (which only proves CPU and GPU agree, not that the value is
/// correct in absolute terms).
#[test]
fn control_status_chronicle_records_carry_correct_kind_and_expiry() {
    const DURATION: u32 = 50;
    const TICK: u32 = 10;
    const EXPECTED_EXPIRES: u32 = TICK + DURATION;
    const N_AGENTS: u32 = 4;

    // Build a registry with the four control-status abilities. Each
    // single-target program emits one Effect{Root,Silence,Fear,Taunt}
    // record per cast.
    let mut builder = AbilityRegistryBuilder::new();
    let stop_id    = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Root { duration_ticks: DURATION }],
    ));
    let mute_id    = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Silence { duration_ticks: DURATION }],
    ));
    let terrify_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Fear { duration_ticks: DURATION }],
    ));
    let provoke_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Taunt { duration_ticks: DURATION }],
    ));
    let registry = builder.build();

    // Per-agent levels = AbilityId.raw_u32 to dispatch — agent 0 fires
    // Stop (Root), agent 1 fires Mute (Silence), etc. The dispatcher
    // reads agent_level[caster_slot] to pick the AbilityId.
    let per_agent_levels = vec![stop_id.raw(), mute_id.raw(), terrify_id.raw(), provoke_id.raw()];
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
                "[control_status_chronicle_pin] skipping: no wgpu adapter available",
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

    // Sort by (kind, target_slot) to make the assertion order-independent
    // — atomicAdd ring slot ordering is workgroup-schedule-dependent.
    let mut sorted: Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> = records.clone();
    sorted.sort_by_key(|r| (r[0], r[3]));

    // After sorting, sorted[0..4] should be (kind=43, target=0..),
    // (kind=44, target=1..), (kind=45, target=2..), (kind=46, target=3..)
    // — but each agent dispatches ITS ability against ITSELF (caster ==
    // target == agent_slot), so the (kind, target) pair maps 1:1.
    let expected: [(u32, u32); 4] = [
        (43, 0), // Root @ agent 0
        (44, 1), // Silence @ agent 1
        (45, 2), // Fear @ agent 2
        (46, 3), // Taunt @ agent 3
    ];
    for (i, &(expected_kind, expected_slot)) in expected.iter().enumerate() {
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
            r[4], EXPECTED_EXPIRES,
            "record {i}: expires_at_tick = tick({TICK}) + duration({DURATION}) = {EXPECTED_EXPIRES}",
        );
        // Tail words zeroed.
        for j in 5..CHRONICLE_STRIDE_U32 as usize {
            assert_eq!(r[j], 0, "record {i}: tail word {j} must be zero");
        }
    }
}
