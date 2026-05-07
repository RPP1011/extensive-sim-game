//! Slice γ tail behavioral pin: the GPU dispatcher's chronicle arms for
//! Buff/Harvest/PlaceVoxel/Reflect write the expected (kind, actor,
//! target?, packed payloads) tuple for a single self-cast.
//!
//! The parity sweep test (`parity_apply_program_sweep.rs`) covers the
//! CPU↔GPU byte-equality matrix across all 29 abilities × 5 ticks. This
//! pin is a smaller, targeted assertion: build a registry with one
//! ability per slice γ tail op (`Empower=Buff`, `Mirror=Reflect`,
//! `Reap_Ore=Harvest`, `Drop_Stone=PlaceVoxel`), pre-seed four agents,
//! dispatch at tick=10, and confirm the resulting event_ring carries
//! kinds 58..61 with the correct per-shape payload offsets. Two distinct
//! shape families:
//!   - Buff/Reflect: target-cast with packed signed payload — the i16
//!     `magnitude_q8` (Buff) and `fraction_q8` (Reflect) must round-trip
//!     correctly through the i16→i32→u32 (Buff) and i16→u16→u32 (Reflect)
//!     paths to land at the dispatcher's raw `payload_a`/`payload_b`
//!     ring slots. We exercise BOTH with negative values — a regression
//!     that loses the sign or swaps the encoding surfaces here.
//!   - Harvest/PlaceVoxel: caster-self with unsigned u32 payloads —
//!     `kind_hash` (Harvest+PlaceVoxel) and `amount` (Harvest) round-trip
//!     verbatim through the dispatcher's payload_a/payload_b stores.
//!
//! Distinct from the parity sweep (which only proves CPU and GPU agree),
//! this pin asserts the absolute payload values round-trip correctly
//! through the raw-u32 pipeline — particularly for the signed packed
//! sub-fields (Buff's magnitude_q8 and Reflect's fraction_q8). A
//! regression that drops the high half of the packed payload, or that
//! sign-casts incorrectly, would surface here even when the parity test
//! still agrees with itself (CPU and GPU could agree on a wrong value).
//!
//! Companion to the existing `extended_status_chronicle_pin.rs`
//! (kinds 54..57) — same fixture pattern, four extended-status arms
//! become four slice-γ-tail arms.

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::program::BuffStat;
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

/// Sanity-pin for the four slice γ tail EffectOps.
///
/// We register four abilities (Empower=Buff, Mirror=Reflect, Reap_Ore=
/// Harvest, Drop_Stone=PlaceVoxel) into a single registry and assign
/// one ability to each of four agents. After one dispatch tick, the
/// event_ring should hold exactly four chronicle records, one per agent
/// — kinds 58..61 in the order each agent's level routes to its
/// registered AbilityId. The per-shape payload offsets get verified
/// per-record:
///   - Kind 58 (Buff, target-cast): packed payload_a (= stat | mag_q8 << 8)
///     at slot 4, raw payload_b (= duration_ticks) at slot 5.
///   - Kind 61 (Reflect, target-cast): raw duration_ticks at slot 4,
///     packed payload_b (low 16 bits = fraction_q8 i16) at slot 5.
///   - Kind 59 (Harvest, caster-self): kind_hash at slot 3, amount at
///     slot 4. No target field on engine event.
///   - Kind 60 (PlaceVoxel, caster-self): kind_hash at slot 3 only.
///     Position implicit from cast's target world position.
#[test]
fn slice_gamma_tail_chronicle_records_carry_correct_kind_and_payloads() {
    // Use signed values that surface sign-cast regressions if the
    // dispatcher drops the high half or zero-extends incorrectly.
    const EMPOWER_STAT: BuffStat = BuffStat::AttackSpeed;     // ordinal 1
    const EMPOWER_MAG_Q8: i16 = -64;                          // negative — exercise sign extend
    const EMPOWER_DURATION: u32 = 50;
    const MIRROR_DURATION: u32 = 50;
    const MIRROR_FRACTION_Q8: i16 = -64;                      // negative — exercise low-16-bit pack
    const REAP_ORE_KIND_HASH: u32 = 0xCAFEBABE;
    const REAP_ORE_AMOUNT: u16 = 5;
    const DROP_STONE_KIND_HASH: u32 = 0xFACEFEED;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 4;

    // Build a registry with the four slice-γ-tail abilities. Each
    // single-target program emits one Effect{Buff,Reflect,Harvest,
    // PlaceVoxel} record per cast.
    let mut builder = AbilityRegistryBuilder::new();
    let empower_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Buff {
            stat: EMPOWER_STAT,
            magnitude_q8: EMPOWER_MAG_Q8,
            duration_ticks: EMPOWER_DURATION,
        }],
    ));
    let mirror_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: true, line_of_sight: false },
        [EffectOp::Reflect {
            duration_ticks: MIRROR_DURATION,
            fraction_q8: MIRROR_FRACTION_Q8,
        }],
    ));
    let reap_ore_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Harvest {
            kind_hash: REAP_ORE_KIND_HASH,
            amount: REAP_ORE_AMOUNT,
        }],
    ));
    let drop_stone_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::PlaceVoxel {
            kind_hash: DROP_STONE_KIND_HASH,
        }],
    ));
    let registry = builder.build();

    // Per-agent levels = AbilityId.raw() to dispatch — agent 0 fires
    // Empower (Buff), agent 1 fires Mirror (Reflect), agent 2 fires
    // Reap_Ore (Harvest), agent 3 fires Drop_Stone (PlaceVoxel). The
    // dispatcher reads agent_level[caster_slot] to pick the AbilityId.
    let per_agent_levels = vec![
        empower_id.raw(),
        mirror_id.raw(),
        reap_ore_id.raw(),
        drop_stone_id.raw(),
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
                "[buff_reflect_chronicle_pin] skipping: no wgpu adapter available",
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
    // (kind=58), (kind=59), (kind=60), (kind=61) in ascending order.
    let mut sorted: Vec<[u32; CHRONICLE_STRIDE_U32 as usize]> = records.clone();
    sorted.sort_by_key(|r| r[0]);

    // Buff at kind=58 (sorted[0]).
    let r = &sorted[0];
    assert_eq!(r[0], 58, "Buff: kind tag — EffectBuffApplied");
    assert_eq!(r[1], TICK, "Buff: tick");
    assert_eq!(r[2], 0, "Buff: actor slot — caster_slot for agent 0");
    assert_eq!(
        r[3], 0,
        "Buff (target-cast): target slot — implicit-target rule writes \
         caster_slot == target_slot"
    );
    // Expected packed payload_a: stat (u8 low byte) | mag_q8 (i16 → i32 → u32 << 8).
    let expected_buff_pa =
        (EMPOWER_STAT as u32) | ((EMPOWER_MAG_Q8 as i32 as u32) << 8);
    assert_eq!(
        r[4], expected_buff_pa,
        "Buff: packed payload_a (= stat | magnitude_q8 << 8). Negative \
         magnitude_q8 must sign-extend i16 → i32 before shift; got 0x{:08x} \
         expected 0x{:08x}",
        r[4], expected_buff_pa,
    );
    assert_eq!(
        r[5], EMPOWER_DURATION,
        "Buff: payload_b = duration_ticks (raw u32)"
    );
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Buff: tail word {i} must be zero");
    }

    // Harvest at kind=59 (sorted[1]).
    let r = &sorted[1];
    assert_eq!(r[0], 59, "Harvest: kind tag — EffectHarvestApplied");
    assert_eq!(r[1], TICK, "Harvest: tick");
    assert_eq!(r[2], 2, "Harvest: actor slot — caster_slot for agent 2");
    assert_eq!(
        r[3], REAP_ORE_KIND_HASH,
        "Harvest (caster-self): kind_hash at payload word 1 (no target field)"
    );
    assert_eq!(
        r[4], REAP_ORE_AMOUNT as u32,
        "Harvest (caster-self): amount at payload word 2 (u16 widened to u32)"
    );
    for i in 5..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Harvest: tail word {i} must be zero");
    }

    // PlaceVoxel at kind=60 (sorted[2]).
    let r = &sorted[2];
    assert_eq!(r[0], 60, "PlaceVoxel: kind tag — EffectPlaceVoxelApplied");
    assert_eq!(r[1], TICK, "PlaceVoxel: tick");
    assert_eq!(r[2], 3, "PlaceVoxel: actor slot — caster_slot for agent 3");
    assert_eq!(
        r[3], DROP_STONE_KIND_HASH,
        "PlaceVoxel (caster-self): kind_hash at payload word 1 \
         (no target / position fields — position implicit from cast target)"
    );
    for i in 4..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "PlaceVoxel: tail word {i} must be zero");
    }

    // Reflect at kind=61 (sorted[3]).
    let r = &sorted[3];
    assert_eq!(r[0], 61, "Reflect: kind tag — EffectReflectApplied");
    assert_eq!(r[1], TICK, "Reflect: tick");
    assert_eq!(r[2], 1, "Reflect: actor slot — caster_slot for agent 1");
    assert_eq!(
        r[3], 1,
        "Reflect (target-cast): target slot — implicit-target rule writes \
         caster_slot == target_slot"
    );
    assert_eq!(
        r[4], MIRROR_DURATION,
        "Reflect: payload_a = duration_ticks (raw u32)"
    );
    // Expected packed payload_b: fraction_q8 (i16 → u16 → u32, zero-extend).
    // For -64_i16: u16 wraps to 0xFFC0; u32 zero-extend → 0x0000_FFC0.
    let expected_reflect_pb = (MIRROR_FRACTION_Q8 as u16) as u32;
    assert_eq!(
        expected_reflect_pb, 0x0000_FFC0,
        "sanity: -64_i16 → 0xFFC0 in low 16 bits"
    );
    assert_eq!(
        r[5], expected_reflect_pb,
        "Reflect: packed payload_b's low 16 bits carry fraction_q8 \
         (i16 → u16 → u32, zero-extend). Consumer sign-extends to recover \
         the negative value; got 0x{:08x} expected 0x{:08x}",
        r[5], expected_reflect_pb,
    );
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Reflect: tail word {i} must be zero");
    }
}
