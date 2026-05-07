//! Slice γ closer behavioral pin: the GPU dispatcher's chronicle arm
//! for Summon (kind 24 → 62) writes the expected (kind, actor,
//! template_hash, count, lifetime_ticks) tuple for a single self-cast.
//!
//! The parity sweep test (`parity_apply_program_sweep.rs`) covers the
//! CPU↔GPU byte-equality matrix across all 46 abilities × 5 ticks
//! (including `EvocationSummon`). This pin is a smaller, targeted
//! assertion: build a registry with one Summon ability, pre-seed one
//! agent, dispatch at tick=10, and confirm the resulting event_ring
//! carries kind=62 with the correct per-shape payload offsets:
//!   - Kind 62 (Summon, caster-self): template_hash at slot 3, count
//!     (u8 widened to u32) at slot 4, lifetime_ticks (raw u32) at slot
//!     5. No target field on the engine event. The dispatcher splits
//!     the packed `payload_b` (= count<<24 | lifetime in low 24 bits)
//!     into distinct ring slots so consumers don't have to redo the
//!     bit-unpack on read.
//!
//! Distinct from the parity sweep (which only proves CPU and GPU agree),
//! this pin asserts the absolute payload values round-trip correctly
//! through the bit-unpack pipeline — particularly that count/lifetime
//! land in the right slots after the dispatcher's `payload_b >> 24` /
//! `payload_b & 0x00FFFFFF` decode. A regression that swaps the slots
//! or drops the bit-mask would surface here even when the parity test
//! still agrees with itself (CPU and GPU could agree on a wrong value).
//!
//! Companion to `buff_reflect_chronicle_pin.rs` (kinds 58..61) — same
//! fixture pattern, one slice-γ-closer arm.

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats, CHRONICLE_STRIDE_U32};
use engine::ability::{
    AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};

/// Sanity-pin for the slice γ closer (Summon) EffectOp.
///
/// We register one ability (`EvocationSummon`) and assign it to a
/// single agent. After one dispatch tick, the event_ring should hold
/// exactly one chronicle record at kind=62. The per-shape payload
/// offsets get verified:
///   - slot 0 = 62 (kind tag)
///   - slot 1 = tick
///   - slot 2 = caster_slot
///   - slot 3 = template_hash (raw u32 = payload_a)
///   - slot 4 = count (u32 widened from u8 = `(payload_b >> 24) & 0xFF`)
///   - slot 5 = lifetime_ticks (raw u32 = `payload_b & 0x00FFFFFF`)
///   - slots 6..9 = zero
#[test]
fn summon_chronicle_record_carries_correct_kind_and_payloads() {
    // The earlier "multi-spawn semantics need a new dispatch shape"
    // deferral was misleading — per `crates/engine/src/ability/apply.rs`,
    // the CPU side writes ONE `ApplyEvent::Summon` per cast carrying
    // packed (count, lifetime). Downstream N-entity spawning is a
    // separate consumer concern, distinct from the dispatcher's
    // single-record chronicle write.
    const TEMPLATE_HASH: u32 = 0xDEADBEEF;
    const COUNT: u8 = 3;
    const LIFETIME: u32 = 120;
    const TICK: u32 = 10;
    const N_AGENTS: u32 = 1;

    let mut builder = AbilityRegistryBuilder::new();
    let evocation_summon_id = builder.register(AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 60, hostile_only: false, line_of_sight: false },
        [EffectOp::Summon {
            template_hash: TEMPLATE_HASH,
            count: COUNT,
            lifetime_ticks: LIFETIME,
        }],
    ));
    let registry = builder.build();

    let per_agent_levels = vec![evocation_summon_id.raw()];
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
                "[summon_chronicle_pin] skipping: no wgpu adapter available",
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

    let r = &records[0];
    assert_eq!(r[0], 62, "Summon: kind tag — EffectSummonApplied");
    assert_eq!(r[1], TICK, "Summon: tick");
    assert_eq!(r[2], 0, "Summon: actor slot — caster_slot for agent 0");
    assert_eq!(
        r[3], TEMPLATE_HASH,
        "Summon (caster-self): template_hash at payload word 1 \
         (raw u32 = payload_a; no target field on engine event)"
    );
    assert_eq!(
        r[4], COUNT as u32,
        "Summon: count at payload word 2 (u8 widened to u32 via \
         `(payload_b >> 24) & 0xFF`)"
    );
    assert_eq!(
        r[5], LIFETIME,
        "Summon: lifetime_ticks at payload word 3 (raw u32 via \
         `payload_b & 0x00FFFFFF` low 24 bits)"
    );
    for i in 6..CHRONICLE_STRIDE_U32 as usize {
        assert_eq!(r[i], 0, "Summon: tail word {i} must be zero");
    }
}
