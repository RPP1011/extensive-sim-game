//! Task #121 (Path B — GPU dispatcher): AOE behavioral E2E pin on
//! real GPU.
//!
//! GPU-side mirror of `engine/tests/aoe_multi_agent_e2e.rs::
//! aoe_circle_hits_two_in_row_three_agent_fixture` (Path A — CPU
//! oracle). Same shape: a row of agents with a Cleave ability
//! (`Damage 30 in circle(<radius>)`) cast from the leftmost agent;
//! assert the dispatcher emits one chronicle record per in-circle
//! target and only those targets.
//!
//! ## Why a separate pin from the parity sweep?
//!
//! The parity sweep (`parity_apply_program_sweep.rs`) ALSO tests
//! Cleave at index 31, but its assertion is "GPU records byte-equal
//! CPU oracle records" — a relative pin. This pin is absolute: it
//! pins the EXACT chronicle record set (kind, actor, target,
//! payload_a) the dispatcher produces for a known fixture. If the
//! CPU oracle and GPU walk both regress in the same direction, the
//! parity sweep stays green but THIS pin trips. Mirrors the role of
//! `engine/tests/aoe_multi_agent_e2e.rs` for Path A.
//!
//! ## Fixture shape
//!
//! 4 agents at (0, 0, 0), (1.5, 0, 0), (3.0, 0, 0), (4.5, 0, 0).
//! Cleave radius=2.0 cast from agent 0 (caster_slot=0,
//! target_slot=0 ⇒ aoe_center=(0,0,0) under the smoke fixture's
//! implicit-target rule). In-radius set: agents 0 (d=0) and 1
//! (d=1.5); agents 2 (d=3.0) and 3 (d=4.5) are out.
//!
//! Only agent 0 is alive (`agent_alive = [1, 0, 0, 0]`) — the
//! dispatcher's `where (self.alive)` gate fires once for caster
//! slot 0; the spatial grid still holds entries for slots 0..3 so
//! the walk finds them as candidates.
//!
//! Expected: 2 chronicle records, both with kind=26
//! (EffectDamageApplied), actor=0, payload_a=bitcast<u32>(30.0)=
//! 0x41F00000, targets in {0, 1} (sort-canonicalized to that order
//! post-readback).

use apply_ability_smoke_runtime::{ApplyAbilitySmokeState, PerAgentStats};
use engine::ability::{
    AbilityId, AbilityProgram, AbilityRegistryBuilder, EffectOp, Gate,
};
use engine::ability::program::{EffectAreaShape, ShapeKind};

#[test]
fn aoe_circle_hits_two_in_row_four_agent_fixture_on_gpu() {
    // Build a Cleave ability program: Damage(30) in Circle(2.0).
    let mut cleave = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 30.0 }],
    );
    cleave.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        args: [2.0, 0.0, 0.0, 0.0],
    }));

    let mut builder = AbilityRegistryBuilder::new();
    let cleave_id = builder.register(cleave);
    assert_eq!(
        cleave_id,
        AbilityId::new(1).unwrap(),
        "Cleave must register at AbilityId(1) — first program in this fixture's registry"
    );
    let registry = builder.build();

    // 4-agent fixture; only agent 0 is alive (the caster).
    const N_AGENTS: u32 = 4;
    let levels: Vec<u32> = vec![cleave_id.raw(); N_AGENTS as usize];
    let stats: Vec<PerAgentStats> = vec![PerAgentStats::default(); N_AGENTS as usize];

    let state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &levels,
        &stats,
    ) {
        Some(s) => s,
        None => {
            eprintln!(
                "[aoe_chronicle_pin] skipping: no wgpu adapter available on this \
                 host. The compile path still validated the AOE Path B emit."
            );
            return;
        }
    };
    let mut state = state;

    state.set_agent_alive(&[1, 0, 0, 0]);
    state.set_agent_positions(&[
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ]);

    state.step(0);

    let tail = state.read_event_tail();
    assert_eq!(
        tail, 2,
        "AOE Cleave (radius=2.0) at center (0,0,0) over row of 4 agents must \
         emit exactly 2 chronicle records (slots 0 + 1, both within radius). \
         Got tail={tail}",
    );

    let mut records = state.read_event_ring(tail);
    // P11 sort: GPU's atomicAdd doesn't preserve target order. Sort by
    // target slot so the assertions can index by sorted position.
    records.sort_by_key(|r| (r[3], r[0]));

    // Both records: kind=26 (EffectDamageApplied), tick=0, actor=0,
    // payload_a=bitcast<u32>(30.0)=0x41F00000.
    let damage_kind: u32 = 26;
    let amount_bits: u32 = 30.0_f32.to_bits();
    for (i, expected_target) in [0u32, 1u32].iter().enumerate() {
        let r = records[i];
        assert_eq!(r[0], damage_kind, "record {i} kind: expected EffectDamageApplied=26 got {}", r[0]);
        assert_eq!(r[1], 0, "record {i} tick must be 0");
        assert_eq!(r[2], 0, "record {i} actor must be caster slot 0");
        assert_eq!(
            r[3], *expected_target,
            "record {i} target: expected slot {expected_target} got {}",
            r[3],
        );
        assert_eq!(
            r[4], amount_bits,
            "record {i} payload_a: expected bitcast<u32>(30.0)=0x{amount_bits:08x} got 0x{:08x}",
            r[4],
        );
    }
}

#[test]
fn aoe_circle_with_zero_radius_hits_only_caster() {
    // Edge case: radius=0 should NOT match anyone (in-circle test is
    // `dot(d, d) <= radius_sq` — strictly inside or on the surface).
    // For self-cast with d=(0,0,0), `dot=0` ≤ `radius_sq=0`, so the
    // caster (agent 0) IS in-circle and receives the chronicle record.
    // A future stricter "open ball" semantic would change this — pin
    // current behavior so a regression on the gating predicate
    // surfaces here.
    let mut cleave = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 30.0 }],
    );
    cleave.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        args: [0.0, 0.0, 0.0, 0.0],
    }));

    let mut builder = AbilityRegistryBuilder::new();
    let cleave_id = builder.register(cleave);
    let registry = builder.build();

    const N_AGENTS: u32 = 2;
    let levels: Vec<u32> = vec![cleave_id.raw(); N_AGENTS as usize];
    let stats: Vec<PerAgentStats> = vec![PerAgentStats::default(); N_AGENTS as usize];

    let state = match ApplyAbilitySmokeState::try_new_with_registry(
        N_AGENTS,
        &registry,
        &levels,
        &stats,
    ) {
        Some(s) => s,
        None => return,
    };
    let mut state = state;

    state.set_agent_alive(&[1, 0]);
    state.set_agent_positions(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]);

    state.step(0);
    let tail = state.read_event_tail();
    assert_eq!(
        tail, 1,
        "Zero-radius Circle from caster slot 0: only the caster itself \
         (dist=0 ≤ 0) is in-circle; agent 1 (dist=1) is out. Expected 1 \
         chronicle record, got tail={tail}",
    );
    let records = state.read_event_ring(tail);
    assert_eq!(records[0][3], 0, "the single record's target must be caster slot 0");
}
