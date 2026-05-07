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

#[test]
fn aoe_cone_degenerate_under_self_cast_emits_no_records() {
    // #178 AOE Path B Cone — GPU pin for the degenerate-cone branch.
    //
    // Smoke-fixture self-cast invariant: `caster_slot == target_slot`,
    // so `apex == target_pos == agent_pos[0]`. The cone's `direction
    // = target_pos - apex = (0,0,0)` is degenerate; the WGSL kernel's
    // `dir_len_sq < 1e-6 → no-op` branch skips the spatial walk and
    // emits zero records. Pins the degenerate-cone semantic
    // explicitly so a future change to the direction calc surfaces
    // here (e.g. a "caster auto-faces nearest enemy on degenerate"
    // re-target would silently break parity with the CPU oracle's
    // empty-set).
    //
    // Fan layout (4 agents): caster at origin, three candidates that
    // a non-degenerate cone(60°, 4) would hit (slots 1, 2, 3 in front
    // of caster). With this fixture the cone WOULD hit them; under
    // self-cast it doesn't, because the direction is undefined.
    let mut slash = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 25.0 }],
    );
    slash.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Cone,
        args: [60.0, 4.0, 0.0, 0.0],
    }));

    let mut builder = AbilityRegistryBuilder::new();
    let slash_id = builder.register(slash);
    let registry = builder.build();

    const N_AGENTS: u32 = 4;
    let levels: Vec<u32> = vec![slash_id.raw(); N_AGENTS as usize];
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
                "[aoe_chronicle_pin] skipping cone test: no wgpu adapter available."
            );
            return;
        }
    };
    let mut state = state;

    // Only caster (slot 0) alive; spatial grid still holds positions
    // for slots 1..3 so the walk would find them as candidates if
    // the direction were non-degenerate.
    state.set_agent_alive(&[1, 0, 0, 0]);
    state.set_agent_positions(&[
        [0.0, 0.0, 0.0],   // caster apex
        [3.0, -1.0, 0.0],  // would-be in-cone bottom (≈18.4°)
        [4.0, 0.0, 0.0],   // would-be on-axis
        [3.0, 1.0, 0.0],   // would-be in-cone top
    ]);

    state.step(0);

    let tail = state.read_event_tail();
    assert_eq!(
        tail, 0,
        "Self-cast cone (caster_slot == target_slot ⇒ direction \
         degenerate) must emit zero chronicle records. The cone \
         WGSL's `dir_len_sq < 1e-6` branch skips the spatial walk; \
         CPU oracle's `apply_program_aoe_cone_filter` matches by \
         returning an empty in-cone set. Got tail={tail}",
    );
}

#[test]
fn aoe_box_hits_within_aabb_extents() {
    // #179 AOE Path B Box — GPU pin for the box branch.
    //
    // Fixture shape: same 4-agent row as Cleave (positions at x=0, 1.5,
    // 3.0, 4.5). Pulverize ability: `Damage(20) in box(1.5, 1.5, 1.5)`.
    // Cast from caster slot 0 (target_slot=0 ⇒ aoe_center=(0,0,0) under
    // the smoke fixture's implicit-target rule). With half-extents
    // (1.5, 1.5, 1.5):
    //   - slot 0 at (0,0,0):       |d|=0 → in-box
    //   - slot 1 at (1.5, 0, 0):   |d.x|=1.5 → in-box (closed AABB,
    //                              candidate at the wall is in)
    //   - slot 2 at (3.0, 0, 0):   |d.x|=3.0 > 1.5 → out
    //   - slot 3 at (4.5, 0, 0):   |d.x|=4.5 > 1.5 → out
    //
    // Expected: 2 chronicle records (kind=26 EffectDamageApplied,
    // actor=0, payload_a=bitcast<u32>(20.0)=0x41A00000), targets in
    // {0, 1} after sort.
    //
    // **Closed-AABB pin.** Agent 1 at exactly the wall (|d.x|=wx)
    // landing in-box validates that the WGSL's `<=` predicate matches
    // the CPU oracle's `<=` semantic. A future "open AABB" change
    // (using `<`) would silently drop slot 1 — this assertion catches
    // it.
    //
    // **Spatial walk constraint pin.** wx=wy=wz=1.5 ≤ cell_size=6.0,
    // so the 27-cell walk visits every cell that could hold a
    // candidate. Larger extents would require additional cell rings
    // (deferred — see the box branch's WGSL doc-comment).
    let mut pulverize = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 20.0 }],
    );
    pulverize.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Box,
        args: [1.5, 1.5, 1.5, 0.0],
    }));

    let mut builder = AbilityRegistryBuilder::new();
    let pulverize_id = builder.register(pulverize);
    assert_eq!(
        pulverize_id,
        AbilityId::new(1).unwrap(),
        "Pulverize must register at AbilityId(1)",
    );
    let registry = builder.build();

    const N_AGENTS: u32 = 4;
    let levels: Vec<u32> = vec![pulverize_id.raw(); N_AGENTS as usize];
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
                "[aoe_chronicle_pin] skipping box test: no wgpu adapter available."
            );
            return;
        }
    };
    let mut state = state;

    state.set_agent_alive(&[1, 0, 0, 0]);
    state.set_agent_positions(&[
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],  // exactly at +x wall — in-box (closed AABB)
        [3.0, 0.0, 0.0],
        [4.5, 0.0, 0.0],
    ]);

    state.step(0);

    let tail = state.read_event_tail();
    assert_eq!(
        tail, 2,
        "AOE Pulverize (box(1.5, 1.5, 1.5)) at center (0,0,0) over row \
         of 4 agents must emit exactly 2 chronicle records (slots 0 + \
         1 are in-box; slot 1 at the +x wall is in-box per the closed-\
         AABB ≤ semantic). Got tail={tail}",
    );

    let mut records = state.read_event_ring(tail);
    // P11 sort: GPU's atomicAdd doesn't preserve target order. Sort by
    // target slot so the assertions can index by sorted position.
    records.sort_by_key(|r| (r[3], r[0]));

    // Both records: kind=26 (EffectDamageApplied), tick=0, actor=0,
    // payload_a=bitcast<u32>(20.0)=0x41A00000.
    let damage_kind: u32 = 26;
    let amount_bits: u32 = 20.0_f32.to_bits();
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
            "record {i} payload_a: expected bitcast<u32>(20.0)=0x{amount_bits:08x} got 0x{:08x}",
            r[4],
        );
    }
}

// ---------------------------------------------------------------- //
// AOE Path B Sphere/Ring/Line GPU pins (#180)
// ---------------------------------------------------------------- //

#[test]
fn aoe_sphere_hits_two_in_row() {
    // #180 AOE Path B Sphere — GPU pin. Sphere is mathematically
    // equivalent to Circle today (3D dist² ≤ radius²), routed through
    // the dispatcher's dedicated Sphere branch (`area_kind == 6u`).
    //
    // Mirror of the Circle pin: 4-agent row at (0, 1.5, 3.0, 4.5),
    // sphere(2.0) cast from caster slot 0; in-sphere = {slot 0 (d=0),
    // slot 1 (d=1.5)}. Expected 2 chronicle records.
    let mut blast_sphere = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 18.0 }],
    );
    blast_sphere.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Sphere,
        args: [2.0, 0.0, 0.0, 0.0],
    }));

    let mut builder = AbilityRegistryBuilder::new();
    let blast_sphere_id = builder.register(blast_sphere);
    let registry = builder.build();

    const N_AGENTS: u32 = 4;
    let levels: Vec<u32> = vec![blast_sphere_id.raw(); N_AGENTS as usize];
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
                "[aoe_chronicle_pin] skipping sphere test: no wgpu adapter available."
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
        "AOE BlastSphere (sphere(2.0)) at center (0,0,0) over row of 4 \
         agents must emit exactly 2 chronicle records (slots 0 + 1, both \
         within radius — Sphere is equivalent to Circle today). Got tail={tail}",
    );

    let mut records = state.read_event_ring(tail);
    records.sort_by_key(|r| (r[3], r[0]));

    let damage_kind: u32 = 26;
    let amount_bits: u32 = 18.0_f32.to_bits();
    for (i, expected_target) in [0u32, 1u32].iter().enumerate() {
        let r = records[i];
        assert_eq!(r[0], damage_kind, "record {i} kind");
        assert_eq!(r[1], 0, "record {i} tick");
        assert_eq!(r[2], 0, "record {i} actor");
        assert_eq!(r[3], *expected_target, "record {i} target");
        assert_eq!(r[4], amount_bits, "record {i} payload_a (18.0)");
    }
}

#[test]
fn aoe_ring_excludes_inner_radius() {
    // #180 AOE Path B Ring — GPU pin for the annulus gate.
    //
    // Fixture: 4-agent row at (0, 1.5, 3.0, 4.5). Ring(0.5, 2.0) cast
    // from caster slot 0:
    //   - slot 0 (d=0):   d² < inner² ⇒ INNER-EXCLUDED
    //   - slot 1 (d=1.5): inner² ≤ d² ≤ outer² ⇒ in
    //   - slot 2 (d=3.0): d² > outer² ⇒ OUT
    //   - slot 3 (d=4.5): d² > outer² ⇒ OUT
    //
    // Expected: 1 chronicle record (kind=26, actor=0, target=1,
    // payload_a=bitcast<u32>(14.0)=0x41600000). The inner-radius
    // exclusion is the distinguishing feature vs Circle/Sphere — pin
    // explicitly so a regression on the inner gate (e.g. `>=` flipped
    // to `>`) surfaces as a count or target mismatch here.
    let mut shockwave_ring = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 14.0 }],
    );
    shockwave_ring.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Ring,
        args: [0.5, 2.0, 0.0, 0.0],
    }));

    let mut builder = AbilityRegistryBuilder::new();
    let shockwave_ring_id = builder.register(shockwave_ring);
    let registry = builder.build();

    const N_AGENTS: u32 = 4;
    let levels: Vec<u32> = vec![shockwave_ring_id.raw(); N_AGENTS as usize];
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
                "[aoe_chronicle_pin] skipping ring test: no wgpu adapter available."
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
        tail, 1,
        "AOE ShockwaveRing (ring(0.5, 2.0)) at center (0,0,0) over row \
         of 4 agents must emit exactly 1 chronicle record (slot 1 only — \
         slot 0 inner-excluded at d=0, slots 2+3 outer-excluded). Got tail={tail}",
    );

    let records = state.read_event_ring(tail);
    let r = records[0];
    let damage_kind: u32 = 26;
    let amount_bits: u32 = 14.0_f32.to_bits();
    assert_eq!(r[0], damage_kind, "ring record kind");
    assert_eq!(r[1], 0, "ring record tick");
    assert_eq!(r[2], 0, "ring record actor must be caster slot 0");
    assert_eq!(r[3], 1, "ring record target must be slot 1 (only in-ring agent)");
    assert_eq!(r[4], amount_bits, "ring record payload_a (14.0)");
}

#[test]
fn aoe_line_degenerate_under_self_cast_emits_no_records() {
    // #180 AOE Path B Line — GPU pin for the degenerate-line branch.
    //
    // Smoke-fixture self-cast invariant: `caster_slot == target_slot`,
    // so `apex == target_pos == agent_pos[0]`. The line's `direction =
    // target_pos - apex = (0,0,0)` is degenerate; the WGSL kernel's
    // `dir_len_sq < 1e-6 → no-op` branch skips the spatial walk and
    // emits zero records. Mirrors the cone degenerate pin shape (#178).
    //
    // Fixture (4 agents): caster at origin, three candidates that a
    // non-degenerate line(5.0, 1.0) facing +X WOULD hit (slots 1, 2 in
    // front of caster within the corridor, slot 3 outside the corridor).
    // With this fixture the line WOULD hit them; under self-cast it
    // doesn't, because the direction is undefined.
    let mut piercing_line = AbilityProgram::new_single_target(
        5.0,
        Gate { cooldown_ticks: 30, hostile_only: true, line_of_sight: false },
        [EffectOp::Damage { amount: 22.0 }],
    );
    piercing_line.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Line,
        args: [5.0, 1.0, 0.0, 0.0],
    }));

    let mut builder = AbilityRegistryBuilder::new();
    let piercing_line_id = builder.register(piercing_line);
    let registry = builder.build();

    const N_AGENTS: u32 = 4;
    let levels: Vec<u32> = vec![piercing_line_id.raw(); N_AGENTS as usize];
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
                "[aoe_chronicle_pin] skipping line test: no wgpu adapter available."
            );
            return;
        }
    };
    let mut state = state;

    // Only caster (slot 0) alive; spatial grid still holds positions
    // for slots 1..3 so the walk would find them as candidates if
    // the direction were non-degenerate.
    state.set_agent_alive(&[1, 0, 0, 0]);
    state.set_agent_positions(&[
        [0.0, 0.0, 0.0],   // caster apex
        [1.0, 0.0, 0.0],   // would-be in-line (along=1, perp=0)
        [3.0, 0.0, 0.0],   // would-be in-line (along=3, perp=0)
        [2.0, 0.6, 0.0],   // would-be out-of-line (perp=0.6 > 0.5)
    ]);

    state.step(0);

    let tail = state.read_event_tail();
    assert_eq!(
        tail, 0,
        "Self-cast line (caster_slot == target_slot ⇒ direction \
         degenerate) must emit zero chronicle records. The line WGSL's \
         `dir_len_sq < 1e-6` branch skips the spatial walk; CPU oracle's \
         `apply_program_aoe_line_filter` matches by returning an empty \
         in-line set. Got tail={tail}",
    );
}
