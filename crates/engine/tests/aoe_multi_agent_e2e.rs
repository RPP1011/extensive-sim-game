//! Task #121 (Path A — CPU oracle): spatial AOE behavioral E2E in a
//! multi-agent fixture.
//!
//! End-to-end pin for the multi-target AOE dispatch path. Stands up a
//! real `SimState` with ≥3 alive agents in a row, fires an AOE Strike
//! (`damage 30 in circle(2.5)`), and asserts the dispatcher emits one
//! `ApplyEvent::Damage` per target inside the spatial circle. The
//! spatial query uses `state.spatial().within_radius(...)` — the same
//! production-side primitive a sim's apply-physics handler would call
//! before invoking `apply_program_aoe` — so this test exercises the
//! actual integration shape, not just the dispatch math.
//!
//! The "multi-agent fixture in a row" shape pins these behaviors:
//!   1. **2+ chronicle records from 1 cast.** A single `apply_program_aoe`
//!      call expands a Circle slot across multiple targets in one pass
//!      (the deferred behavior #121 closes).
//!   2. **Spatial query ordering is preserved.** `within_radius` returns
//!      ids sorted ascending by `AgentId::raw()`; the dispatcher emits
//!      events in the same order (P11 — output is reproducible across
//!      runs and platforms).
//!   3. **Spatial cutoff is honored.** Agents outside the circle radius
//!      do NOT receive events — the AOE expansion is bounded by the
//!      caller's spatial query result, not the full agent set.
//!
//! **GPU parity.** GPU dispatcher remains single-target only — multi-
//! target AOE expansion on GPU is deferred (Path B, future slice).
//! Existing CPU↔GPU parity sweep at 29 abilities × 5 ticks stays valid
//! because that sweep uses no AOE-shape abilities; this slice does NOT
//! extend that sweep (would require GPU-side spatial expansion).

use engine::ability::apply::{
    apply_program_aoe, apply_program_aoe_box_filter, apply_program_aoe_cone_filter, ApplyEvent,
};
use engine::ability::program::{
    AbilityProgram, CasterStats, EffectAreaShape, EffectOp, Gate, ShapeKind,
};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};

/// Spawn `n` agents in a row at `(i, 0, 0)` for `i in 0..n`. Returns
/// the list of allocated AgentIds in spawn order.
fn spawn_row(state: &mut SimState, n: u32) -> Vec<AgentId> {
    (0..n)
        .map(|i| {
            state
                .spawn_agent(AgentSpawn {
                    pos: glam::Vec3::new(i as f32, 0.0, 0.0),
                    ..Default::default()
                })
                .expect("agent_cap large enough for n agents")
        })
        .collect()
}

#[test]
fn aoe_circle_hits_two_in_row_three_agent_fixture() {
    // 3 agents in a row at x = 0, 1, 2. Caster = ids[0]. Cast center
    // = ids[1]'s position (x=1). Circle radius = 0.5 — covers ids[1]
    // (distance 0) and NOT ids[0] (distance 1) or ids[2] (distance 1).
    // Expect exactly 1 chronicle record (ids[1]).
    //
    // Then re-fire with radius 1.5 — covers ids[0], ids[1], ids[2]
    // (all within 1.0 of x=1). Expect 3 chronicle records.
    //
    // Then re-fire with radius 1.5 but cast center = ids[0]'s position
    // (x=0). Covers ids[0] (distance 0) and ids[1] (distance 1) but
    // NOT ids[2] (distance 2). Expect 2 chronicle records — the "2+
    // chronicle records from 1 cast" shape the task title calls out.
    let mut state = SimState::new(8, 0xCAFE);
    let ids = spawn_row(&mut state, 3);
    assert_eq!(ids.len(), 3, "fixture must have 3 agents in a row");

    let mut prog = AbilityProgram::new_single_target(
        5.0,
        Gate {
            cooldown_ticks: 10,
            hostile_only: false,
            line_of_sight: false,
        },
        [EffectOp::Damage { amount: 30.0 }],
    );

    // Tight circle around ids[1]: only ids[1] is hit.
    prog.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        args: [0.5, 0.0, 0.0, 0.0],
    }));
    let center = state.agent_pos(ids[1]).expect("ids[1] alive");
    let aoe_targets = state.spatial().within_radius(&state, center, 0.5);
    assert_eq!(
        aoe_targets,
        vec![ids[1]],
        "tight circle must capture only ids[1]"
    );
    let events = apply_program_aoe(
        &prog,
        ids[0],
        ids[1],
        &aoe_targets,
        0,
        0xCAFE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    assert_eq!(events.len(), 1, "tight circle → 1 event");
    assert!(matches!(
        events[0],
        ApplyEvent::Damage { target, .. } if target == ids[1]
    ));

    // Wide circle around ids[1]: hits all three.
    prog.per_effect_areas.clear();
    prog.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        args: [1.5, 0.0, 0.0, 0.0],
    }));
    let aoe_targets = state.spatial().within_radius(&state, center, 1.5);
    assert_eq!(
        aoe_targets.len(),
        3,
        "wide circle must capture all three (sorted by AgentId): got {aoe_targets:?}"
    );
    let events = apply_program_aoe(
        &prog,
        ids[0],
        ids[1],
        &aoe_targets,
        0,
        0xCAFE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    assert_eq!(
        events.len(),
        3,
        "wide circle → 3 chronicle records from 1 cast (≥2 satisfies #121 task contract)"
    );
    // P11: events in `aoe_targets` order — sorted by AgentId ascending.
    for (i, expected_target) in aoe_targets.iter().enumerate() {
        match &events[i] {
            ApplyEvent::Damage { source, target, amount } => {
                assert_eq!(
                    *source, ids[0],
                    "event[{i}] source must be caster ids[0]={:?}",
                    ids[0]
                );
                assert_eq!(
                    *target, *expected_target,
                    "event[{i}] target must match aoe_targets[{i}]"
                );
                assert_eq!(*amount, 30.0, "event[{i}] amount must be 30.0");
            }
            other => panic!("expected Damage, got {other:?}"),
        }
    }

    // Edge circle around ids[0]: hits ids[0] (d=0) and ids[1] (d=1)
    // but NOT ids[2] (d=2). The "2+ records" shape — a multi-agent
    // AOE that doesn't include the entire row.
    let edge_center = state.agent_pos(ids[0]).expect("ids[0] alive");
    let aoe_targets = state.spatial().within_radius(&state, edge_center, 1.5);
    assert_eq!(
        aoe_targets,
        vec![ids[0], ids[1]],
        "edge circle must hit ids[0] and ids[1] only"
    );
    let events = apply_program_aoe(
        &prog,
        ids[0],
        ids[1],
        &aoe_targets,
        0,
        0xCAFE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    assert_eq!(events.len(), 2, "edge circle → 2 chronicle records");
    assert!(matches!(events[0], ApplyEvent::Damage { target, .. } if target == ids[0]));
    assert!(matches!(events[1], ApplyEvent::Damage { target, .. } if target == ids[1]));
}

#[test]
fn aoe_strike_shape_in_4_agent_row_drops_hp_on_each_target() {
    // 4 agents in a row at x = 0..4. Caster = ids[0]. Cast center =
    // ids[1] (x=1). Circle radius = 1.5 → ids[0] (d=1), ids[1] (d=0),
    // ids[2] (d=1) ; ids[3] (d=2) is OUT.
    //
    // Drives an end-to-end behavioral assertion: each in-circle agent's
    // hp drops after the apply path's `Damage` events are drained into
    // SoA mutations (hp -= 30). This pins the "hp drops on multiple
    // agents" leg of the task's behavioral contract.
    let mut state = SimState::new(8, 0xCAFE);
    let ids = spawn_row(&mut state, 4);
    assert_eq!(ids.len(), 4);

    let mut prog = AbilityProgram::new_single_target(
        5.0,
        Gate {
            cooldown_ticks: 10,
            hostile_only: false,
            line_of_sight: false,
        },
        [EffectOp::Damage { amount: 30.0 }],
    );
    prog.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Circle,
        args: [1.5, 0.0, 0.0, 0.0],
    }));

    let center = state.agent_pos(ids[1]).expect("ids[1] alive");
    let aoe_targets = state.spatial().within_radius(&state, center, 1.5);
    assert_eq!(
        aoe_targets,
        vec![ids[0], ids[1], ids[2]],
        "circle around ids[1] r=1.5 must hit ids[0..=2] (sorted ascending)"
    );

    // Snapshot starting hp so we can verify the drop.
    let hp_before: Vec<f32> = ids.iter().map(|id| state.agent_hp(*id).unwrap()).collect();
    assert!(
        hp_before.iter().all(|h| (*h - 100.0).abs() < f32::EPSILON),
        "all agents start at full hp (default spawn)"
    );

    // Dispatch the AOE.
    let events = apply_program_aoe(
        &prog,
        ids[0],
        ids[1],
        &aoe_targets,
        0,
        0xCAFE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    assert_eq!(events.len(), 3, "AOE expansion → 3 Damage events");

    // Apply the events to hp SoA — mirrors what an apply-physics
    // handler does (the registry-driven dispatcher's contract is "emit
    // ApplyEvents; sim drains them"). For this E2E the test does the
    // drain inline: `state.set_agent_hp(t, hp - amount)`.
    for ev in &events {
        if let ApplyEvent::Damage { target, amount, .. } = ev {
            let hp = state.agent_hp(*target).unwrap();
            state.set_agent_hp(*target, hp - amount);
        }
    }

    // Verify hp dropped on ids[0..=2] but not on ids[3].
    assert!(
        (state.agent_hp(ids[0]).unwrap() - 70.0).abs() < f32::EPSILON,
        "ids[0] hp must drop 100 → 70 from in-circle Damage"
    );
    assert!(
        (state.agent_hp(ids[1]).unwrap() - 70.0).abs() < f32::EPSILON,
        "ids[1] hp must drop 100 → 70 from in-circle Damage"
    );
    assert!(
        (state.agent_hp(ids[2]).unwrap() - 70.0).abs() < f32::EPSILON,
        "ids[2] hp must drop 100 → 70 from in-circle Damage"
    );
    assert!(
        (state.agent_hp(ids[3]).unwrap() - 100.0).abs() < f32::EPSILON,
        "ids[3] is OUTSIDE the circle — hp must remain at 100"
    );
}

#[test]
fn aoe_cone_hits_three_in_fan_five_agent_fixture() {
    // 5 agents in a fan around the caster (`ids[0]` at origin):
    //   ids[1] at ( 3, -1, 0)  in-cone (≈18.4° below +X axis)
    //   ids[2] at ( 4,  0, 0)  on-axis target (cone facing +X)
    //   ids[3] at ( 3,  1, 0)  in-cone (≈18.4° above +X axis)
    //   ids[4] at ( 1,  5, 0)  out-of-cone (≈78.7° off-axis — outside 60° half-angle)
    //
    // Cone(60°, 5) cast from ids[0] facing ids[2] hits {ids[1], ids[2],
    // ids[3]} but NOT ids[4]. Drives the cone path through the same
    // `apply_program_aoe` dispatcher Circle uses (the user-facing
    // contract: caller pre-filters via
    // `apply_program_aoe_cone_filter`, dispatcher emits one record per
    // target). End-to-end: chronicle records produced for in-cone
    // agents only.
    let mut state = SimState::new(8, 0xCAFE);
    let positions = [
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::new(3.0, -1.0, 0.0),
        glam::Vec3::new(4.0, 0.0, 0.0),
        glam::Vec3::new(3.0, 1.0, 0.0),
        glam::Vec3::new(1.0, 5.0, 0.0),
    ];
    let ids: Vec<AgentId> = positions
        .iter()
        .map(|&pos| {
            state
                .spawn_agent(AgentSpawn { pos, ..Default::default() })
                .expect("agent_cap large enough for 5 agents")
        })
        .collect();
    assert_eq!(ids.len(), 5);

    let mut prog = AbilityProgram::new_single_target(
        5.0,
        Gate {
            cooldown_ticks: 10,
            hostile_only: false,
            line_of_sight: false,
        },
        [EffectOp::Damage { amount: 25.0 }],
    );
    prog.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Cone,
        args: [60.0, 5.0, 0.0, 0.0],
    }));

    // Caller-side cone filter (mirrors the GPU kernel's WGSL math).
    // Use the spatial query as the candidate superset so the test
    // exercises the production-shape integration.
    let apex = state.agent_pos(ids[0]).expect("ids[0] alive");
    let target_pos = state.agent_pos(ids[2]).expect("ids[2] alive");
    let candidates: Vec<(AgentId, glam::Vec3)> = state
        .spatial()
        .within_radius(&state, apex, 5.0)
        .into_iter()
        .map(|id| (id, state.agent_pos(id).expect("alive")))
        .collect();
    let aoe_targets = apply_program_aoe_cone_filter(
        apex,
        target_pos,
        /*half_angle_deg*/ 60.0,
        /*range*/ 5.0,
        &candidates,
    );
    assert_eq!(
        aoe_targets,
        vec![ids[1], ids[2], ids[3]],
        "cone(60°, 5) facing +X must hit ids[1..=3] only (ids[4] outside angle, \
         ids[0] is the caster apex — excluded), got {aoe_targets:?}",
    );

    let events = apply_program_aoe(
        &prog,
        ids[0],
        ids[2],
        &aoe_targets,
        0,
        0xCAFE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    assert_eq!(events.len(), 3, "cone AOE → 3 chronicle records (one per in-cone target)");
    for (i, expected_target) in [ids[1], ids[2], ids[3]].iter().enumerate() {
        match &events[i] {
            ApplyEvent::Damage { source, target, amount } => {
                assert_eq!(*source, ids[0], "event[{i}] source must be caster ids[0]");
                assert_eq!(*target, *expected_target, "event[{i}] target");
                assert_eq!(*amount, 25.0, "event[{i}] amount must be 25.0");
            }
            other => panic!("expected Damage, got {other:?}"),
        }
    }
}

#[test]
fn aoe_box_hits_agents_within_aabb_extents_six_agent_3d_scatter() {
    // 6 agents scattered in 3D around the origin. Cast center =
    // ids[0]'s position (origin). Box half-extents (wx=2, wy=2, wz=2)
    // — a 4×4×4 cube. Expected in-box: agents at distances ≤ 2 along
    // each axis. The fixture is a mix of in-box / out-of-box positions
    // covering the wall-inclusive (≤) semantic and per-axis filtering.
    //
    //   ids[0] at (0, 0, 0)     in-box (origin)
    //   ids[1] at (2, 0, 0)     in-box (|x|=2 — wall, ≤ semantic)
    //   ids[2] at (1, 1, 1)     in-box (all three axes < 2)
    //   ids[3] at (0, -2, 0)    in-box (|y|=2 — wall, ≤ semantic)
    //   ids[4] at (3, 0, 0)     out (|x|=3 > 2)
    //   ids[5] at (1, 0, 5)     out (|z|=5 > 2)
    let mut state = SimState::new(8, 0xCAFE);
    let positions = [
        glam::Vec3::new(0.0, 0.0, 0.0),
        glam::Vec3::new(2.0, 0.0, 0.0),
        glam::Vec3::new(1.0, 1.0, 1.0),
        glam::Vec3::new(0.0, -2.0, 0.0),
        glam::Vec3::new(3.0, 0.0, 0.0),
        glam::Vec3::new(1.0, 0.0, 5.0),
    ];
    let ids: Vec<AgentId> = positions
        .iter()
        .map(|&pos| {
            state
                .spawn_agent(AgentSpawn { pos, ..Default::default() })
                .expect("agent_cap large enough for 6 agents")
        })
        .collect();
    assert_eq!(ids.len(), 6);

    let mut prog = AbilityProgram::new_single_target(
        5.0,
        Gate {
            cooldown_ticks: 10,
            hostile_only: false,
            line_of_sight: false,
        },
        [EffectOp::Damage { amount: 20.0 }],
    );
    prog.per_effect_areas.push(Some(EffectAreaShape {
        kind: ShapeKind::Box,
        args: [2.0, 2.0, 2.0, 0.0],
    }));

    // Caller-side box filter — candidates from the spatial query
    // (radius bounded by max extent · √3 ≈ 3.464 to cover the AABB
    // diagonal). The box filter then picks the in-AABB subset.
    let center = state.agent_pos(ids[0]).expect("ids[0] alive");
    let candidates: Vec<(AgentId, glam::Vec3)> = state
        .spatial()
        .within_radius(&state, center, 4.0)
        .into_iter()
        .map(|id| (id, state.agent_pos(id).expect("alive")))
        .collect();
    let aoe_targets = apply_program_aoe_box_filter(
        center, /*wx*/ 2.0, /*wy*/ 2.0, /*wz*/ 2.0, &candidates,
    );
    assert_eq!(
        aoe_targets,
        vec![ids[0], ids[1], ids[2], ids[3]],
        "box(2,2,2) at origin must hit ids[0..=3] only (sorted ascending; \
         ids[4] outside x extent, ids[5] outside z extent); got {aoe_targets:?}",
    );

    let events = apply_program_aoe(
        &prog,
        ids[0],
        ids[0],
        &aoe_targets,
        0,
        0xCAFE,
        &CasterStats::default(),
        &CasterStats::default(),
    );
    assert_eq!(
        events.len(),
        4,
        "box AOE → 4 chronicle records (one per in-box target)"
    );
    for (i, expected_target) in [ids[0], ids[1], ids[2], ids[3]].iter().enumerate() {
        match &events[i] {
            ApplyEvent::Damage { source, target, amount } => {
                assert_eq!(*source, ids[0], "event[{i}] source must be caster ids[0]");
                assert_eq!(*target, *expected_target, "event[{i}] target");
                assert_eq!(*amount, 20.0, "event[{i}] amount must be 20.0");
            }
            other => panic!("expected Damage, got {other:?}"),
        }
    }
}

#[test]
fn aoe_box_filter_zero_extent_collapses_to_axis_equality() {
    // Edge case: zero half-extent on an axis → strict equality semantic
    // on that axis (only candidates with `cand.<axis> == center.<axis>`
    // match). Pin the documented behavior.
    let center = glam::Vec3::new(0.0, 0.0, 0.0);
    let candidates = vec![
        // In-box (z=0 exactly): wx/wy slack admits everything else.
        (AgentId::new(1).unwrap(), glam::Vec3::new(1.0, 1.0, 0.0)),
        (AgentId::new(2).unwrap(), glam::Vec3::new(-1.0, -1.0, 0.0)),
        // Out: any non-zero z drops them from the in-box set.
        (AgentId::new(3).unwrap(), glam::Vec3::new(0.0, 0.0, 0.001)),
        (AgentId::new(4).unwrap(), glam::Vec3::new(0.0, 0.0, -0.001)),
    ];
    let hits = apply_program_aoe_box_filter(
        center, /*wx*/ 5.0, /*wy*/ 5.0, /*wz*/ 0.0, &candidates,
    );
    assert_eq!(
        hits,
        vec![AgentId::new(1).unwrap(), AgentId::new(2).unwrap()],
        "wz=0 collapses z-axis to equality with center.z=0; only z==0 \
         candidates remain in-box; got {hits:?}"
    );
}

#[test]
fn aoe_cone_caster_targets_self_returns_empty_set() {
    // Edge case: caster targets self ⇒ direction = (0,0,0) is
    // degenerate. CPU oracle returns empty; GPU's `dir_len_sq < 1e-6`
    // branch matches by emitting zero records. Pin so the deferred
    // semantic doesn't drift.
    let mut state = SimState::new(8, 0xCAFE);
    let ids = spawn_row(&mut state, 3);

    let apex = state.agent_pos(ids[0]).expect("ids[0] alive");
    let candidates: Vec<(AgentId, glam::Vec3)> = ids
        .iter()
        .map(|&id| (id, state.agent_pos(id).expect("alive")))
        .collect();
    let aoe_targets = apply_program_aoe_cone_filter(
        apex,
        /*target_pos*/ apex, // caster targets self
        60.0,
        5.0,
        &candidates,
    );
    assert!(
        aoe_targets.is_empty(),
        "degenerate cone (caster targets self) → empty in-cone set, got {aoe_targets:?}",
    );
}
