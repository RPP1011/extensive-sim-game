//! Combat Foundation Task 5 — engagement symmetry + range + determinism
//! proptest. Covers acceptance criterion (5) in the plan header.

use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::{Event, EventRing};
use engine::ids::AgentId;
use engine::state::{AgentSpawn, SimState};
use engine_data::config::Config;
use glam::Vec3;
use proptest::prelude::*;

const CREATURES: [CreatureType; 4] = [
    CreatureType::Human,
    CreatureType::Wolf,
    CreatureType::Deer,
    CreatureType::Dragon,
];

fn arb_creature() -> impl Strategy<Value = CreatureType> {
    (0u8..4).prop_map(|i| CREATURES[i as usize])
}

fn arb_pos() -> impl Strategy<Value = (f32, f32, f32)> {
    (-10.0f32..10.0, -10.0f32..10.0, -10.0f32..10.0)
}

fn arb_population() -> impl Strategy<Value = Vec<(CreatureType, (f32, f32, f32))>> {
    prop::collection::vec((arb_creature(), arb_pos()), 3..=10)
}

fn build_state(pop: &[(CreatureType, (f32, f32, f32))]) -> (SimState, Vec<AgentId>) {
    let cap = (pop.len() + 2) as u32;
    let mut state = SimState::new(cap, 42);
    let mut ids = Vec::with_capacity(pop.len());
    for (ct, (x, y, z)) in pop {
        let id = state.spawn_agent(AgentSpawn {
            creature_type: *ct,
            pos: Vec3::new(*x, *y, *z),
            hp: 100.0,
            ..Default::default()
        }).unwrap();
        ids.push(id);
    }
    (state, ids)
}

fn run_tick_start(state: &mut SimState) {
    // Task 139 + Task 143 — engagement recompute lives on the event-driven
    // path. This helper inlines the retired `recompute_all_engagements`
    // shim: emit one synthetic `AgentMoved` at each alive agent's current
    // position so the `engagement_on_move` physics rule (see
    // `assets/sim/physics.sim`) recomputes nearest-hostile pairings via
    // the cascade, then run the registry to fixed point.
    let mut ring = EventRing::with_cap(256);
    drain_initial_engagements(state, &mut ring);
}

/// Emit a synthetic `AgentMoved` at each alive agent's current position,
/// then run the engine cascade to fixed point. This plays the moves that
/// `step_full`'s movement phase would normally emit so the engagement
/// physics rule fires once per agent and converges to steady state.
fn drain_initial_engagements(state: &mut SimState, events: &mut EventRing) {
    let registry = CascadeRegistry::with_engine_builtins();
    let tick = state.tick;
    let alive: Vec<AgentId> = state.agents_alive().collect();
    for id in alive {
        let pos = state.agent_pos(id).unwrap_or(Vec3::ZERO);
        events.push(Event::AgentMoved { actor: id, from: pos, location: pos, tick });
    }
    registry.run_fixed_point(state, events);
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 200,
        max_shrink_iters: 5000,
        .. ProptestConfig::default()
    })]

    /// Property 1: engaged_with is bidirectional — a <-> b iff b <-> a.
    #[test]
    fn engagement_is_bidirectional(pop in arb_population()) {
        let (mut state, ids) = build_state(&pop);
        run_tick_start(&mut state);
        for &a in &ids {
            if let Some(b) = state.agent_engaged_with(a) {
                let back = state.agent_engaged_with(b);
                prop_assert_eq!(
                    back, Some(a),
                    "engaged_with[{:?}]=Some({:?}) but engaged_with[{:?}]={:?}",
                    a, b, b, back
                );
            }
        }
    }

    /// Property 2: no engagement between non-hostile pairs.
    #[test]
    fn engagement_requires_hostility(pop in arb_population()) {
        let (mut state, ids) = build_state(&pop);
        run_tick_start(&mut state);
        for &a in &ids {
            if let Some(b) = state.agent_engaged_with(a) {
                let ct_a = state.agent_creature_type(a).unwrap();
                let ct_b = state.agent_creature_type(b).unwrap();
                prop_assert!(ct_a.is_hostile_to(ct_b),
                    "engagement between non-hostile pair {:?}⟷{:?}", ct_a, ct_b);
            }
        }
    }

    /// Property 3: no engagement at distance > engagement_range.
    #[test]
    fn engagement_respects_range(pop in arb_population()) {
        let (mut state, ids) = build_state(&pop);
        run_tick_start(&mut state);
        let engagement_range = Config::default().combat.engagement_range;
        for &a in &ids {
            if let Some(b) = state.agent_engaged_with(a) {
                let pa = state.agent_pos(a).unwrap();
                let pb = state.agent_pos(b).unwrap();
                let d = pa.distance(pb);
                prop_assert!(d <= engagement_range,
                    "engaged pair at distance {}m exceeds engagement_range={}",
                    d, engagement_range);
            }
        }
    }

    /// Property 4: two SimStates built from the same population produce
    /// byte-identical engaged_with slices.
    #[test]
    fn engagement_is_cross_instance_deterministic(pop in arb_population()) {
        let (mut s1, _) = build_state(&pop);
        let (mut s2, _) = build_state(&pop);
        run_tick_start(&mut s1);
        run_tick_start(&mut s2);
        prop_assert_eq!(
            s1.hot_engaged_with(), s2.hot_engaged_with(),
            "engagement differs between two instances with identical input"
        );
    }

    /// Property 5: perturbation clears out-of-range pairs.
    #[test]
    fn perturbation_clears_out_of_range_pairs(
        pop in arb_population(),
        perturb in (-5.0f32..5.0, -5.0f32..5.0, -5.0f32..5.0),
        target_idx in 0usize..10,
    ) {
        let (mut state, ids) = build_state(&pop);
        run_tick_start(&mut state);

        // Pick an agent to move; if index out of range, skip to last.
        let idx = target_idx.min(ids.len() - 1);
        let victim = ids[idx];

        // If the victim is engaged, move it far enough that the pair exceeds
        // engagement_range. Otherwise, this test vacuously holds.
        if let Some(other) = state.agent_engaged_with(victim) {
            let pv = state.agent_pos(victim).unwrap();
            let po = state.agent_pos(other).unwrap();
            // Move victim to a position we KNOW is > engagement_range from other.
            // Displacement from `other` along the x-axis at distance 3.0.
            let engagement_range = Config::default().combat.engagement_range;
            let new_pos = po + Vec3::new(engagement_range + 1.0 + perturb.0.abs(), 0.0, 0.0);
            state.set_agent_pos(victim, new_pos);
            let new_d = state.agent_pos(victim).unwrap().distance(po);
            prop_assert!(new_d > engagement_range);
            run_tick_start(&mut state);
            // The previously-engaged pair should now be broken (at least on
            // victim's side). It may have re-engaged with a different closer
            // hostile, but the pair `victim<->other` is no longer in force.
            let e_after = state.agent_engaged_with(victim);
            prop_assert_ne!(e_after, Some(other),
                "out-of-range pair still engaged after perturbation: {:?}⟷{:?} distance={}m",
                victim, other, new_d);
            let _ = pv;
        }
    }
}
