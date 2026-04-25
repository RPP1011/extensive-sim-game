//! Proptest baseline — establishes the style for the rest of the proptest
//! suite. Any proptest file in `crates/engine/tests/` should start with
//! `proptest_` so `cargo test -p engine proptest_` runs the whole set.
use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch};
use glam::Vec3;
use proptest::prelude::*;

fn run_engine(seed: u64, n_agents: u32, ticks: u32) {
    let cap = n_agents + 4;
    let mut state = SimState::new(cap, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1_000_000);
    let cascade = CascadeRegistry::<Event>::new();
    for i in 0..n_agents {
        let angle = (i as f32 / (n_agents.max(1) as f32)) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 200,              // engine step is expensive; 200 is plenty for baseline
        max_shrink_iters: 1000,
        .. ProptestConfig::default()
    })]

    /// For any random `(seed, n_agents, ticks)` in the supported range,
    /// `step_full` via the `step` convenience wrapper does not panic. Catches:
    /// arithmetic overflow in shuffle keying; divide-by-zero in
    /// `fraction_true`; slice-bounds bugs in mask construction under very
    /// small or near-capacity agent counts.
    #[test]
    fn step_never_panics_under_random_sizing(
        seed in any::<u64>(),
        n_agents in 1u32..=20,
        ticks in 1u32..=50,
    ) {
        run_engine(seed, n_agents, ticks);
    }
}
