use engine::cascade::CascadeRegistry;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::policy::UtilityBackend;
use engine::state::{SimState, AgentSpawn};
use engine_data::entities::CreatureType;
use engine::step::{step, SimScratch}; // Plan B1' Task 11: step is unimplemented!() stub
use glam::Vec3;

fn run(seed: u64, n_agents: u32, ticks: u32) -> [u8; 32] {
    let mut state = SimState::new(n_agents + 10, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(1_000_000);
    let cascade = CascadeRegistry::<Event>::new();
    for i in 0..n_agents {
        let angle = (i as f32 / n_agents as f32) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(50.0 * angle.cos(), 50.0 * angle.sin(), 10.0),
            hp: 100.0,
            ..Default::default()
        });
    }
    for _ in 0..ticks { step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade); }
    events.replayable_sha256()
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn hundred_agents_thousand_ticks_deterministic() {
    let h1 = run(42, 100, 1000);
    let h2 = run(42, 100, 1000);
    assert_eq!(h1, h2, "same seed → same replayable trace hash");
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn different_seeds_diverge_under_load() {
    let h1 = run(42, 100, 1000);
    let h2 = run(43, 100, 1000);
    assert_ne!(h1, h2, "different seeds → different trace hashes (via position differences propagating through events)");
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn small_and_large_agent_counts_both_deterministic() {
    let small_1 = run(42, 10, 500);
    let small_2 = run(42, 10, 500);
    assert_eq!(small_1, small_2);
    let large_1 = run(42, 100, 500);
    let large_2 = run(42, 100, 500);
    assert_eq!(large_1, large_2);
    // Different N should produce different hashes (more events).
    assert_ne!(small_1, large_1);
}
