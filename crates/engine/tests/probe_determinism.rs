//! Plan 3, Task 10 — probe determinism. Two runs with identical seed + spawn
//! produce bit-identical event replay hashes.

use engine::cascade::CascadeRegistry;
use engine_data::entities::CreatureType;
use engine::event::EventRing;
use engine_data::events::Event;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::{step, SimScratch}; // Plan B1' Task 11: step is unimplemented!() stub
use glam::Vec3;

fn hash_of_run(seed: u64, ticks: u32, ring_cap: usize) -> [u8; 32] {
    let mut state = SimState::new(64, seed);
    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::<Event>::with_cap(ring_cap);
    let cascade = CascadeRegistry::<Event>::new();
    // Ring of 16 agents in a 20m-radius circle at z=10.
    for i in 0..16 {
        let angle = (i as f32 / 16.0) * std::f32::consts::TAU;
        state.spawn_agent(AgentSpawn {
            creature_type: CreatureType::Human,
            pos: Vec3::new(20.0 * angle.cos(), 20.0 * angle.sin(), 10.0),
            hp: 100.0,
            max_hp: 100.0,
        });
    }
    for _ in 0..ticks {
        step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);
    }
    events.replayable_sha256()
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn same_seed_same_hash() {
    assert_eq!(hash_of_run(42, 200, 8192), hash_of_run(42, 200, 8192));
}

    #[ignore] // Re-enable after B1' Task 11 emits engine_rules::step::step.
#[test]
fn different_seed_different_hash() {
    // Not strictly required (two seeds could in principle collide) but very
    // unlikely for a 200-tick 16-agent run. Useful sanity that seed is
    // actually threaded through.
    let h1 = hash_of_run(1, 200, 8192);
    let h2 = hash_of_run(2, 200, 8192);
    assert_ne!(h1, h2, "seeds 1 and 2 produced identical event hashes");
}
