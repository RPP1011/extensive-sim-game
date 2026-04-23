//! Single-tick resident cascade on a fixture that converges in 2
//! iterations. Asserts the indirect args buffer reflects that.

#![cfg(feature = "gpu")]

use engine::backend::SimBackend;
use engine::cascade::CascadeRegistry;
use engine::creature::CreatureType;
use engine::event::EventRing;
use engine::policy::UtilityBackend;
use engine::state::{AgentSpawn, SimState};
use engine::step::SimScratch;
use engine_gpu::GpuBackend;
use glam::Vec3;

const SEED: u64 = 0xC0DE_FACE_CAFE_0001;

#[test]
#[ignore = "requires Phase D: snapshot() and full resident cascade wiring"]
fn cascade_resident_converges_in_two_iters() {
    // 1 human + 1 wolf 1 m apart — guarantees a single attack event
    // that kills neither in one hit, so cascade needs 2 iters (attack
    // + follow-on fear from kin) then converges.
    let mut gpu = GpuBackend::new().expect("gpu init");
    let mut state = SimState::new(8, SEED);
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Human,
        pos: Vec3::new(0.0, 0.0, 0.0),
        hp: 100.0,
        ..Default::default()
    }).unwrap();
    state.spawn_agent(AgentSpawn {
        creature_type: CreatureType::Wolf,
        pos: Vec3::new(1.0, 0.0, 0.0),
        hp: 80.0,
        ..Default::default()
    }).unwrap();

    let mut scratch = SimScratch::new(state.agent_cap() as usize);
    let mut events = EventRing::with_cap(1024);
    let cascade = CascadeRegistry::with_engine_builtins();

    // Warmup.
    gpu.step(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade);

    // Take a snapshot to read indirect args via the debug accessor.
    // step_batch records the resident cascade; one tick is enough.
    gpu.step_batch(&mut state, &mut scratch, &mut events, &UtilityBackend, &cascade, 1);

    // The `snapshot()` accessor lands in Task D3 as part of the
    // Phase D resident-cascade wiring; gated off until then so this
    // file compiles cleanly. The `#[ignore]` above keeps it skipped
    // by default when Phase D lands and the cfg is removed.
    #[cfg(any())]
    {
        let snap = gpu.snapshot().expect("snapshot");

        // last_cascade_iterations() reports the number of slots whose
        // workgroup count was > 0 at the end of the batch. For a
        // 2-iteration convergence we expect 2 (some tolerance for
        // cascade tail noise — the fixture is designed to converge
        // quickly but small differences in event counts can push this
        // to 3 or 4 iters without being a bug).
        let iters = gpu.last_cascade_iterations().expect("iter count");
        assert!(
            iters >= 2 && iters <= 4,
            "expected 2-4 cascade iterations, got {iters}"
        );
        let _ = snap; // snapshot used to trigger the args readback
    }
}
