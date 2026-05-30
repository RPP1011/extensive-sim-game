//! edgeworld Phase 0 behavioral pin. Phase 0 = hunger + food +
//! forage/eat/starve/regrow. This file grows task-by-task; Task 1
//! only asserts the fixture compiles and the runtime constructs.

use sims::edgeworld::GeneratedRuntime;

const SEED: u64 = 0xED6E_0001;
const N_TOTAL: u32 = 4;

#[test]
fn edgeworld_runtime_constructs() {
    let state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[edgeworld] skipping: no wgpu adapter on host.");
            return;
        }
    };
    // Constructing + dropping the runtime is the Task 1 assertion:
    // the fixture compiled and the GPU pipeline built.
    drop(state);
}
