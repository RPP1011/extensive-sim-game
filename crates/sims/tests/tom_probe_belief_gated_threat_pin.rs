//! Mega-crate-hosted port of the belief-gated threat-awareness GPU pin
//! (formerly `crates/tom_probe_runtime/tests/belief_gated_threat_awareness_gpu.rs`).
//!
//! Confirms that per-cell flags written into `beliefs_flags_buf` round-
//! trip through GPU storage and drive divergent host-side fold-gate
//! decisions per observer. The fold predicate is mocked here against the
//! GPU bytes; once the compiler swap to GPU-side belief-gated fold lands
//! at `crates/dsl_compiler/src/cg/emit/kernel.rs:2645`, the host
//! evaluation becomes redundant — the test stays valid as the spec.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 3;

const BELIEF_BIT_OBSERVED_BUSY: u32 = 7;

fn would_pick_flee(threats_for_observer: f32) -> bool {
    threats_for_observer > 0.5
}

fn run_belief_gated_threats_fold(agent_cap: u32, beliefs_flags: &[u32]) -> Vec<f32> {
    let mut out: Vec<f32> = vec![0.0; agent_cap as usize];
    for observer in 0..agent_cap {
        for source in 0..agent_cap {
            let cell = beliefs_flags[(observer * agent_cap + source) as usize];
            let bit = 1u32 << BELIEF_BIT_OBSERVED_BUSY;
            if (cell & bit) == 0 {
                continue;
            }
            out[observer as usize] += 1.0;
        }
    }
    out
}

#[test]
fn belief_flag_writes_to_gpu_storage_drive_per_observer_threat_divergence() {
    let mut state = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_belief_gated_threat_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let cap = (N * N) as usize;
    let mut seed = vec![0u32; cap];
    let bit = 1u32 << BELIEF_BIT_OBSERVED_BUSY;
    seed[(1 * N + 0) as usize] = bit;

    h::seed_beliefs_flags(&mut state, &seed);

    state.step();

    let live = h::read_beliefs_flags(&state);

    assert_eq!(
        live[(1 * N + 0) as usize] & bit, bit,
        "GPU-stored cell (A=1, caster=0) must carry the BUSY bit (cell = 0x{:08X})",
        live[(1 * N + 0) as usize]
    );
    assert_eq!(
        live[(2 * N + 0) as usize] & bit, 0,
        "GPU-stored cell (B=2, caster=0) must NOT carry the bit (cell = 0x{:08X})",
        live[(2 * N + 0) as usize]
    );
    assert_eq!(
        live[(0 * N + 0) as usize] & bit, 0,
        "GPU-stored cell (caster=0, caster=0): OBSERVED_BUSY bit must be clear (cell = 0x{:08X})",
        live[(0 * N + 0) as usize]
    );

    let threats = run_belief_gated_threats_fold(N, &live);

    assert_eq!(
        threats[1], 1.0,
        "observer A: GPU-stored belief drives threats[1] = 1.0",
    );
    assert_eq!(
        threats[2], 0.0,
        "observer B: GPU-stored belief absence drives threats[2] = 0.0",
    );

    assert!(would_pick_flee(threats[1]), "observer A picks Flee");
    assert!(!would_pick_flee(threats[2]), "observer B picks Idle");

    eprintln!(
        "[tom_probe_belief_gated_threat_pin] divergence proven against GPU storage: \
         threats = {threats:?}, A→Flee, B→Idle"
    );
}
