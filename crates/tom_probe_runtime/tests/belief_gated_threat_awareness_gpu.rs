//! GPU pin: the `beliefs_flags` storage layer (per-(observer, subject)
//! u32 bitset) faithfully carries a per-observer "knows source is
//! busy" bit, and that bit drives divergent per-observer behavior
//! when consumed through the same belief-gate predicate the future
//! GPU threats fold will use.
//!
//! Companion to `dsl_compiler/tests/threats_belief_gated_fold.rs`
//! (host-side semantic spec). That test models the fold predicate
//! on host; this one proves the GPU storage layer the predicate will
//! READ FROM works correctly: write divergent flags to the GPU,
//! read them back, run the same belief-gate the host spec uses, and
//! confirm the per-observer behavioral divergence happens against
//! real GPU bytes.
//!
//! When the compiler swap to belief-gated fold lands at
//! `crates/dsl_compiler/src/cg/emit/kernel.rs:2645`, the divergence
//! this test asserts moves from "host-evaluated against GPU bytes"
//! to "GPU-evaluated end-to-end" — the test stays valid as the
//! behavioural specification, the gate just runs on GPU instead of
//! being mocked on host.

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xCAFE_FEED_BEEF_F00D;
const N: u32 = 3;

/// Bit position in `beliefs_flags[observer * N + subject]` reserved
/// for "observer believes subject is busy casting".
///
/// Note: the host-side spec at
/// `dsl_compiler/tests/threats_belief_gated_fold.rs` uses bit 0
/// because no other writer competes there. In the live tom_probe
/// runtime, `physics_WhatIBelieve` writes bit 0 at `(self, self)`
/// cells each tick (a per-Knower self-belief stamp); we use bit 7
/// here so the OBSERVED_BUSY bit stays isolated from the runtime's
/// own bookkeeping. The bit POSITION is convention; the predicate
/// shape is identical.
const BELIEF_BIT_OBSERVED_BUSY: u32 = 7;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

/// Mirrors `would_pick_flee` from the host-side spec — Flee scores
/// `threats[obs]`, Idle scores 0.5, argmax wins.
fn would_pick_flee(threats_for_observer: f32) -> bool {
    threats_for_observer > 0.5
}

/// Mock fold gate using the SAME predicate the future GPU emit will
/// use, but evaluated host-side against GPU-stored flags. When the
/// real GPU emit lands, this becomes redundant with the GPU-side
/// evaluation, but stays valid as the spec.
fn run_belief_gated_threats_fold(
    agent_cap: u32,
    beliefs_flags: &[u32],
) -> Vec<f32> {
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
    let mut state = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[belief_gated_gpu_pin] skipping: no wgpu adapter on host.");
            return;
        }
    };

    // Slot layout: caster=0, observer A=1, observer B=2.
    //
    // Plant the BUSY bit on (A → caster); leave (B → caster) clear.
    // This is what `EffectOp::PlantBelief { subject_idx: 0, fact_bit: 0 }`
    // would do via atomic-OR on the (caster=A, subject=0) cell — we
    // skip the chronicle round-trip and seed directly so the test
    // isolates the per-cell GPU storage semantics from the dispatch
    // chain.
    let cap = (N * N) as usize;
    let mut seed = vec![0u32; cap];
    let bit = 1u32 << BELIEF_BIT_OBSERVED_BUSY;
    seed[(1 * N + 0) as usize] = bit; // A knows caster is busy

    state.seed_beliefs_flags(&seed);

    // Run a step so the GPU's view is settled (the seed write is a
    // queue.write_buffer; the next step ensures any pending writes
    // are flushed and the readback reflects the post-step state).
    state.step();

    // Read the flags back from GPU.
    let live = state.beliefs_flags().to_vec();

    // Pin: GPU storage faithfully carries the per-cell divergence.
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
    // Note: cell (caster=0, caster=0) may have OTHER bits set by the
    // runtime's `physics_WhatIBelieve` self-stamp; we only assert
    // that OUR bit (BELIEF_BIT_OBSERVED_BUSY=7) is unset here, not
    // that the cell is zero.
    assert_eq!(
        live[(0 * N + 0) as usize] & bit, 0,
        "GPU-stored cell (caster=0, caster=0): OBSERVED_BUSY bit must be clear (cell = 0x{:08X})",
        live[(0 * N + 0) as usize]
    );

    // Drive the future-gated fold against the GPU bytes. When the
    // compiler swap lands, this `run_belief_gated_threats_fold` host
    // call becomes unnecessary — the GPU's PerAgentEventScan kernel
    // will compute the same vector. Until then, the test demonstrates
    // the architecture works against real GPU storage.
    let threats = run_belief_gated_threats_fold(N, &live);

    assert_eq!(
        threats[1], 1.0,
        "observer A: GPU-stored belief drives threats[1] = 1.0 (one believed-busy source)"
    );
    assert_eq!(
        threats[2], 0.0,
        "observer B: GPU-stored belief absence drives threats[2] = 0.0"
    );

    // Behavioural divergence: same physical world (caster IS the
    // busy source), same belief storage layer, but the per-observer
    // view of that storage produces different scoring choices.
    assert!(
        would_pick_flee(threats[1]),
        "observer A's GPU-stored belief drives Flee"
    );
    assert!(
        !would_pick_flee(threats[2]),
        "observer B's GPU-stored belief absence drives Idle"
    );

    eprintln!(
        "[belief_gated_gpu_pin] divergence proven against GPU storage: \
         threats = {threats:?}, A→Flee, B→Idle"
    );
}
