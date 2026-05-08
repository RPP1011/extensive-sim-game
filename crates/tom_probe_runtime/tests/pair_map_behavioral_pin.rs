//! Wave 3 ToM Phase 1 behavioral pin on the existing `pair_map`-storage
//! belief view.
//!
//! ## What this exercises
//!
//! The `view beliefs(observer, subject) -> u32` materialised view in
//! `assets/sim/tom_probe.sim` uses `storage = pair_map`, which composes
//! a flat `agent_count × agent_count × u32` cell array at slot
//! `[observer * agent_count + subject]`. The folded accumulator is a
//! bit-OR (`self |= b`), implemented in WGSL as native `atomicOr` on
//! the primary storage buffer (no CAS retry — bit-OR is commutative
//! and associative, P11 trivial).
//!
//! Pre-fix (commit 51b5853b shipped the basic mechanism), the
//! `tom_probe_app` binary asserts the FULL FIRE shape end-to-end. This
//! test pins the *behaviour* via a smaller fixture inside
//! `cargo test`'s harness so a regression in the fold path surfaces in
//! the schema/physics CI gate rather than only in the binary.
//!
//! ## Fixture
//!
//! 4 agents, run for 1 tick (the `physics_WhatIBelieve` rule emits one
//! `BeliefAcquired { observer: self, subject: self, fact_bit: 1 }`
//! per alive agent per tick — the fold body OR's `fact_bit` into the
//! `[observer, subject]` cell). The agent SoA's `agent_alive` field
//! starts all-1, so all 4 producers fire.
//!
//! ## Asserts
//!
//! - **Diagonal isolation** — each observer's `(self, self)` cell
//!   acquires bit 0 (= `fact_bit = 1`); no cell other than the
//!   diagonal is touched.
//! - **No bleed across observers** — `pair_map[A * N + B]` for
//!   `A != B` stays 0 even though A is alive and producing events.
//!   This is the critical isolation property: the fold's `[observer *
//!   N + subject]` index composition correctly routes each producer's
//!   record to its own (self, self) cell, NOT into a neighbour's row.
//! - **Idempotence** — running a second tick leaves the diagonal at
//!   bit 0 (OR-ing a set bit is a no-op), and off-diagonals stay 0.
//!
//! Skip path: GPU adapter unavailable (CI minus `--all-targets`,
//! sandbox without a wgpu adapter). The constructor panics on a
//! missing adapter; we guard the call with a `catch_unwind` and log
//! a skip message so the test passes when the runtime simply can't
//! spin up a device.

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xC0FFEE_BAD_BEEF;
const N: u32 = 4;

/// Construct a `TomProbeState` if the host has a wgpu adapter; return
/// `None` otherwise so the test logs a skip message and exits 0. The
/// constructor uses `expect("init wgpu adapter + device")` internally;
/// we wrap in `catch_unwind` to convert that into a skip rather than a
/// hard failure on a CI runner without GPU.
fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

#[test]
fn pair_map_diagonal_lights_up_after_one_tick() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[pair_map_behavioral_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // One tick — every alive Knower emits BeliefAcquired { observer:
    // self, subject: self, fact_bit: 1 } and the fold body OR's it into
    // the diagonal cell.
    sim.step();

    let beliefs = sim.beliefs().to_vec();
    let n = N as usize;
    assert_eq!(
        beliefs.len(),
        n * n,
        "pair_map sized agent_count^2 = {}",
        n * n,
    );

    // Diagonal: every (i, i) cell must equal 1u after one tick.
    for i in 0..n {
        let v = beliefs[i * n + i];
        assert_eq!(
            v, 1,
            "diagonal cell pair_map[{i} * N + {i}] = {v} (expected 1u after one tick)",
        );
    }

    // Isolation: every off-diagonal cell must stay 0u — no BeliefAcquired
    // event in the fixture has (observer=i, subject=j) for i != j, so
    // the [observer * N + subject] index composition must NOT bleed any
    // bits into neighbouring rows / columns.
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = beliefs[i * n + j];
            assert_eq!(
                v, 0,
                "off-diagonal cell pair_map[{i} * N + {j}] = {v} (expected 0u — no observer→subject event for i != j)",
            );
        }
    }
}

#[test]
fn pair_map_second_tick_is_idempotent() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[pair_map_behavioral_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // Two ticks — the second tick OR's bit 0 again into a cell that
    // already holds bit 0. The fold body's `self |= b` is idempotent
    // on a bit that's already set, so the diagonal must STILL equal 1u
    // and off-diagonals must STILL be 0u.
    sim.step();
    sim.step();

    let beliefs = sim.beliefs().to_vec();
    let n = N as usize;

    for i in 0..n {
        assert_eq!(
            beliefs[i * n + i],
            1,
            "diagonal cell pair_map[{i} * N + {i}] must remain 1u under bit-OR idempotence",
        );
        for j in 0..n {
            if i == j {
                continue;
            }
            assert_eq!(
                beliefs[i * n + j],
                0,
                "off-diagonal cell pair_map[{i} * N + {j}] must stay 0u after two ticks",
            );
        }
    }
}
