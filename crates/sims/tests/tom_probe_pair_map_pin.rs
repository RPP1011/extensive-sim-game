//! Mega-crate-hosted port of the Wave 3 ToM Phase 1 `pair_map`
//! storage-sizing pin (formerly
//! `crates/tom_probe_runtime/tests/pair_map_behavioral_pin.rs`).
//!
//! ## What this exercises
//!
//! The `view beliefs_flags(observer, subject) -> u32` materialised
//! view in `assets/sim/tom_probe.sim` uses `storage = pair_map`,
//! which composes a flat `agent_count × agent_count × u32` cell
//! array at slot `[observer * agent_count + subject]`. The folded
//! accumulator is a bit-OR (`self |= b`).
//!
//! ### Sizing scope (this pin) vs. behavioural pin
//!
//! The original per-fixture-runtime test asserted the FULL FIRE shape
//! (one tick → diagonal lights up to bit 0, off-diagonals stay zero)
//! against `sim.beliefs_flags()`. That worked because
//! `tom_probe_runtime`'s hand-rolled `step()`:
//!   * runs `physics_WhatIBelieve` BEFORE `fold_beliefs_flags`, and
//!   * writes `second_key_pop = agent_count` into the fold's cfg
//!     uniform so the `[observer * second_key_pop + subject]` index
//!     matches the host-side N² layout.
//!
//! The auto-emitted `sims::tom_probe::GeneratedRuntime` shares neither
//! property today — its `step()` runs the schedule in compiler-default
//! order (fold first, physics second) and writes a single shared cfg
//! constant of `1u32` into every kernel's `slot 2`. Both gaps are
//! tracked separately from this unit's pair-map sizing fix.
//!
//! What this pin DOES exercise: the auto-emitted
//! `view_storage_primary_buf` is sized for the full pair-map shape
//! (`agent_count * agent_count` u32 cells = `4 * agent_count^2`
//! bytes). Pre-fix the buffer was sized as `agent_count * 4` bytes —
//! a 4× under-allocation at `N = 4` that silently corrupted memory
//! when ANY read or write touched a cell past the per-agent prefix.
//! The seed/readback round-trip below proves the buffer accepts
//! writes across all N² cells WITHOUT a wgpu validation panic and
//! WITHOUT bleed across cells.
//!
//! Skip path: GPU adapter unavailable (CI minus `--all-targets`,
//! sandbox without a wgpu adapter). The constructor returns `None` on
//! a missing adapter; we log a skip message and exit 0 in that case.

#![allow(non_snake_case)]

mod tom_probe_helpers;
use tom_probe_helpers as h;

const SEED: u64 = 0xC0FFEE_BAD_BEEF;
const N: u32 = 4;

#[test]
fn view_storage_primary_holds_n_squared_cells() {
    let rt = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_pair_map_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // The readback helper requests `cell_count(rt) = N*N` u32 words.
    // Pre-fix this would panic at the COPY_SRC step (the buffer was
    // 16 bytes for N=4 = 4 words; readback asked for 64 bytes = 16
    // words). Post-fix the buffer is sized `agent_count^2 * 4` so the
    // copy succeeds and we get N² zeros back from a freshly-allocated
    // buffer.
    let beliefs = h::read_view_storage_primary(&rt);
    let n = N as usize;
    assert_eq!(
        beliefs.len(),
        n * n,
        "view_storage_primary must hold agent_count^2 = {} u32 cells",
        n * n,
    );
    for (i, v) in beliefs.iter().enumerate() {
        assert_eq!(
            *v, 0,
            "freshly-allocated view_storage_primary cell {i} should \
             be 0u (zero-init)",
        );
    }
}

#[test]
fn view_storage_primary_round_trips_per_pair_cells() {
    let rt = match h::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[tom_probe_pair_map_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // Seed every cell with a distinct sentinel encoding (observer,
    // subject), then read back. Pre-fix the upper rows (observer >=
    // 1) would either fail to write (out-of-bounds) or wrap into
    // earlier rows, so the readback would not match the seeded
    // pattern. Post-fix every cell is independently addressable.
    let n = N as usize;
    let cells = n * n;
    let seeded: Vec<u32> = (0..cells)
        .map(|i| {
            let observer = (i / n) as u32;
            let subject = (i % n) as u32;
            (observer << 16) | (subject & 0xFFFF)
        })
        .collect();

    // Per-view storage post the aliasing-gap fix — `beliefs_flags`
    // is the pair-keyed view in this fixture; backing buffer is
    // `view_storage_beliefs_flags_primary_buf`.
    rt.gpu.queue.write_buffer(
        &rt.view_storage_beliefs_flags_primary_buf,
        0,
        bytemuck::cast_slice(&seeded),
    );

    let read_back = h::read_view_storage_primary(&rt);
    assert_eq!(read_back.len(), cells);
    for observer in 0..n {
        for subject in 0..n {
            let idx = observer * n + subject;
            let expected = ((observer as u32) << 16) | (subject as u32 & 0xFFFF);
            assert_eq!(
                read_back[idx], expected,
                "view_storage_primary[{observer} * N + {subject}] \
                 round-trip mismatch (got 0x{:08x}, expected 0x{:08x})",
                read_back[idx], expected,
            );
        }
    }
}
