//! Wave 3 ToM Phase 2 behavioral pin on the multi-field BeliefState
//! SoA storage shape.
//!
//! ## What this exercises
//!
//! Phase 2 introduced 5 additional per-(observer, target) storage
//! columns alongside the Phase 1 `flags` bit-OR slot:
//!
//! | Column     | Type      | Bytes per cell |
//! |------------|-----------|----------------|
//! | flags      | u32       | 4              |
//! | last_known_pos      | vec4 f32  | 16             |
//! | last_known_creature_type | u8        | 1              |
//! | last_seen_tick      | u32       | 4              |
//! | confidence | u8 (q8)   | 1              |
//! | suspicion  | u8 (q8)   | 1              |
//!
//! All 6 columns share the `observer * agent_count + target`
//! indexing convention (the existing `pair_map` storage hint shape).
//! Phase 2 ships the storage + readback accessors only — there are
//! no new chronicle event kinds and no new writer verbs (those land
//! in Phase 3 alongside the spec's `agents.beliefs(o, s).<field>`
//! struct-then-field DSL access syntax).
//!
//! This test pre-seeds each column with a distinguishable per-cell
//! pattern via the `seed_beliefs_<field>()` test-only setters,
//! reads back via the matching accessors, and asserts:
//!
//!   1. **Per-cell isolation across observers.** Observer A's
//!      beliefs about target T do not bleed into observer B's
//!      beliefs about target T. The seed pattern uses
//!      `observer_idx * 100 + target_idx` so a row-major bleed
//!      shows up as an off-by-one neighbouring value.
//!   2. **Per-cell isolation across targets.** Same observer's
//!      beliefs about target T1 do not bleed into target T2.
//!   3. **Per-column isolation.** Setting the `flags` column does
//!      not corrupt the `pos` / `type` / `tick` / `confidence` /
//!      `suspicion` columns and vice-versa. Each column's readback
//!      matches exactly the bytes the test seeded into it.
//!   4. **Round-trip identity.** What goes in via
//!      `seed_beliefs_<field>()` comes back out via
//!      `beliefs_<field>()` — no truncation or sign-extension when
//!      the host slice writes into the GPU buffer and the staging
//!      readback reverses the trip.
//!
//! ## Skip path
//!
//! GPU adapter unavailable (CI minus `--all-targets`, sandbox
//! without a wgpu adapter). The constructor panics on a missing
//! adapter; we wrap in `catch_unwind` and log a skip message so the
//! test passes when the runtime simply can't spin up a device —
//! same convention as `pair_map_behavioral_pin.rs`.

use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const N: u32 = 4;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

/// Per-cell expected `flags` value: pattern `(observer << 16) |
/// target` so a row vs column swap is visible in the high vs low
/// bits.
fn expected_flags(observer: usize, target: usize) -> u32 {
    ((observer as u32) << 16) | (target as u32)
}

/// Per-cell expected `pos` vec4: x = observer, y = target, z =
/// observer*target, w = 1.0 (sentinel; rules out half-zeroed
/// reads).
fn expected_pos(observer: usize, target: usize) -> [f32; 4] {
    [
        observer as f32,
        target as f32,
        (observer * target) as f32,
        1.0,
    ]
}

/// Per-cell expected `type`: `(observer * 7 + target) & 0xFF`.
/// Multiplier coprime with N keeps neighbour-cell collisions away.
fn expected_type(observer: usize, target: usize) -> u8 {
    ((observer * 7 + target) & 0xFF) as u8
}

/// Per-cell expected `tick`: `100 + observer * 1000 + target`.
fn expected_tick(observer: usize, target: usize) -> u32 {
    (100 + observer * 1000 + target) as u32
}

/// Per-cell expected `confidence`: `((observer * 11 + target * 3) &
/// 0xFF)`.
fn expected_confidence(observer: usize, target: usize) -> u8 {
    ((observer * 11 + target * 3) & 0xFF) as u8
}

/// Per-cell expected `suspicion`: `((observer * 13 + target * 5) &
/// 0xFF)`.
fn expected_suspicion(observer: usize, target: usize) -> u8 {
    ((observer * 13 + target * 5) & 0xFF) as u8
}

#[test]
fn belief_state_soa_round_trips_all_columns() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[belief_state_soa_pin] skipping: no wgpu adapter");
            return;
        }
    };
    let n = N as usize;
    let cells = n * n;

    // Build per-column seed vectors with distinguishable patterns.
    let flags: Vec<u32> = (0..cells)
        .map(|i| expected_flags(i / n, i % n))
        .collect();
    let pos: Vec<[f32; 4]> = (0..cells)
        .map(|i| expected_pos(i / n, i % n))
        .collect();
    let creature_types: Vec<u8> = (0..cells)
        .map(|i| expected_type(i / n, i % n))
        .collect();
    let ticks: Vec<u32> = (0..cells)
        .map(|i| expected_tick(i / n, i % n))
        .collect();
    let confidence: Vec<u8> = (0..cells)
        .map(|i| expected_confidence(i / n, i % n))
        .collect();
    let suspicion: Vec<u8> = (0..cells)
        .map(|i| expected_suspicion(i / n, i % n))
        .collect();

    // Pre-seed every column.
    sim.seed_beliefs_flags(&flags);
    sim.seed_beliefs_pos(&pos);
    sim.seed_beliefs_type(&creature_types);
    sim.seed_beliefs_tick(&ticks);
    sim.seed_beliefs_confidence(&confidence);
    sim.seed_beliefs_suspicion(&suspicion);

    // Round-trip via the readback accessors. Each accessor surfaces
    // its column independently of the others — Phase 2's per-column
    // dirty flag + paired (primary, staging) buffers guarantee no
    // shared-buffer aliasing across columns.
    let read_flags = sim.beliefs_flags().to_vec();
    let read_pos = sim.beliefs_pos().to_vec();
    let read_type = sim.beliefs_type().to_vec();
    let read_tick = sim.beliefs_tick().to_vec();
    let read_confidence = sim.beliefs_confidence().to_vec();
    let read_suspicion = sim.beliefs_suspicion().to_vec();

    assert_eq!(
        read_flags.len(),
        cells,
        "flags column must surface agent_count^2 = {cells} cells",
    );
    assert_eq!(
        read_pos.len(),
        cells,
        "pos column must surface agent_count^2 = {cells} cells",
    );
    assert_eq!(
        read_type.len(),
        cells,
        "type column must surface agent_count^2 = {cells} cells",
    );
    assert_eq!(
        read_tick.len(),
        cells,
        "tick column must surface agent_count^2 = {cells} cells",
    );
    assert_eq!(
        read_confidence.len(),
        cells,
        "confidence column must surface agent_count^2 = {cells} cells",
    );
    assert_eq!(
        read_suspicion.len(),
        cells,
        "suspicion column must surface agent_count^2 = {cells} cells",
    );

    // Per-cell round-trip + isolation assertions. A single
    // mismatched cell on any column proves either (a) row-vs-column
    // swap in the GPU index, (b) cross-column buffer aliasing, or
    // (c) staging-buffer truncation — all P11 / determinism
    // failures.
    for observer in 0..n {
        for target in 0..n {
            let idx = observer * n + target;

            assert_eq!(
                read_flags[idx],
                expected_flags(observer, target),
                "flags[{observer}, {target}] = {} (expected {})",
                read_flags[idx],
                expected_flags(observer, target),
            );

            let want_pos = expected_pos(observer, target);
            assert_eq!(
                read_pos[idx], want_pos,
                "pos[{observer}, {target}] = {:?} (expected {:?})",
                read_pos[idx], want_pos,
            );

            assert_eq!(
                read_type[idx],
                expected_type(observer, target),
                "type[{observer}, {target}] = {} (expected {})",
                read_type[idx],
                expected_type(observer, target),
            );

            assert_eq!(
                read_tick[idx],
                expected_tick(observer, target),
                "tick[{observer}, {target}] = {} (expected {})",
                read_tick[idx],
                expected_tick(observer, target),
            );

            assert_eq!(
                read_confidence[idx],
                expected_confidence(observer, target),
                "confidence[{observer}, {target}] = {} (expected {})",
                read_confidence[idx],
                expected_confidence(observer, target),
            );

            assert_eq!(
                read_suspicion[idx],
                expected_suspicion(observer, target),
                "suspicion[{observer}, {target}] = {} (expected {})",
                read_suspicion[idx],
                expected_suspicion(observer, target),
            );
        }
    }
}

#[test]
fn belief_state_soa_per_observer_isolation() {
    // Even tighter focus: only seed observer 0's row of every
    // column, leave the rest at the default 0. Then assert
    // observer 0 has the seeded values AND every other observer's
    // cells stayed at 0 — no cross-observer bleed.
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[belief_state_soa_pin] skipping: no wgpu adapter");
            return;
        }
    };
    let n = N as usize;
    let cells = n * n;

    // Build a "observer-0-only" seed: cells 0..n carry the marker
    // pattern; cells n..cells stay at 0.
    let mut flags = vec![0u32; cells];
    let mut pos = vec![[0.0f32; 4]; cells];
    let mut creature_types = vec![0u8; cells];
    let mut ticks = vec![0u32; cells];
    let mut confidence = vec![0u8; cells];
    let mut suspicion = vec![0u8; cells];

    for target in 0..n {
        let idx = target; // observer 0
        flags[idx] = 0xDEAD_0000 | (target as u32);
        pos[idx] = [99.0, target as f32, -1.0, 7.0];
        creature_types[idx] = 0xA0 | (target as u8 & 0x0F);
        ticks[idx] = 5000 + target as u32;
        confidence[idx] = 0xC0 | (target as u8 & 0x0F);
        suspicion[idx] = 0x50 | (target as u8 & 0x0F);
    }

    sim.seed_beliefs_flags(&flags);
    sim.seed_beliefs_pos(&pos);
    sim.seed_beliefs_type(&creature_types);
    sim.seed_beliefs_tick(&ticks);
    sim.seed_beliefs_confidence(&confidence);
    sim.seed_beliefs_suspicion(&suspicion);

    let read_flags = sim.beliefs_flags().to_vec();
    let read_pos = sim.beliefs_pos().to_vec();
    let read_type = sim.beliefs_type().to_vec();
    let read_tick = sim.beliefs_tick().to_vec();
    let read_confidence = sim.beliefs_confidence().to_vec();
    let read_suspicion = sim.beliefs_suspicion().to_vec();

    // Observer 0's row: must match the seeded marker pattern.
    for target in 0..n {
        let idx = target;
        assert_eq!(
            read_flags[idx],
            0xDEAD_0000 | (target as u32),
            "observer 0's flags[target={target}] mismatch",
        );
        assert_eq!(
            read_pos[idx],
            [99.0, target as f32, -1.0, 7.0],
            "observer 0's pos[target={target}] mismatch",
        );
        assert_eq!(
            read_type[idx],
            0xA0 | (target as u8 & 0x0F),
            "observer 0's type[target={target}] mismatch",
        );
        assert_eq!(
            read_tick[idx],
            5000 + target as u32,
            "observer 0's tick[target={target}] mismatch",
        );
        assert_eq!(
            read_confidence[idx],
            0xC0 | (target as u8 & 0x0F),
            "observer 0's confidence[target={target}] mismatch",
        );
        assert_eq!(
            read_suspicion[idx],
            0x50 | (target as u8 & 0x0F),
            "observer 0's suspicion[target={target}] mismatch",
        );
    }

    // Every other observer's cells: must stay at the all-zero
    // default. Any non-zero readback proves the observer-keyed
    // index is leaking writes into a neighbour row (i.e. the
    // `observer * agent_count + target` indexing got confused with
    // `target * agent_count + observer`, or the seed slice was
    // shifted by an off-by-one).
    for observer in 1..n {
        for target in 0..n {
            let idx = observer * n + target;
            assert_eq!(
                read_flags[idx], 0,
                "observer {observer}'s flags[target={target}] bled non-zero (= {}); \
                 expected 0 (only observer 0 was seeded)",
                read_flags[idx],
            );
            assert_eq!(
                read_pos[idx], [0.0; 4],
                "observer {observer}'s pos[target={target}] bled non-zero (= {:?})",
                read_pos[idx],
            );
            assert_eq!(
                read_type[idx], 0,
                "observer {observer}'s type[target={target}] bled non-zero (= {})",
                read_type[idx],
            );
            assert_eq!(
                read_tick[idx], 0,
                "observer {observer}'s tick[target={target}] bled non-zero (= {})",
                read_tick[idx],
            );
            assert_eq!(
                read_confidence[idx], 0,
                "observer {observer}'s confidence[target={target}] bled non-zero (= {})",
                read_confidence[idx],
            );
            assert_eq!(
                read_suspicion[idx], 0,
                "observer {observer}'s suspicion[target={target}] bled non-zero (= {})",
                read_suspicion[idx],
            );
        }
    }
}

#[test]
fn belief_state_soa_per_column_independence() {
    // Cross-column independence pin: seed ONLY the `tick` column;
    // all other columns must surface as zero. Catches a bug where
    // a single buffer is shared across columns (one column's writes
    // would corrupt every other column's readback).
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[belief_state_soa_pin] skipping: no wgpu adapter");
            return;
        }
    };
    let n = N as usize;
    let cells = n * n;

    let only_ticks: Vec<u32> = (0..cells).map(|i| 1000 + i as u32).collect();
    sim.seed_beliefs_tick(&only_ticks);

    let read_flags = sim.beliefs_flags().to_vec();
    let read_pos = sim.beliefs_pos().to_vec();
    let read_type = sim.beliefs_type().to_vec();
    let read_tick = sim.beliefs_tick().to_vec();
    let read_confidence = sim.beliefs_confidence().to_vec();
    let read_suspicion = sim.beliefs_suspicion().to_vec();

    for i in 0..cells {
        assert_eq!(
            read_flags[i], 0,
            "flags column corrupted by tick seed at cell {i}",
        );
        assert_eq!(
            read_pos[i], [0.0; 4],
            "pos column corrupted by tick seed at cell {i}",
        );
        assert_eq!(
            read_type[i], 0,
            "type column corrupted by tick seed at cell {i}",
        );
        assert_eq!(
            read_tick[i],
            1000 + i as u32,
            "tick column round-trip mismatch at cell {i}",
        );
        assert_eq!(
            read_confidence[i], 0,
            "confidence column corrupted by tick seed at cell {i}",
        );
        assert_eq!(
            read_suspicion[i], 0,
            "suspicion column corrupted by tick seed at cell {i}",
        );
    }
}
