//! Unit test for the GPU exclusive prefix-scan kernel in spatial_gpu.
//! Replaces the CPU exclusive-scan at spatial_gpu.rs:917-926.

#![cfg(feature = "gpu")]

use engine_gpu::spatial_gpu::{GpuSpatialHash, GRID_CELLS};

fn cpu_exclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; input.len()];
    let mut running: u32 = 0;
    for i in 0..input.len() {
        out[i] = running;
        running = running.saturating_add(input[i]);
    }
    out
}

#[test]
fn prefix_scan_matches_cpu_reference() {
    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");

    // Deterministic pseudo-random counts.
    let mut counts = vec![0u32; GRID_CELLS as usize];
    let mut s: u32 = 0x1234_5678;
    for c in counts.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *c = s % 64;
    }

    let expected = cpu_exclusive_scan(&counts);
    let actual = hash.run_scan_for_test(&counts).expect("scan dispatch");

    assert_eq!(actual, expected, "GPU scan must equal CPU exclusive-scan");
}

#[test]
fn prefix_scan_all_zeros() {
    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");
    let counts = vec![0u32; GRID_CELLS as usize];
    let actual = hash.run_scan_for_test(&counts).expect("scan dispatch");
    assert!(actual.iter().all(|&x| x == 0));
}

#[test]
fn prefix_scan_saturates() {
    // If total > u32::MAX, cpu_exclusive_scan saturates; GPU must too.
    let mut hash = GpuSpatialHash::new_for_test().expect("spatial init");
    let mut counts = vec![0u32; GRID_CELLS as usize];
    counts[0] = u32::MAX;
    counts[1] = 10;
    let expected = cpu_exclusive_scan(&counts);
    let actual = hash.run_scan_for_test(&counts).expect("scan dispatch");
    assert_eq!(actual, expected);
}
