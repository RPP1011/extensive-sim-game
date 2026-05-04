//! ToM probe harness — drives `tom_probe_runtime` for N ticks and
//! asserts the per-(observer, subject) `beliefs` storage matches the
//! seed bit pattern produced by `physics_WhatIBelieve`.
//!
//! ## Expected (FULL FIRE)
//!
//! With `agent_count = 32`, `ticks = 100`, every alive Knower emits
//! one `BeliefAcquired { observer: self, subject: self, fact_bit: 1 }`
//! per tick. The view's `self |= b` accumulator collapses every
//! emission into the same diagonal cell, so after the FIRST tick:
//!
//!   - `beliefs[i * N + i] == 1u` for every i in 0..N (diagonal).
//!   - `beliefs[i * N + j] == 0u` for every i != j (off-diagonal).
//!
//! Subsequent ticks are idempotent (OR-ing a set bit is a no-op).
//! The probe runs for TICKS=100 anyway to exercise the per-tick
//! producer + fold dispatch chain end-to-end.
//!
//! ## OUTCOME classification
//!
//! - **(a) FULL FIRE** — diagonal == 1 everywhere AND off-diagonal
//!   == 0 everywhere. The belief read/write path is end-to-end.
//! - **(b) NO FIRE** — diagonal stayed 0; producer or fold kernel
//!   dropped at compile time. (See git history for the pre-fix
//!   discovery shape that surfaced this outcome.)
//! - **(c) PARTIAL FIRE** — anything else; fold landed but the bit
//!   pattern doesn't match the producer's emit. Indicates a
//!   second_key_pop / event-field-offset mismatch between the
//!   compiler's emit and the runtime's cfg uniform.

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xCAFE_BABE_DEAD_BEEF;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;

fn main() {
    let mut sim = TomProbeState::new(SEED, AGENT_COUNT);
    println!(
        "tom_probe_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let beliefs = sim.beliefs().to_vec();
    let n = AGENT_COUNT as usize;
    println!(
        "tom_probe_app: finished — final tick={} agents={} beliefs.len()={}",
        sim.tick(),
        sim.agent_count(),
        beliefs.len(),
    );
    assert_eq!(
        beliefs.len(),
        n * n,
        "beliefs must be sized agent_count^2 = {}",
        n * n,
    );

    // Diagonal: every (i, i) cell must equal 1u (the producer's
    // `fact_bit = 1` OR'd into every (self, self) slot every tick).
    let mut diagonal_ok = 0usize;
    let mut diagonal_bad: Vec<(usize, u32)> = Vec::new();
    for i in 0..n {
        let v = beliefs[i * n + i];
        if v == 1 {
            diagonal_ok += 1;
        } else {
            diagonal_bad.push((i, v));
        }
    }

    // Off-diagonal: every (i, j != i) cell must stay 0u (no producer
    // ever emits `BeliefAcquired { observer: i, subject: j }` for i
    // != j).
    let mut offdiag_ok = 0usize;
    let mut offdiag_bad: Vec<(usize, usize, u32)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = beliefs[i * n + j];
            if v == 0 {
                offdiag_ok += 1;
            } else {
                offdiag_bad.push((i, j, v));
            }
        }
    }

    println!(
        "tom_probe_app: diagonal — {}/{} slots == 1u",
        diagonal_ok, n,
    );
    println!(
        "tom_probe_app: off-diagonal — {}/{} slots == 0u",
        offdiag_ok,
        n * (n - 1),
    );
    if !diagonal_bad.is_empty() {
        println!(
            "tom_probe_app: diagonal mismatches (first 8): {:?}",
            &diagonal_bad[..diagonal_bad.len().min(8)],
        );
    }
    if !offdiag_bad.is_empty() {
        println!(
            "tom_probe_app: off-diagonal mismatches (first 8): {:?}",
            &offdiag_bad[..offdiag_bad.len().min(8)],
        );
    }

    let diagonal_total = beliefs.iter().filter(|&&v| v != 0).count();
    if diagonal_ok == n && offdiag_ok == n * (n - 1) {
        println!(
            "tom_probe_app: OUTCOME = (a) FULL FIRE — diagonal == 1u, \
             off-diagonal == 0u; ToM belief read/write path lights up \
             end-to-end (atomicOr fold body, pair_map storage, u32 \
             view return type, |= self-update operator)",
        );
    } else if diagonal_total == 0 {
        println!(
            "tom_probe_app: OUTCOME = (b) NO FIRE — every cell stayed 0u; \
             producer rule or fold kernel dropped at compile time. See \
             docs/superpowers/notes/2026-05-04-tom-probe.md for the \
             original gap-chain discovery doc.",
        );
        std::process::exit(1);
    } else {
        println!(
            "tom_probe_app: OUTCOME = (c) PARTIAL FIRE — diagonal_ok={} \
             offdiag_ok={} (total nonzero = {}). Likely cfg mismatch \
             (second_key_pop or event-field offsets) between compiler \
             emit and runtime cfg uniform.",
            diagonal_ok, offdiag_ok, diagonal_total,
        );
        std::process::exit(1);
    }
}
