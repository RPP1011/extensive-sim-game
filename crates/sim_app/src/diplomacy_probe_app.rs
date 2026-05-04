//! Diplomacy probe harness — drives `diplomacy_probe_runtime` for N
//! ticks and reports the three observables:
//!   - `trust` (pair_map u32) diagonal — should converge to 1u
//!   - `alliances_proposed` (f32 per agent) — should be ≈34.0
//!   - `betrayals_committed` (f32 per agent) — should be ≈66.0
//!
//! ## Predicted observables
//!
//! ### (a) FULL FIRE
//!
//! With AGENT_COUNT=32, TICKS=100, observation_tick_mod=3:
//!
//!   - trust[i*N + i] = 1u for every i in 0..N (diagonal converged)
//!   - trust[i*N + j] = 0u for every i != j (off-diagonal stays at 0
//!     — placeholder self-routing, no spatial broadcast yet)
//!   - alliances_proposed[i] = 34.0 (ticks 0,3,...,99 → 34 fires)
//!   - betrayals_committed[i] = 66.0 (TICKS - 34 = 66)
//!
//! ### (b) Likely gap shapes
//!
//!   - 3 distinct event kinds in one ring: per-handler tag filter
//!     scaling test — if a fold accumulates wrong totals (e.g.,
//!     alliances_proposed[i] = 100 because the tag-filter rejected
//!     nothing), the per-kind partition is broken.
//!   - Mod-tick mask predicates: if both folds ≈ TICKS, the mask gates
//!     never engaged; if both ≈ 0, the mask gates rejected everything.
//!   - pair_map u32 + 2 f32 views in one program: if trust diagonal
//!     stays 0u while the f32 folds work, the mixed-storage path
//!     mis-routed the bit-OR fold's primary buffer.
//!
//! Discovery doc: `docs/superpowers/notes/2026-05-04-diplomacy_probe.md`.

use diplomacy_probe_runtime::DiplomacyProbeState;
use engine::CompiledSim;

const SEED: u64 = 0xD1_1051A_C0A11710;
const AGENT_COUNT: u32 = 32;
const TICKS: u64 = 100;
const OBSERVATION_TICK_MOD: u64 = 3;

fn main() {
    let mut sim = DiplomacyProbeState::new(SEED, AGENT_COUNT);
    println!(
        "diplomacy_probe_app: starting — seed=0x{:016X} agents={} ticks={}",
        SEED, AGENT_COUNT, TICKS,
    );

    for _ in 0..TICKS {
        sim.step();
    }

    let trust = sim.trust().to_vec();
    let alliances = sim.alliances_proposed().to_vec();
    let betrayals = sim.betrayals_committed().to_vec();

    println!(
        "diplomacy_probe_app: finished — final tick={} agents={} trust.len()={} alliances.len()={} betrayals.len()={}",
        sim.tick(),
        sim.agent_count(),
        trust.len(),
        alliances.len(),
        betrayals.len(),
    );

    let n = AGENT_COUNT as usize;

    // -- Trust diagonal check --
    let mut diag_set = 0usize;
    let mut offdiag_set = 0usize;
    for i in 0..n {
        for j in 0..n {
            let v = trust[i * n + j];
            if v != 0 {
                if i == j {
                    diag_set += 1;
                } else {
                    offdiag_set += 1;
                }
            }
        }
    }
    println!(
        "diplomacy_probe_app: trust — diagonal-set: {}/{} ; off-diagonal-set: {}/{}",
        diag_set, n, offdiag_set, n * (n - 1),
    );

    // -- Per-Diplomat alliance + betrayal counters --
    let expected_alliances =
        ((TICKS + OBSERVATION_TICK_MOD - 1) / OBSERVATION_TICK_MOD) as f32; // ceil(TICKS/MOD)
    let expected_betrayals = (TICKS as f32) - expected_alliances;

    let (a_min, a_mean, a_max, a_sum, a_zero) = stats(&alliances);
    let (b_min, b_mean, b_max, b_sum, b_zero) = stats(&betrayals);
    println!(
        "diplomacy_probe_app: alliances_proposed — min={:.3} mean={:.3} max={:.3} sum={:.3} zero_slots={}",
        a_min, a_mean, a_max, a_sum, a_zero,
    );
    println!(
        "diplomacy_probe_app: betrayals_committed — min={:.3} mean={:.3} max={:.3} sum={:.3} zero_slots={}",
        b_min, b_mean, b_max, b_sum, b_zero,
    );
    println!(
        "diplomacy_probe_app: expected per-slot — alliances={:.3} betrayals={:.3} (sum={:.3})",
        expected_alliances,
        expected_betrayals,
        expected_alliances + expected_betrayals,
    );

    let trust_ok = diag_set == n && offdiag_set == 0;
    let alliances_ok =
        (a_min - expected_alliances).abs() < 0.5 && (a_max - expected_alliances).abs() < 0.5;
    let betrayals_ok =
        (b_min - expected_betrayals).abs() < 0.5 && (b_max - expected_betrayals).abs() < 0.5;

    if trust_ok && alliances_ok && betrayals_ok {
        println!(
            "diplomacy_probe_app: OUTCOME = (a) FULL FIRE — all three observables match analytical predictions.\n  \
             - pair_map u32 view fold (trust) lights up diagonal cleanly\n  \
             - verb cascade with TWO competing verbs gated on disjoint Mod-tick predicates routes correctly\n  \
             - per-handler tag filter scales to THREE distinct event kinds in one ring\n  \
             - 2 Group entity declarations + 1 Agent entity + 3 views (mixed u32 + 2 f32) compile + dispatch cleanly",
        );
    } else {
        println!(
            "diplomacy_probe_app: OUTCOME = (b) PARTIAL/NO FIRE — gaps surfaced:\n  \
             - trust diag: {}/{} (expected {}); off-diag: {} (expected 0)\n  \
             - alliances_proposed [{:.3}, {:.3}] (expected {:.3} per slot)\n  \
             - betrayals_committed [{:.3}, {:.3}] (expected {:.3} per slot)",
            diag_set, n, n,
            offdiag_set,
            a_min, a_max, expected_alliances,
            b_min, b_max, expected_betrayals,
        );
    }

    // Preview first 8 slots so the failure mode is visible at a glance.
    let alliances_preview: Vec<f32> = alliances.iter().take(8).copied().collect();
    let betrayals_preview: Vec<f32> = betrayals.iter().take(8).copied().collect();
    println!(
        "diplomacy_probe_app: preview alliances_proposed[0..8] = {:?}\n  \
         preview betrayals_committed[0..8] = {:?}",
        alliances_preview, betrayals_preview,
    );
    let trust_diag_preview: Vec<u32> =
        (0..8.min(n)).map(|i| trust[i * n + i]).collect();
    println!(
        "diplomacy_probe_app: preview trust[i*N + i] for i in 0..8 = {:?}",
        trust_diag_preview,
    );
}

fn stats(v: &[f32]) -> (f32, f32, f32, f32, usize) {
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut zero = 0usize;
    for &x in v {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
        sum += x;
        if x == 0.0 {
            zero += 1;
        }
    }
    let mean = if v.is_empty() { 0.0 } else { sum / v.len() as f32 };
    (min, mean, max, sum, zero)
}
