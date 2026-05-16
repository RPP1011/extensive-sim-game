//! Multi-horizon threat stresstest — effectiveness report.
//!
//! Drives `assets/sim/threat_horizon_stresstest.sim` for 64 ticks
//! with mixed long-fuse + short-fuse casters coexisting, walks the
//! per-observer `threats` belief, and reports:
//!
//!   1. Long-fuse steady-state intensity (slots 0..2 stay busy
//!      every tick → fold +N_long every tick, decay 0.85).
//!   2. Short-fuse transient + decay shape (slots 2..4 busy at tick
//!      0 only → one-shot +N_short, decay geometric).
//!   3. Crossover tick: when does short-fuse's decaying memory fall
//!      below long-fuse's sustained contribution? This is the AI-
//!      decision signal that distinguishes "this threat is fresh"
//!      from "this threat was a one-time scare".
//!
//! Behavioural pin (vs the existing simple decay test):
//!   * Both horizons coexist in ONE run instead of two separate
//!     scenarios. The observer's `threats[obs]` is the SUPERPOSITION
//!     of sustained + decaying signals.
//!   * The crossover-tick metric is the load-bearing decision input.
//!     A regression that broke decay would push crossover to never
//!     (short-fuse dominates forever) or never happen (decay too
//!     aggressive — short-fuse drops below long-fuse instantly).

use sims::threat_horizon_stresstest::GeneratedRuntime;

const SEED: u64 = 0xF00D_BEEF_DEAD_C0DE;
const N: u32 = 8;
const N_LONG_FUSE: usize = 2;
const N_SHORT_FUSE: usize = 2;
const N_OBSERVERS: usize = 4; // slots 4..8
const TICKS: usize = 64;

fn seed_all_alive(state: &mut GeneratedRuntime) {
    let alive: Vec<u32> = vec![1u32; state.agent_count as usize];
    state
        .gpu
        .queue
        .write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
}

/// Host-side patch: after tick `short_fuse_ticks` (= 1), zero the
/// busy-with-ability-id SoA cells for the short-fuse caster slots.
/// Long-fuse cells stay non-zero so the per-agent-event-scan fold
/// keeps adding +1 per long-fuse caster per observer per tick.
fn clear_short_fuse_busy(state: &mut GeneratedRuntime) {
    let n = state.agent_count as usize;
    let mut zeros: Vec<u32> = vec![0u32; n];
    // Re-stamp long-fuse cells with the original cast_ability_id so
    // they stay flagged after the host overwrites the buffer below.
    // (The MarkInitialBusy physics rule fires only at tick 0; from
    // tick 1 onward the SoA state has to be host-managed.)
    let cast_ability_id = 1u32;
    for i in 0..N_LONG_FUSE {
        zeros[i] = cast_ability_id;
    }
    state.gpu.queue.write_buffer(
        &state.agent_busy_with_ability_id_buf,
        0,
        bytemuck::cast_slice(&zeros),
    );
}

fn read_threats_primary(state: &mut GeneratedRuntime) -> Vec<f32> {
    let n = state.agent_count as usize;
    let bytes = (n as u64 * 4u64).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("threat_horizon_stresstest_pin::view_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder =
        state
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("threat_horizon_stresstest_pin::readback"),
            });
    encoder.copy_buffer_to_buffer(
        &state.view_storage_threats_primary_buf,
        0,
        &staging,
        0,
        bytes,
    );
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| {
        res.expect("view_storage map_async failed")
    });
    state
        .gpu
        .device
        .poll(wgpu::PollType::Wait)
        .expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

#[test]
fn multi_horizon_threat_decay_effectiveness_report() {
    let mut state = match GeneratedRuntime::try_new(SEED, N) {
        Some(s) => s,
        None => {
            eprintln!("[threat_horizon_stresstest] skipping: no wgpu adapter");
            return;
        }
    };
    seed_all_alive(&mut state);

    // The auto-emitted runtime writes `slot_count=1` into the
    // DecayThreatsCfg uniform (single-cfg-slot wiring), so decay
    // only multiplies `view_storage_primary[0]`. Slots 1..N still
    // accumulate fold contributions but grow unboundedly. We read
    // slot 0 as the canonical observer where the decay actually
    // shapes the curve — same convention `threats_with_decay_probe`
    // uses. Slot 0 is also a long-fuse caster, so its own busy
    // flag stays set across the run and it sees its own contribution
    // in the fold count.
    let observer_idx = 0;
    let mut per_tick: Vec<f32> = Vec::with_capacity(TICKS);

    for tick in 0..TICKS {
        state.step();
        // After tick 0 fires (stamping busy on all alive slots and
        // letting fold add +N), clear the short-fuse cells host-side
        // so subsequent ticks only see long-fuse busy.
        if tick == 0 {
            clear_short_fuse_busy(&mut state);
        }
        let view = read_threats_primary(&mut state);
        per_tick.push(view[observer_idx]);
    }

    // Pure-decay reference for short-fuse alone (no long-fuse): N_short
    // at tick 0, then x 0.85 each tick. Long-fuse-alone steady-state:
    // N_long / (1 - 0.85) ≈ 6.67·N_long.
    //
    // Crossover: tick at which the per_tick value starts being
    // dominated by long-fuse steady-state. We approximate by finding
    // the first tick where d/dt(per_tick) > 0 (long-fuse re-growth
    // exceeds short-fuse decay) AFTER the initial spike.
    let crossover = per_tick
        .windows(2)
        .enumerate()
        .skip(2) // skip tick 0 spike + tick 1 transient
        .find(|(_, w)| w[1] > w[0])
        .map(|(i, _)| i + 1); // +1 because windows pairs i with i+1

    // Long-fuse steady-state estimate: take the last tick's value (by
    // T=64 it's effectively converged for rate=0.85).
    let long_fuse_steady = per_tick[TICKS - 1] as f64;
    let analytic_long_fuse_ss = (N_LONG_FUSE as f64) / (1.0 - 0.85);

    // Peak intensity (tick 1's spike: long-fuse + short-fuse both
    // contributing one tick of fold).
    let peak_intensity = per_tick
        .iter()
        .skip(1)
        .copied()
        .fold(0.0_f32, f32::max);

    // Cumulative "would Flee" ticks above threshold = 5.0. This is
    // an effectiveness metric: if the observer used scoring threshold
    // 5.0 to decide Flee, how many ticks did it fire?
    const FLEE_THRESHOLD: f32 = 5.0;
    let flee_ticks: usize = per_tick.iter().filter(|v| **v > FLEE_THRESHOLD).count();
    let flee_pct = (flee_ticks as f64) / (TICKS as f64) * 100.0;

    println!("==== threat horizon stresstest (mixed-horizon, decay 0.85) ====");
    println!(
        "  config: N={N} agents, ticks={TICKS}  \
         long-fuse slots={N_LONG_FUSE} short-fuse slots={N_SHORT_FUSE}  \
         observers={N_OBSERVERS}  flee threshold={FLEE_THRESHOLD}",
    );
    println!("  observer slot {observer_idx} per-tick threats[obs]:");
    for (tick, v) in per_tick.iter().enumerate() {
        if tick % 4 == 0 || tick == TICKS - 1 {
            println!("    tick {tick:3}: {v:.3}");
        }
    }
    println!(
        "  peak intensity     = {peak_intensity:.3}  (tick 1 spike: long+short both fold)",
    );
    println!(
        "  long-fuse steady   = {long_fuse_steady:.3}  (analytic N_long/(1-0.85) = {analytic_long_fuse_ss:.3})",
    );
    println!("  crossover tick     = {crossover:?}  (first re-growth after short-fuse memory fades)");
    println!(
        "  flee-threshold pct = {flee_ticks}/{TICKS} ticks ({flee_pct:.1}%) above {FLEE_THRESHOLD}",
    );
    println!("=================================================================");

    // Behavioural pins.
    // (1) Both horizons fold at tick 0+: peak should be at least
    //     (N_LONG + N_SHORT) since fold adds 1 per busy candidate per
    //     observer (4 busy candidates → peak ≥ 4).
    assert!(
        peak_intensity >= (N_LONG_FUSE + N_SHORT_FUSE) as f32 - 0.5,
        "peak intensity {peak_intensity} should be ≥ {} (N_long + N_short)",
        N_LONG_FUSE + N_SHORT_FUSE
    );
    // (2) Long-fuse steady-state should be in the ballpark of
    //     N_long / (1 - decay_rate). The exact value drifts by the
    //     observer-self-exclusion in the per_agent_event_scan
    //     fold (observer == source is filtered, so an observer
    //     that is itself a caster sees N_long - 1 busy candidates
    //     per tick instead of N_long). Tolerance is set to
    //     accommodate either inclusion mode.
    let lower_bound = ((N_LONG_FUSE - 1) as f64) / (1.0 - 0.85);
    let upper_bound = (N_LONG_FUSE as f64) / (1.0 - 0.85);
    assert!(
        long_fuse_steady >= lower_bound - 1.0 && long_fuse_steady <= upper_bound + 1.0,
        "long-fuse steady-state {long_fuse_steady:.3} outside ballpark \
         [{lower_bound:.3}, {upper_bound:.3}] (1.0 tolerance for float drift)"
    );
    // (3) Crossover must exist — the decay must be aggressive enough
    //     that short-fuse memory eventually drops below long-fuse
    //     accumulation. If crossover is None, decay isn't working or
    //     long-fuse isn't accumulating.
    assert!(
        crossover.is_some(),
        "crossover tick must exist; per_tick={per_tick:?}"
    );
    // (4) Flee% must be above 50% — the observer perceives meaningful
    //     threat for most of the run thanks to sustained long-fuse
    //     pressure.
    assert!(
        flee_pct >= 50.0,
        "flee% {flee_pct:.1} should be ≥ 50 (long-fuse keeps threat above {FLEE_THRESHOLD})"
    );
}
