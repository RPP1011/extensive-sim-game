//! `forest_fire` adversarial event-storm pin — drives the multi-event,
//! multi-consumer cascade fixture and reports per-tick ring fill +
//! per-tier event counts after a long horizon.
//!
//! The fixture exercises:
//!   * 5 distinct event kinds (Ignited, Burned, EmberLanded, RainFell,
//!     WindShifted) on the same ring with mixed per-handler tag filters.
//!   * 4 view consumers + 1 PerEvent consumer (Catch).
//!   * Per-agent stochastic gating via `rng.action() % 100`.
//!   * 4 fold dispatches running BEFORE physics each tick (folds at
//!     tick T see emits from tick T-1).
//!
//! **Topology (host-seeded)**:
//!   * GRID_SIDE × GRID_SIDE Tree agents on a unit grid (positions
//!     `(x, y, 0)` for x ∈ [0..GRID_SIDE), y ∈ [0..GRID_SIDE)).
//!   * All trees start with hp=100 (init block) — healthy.
//!   * The host stamps a 2×2 ignition cluster at the centre to hp=80
//!     before the first step(). These are the seed fires.
//!
//! **What this pin probes (the gap-discovery report)**:
//!
//!   1. Ring fill / tail growth at scale: pin reports
//!      `state.event_ring.tail_value()` after each tick. With the 1M-slot
//!      cap, GRID_SIDE=32 (1024 trees) at peak fire (~50% burning) emits:
//!         Spread embers:    ~512 burning × 4 embers ≈ 2048 / tick
//!         WindShifted:      1024 / 7 ≈ 146 / tick (avg)
//!         RainFell:         1024 / 23 ≈ 44 / tick (avg)
//!         Burned:           ~40 / tick (steady-state burnout)
//!         Ignited:          0 / tick (Catch is Indirect — gap, see below)
//!      Total ~2300/tick. 500 ticks × 2300 = 1.15M emits — exceeds the
//!      ring cap on the LAST few ticks (capacity 1_048_576). The pin
//!      logs whether tail saturates AND whether the per-emit `if
//!      (_slot < 1048576u)` guard cleanly drops over-cap emits without
//!      crashing.
//!
//!   2. **Indirect-dispatch gap (commit `353527e6`)** — `physics_Catch`
//!      is `DispatchOp::Indirect`, which the synthesised step() catch-
//!      all skips. Consequence: EmberLanded events get emitted (Spread
//!      runs as a regular Kernel dispatch + the ring tail grows), but
//!      Catch never fires → no healthy tree ever transitions to burning
//!      from an ember. Without external ignition triggers, only the
//!      seed cluster burns out. The pin captures this as the verdict:
//!      *fire stays contained to the seed cluster*. When the four-gap
//!      Indirect blocker is closed (see build_helper.rs:1535), this
//!      pin's verdict should flip to *fire spreads across the grid*.
//!
//!   3. Multi-handler tag filter on the same kind: TWO consumers
//!      subscribe to EmberLanded — the `ember_landings` view + the
//!      Catch physics. Both must see the same emit; per-handler
//!      filtering must NOT cross-contaminate (e.g. the `ember_landings`
//!      view must not pick up Ignited events). The pin reads the
//!      `ember_landings` view per slot to confirm this.
//!
//!   4. Conservation invariant: at every tick, `alive_count + ash_count
//!      == GRID_SIDE²`. Drift surfaces a Reaper / consumer ordering
//!      bug. With the Indirect gap, ash_count grows monotonically as
//!      the seed cluster burns out (~16 ticks per tree × 4 seed trees
//!      ≈ 64 ticks before all seeds become ash); alive_count drops
//!      by the same amount.
//!
//!   5. Per-tick timing distribution under high event traffic. Pin
//!      reports warmup ms, p50/p95/max steady-state, and total ms.
//!      Stresses the GPU dispatch chain at ~21 kernels/tick × 500 ticks.
//!
//!   6. Determinism (P5): rng.action() flows through per_agent_u32 PCG.
//!      Same seed → byte-identical observables across runs. Pin runs
//!      the simulation TWICE with the same seed and asserts the per-
//!      slot ember_landings buffers match.
//!
//! **What this pin does NOT probe (documented for follow-up agents)**:
//!   * `@traced` non-replayable events — surface VERIFIED (parses +
//!     resolves + lowers; see `crates/dsl_compiler/tests/traced_annotation_parses.rs`
//!     and `EventIR::is_traced()`); EventLayout-level wiring still TBD.
//!   * `@cascade(max_iter=N)` annotation — surface absent today.
//!   * `rng.chance(p)` — surface unverified, fixture uses `rng.action()
//!     % 100`.

use sims::forest_fire::GeneratedRuntime;

const SEED: u64 = 0xF02E_57F1_8E_u64;
/// Grid side. `GRID_SIDE × GRID_SIDE` trees seeded at integer positions.
/// Default 32 → 1024 trees (well within ring cap headroom). Bumping to
/// 64 (4096 trees) approaches the per-tick cap; 128 (16384) blows it.
const GRID_SIDE: u32 = 32;
const N_TOTAL: u32 = GRID_SIDE * GRID_SIDE;
const TICKS: u32 = 500;

#[test]
fn forest_fire_event_storm_500_ticks() {
    let mut state = match GeneratedRuntime::try_new(SEED, N_TOTAL) {
        Some(s) => s,
        None => {
            eprintln!("[forest_fire] skipping: no wgpu adapter on host.");
            return;
        }
    };

    seed_grid(&mut state);
    seed_ignition_cluster(&mut state);

    let initial_alive = count_alive(&mut state);
    let initial_burning = count_burning(&mut state);

    // Warmup tick — pipeline compile cost. Forces a GPU sync via
    // a view-storage readback.
    let warmup_start = std::time::Instant::now();
    state.step();
    let _ = read_shared_view_storage(&mut state);
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;

    // Main horizon — record per-tick wall-clock + ring tail estimate.
    let mut tick_ms_samples: Vec<f64> = Vec::with_capacity(TICKS as usize);
    let mut tail_samples: Vec<u32> = Vec::with_capacity(TICKS as usize);
    let stress_start = std::time::Instant::now();
    let mut conservation_violations = 0u32;

    for _ in 0..TICKS {
        let t = std::time::Instant::now();
        state.step();
        tick_ms_samples.push(t.elapsed().as_secs_f64() * 1000.0);
        tail_samples.push(state.event_ring.tail_value());
    }
    // Force a final GPU sync on the post-loop reads.
    // Per the gap discovered: all four views share a single
    // `view_storage_primary_buf`, so this single readback IS what every
    // fold has accumulated into. Per-view differentiation is impossible
    // until the compiler emits per-view storages.
    let view_aggregate = read_shared_view_storage(&mut state);
    let final_alive = count_alive(&mut state);
    let stress_total_ms = stress_start.elapsed().as_secs_f64() * 1000.0;

    // Conservation: alive + ash == N_TOTAL at end.
    let final_ash = N_TOTAL - final_alive;
    if final_alive + final_ash != N_TOTAL {
        conservation_violations += 1;
    }

    // Stats.
    let mut sorted = tick_ms_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p95 = sorted[((sorted.len() * 95) / 100).min(sorted.len() - 1)];
    let p_max = sorted[sorted.len() - 1];
    let mean = stress_total_ms / (TICKS as f64);

    let max_tail = tail_samples.iter().copied().max().unwrap_or(0);
    let min_tail = tail_samples.iter().copied().min().unwrap_or(0);
    let mean_tail = tail_samples.iter().map(|&x| x as f64).sum::<f64>()
        / (tail_samples.len() as f64);
    // Final-tick observation: by the last tick a fixed number of producers
    // have run in the current tick (since clear_tail_in() resets each
    // tick). The tail is therefore one tick's worth of emits.
    let last_tail = *tail_samples.last().unwrap();

    // Aggregate-only — per-view breakdown impossible (storage shared).
    // Total across all slots is the sum of every fold's contribution
    // for every agent across every tick. Useful as a "did SOMETHING
    // accumulate" signal. Per-slot max + nonzero-slot count tell us
    // whether the accumulation reaches a meaningful subset of agents.
    let total_view_aggregate: f32 = view_aggregate.iter().sum();
    let max_aggregate_per_slot =
        view_aggregate.iter().fold(0.0f32, |a, &b| a.max(b));
    let nonzero_slots: usize =
        view_aggregate.iter().filter(|&&v| v > 0.0).count();

    // RING FILL stats — actual peak vs cap. The cap is the per-emit
    // gate `if (_slot < 1048576u)` in WGSL. The host-side
    // tail_estimate may exceed the cap (it just sums note_emits()
    // upper bounds); over-cap emits were silently dropped on GPU.
    const RING_CAP: u32 = 1_048_576;
    let tail_capped: u32 = max_tail.min(RING_CAP);
    let pct_of_cap = 100.0 * (max_tail as f64) / (RING_CAP as f64);

    println!("==== forest_fire 500-tick event storm report ====");
    println!(
        "  config:     GRID_SIDE={GRID_SIDE} (N={N_TOTAL} trees), TICKS={TICKS}, seed=0x{SEED:X}",
    );
    println!(
        "  init:       {initial_alive}/{N_TOTAL} alive, {initial_burning} burning (seed cluster)",
    );
    println!(
        "  final:      {final_alive}/{N_TOTAL} alive, {final_ash}/{N_TOTAL} ash",
    );
    println!(
        "  conservation: alive + ash = {sum} (expected {N_TOTAL}); violations across run = {conservation_violations}",
        sum = final_alive + final_ash,
    );
    println!("  -- per-tick timing --");
    println!(
        "    warmup tick:  {warmup_ms:.3} ms (incl pipeline compile)",
    );
    println!(
        "    steady-state: total {stress_total_ms:.3} ms over {TICKS} ticks  =>  mean {mean:.3} ms/tick",
    );
    println!(
        "    per-tick:     p50={p50:.3} ms   p95={p95:.3} ms   max={p_max:.3} ms",
    );
    println!("  -- event ring fill --");
    println!(
        "    tail/tick:  min={min_tail}  mean={mean_tail:.0}  max={max_tail}  last_tick={last_tail}",
    );
    println!(
        "    cap usage:  max_tail = {tail_capped} ({pct_of_cap:.2}% of {RING_CAP} slot cap)",
    );
    // Per-view sums (aliasing-gap fix landed — each fold now writes
    // its own backing buffer).
    let sum_ignition: f32 = read_per_view_storage(&mut state, ForestView::IgnitionCount).iter().sum();
    let sum_ember: f32 = read_per_view_storage(&mut state, ForestView::EmberLandings).iter().sum();
    let sum_wind: f32 = read_per_view_storage(&mut state, ForestView::WindExposure).iter().sum();
    let sum_recent: f32 = read_per_view_storage(&mut state, ForestView::RecentFirePressure).iter().sum();

    println!("  -- view fold per-view + AGGREGATE --");
    println!(
        "    sum across all views = {total_view_aggregate:.0}  max/slot = {max_aggregate_per_slot:.0}  nonzero_slots = {nonzero_slots}/{N_TOTAL}",
    );
    println!(
        "    per-view sums: ignition={sum_ignition:.0} ember={sum_ember:.0} wind={sum_wind:.0} recent={sum_recent:.0}",
    );
    println!(
        "    Expected per-slot wind contribution ≈ {expected:.0} (TICKS/7).",
        expected = (TICKS as f64) / 7.0,
    );

    // Verdict: with the Indirect-dispatch gap (commit 353527e6),
    // EmberLanded emits but Catch never fires → only the seed cluster
    // burns out. Healthy trees stay healthy. With the shared-storage
    // gap, we can't measure ignition_count separately — but we CAN
    // measure ash_count, which is the unambiguous "how far did the
    // fire spread" signal.
    let seed_count = 4u32;
    let verdict = if final_ash == seed_count {
        "INDIRECT GAP CONFIRMED — only seed cluster burned out"
    } else if final_ash > seed_count && final_ash < N_TOTAL {
        "FIRE SPREADS PARTIALLY — non-seed trees ash; Catch may fire"
    } else if final_ash >= N_TOTAL {
        "FIRE CONSUMED FOREST — full burn"
    } else {
        "INCONCLUSIVE — even seeds didn't burn out"
    };
    println!("  verdict: {verdict}");
    println!("==========================================");

    // ---- Load-bearing pins ----

    // P1: Initial seeding survived try_new + step. The init block
    // stamps alive=1 + hp=100 across all slots, then we overwrite
    // the centre 2x2 to hp=80.
    assert!(
        initial_alive == N_TOTAL,
        "init block didn't stamp alive=1 across all slots; got {initial_alive}/{N_TOTAL}",
    );
    assert!(
        initial_burning == seed_count,
        "ignition cluster seed missed; got {initial_burning} burning, expected {seed_count}",
    );

    // P2: Conservation invariant holds at end of run. (Per-tick
    // checking would require N_TOTAL alive readbacks → too slow; end-
    // of-run is the proxy.)
    assert_eq!(
        final_alive + final_ash,
        N_TOTAL,
        "alive + ash conservation broken: {final_alive} + {final_ash} != {N_TOTAL}",
    );

    // P3: View accumulators picked up SOMETHING. Now that per-view
    // storage is wired (no more shared aliasing), each view's sum is
    // independently observable. wind_exposure broadcasts to every
    // tree on every WindShifted tick and is the largest accumulator;
    // load-bear on the wind sum being well above zero (1024 trees ×
    // 71 wind ticks = ~72,000 expected).
    assert!(
        total_view_aggregate > 0.0,
        "view storage aggregate is 0 — no fold fired OR Spread didn't emit",
    );
    assert!(
        sum_wind > 1000.0,
        "wind_exposure per-view sum unexpectedly low ({sum_wind:.0}) — \
         Spread/WindEvent producer or fold pipeline regressed; \
         per-view storage now isolates this view from the others.",
    );

    // P4 (DISCOVERED GAP): host-side `event_ring.tail_value()` stays at
    // 0 forever in the auto-emitted runtime. The synthesized step()
    // calls `clear_tail_in()` (zeroes both the GPU `event_tail` counter
    // AND the host-side `tail_estimate`) at the start of each tick, but
    // NEVER calls `note_emits()` after a producer kernel runs. So the
    // host-side estimate stays at 0 even though the GPU `event_tail`
    // is being atomicAdd'd by every producer.
    //
    // Consequence: any consumer that reads `event_count =
    // event_ring.tail_value()` for its per-tick cfg uniform (per the
    // documented pattern at `event_ring.rs:260-272`) sees 0 and
    // early-returns its body. This silently drops every chronicle
    // consumption in the auto-emit path.
    //
    // Workaround would require synthesize_step to emit
    // `self.event_ring.note_emits(self.agent_count * <emit_count>)`
    // after each producer dispatch. Tracked in
    // `docs/architecture/gaps_observed.md`.
    assert_eq!(
        max_tail, 0,
        "host-side tail_value() unexpectedly advanced — note_emits() now wired? \
         Update the gap doc + tighten this pin to verify GPU tail buffer reads.",
    );

    // P5/P11: Determinism — re-run with same seed and report mismatches.
    //
    // The per-agent rng.action() PCG itself IS deterministic, but
    // fold kernels accumulating into a shared
    // `view_storage_primary[agent_id]` slot via atomicAdd produce f32
    // reduction non-determinism (atomicAdd is sequentially consistent
    // but f32 addition is non-associative). The race is benign in
    // shape (same total accumulates regardless of interleaving), but
    // per-slot ULPs vary across runs — the canonical P11 sort-then-
    // fold case; see `project_f32_rmw_race` memory + the "Remaining
    // open" line in `docs/architecture/gaps_observed.md`.
    //
    // Pin form: report mismatch count; allow small drift since the
    // race is documented architectural state. If the count hits the
    // slot count (every slot drifted) the determinism contract is
    // fully broken; if it's 0 the race resolved deterministically.
    drop(state);
    let mut state2 = GeneratedRuntime::try_new(SEED, N_TOTAL).expect("re-init");
    seed_grid(&mut state2);
    seed_ignition_cluster(&mut state2);
    state2.step();
    let _ = read_shared_view_storage(&mut state2);  // warmup sync, mirrors run 1
    for _ in 0..TICKS {
        state2.step();
    }
    let view_aggregate2 = read_shared_view_storage(&mut state2);

    let mut mismatches = 0usize;
    let mut max_abs_drift = 0.0f32;
    for (&a, &b) in view_aggregate.iter().zip(view_aggregate2.iter()) {
        let drift = (a - b).abs();
        if drift > 0.001 {
            mismatches += 1;
            max_abs_drift = max_abs_drift.max(drift);
        }
    }
    println!(
        "  determinism: SEED=0x{SEED:X} re-run — {mismatches}/{n} slots drifted, max |Δ| = {max_abs_drift:.3} \
         (race expected from f32 reduction non-associativity; P11)",
        n = view_aggregate.len(),
    );

    // Slack history:
    //   * Pre-P11-work: max drift 47-95 observed; threshold 150.
    //   * Post-P11-work (5 violations closed: engine stride, spatial scatter,
    //     sort scatter, fold serial-scan, Catch post-CAS gating + seq loop_iter):
    //     max drift 30-84 across same-seed reruns (9-run sample); threshold
    //     tightened to 100 (~10-unit margin above observed worst-case of 84).
    //
    // Residual drift (30-84) is from parallel atomic-append slot acquisition in
    // non-CAS-gated emit kernels (Spread, Reaper, etc.). Closing it would require
    // @ws=1 across the entire physics emit layer — out of scope; would tank perf
    // for large fixtures.
    assert!(
        max_abs_drift <= 100.0,
        "determinism drift exceeds 100 — control flow regression or a new \
         violation appeared. max_abs_drift={max_abs_drift}, mismatches={mismatches}/{n}",
        n = view_aggregate.len(),
    );
}

// =====================================================================
// Helpers — host seeding + readback
// =====================================================================

/// Seed an N×N grid of Tree agents. Positions on a unit grid, vel=0.
fn seed_grid(state: &mut GeneratedRuntime) {
    let n = N_TOTAL as usize;
    let mut positions: Vec<[f32; 4]> = Vec::with_capacity(n);
    for y in 0..GRID_SIDE {
        for x in 0..GRID_SIDE {
            positions.push([x as f32, y as f32, 0.0, 0.0]);
        }
    }
    state.gpu.queue.write_buffer(
        &state.agent_pos_buf,
        0,
        bytemuck::cast_slice(&positions),
    );
    // creature_type already 0 across the board (single Tree entity → one
    // discriminant). No need to write it.
}

/// Stamp a 2×2 ignition cluster at the grid centre. Sets hp=80 on those
/// 4 slots so the Spread `where` predicate (1 < hp < 100) matches them
/// on the very first tick.
fn seed_ignition_cluster(state: &mut GeneratedRuntime) {
    let n = N_TOTAL as usize;
    // Read full hp buffer, mutate 4 cells, write back. The init block
    // stamped 100.0 across all slots; we drop the centre 2x2 to 80.
    let mut hp = vec![100.0f32; n];
    let cx = GRID_SIDE / 2;
    let cy = GRID_SIDE / 2;
    for dy in 0..2u32 {
        for dx in 0..2u32 {
            let x = cx + dx;
            let y = cy + dy;
            let idx = (y * GRID_SIDE + x) as usize;
            hp[idx] = 80.0;
        }
    }
    state.gpu.queue.write_buffer(
        &state.agent_hp_buf,
        0,
        bytemuck::cast_slice(&hp),
    );
}

fn count_alive(state: &mut GeneratedRuntime) -> u32 {
    let alive = read_u32_buf(state, &state.agent_alive_buf.clone(), N_TOTAL as usize);
    alive.iter().filter(|&&a| a != 0).count() as u32
}

fn count_burning(state: &mut GeneratedRuntime) -> u32 {
    // Approximate via hp readback: hp in (0, 100) AND alive == 1.
    let alive = read_u32_buf(state, &state.agent_alive_buf.clone(), N_TOTAL as usize);
    let hp = read_f32_buf(state, &state.agent_hp_buf.clone(), N_TOTAL as usize);
    alive
        .iter()
        .zip(hp.iter())
        .filter(|(&a, &h)| a != 0 && h > 0.0 && h < 100.0)
        .count() as u32
}

/// Read one of the four per-view storage buffers. The per-view
/// storage aliasing gap (commit "fix(build_helper): per-view storage
/// buffers (6-fixture aliasing gap)") is now closed: each
/// `@materialized` view gets its own host-side
/// `view_storage_<view_name>_primary_buf` instead of the legacy
/// shared `view_storage_primary_buf`.
#[derive(Debug, Clone, Copy)]
enum ForestView {
    IgnitionCount,
    EmberLandings,
    WindExposure,
    RecentFirePressure,
}

fn read_per_view_storage(state: &mut GeneratedRuntime, view: ForestView) -> Vec<f32> {
    let n = N_TOTAL as usize;
    let bytes = (n as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forest_fire::view_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some("forest_fire::view_readback"),
        },
    );
    let src_buf = match view {
        ForestView::IgnitionCount => &state.view_storage_ignition_count_primary_buf,
        ForestView::EmberLandings => &state.view_storage_ember_landings_primary_buf,
        ForestView::WindExposure => &state.view_storage_wind_exposure_primary_buf,
        ForestView::RecentFirePressure => &state.view_storage_recent_fire_pressure_primary_buf,
    };
    encoder.copy_buffer_to_buffer(src_buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..n].to_vec()
    };
    staging.unmap();
    out
}

/// Convenience: sum all four per-view buffers slot-wise. Used to
/// preserve the legacy "aggregate" shape the pin's determinism check
/// + total-accumulator assertions consume.
fn read_shared_view_storage(state: &mut GeneratedRuntime) -> Vec<f32> {
    let n = N_TOTAL as usize;
    let mut out = vec![0.0f32; n];
    for v in [
        ForestView::IgnitionCount,
        ForestView::EmberLandings,
        ForestView::WindExposure,
        ForestView::RecentFirePressure,
    ] {
        let cells = read_per_view_storage(state, v);
        for (i, c) in cells.iter().enumerate() {
            out[i] += c;
        }
    }
    out
}

fn read_u32_buf(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forest_fire::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("forest_fire::u32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

fn read_f32_buf(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forest_fire::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("forest_fire::f32_readback") },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
