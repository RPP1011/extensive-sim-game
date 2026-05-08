//! Wave 3 ToM Phase 3 decay-slope pin: confidence decays linearly at
//! `decay_per_tick = 1` per tick of staleness, saturating at 0.
//!
//! ## What this exercises
//!
//! Phase 3 introduced `TomProbeState::decay_step()` — the runtime-side
//! per-tick decay phase that walks every (observer, target) cell and
//! decrements `confidence` by `(world.tick - last_seen_tick) *
//! decay_per_tick`, saturating at 0. The decay phase mirrors the
//! "memory fades" half of the ToM model: a fresh observation pegs
//! confidence at 255 (q8 max), and each tick of staleness peels off
//! one unit until the cell hits 0 (no information).
//!
//! The DSL `@decay(beliefs)` annotation that would synthesize this
//! kernel from the .sim surface is deferred to Phase 4. Today the
//! runtime hand-rolls the sweep so the slope is pinnable without
//! expanding the DSL surface.
//!
//! ## Fixture
//!
//! 4 agents. Pre-seed cell `[0 * N + 1]` with `confidence = 255` and
//! `last_seen_tick = 0` (a fresh observation made at tick 0). Run
//! 100 ticks; after each `step()`, call `decay_step()`. The cell's
//! confidence should drop monotonically by 1 per tick.
//!
//! ## Asserts
//!
//! - **Slope (the load-bearing claim)**: after 100 ticks, the cell's
//!   confidence is `255 - 100 = 155`. Pin this exact value so a
//!   change to `decay_per_tick` (= 1 today) shows up loudly in the
//!   regression diff.
//! - **Saturation**: after 300 ticks the cell would algebraically
//!   need a -45 confidence; the saturating arithmetic clamps at 0
//!   instead. Pin `confidence == 0` after 300 ticks.
//! - **Per-cell isolation**: the decay sweep walks all N² cells but
//!   only seeded cells should change. Cells that were never seeded
//!   (last_seen_tick=0 AND confidence=0) stay at 0; the seeded cell
//!   decays. No bleeding into neighbouring rows / columns.
//! - **Fresh observation resets**: after decay drops a cell to 100
//!   at tick 155, calling `observe(0, 1)` re-pegs confidence at 255
//!   AND sets `last_seen_tick = 155`. Subsequent decay sweeps
//!   compute `delta = current_tick - 155`, not `delta =
//!   current_tick - 0`.
//!
//! ## Skip path
//!
//! GPU adapter unavailable — same convention as
//! `pair_map_behavioral_pin.rs`.

use engine::CompiledSim;
use tom_probe_runtime::TomProbeState;

const SEED: u64 = 0xDECA_F00D_BEEF_BAD;
const N: u32 = 4;
const OBSERVER: u32 = 0;
const TARGET: u32 = 1;

fn try_new() -> Option<TomProbeState> {
    std::panic::catch_unwind(|| TomProbeState::new(SEED, N)).ok()
}

#[test]
fn decay_slope_is_one_per_tick() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[decay_pin] skipping: no wgpu adapter");
            return;
        }
    };

    // Seed cell [OBSERVER * N + TARGET] with a fresh observation made
    // at tick 0. Other cells stay at default zero (will be no-op'd by
    // the decay sweep's `last == 0 && confidence == 0` guard).
    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);

    let cell_count = (N * N) as usize;
    let mut confidence_seed = vec![0u8; cell_count];
    confidence_seed[cell] = 255;
    sim.seed_beliefs_confidence(&confidence_seed);
    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 0;
    sim.seed_beliefs_tick(&tick_seed);

    // Run 100 ticks. After each step, run the decay phase. The Phase
    // 1 producer fires on every step but writes into beliefs_FLAGS,
    // not beliefs_confidence — orthogonal column.
    for _ in 0..100 {
        sim.step();
        sim.decay_step();
    }
    assert_eq!(sim.tick(), 100);

    let confidence = sim.beliefs_confidence().to_vec();
    assert_eq!(
        confidence[cell], 155,
        "after 100 ticks at decay_per_tick=1, confidence should be 255 - 100 = 155 (got {})",
        confidence[cell],
    );

    // Per-cell isolation: every other cell stayed at 0 (the no-op
    // guard prevented the decay sweep from corrupting unseeded cells).
    for c in 0..cell_count {
        if c == cell {
            continue;
        }
        assert_eq!(
            confidence[c], 0,
            "decay sweep should not corrupt unseeded cell {c} (got {})",
            confidence[c],
        );
    }
}

#[test]
fn decay_saturates_at_zero() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[decay_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);
    let cell_count = (N * N) as usize;

    let mut confidence_seed = vec![0u8; cell_count];
    confidence_seed[cell] = 255;
    sim.seed_beliefs_confidence(&confidence_seed);
    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 0;
    sim.seed_beliefs_tick(&tick_seed);

    // 300 ticks would algebraically take confidence to -45; the
    // saturating_sub clamps at 0.
    for _ in 0..300 {
        sim.step();
        sim.decay_step();
    }

    let confidence = sim.beliefs_confidence().to_vec();
    assert_eq!(
        confidence[cell], 0,
        "after 300 ticks, decay should saturate confidence at 0 (got {})",
        confidence[cell],
    );
}

#[test]
fn fresh_observation_resets_decay() {
    let mut sim = match try_new() {
        Some(s) => s,
        None => {
            eprintln!("[decay_pin] skipping: no wgpu adapter");
            return;
        }
    };

    let n = N as usize;
    let cell = (OBSERVER as usize) * n + (TARGET as usize);

    // Seed target B's agent SoA so observe(0, 1) has data to copy.
    let mut pos_seed = vec![[0.0f32; 4]; N as usize];
    pos_seed[TARGET as usize] = [42.0, 0.0, 0.0, 1.0];
    sim.seed_agent_pos(&pos_seed);
    let mut type_seed = vec![0u8; N as usize];
    type_seed[TARGET as usize] = 7;
    sim.seed_agent_creature_type(&type_seed);

    // Run 155 ticks with no observations — decay drives confidence
    // from initial 0 (default) to ... still 0, since the cell was
    // never seeded with confidence > 0. Use an explicit seed instead
    // so the decay sweep has something to chew on.
    let cell_count = (N * N) as usize;
    let mut confidence_seed = vec![0u8; cell_count];
    confidence_seed[cell] = 255;
    sim.seed_beliefs_confidence(&confidence_seed);
    let mut tick_seed = vec![0u32; cell_count];
    tick_seed[cell] = 0;
    sim.seed_beliefs_tick(&tick_seed);

    for _ in 0..155 {
        sim.step();
        sim.decay_step();
    }
    assert_eq!(sim.tick(), 155);
    let conf_after_decay = sim.beliefs_confidence().to_vec();
    assert_eq!(
        conf_after_decay[cell], 100,
        "after 155 ticks decay should land confidence at 255 - 155 = 100 (got {})",
        conf_after_decay[cell],
    );

    // Fresh observation: re-pegs confidence to 255 AND updates
    // last_seen_tick to 155. Subsequent decay sweeps should walk the
    // delta from 155, not 0.
    sim.observe(OBSERVER, TARGET);
    let conf_after_observe = sim.beliefs_confidence().to_vec();
    assert_eq!(
        conf_after_observe[cell], 255,
        "fresh observation should re-peg confidence at 255",
    );
    let ticks_after_observe = sim.beliefs_tick().to_vec();
    assert_eq!(
        ticks_after_observe[cell], 155,
        "fresh observation should set last_seen_tick to current tick (155)",
    );

    // Run 50 more ticks of decay. Slope is `255 - (current_tick -
    // 155) * 1` = `255 - 50 = 205`.
    for _ in 0..50 {
        sim.step();
        sim.decay_step();
    }
    assert_eq!(sim.tick(), 205);
    let conf_after_more_decay = sim.beliefs_confidence().to_vec();
    assert_eq!(
        conf_after_more_decay[cell], 205,
        "after 50 more ticks, decay should land confidence at 255 - 50 = 205 (got {}); \
         the post-observe last_seen_tick=155 should be the decay anchor, not the original 0",
        conf_after_more_decay[cell],
    );
}
