//! Three-tier ecosystem-cascade harness — runs the
//! [`ecosystem_runtime`] fixture and logs per-tier centroid summaries
//! every `LOG_INTERVAL_TICKS` ticks. After the run, reads back each of
//! the three @decay-anchored views and asserts they converge to the
//! analytical steady state predicted by their decay rate and the
//! per-tick emit count from the corresponding tier.
//!
//! ## Tier layout
//!
//! Slots are split into roughly equal thirds in declaration order
//! (Plant=0, Herbivore=1, Carnivore=2). With AGENT_COUNT=64 this gives
//! 21 plants + 21 herbivores + 22 carnivores; the per-tier dispatches
//! commit writes only when their `where (self.creature_type == X)`
//! guard fires.
//!
//! ## Observable expectations vs the @decay analytical steady state
//!
//! Each tier emits exactly one event per agent per tick (placeholder
//! self-targeting emits — real per-pair strike-radius work lands when
//! spatial body-iter in physics bodies works). Per-event the matching
//! view increments by 1.0 BEFORE the next per-tick anchor multiply
//! lands, so the per-slot steady-state is `1 / (1 - decay_rate)`:
//!
//! - recent_browse           (rate=0.92): 1 / 0.08 = 12.5
//! - predator_pressure       (rate=0.95): 1 / 0.05 = 20.0
//! - plant_pressure          (rate=0.90): 1 / 0.10 = 10.0
//!
//! ## Compiler-emitted shape this app surfaces
//!
//! - **Single shared event ring** for both PlantEaten + HerbivoreEaten
//!   — the schedule synthesizer emitted ONE EventRing the runtime
//!   binds to both physics emitters and all three folds. Tag bytes
//!   live at `event_ring[idx*10+0]` (1u for PlantEaten, 2u for
//!   HerbivoreEaten) but the fold kernels currently DO NOT filter on
//!   them — they all read `event_ring[idx*10+3]` (the `by` field of
//!   either kind) blindly. So:
//!     * **recent_browse[herbivore_slot]** picks up the herbivore's
//!       PlantEaten emit (by=self) → +1/tick, steady-state ≈ 12.5.
//!     * **recent_browse[carnivore_slot]** also picks up the
//!       carnivore's HerbivoreEaten emit (by=self) → +1/tick, also
//!       ≈ 12.5. Crosstalk; the .sim view spec is "Per-Herbivore".
//!     * Symmetrically for predator_pressure (≈20 on both
//!       herbivore + carnivore slots).
//!     * plant_pressure reads `event[idx*10+3]` which is the same
//!       `by` field (placeholder emit makes plant=by=self), so
//!       plant slots stay 0 here too.
//! - **Plants** never appear as `by` in any event → all three views
//!   stay 0.0 on plant slots, regardless of tag-discrimination state.
//!
//! Compiler gap to surface (DO NOT fix in this task — see the runtime
//! crate's lib.rs head doc): the fold kernels should add a tag-filter
//! `if (event_ring[idx*10+0] != <expected_kind>) { return; }` so views
//! materialize only off the event kinds named in their `on <Event>`
//! handler. Today they don't, and recent_browse + predator_pressure
//! pick up both event kinds.

use ecosystem_runtime::EcosystemState;
use engine::CompiledSim;
use glam::Vec3;

const SEED: u64 = 0xEC0_5_71E3_42A8;
const AGENT_COUNT: u32 = 64;
const TICKS: u64 = 200;
const LOG_INTERVAL_TICKS: u64 = 25;

fn main() {
    let mut sim = EcosystemState::new(SEED, AGENT_COUNT);
    let plant_count = sim.plant_count();
    let herbivore_count = sim.herbivore_count();
    let carnivore_count = sim.carnivore_count();
    println!(
        "ecosystem_app: starting run — seed=0x{:016X} agents={} \
         plants={} herbivores={} carnivores={} ticks={}",
        SEED, AGENT_COUNT, plant_count, herbivore_count, carnivore_count, TICKS,
    );
    log_sample(&mut sim);

    for _ in 0..TICKS {
        sim.step();
        if sim.tick() % LOG_INTERVAL_TICKS == 0 {
            log_sample(&mut sim);
        }
    }

    println!(
        "ecosystem_app: finished — final tick={} agents={} ({} plants + {} herbivores + {} carnivores)",
        sim.tick(),
        sim.agent_count(),
        plant_count,
        herbivore_count,
        carnivore_count,
    );

    // ---- View readbacks + steady-state assertions ----
    let creature_types = sim.creature_types();
    let rb = sim.recent_browses().to_vec();
    let pp = sim.predator_pressures().to_vec();
    let plp = sim.plant_pressures().to_vec();
    let third = AGENT_COUNT / 3;

    let rb_steady = 1.0_f32 / (1.0 - 0.92);
    let pp_steady = 1.0_f32 / (1.0 - 0.95);
    let plp_steady = 1.0_f32 / (1.0 - 0.90);

    let (rb_p, rb_h, rb_c) = per_tier_stats(&rb, &creature_types);
    let (pp_p, pp_h, pp_c) = per_tier_stats(&pp, &creature_types);
    let (plp_p, plp_h, plp_c) = per_tier_stats(&plp, &creature_types);

    println!(
        "ecosystem_app: recent_browse      (rate=0.92, steady ~{:.2}/slot) \
         — plants(min/mean/max)={:.2}/{:.2}/{:.2}  herb={:.2}/{:.2}/{:.2}  carn={:.2}/{:.2}/{:.2}",
        rb_steady,
        rb_p.0, rb_p.1, rb_p.2,
        rb_h.0, rb_h.1, rb_h.2,
        rb_c.0, rb_c.1, rb_c.2,
    );
    println!(
        "ecosystem_app: predator_pressure  (rate=0.95, steady ~{:.2}/slot) \
         — plants(min/mean/max)={:.2}/{:.2}/{:.2}  herb={:.2}/{:.2}/{:.2}  carn={:.2}/{:.2}/{:.2}",
        pp_steady,
        pp_p.0, pp_p.1, pp_p.2,
        pp_h.0, pp_h.1, pp_h.2,
        pp_c.0, pp_c.1, pp_c.2,
    );
    println!(
        "ecosystem_app: plant_pressure     (rate=0.90, steady ~{:.2}/slot) \
         — plants(min/mean/max)={:.2}/{:.2}/{:.2}  herb={:.2}/{:.2}/{:.2}  carn={:.2}/{:.2}/{:.2}",
        plp_steady,
        plp_p.0, plp_p.1, plp_p.2,
        plp_h.0, plp_h.1, plp_h.2,
        plp_c.0, plp_c.1, plp_c.2,
    );

    // ---- Acceptance assertions (±10% per the task spec) ----
    // recent_browse[herbivore]: each herbivore emits 1 PlantEaten/tick
    // with by=self → matching slot increments by 1/tick → steady=12.5.
    let tol_rb = rb_steady * 0.10;
    assert!(
        (rb_h.1 - rb_steady).abs() <= tol_rb,
        "recent_browse[herbivore] mean {:.3} not within ±10% of analytical {:.3}",
        rb_h.1, rb_steady,
    );
    // predator_pressure[carnivore]: each carnivore emits 1
    // HerbivoreEaten/tick with by=self → +1/tick → steady=20.0.
    let tol_pp = pp_steady * 0.10;
    assert!(
        (pp_c.1 - pp_steady).abs() <= tol_pp,
        "predator_pressure[carnivore] mean {:.3} not within ±10% of analytical {:.3}",
        pp_c.1, pp_steady,
    );
    // Plants emit nothing → plant slots stay at 0 in all three views.
    assert!(
        rb_p.2 == 0.0 && pp_p.2 == 0.0 && plp_p.2 == 0.0,
        "plant slots should be 0 in all views (plants emit nothing) but got \
         rb_max={:.3} pp_max={:.3} plp_max={:.3}",
        rb_p.2, pp_p.2, plp_p.2,
    );

    // Sanity: the 3-way split totals should match agent_count.
    assert_eq!(
        plant_count + herbivore_count + carnivore_count,
        AGENT_COUNT,
        "tier counts should total agent_count",
    );
    assert_eq!(plant_count, third, "plant_count should be agent_count/3");

    println!(
        "ecosystem_app: assertions OK — recent_browse[herb] ≈ {:.2} \
         (target {:.2}), predator_pressure[carn] ≈ {:.2} (target {:.2}), \
         plant slots all 0.",
        rb_h.1, rb_steady, pp_c.1, pp_steady,
    );
}

/// `(min, mean, max)` of `values` filtered to slots whose
/// `creature_types[slot]` matches `tier`.
fn tier_stats(values: &[f32], creature_types: &[u32], tier: u32) -> (f32, f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = 0.0_f32;
    let mut sum = 0.0_f32;
    let mut count = 0u32;
    for (slot, &v) in values.iter().enumerate() {
        if creature_types[slot] != tier {
            continue;
        }
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v;
        count += 1;
    }
    if count == 0 {
        return (0.0, 0.0, 0.0);
    }
    (min, sum / count as f32, max)
}

/// `((plant_min, plant_mean, plant_max), (herb…), (carn…))` —
/// per-tier stats for one view.
fn per_tier_stats(
    values: &[f32],
    creature_types: &[u32],
) -> ((f32, f32, f32), (f32, f32, f32), (f32, f32, f32)) {
    (
        tier_stats(values, creature_types, 0),
        tier_stats(values, creature_types, 1),
        tier_stats(values, creature_types, 2),
    )
}

/// One per-tick log line: per-tier centroid + max axis-aligned span.
fn log_sample(sim: &mut EcosystemState) {
    let tick = sim.tick();
    let creature_types = sim.creature_types();
    let positions = sim.positions().to_vec();
    if positions.is_empty() {
        println!("  tick {:>4}: (no agents)", tick);
        return;
    }
    let (plant_centroid, plant_span) =
        group_stats(&positions, |slot| creature_types[slot] == 0);
    let (herb_centroid, herb_span) =
        group_stats(&positions, |slot| creature_types[slot] == 1);
    let (carn_centroid, carn_span) =
        group_stats(&positions, |slot| creature_types[slot] == 2);
    println!(
        "  tick {:>4}: plant centroid=({:>+5.2},{:>+5.2},{:>+5.2}) span={:>+4.2} | \
         herb=({:>+5.2},{:>+5.2},{:>+5.2}) span={:>+4.2} | \
         carn=({:>+5.2},{:>+5.2},{:>+5.2}) span={:>+4.2}",
        tick,
        plant_centroid.x, plant_centroid.y, plant_centroid.z, plant_span,
        herb_centroid.x, herb_centroid.y, herb_centroid.z, herb_span,
        carn_centroid.x, carn_centroid.y, carn_centroid.z, carn_span,
    );
}

/// Centroid + max axis-aligned span across the slot positions whose
/// indices match `pred`. Returns (`Vec3::ZERO`, 0.0) when no slots
/// match.
fn group_stats(positions: &[Vec3], pred: impl Fn(usize) -> bool) -> (Vec3, f32) {
    let mut min = None::<Vec3>;
    let mut max = None::<Vec3>;
    let mut sum = Vec3::ZERO;
    let mut count = 0u32;
    for (slot, p) in positions.iter().enumerate() {
        if !pred(slot) {
            continue;
        }
        min = Some(min.map_or(*p, |m| m.min(*p)));
        max = Some(max.map_or(*p, |m| m.max(*p)));
        sum += *p;
        count += 1;
    }
    if count == 0 {
        return (Vec3::ZERO, 0.0);
    }
    let centroid = sum / count as f32;
    let span_v = max.unwrap() - min.unwrap();
    let span = span_v.x.max(span_v.y).max(span_v.z);
    (centroid, span)
}
