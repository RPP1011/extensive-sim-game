//! Stress test for the compiler-emitted `metrics` module.
//!
//! Closes the silent-drop gap on `metric { metric <name> = <expr> ... }`
//! declarations (Slice C of the verb/probe/metric emit plan,
//! `docs/superpowers/plans/2026-05-03-verb-probe-metric-emit.md`).
//! `assets/sim/swarm_event_storm.sim` declares three metrics
//! (`tick_gauge`, `tick_counter`, `tick_histogram`) that all parsed
//! and resolved cleanly but produced no compiler artifact — the
//! emitted `metrics.rs` module now synthesises a `MetricsSink` struct
//! plus a `record_tick(world_tick)` driver. This test proves the
//! artifact actually fires by driving the auto-driver across N ticks
//! and asserting the recorded values match the metric declarations.
//!
//! The driver path is pure host code — no GPU coupling — so the test
//! runs on the CI bench machine without an adapter. The full GPU
//! integration is exercised end-to-end by the runtime's
//! `SwarmStormState::step()`, which calls `record_tick` after every
//! GPU submission. The split here is deliberate: the auto-driven
//! metric semantics belong to the compiler-emitted module; the test
//! pins the contract without needing to spin up a `wgpu::Adapter`.
//!
//! Cross-references:
//! - Emitter: `crates/dsl_compiler/src/cg/emit/metrics.rs`
//! - Wiring (build.rs): `crates/swarm_storm_runtime/build.rs`
//! - Wiring (runtime call site): `SwarmStormState::step` in
//!   `crates/swarm_storm_runtime/src/lib.rs` —
//!   `self.metrics_sink.record_tick(self.tick)`

use swarm_storm_runtime::metrics::MetricsSink;

#[test]
fn metric_records_per_tick_value() {
    // Drive 100 ticks through the auto-driver. Tick values are
    // 0..100 (mirroring the runtime's pre-increment record pattern).
    let mut sink = MetricsSink::default();
    for tick in 0..100u64 {
        sink.record_tick(tick);
    }

    // tick_gauge: gauge(world.tick) emit_every 1.
    // After 100 ticks: count = 100, last = Some(99) (the final tick
    // recorded, before the runtime would increment).
    assert_eq!(
        sink.tick_gauge.count, 100,
        "tick_gauge should record once per tick"
    );
    assert_eq!(
        sink.tick_gauge.last,
        Some(99.0),
        "tick_gauge.last should mirror the most recent recorded tick"
    );

    // tick_counter: counter(1.0) emit_every 1.
    // After 100 ticks: count = 100, total = 100.0.
    assert_eq!(
        sink.tick_counter.count, 100,
        "tick_counter should record once per tick"
    );
    assert!(
        (sink.tick_counter.total - 100.0).abs() < 1e-9,
        "tick_counter.total should be exactly 100 after 100 ticks (got {})",
        sink.tick_counter.total,
    );

    // tick_histogram: histogram(world.tick) emit_every 10.
    // The auto-driver gates on `world_tick % 10 == 0`. Across ticks
    // 0..100 this fires at 0, 10, 20, ..., 90 — exactly 10 samples.
    assert_eq!(
        sink.tick_histogram.samples.len(),
        10,
        "tick_histogram should record on every 10th tick (0..100 → 10 samples)"
    );
    let expected: Vec<f32> = (0..10).map(|i| (i as f32) * 10.0).collect();
    assert_eq!(
        sink.tick_histogram.samples, expected,
        "tick_histogram samples should be 0, 10, 20, ..., 90"
    );
}

#[test]
fn metric_sink_starts_empty() {
    // Defense in depth: the `Default::default()` shape is the
    // contract the runtime field-init relies on. A non-zero starting
    // sink would skew every downstream observable.
    let sink = MetricsSink::default();
    assert_eq!(sink.tick_gauge.count, 0);
    assert_eq!(sink.tick_gauge.last, None);
    assert_eq!(sink.tick_counter.count, 0);
    assert_eq!(sink.tick_counter.total, 0.0);
    assert!(sink.tick_histogram.samples.is_empty());
}

#[test]
fn metric_emit_every_modulo_handles_tick_zero() {
    // Tick 0 satisfies `0 % N == 0` for every N — the auto-driver
    // must record the histogram sample at tick 0 and not miss it.
    // This is a regression test for the most-likely off-by-one in
    // the `if world_tick % 10 == 0` gate.
    let mut sink = MetricsSink::default();
    sink.record_tick(0);
    assert_eq!(sink.tick_histogram.samples, vec![0.0_f32]);
    assert_eq!(sink.tick_gauge.last, Some(0.0));
    assert!((sink.tick_counter.total - 1.0).abs() < 1e-9);
}

#[test]
fn metric_per_metric_setter_overrides_auto_drive() {
    // The compiler emits a per-metric `record_<name>` setter for
    // every metric — even ones the auto-driver handles. Hand-driven
    // tests + alternate harnesses can override auto-drive by calling
    // the setter directly with a host-computed scalar.
    let mut sink = MetricsSink::default();
    sink.record_tick_gauge(42.0);
    sink.record_tick_counter(2.5);
    sink.record_tick_histogram(7.0);
    assert_eq!(sink.tick_gauge.last, Some(42.0));
    assert_eq!(sink.tick_gauge.count, 1);
    assert!((sink.tick_counter.total - 2.5).abs() < 1e-9);
    assert_eq!(sink.tick_histogram.samples, vec![7.0_f32]);
}
