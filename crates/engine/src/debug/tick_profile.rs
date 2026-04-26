//! Phase timing histogram.
//!
//! Wraps `Instant` measurements per phase; emits via `TelemetrySink` if
//! installed; persists raw samples for offline analysis.
//!
//! Per `spec/runtime.md` §24 and Plan 4 Task 5.

use crate::telemetry::{metrics, NullSink, TelemetrySink};
use std::collections::BTreeMap;
use std::time::Instant;

/// Accumulates per-phase nanosecond samples across ticks.
///
/// Usage pattern — call `enter(phase)` at the start of a phase, then
/// `exit(&sink)` (or `exit_with_null()`) at the end. Samples are stored in
/// `samples()` and also forwarded to the provided `TelemetrySink`.
///
/// Not `Send` by itself; callers wrap in `Arc<Mutex<TickProfile>>` when they
/// need shared access across a tick boundary.
#[derive(Debug)]
pub struct TickProfile {
    samples:   BTreeMap<&'static str, Vec<u128>>,
    in_flight: Option<(&'static str, Instant)>,
}

impl Default for TickProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl TickProfile {
    /// Create an empty profile with no samples.
    pub fn new() -> Self {
        Self {
            samples:   BTreeMap::new(),
            in_flight: None,
        }
    }

    /// Record the start of `phase`. Any previous in-flight phase is silently
    /// discarded (mismatched pairs should not happen in correct usage).
    pub fn enter(&mut self, phase: &'static str) {
        self.in_flight = Some((phase, Instant::now()));
    }

    /// Record the end of the current in-flight phase, push the elapsed
    /// nanoseconds into `samples`, and forward to `telemetry`.
    ///
    /// No-op if `enter` was never called.
    pub fn exit(&mut self, telemetry: &dyn TelemetrySink) {
        if let Some((phase, t0)) = self.in_flight.take() {
            let ns = t0.elapsed().as_nanos();
            self.samples.entry(phase).or_default().push(ns);
            telemetry.emit_histogram(metrics::DEBUG_PHASE_NS, ns as f64);
        }
    }

    /// Convenience: `exit` with a `NullSink` (no telemetry forwarding).
    pub fn exit_with_null(&mut self) {
        self.exit(&NullSink);
    }

    /// Borrow the raw sample map: phase name → nanoseconds-per-tick vec.
    pub fn samples(&self) -> &BTreeMap<&'static str, Vec<u128>> {
        &self.samples
    }
}
