use super::TelemetrySink;

/// Discard-all telemetry sink. Used as the default when callers don't care
/// about metrics.
pub struct NullSink;

impl TelemetrySink for NullSink {
    fn emit(&self, _: &'static str, _: f64, _: &[(&'static str, &'static str)]) {}
    fn emit_histogram(&self, _: &'static str, _: f64) {}
    fn emit_counter(&self, _: &'static str, _: i64) {}
}

/// Stub — Task 7 replaces.
pub struct VecSink;

/// Stub — Task 7 replaces.
pub struct FileSink;
