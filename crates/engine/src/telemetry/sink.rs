//! Telemetry sink trait. Minimal surface; production implementations live
//! downstream. Engine ships NullSink, VecSink (in-memory for tests), and
//! FileSink (JSONL writer).

pub trait TelemetrySink: Send + Sync {
    fn emit(&self, metric: &'static str, value: f64, tags: &[(&'static str, &'static str)]);
    fn emit_histogram(&self, metric: &'static str, value: f64);
    fn emit_counter(&self, metric: &'static str, delta: i64);
}
