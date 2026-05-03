use super::TelemetrySink;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;

/// Discard-all telemetry sink. Used as the default when callers don't care
/// about metrics.
pub struct NullSink;

impl TelemetrySink for NullSink {
    fn emit(&self, _: &'static str, _: f64, _: &[(&'static str, &'static str)]) {}
    fn emit_histogram(&self, _: &'static str, _: f64) {}
    fn emit_counter(&self, _: &'static str, _: i64) {}
}

/// A single recorded metric row captured by [`VecSink`].
#[derive(Clone, Debug, PartialEq)]
pub struct MetricRow {
    pub metric: String,
    pub value:  f64,
    /// Discriminator: `"gauge"`, `"hist"`, or `"counter"`.
    pub kind:   &'static str,
    pub tags:   Vec<(String, String)>,
}

/// In-memory telemetry sink — collects every emit into a `Vec<MetricRow>`.
///
/// Useful for tests and for short-lived inspection tools.
pub struct VecSink {
    inner: Mutex<Vec<MetricRow>>,
}

impl VecSink {
    pub fn new() -> Self {
        Self { inner: Mutex::new(Vec::new()) }
    }

    /// Take the collected rows, leaving the sink empty.
    pub fn drain(&self) -> Vec<MetricRow> {
        std::mem::take(&mut *self.inner.lock().unwrap())
    }
}

impl Default for VecSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySink for VecSink {
    fn emit(&self, metric: &'static str, value: f64, tags: &[(&'static str, &'static str)]) {
        let tags: Vec<(String, String)> = tags
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        self.inner.lock().unwrap().push(MetricRow {
            metric: metric.to_string(),
            value,
            kind: "gauge",
            tags,
        });
    }

    fn emit_histogram(&self, metric: &'static str, value: f64) {
        self.inner.lock().unwrap().push(MetricRow {
            metric: metric.to_string(),
            value,
            kind: "hist",
            tags: vec![],
        });
    }

    fn emit_counter(&self, metric: &'static str, delta: i64) {
        self.inner.lock().unwrap().push(MetricRow {
            metric: metric.to_string(),
            value: delta as f64,
            kind: "counter",
            tags: vec![],
        });
    }
}

/// JSON-lines telemetry sink — one flat JSON object per line.
pub struct FileSink {
    inner: Mutex<BufWriter<File>>,
}

impl FileSink {
    pub fn create<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        Ok(Self {
            inner: Mutex::new(BufWriter::new(File::create(path)?)),
        })
    }

    /// Flush the buffered writer.
    pub fn flush(&self) {
        let _ = self.inner.lock().unwrap().flush();
    }

    fn write_json_line(&self, s: &str) {
        let mut w = self.inner.lock().unwrap();
        let _ = writeln!(&mut *w, "{}", s);
    }
}

impl TelemetrySink for FileSink {
    fn emit(&self, metric: &'static str, value: f64, _tags: &[(&'static str, &'static str)]) {
        self.write_json_line(&format!(
            "{{\"kind\":\"gauge\",\"metric\":\"{}\",\"value\":{}}}",
            metric, value,
        ));
    }

    fn emit_histogram(&self, metric: &'static str, value: f64) {
        self.write_json_line(&format!(
            "{{\"kind\":\"hist\",\"metric\":\"{}\",\"value\":{}}}",
            metric, value,
        ));
    }

    fn emit_counter(&self, metric: &'static str, delta: i64) {
        self.write_json_line(&format!(
            "{{\"kind\":\"counter\",\"metric\":\"{}\",\"value\":{}}}",
            metric, delta,
        ));
    }
}
