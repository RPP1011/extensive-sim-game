use engine::telemetry::TelemetrySink;

struct CountingSink {
    inner: std::sync::Mutex<Vec<(String, f64)>>,
}

impl CountingSink {
    fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(Vec::new()),
        }
    }
    fn samples(&self) -> Vec<(String, f64)> {
        self.inner.lock().unwrap().clone()
    }
}

impl TelemetrySink for CountingSink {
    fn emit(&self, metric: &'static str, value: f64, _tags: &[(&'static str, &'static str)]) {
        self.inner.lock().unwrap().push((metric.to_string(), value));
    }
    fn emit_histogram(&self, metric: &'static str, value: f64) {
        self.inner
            .lock()
            .unwrap()
            .push((format!("hist:{}", metric), value));
    }
    fn emit_counter(&self, metric: &'static str, delta: i64) {
        self.inner
            .lock()
            .unwrap()
            .push((format!("ctr:{}", metric), delta as f64));
    }
}

#[test]
fn object_safe_and_basic_emit() {
    let s = CountingSink::new();
    let sink: &dyn TelemetrySink = &s;
    sink.emit("foo", 1.0, &[]);
    sink.emit_histogram("latency", 12.3);
    sink.emit_counter("events", 5);
    let samples = s.samples();
    assert_eq!(samples.len(), 3);
    assert_eq!(samples[0].0, "foo");
    assert_eq!(samples[1].0, "hist:latency");
    assert_eq!(samples[2].0, "ctr:events");
}

#[test]
fn null_sink_accepts_emits_without_panic() {
    use engine::telemetry::NullSink;
    let s = NullSink;
    s.emit("foo", 1.0, &[]);
    s.emit_histogram("bar", 2.0);
    s.emit_counter("baz", 3);
}
