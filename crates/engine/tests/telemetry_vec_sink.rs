use engine::telemetry::{TelemetrySink, VecSink};

#[test]
fn vec_sink_collects_emits_in_order() {
    let s = VecSink::new();
    s.emit("a", 1.0, &[("k", "v")]);
    s.emit_histogram("b", 2.0);
    s.emit_counter("c", 3);
    let rows = s.drain();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].metric, "a");
    assert!((rows[0].value - 1.0).abs() < 1e-9);
    assert_eq!(rows[0].kind, "gauge");
    assert_eq!(rows[1].kind, "hist");
    assert_eq!(rows[2].kind, "counter");
}
