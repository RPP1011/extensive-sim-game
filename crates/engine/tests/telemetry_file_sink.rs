use engine::telemetry::{FileSink, TelemetrySink};
use std::fs;

#[test]
fn file_sink_writes_json_lines() {
    let tmp = std::env::temp_dir().join("engine_file_sink_test.jsonl");
    let _ = fs::remove_file(&tmp);

    {
        let sink = FileSink::create(&tmp).unwrap();
        sink.emit("foo", 42.0, &[]);
        sink.emit_histogram("bar", 1.5);
        sink.emit_counter("baz", 7);
        sink.flush();
    }

    let text = fs::read_to_string(&tmp).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].contains("\"metric\":\"foo\""));
    assert!(lines[0].contains("\"value\":42"));
    assert!(lines[1].contains("\"hist\""));
    assert!(lines[2].contains("\"counter\""));
    fs::remove_file(&tmp).ok();
}
