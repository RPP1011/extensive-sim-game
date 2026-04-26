use engine::debug::trace_mask::TraceMaskCollector;
use engine::mask::MaskBuffer;

#[test]
fn collector_rings_at_max_ticks() {
    let mut c = TraceMaskCollector::new(3);
    let buf = MaskBuffer::new(2); // 2 agents, 18 kinds (MicroKind::ALL.len())
    for tick in 0..5 {
        c.record(tick, &buf);
    }
    assert_eq!(c.all().count(), 3);
    assert_eq!(c.all().next().unwrap().tick, 2);
    assert_eq!(c.all().last().unwrap().tick, 4);
}
