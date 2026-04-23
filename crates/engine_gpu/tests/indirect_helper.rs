//! Unit test for gpu_util::indirect::IndirectArgsBuffer.

#![cfg(feature = "gpu")]

use engine_gpu::gpu_util::indirect::IndirectArgsBuffer;

#[test]
fn indirect_args_buffer_initial_state_is_noop() {
    let (device, queue) = engine_gpu::test_device().expect("test device");
    let ia = IndirectArgsBuffer::new(&device, 4); // 4 slots
    let vals = ia.read(&device, &queue);
    assert_eq!(vals.len(), 4);
    for v in &vals {
        assert_eq!(v.x, 0);
        assert_eq!(v.y, 1);
        assert_eq!(v.z, 1);
    }
}

#[test]
fn indirect_args_buffer_slot_offset() {
    let (device, _queue) = engine_gpu::test_device().expect("test device");
    let ia = IndirectArgsBuffer::new(&device, 8);
    assert_eq!(ia.slot_offset(0), 0);
    assert_eq!(ia.slot_offset(1), 12);
    assert_eq!(ia.slot_offset(7), 84);
}
