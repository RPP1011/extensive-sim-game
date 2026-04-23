//! Unit test for gpu_util::readback::readback_typed.

#![cfg(feature = "gpu")]

use engine_gpu::gpu_util::readback::readback_typed;
use wgpu::util::DeviceExt;

#[test]
fn readback_typed_roundtrips_u32_slice() {
    let (device, queue) = engine_gpu::test_device().expect("test device");
    let src: Vec<u32> = (0..256u32).collect();
    let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test::src"),
        contents: bytemuck::cast_slice(&src),
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
    });
    let out: Vec<u32> = readback_typed(&device, &queue, &buf, src.len() * 4).expect("readback");
    assert_eq!(out, src);
}
