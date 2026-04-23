//! Indirect dispatch args buffer layout + helper.
//!
//! An indirect dispatch reads `(workgroup_x, workgroup_y, workgroup_z)`
//! as three consecutive u32s from a device buffer at
//! `dispatch_workgroups_indirect(buf, offset)` time. A buffer of N
//! slots × 12 B can stage N consecutive indirect dispatches whose
//! workgroup counts are computed by preceding kernels.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// One (x, y, z) tuple. Writable by WGSL via
/// `array<vec3<u32>>` binding.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Pod, Zeroable)]
pub struct IndirectArgs {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub struct IndirectArgsBuffer {
    buf: wgpu::Buffer,
    slots: u32,
}

impl IndirectArgsBuffer {
    /// `slots` = how many consecutive indirect dispatches this buffer
    /// can drive. Initialised to `(0, 1, 1)` everywhere so dispatches
    /// that read from uninitialised slots no-op.
    pub fn new(device: &wgpu::Device, slots: u32) -> Self {
        let initial: Vec<IndirectArgs> = (0..slots)
            .map(|_| IndirectArgs { x: 0, y: 1, z: 1 })
            .collect();
        let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_util::indirect::args"),
            contents: bytemuck::cast_slice(&initial),
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Self { buf, slots }
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Byte offset for the given slot. Panics if `slot >= slots()`.
    /// Used by `dispatch_workgroups_indirect(buf, offset)`.
    pub fn slot_offset(&self, slot: u32) -> u64 {
        assert!(slot < self.slots, "slot {slot} out of range (slots={})", self.slots);
        (slot as u64) * 12
    }

    /// Test-only blocking readback. Not for the hot path.
    #[doc(hidden)]
    pub fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<IndirectArgs> {
        crate::gpu_util::readback::readback_typed(
            device,
            queue,
            &self.buf,
            (self.slots as usize) * 12,
        )
        .expect("indirect args readback")
    }
}
