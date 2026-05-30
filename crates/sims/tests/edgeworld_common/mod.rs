//! Shared constants + staging-buffer readback helpers for the edgeworld
//! Phase 0 tests. Lives in a sibling `mod` directory so both the
//! behavioral pin (`edgeworld_pin.rs`) and the future render test
//! (Task 6) can `mod edgeworld_common;` + `use edgeworld_common::*;`
//! without duplicating the GPU readback plumbing.
//!
//! Not every consumer uses every helper (the pin reads positions/hunger/
//! mana/alive; the render test will reach for creature_types), so the
//! module allows dead code rather than forcing each binary to touch all
//! of them.
#![allow(dead_code)]

use sims::edgeworld::GeneratedRuntime;

/// creature_type discriminant for `FoodNode` (alphabetical entity decl
/// order → FoodNode = 0).
pub const CT_FOOD: u32 = 0;
/// creature_type discriminant for `Survivor` (= 1).
pub const CT_SURVIVOR: u32 = 1;

/// Staging-buffer readback of `state.agent_hunger_buf` as f32.
pub fn read_hunger(state: &mut GeneratedRuntime, count: usize) -> Vec<f32> {
    read_f32(state, &state.agent_hunger_buf.clone(), count, "hunger")
}

/// Staging-buffer readback of `state.agent_mana_buf` as f32 — the
/// FoodNode-quantity column (quantity is repurposed onto the `mana` f32
/// SoA column per the Task 4 decision).
pub fn read_mana(state: &mut GeneratedRuntime, count: usize) -> Vec<f32> {
    read_f32(state, &state.agent_mana_buf.clone(), count, "mana")
}

/// Staging-buffer readback of `state.agent_alive_buf` as u32.
pub fn read_alive(state: &mut GeneratedRuntime, count: usize) -> Vec<u32> {
    read_u32(state, &state.agent_alive_buf.clone(), count, "alive")
}

/// Staging-buffer readback of `state.agent_creature_type_buf` as u32.
pub fn read_creature_types(state: &mut GeneratedRuntime, count: usize) -> Vec<u32> {
    read_u32(state, &state.agent_creature_type_buf.clone(), count, "creature_type")
}

/// Staging-buffer readback of `state.agent_pos_buf` as stride-16
/// `[f32; 4]` rows (x, y, z, pad).
pub fn read_positions(state: &mut GeneratedRuntime, count: usize) -> Vec<[f32; 4]> {
    let bytes = (count as u64 * 16).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::pos_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("edgeworld::pos_readback") },
    );
    let buf = state.agent_pos_buf.clone();
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[[f32; 4]] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

/// Generic f32-column staging readback.
fn read_f32(
    state: &mut GeneratedRuntime,
    buf: &wgpu::Buffer,
    count: usize,
    label: &str,
) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::f32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some(&format!("edgeworld::{label}_readback")),
        },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}

/// Generic u32-column staging readback.
fn read_u32(
    state: &mut GeneratedRuntime,
    buf: &wgpu::Buffer,
    count: usize,
    label: &str,
) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::u32_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor {
            label: Some(&format!("edgeworld::{label}_readback")),
        },
    );
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map_async"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
