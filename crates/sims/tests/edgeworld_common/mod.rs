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

/// FoodNode quantity ceiling used by the seeder (mirrors
/// `config.edgeworld.food_max`). Frame brightness in the render test
/// scales by this.
pub const FOOD_MAX: f32 = 5.0;

/// Initial FoodNode quantity at seed time (the standing larder). Lower
/// than `FOOD_MAX` so the larder is a finite buffer that can be drawn
/// down by the seeded overshoot crowd, producing the crash window.
pub const FOOD_SEED: f32 = 4.0;

/// Seed a compact survival world deterministically (no rng / time).
///
/// Slot layout: `[0..n_food)` are FoodNodes spread on a grid across the
/// inner world at full quantity; `[n_food..n_food+n_survivors)` are
/// Survivors arranged in concentric rings near the centre (inside the
/// ~6-unit perception ring of the inner food grid) at zero hunger.
///
/// Both the boom/bust pin and the render test call this so the seeding
/// stays DRY and identical. `world_half` is the half-extent of the
/// square world (`world` spans `[-world_half, world_half]`).
pub fn seed_world(
    state: &mut GeneratedRuntime,
    n_survivors: usize,
    n_food: usize,
    world_half: f32,
) {
    let n = n_food + n_survivors;
    let food_base = 0usize;
    let survivor_base = n_food;

    let mut positions: Vec<[f32; 4]> = vec![[0.0; 4]; n];
    let mut types: Vec<u32> = vec![CT_FOOD; n];
    let alive: Vec<u32> = vec![1u32; n];
    let mut hunger: Vec<f32> = vec![0.0f32; n];
    let mut mana: Vec<f32> = vec![0.0f32; n];
    let hp: Vec<f32> = vec![1.0f32; n]; // for the starvation fallback path

    // Food in a tight cluster near the origin (a single "oasis"). The
    // perception ring (~6 world units) caps foraging, so survivors that
    // start near this oasis can feed; survivors seeded out toward the
    // rim are beyond perception and cannot reach it — they form the
    // overshoot that starves in the crash. This spatial heterogeneity
    // is what produces a STABLE remnant (the oasis survivors) instead of
    // an all-or-nothing flip.
    let food_span = 3.0; // oasis radius — spread so remnant settles in
                         // distinct clusters (legible in the render) while
                         // still inside survivor perception of the centre.
    for f in 0..n_food {
        let slot = food_base + f;
        let theta = (f as f32) / (n_food.max(1) as f32) * std::f32::consts::TAU;
        let r = if n_food <= 1 { 0.0 } else { food_span };
        let fx = r * theta.cos();
        let fz = r * theta.sin();
        positions[slot] = [fx, 0.0, fz, 0.0];
        types[slot] = CT_FOOD;
        // Seed food at a modest standing buffer (not the ceiling). The
        // overshoot crowd strips this buffer down in the opening ticks;
        // it then recovers via regrowth to the level that sustains the
        // post-crash remnant.
        mana[slot] = FOOD_SEED;
    }

    // Survivors spread on a deterministic golden-angle spiral spanning
    // the inner world (kept inside the ~6-unit perception ring of the
    // oasis so foraging works), with a GRADED initial hunger ramp.
    //
    // The food pool behaves as a shared larder (the engine's N-to-1
    // spatial fan-in feeds every survivor near-equally each tick, and
    // the eat query spans the compact world), so position alone cannot
    // single out a remnant — every survivor's hunger would otherwise
    // move in lockstep and the population would be all-or-nothing. The
    // graded initial-hunger RAMP breaks that symmetry: survivors seeded
    // well above the starvation threshold (1.0) are the overshoot — they
    // cannot be pulled back under in the opening ticks and are culled
    // (the crash); survivors seeded low are the remnant the recovered
    // larder then sustains at carrying capacity. The ramp climbs 0.0 →
    // RAMP_TOP across the cohort; RAMP_TOP sets how deep the cull goes
    // (higher → smaller remnant). Deterministic, no rng.
    let ramp_top = 2.8;
    let rim = world_half * 0.7; // survivors spread across the inner world,
                                // inside SeekFood perception of the oasis.
    let golden = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt()); // ~2.399963
    let denom = (n_survivors.max(1)) as f32;
    for s in 0..n_survivors {
        let slot = survivor_base + s;
        // sqrt spacing → uniform areal density from centre to rim.
        let frac = ((s as f32) + 0.5) / denom;
        let r = rim * frac.sqrt();
        let theta = (s as f32) * golden;
        let sx = r * theta.cos();
        let sz = r * theta.sin();
        positions[slot] = [sx, 0.0, sz, 0.0];
        types[slot] = CT_SURVIVOR;
        hunger[slot] = ramp_top * (s as f32) / denom;
    }

    state
        .gpu
        .queue
        .write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&positions));
    state.gpu.queue.write_buffer(
        &state.agent_creature_type_buf,
        0,
        bytemuck::cast_slice(&types),
    );
    state
        .gpu
        .queue
        .write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    state
        .gpu
        .queue
        .write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&hunger));
    state
        .gpu
        .queue
        .write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&mana));
    state
        .gpu
        .queue
        .write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
}

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
