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
/// creature_type discriminant for `Wolf` (= 2; W sorts after Survivor so
/// the existing `== 1` survivor guards keep their discriminant).
pub const CT_WOLF: u32 = 2;

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
/// ~6-unit perception ring of the inner food grid) at zero hunger;
/// `[n_food+n_survivors..+n_wolves)` are Wolves placed on the world rim
/// (radius ~world_half*0.95) away from the central survivor cluster, at
/// zero hunger and alive.
///
/// Both the boom/bust pin and the render test call this so the seeding
/// stays DRY and identical. `world_half` is the half-extent of the
/// square world (`world` spans `[-world_half, world_half]`).
pub fn seed_world(
    state: &mut GeneratedRuntime,
    n_survivors: usize,
    n_food: usize,
    n_wolves: usize,
    world_half: f32,
) {
    let n = n_food + n_survivors + n_wolves;
    let food_base = 0usize;
    let survivor_base = n_food;
    let wolf_base = n_food + n_survivors;

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

    // Wolves on the world rim, evenly spaced around the perimeter at
    // radius ~world_half*0.95 — well outside the central survivor
    // cluster (rim 0.7*world_half). Zero hunger, alive. Deterministic.
    let wolf_r = world_half * 0.95;
    let w_denom = (n_wolves.max(1)) as f32;
    for w in 0..n_wolves {
        let slot = wolf_base + w;
        let theta = (w as f32) / w_denom * std::f32::consts::TAU;
        let wx = wolf_r * theta.cos();
        let wz = wolf_r * theta.sin();
        positions[slot] = [wx, 0.0, wz, 0.0];
        types[slot] = CT_WOLF;
        hunger[slot] = 0.0;
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

/// Seed a REPRODUCTION world deterministically (Phase 3 Task 1).
///
/// Each "breeder" survivor is pre-linked 1:1 to a unique DEAD "offspring"
/// slot via the `engaged_with` column. A well-fed breeder revives ITS OWN
/// offspring slot (Reproduce → Born → BornRevive), so no two breeders ever
/// target the same slot — the revive is allocation-race-free.
///
/// Slot layout (total N = `n_food + 2 * n_breeders`):
///   `[0..n_food)`                              FoodNodes (type 0, alive),
///   `[n_food..n_food+n_breeders)`              Breeders (type 1, alive,
///                                              hunger 0, clustered on the
///                                              food oasis so they stay fed),
///   `[n_food+n_breeders..n_food+2*n_breeders)` Offspring slots (type 1,
///                                              **alive = 0**, hunger 0).
///
/// Breeder N's `engaged_with` cell (written as a raw u32 absolute slot
/// index — edgeworld reads it directly, no OptAgentId +1 sentinel) points
/// at offspring slot N. Deterministic, no rng / time.
pub fn seed_repro_world(
    state: &mut GeneratedRuntime,
    n_breeders: usize,
    n_food: usize,
    world_half: f32,
) {
    seed_full_world(state, n_breeders, n_food, 0, world_half);
}

/// Seed a FULL ecosystem world deterministically (Phase 3 Task 2):
/// reproduction (breeders + pre-linked dead offspring slots) AND predators
/// (a wolf pack on the rim) together, so the population can both GROW (via
/// births) and be CULLED (via predation + starvation) — the living
/// oscillation cycle.
///
/// Slot layout (total N = `n_food + 2 * n_breeders + n_wolves`):
///   `[0..n_food)`                              FoodNodes (type 0, alive),
///   `[n_food..n_food+n_breeders)`              Breeders (type 1, alive,
///                                              hunger 0, on the oasis),
///   `[n_food+n_breeders..n_food+2*n_breeders)` Offspring slots (type 1,
///                                              **alive = 0**, hunger 0),
///   `[..+n_wolves)`                            Wolves (type 2, alive,
///                                              hunger 0, on the world rim).
///
/// Breeder N is wired 1:1 to offspring slot N via `engaged_with` (raw u32
/// absolute slot index). Wolves are placed exactly as in `seed_world`:
/// evenly spaced on the rim at radius ~`world_half*0.95`. Deterministic,
/// no rng / time.
pub fn seed_full_world(
    state: &mut GeneratedRuntime,
    n_breeders: usize,
    n_food: usize,
    n_wolves: usize,
    world_half: f32,
) {
    let n = n_food + 2 * n_breeders + n_wolves;
    let food_base = 0usize;
    let breeder_base = n_food;
    let offspring_base = n_food + n_breeders;
    let wolf_base = n_food + 2 * n_breeders;

    let mut positions: Vec<[f32; 4]> = vec![[0.0; 4]; n];
    let mut types: Vec<u32> = vec![CT_FOOD; n];
    let mut alive: Vec<u32> = vec![1u32; n];
    let hunger: Vec<f32> = vec![0.0f32; n];
    let mut mana: Vec<f32> = vec![0.0f32; n];
    let hp: Vec<f32> = vec![1.0f32; n];
    let mut engaged: Vec<u32> = vec![0u32; n];

    // Food spread BROADLY across the inner world on a golden-angle spiral
    // (full larder). A wide food field — rather than one tight oasis — lets
    // the breeders disperse over the arena instead of stacking on a single
    // pixel, so an arriving wolf cannot wipe the whole breeding core in one
    // pass: it culls the locals and the rest flee. Every breeder always has
    // food within perception/eat range, so the core stays well-fed (hunger
    // below birth_hunger_max) and keeps re-reviving culled offspring — the
    // recovery half of the cycle.
    let food_rim = world_half * 0.5; // food field half-extent (inside the
                                     // 0.95*world_half wolf rim).
    let golden = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt()); // ~2.399963
    let fdenom = (n_food.max(1)) as f32;
    for f in 0..n_food {
        let slot = food_base + f;
        let frac = ((f as f32) + 0.5) / fdenom;
        let r = if n_food <= 1 { 0.0 } else { food_rim * frac.sqrt() };
        let theta = (f as f32) * golden;
        positions[slot] = [r * theta.cos(), 0.0, r * theta.sin(), 0.0];
        types[slot] = CT_FOOD;
        mana[slot] = FOOD_MAX; // full larder — breeders never go hungry.
    }

    // Breeders spread across the SAME inner field on an interleaved spiral at
    // zero hunger, alive. Each points at its unique offspring slot. Spreading
    // them (rather than stacking) gives them room to flee a closing wolf
    // (flee_speed > wolf_move_speed), so the breeding core survives the cull
    // and persists to re-revive offspring.
    let breeder_rim = world_half * 0.34;
    let bdenom = (n_breeders.max(1)) as f32;
    for b in 0..n_breeders {
        let slot = breeder_base + b;
        let frac = ((b as f32) + 0.5) / bdenom;
        let r = if n_breeders <= 1 { 0.0 } else { breeder_rim * frac.sqrt() };
        // Offset the breeder spiral half a golden step off the food spiral so
        // breeders interleave with — rather than land exactly on — food nodes.
        let theta = (b as f32) * golden + 0.5 * golden;
        positions[slot] = [r * theta.cos(), 0.0, r * theta.sin(), 0.0];
        types[slot] = CT_SURVIVOR;
        // engaged_with = absolute offspring slot index (raw, no sentinel).
        engaged[slot] = (offspring_base + b) as u32;
    }

    // Offspring slots: type 1, DEAD (alive = 0), hunger 0. Parked off the
    // field so they don't pre-feed; BornRevive teleports them onto the
    // parent's position on birth (so a newborn spawns in its parent's fed
    // neighbourhood, then forages/flees like any survivor).
    for o in 0..n_breeders {
        let slot = offspring_base + o;
        let theta = (o as f32) / bdenom * std::f32::consts::TAU;
        positions[slot] = [world_half * 0.5 * theta.cos(), 0.0, world_half * 0.5 * theta.sin(), 0.0];
        types[slot] = CT_SURVIVOR;
        alive[slot] = 0; // dead until a breeder revives it.
    }

    // Wolves on the world rim, evenly spaced around the perimeter at radius
    // ~world_half*0.95 — well outside the central breeder/offspring cluster.
    // Zero hunger, alive. Identical placement to seed_world's wolf layout.
    let wolf_r = world_half * 0.95;
    let w_denom = (n_wolves.max(1)) as f32;
    for w in 0..n_wolves {
        let slot = wolf_base + w;
        let theta = (w as f32) / w_denom * std::f32::consts::TAU;
        positions[slot] = [wolf_r * theta.cos(), 0.0, wolf_r * theta.sin(), 0.0];
        types[slot] = CT_WOLF;
        // hunger already 0 (vec init); alive already 1.
    }

    let q = &state.gpu.queue;
    q.write_buffer(&state.agent_pos_buf, 0, bytemuck::cast_slice(&positions));
    q.write_buffer(&state.agent_creature_type_buf, 0, bytemuck::cast_slice(&types));
    q.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    q.write_buffer(&state.agent_hunger_buf, 0, bytemuck::cast_slice(&hunger));
    q.write_buffer(&state.agent_mana_buf, 0, bytemuck::cast_slice(&mana));
    q.write_buffer(&state.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
    q.write_buffer(&state.agent_engaged_with_buf, 0, bytemuck::cast_slice(&engaged));
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

/// Staging-buffer readback of the hand-rolled FEAR column — Phase 2 Task 2
/// repurposes the free `shield_hp` f32 SoA column as each survivor's
/// decaying fear level (RISES +1.0 per wolf sighting via Perceive, DECAYS
/// *0.90/tick via DecayFear). The behaviour gates (Flee / SeekFood-
/// suppression) read THIS column — the `threats` belief's value is not
/// readable from a physics rule (see edgeworld.sim PATH B finding), so the
/// belief stays a host-readable observable while the column drives action.
pub fn read_fear(state: &mut GeneratedRuntime, count: usize) -> Vec<f32> {
    read_f32(state, &state.agent_shield_hp_buf.clone(), count, "fear")
}

/// Staging-buffer readback of `state.agent_creature_type_buf` as u32.
pub fn read_creature_types(state: &mut GeneratedRuntime, count: usize) -> Vec<u32> {
    read_u32(state, &state.agent_creature_type_buf.clone(), count, "creature_type")
}

/// Staging-buffer readback of the `threats` belief storage —
/// `view_storage_threats_primary_buf` holds `count` packed f32 cells
/// (one per agent slot, keyed on the observer). Mirrors the readback in
/// `threat_stresstest_pin.rs::read_threats_primary_*`. Phase 2 Task 1
/// reads this to assert the per-survivor threat level rises on wolf
/// sightings and decays once the wolf leaves; Task 2/3 will read the
/// same buffer to gate flee behaviour.
pub fn read_threats(state: &mut GeneratedRuntime, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("edgeworld::threats_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state.gpu.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("edgeworld::threats_readback") },
    );
    encoder.copy_buffer_to_buffer(
        &state.view_storage_threats_primary_buf,
        0,
        &staging,
        0,
        bytes,
    );
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
