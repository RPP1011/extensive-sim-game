# GPU Terrain Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move voxel terrain materialization (base layers, caves, rivers, surface features) from CPU to GPU compute shader so the renderer can generate 64³ chunks in milliseconds instead of ~150ms each.

**Architecture:** A compute shader takes `(chunk_pos, seed)` as push constants and reads region plan data from storage buffers. It writes the resulting voxel materials directly into a storage buffer that's then copied into the existing `VoxelGrid` 3D texture used for rendering. CPU and GPU stay deterministic by sharing the same algorithm — both generate identical chunks from identical inputs, so gameplay code can lazily generate chunks on the CPU when it needs to query specific voxels without ever syncing.

**Tech Stack:** Rust, Vulkan via `ash`, GLSL compute shaders compiled by `shaderc` in build.rs, existing `voxel_engine` crate.

**Key constraint:** The Rust hash uses `u64` arithmetic with the constant `6364136223846793005` (>u32::MAX). GLSL only reliably supports `uint`. We replace `hash_3d` on both sides with a `u32`-only PCG-based hash so CPU and GPU produce bit-identical output.

---

## Phase 1: Compute pipeline plumbing

### Task 1: Add u32-only hash function

**Files:**
- Modify: `src/world_sim/terrain/noise.rs` (add new fn, keep old for now)
- Test: `src/world_sim/terrain/noise.rs` (in mod tests)

- [ ] **Step 1: Write the failing test**

Add this to the existing `mod tests` block in `src/world_sim/terrain/noise.rs`:

```rust
#[test]
fn hash_u32_deterministic_and_uniform() {
    // Same input → same output
    assert_eq!(hash_u32(10, 20, 30, 42), hash_u32(10, 20, 30, 42));
    // Different input → different output (with high probability)
    assert_ne!(hash_u32(10, 20, 30, 42), hash_u32(11, 20, 30, 42));
    // Output spans the full u32 range (sample mean ~ 0.5)
    let mut sum = 0.0f64;
    let n = 10_000;
    for i in 0..n {
        sum += hash_u32_to_f32(i, i * 7, i / 3, 999) as f64;
    }
    let mean = sum / n as f64;
    assert!((mean - 0.5).abs() < 0.05, "hash_u32 distribution skewed: mean={mean}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib -- terrain::noise::tests::hash_u32_deterministic_and_uniform`
Expected: FAIL (function not defined)

- [ ] **Step 3: Add the new hash functions**

Add at the top of `src/world_sim/terrain/noise.rs` (above the existing `hash_3d`):

```rust
/// PCG-based deterministic hash using only u32 arithmetic so the same
/// algorithm can be ported byte-for-byte to GLSL (no int64 extension needed).
#[inline]
pub fn hash_u32(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed;
    h = h.wrapping_mul(0x85ebca6b).wrapping_add(x as u32);
    h = h ^ (h >> 16);
    h = h.wrapping_mul(0xc2b2ae35).wrapping_add(y as u32);
    h = h ^ (h >> 16);
    h = h.wrapping_mul(0x27d4eb2f).wrapping_add(z as u32);
    h = h ^ (h >> 16);
    h
}

/// u32 hash → f32 in [0, 1).
#[inline]
pub fn hash_u32_to_f32(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    hash_u32(x, y, z, seed) as f32 / u32::MAX as f32
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --lib -- terrain::noise::tests::hash_u32_deterministic_and_uniform`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/terrain/noise.rs
git commit -m "feat(terrain): add u32-only hash for GPU/CPU parity"
```

---

### Task 2: Switch noise functions to use the u32 hash

**Files:**
- Modify: `src/world_sim/terrain/noise.rs` (replace hash_f32 calls in value_noise_*)

- [ ] **Step 1: Replace `hash_3d`/`hash_f32` callers in `value_noise_2d` and `value_noise_3d`**

Edit `src/world_sim/terrain/noise.rs`. Change `value_noise_2d`:

```rust
pub fn value_noise_2d(x: f32, y: f32, seed: u64, scale: f32) -> f32 {
    let sx = x / scale;
    let sy = y / scale;
    let ix = sx.floor() as i32;
    let iy = sy.floor() as i32;
    let fx = smoothstep(sx - sx.floor());
    let fy = smoothstep(sy - sy.floor());
    let s32 = seed as u32 ^ (seed >> 32) as u32;
    let h00 = hash_u32_to_f32(ix, iy, 0, s32);
    let h10 = hash_u32_to_f32(ix + 1, iy, 0, s32);
    let h01 = hash_u32_to_f32(ix, iy + 1, 0, s32);
    let h11 = hash_u32_to_f32(ix + 1, iy + 1, 0, s32);
    let a = h00 + (h10 - h00) * fx;
    let b = h01 + (h11 - h01) * fx;
    a + (b - a) * fy
}
```

Change `value_noise_3d` analogously: compute `s32` once, replace all 8 `hash_f32(..., seed)` calls with `hash_u32_to_f32(..., s32)`.

- [ ] **Step 2: Run noise tests to verify they still pass**

Run: `cargo test --lib -- terrain::noise::tests`
Expected: All pass (the smoothness tests don't care about the underlying hash, only that nearby samples are close)

- [ ] **Step 3: Run terrain tests to verify generation still works**

Run: `cargo test --lib -- terrain::`
Expected: All pass. (Generation output will differ from old seeds, but the structural properties — biome variety, river presence, settlement count — should still hold.)

- [ ] **Step 4: Re-render the gallery to confirm visual sanity**

Run: `cargo test --test voxel_render_inspect terrain_gallery --features app --release -- --nocapture`
Expected: PASS. Open `generated/render_tests/gallery.png` and verify all 8 biomes still look reasonable.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/terrain/noise.rs
git commit -m "refactor(terrain): switch noise functions to u32 hash"
```

---

### Task 3: Create the terrain compute shader skeleton

**Files:**
- Create: `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`
- Create: `/home/ricky/Projects/voxel_engine/shaders/terrain_common.glsl` (shared helpers)

- [ ] **Step 1: Create the shared helpers file**

Write `/home/ricky/Projects/voxel_engine/shaders/terrain_common.glsl`:

```glsl
// Shared GLSL helpers for terrain generation. Mirrors src/world_sim/terrain/noise.rs.

#ifndef TERRAIN_COMMON_GLSL
#define TERRAIN_COMMON_GLSL

// PCG-based hash matching Rust noise::hash_u32 byte-for-byte.
uint hash_u32(int x, int y, int z, uint seed) {
    uint h = seed;
    h = h * 0x85ebca6bu + uint(x);
    h = h ^ (h >> 16);
    h = h * 0xc2b2ae35u + uint(y);
    h = h ^ (h >> 16);
    h = h * 0x27d4eb2fu + uint(z);
    h = h ^ (h >> 16);
    return h;
}

float hash_f32(int x, int y, int z, uint seed) {
    return float(hash_u32(x, y, z, seed)) / 4294967295.0;
}

float smoothstep_t(float t) {
    return t * t * (3.0 - 2.0 * t);
}

float value_noise_2d(float x, float y, uint seed, float scale) {
    float sx = x / scale;
    float sy = y / scale;
    int ix = int(floor(sx));
    int iy = int(floor(sy));
    float fx = smoothstep_t(sx - floor(sx));
    float fy = smoothstep_t(sy - floor(sy));
    float h00 = hash_f32(ix,     iy,     0, seed);
    float h10 = hash_f32(ix + 1, iy,     0, seed);
    float h01 = hash_f32(ix,     iy + 1, 0, seed);
    float h11 = hash_f32(ix + 1, iy + 1, 0, seed);
    float a = h00 + (h10 - h00) * fx;
    float b = h01 + (h11 - h01) * fx;
    return a + (b - a) * fy;
}

float value_noise_3d(float x, float y, float z, uint seed, float scale) {
    float sx = x / scale;
    float sy = y / scale;
    float sz = z / scale;
    int ix = int(floor(sx));
    int iy = int(floor(sy));
    int iz = int(floor(sz));
    float fx = smoothstep_t(sx - floor(sx));
    float fy = smoothstep_t(sy - floor(sy));
    float fz = smoothstep_t(sz - floor(sz));
    float c000 = hash_f32(ix,     iy,     iz,     seed);
    float c100 = hash_f32(ix + 1, iy,     iz,     seed);
    float c010 = hash_f32(ix,     iy + 1, iz,     seed);
    float c110 = hash_f32(ix + 1, iy + 1, iz,     seed);
    float c001 = hash_f32(ix,     iy,     iz + 1, seed);
    float c101 = hash_f32(ix + 1, iy,     iz + 1, seed);
    float c011 = hash_f32(ix,     iy + 1, iz + 1, seed);
    float c111 = hash_f32(ix + 1, iy + 1, iz + 1, seed);
    float a0 = c000 + (c100 - c000) * fx;
    float b0 = c010 + (c110 - c010) * fx;
    float a1 = c001 + (c101 - c001) * fx;
    float b1 = c011 + (c111 - c011) * fx;
    float c0 = a0 + (b0 - a0) * fy;
    float c1 = a1 + (b1 - a1) * fy;
    return c0 + (c1 - c0) * fz;
}

float fbm_2d(float x, float y, uint seed, int octaves, float lacunarity, float gain) {
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    float max_amp = 0.0;
    for (int i = 0; i < octaves; i++) {
        sum += amp * value_noise_2d(x * freq, y * freq, seed + uint(i * 31337), 1.0);
        max_amp += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return sum / max_amp;
}

#endif // TERRAIN_COMMON_GLSL
```

- [ ] **Step 2: Create the materialization shader skeleton**

Write `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`:

```glsl
#version 450

#include "terrain_common.glsl"

// 64³ = 262144 voxels per chunk. 8×8×8 workgroup → 512 invocations per group.
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

// Output: u8 material per voxel, packed into uint storage buffer (4 voxels per uint).
// Index by linear voxel index, then unpack/pack the byte.
layout(set = 0, binding = 0) buffer OutputBuf {
    uint voxels[]; // length = chunk_size^3 / 4
} out_buf;

layout(push_constant) uniform PushConstants {
    ivec4 chunk_pos;       // (cx, cy, cz, _)
    uvec4 params;          // (seed, chunk_size, _, _)
} pc;

const uint MAT_AIR = 0u;
const uint MAT_STONE = 2u;
const uint MAT_DIRT = 1u;

void write_voxel(uint idx, uint material) {
    uint word_idx = idx / 4u;
    uint byte_off = (idx & 3u) * 8u;
    uint mask = 0xffu << byte_off;
    uint val = (material & 0xffu) << byte_off;
    // Read-modify-write — safe because each invocation owns its own voxel.
    uint old = out_buf.voxels[word_idx];
    out_buf.voxels[word_idx] = (old & ~mask) | val;
}

void main() {
    uint cs = pc.params.y;
    uvec3 lp = gl_GlobalInvocationID;
    if (lp.x >= cs || lp.y >= cs || lp.z >= cs) return;

    uint idx = lp.z * cs * cs + lp.y * cs + lp.x;

    int vx = pc.chunk_pos.x * int(cs) + int(lp.x);
    int vy = pc.chunk_pos.y * int(cs) + int(lp.y);
    int vz = pc.chunk_pos.z * int(cs) + int(lp.z);

    // SKELETON: write a stone block below z=0, air above. Real materializer comes in Task 5.
    uint mat = (vz < 0) ? MAT_STONE : MAT_AIR;
    write_voxel(idx, mat);
}
```

- [ ] **Step 3: Build voxel_engine to compile the shader**

Run: `cd /home/ricky/Projects/voxel_engine && cargo build --features compile-shaders 2>&1 | tail -10`
Expected: Build succeeds. The new `.spv` file appears in `shaders/compiled/terrain_materialize.comp.spv`.

If the build fails because the compiler doesn't follow `#include`, replace the `#include "terrain_common.glsl"` line in the shader by inlining the contents of `terrain_common.glsl` directly. (Vulkan GLSL doesn't natively support `#include` — shaderc supports it via `set_include_callback`. Check `voxel_engine/build.rs` to see if includes are configured. If not, do the inline workaround.)

- [ ] **Step 4: Verify the .spv exists**

Run: `ls /home/ricky/Projects/voxel_engine/shaders/compiled/terrain_materialize.comp.spv`
Expected: file exists.

- [ ] **Step 5: Commit**

```bash
cd /home/ricky/Projects/voxel_engine
git add shaders/terrain_materialize.comp shaders/terrain_common.glsl shaders/compiled/terrain_materialize.comp.spv
git commit -m "feat: add terrain materialization compute shader skeleton"
```

---

### Task 4: Rust-side terrain compute pipeline wrapper

**Files:**
- Create: `/home/ricky/Projects/voxel_engine/src/terrain_compute.rs`
- Modify: `/home/ricky/Projects/voxel_engine/src/lib.rs` (export module)

- [ ] **Step 1: Create the terrain_compute module**

Write `/home/ricky/Projects/voxel_engine/src/terrain_compute.rs`:

```rust
//! GPU terrain materialization pipeline.
//!
//! Dispatches a compute shader that fills a chunk's worth of voxel materials
//! based on chunk position, seed, and (eventually) region plan data.

use anyhow::{Context, Result};
use ash::vk;

use crate::vulkan::allocator::{VulkanAllocator, AllocatedBuffer};
use crate::vulkan::instance::VulkanContext;

const CHUNK_SIZE: u32 = 64;
const CHUNK_VOLUME: u32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const OUTPUT_WORDS: u32 = CHUNK_VOLUME / 4; // 4 u8 voxels per u32

pub struct TerrainComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    output_buffer: AllocatedBuffer,
    shader_module: vk::ShaderModule,
}

impl TerrainComputePipeline {
    pub fn new(ctx: &VulkanContext, alloc: &mut VulkanAllocator) -> Result<Self> {
        let device = ctx.device();

        // Load precompiled SPIR-V from OUT_DIR/shaders.
        let spirv_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/terrain_materialize.comp.spv"));
        let spirv_words: Vec<u32> = spirv_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe { device.create_shader_module(&shader_ci, None) }
            .context("create terrain compute shader")?;

        // Descriptor layout: single STORAGE_BUFFER at binding 0.
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let bindings = [binding];
        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_ci, None) }
            .context("descriptor set layout")?;

        // Push constants: ivec4 + uvec4 = 32 bytes.
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(32);

        let set_layouts = [descriptor_set_layout];
        let push_ranges = [push_range];
        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_ci, None) }
            .context("pipeline layout")?;

        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let pipeline_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_ci], None)
        }
        .map_err(|(_, e)| e)
        .context("compute pipeline")?[0];

        // Descriptor pool with one storage buffer descriptor.
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1);
        let pool_sizes = [pool_size];
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }
            .context("descriptor pool")?;

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&set_layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("alloc descriptor set")?[0];

        // Output buffer: STORAGE_BUFFER + TRANSFER_SRC, host-visible for readback during testing.
        let buffer_size = (OUTPUT_WORDS * 4) as vk::DeviceSize;
        let output_buffer = alloc
            .create_buffer(
                ctx,
                buffer_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .context("output buffer")?;

        // Bind buffer to descriptor.
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(output_buffer.buffer)
            .offset(0)
            .range(buffer_size);
        let buffer_infos = [buffer_info];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buffer_infos);
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            output_buffer,
            shader_module,
        })
    }

    /// Dispatch the compute shader for one chunk and return the materials as a flat Vec.
    /// Index ordering: `[z * cs * cs + y * cs + x]`.
    pub fn generate_chunk(
        &self,
        ctx: &VulkanContext,
        chunk_pos: [i32; 3],
        seed: u32,
    ) -> Result<Vec<u8>> {
        let device = ctx.device();

        // Allocate a one-shot command buffer.
        let cmd_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(ctx.queue_family_index())
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_ci, None) }?;
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&cmd_alloc) }?[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device.begin_command_buffer(cmd, &begin)?;
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            // Push constants: ivec4 chunk_pos, uvec4 params (seed, chunk_size, _, _).
            let mut push = [0u8; 32];
            push[0..4].copy_from_slice(&chunk_pos[0].to_le_bytes());
            push[4..8].copy_from_slice(&chunk_pos[1].to_le_bytes());
            push[8..12].copy_from_slice(&chunk_pos[2].to_le_bytes());
            push[16..20].copy_from_slice(&seed.to_le_bytes());
            push[20..24].copy_from_slice(&CHUNK_SIZE.to_le_bytes());
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );

            // Dispatch: 64³ voxels with 8×8×8 workgroups = 8×8×8 groups.
            let groups = CHUNK_SIZE / 8;
            device.cmd_dispatch(cmd, groups, groups, groups);

            device.end_command_buffer(cmd)?;
        }

        // Submit and wait.
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe {
            device.queue_submit(ctx.queue(), &[submit], vk::Fence::null())?;
            device.queue_wait_idle(ctx.queue())?;
            device.destroy_command_pool(cmd_pool, None);
        }

        // Read back the buffer.
        let buffer_size = (CHUNK_VOLUME) as usize;
        let mut materials = vec![0u8; buffer_size];
        let mapped = self.output_buffer.mapped_ptr.context("buffer not mapped")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped as *const u8,
                materials.as_mut_ptr(),
                buffer_size,
            );
        }
        Ok(materials)
    }

    pub fn destroy(self, ctx: &VulkanContext, alloc: &mut VulkanAllocator) {
        let device = ctx.device();
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_shader_module(self.shader_module, None);
        }
        self.output_buffer.destroy(ctx, alloc);
    }
}
```

- [ ] **Step 2: Export from lib.rs**

Edit `/home/ricky/Projects/voxel_engine/src/lib.rs` and add:

```rust
pub mod terrain_compute;
```

If the file uses a different module organization, add the line near the other `pub mod` declarations.

- [ ] **Step 3: Verify the AllocatedBuffer API matches**

Check `/home/ricky/Projects/voxel_engine/src/vulkan/allocator.rs` for the actual `AllocatedBuffer` struct and `create_buffer` signature. If `mapped_ptr` is named differently (e.g., `mapping`, `mapped`, `host_ptr`), fix the call sites in `terrain_compute.rs`. If `create_buffer` returns a different type or takes different params, adjust accordingly. If the allocator doesn't auto-map host-visible buffers, add a `map_memory` step before reading.

Run: `grep -n "AllocatedBuffer\|fn create_buffer\|mapped" /home/ricky/Projects/voxel_engine/src/vulkan/allocator.rs | head -20`

Then update `terrain_compute.rs` to match the actual API.

- [ ] **Step 4: Build voxel_engine**

Run: `cd /home/ricky/Projects/voxel_engine && cargo build 2>&1 | tail -15`
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
cd /home/ricky/Projects/voxel_engine
git add src/terrain_compute.rs src/lib.rs
git commit -m "feat: add terrain compute pipeline wrapper"
```

---

### Task 5: Smoke test the GPU pipeline produces a recognizable shape

**Files:**
- Create: `/home/ricky/Projects/voxel_engine/tests/terrain_compute_smoke.rs`

- [ ] **Step 1: Write a smoke test**

```rust
//! Smoke test: GPU compute generates a chunk with the expected skeleton output
//! (stone below z=0, air above). Skips if no Vulkan GPU available.

use voxel_engine::terrain_compute::TerrainComputePipeline;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;

const CHUNK_SIZE: usize = 64;

#[test]
fn gpu_skeleton_chunk_has_stone_below_zero() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mut alloc = VulkanAllocator::new(&ctx).expect("alloc");
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    // Chunk at z=-1 should be entirely stone.
    let mats = pipeline.generate_chunk(&ctx, [0, 0, -1], 42).expect("dispatch");
    assert_eq!(mats.len(), CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);
    let stone_count = mats.iter().filter(|&&m| m == 2).count();
    assert!(stone_count > mats.len() * 9 / 10,
        "expected mostly stone in below-zero chunk, got {}/{}", stone_count, mats.len());

    // Chunk at z=1 should be entirely air.
    let mats = pipeline.generate_chunk(&ctx, [0, 0, 1], 42).expect("dispatch");
    let air_count = mats.iter().filter(|&&m| m == 0).count();
    assert_eq!(air_count, mats.len(), "above-zero chunk should be all air");

    pipeline.destroy(&ctx, &mut alloc);
}
```

- [ ] **Step 2: Run the test**

Run: `cd /home/ricky/Projects/voxel_engine && cargo test --test terrain_compute_smoke -- --nocapture`
Expected: PASS (or skip if no Vulkan).

If it fails, the most likely culprits are: (a) the byte-packing in `write_voxel` is wrong, (b) the buffer isn't actually host-visible after dispatch, (c) the queue family doesn't support compute. Fix and retry.

- [ ] **Step 3: Commit**

```bash
cd /home/ricky/Projects/voxel_engine
git add tests/terrain_compute_smoke.rs
git commit -m "test: smoke test for terrain compute pipeline"
```

---

## Phase 2: Port noise + base materializer to GLSL

### Task 6: Port `surface_height_at` to GLSL

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`

- [ ] **Step 1: Add surface_height_at function to the shader**

The CPU version (`src/world_sim/terrain/materialize.rs:21-46`) reads the region plan to get `(base_height, terrain)`. For Phase 2 we don't have the region plan on GPU yet — hardcode `base_height = 0.3` and `large_amp/medium_amp/small_amp = 30/12/3` (forest defaults). We'll wire region plan in Task 8.

Add to `terrain_materialize.comp` (above `main`):

```glsl
int surface_height_at(float vx, float vy, uint seed) {
    float base_height = 0.3; // TODO: sample from region plan in Task 8

    float large = fbm_2d(vx * 0.004, vy * 0.004, seed + 0xface_cafeu, 4, 2.0, 0.5);
    float medium = fbm_2d(vx * 0.015, vy * 0.015, seed + 0xdead_beefu, 3, 2.0, 0.5);
    float small = fbm_2d(vx * 0.06, vy * 0.06, seed + 0xcafe_babeu, 2, 2.0, 0.5);

    float large_amp = 30.0;
    float medium_amp = 12.0;
    float small_amp = 3.0;

    float detail = (large * 2.0 - 1.0) * large_amp
                 + (medium * 2.0 - 1.0) * medium_amp
                 + (small * 2.0 - 1.0) * small_amp;

    const int MAX_SURFACE_Z = 2000;
    return int(round(base_height * float(MAX_SURFACE_Z) + detail));
}
```

Note: GLSL doesn't allow `_` in numeric literals — use `0xfacecafeu` etc.

- [ ] **Step 2: Replace the placeholder material in main()**

Change `main()` in `terrain_materialize.comp`:

```glsl
void main() {
    uint cs = pc.params.y;
    uvec3 lp = gl_GlobalInvocationID;
    if (lp.x >= cs || lp.y >= cs || lp.z >= cs) return;

    uint idx = lp.z * cs * cs + lp.y * cs + lp.x;

    int vx = pc.chunk_pos.x * int(cs) + int(lp.x);
    int vy = pc.chunk_pos.y * int(cs) + int(lp.y);
    int vz = pc.chunk_pos.z * int(cs) + int(lp.z);
    uint seed = pc.params.x;

    int surface_z = surface_height_at(float(vx), float(vy), seed);
    int depth = surface_z - vz;

    uint mat;
    if (vz < -500) {
        mat = 3u; // Granite
    } else if (depth > 80) {
        mat = 2u; // Stone (deep)
    } else if (depth > 20) {
        mat = 2u; // Stone (subsoil zone, simplified)
    } else if (depth > 0) {
        mat = 1u; // Dirt (subsoil)
    } else if (depth >= -1) {
        mat = 7u; // Grass
    } else {
        mat = 0u; // Air
    }
    write_voxel(idx, mat);
}
```

- [ ] **Step 3: Recompile shaders**

Run: `cd /home/ricky/Projects/voxel_engine && cargo build --features compile-shaders 2>&1 | tail -10`
Expected: Build succeeds.

- [ ] **Step 4: Update the smoke test to expect a layered chunk**

Edit `/home/ricky/Projects/voxel_engine/tests/terrain_compute_smoke.rs`:

```rust
#[test]
fn gpu_chunk_has_grass_dirt_stone_layers() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mut alloc = VulkanAllocator::new(&ctx).expect("alloc");
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    // Surface chunk: at depth ~0 we should see grass/dirt/stone all present.
    // base_height=0.3 * MAX_SURFACE_Z=2000 = 600. Chunk at cz=9 covers vz=576..640.
    let mats = pipeline.generate_chunk(&ctx, [0, 0, 9], 42).expect("dispatch");
    let grass_count = mats.iter().filter(|&&m| m == 7).count();
    let dirt_count = mats.iter().filter(|&&m| m == 1).count();
    let stone_count = mats.iter().filter(|&&m| m == 2).count();
    let air_count = mats.iter().filter(|&&m| m == 0).count();
    eprintln!("grass={grass_count} dirt={dirt_count} stone={stone_count} air={air_count}");
    assert!(grass_count > 0, "no grass found in surface chunk");
    assert!(dirt_count > 0, "no dirt found in surface chunk");
    assert!(stone_count > 0, "no stone found in surface chunk");
    assert!(air_count > 0, "no air found in surface chunk");

    pipeline.destroy(&ctx, &mut alloc);
}
```

- [ ] **Step 5: Run the test**

Run: `cd /home/ricky/Projects/voxel_engine && cargo test --test terrain_compute_smoke -- --nocapture`
Expected: PASS — terrain has all four expected material types.

- [ ] **Step 6: Commit**

```bash
cd /home/ricky/Projects/voxel_engine
git add shaders/terrain_materialize.comp shaders/compiled/terrain_materialize.comp.spv tests/terrain_compute_smoke.rs
git commit -m "feat: GPU surface_height_at and base layer materialization"
```

---

### Task 7: GPU/CPU parity test for `surface_height_at`

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/tests/terrain_compute_smoke.rs`

- [ ] **Step 1: Add a parity test that compares GPU output to CPU `materialize_chunk`**

This requires the `game` crate as a dev-dependency of voxel_engine, which is a circular dependency we can't have. Instead, write the parity test in the `game` crate.

Create `/home/ricky/Projects/game/.claude/worktrees/sprightly-growing-gadget/tests/gpu_terrain_parity.rs`:

```rust
//! Parity test: GPU compute produces the same materials as CPU materialize_chunk.
//! Required for the GPU pipeline to be a drop-in replacement.

#![cfg(feature = "app")]

use game::world_sim::terrain::{generate_continent, materialize_chunk};
use game::world_sim::voxel::{ChunkPos, CHUNK_SIZE};

use voxel_engine::terrain_compute::TerrainComputePipeline;
use voxel_engine::vulkan::allocator::VulkanAllocator;
use voxel_engine::vulkan::instance::VulkanContext;

#[test]
fn gpu_matches_cpu_for_simple_chunk() {
    let ctx = match VulkanContext::new() {
        Ok(c) => c,
        Err(_) => { eprintln!("SKIP: no Vulkan"); return; }
    };
    let mut alloc = VulkanAllocator::new(&ctx).expect("alloc");
    let pipeline = TerrainComputePipeline::new(&ctx, &mut alloc).expect("pipeline");

    let plan = generate_continent(60, 40, 42);
    let cp = ChunkPos::new(0, 0, 9);
    let cpu_chunk = materialize_chunk(cp, &plan, 42);

    let gpu_mats = pipeline.generate_chunk(&ctx, [cp.x, cp.y, cp.z], 42).expect("dispatch");

    // Compare voxel-by-voxel.
    let mut mismatches = 0;
    let cs = CHUNK_SIZE;
    for lz in 0..cs {
        for ly in 0..cs {
            for lx in 0..cs {
                let cpu_mat = cpu_chunk.get(lx, ly, lz).material as u8;
                let gpu_mat = gpu_mats[lz * cs * cs + ly * cs + lx];
                if cpu_mat != gpu_mat {
                    mismatches += 1;
                }
            }
        }
    }
    let total = cs * cs * cs;
    let pct = mismatches as f32 / total as f32 * 100.0;
    eprintln!("Mismatches: {}/{} ({:.1}%)", mismatches, total, pct);
    // Phase 2 GPU still hardcodes base_height=0.3 (no region plan), so we expect
    // mismatches but the order of magnitude should match a typical surface chunk.
    // After Task 8 (region plan upload) this should be 0.
    assert!(mismatches < total, "all voxels mismatched — pipeline broken");

    pipeline.destroy(&ctx, &mut alloc);
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test --test gpu_terrain_parity --features app -- --nocapture`
Expected: PASS with non-zero but bounded mismatches (logged for review).

- [ ] **Step 3: Commit**

```bash
git add tests/gpu_terrain_parity.rs
git commit -m "test: GPU/CPU terrain parity baseline"
```

---

## Phase 3: Region plan upload + full materializer parity

### Task 8: Upload region plan to GPU as a storage buffer

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/src/terrain_compute.rs`
- Modify: `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`

- [ ] **Step 1: Add a packed `RegionCellGpu` struct in `terrain_compute.rs`**

Pack the cell data the shader needs into a fixed-size struct. Looking at `src/world_sim/terrain/region_plan.rs` `RegionCell`: we need `height`, `moisture`, `temperature`, `terrain` (enum → u32), `sub_biome` (enum → u32). 5 × 4 bytes = 20 bytes, padded to 24 or 32 for alignment.

```rust
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RegionCellGpu {
    pub height: f32,
    pub moisture: f32,
    pub temperature: f32,
    pub terrain: u32,    // Terrain enum as u32
    pub sub_biome: u32,  // SubBiome enum as u32
    pub _pad: [u32; 3],  // pad to 32 bytes for std430 alignment
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RegionPlanHeader {
    pub cols: u32,
    pub rows: u32,
    pub cell_size: u32,
    pub _pad: u32,
}
```

Add an `upload_region_plan` method to `TerrainComputePipeline`:

```rust
pub fn upload_region_plan(
    &mut self,
    ctx: &VulkanContext,
    alloc: &mut VulkanAllocator,
    cols: u32,
    rows: u32,
    cell_size: u32,
    cells: &[RegionCellGpu],
) -> Result<()> {
    // ... allocate a STORAGE_BUFFER, copy header + cells, bind to descriptor binding 1.
    // Create the buffer if first call, or destroy + recreate if size changed.
    // Update the descriptor set to include the new binding.
    todo!("implement: see existing upload patterns in voxel_gpu.rs")
}
```

The actual implementation pattern: create a host-visible staging buffer, memcpy header+cells, then bind to the descriptor at binding 1. Reuse the same descriptor pool.

This is the trickiest task — look at `voxel_engine/src/vulkan/voxel_gpu.rs::upload_grid_to_gpu` for the reference pattern of host-visible buffer creation and data copy.

Update the descriptor set layout in `TerrainComputePipeline::new` to include 3 bindings: output (0), region cells (1), region header (2). Update the pool size to 3 STORAGE_BUFFER descriptors.

- [ ] **Step 2: Update the shader to read from region plan**

Add bindings to `terrain_materialize.comp`:

```glsl
struct RegionCell {
    float height;
    float moisture;
    float temperature;
    uint terrain;
    uint sub_biome;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(set = 0, binding = 1) readonly buffer RegionCells {
    RegionCell cells[];
} region;

layout(set = 0, binding = 2) readonly buffer RegionHeader {
    uint cols;
    uint rows;
    uint cell_size;
    uint _pad;
} header;
```

Replace `surface_height_at` to sample the plan:

```glsl
RegionCell sample_cell(float vx, float vy) {
    int col = clamp(int(floor(vx / float(header.cell_size))), 0, int(header.cols) - 1);
    int row = clamp(int(floor(vy / float(header.cell_size))), 0, int(header.rows) - 1);
    return region.cells[row * int(header.cols) + col];
}

float interpolate_height(float vx, float vy) {
    float cs = float(header.cell_size);
    float cx = vx / cs;
    float cy = vy / cs;
    int col0 = clamp(int(floor(cx)), 0, int(header.cols) - 1);
    int row0 = clamp(int(floor(cy)), 0, int(header.rows) - 1);
    int col1 = min(col0 + 1, int(header.cols) - 1);
    int row1 = min(row0 + 1, int(header.rows) - 1);
    float tx = cx - floor(cx);
    float ty = cy - floor(cy);
    float h00 = region.cells[row0 * int(header.cols) + col0].height;
    float h10 = region.cells[row0 * int(header.cols) + col1].height;
    float h01 = region.cells[row1 * int(header.cols) + col0].height;
    float h11 = region.cells[row1 * int(header.cols) + col1].height;
    float h0 = h00 + (h10 - h00) * tx;
    float h1 = h01 + (h11 - h01) * tx;
    return h0 + (h1 - h0) * ty;
}

int surface_height_at(float vx, float vy, uint seed) {
    float base_height = interpolate_height(vx, vy);
    RegionCell cell = sample_cell(vx, vy);
    uint terrain = cell.terrain;

    float large = fbm_2d(vx * 0.004, vy * 0.004, seed + 0xfacecafeu, 4, 2.0, 0.5);
    float medium = fbm_2d(vx * 0.015, vy * 0.015, seed + 0xdeadbeefu, 3, 2.0, 0.5);
    float small = fbm_2d(vx * 0.06, vy * 0.06, seed + 0xcafebabeu, 2, 2.0, 0.5);

    // Terrain enum values from src/world_sim/state.rs Terrain enum.
    // 0=Plains 1=Forest 2=Jungle 3=Desert 4=Badlands 5=Mountains 6=Tundra ...
    // CHECK src/world_sim/state.rs FOR ACTUAL ORDERING when implementing.
    float large_amp = 25.0;
    float medium_amp = 10.0;
    float small_amp = 3.0;
    if (terrain == 5u || terrain == 7u) { large_amp = 80.0; medium_amp = 30.0; small_amp = 5.0; } // Mountains/Glacier
    else if (terrain == 4u) { large_amp = 50.0; medium_amp = 35.0; small_amp = 10.0; } // Badlands
    else if (terrain == 1u) { large_amp = 30.0; medium_amp = 12.0; small_amp = 3.0; } // Forest
    else if (terrain == 2u) { large_amp = 45.0; medium_amp = 20.0; small_amp = 6.0; } // Jungle
    else if (terrain == 3u) { large_amp = 35.0; medium_amp = 20.0; small_amp = 4.0; } // Desert
    // ... etc, mirroring the Rust match in materialize.rs:30-40

    float detail = (large * 2.0 - 1.0) * large_amp
                 + (medium * 2.0 - 1.0) * medium_amp
                 + (small * 2.0 - 1.0) * small_amp;

    return int(round(base_height * 2000.0 + detail));
}
```

The Terrain enum mapping must match exactly. Before writing the shader values, run `grep -n "pub enum Terrain" -A 30 src/world_sim/state.rs` to get the actual ordering.

- [ ] **Step 3: Add a `from_region_plan` helper in Rust**

Add to `terrain_compute.rs`:

```rust
/// Convert game::world_sim::terrain::region_plan::RegionPlan into a flat
/// `Vec<RegionCellGpu>` ready for upload. The terrain enum mapping must match
/// the GLSL shader.
pub fn pack_region_cells(/* take a slice of opaque cells */) -> Vec<RegionCellGpu> {
    todo!()
}
```

The plumbing detail: voxel_engine can't depend on `game`, so the conversion has to live in the `game` crate or use a trait. Simpler: define `RegionCellGpu` in `voxel_engine` and have the `game` crate construct the Vec directly. Add a small helper `to_gpu_cells()` in `src/world_sim/terrain/region_plan.rs` that returns `Vec<voxel_engine::terrain_compute::RegionCellGpu>`.

- [ ] **Step 4: Update the parity test to upload the plan first**

```rust
let plan = generate_continent(60, 40, 42);
let gpu_cells = plan.to_gpu_cells();
pipeline.upload_region_plan(&ctx, &mut alloc, plan.cols as u32, plan.rows as u32, CELL_SIZE as u32, &gpu_cells)?;
let gpu_mats = pipeline.generate_chunk(&ctx, [cp.x, cp.y, cp.z], 42).expect("dispatch");
```

- [ ] **Step 5: Run the parity test and aim for <5% mismatches**

Run: `cargo test --test gpu_terrain_parity --features app -- --nocapture`
Expected: Mismatches drop dramatically (was ~50%, now should be <5% — caves/rivers/features still missing).

If mismatches are still >50%, the noise function ports are wrong. Use the smoke test to dump the same `surface_height_at` value from both CPU and GPU at known coordinates and compare.

- [ ] **Step 6: Commit**

```bash
git add /home/ricky/Projects/voxel_engine/src/terrain_compute.rs \
        /home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp \
        /home/ricky/Projects/voxel_engine/shaders/compiled/terrain_materialize.comp.spv \
        src/world_sim/terrain/region_plan.rs \
        tests/gpu_terrain_parity.rs
git commit -m "feat: GPU samples region plan for biome-aware terrain"
```

---

### Task 9: Port full layer-assignment match from `materialize_chunk`

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`

- [ ] **Step 1: Replace simplified material assignment with full version**

Read `src/world_sim/terrain/materialize.rs` lines 114-260. Port every branch (Mountains, Tundra, Plains, Forest, Jungle, Badlands, Desert surface variants; subsoil/stone layer ore mixing; biome-aware materials; geological banding for mountains; etc.) into the GLSL shader.

Material constants (from `src/world_sim/voxel.rs` `VoxelMaterial` enum, repr(u8) ordering):

```glsl
const uint MAT_AIR = 0u;
const uint MAT_DIRT = 1u;
const uint MAT_STONE = 2u;
const uint MAT_GRANITE = 3u;
const uint MAT_SAND = 4u;
const uint MAT_CLAY = 5u;
const uint MAT_GRAVEL = 6u;
const uint MAT_GRASS = 7u;
const uint MAT_WATER = 8u;
// ... full list, copy from voxel.rs and verify discriminant order
```

- [ ] **Step 2: Verify enum values**

Run: `grep -n "VoxelMaterial::" src/world_sim/voxel.rs | head -50`
Match the constants in the shader to the actual `repr(u8)` ordering.

- [ ] **Step 3: Recompile and run parity test**

Run: `cd /home/ricky/Projects/voxel_engine && cargo build --features compile-shaders && cd - && cargo test --test gpu_terrain_parity --features app -- --nocapture`
Expected: Mismatches drop further. Should be <1% (only caves/rivers/features now).

- [ ] **Step 4: Commit**

```bash
git add /home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp \
        /home/ricky/Projects/voxel_engine/shaders/compiled/terrain_materialize.comp.spv
git commit -m "feat: full biome-aware layer assignment in GPU shader"
```

---

## Phase 4: Caves, rivers, surface features

### Task 10: Cave carving on GPU

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`

- [ ] **Step 1: Port `caves::carve_caves` logic**

Read `src/world_sim/terrain/caves.rs` for the cave-carving algorithm. Port the noise check to GLSL. The carve happens in the same compute pass — after computing the base material, check if this voxel should be cave (set to MAT_AIR).

Pseudo:
```glsl
bool is_cave(int vx, int vy, int vz, uint seed) {
    // worm cave: two 3D noise fields, both near 0.5 → carve
    float a = value_noise_3d(float(vx), float(vy), float(vz), seed + 0x1000u, 16.0);
    float b = value_noise_3d(float(vx), float(vy), float(vz), seed + 0x2000u, 16.0);
    return abs(a - 0.5) < 0.06 && abs(b - 0.5) < 0.06;
}
```

Apply only when depth > 20 and the chunk's center is below surface (matches CPU logic).

- [ ] **Step 2: Run parity test**

Run: `cargo test --test gpu_terrain_parity --features app -- --nocapture`
Expected: Mismatches drop further if the chunk being tested is in a cave-eligible area.

- [ ] **Step 3: Commit**

```bash
git add /home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp \
        /home/ricky/Projects/voxel_engine/shaders/compiled/terrain_materialize.comp.spv
git commit -m "feat: cave carving in GPU shader"
```

---

### Task 11: River carving on GPU

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/src/terrain_compute.rs` (river buffer)
- Modify: `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`

- [ ] **Step 1: Add river polyline storage buffer**

Add binding 3: a flat buffer of `vec4(point.x, point.y, width, _)` for each river point. Use a separate offsets buffer (binding 4) so the shader knows where each river starts/ends in the points buffer.

Or simpler: pack as `vec4` with `w` = river_id, and use a parallel `uvec2[]` of `(start_idx, length)` per river.

- [ ] **Step 2: Port `rivers::carve_river_in_chunk` to GLSL**

For each voxel near the surface, iterate over rivers and check distance to the polyline. If within `width`, carve to water.

This is O(voxels × rivers × points). At 262K voxels × ~10 rivers × ~50 points each = 130M ops per chunk — borderline. Optimization: bounding box per river, skip rivers whose bbox doesn't overlap the chunk.

- [ ] **Step 3: Update upload code to package river data**

Add `upload_rivers` method on `TerrainComputePipeline` similar to `upload_region_plan`.

- [ ] **Step 4: Run parity test**

Run: `cargo test --test gpu_terrain_parity --features app -- --nocapture`

- [ ] **Step 5: Commit**

```bash
git add /home/ricky/Projects/voxel_engine/src/terrain_compute.rs \
        /home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp \
        /home/ricky/Projects/voxel_engine/shaders/compiled/terrain_materialize.comp.spv
git commit -m "feat: river carving in GPU shader"
```

---

### Task 12: Surface features on GPU (trees, boulders, pillars)

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp`

- [ ] **Step 1: Halo-based feature stamping**

For each voxel in the output chunk, scan a 1-chunk halo of feature *origins* (not voxels — just the (vx, vy) where a tree might root). For each candidate origin in the halo, check if the current voxel falls inside that tree's footprint. If yes, write the canopy/trunk material.

Pseudocode:
```glsl
// Scan a halo of CHUNK_SIZE around the chunk for tree origins.
for (int oy = -CHUNK_SIZE; oy < CHUNK_SIZE * 2; oy++) {
    for (int ox = -CHUNK_SIZE; ox < CHUNK_SIZE * 2; ox++) {
        int origin_vx = base_vx + ox;
        int origin_vy = base_vy + oy;
        // Get the surface at the origin.
        int origin_surface = surface_height_at(float(origin_vx), float(origin_vy), seed);
        // Check tree density for this column.
        float td_hash = hash_f32(origin_vx, origin_vy, 0, seed + TREE_DENSITY_SALT);
        if (td_hash < tree_density_for_biome) {
            // This origin has a tree. Compute its size.
            // Check if our voxel (vx, vy, vz) is inside this tree's footprint.
            int dx = vx - origin_vx;
            int dy = vy - origin_vy;
            int dz = vz - (origin_surface + 1);
            // ... apply trunk + canopy logic
        }
    }
}
```

Halo size: tree canopies are up to 30 voxels radius = 30 voxels. So scan ±30 voxels around the chunk = much less than a full chunk halo. This makes the loop ~64 voxels per axis × 64 voxels per axis = 4096 candidates per voxel = 1B ops per chunk. Too much.

Better: scan only tree-spawn columns. Use a coarser grid: every voxel is a potential tree origin, but tree density is ~0.005, so 99.5% will skip immediately. The hash check is cheap. Should be ~4096 candidates × 0.005 ≈ 20 actual stamps per voxel.

- [ ] **Step 2: Port `feature_params` table to GLSL**

```glsl
float tree_density(uint terrain, uint sub_biome) {
    if (terrain == 1u) { // Forest
        if (sub_biome == 1u) return 0.025; // DenseForest
        return 0.010;
    }
    if (terrain == 2u) return 0.035; // Jungle
    // ... etc
    return 0.0;
}
```

- [ ] **Step 3: Run parity test**

Run: `cargo test --test gpu_terrain_parity --features app -- --nocapture`
Expected: Mismatches near 0 for any chunk.

- [ ] **Step 4: Re-run gallery render**

Run: `cargo test --test voxel_render_inspect terrain_gallery --features app --release -- --nocapture`

Compare `generated/render_tests/gallery.png` to a known-good baseline. The gallery test still uses CPU `materialize_chunk` so this should be unchanged — confirms we haven't broken the CPU path.

- [ ] **Step 5: Commit**

```bash
git add /home/ricky/Projects/voxel_engine/shaders/terrain_materialize.comp \
        /home/ricky/Projects/voxel_engine/shaders/compiled/terrain_materialize.comp.spv
git commit -m "feat: surface features (trees, boulders) in GPU shader"
```

---

## Phase 5: Wire GPU into the render loop

### Task 13: Replace `generate_camera_chunks` CPU path with GPU dispatch

**Files:**
- Modify: `src/world_sim/voxel_app.rs` (the `generate_camera_chunks` function)

- [ ] **Step 1: Hold a `TerrainComputePipeline` on `WorldSimVoxelApp`**

Add a field:

```rust
terrain_compute: Option<TerrainComputePipeline>,
```

Initialize it in `WorldSimVoxelApp::new` (with the existing Vulkan context). On first use, upload the region plan.

- [ ] **Step 2: Replace `materialize_chunk` call with GPU dispatch**

In `generate_camera_chunks`, replace:

```rust
let chunk = crate::world_sim::terrain::materialize_chunk(cp, &plan, seed);
```

with:

```rust
let gpu_mats = self.terrain_compute.as_ref().unwrap()
    .generate_chunk(&self.ctx, [cp.x, cp.y, cp.z], seed as u32)?;
// Convert flat material array into a Chunk struct.
let mut chunk = Chunk::new_air(cp);
for lz in 0..CHUNK_SIZE {
    for ly in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            let mat = gpu_mats[lz * CHUNK_SIZE * CHUNK_SIZE + ly * CHUNK_SIZE + lx];
            if mat != 0 {
                chunk.voxels[local_index(lx, ly, lz)] = Voxel::new(unsafe { std::mem::transmute(mat) });
            }
        }
    }
}
```

(Use a safer conversion than `transmute` — write a `VoxelMaterial::from_u8` helper.)

Increase the budget back to 8 chunks/frame since GPU dispatch is fast.

- [ ] **Step 3: Run the renderer**

Run: `cargo run --release --features app --bin xtask -- world-sim --render --ticks 100`

Expected: window opens, camera generates chunks visibly faster than before, FPS noticeably higher.

- [ ] **Step 4: Profile and verify the speedup**

Look at the `[perf]` log lines. Compare to the baseline:
- Before GPU: ~6 FPS, total ~170ms
- After GPU: target >30 FPS, total <33ms

If the speedup isn't there, the GPU dispatch is probably synchronous (queue_wait_idle blocking the frame). Phase 6 work, but note it.

- [ ] **Step 5: Commit**

```bash
git add src/world_sim/voxel_app.rs
git commit -m "feat: use GPU terrain compute in render loop"
```

---

### Task 14: Async GPU dispatch (no per-frame stall)

**Files:**
- Modify: `/home/ricky/Projects/voxel_engine/src/terrain_compute.rs`
- Modify: `src/world_sim/voxel_app.rs`

- [ ] **Step 1: Make `generate_chunk` non-blocking**

Refactor `TerrainComputePipeline::generate_chunk` to:
- `submit_chunk(chunk_pos, seed) -> ChunkRequestId` — submit dispatch with a fence
- `try_take_completed() -> Option<(ChunkRequestId, Vec<u8>)>` — poll fences, return ready chunks

Use a ring of N output buffers (e.g., 8) so multiple chunks can be in flight.

- [ ] **Step 2: Update voxel_app to use the async API**

In `generate_camera_chunks`, submit chunks but don't wait. In a separate phase of the frame loop, drain completed chunks and write them into the world.

- [ ] **Step 3: Run renderer and verify FPS climbs further**

Run: `cargo run --release --features app --bin xtask -- world-sim --render --ticks 100`

Expected: FPS approaches the engine's intrinsic rate (60+).

- [ ] **Step 4: Commit**

```bash
git add /home/ricky/Projects/voxel_engine/src/terrain_compute.rs src/world_sim/voxel_app.rs
git commit -m "feat: async GPU chunk generation with multi-buffer pipeline"
```

---

## Phase 6: Cleanup

### Task 15: Verify all tests still pass

- [ ] **Step 1: Run the full lib test suite**

Run: `cargo test --lib --release`
Expected: 391 passed.

- [ ] **Step 2: Run the integration tests**

Run: `cargo test --test voxel_render_inspect --features app --release -- --nocapture`
Expected: PASS.

Run: `cargo test --test gpu_terrain_parity --features app --release -- --nocapture`
Expected: PASS with <1% mismatches.

- [ ] **Step 3: Re-render the gallery and verify no visual regressions**

Run: `cargo test --test voxel_render_inspect terrain_gallery --features app --release -- --nocapture`
Open `generated/render_tests/gallery.png`. Compare to the version committed in `0d8572b5` — should look identical (the gallery uses CPU materialize_chunk, which we haven't changed).

- [ ] **Step 4: Commit any cleanup**

If there are stale comments, dead helper functions, or `todo!()` calls left in `terrain_compute.rs`, remove them.

```bash
git add -u
git commit -m "chore: clean up after GPU terrain implementation"
```

---

## Things to verify during execution

- **Hash parity:** add a Rust+GLSL pair test that calls `hash_u32(10, 20, 30, 42)` from both sides and asserts equal. The first time the parity test fails, this is the most likely culprit.
- **Push constant alignment:** Vulkan push constants must be 4-byte aligned. ivec4 + uvec4 = 32 bytes, both naturally aligned. If the shader sees garbage values, double-check the byte offsets in the Rust `cmd_push_constants` call.
- **Buffer storage qualifier:** the shader uses `std430` implicitly for storage buffers. The `RegionCellGpu` struct must match `std430` layout (no `vec3` padding surprises — use `vec4` and `_pad`).
- **Determinism between runs:** if the GPU produces slightly different results each run, you have undefined behavior somewhere (uninitialized push constants, race condition in `write_voxel`, etc.).
- **The gallery test still uses CPU:** the gallery test (`voxel_render_inspect.rs`) calls `world.generate_chunk()` which calls the CPU `materialize_chunk`. Don't change this in Phase 5 — keep both paths functional. Only the renderer's interactive path moves to GPU.
