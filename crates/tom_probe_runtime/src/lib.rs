//! Per-fixture runtime for `assets/sim/tom_probe.sim` — Theory-of-Mind
//! end-to-end probe (post-fix shape).
//!
//! ## What lights up
//!
//! - The Knower SoA (`agent_alive`).
//! - The `BeliefAcquired` event ring + per-tick tail clear.
//! - The `physics_WhatIBelieve` kernel (per-Knower; emits one
//!   `BeliefAcquired { observer: self, subject: self, fact_bit: 1 }`
//!   per tick into the ring).
//! - The `fold_beliefs_flags` kernel (per-event; OR's `fact_bit` into
//!   `view_storage_primary[observer * agent_cap + subject]` via
//!   WGSL native `atomicOr` — no CAS retry, P11 trivial). Renamed
//!   from `fold_beliefs` in Wave 3 ToM Phase 2 (see below).
//! - Per-(observer, subject) belief storage: see "BeliefState SoA"
//!   below for the multi-field shape introduced by Wave 3 ToM
//!   Phase 2. The `flags` column is the same `array<atomic<u32>>`
//!   BGL shape the fold kernel expects, so binding parity holds.
//!
//! ## BeliefState SoA (Wave 3 ToM Phase 2)
//!
//! Each (observer, target) pair is described by 6 parallel SoA
//! columns mirroring the spec's BeliefState block
//! (`docs/spec/ability_dsl_test_sims.md` spy_network):
//!
//! | Field        | Type       | Buffer                          | Notes                                  |
//! |--------------|------------|---------------------------------|----------------------------------------|
//! | flags        | u32        | `beliefs_flags_primary`         | Wave 3 Phase 1 bit-OR slot (kept)      |
//! | last_known_pos        | vec3 (4×f32 padded) | `beliefs_pos_primary`           | observer's last sighting               |
//! | last_known_creature_type | u8 → u32     | `beliefs_type_primary`          | observer's last sighted classification |
//! | last_seen_tick        | u32       | `beliefs_tick_primary`          | tick of last update; powers decay      |
//! | confidence            | u8 (q8)   | `beliefs_confidence_primary`    | 0..255 (≈ 0.0..1.0); how sure observer |
//! | suspicion             | u8 (q8)   | `beliefs_suspicion_primary`     | 0..255; how much observer suspects     |
//!
//! Phase 2 ships the storage shape + readback accessors only — the
//! writer verbs (`scry` / `disguise` / `observe` / etc.) are
//! Phase 3, and the spec's `agents.beliefs(o, s).<field>`
//! struct-then-field DSL access syntax is also deferred there. The
//! 5 non-`flags` columns are NOT folded over any event today; they
//! exist as runtime-allocated GPU storage so test fixtures can
//! pre-seed them with `seed_beliefs_<field>()` and assert
//! per-(observer, target) cell isolation via the readback accessors.
//!
//! Cell indexing across all 6 columns is `observer * agent_count +
//! target` — identical to `beliefs_flags`, so a fold kernel that
//! eventually folds into one of these columns can reuse the existing
//! `pair_map` index expression unchanged.
//!
//! ## Expected outcome (FULL FIRE)
//!
//! After N ticks at agent_count = N: `beliefs_flags(i, i) = 1u` for
//! every alive Knower; every off-diagonal `beliefs_flags(i, j != i)
//! = 0u`. The tom_probe_app driver asserts both halves and reports
//! OUTCOME (a) FULL FIRE on success.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::EventRing;

/// Per-fixture state for the ToM probe. Carries the Knower SoA
/// (`agent_alive`), the `BeliefAcquired` event ring, the 6-column
/// per-(observer, subject) BeliefState SoA storage (Wave 3 ToM
/// Phase 2 — see module docstring for the column inventory), and
/// the cfg uniforms for the producer (`physics_WhatIBelieve`) +
/// consumer (`fold_beliefs_flags`) kernels.
pub struct TomProbeState {
    gpu: GpuContext,

    // -- Agent SoA (read by physics_WhatIBelieve to gate `self.alive`) --
    /// 1 = alive, 0 = dead. Initialised all-1 so every Knower fires
    /// the producer rule each tick.
    agent_alive_buf: wgpu::Buffer,

    // -- Agent SoA (Wave 3 ToM Phase 3) — addressed by the `observe`
    // consumer to read target's CURRENT pos / creature_type at consume
    // tick. These columns aren't read by any DSL kernel today (the
    // existing Knower entity declares `pos: vec3, vel: vec3` but the
    // .sim consumer doesn't bind them); they're allocated runtime-side
    // so the observe round-trip pin can pre-seed target's state and
    // verify the consumer copies it into the BeliefState SoA.
    //
    // `agent_pos`: vec4 f32 padded (vec3 padded to vec4 for std430 +
    // future-proofing parity with the existing `beliefs_pos` column).
    // `agent_creature_type`: u8 (one byte per slot). Both share the
    // same `agent_count`-sized footprint.
    agent_pos_buf: wgpu::Buffer,
    agent_pos_staging: wgpu::Buffer,
    agent_pos_cache: Vec<[f32; 4]>,
    agent_pos_dirty: bool,

    agent_creature_type_buf: wgpu::Buffer,
    agent_creature_type_staging: wgpu::Buffer,
    agent_creature_type_cache: Vec<u8>,
    agent_creature_type_dirty: bool,

    // -- BeliefState SoA: `flags` column (Wave 3 ToM Phase 1 + Phase 2) --
    //
    // `pair_map`-keyed: `agent_cap × agent_cap × u32`. The fold body
    // indexes `view_storage_primary[observer * cfg.second_key_pop +
    // subject]`. We allocate this locally (instead of via
    // `engine::gpu::ViewStorage`) so the host-side readback can
    // surface a `&[u32]` directly without an f32 bitcast round-trip.
    // Renamed from `beliefs_*` → `beliefs_flags_*` in Phase 2 to
    // disambiguate from the other 5 BeliefState columns below.
    beliefs_flags_primary: wgpu::Buffer,
    beliefs_flags_staging: wgpu::Buffer,
    beliefs_flags_cache: Vec<u32>,
    beliefs_flags_dirty: bool,

    // -- BeliefState SoA: `last_known_pos` column (Phase 2) --
    //
    // 4×f32 per cell (vec3 padded to vec4 for std430 alignment) so
    // `agent_cap × agent_cap` cells take `agent_cap^2 * 16` bytes.
    // The host-side cache is `Vec<[f32; 4]>` so the readback can
    // surface positions without a per-row reshape.
    beliefs_pos_primary: wgpu::Buffer,
    beliefs_pos_staging: wgpu::Buffer,
    beliefs_pos_cache: Vec<[f32; 4]>,
    beliefs_pos_dirty: bool,

    // -- BeliefState SoA: `last_known_creature_type` column (Phase 2) --
    //
    // u8 per cell, packed as one byte per slot. Host cache is
    // `Vec<u8>` (no padding); GPU buffer rounds size to a 16-byte
    // multiple to satisfy std430 + the wgpu min-binding-size guards.
    beliefs_type_primary: wgpu::Buffer,
    beliefs_type_staging: wgpu::Buffer,
    beliefs_type_cache: Vec<u8>,
    beliefs_type_dirty: bool,

    // -- BeliefState SoA: `last_seen_tick` column (Phase 2) --
    //
    // u32 per cell — same shape as `flags` minus the `atomic<>`
    // qualifier (Phase 2 has no per-event writers contending on this
    // column; Phase 3 will introduce them and may need to switch to
    // `atomic<u32>` per usage pattern).
    beliefs_tick_primary: wgpu::Buffer,
    beliefs_tick_staging: wgpu::Buffer,
    beliefs_tick_cache: Vec<u32>,
    beliefs_tick_dirty: bool,

    // -- BeliefState SoA: `confidence` column (Phase 2) --
    //
    // u8 per cell (q8 fixed-point in 0..255 representing 0.0..1.0).
    // Same packing strategy as `creature_type`.
    beliefs_confidence_primary: wgpu::Buffer,
    beliefs_confidence_staging: wgpu::Buffer,
    beliefs_confidence_cache: Vec<u8>,
    beliefs_confidence_dirty: bool,

    // -- BeliefState SoA: `suspicion` column (Phase 2) --
    //
    // u8 per cell (q8 — observer's hostility / lying suspicion of
    // the target). Same packing strategy as `creature_type`.
    beliefs_suspicion_primary: wgpu::Buffer,
    beliefs_suspicion_staging: wgpu::Buffer,
    beliefs_suspicion_cache: Vec<u8>,
    beliefs_suspicion_dirty: bool,

    // -- Event ring + per-kernel cfg uniforms --
    event_ring: EventRing,
    physics_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

/// Allocate a paired (primary, staging) buffer of `bytes` bytes for
/// a single BeliefState SoA column. The primary carries
/// `STORAGE | COPY_SRC | COPY_DST` (kernel can write, host can
/// readback + seed); staging is `MAP_READ | COPY_DST`. The pair is
/// the standard wgpu shape for "GPU-resident readable buffer with
/// off-line CPU mapping".
fn alloc_belief_column_pair(
    device: &wgpu::Device,
    label_primary: &str,
    label_staging: &str,
    bytes: u64,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let bytes = bytes.max(16); // wgpu rejects sub-16B storage allocations
    let primary = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label_primary),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label_staging),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (primary, staging)
}

impl TomProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Knower SoA — `agent_alive` is the only field the producer
        // rule reads (`where (self.alive)`). Every slot starts alive
        // so every tick fires.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Wave 3 ToM Phase 3 agent SoA — `agent_pos` (vec4 padded) and
        // `agent_creature_type` (u8). Allocated paired (primary +
        // staging) so the observe consumer can read them off-line into
        // host memory and write the result into the BeliefState SoA's
        // 6 columns. Both start zero-filled; tests pre-seed via
        // `seed_agent_pos` / `seed_agent_creature_type`.
        let agent_pos_bytes = (agent_count as u64) * 16;
        let (agent_pos_buf, agent_pos_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::agent_pos_primary",
            "tom_probe_runtime::agent_pos_staging",
            agent_pos_bytes,
        );
        let agent_type_bytes = (((agent_count as u64) + 15) / 16 * 16).max(16);
        let (agent_creature_type_buf, agent_creature_type_staging) =
            alloc_belief_column_pair(
                &gpu.device,
                "tom_probe_runtime::agent_creature_type_primary",
                "tom_probe_runtime::agent_creature_type_staging",
                agent_type_bytes,
            );

        // -- BeliefState SoA columns --
        //
        // All 6 columns share the same `agent_count × agent_count`
        // cell-count footprint and the `observer * agent_count +
        // target` indexing convention. The byte sizes differ per
        // column type:
        //   - flags / tick      : 4 B per cell  (u32)
        //   - pos               : 16 B per cell (vec4f, vec3 padded)
        //   - type / confidence / suspicion : 1 B per cell (u8)
        let cell_count = (agent_count as u64) * (agent_count as u64);

        let flags_bytes = cell_count * 4;
        let (beliefs_flags_primary, beliefs_flags_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_flags_primary",
            "tom_probe_runtime::beliefs_flags_staging",
            flags_bytes,
        );

        let pos_bytes = cell_count * 16;
        let (beliefs_pos_primary, beliefs_pos_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_pos_primary",
            "tom_probe_runtime::beliefs_pos_staging",
            pos_bytes,
        );

        // u8 columns share an "rounded up to a 16-byte multiple"
        // size so std430 alignment + min-binding-size guards stay
        // happy even at small agent counts. The CPU cache is sized
        // to the exact cell count (no padding); readback only copies
        // `cell_count` bytes back into the cache.
        let u8_buf_bytes = ((cell_count + 15) / 16 * 16).max(16);
        let (beliefs_type_primary, beliefs_type_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_type_primary",
            "tom_probe_runtime::beliefs_type_staging",
            u8_buf_bytes,
        );

        let tick_bytes = cell_count * 4;
        let (beliefs_tick_primary, beliefs_tick_staging) = alloc_belief_column_pair(
            &gpu.device,
            "tom_probe_runtime::beliefs_tick_primary",
            "tom_probe_runtime::beliefs_tick_staging",
            tick_bytes,
        );

        let (beliefs_confidence_primary, beliefs_confidence_staging) =
            alloc_belief_column_pair(
                &gpu.device,
                "tom_probe_runtime::beliefs_confidence_primary",
                "tom_probe_runtime::beliefs_confidence_staging",
                u8_buf_bytes,
            );

        let (beliefs_suspicion_primary, beliefs_suspicion_staging) =
            alloc_belief_column_pair(
                &gpu.device,
                "tom_probe_runtime::beliefs_suspicion_primary",
                "tom_probe_runtime::beliefs_suspicion_staging",
                u8_buf_bytes,
            );

        let event_ring = EventRing::new(&gpu, "tom_probe_runtime");

        let physics_cfg_init =
            physics_WhatIBelieve::PhysicsWhatIBelieveCfg {
                agent_cap: agent_count,
                tick: 0,
                seed: 0,
                _pad: 0,
            };
        let physics_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::physics_WhatIBelieve_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let fold_cfg_init = fold_beliefs_flags::FoldBeliefsFlagsCfg {
            event_count: 0,
            tick: 0,
            // `beliefs_flags(observer: Agent, subject: Agent)` —
            // both keys are Agent, so second_key_pop = agent_count.
            // The fold body composes
            // `view_storage_primary[k1 * second_key_pop + k2]` so
            // this MUST equal agent_count for the diagonal to land
            // at index `i * N + i`.
            second_key_pop: agent_count,
            _pad: 0,
        };
        let fold_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("tom_probe_runtime::fold_beliefs_flags_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_alive_buf,
            agent_pos_buf,
            agent_pos_staging,
            agent_pos_cache: vec![[0.0f32; 4]; agent_count as usize],
            agent_pos_dirty: false,
            agent_creature_type_buf,
            agent_creature_type_staging,
            agent_creature_type_cache: vec![0u8; agent_count as usize],
            agent_creature_type_dirty: false,
            beliefs_flags_primary,
            beliefs_flags_staging,
            beliefs_flags_cache: vec![0u32; cell_count as usize],
            beliefs_flags_dirty: false,
            beliefs_pos_primary,
            beliefs_pos_staging,
            beliefs_pos_cache: vec![[0.0f32; 4]; cell_count as usize],
            beliefs_pos_dirty: false,
            beliefs_type_primary,
            beliefs_type_staging,
            beliefs_type_cache: vec![0u8; cell_count as usize],
            beliefs_type_dirty: false,
            beliefs_tick_primary,
            beliefs_tick_staging,
            beliefs_tick_cache: vec![0u32; cell_count as usize],
            beliefs_tick_dirty: false,
            beliefs_confidence_primary,
            beliefs_confidence_staging,
            beliefs_confidence_cache: vec![0u8; cell_count as usize],
            beliefs_confidence_dirty: false,
            beliefs_suspicion_primary,
            beliefs_suspicion_staging,
            beliefs_suspicion_cache: vec![0u8; cell_count as usize],
            beliefs_suspicion_dirty: false,
            event_ring,
            physics_cfg_buf,
            fold_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-(observer, subject) belief flag bitset, flattened
    /// row-major: slot `[observer * agent_count + subject]` holds
    /// the OR-folded fact bits the observer believes about the
    /// subject. Length = `agent_count × agent_count`. This is the
    /// `flags` column of the BeliefState SoA — the only column
    /// folded over a chronicle event in Phase 2 (Phase 1's
    /// `BeliefAcquired` shape preserved).
    pub fn beliefs_flags(&mut self) -> &[u32] {
        if self.beliefs_flags_dirty {
            let bytes = (self.beliefs_flags_cache.len() as u64) * 4;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_flags::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_flags_primary,
                0,
                &self.beliefs_flags_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_flags_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.beliefs_flags_cache.copy_from_slice(raw);
            drop(mapped);
            self.beliefs_flags_staging.unmap();
            self.beliefs_flags_dirty = false;
        }
        &self.beliefs_flags_cache
    }

    /// Per-(observer, subject) `last_known_pos` column. Each cell is
    /// a `[f32; 4]` (vec3 padded to vec4). Wave 3 ToM Phase 2
    /// runtime-only storage; no fold writer today (writer lands in
    /// Phase 3).
    pub fn beliefs_pos(&mut self) -> &[[f32; 4]] {
        if self.beliefs_pos_dirty {
            let bytes = (self.beliefs_pos_cache.len() as u64) * 16;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_pos::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_pos_primary,
                0,
                &self.beliefs_pos_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_pos_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            self.beliefs_pos_cache.copy_from_slice(raw);
            drop(mapped);
            self.beliefs_pos_staging.unmap();
            self.beliefs_pos_dirty = false;
        }
        &self.beliefs_pos_cache
    }

    /// Per-(observer, subject) `last_known_creature_type` column.
    /// Each cell is a `u8` classification ordinal. Wave 3 ToM
    /// Phase 2 runtime-only storage.
    pub fn beliefs_type(&mut self) -> &[u8] {
        if self.beliefs_type_dirty {
            let cell_count = self.beliefs_type_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_type::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_type_primary,
                0,
                &self.beliefs_type_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_type_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.beliefs_type_cache.copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.beliefs_type_staging.unmap();
            self.beliefs_type_dirty = false;
        }
        &self.beliefs_type_cache
    }

    /// Per-(observer, subject) `last_seen_tick` column. Each cell
    /// is a `u32` tick number powering decay computations. Wave 3
    /// ToM Phase 2 runtime-only storage.
    pub fn beliefs_tick(&mut self) -> &[u32] {
        if self.beliefs_tick_dirty {
            let bytes = (self.beliefs_tick_cache.len() as u64) * 4;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_tick::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_tick_primary,
                0,
                &self.beliefs_tick_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_tick_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.beliefs_tick_cache.copy_from_slice(raw);
            drop(mapped);
            self.beliefs_tick_staging.unmap();
            self.beliefs_tick_dirty = false;
        }
        &self.beliefs_tick_cache
    }

    /// Per-(observer, subject) `confidence` column. Each cell is a
    /// `u8` q8 value in 0..255 representing 0.0..1.0. Wave 3 ToM
    /// Phase 2 runtime-only storage.
    pub fn beliefs_confidence(&mut self) -> &[u8] {
        if self.beliefs_confidence_dirty {
            let cell_count = self.beliefs_confidence_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_confidence::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_confidence_primary,
                0,
                &self.beliefs_confidence_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_confidence_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.beliefs_confidence_cache
                .copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.beliefs_confidence_staging.unmap();
            self.beliefs_confidence_dirty = false;
        }
        &self.beliefs_confidence_cache
    }

    /// Per-(observer, subject) `suspicion` column. Each cell is a
    /// `u8` q8 value in 0..255 representing how much the observer
    /// suspects the target is hostile / lying. Wave 3 ToM Phase 2
    /// runtime-only storage.
    pub fn beliefs_suspicion(&mut self) -> &[u8] {
        if self.beliefs_suspicion_dirty {
            let cell_count = self.beliefs_suspicion_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::beliefs_suspicion::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.beliefs_suspicion_primary,
                0,
                &self.beliefs_suspicion_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.beliefs_suspicion_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.beliefs_suspicion_cache
                .copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.beliefs_suspicion_staging.unmap();
            self.beliefs_suspicion_dirty = false;
        }
        &self.beliefs_suspicion_cache
    }

    // -- Test-only seed helpers (Wave 3 ToM Phase 2) --
    //
    // The pre-fix storage path (Phase 1) was driven entirely by the
    // `BeliefAcquired` event flowing through `fold_beliefs_flags`.
    // Phase 2 adds 5 more columns with no event-driven writers
    // today, so the regression pin needs an off-band channel to seed
    // each column with known bytes and assert the readback shape.
    //
    // Each `seed_*` method takes a host-side slice sized to
    // `agent_count^2` and writes it into the corresponding primary
    // GPU buffer via `Queue::write_buffer`. The dirty flag is
    // toggled so the next read-back picks up the seeded bytes.

    /// Test-only: seed the entire `flags` column.
    pub fn seed_beliefs_flags(&mut self, values: &[u32]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_flags expected {expected} cells, got {}",
            values.len(),
        );
        self.gpu.queue.write_buffer(
            &self.beliefs_flags_primary,
            0,
            bytemuck::cast_slice(values),
        );
        self.beliefs_flags_dirty = true;
    }

    /// Test-only: seed the entire `last_known_pos` column. Each
    /// cell is a `[f32; 4]` (vec3 padded to vec4).
    pub fn seed_beliefs_pos(&mut self, values: &[[f32; 4]]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_pos expected {expected} cells, got {}",
            values.len(),
        );
        self.gpu.queue.write_buffer(
            &self.beliefs_pos_primary,
            0,
            bytemuck::cast_slice(values),
        );
        self.beliefs_pos_dirty = true;
    }

    /// Test-only: seed the entire `last_known_creature_type`
    /// column.
    pub fn seed_beliefs_type(&mut self, values: &[u8]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_type expected {expected} cells, got {}",
            values.len(),
        );
        // Pad to the underlying buffer size (16-byte aligned) with
        // zero bytes so write_buffer doesn't reject a sub-buffer
        // write.
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_type_primary, 0, &padded);
        self.beliefs_type_dirty = true;
    }

    /// Test-only: seed the entire `last_seen_tick` column.
    pub fn seed_beliefs_tick(&mut self, values: &[u32]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_tick expected {expected} cells, got {}",
            values.len(),
        );
        self.gpu.queue.write_buffer(
            &self.beliefs_tick_primary,
            0,
            bytemuck::cast_slice(values),
        );
        self.beliefs_tick_dirty = true;
    }

    /// Test-only: seed the entire `confidence` column (q8 in
    /// 0..255).
    pub fn seed_beliefs_confidence(&mut self, values: &[u8]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_confidence expected {expected} cells, got {}",
            values.len(),
        );
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_confidence_primary, 0, &padded);
        self.beliefs_confidence_dirty = true;
    }

    /// Test-only: seed the entire `suspicion` column (q8 in
    /// 0..255).
    pub fn seed_beliefs_suspicion(&mut self, values: &[u8]) {
        let expected = self.cell_count() as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_beliefs_suspicion expected {expected} cells, got {}",
            values.len(),
        );
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_suspicion_primary, 0, &padded);
        self.beliefs_suspicion_dirty = true;
    }

    /// Number of (observer, target) cells in each BeliefState SoA
    /// column = `agent_count^2`. Convenience for test code that
    /// pre-allocates seed vectors.
    pub fn cell_count(&self) -> u32 {
        self.agent_count * self.agent_count
    }

    // -- Wave 3 ToM Phase 3 agent SoA seeders + readers --
    //
    // The observe consumer reads target's CURRENT pos / creature_type
    // from the agent SoA at consume tick. These are runtime-only
    // helpers (the .sim consumer doesn't bind these columns today —
    // the existing Knower entity declares `pos: vec3, vel: vec3` but
    // no kernel reads them). The closed-loop `observe_round_trip_pin`
    // pre-seeds target B's state via `seed_agent_pos` /
    // `seed_agent_creature_type`, calls `observe(observer_a, target_b,
    // tick)`, then asserts the BeliefState SoA's 6 columns at
    // `[a * N + b]` reflect what was seeded.

    /// Test-only: seed agent SoA `pos` column. Each cell is a
    /// `[f32; 4]` (vec3 padded to vec4 for std430 alignment).
    pub fn seed_agent_pos(&mut self, values: &[[f32; 4]]) {
        let expected = self.agent_count as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_agent_pos expected {expected} agents, got {}",
            values.len(),
        );
        self.gpu
            .queue
            .write_buffer(&self.agent_pos_buf, 0, bytemuck::cast_slice(values));
        self.agent_pos_dirty = true;
    }

    /// Test-only: seed agent SoA `creature_type` column. Each cell is
    /// a `u8` classification ordinal.
    pub fn seed_agent_creature_type(&mut self, values: &[u8]) {
        let expected = self.agent_count as usize;
        assert_eq!(
            values.len(),
            expected,
            "seed_agent_creature_type expected {expected} agents, got {}",
            values.len(),
        );
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(values);
        self.gpu
            .queue
            .write_buffer(&self.agent_creature_type_buf, 0, &padded);
        self.agent_creature_type_dirty = true;
    }

    /// Read the agent SoA `pos` column back to host memory. The cache
    /// is staged on first read and reused until any seeder dirties it.
    pub fn agent_pos(&mut self) -> &[[f32; 4]] {
        if self.agent_pos_dirty {
            let bytes = (self.agent_pos_cache.len() as u64) * 16;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::agent_pos::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.agent_pos_buf,
                0,
                &self.agent_pos_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.agent_pos_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
            self.agent_pos_cache.copy_from_slice(raw);
            drop(mapped);
            self.agent_pos_staging.unmap();
            self.agent_pos_dirty = false;
        }
        &self.agent_pos_cache
    }

    /// Read the agent SoA `creature_type` column back to host memory.
    pub fn agent_creature_type(&mut self) -> &[u8] {
        if self.agent_creature_type_dirty {
            let cell_count = self.agent_creature_type_cache.len();
            let bytes = cell_count as u64;
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("tom_probe_runtime::agent_creature_type::readback"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.agent_creature_type_buf,
                0,
                &self.agent_creature_type_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.agent_creature_type_staging.slice(..bytes);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            self.agent_creature_type_cache
                .copy_from_slice(&mapped[..cell_count]);
            drop(mapped);
            self.agent_creature_type_staging.unmap();
            self.agent_creature_type_dirty = false;
        }
        &self.agent_creature_type_cache
    }

    /// Wave 3 ToM Phase 3 — runtime-side observe consumer. Mirrors the
    /// CHRONICLE consumer rule that a future `EffectObserveApplied`
    /// PerEvent handler would run on GPU: reads target's current pos
    /// + creature_type from the agent SoA, and writes into ALL 6
    /// BeliefState SoA columns at `[observer * agent_count + target]`:
    ///
    ///   - `last_known_pos[idx]` ← target's current pos
    ///   - `last_known_creature_type[idx]` ← target's current type
    ///   - `last_seen_tick[idx]` ← `current_tick`
    ///   - `confidence[idx]` ← 255 (max — fresh observation)
    ///   - `flags[idx]` ← unchanged (Phase 1 bit-OR slot; observe
    ///     doesn't set bits)
    ///   - `suspicion[idx]` ← unchanged (Phase 4 deception verbs
    ///     mutate this; observe leaves it alone)
    ///
    /// `observer` and `target` are 0-based agent slot indices into the
    /// `agent_count`-sized SoA. The current sim tick (`self.tick`) is
    /// stamped into `last_seen_tick` so subsequent decay_step() calls
    /// can compute the `delta = world.tick - last_seen_tick` decay
    /// driver.
    ///
    /// The DSL view-call lowering for kernel-side reads of
    /// `agents.beliefs_<field>(observer, subject)` is deferred to
    /// Phase 4 alongside the `scry` / `reveal` verbs that pair with
    /// the deception verbs (disguise / decoy / erase_belief). Today
    /// the runtime hand-rolls the consumer logic so the closed-loop
    /// pin lands without expanding the DSL surface.
    pub fn observe(&mut self, observer: u32, target: u32) {
        assert!(
            observer < self.agent_count,
            "observe: observer {observer} >= agent_count {}",
            self.agent_count,
        );
        assert!(
            target < self.agent_count,
            "observe: target {target} >= agent_count {}",
            self.agent_count,
        );
        // Read target's current pos / creature_type from the agent SoA
        // (CPU-side staging readback). `agent_pos()` /
        // `agent_creature_type()` are dirty-cached; the first call
        // each tick pays the staging copy, subsequent calls reuse the
        // cache.
        let target_pos = self.agent_pos()[target as usize];
        let target_type = self.agent_creature_type()[target as usize];

        let n = self.agent_count as usize;
        let cell = (observer as usize) * n + (target as usize);
        let cell_offset = cell as u64;
        let tick_u32 = self.tick as u32;

        // Write each column at the (observer, target) cell. Each write
        // is a 1-cell partial buffer write (queue.write_buffer accepts
        // any byte-aligned offset). The writes go through the GPU
        // queue (stamped at submit time); the readback accessors mark
        // the columns dirty so subsequent reads see the new bytes.

        // last_known_pos: 16 B per cell.
        self.gpu.queue.write_buffer(
            &self.beliefs_pos_primary,
            cell_offset * 16,
            bytemuck::bytes_of(&target_pos),
        );
        // last_known_creature_type: 1 B per cell. wgpu requires the
        // write size to be a multiple of 4, so we widen to a u32 word
        // and write 4 bytes — but only if the write doesn't span a
        // cell boundary. With u8 cells packed contiguously, writing 1
        // byte at offset N would clobber 3 neighbouring cells. So we
        // staging-read the 4-byte aligned word, patch the target byte,
        // and write the whole word back.
        //
        // Simpler: pre-fetch the entire u8 column (already cached in
        // `beliefs_type_cache` if dirty), patch the byte, and write
        // back the entire column. This is O(N²) per observe call but
        // the agent_count is small in fixtures (≤ 16 typically) — the
        // round-trip pin uses N=4, so total is 16 bytes per call.
        let mut type_bytes = self.beliefs_type().to_vec();
        type_bytes[cell] = target_type;
        let padded_size = ((type_bytes.len() + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..type_bytes.len()].copy_from_slice(&type_bytes);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_type_primary, 0, &padded);

        // last_seen_tick: 4 B per cell.
        self.gpu.queue.write_buffer(
            &self.beliefs_tick_primary,
            cell_offset * 4,
            bytemuck::bytes_of(&tick_u32),
        );
        // confidence: 1 B per cell — same column-rewrite trick as
        // creature_type.
        let mut conf_bytes = self.beliefs_confidence().to_vec();
        conf_bytes[cell] = 255u8;
        let padded_size = ((conf_bytes.len() + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..conf_bytes.len()].copy_from_slice(&conf_bytes);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_confidence_primary, 0, &padded);

        // Mark all touched columns dirty so the next readback restages
        // the bytes.
        self.beliefs_pos_dirty = true;
        self.beliefs_type_dirty = true;
        self.beliefs_tick_dirty = true;
        self.beliefs_confidence_dirty = true;
    }

    /// Wave 3 ToM Phase 3 — per-tick decay phase. Walks every
    /// (observer, target) cell and decrements `confidence` by exactly
    /// `decay_per_tick`, saturating at 0. Cells observed THIS tick
    /// (`last_seen_tick == current_tick`) skip — the observe consumer
    /// already pegged confidence at 255 and a same-tick decay would
    /// drop it back to 254 immediately.
    ///
    /// **Decay slope rationale.** The spec sketch in the Phase 3 brief
    /// initially read `new_confidence = saturating_sub(confidence,
    /// (world.tick - last_seen_tick) * decay_per_tick)` — but applying
    /// that formula every tick double-counts the delta (`tick 1` =
    /// drop 1, `tick 2` = drop 2 more, `tick 3` = drop 3 more, etc.,
    /// summing to N(N+1)/2 instead of the linear N). The companion
    /// pin (`decay_pin.rs::decay_slope_is_one_per_tick`) explicitly
    /// asserts `confidence == 155` after 100 ticks (= 255 - 100), so
    /// the implementation must produce a *linear* slope. The simplest
    /// linear formulation is "decrement by `decay_per_tick` per call,
    /// skipping cells observed this tick" — which matches the
    /// observable target slope without needing a baseline-confidence
    /// register. A future refinement might gate decay on `delta` but
    /// would still need to guarantee `slope == decay_per_tick * ticks`
    /// for the pin to hold.
    ///
    /// Cells with `last_seen_tick == 0` AND `confidence == 0` are
    /// no-ops (no observation has happened — both at default — and the
    /// saturating arithmetic would no-op anyway; the guard is just an
    /// efficiency + intent hint).
    ///
    /// Run AFTER `observe` calls for the same tick so a cell observed
    /// at tick T (confidence=255, last_seen_tick=T) skips this round
    /// via the same-tick guard. The runtime sequence in `step()` is:
    /// observe injection (test API) → physics + fold kernels → decay
    /// sweep.
    ///
    /// `decay_per_tick = 1` is the canonical rate — confidence drops
    /// by 1 per tick of staleness. Pinned by `decay_pin.rs`.
    pub fn decay_step(&mut self) {
        let decay_per_tick: u8 = 1;
        let current_tick = self.tick as u32;

        // Pull both columns into host memory. The observe-side writes
        // for THIS tick haven't necessarily been staged yet (the GPU
        // queue is async), but `beliefs_*()` issues a fresh staging
        // copy when dirty so we see the latest bytes.
        let confidence = self.beliefs_confidence().to_vec();
        let last_seen_tick = self.beliefs_tick().to_vec();

        // Compute new confidence per cell.
        let mut new_confidence = confidence.clone();
        for cell in 0..(self.cell_count() as usize) {
            let last = last_seen_tick[cell];
            // Skip never-observed cells (intent + perf — the
            // saturating arithmetic would no-op anyway since confidence
            // is already 0).
            if last == 0 && confidence[cell] == 0 {
                continue;
            }
            // Skip cells observed THIS tick — observe just pegged
            // confidence at 255 and the same-tick decay would drop it
            // to 254 immediately (the observe consumer's writeback
            // shouldn't be silently undone by the next decay sweep).
            if last == current_tick {
                continue;
            }
            new_confidence[cell] = confidence[cell].saturating_sub(decay_per_tick);
        }

        // Write back the new confidence column in a single buffer
        // write. Same padding strategy as `seed_beliefs_confidence` /
        // `observe`'s confidence patch (rounds up to 16-byte multiple).
        let expected = self.cell_count() as usize;
        let padded_size = ((expected + 15) / 16 * 16).max(16);
        let mut padded = vec![0u8; padded_size];
        padded[..expected].copy_from_slice(&new_confidence);
        self.gpu
            .queue
            .write_buffer(&self.beliefs_confidence_primary, 0, &padded);
        self.beliefs_confidence_dirty = true;
    }

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl CompiledSim for TomProbeState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("tom_probe_runtime::step"),
            },
        );

        // (1) Per-tick clear of event_tail. The producer rule
        // atomicAdd's against it during physics_WhatIBelieve to
        // acquire write slots; the count accumulates over the tick
        // and the fold kernel reads it via cfg.event_count. Clearing
        // here guarantees a fresh per-tick slot count even though
        // event slots from prior ticks linger in the ring (the fold
        // kernel's `event_idx >= cfg.event_count` early-return
        // filters stale slots).
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) physics_WhatIBelieve — per-Knower; emits one
        // `BeliefAcquired { observer: self, subject: self, fact_bit:
        // 1 }` per tick when `self.alive`.
        let physics_cfg = physics_WhatIBelieve::PhysicsWhatIBelieveCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.physics_cfg_buf,
            0,
            bytemuck::bytes_of(&physics_cfg),
        );
        let physics_bindings =
            physics_WhatIBelieve::PhysicsWhatIBelieveBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_alive: &self.agent_alive_buf,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_physics_whatibelieve(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) fold_beliefs_flags — per-event; OR's `fact_bit` into
        // `beliefs_flags_primary[observer * agent_cap + subject]`
        // via `atomicOr`. We size event_count = agent_count: every
        // alive Knower emits exactly one event per tick, so this is
        // the exact upper bound (no skip / no over-dispatch). The
        // kernel's `event_idx >= cfg.event_count` early-return
        // filters anything beyond the producer's per-tick batch even
        // if leftover slots from prior ticks remain in the ring.
        let event_count_estimate = self.agent_count;
        let fold_cfg = fold_beliefs_flags::FoldBeliefsFlagsCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: self.agent_count,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.fold_cfg_buf,
            0,
            bytemuck::bytes_of(&fold_cfg),
        );
        let fold_bindings = fold_beliefs_flags::FoldBeliefsFlagsBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.beliefs_flags_primary,
            // No `@decay` and no top-K → no anchor / no ids; the
            // generated `record()` body falls back to primary via
            // `unwrap_or(primary_buf)` per `kernel.rs`'s slot-aliasing
            // convention.
            view_storage_anchor: None,
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fold_cfg_buf,
        };
        dispatch::dispatch_fold_beliefs_flags(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.beliefs_flags_dirty = true;
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice. Same shape
        // as verb_probe_runtime (which has the same comment).
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(TomProbeState::new(seed, agent_count))
}
