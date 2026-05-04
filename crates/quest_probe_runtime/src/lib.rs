//! Per-fixture runtime for `assets/sim/quest_probe.sim` — discovery
//! probe for `entity X : Quest` AND the `quests.*` namespace. Both
//! surfaces are documented in spec but never exercised in any
//! fixture today (see `docs/superpowers/notes/2026-05-04-quest_probe.md`
//! for the full discovery report).
//!
//! ## What this runtime exercises
//!
//! 1. **`Mission : Item` declaration** (live, fall-back analog of
//!    the rejected `Mission : Quest`). The Item entity-root
//!    reaches `populate_entity_field_catalog` and would allocate a
//!    per-Item `mission_reward` SoA buffer if any rule body ever
//!    read it via `items.reward(<idx>)`. No body does today, so
//!    this runtime does NOT bind an item buffer — the catalog is
//!    populated but the kernel surface is unread.
//!
//! 2. **`+= 1u` on a `u32`-result view** (live, observable as the
//!    LIVE GAP). The fold body lowers cleanly via the `+=` operator
//!    gate at `crates/dsl_compiler/src/cg/lower/view.rs:564`, but
//!    the WGSL emitter routes `self += <rhs>` on a u32-result view
//!    through the same `atomicOr(&storage[idx], rhs)` branch as
//!    `self |= <rhs>` (`crates/dsl_compiler/src/cg/emit/wgsl_body.rs:
//!    1326-1338`). With rhs = `1u` constant, this is idempotent —
//!    every emit ORs `1u` into the slot, leaving the per-slot value
//!    at `1u` regardless of fire count. The runtime reads back the
//!    u32 storage and the harness app contrasts the OBSERVED
//!    (gap-actual) value `1u` with the EXPECTED (operator-intent)
//!    value `100u`.
//!
//! ## Per-tick chain (mirrors `tom_probe_runtime` shape — no
//! `seed_indirect` step since the fold consumes the event_count
//! directly via `cfg.event_count`):
//!
//! 1. `clear_tail` — event_tail = 0 so `atomicAdd` slots restart.
//! 2. `physics_ProgressAndComplete` — reads `agent_alive`, emits
//!    `ProgressTick { agent: self, quest: 0 }` per alive Adventurer
//!    per tick.
//! 3. `fold_progress` — per-event tag-filter on `ProgressTick`
//!    (kind = 1u), `atomicOr(&progress_primary[agent], 1u)` per
//!    event.
//!
//! ## Observable
//!
//! With AGENT_COUNT=32, TICKS=100:
//!   - GAP-ACTUAL: `progress[N] = 1u` for every N (atomicOr
//!     idempotent on the same bit).
//!   - OPERATOR-INTENT: `progress[N] = 100u` (the `+=` semantics
//!     suggest atomicAdd, accumulating one per tick).
//!
//! Runtime reports both numbers; the harness OUTCOME line classifies
//! the result. See `docs/superpowers/notes/2026-05-04-quest_probe.md`.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::EventRing;

/// Per-fixture state for the quest probe. Owns:
///   - Agent SoA (`alive` only — the producer rule reads no other field)
///   - Event ring + per-view u32 storage (`progress` per agent)
///   - Per-kernel cfg uniforms
///
/// No per-Item SoA is allocated — the `Mission : Item` declaration
/// reaches the entity_field_catalog at compile time but no kernel
/// reads `items.reward(...)`, so there's no binding to provide.
pub struct QuestProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    /// 1 = alive, 0 = dead. All-1 init so every slot's `self.alive`
    /// gate evaluates true.
    agent_alive_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    /// Per-agent `progress` u32 view storage. Sized = `agent_count`,
    /// no @decay anchor, no top-K ids. Bound to `view_storage_primary`
    /// in the fold kernel; the host-side readback bypasses
    /// `engine::ViewStorage` (which returns `&[f32]`) and rolls a
    /// raw u32 staging buffer following `tom_probe_runtime`'s pattern.
    progress_primary: wgpu::Buffer,
    progress_staging: wgpu::Buffer,
    progress_cache: Vec<u32>,
    progress_dirty: bool,

    // -- Per-kernel cfg uniforms --
    physics_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl QuestProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — `alive` is read by physics_ProgressAndComplete
        // (BGL slot 2). Initialised to all-1 so every slot fires its
        // `where (self.alive)` gate.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quest_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Per-agent `progress` u32 storage. Single-key view (`view
        // progress(agent: Agent) -> u32`), so the slot count is
        // exactly `agent_count`. The buffer is the same `array<atomic
        // <u32>>` BGL shape the fold kernel expects; we expose
        // `STORAGE | COPY_SRC | COPY_DST` so the per-readback
        // `copy_buffer_to_buffer → staging map` dance works.
        let progress_slot_count = agent_count as u64;
        let progress_bytes = (progress_slot_count * 4).max(16);
        let progress_primary = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quest_probe_runtime::progress_primary"),
            size: progress_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let progress_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quest_probe_runtime::progress_staging"),
            size: progress_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let event_ring = EventRing::new(&gpu, "quest_probe_runtime");

        // Per-kernel cfg uniforms.
        let physics_cfg_init =
            physics_ProgressAndComplete::PhysicsProgressAndCompleteCfg {
                agent_cap: agent_count,
                tick: 0,
                seed: 0,
                _pad: 0,
            };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("quest_probe_runtime::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let fold_cfg_init = fold_progress::FoldProgressCfg {
            event_count: 0,
            tick: 0,
            // Single-key view → `second_key_pop` is unused by the
            // fold body (the index expr is `local_<last>` per
            // `wgsl_body.rs:1294-1298`). Set to 1 for parity with
            // cooldown_probe's no-pair-map cfg.
            second_key_pop: 1,
            _pad: 0,
        };
        let fold_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("quest_probe_runtime::fold_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
            event_ring,
            progress_primary,
            progress_staging,
            progress_cache: vec![0u32; progress_slot_count as usize],
            progress_dirty: false,
            physics_cfg_buf,
            fold_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-agent `progress` u32 readback. After T ticks under the
    /// LIVE GAP path: `progress[N] = 1u` for every N (atomicOr
    /// idempotent). Under the OPERATOR-INTENT shape (would require
    /// a future fix): `progress[N] = T` for every N.
    pub fn progress(&mut self) -> &[u32] {
        if self.progress_dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("quest_probe_runtime::progress::readback"),
                },
            );
            let bytes = (self.progress_cache.len() as u64) * 4;
            encoder.copy_buffer_to_buffer(
                &self.progress_primary,
                0,
                &self.progress_staging,
                0,
                bytes,
            );
            self.gpu.queue.submit(Some(encoder.finish()));
            let slice = self.progress_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
            let mapped = slice.get_mapped_range();
            let raw: &[u32] = bytemuck::cast_slice(&mapped);
            self.progress_cache.copy_from_slice(raw);
            drop(mapped);
            self.progress_staging.unmap();
            self.progress_dirty = false;
        }
        &self.progress_cache
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

impl CompiledSim for QuestProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("quest_probe_runtime::step"),
                });

        // (1) Per-tick clear of event_tail.
        self.event_ring.clear_tail_in(&mut encoder);

        // (2) physics_ProgressAndComplete — per-Adventurer; emits
        // one `ProgressTick { agent: self, quest: 0 }` per tick when
        // `self.alive`. No agent SoA reads beyond `alive`.
        let physics_cfg =
            physics_ProgressAndComplete::PhysicsProgressAndCompleteCfg {
                agent_cap: self.agent_count,
                tick: self.tick as u32,
                seed: 0,
                _pad: 0,
            };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));
        let physics_bindings =
            physics_ProgressAndComplete::PhysicsProgressAndCompleteBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_alive: &self.agent_alive_buf,
                cfg: &self.physics_cfg_buf,
            };
        dispatch::dispatch_physics_progressandcomplete(
            &mut self.cache,
            &physics_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) fold_progress — per-event tag-filter on ProgressTick
        // (kind = 1u). Body composes `view_storage_primary[local_0]`
        // (single-key — last AgentId binder is `agent`) and emits
        // `atomicOr(&storage[_idx], (1u))` per the u32-view emit
        // branch. event_count = agent_count: every alive Adventurer
        // emits exactly one event per tick.
        let event_count_estimate = self.agent_count;
        let fold_cfg = fold_progress::FoldProgressCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            second_key_pop: 1,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.fold_cfg_buf, 0, bytemuck::bytes_of(&fold_cfg));
        let fold_bindings = fold_progress::FoldProgressBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: &self.progress_primary,
            // No `@decay` and no top-K → no anchor / no ids; the
            // generated `record()` body falls back to primary via
            // `unwrap_or(primary_buf)` per `kernel.rs`'s slot-aliasing
            // convention (matches tom_probe shape).
            view_storage_anchor: None,
            view_storage_ids: None,
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.fold_cfg_buf,
        };
        dispatch::dispatch_fold_progress(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate.max(1),
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.progress_dirty = true;
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
        // as tom_probe_runtime / cooldown_probe_runtime.
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(QuestProbeState::new(seed, agent_count))
}
