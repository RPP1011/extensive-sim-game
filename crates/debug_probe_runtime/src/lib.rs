//! Per-fixture runtime for `assets/sim/debug_probe.sim` — Phase 2
//! `DebugWgslFlags` end-to-end exercise (task #242).
//!
//! Opts in to ALL 3 instrumentation axes via `LowerOpts.debug_wgsl =
//! DebugWgslFlags::ALL` (see `build.rs`); allocates the matching
//! `event_kind_counts` / `mask_total` / `mask_passed` /
//! `score_kernel_visits` storage buffers and threads them into the
//! synthesised kernel `Bindings` structs. Per-tick chain mirrors
//! `stress_agent_count_runtime` (mask → score → chronicle dispatcher
//! → consumer → seed_indirect) but uses `damage 0.0` rather than
//! `self_damage 0.0` so the dispatcher emits EffectDamageApplied
//! (kind=26) — the same kind that dominates the cast_density
//! fixture's chronicle ring.
//!
//! ## Behavioral pin (`viz_tests::all_three_axes_record_nonzero_data`)
//!
//! At agent_cap=100, after one tick:
//!   - `event_kind_histogram`: at least one kind slot > 0
//!     (EffectDamageApplied kind=26 should accumulate ~100 events).
//!   - `mask_hit_rate`: `total > 0` (mask kernel ran) AND `passed > 0`
//!     (every alive agent qualifies).
//!   - `score_kernel_visits`: at least one agent visited > 0 times.
//!
//! Failure mode: if the BGL composer regresses (i.e. removes the
//! debug-instrumentation bindings the WGSL `atomicAdd` sites
//! reference), the kernel either fails to compile (caught at build
//! time) or panics at dispatch (`Bindings { ... }` initializer
//! complains about the missing field). Either way, the test crashes
//! before reaching the readback assertions.

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext};

/// Slot count for the WGSL `event_kind_counts: array<atomic<u32>>`
/// instrumentation buffer. Sized for `EventKindId::ChronicleEntry =
/// 128 + 1` so every emitted EventKind discriminant has a dedicated
/// counter slot. See `crates/engine/src/cascade/handler.rs`.
pub const N_EVENT_KIND_SLOTS: usize = 129;

/// Mask slot count. The fixture has one verb (Pulse) → one mask
/// (mask_0). Sized at 4 to give a bit of headroom in case a future
/// edit to debug_probe.sim adds more verbs without re-counting.
pub const N_MASK_SLOTS: usize = 4;

/// Per-tick `event_kind_histogram` readback as a fixed-size array.
/// Indexed by `EventKindId` discriminant.
pub type EventKindHistogram = [u32; N_EVENT_KIND_SLOTS];

/// Per-mask `(total, passed)` rate accumulators. Indexed by `MaskId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskHitStats {
    pub mask_id: u32,
    pub total: u32,
    pub passed: u32,
}

/// Per-fixture state.
pub struct DebugProbeState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,
    agent_max_hp_buf: wgpu::Buffer,
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask + scoring --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring --
    event_ring: EventRing,

    // -- Cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    // -- Packed AbilityRegistry --
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Phase 2 debug instrumentation buffers. The WGSL emit writes
    //    `atomicAdd(&event_kind_counts[k], 1u)` /
    //    `atomicAdd(&mask_total[m], 1u)` /
    //    `atomicAdd(&mask_passed[m], 1u)` /
    //    `atomicAdd(&score_kernel_visits[a], 1u)` into these. Reset
    //    to zero at the start of every step() so each readback
    //    captures the most recent tick's totals only. --
    event_kind_counts_buf: wgpu::Buffer,
    mask_total_buf: wgpu::Buffer,
    mask_passed_buf: wgpu::Buffer,
    score_kernel_visits_buf: wgpu::Buffer,
    event_kind_counts_zero_buf: wgpu::Buffer,
    mask_total_zero_buf: wgpu::Buffer,
    mask_passed_zero_buf: wgpu::Buffer,
    score_kernel_visits_zero_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl DebugProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert!(agent_count > 0, "agent_count must be > 0");
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let built = build_debug_probe_registry();
        let packed = PackedAbilityRegistry::pack(&built);
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "debug_probe_runtime");

        let n = agent_count as usize;
        let hp_init = vec![100.0_f32; n];
        let alive_init = vec![1u32; n];
        let mana_init = vec![0.0_f32; n];
        let max_hp_init = vec![100.0_f32; n];
        let zeros_f32 = vec![0.0_f32; n];

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug_probe::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_probe::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let agent_mana_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_probe::agent_mana"),
                contents: bytemuck::cast_slice(&mana_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let agent_max_hp_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_probe::agent_max_hp"),
                contents: bytemuck::cast_slice(&max_hp_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let mk_stat = |label: &'static str| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&zeros_f32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let agent_attack_damage_buf = mk_stat("debug_probe::agent_attack_damage");
        let agent_ability_power_buf = mk_stat("debug_probe::agent_ability_power");
        let agent_armor_buf = mk_stat("debug_probe::agent_armor");
        let agent_magic_resist_buf = mk_stat("debug_probe::agent_magic_resist");
        let agent_move_speed_buf = mk_stat("debug_probe::agent_move_speed");

        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_0_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_probe::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_probe::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("debug_probe::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> =
            vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_probe::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let event_ring = EventRing::new(&gpu, "debug_probe");

        let mask_cfg_init = mask_verb_Pulse::MaskVerbPulseCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug_probe::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let scoring_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_probe::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let chronicle_cfg_init = physics_ApplyDamageFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyDamageFromChronicleAndVerbChroniclePulseCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            agent_cap: 0,
        };
        let chronicle_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("debug_probe::chronicle_cfg"),
                contents: bytemuck::bytes_of(&chronicle_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("debug_probe::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // -- Phase 2 debug instrumentation buffers. Each axis owns a
        //    pair: a `*_buf` storage destination + a matching
        //    `*_zero_buf` COPY_SRC source for the per-tick reset. --
        let mk_pair = |slots: usize, base: &str| -> (wgpu::Buffer, wgpu::Buffer) {
            let zeros: Vec<u32> = vec![0u32; slots];
            let dst = gpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("debug_probe::{base}")),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                },
            );
            let zero = gpu.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("debug_probe::{base}_zero")),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::COPY_SRC,
                },
            );
            (dst, zero)
        };
        let (event_kind_counts_buf, event_kind_counts_zero_buf) =
            mk_pair(N_EVENT_KIND_SLOTS, "event_kind_counts");
        let (mask_total_buf, mask_total_zero_buf) = mk_pair(N_MASK_SLOTS, "mask_total");
        let (mask_passed_buf, mask_passed_zero_buf) = mk_pair(N_MASK_SLOTS, "mask_passed");
        let (score_kernel_visits_buf, score_kernel_visits_zero_buf) =
            mk_pair(n, "score_kernel_visits");

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_max_hp_buf,
            agent_attack_damage_buf,
            agent_ability_power_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            mask_0_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_cfg_buf,
            seed_cfg_buf,
            registry_gpu,
            event_kind_counts_buf,
            mask_total_buf,
            mask_passed_buf,
            score_kernel_visits_buf,
            event_kind_counts_zero_buf,
            mask_total_zero_buf,
            mask_passed_zero_buf,
            score_kernel_visits_zero_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
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

    /// Read back the per-EventKindId histogram populated by the
    /// chronicle producer's compiler-emitted
    /// `atomicAdd(&event_kind_counts[k], 1u)` site. Resets to zero at
    /// the start of every step(); this returns the most recent tick's
    /// histogram only.
    pub fn event_kind_histogram(&self) -> EventKindHistogram {
        let bytes = (N_EVENT_KIND_SLOTS as u64) * 4;
        let v = self.read_u32_buf(&self.event_kind_counts_buf, bytes, "event_kind_counts");
        let mut out = [0u32; N_EVENT_KIND_SLOTS];
        out.copy_from_slice(&v[..N_EVENT_KIND_SLOTS]);
        out
    }

    /// Read back the per-mask `(total, passed)` accumulators. Index =
    /// MaskId. Returns one [`MaskHitStats`] per occupied slot
    /// (skipping zero-total slots so the output stays compact for
    /// fixtures with > 1 mask).
    pub fn mask_hit_rate(&self) -> Vec<MaskHitStats> {
        let bytes = (N_MASK_SLOTS as u64) * 4;
        let total = self.read_u32_buf(&self.mask_total_buf, bytes, "mask_total");
        let passed = self.read_u32_buf(&self.mask_passed_buf, bytes, "mask_passed");
        (0..N_MASK_SLOTS)
            .filter_map(|i| {
                let t = total[i];
                if t == 0 && passed[i] == 0 {
                    None
                } else {
                    Some(MaskHitStats {
                        mask_id: i as u32,
                        total: t,
                        passed: passed[i],
                    })
                }
            })
            .collect()
    }

    /// Read back the per-agent `score_kernel_visits` histogram. Index
    /// = AgentId; value = number of times the scoring kernel visited
    /// the agent's row this tick. Self-target verb → expect 1
    /// visit/agent for every alive agent.
    pub fn score_kernel_visits(&self) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4;
        self.read_u32_buf(&self.score_kernel_visits_buf, bytes, "score_kernel_visits")
    }

    fn read_u32_buf(&self, buf: &wgpu::Buffer, bytes: u64, label: &str) -> Vec<u32> {
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("debug_probe::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("debug_probe::read_u32_buf"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }
}

impl CompiledSim for DebugProbeState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("debug_probe::step"),
            },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = (self.agent_count.saturating_mul(4)).min(60_000);
        self.event_ring.clear_ring_headers_in(
            &self.gpu,
            &mut encoder,
            max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        encoder.copy_buffer_to_buffer(
            &self.mask_bitmap_zero_buf,
            0,
            &self.mask_0_bitmap_buf,
            0,
            mask_bytes.max(4),
        );
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf,
            0,
            &self.scoring_output_buf,
            0,
            scoring_output_bytes.max(16),
        );
        // Phase 2 instrumentation reset — every counter starts at 0
        // for each tick so the readback captures THIS tick's totals
        // only. Same encoder.copy as the other resets above to keep
        // them clustered.
        encoder.copy_buffer_to_buffer(
            &self.event_kind_counts_zero_buf,
            0,
            &self.event_kind_counts_buf,
            0,
            (N_EVENT_KIND_SLOTS as u64) * 4,
        );
        encoder.copy_buffer_to_buffer(
            &self.mask_total_zero_buf,
            0,
            &self.mask_total_buf,
            0,
            (N_MASK_SLOTS as u64) * 4,
        );
        encoder.copy_buffer_to_buffer(
            &self.mask_passed_zero_buf,
            0,
            &self.mask_passed_buf,
            0,
            (N_MASK_SLOTS as u64) * 4,
        );
        encoder.copy_buffer_to_buffer(
            &self.score_kernel_visits_zero_buf,
            0,
            &self.score_kernel_visits_buf,
            0,
            (self.agent_count as u64) * 4,
        );

        // Shared once per tick; each dispatch below adds only its
        // fixture-specific `*Extras` (mask bitmap, scoring output,
        // instrumentation buffers, indirect args).
        let agent_buffers = AgentBuffers {
            hp_buf: Some(&self.agent_hp_buf),
            max_hp_buf: Some(&self.agent_max_hp_buf),
            alive_buf: Some(&self.agent_alive_buf),
            mana_buf: Some(&self.agent_mana_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            armor_buf: Some(&self.agent_armor_buf),
            magic_resist_buf: Some(&self.agent_magic_resist_buf),
            move_speed_buf: Some(&self.agent_move_speed_buf),
            ..Default::default()
        };
        let ctx = KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
        };

        // (2) Mask round.
        let mask_cfg = mask_verb_Pulse::MaskVerbPulseCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf,
            0,
            bytemuck::bytes_of(&mask_cfg),
        );
        let mask_extras = mask_verb_Pulse::MaskVerbPulseExtras {
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_total: &self.mask_total_buf,
            mask_passed: &self.mask_passed_buf,
            cfg: &self.mask_cfg_buf,
        };
        let mask_bindings =
            mask_verb_Pulse::MaskVerbPulseBindings::from_context_with_extras(
                &ctx,
                &mask_extras,
            );
        dispatch::dispatch_mask_verb_pulse(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3) Scoring.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_extras = scoring::ScoringExtras {
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            score_kernel_visits: &self.score_kernel_visits_buf,
            cfg: &self.scoring_cfg_buf,
        };
        let scoring_bindings =
            scoring::ScoringBindings::from_context_with_extras(&ctx, &scoring_extras);
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) Chronicle dispatcher + consumer (fused PerEvent).
        let event_count_estimate = self.agent_count.saturating_mul(4).min(60_000);
        let chronicle_cfg = physics_ApplyDamageFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyDamageFromChronicleAndVerbChroniclePulseCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: self.agent_count,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_cfg_buf,
            0,
            bytemuck::bytes_of(&chronicle_cfg),
        );
        let chronicle_extras = physics_ApplyDamageFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyDamageFromChronicleAndVerbChroniclePulseExtras {
            event_kind_counts: &self.event_kind_counts_buf,
            cfg: &self.chronicle_cfg_buf,
        };
        let chronicle_bindings = physics_ApplyDamageFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyDamageFromChronicleAndVerbChroniclePulseBindings::from_context_with_extras(
            &ctx,
            &chronicle_extras,
        );
        dispatch::dispatch_physics_applydamagefromchronicle_and_verb_chronicle_pulse(
            &mut self.cache,
            &chronicle_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (5) seed_indirect_0.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf,
            0,
            bytemuck::bytes_of(&seed_cfg),
        );
        let seed_extras = seed_indirect_0::SeedIndirect0Extras {
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        let seed_bindings =
            seed_indirect_0::SeedIndirect0Bindings::from_context_with_extras(
                &ctx,
                &seed_extras,
            );
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("poll after step submit");

        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        &[]
    }

    fn snapshot(&mut self) -> AgentSnapshot {
        AgentSnapshot {
            positions: vec![Vec3::ZERO; self.agent_count as usize],
            creature_types: vec![1u32; self.agent_count as usize],
            alive: vec![1u32; self.agent_count as usize],
        }
    }

    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![VizGlyph::new('?', 240), VizGlyph::new('.', 27)]
    }
}

const PULSE_ABILITY_SRC: &str =
    include_str!("../../../assets/ability_test/debug_probe/Pulse.ability");

fn build_debug_probe_registry() -> engine::ability::registry::AbilityRegistry {
    let parse = |name: &str, src: &str| {
        dsl_ast::parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {name}: {e:?}"))
    };

    let files = vec![(
        "Pulse.ability".to_string(),
        parse("Pulse.ability", PULSE_ABILITY_SRC),
    )];

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over debug_probe corpus")
        .registry
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(DebugProbeState::new(seed, agent_count))
}

// ---------------------------------------------------------------------------
// Behavioral pin tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod viz_tests {
    use super::*;

    /// At agent_cap=100, a single tick must accumulate non-zero data in
    /// all 3 Phase 2 instrumentation buffers. Pin guards against a
    /// regression in the BGL composer dropping any of the four debug
    /// bindings (event_kind_counts, mask_total, mask_passed,
    /// score_kernel_visits).
    #[test]
    fn all_three_axes_record_nonzero_data() {
        let mut state = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            DebugProbeState::new(0, 100)
        })) {
            Ok(s) => s,
            Err(_) => {
                eprintln!(
                    "[debug_probe all_three_axes_record_nonzero_data] \
                     skipping: GPU init failed (no wgpu adapter on host?)"
                );
                return;
            }
        };

        state.step();

        // Axis 1: event_kind_histogram. The fixture's chronicle
        // dispatcher emits EffectDamageApplied (kind=26) per cast.
        let hist = state.event_kind_histogram();
        let nonzero_kinds: Vec<(usize, u32)> = hist
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| (c > 0).then_some((i, c)))
            .collect();
        assert!(
            !nonzero_kinds.is_empty(),
            "event_kind_histogram should have at least one non-zero kind \
             after 1 tick at agent_cap=100; got all zeros"
        );
        eprintln!(
            "[debug_probe] event_kind_histogram nonzero: {:?}",
            nonzero_kinds
        );

        // Axis 2: mask_hit_rate. With 100 alive agents on a self-target
        // mask, total > 0 AND passed > 0.
        let stats = state.mask_hit_rate();
        assert!(
            !stats.is_empty(),
            "mask_hit_rate should have at least one non-zero mask after 1 \
             tick at agent_cap=100; got all zeros (the mask kernel did \
             not fire OR the BGL composer lost the mask_total/passed \
             bindings)"
        );
        for s in &stats {
            assert!(
                s.total > 0,
                "mask {} reported passed={} but total=0 — atomic-counter \
                 ordering invariant violated",
                s.mask_id, s.passed,
            );
            assert!(
                s.passed > 0,
                "mask {} total={} but passed=0 — every alive agent should \
                 satisfy `self.alive` for the Pulse mask",
                s.mask_id, s.total,
            );
        }
        eprintln!("[debug_probe] mask_hit_rate: {:?}", stats);

        // Axis 3: score_kernel_visits. Self-target verb → each alive
        // agent visits the row exactly once per tick. Assert at least
        // one agent has a non-zero count (the strong "= 1 per agent"
        // claim is held back to avoid coupling the test to the
        // scoring kernel's iteration shape).
        let visits = state.score_kernel_visits();
        let nonzero_count = visits.iter().filter(|&&v| v > 0).count();
        assert!(
            nonzero_count > 0,
            "score_kernel_visits should have at least one non-zero agent \
             after 1 tick at agent_cap=100; got all zeros (scoring kernel \
             did not fire OR the BGL composer lost the \
             score_kernel_visits binding)"
        );
        eprintln!(
            "[debug_probe] score_kernel_visits: {} of {} agents non-zero, \
             min={} max={}",
            nonzero_count,
            visits.len(),
            visits.iter().min().copied().unwrap_or(0),
            visits.iter().max().copied().unwrap_or(0),
        );
    }
}
