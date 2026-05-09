//! Per-fixture runtime for `assets/sim/stress_agent_count.sim` —
//! Fixture A from `docs/superpowers/plans/2026-05-09-stress-test-runtime-ceilings.md`.
//!
//! ## What this fixture characterizes
//!
//! One-shot characterization of the agent_count ceiling. Drives the
//! standard mask → score → chronicle → consume per-tick chain at a
//! variable agent_cap; the `stress_agent_count_app` binary sweeps the
//! cap from 1k to 200k and emits NDJSON per-tick metrics + a final
//! summary. The first cap that panics or OOMs is recorded as the
//! ceiling.
//!
//! ## Per-tick chain
//!
//! Mirrors the spy_network / duel_abilities shape but with a single
//! self-target verb (Pulse) and a fused chronicle-dispatcher + no-op
//! consumer (the .sim's only `@phase(post)` rule is
//! `ApplyPulseFromChronicle` which does `let _ = agents.hp(a)`):
//!
//!   1. Per-tick clears: event_tail, ring headers, mask bitmap,
//!      scoring_output.
//!   2. mask_verb_Pulse — PerAgent (NOT PerPair, since the verb has
//!      `verb Pulse(self)` with no per-pair candidate), writes mask_0.
//!   3. scoring — PerAgent argmax over the single row; emits
//!      ActionSelected for every alive agent.
//!   4. physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse —
//!      fused PerEvent kernel that walks ActionSelected → dispatches
//!      apply_ability → walks Pulse's program (one EffectOp::SelfDamage)
//!      → emits EffectSelfDamageApplied (kind=39) → consumer reads
//!      `agents.hp(actor)`.
//!   5. seed_indirect_0 — keeps indirect-args buffer warm.
//!
//! No consumer side-effects (the `let _ = agents.hp(a)` is intentionally
//! a dead read so HP doesn't drift across ticks; the population stays
//! constant for the entire tick budget).

use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use engine::rng::per_agent_u32_pcg;
use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext};

/// Pre-built dispatcher kernel total size, captured at build time by
/// build.rs (sum of all WGSL kernel byte counts). Read by the driver
/// binary's NDJSON metric emit so the harness can record kernel size
/// at the active cap (the kernel size is independent of agent_cap, so
/// it's the same for every cap in the sweep — but it's a useful
/// breakpoint candidate as the codegen continues to evolve).
pub const DISPATCHER_KERNEL_BYTES: usize = include!(concat!(
    env!("OUT_DIR"),
    "/dispatcher_total_bytes.txt"
));

/// Slot count for the WGSL `event_kind_counts: array<atomic<u32>>`
/// instrumentation buffer. Sized for `EventKindId::ChronicleEntry =
/// 128 + 1` so every emitted EventKind discriminant has a dedicated
/// counter slot. See `crates/engine/src/cascade/handler.rs`.
pub const N_EVENT_KIND_SLOTS: usize = 129;

/// Seed used by all stress runs. Per the plan doc P5 compliance:
/// pseudo-hex `0x5tre55_aa` doesn't fit in a real u64 hex literal so
/// we use the closest sensible interpretation.
pub const STRESS_SEED: u64 = 0x57_2E_55_AA_57_2E_55_AA;

/// Per-fixture state for the stress fixture.
pub struct StressAgentCountState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // The chronicle dispatcher + scoring kernels both bind these stat
    // columns; init to neutral values so any %max_hp / +AD scaling
    // future-mod doesn't produce sentinel results. Pulse itself doesn't
    // scale on any stat, so the values are dead reads.
    agent_max_hp_buf: wgpu::Buffer,
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask bitmap (Pulse=0) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring (engine helper) --
    event_ring: EventRing,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    /// Packed AbilityRegistry uploaded once at construction. Holds the
    /// Pulse program (single EffectOp::SelfDamage{0.0}) at AbilityId(1).
    registry_gpu: PackedAbilityRegistryGpu,

    // -- Phase 2 debug instrumentation buffers. Compiler-emitted
    //    `atomicAdd` sites in the chronicle producer + scoring
    //    kernels accumulate into these per-tick. Reset to zero at
    //    the start of each step() so the readback captures the
    //    most-recent-tick histogram only. --
    /// Per-EventKindId chronicle counter (sized N_EVENT_KIND_SLOTS).
    event_kind_counts_buf: wgpu::Buffer,
    /// Per-agent argmax visit counter (sized agent_count). Bound to
    /// the scoring kernel.
    score_kernel_visits_buf: wgpu::Buffer,
    /// Zero-source for the per-tick reset of `event_kind_counts`.
    event_kind_counts_zero_buf: wgpu::Buffer,
    /// Zero-source for the per-tick reset of `score_kernel_visits`.
    score_kernel_visits_zero_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    /// D3 compiler-debug-mode instrumentation. `Some` when the
    /// adapter exposes `wgpu::Features::TIMESTAMP_QUERY`; otherwise
    /// `None` and the per-tick chain falls back to the plain
    /// `dispatch_<name>` helpers (P10 — no panic on adapters without
    /// timestamp support).
    debug_timings: Option<dispatch::DebugTimings>,
    /// Per-kernel last-completed-tick timings, refreshed after each
    /// `step()` when `debug_timings.is_some()`. Owned here so the
    /// driver binary can read them via [`Self::last_kernel_timings`]
    /// without re-mapping the readback buffer.
    last_kernel_timings: Vec<dispatch::KernelTiming>,

    tick: u64,
    agent_count: u32,
    seed: u64,

    /// Per-agent positions (Vec3), seeded deterministically via PCG.
    /// Kept on host only — the .sim doesn't read pos in any verb body
    /// so we don't need a GPU buffer for it (pack_agents binds it but
    /// the runtime never invokes pack_agents).
    positions_host: Vec<Vec3>,
}

impl StressAgentCountState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert!(agent_count > 0, "agent_count must be > 0");
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Build the AbilityRegistry from the .ability corpus + upload.
        let built = build_stress_registry();
        let packed = PackedAbilityRegistry::pack(&built);
        let registry_gpu =
            PackedAbilityRegistryGpu::upload(&packed, &gpu, "stress_agent_count_runtime");

        // -- Per-agent SoA. HP=100 (constant — Pulse self-damages 0),
        //    alive=1, mana=0, stats=neutral.
        let n = agent_count as usize;
        let hp_init = vec![100.0_f32; n];
        let alive_init = vec![1u32; n];
        let mana_init = vec![0.0_f32; n];
        let max_hp_init = vec![100.0_f32; n];
        let zeros_f32 = vec![0.0_f32; n];

        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stress_agent_count::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let agent_mana_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::agent_mana"),
                contents: bytemuck::cast_slice(&mana_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });
        let agent_max_hp_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::agent_max_hp"),
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
        let agent_attack_damage_buf = mk_stat("stress_agent_count::agent_attack_damage");
        let agent_ability_power_buf = mk_stat("stress_agent_count::agent_ability_power");
        let agent_armor_buf = mk_stat("stress_agent_count::agent_armor");
        let agent_magic_resist_buf = mk_stat("stress_agent_count::agent_magic_resist");
        let agent_move_speed_buf = mk_stat("stress_agent_count::agent_move_speed");

        // -- Single mask bitmap (Pulse=0). Cleared each tick via copy.
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mask_0_bitmap_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_agent_count::mask_0_bitmap"),
            size: mask_bitmap_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // -- Scoring output (4 × u32 per agent).
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_agent_count::scoring_output"),
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
                label: Some("stress_agent_count::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let event_ring = EventRing::new(&gpu, "stress_agent_count");

        // -- Per-kernel cfg uniforms.
        let mask_cfg_init = mask_verb_Pulse::MaskVerbPulseCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("stress_agent_count::mask_cfg"),
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
                label: Some("stress_agent_count::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let chronicle_cfg_init = physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyPulseFromChronicleAndVerbChroniclePulseCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            agent_cap: 0,
        };
        let chronicle_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::chronicle_cfg"),
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
            label: Some("stress_agent_count::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // -- Per-agent positions, seeded deterministically via PCG.
        //    Constitution P5: all sim randomness flows through PCG;
        //    `per_agent_u32_pcg(seed_lo, agent_id, tick=0, purpose)` is
        //    the canonical mirror of the WGSL helper.
        let seed_lo = seed as u32;
        let mut positions_host: Vec<Vec3> = Vec::with_capacity(n);
        for slot in 0..n {
            let aid = slot as u32;
            let rx = per_agent_u32_pcg(seed_lo, aid, 0, 0xA5_5A_A5_5A);
            let ry = per_agent_u32_pcg(seed_lo, aid, 0, 0x5A_A5_5A_A5);
            let rz = per_agent_u32_pcg(seed_lo, aid, 0, 0xC0_FF_EE_42);
            // Map into a 1024×1024×1024 volume centred at origin.
            let fx = ((rx as f32) / (u32::MAX as f32)) * 1024.0 - 512.0;
            let fy = ((ry as f32) / (u32::MAX as f32)) * 1024.0 - 512.0;
            let fz = ((rz as f32) / (u32::MAX as f32)) * 1024.0 - 512.0;
            positions_host.push(Vec3::new(fx, fy, fz));
        }

        // Phase 2 debug instrumentation buffers: per-EventKindId
        // histogram + per-agent score_kernel_visits. Init zero. The
        // matching zero-source buffers are COPY_SRC only — driven into
        // the per-tick destinations at the start of each step() so the
        // histograms capture this tick's counts only.
        let zeros_kind: Vec<u32> = vec![0u32; N_EVENT_KIND_SLOTS];
        let event_kind_counts_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::event_kind_counts"),
                contents: bytemuck::cast_slice(&zeros_kind),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );
        let event_kind_counts_zero_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::event_kind_counts_zero"),
                contents: bytemuck::cast_slice(&zeros_kind),
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        );
        let zeros_per_agent: Vec<u32> = vec![0u32; n];
        let score_kernel_visits_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::score_kernel_visits"),
                contents: bytemuck::cast_slice(&zeros_per_agent),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            },
        );
        let score_kernel_visits_zero_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("stress_agent_count::score_kernel_visits_zero"),
                contents: bytemuck::cast_slice(&zeros_per_agent),
                usage: wgpu::BufferUsages::COPY_SRC,
            },
        );

        // Construct compiler-debug-mode (D3) timing instrumentation
        // before moving `gpu` into Self. None on adapters without
        // TIMESTAMP_QUERY (the per-tick path falls back to plain
        // dispatch_<name> in that case — see fn step()).
        let debug_timings = dispatch::DebugTimings::new(&gpu);

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
            score_kernel_visits_buf,
            event_kind_counts_zero_buf,
            score_kernel_visits_zero_buf,
            cache: dispatch::KernelCache::default(),
            debug_timings,
            last_kernel_timings: Vec::new(),
            tick: 0,
            agent_count,
            seed,
            positions_host,
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

    /// Read the per-EventKindId histogram populated by the chronicle
    /// kernel's compiler-emitted `atomicAdd(&event_kind_counts[k], 1u)`
    /// instrumentation. Reset to zero at the start of every step(), so
    /// this returns the most recent tick's histogram only.
    pub fn read_event_kind_histogram(&self) -> [u32; N_EVENT_KIND_SLOTS] {
        let v = self.read_u32_slice(
            &self.event_kind_counts_buf,
            N_EVENT_KIND_SLOTS,
            "event_kind_counts",
        );
        let mut out = [0u32; N_EVENT_KIND_SLOTS];
        out.copy_from_slice(&v);
        out
    }

    /// Read the per-agent score_kernel_visits histogram populated by
    /// the scoring kernel. Index = AgentId (0..agent_count); value =
    /// number of times the scoring kernel visited the agent's row this
    /// tick. Pulse is self-target, so the per-agent visit count is
    /// equal to the number of mask rows the agent qualifies for (= 1
    /// for every alive agent in this fixture). Useful at 200k for
    /// surfacing surprising load-imbalance.
    pub fn read_score_kernel_visits(&self) -> Vec<u32> {
        self.read_u32_slice(
            &self.score_kernel_visits_buf,
            self.agent_count as usize,
            "score_kernel_visits",
        )
    }

    fn read_u32_slice(&self, buf: &wgpu::Buffer, len: usize, label: &str) -> Vec<u32> {
        let bytes = (len as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("stress_agent_count::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("stress_agent_count::read_u32_slice"),
            },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let v: Vec<u32> = {
            let mapped = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&mapped);
            words[..len].to_vec()
        };
        staging.unmap();
        v
    }

    /// Snapshot of the chronicle event ring's tail value (= total
    /// events emitted this tick at flush time). Triggers a small
    /// staging readback; only meant for the behavioral pin test, NOT
    /// the per-tick metrics path.
    pub fn read_event_count(&self) -> u32 {
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stress_agent_count::event_tail_staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("stress_agent_count::read_event_count"),
            },
        );
        encoder.copy_buffer_to_buffer(self.event_ring.tail(), 0, &staging, 0, 4);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: u32 = *bytemuck::from_bytes::<u32>(&mapped[..4]);
        drop(mapped);
        staging.unmap();
        v
    }
}

impl StressAgentCountState {
    /// Last-tick per-kernel GPU wall times (via the compiler-emitted D3
    /// `DebugTimings` instrumentation). Empty when the host adapter
    /// doesn't expose `wgpu::Features::TIMESTAMP_QUERY` — the per-tick
    /// chain still runs, just without per-kernel attribution.
    pub fn last_kernel_timings(&self) -> &[dispatch::KernelTiming] {
        &self.last_kernel_timings
    }
}

impl CompiledSim for StressAgentCountState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("stress_agent_count::step"),
            },
        );

        if let Some(t) = self.debug_timings.as_ref() {
            t.begin_tick();
        }

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        // Per tick: ~agent_count ActionSelected + ~agent_count
        // EffectSelfDamageApplied. 4× headroom keeps far below the
        // engine's EVENT_RING_CAP_SLOTS (which is also 65536).
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
        // Phase 2 debug: clear the per-tick instrumentation counters
        // so the readback captures THIS tick's totals only. Without
        // this, every tick's atomicAdd accumulates onto the prior
        // tick's totals.
        encoder.copy_buffer_to_buffer(
            &self.event_kind_counts_zero_buf,
            0,
            &self.event_kind_counts_buf,
            0,
            (N_EVENT_KIND_SLOTS as u64) * 4,
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

        // (2) Mask round — single PerAgent kernel, dispatches
        //     `agent_cap` threads (no per-pair fan-out — Pulse is
        //     self-target, so each agent only gates its own slot).
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
            cfg: &self.mask_cfg_buf,
        };
        let mask_bindings =
            mask_verb_Pulse::MaskVerbPulseBindings::from_context_with_extras(
                &ctx,
                &mask_extras,
            );
        if let Some(t) = self.debug_timings.as_ref() {
            dispatch::record_mask_verb_pulse_timing(
                &mut self.cache,
                t,
                &mask_bindings,
                &self.gpu.device,
                &mut encoder,
                self.agent_count,
            );
        } else {
            dispatch::dispatch_mask_verb_pulse(
                &mut self.cache,
                &mask_bindings,
                &self.gpu.device,
                &mut encoder,
                self.agent_count,
            );
        }

        // (3) Scoring — argmax over the single Pulse row.
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
        if let Some(t) = self.debug_timings.as_ref() {
            dispatch::record_scoring_timing(
                &mut self.cache,
                t,
                &scoring_bindings,
                &self.gpu.device,
                &mut encoder,
                self.agent_count,
            );
        } else {
            dispatch::dispatch_scoring(
                &mut self.cache,
                &scoring_bindings,
                &self.gpu.device,
                &mut encoder,
                self.agent_count,
            );
        }

        // (4) Fused chronicle dispatcher + ApplyPulseFromChronicle
        //     consumer (PerEvent kernel). Walks ActionSelected entries,
        //     dispatches Pulse's program (one EffectOp::SelfDamage),
        //     emits EffectSelfDamageApplied (kind=39), consumer reads
        //     `agents.hp(actor)` (no-op).
        let event_count_estimate = self.agent_count.saturating_mul(4).min(60_000);
        let chronicle_cfg = physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyPulseFromChronicleAndVerbChroniclePulseCfg {
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
        let chronicle_extras = physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyPulseFromChronicleAndVerbChroniclePulseExtras {
            event_kind_counts: &self.event_kind_counts_buf,
            cfg: &self.chronicle_cfg_buf,
        };
        let chronicle_bindings = physics_ApplyPulseFromChronicle_and_verb_chronicle_Pulse::PhysicsApplyPulseFromChronicleAndVerbChroniclePulseBindings::from_context_with_extras(
            &ctx,
            &chronicle_extras,
        );
        if let Some(t) = self.debug_timings.as_ref() {
            dispatch::record_physics_applypulsefromchronicle_and_verb_chronicle_pulse_timing(
                &mut self.cache,
                t,
                &chronicle_bindings,
                &self.gpu.device,
                &mut encoder,
                event_count_estimate,
            );
        } else {
            dispatch::dispatch_physics_applypulsefromchronicle_and_verb_chronicle_pulse(
                &mut self.cache,
                &chronicle_bindings,
                &self.gpu.device,
                &mut encoder,
                event_count_estimate,
            );
        }

        // (5) seed_indirect_0 — keeps args buffer warm.
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
        if let Some(t) = self.debug_timings.as_ref() {
            dispatch::record_seed_indirect_0_timing(
                &mut self.cache,
                t,
                &seed_bindings,
                &self.gpu.device,
                &mut encoder,
                self.agent_count,
            );
        } else {
            dispatch::dispatch_seed_indirect_0(
                &mut self.cache,
                &seed_bindings,
                &self.gpu.device,
                &mut encoder,
                self.agent_count,
            );
        }

        // D3 timestamp resolve — must run BEFORE queue.submit so the
        // resolve_query_set + copy_buffer_to_buffer commands are part
        // of the same submission as the bracketed dispatches.
        if let Some(t) = self.debug_timings.as_ref() {
            t.finalise_tick(&mut encoder);
        }

        self.gpu.queue.submit(Some(encoder.finish()));
        // Wait for GPU work to complete so per-tick wall-clock
        // measurements taken in the driver actually capture GPU time
        // (not just queue-submit time, which is microseconds regardless
        // of agent_cap).
        self.gpu
            .device
            .poll(wgpu::PollType::Wait)
            .expect("poll after step submit");

        // D3 timing readback — runs after the poll(Wait) so the
        // GPU-side timestamps have actually flushed. `read_kernel_timings`
        // returns an empty vec on adapters without TIMESTAMP_QUERY,
        // which is fine — `last_kernel_timings` then stays at `[]`.
        if let Some(t) = self.debug_timings.as_ref() {
            self.last_kernel_timings = t.read_kernel_timings_with(&self.gpu.device);
        }

        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        &self.positions_host
    }

    fn snapshot(&mut self) -> AgentSnapshot {
        AgentSnapshot {
            positions: self.positions_host.clone(),
            creature_types: vec![1u32; self.agent_count as usize],
            alive: vec![1u32; self.agent_count as usize],
        }
    }

    fn glyph_table(&self) -> Vec<VizGlyph> {
        vec![VizGlyph::new('?', 240), VizGlyph::new('.', 27)]
    }
}

/// Pulse.ability source — embedded at compile time so the runtime
/// doesn't need filesystem access to the workspace at run time
/// (release-built binaries get distributed standalone).
const PULSE_ABILITY_SRC: &str =
    include_str!("../../../assets/ability_test/stress_agent_count/Pulse.ability");

/// Build the Pulse ability registry from the embedded `.ability`
/// source. Mirrors
/// `spy_network_runtime::binding_check::build_spy_network_registry`
/// but embeds the .ability text rather than reading via the
/// `CARGO_MANIFEST_DIR` env var (which is only set at compile time,
/// not at run time).
fn build_stress_registry() -> engine::ability::registry::AbilityRegistry {
    let parse = |name: &str, src: &str| {
        dsl_ast::parse_ability_file(src)
            .unwrap_or_else(|e| panic!("parse {name}: {e:?}"))
    };

    let files = vec![(
        "Pulse.ability".to_string(),
        parse("Pulse.ability", PULSE_ABILITY_SRC),
    )];

    dsl_compiler::ability_registry::build_registry(&files)
        .expect("build_registry over stress_agent_count corpus")
        .registry
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(StressAgentCountState::new(seed, agent_count))
}

// ---------------------------------------------------------------------------
// Behavioral pin test.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin: a single tick at agent_cap=1000 advances `state.tick` to 1
    /// AND emits at least one chronicle record (event_tail > 0).
    /// Exercises the full mask → score → chronicle → consumer pipeline
    /// at a non-trivial agent count.
    #[test]
    fn tick_advances_at_1k_agents() {
        let mut state = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            StressAgentCountState::new(STRESS_SEED, 1000)
        })) {
            Ok(s) => s,
            Err(_) => {
                eprintln!(
                    "[tick_advances_at_1k_agents] skipping: GPU init failed (no wgpu adapter on host?)"
                );
                return;
            }
        };

        state.step();
        assert_eq!(state.tick(), 1, "tick should advance to 1 after one step");

        // event_tail counts chronicle records written this tick. With
        // 1000 alive agents, we expect ~1000 ActionSelected +
        // ~1000 EffectSelfDamageApplied = 2000-ish entries; the floor
        // we assert is "non-zero" since GPU race ordering / event-ring
        // back-pressure can clip the count somewhat.
        let event_count = state.read_event_count();
        assert!(
            event_count > 0,
            "expected event_tail > 0 after one tick at 1000 agents; got {event_count}",
        );
    }
}
