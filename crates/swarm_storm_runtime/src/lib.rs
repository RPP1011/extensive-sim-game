//! Per-fixture runtime for `assets/sim/swarm_event_storm.sim`.
//! Stress fixture for the Phase 7+8 wired event-ring + view-fold
//! storage primitives under load: 4 emits per agent per tick fed
//! into a plain accumulator (`pulse_count`) and a @decay-anchored
//! accumulator (`recent_pulse_intensity` at rate=0.85). Mirrors
//! the crowd_navigation runtime shape; the additional view +
//! @decay anchor are the new pieces.

use engine::ids::AgentId;
use engine::rng::per_agent_u32;
use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Vec3Padded {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}

impl From<Vec3> for Vec3Padded {
    fn from(v: Vec3) -> Self {
        Self { x: v.x, y: v.y, z: v.z, _pad: 0.0 }
    }
}

impl From<Vec3Padded> for Vec3 {
    fn from(p: Vec3Padded) -> Self {
        Vec3::new(p.x, p.y, p.z)
    }
}

pub struct SwarmStormState {
    gpu: GpuContext,
    pos_buf: wgpu::Buffer,
    vel_buf: wgpu::Buffer,
    alive_buf: wgpu::Buffer,
    cfg_buf: wgpu::Buffer,
    pos_staging: wgpu::Buffer,

    event_ring: EventRing,

    /// Plain accumulator: pulse_count[slot] += 1 per matched Pulse.
    /// Same shape as cn's stuck_count, just at 4x the throughput.
    pulse_count: ViewStorage,
    pulse_count_cfg_buf: wgpu::Buffer,

    /// @decay-anchored accumulator: recent_pulse_intensity[slot]
    /// applies anchor *= 0.85 each tick before the deltas land.
    /// Steady state ≈ per_tick_emits / (1 - decay) = 4 / 0.15 ≈ 26.67.
    /// has_anchor=true to allocate the anchor buffer the @decay
    /// pattern needs.
    recent_pulse_intensity: ViewStorage,
    recent_pulse_intensity_cfg_buf: wgpu::Buffer,
    /// Cfg uniform for the per-tick decay kernel (B2). One thread per
    /// slot multiplies `view_storage_primary[k]` by the compile-time
    /// rate (0.85) before the per-event fold lands. The kernel reads
    /// `cfg.agent_cap` for the early-return.
    recent_pulse_intensity_decay_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,
    pos_cache: Vec<Vec3>,
    dirty: bool,
    tick: u64,
    agent_count: u32,
    seed: u64,

    /// Compiler-emitted metrics sink (Slice C of the verb/probe/metric
    /// emit plan). The `MetricsSink` struct comes from the
    /// `metrics.rs` artifact `swarm_storm_runtime/build.rs` injects
    /// alongside the kernel modules; per-tick `record_tick(self.tick)`
    /// drives every metric whose value source the emitter recognises
    /// (constant + `world.tick`). Tests + examples read fields off
    /// the sink directly.
    metrics_sink: metrics::MetricsSink,
}

fn normalise(u: u32) -> f32 {
    (u as f32 / u32::MAX as f32) * 2.0 - 1.0
}

impl SwarmStormState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let n = agent_count as usize;
        let spread = (agent_count as f32).cbrt().max(1.0);

        let mut pos_host: Vec<Vec3> = Vec::with_capacity(n);
        let mut pos_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        let mut vel_padded: Vec<Vec3Padded> = Vec::with_capacity(n);
        for slot in 0..agent_count {
            let agent_id =
                AgentId::new(slot + 1).expect("slot+1 is non-zero by construction");
            let nudge = 0.05_f32;
            let p = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"ses_init_pos_x")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"ses_init_pos_y")) * spread,
                normalise(per_agent_u32(seed, agent_id, 0, b"ses_init_pos_z")) * spread,
            );
            let v = Vec3::new(
                normalise(per_agent_u32(seed, agent_id, 0, b"ses_init_vel_x")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"ses_init_vel_y")) * nudge,
                normalise(per_agent_u32(seed, agent_id, 0, b"ses_init_vel_z")) * nudge,
            );
            pos_host.push(p);
            pos_padded.push(p.into());
            vel_padded.push(v.into());
        }
        let alive_init: Vec<u32> = vec![1u32; n];

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let pos_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("swarm_storm_runtime::pos"),
            contents: bytemuck::cast_slice(&pos_padded),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let vel_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("swarm_storm_runtime::vel"),
            contents: bytemuck::cast_slice(&vel_padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("swarm_storm_runtime::alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let cfg = physics_PulseAndDrift::PhysicsPulseAndDriftCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("swarm_storm_runtime::cfg"),
            contents: bytemuck::bytes_of(&cfg),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let pos_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("swarm_storm_runtime::pos_staging"),
            size: (n * std::mem::size_of::<Vec3Padded>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let event_ring = EventRing::new(&gpu, "swarm_storm_runtime");
        // pulse_count: plain accumulator, no @decay → has_anchor=false.
        let pulse_count = ViewStorage::new(
            &gpu,
            "swarm_storm_runtime::pulse_count",
            agent_count,
            false,
            false,
        );
        // recent_pulse_intensity: has @decay → has_anchor=true.
        let recent_pulse_intensity = ViewStorage::new(
            &gpu,
            "swarm_storm_runtime::recent_pulse_intensity",
            agent_count,
            true,
            false,
        );
        let pc_cfg = fold_pulse_count::FoldPulseCountCfg {
            event_count: 0,
            tick: 0,
            _pad: [0; 2],
        };
        let pulse_count_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("swarm_storm_runtime::pulse_count_cfg"),
                contents: bytemuck::bytes_of(&pc_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let rpi_cfg = fold_recent_pulse_intensity::FoldRecentPulseIntensityCfg {
            event_count: 0,
            tick: 0,
            _pad: [0; 2],
        };
        let recent_pulse_intensity_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("swarm_storm_runtime::recent_pulse_intensity_cfg"),
                contents: bytemuck::bytes_of(&rpi_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );
        let rpi_decay_cfg = decay_recent_pulse_intensity::DecayRecentPulseIntensityCfg {
            agent_cap: agent_count,
            tick: 0,
            _pad: [0; 2],
        };
        let recent_pulse_intensity_decay_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("swarm_storm_runtime::recent_pulse_intensity_decay_cfg"),
                contents: bytemuck::bytes_of(&rpi_decay_cfg),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            pos_buf,
            vel_buf,
            alive_buf,
            cfg_buf,
            pos_staging,
            event_ring,
            pulse_count,
            pulse_count_cfg_buf,
            recent_pulse_intensity,
            recent_pulse_intensity_cfg_buf,
            recent_pulse_intensity_decay_cfg_buf,
            cache: dispatch::KernelCache::default(),
            pos_cache: pos_host,
            dirty: false,
            tick: 0,
            agent_count,
            seed,
            metrics_sink: metrics::MetricsSink::default(),
        }
    }

    /// Read-only handle to the compiler-emitted metrics sink. Tests
    /// + examples introspect the sink fields (`tick_gauge.last`,
    /// `tick_counter.total`, `tick_histogram.samples`, ...) to
    /// observe the per-tick recorded values.
    pub fn metrics(&self) -> &metrics::MetricsSink {
        &self.metrics_sink
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn pulse_counts(&mut self) -> &[f32] {
        self.pulse_count.readback(&self.gpu)
    }

    pub fn recent_pulse_intensities(&mut self) -> &[f32] {
        self.recent_pulse_intensity.readback(&self.gpu)
    }

    fn read_positions(&mut self) -> &[Vec3] {
        if self.dirty {
            let mut encoder = self.gpu.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("swarm_storm_runtime::positions::copy"),
                },
            );
            encoder.copy_buffer_to_buffer(
                &self.pos_buf,
                0,
                &self.pos_staging,
                0,
                (self.agent_count as u64) * std::mem::size_of::<Vec3Padded>() as u64,
            );
            self.gpu.queue.submit(Some(encoder.finish()));

            let slice = self.pos_staging.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");

            let bytes = slice.get_mapped_range();
            let padded: &[Vec3Padded] = bytemuck::cast_slice(&bytes);
            for (cache, p) in self.pos_cache.iter_mut().zip(padded.iter()) {
                *cache = (*p).into();
            }
            drop(bytes);
            self.pos_staging.unmap();
            self.dirty = false;
        }
        &self.pos_cache
    }
}

impl CompiledSim for SwarmStormState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("swarm_storm_runtime::step"),
            });

        self.event_ring.clear_tail_in(&mut encoder);

        // (1) PulseAndDrift — emits 4 Pulse events per alive agent.
        let bindings = physics_PulseAndDrift::PhysicsPulseAndDriftBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_pos: &self.pos_buf,
            agent_alive: &self.alive_buf,
            agent_vel: &self.vel_buf,
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_physics_pulseanddrift(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // 4 Pulse events per agent per tick.
        let event_count = self.agent_count * 4;
        let pc_cfg = fold_pulse_count::FoldPulseCountCfg {
            event_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.pulse_count_cfg_buf,
            0,
            bytemuck::bytes_of(&pc_cfg),
        );
        let rpi_cfg = fold_recent_pulse_intensity::FoldRecentPulseIntensityCfg {
            event_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.recent_pulse_intensity_cfg_buf,
            0,
            bytemuck::bytes_of(&rpi_cfg),
        );

        // (2) fold_pulse_count — RMW primary by 1.0 per Pulse.
        // Dispatch dimension is the EVENT count (= agent_count * 4),
        // not agent_count. The compiler-emitted dispatch helper
        // takes its size arg as `agent_cap` for hysterical raisins,
        // but the underlying kernel uses `gid.x` as `event_idx`
        // (one thread per event) and early-returns past
        // `cfg.event_count`. Without this, only the first
        // agent_count events would be processed and 3 of every 4
        // emits would be dropped on the floor — which is exactly
        // what swarm_event_storm surfaces (pulse_count = 1×ticks
        // per slot instead of 4×ticks). When the event-ring fold
        // dispatch migrates to dispatch_workgroups_indirect via
        // seed_indirect_0's args buffer, this caller-side sizing
        // becomes irrelevant.
        let pc_bindings = fold_pulse_count::FoldPulseCountBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.pulse_count.primary(),
            view_storage_anchor: self.pulse_count.anchor(),
            view_storage_ids: self.pulse_count.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.pulse_count_cfg_buf,
        };
        dispatch::dispatch_fold_pulse_count(
            &mut self.cache,
            &pc_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count,
        );

        // (3a) decay_recent_pulse_intensity — B2 anchor multiply.
        // PerAgent dispatch (one thread per slot). MUST run before
        // the fold so the per-event deltas land on the decayed value.
        // Steady-state per slot ≈ per_tick_emits / (1 - decay_rate)
        //                       = 4 / 0.15 ≈ 26.67. Without this
        // multiply the value grew unbounded across ticks (the prior
        // "KNOWN GAP B2" diagnostic).
        let rpi_decay_cfg = decay_recent_pulse_intensity::DecayRecentPulseIntensityCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            _pad: [0; 2],
        };
        self.gpu.queue.write_buffer(
            &self.recent_pulse_intensity_decay_cfg_buf,
            0,
            bytemuck::bytes_of(&rpi_decay_cfg),
        );
        let rpi_decay_bindings =
            decay_recent_pulse_intensity::DecayRecentPulseIntensityBindings {
                view_storage_primary: self.recent_pulse_intensity.primary(),
                cfg: &self.recent_pulse_intensity_decay_cfg_buf,
            };
        dispatch::dispatch_decay_recent_pulse_intensity(
            &mut self.cache,
            &rpi_decay_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (3b) fold_recent_pulse_intensity — same Pulse stream,
        // with @decay anchor. Same event-count dispatch sizing.
        let rpi_bindings =
            fold_recent_pulse_intensity::FoldRecentPulseIntensityBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.recent_pulse_intensity.primary(),
                view_storage_anchor: self.recent_pulse_intensity.anchor(),
                view_storage_ids: self.recent_pulse_intensity.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.recent_pulse_intensity_cfg_buf,
            };
        dispatch::dispatch_fold_recent_pulse_intensity(
            &mut self.cache,
            &rpi_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.dirty = true;
        self.pulse_count.mark_dirty();
        self.recent_pulse_intensity.mark_dirty();

        // Slice C metric emit — drive every auto-driveable metric the
        // compiler synthesised. Pre-increment so `world.tick` mirrors
        // the tick the just-completed step belonged to (tick 0 ran,
        // record_tick(0) fires, then tick advances). The compiler-
        // emitted `record_tick` honors per-metric `emit_every`.
        self.metrics_sink.record_tick(self.tick);

        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        self.read_positions()
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(SwarmStormState::new(seed, agent_count))
}
