//! Per-fixture runtime for `assets/sim/for_each_agent_probe.sim` —
//! end-to-end behavioural pin for the `for_each_agent <binder>`
//! body-shape primitive (Task #229).
//!
//! ## What this exercises
//!
//! - `for_each_agent <binder> { <body> }` parses + resolves + lowers
//!   to `CgStmt::ForEachAgentBody`.
//! - The containing per-agent rule's dispatch auto-retags from
//!   `PerAgent` to `OneShot` (single thread serialises the linear
//!   scan).
//! - The WGSL emit produces the alive-gated linear-scan loop
//!   (`for (var per_pair_candidate = 0u; … < cfg.agent_cap; …)`
//!   with `if (agent_alive[per_pair_candidate] != 0u)`).
//! - The body fires once per alive slot per tick: each tick bumps
//!   every alive agent's `mana` by +1 via
//!   `agents.set_mana(slot, agents.mana(slot) + 1.0)`.
//!
//! ## Per-tick chain
//!
//! Only the physics kernel matters for the behavioural pin. The
//! pack/unpack/snapshot kernels the schedule synthesizes are skipped
//! at dispatch time — they are scaffolding for the snapshot subsystem
//! and don't contribute to the mana increment observable.
//!
//! ## Observable
//!
//! With AGENT_COUNT=N, TICKS=T, every alive slot's mana ends at T
//! (init: 0). Dead slots stay at their initial value. The harness
//! test (`tests/probe.rs`) uses this exact contract.

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

/// Per-fixture state for the for_each_agent probe. Owns:
///   - Agent SoA (alive + mana — two u32/f32 buffers)
///   - sim_cfg buffer (the schedule's UploadSimCfg op writes through
///     this; we allocate a zeroed 4-u32 buffer that satisfies the BGL
///     storage binding)
///   - Per-kernel cfg uniform
pub struct ForEachAgentProbeState {
    gpu: GpuContext,

    /// 1 = alive, 0 = dead. The runtime initialises every slot alive
    /// so the alive gate fires for all of them; the inline test runs a
    /// second pass that marks half of the slots dead to confirm the
    /// gate actually filters.
    agent_alive_buf: wgpu::Buffer,
    /// Per-agent mana (f32). Initialised to 0; each tick bumps every
    /// alive slot by +1 via the for_each_agent body.
    agent_mana_buf: wgpu::Buffer,
    /// `sim_cfg` storage buffer — the schedule's `UploadSimCfg` op
    /// declares this as a writable storage binding. Zeroed at init
    /// time; the kernel doesn't read from it (the for_each_agent body
    /// only touches `agent_mana` + `agent_alive`).
    sim_cfg_buf: wgpu::Buffer,
    /// Per-kernel cfg uniform — `agent_cap`, `tick`, `seed`, `_pad`.
    physics_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,
    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl ForEachAgentProbeState {
    /// Construct a fresh probe state with `agent_count` slots, all
    /// initialised alive with mana=0.
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("for_each_agent_probe_runtime::agent_alive"),
                contents: bytemuck::cast_slice(&alive_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let mana_init: Vec<f32> = vec![0.0f32; agent_count as usize];
        let agent_mana_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("for_each_agent_probe_runtime::agent_mana"),
                contents: bytemuck::cast_slice(&mana_init),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // sim_cfg — declared as storage in the BGL. Allocate a zeroed
        // 4-u32 buffer (16 bytes) to satisfy the binding; the
        // for_each_agent body never reads from it.
        let sim_cfg_init: [u32; 4] = [agent_count, 0, 0, 0];
        let sim_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("for_each_agent_probe_runtime::sim_cfg"),
            contents: bytemuck::bytes_of(&sim_cfg_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let physics_cfg_init = fused_BumpAllMana::FusedBumpAllManaCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let physics_cfg_buf = gpu.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("for_each_agent_probe_runtime::physics_cfg"),
                contents: bytemuck::bytes_of(&physics_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        Self {
            gpu,
            agent_alive_buf,
            agent_mana_buf,
            sim_cfg_buf,
            physics_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Read the current mana values back from the GPU. The buffer is
    /// `agent_count` f32 entries.
    pub fn mana(&mut self) -> Vec<f32> {
        let n = self.agent_count as usize;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("for_each_agent_probe_runtime::mana_staging"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("for_each_agent_probe_runtime::mana::copy"),
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.agent_mana_buf,
            0,
            &staging,
            0,
            (n * std::mem::size_of::<f32>()) as u64,
        );
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let bytes = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
        drop(bytes);
        staging.unmap();
        out
    }

    /// Mark a single slot dead. Used by the inline test's negative-
    /// path coverage (the alive gate must filter out dead slots).
    pub fn kill(&mut self, slot: u32) {
        let zero = [0u32];
        self.gpu.queue.write_buffer(
            &self.agent_alive_buf,
            (slot as u64) * std::mem::size_of::<u32>() as u64,
            bytemuck::bytes_of(&zero[0]),
        );
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

impl CompiledSim for ForEachAgentProbeState {
    fn step(&mut self) {
        let mut encoder =
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("for_each_agent_probe_runtime::step"),
                });

        // Update the per-kernel cfg uniform with the current tick.
        let physics_cfg = fused_BumpAllMana::FusedBumpAllManaCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu
            .queue
            .write_buffer(&self.physics_cfg_buf, 0, bytemuck::bytes_of(&physics_cfg));

        let bindings = fused_BumpAllMana::FusedBumpAllManaBindings {
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            sim_cfg: &self.sim_cfg_buf,
            cfg: &self.physics_cfg_buf,
        };
        dispatch::dispatch_fused_bumpallmana(
            &mut self.cache,
            &bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // Pack/unpack/snapshot kernels are skipped — they are
        // snapshot-subsystem scaffolding the schedule synthesizes for
        // every fixture and don't contribute to the mana increment
        // observable. Skipping keeps the runtime focused on the body-
        // shape primitive under test.

        self.gpu.queue.submit(Some(encoder.finish()));
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        // No positions tracked — return an empty slice.
        &[]
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(ForEachAgentProbeState::new(seed, agent_count))
}

// ---------------------------------------------------------------------------
// Tests — behavioural pin for Task #229
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// After T ticks of stepping the probe, every alive agent's mana
    /// must equal exactly T (the per-tick increment is +1.0). This is
    /// the proof that the `for_each_agent` body actually runs per
    /// agent on the GPU.
    ///
    /// The test deliberately uses a small AGENT_COUNT so the
    /// behavioural pin runs cheaply, and tests both:
    ///   - All-alive: every slot bumps each tick.
    ///   - Mixed alive/dead: dead slots stay at their initial value
    ///     (the alive gate inside the linear scan filters them out).
    #[test]
    fn for_each_agent_body_increments_every_alive_slot_per_tick() {
        const AGENT_COUNT: u32 = 8;
        const TICKS: u64 = 10;

        let mut state = ForEachAgentProbeState::new(0, AGENT_COUNT);
        for _ in 0..TICKS {
            state.step();
        }
        let mana = state.mana();
        assert_eq!(
            mana.len(),
            AGENT_COUNT as usize,
            "mana buffer length must match agent_count"
        );
        for (slot, value) in mana.iter().enumerate() {
            assert_eq!(
                *value, TICKS as f32,
                "alive slot {slot} expected mana = {TICKS} after {TICKS} ticks, \
                 got {value} — the for_each_agent body must fire once per slot per tick"
            );
        }
    }

    #[test]
    fn for_each_agent_body_skips_dead_slots() {
        const AGENT_COUNT: u32 = 8;
        const TICKS: u64 = 10;

        let mut state = ForEachAgentProbeState::new(0, AGENT_COUNT);
        // Kill every odd slot before any ticks fire.
        for slot in (1..AGENT_COUNT).step_by(2) {
            state.kill(slot);
        }
        for _ in 0..TICKS {
            state.step();
        }
        let mana = state.mana();
        for (slot, value) in mana.iter().enumerate() {
            let expected = if slot % 2 == 0 { TICKS as f32 } else { 0.0 };
            assert_eq!(
                *value, expected,
                "slot {slot} (alive={}) expected mana = {expected}, got {value} — \
                 the alive gate inside the for_each_agent body must filter dead slots",
                slot % 2 == 0
            );
        }
    }
}
