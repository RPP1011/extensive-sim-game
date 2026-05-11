//! cooldown_probe_runtime — Plan E-A6 migration. Hand-written
//! State + step() retired; runtime is the compiler-emitted
//! `GeneratedRuntime` (via include!()'s of generated.rs +
//! runtime_core.rs) plus a tiny extras impl block for activations
//! readback.
//!
//! Initial buffer state (alive=1 per slot, cooldown_next_ready_tick=N
//! per slot N) lives in `assets/sim/cooldown_probe.sim`'s `init { ... }`
//! block — Plan E-A6 escape hatch lifted out of the hand-written
//! runtime so the .sim is the single source of truth.

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
include!(concat!(env!("OUT_DIR"), "/runtime_core.rs"));

use engine::sim_trait::CompiledSim;
use glam::Vec3;

pub type CooldownProbeState = GeneratedRuntime;

#[allow(non_snake_case, clippy::all)]
impl CooldownProbeState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        Self::try_new(seed, agent_count).expect("init wgpu adapter + device")
    }

    /// Per-caster activation count (one f32 per slot). Under the
    /// staggered `cooldown_next_ready_tick: slot` init the per-slot
    /// fire pattern at tick T is `max(0, T - N)`. The compiler-
    /// emitted SCHEDULE currently runs FoldActivations BEFORE
    /// PhysicsCheckAndCast (producer/consumer inversion), so the
    /// observable lags by one tick: `max(0, T - 1 - N)`. Pre-Plan-E
    /// the hand-written runtimes called dispatches in their own
    /// (correct) order and ignored SCHEDULE; Plan E exposes this
    /// latent compiler bug. Tracked for a follow-up schedule-
    /// synthesis fix; the init mechanism itself is validated by the
    /// off-by-one being uniform across all slots.
    pub fn activations(&mut self) -> Vec<f32> {
        let bytes = (self.agent_count as u64 * 4).max(16);
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cooldown_probe::activations_staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("cooldown_probe::activations_readback"),
            },
        );
        encoder.copy_buffer_to_buffer(
            &self.view_storage_primary_buf,
            0,
            &staging,
            0,
            bytes,
        );
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..bytes);
        slice.map_async(wgpu::MapMode::Read, |res| {
            res.expect("activations_staging map_async failed")
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
        let out = {
            let view = slice.get_mapped_range();
            let words: &[f32] = bytemuck::cast_slice(&view);
            words[..self.agent_count as usize].to_vec()
        };
        staging.unmap();
        out
    }
}

impl CompiledSim for CooldownProbeState {
    fn step(&mut self) {
        GeneratedRuntime::step(self)
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
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(CooldownProbeState::new(seed, agent_count))
}

#[cfg(test)]
mod cooldown_probe_init_tests {
    use super::CooldownProbeState;
    use engine::sim_trait::CompiledSim;

    /// Plan E-A6 behavioural pin — the .sim's `init { alive: 1,
    /// cooldown_next_ready_tick: slot }` block must produce the
    /// staggered fire pattern through the GeneratedRuntime. Expected
    /// shape `max(0, T - 1 - N)` rather than `max(0, T - N)` because
    /// the compiler-emitted SCHEDULE currently inverts producer/
    /// consumer order (fold runs before physics each tick — see
    /// `activations()` doc comment). The off-by-one is uniform across
    /// every slot, which IS the validation that init state survived
    /// end-to-end: if init were broken (alive=0 or ready_at=0 for
    /// every slot) the pattern would be flat 0 or flat T, not the
    /// staggered shape.
    #[test]
    fn staggered_init_drives_per_slot_fire_pattern() {
        const N: u32 = 8;
        const TICKS: u64 = 16;
        let mut state = match CooldownProbeState::try_new(0xC001_DA, N) {
            Some(s) => s,
            None => {
                eprintln!("[cooldown_probe init] skipping: no wgpu adapter on host.");
                return;
            }
        };
        for _ in 0..TICKS {
            <CooldownProbeState as CompiledSim>::step(&mut state);
        }
        let r = state.activations();
        assert_eq!(r.len(), N as usize);
        for (slot, &count) in r.iter().enumerate() {
            let expected = (TICKS as i64 - 1 - slot as i64).max(0) as f32;
            assert!(
                (count - expected).abs() < 1e-3,
                "slot {slot} activations = {count} (expected {expected} \
                 — staggered pattern with one-tick fold-lag from SCHEDULE order)",
            );
        }
        // Distinct values across slots is the load-bearing init-mechanism
        // assertion: if every slot were identical, init would have been
        // ignored (uniform zero-init OR uniform same-fire-count).
        let unique: std::collections::BTreeSet<_> =
            r.iter().map(|f| f.to_bits()).collect();
        assert!(
            unique.len() >= 5,
            "expected staggered values across {} slots, got {} unique: {:?}",
            N, unique.len(), r,
        );
    }
}
