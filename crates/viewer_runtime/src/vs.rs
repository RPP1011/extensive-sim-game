//! Vampire Survivors voxel-viewer path — parallel to the dungeon_horde ViewerApp.
//! Sim-side here (state, seeding, step+drain, mana-band snapshot); rendering in VsBridge (VD2).
use sims::vampire_survivors::GeneratedRuntime;
use sims::vampire_survivors_seed::seed_initial_state;
use sims::summon_alloc::{drain_summons, DrainCtx};

pub const VS_AGENT_COUNT: u32 = 512;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VsRole { Player, Enemy, Spawner }

pub fn role_for_mana(mana: f32) -> VsRole {
    if mana < 1.5 { VsRole::Player } else if mana < 2.5 { VsRole::Enemy } else { VsRole::Spawner }
}

#[derive(Clone, Copy)]
pub struct VsAgent { pub pos: [f32; 3], pub hp: f32, pub role: VsRole }

pub struct VsViewerApp {
    pub state: GeneratedRuntime,
    pub seed: u64,
    pub agent_count: u32,
    agents: Vec<VsAgent>,
    pub terminated_at_tick: Option<u64>,
}

impl VsViewerApp {
    pub fn try_new(seed: u64) -> Option<Self> {
        let mut state = GeneratedRuntime::try_new(seed, VS_AGENT_COUNT)?;
        seed_initial_state(&mut state);
        let mut app = Self { state, seed, agent_count: VS_AGENT_COUNT, agents: Vec::new(), terminated_at_tick: None };
        app.refresh_snapshot();
        Some(app)
    }
    pub fn sim_tick(&self) -> u64 { self.state.tick }
    pub fn agents(&self) -> &[VsAgent] { &self.agents }
    pub fn step(&mut self) {
        self.state.step();
        let _ = {
            let device = &self.state.gpu.device;
            let queue = &self.state.gpu.queue;
            let event_ring = &self.state.event_ring;
            let agent_alive_buf = &self.state.agent_alive_buf;
            let agent_pos_buf = &self.state.agent_pos_buf;
            let agent_count = self.state.agent_count;
            let seed = self.state.seed;
            let tick = self.state.tick;
            drain_summons(DrainCtx {
                device,
                queue,
                event_ring,
                agent_alive_buf,
                agent_pos_buf,
                agent_count,
                seed,
                tick,
            })
        };
        self.refresh_snapshot();
        if self.terminated_at_tick.is_none() && !self.agents.iter().any(|a| a.role == VsRole::Player) {
            self.terminated_at_tick = Some(self.state.tick);
        }
    }
    fn refresh_snapshot(&mut self) {
        let n = self.agent_count;
        // Clone buffer handles before borrowing &mut self.state — wgpu::Buffer is Arc-backed, clone is cheap.
        let pos_buf  = self.state.agent_pos_buf.clone();
        let alive_buf = self.state.agent_alive_buf.clone();
        let hp_buf   = self.state.agent_hp_buf.clone();
        let mana_buf = self.state.agent_mana_buf.clone();
        let pos  = read_vec4(&mut self.state, &pos_buf, n);
        let alive = read_u32(&mut self.state, &alive_buf, n);
        let hp   = read_f32(&mut self.state, &hp_buf, n);
        let mana = read_f32(&mut self.state, &mana_buf, n);
        self.agents.clear();
        for i in 0..n as usize {
            if alive[i] == 1 {
                self.agents.push(VsAgent {
                    pos: [pos[i][0], pos[i][1], pos[i][2]],
                    hp: hp[i],
                    role: role_for_mana(mana[i]),
                });
            }
        }
    }
}

// Inline staging-buffer readbacks (synchronous map via device.poll(Wait)).
fn read_raw_u32(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, bytes: u64) -> Vec<u32> {
    let bytes = bytes.max(16);
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vs::rb"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rt.gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    rt.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    out
}

fn read_u32(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<u32> {
    read_raw_u32(rt, buf, n as u64 * 4)
}

fn read_f32(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<f32> {
    read_raw_u32(rt, buf, n as u64 * 4)
        .into_iter()
        .map(f32::from_bits)
        .collect()
}

/// Returns `n` vec4 values as `Vec<[f32; 4]>`. agent_pos_buf stride is 16 bytes (vec3<f32> padded to vec4).
fn read_vec4(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<[f32; 4]> {
    let raw = read_raw_u32(rt, buf, n as u64 * 16);
    raw.chunks_exact(4)
        .map(|c| [f32::from_bits(c[0]), f32::from_bits(c[1]), f32::from_bits(c[2]), f32::from_bits(c[3])])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn role_bands() {
        assert_eq!(role_for_mana(1.0), VsRole::Player);
        assert_eq!(role_for_mana(2.0), VsRole::Enemy);
        assert_eq!(role_for_mana(3.0), VsRole::Spawner);
    }
}
