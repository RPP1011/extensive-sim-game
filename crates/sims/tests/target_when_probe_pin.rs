//! Regression pin: scoring honours a verb's target-dependent `when` per
//! candidate (`assets/sim/target_when_probe.sim`). Before the fix the
//! argmax walked every candidate without re-checking `when` (PickEven
//! chose the rejected slot 15) and a target-free score produced
//! NO_TARGET (PickAny).
//!
//! GPU-backed: skips (passes silently) when no wgpu adapter exists.

use sims::target_when_probe::GeneratedRuntime;

fn read_u32(st: &GeneratedRuntime, buf: &wgpu::Buffer, n: usize) -> Vec<u32> {
    let bytes = (n as u64 * 4).max(16);
    let staging = st.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("target_when_probe::readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = st.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    st.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    st.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range())[..n].to_vec();
    staging.unmap();
    out
}

#[test]
fn scoring_only_picks_targets_the_verb_when_admits() {
    const N: u32 = 16;
    let Some(mut st) = GeneratedRuntime::try_new(0x7A26E7, N) else {
        eprintln!("[target_when_probe] skipping: no wgpu adapter on host.");
        return;
    };
    for _ in 0..3 {
        st.step();
    }
    let picked = read_u32(&st, &st.agent_picked_buf, N as usize);
    let verb = read_u32(&st, &st.agent_verb_id_buf, N as usize);
    for a in 0..N {
        let (want_verb, want_target) = if a < 8 {
            // Highest even idx other than self.
            (1, if a == 14 { 12 } else { 14 })
        } else {
            // Lowest idx with idx % 3 == 1 other than self (constant score).
            (2, if a == 1 { 4 } else { 1 })
        };
        assert_eq!(verb[a as usize], want_verb, "actor {a}: verb");
        assert_eq!(
            picked[a as usize],
            want_target + 1,
            "actor {a}: picked target (idx+1); 0 or a wrapped value means NO_TARGET, \
             an odd / non-(3k+1) idx means a `when`-rejected target"
        );
    }
}
