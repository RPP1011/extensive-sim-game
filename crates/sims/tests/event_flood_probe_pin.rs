//! Regression pin: per-event consumers consume every live ring record
//! exactly once, even when the ring holds far more records than there
//! are agents (`assets/sim/event_flood_probe.sim`: 4 events per agent
//! per tick). Per-event kernels are dispatched with an agent-sized
//! thread count; before the grid-stride fix every record at ring slot
//! >= agent_cap was silently never consumed.
//!
//! GPU-backed: skips (passes silently) when no wgpu adapter exists.

use sims::event_flood_probe::GeneratedRuntime;

fn read_u32(st: &GeneratedRuntime, buf: &wgpu::Buffer, n: usize) -> Vec<u32> {
    let bytes = (n as u64 * 4).max(16);
    let staging = st.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("event_flood_probe::readback"),
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

fn run(agents: u32, ticks: u32) -> Option<Vec<Vec<u32>>> {
    let mut st = GeneratedRuntime::try_new(0xF100D, agents)?;
    for _ in 0..ticks {
        st.step();
    }
    let n = agents as usize;
    Some(vec![
        read_u32(&st, &st.agent_cnt0_buf, n),
        read_u32(&st, &st.agent_cnt1_buf, n),
        read_u32(&st, &st.agent_cnt2_buf, n),
        read_u32(&st, &st.agent_cnt3_buf, n),
    ])
}

#[test]
fn every_event_is_consumed_exactly_once() {
    // Small (one workgroup, 16 threads active for 64 events), mid, and a
    // count that is not a multiple of the workgroup size.
    for &(agents, ticks) in &[(16u32, 12u32), (256, 20), (1000, 8)] {
        let Some(cnts) = run(agents, ticks) else {
            eprintln!("[event_flood_probe] skipping: no wgpu adapter on host.");
            return;
        };
        for (k, c) in cnts.iter().enumerate() {
            let bad: Vec<(usize, u32)> =
                c.iter().copied().enumerate().filter(|&(_, v)| v != ticks).take(8).collect();
            assert!(
                bad.is_empty(),
                "agents={agents}: counter cnt{k} != {ticks} ticks for (slot, value) {bad:?} \
                 — events dropped (< ticks) or double-consumed (> ticks)"
            );
        }
        // Determinism: a second run is identical.
        assert_eq!(Some(cnts), run(agents, ticks), "agents={agents}: rerun differs");
    }
}
