//! Task C3 — keystone integration test: DSL waves → chronicle → host allocator → live GPU enemies.
//!
//! Verifies the full spawner pipeline end-to-end:
//!   1. seed_initial_state puts player (slot 0), 6 spawners (slots 1..=6), enemy pool dead (slots 7..N)
//!   2. step() drives DSL kernels including SpawnSmall verb chronicle emission
//!   3. drain_summons() reads kind-62 ring records, claims dead slots, writes alive=1+pos
//!   4. After TICKS steps (covering ticks 30, 60, 90 where wave_period=30 fires), enemy_count > 0

use sims::vampire_survivors::GeneratedRuntime;
use sims::vampire_survivors_seed::seed_initial_state;
use sims::summon_alloc::{drain_summons, DrainCtx};

const SEED: u64 = 0x5_F00D_CAFE_0001;
const N: u32 = 512;
const TICKS: u64 = 120; // > wave_period (30): several SpawnSmall waves fire

fn read_alive(rt: &mut GeneratedRuntime) -> Vec<u32> {
    let bytes = (N as u64 * 4).max(16);
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test::alive_rb"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rt.gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&rt.agent_alive_buf, 0, &staging, 0, bytes);
    rt.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let out = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    out
}

#[test]
fn vampire_survivors_spawns_and_runs() {
    let mut rt = match GeneratedRuntime::try_new(SEED, N) {
        Some(r) => r,
        None => {
            eprintln!("[vampire_survivors] skip: no wgpu adapter");
            return;
        }
    };
    seed_initial_state(&mut rt);

    let enemy_count = |alive: &[u32]| alive[7..].iter().filter(|&&a| a == 1).count();

    let alive0 = read_alive(&mut rt);
    assert_eq!(enemy_count(&alive0), 0, "no enemies before any wave");

    let mut max_enemy_count = 0usize;

    for i in 0..TICKS {
        rt.step();

        // Destructure needed refs into locals to avoid simultaneous borrow of rt
        let drained = {
            let device = &rt.gpu.device;
            let queue = &rt.gpu.queue;
            let event_ring = &rt.event_ring;
            let agent_alive_buf = &rt.agent_alive_buf;
            let agent_pos_buf = &rt.agent_pos_buf;
            let agent_count = rt.agent_count;
            let seed = rt.seed;
            let tick = rt.tick;
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

        if drained > 0 {
            let alive = read_alive(&mut rt);
            let ec = enemy_count(&alive);
            eprintln!("[vampire_survivors] tick {} (rt.tick={}): drained={} enemy_count={}", i + 1, rt.tick, drained, ec);
            max_enemy_count = max_enemy_count.max(ec);
        }
    }

    let alive_end = read_alive(&mut rt);
    let final_count = enemy_count(&alive_end);
    eprintln!(
        "[vampire_survivors] after {} ticks: final enemy_count={}, max seen={}",
        TICKS, final_count, max_enemy_count
    );

    // Primary assertion: enemies must exist at end OR have existed at some point during the run
    // (the player's weapons kill them after spawning — if max_enemy_count > 0, the spawn path worked).
    let spawns_worked = final_count > 0 || max_enemy_count > 0;
    assert!(
        spawns_worked,
        "expected DSL waves to spawn live enemies after {} ticks; final_count={} max_seen={}",
        TICKS, final_count, max_enemy_count,
    );
}
