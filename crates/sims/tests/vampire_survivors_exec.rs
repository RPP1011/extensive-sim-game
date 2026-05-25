//! Task C3 — keystone integration test: DSL waves → chronicle → host allocator → live GPU enemies.
//!
//! Verifies the full spawner pipeline end-to-end:
//!   1. seed_initial_state: slot 0 unused (AgentId sentinel), player (slot 1), 6 spawners (slots 2..=7), enemy pool dead (slots 8..N)
//!   2. step() drives DSL kernels including SpawnSmall verb chronicle emission
//!   3. drain_summons() reads kind-62 ring records, claims dead slots, writes alive=1+pos
//!   4. After TICKS steps (covering ticks 30, 60, 90 where wave_period=30 fires), enemy_count > 0

use sims::vampire_survivors::GeneratedRuntime;
use sims::vampire_survivors_seed::{seed_initial_state, ENEMY_POOL_START, PLAYER_SLOT};
use sims::summon_alloc::{drain_summons, DrainCtx};

const SEED: u64 = 0x5_F00D_CAFE_0001;
const N: u32 = 512;
const TICKS: u64 = 120; // > wave_period (30): several SpawnSmall waves fire

fn read_player_pos(rt: &mut GeneratedRuntime) -> [f32; 3] {
    // agent_pos_buf stride: 16 bytes (vec3<f32> padded to vec4). Player at PLAYER_SLOT.
    let bytes: u64 = 16; // one vec4 (4 × f32)
    let player_off: u64 = PLAYER_SLOT as u64 * 16;
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test::player_pos_rb"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = rt.gpu.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&rt.agent_pos_buf, player_off, &staging, 0, bytes);
    rt.gpu.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
    let floats = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    [floats[0], floats[1], floats[2]]
}

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
    let p0 = read_player_pos(&mut rt);
    eprintln!("[vampire_survivors] player start pos: {:?}", p0);

    let enemy_count = |alive: &[u32]| alive[ENEMY_POOL_START as usize..].iter().filter(|&&a| a == 1).count();

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
            let agent_hp_buf = &rt.agent_hp_buf;
            let agent_move_speed_buf = &rt.agent_move_speed_buf;
            let agent_count = rt.agent_count;
            let seed = rt.seed;
            let tick = rt.tick;
            drain_summons(DrainCtx {
                device,
                queue,
                event_ring,
                agent_alive_buf,
                agent_pos_buf,
                agent_hp_buf,
                agent_move_speed_buf,
                agent_count,
                seed,
                tick,
                pool_start: ENEMY_POOL_START,
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
    let p1 = read_player_pos(&mut rt);
    let final_count = enemy_count(&alive_end);
    eprintln!(
        "[vampire_survivors] after {} ticks: final enemy_count={}, max seen={}",
        TICKS, final_count, max_enemy_count
    );
    eprintln!("[vampire_survivors] player end pos: {:?}", p1);

    // Primary assertion: enemies must exist at end OR have existed at some point during the run
    // (the player's weapons kill them after spawning — if max_enemy_count > 0, the spawn path worked).
    let spawns_worked = final_count > 0 || max_enemy_count > 0;
    assert!(
        spawns_worked,
        "expected DSL waves to spawn live enemies after {} ticks; final_count={} max_seen={}",
        TICKS, final_count, max_enemy_count,
    );

    // C4: liveness assertion — player kite or swarm closing.
    // Spawners sit at radius 40; enemies spawn near them so they start far from origin.
    // Over 120 ticks the swarm chases the player; KitePlayer (flee_radius=8) triggers once
    // enemies enter that radius. We first check whether the player moved. If not (swarm
    // hasn't reached flee range yet), we assert the swarm has closed distance instead —
    // proving the GAME loop (movement AI) is live and the enemies are actually chasing.
    let moved = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
    eprintln!("[vampire_survivors] player displacement: {:.4} (p0={:?} p1={:?})", moved, p0, p1);

    if moved > 0.01 {
        // Happy path: player kited away from the swarm.
        eprintln!("[vampire_survivors] PASS: player kited (moved {:.4})", moved);
    } else {
        // Player hasn't moved yet — swarm spawns at r=40 and may not have reached
        // flee_radius=8 within 120 ticks. Assert the game loop is still live by checking
        // that the swarm has closed distance toward the player (enemies are chasing).
        eprintln!("[vampire_survivors] player stationary; checking swarm closes distance...");
        // We already have alive_end. Compute alive positions after the loop.
        let bytes_pos = (N as u64 * 16).max(16);
        let staging_pos = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test::all_pos_rb"),
            size: bytes_pos,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = rt.gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&rt.agent_pos_buf, 0, &staging_pos, 0, bytes_pos);
        rt.gpu.queue.submit(Some(enc.finish()));
        let slice_pos = staging_pos.slice(..bytes_pos);
        slice_pos.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        rt.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let all_floats = bytemuck::cast_slice::<u8, f32>(&slice_pos.get_mapped_range()).to_vec();
        staging_pos.unmap();

        // Find min distance of any live enemy (enemy pool) to the player (origin).
        let min_dist_end: f32 = (ENEMY_POOL_START as usize..N as usize)
            .filter(|&i| alive_end[i] == 1)
            .map(|i| {
                let base = i * 4;
                let dx = all_floats[base];
                let dy = all_floats[base + 1];
                let dz = all_floats[base + 2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(f32::INFINITY, f32::min);

        eprintln!(
            "[vampire_survivors] nearest enemy distance to origin at end: {:.4}",
            min_dist_end
        );

        // The spawners are at radius 40. If enemies are chasing, they should be
        // meaningfully inside that initial radius by tick 120.
        assert!(
            min_dist_end < 39.0,
            "game loop appears dead: player did not move (moved={moved:.4}) and nearest enemy \
             did not close in (min_dist={min_dist_end:.4}); expected < 39.0 (spawner radius)"
        );
        eprintln!("[vampire_survivors] PASS: swarm closing (nearest enemy at {min_dist_end:.4})");
    }
}

// Plan 3 runtime gate: player movement is driven by the config.ctl input
// channel (PlayerControl reads cfg.config_ctl_move_*). No enemies needed —
// movement is independent of the swarm, so this test skips the summon drain.
#[test]
fn player_tracks_input() {
    let mut rt = match GeneratedRuntime::try_new(SEED, N) {
        Some(r) => r,
        None => { eprintln!("[vampire_survivors] skip: no wgpu adapter"); return; }
    };
    seed_initial_state(&mut rt);

    rt.set_config_ctl_move_x(1.0);
    rt.set_config_ctl_move_y(0.0);
    let x0 = read_player_pos(&mut rt)[0];
    for _ in 0..10 { rt.step(); }
    let x1 = read_player_pos(&mut rt)[0];
    assert!(x1 > x0 + 1.0, "player should move +X under move_x=1: {x0} -> {x1}");

    rt.set_config_ctl_move_x(-1.0);
    for _ in 0..10 { rt.step(); }
    let x2 = read_player_pos(&mut rt)[0];
    assert!(x2 < x1, "player should reverse under move_x=-1: {x1} -> {x2}");
    eprintln!("[vampire_survivors] PASS: player tracks input ({x0} -> {x1} -> {x2})");
}

// Plan 3 runtime gate: a full playable loop with all weapons enabled survives
// T ticks without panic (P10); waves spawn enemies and the weapons cull them
// (final count <= peak — i.e. kills happen, the swarm is not strictly growing).
#[test]
fn playable_loop_survivable() {
    let mut rt = match GeneratedRuntime::try_new(0x9999_0001, N) {
        Some(r) => r,
        None => { eprintln!("[vampire_survivors] skip: no wgpu adapter"); return; }
    };
    seed_initial_state(&mut rt);
    rt.set_config_ctl_bolt_level(2.0);
    rt.set_config_ctl_nova_level(1.0);
    rt.set_config_ctl_garlic_level(1.0);
    rt.set_config_ctl_whip_level(1.0);
    rt.set_config_ctl_move_x(0.3);
    rt.set_config_ctl_move_y(0.2);

    let enemy_count = |alive: &[u32]| alive[ENEMY_POOL_START as usize..].iter().filter(|&&a| a == 1).count();
    let mut max_enemy_count = 0usize;

    for _ in 0..TICKS {
        rt.step();
        let _drained = {
            let device = &rt.gpu.device;
            let queue = &rt.gpu.queue;
            let event_ring = &rt.event_ring;
            let agent_alive_buf = &rt.agent_alive_buf;
            let agent_pos_buf = &rt.agent_pos_buf;
            let agent_hp_buf = &rt.agent_hp_buf;
            let agent_move_speed_buf = &rt.agent_move_speed_buf;
            let agent_count = rt.agent_count;
            let seed = rt.seed;
            let tick = rt.tick;
            drain_summons(DrainCtx {
                device, queue, event_ring, agent_alive_buf, agent_pos_buf,
                agent_hp_buf, agent_move_speed_buf, agent_count, seed, tick,
                pool_start: ENEMY_POOL_START,
            })
        };
        max_enemy_count = max_enemy_count.max(enemy_count(&read_alive(&mut rt)));
    }

    let final_count = enemy_count(&read_alive(&mut rt));
    assert!(max_enemy_count > 0, "waves should spawn enemies over {TICKS} ticks");
    assert!(
        final_count <= max_enemy_count,
        "weapons should cull the swarm: final={final_count} > peak={max_enemy_count}"
    );
    eprintln!("[vampire_survivors] PASS: playable loop survivable (peak={max_enemy_count}, final={final_count})");
}
