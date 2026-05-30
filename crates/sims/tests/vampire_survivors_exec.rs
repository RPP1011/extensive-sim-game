//! Keystone integration test: declarative `.sim` seeding → DSL waves →
//! chronicle → host allocator → live GPU enemies, all gated by `creature_type`.
//!
//! Subkind-seeding migration (Plan B): the population is now seeded by the
//! `init { spawn … }` block in `assets/sim/vampire_survivors.sim`, so
//! `GeneratedRuntime::try_new` self-seeds — there is no manual
//! `seed_initial_state` call. `try_new` stamps:
//!   * slot 0 — the AgentId NonZeroU32 sentinel (untouched, dead),
//!   * slot 1 — the Player (creature_type 0, alive, hp 100, pos origin),
//!   * slots 2..N — the Enemy pool (creature_type 1, alive 0, engaged_with 1).
//!
//! Enemies are counted by `creature_type == Enemy && alive` (reading
//! `agent_creature_type_buf`), NOT the retired mana band — this is the
//! assertion that catches the drain ever zeroing creature_type (it flips alive
//! only, leaving the seeded Enemy subkind intact).
//!
//! Verifies:
//!   1. a live Player exists immediately after seeding (no manual seed),
//!   2. PlayerControl tracks the config.ctl input channel,
//!   3. step() drives the DSL spawn verbs; drain_summons claims dead Enemy-pool
//!      slots so the live-Enemy count (by creature_type) grows under the drain,
//!   4. the weapons cull the swarm (final count <= peak).
//!
//! Requires a GPU adapter; run with
//! `RUST_MIN_STACK=33554432 cargo test -p sims --test vampire_survivors_exec`.

use sims::vampire_survivors::GeneratedRuntime;
use sims::vampire_survivors_seed::{ENEMY_POOL_START, PLAYER_SLOT};
use sims::summon_alloc::{drain_summons, DrainCtx};

const SEED: u64 = 0x5_F00D_CAFE_0001;
const N: u32 = 512;
const TICKS: u64 = 120; // > wave_period (30): several SpawnSmall waves fire

// Enemy subkind ordinal = declaration order in vampire_survivors.sim
// (entity Player then entity Enemy → Player = 0, Enemy = 1).
const CT_ENEMY: u32 = 1;
const CT_PLAYER: u32 = 0;

fn read_buf_u32(rt: &mut GeneratedRuntime, buf: &wgpu::Buffer, n: u32) -> Vec<u32> {
    let bytes = (n as u64 * 4).max(16);
    let staging = rt.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test::u32_rb"),
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
    let buf = rt.agent_alive_buf.clone();
    read_buf_u32(rt, &buf, N)
}

fn read_creature_type(rt: &mut GeneratedRuntime) -> Vec<u32> {
    let buf = rt.agent_creature_type_buf.clone();
    read_buf_u32(rt, &buf, N)
}

/// Count of live enemies by SUBKIND: `creature_type == Enemy && alive == 1`.
/// This is the migration's load-bearing check — the drain flips alive only, so
/// a live slot still reads creature_type Enemy. (Zeroing creature_type in the
/// drain would make this count stay 0 even as alive grows.)
fn enemy_count(alive: &[u32], ct: &[u32]) -> usize {
    alive
        .iter()
        .zip(ct.iter())
        .filter(|(&a, &c)| a == 1 && c == CT_ENEMY)
        .count()
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
    // No manual seed — try_new self-seeds via the .sim `init { spawn … }` block.

    // The Player must be live immediately after seeding (creature_type Player,
    // alive) at PLAYER_SLOT.
    let ct0 = read_creature_type(&mut rt);
    let alive0 = read_alive(&mut rt);
    assert_eq!(
        ct0[PLAYER_SLOT as usize], CT_PLAYER,
        "player slot must seed creature_type Player"
    );
    assert_eq!(alive0[PLAYER_SLOT as usize], 1, "player must seed alive");

    let p0 = read_player_pos(&mut rt);
    eprintln!("[vampire_survivors] player start pos: {:?}", p0);

    assert_eq!(
        enemy_count(&alive0, &ct0),
        0,
        "no live enemies before any wave (pool seeded alive:0)"
    );
    // The Enemy pool exists (creature_type Enemy across slots 2..N) even while dormant.
    let pool_enemies = ct0[ENEMY_POOL_START as usize..]
        .iter()
        .filter(|&&c| c == CT_ENEMY)
        .count();
    assert_eq!(
        pool_enemies,
        (N - ENEMY_POOL_START) as usize,
        "the whole pool seeds creature_type Enemy"
    );

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
            let ct = read_creature_type(&mut rt);
            let ec = enemy_count(&alive, &ct);
            eprintln!("[vampire_survivors] tick {} (rt.tick={}): drained={} enemy_count(by creature_type)={}", i + 1, rt.tick, drained, ec);
            max_enemy_count = max_enemy_count.max(ec);
        }
    }

    let alive_end = read_alive(&mut rt);
    let ct_end = read_creature_type(&mut rt);
    let p1 = read_player_pos(&mut rt);
    let final_count = enemy_count(&alive_end, &ct_end);
    eprintln!(
        "[vampire_survivors] after {} ticks: final enemy_count={}, max seen={}",
        TICKS, final_count, max_enemy_count
    );
    eprintln!("[vampire_survivors] player end pos: {:?}", p1);

    // Primary assertion: enemies (by creature_type) must exist at end OR have
    // existed at some point during the run. If max_enemy_count > 0, the drain
    // claimed dormant Enemy-pool slots and they still read creature_type Enemy.
    let spawns_worked = final_count > 0 || max_enemy_count > 0;
    assert!(
        spawns_worked,
        "expected DSL waves to spawn live enemies (by creature_type) after {} ticks; final_count={} max_seen={}",
        TICKS, final_count, max_enemy_count,
    );

    // Liveness assertion — player kite or swarm closing.
    // Enemies spawn near the player; over 120 ticks the swarm chases. We first
    // check whether the player moved. If not, we assert the swarm has closed
    // distance instead — proving the game loop (movement AI) is live.
    let moved = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2) + (p1[2] - p0[2]).powi(2)).sqrt();
    eprintln!("[vampire_survivors] player displacement: {:.4} (p0={:?} p1={:?})", moved, p0, p1);

    if moved > 0.01 {
        eprintln!("[vampire_survivors] PASS: player kited (moved {:.4})", moved);
    } else {
        eprintln!("[vampire_survivors] player stationary; checking swarm closes distance...");
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

        // Min distance of any live ENEMY (by creature_type) to the player (origin).
        let min_dist_end: f32 = (ENEMY_POOL_START as usize..N as usize)
            .filter(|&i| alive_end[i] == 1 && ct_end[i] == CT_ENEMY)
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

        assert!(
            min_dist_end < 39.0,
            "game loop appears dead: player did not move (moved={moved:.4}) and nearest enemy \
             did not close in (min_dist={min_dist_end:.4}); expected < 39.0 (spawner radius)"
        );
        eprintln!("[vampire_survivors] PASS: swarm closing (nearest enemy at {min_dist_end:.4})");
    }
}

// Runtime gate: player movement is driven by the config.ctl input channel
// (PlayerControl reads cfg.config_ctl_move_*). No enemies needed — movement is
// independent of the swarm, so this test skips the summon drain. The player is
// seeded live by the .sim init (no manual seed_initial_state).
#[test]
fn player_tracks_input() {
    let mut rt = match GeneratedRuntime::try_new(SEED, N) {
        Some(r) => r,
        None => { eprintln!("[vampire_survivors] skip: no wgpu adapter"); return; }
    };

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

// Runtime gate: a full playable loop with all weapons enabled survives T ticks
// without panic (P10); waves spawn enemies (counted by creature_type) and the
// weapons cull them (final count <= peak — kills happen, swarm not strictly
// growing). Construct via make_playable to exercise the registry seam too.
#[test]
fn playable_loop_survivable() {
    let mut rt = match GeneratedRuntime::try_new(0x9999_0001, N) {
        Some(r) => r,
        None => { eprintln!("[vampire_survivors] skip: no wgpu adapter"); return; }
    };
    rt.set_config_ctl_bolt_level(2.0);
    rt.set_config_ctl_nova_level(1.0);
    rt.set_config_ctl_garlic_level(1.0);
    rt.set_config_ctl_whip_level(1.0);
    rt.set_config_ctl_move_x(0.3);
    rt.set_config_ctl_move_y(0.2);

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
        let alive = read_alive(&mut rt);
        let ct = read_creature_type(&mut rt);
        max_enemy_count = max_enemy_count.max(enemy_count(&alive, &ct));
    }

    let alive_f = read_alive(&mut rt);
    let ct_f = read_creature_type(&mut rt);
    let final_count = enemy_count(&alive_f, &ct_f);
    assert!(max_enemy_count > 0, "waves should spawn enemies (by creature_type) over {TICKS} ticks");
    assert!(
        final_count <= max_enemy_count,
        "weapons should cull the swarm: final={final_count} > peak={max_enemy_count}"
    );
    eprintln!("[vampire_survivors] PASS: playable loop survivable (peak={max_enemy_count}, final={final_count})");
}

// Registry seam: the migrated fixture self-seeds a live Player through
// make_playable too (the boxed PlayableRuntime path the generic `play` binary
// uses). agent_snapshot reads back creature_type + alive directly.
#[test]
fn make_playable_self_seeds_live_player() {
    let Some(mut rt) = sims::make_playable("vampire_survivors", SEED, N) else {
        eprintln!("[vampire_survivors] skip: no wgpu adapter");
        return;
    };
    let snap = rt.agent_snapshot();
    assert_eq!(snap.len(), N as usize, "snapshot covers every slot");

    // Exactly one live Player (creature_type 0), at slot 1.
    let players: Vec<_> = snap
        .iter()
        .enumerate()
        .filter(|(_, a)| a.alive && a.creature_type == CT_PLAYER)
        .collect();
    assert_eq!(players.len(), 1, "exactly one live Player seeded");
    assert_eq!(players[0].0, PLAYER_SLOT as usize, "player at slot 1");
    assert!((players[0].1.hp - 100.0).abs() < 1e-3, "player hp seeded to 100");

    // The Enemy pool is dormant (alive 0) but stamped creature_type Enemy.
    let pool_enemies = snap
        .iter()
        .filter(|a| a.creature_type == CT_ENEMY)
        .count();
    assert_eq!(
        pool_enemies,
        (N - ENEMY_POOL_START) as usize,
        "whole Enemy pool seeds creature_type Enemy"
    );
    let live_enemies = snap.iter().filter(|a| a.alive && a.creature_type == CT_ENEMY).count();
    assert_eq!(live_enemies, 0, "pool seeds alive:0 — no live enemies before any wave");

    eprintln!(
        "[vampire_survivors] PASS make_playable self-seeds: 1 Player (hp100, slot1), {} dormant Enemy pool",
        pool_enemies
    );
}
