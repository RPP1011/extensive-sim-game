//! Initial-state seeding for the vampire_survivors runtime.
use crate::vampire_survivors::GeneratedRuntime;

pub const PLAYER_SLOT: u32 = 0;
pub const SPAWNER_COUNT: u32 = 6;
pub const SPAWNER_SLOT_START: u32 = 1;
pub const ENEMY_POOL_START: u32 = SPAWNER_SLOT_START + SPAWNER_COUNT; // 7
pub const SPAWNER_RING_RADIUS: f32 = 40.0;

const PLAYER_MANA: f32 = 1.0;   // player band [0.5,1.5]
const ENEMY_MANA: f32 = 2.0;    // enemy band  [1.5,2.5]
const SPAWNER_MANA: f32 = 3.0;  // spawner band[2.5,3.5]
const PLAYER_HP: f32 = 100.0;
const SPAWNER_HP: f32 = 1.0e6;
const ENEMY_HP: f32 = 12.0;     // == config.vs.enemy_spawn_hp

pub fn seed_initial_state(rt: &mut GeneratedRuntime) {
    let n = rt.agent_count as usize;
    let mut alive = vec![0u32; n];
    let mut mana = vec![ENEMY_MANA; n];
    let mut hp = vec![ENEMY_HP; n];
    // agent_pos_buf stride is 16 bytes (vec3<f32> padded to vec4 in WGSL storage)
    let mut pos = vec![0.0f32; n * 4];

    // Player (slot 0)
    alive[0] = 1;
    mana[0] = PLAYER_MANA;
    hp[0] = PLAYER_HP;

    // Spawners (slots 1..=6) on a ring at the arena edge
    for i in 0..SPAWNER_COUNT {
        let slot = (SPAWNER_SLOT_START + i) as usize;
        let ang = i as f32 / SPAWNER_COUNT as f32 * std::f32::consts::TAU;
        alive[slot] = 1;
        mana[slot] = SPAWNER_MANA;
        hp[slot] = SPAWNER_HP;
        pos[slot * 4] = SPAWNER_RING_RADIUS * ang.cos();
        pos[slot * 4 + 1] = SPAWNER_RING_RADIUS * ang.sin();
    }
    // Enemy pool (slots 7..n) stays alive=0, mana=ENEMY_MANA, hp=ENEMY_HP.
    let _ = ENEMY_POOL_START;

    rt.gpu.queue.write_buffer(&rt.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    rt.gpu.queue.write_buffer(&rt.agent_mana_buf, 0, bytemuck::cast_slice(&mana));
    rt.gpu.queue.write_buffer(&rt.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
    rt.gpu.queue.write_buffer(&rt.agent_pos_buf, 0, bytemuck::cast_slice(&pos));
}
