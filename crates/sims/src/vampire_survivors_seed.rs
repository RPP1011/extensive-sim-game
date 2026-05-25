//! Initial-state seeding for the vampire_survivors runtime.
use crate::vampire_survivors::GeneratedRuntime;

// Slot 0 is left UNUSED: AgentId is NonZeroU32, so slot 0 is the "absent"
// sentinel. The player lives at a nonzero slot so enemies can reference it by
// a valid AgentId (`engaged_with`) for a direct cross-agent position read.
// There are no spawner agents — the player itself emits the wave Summon and
// the host drain places summoned enemies in a circle AROUND the player.
pub const PLAYER_SLOT: u32 = 1;
pub const ENEMY_POOL_START: u32 = 2; // slots 2..N are the enemy pool

const PLAYER_MANA: f32 = 1.0; // player band [0.5,1.5]
const ENEMY_MANA: f32 = 2.0; // enemy band  [1.5,2.5]
const PLAYER_HP: f32 = 100.0;
const ENEMY_HP: f32 = 12.0; // default; the drain overrides per enemy type
const ENEMY_MOVE_SPEED: f32 = 0.4; // default; the drain overrides per enemy type

pub fn seed_initial_state(rt: &mut GeneratedRuntime) {
    let n = rt.agent_count as usize;
    let mut alive = vec![0u32; n];
    let mut mana = vec![ENEMY_MANA; n];
    let mut hp = vec![ENEMY_HP; n];
    let move_speed = vec![ENEMY_MOVE_SPEED; n];
    // agent_pos_buf stride is 16 bytes (vec3<f32> padded to vec4 in WGSL storage).
    // Player spawns at origin (all-zero); enemies are positioned by the drain.
    let pos = vec![0.0f32; n * 4];
    // Every agent's `engaged_with` points at the player's slot, so the chase
    // rule's `agents.pos(self.engaged_with)` reads the player directly. The
    // player never changes slots, so this is set once at seed time.
    let engaged = vec![PLAYER_SLOT; n];

    // Player (slot 1, arena centre/origin).
    let p = PLAYER_SLOT as usize;
    alive[p] = 1;
    mana[p] = PLAYER_MANA;
    hp[p] = PLAYER_HP;

    // Enemy pool (slots ENEMY_POOL_START..n) stays alive=0; the summon drain
    // claims these slots and overrides mana/hp/move_speed/pos per enemy type.
    let _ = ENEMY_POOL_START;

    rt.gpu.queue.write_buffer(&rt.agent_alive_buf, 0, bytemuck::cast_slice(&alive));
    rt.gpu.queue.write_buffer(&rt.agent_mana_buf, 0, bytemuck::cast_slice(&mana));
    rt.gpu.queue.write_buffer(&rt.agent_hp_buf, 0, bytemuck::cast_slice(&hp));
    rt.gpu.queue.write_buffer(&rt.agent_move_speed_buf, 0, bytemuck::cast_slice(&move_speed));
    rt.gpu.queue.write_buffer(&rt.agent_pos_buf, 0, bytemuck::cast_slice(&pos));
    rt.gpu.queue.write_buffer(&rt.agent_engaged_with_buf, 0, bytemuck::cast_slice(&engaged));
}
