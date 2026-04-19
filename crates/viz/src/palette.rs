//! Voxel palette. Indices are chosen so humans are blue and wolves red
//! exactly as in `src/world_sim/voxel_bridge.rs`, so visual meaning
//! transfers between the legacy renderer and this viz.

pub const PAL_AIR:      u8 = 0;
pub const PAL_GROUND:   u8 = 1;
pub const PAL_HUMAN:    u8 = 10; // CreatureType::Human  = 0
pub const PAL_WOLF:     u8 = 11; // CreatureType::Wolf   = 1
pub const PAL_DEER:     u8 = 12; // CreatureType::Deer   = 2
pub const PAL_DRAGON:   u8 = 13; // CreatureType::Dragon = 3
pub const PAL_ATTACK:   u8 = 20;
pub const PAL_DEATH:    u8 = 21;
pub const PAL_ANNOUNCE: u8 = 22;

pub fn build_palette_rgba() -> [[u8; 4]; 256] {
    let mut p = [[0u8, 0, 0, 0]; 256];
    p[PAL_GROUND as usize]   = [120, 125, 128, 255]; // matches voxel_bridge::Stone
    p[PAL_HUMAN as usize]    = [ 60, 120, 220, 255]; // matches voxel_bridge::NpcIdle
    p[PAL_WOLF as usize]     = [180,  30,  30, 255]; // matches voxel_bridge::MonsterMarker
    p[PAL_DEER as usize]     = [210, 180, 120, 255];
    p[PAL_DRAGON as usize]   = [220,  80,  20, 255];
    p[PAL_ATTACK as usize]   = [230,  60,  40, 255];
    p[PAL_DEATH as usize]    = [ 10,  10,  10, 255];
    p[PAL_ANNOUNCE as usize] = [240, 245, 250, 255];
    p
}

pub fn creature_palette_index(ct: engine::creature::CreatureType) -> u8 {
    use engine::creature::CreatureType as CT;
    match ct {
        CT::Human  => PAL_HUMAN,
        CT::Wolf   => PAL_WOLF,
        CT::Deer   => PAL_DEER,
        CT::Dragon => PAL_DRAGON,
    }
}
