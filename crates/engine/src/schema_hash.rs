//! Compile-time schema hash — a baselined fingerprint over the public layout-
//! relevant types. CI compares this against `crates/engine/.schema_hash` to
//! catch unintended schema drift. When the schema intentionally changes, update
//! the baseline file.

use sha2::{Digest, Sha256};

pub fn schema_hash() -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"SimState:SoA{hot_pos=vec3,hot_hp=f32,hot_max_hp=f32,hot_alive=bool,hot_movement_mode=u8};cold{creature_type=u8,channels=smallvec4,spawn_tick=u32}");
    h.update(b"Event:AgentMoved,AgentAttacked,AgentDied,ChronicleEntry");
    h.update(b"MicroKind:Hold,MoveToward,Flee,Attack,Cast,UseItem,Harvest,Eat,Drink,Rest,PlaceTile,PlaceVoxel,HarvestVoxel,Converse,ShareStory,Communicate,Ask,Remember");
    h.update(b"CommunicationChannel:Speech,PackSignal,Pheromone,Song,Telepathy,Testimony");
    h.update(b"CreatureType:Human,Wolf,Deer,Dragon");
    h.update(b"MovementMode:Walk,Fly,Swim,Climb");
    h.finalize().into()
}
