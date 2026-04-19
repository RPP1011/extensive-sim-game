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
    h.update(b"MacroKind:NoOp,PostQuest,AcceptQuest,Bid,Announce");
    h.update(b"AnnounceAudience:Group,Area,Anyone");
    h.update(b"Resolution:HighestBid,FirstAcceptable,MutualAgreement,Coalition,Majority");
    h.update(b"QuestCategory:Physical,Political,Personal,Economic,Narrative");
    h.update(b"QueryKind:AboutEntity,AboutKind,AboutAll");
    h.update(b"MemoryKind:Combat,Trade,Social,Political,Other");
    h.update(b"EventKindId:AgentMoved=0,AgentAttacked=1,AgentDied=2,ChronicleEntry=128");
    h.update(b"Lane:Validation=0,Effect=1,Reaction=2,Audit=3");
    h.update(b"MAX_CASCADE_ITERATIONS=8");
    h.finalize().into()
}
