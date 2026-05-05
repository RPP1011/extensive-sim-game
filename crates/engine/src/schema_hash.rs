//! Compile-time schema hash — a baselined fingerprint over the public layout-
//! relevant types. CI compares this against `crates/engine/.schema_hash` to
//! catch unintended schema drift. When the schema intentionally changes, update
//! the baseline file.

use sha2::{Digest, Sha256};

pub fn schema_hash() -> [u8; 32] {
    let mut h = Sha256::new();
    // SimState SoA layout — expanded by the 2026-04-19 state-port plan to
    // cover state.md's full agent catalogue. Hot fields are read every tick;
    // cold fields land on spawn/chronicle/debug paths. See
    // `docs/superpowers/plans/2026-04-19-engine-plan-state-port.md`.
    h.update(b"SimState:SoA{");
    h.update(b"hot_pos=vec3,hot_hp=f32,hot_max_hp=f32,hot_alive=bool,hot_movement_mode=u8,");
    h.update(b"hot_level=u32,hot_move_speed=f32,hot_move_speed_mult=f32,");
    h.update(b"hot_shield_hp=f32,hot_armor=f32,hot_magic_resist=f32,hot_attack_damage=f32,hot_attack_range=f32,hot_mana=f32,hot_max_mana=f32,");
    h.update(b"hot_hunger=f32,hot_thirst=f32,hot_rest_timer=f32,");
    h.update(b"hot_safety=f32,hot_shelter=f32,hot_social=f32,hot_purpose=f32,hot_esteem=f32,");
    h.update(b"hot_risk_tolerance=f32,hot_social_drive=f32,hot_ambition=f32,hot_altruism=f32,hot_curiosity=f32,");
    h.update(b"hot_engaged_with=OptionAgentId,hot_stun_expires_at_tick=u32,hot_slow_expires_at_tick=u32,hot_slow_factor_q8=i16,hot_cooldown_next_ready_tick=u32,hot_root_expires_at_tick=u32,hot_silence_expires_at_tick=u32,hot_fear_expires_at_tick=u32,hot_taunt_expires_at_tick=u32");
    h.update(b"};cold{");
    h.update(b"creature_type=u8,channels=smallvec4,spawn_tick=u32,");
    h.update(b"grid_id=Option<u32>,local_pos=Option<vec3>,move_target=Option<vec3>,");
    h.update(b"status_effects=smallvec8<StatusEffect>,memberships=smallvec4<Membership>,inventory=Inventory,memory=views::Memory{per_entity_ring(K=64)},relationships=smallvec8<Relationship>,");
    h.update(b"class_definitions=[ClassSlot;4],creditor_ledger=smallvec16<Creditor>,mentor_lineage=[Option<AgentId>;8]");
    h.update(b"}");
    h.update(b"StatusEffect{kind=u8,source=AgentId,remaining_ticks=u32,payload_q8=i16}");
    h.update(b"StatusEffectKind:Stun=0,Slow=1,Root=2,Silence=3,Dot=4,Hot=5,Buff=6,Debuff=7");
    h.update(b"Membership{group=GroupId,role=u8,joined_tick=u32,standing_q8=i16}");
    h.update(b"GroupRole:Member=0,Officer=1,Leader=2,Founder=3,Apprentice=4,Outcast=5");
    h.update(b"Inventory{gold=i32,commodities=[u16;8]}");
    h.update(b"MemoryEvent{source=AgentId,kind=u8,payload=u64,confidence_q8=u8,tick=u32}");
    h.update(b"Relationship{other=AgentId,valence_q8=i16,tenure_ticks=u32}");
    h.update(b"ClassSlot{class_tag=u32,level=u8}");
    h.update(b"Creditor{creditor=AgentId,amount=u32}");
    h.update(b"MentorLink{mentor=AgentId,discipline=u8}");
    h.update(b"LanguageId=NonZeroU16;Capabilities{channels,languages=smallvec4<LanguageId>,can_fly,can_build,can_trade,can_climb,can_tunnel,can_marry,max_spouses=u8}");
    h.update(b"Event:AgentMoved,AgentAttacked,AgentDied,AgentFled,AgentAte,AgentDrank,AgentRested,AgentCast{caster,ability,target,depth=u8,tick},AgentUsedItem,AgentHarvested,AgentPlacedTile,AgentPlacedVoxel,AgentHarvestedVoxel,AgentConversed,AgentSharedStory,AgentCommunicated,InformationRequested,AgentRemembered,QuestPosted,QuestAccepted,BidPlaced,AnnounceEmitted,RecordMemory,OpportunityAttackTriggered,EffectDamageApplied,EffectHealApplied,EffectShieldApplied,EffectStunApplied,EffectSlowApplied,EffectGoldTransfer,EffectStandingDelta,CastDepthExceeded,ChronicleEntry");
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
    h.update(b"EventKindId:AgentMoved=0,AgentAttacked=1,AgentDied=2,AgentFled=3,AgentAte=4,AgentDrank=5,AgentRested=6,AgentCast=7,AgentUsedItem=8,AgentHarvested=9,AgentPlacedTile=10,AgentPlacedVoxel=11,AgentHarvestedVoxel=12,AgentConversed=13,AgentSharedStory=14,AgentCommunicated=15,InformationRequested=16,AgentRemembered=17,QuestPosted=18,QuestAccepted=19,BidPlaced=20,AnnounceEmitted=21,RecordMemory=22,OpportunityAttackTriggered=25,EffectDamageApplied=26,EffectHealApplied=27,EffectShieldApplied=28,EffectStunApplied=29,EffectSlowApplied=30,EffectGoldTransfer=31,EffectStandingDelta=32,CastDepthExceeded=33,ChronicleEntry=128");
    h.update(b"AbilityId=NonZeroU32;MAX_EFFECTS_PER_PROGRAM=4");
    h.update(b"Delivery:Instant=0;Area:SingleTarget{range=f32};Gate{cooldown_ticks=u32,hostile_only=bool,line_of_sight=bool};TargetSelector:Target=0,Caster=1");
    h.update(b"EffectOp:Damage=0{amount=f32},Heal=1{amount=f32},Shield=2{amount=f32},Stun=3{duration_ticks=u32},Slow=4{duration_ticks=u32,factor_q8=i16},TransferGold=5{amount=i32},ModifyStanding=6{delta=i16},CastAbility=7{ability=AbilityId,selector=TargetSelector},Root=8{duration_ticks=u32},Silence=9{duration_ticks=u32},Fear=10{duration_ticks=u32},Taunt=11{duration_ticks=u32}");
    h.update(b"PackedAbilityRegistry:SoA{hints=Vec<u32>,cooldown_ticks=Vec<u32>,range=Vec<f32>,gate_flags=Vec<u32>{bit0=hostile_only,bit1=los},delivery_kind=Vec<u32>,effect_kinds=Vec<u32>{stride=MAX_EFFECTS_PER_PROGRAM,empty=0xFF},effect_payload_a=Vec<u32>,effect_payload_b=Vec<u32>,tag_values=Vec<f32>{stride=NUM_ABILITY_TAGS=6}};HINT_NONE_SENTINEL=0xFFFFFFFF;EFFECT_KIND_EMPTY=0xFF");
    h.update(b"MicroTarget:None,Agent,Position,ItemSlot,AbilityIdx,Ability{id=AbilityId,target=AgentId},Query,Opaque");
    h.update(b"EventPacking:QuestPosted:resolution_tag+min_parties_byte,BidPlaced:amount_f32bits,AnnounceEmitted:audience_tag_u8+fact_payload_u64le,RecordMemory:confidence_f32bits,AgentCast:depth_u8");
    h.update(b"Lane:Validation=0,Effect=1,Reaction=2,Audit=3");
    h.update(b"MAX_CASCADE_ITERATIONS=8");
    h.update(b"OVERHEAR_RANGE=30");
    h.update(b"ENGAGEMENT_RANGE=2.0,ENGAGEMENT_SLOW_FACTOR=0.3,GLOBAL_COOLDOWN_TICKS=5,ATTACK_DAMAGE=10,ATTACK_RANGE=2,MOVE_SPEED_MPS=1");
    h.update(b"BuiltinMetrics:tick_ms,event_count,agent_alive,cascade_iterations,mask_true_frac");
    h.update(b"BuiltinInvariants:mask_validity,pool_non_overlap");
    h.update(b"FailureMode:Panic,Log");
    // Plan 3 surface — snapshot format, observation packer, probe harness.
    // Bumping any of these strings invalidates older snapshot files and
    // forces a migration registration.
    h.update(b"SnapshotFormat:v1:WSIMSV01:hot+cold_scalars+pod_collections+ring_meta;header=64B");
    h.update(b"FeatureSource:Vitals=4,Position=7,Neighbor<K>=6K");
    h.update(b"ProbeHarness:v1:DEFAULT_AGENT_CAP=256,DEFAULT_EVENT_CAP=4096");
    // T16 P2 coupling: engine snapshots reject if GPU rules changed without
    // a coordinated bump. Cross-crate hash dependency — `compile-dsl`
    // regenerates `engine_gpu_rules/.schema_hash`, and any drift there
    // invalidates the engine schema hash too.
    let gpu_hash_str = include_str!("../../engine_gpu_rules/.schema_hash").trim();
    h.update(b"engine_gpu_rules.schema_hash=");
    h.update(gpu_hash_str.as_bytes());
    h.finalize().into()
}
