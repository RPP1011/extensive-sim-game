//! Compile-time schema hash — a baselined fingerprint over the public layout-
//! relevant types. CI compares this against `crates/engine/.schema_hash` to
//! catch unintended schema drift. When the schema intentionally changes, update
//! the baseline file.

use engine_data::entities::CreatureType;
use sha2::{Digest, Sha256};

pub fn schema_hash() -> [u8; 32] {
    let mut h = Sha256::new();
    // SimState SoA layout. Hot fields are read every tick; cold fields land
    // on spawn/chronicle/debug paths. See `docs/spec/state.md`.
    h.update(b"SimState:SoA{");
    h.update(b"hot_pos=vec3,hot_hp=f32,hot_max_hp=f32,hot_alive=bool,hot_movement_mode=u8,");
    h.update(b"hot_level=u32,hot_move_speed=f32,hot_move_speed_mult=f32,");
    h.update(b"hot_shield_hp=f32,hot_armor=f32,hot_magic_resist=f32,hot_attack_damage=f32,hot_attack_range=f32,hot_mana=f32,hot_max_mana=f32,hot_ability_power=f32,");
    h.update(b"hot_hunger=f32,hot_thirst=f32,hot_rest_timer=f32,");
    h.update(b"hot_safety=f32,hot_shelter=f32,hot_social=f32,hot_purpose=f32,hot_esteem=f32,");
    h.update(b"hot_risk_tolerance=f32,hot_social_drive=f32,hot_ambition=f32,hot_altruism=f32,hot_curiosity=f32,");
    h.update(b"hot_engaged_with=OptionAgentId,hot_stun_expires_at_tick=u32,hot_slow_expires_at_tick=u32,hot_slow_factor_q8=i16,hot_cooldown_next_ready_tick=u32,hot_root_expires_at_tick=u32,hot_silence_expires_at_tick=u32,hot_fear_expires_at_tick=u32,hot_taunt_expires_at_tick=u32,hot_lifesteal_frac_q8=i16,hot_lifesteal_expires_at_tick=u32,hot_damage_taken_mult_q8=i16,hot_damage_taken_mult_expires_at_tick=u32");
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
    h.update(b"Event:AgentMoved,AgentAttacked,AgentDied,AgentFled,AgentAte,AgentDrank,AgentRested,AgentCast{caster,ability,target,depth=u8,tick},AgentUsedItem,AgentHarvested,AgentPlacedTile,AgentPlacedVoxel,AgentHarvestedVoxel,AgentConversed,AgentSharedStory,AgentCommunicated,InformationRequested,AgentRemembered,QuestPosted,QuestAccepted,BidPlaced,AnnounceEmitted,RecordMemory,OpportunityAttackTriggered,EffectDamageApplied,EffectHealApplied,EffectShieldApplied,EffectStunApplied,EffectSlowApplied,EffectGoldTransfer,EffectStandingDelta,CastDepthExceeded,EngagementCommitted{actor,target,tick},EngagementBroken{actor,former_target,reason=u8,tick},FearSpread{observer,dead_kin,tick},PackAssist{observer,target,tick},RallyCall{observer,wounded_kin,tick},EffectSelfDamageApplied{actor,target,amount,tick},EffectLifeStealApplied{actor,target,expires_at_tick,fraction_q8=i16,tick},EffectDamageModifyApplied{actor,target,expires_at_tick,multiplier_q8=i16,tick},EffectExecuteApplied{actor,target,hp_threshold=f32,tick},EffectRootApplied{actor,target,expires_at_tick,tick},EffectSilenceApplied{actor,target,expires_at_tick,tick},EffectFearApplied{actor,target,expires_at_tick,tick},EffectTauntApplied{actor,target,expires_at_tick,tick},EffectDashApplied{actor,distance=f32,tick},EffectBlinkApplied{actor,distance=f32,tick},EffectKnockbackApplied{actor,target,distance=f32,tick},EffectPullApplied{actor,target,distance=f32,tick},EffectDamageOverTimeApplied{actor,target,amount_per_tick=f32,duration_ticks=u32,tick},EffectHealOverTimeApplied{actor,target,amount_per_tick=f32,duration_ticks=u32,tick},EffectTimedShieldApplied{actor,target,amount=f32,duration_ticks=u32,tick},EffectStealthApplied{actor,duration_ticks=u32,tick},EffectCharmApplied{actor,target,duration_ticks=u32,tick},EffectGroundedApplied{actor,target,duration_ticks=u32,tick},EffectSuppressApplied{actor,target,duration_ticks=u32,tick},EffectBuffApplied{actor,target,stat_ordinal=u8,magnitude_q8=i16,duration_ticks=u32,tick},EffectHarvestApplied{actor,kind_hash=u32,amount=u32,tick},EffectPlaceVoxelApplied{actor,kind_hash=u32,tick},EffectReflectApplied{actor,target,duration_ticks=u32,fraction_q8=i16,tick},EffectSummonApplied{actor,template_hash=u32,count=u8,lifetime_ticks=u32,tick},EffectPlantBeliefApplied{actor,target,subject_idx=u32,fact_bit_mask=u32,tick},EffectObserveApplied{actor,target,target_observer=u8,tick},EffectScryApplied{actor,subject,target_observer=u8,subject_idx=u32,tick},EffectRevealApplied{actor,subject,subject_idx=u32,tick},EffectDisguiseApplied{actor,fake_type=u8,duration_ticks=u32,tick},EffectDecoyApplied{actor,target,subject_idx=u32,fake_pos=u32,tick},EffectEraseBeliefApplied{actor,target,subject_idx=u32,fields=u8,tick},EffectTravelToApplied{actor,dest_x_q8=i16,dest_y_q8=i16,eta_ticks=u32,tick},EffectRecipeApplied{actor,recipe_id=u16,target_tool=u8,tick},EffectWearToolApplied{actor,tool_kind=u8,amount=u16,tick},EffectProposeApplied{actor,target,contract_kind=u8,expires_at_tick=u32,tick},EffectAnnounceApplied{actor,announcement_kind=u8,radius_q8=u16,tick},EffectGainSkillApplied{actor,skill_id=u8,amount_q8=u16,tick},EffectCreateObligationApplied{actor,target,obligation_id=u16,kind=u8,tick},EffectCastBeginApplied{actor,ability_id=u16,duration_ticks=u16,target_slot=u32,target_x_q8=i16,target_y_q8=i16,tick},CastBegan{actor,ability_id=u16,duration_ticks=u16,target_slot=u32,target_x_q8=i16,target_y_q8=i16,tick},CastResolved{actor,ability_id=u16,tick},CastInterrupted{actor,ability_id=u16,interrupt_kind=u8,tick},ChronicleEntry");
    h.update(b"MicroKind:Hold,MoveToward,Flee,Attack,Cast,UseItem,Harvest,Eat,Drink,Rest,PlaceTile,PlaceVoxel,HarvestVoxel,Converse,ShareStory,Communicate,Ask,Remember");
    h.update(b"CommunicationChannel:Speech,PackSignal,Pheromone,Song,Telepathy,Testimony");
    // Built from engine_data::entities::CreatureType::ALL, not hand-copied,
    // so this can't silently drift from the vocabulary engine_data actually
    // owns. Byte-identical to the old literal today (Human,Wolf,Deer,Dragon
    // in ALL's declared order) — no baseline bump.
    h.update(b"CreatureType:");
    let creature_names: Vec<&str> = CreatureType::ALL.iter().map(|ct| ct.name()).collect();
    h.update(creature_names.join(",").as_bytes());
    h.update(b"MovementMode:Walk,Fly,Swim,Climb");
    h.update(b"MacroKind:NoOp,PostQuest,AcceptQuest,Bid,Announce");
    h.update(b"AnnounceAudience:Group,Area,Anyone");
    h.update(b"Resolution:HighestBid,FirstAcceptable,MutualAgreement,Coalition,Majority");
    h.update(b"QuestCategory:Physical,Political,Personal,Economic,Narrative");
    h.update(b"QueryKind:AboutEntity,AboutKind,AboutAll");
    h.update(b"MemoryKind:Combat,Trade,Social,Political,Other");
    h.update(b"EventKindId:AgentMoved=0,AgentAttacked=1,AgentDied=2,AgentFled=3,AgentAte=4,AgentDrank=5,AgentRested=6,AgentCast=7,AgentUsedItem=8,AgentHarvested=9,AgentPlacedTile=10,AgentPlacedVoxel=11,AgentHarvestedVoxel=12,AgentConversed=13,AgentSharedStory=14,AgentCommunicated=15,InformationRequested=16,AgentRemembered=17,QuestPosted=18,QuestAccepted=19,BidPlaced=20,AnnounceEmitted=21,RecordMemory=22,OpportunityAttackTriggered=25,EffectDamageApplied=26,EffectHealApplied=27,EffectShieldApplied=28,EffectStunApplied=29,EffectSlowApplied=30,EffectGoldTransfer=31,EffectStandingDelta=32,CastDepthExceeded=33,EngagementCommitted=34,EngagementBroken=35,FearSpread=36,PackAssist=37,RallyCall=38,EffectSelfDamageApplied=39,EffectLifeStealApplied=40,EffectDamageModifyApplied=41,EffectExecuteApplied=42,EffectRootApplied=43,EffectSilenceApplied=44,EffectFearApplied=45,EffectTauntApplied=46,EffectDashApplied=47,EffectBlinkApplied=48,EffectKnockbackApplied=49,EffectPullApplied=50,EffectDamageOverTimeApplied=51,EffectHealOverTimeApplied=52,EffectTimedShieldApplied=53,EffectStealthApplied=54,EffectCharmApplied=55,EffectGroundedApplied=56,EffectSuppressApplied=57,EffectBuffApplied=58,EffectHarvestApplied=59,EffectPlaceVoxelApplied=60,EffectReflectApplied=61,EffectSummonApplied=62,EffectPlantBeliefApplied=63,EffectObserveApplied=64,EffectScryApplied=65,EffectRevealApplied=66,EffectDisguiseApplied=67,EffectDecoyApplied=68,EffectEraseBeliefApplied=69,EffectTravelToApplied=70,EffectRecipeApplied=71,EffectWearToolApplied=72,EffectProposeApplied=73,EffectAnnounceApplied=74,EffectGainSkillApplied=75,EffectCreateObligationApplied=76,EffectCastBeginApplied=77,CastBegan=78,CastResolved=79,CastInterrupted=80,ChronicleEntry=128");
    h.update(b"AbilityId=NonZeroU32;MAX_EFFECTS_PER_PROGRAM=6");
    h.update(b"Delivery:Instant|Method{kind=DeliveryMethodKind,raw=String,hooks=SmallVec<DeliveryHook;4>};DeliveryMethodKind:Projectile=0,Channel=1,Zone=2,Chain=3,Tether=4,Trap=5;DeliveryHook{kind=DeliveryHookKind,effects=SmallVec<EffectOp;MAX_EFFECTS_PER_PROGRAM>};DeliveryHookKind:OnHit=0,OnTick=1,OnArrival=2,OnComplete=3,OnTrigger=4,OnKill=5,OnDamage=6,OnDamageDealt=7,OnDamageTaken=8,OnDeath=9,OnAutoAttack=10,OnAbilityUsed=11;Area:SingleTarget{range=f32};Gate{cooldown_ticks=u32,hostile_only=bool,line_of_sight=bool};TargetSelector:Target=0,Caster=1");
    h.update(b"EffectOp:Damage=0{amount=f32},Heal=1{amount=f32},Shield=2{amount=f32},Stun=3{duration_ticks=u32},Slow=4{duration_ticks=u32,factor_q8=i16},TransferGold=5{amount=i32},ModifyStanding=6{delta=i16},CastAbility=7{ability=AbilityId,selector=TargetSelector},Root=8{duration_ticks=u32},Silence=9{duration_ticks=u32},Fear=10{duration_ticks=u32},Taunt=11{duration_ticks=u32},Dash=12{distance=f32},Blink=13{distance=f32},Knockback=14{distance=f32},Pull=15{distance=f32},Execute=16{hp_threshold=f32},SelfDamage=17{amount=f32},LifeSteal=18{duration_ticks=u32,fraction_q8=i16},DamageModify=19{duration_ticks=u32,multiplier_q8=i16},DamageOverTime=20{amount=f32,duration_ticks=u32},HealOverTime=21{amount=f32,duration_ticks=u32},TimedShield=22{amount=f32,duration_ticks=u32},Buff=23{stat=BuffStat{MoveSpeed=0,AttackSpeed=1},magnitude_q8=i16,duration_ticks=u32},Summon=24{template_hash=u32,count=u8,lifetime_ticks=u32},Harvest=25{kind_hash=u32,amount=u16},PlaceVoxel=26{kind_hash=u32},Stealth=27{duration_ticks=u32},Charm=28{duration_ticks=u32},Grounded=29{duration_ticks=u32},Suppress=30{duration_ticks=u32},Reflect=31{duration_ticks=u32,fraction_q8=i16},PlantBelief=32{subject_idx=u32,fact_bit=u8},Observe=33{target_observer=u8},Scry=34{target_observer=u8,subject_idx=u32},Reveal=35{subject_idx=u32},Disguise=36{fake_type=u8,duration_ticks=u32},Decoy=37{subject_idx=u32,fake_pos=u32},EraseBelief=38{subject_idx=u32,fields=u8},TravelTo=39{dest_x_q8=i16,dest_y_q8=i16,eta_ticks=u32},Recipe=40{recipe_id=u16,target_tool=u8},WearTool=41{tool_kind=u8,amount=u16},Propose=42{contract_kind=u8,expires_at_tick=u32},Announce=43{announcement_kind=u8,radius_q8=u16},GainSkill=44{skill_id=u8,amount_q8=u16},CreateObligation=45{obligation_id=u16,kind=u8},CastBegin=46{ability_id=u16,duration_ticks=u16,target_slot=u32,target_x_q8=i16,target_y_q8=i16}");
    h.update(b"StackingMode:Refresh=0,Stack=1,Extend=2");
    h.update(b"LifetimeMode:UntilCasterDies=0,DamageableHp=1,BreakOnDamage=2;LIFETIME_KIND_NONE_SENTINEL=0xFF");
    h.update(b"ShapeKind:Circle=0,Cone=1,Line=2,Ring=3,Spread=4,Box=5,Sphere=6,Column=7,Wall=8,Cylinder=9,Dome=10,Hull=11;SHAPE_KIND_NONE_SENTINEL=0xFF");
    h.update(b"ScalingStatRef:AttackDamage=0,AbilityPower=1,MaxHp=2,Hp=3,Armor=4,MagicResist=5,MoveSpeed=6,Mana=7;MAX_SCALINGS_PER_EFFECT=2;SCALING_STAT_NONE_SENTINEL=0xFF");
    h.update(b"AbilityHint:Damage=0,Defense=1,CrowdControl=2,Utility=3,Heal=4,Buff=5");
    h.update(b"EffectPredicate{binder=EffectPredicateBinder{SelfBinder=0,Target=1},field=u8,op=EffectPredicateOp{Lt=0,Le=1,Gt=2,Ge=3,Eq=4,Ne=5},literal=f32}");
    h.update(b"WhenPredicate:Atom(EffectPredicate)|And(Box<WhenPredicate>,Box<WhenPredicate>)|Or(Box<WhenPredicate>,Box<WhenPredicate>)|Not(Box<WhenPredicate>);MAX_PRED_NODES_PER_EFFECT=12");
    h.update(b"AbilityProgram{delivery,area,gate,effects=SmallVec<EffectOp;MAX_EFFECTS_PER_PROGRAM>,hint=Option<AbilityHint>,tags=SmallVec<(AbilityTag,f32);MAX_TAGS_PER_PROGRAM>,stackings=SmallVec<Option<StackingMode>;MAX_EFFECTS_PER_PROGRAM>,chances=SmallVec<Option<u16>;MAX_EFFECTS_PER_PROGRAM>,lifetimes=SmallVec<Option<LifetimeMode>;MAX_EFFECTS_PER_PROGRAM>,per_effect_areas=SmallVec<Option<EffectAreaShape>;MAX_EFFECTS_PER_PROGRAM>,scalings_per_effect=SmallVec<SmallVec<EffectScaling;MAX_SCALINGS_PER_EFFECT>;MAX_EFFECTS_PER_PROGRAM>,when_per_effect=SmallVec<Option<EffectWhenCondition{when_cond:String,else_cond:Option<String>,when_compiled:Option<EffectPredicate>,when_compound:Option<WhenPredicate>}>;MAX_EFFECTS_PER_PROGRAM>,nested_per_effect=SmallVec<SmallVec<EffectOp;MAX_NESTED_PER_EFFECT=2>;MAX_EFFECTS_PER_PROGRAM>,cost=Option<AbilityCost{resource:CostResource{Mana=0,Stamina=1,Hp=2,Gold=3},amount:CostAmount{Flat(f32)|PercentOfMax(f32)}}>,charges=Option<u32>,recharge_ticks=Option<u32>,is_toggle=bool,recast=Option<RecastKind{Count(u32)|CooldownTicks(u32)}>,recast_window_ticks=Option<u32>,target_mode=TargetModeKind{Enemy=0,SelfCast=1,Ally=2,SelfAoe=3,Ground=4,Direction=5,Vector=6,Global=7},telegraph_kind=u8,telegraph_params=[f32;4]}");
    h.update(b"TelegraphKind:Circle=1,Line=2;TELEGRAPH_KIND_NONE=0");
    h.update(b"PackedAbilityRegistry:SoA{hints=Vec<u32>,cooldown_ticks=Vec<u32>,range=Vec<f32>,gate_flags=Vec<u32>{bit0=hostile_only,bit1=los},delivery_kind=Vec<u32>,effect_kinds=Vec<u32>{stride=MAX_EFFECTS_PER_PROGRAM,empty=0xFF},effect_payload_a=Vec<u32>,effect_payload_b=Vec<u32>,tag_values=Vec<f32>{stride=NUM_ABILITY_TAGS=6},stackings=Vec<u8>{stride=MAX_EFFECTS_PER_PROGRAM,none=0xFF},chances=Vec<u16>{stride=MAX_EFFECTS_PER_PROGRAM,sentinel=0xFFFF},lifetime_kinds=Vec<u8>{stride=MAX_EFFECTS_PER_PROGRAM,sentinel=0xFF},lifetime_payloads=Vec<f32>{stride=MAX_EFFECTS_PER_PROGRAM},area_kinds=Vec<u8>{stride=MAX_EFFECTS_PER_PROGRAM,sentinel=0xFF},area_args=Vec<f32>{stride=MAX_EFFECTS_PER_PROGRAM*4},scaling_stat_refs=Vec<u8>{stride=MAX_EFFECTS_PER_PROGRAM*MAX_SCALINGS_PER_EFFECT,sentinel=0xFF},scaling_percents=Vec<f32>{stride=MAX_EFFECTS_PER_PROGRAM*MAX_SCALINGS_PER_EFFECT},nested_effect_kinds=Vec<u32>{stride=MAX_EFFECTS_PER_PROGRAM*MAX_NESTED_PER_EFFECT,empty=0xFF},nested_effect_payload_a=Vec<u32>,nested_effect_payload_b=Vec<u32>,when_pred_binder=Vec<u8>{stride=MAX_EFFECTS_PER_PROGRAM*MAX_PRED_NODES_PER_EFFECT,sentinel=0xFF,op_and=0xFE,op_or=0xFD,op_not=0xFC},when_pred_field=Vec<u8>{stride=MAX_EFFECTS_PER_PROGRAM*MAX_PRED_NODES_PER_EFFECT},when_pred_op=Vec<u8>{stride=MAX_EFFECTS_PER_PROGRAM*MAX_PRED_NODES_PER_EFFECT},when_pred_literal=Vec<f32>{stride=MAX_EFFECTS_PER_PROGRAM*MAX_PRED_NODES_PER_EFFECT},telegraph_kind=Vec<u8>{stride=1,sentinel=TELEGRAPH_KIND_NONE=0},telegraph_params=Vec<[f32;4]>{stride=4},pending_effect_kinds=Vec<u32>{stride=MAX_EFFECTS_PER_PROGRAM,empty=0xFF},pending_effect_payload_a=Vec<u32>,pending_effect_payload_b=Vec<u32>,interrupt_mask=Vec<u8>{stride=1,sentinel=INTERRUPT_MASK_NONE_SENTINEL=0xFF}};HINT_NONE_SENTINEL=0xFFFFFFFF;EFFECT_KIND_EMPTY=0xFF;STACKING_NONE_SENTINEL=0xFF;CHANCE_NONE_SENTINEL=0xFFFF;LIFETIME_KIND_NONE_SENTINEL=0xFF;SHAPE_KIND_NONE_SENTINEL=0xFF;SCALING_STAT_NONE_SENTINEL=0xFF;WHEN_PRED_NONE_SENTINEL=0xFF;WHEN_PRED_OP_AND_SENTINEL=0xFE;WHEN_PRED_OP_OR_SENTINEL=0xFD;WHEN_PRED_OP_NOT_SENTINEL=0xFC;TELEGRAPH_KIND_NONE=0;INTERRUPT_MASK_NONE_SENTINEL=0xFF");
    h.update(b"MicroTarget:None,Agent,Position,ItemSlot,AbilityIdx,Ability{id=AbilityId,target=AgentId},Query,Opaque");
    h.update(b"EventPacking:QuestPosted:resolution_tag+min_parties_byte,BidPlaced:amount_f32bits,AnnounceEmitted:audience_tag_u8+fact_payload_u64le,RecordMemory:confidence_f32bits,AgentCast:depth_u8");
    h.update(b"Lane:Validation=0,Effect=1,Reaction=2,Audit=3");
    h.update(b"MAX_CASCADE_ITERATIONS=8");
    h.update(b"OVERHEAR_RANGE=30");
    h.update(b"ENGAGEMENT_RANGE=2.0,ENGAGEMENT_SLOW_FACTOR=0.3,GLOBAL_COOLDOWN_TICKS=5,ATTACK_DAMAGE=10,ATTACK_RANGE=2,MOVE_SPEED_MPS=1");
    h.update(b"BuiltinMetrics:tick_ms,event_count,agent_alive,cascade_iterations,mask_true_frac");
    h.update(b"BuiltinInvariants:mask_validity,pool_non_overlap");
    h.update(b"FailureMode:Panic,Log");
    // P5: RngPurpose discriminant table (closed-set FIXED-FOREVER ids
    // — see `dsl_compiler::cg::data_handle::RngPurpose::wgsl_id`).
    // Adding/reordering variants is a determinism-breaking change
    // requiring a schema-hash bump. WGSL prelude
    // (`RNG_WGSL_PRELUDE` in `dsl_compiler::cg::emit::program`)
    // mirrors this exact mapping. `Chance=10` is for the apply_ability
    // dispatcher's per-effect chance gate (CPU/GPU PCG parity).
    h.update(b"RngPurpose:Action=1,Sample=2,Shuffle=3,Conception=4,Uniform=5,Gauss=6,Coin=7,UniformInt=8,GaussB=9,Chance=10");
    // GPU chronicle record layout (compiler-emitted in
    // `cg::emit::wgsl_body::emit_chronicle_arm_chain`). Per-record
    // stride = 11 u32 words; layout:
    //   [0]=kind_tag, [1]=tick, [2]=actor, [3]=target,
    //   [4..6]=per-variant payload, [6]=ability_id, [7..9]=reserved,
    //   [10]=seq (sort-then-fold sequence trailer, written by producers).
    // Slot 6 holds ability_id so chronicle consumers can discriminate ability
    // source. Slot 10 (seq) is the sort-then-fold pass sequence trailer.
    // Bumping any of these values requires updating every per-arm `atomicStore`
    // in `emit_chronicle_arm_chain` AND the CPU reference in
    // `cpu_chronicle_reference::apply_event_to_chronicle_record`.
    h.update(b"ChronicleRecord:stride=11u32{kind=0,tick=1,actor=2,target=3,payload_a=4,payload_b=5,ability_id=6,reserved=7..=9,seq=10}");
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
