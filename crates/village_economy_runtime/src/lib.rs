//! Per-fixture runtime for `assets/sim/village_economy.sim` —
//! noncombat acceptance fixture composing all four architectural
//! lift primitives (Lift A travel, Lift B recipes/wear, Lift C
//! consent/announce, Lift D skills/obligations) in a single sim.
//!
//! ## Architectural composition pin
//!
//! The 7 lift primitives split across 3 abilities:
//!
//!   ForgeIron.ability (4 effects):
//!     [0] cast_recipe 1           → EffectOp::Recipe   (40)  → kind=71
//!     [1] wear_tool 0 50          → EffectOp::WearTool (41)  → kind=72
//!     [2] gain_skill 0 25         → EffectOp::GainSkill (44) → kind=75
//!     [3] announce 0 radius 3     → EffectOp::Announce (43)  → kind=74
//!
//!   OfferDeal.ability (2 effects):
//!     [0] propose 4 expires_at 200    → EffectOp::Propose (42) → kind=73
//!     [1] create_obligation 1 0       → EffectOp::CreateObligation (45) → kind=76
//!
//!   WalkToWorkshop.ability (1 effect):
//!     [0] travel_to 5 5 for 5s    → EffectOp::TravelTo (39)  → kind=70
//!
//! Together: 7 lift primitives across 3 abilities, all chronicled into
//! one tick.
//!
//! ## Per-tick chain
//!
//!   1. Per-tick clears (event_tail, ring headers, mask bitmaps,
//!      scoring_output)
//!   2. fused_mask_verb_WalkToWorkshop — PerPair, writes mask_0
//!      (WalkToWorkshop) + mask_1 (ForgeIron) + mask_2 (OfferDeal).
//!   3. scoring — PerAgent argmax over 3 rows; emits ActionSelected.
//!   4. physics_verb_chronicle_ForgeIron — gates action_id==1u, walks
//!      ForgeIron's program; emits 4 chronicle records per cast (kinds
//!      71/72/74/75).
//!   5. physics_verb_chronicle_OfferDeal — gates action_id==2u, walks
//!      OfferDeal's program; emits 2 chronicle records per cast (kinds
//!      73/76).
//!   6. physics_verb_chronicle_WalkToWorkshop — gates action_id==0u,
//!      walks WalkToWorkshop's program; emits 1 chronicle record per
//!      cast (kind=70). Task #235 split this kernel out of the fused
//!      Fold kernel below; see the FoldTravel commentary in
//!      `assets/sim/village_economy.sim` for the analyzer fix that
//!      enabled the split.
//!   7. physics_FoldTravel_and_FoldRecipe_and_FoldWear_and_FoldPropose_and_
//!      FoldAnnounce_and_FoldSkill_and_FoldObligation — fused PerEvent
//!      kernel that drains the 7 chronicle kinds (70-76) emitted in
//!      steps 4-6 and folds them into the per-agent SoA counters.
//!   8. seed_indirect_0
//!
//! ## Fusion-correctness note
//!
//! Pre-Task #235 the compiler fused verb_chronicle_WalkToWorkshop with
//! every FoldX consumer rule into ONE PerEvent kernel because the
//! schedule analyzer's rule 4 (producer/consumer event-ring split) only
//! recognised explicit `Emit` statements as event-ring writes; an
//! `apply_ability <literal>` whose program emits a chronicle kind landed
//! a placeholder `EventKindId(0)` instead, missing the real kind=70
//! emission. With the analyzer now enumerating ApplyAbility's
//! dispatched kinds via the AbilityRegistry (see `dsl_compiler::cg::
//! schedule::fusion::push_apply_ability_event_kinds`), rule 4 fires and
//! WalkToWorkshop is dispatched as its own kernel BEFORE the fold pass —
//! the per-kernel dispatch barrier makes the kind=70 records visible to
//! the FoldTravel consumer. The same rule already protected ForgeIron /
//! OfferDeal (their producer kernels were ordered before the consumer
//! group naturally because the topo walk found their adjacent
//! producer/consumer pair).
//!
//! ## Faction layout (8 agents)
//!
//! Slot 0..1 → blacksmith (creature_type=1) — fires Walk + Forge
//! Slot 2..3 → noble      (creature_type=2) — fires OfferDeal
//! Slot 4..7 → villager   (creature_type=3) — bystanders
//!
//! ## SoA re-purpose table (mirrors spy_network's mana → suspicion)
//!
//!   * `mana`                       → recipes-completed counter (f32)
//!   * `shield_hp`                  → tool wear accumulation     (f32)
//!   * `hp`                         → skill level [0.0..1.0]      (f32)
//!   * `hunger`                     → consent / obligation count (f32)
//!   * `disguise_expires_at_tick`   → travel-completed counter   (u32)
//!
//! No engine SoA columns added. Engine SoA work for `busy_until_tick`
//! / `travel_dest_*` / `inventory_*` / `tool_wear_*` / `skill_*` is
//! deferred per the lift foundation slices' charter.

use engine::sim_trait::{AgentSnapshot, CompiledSim, VizGlyph};
use engine::GpuContext;
use engine::ability::registry_gpu::PackedAbilityRegistryGpu;
use engine::ability::PackedAbilityRegistry;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{AgentBuffers, EventRing, KernelBindingsContext};

mod binding_check;

pub use binding_check::{
    assert_ability_registry_matches_sim_constants,
    FORGE_IRON_EXPECTED_ABILITY_ID, OFFER_DEAL_EXPECTED_ABILITY_ID,
    WALK_TO_WORKSHOP_EXPECTED_ABILITY_ID,
};

/// Faction discriminants — must match `assets/sim/village_economy.sim`'s
/// `config.combat.type_*` fields.
pub const CREATURE_TYPE_BLACKSMITH: u32 = 1;
pub const CREATURE_TYPE_NOBLE: u32 = 2;
pub const CREATURE_TYPE_VILLAGER: u32 = 3;

/// Per-fixture state.
pub struct VillageEconomyState {
    gpu: GpuContext,

    // -- Agent SoA (REPURPOSED columns; see file-level docs) --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,
    agent_shield_hp_buf: wgpu::Buffer,
    agent_hunger_buf: wgpu::Buffer,
    agent_disguise_expires_at_tick_buf: wgpu::Buffer,
    agent_creature_type_buf: wgpu::Buffer,

    // -- Stat columns (BGL contract requires; init zero) --
    agent_max_hp_buf: wgpu::Buffer,
    agent_attack_damage_buf: wgpu::Buffer,
    agent_ability_power_buf: wgpu::Buffer,
    agent_armor_buf: wgpu::Buffer,
    agent_magic_resist_buf: wgpu::Buffer,
    agent_move_speed_buf: wgpu::Buffer,

    // -- Mask bitmaps. The compiler binds them positionally; the mask
    // kernel WGSL writes mask_0 = WalkToWorkshop (op#0), mask_1 =
    // ForgeIron (op#1), mask_2 = OfferDeal (op#2) — source order in
    // the .sim. ActionSelected.action_id matches.
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring --
    event_ring: EventRing,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_forge_cfg_buf: wgpu::Buffer,
    chronicle_offer_cfg_buf: wgpu::Buffer,
    chronicle_walk_cfg_buf: wgpu::Buffer,
    fold_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    /// Packed AbilityRegistry uploaded to the GPU. Built once at
    /// construction from the `.ability` corpus.
    registry_gpu: PackedAbilityRegistryGpu,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl VillageEconomyState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        assert_eq!(
            agent_count, 8,
            "village_economy expects exactly 8 agents (2 blacksmith + 2 noble + 4 villager); \
             got {agent_count}",
        );

        binding_check::assert_ability_registry_matches_sim_constants();

        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        let built_registry = binding_check::build_village_economy_registry();
        let packed = PackedAbilityRegistry::pack(&built_registry.registry);
        let registry_gpu = PackedAbilityRegistryGpu::upload(
            &packed,
            &gpu,
            "village_economy_runtime",
        );

        let n = agent_count as usize;
        let zero_f32 = vec![0.0_f32; n];
        let zero_u32 = vec![0u32; n];
        let alive_init = vec![1u32; n];
        let mut creature_init = vec![0u32; n];
        for (slot, ct) in creature_init.iter_mut().enumerate() {
            *ct = match slot {
                0..=1 => CREATURE_TYPE_BLACKSMITH,
                2..=3 => CREATURE_TYPE_NOBLE,
                _ => CREATURE_TYPE_VILLAGER,
            };
        }

        let mk_storage_f32 = |label: &'static str, contents: &[f32]| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(contents),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };
        let mk_storage_u32 = |label: &'static str, contents: &[u32]| -> wgpu::Buffer {
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(contents),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        };

        let agent_hp_buf = mk_storage_f32("village_economy::agent_hp", &zero_f32);
        let agent_alive_buf = mk_storage_u32("village_economy::agent_alive", &alive_init);
        let agent_mana_buf = mk_storage_f32("village_economy::agent_mana", &zero_f32);
        let agent_shield_hp_buf = mk_storage_f32("village_economy::agent_shield_hp", &zero_f32);
        let agent_hunger_buf = mk_storage_f32("village_economy::agent_hunger", &zero_f32);
        let agent_disguise_expires_at_tick_buf = mk_storage_u32(
            "village_economy::agent_disguise_expires_at_tick",
            &zero_u32,
        );
        let agent_creature_type_buf =
            mk_storage_u32("village_economy::agent_creature_type", &creature_init);

        let agent_max_hp_buf = mk_storage_f32("village_economy::agent_max_hp", &zero_f32);
        let agent_attack_damage_buf =
            mk_storage_f32("village_economy::agent_attack_damage", &zero_f32);
        let agent_ability_power_buf =
            mk_storage_f32("village_economy::agent_ability_power", &zero_f32);
        let agent_armor_buf = mk_storage_f32("village_economy::agent_armor", &zero_f32);
        let agent_magic_resist_buf =
            mk_storage_f32("village_economy::agent_magic_resist", &zero_f32);
        let agent_move_speed_buf =
            mk_storage_f32("village_economy::agent_move_speed", &zero_f32);

        // Three mask bitmaps (source order: WalkToWorkshop=0, ForgeIron=1, OfferDeal=2).
        let mask_bitmap_words = (agent_count + 31) / 32;
        let mask_bitmap_bytes = (mask_bitmap_words as u64) * 4;
        let mk_mask = |label: &str| -> wgpu::Buffer {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: mask_bitmap_bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mask_0_bitmap_buf = mk_mask("village_economy::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("village_economy::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("village_economy::mask_2_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_economy::mask_bitmap_zero"),
                contents: bytemuck::cast_slice(&zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("village_economy::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_economy::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let event_ring = EventRing::new(&gpu, "village_economy");

        let mask_cfg_init = fused_mask_verb_WalkToWorkshop::FusedMaskVerbWalkToWorkshopCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_economy::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let scoring_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_economy::scoring_cfg"),
                contents: bytemuck::bytes_of(&scoring_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let forge_cfg_init = physics_verb_chronicle_ForgeIron::PhysicsVerbChronicleForgeIronCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            agent_cap: 0,
        };
        let chronicle_forge_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_economy::chronicle_forge_cfg"),
                contents: bytemuck::bytes_of(&forge_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let offer_cfg_init = physics_verb_chronicle_OfferDeal::PhysicsVerbChronicleOfferDealCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            agent_cap: 0,
        };
        let chronicle_offer_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_economy::chronicle_offer_cfg"),
                contents: bytemuck::bytes_of(&offer_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let walk_cfg_init = physics_verb_chronicle_WalkToWorkshop::PhysicsVerbChronicleWalkToWorkshopCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            agent_cap: 0,
        };
        let chronicle_walk_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_economy::chronicle_walk_cfg"),
                contents: bytemuck::bytes_of(&walk_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let fold_cfg_init = physics_FoldTravel_and_FoldRecipe_and_FoldWear_and_FoldPropose_and_FoldAnnounce_and_FoldSkill_and_FoldObligation::PhysicsFoldTravelAndFoldRecipeAndFoldWearAndFoldProposeAndFoldAnnounceAndFoldSkillAndFoldObligationCfg {
            event_count: 0,
            tick: 0,
            seed: 0,
            agent_cap: 0,
        };
        let fold_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("village_economy::fold_cfg"),
                contents: bytemuck::bytes_of(&fold_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count,
            tick: 0,
            seed: 0,
            _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("village_economy::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            agent_shield_hp_buf,
            agent_hunger_buf,
            agent_disguise_expires_at_tick_buf,
            agent_creature_type_buf,
            agent_max_hp_buf,
            agent_attack_damage_buf,
            agent_ability_power_buf,
            agent_armor_buf,
            agent_magic_resist_buf,
            agent_move_speed_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_forge_cfg_buf,
            chronicle_offer_cfg_buf,
            chronicle_walk_cfg_buf,
            fold_cfg_buf,
            seed_cfg_buf,
            registry_gpu,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    /// Per-agent skill level [0.0..1.0]. REPURPOSED `agent_hp` SoA.
    pub fn read_skill_level(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "skill_level")
    }

    /// Per-agent recipes-completed counter. REPURPOSED `agent_mana` SoA.
    pub fn read_recipes_completed(&self) -> Vec<f32> {
        self.read_f32(&self.agent_mana_buf, "recipes_completed")
    }

    /// Per-agent accumulated tool wear. REPURPOSED `agent_shield_hp` SoA.
    pub fn read_tool_wear(&self) -> Vec<f32> {
        self.read_f32(&self.agent_shield_hp_buf, "tool_wear")
    }

    /// Per-agent consent/obligation counter. REPURPOSED `agent_hunger`
    /// SoA. Bumped by Propose, Announce, AND CreateObligation chronicles.
    pub fn read_consent_count(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hunger_buf, "consent_count")
    }

    /// Per-agent travel-completed counter. REPURPOSED
    /// `agent_disguise_expires_at_tick` SoA.
    pub fn read_travels_completed(&self) -> Vec<u32> {
        self.read_u32(&self.agent_disguise_expires_at_tick_buf, "travels_completed")
    }

    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    pub fn read_creature_type(&self) -> Vec<u32> {
        self.read_u32(&self.agent_creature_type_buf, "creature_type")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("village_economy::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("village_economy::read_f32"),
            },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    fn read_u32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<u32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("village_economy::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("village_economy::read_u32"),
            },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    pub fn agent_count(&self) -> u32 {
        self.agent_count
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl CompiledSim for VillageEconomyState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("village_economy::step"),
            },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = self.agent_count * 8;
        self.event_ring
            .clear_ring_headers_in(&self.gpu, &mut encoder, max_slots_per_tick);
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [
            &self.mask_0_bitmap_buf,
            &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf,
        ] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf,
                0,
                buf,
                0,
                mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf,
            0,
            &self.scoring_output_buf,
            0,
            scoring_output_bytes.max(16),
        );

        // Shared once per tick; each dispatch below adds only its
        // fixture-specific `*Extras` (mask bitmaps, scoring output,
        // creature_type / hunger / disguise_expires_at_tick repurposed
        // SoA columns, indirect args).
        let agent_buffers = AgentBuffers {
            hp_buf: Some(&self.agent_hp_buf),
            max_hp_buf: Some(&self.agent_max_hp_buf),
            alive_buf: Some(&self.agent_alive_buf),
            mana_buf: Some(&self.agent_mana_buf),
            shield_hp_buf: Some(&self.agent_shield_hp_buf),
            attack_damage_buf: Some(&self.agent_attack_damage_buf),
            ability_power_buf: Some(&self.agent_ability_power_buf),
            armor_buf: Some(&self.agent_armor_buf),
            magic_resist_buf: Some(&self.agent_magic_resist_buf),
            move_speed_buf: Some(&self.agent_move_speed_buf),
            ..Default::default()
        };
        let ctx = KernelBindingsContext {
            state: &agent_buffers,
            event_ring: &self.event_ring,
            registry: &self.registry_gpu,
        };

        // (2) Mask round — fused PerPair kernel writes all 3 mask bitmaps.
        let mask_cfg = fused_mask_verb_WalkToWorkshop::FusedMaskVerbWalkToWorkshopCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_cfg_buf,
            0,
            bytemuck::bytes_of(&mask_cfg),
        );
        let mask_extras = fused_mask_verb_WalkToWorkshop::FusedMaskVerbWalkToWorkshopExtras {
            agent_creature_type: &self.agent_creature_type_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        let mask_bindings =
            fused_mask_verb_WalkToWorkshop::FusedMaskVerbWalkToWorkshopBindings::from_context_with_extras(
                &ctx,
                &mask_extras,
            );
        dispatch::dispatch_fused_mask_verb_walktoworkshop(
            &mut self.cache,
            &mask_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count * self.agent_count,
        );

        // (3) Scoring — argmax over 3 rows.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf,
            0,
            bytemuck::bytes_of(&scoring_cfg),
        );
        let scoring_extras = scoring::ScoringExtras {
            agent_creature_type: &self.agent_creature_type_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        let scoring_bindings =
            scoring::ScoringBindings::from_context_with_extras(&ctx, &scoring_extras);
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (4) ForgeIron chronicle dispatch — gates action_id==1u, walks
        // ForgeIron's program (4 effects: Recipe + WearTool + GainSkill
        // + Announce); writes 4 chronicle records per cast.
        let forge_cfg = physics_verb_chronicle_ForgeIron::PhysicsVerbChronicleForgeIronCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_forge_cfg_buf,
            0,
            bytemuck::bytes_of(&forge_cfg),
        );
        let forge_extras =
            physics_verb_chronicle_ForgeIron::PhysicsVerbChronicleForgeIronExtras {
                cfg: &self.chronicle_forge_cfg_buf,
            };
        let forge_bindings =
            physics_verb_chronicle_ForgeIron::PhysicsVerbChronicleForgeIronBindings::from_context_with_extras(
                &ctx,
                &forge_extras,
            );
        dispatch::dispatch_physics_verb_chronicle_forgeiron(
            &mut self.cache,
            &forge_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (5) OfferDeal chronicle dispatch — gates action_id==2u, walks
        // OfferDeal's program (2 effects: Propose + CreateObligation);
        // writes 2 chronicle records per cast (kinds 73, 76).
        let offer_cfg = physics_verb_chronicle_OfferDeal::PhysicsVerbChronicleOfferDealCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_offer_cfg_buf,
            0,
            bytemuck::bytes_of(&offer_cfg),
        );
        let offer_extras =
            physics_verb_chronicle_OfferDeal::PhysicsVerbChronicleOfferDealExtras {
                cfg: &self.chronicle_offer_cfg_buf,
            };
        let offer_bindings =
            physics_verb_chronicle_OfferDeal::PhysicsVerbChronicleOfferDealBindings::from_context_with_extras(
                &ctx,
                &offer_extras,
            );
        dispatch::dispatch_physics_verb_chronicle_offerdeal(
            &mut self.cache,
            &offer_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (6) WalkToWorkshop chronicle dispatch — gates action_id==0u,
        // walks WalkToWorkshop's program (1 effect: TravelTo); writes
        // 1 chronicle record per cast (kind=70). Task #235 split this
        // out from the fused FoldTravel kernel so the runtime can
        // dispatch the producer kernel before the FoldX consumer
        // kernel and the per-dispatch barrier makes kind=70 records
        // visible to the FoldTravel reader.
        let walk_cfg = physics_verb_chronicle_WalkToWorkshop::PhysicsVerbChronicleWalkToWorkshopCfg {
            event_count: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_walk_cfg_buf,
            0,
            bytemuck::bytes_of(&walk_cfg),
        );
        let walk_extras =
            physics_verb_chronicle_WalkToWorkshop::PhysicsVerbChronicleWalkToWorkshopExtras {
                cfg: &self.chronicle_walk_cfg_buf,
            };
        let walk_bindings =
            physics_verb_chronicle_WalkToWorkshop::PhysicsVerbChronicleWalkToWorkshopBindings::from_context_with_extras(
                &ctx,
                &walk_extras,
            );
        dispatch::dispatch_physics_verb_chronicle_walktoworkshop(
            &mut self.cache,
            &walk_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        // (7) Fused PerEvent fold kernel — folds kinds 70-76 emitted in
        // steps 4-6. Now that the producer kernels (ForgeIron / OfferDeal /
        // WalkToWorkshop) are all sequenced BEFORE this fold pass, the
        // per-dispatch GPU barrier guarantees their chronicle records are
        // visible to every FoldX consumer. The pre-#235 double-dispatch
        // workaround is gone — see `assets/sim/village_economy.sim::FoldTravel`
        // for the analyzer fix that enabled the split.
        let event_count_estimate = self.agent_count * 8;
        let fold_cfg = physics_FoldTravel_and_FoldRecipe_and_FoldWear_and_FoldPropose_and_FoldAnnounce_and_FoldSkill_and_FoldObligation::PhysicsFoldTravelAndFoldRecipeAndFoldWearAndFoldProposeAndFoldAnnounceAndFoldSkillAndFoldObligationCfg {
            event_count: event_count_estimate,
            tick: self.tick as u32,
            seed: 0,
            agent_cap: 0,
        };
        self.gpu.queue.write_buffer(
            &self.fold_cfg_buf,
            0,
            bytemuck::bytes_of(&fold_cfg),
        );
        let fold_extras = physics_FoldTravel_and_FoldRecipe_and_FoldWear_and_FoldPropose_and_FoldAnnounce_and_FoldSkill_and_FoldObligation::PhysicsFoldTravelAndFoldRecipeAndFoldWearAndFoldProposeAndFoldAnnounceAndFoldSkillAndFoldObligationExtras {
            agent_hunger: &self.agent_hunger_buf,
            agent_disguise_expires_at_tick: &self.agent_disguise_expires_at_tick_buf,
            cfg: &self.fold_cfg_buf,
        };
        let fold_bindings = physics_FoldTravel_and_FoldRecipe_and_FoldWear_and_FoldPropose_and_FoldAnnounce_and_FoldSkill_and_FoldObligation::PhysicsFoldTravelAndFoldRecipeAndFoldWearAndFoldProposeAndFoldAnnounceAndFoldSkillAndFoldObligationBindings::from_context_with_extras(
            &ctx,
            &fold_extras,
        );
        dispatch::dispatch_physics_foldtravel_and_foldrecipe_and_foldwear_and_foldpropose_and_foldannounce_and_foldskill_and_foldobligation(
            &mut self.cache,
            &fold_bindings,
            &self.gpu.device,
            &mut encoder,
            event_count_estimate,
        );

        // (7) seed_indirect_0 — keeps args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count,
            tick: self.tick as u32,
            seed: 0,
            _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf,
            0,
            bytemuck::bytes_of(&seed_cfg),
        );
        let seed_extras = seed_indirect_0::SeedIndirect0Extras {
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        let seed_bindings =
            seed_indirect_0::SeedIndirect0Bindings::from_context_with_extras(
                &ctx,
                &seed_extras,
            );
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_bindings,
            &self.gpu.device,
            &mut encoder,
            self.agent_count,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 {
        self.agent_count
    }

    fn tick(&self) -> u64 {
        self.tick
    }

    fn positions(&mut self) -> &[Vec3] {
        &[]
    }

    fn snapshot(&mut self) -> AgentSnapshot {
        let alive_raw = self.read_alive();
        let creature_types_raw = self.read_creature_type();
        let positions: Vec<Vec3> = (0..self.agent_count as usize)
            .map(|i| Vec3::new(i as f32 * 5.0, 0.0, 0.0))
            .collect();
        AgentSnapshot {
            positions,
            creature_types: creature_types_raw,
            alive: alive_raw,
        }
    }

    fn glyph_table(&self) -> Vec<VizGlyph> {
        // 1=blacksmith, 2=noble, 3=villager.
        vec![
            VizGlyph::new('?', 240),
            VizGlyph::new('B', 33),
            VizGlyph::new('N', 196),
            VizGlyph::new('v', 250),
        ]
    }

    fn default_viewport(&self) -> Option<(Vec3, Vec3)> {
        Some((
            Vec3::new(-1.5, -1.5, 0.0),
            Vec3::new(((self.agent_count as f32) * 5.0) + 1.5, 1.5, 0.0),
        ))
    }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(VillageEconomyState::new(seed, agent_count))
}

// ---------------------------------------------------------------------------
// Behavioral pin tests.
// ---------------------------------------------------------------------------

/// Per-counter summary of a finished village_economy run.
pub struct VillageEconomyResult {
    pub travels_completed: u64,
    pub recipes_completed: u64,
    pub total_tool_wear: f64,
    pub max_skill_level_reached: f32,
    pub consent_records_emitted: u64,
    pub skill_records_emitted: u64,
}

/// Public entry point used by the behavioral pin test below + any
/// downstream regression harness.
pub fn run_village_economy(agent_count: u32, ticks: u64) -> VillageEconomyResult {
    let mut state = VillageEconomyState::new(0, agent_count);
    for _ in 0..ticks {
        state.step();
    }
    summarize(&state)
}

fn summarize(state: &VillageEconomyState) -> VillageEconomyResult {
    let travels = state.read_travels_completed();
    let recipes = state.read_recipes_completed();
    let wear = state.read_tool_wear();
    let skill = state.read_skill_level();
    let consent = state.read_consent_count();

    VillageEconomyResult {
        travels_completed: travels.iter().map(|&v| v as u64).sum(),
        recipes_completed: recipes.iter().map(|&v| v as u64).sum(),
        total_tool_wear: wear.iter().map(|&v| v as f64).sum(),
        max_skill_level_reached: skill.iter().cloned().fold(0.0_f32, f32::max),
        consent_records_emitted: consent.iter().map(|&v| v as u64).sum(),
        skill_records_emitted: skill
            .iter()
            .map(|&v| if v > 0.0 { 1 } else { 0 })
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Behavioral pin — exercises all four lift primitives in a single
    /// runtime tick chain. Asserts every per-lift counter is non-zero
    /// after 200 ticks.
    ///
    /// The pin is the architectural acceptance for Lifts A-D as a unified
    /// surface: if any of the seven chronicle event kinds (70..76) fails
    /// to round-trip from the dispatcher arm to the consumer SoA write,
    /// the matching counter stays at 0 and the assert fires.
    #[test]
    fn village_economy_acceptance() {
        let result = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_village_economy(8, 200)
        })) {
            Ok(r) => r,
            Err(_) => {
                eprintln!(
                    "[village_economy_acceptance] skipping: GPU init failed (no wgpu adapter on host?)"
                );
                return;
            }
        };

        eprintln!("[village_economy] post-run counters @ tick=200:");
        eprintln!("  travels_completed       = {}", result.travels_completed);
        eprintln!("  recipes_completed       = {}", result.recipes_completed);
        eprintln!("  total_tool_wear         = {:.4}", result.total_tool_wear);
        eprintln!("  max_skill_level_reached = {:.4}", result.max_skill_level_reached);
        eprintln!("  consent_records_emitted = {}", result.consent_records_emitted);
        eprintln!("  skill_records_emitted   = {}", result.skill_records_emitted);

        assert!(
            result.travels_completed >= 1,
            "expected >= 1 travel completion; got {}. \
             EffectTravelToApplied (kind=70) → FoldTravel consumer is broken.",
            result.travels_completed,
        );
        assert!(
            result.recipes_completed >= 1,
            "expected >= 1 recipe completion; got {}. \
             EffectRecipeApplied (kind=71) → FoldRecipe consumer is broken.",
            result.recipes_completed,
        );
        assert!(
            result.total_tool_wear > 0.0,
            "expected > 0 total tool wear; got {}. \
             EffectWearToolApplied (kind=72) → FoldWear consumer is broken.",
            result.total_tool_wear,
        );
        assert!(
            result.max_skill_level_reached > 0.0,
            "expected > 0 max skill level; got {}. \
             EffectGainSkillApplied (kind=75) → FoldSkill consumer is broken.",
            result.max_skill_level_reached,
        );
        assert!(
            result.consent_records_emitted >= 1,
            "expected >= 1 consent record (propose / announce / \
             create_obligation); got {}. Lift C+D consumers broken.",
            result.consent_records_emitted,
        );
        assert!(
            result.skill_records_emitted >= 1,
            "expected >= 1 agent's skill_level > 0; got {}. \
             EffectGainSkillApplied → FoldSkill failed for every agent.",
            result.skill_records_emitted,
        );
    }
}
