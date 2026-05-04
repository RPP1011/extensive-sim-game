//! Per-fixture runtime for `assets/sim/quest_arc_real.sim` — the
//! EIGHTH real gameplay-shaped fixture and FIRST sim with multi-
//! stage per-agent state machines.
//!
//! 30 Adventurers each independently cycle through a 5-stage quest
//! arc (Accept → Hunt → Collect → Return → Complete → reset). The
//! per-agent stage is encoded in the `mana` SoA field (f32, values
//! 0.0..=4.0). Cumulative quests-completed per agent comes from the
//! `quests_completed` view (CAS-add f32 fold path, NOT the u32
//! atomicOr path that quest_probe surfaced as broken).
//!
//! The runtime uses **Conservative** schedule synthesis (one kernel
//! per op) rather than Default. Default fuses the AcceptQuest verb
//! chronicle (op#9 — emits StageAdvanced) with the ApplyStageAdvance
//! / ApplyQuestCompleted physics handlers (op#7/8 — read those same
//! events). With per-event-idx single-pass semantics, the Apply ops
//! ran BEFORE the chronicle emits in the same pass, so mana never
//! advanced. Conservative breaks the kernel apart so the chronicle
//! emits land in the ring before the apply kernels read it — matching
//! duel_1v1's pattern (chronicle Strike/Spell/Heal kept SEPARATE from
//! the ApplyDamage/ApplyHeal kernel in duel_1v1's emit set).
//!
//! ## Per-tick chain
//!
//!   1. clear_tail + clear 5 mask bitmaps + zero scoring_output
//!   2. mask_verb_AcceptQuest..mask_verb_CompleteQuest — 5 PerPair
//!      kernels, one per verb. Each writes its own mask_<N>_bitmap.
//!   3. scoring — PerAgent argmax over 5 mutually-exclusive rows.
//!   4. physics_verb_chronicle_AcceptQuest..physics_verb_chronicle_CompleteQuest
//!      — 5 PerEvent kernels, each gating on action_id == N and
//!      emitting StageAdvanced (or QuestCompleted for op#4).
//!   5. physics_ApplyStageAdvance — PerEvent, reads StageAdvanced
//!      events, writes mana = mana + 1.0.
//!   6. physics_ApplyQuestCompleted — PerEvent, reads QuestCompleted
//!      events, writes mana = 0.0.
//!   7. seed_indirect_0 — keeps indirect-args buffer warm.
//!   8. fold_quests_completed + fold_stage_advances — per-agent f32
//!      accumulators (CAS+add path).

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

/// Per-fixture state for the quest arc.
pub struct QuestArcRealState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,
    agent_mana_staging: wgpu::Buffer,

    // -- Mask bitmaps (one per verb in source order) --
    mask_0_bitmap_buf: wgpu::Buffer, // AcceptQuest
    mask_1_bitmap_buf: wgpu::Buffer, // HuntMonster
    mask_2_bitmap_buf: wgpu::Buffer, // CollectItem
    mask_3_bitmap_buf: wgpu::Buffer, // ReturnHome
    mask_4_bitmap_buf: wgpu::Buffer, // CompleteQuest
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    quests_completed: ViewStorage,
    quests_completed_cfg_buf: wgpu::Buffer,
    stage_advances: ViewStorage,
    stage_advances_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms (one per kernel) --
    mask_accept_cfg_buf: wgpu::Buffer,
    mask_hunt_cfg_buf: wgpu::Buffer,
    mask_collect_cfg_buf: wgpu::Buffer,
    mask_return_cfg_buf: wgpu::Buffer,
    mask_complete_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_accept_cfg_buf: wgpu::Buffer,
    chronicle_hunt_cfg_buf: wgpu::Buffer,
    chronicle_collect_cfg_buf: wgpu::Buffer,
    chronicle_return_cfg_buf: wgpu::Buffer,
    chronicle_complete_cfg_buf: wgpu::Buffer,
    apply_stage_cfg_buf: wgpu::Buffer,
    apply_complete_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,
}

impl QuestArcRealState {
    pub fn new(seed: u64, agent_count: u32) -> Self {
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Agent SoA — alive=1, mana=0 (stage 0 = Accept) for every slot.
        let alive_init: Vec<u32> = vec![1u32; agent_count as usize];
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quest_arc_real_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let mana_init: Vec<f32> = vec![0.0_f32; agent_count as usize];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quest_arc_real_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let mana_staging_bytes = ((agent_count as u64) * 4).max(16);
        let agent_mana_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quest_arc_real_runtime::agent_mana_staging"),
            size: mana_staging_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Five mask bitmaps — one per verb. Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("quest_arc_real_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("quest_arc_real_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("quest_arc_real_runtime::mask_2_bitmap");
        let mask_3_bitmap_buf = mk_mask("quest_arc_real_runtime::mask_3_bitmap");
        let mask_4_bitmap_buf = mk_mask("quest_arc_real_runtime::mask_4_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quest_arc_real_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output — 4 × u32 per agent.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quest_arc_real_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quest_arc_real_runtime::scoring_output_zero"),
                contents: bytemuck::cast_slice(&scoring_zero_words),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        // Event ring + view storage.
        let event_ring = EventRing::new(&gpu, "quest_arc_real_runtime");
        let quests_completed = ViewStorage::new(
            &gpu,
            "quest_arc_real_runtime::quests_completed",
            agent_count,
            false,
            false,
        );
        let stage_advances = ViewStorage::new(
            &gpu,
            "quest_arc_real_runtime::stage_advances",
            agent_count,
            false,
            false,
        );

        // Per-kernel cfg uniforms. Initialise each with placeholder
        // values; the per-tick step rewrites them with current
        // tick / event_count.
        let mk_mask_cfg = |label: &str| -> wgpu::Buffer {
            let init = mask_verb_AcceptQuest::MaskVerbAcceptQuestCfg {
                agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
            };
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mask_accept_cfg_buf = mk_mask_cfg("quest_arc_real_runtime::mask_accept_cfg");
        let mask_hunt_cfg_buf = mk_mask_cfg("quest_arc_real_runtime::mask_hunt_cfg");
        let mask_collect_cfg_buf = mk_mask_cfg("quest_arc_real_runtime::mask_collect_cfg");
        let mask_return_cfg_buf = mk_mask_cfg("quest_arc_real_runtime::mask_return_cfg");
        let mask_complete_cfg_buf = mk_mask_cfg("quest_arc_real_runtime::mask_complete_cfg");

        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quest_arc_real_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mk_per_event_cfg = |label: &str| -> wgpu::Buffer {
            let init = physics_verb_chronicle_AcceptQuest::PhysicsVerbChronicleAcceptQuestCfg {
                event_count: 0, tick: 0, seed: 0, _pad0: 0,
            };
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
        };
        let chronicle_accept_cfg_buf =
            mk_per_event_cfg("quest_arc_real_runtime::chronicle_accept_cfg");
        let chronicle_hunt_cfg_buf =
            mk_per_event_cfg("quest_arc_real_runtime::chronicle_hunt_cfg");
        let chronicle_collect_cfg_buf =
            mk_per_event_cfg("quest_arc_real_runtime::chronicle_collect_cfg");
        let chronicle_return_cfg_buf =
            mk_per_event_cfg("quest_arc_real_runtime::chronicle_return_cfg");
        let chronicle_complete_cfg_buf =
            mk_per_event_cfg("quest_arc_real_runtime::chronicle_complete_cfg");
        let apply_stage_cfg_buf = mk_per_event_cfg("quest_arc_real_runtime::apply_stage_cfg");
        let apply_complete_cfg_buf =
            mk_per_event_cfg("quest_arc_real_runtime::apply_complete_cfg");

        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quest_arc_real_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let quests_cfg_init = fold_quests_completed::FoldQuestsCompletedCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let quests_completed_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quest_arc_real_runtime::quests_completed_cfg"),
                contents: bytemuck::bytes_of(&quests_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let stage_cfg_init = fold_stage_advances::FoldStageAdvancesCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let stage_advances_cfg_buf =
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("quest_arc_real_runtime::stage_advances_cfg"),
                contents: bytemuck::bytes_of(&stage_cfg_init),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            agent_alive_buf,
            agent_mana_buf,
            agent_mana_staging,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_3_bitmap_buf,
            mask_4_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            quests_completed,
            quests_completed_cfg_buf,
            stage_advances,
            stage_advances_cfg_buf,
            mask_accept_cfg_buf,
            mask_hunt_cfg_buf,
            mask_collect_cfg_buf,
            mask_return_cfg_buf,
            mask_complete_cfg_buf,
            scoring_cfg_buf,
            chronicle_accept_cfg_buf,
            chronicle_hunt_cfg_buf,
            chronicle_collect_cfg_buf,
            chronicle_return_cfg_buf,
            chronicle_complete_cfg_buf,
            apply_stage_cfg_buf,
            apply_complete_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
        }
    }

    pub fn quests_completed(&mut self) -> &[f32] {
        self.quests_completed.readback(&self.gpu)
    }

    pub fn stage_advances(&mut self) -> &[f32] {
        self.stage_advances.readback(&self.gpu)
    }

    pub fn read_mana(&self) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("quest_arc_real_runtime::read_mana") },
        );
        encoder.copy_buffer_to_buffer(&self.agent_mana_buf, 0, &self.agent_mana_staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = self.agent_mana_staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.agent_mana_staging.unmap();
        v
    }

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }
}

impl CompiledSim for QuestArcRealState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("quest_arc_real_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [
            &self.mask_0_bitmap_buf,
            &self.mask_1_bitmap_buf,
            &self.mask_2_bitmap_buf,
            &self.mask_3_bitmap_buf,
            &self.mask_4_bitmap_buf,
        ] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf, 0, buf, 0, mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round — 5 PerPair kernels, one per verb. Each
        // writes its own mask_<N>_bitmap. mask_k=1u (TODO task-5.7
        // in the compiler), so the per-pair grid degenerates to one
        // thread per agent. Each row gates on `self.mana == STAGE_N`
        // plus a per-row cooldown.
        let tick = self.tick as u32;
        let event_count_estimate = self.agent_count * 4;

        let mask_cfg_accept = mask_verb_AcceptQuest::MaskVerbAcceptQuestCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_accept_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_accept),
        );
        dispatch::dispatch_mask_verb_acceptquest(
            &mut self.cache,
            &mask_verb_AcceptQuest::MaskVerbAcceptQuestBindings {
                agent_alive: &self.agent_alive_buf,
                agent_mana: &self.agent_mana_buf,
                mask_0_bitmap: &self.mask_0_bitmap_buf,
                cfg: &self.mask_accept_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_hunt = mask_verb_HuntMonster::MaskVerbHuntMonsterCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_hunt_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_hunt),
        );
        dispatch::dispatch_mask_verb_huntmonster(
            &mut self.cache,
            &mask_verb_HuntMonster::MaskVerbHuntMonsterBindings {
                agent_alive: &self.agent_alive_buf,
                agent_mana: &self.agent_mana_buf,
                mask_1_bitmap: &self.mask_1_bitmap_buf,
                cfg: &self.mask_hunt_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_collect = mask_verb_CollectItem::MaskVerbCollectItemCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_collect_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_collect),
        );
        dispatch::dispatch_mask_verb_collectitem(
            &mut self.cache,
            &mask_verb_CollectItem::MaskVerbCollectItemBindings {
                agent_alive: &self.agent_alive_buf,
                agent_mana: &self.agent_mana_buf,
                mask_2_bitmap: &self.mask_2_bitmap_buf,
                cfg: &self.mask_collect_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_return = mask_verb_ReturnHome::MaskVerbReturnHomeCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_return_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_return),
        );
        dispatch::dispatch_mask_verb_returnhome(
            &mut self.cache,
            &mask_verb_ReturnHome::MaskVerbReturnHomeBindings {
                agent_alive: &self.agent_alive_buf,
                agent_mana: &self.agent_mana_buf,
                mask_3_bitmap: &self.mask_3_bitmap_buf,
                cfg: &self.mask_return_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        let mask_cfg_complete = mask_verb_CompleteQuest::MaskVerbCompleteQuestCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.mask_complete_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg_complete),
        );
        dispatch::dispatch_mask_verb_completequest(
            &mut self.cache,
            &mask_verb_CompleteQuest::MaskVerbCompleteQuestBindings {
                agent_alive: &self.agent_alive_buf,
                agent_mana: &self.agent_mana_buf,
                mask_4_bitmap: &self.mask_4_bitmap_buf,
                cfg: &self.mask_complete_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (3) Scoring — argmax over the 5 rows. Mutually-exclusive
        // mana-stage gates ensure at most one row's mask is set per
        // agent per tick.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg),
        );
        dispatch::dispatch_scoring(
            &mut self.cache,
            &scoring::ScoringBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                mask_0_bitmap: &self.mask_0_bitmap_buf,
                mask_1_bitmap: &self.mask_1_bitmap_buf,
                mask_2_bitmap: &self.mask_2_bitmap_buf,
                mask_3_bitmap: &self.mask_3_bitmap_buf,
                mask_4_bitmap: &self.mask_4_bitmap_buf,
                scoring_output: &self.scoring_output_buf,
                cfg: &self.scoring_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (4) Five chronicle kernels — each gates on action_id == N
        // and emits StageAdvanced (or QuestCompleted for op#4).
        let chronicle_accept_cfg = physics_verb_chronicle_AcceptQuest::PhysicsVerbChronicleAcceptQuestCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_accept_cfg_buf, 0, bytemuck::bytes_of(&chronicle_accept_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_acceptquest(
            &mut self.cache,
            &physics_verb_chronicle_AcceptQuest::PhysicsVerbChronicleAcceptQuestBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_accept_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chronicle_hunt_cfg = physics_verb_chronicle_HuntMonster::PhysicsVerbChronicleHuntMonsterCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_hunt_cfg_buf, 0, bytemuck::bytes_of(&chronicle_hunt_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_huntmonster(
            &mut self.cache,
            &physics_verb_chronicle_HuntMonster::PhysicsVerbChronicleHuntMonsterBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_hunt_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chronicle_collect_cfg = physics_verb_chronicle_CollectItem::PhysicsVerbChronicleCollectItemCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_collect_cfg_buf, 0, bytemuck::bytes_of(&chronicle_collect_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_collectitem(
            &mut self.cache,
            &physics_verb_chronicle_CollectItem::PhysicsVerbChronicleCollectItemBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_collect_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chronicle_return_cfg = physics_verb_chronicle_ReturnHome::PhysicsVerbChronicleReturnHomeCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_return_cfg_buf, 0, bytemuck::bytes_of(&chronicle_return_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_returnhome(
            &mut self.cache,
            &physics_verb_chronicle_ReturnHome::PhysicsVerbChronicleReturnHomeBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_return_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        let chronicle_complete_cfg = physics_verb_chronicle_CompleteQuest::PhysicsVerbChronicleCompleteQuestCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_complete_cfg_buf, 0, bytemuck::bytes_of(&chronicle_complete_cfg),
        );
        dispatch::dispatch_physics_verb_chronicle_completequest(
            &mut self.cache,
            &physics_verb_chronicle_CompleteQuest::PhysicsVerbChronicleCompleteQuestBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                cfg: &self.chronicle_complete_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (5) ApplyStageAdvance — reads StageAdvanced (kind=1u),
        // writes mana = mana + 1.0. Run AFTER the chronicle kernels
        // so the StageAdvanced events are present in the ring.
        let apply_stage_cfg = physics_ApplyStageAdvance::PhysicsApplyStageAdvanceCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_stage_cfg_buf, 0, bytemuck::bytes_of(&apply_stage_cfg),
        );
        dispatch::dispatch_physics_applystageadvance(
            &mut self.cache,
            &physics_ApplyStageAdvance::PhysicsApplyStageAdvanceBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_mana: &self.agent_mana_buf,
                cfg: &self.apply_stage_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (6) ApplyQuestCompleted — reads QuestCompleted (kind=2u),
        // writes mana = 0.0.
        let apply_complete_cfg = physics_ApplyQuestCompleted::PhysicsApplyQuestCompletedCfg {
            event_count: event_count_estimate, tick, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.apply_complete_cfg_buf, 0, bytemuck::bytes_of(&apply_complete_cfg),
        );
        dispatch::dispatch_physics_applyquestcompleted(
            &mut self.cache,
            &physics_ApplyQuestCompleted::PhysicsApplyQuestCompletedBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                agent_mana: &self.agent_mana_buf,
                cfg: &self.apply_complete_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (7) seed_indirect_0 — keeps indirect-args buffer warm.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count, tick, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg),
        );
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache,
            &seed_indirect_0::SeedIndirect0Bindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                indirect_args_0: self.event_ring.indirect_args_0(),
                cfg: &self.seed_cfg_buf,
            },
            &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (8a) fold_quests_completed — RMW per QuestCompleted event.
        let quests_cfg = fold_quests_completed::FoldQuestsCompletedCfg {
            event_count: event_count_estimate, tick, second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.quests_completed_cfg_buf, 0, bytemuck::bytes_of(&quests_cfg),
        );
        dispatch::dispatch_fold_quests_completed(
            &mut self.cache,
            &fold_quests_completed::FoldQuestsCompletedBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.quests_completed.primary(),
                view_storage_anchor: self.quests_completed.anchor(),
                view_storage_ids: self.quests_completed.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.quests_completed_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (8b) fold_stage_advances — RMW per StageAdvanced event.
        let stage_cfg = fold_stage_advances::FoldStageAdvancesCfg {
            event_count: event_count_estimate, tick, second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.stage_advances_cfg_buf, 0, bytemuck::bytes_of(&stage_cfg),
        );
        dispatch::dispatch_fold_stage_advances(
            &mut self.cache,
            &fold_stage_advances::FoldStageAdvancesBindings {
                event_ring: self.event_ring.ring(),
                event_tail: self.event_ring.tail(),
                view_storage_primary: self.stage_advances.primary(),
                view_storage_anchor: self.stage_advances.anchor(),
                view_storage_ids: self.stage_advances.ids(),
                sim_cfg: self.event_ring.sim_cfg(),
                cfg: &self.stage_advances_cfg_buf,
            },
            &self.gpu.device, &mut encoder, event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.quests_completed.mark_dirty();
        self.stage_advances.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(QuestArcRealState::new(seed, agent_count))
}
