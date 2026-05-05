//! Per-fixture runtime for `assets/sim/dungeon_crawl.sim` — the 20th
//! real sim and the FIRST with multi-encounter sequencing.
//!
//! Five Hero combatants (slots 0-4) traverse THREE rooms in series:
//!
//!   - Room 1: 6 weak enemies (HP=50)   in slots 5-10
//!   - Room 2: 10 medium enemies (HP=100) in slots 11-20
//!   - Room 3: 1 boss-like enemy (HP=800) in slot 21
//!
//! Total agent capacity: 35 slots. The DSL is the same Strike/Spell/
//! Heal verb cascade used by `duel_1v1.sim`. CPU-side encounter pacing
//! lives in this crate: after each `step()`, the runtime reads the
//! `agent_alive` buffer and, when no enemy in the current room is
//! alive AND another room remains, writes the next room's enemies
//! into the GPU buffers (alive=1, hp=room_hp).
//!
//! ## Termination conditions
//!
//!   - VICTORY: room 3 boss is dead and all rooms have been spawned
//!   - DEFEAT:  all 5 heroes are dead
//!   - else: keep stepping (capped at MAX_TICKS in the harness)

use engine::sim_trait::CompiledSim;
use engine::GpuContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

include!(concat!(env!("OUT_DIR"), "/generated.rs"));

use engine::gpu::{EventRing, ViewStorage};

pub const HERO_COUNT: u32 = 5;
pub const HERO_HP: f32 = 400.0;
pub const TOTAL_SLOTS: u32 = 36;

/// Slot 0 is the "anchor" combatant. It exists ONLY to satisfy the
/// PerPair mask kernel's hardcoded `cand=0` candidate (TODO task-5.7
/// in the compiler — the mask emitter writes
/// `mask_k = 1u; cand = pair % 1 = 0` so the cand axis collapses to
/// slot 0). Every other agent's verb mask checks `alive[0] != 0u` —
/// if slot 0 ever dies, the whole combat halts. The CPU-side advance
/// step refreshes anchor.alive=1 + anchor.hp=HERO_HP each tick so the
/// anchor never dies. The anchor itself never acts (Strike/Spell mask
/// require `cand != agent_id`, which fails for agent_id=0=cand=0).
pub const ANCHOR_SLOT: u32 = 0;
/// Hero slots are 1..1+HERO_COUNT.
pub const HERO_FIRST_SLOT: u32 = 1;

/// Per-room layout: how many enemies + their starting HP + slot range.
#[derive(Copy, Clone, Debug)]
pub struct RoomSpec {
    pub label: &'static str,
    pub start_slot: u32,
    pub count: u32,
    pub hp: f32,
}

// Slot layout: 0=anchor, 1..6=heroes (5 of them), 6..36=enemy slots.
//
// HP values are tuned for the lowest-HP-wins Spell scoring:
//   - Each room's enemies are LOWER HP than heroes so Spell argmax
//     prefers enemies as targets while any are alive.
//   - When the room is cleared, all that's left is heroes (HP~200) +
//     the next room's spawn. The boss (room 3) is set BELOW hero HP
//     too so heroes (and the boss) keep targeting each other instead
//     of friendly-firing teammates. Single boss = focused fire.
pub const ROOMS: [RoomSpec; 3] = [
    RoomSpec { label: "Room 1 (weaklings)", start_slot: 6,  count: 6,  hp: 50.0 },
    RoomSpec { label: "Room 2 (medium)",    start_slot: 12, count: 10, hp: 100.0 },
    RoomSpec { label: "Room 3 (BOSS)",      start_slot: 22, count: 1,  hp: 180.0 },
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DungeonOutcome {
    InProgress,
    Victory,
    Defeat,
}

pub struct DungeonCrawlState {
    gpu: GpuContext,

    // -- Agent SoA --
    agent_hp_buf: wgpu::Buffer,
    agent_alive_buf: wgpu::Buffer,
    agent_mana_buf: wgpu::Buffer,

    // -- Mask bitmaps (Strike=0, Spell=1, Heal=2) --
    mask_0_bitmap_buf: wgpu::Buffer,
    mask_1_bitmap_buf: wgpu::Buffer,
    mask_2_bitmap_buf: wgpu::Buffer,
    mask_bitmap_zero_buf: wgpu::Buffer,
    mask_bitmap_words: u32,

    // -- Scoring output (4 × u32 per agent) --
    scoring_output_buf: wgpu::Buffer,
    scoring_output_zero_buf: wgpu::Buffer,

    // -- Event ring + per-view storage --
    event_ring: EventRing,
    damage_dealt: ViewStorage,
    damage_dealt_cfg_buf: wgpu::Buffer,
    healing_done: ViewStorage,
    healing_done_cfg_buf: wgpu::Buffer,

    // -- Per-kernel cfg uniforms --
    mask_cfg_buf: wgpu::Buffer,
    scoring_cfg_buf: wgpu::Buffer,
    chronicle_strike_cfg_buf: wgpu::Buffer,
    chronicle_spell_cfg_buf: wgpu::Buffer,
    chronicle_heal_cfg_buf: wgpu::Buffer,
    apply_cfg_buf: wgpu::Buffer,
    seed_cfg_buf: wgpu::Buffer,

    cache: dispatch::KernelCache,

    tick: u64,
    agent_count: u32,
    seed: u64,

    /// Index into ROOMS — which room is currently spawned. `None`
    /// means no room is active yet (initialisation state).
    current_room: Option<usize>,
    /// Highest room ever spawned, used to detect VICTORY (boss dead).
    rooms_cleared: u32,
    outcome: DungeonOutcome,
}

impl DungeonCrawlState {
    pub fn new(seed: u64) -> Self {
        let agent_count = TOTAL_SLOTS;
        let gpu = GpuContext::new_blocking().expect("init wgpu adapter + device");

        // Heroes alive @ HP=200. Dead enemy slots start at a HUGE
        // sentinel HP so they never win the lowest-HP-target argmax in
        // the Spell scoring row (which iterates ALL slots, regardless
        // of `alive` — that gate lives in the mask kernel, not the
        // pair-field score iteration). When a room spawns, those slots
        // get overwritten with their real room HP; when they die, the
        // ApplyDamage chronicle drops them to <=0 — but by then they
        // are alive=0, so the mask predicate skips them as targets.
        const DEAD_SLOT_HP_SENTINEL: f32 = 1.0e9;
        let mut hp_init: Vec<f32> = vec![DEAD_SLOT_HP_SENTINEL; agent_count as usize];
        let mut alive_init: Vec<u32> = vec![0u32; agent_count as usize];
        // Slot 0 = anchor (always alive, never acts, sentinel HP so the
        // Spell scoring kernel's lowest-HP-target argmax never picks
        // the anchor as a victim).
        hp_init[ANCHOR_SLOT as usize] = DEAD_SLOT_HP_SENTINEL;
        alive_init[ANCHOR_SLOT as usize] = 1;
        // Heroes occupy slots HERO_FIRST_SLOT..HERO_FIRST_SLOT+HERO_COUNT.
        for h in 0..HERO_COUNT {
            let s = (HERO_FIRST_SLOT + h) as usize;
            hp_init[s] = HERO_HP;
            alive_init[s] = 1;
        }
        let agent_hp_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::agent_hp"),
            contents: bytemuck::cast_slice(&hp_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        let agent_alive_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::agent_alive"),
            contents: bytemuck::cast_slice(&alive_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        // Mana not used as a real resource (Spell `when` checks
        // `self.mana >= 5.0` and never decrements). Initialise to 100
        // so all slots can always cast.
        let mana_init: Vec<f32> = vec![100.0_f32; agent_count as usize];
        let agent_mana_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::agent_mana"),
            contents: bytemuck::cast_slice(&mana_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Three mask bitmaps — one per verb. Cleared each tick.
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
        let mask_0_bitmap_buf = mk_mask("dungeon_crawl_runtime::mask_0_bitmap");
        let mask_1_bitmap_buf = mk_mask("dungeon_crawl_runtime::mask_1_bitmap");
        let mask_2_bitmap_buf = mk_mask("dungeon_crawl_runtime::mask_2_bitmap");
        let zero_words: Vec<u32> = vec![0u32; mask_bitmap_words.max(4) as usize];
        let mask_bitmap_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::mask_bitmap_zero"),
            contents: bytemuck::cast_slice(&zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        // Scoring output — 4 × u32 per agent. Cleared each tick.
        let scoring_output_words = (agent_count as u64) * 4;
        let scoring_output_bytes = scoring_output_words * 4;
        let scoring_output_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dungeon_crawl_runtime::scoring_output"),
            size: scoring_output_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scoring_zero_words: Vec<u32> = vec![0u32; (scoring_output_words as usize).max(4)];
        let scoring_output_zero_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::scoring_output_zero"),
            contents: bytemuck::cast_slice(&scoring_zero_words),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        let event_ring = EventRing::new(&gpu, "dungeon_crawl_runtime");
        let damage_dealt = ViewStorage::new(
            &gpu, "dungeon_crawl_runtime::damage_dealt", agent_count, false, false,
        );
        let healing_done = ViewStorage::new(
            &gpu, "dungeon_crawl_runtime::healing_done", agent_count, false, false,
        );

        let mask_cfg_init = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let mask_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::mask_cfg"),
            contents: bytemuck::bytes_of(&mask_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let scoring_cfg_init = scoring::ScoringCfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let scoring_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::scoring_cfg"),
            contents: bytemuck::bytes_of(&scoring_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_strike_cfg_init = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let chronicle_strike_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::chronicle_strike_cfg"),
            contents: bytemuck::bytes_of(&chronicle_strike_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_spell_cfg_init = physics_verb_chronicle_Spell::PhysicsVerbChronicleSpellCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let chronicle_spell_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::chronicle_spell_cfg"),
            contents: bytemuck::bytes_of(&chronicle_spell_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let chronicle_heal_cfg_init = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let chronicle_heal_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::chronicle_heal_cfg"),
            contents: bytemuck::bytes_of(&chronicle_heal_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let apply_cfg_init = physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
            event_count: 0, tick: 0, seed: 0, _pad0: 0,
        };
        let apply_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::apply_cfg"),
            contents: bytemuck::bytes_of(&apply_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let seed_cfg_init = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: agent_count, tick: 0, seed: 0, _pad: 0,
        };
        let seed_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::seed_cfg"),
            contents: bytemuck::bytes_of(&seed_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let damage_cfg_init = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let damage_dealt_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::damage_dealt_cfg"),
            contents: bytemuck::bytes_of(&damage_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let healing_cfg_init = fold_healing_done::FoldHealingDoneCfg {
            event_count: 0, tick: 0, second_key_pop: 1, _pad: 0,
        };
        let healing_done_cfg_buf = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dungeon_crawl_runtime::healing_done_cfg"),
            contents: bytemuck::bytes_of(&healing_cfg_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let mut state = Self {
            gpu,
            agent_hp_buf,
            agent_alive_buf,
            agent_mana_buf,
            mask_0_bitmap_buf,
            mask_1_bitmap_buf,
            mask_2_bitmap_buf,
            mask_bitmap_zero_buf,
            mask_bitmap_words,
            scoring_output_buf,
            scoring_output_zero_buf,
            event_ring,
            damage_dealt,
            damage_dealt_cfg_buf,
            healing_done,
            healing_done_cfg_buf,
            mask_cfg_buf,
            scoring_cfg_buf,
            chronicle_strike_cfg_buf,
            chronicle_spell_cfg_buf,
            chronicle_heal_cfg_buf,
            apply_cfg_buf,
            seed_cfg_buf,
            cache: dispatch::KernelCache::default(),
            tick: 0,
            agent_count,
            seed,
            current_room: None,
            rooms_cleared: 0,
            outcome: DungeonOutcome::InProgress,
        };

        // Spawn room 1 immediately so the first tick already has
        // enemies on the field.
        state.spawn_room(0);
        state
    }

    pub fn current_room(&self) -> Option<usize> { self.current_room }
    pub fn rooms_cleared(&self) -> u32 { self.rooms_cleared }
    pub fn outcome(&self) -> DungeonOutcome { self.outcome }

    /// Spawn the enemies for the given room index. CPU writes alive=1
    /// + hp=room_hp into the GPU buffers for the room's slot range.
    pub fn spawn_room(&mut self, room_idx: usize) {
        let spec = ROOMS[room_idx];
        // Splat updates: build a Vec the size of the slot range and
        // upload via `queue.write_buffer` at the slot offset.
        let hp_splat: Vec<f32> = vec![spec.hp; spec.count as usize];
        let alive_splat: Vec<u32> = vec![1u32; spec.count as usize];
        let hp_offset = (spec.start_slot as u64) * 4;
        let alive_offset = (spec.start_slot as u64) * 4;
        self.gpu.queue.write_buffer(
            &self.agent_hp_buf, hp_offset, bytemuck::cast_slice(&hp_splat),
        );
        self.gpu.queue.write_buffer(
            &self.agent_alive_buf, alive_offset, bytemuck::cast_slice(&alive_splat),
        );
        self.current_room = Some(room_idx);
    }

    /// Returns true if any enemy slot in the current room is alive.
    /// Reads back the agent_alive buffer (synchronously) — called after
    /// each `step()` to detect room-cleared events.
    fn current_room_has_alive(&self) -> bool {
        let alive = self.read_alive();
        if let Some(idx) = self.current_room {
            let spec = ROOMS[idx];
            for s in spec.start_slot..(spec.start_slot + spec.count) {
                if alive.get(s as usize).copied() == Some(1) {
                    return true;
                }
            }
        }
        false
    }

    /// Returns true if any hero slot (excluding the anchor) is alive.
    fn any_hero_alive(&self) -> bool {
        let alive = self.read_alive();
        (0..HERO_COUNT).any(|h| {
            let s = (HERO_FIRST_SLOT + h) as usize;
            alive.get(s).copied() == Some(1)
        })
    }

    /// Quarantine HP of dead slots — overwrite with the sentinel
    /// (1e9) so they never win the lowest-HP-target argmax in the
    /// Spell scoring row's inner-candidate loop. Also keep the anchor
    /// slot (0) refreshed to alive=1 + hp=HERO_HP so the mask kernel's
    /// hardcoded `cand=0` candidate never gates off the entire combat.
    /// Both are CPU-side workarounds documented in `lib.rs::new` and
    /// at `ANCHOR_SLOT`.
    fn quarantine_dead_slots(&self) {
        const DEAD_SLOT_HP_SENTINEL: f32 = 1.0e9;
        // Refresh anchor every tick (cheap, deterministic). Sentinel
        // HP so the anchor never wins lowest-HP argmax.
        let anchor_one: u32 = 1;
        let anchor_hp: f32 = DEAD_SLOT_HP_SENTINEL;
        self.gpu.queue.write_buffer(
            &self.agent_alive_buf,
            (ANCHOR_SLOT as u64) * 4,
            bytemuck::bytes_of(&anchor_one),
        );
        self.gpu.queue.write_buffer(
            &self.agent_hp_buf,
            (ANCHOR_SLOT as u64) * 4,
            bytemuck::bytes_of(&anchor_hp),
        );
        let alive = self.read_alive();
        let hp = self.read_hp();
        let mut writes: Vec<(u64, f32)> = Vec::new();
        for slot in 0..self.agent_count {
            if slot == ANCHOR_SLOT { continue; }
            let s = slot as usize;
            if alive[s] == 0 && hp[s] < DEAD_SLOT_HP_SENTINEL * 0.5 {
                writes.push(((slot as u64) * 4, DEAD_SLOT_HP_SENTINEL));
            }
        }
        for (offset, val) in writes {
            self.gpu.queue.write_buffer(
                &self.agent_hp_buf, offset, bytemuck::bytes_of(&val),
            );
        }
    }

    /// CPU-side encounter pacing — call after each `step()`. Advances
    /// `current_room` when the active room is cleared, spawns the
    /// next room's enemies, and updates `outcome`.
    pub fn advance_encounter(&mut self) {
        if !matches!(self.outcome, DungeonOutcome::InProgress) {
            return;
        }
        // Quarantine first — keeps the next tick's Spell scoring from
        // chasing dead slots.
        self.quarantine_dead_slots();
        if !self.any_hero_alive() {
            self.outcome = DungeonOutcome::Defeat;
            return;
        }
        if self.current_room_has_alive() {
            return;
        }
        // Active room cleared. Promote rooms_cleared, then either spawn
        // next or declare VICTORY.
        let cleared_idx = self.current_room.expect("current_room set after init");
        self.rooms_cleared = (cleared_idx as u32) + 1;
        if cleared_idx + 1 < ROOMS.len() {
            self.spawn_room(cleared_idx + 1);
        } else {
            // All 3 rooms cleared — boss is dead.
            self.outcome = DungeonOutcome::Victory;
        }
    }

    pub fn damage_dealt(&mut self) -> &[f32] {
        self.damage_dealt.readback(&self.gpu)
    }
    pub fn healing_done(&mut self) -> &[f32] {
        self.healing_done.readback(&self.gpu)
    }
    pub fn read_hp(&self) -> Vec<f32> {
        self.read_f32(&self.agent_hp_buf, "hp")
    }
    pub fn read_alive(&self) -> Vec<u32> {
        self.read_u32(&self.agent_alive_buf, "alive")
    }

    fn read_f32(&self, buf: &wgpu::Buffer, label: &str) -> Vec<f32> {
        let bytes = (self.agent_count as u64) * 4;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("dungeon_crawl_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("dungeon_crawl_runtime::read_f32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
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
            label: Some(&format!("dungeon_crawl_runtime::{label}_staging")),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("dungeon_crawl_runtime::read_u32") },
        );
        encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
        self.gpu.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = sender.send(r); });
        self.gpu.device.poll(wgpu::PollType::Wait).expect("poll");
        let _ = receiver.recv().expect("map_async result");
        let mapped = slice.get_mapped_range();
        let v: Vec<u32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        v
    }

    pub fn agent_count(&self) -> u32 { self.agent_count }
    pub fn tick(&self) -> u64 { self.tick }
    pub fn seed(&self) -> u64 { self.seed }
}

impl CompiledSim for DungeonCrawlState {
    fn step(&mut self) {
        let mut encoder = self.gpu.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("dungeon_crawl_runtime::step") },
        );

        // (1) Per-tick clears.
        self.event_ring.clear_tail_in(&mut encoder);
        let max_slots_per_tick = self.agent_count * 4;
        self.event_ring.clear_ring_headers_in(
            &self.gpu, &mut encoder, max_slots_per_tick,
        );
        let mask_bytes = (self.mask_bitmap_words as u64) * 4;
        for buf in [&self.mask_0_bitmap_buf, &self.mask_1_bitmap_buf, &self.mask_2_bitmap_buf] {
            encoder.copy_buffer_to_buffer(
                &self.mask_bitmap_zero_buf, 0, buf, 0, mask_bytes.max(4),
            );
        }
        let scoring_output_bytes = (self.agent_count as u64) * 4 * 4;
        encoder.copy_buffer_to_buffer(
            &self.scoring_output_zero_buf, 0, &self.scoring_output_buf,
            0, scoring_output_bytes.max(16),
        );

        // (2) Mask round.
        let mask_cfg = fused_mask_verb_Strike::FusedMaskVerbStrikeCfg {
            agent_cap: self.agent_count, tick: self.tick as u32, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.mask_cfg_buf, 0, bytemuck::bytes_of(&mask_cfg));
        let mask_bindings = fused_mask_verb_Strike::FusedMaskVerbStrikeBindings {
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            agent_mana: &self.agent_mana_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            cfg: &self.mask_cfg_buf,
        };
        dispatch::dispatch_fused_mask_verb_strike(
            &mut self.cache, &mask_bindings, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (3) Scoring.
        let scoring_cfg = scoring::ScoringCfg {
            agent_cap: self.agent_count, tick: self.tick as u32, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.scoring_cfg_buf, 0, bytemuck::bytes_of(&scoring_cfg));
        let scoring_bindings = scoring::ScoringBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            mask_0_bitmap: &self.mask_0_bitmap_buf,
            mask_1_bitmap: &self.mask_1_bitmap_buf,
            mask_2_bitmap: &self.mask_2_bitmap_buf,
            scoring_output: &self.scoring_output_buf,
            cfg: &self.scoring_cfg_buf,
        };
        dispatch::dispatch_scoring(
            &mut self.cache, &scoring_bindings, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (4) Strike chronicle.
        let strike_cfg = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_strike_cfg_buf, 0, bytemuck::bytes_of(&strike_cfg),
        );
        let strike_bindings = physics_verb_chronicle_Strike::PhysicsVerbChronicleStrikeBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_strike_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_strike(
            &mut self.cache, &strike_bindings, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (5) Spell chronicle.
        let spell_cfg = physics_verb_chronicle_Spell::PhysicsVerbChronicleSpellCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_spell_cfg_buf, 0, bytemuck::bytes_of(&spell_cfg),
        );
        let spell_bindings = physics_verb_chronicle_Spell::PhysicsVerbChronicleSpellBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_spell_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_spell(
            &mut self.cache, &spell_bindings, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (6) Heal chronicle.
        let heal_cfg = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealCfg {
            event_count: self.agent_count, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(
            &self.chronicle_heal_cfg_buf, 0, bytemuck::bytes_of(&heal_cfg),
        );
        let heal_bindings = physics_verb_chronicle_Heal::PhysicsVerbChronicleHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            cfg: &self.chronicle_heal_cfg_buf,
        };
        dispatch::dispatch_physics_verb_chronicle_heal(
            &mut self.cache, &heal_bindings, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (7) ApplyDamage_and_ApplyHeal.
        let event_count_estimate = self.agent_count * 4;
        let apply_cfg = physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealCfg {
            event_count: event_count_estimate, tick: self.tick as u32, seed: 0, _pad0: 0,
        };
        self.gpu.queue.write_buffer(&self.apply_cfg_buf, 0, bytemuck::bytes_of(&apply_cfg));
        let apply_bindings = physics_ApplyDamage_and_ApplyHeal::PhysicsApplyDamageAndApplyHealBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            agent_hp: &self.agent_hp_buf,
            agent_alive: &self.agent_alive_buf,
            cfg: &self.apply_cfg_buf,
        };
        dispatch::dispatch_physics_applydamage_and_applyheal(
            &mut self.cache, &apply_bindings, &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (8) seed_indirect_0.
        let seed_cfg = seed_indirect_0::SeedIndirect0Cfg {
            agent_cap: self.agent_count, tick: self.tick as u32, seed: 0, _pad: 0,
        };
        self.gpu.queue.write_buffer(&self.seed_cfg_buf, 0, bytemuck::bytes_of(&seed_cfg));
        let seed_bindings = seed_indirect_0::SeedIndirect0Bindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            indirect_args_0: self.event_ring.indirect_args_0(),
            cfg: &self.seed_cfg_buf,
        };
        dispatch::dispatch_seed_indirect_0(
            &mut self.cache, &seed_bindings, &self.gpu.device, &mut encoder, self.agent_count,
        );

        // (9) fold_damage_dealt.
        let damage_cfg = fold_damage_dealt::FoldDamageDealtCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.damage_dealt_cfg_buf, 0, bytemuck::bytes_of(&damage_cfg),
        );
        let damage_bindings = fold_damage_dealt::FoldDamageDealtBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.damage_dealt.primary(),
            view_storage_anchor: self.damage_dealt.anchor(),
            view_storage_ids: self.damage_dealt.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.damage_dealt_cfg_buf,
        };
        dispatch::dispatch_fold_damage_dealt(
            &mut self.cache, &damage_bindings, &self.gpu.device, &mut encoder, event_count_estimate,
        );

        // (10) fold_healing_done.
        let healing_cfg = fold_healing_done::FoldHealingDoneCfg {
            event_count: event_count_estimate, tick: self.tick as u32,
            second_key_pop: 1, _pad: 0,
        };
        self.gpu.queue.write_buffer(
            &self.healing_done_cfg_buf, 0, bytemuck::bytes_of(&healing_cfg),
        );
        let healing_bindings = fold_healing_done::FoldHealingDoneBindings {
            event_ring: self.event_ring.ring(),
            event_tail: self.event_ring.tail(),
            view_storage_primary: self.healing_done.primary(),
            view_storage_anchor: self.healing_done.anchor(),
            view_storage_ids: self.healing_done.ids(),
            sim_cfg: self.event_ring.sim_cfg(),
            cfg: &self.healing_done_cfg_buf,
        };
        dispatch::dispatch_fold_healing_done(
            &mut self.cache, &healing_bindings, &self.gpu.device, &mut encoder, event_count_estimate,
        );

        self.gpu.queue.submit(Some(encoder.finish()));
        self.damage_dealt.mark_dirty();
        self.healing_done.mark_dirty();
        self.tick += 1;
    }

    fn agent_count(&self) -> u32 { self.agent_count }
    fn tick(&self) -> u64 { self.tick }
    fn positions(&mut self) -> &[Vec3] { &[] }
}

pub fn make_sim(seed: u64, _agent_count: u32) -> Box<dyn CompiledSim> {
    Box::new(DungeonCrawlState::new(seed))
}
